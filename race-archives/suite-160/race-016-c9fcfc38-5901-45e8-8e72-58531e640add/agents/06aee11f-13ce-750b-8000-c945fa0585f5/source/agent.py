import itertools
import json
import math
import re
import time

from src.agent.proxy_client import ProxyClient

DEEPSEEK, LUNA, SOL = "deepseek/deepseek-v4.1-flash", "openai/gpt-6-luna", "openai/gpt-6-sol"
BACKUP = {SOL: LUNA, LUNA: DEEPSEEK, DEEPSEEK: LUNA}
CHAT, CALL = "/inference/chat/completions", "/environment/call"
PLATFORM, CALL_TIMEOUT, ENV_TIMEOUT, LOOP, RETRIES = 300, 60, 100, 1000, 3
MAP_TRIES, ODDS = 4, math.log(100)
SHELF, FREE_TEXT, NAME_WORDS, JUDGE_BATCH, OPTIONS, EXAMPLES = 60, 40, 4, 12, 16, 4
HIDDEN, AGAIN, VARIED = 0.5, 0.1, 1.2
FIELDS = ("price", "stock", "category", "brand", "title", "description")
ROLE = {"search": "finds products by keywords and/or filters", "view": "shows one product's listing (title, price, stock)", "details": "shows one product's properties, attributes or specifications",
        "compare": "shows the properties of several products at once, given a list of product ids", "add": "puts a product into the cart", "remove": "takes a product out of the cart (or lowers its quantity)",
        "review": "shows the cart", "message": "sends a message or question to the shopper", "approve": "asks the shopper to approve the basket or order before checkout",
        "status": "reports how many messages to the shopper, or steps, have been used or are left", "checkout": "buys what is in the cart, or the products it is given, and finishes the task"}
ROLES = tuple(r for r in ROLE if r != "search")
PLAN = ("add", "remove", "checkout")
NEEDS = {"view": "<ID>", "details": "<ID>", "compare": "<ID>", "add": "<ID>", "remove": "<ID>", "message": "<TEXT>"}
WIDEN = {"dropped", "changed", "raised"}
SHIFTS = {"need", "changed", "dropped", "budget", "raised", "quantity", "price", "out_of_stock", "other"}
READS = ("search", "view", "details", "compare", "review")
TURNS, MENDS, RAW, STEP = {"search": 25, "read": 60, "cart": 15, "final": 6}, 10, 6000, {"search": "search", "view": "read", "details": "read", "compare": "read", "message": "ask", "approve": "ask", "add": "cart", "remove": "cart", "review": "cart", "checkout": "final"}
ITEM = ("product", "reqs", "qty", "budget", "per_unit", "details", "cat_ok", "searched", "search_rows", "asked", "known", "revealed", "lost", "objective", "best_by", "best_high", "measure")


class Prompts:
    TOOLS = """You map the tools of an online store to the roles a shopping agent needs. Read each tool's name, description and JSON parameter schema.
Roles:
""" + "\n".join("- %s: %s." % kv for kv in ROLE.items()) + """
Return one JSON object with every role as a key.
"search" is a list with one entry per search tool: {"tool": "<name>", "page": "<argument that selects the result page, or null>", "page_kind": "number when it counts pages, offset when it counts results to skip", "price_cap": "<argument that sets a maximum price, or null>"}.
Every other role is null when no tool fits, else {"tool": "<name>", "args": {...}}: a template with every required argument and every argument the role needs, using the placeholders "<ID>" for the product id, "<QTY>" for the quantity and "<TEXT>" for free text; write fixed values (such as the enum value that selects adding or removing) literally. When a tool takes a list of cart lines or of product ids, its template holds that list with one entry made of the placeholders. When no tool puts products into a cart and the tool that buys takes the products itself, add is null and the checkout template holds the product and quantity placeholders. The add entry also gives "replaces": true when one call sets the whole cart to exactly the lines it is given, false when it adds them to what the cart already holds. A tool that either buys or ends without buying is checkout with the fixed value that buys; one tool may be both view and details. The checkout template also gives "<TEXT>" to an argument that takes the agent's own words (such as a note, reason or justification), also when that argument is optional."""
    SPEC = """You read a shopper's request to an online store and extract the shopping spec.
Return one JSON object:
{"items": [{"product": "",
            "requirements": [{"attribute": "", "values": [""], "exclude": false, "min": null, "max": null, "unit": ""}],
            "quantity": 1,
            "budget": null,
            "budget_per_unit": false,
            "objective": null,
            "best_by": "",
            "best_high": true}],
 "budget": null,
 "budget_unit": "",
 "budget_per_unit": false,
 "objective": "any",
 "best_by": "",
 "best_high": true,
 "messages_allowed": null,
 "actions_allowed": null,
 "seconds_allowed": null,
 "purpose": "",
 "stock_rule": "in_stock",
 "references": [],
 "wishes": [],
 "rules": [],
 "basket_rules": [{"quote": "", "item": null, "on": null}]}
- items: one entry per different product the shopper asks to buy in this order, in the request's order; most requests want one product. Several units of one product are one entry with their quantity, and a product the request names again, also by another, broader or narrower name (a kind, form or shape of it), is the same entry. Kinds named together as one thing to buy, joined by "or", "and" or commas, with one shared quantity or none and the same requirements ("binoculars or a telescope", "pots, pans and lids that are oven-safe"), are one entry whose product is the whole phrase, as a store category names them; a kind gets its own entry only with its own quantity or requirements. Something the shopper already owns, or that a product must go with or fit, is not an entry.
- product: the product type or category wanted, with every word of its name as the request gives it, but without words that only count, package or call it a thing (a pack of, a set of, an item, items, a kit) or join and explain it (so, that is); a category followed by the kinds it holds ("cookware, so pots, pans and lids") is the category name alone. Never split words of the product's name into requirements; a value the request also gives for one of its properties (such as its type, or a kind within the category) goes only in that requirement.
- requirements: every property the request requires of that product or rules out, one property per entry, for the last product as fully as for the first; a property stated again in other words (a restatement, a reason, a story) is the same entry. attribute: the property's name as the request gives it, else the plain name of the property the value describes (material for "made of"; for a part made for a maker or a vehicle: make only for the maker's name, model for a vehicle named by its model name, so "for a Corolla" or "for a Civic" gives model, not make). values: the value in the request's words, without the property's name, without a marker that it is the only value accepted (only, just, nothing but), and without any comparison, simile or decoration around it. In a "<value> <property>" phrase the property is the whole tail that names it, with the word saying which property is meant ("large bag pack type" gives pack type: large bag, "for the coastal route series" gives route series: coastal), while a phrase naming a kind of a property keeps all its words as the value ("a dealer warranty" gives warranty type: dealer warranty). When the shopper accepts any one of several values, list them all in one entry, also when each comes with its own property word ("a classic style or nailhead trim": style). A value marked as the only one accepted, also inside such a phrase ("a gas only hob, nothing electric in it"), is the value alone (fuel: gas) plus an entry with exclude true for what is ruled out (electric), never "gas only". exclude: true when the shopper rules the listed values out; a value ruled out next to a required value of the same property is its own entry with exclude true, while a negative word that is part of the stated value (a series that "is not available") stays in the value. A property that holds only on a condition about another product of the order is a basket rule, not a requirement. min, max: when the property must be at least or at most some number, those bound numbers, with unit: the unit as written; else null, null and "".
- A named model of a device or vehicle outside this order that a product must fit, suit or work with is a requirement of that product: attribute the property the request names for it, else "compatibility"; values that model as written, without the words around it that name the property or what it belongs to ("for a bike of the coastal route series" gives route series: coastal). A place or a use is no such model.
- Background about the shopper, their plans, or what they will do with the product is not a requirement unless the request says the product itself must have that property.
- Never a requirement: the product type or category itself (a category the request names goes in product), price, budget or any money amount, stock or availability (stock_rule holds what the request says about stock), quantity, delivery, or how the agent should work.
- Each requirement is a property of the product itself with a value the request states plainly. A word that only says the product should suit the shopper, or compares it with something (such as suitable, nicer or bigger), is not a value.
- quantity: how many units of that product to buy. A set, pack, kit or lot of several pieces is one unit; its piece count is not the quantity.
- budget in an item: a price limit the request sets for that product alone, as a number in budget_unit; else null. budget_per_unit there: true when that limit is for each unit.
- budget: the most the shopper will pay for everything bought, as a number in budget_unit; null when the request only limits products one by one. budget_unit: the smallest unit of the request's currency (its hundredth part when the currency has one), named in the request's words; an amount in a larger unit is converted into it exactly, and an amount written in words is written in digits. budget_per_unit: true only when the budget is for each unit.
- objective: "cheapest" only when the shopper asks for the lowest-priced acceptable product or order (a price limit or a wish to save money alone is not that), "best" when the best by some measure (best_by names the measure, best_high is false when a lower value is better), else "any".
- objective, best_by and best_high in an item: the same, when the request gives that product a way to choose between fitting listings of its own (such as the lowest price, or the most or least of some property) that differs from the whole order's; else null, "" and true.
- messages_allowed: the number of messages the request or a tool description lets the agent send to the shopper, as a number (also when written in words); 0 when the request forbids messaging the shopper; null only when none of them states such a number.
- actions_allowed: the number of actions or steps the request or a tool description allows the agent, as a number; null when not stated.
- seconds_allowed: the time limit for the whole task in seconds, when the request, a tool description or the input's limits state one; null when none is stated.
- purpose: who or what the products are for, in a few words.
- stock_rule: "any" when the request accepts products that are out of stock now (it can wait for them or does not mind); else "in_stock".
- references: one entry per property whose value must be read from another listing the request points to (the same as, matching, going with or compatible with, within some amount of, at least, at most, or other than the value on that listing's page): {"item": <0-based index of the items entry it is for>, "attribute": "<the property>", "relation": "same" or "within" or "at_least" or "at_most" or "different", "margin": <the allowed difference for within, else null>, "unit": "<the margin's unit>", "listing": {"seller": "<the seller, shop or maker named>", "product": "<that listing's product type>", "facts": {"<property>": "<value as written>"}}}. attribute: the property the request says to take from that page (often named with the pointer, such as "that's the <property>" or "as its <property>"); the properties and values that only tell which listing is meant are its facts, never an attribute, so a pointer that names some facts and one property to take is one entry. relation: "same" also when the value must go with, fit or be compatible with that page's. listing: a name before the product type or in the possessive is the seller; product is that listing's product type alone; facts holds every other detail the request gives about that listing, one key per property: the ones after "with" or "for", and the ones written into its product phrase (such as who it is for, its material or its style). Such a requirement is not in requirements, and the facts that only describe that other listing are no item's requirements.
- wishes: every sentence or phrase of the request that only tells a story, a feeling, a taste or a wish about the purchase (such as comfort, delivery, wrapping, or how the shopper will use or enjoy it) and states no property the product must have, copied exactly as written. A wish is never a requirement, and a sentence that states a requirement is not a wish.
- rules: every instruction in the request or a tool description about how the agent must work or finish the task rather than what to buy, other than the message, step, time, stock and budget limits above: {"quote": "<the instruction copied exactly as written>", "before": [the agent_actions it must be followed before: the action it names or has to come first, "checkout" when it must hold when the order is placed, "message" only when it limits when or how the agent may message the shopper], "messages": <how many messages to the shopper it takes: only to tell or ask the shopper something>, "calls": <how many other agent actions it takes, such as looking at listings or the cart>, "per_line": true when those actions repeat for each product of the order}. An invitation or permission to ask (such as "ask me anything you need") or a remark or rhetorical question about what matters for a product is no rule, and neither is a request to check or ask with the shopper before choosing or picking a product: the agent's questions about that product meet it. A sentence that only restates a limit above (stock, budget, messages, steps, time) or only says where a reference's value may be read is no rule. A rule is never a requirement or a wish.
- basket_rules: every condition the request sets on the whole basket or between its products, other than the total budget and the lowest total, such as one shop or one brand for everything, one product having to work with, fit or match another product of the order, or a requirement on one product that applies only when another product turns out to have some property: {"quote": "<the condition copied exactly as written>", "item": <when the condition ties one product to another product of the order: the 0-based index of the items entry it limits (the one that must fit, match or meet the requirement), else null>, "on": <the 0-based index of the items entry that product is tied to, else null>}.
- ungrounded (only when the input has it): entries of an earlier reading of this request whose words or numbers are not in the request, that come only from its wishes or from the facts of a reference, or references that give no facts about the listing they point to. Read the request again and give each such entry only in the request's own words and numbers when the request states it; else leave it out."""
    SEND = """You check, before a shopping agent sends a message to the shopper, whether sending it now follows the shopper's instructions about messaging. The input holds those instructions, the next message, the messages already sent with the replies, the order's products and those for which the agent has already found a fitting listing. An instruction that sets no limit on when, how or how often the agent may message the shopper, such as one about what to say when nothing fits, never stops a message.
Return one JSON object: {"send": true} when sending the next message now follows every instruction, else {"send": false}."""
    USAGE = """You read what a store's usage tool returned to a shopping agent. The input holds the tool's description, its result, the number of messages the agent has sent to the shopper and the number of actions it has taken.
Return one JSON object: {"messages_left": <the number of messages the agent may still send to the shopper, or null when the result does not show it>, "steps_left": <the number of actions or tool calls the agent may still take, or null when the result does not show it>, "money_left": <the amount of money the shopper may still spend, the number as the result gives it, not converted, or null when the result does not show it>}.
Each number is what the result shows as left, or as allowed minus used, read with the tool's description; a count of something else is null."""
    TELL = """You read instructions a shopper gave a shopping agent about how to work and tell which of them need a message of their own. The input holds the numbered instructions.
Return one JSON object: {"tell": [<instruction number>, ...], "unless": [<instruction number>, ...]}
- unless: the instructions that apply only when the agent finds nothing that fits and places no order, such as what to say or offer then (to say so, to offer or not to offer a substitute or the closest listing). The agent places an order, so it never follows them.
- tell: the other instructions that require the agent to send the shopper a message that is not a question about what the shopper wants: to tell, show, report or confirm something about the basket or the order (such as the basket, its total or stock), or to get the shopper's approval of the basket or order. Not an instruction about what the agent should learn from the shopper before choosing: one that invites, allows or asks the agent to ask what it needs, to ask about a product, a part of it, its use, its user or its place, or to settle or explain a detail first, one that asks to be shown or told what the store offers (its kinds, forms or options) so that the shopper can choose, one that says the shopper will answer or tell when asked, or one that is itself a question about what to ask, settle or check: the agent's questions about the products' properties follow those."""
    NOTE = """\n- The agent's last action, which places the order, carries a written note to the shopper. An instruction that only asks the agent to say, explain, report or offer something about its outcome as it finishes (what it orders and why, that nothing fits, the listing that comes closest) is met by that note and is not in tell; one that needs the shopper's answer or approval, or asks to tell the shopper something before the order is placed, still is."""
    ACCEPT = """You read the shopper's reply to the final basket a shopping agent showed before ordering. The input holds the agent's message, the reply and the basket lines.
Return one JSON object: {"accepted": true, "rejected": [<line number>, ...]}.
- accepted: true when the reply agrees, approves or only acknowledges the basket without objecting; false when it refuses, objects or asks for something else.
- rejected: when accepted is false, the numbers of the basket lines whose listing the shopper turns down; [] when the reply names none."""
    GATE = """You check, just before a shopping agent takes its next action, whether it has followed each instruction about how to work or finish, and plan what it must do first. The input holds the next action and what it does, the numbered instructions (quoted from the request or the tools), the basket (each line with its count, price and stock as last seen, and whether its listing was viewed after its last cart change; the total), the agent's recent actions, the messages it sent with the shopper's replies, the actions it can take now with what each does, the messages and steps it has left for these instructions, and, when given, not_yet_told: the instructions that take a message to the shopper and have none sent for them since the last cart change (check each of them again).
Return one JSON object: {"unmet": [<instruction number>, ...], "steps": []}
- unmet: the instructions not yet followed. An instruction is followed when what the agent has done already does what it asks for the basket as it is now (a message that asked or told what it asks follows it whatever the reply, unless it asks for the shopper's approval and the reply gives none), when it is about what to buy rather than how to work, when it does not apply before this action, or when it can no longer be followed. The basket is the agent's own data: the shopper has seen only the messages listed, and one marked after_last_cart_change false told an older basket, so an instruction to tell, show or confirm something about the basket that holds before this action needs a new message now.
- steps: one plan that follows every unmet instruction together with the fewest actions, each {"action": "<one of the actions it can take>", "target": "<a listing id, \"each\" for every listing in the basket, or \"\">", "text": "<for an action that writes to the shopper, the whole message in plain words>", "reply": true when an instruction needs the shopper's answer, such as their approval}, in this order:
  - first each check the unmet instructions ask for: the action that shows it, only for what was not already seen after the last cart change;
  - then, when an unmet instruction asks to tell, show, confirm or ask the shopper something, one message that does all of it at once and also states the result of every check the instructions ask for, with the basket's facts as the instructions ask (such as each line with its count, price and stock, and the exact total), every amount written exactly as the basket gives it; a second message only when it needs the shopper's answer to the first; never more messages than messages_left;
  - no message when no unmet instruction asks to tell or ask the shopper anything. A step never asks about a product property and is never the next action itself."""
    ORDER = """You check a shopping agent's planned order just before it is placed, strictly and literally. The input holds the shopper's request, the order lines (each with the wanted product, the quantity, its numbered requirements from the request and from the shopper's replies, and the planned listing: name, attribute data, notes and price) and the order total.
Return one JSON object: {"lines": [{"line": 0, "fails": [{"requirement": "1", "quote": ""}]}], "rules": [{"rule": 1, "pass": true, "replace": []}]}:
- lines: one entry per order line. fails lists each numbered requirement that the planned listing's own data clearly breaks: its name, attributes or notes state another value, its notes say the property is taken from another product whose value is different, or its notes deny that it is what the requirement asks. quote: the listing's words that show it, copied exactly. When the notes say a property is taken from another product, only what the notes say about that product counts for it, never the listing's own value. A requirement the listing does not mention, or meets in other words, units or forms, is not a fail; a listing that gives several values for the property (in one value, such as a list or "upper and lower", or under more than one key) does not fail a requirement for one value when one of them is that value; and a key or a part of the name that lists variants or options to choose from (such as several sizes, lengths or colours on sale) states no value of the planned listing.
- rules: one entry per numbered basket rule, when the input has them. pass: true when the planned order keeps it, false when its data shows a breach, null when the data cannot tell. replace: when pass is false, the line numbers whose listing causes the breach; for a product that must work with or fit another, the line of the dependent product."""
    FIT = """You check which listings of one product of a shopper's order keep a rule that ties that product to another product of the same order. The input holds the rule as the shopper wrote it, the product the listings are for, the listings chosen for the other product (by id: its product, name, attributes, notes, and continued: the data of the listings its text sent the agent to for the rest of its list) and the candidate listings (name, attributes and notes, by id).
Return one JSON object with one entry per chosen listing: {"<chosen id>": {"applies": true, "codes": [], "more": null, "words": [], "fit": {"<candidate id>": "<quote>"}}}, written compactly on one line; leave out "codes", "more", "words", "needs" and "fit" when they would be empty or null. When a candidate's own data says that the rest of what it fits or works with is on another listing, part or page, also add the key "rest": {"<candidate id>": "<the part code or name its data gives for that listing, copied exactly>"}:
- applies: false only when the rule sets a requirement only if the chosen listing has some property and the chosen listing does not have it; else true.
- codes: when the rule asks that the product fits, suits or works with the other one: the model, series or part codes and names that the chosen listing's data (its own and its continued data) gives for the parts that fit or work with it, and the codes and names it gives for itself (under all its names), each copied exactly as written: a fitting candidate carries one of them in its name, attributes or notes, alone or as the first part of its own longer part code (a code starting with the same letters and digits marks a part of that series), or lists it among what it fits or works with. A code given for another part, or for the listing the rest of the list is on, is not one of them. [] for any other rule, or when the data gives no such codes.
- more: when the chosen listing's data says that the rest of its list is on another listing, part or page: {"for": "<what is found there, in a few words>", "id": "<that listing's id when given, copied exactly, else empty>", "seller": "<the shop or maker named for it, else empty>", "product": "<its product type when named, else empty>", "facts": {"<property>": "<value as written>"}}, where the facts hold what names that listing, such as its part code; null when nothing points further, or when the listing pointed to is already among continued.
- words: when the rule asks that the product fits, suits or works with the other one: the search words that listings of the product made for the chosen listing would carry in their own name or description, one entry per alternative, copied from the chosen listing's data: each series or part code it gives for the parts that fit it, else the type, standard, socket, connector or size both must state (not its own model name); [] for any other rule.
- needs: when the rule does not ask for a fit but sets the value some property of the product must have when the chosen listing has some property, and applies is true: that value as the rule words it, such as ["blue"] for "if the pump is electric, the filter must be blue"; else [].
- fit: when applies is true, every candidate that is that product and states what keeps the rule with the chosen listing, with the exact words of its data that state it: one of the codes, carried or listed as said above, or a model range or pattern that covers one of them (a code in place of its x or wildcard characters) and excepts none of them. A code the candidate names only as an exception does not count for it, and neither does a note that another part, named by its own part code, fits some models (such as "part <code> also fits: <models>"): that note is about the other part only. A candidate that names the models, series or parts it fits or works with keeps the rule only through those. A candidate that names none keeps it through a size, connector, standard or other property both listings state, or the requirement the rule sets. A candidate whose data does not state it is left out.
Example: the rule "Make sure the filter fits the pump you end up buying.", chosen A1 named "Garden Pump GP-20" with the note "Model no. QX41. Takes filters of the FL7 series.", and candidates 1 named "Mesh Filter FL7-3301", 2 with the note "Fits every QX4x model, except QX43.", 3 with "Fits QX52. Part FL9-100 also fits: QX41.", 4 with "Fits every QX4x model, except QX41." and 5 with "Works with QX52. See part FL9-200 for the rest." give {"A1":{"applies":true,"codes":["FL7","QX41","GP-20"],"words":["FL7"],"fit":{"1":"FL7-3301","2":"every QX4x model, except QX43"}},"rest":{"5":"FL9-200"}}: 1 carries the series FL7 as the first part of its code, 2 covers QX41 and excepts only QX43, 3 names QX41 only for part FL9-100, 4 excepts QX41 itself, and 5 does not state it (the rest of its list is on part FL9-200)."""
    PRICES = """You compare the money amounts of a shopper's request with an online store's prices. The input holds the request, the budget numbers the agent read from it with their unit, the store's search tools with their descriptions and schemas, the key that holds a listing's price, and some listings.
Return one JSON object: {"factor": 1}: the number that turns an amount in the budget's unit into the unit of the listing price numbers (the number in the price value, without any currency sign or word): 1 when both use the same unit, 0.01 when the budget is in hundredths of the currency and the listings in whole units, 100 for the opposite. Judge the listings' unit by the tool descriptions, and any currency sign or word in the price values; 1 when nothing shows another unit."""
    FIELDS = """You map the keys of an online store's product listing records to what they hold. The input holds a few listing records.
Return one JSON object {"price": "", "stock": "", "category": "", "brand": "", "title": "", "description": ""}: the key that holds the product's price, its stock status or count, its category, its brand or maker, its name and its free-text description, each written exactly as in the records; null for a key that no record has."""
    KEYWORDS = """You set up a keyword search for a shopping agent. The input holds the store's search tools with their JSON schemas.
Return one JSON object: {"tool": "<search tool>", "args": {...}}: a search across all products by free-text keywords that matches its tool's schema exactly, with the keyword argument set to the string "<TEXT>" and every other required argument set for a keyword search; no category, brand, price or page argument."""
    REFER = """You follow a reference for a shopping agent. The shopper wants the value of one property taken from one particular listing of the store: the wanted listing, given by its seller, product type and facts. The facts only tell which listing is meant; the property is the one whose value is wanted. The input holds the property, that description, and opened listings with all their data, including any free text.
Return one JSON object: {"checks": {}, "match": null, "redirect": null, "value": [], "pairs_with": []}
- checks: every opened listing's id mapped to one entry per given fact of the wanted listing (never the property asked about): {"<fact's property>": {"has": "<that listing's own value for it, as its data writes it, or empty>", "fits": true when that value is the given one (possibly in another wording, or in another unit that converts to it), false when its name, attributes or notes give another value for it or deny being the described one (such as notes saying it is not that colour or not that version), null when its data says nothing about it}}.
- match: the id of the opened listing that is the wanted one: from that seller, of that product type, no given fact false in its checks and the most of them true. When the wanted listing gives an id, the opened listing with that id is the wanted one. null when none fits.
- redirect: when the matched listing's text says to take this very property from another listing instead of itself: {"for": "<the property the text names for that pointer>", "id": "", "seller": "", "product": "", "facts": {"<property>": "<value>"}}: id is the listing id when the text gives one (copied exactly), else ""; the seller is the shop or maker name given for it (a name before the product type or in the possessive), and each fact is one property with its value as the text writes it, never the property asked about. A pointer for any other property does not count: then redirect is null and the matched listing's own data gives the value. Else null.
- value: the matched listing's own value of exactly this property, as its data writes it: an attribute, or its name or free text when they state it; never the value of another property; [] when it states none.
- pairs_with: the matched listing's notes may give a rule or list that pairs values of this property, each pair as one value going with, matching or combined with another (a list may name the link once and then give only the pairs). The values paired with the value above: for each pair that holds the value above, its other value. Else []."""
    PROFILE = """- search_profile holds what the agent measured of each search tool so far (a missing field is not measured yet): rows (rows per page), paging (page, offset or token), reach (the most rows one search can show; a search whose found count is above reach cannot be read to its end, so narrow it), chars (the longest keyword text), match ("all": a row holds every keyword, so each added word narrows the result; "any": a row may hold any one keyword, so an added word widens it and only a filter, a label or a more distinctive word narrows it; not measured: keep searches short), order per mode (id, price or relevance; a mode ordered by price lists the cheapest matches first), filters (ignored: that filter does not filter in that mode, so put a word of its label into the keywords instead) and cheapest_route (the strongest way measured to cover every product under a price: price order, price cap, shelf paging or entry queries)."""
    SEARCH = """You plan the first searches of a shopping agent. The input holds the shopping spec and the store's search tools with their JSON schemas.
Return one JSON object: {"searches": [{"tool": "<search tool>", "args": {...}}]}
- At most searches_allowed (a number in the input) searches, each matching its tool's schema exactly. Never set the argument that selects the result page: the agent pages on its own.
- Each search shows only a few rows at a time, so keep each search narrow: one stated value (a word, label or figure with its unit, exactly as the request writes it; a bound is searched by its written figure) followed by the spec's product word (never a word from the purpose or story), in as few plain words as hold them. A requirement that accepts any of several values gives each value its own search. Never search an excluded value, a figure the agent computed (from a reference, a tolerance or a budget) or a price. Most distinctive values first.
- Search the product word alone, or list a whole category, when the spec states no value to search. No prices, quotes or filler words in keywords.
- Use a category, brand or other label filter only with a label stated in the request. A mode that lists a whole category (as the tool description says) takes only the category label, no keywords.
- Every search carries keywords or a label; a price cap or a page alone finds nothing.
- When the objective is cheapest and a search tool takes an argument that sets the order of its results, set it to list the lowest price first.
""" + PROFILE
    LINKS = " Who it is for, who asked for it or who will wear or use it points to gender or age; its looks, being on show or seen, or what it sits beside or must match points to colour; feel or skin contact to material; the space it goes in to dimensions; which part, side or place it is for to position; power, plugs, leads or batteries to power source, plug or voltage; carrying, storing, filling or topping up to packaging, pack size or volume; what is inside or how it tastes to flavour or contents; files, use or installing to capacity or format; what went wrong before to the property that failed."
    QUESTION = """You plan what a shopping agent asks the shopper about one product of an order before buying. The shopper may hold needs about this product that the request does not state, and states such a need only when a question names that property, one property per question. Every hidden need must be found, so the agent asks the most likely ones first.
The input holds the shopper's full request, the product asked about now, the other products of the order, the requirements already known, the properties already asked about, and the candidate pool: properties found on opened listings of this product, each with the number of listings that carry it, example values, and share (the estimated share of the fitting listings that state it). Hidden needs are almost always properties that most listings state, whatever kind of property they are.
Return one JSON object: {"hinted": [{"property": "<property>", "words": "<the story's words>"}, ...], "core": ["<candidate property>", ...], "minor": ["<candidate property>", ...], "skip": ["<candidate property>", ...], "more": ["<property>", ...]}
- hinted: the properties the request's story points to for this product without stating their value. Go through the sentences that are not part of the product list, its requirements or the shopping rules, and for each one about this product or the whole order name the one property it points to most directly, with at most 8 of the sentence's words.""" + LINKS + """ A sentence that only sets the scene of the shopper's day or is about another product points to nothing, and a rule the request states (that it fits or works with another product, stock, price) is not a hint. Use the pool's label when the pool holds the property, else its plain name. In the story's order.
- core: the other candidates that are real properties a buyer may want one particular value of: the product's kind, type, style, form or series, its material, colour, pattern, finish, shape, closure, size, dimensions, capacity, volume, weight, pack size or packaging, who it is for, the part, side or vehicle it fits, and its specifications and features (a measure, rating, standard, format, function, connection or flavour). Most likely first: a property most listings carry and buyers of this kind of product choose by before one few listings carry.
- minor: the other candidates that only describe care, handling or delivery, which a shopper seldom picks.
- skip: candidates no shopper picks a value for: a model or product name, an id, code or part number, a field whose values list the variants on sale or join several facets in one value (such as a colour with a size), a certification or a warranty; a property whose value the request already states for this product, or that a rule of the request already covers; a second label for a property already asked or already listed (another spelling, a synonym or a narrower name of it).
- more: properties the pool lacks that listings of this kind of product name and its buyers typically choose by, not stated in the request, named the way listings name them, most likely first.
- Every candidate goes in exactly one of hinted, core, minor and skip, written exactly as in the pool. Values the request gives only to describe another listing it refers to (a page it points to) are not stated for this product. Hints and statements about another product of the order do not count."""
    VERIFY = """You check products for a shopping agent against the shopper's requirements, strictly and literally. The input holds the wanted product type, the numbered requirements, category values to judge, and products with their name, their attribute data (key: value) and the numbers of the requirements to check.
Return one JSON object:
{"categories": ["<category value>"],
 "products": {"<id>": {"<requirement number>": {"key": "<attribute key or null>", "pass": true, "loose": false, "quote": ""}}}}
- categories: when category values are listed, the values whose name matches the wanted product type: the one closest value, or one value per product type when the wanted type names alternatives ("X or Y"); every other, narrower, broader or related category fails. [] when none is listed or none matches.
- For each product and each requirement number listed for it:
  - key: the attribute key that names the requirement's property: the same name, possibly written another way or with a prefix or qualifier. Never a key about another property or another part of the product, even when its value holds the wanted word. When several keys name the property, use the one closest to the property name. When no attribute names the property but the category's name states the required value, use the key "category". When no attribute names the property but the product's notes state this product's own value of it (written as the property and its value, or as the models, parts or devices the product is compatible with, fits or works with, including a pattern or range of models that covers a listed value), use the key "notes" and judge that value. null when no key names it.
  - pass for required values: the key's value is one of the listed values: the same words apart from letter case, punctuation, spacing and word forms (singular, plural, -ed, -ing, -y: "printed" and "print" are one word), or with the property's or the product type's name added; a list value passes when one of its items does. It also passes when it holds a listed value as a whole phrase among other words or as one item of a list; a word inserted inside the phrase breaks it. It also passes when the value is the same value written in a short or long form: an abbreviation or initials, a shortened word, a hyphenated, joined or split spelling, or the same words with a filler word added. For a colour, it also passes when the value is another name for the same shade and keeps the listed colour's base colour word, but not when it names a lighter, darker or other shade or another colour; a value that holds a listed colour as a whole phrase still passes. A different word with a similar meaning, a broader value, a near value, or no key: false. Keys named by the property's own words, alone or with the product type's name, decide: it passes when any of them holds a listed value; a key with another qualifier counts only when no such key exists.
  - min, max, and values with a number: when the key's number has another unit than the requirement (such as mm, cm or inches), first convert it to the requirement's unit, then compare the numbers; a value equal to the bound after an exact conversion satisfies it; never round the converted number or the bound. A required value (not a bound) in another unit than the key's is often a rounded conversion: it passes when the key's converted number rounds to it at the precision the requirement writes it. A value that states a band of numbers (two numbers, or a number with words meaning and up or and under) gives the product the band's lowest number: compare that number with the bound.
  - exclude true: pass only when a key names the property and its value is none of the listed values; false when no key names the property, or when its value is one of them or holds one as a whole phrase.
  - The product name never makes a requirement pass. pass is false when another key that names the same property gives this product a different value of that property than the key you judged (for exclude: gives it a listed value); a key that lists variants or options to choose from does not count. The product name decides only when no key names the property, and then only when it gives a different value of that property outright, not a broader word for the same value or a word that may describe something else (such as a piece or pack count, a size or another part).
  - loose, when pass is false: true when the product still has what the requirement means, read with its plain meaning (a figure of speech or a comparison in the request means its plain property value), because one of these states it: another key whose name is a broader or related property and whose value holds the required value together with the requirement's own word (requirement "sugar: low" and "dietary needs: low sugar"), a key that holds the kind of property the value belongs to, the option list, or the product name; or because the named key holds the same value in another form: converted to another unit, the same words with filler or intensifier words dropped, every part of the value contained, a shade of the same base colour, or another name the trade uses for the very same value (a synonym, not another shade, grade, style or kind). For exclude true: loose is true when a key that holds the kind of property the excluded value belongs to gives another value, and nothing in the data gives an excluded value. loose is false when nothing in the data states the value, when any key or the product name gives a different value of the property, for a broader, narrower or near value, for a different word or technical term of related meaning, and for min or max requirements whose number the data does not give. quote: the exact text from the attribute data or the product name that shows it (for the key "notes": the note's words that state the value), else "".
  - notes, when a product has them, are its own free text. When a note says that the value of a requirement's property is to be taken from, read from or checked on another listing instead of this product, that requirement is judged by that other listing, never by this product's own data: give "source": {"seller": "<the seller, shop or maker named>", "product": "<that listing's product type>", "facts": {"<property>": "<value as written>"}} with one fact per detail the note gives about that listing, and pass null; when the note itself states that other listing's value of the requirement's property, judge the requirement by that value instead and give no source. When a note says this product is not, or does not have, a value of the property, that value fails for it; quote: the words of the note that say so.
- When the input names a measure, also give each product listed in "measure_for" a "measure": its number for that measure from the attribute data, in the unit of measure_like when that is given (convert another unit to it), or null."""
    CHANGE = """You read what the shopper or the store said during a shopping task and turn it into structured changes. The input holds the current requirement list, the budget as stated, the quantity, the products in the cart, the properties asked about, and the new lines: replies to questions, notices, or the reasons an order was refused.
Return one JSON object: {"changes": [{"type": "", "attribute": "", "values": [], "exclude": false, "min": null, "max": null, "unit": "", "number": null}]} with one entry per change:
- "need": the shopper states what they want or rule out for a property: attribute (the property as the line itself names it; a reply may name another property than the one asked about, and then the reply's name is the attribute, never the asked one), values (the value itself, a value that property of the product can have, without the property's name and without any comparison, simile or figure of speech around it; several only when the line joins them as alternatives; a word that addresses the reader, such as a term of endearment or a name, is never a value, even when the same word could name one), exclude true only when the shopper does not want those values; min, max and unit when the value is a numeric bound (at least gives min, at most gives max, with the unit as written). Read the meaning, not the form or the negative words: a value the shopper insists on, will not give up, wants kept or treated as settled, or will only accept is wanted (exclude false), also when the line comes as a question ("surely you see the strap is leather for me?"), a request ("please take the strap as decided on leather") or a comparison ("as a kettle needs water, the strap needs leather"); when the line speaks of everything other than a value, read what it does with those others: refused, thrown out or not to be shown ("any strap apart from leather goes straight back") means the value is wanted (exclude false), accepted ("any strap but leather will do") means it is ruled out (exclude true); the value is the words the line gives for the property, even words that could also mean that a value is missing.
- "no_preference": the shopper does not mind about a property, whatever else the line adds: attribute.
- A line that only asks the agent to rephrase, explain or repeat its message, or to ask its questions one at a time, in any wording and with or without a comparison, gives no entry. Words about something other than the order's products (the shopper's day, body or mood, other things) give no need, even when they use a property's name or a word that could be a value.
- "dropped": the shopper withdraws or loosens a requirement, in any wording: attribute as written in the requirement list.
- "changed": a requirement now has other values: attribute as written in the requirement list, and values.
- "budget": the line states a money amount the shopper now has or may spend, higher or lower than before, in any wording: number = the new amount in the unit of budget_unit in the input, converted only when the line states it in another unit of that currency; when budget_unit is empty, the amount as the line states it. Use the number stated, whatever the tone of the line.
- "quantity": number = the new quantity; "per_unit": true when the line keeps a money limit per piece (the amount per piece stays), else false.
- "out_of_stock": the line says a product in the cart, or the agent's pick, can no longer be bought (sold out, out of stock, the last one gone), in any wording, also when it asks about it or complains about it in a question: "id" = the id of that cart product when the line or its context says which one, else null.
- "price": the line says the price of a product is now different (higher, lower, or only changed), in any wording, also when it gives no new number: number = the new price when the line states it, else null. A price change is not a stock problem: the product can still be bought.
- "reference": instead of naming the value, the line points to what the page of another listing of the store says (directly or through a listing that page names): attribute (the property the line itself names as the one it constrains, also when another was asked about), "relation": "same" when the value must be the one that page gives or one that matches, fits or works with it, "different" when the shopper accepts anything except that page's value (anything but, bar, besides or other than it), "at_least" or "at_most" when it must be at least or at most that page's value, "within" with "margin" and unit when it may differ by up to some amount; and "listing": {"seller": "<the seller, shop or maker named>", "product": "<that listing's product type, as the line names it>", "facts": {"<property>": "<value as written>"}} with one fact per detail the line gives about that listing (every property and value it names for it, such as the ones after "with"). The relation comes from the words that point to the page, whatever the words around them (insisting on anything but that page's value is "different"). The property then gets this one entry only: never a "need" with the reference wording as its value, and the details of that listing go only in its facts: never in another entry.
- "other": anything else that matters: values = [a short summary].
A line that begins with a product id and a colon is what the store's page or cart line for that product newly shows: read it as a notice about that product and give that id as "id". Text that only describes the product gives no entry.
One line may hold several changes: give each its own entry. A line says one thing about each property of each product: one entry per property and product, holding what the whole line means for it; words that only stress, repeat or decorate it (that the shopper will take nothing else, that it is fixed, settled or firm, or a comparison) add no entry and no value of their own, and the value the line names keeps its entry. Lines that change nothing give no entry.
When the input lists the products of the order, each entry also gives "product": the product of the order that the line names for this change, also when it is not the one asked about (a line about "the color of the X" is about X); empty when the line names none. The product type of a listing a reference points to is not the product of the change.
Every entry also gives "quote": the words of the new line that state the change, copied exactly. A change no new line states gives no entry."""
    STOCK = """You read the stock values of an online store's product listings. The input holds the key that holds each listing's stock or availability and the distinct values seen under it.
Return one JSON object with every value, written exactly as given, as a key: true when a product showing that value can be bought now, false when it cannot, null when the value does not say."""
    UNITS = """You convert measured values for a shopping agent. The input holds values as written in product data, each with the unit to express it in.
Return one JSON object: {"numbers": [<number or null>, ...]} with one entry per value, in input order. Give the value's number converted exactly to the target unit (such as cm to inches: divide by 2.54); keep all decimals, never round. When the value already is in the target unit written another way (singular or plural, short or long form), give its number unchanged. A value that states a band or several numbers gives its lowest number. null when the value holds no number, or when its unit measures another kind of quantity than the target unit."""
    RECOVER = """You help a shopping agent that works through the tools of an online store when one of its steps got a result it could not use. The input holds the step, its goal, the call the agent made, the raw result of that call as the store returned it (cut off when long), the store's tools (name, what each does, its arguments) and the calls the agent already tried for this goal; rejected, when given, is an earlier answer that did not fit the result or the tools, with the reason why.
Find why the result does not serve the goal and the one next action that will. Return one JSON object: {"reason": "<one short sentence>", "action": "", "mapping": {"path": [], "records": null, "format": "", "split": "", "pair": "", "fields": [{"name": "", "split": null, "pair": null}], "header": false, "keyed": false, "id": "", "price": "", "stock": "", "title": ""}, "call": {"tool": "", "args": {}}}. action is one of:
- map: the result holds product records (or one record) written in a form the agent could not read, such as records written as text instead of JSON objects, or records nested under other keys or beside notices; choose map for such a result even when its records show less than the goal hopes for, since the agent reads them before it plans its next call. mapping tells the agent's code how to read this result and every later result of the same shape, whatever its records hold: path, the keys and list positions that lead from the top of the result to the records: a list with one record per item, an object whose values are the records, or a text that holds one record or several ([] when the result itself is it); records, the exact characters between one record and the next when one text holds several, else null; format, "pairs" when a record writes each field as its name, fixed characters and its value, "columns" when a record tells its fields apart only by their position, "json" when each record is a JSON object or JSON text; split, the exact characters between one field and the next inside a record; pair, for pairs, the exact characters between a field's name and its value; fields, for pairs every field any record uses, for columns one per position in order: name, the field's name exactly as written, and, when its value is itself a list of names with values (such as "color: red, size: M" or "width=2;depth=3"), split, the exact characters between the value's entries, and pair, the exact characters between each name and its value (else null); header, true when the first record is a row that names the columns; keyed, true when the records are the values of an object whose keys are the product ids; id, the field that holds the product id, the code the store's other tools take to open, compare or buy that product; price, stock and title, the fields that hold the price, the stock or availability and the product's name, or null. When the result holds no product record at all (only notices, an error or an empty list), choose skip.
- call: another call (another tool, or the same tool with other arguments) will reach the goal; call gives the tool's name and its arguments as the tool takes them. retry: the same call made again will likely work, because the result shows a passing failure. skip: no call can reach this goal now; the agent moves on without it. finish: the agent should end this step and finish the task with what it has.
Use only tools and arguments listed in the input, and never a call listed in tried. Leave the fields the action does not use null."""
    RECORDS = """You read a result that a tool of an online store returned to a shopping agent and that the agent's code could not read. The input holds the tool's name and the result as the store returned it (cut off when long).
Return one JSON object: {"id": "<the field that holds the product id, the code the store's other tools take to open, compare or buy the product>", "records": [{"<field>": "<value>"}]} with one object per product record in the result, in order, holding every field the record gives, named and valued exactly as the record writes them (numbers as JSON numbers; true, false and null as JSON); a field whose value holds its own names and values becomes an object of them. "records": [] when the result holds no product record."""


class Util:
    def field(key):
        return property(lambda self: self.items[self.at][key], lambda self, value: self.items[self.at].__setitem__(key, value))
    def number(value):
        if isinstance(value, bool) or value is None: return None
        if not isinstance(value, (int, float)):
            found = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
            value = float(found.group()) if found else None
        return value if value is not None and abs(value) < 1e15 else None
    def kind(schema):
        kind = schema.get("type") if isinstance(schema, dict) else None
        if not kind and isinstance(schema, dict): kind = [Util.kind(s) for s in schema.get("anyOf") or schema.get("oneOf") or []]
        return next((k for k in kind if k and k != "null"), None) if isinstance(kind, list) else kind
    def fit(schema, value):
        kind = Util.kind(schema)
        if "enum" in schema:
            hits = [o for o in schema["enum"] if o == value or str(o).lower() == str(value).lower()]
            return hits[0] if hits else None
        if kind == "object":
            if not isinstance(value, dict): return None
            props, out = schema.get("properties", {}), {}
            for key, item in value.items():
                if key in props:
                    item = Util.fit(props[key], item)
                    if item is not None: out[key] = item
                elif schema.get("additionalProperties", True) is not False: out[key] = item
            return out if all(k in out for k in schema.get("required", [])) else None
        if kind == "array":
            items = [Util.fit(schema.get("items", {}), v) for v in (value if isinstance(value, list) else [value])]
            items = [v for v in items if v is not None][: schema.get("maxItems", 1000)]
            return items if len(items) >= schema.get("minItems", 0) else None
        if kind in ("integer", "number"):
            found = Util.number(value)
            if found is None: return None
            found = min(max(found, schema.get("minimum", found)), schema.get("maximum", found))
            return int(found) if kind == "integer" or found == int(found) else found
        if kind == "boolean": return value if isinstance(value, bool) else str(value).strip().lower() == "true"
        if kind == "string":
            if value is None or isinstance(value, (dict, list)): return None
            value = str(value)[: schema.get("maxLength", 100000)]
            if len(value) < schema.get("minLength", 0) or ("pattern" in schema and not re.search(schema["pattern"], value)): return None
            return value
        return value
    def fill(node, pid, qty, text, lines=None):
        if isinstance(node, dict): return {k: Util.fill(v, pid, qty, text, lines) for k, v in node.items()}
        if isinstance(node, list): return [Util.fill(v, p, q, text) for p, q in (lines.items() if isinstance(lines, dict) else lines) for _ in range(1 if "<QTY>" in json.dumps(node) else q) for v in node] if lines is not None and Util.slots(node) else [Util.fill(v, pid, qty, text, lines) for v in node]
        if node == "<QTY>": return qty
        if node == "<ID>" and pid is not None: return pid
        if isinstance(node, str): return node.replace("<ID>", "" if pid is None else str(pid)).replace("<TEXT>", text or "").replace("<QTY>", str(qty))
        return node
    def slots(node):
        if isinstance(node, list): return "<ID>" in json.dumps(node)
        return isinstance(node, dict) and any(Util.slots(v) for v in node.values())
    def dicts(node):
        return [node] + [d for v in node.values() for d in Util.dicts(v)] if isinstance(node, dict) else [d for v in node for d in Util.dicts(v)] if isinstance(node, list) else []
    def id_key(node, names=()):
        stack = [node]
        while stack:
            item = stack.pop()
            if isinstance(item, dict): stack.extend(item.values())
            elif isinstance(item, list):
                rows = [r for r in item if isinstance(r, dict)]
                stack.extend(item)
                keys = [k for k in (rows[0] if rows else []) if all(Util.ident(r.get(k)) for r in rows) and len({str(r[k]) for r in rows}) == len(rows)]
                if keys:
                    fraction = lambda k: any(Util.number(r[k]) is not None and Util.number(r[k]) % 1 for r in rows)
                    return min(keys, key=lambda k: (Util.flat(k) not in names, fraction(k), "id" not in Util.keyname(k).split() and not Util.flat(k).endswith("id"), not all(re.search(r"\d", str(r[k])) for r in rows), isinstance(rows[0][k], int)))
        return None
    def ident(value):
        if isinstance(value, bool): return False
        return isinstance(value, int) or isinstance(value, str) and bool(re.fullmatch(r"\S{3,80}", value))
    def ids(node, key):
        for item in Util.dicts(node) if key else []:
            if Util.ident(item.get(key)) and isinstance(item[key], int): item[key] = str(item[key])
        return node
    def scalar(text):
        try: out = {k: v for v in (True, False, None) for k in (repr(v), json.dumps(v))}[text] if text.isidentifier() else json.loads(text)
        except (KeyError, ValueError): return text
        return text if isinstance(out, str) or isinstance(out, float) and not math.isfinite(out) else out
    def sig(node):
        return {k: Util.sig(v) for k, v in node.items()} if isinstance(node, dict) else [Util.sig(v) for v in node[:1]] if isinstance(node, list) else "text" if isinstance(node, str) else "value"
    def unescape(text):
        try: return json.loads('"%s"' % text) if "\\" in text else text
        except ValueError: return text
    def walk(node, path):
        for key in path: node = node.get(key) if isinstance(node, dict) else node[int(key)] if isinstance(node, list) and str(key).lstrip("-").isdigit() and -len(node) <= int(key) < len(node) else None
        return node
    def put(node, path, value):
        return value if not path else {**node, path[0]: Util.put(node[path[0]], path[1:], value)} if isinstance(node, dict) else [Util.put(v, path[1:], value) if i == int(path[0]) % len(node) else v for i, v in enumerate(node)]
    def nodes(node, path=()):
        return [list(path)] + ([p for k, v in node.items() for p in Util.nodes(v, path + (k,))] if isinstance(node, dict) else [])
    def pairs(parts, split, pair, names=None):
        out, key = {}, None
        for part in parts:
            name, has, value = part.partition(pair.strip() or pair)
            if has and name.strip() not in out and (names is None or name.strip() in names): key, out[name.strip()] = name.strip(), value
            elif key: out[key] += split + part
            else: return {}
        return out
    def cells(text, how):
        return [c.strip() for c in text.strip().strip(how["split"].strip()).split(how["split"])] if how["format"] == "columns" else text.split(how["split"])
    def row(item, how, names):
        if not isinstance(item, str) or how["format"] == "json": return dict(item) if isinstance(item := Util.scalar(item.strip()) if isinstance(item, str) else item, dict) else {}
        parts = Util.cells(item, how)
        out = Util.pairs(parts, how["split"], how["pair"], names) if how["format"] == "pairs" else dict(zip(names, parts)) if len(parts) == len(names) else {}
        return {k: Util.value(v, how["nested"].get(k)) for k, v in out.items()}
    def value(text, how):
        inner = Util.pairs(text.split(how["split"]), how["split"], how["pair"]) if how else {}
        return {k: Util.scalar(v.strip()) for k, v in inner.items()} if inner else Util.scalar(text.strip())
    def lay(node, how, pid):
        at, rows, lost, names = Util.walk(node, how["path"]), [], 0, how["fields"]
        items = list(at.items()) if how["keyed"] and isinstance(at, dict) else [(None, v) for v in at] if isinstance(at, list) else [(None, v) for v in (at.split(how["records"]) if how["records"] else [at])] if isinstance(at, str) else []
        for n, (key, item) in enumerate(items):
            if how["header"] and not n and isinstance(item, str): names = Util.cells(item, how); continue
            row = Util.row(item, how, names)
            if row and (key is not None or pid and len(items) == 1): row.setdefault(how["id"], key if key is not None else pid)
            if Util.ident(row.get(how["id"])): rows.append(row)
            else: lost += isinstance(item, str) and bool(item.strip()) and (how["format"] == "json" or any(c and c in item for c in (how["split"], how["pair"])))
        return rows, lost, at
    def most(node, schema, edge="maximum"):
        if node == "<QTY>": return Util.number(schema.get(edge))
        children = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else []
        for key, value in children:
            found = Util.most(value, schema.get("properties", {}).get(key, {}) if isinstance(node, dict) else schema.get("items", {}), edge)
            if found is not None: return found
        return None
    def texts(node, idk, loose, listed=False):
        if isinstance(node, str): return [node] if (loose or listed) and " " in node.strip() else []
        if isinstance(node, list): return [t for v in node for t in Util.texts(v, idk, loose is True, True)]
        if isinstance(node, dict) and not (idk and isinstance(node.get(idk), str)): return [t for v in node.values() for t in Util.texts(v, idk, loose)]
        return []
    def clip(node, size=160):
        if isinstance(node, str): return node if len(node) <= size else node[:size] + "..."
        if isinstance(node, list): return [Util.clip(v, size) for v in node[:12]]
        if isinstance(node, dict): return {k: Util.clip(v, size) for k, v in node.items() if v not in (None, "", [], {})}
        return node
    def attrs(node, out):
        for key, value in node.items():
            if isinstance(value, dict): Util.attrs(value, out)
            elif isinstance(value, list):
                if value and all(not isinstance(v, (dict, list)) for v in value): out.setdefault(key, ", ".join(map(str, value)))
                pairs = [list(v.values()) for v in value if isinstance(v, dict)]
                if pairs and len(pairs) == len(value) and all(len(p) > 1 and isinstance(p[0], str) and all(not isinstance(x, (dict, list)) for x in p) for p in pairs):
                    for name, *rest in pairs: Util.attrs({name: " ".join(str(x) for x in rest if x not in (None, ""))}, out)
            elif value not in (None, "") and len(str(value)) <= 150: out.setdefault(key, str(value))
        return out
    def parse(reply):
        try:
            text = reply["choices"][0]["message"]["content"] or ""
            out = json.JSONDecoder().raw_decode(text[text.index("{"):])[0]
        except (TypeError, KeyError, IndexError, ValueError, AttributeError): return None
        return out if isinstance(out, dict) else None
    def items(value):
        return list(value) if isinstance(value, (list, dict, str)) else []
    def norm(text):
        return " ".join(re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split())
    def flat(text):
        return Util.norm(text).replace(" ", "")
    def keyname(key):
        return Util.norm(re.sub(r"([a-z])([A-Z])", r"\1 \2", key))
    def figure(text):
        found = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z%]*)", re.sub(r"[\s,]", "", str(text).lower()))
        return (float(found.group(1)), found.group(2)) if found else None
    def grounded(text, have):
        words, flat, roots = Util.norm(text).split(), "".join(have), {Util.root(w) for w in have}
        whole, ends = "".join(words), {0, *itertools.accumulate(map(len, have))}
        return any(flat.startswith(whole, s) and s + len(whole) in ends for s in ends) or all(Util.root(w) in roots for w in words)
    def without(have, parts):
        have = list(have)
        for cut in filter(None, (Util.norm(p).split() for p in parts)):
            for i in range(len(have) - len(cut) + 1):
                if have[i:i + len(cut)] == cut: have[i:i + len(cut)] = ["\0"] * len(cut)
        return have
    def spans(words, value):
        value = Util.norm(value).split()
        return [(i, i + len(value)) for i in range(len(words) - len(value) + 1) if value and words[i:i + len(value)] == value]
    def joined(values, text):
        words = Util.norm(text).split()
        spans = [(Util.spans(words, v) or [None])[0] for v in values]
        if None in spans or len(spans) < 2: return values
        low, high = min(a for a, _ in spans), max(b for _, b in spans)
        return values if set(range(low, high)) - {i for a, b in spans for i in range(a, b)} else [" ".join(words[low:high])]
    def stems(text):
        return {w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w for w in Util.norm(text).split()}
    def overlaps(one, other):
        a, b = Util.stems(one), Util.stems(other)
        return bool(a and b) and (a <= b or b <= a)
    def same(label, name):
        a, b = Util.stems(label), Util.stems(name)
        return bool(a) and a <= b and all(len(w) <= 2 for w in b - a)
    def alike(value, wanted, name):
        return Util.flat(value) == Util.flat(wanted) or (Util.stems(value) - Util.stems(name) or Util.stems(value)) == (Util.stems(wanted) - Util.stems(name) or Util.stems(wanted))
    def root(word):
        if word.endswith("ss"): return word
        return next((word[:-len(s)] for s in ("ies", "ied", "ing", "ed", "es", "s", "y", "d", "e") if word.endswith(s) and len(word) - len(s) >= 4), word)
    def holds(value, wanted):
        return (lambda have, want: bool(want) and any(have[i:i + len(want)] == want for i in range(len(have) - len(want) + 1)))(*([Util.root(w) for w in Util.norm(t).split()] for t in (value, wanted)))
    def spelled(text, ref):
        return {w if w in ref else next((r for r in ref if min(len(w), len(r)) > 4 and len(w) + len(r) & 1 and any(w[:i] + w[i + 1:] == r or r[:i] + r[i + 1:] == w for i in range(len(w) + 1))), w) for w in Util.stems(text)}
    def related(value, wanted, name):
        have, want = ({w for i, x in enumerate(words) for w in (x, "".join(words[i:i + 2]))} for words in ([w for w in Util.norm(t).split() if w not in Util.norm(name).split()] for t in (value, wanted)))
        return Util.flat(value) == Util.flat(wanted) or any(a == b or len(a) > 4 and len(b) > 4 and a[:5] == b[:5] for a in have for b in want)
    def variant(value, wanted, name):
        if Util.holds(value, wanted): return 2 * ("+" in set(str(value)) - set(str(wanted)))
        skip, root, parts = set(Util.norm(name).split()), Util.root, [Util.norm(c).split() for c in str(wanted).split()]
        want, tight = [w for p in parts for w in p if w not in skip], {(root(a), root(b)) for p in parts for a, b in zip(p, p[1:])}
        have = (lambda words: [w for i, w in enumerate(words) if not 0 < i < len(words) - 1 or (root(words[i - 1]), root(words[i + 1])) not in tight])([w for w in Util.norm(value).split() if w not in skip])
        roots, near = [root(w) for w in have], lambda a, b: Util.related(a, b, "") or len(a) == len(b) > 3 and sum(x != y for x, y in zip(a, b)) == 1 or any(len(x) > 1 and x[0] == y[0] and (lambda rest: all(c in rest for c in x))(iter(y)) for x, y in ((a, b), (b, a)))
        if any(path[-1] < len(roots) and path[-1] - path[0] >= len(want) for start, w in enumerate(roots) if len(want) > 1 and w == root(want[0]) for path in [list(itertools.accumulate(want[1:], lambda at, x: next((j for j in range(at + 1, min(at + 4, len(roots))) if roots[j] == root(x)), len(roots)), initial=start))]): return 2
        return int(len(have) == len(want) > 1 and any(near(a, b) and not a.isdigit() for a in have for b in want) and not all(any(near(a, b) for b in want) for a in have))
    def ranged(text, code):
        text, code, head = Util.flat(text), Util.flat(code), next((i for i, c in enumerate(Util.flat(code)) if not c.isalpha()), 0)
        spots = {text[i:i + len(code)] for i in range(len(text)) if text.startswith(code[:head], i)} if head and code[head:].isdigit() else set()
        covered = [all(a in (b, "x") for a, b in zip(s, code)) for s in spots if len(s) == len(code) and "x" in s[head:] and all(c.isdigit() or c == "x" for c in s[head:])]
        return None if not covered or code in spots and not any(covered) else any(covered) and code not in spots
    def meets(value, req, product):
        if req["min"] is not None or req["max"] is not None:
            found = Util.figure(value)
            return None if not found or found[1] != Util.flat(req["unit"]) else (req["min"] is None or found[0] >= req["min"]) and (req["max"] is None or found[0] <= req["max"])
        name, verdicts = req["attribute"] + " " + product, []
        for item, wanted in itertools.product([value] + (value.split(", ") if ", " in value else []), req["values"]):
            a, b, shape = Util.figure(item), Util.figure(wanted), Util.variant(item, wanted, name)
            verdicts.append((a[0] == b[0] if a[1] == b[1] else None) if a and b else True if Util.alike(item, wanted, name) or Util.holds(item, wanted) and (req["exclude"] or shape < 2) else False if (not Util.related(item, wanted, req["attribute"]) if req["exclude"] else shape == 2 and Util.holds(item, wanted)) else None)
        return True if True in verdicts else False if verdicts and None not in verdicts else None
    def singular(label):
        *head, last = Util.norm(label).split() or [""]
        cut = 2 if len(last) > 4 and last.endswith(("ches", "shes", "sses", "xes")) else 1 if len(last) > 3 and last.endswith("s") and not last.endswith(("ss", "us", "is", "ies")) else 0
        return " ".join(head + [last[:len(last) - cut]])
    def requirement(item):
        if not isinstance(item, dict): return None
        attribute = str(item.get("attribute") or "").strip()
        values = item.get("values") if isinstance(item.get("values"), list) else [item.get("value")]
        values = [str(v).strip() for v in values if isinstance(v, (str, int, float)) and str(v).strip()]
        low, high = Util.number(item.get("min")), Util.number(item.get("max"))
        if not attribute or not values and low is None and high is None: return None
        return {"attribute": attribute, "values": values, "exclude": item.get("exclude") is True, "min": low, "max": high, "unit": str(item.get("unit") or "").strip()}
    def listed(value):
        return [str(v).strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip()]
    def bare(key, value):
        words, own = " ".join(Util.listed(value)).split(), set(Util.norm(key).split())
        while len(words) > 1 and set(Util.norm(words[-1]).split()) <= own: words.pop()
        return " ".join(words)
    def described(listing):
        facts = listing.get("facts")
        return Util.listed(list(facts.values()) if isinstance(facts, dict) else facts)
    def sign(req):
        return json.dumps([Util.norm(req["attribute"]), sorted(map(Util.norm, req["values"])), req["exclude"], req["min"], req["max"], Util.flat(req["unit"])])


class Core:
    def __init__(self, problem):
        env = problem["environment"]
        view = env["policy_view"]
        self.binding, self.query, self.opening = env["binding"], view.get("query") or "", view.get("opening")
        self.tools = {}
        for tool in view.get("tools") or []:
            spec = tool.get("function", tool)
            self.tools[spec["name"]] = {"description": spec.get("description", ""), "parameters": spec.get("parameters") or {"type": "object", "properties": {}}}
        self.max_steps = self.hard = int(view.get("max_steps") or LOOP)
        self.view_limits = {k: v for k, v in view.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self.finished = bool(view.get("finished"))
        spread = sum(map(ord, json.dumps(self.binding, sort_keys=True, default=str))) % 100 / 100
        self.llm = ProxyClient(timeout=CALL_TIMEOUT, max_retries=1, retry_delay=1 + spread, rate_limit_retry_delay=2 + 3 * spread)
        self.env, self.lat, self.slow = ProxyClient(timeout=ENV_TIMEOUT, max_retries=3), {}, 0.0
        self.start, self.step, self.dead, self.tries, self.seconds, self.remapped = time.monotonic(), 0, False, 0, None, False
        self.cost = {"act": [0, 0.0], "model": [0, 0.0]}
        self.roles, self.search_tools, self.products, self.cart, self.fields, self.cached = {}, {}, {}, {}, {}, {}
        self.judged, self.decided, self.version = {}, {"code": 0, "model": 0}, 0
        self.pending, self.trace, self.heard, self.hearing, self.gone, self.ignored = [], [], set(), [], set(), set()
        self.idk, self.last, self.page_size, self.refs, self.job, self.sources, self.inherited = None, {}, 1, [], None, {}, {}
        self.purpose, self.unit, self.scale = "", "", 1
        self.items, self.at, self.total, self.spent, self.stop, self.questions, self.allowed = [], None, None, 0, None, 0, 0
        self.shapes, self.faults, self.shifted, self.readings, self.rules, self.said, self.verdicts, self.permits, self.changes, self.basket_rules, self.vetoed = {}, set(), set(), {}, [], [], {}, {}, 0, [], set()
        self.assigned, self.replaces, self.audited, self.lean, self.sets, self.looked, self.retried, self.blind, self.trusted = {}, True, {}, False, None, None, set(), False, False
        self.touched, self.refused, self.mend, self.closing, self.failed, self.read_at, self.widened, self.rescued = set(), {}, 1, False, set(), {}, set(), set()
        self.links, self.fitted, self.unfit, self.full, self.dual, self.read_by, self.raws = [], {}, None, set(), None, {}, []
        self.in_stock, self.priced, self.memo, self.texts, self.blank, self.plans, self.consulted, self.batch, self.moments, self.gating, self.placed, self.urgent, self.sight = True, False, (None, False), {}, set(), {}, False, None, set(), False, {}, False, {}
        self.mends, self.mended = {}, 0
        self.world = {"rows": None, "paging": None, "reach": None, "chars": None, "words": None, "order": {}, "filters": {}}
        self.readers = {}
    def reserve(self, least=False):
        qty, add = [s["qty"] for s in self.items] or [1], self.roles.get("add")
        slots = bool(add) and Util.slots(add["args"])
        adds = [1 if slots else len(self.chunks(q)) if add else 0 for q in qty]
        left = [n for i, n in enumerate(adds) if self.assigned.get(i) not in self.cart]
        cart, tail, again = max(left, default=0) if slots else sum(left), self.owed() + (not self.refused), bool(self.retried)
        if least or self.urgent and not self.unasked(): return cart + 1 + again + tail
        cap, swap, look = max(cart + 1, self.hard - 2 * len(qty)), max(adds) + ("remove" in self.roles and bool(add) and not slots), int("review" in self.roles and bool(add))
        view = int(bool({"view", "details"} & set(self.roles)))
        find = view * (len(qty) > 1)
        if self.lean: return min((max(adds) if slots else sum(adds)) + look + 1 + again + swap + find + 1, cap) + tail
        if self.tries: return min(look + swap + find + view * (2 - self.backed()) + (self.objective != "any") + 1, cap) + tail
        return min(cart + sum(len(self.unread(p)) for p in self.cart) + view * len(left) + look + 1 + again + swap, cap) + tail
    def backed(self):
        key = (self.step, self.at, self.version, len(self.details))
        if self.memo[0] != key: self.memo = (key, sum(map(self.passes, list(self.details))) > 1)
        return self.memo[1]
    def unasked(self):
        if self.consulted or "message" not in self.roles or self.allowed == 0: return 0
        return len(self.items) if self.allowed is None else max(0, min(self.left(), sum(map(len, self.plans.values())) if self.plans else math.inf))
    def left(self):
        return math.inf if self.allowed is None else self.allowed - self.sent() - self.reserved()
    def elapsed(self):
        return time.monotonic() - self.start
    def spend(self, began, *kinds):
        seconds = time.monotonic() - began
        if "act" in kinds: self.slow = max(self.slow, seconds)
        for kind in kinds:
            count, total = self.cost.get(kind, (0, 0.0))
            self.cost[kind] = [count + 1, total + seconds]
        return seconds
    def pace(self, kind):
        count, seconds = self.cost[kind]
        return seconds / count if count else 0.0
    def deadline(self):
        seen = [s for v in self.lat.values() for s in v]
        return (self.seconds or PLATFORM) - (max(seen + [self.slow]) if seen else CALL_TIMEOUT)
    def late(self, calls=0):
        done = [self.cost[r] for r in ("add", "view", "details", "review", "checkout") if r in self.cost]
        count, seconds = sum(c for c, _ in done), sum(t for _, t in done)
        each, steps = (seconds / count, self.reserve()) if count else (self.pace("act"), self.reserve(True))
        return self.elapsed() + steps * each + (calls + 1 + (not self.tries)) * self.pace("model") > self.deadline()
    def room(self, step=None):
        room = min(self.hard - self.reserve() - self.unasked(), self.stop if self.stop is not None else self.hard) - self.step
        return room if step is None else min(room, self.plan().get(step, math.inf) - self.used(step))
    def busy(self):
        return self.room() <= 0 or self.late() or self.finished or self.dead
    def allowance(self, step=None):
        return max(1, self.room(step) // 2)
    def plan(self):
        steps = list(dict.fromkeys(STEP.values()))
        used, share = {s: self.used(s) for s in steps}, {s: math.ceil(self.hard * TURNS[s] / 100) if s in TURNS else self.allowed if self.allowed is not None else self.sent() + self.unasked() for s in steps}
        late, spare = max(0, share["cart"] + share["final"] - used["cart"] - used["final"] - self.reserve(True)), {s: max(0, share[s] - used[s]) for s in steps}
        return {s: share[s] + (s in TURNS) * (sum(spare[t] for t in steps[:i]) + max(late * (i < 2), int(sum(spare[t] * min(1, used[t] / max(1, share[t])) for t in steps[i + 1:] if t in TURNS)))) for i, s in enumerate(steps)}
    def used(self, step):
        return sum(STEP.get(t.get("role")) == step for t in self.trace)
    def over(self, step):
        return self.used(step) > self.plan().get(step, math.inf)
    def recover(self, step, goal, raw, tried=(), pid=None):
        last, sign, tried, reading = next((t for t in reversed(self.trace) if "tool" in t), {}), lambda v: json.dumps(v, sort_keys=True, default=str), list(tried), step == "read-shape"
        if (key := sign([step, last.get("tool"), Util.sig(raw)])) in self.mends and sign(self.mends[key]["call"]) not in map(sign, tried): return self.mends[key]
        data, out = {"step": step, "goal": goal, "call": {k: last.get(k) for k in ("tool", "args")}, "result": (raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str))[:RAW], "tools": {n: {"does": Util.clip(t["description"], 200), "args": {k: v.get("enum") or Util.kind(v) for k, v in self.props(n).items()}, "required": t["parameters"].get("required", [])} for n, t in self.tools.items()}, "tried": [Util.clip(t, 300) for t in tried[-8:]]}, None
        for name in (LUNA, SOL)[:2 if reading else max(0, math.ceil(self.hard * MENDS / 100) - self.mended)]:
            self.mended += not reading
            out = self.model(Prompts.RECOVER, {**data, **({"rejected": out} if out else {})}, 500, name) or {}
            _, m, why = self.laid(out.get("mapping"), raw, pid)
            call = out.get("call") if isinstance(out.get("call"), dict) else {}
            tool = call.get("tool") if call.get("tool") in self.tools and isinstance(call.get("args"), dict) else None
            args = Util.fit(self.tools[tool]["parameters"], {k: v for k, v in call["args"].items() if k in self.props(tool)}) if tool else None
            call, action = {"tool": tool, "args": args} if args is not None and sign({"tool": tool, "args": args}) not in map(sign, tried) else {}, out.get("action")
            if action in ("retry", "skip", "finish") or action == "map" and not why or action == "call" and call:
                fix = {"reason": str(out.get("reason") or "")[:300], "action": action, "mapping": m if action == "map" else {}, "call": call or ({"tool": last["tool"], "args": last["args"]} if action == "retry" and last else {})}
                if not reading: self.mends[key] = fix
                return fix
            out = {**out, "why": why if action == "map" else "its action or call does not fit the tools or was tried before"}
        return {"reason": str((out or {}).get("reason") or "")[:300], "action": "fail", "mapping": {}, "call": {}}
    def model(self, prompt, data, tokens, name=LUNA):
        body = {"messages": [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps(data, ensure_ascii=False)}], "temperature": 0, "max_tokens": tokens, "reasoning": {"enabled": False}}
        for name in dict.fromkeys((name, BACKUP[name], BACKUP[BACKUP[name]])):
            left, seen = self.deadline() - self.elapsed(), sorted(self.lat.get(name, []))
            if left <= (seen[len(seen) // 2] if seen else 0): return None
            wait = min(CALL_TIMEOUT, left / 2)
            self.llm.timeout = wait
            self.llm.max_retries = next(k for k in range(RETRIES, 0, -1) if k == 1 or k * wait + 10 * (k - 1) <= left)
            began = time.monotonic()
            raw = self.llm.post(CHAT, json_data={**body, "model": name})
            out, seconds = Util.parse(raw), self.spend(began, "model")
            self.trace.append({"model": name, "seconds": round(seconds, 1), "ok": out is not None})
            if out is not None:
                self.lat.setdefault(name, []).append(seconds)
                return out
            if raw is not None: body = {**body, "messages": body["messages"] + [{"role": "user", "content": "Reply with the JSON object only."}]}
        return None
    def act(self, role, pid=None, qty=None, text=None, tool=None, args=None, final=False, again=False):
        if self.finished or self.dead or self.step >= (self.hard if final else self.hard - self.reserve()): return None
        if role in self.moments and not self.gating: self.gate(dict(self.cart), role)
        if tool is None:
            entry = self.roles.get(role)
            if not entry: return None
            tool, args = entry["tool"], Util.fill(entry["args"], pid, qty, text)
        args = Util.fit(self.tools[tool]["parameters"], args or {})
        sign = json.dumps([tool, args], sort_keys=True, default=str)
        if args is None or len(json.dumps(args)) > 4000 or sign in self.failed: return None
        self.step += 1
        payload = {**self.binding, "step": self.step, "action": {"name": tool, "args": args}}
        began = time.monotonic()
        self.env.timeout = max(1, min(ENV_TIMEOUT, ((self.seconds or PLATFORM) - self.elapsed() - 10) / 3))
        reply = self.env.post(CALL, json_data=payload)
        for _ in range(LOOP):
            if reply is not None or self.elapsed() >= self.deadline(): break
            reply = self.env.post(CALL, json_data=payload)
        self.spend(began, "act", role)
        if not isinstance(reply, dict):
            self.dead = True
            return None
        self.finished = bool(reply.get("finished"))
        obs = reply.get("observation")
        self.raws.append({"role": role, "tool": tool, "args": args, "result": Util.clip(obs, 300)})
        self.trace.append({"step": self.step, "role": role, "tool": tool, "args": args, "seconds": round(self.elapsed(), 1)})
        obs = self.observe(role, tool, obs, pid)
        fault = self.fault(role, tool, obs)
        self.absorb(role, obs, pid, fault)
        if fault and role in READS[:3] + ("add", "remove", "checkout") and tool not in self.shapes and not again and self.step < (self.hard if final else self.hard - self.reserve() - 1):
            self.retried.add(tool)
            return self.act(role, pid, qty, text, tool, args, final, True)
        if fault and role in READS[1:4]: self.failed.add(sign)
        elif role in READS[1:4]:
            self.read_at.update(dict.fromkeys([pid] if pid else self.last["records"], self.step))
            if pid: self.read_by.setdefault(role, set()).add(pid)
        return obs
    def fault(self, role, tool, obs):
        if role == "message": return not Util.texts(obs, None, True)
        if isinstance(obs, str): return bool(obs.strip()) and (role in READS or tool in self.shapes)
        if not isinstance(obs, dict) or not obs: return False
        keys, seen = set(obs), self.shapes.get(tool, set())
        flat = lambda v: isinstance(v, str) or isinstance(v, list) and all(isinstance(x, str) for x in v) or isinstance(v, dict) and all(isinstance(x, (str, int)) and not isinstance(x, bool) for x in v.values())
        text = all(map(flat, obs.values())) and any(isinstance(v, (str, list)) or isinstance(v, dict) and any(isinstance(x, str) for x in v.values()) for v in obs.values())
        if text and not (self.idk and self.idk in keys) and (role in READS or (not keys & seen if seen else keys <= self.faults)):
            self.faults |= keys
            return True
        self.shapes.setdefault(tool, set()).update(keys)
        return False
    def absorb(self, role, obs, pid, fault):
        listing = role in ("search", "view")
        records = [r for r in Util.dicts(obs) if self.idk and isinstance(r.get(self.idk), str)]
        watched = ({r[self.idk] for r in records} | {pid}) & (set(self.cart) | set(self.cached) | {p for s in self.items for p in s["details"]})
        before = {p: (self.price(p), self.field(p, "stock")) for p in watched}
        for record in records if listing else []: self.products.setdefault(record[self.idk], {}).update(record)
        whole, read = False, {}
        if role == "compare" or pid and (role == "details" or role == "view" and "details" not in self.roles):
            read = {r[self.idk]: r for r in records if role == "compare" or r[self.idk] == pid}
            if pid and pid not in read and isinstance(obs, dict) and obs and not fault: read[pid], whole = obs, True
        tables = lambda r: {k for k, v in r.items() if isinstance(v, dict)}
        if role != "compare": self.full |= {k for r in read.values() if self.idk in r for k in tables(r)}
        elif any(not tables(r) or not self.full <= tables(r) for r in read.values()):
            read = {}
            self.roles.pop("compare", None)
        for p, record in read.items():
            self.cached[p] = {k: v for k, v in record.items() if k != self.idk}
            if p == pid or p in self.details: self.details[p] = self.cached[p]
            self.products.setdefault(p, {}).update({k: v for k, v in record.items() if not isinstance(v, (dict, list))})
        for record in records if role == "review" else []:
            keys = [self.fields.get("stock")] + [self.fields.get("price")] * (self.cart.get(record[self.idk], 1) == 1)
            self.products.setdefault(record[self.idk], {}).update({k: record[k] for k in keys if k in record})
        for p, (price, stock) in before.items():
            if price is not None and self.price(p) is not None and self.price(p) != price: self.shift("price")
            if stock != self.field(p, "stock") and self.stocked(p) is False: self.shift("out_of_stock")
        last = next(t for t in reversed(self.trace) if "role" in t)
        said = [Util.norm(v) for v in (last.get("args") or {}).values() if role == "search" and isinstance(v, str) and Util.norm(v)]
        found, noted, tool = Util.texts(obs, self.idk, True if role not in READS else not whole and "top"), [], last["tool"]
        for p, record in [(r[self.idk], r) for r in records] + ([(pid, obs)] if whole else []):
            texts = Util.texts(record, None, True)
            seen = self.sight.get((tool, p), self.sight.get(p) if role == "review" else None)
            noted += [(p, t) for t in texts if seen is not None and t not in seen]
            self.sight[(tool, p)], self.sight[p] = self.sight.get((tool, p), set()) | set(texts), self.sight.get(p, set()) | set(texts)
        found += ["%s: %s" % (p, t) for p, t in noted if t not in found]
        lines = [t for t in dict.fromkeys(found) if t not in self.heard and not any(" %s " % s in " %s " % Util.norm(t) for s in said)]
        if role == "message": self.heard.update(lines)
        self.last = {"role": role, "records": [r[self.idk] for r in records], "lines": lines, "fault": fault}
        if role != "message": self.pending += lines
    def shift(self, kind):
        self.shifted.add(kind)
        self.version += 1
    def focus(self, index, share=0, room=None):
        if self.at is not None:
            pick = sorted([p for p in self.details if self.passes(p)], key=lambda p: self.price(p) or 0)[:1]
            self.items[self.at]["low"] = (self.price(pick[0]) or 0) * self.qty if pick else None
        self.at = index
        self.spent = sum(s.get("low") or 0 for i, s in enumerate(self.items) if i != index)
        self.stop = self.step + int((self.hard - self.reserve() - self.unasked() - self.step if room is None else room) / share) if share else None
    def each(self, test):
        at, stop = self.at, self.stop
        out = [test() for i in range(len(self.items)) if self.focus(i) is None]
        self.focus(at)
        self.stop = stop
        return out
    def within(self, index, test):
        at, stop = self.at, self.stop
        self.focus(index)
        out = test()
        self.focus(at)
        self.stop = stop
        return out
    def aside(self, work, keep=(), urgent=False):
        job, self.urgent = {**self.searcher(), "began": len(self.searched), "opened": set()}, urgent
        try:
            out = work(job)
        finally:
            self.urgent = False
        del self.searched[job["began"]:], self.search_rows[job["began"]:]
        for pid in job["opened"] - set(keep): self.details.pop(pid, None)
        return out
    def talk(self):
        return [{"message": a["question"], "reply": a["reply"]} for s in self.items for a in s["asked"]]
    def sent(self):
        return sum(len(s["asked"]) for s in self.items) + len(self.said)
    def brief(self):
        budget, per_unit = (self.budget, self.per_unit) if self.total is None else (self.cap() / self.scale, False)
        wanted = {Util.flat(r["attribute"]) for r in self.reqs}
        return {"product": self.product, "requirements": self.reqs, "no_preference": sorted(k for k in self.known if Util.flat(k) not in wanted), "budget": budget, "budget_unit": self.unit, "budget_per_unit": per_unit, "quantity": self.qty, "objective": self.objective, "best_by": self.best_by, "purpose": self.purpose}
    def props(self, tool):
        return self.tools[tool]["parameters"].get("properties", {})
    def page(self, tool):
        return self.search_tools[tool].get("page")
    def moded(self, tool, key):
        spec, need = self.search_tools.get(tool, {}), self.tools[tool]["parameters"].get("required", [])
        return key in {k for k, v in self.props(tool).items() if v.get("enum") and k not in spec.get("labels", []) and (k in need or spec.get("seen"))}
    def field(self, pid, name):
        return self.products.get(pid, {}).get(self.fields.get(name))
    def price(self, pid):
        return Util.number(self.field(pid, "price"))
    def category(self, pid):
        value = self.field(pid, "category")
        return value if isinstance(value, str) and value.strip() else None
    def stocked(self, pid):
        value = self.field(pid, "stock")
        if isinstance(value, bool): return value
        return value > 0 if isinstance(value, (int, float)) else self.reading(value)
    def cap(self):
        caps = ([] if self.budget is None else [self.budget * self.scale * (self.qty if self.per_unit else 1)]) + ([] if self.total is None else [self.total * self.scale - self.spent])
        return min(caps) if caps else None
    def eligible(self, pid):
        price, cap = self.price(pid), self.cap()
        return pid not in self.gone and pid not in self.vetoed and (not self.in_stock or self.stocked(pid) is not False) and self.cat_ok.get(self.category(pid)) is not False and (cap is None or price is None or price * self.qty <= cap + 1e-9)
    def under(self, pid, top):
        price = self.price(pid)
        if price is None: return top is None or self.objective == "cheapest" or not self.backed()
        return top is None or price < top or price == top and not self.backed()
    def prefer(self, pids):
        return [p for p in pids if self.cat_ok.get(self.category(p))] or pids
    def verdict(self, pid, req):
        return self.ruled(pid, Util.sign(req))
    def ruled(self, pid, sign):
        return self.judged.get((*self.mark(pid), sign))
    def mark(self, pid):
        return pid, json.dumps([self.field(pid, "title"), self.attributes(pid), self.notes(pid)], sort_keys=True, ensure_ascii=False, default=str)
    def pool(self, size=40):
        order, hits = {p: i for i, p in enumerate(self.products)}, self.evidence()
        cheap = self.objective == "cheapest"
        return sorted([p for p in self.products if self.eligible(p)], key=lambda p: (p in self.details, -hits.get(p, 0), self.price(p) or 0 if cheap else order[p]))[:size]
    def evidence(self, sets=False):
        groups = [g for g in ([v for v in map(Util.stems, r["values"]) if v - Util.stems(self.product)] for r in self.reqs if not r["exclude"]) if g]
        said = [Util.stems(self.keywords(s["tool"], s["args"])) for s in self.searched]
        found = [{p for t, ids in zip(said, self.search_rows) if any(v <= t for v in g) for p in ids} for g in groups]
        out = {}
        for pid, row in self.products.items():
            if pid not in self.texts: self.texts[pid] = Util.stems(json.dumps(row, ensure_ascii=False, default=str))
            shown = frozenset(i for i, (g, ids) in enumerate(zip(groups, found)) if pid in ids or any(v <= self.texts[pid] for v in g))
            if shown: out[pid] = shown if sets else len(shown)
        return out
    def fresh(self, shown=False):
        hits, bounds = self.evidence(), [(r, Util.flat(r["unit"])) for r in self.reqs if not r["exclude"] and (r["min"] is not None or r["max"] is not None) and len(Util.flat(r["unit"])) > 1]
        sizes = lambda text, r, unit: [(r["min"] is None or float(n) >= r["min"]) and (r["max"] is None or float(n) <= r["max"]) for n, u in re.findall(r"(?<![a-z\d.])(\d+(?:\.\d+)?)[\s-]*([a-z]+)(?![\s-]*\d)", re.sub(r"(?<=\d),(?=\d)", "", text.lower())) for a, b in [sorted((u, unit), key=len)] if len(a) > 1 and a[0] == b[0] and (lambda rest: all(c in rest for c in a))(iter(b))]
        place, failed = {p: i for i, p in sorted(((i, p) for rows in self.search_rows for i, p in enumerate(rows)), reverse=True)} if shown else {}, {(self.price(p), self.field(p, "title")) for p in self.details if any(self.verdict(p, r) is False for r in self.reqs)}
        rank = {p: ((wrong or (self.price(p), self.field(p, "title")) in failed, -hits.get(p, 0), self.price(p) is None, self.price(p) or 0, -sized) if self.objective == "cheapest" else (-hits.get(p, 0) - sized, wrong)) + (place.get(p, math.inf), hits.get(p, 0) + sized) for p in self.pool(None) if p not in self.details for found in [[[sizes(v, r, unit) for v in self.products[p].values() if isinstance(v, str)] for r, unit in bounds]] for sized, wrong in [(sum(any(map(any, f)) for f in found), any(m and not any(m) for f in found for m in f))]}
        return sorted((p for p in rank if rank[p][-1] or not shown), key=lambda p: rank[p][:-1])
    def open_item(self, pid, final=False):
        batch, role = self.roles.get("compare"), "details" if "details" in self.roles else "view" if "view" in self.roles else None
        if batch and pid not in self.cached:
            ids = ([pid] + sorted((p for p in self.fresh() if p not in self.cached and p != pid), key=lambda p: (self.cat_ok.get(self.category(p)) is not True, self.foreign(p))))[:self.batch]
            args = Util.fit(self.tools[batch["tool"]]["parameters"], Util.fill(batch["args"], None, None, None, lines=[(p, 1) for p in ids])) or {}
            sent = sum(p in json.dumps(args) for p in ids)
            if self.act("compare", tool=batch["tool"], args=args, final=final) is not None and not self.last["records"]:
                if sent > 1: self.batch = sent // 2
                else: self.roles.pop("compare")
        if pid in self.cached:
            self.details[pid] = self.cached[pid]
            return
        began, obs = len(self.trace), None
        if role == "details" and self.dual is None and self.roles.get("view", {}).get("tool") not in (None, self.roles["details"]["tool"]): self.dual, obs = self.weigh_reads(pid, final)
        elif role: obs = self.act(role, pid=pid, final=final)
        tried = [(t["tool"], t["args"]) for t in self.trace[began:] if "role" in t]
        if tried and pid not in self.cached and not (self.last["fault"] and tried[-1][0] in self.shapes): self.remedy("read", "read the attributes of product %s with the %s tool" % (pid, role), obs, tried, lambda name, given: self.act(role, pid=pid, tool=name, args=given, final=final))
        if role != "details": self.details[pid] = dict(self.products.get(pid, {}))
    def weigh_reads(self, pid, final):
        row, raw = set(Util.attrs(self.products.get(pid, {}), {})), self.act("details", pid=pid, final=final)
        facts = lambda obs: len(set(Util.attrs(obs, {})) - row) if isinstance(obs, dict) and not self.last.get("fault") else -1
        shown, obs = facts(raw), self.act("view", pid=pid, final=final)
        if facts(obs) > shown:
            self.roles["view"], self.roles["details"] = self.roles["details"], self.roles["view"]
            self.absorb("details", obs, pid, False)
        return shown >= 0, raw


for key in ITEM: setattr(Core, key, Util.field(key))

class Read:
    def observe(self, role, tool, obs, pid):
        if role in ("search", "view") and not self.idk: self.idk = Util.id_key(obs, {Util.flat(k) for r in self.roles.values() for d in Util.dicts(r["args"]) for k, v in d.items() if v == "<ID>"})
        texts, known = Util.texts(obs, None, True), {pid, *self.products} - {None}
        listed = role in ("search", "compare") and any(isinstance(v, list) and any(isinstance(x, str) for x in v) for v in [obs, *(w for d in Util.dicts(obs) for w in d.values())])
        if role not in READS[:4] or not texts or self.idk and any(Util.ident(d.get(self.idk)) for d in Util.dicts(obs)) or not (self.idk is None or listed or self.readers or any(p in t for t in texts + [k for d in Util.dicts(obs) for k in d] for p in known)): return Util.ids(obs, self.idk)
        key = (tool, json.dumps(Util.sig(obs), sort_keys=True))
        if key not in self.readers:
            again = next((m for (t, _), how in list(self.readers.items()) if t == tool and how for m in ({**how, "path": q} for p in Util.nodes(obs) for q in (p + how["path"], p)) if not self.laid(m, obs, pid)[2]), None)
            fix = {"action": "map", "mapping": again} if again else self.recover("read-shape", "read the product records (id, title, price, stock, attributes) in this %s result" % tool, obs, (), pid)
            if fix.get("action") != "skip": self.readers[key] = {**fix["mapping"], "nested": {**{k: v for how in self.readers.values() if how for k, v in how["nested"].items()}, **fix["mapping"]["nested"]}} if fix.get("mapping") else {}
        out, how, _ = self.laid(self.readers[key], obs, pid) if self.readers.get(key) else (None, {}, "")
        if out is None and key in self.readers: out, how = self.extract(tool, obs), {}
        self.idk = self.idk or how.get("id")
        return Util.ids(obs if out is None else out, self.idk)
    def laid(self, given, obs, pid):
        m, text = given if isinstance(given, dict) else {}, lambda d, k: Util.unescape(d[k]) if isinstance(d.get(k), str) and d[k] else None
        path = m.get("path") if isinstance(m.get("path"), list) else [m["path"]] if isinstance(m.get("path"), (str, int)) and m["path"] != "" else []
        fields = [f if isinstance(f, dict) else {"name": f} for f in Util.items(m.get("fields")) if isinstance(f, (str, dict))]
        inner = list(m["nested"].items() if isinstance(m.get("nested"), dict) else []) + [(f.get("name"), f) for f in fields]
        how = {"path": [k for k in path if isinstance(k, (str, int))], "records": text(m, "records"), "format": m.get("format") if m.get("format") in ("pairs", "columns", "json") else None, "split": text(m, "split"), "pair": text(m, "pair"), "fields": [f["name"] for f in fields if text(f, "name")], "header": m.get("header") is True and m.get("format") == "columns",
               "keyed": m.get("keyed") is True, "id": text(m, "id") or self.idk or "id", "price": text(m, "price"), "nested": {k: {"split": v["split"], "pair": v["pair"]} for k, v in inner if isinstance(k, str) and isinstance(v, dict) and text(v, "split") and text(v, "pair")}}
        if not (how["format"] == "json" or how["split"] and (how["pair"] if how["format"] == "pairs" else how["format"] and (how["fields"] or how["header"]))): return None, how, "it does not say how the fields of a record are separated and named"
        rows, lost, at = Util.lay(obs, how, pid)
        blob, ids, known = json.dumps(at, ensure_ascii=False, default=str), [str(r[how["id"]]) for r in rows], set(self.products) | set(self.cart)
        lost, prices = max(lost, blob.count(how["id"] + (how["pair"] or "\0")) - len(rows)), [r[how["price"]] for r in rows if r.get(how["price"]) not in (None, "")]
        why = ("it reads no record holding the id field %s at that path" % how["id"] if not rows else "%d records of the result were not read" % lost if lost else "its records hold nothing but the id" if any(len(r) < 2 for r in rows) else "entries of %s still hold its pair characters, so its split is not the one between its entries" % bad if (bad := next((k for k, v in how["nested"].items() if sum(x.count(v["pair"].strip() or v["pair"]) for r in rows if isinstance(r.get(k), dict) for x in r[k].values() if isinstance(x, str)) >= max(1, sum(len(r[k]) for r in rows if isinstance(r.get(k), dict)))), None)) else "the id field %s does not hold unique product codes: %s" % (how["id"], ids[:3]) if len(set(ids)) < len(ids) else
               "the record of product %s, which this call asked for, is missing" % pid if pid and pid not in ids else "it reads ids such as %s, but the store's ids look like %s" % (ids[0], next(p for p in known if p in blob)) if not known & set(ids) and any(p in blob for p in known) else
               "the price field %s holds a value that is not a number: %s" % (how["price"], next(v for v in prices if Util.number(v) is None)) if any(Util.number(v) is None for v in prices) else None)
        name = self.idk or how["id"]
        rows = [{name: r[how["id"]], **{k: v for k, v in r.items() if k != how["id"]}} for r in rows]
        return (Util.put(obs, how["path"], rows if isinstance(at, (list, dict)) or len(rows) > 1 else rows[0]) if rows else None), how, why
    def extract(self, tool, obs):
        if self.mended >= math.ceil(self.hard * MENDS / 100): return None
        self.mended, raw = self.mended + 1, json.dumps(obs, ensure_ascii=False, default=str)[:RAW]
        out = self.model(Prompts.RECORDS, {"tool": tool, "result": raw}, len(raw) // 2 + 100) or {}
        key = out.get("id") if isinstance(out.get("id"), str) else None
        if not (rows := [r for r in Util.items(out.get("records")) if isinstance(r, dict) and Util.ident(r.get(key)) and str(r[key]) in raw]): return None
        self.idk, path = self.idk or key, max((p for p in Util.nodes(obs) if all(str(r[key]) in json.dumps(Util.walk(obs, p), ensure_ascii=False) for r in rows)), key=len)
        rows = [{self.idk: r[key], **{k: v for k, v in r.items() if k != key}} for r in rows]
        return Util.put(obs, path, rows if len(rows) > 1 or isinstance(Util.walk(obs, path), list) else rows[0])
    def read(self):
        mapped = lambda: self.search_tools and "checkout" in self.roles and ("add" in self.roles or "<ID>" in json.dumps(self.roles["checkout"]["args"]))
        self.map_tools()
        while not mapped() and self.elapsed() < self.deadline() / 3:
            kept, search = dict(self.roles), dict(self.search_tools)
            self.map_tools()
            self.roles, self.search_tools = {**kept, **self.roles}, {**search, **self.search_tools}
            self.unload()
        if not mapped(): return False
        for _ in range(2):
            self.items, self.at = [], None
            self.parse_request()
            if any(s["product"] for s in self.items): break
        if self.allowed is None or self.hard == LOOP: self.usage()
        self.resolve()
        self.rescale()
        return True
    def map_tools(self, problems=()):
        data = {"tools": [{"name": n, **t} for n, t in self.tools.items()], **({"problems_with_your_last_answer": list(problems)} if problems else {})}
        roles, search, kept = {}, {}, (dict(self.roles), dict(self.search_tools))
        for attempt in range(MAP_TRIES):
            out = self.model(Prompts.TOOLS, data, 900, (LUNA, SOL, DEEPSEEK)[attempt % 3])
            if out is None: time.sleep(attempt + 1)
            errors = self.check_tools(out)
            roles, search = {**roles, **self.roles}, {**search, **{n: {**search.get(n, {}), **f} for n, f in self.search_tools.items()}}
            if not errors: break
            data["problems_with_your_last_answer"] = errors
        self.roles, self.search_tools = (roles, search) if errors else (self.roles, {n: search[n] for n in self.search_tools})
        if problems: self.roles, self.search_tools = {**kept[0], **self.roles}, {**kept[1], **{n: {**kept[1].get(n, {}), **s} for n, s in self.search_tools.items()}}
        idle = [n for n in self.tools if self.props(n) and n not in self.search_tools and n not in {r["tool"] for r in self.roles.values()}]
        if problems or errors or not idle or {"view", "details"} <= set(self.roles): return
        kept, search = dict(self.roles), dict(self.search_tools)
        data["problems_with_your_last_answer"] = ["tools %s have no role; a tool that shows one product takes the view or details role that is still null" % idle]
        if self.check_tools(self.model(Prompts.TOOLS, data, 900)) or not set(kept) < set(self.roles): self.roles, self.search_tools = kept, search
    def check_tools(self, out):
        if not isinstance(out, dict): return ["the reply was not a JSON object"]
        items = out.get("search") if isinstance(out.get("search"), list) else [out.get("search")]
        self.search_tools, self.roles = {}, {}
        for item in items:
            name = item.get("tool") if isinstance(item, dict) else item
            if isinstance(name, str) and name in self.tools:
                props = self.props(name)
                kinds = {"page": ("integer", "number", "string"), "price_cap": ("integer", "number")}
                self.search_tools[name] = {k: item[k] for k in kinds if isinstance(item, dict) and isinstance(item.get(k), str) and item[k] in props and Util.kind(props[item[k]]) in kinds[k]}
                page = props.get(self.page(name), {})
                kind = item.get("page_kind") if isinstance(item, dict) else None
                self.search_tools[name]["offset"] = kind == "offset" if kind in ("number", "offset") else page.get("minimum") == 0
        errors = [] if self.search_tools else ["search: no valid tool"]
        for role in ROLES:
            entry = out.get(role)
            problem = self.bad_entry(role, entry) if entry else ""
            if problem: errors.append(role + ": " + problem)
            elif entry: self.roles[role] = {"tool": entry["tool"], "args": entry.get("args") or {}}
        if not Util.slots(self.roles.get("compare", {}).get("args")): self.roles.pop("compare", None)
        self.replaces = not isinstance(out.get("add"), dict) or out["add"].get("replaces") is not False
        self.unload()
        if "checkout" not in self.roles: errors.append("checkout: missing")
        elif "add" not in self.roles and "<ID>" not in json.dumps(self.roles["checkout"]["args"]): errors.append("add: missing; when the tool that buys takes the products itself, its template needs the <ID> placeholder")
        if "message" in self.roles and self.roles["message"]["tool"] in {self.roles[r]["tool"] for r in ("add", "checkout") if r in self.roles}: errors.append("message must not be a cart or checkout tool")
        return errors
    def unload(self):
        add, drop = self.roles.get("add"), self.roles.get("remove")
        if add and drop == {"tool": add["tool"], "args": add["args"]}: self.roles.pop("remove")
        low = Util.most(add["args"], self.tools[add["tool"]]["parameters"], "minimum") if add and "<QTY>" in json.dumps(add["args"]) else 1
        if low is not None and low <= 0: self.sets = True
        if (low is None or low <= 0) and "remove" not in self.roles: self.roles["remove"] = {"tool": add["tool"], "args": json.loads(json.dumps(add["args"]).replace('"<QTY>"', str(int(low or 0)))), **({"trial": True} if low is None else {})}
    def bad_entry(self, role, entry):
        if not isinstance(entry, dict) or not isinstance(entry.get("tool"), str) or entry["tool"] not in self.tools: return "unknown tool"
        args = entry.get("args") or {}
        if not isinstance(args, dict): return "args must be an object"
        schema = self.tools[entry["tool"]]["parameters"]
        extra = [k for k in args if k not in schema.get("properties", {})]
        missing = [k for k in schema.get("required", []) if k not in args]
        if extra or missing: return "unknown arguments %s, missing required arguments %s" % (extra, missing)
        if role in NEEDS and NEEDS[role] not in json.dumps(args): return "needs the " + NEEDS[role] + " placeholder"
        sample = Util.fill(args, "1", 1, "hello")
        if Util.norm(json.dumps(Util.fit(schema, sample), sort_keys=True)) != Util.norm(json.dumps(sample, sort_keys=True)): return "argument values do not match the schema"
        return ""
    def parse_request(self):
        tools = [{"name": n, "description": Util.clip(t["description"], 600)} for n, t in self.tools.items()]
        data = {"request": self.query, "opening": self.opening, "limits": self.view_limits, "tools": tools, "can_message_shopper": "message" in self.roles, "agent_actions": self.acts(True)}
        read = lambda out: isinstance(out, dict) and any(isinstance(i, dict) and str(i.get("product") or "").strip() for i in Util.items(out.get("items")) or [out])
        sure, out = lambda o: read(o) and ("messages_allowed" in o and o["messages_allowed"] is None or Util.number(o.get("messages_allowed")) is not None), {}
        for name in (SOL, LUNA):
            if sure(out): break
            new = self.model(Prompts.SPEC, data, 3000, name)
            if read(new) and (not read(out) or sure(new)): out = new
        out = self.ground(out, data)
        valid = lambda value: value if value and value > 0 else None
        wanted = [i for i in Util.items(out.get("items")) if isinstance(i, dict)] or [out]
        aim = lambda o, base: (o["objective"], str(o.get("best_by") or base[1]), o.get("best_high") is not False) if o.get("objective") in ("cheapest", "best") else base
        whole = aim(out, ("any", "", True))
        budget, per_unit = valid(Util.number(out.get("budget"))), out.get("budget_per_unit") is True
        self.total = budget if len(wanted) > 1 and not per_unit else None
        for item in wanted:
            own = valid(Util.number(item.get("budget")))
            mine, unit = (own, item.get("budget_per_unit") is True) if own else (budget, per_unit) if self.total is None else (None, False)
            qty = Util.number(item.get("quantity"))
            reqs = [r for r in map(Util.requirement, Util.items(item.get("requirements"))) if r and not ({mine, budget} & {r["min"], r["max"]} - {None})]
            self.items.append({"product": str(item.get("product") or ""), "reqs": reqs, "qty": int(qty) if qty and qty >= 1 and qty == int(qty) else 1, "budget": mine, "per_unit": unit,
                               "details": {}, "cat_ok": {}, "searched": [], "search_rows": [], "asked": [], "known": set(), "revealed": [], "lost": set(), "measure": {},
                               **dict(zip(("objective", "best_by", "best_high"), aim(item, whole)))})
        self.focus(0)
        self.unit = str(out.get("budget_unit") or "")
        self.refs = [r for r in Util.items(out.get("references")) if isinstance(r, dict) and isinstance(r.get("listing"), dict) and str(r.get("attribute") or "").strip()]
        self.purpose = str(out.get("purpose") or "")
        steps, seconds = Util.number(out.get("actions_allowed")), Util.number(out.get("seconds_allowed"))
        if steps and 1 <= steps <= self.max_steps: self.hard = int(steps)
        self.seconds = seconds if seconds and seconds > 0 else None
        source = Util.norm(" ".join([self.query] + [t["description"] for t in self.tools.values()]))
        line = lambda v: v if isinstance(v, int) and not isinstance(v, bool) and 0 <= v < len(self.items) else None
        ties = [{"quote": str(r["quote"]).strip(), "item": line(r.get("item")), "on": line(r.get("on"))} for r in Util.items(out.get("basket_rules")) if isinstance(r, dict) and Util.norm(r.get("quote") or "") and Util.norm(r["quote"]) in source]
        self.links = [t for t in ties if None not in (t["item"], t["on"]) and t["item"] != t["on"]]
        self.basket_rules = [t["quote"] for t in ties if t not in self.links]
        count = lambda value: max(0, int(Util.number(value) or 0))
        rules = [r for r in ({"quote": str(r["quote"]).strip(), "before": set(Util.listed(r.get("before"))) & set(self.acts(True)) or {"checkout"}, "messages": count(r.get("messages")), "calls": count(r.get("calls")), "per_line": r.get("per_line") is True, "done": set()}
                 for r in Util.items(out.get("rules")) if isinstance(r, dict) and Util.norm(r.get("quote") or "") and Util.norm(r["quote"]) in source) if r["messages"] + r["calls"] or "message" in r["before"]]
        told = self.model(Prompts.TELL + Prompts.NOTE * ("<TEXT>" in json.dumps(self.roles["checkout"]["args"])), {"instructions": {str(n + 1): r["quote"] for n, r in enumerate(rules)}}, 80) if rules and "message" in self.roles else None
        tell, unless = ({str(n) for n in Util.items(told.get(k))} for k in ("tell", "unless")) if told is not None else (None, set())
        for n, r in enumerate(rules if tell is not None else []):
            if r["messages"] and str(n + 1) not in tell: r["messages"], r["before"] = 0, r["before"] - {"message"}
        self.rules = [r for n, r in enumerate(rules) if str(n + 1) not in unless]
        self.moments = set().union(*[r["before"] for r in self.rules if r["messages"] + r["calls"]]) - set(PLAN)
        asks = Util.number(out.get("messages_allowed"))
        self.allowed = 0 if "message" not in self.roles else int(asks) if asks is not None and asks >= 0 else None
        self.questions = None if self.allowed is None else max(0, self.allowed - self.reserved())
        self.in_stock = out.get("stock_rule") != "any"
    def founded(self, spec, have, numbers, cited=()):
        rest = Util.without(have, Util.listed(spec.get("wishes")))
        facts = [Util.norm(v).split() for r in Util.items(spec.get("references")) + list(cited) if isinstance(r, dict) and isinstance(r.get("listing"), dict) for v in Util.described(r["listing"])]
        sure = lambda values, bounds, name="": all(Util.grounded(v, rest) for v in values) and all(abs(n) in numbers or bool(name) and Util.grounded(name, rest) for n in bounds if n is not None)
        echo = lambda values: bool(values) and all(len(Util.spans(rest, v)) <= sum(len(Util.spans(f, v)) for f in facts) for v in values)
        lines, refs, lost = [dict(i) for i in Util.items(spec.get("items")) if isinstance(i, dict)] or [dict(spec)], [], []
        for at, line in enumerate(lines):
            if not Util.grounded(str(line.get("product") or ""), have): lost.append(("product", at, line.get("product")))
            reqs = [(e, Util.requirement(e)) for e in Util.items(line.get("requirements"))]
            line["requirements"] = [e for e, r in reqs if r and sure(r["values"], (r["min"], r["max"]), r["attribute"]) and not echo(r["values"])]
            lost += [("requirement", at, e) for e, r in reqs if r and e not in line["requirements"]]
        for ref in filter(lambda r: isinstance(r, dict), Util.items(spec.get("references"))):
            listing = ref.get("listing") if isinstance(ref.get("listing"), dict) else {}
            grounded = sure(Util.listed(listing.get("seller")) + Util.listed(listing.get("product")) + Util.described(listing), (Util.number(ref.get("margin")),))
            if grounded: refs.append(ref)
            if not grounded or not Util.described(listing): lost.append(("reference", ref.get("item"), ref))
        return {**spec, "items": lines, "references": refs}, lost
    def ground(self, spec, data):
        text = self.query + " " + json.dumps(self.opening or "", ensure_ascii=False)
        have, numbers = Util.norm(text).split(), {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))}
        spec, lost = self.founded(spec, have, numbers)
        if not lost: return spec
        again = self.founded(self.model(Prompts.SPEC, {**data, "ungrounded": [{"field": k, "line": at, "value": e} for k, at, e in lost]}, 3000, LUNA) or {}, have, numbers, spec["references"])[0]
        pair = lambda at: again["items"][at] if len(again["items"]) == len(spec["items"]) else {}
        alike = lambda one, other: Util.overlaps(one["attribute"], other["attribute"]) or bool({Util.flat(v) for v in one["values"]} & {Util.flat(v) for v in other["values"]})
        for kind, at, entry in lost:
            if kind == "product":
                new = str(pair(at).get("product") or "")
                spec["items"][at]["product"] = new if new and Util.grounded(new, have) else " ".join(w for w in str(entry).split() if Util.grounded(w, have)) or entry
            elif kind == "requirement":
                reqs = spec["items"][at]["requirements"]
                reqs += [e for e in Util.items(pair(at).get("requirements")) if alike(Util.requirement(e), Util.requirement(entry)) and Util.sign(Util.requirement(e)) not in {Util.sign(Util.requirement(r)) for r in reqs}]
            else:
                found = [r for r in again["references"] if r not in spec["references"] and Util.overlaps(str(r.get("attribute") or ""), str(entry.get("attribute") or ""))]
                if found: spec["references"] = [r for r in spec["references"] if r is not entry] + found
        return spec
    def rescale(self):
        price = self.fields.get("price")
        amounts = sorted({a for a in [self.total, self.budget] + [s["budget"] for s in self.items] if a})
        rows = [p for p in self.products if (self.price(p) or 0) > 0][:2 * EXAMPLES]
        if self.priced or not price or not rows or not amounts: return
        self.priced = True
        out = self.model(Prompts.PRICES, {"request": self.query, "budgets": amounts, "budget_unit": self.unit, "search_tools": self.search_specs(), "price_key": price, "listings": [Util.clip(self.products[p], 60) for p in rows]}, 100) or {}
        factor, prices = Util.number(out.get("factor")), sorted(map(self.price, rows))
        budget = amounts[-1] / (sum(s["qty"] for s in self.items) if amounts[-1] == self.total else 1)
        gap = lambda value: max(value / prices[len(prices) // 2], prices[len(prices) // 2] / value)
        if factor and factor > 0 and math.isfinite(factor) and gap(budget * factor) < gap(budget) and prices[0] <= budget * factor and (factor > 1 or any(p != int(p) for p in prices)): self.scale = factor
    def resolve(self):
        self.anchors = []
        if not self.refs: return
        told, query = {Util.flat(v) for r in self.refs for v in Util.described(r["listing"])}, Util.flat(self.query)
        for item in self.items: item["reqs"][:] = [q for q in item["reqs"] if not q["values"] or not all(Util.flat(v) in told and query.count(Util.flat(v)) == 1 for v in q["values"])]
    def owner(self, ref):
        return next((int(i) for i in [Util.number(ref.get("item"))] if i is not None and 0 <= i < len(self.items)), 0)
    def searcher(self):
        if not self.job:
            out = self.model(Prompts.KEYWORDS, {"search_tools": self.search_specs()}, 300)
            tool, args = (out.get("tool"), out.get("args")) if isinstance(out, dict) else (None, None)
            key = next((k for k, v in args.items() if "<TEXT>" in json.dumps(v)), None) if isinstance(args, dict) else None
            if isinstance(tool, str) and tool in self.search_tools and key is not None: self.job = {"tool": tool, "key": key, "args": {k: v for k, v in args.items() if k == key or k != self.page(tool) and self.moded(tool, k)}}
        return self.job
    def pursue(self, refs):
        if not self.searcher(): return []
        found = [(self.owner(ref), new) for ref, new in self.aside(lambda job: [(ref, self.referred(ref, self.follow(ref, job))) for ref in refs], urgent=True) if new]
        for index, new in found: self.items[index]["reqs"] = [q for q in self.items[index]["reqs"] if Util.stems(q["attribute"]) != Util.stems(new["attribute"])] + [new]
        return [new for _, new in found]
    def follow(self, ref, job):
        target, seen, line, key = ref["listing"], [], "", None
        for _ in range(LOOP):
            pid, out = self.find(target, str(ref["attribute"]), job, seen, (line, key))
            if pid is None: return {}
            seen.append(pid)
            after = out.get("redirect")
            if isinstance(after, dict) and Util.ident(after.get("id")) and not self.cited(after["id"], pid): after = {**after, "id": ""}
            if not isinstance(after, dict) or not (Util.listed(after.get("seller")) or Util.described(after) or Util.ident(after.get("id")) and after["id"] not in seen): return out
            line, key = " ".join(s for s in re.split(r"(?<=[.!?;])\s+", " ".join(self.sheet(pid)["notes"])) if Util.holds(s, ref["attribute"])), key or next((k for k, v in self.products.get(pid, {}).items() if isinstance(v, str) and Util.listed(target.get("seller")) and Util.norm(v) == Util.norm(" ".join(Util.listed(target["seller"])))), None)
            target = {k: "" if k == "seller" and line and Util.described(after) and not Util.holds(line, " ".join(Util.listed(after.get(k)))) else after.get(k) for k in ("id", "seller", "product", "facts")}
        return {}
    def find(self, target, attribute, job, seen, pointer=("", None)):
        named, (line, key) = target.get("id") if Util.ident(target.get("id")) and target.get("id") not in seen else None, pointer
        if named and not self.busy():
            self.open_item(named)
            if named in self.details:
                job["opened"].add(named)
                self.open_page(named)
                out = self.refer(attribute, target, [named])
                if out.get("match") == named and self.told_by(out): return named, out
        seller, product, facts = " ".join(Util.listed(target.get("seller"))), " ".join(Util.listed(target.get("product"))), [f for k, v in (target["facts"].items() if isinstance(target.get("facts"), dict) else []) for f in [Util.bare(k, v)] if f] or Util.described(target)
        trim = lambda n, limit=self.props(job["tool"]).get(job["key"], {}).get("maxLength", 1000): n if len(n) <= limit else n[:limit + 1].rsplit(" ", 1)[0]
        forms = ([[g] for f in sorted(facts, key=lambda f: (not re.search(r"\d", f), Util.overlaps(f, product))) for g in dict.fromkeys([f, re.sub(r"(\d) (?=[a-z])", r"\1", f)])] or [[product]]) + [sorted(facts, key=len)] * (len(facts) > 1)
        precise = [trim(" ".join(dict.fromkeys(filter(None, (seller, *f, "" if seller else product))))) for f in forms]
        needles, hits, opened, wrong, empty, loose, last, held = list(dict.fromkeys(n for n in precise + [trim(" ".join(filter(None, (seller, product)))), seller] if n)), set(), [], [], None, None, set(), False
        stop = lambda: self.busy() and (not held or self.late() or self.hard - self.reserve() <= self.step)
        for again in range(LOOP):
            if loose and not (needles := [n for n in needles if n == precise[-1]] * (len(facts) > 1)): break
            fetched = False
            while not self.busy() and ((again or key and not seller) and not fetched or not any(p in hits or self.told(p, facts) for p in self.sellers(seller, product, facts, seen + opened, hits, line))):
                found = self.fetch(job, needles, precise)
                if found is None: break
                hits.update(found)
                fetched, last = True, set(found)
                if found and not set(found) & set(self.sellers(seller, product, facts, [], set())): self.searched[-1]["next"] = None
            if key and not seller and len(names := {r[key] for r in self.products.values() if isinstance(r.get(key), str) and Util.holds(line, r[key])}) == 1: return self.find({**target, "seller": names.pop()}, attribute, job, seen)
            fresh = [p for p in self.sellers(seller, product, facts, seen + opened, hits, line) if not loose or p in last]
            if again and (not fresh or stop()): break
            for pid in fresh:
                if stop(): break
                self.open_item(pid)
                if pid not in self.details: continue
                opened.append(pid)
                job["opened"].add(pid)
                out = self.refer(attribute, target, [pid])
                if out.get("match") == pid and not self.denied(pid, target):
                    held = True
                    if all(self.accord(out, pid, target, True)) and (not self.open_page(pid, not stop()) or (out := self.refer(attribute, target, [pid])).get("match") == pid) and self.told_by(out) and all(self.accord(out, pid, target)): return pid, out
                if all(self.grounds(pid, f) for f in facts): break
            live = [p for p in opened if p not in wrong]
            if not live: break
            while live:
                out = self.refer(attribute, target, live)
                pick = self.near(out, live, target)
                if pick is None: break
                paged, ask = self.open_page(pick, not stop()) or pick != out.get("match"), {**target, "facts": {k: v for k, v in target["facts"].items() if not self.grounds(pick, Util.bare(k, v))}} if pick != out.get("match") and isinstance(target.get("facts"), dict) else target
                one = self.refer(attribute, ask, [pick]) if paged or len(live) > 1 else out
                out = one if paged or not self.told_by(out) or one.get("match") == pick and isinstance(one.get("redirect"), dict) > isinstance(out.get("redirect"), dict) else out
                if out.get("match") == pick and not self.denied(pick, target):
                    if self.told_by(out) and all(self.accord(out, pick, target)) or pick == empty: return pick, out
                    loose, wrong, empty = (loose or (pick, out), wrong + [pick], empty) if self.told_by(out) else (loose, wrong, pick)
                    break
                wrong.append(pick)
                live.remove(pick)
        return loose or (None, {})
    def open_page(self, pid, spare=False):
        return "view" in self.roles and self.roles["view"]["tool"] != self.roles.get("details", {}).get("tool") and pid not in self.read_by.get("view", ()) and (spare or not self.busy()) and self.act("view", pid=pid) is not None
    def near(self, out, live, target):
        sure, held = [p for p in live if all(self.accord(out, p, target))], lambda p: self.accord(out, p, target).count(True) - len(self.accord(out, p, target))
        return out.get("match") if out.get("match") in sure else sure[0] if len(sure) == 1 else out.get("match") if out.get("match") in live else next((p for p in sorted(live, key=held, reverse=True) if held(p) >= -1 and self.open_page(p)), None)
    def accord(self, out, pid, target, strict=False):
        mine, sheet = (out.get("checks") if isinstance(out.get("checks"), dict) else {}).get(pid), self.sheet(pid)
        text = Util.norm(" ".join([str(sheet["name"] or ""), *sheet["attributes"].values(), *sheet["notes"]]))
        said = lambda k: mine.get(k) if isinstance(mine, dict) and isinstance(mine.get(k), dict) else {}
        return [True if self.grounds(pid, f) or said(k).get("fits") is True and (not strict or self.grounds(pid, has := str(said(k).get("has") or "")) and Util.flat(has) and (Util.flat(has) in Util.flat(f) or Util.flat(f) in Util.flat(has))) or not strict and said(k).get("fits") is None and Util.holds(text, f) else said(k).get("fits") and None for k, v in (target["facts"].items() if isinstance(target.get("facts"), dict) else []) for f in [Util.bare(k, v)] if f]
    def grounds(self, pid, fact):
        return any(Util.flat(v) == Util.flat(fact) or Util.holds(v, fact) for k, v in self.attributes(pid).items() if k not in ("category", "brand") and len(v.split()) <= 6)
    def cited(self, value, pid):
        return str(value) in json.dumps({**self.products.get(pid, {}), **self.details.get(pid, {})}, ensure_ascii=False, default=str) and re.sub(r"[^\W_]", "x", str(value)) in {re.sub(r"[^\W_]", "x", p) for p in self.products}
    def refer(self, attribute, target, pids):
        out = self.model(Prompts.REFER, {"property": attribute, "wanted_listing": target, "opened": {p: Util.clip({**self.products.get(p, {}), **self.details[p]}, 400) for p in pids}}, 300 + 100 * len(pids), DEEPSEEK) or {}
        if out.get("match") not in pids or False in self.accord(out, out["match"], target): out["match"] = None
        return out
    def told_by(self, out):
        return isinstance(out.get("redirect"), dict) or bool(Util.listed(out.get("pairs_with")) or Util.listed(out.get("value")))
    def denied(self, pid, target):
        return any(self.decide(pid, {"attribute": str(k), "values": Util.listed(v), "exclude": False, "min": None, "max": None, "unit": ""}) is False and not self.grounds(pid, Util.bare(k, v)) for k, v in (target["facts"].items() if isinstance(target.get("facts"), dict) else []) if str(k).strip() and Util.listed(v))
    def fetch(self, job, needles, precise):
        last, page = self.searched[-1] if len(self.searched) > job["began"] else {}, self.page(job["tool"])
        if needles and (needles[0] in precise or last.get("next") is None):
            args = Util.fit(self.tools[job["tool"]]["parameters"], Util.fill(job["args"], None, None, needles.pop(0)))
        elif last.get("next") is not None and page:
            args, last["next"] = {**last["args"], page: last["next"]}, None
        else: return None
        return found[0] if (found := self.run_search(job["tool"], args) if args else None) is not None else [] if needles else None
    def told(self, pid, facts):
        return sum(Util.flat(f) in Util.flat(" ".join(v for v in self.products.get(pid, {}).values() if isinstance(v, str))) for f in facts if Util.flat(f))
    def sellers(self, seller, product, facts, skip, hits, line=""):
        texts = lambda p: [v for v in self.products[p].values() if isinstance(v, str)]
        return sorted([p for p in self.products if p not in skip and (any(Util.norm(v) == Util.norm(seller) for v in texts(p)) if seller else self.told(p, facts))], key=lambda p: (-sum(Util.holds(line, v) for v in texts(p) if len(v.split()) <= 3), -self.told(p, facts), p not in hits, not any(Util.overlaps(v, product) for v in texts(p) if len(v.split()) <= 4)))
    def referred(self, ref, out):
        values = Util.listed(out.get("pairs_with")) or Util.listed(out.get("value"))
        relation, margin, base = str(ref.get("relation") or "").lower(), Util.number(ref.get("margin")), Util.number(values[0]) if values else None
        sides = {"within": (-1, 1), "at_least": (0, None), "at_most": (None, 0)}.get(relation) if margin is not None or relation != "within" else None
        if not values or sides and base is None and relation == "within": return None
        if sides and base is not None:
            unit = re.search(r"\d[\d.,]*[\s-]*([a-z%][a-z%/]*)", values[0].lower())
            low, high = (None if side is None else round(base + side * (margin or 0), 6) for side in sides)
            return {"attribute": str(ref["attribute"]).strip(), "values": [], "exclude": False, "min": low, "max": high, "unit": unit.group(1) if unit else str(ref.get("unit") or "")}
        return {"attribute": str(ref["attribute"]).strip(), "values": values, "exclude": relation == "different", "min": None, "max": None, "unit": ""}


class Probe:
    def kit(self, tool):
        return self.world.setdefault("tools", {}).setdefault(tool, {"rows": None, "first": None, "end": None, "match": None, "votes": [0, 0]})
    def size(self, tool):
        return self.kit(tool)["rows"] or self.page_size
    def shelves(self, tool, rows):
        spec, skip = self.search_tools[tool], self.fields.get("description")
        values = [v for p in rows for k, v in self.products.get(p, {}).items() if isinstance(v, str) and k != skip and v.strip()]
        for key, prop in self.props(tool).items():
            if prop.get("enum") and key not in spec.get("labels", []) and any(Util.flat(e) == Util.flat(v) or Util.holds(v, str(e)) for e in prop["enum"] for v in values): spec.setdefault("labels", []).append(key)
        spec["seen"] = spec.get("seen") or bool(rows)
    def pieces(self, text):
        return list(dict.fromkeys(p for p in re.findall(r"[^\W_]+(?:['’][^\W_]+)*", str(text).lower()) if len(p) > 1))
    def learn(self, tool, args, rows):
        spec, page, kit = self.search_tools[tool], self.page(tool), self.kit(tool)
        if rows and not self.fields.get("price"): self.fieldmap()
        self.shelves(tool, rows)
        keys = [k for k in self.frees(tool) if isinstance(args.get(k), str) and not all(any(Util.flat(x) == Util.flat(args[k]) for x in self.products.get(p, {}).values() if isinstance(x, str)) for p in rows)]
        key = max(keys, key=lambda k: len([w for w in str(args[k]).split() if re.search(r"[a-z0-9]{2}", w, re.I)]), default=None)
        if key and not spec.get("text"): spec["text"] = key
        done, mine = self.searched[-1], self.pieces(args[key]) if key else []
        stems, others, whole = lambda pieces: [Util.root(p) for p in pieces], lambda args: {k: v for k, v in args.items() if k not in (key, page)}, lambda text: len(self.pieces(text)) == len(dict.fromkeys(re.findall(r"\d+[.,]\d+|[^\W_]+(?:['’][^\W_]+)*", str(text).lower())))
        for old, seen in zip(self.searched[:-1], self.search_rows[:-1]):
            if old["tool"] != tool or page in old["args"] or page in args: continue
            twin = bool(rows) and seen == rows and old["found"] == done["found"]
            done["continued"] = done["continued"] or twin
            if not key or others(old["args"]) != others(args) or None in (old["found"], done["found"]): continue
            a, b = stems(self.pieces(old["args"].get(key, ""))), stems(mine)
            common = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            if common and a != b and old["found"] != done["found"] and whole(old["args"].get(key, "")) and whole(args[key]) and common == min(len(a), len(b)): kit["votes"][(old["found"] < done["found"]) == (len(a) < len(b))] += 1
        if key and mine and done["found"] == 0 and not rows and all(self.moded(tool, k) for k in others(args)):
            names = {Util.root(w) for p in self.products.values() for f in ("title", "brand", "category") if isinstance(p.get(self.fields.get(f)), str) for w in re.findall(r"[^\W_]+", p[self.fields[f]].lower())}
            kit["votes"][0] += Util.root(mine[0]) in names
        votes = kit["votes"]
        kit["match"] = "all" if votes[0] > votes[1] else "any" if votes[1] > votes[0] else None
        self.surface()
        self.gauge()
    def fieldmap(self):
        self.mapped = getattr(self, "mapped", -1) + 1
        out = self.model(Prompts.FIELDS, {"records": [Util.clip(r, 60) for r in list(self.products.values())[self.mapped % 2: self.mapped % 2 + 3]]}, 300, (LUNA, SOL, DEEPSEEK)[self.mapped % 3]) or {}
        for key in [k for k in FIELDS if isinstance(out.get(k), str) and any(out[k] in r for r in self.products.values())]: self.fields[key] = out[key]
    def frees(self, tool):
        return [k for k, v in self.props(tool).items() if Util.kind(v) == "string" and "enum" not in v and k != self.page(tool)]
    def first(self, tool):
        schema = self.props(tool).get(self.page(tool), {})
        return next((v for v in (schema.get("minimum"), schema.get("default"), self.kit(tool)["first"]) if v in (0, 1) and not isinstance(v, bool)), None)
    def surface(self):
        out = {}
        for tool, spec in self.search_tools.items():
            kit, props = self.kit(tool), self.props(tool)
            page, size = props.get(spec.get("page"), {}), kit["rows"]
            top = None if Util.kind(page) == "string" else page.get("maximum")
            kit["end"] = max([kit["end"] or 0] + [s["read"] for s in self.searched if s["tool"] == tool and s["next"] is None and s["found"] and s["found"] > s["read"] >= (size or 1)]) or None
            caps = [props[k]["maxLength"] for k in self.frees(tool) if "maxLength" in props[k]]
            kit.update(paging=None if not page else "token" if Util.kind(page) == "string" else "offset" if spec.get("offset") else "page", chars=min(caps) if caps else None,
                       reach=kit["end"] or (None if top is None or not size else top + size if spec.get("offset") else (top - (self.first(tool) if self.first(tool) is not None else 1) + 1) * size))
            out[tool] = {k: v for k, v in kit.items() if v is not None and k not in ("first", "end", "votes")}
        self.world.update({k: self.kit(next((t for t, s in self.search_tools.items() if s.get("text")), next(iter(self.search_tools))))[k] for k in ("rows", "paging", "reach", "chars")})
        return {"tools": out, "order": self.world["order"], "filters": self.world["filters"], "cheapest_route": self.cheapest()}
    def cheapest(self):
        caps = {s["price_cap"] for s in self.search_tools.values() if s.get("price_cap")}
        marks = {k.split(" with ")[0]: v for k, v in self.world["filters"].items() if v == "filters"}
        if "price" in self.world["order"].values(): return "price order"
        if caps and not caps & {k.split(" with ")[0] for k, v in self.world["filters"].items() if v == "ignored"} - set(marks): return "price cap"
        return "shelf paging" if set(marks) - caps or any(s.get("labels") for s in self.search_tools.values()) else "entry queries"
    def label(self, tool, args):
        named = ["%s=%s" % (k, v) for k, v in sorted(args.items()) if self.moded(tool, k)]
        return " ".join(([tool] if len(self.search_tools) > 1 else []) + named) or "every search"
    def gauge(self):
        votes, marks, pages = {}, {}, []
        for done, ids in zip(self.searched, self.search_rows):
            tool, args = done["tool"], done["args"]
            spec, mode = self.search_tools[tool], self.label(tool, args)
            prices = [self.price(p) for p in ids]
            if len(ids) > 2 and ids not in pages:
                pages.append(ids)
                odds = {"id": math.lgamma(len(ids) + 1)} if ids == sorted(ids) else {}
                if None not in prices and prices == sorted(prices): odds["price"] = math.lgamma(len(ids) + 1) - sum(math.lgamma(prices.count(v) + 1) for v in set(prices))
                votes.setdefault(mode, []).append(odds)
            for key, value in args.items() if len(ids) > 1 else []:
                if key == spec.get("price_cap") and None not in prices and Util.number(value) is not None:
                    marks.setdefault(key + " with " + mode, []).append("filters" if max(prices) <= Util.number(value) + 1e-9 else "ignored")
                elif isinstance(value, str) and key != spec.get("page") and not self.moded(tool, key):
                    has = sum(any(Util.flat(x) == Util.flat(value) for x in self.products.get(p, {}).values() if isinstance(x, str)) for p in ids)
                    marks.setdefault(key + " with " + mode, []).append("filters" if has == len(ids) else "ignored" if has else "none")
        order = {}
        for mode, seen in votes.items():
            fits = [k for k in ("id", "price") if all(k in odds for odds in seen)]
            sure = [k for k in fits if sum(odds[k] for odds in seen) > ODDS]
            if len(sure) == 1 and len(fits) == 1 or not fits and self.fields.get("price"): order[mode] = sure[0] if fits else "relevance"
        self.world.update(order=order, filters={k: "ignored" if {"ignored", "none"} & set(m) else "filters" for k, m in marks.items() if m.count("filters") > m.count("none")})
    def sorted_by(self, done):
        return self.world["order"].get(self.label(done["tool"], done["args"]))
    def cursor(self, tool, args, obs, rows):
        spec = self.search_tools[tool]
        holder = next((d for d in Util.dicts(obs) if any(isinstance(v, list) and (not v or isinstance(v[0], dict)) for v in d.values())), None)
        if holder is None: return False, None, None
        scalar = lambda v: v is None or isinstance(v, (str, int, float)) and not isinstance(v, bool)
        sides = {**{k: v for d in holder.values() if isinstance(d, dict) for k, v in d.items() if scalar(v)}, **{k: v for k, v in holder.items() if scalar(v)}}
        nums = {k: v for k, v in sides.items() if isinstance(v, (int, float))}
        page, key = spec.get("page"), spec.get("next_key")
        if page and key not in sides:
            text = Util.kind(self.props(tool)[page]) == "string"
            now = None if text else Util.number(args.get(page))
            base = self.first(tool)
            hops, given = {(now or 0) + len(rows), (now if now is not None else 1 if base is None else base) + 1}, {json.dumps(v) for v in args.values()}
            tokens = [k for k, v in sides.items() if text and isinstance(v, str) and re.fullmatch(r"\S{1,80}", v) and re.search(r"\d", v) and json.dumps(v) not in given]
            steps = [k for k, v in nums.items() if not text and v in hops and (v != max(nums.values()) or len(nums) == 1 and None not in sides.values())]
            key = (tokens or steps or [None])[0]
            if key: spec["next_key"] = key
            elif None in sides.values(): key = next(k for k, v in sides.items() if v is None)
        count = spec.get("count_key") or max((k for k in nums if k != key), key=nums.get, default=None)
        if count and spec.get("next_key"): spec["count_key"] = count
        found = nums.get(count, 0 if not rows else None) if count else 0 if not rows else None
        return bool(page and key), sides.get(key), found
    def next_page(self, tool, args):
        schema = self.props(tool).get(page, {}) if (page := self.page(tool)) else {}
        if Util.kind(schema) not in ("integer", "number"): return None
        now = Util.number(args.get(page))
        value = (now or 0) + self.size(tool) if self.search_tools[tool].get("offset") else (now if now is not None else self.first(tool) or 0) + 1
        return int(value) if value <= schema.get("maximum", value) else None
    def mode(self, tool, args):
        return json.dumps(sorted((k, v) for k, v in args.items() if self.moded(tool, k)))
    def mark_modes(self, tool, args, rows, found):
        page = self.page(tool)
        if len(rows) < max(self.size(tool), 2) or page in args or found is not None and found <= self.size(tool): return
        mode = self.mode(tool, args)
        for done, seen in zip(self.searched[:-1], self.search_rows[:-1]):
            if done["tool"] != tool or seen != rows or done["found"] != found or page in done["args"] or self.mode(tool, done["args"]) != mode: continue
            diff = [k for k in args.keys() | done["args"].keys() if args.get(k) != done["args"].get(k)]
            if len(diff) == 1 and not Util.stems(str(args.get(diff[0]) or "")) & Util.stems(str(done["args"].get(diff[0]) or "")): self.ignored.add((tool, mode, diff[0]))
    def strip(self, tool, args):
        mode = self.mode(tool, args)
        return {k: v for k, v in args.items() if (tool, mode, k) not in self.ignored}
    def chain(self, done):
        page = self.page(done["tool"])
        return json.dumps([done["tool"], sorted((k, v) for k, v in self.strip(done["tool"], done["args"]).items() if k != page)])
    def capless(self, tool, args):
        key, mode = self.search_tools[tool].get("price_cap"), self.mode(tool, args)
        return any(s["tool"] == tool and self.mode(tool, s["args"]) == mode and Util.number(s["args"].get(key)) is not None and any((self.price(p) or 0) > Util.number(s["args"][key]) for p in rows) for s, rows in zip(self.searched, self.search_rows))

class Search:
    def first_search(self):
        allowed = max(1, min(len(kinds := [u["attribute"] for u in self.unsearched()]) - kinds.count("") * (kinds.count("") > 1), self.allowance()))
        out = self.model(Prompts.SEARCH, {"spec": self.brief(), "search_tools": self.search_specs(), "searches_allowed": allowed, "search_profile": self.surface()}, 500) or {}
        for tool, args in [e for e in map(self.search_entry, Util.items(out.get("searches"))[:allowed]) if e]:
            if not self.dry(tool, args): self.seek(tool, args)
    def family(self, tool, args):
        return json.dumps([self.label(tool, args), sorted(Util.stems(re.sub(r"\d", " ", self.keywords(tool, args))))])
    def keywords(self, tool, args):
        return " ".join(v for k, v in sorted(self.strip(tool, args).items()) if isinstance(v, str) and k != self.page(tool) and not self.moded(tool, k))
    def yields(self):
        out, seen = {}, set()
        for done, rows in zip(self.searched, self.search_rows):
            if self.page(done["tool"]) not in done["args"]: out[self.family(done["tool"], done["args"])] = sum(p not in seen and self.cat_ok.get(self.category(p)) is not False for p in rows)
            seen.update(rows)
        return out
    def dry(self, tool, args):
        return self.page(tool) not in args and self.yields().get(self.family(tool, args), 1) == 0
    def search_specs(self):
        return [{"name": n, **self.tools[n], **{k: s[k] for k in ("page", "price_cap") if s.get(k)}} for n, s in self.search_tools.items()]
    def search_entry(self, item, paged=False):
        if not isinstance(item, dict): return None
        tool, args = (item.get("tool"), item.get("args")) if "args" in item else (None, item)
        tool = tool if isinstance(tool, str) and tool in self.search_tools else next(iter(self.search_tools))
        schema, page, props, cut = self.tools[tool]["parameters"], self.page(tool), self.props(tool), lambda k, v: v[:props[k]["maxLength"] + 1].rsplit(" ", 1)[0] if isinstance(v, str) and len(v) > props.get(k, {}).get("maxLength", len(v)) else v
        args = Util.fit(schema, Util.fill({k: cut(k, v) for k, v in args.items()} if isinstance(args, dict) else args, None, self.qty, None))
        args = args if args is None else Util.fit(schema, self.strip(tool, {k: v for k, v in args.items() if k != page or paged}))
        if args is not None and (key := self.search_tools[tool].get("price_cap")) and not paged: args = Util.fit(schema, self.strip(tool, {**{k: v for k, v in args.items() if k != key}, **({} if (top := self.ceiling(tool)) is None else {key: top})}))
        if args is None or any(s["tool"] == tool and self.strip(tool, s["args"]) == args for s in self.searched): return None
        words, left = [v for k, v in args.items() if isinstance(v, str) and v.strip() and k != page and not self.moded(tool, k)], [Util.stems(u["value"]) for u in self.unsearched()] if self.frees(tool) and not paged else []
        return None if not words or left and any(s["tool"] == tool for s in self.searched) and not any(Util.stems(v) - Util.stems(self.product) or any(u <= Util.stems(v) for u in left) for v in words) else (tool, args)
    def run_search(self, tool, args, again=False):
        same = lambda given: json.dumps([tool, {k: Util.norm(v) if isinstance(v, str) else v for k, v in self.strip(tool, given).items()}], sort_keys=True)
        free, sign = self.room() if self.search_rows else self.hard - self.reserve(True) - self.step, same(args)
        if free <= 0 or self.late() and self.search_rows or sign in self.blank or any(s["tool"] == tool and same(s["args"]) == sign for s in self.searched): return None
        obs = self.act("search", tool=tool, args=args)
        page, link = self.page(tool), self.chain({"tool": tool, "args": args})
        chain = [s for s in self.searched if page in args and s["next"] == args[page] and self.chain(s) == link]
        for done in chain: done["continued"] = True
        mend = lambda why: None if again else self.remedy("search", "find products for %s; the search %s" % (self.product, why), obs, [(tool, args)], lambda name, given: self.run_search(name, given, True) if name in self.search_tools else None)
        if obs is None or self.last["fault"]:
            if obs is not None:
                self.searched.append({"tool": tool, "args": args, "rows": 0, "found": None, "read": 0, "more_pages": False, "next": None, "continued": True})
                self.search_rows.append([])
            return mend("failed on the first use of this tool") if obs is not None and tool not in self.shapes else None
        rows, kit = self.last.get("records", []), self.kit(tool)
        if not rows: self.blank.add(sign)
        self.page_size, kit["rows"] = max(self.page_size, len(rows)), max(kit["rows"] or 0, len(rows))
        field, value, found = self.cursor(tool, args, obs, rows)
        again = bool(rows) and not field and any(page not in s["args"] and r == rows for s, r in zip(self.searched, self.search_rows) if s in chain)
        if again: kit["first"] = Util.number(args[page])
        if not field: value = self.next_page(tool, args) if len(rows) >= self.size(tool) else None
        read = len(rows) * (not again) + max((s["read"] for s in chain), default=0)
        if value is not None and (found is not None and read >= found or Util.fit(self.props(tool)[page], value) != value): value = None
        self.searched.append({"tool": tool, "args": args, "rows": len(rows), "found": found, "read": read, "more_pages": value is not None, "next": value, "continued": False})
        self.search_rows.append(rows)
        self.mark_modes(tool, args, rows, found)
        self.learn(tool, args, rows)
        shown, top, seen = self.evidence(True), self.best(), {p for ids in self.search_rows[:-1] for p in ids}
        rivals = [] if self.objective == "cheapest" else [shown.get(p, frozenset()) for p in seen if self.live(p)]
        self.searched[-1]["gain"] = [p for p in dict.fromkeys(rows) if p not in seen and self.live(p) and self.under(p, top) and not any(s >= shown.get(p, frozenset()) for s in rivals)]
        if not rows and (found or isinstance(obs, list) and obs or any(isinstance(v, list) and v for d in Util.dicts(obs) for v in d.values())): return mend("showed results but no readable product id")
        if not self.products and self.over("search"): return mend("used its turn budget without finding any product")
        return rows, value is None if field else len(rows) < self.page_size
    def remedy(self, step, goal, raw, tried, run):
        fix = self.recover(step, goal, raw, [{"tool": t, "args": a} for t, a in tried])
        action, call = (fix.get("action"), fix.get("call")) if isinstance(fix, dict) else (None, None)
        if action == "map" and not self.remapped:
            self.remapped = True
            self.map_tools(["%s; tool %s returned %s; %s" % (goal, tried[-1][0], json.dumps(Util.clip(raw), ensure_ascii=False, default=str)[:900], fix.get("reason") or "")])
        if action == "finish": self.stop = self.step
        tool, args = (call.get("tool"), call.get("args")) if action == "call" and isinstance(call, dict) else tried[-1] if action == "retry" else (None, None)
        return run(tool, args) if tool in self.tools and isinstance(args, dict) else None
    def live(self, pid):
        return self.eligible(pid) and not any(self.verdict(pid, r) is False for r in self.reqs)
    def ceiling(self, tool):
        return None if (top := min([c / self.qty for c in [self.cap()] if c is not None] + [b for b in [self.best()] if b is not None], default=None)) is None or self.search_tools[tool].get("capped", {}).get(self.scale) is False else math.ceil(round(top, 6))
    def turn(self, done):
        tool, page, own, head, key = done["tool"], self.page(done["tool"]), Util.stems(self.product), next(s for s in self.searched if self.chain(s) == self.chain(done)), self.search_tools[done["tool"]].get("price_cap")
        top, cap, said, wanted = self.ceiling(tool), Util.number(done["args"].get(key)), Util.stems(self.keywords(tool, head["args"])), {Util.norm(v) for r in self.reqs if not r["exclude"] for v in r["values"]}
        if any(v not in wanted for sign in head.get("known", ()) for v in json.loads(sign)[1] if not json.loads(sign)[2] and Util.stems(v) - own and Util.stems(v) <= said): return None
        fresh = [Util.stems(v) - own for r in self.reqs if not r["exclude"] and r["min"] is None and r["max"] is None and Util.sign(r) not in head.get("known", [Util.sign(r)]) for v in r["values"] if Util.stems(v) - own and not Util.stems(v) - own <= said]
        if (carrier := next((s for s in self.searched[self.searched.index(head) + 1:] if fresh and s["tool"] == tool and s["next"] is not None and not s["continued"] and any(v <= Util.stems(self.keywords(tool, s["args"])) for v in fresh)), None)) and (entry := self.turn(carrier)): return entry
        if None not in (top, cap) and top > cap: return self.search_entry({"tool": tool, "args": {k: v for k, v in done["args"].items() if k != page}})
        return self.search_entry({"tool": tool, "args": {**done["args"], page: done["next"]}}, paged=True)
    def seek(self, tool, args):
        began, found, spec = len(self.searched), self.run_search(tool, args), self.search_tools[tool]
        cap, trust, modes, text = args.get(spec.get("price_cap")), spec.setdefault("capped", {}), {k: v for k, v in args.items() if self.moded(tool, k)}, spec.get("text")
        if found and cap is not None and trust.get(self.scale) is None:
            trust[self.scale] = bool(found[0])
            again = None if found[0] else self.search_entry({"tool": tool, "args": args})
            retry = self.run_search(*again) if again else None
            trust[self.scale] = trust[self.scale] or (None if retry is None else not any((self.price(p) or 0) <= cap for p in retry[0]))
            if retry: tool, args, found = *again, retry
        for entry in (lambda: self.relax(tool, args), lambda: self.search_entry({"tool": tool, "args": {**modes, text: self.product}}, paged=True) if text else None) if found is not None and not found[0] else ():
            entry = entry()
            if entry and not self.search_rows[-1] and (Util.stems(entry[1].get(text, "")) <= Util.stems(self.product) or not self.dry(*entry)): self.run_search(*entry)
        for done in self.searched[began:]: done["known"] = [Util.sign(r) for r in self.reqs]
    def relax(self, tool, args):
        props, page = self.props(tool), self.page(tool)
        if not (texts := [k for k, v in args.items() if isinstance(v, str) and len(v.split()) > 2 and k != page and "enum" not in props.get(k, {})]): return None
        words, own = args[key := max(texts, key=lambda k: len(args[k].split()))].split(), Util.stems(self.product)
        loose = [i for i in reversed(range(len(words))) if not Util.stems(words[i]) <= own]
        return next(filter(None, (self.search_entry({"tool": tool, "args": {**args, key: " ".join(words[:i] + words[i + 1:])}}) for i in loose if len(loose) > 1)), None)
    def fits(self, done):
        found, cheap = done["found"] or 0, self.objective == "cheapest" and self.sorted_by(done) == "id"
        left = (shown := min(found, self.kit(done["tool"])["reach"] or found)) - done["read"]
        return done["found"] is not None and self.sorted_by(done) != "relevance" and (2 * shown >= found if cheap else shown == found and left <= done["read"]) and -(-left // self.size(done["tool"])) <= self.allowance()
    def best(self):
        return min([self.price(p) for p in self.details if self.passes(p) and self.price(p) is not None], default=None) if self.objective == "cheapest" else None
    def pages(self):
        chains, top, hits, bare, doubt, cheap, members = [], self.best(), self.evidence(), not self.passing(), self.suspect()[0], self.objective == "cheapest", {}
        groups, promise = [g for g in ([v for v in map(Util.stems, r["values"]) if v - Util.stems(self.product)] for r in self.reqs if not r["exclude"]) if g], lambda rows: sum(self.eligible(p) and (not hits or p in hits) for p in rows) / max(len(rows), 1)
        for done, rows in zip(self.searched, self.search_rows): members.setdefault(self.chain(done), set()).update(rows)
        heads = [(Util.stems(self.keywords(s["tool"], s["args"])), s["found"]) for s in self.searched if s["found"] is not None and self.page(s["tool"]) not in s["args"]]
        for done, rows in zip(self.searched, self.search_rows):
            if done["next"] is None or done["continued"] or not rows: continue
            mine, order, said, fit = members[self.chain(done)], self.sorted_by(done), Util.stems(self.keywords(done["tool"], done["args"])), self.fits(done)
            fitting = [p for p in mine if self.eligible(p)]
            sharp = cheap and bool(groups and fitting) and order != "price" and done["found"] is not None and any(w < said and n > done["found"] for w, n in heads) and not mine & doubt and all(any(v <= self.texts.get(p, set()) for v in g) for p in fitting for g in groups) and -(-(min(done["found"], self.kit(done["tool"])["reach"] or done["found"]) - done["read"]) // self.size(done["tool"])) <= self.allowance()
            if top is not None and order == "price":
                if (self.price(rows[-1]) or 0) < top: chains.append((False, -2, 0, done, False))
            elif sharp or fit or any(map(self.eligible, done["gain"])) or (bare and any(self.cat_ok.get(self.category(p)) for p in rows) or self.objective != "any" and order == "id" and any(v and v <= Util.stems(" ".join(str(x) for x in done["args"].values() if isinstance(x, str))) for v in (Util.stems(v) for r in self.reqs if not r["exclude"] for v in r["values"]))) and promise(rows):
                chains.append((not (sharp or fit), -promise(rows), done["found"] - done["read"] if done["found"] is not None else self.page_size, done, fit and order == "id"))
        jobs, covered = [], set()
        for wide, _, _, done, lead in sorted(chains, key=lambda c: c[:3]):
            entry, rows = self.turn(done), members[self.chain(done)]
            if not entry or any(entry == e for _, e, _ in jobs) or cheap and 2 * len(rows & covered) >= len(rows): continue
            jobs.append((wide, entry, lead))
            covered |= rows
        return jobs
    def suspect(self):
        own, members, out, named = Util.stems(self.product), {}, set(), []
        for done, rows in zip(self.searched, self.search_rows): members.setdefault(self.chain(done), set()).update(rows)
        for req, group in [(r, [v for v in map(Util.stems, r["values"]) if v - own]) for r in self.reqs if not r["exclude"]]:
            hidden, seen = lambda p: not any(v <= self.texts.get(p, set()) for v in group), lambda rows, mark: sum(p in self.details and hidden(p) and self.verdict(p, req) is mark for p in rows)
            bad = {c for c, rows in members.items() if seen(rows, False) > 1 and not seen(rows, True)} if group and any(not hidden(p) and p not in self.details and self.eligible(p) for rows in members.values() for p in rows) else set()
            named += group * bool(bad)
            out |= {p for c in bad for p in members[c] if hidden(p) and all(k in bad for k, rows in members.items() if p in rows)}
        return out, named
    def chase(self, tool, args):
        for _ in range(LOOP):
            if self.busy() or not self.run_search(tool, args): return
            done, rows, top = self.searched[-1], self.search_rows[-1], self.best()
            gain, fit, order = [p for p in done["gain"] if self.eligible(p)], self.fits(done), self.sorted_by(done)
            if done["next"] is None or not fit and not gain or top is not None and order == "price" and (self.price(rows[-1]) or 0) >= top or self.objective == "cheapest" and not (fit and order == "id") and any(p not in self.details for p in gain) or not (entry := self.turn(done)): return
            tool, args = entry
    def unsearched(self):
        own, texts = Util.stems(self.product), [(Util.stems(self.keywords(s["tool"], s["args"])), any(self.cat_ok.get(self.category(p)) is not False for p in rows)) for s, rows in zip(self.searched, self.search_rows)]
        found, dead, left = lambda v: any(ok and v <= t and (t & own or not own) for t, ok in texts), lambda v: any(not ok and v <= t <= v | own for t, ok in texts), lambda pairs: [{"attribute": a, "value": v} for a, v in pairs if Util.stems(v) - own and not found(Util.stems(v)) and not dead(Util.stems(v))]
        stated, new, failed, say = [r for r in self.reqs if not r["exclude"] and r["min"] is None and r["max"] is None], lambda r: self.searched and not any(Util.sign(r) in s.get("known", [Util.sign(r)]) for s in self.searched), lambda r: self.details and not any(self.verdict(p, r) for p in self.details), lambda v: (lambda f: f and self.written(*f) or v)(Util.figure(v))
        values, groups = [(r["attribute"], say(v)) for r in stated for v in r["values"] if Util.stems(v) - own], [g for g in ([say(v) for v in r["values"] if Util.stems(v) - own and not dead(Util.stems(say(v)))] for r in stated) if g]
        both, wide = [("", " ".join(c)) for c in itertools.product(*groups)] * (len(groups) > 1), any(self.kit(t)["match"] == "any" for t in self.search_tools) or any(map(new, stated)) and not any(map(self.passes, self.details))
        hits, lead, bounds = self.evidence() if len(both) > 1 else {}, " ".join(g[0] for g in groups), [r for r in self.reqs if r not in stated and not r["exclude"] and (new(r) or failed(r))]
        if len(both) > 1 and self.room() <= len(both) + sum(p not in self.details and self.live(p) and (self.objective != "any" or hits.get(p, 0) >= len(groups)) for p in {p for ids in self.search_rows for p in ids}): both = [("", " ".join(g[0] for g in groups if len(g) == 1))] if sum(len(g) == 1 for g in groups) > 1 else both[:1]
        late = [(r["attribute"], lead + " " + next((say(v) for v in r["values"] if Util.number(v) == edge), self.written(edge, Util.flat(r["unit"])) or "%g" % edge)) for r in bounds if lead and None in (r["min"], r["max"]) for edge in [r["max"] if r["min"] is None else r["min"]]] + [(r["attribute"], r["attribute"]) for r in bounds if not values and failed(r)]
        return left(both + values if wide else values + both) or left(late)
    def cheaper(self, top):
        hits, pick = self.evidence(), self.passing()[:1]
        rows = dict.fromkeys([p for ids in self.search_rows for p in ids] + list(hits))
        title, twin = Util.stems(self.field(pick[0], "title") or "") if pick else set(), lambda p: len(title & Util.stems(self.field(p, "title") or "")) if self.price(p) == top and not self.backed() else -1
        failed = {(self.price(p), self.field(p, "title")) for p in self.details if any(self.verdict(p, r) is False for r in self.reqs)}
        return sorted([p for p in rows if p not in self.details and self.live(p) and self.under(p, top)], key=lambda p: (not (hits.get(p) or self.cat_ok.get(self.category(p)) is True), (self.price(p), self.field(p, "title")) in failed, -hits.get(p, 0), self.price(p) is None, self.price(p) or 0, -twin(p)))
    def routes(self):
        pick = self.passing()[:1]
        label = self.category(pick[0]) if pick else next((c for c, ok in self.cat_ok.items() if ok), self.product)
        wanted = not pick or self.objective != "any" and not any(self.sorted_by(s) == "price" for s in self.searched)
        bare = lambda tool, args: {k: v for k, v in self.strip(tool, args).items() if k not in (self.page(tool), self.search_tools[tool].get("price_cap"))}
        out = []
        for tool in self.search_tools if label and wanted else []:
            props, need, page, text = self.props(tool), self.tools[tool]["parameters"].get("required", []), self.page(tool), self.search_tools[tool].get("text")
            model = next((s["args"] for s in self.searched if s["tool"] == tool and text in s["args"]), None)
            modes, shelf = {k: v for k, v in (model or {}).items() if self.moded(tool, k)}, model is not None and self.capless(tool, model) and (self.objective == "cheapest" or not any(self.eligible(p) for ids in self.search_rows for p in ids if self.cat_ok.get(self.category(p))))
            if model is not None: options = [{**modes, text: label}] + [{**modes, key: other, free: label} for free in [k for k in self.frees(tool) if k != text][:1] if shelf for key in modes for other in props[key]["enum"] if other != modes[key]]
            elif any(s["tool"] == tool for s in self.searched): continue
            else:
                fill = lambda prop: next(iter([e for e in prop["enum"] if Util.flat(e) == Util.flat(label)] or [e for e in prop["enum"] if isinstance(e, str) and Util.holds(label, e)]), None) if prop.get("enum") else label if Util.kind(prop) == "string" else None
                options = [{k: fill(props.get(k, {})) for k in [k for k in need if k != page] or [k for k in props if k != page and fill(props[k]) is not None][:1]}]
            for entry in [self.search_entry({"tool": tool, "args": args}) for args in options if args and None not in args.values()]:
                if entry and not any(s["tool"] == tool and bare(tool, s["args"]) == bare(*entry) for s in self.searched): out.append(entry)
        return sorted(out, key=lambda e: self.objective == "cheapest" and (self.search_tools[e[0]].get("price_cap") not in e[1] or self.capless(*e)))
    def explore(self, accepted):
        deep, wide = bool(accepted) and self.objective != "any", not accepted or self.objective != "any"
        if wide and self.target(): return 1
        top, turns, cheap = self.best(), self.pages() if wide else [], self.objective == "cheapest"
        opens, paging, (doubt, named) = self.cheaper(top) if cheap else [], [e for _, e, _ in turns], self.suspect() if cheap else (set(), [])
        searches = paging + self.routes() if wide else []
        if not accepted and not opens: opens += self.fresh()
        opens = sorted([p for p in dict.fromkeys(p for p in opens if isinstance(p, str)) if p in self.products and p not in self.details and self.eligible(p) and (top is None or self.under(p, top))], key=doubt.__contains__)
        opening, searching, hits = [("open", p) for p in opens[:self.allowance()]], [("chase" if deep and e in paging else "search",) + e for e in searches], self.evidence()
        lead = bool(opens) and hits.get(opens[0], 0) >= max(hits.values(), default=1)
        jobs = searching[:1] + opening + searching[1:] if not accepted and not lead and (self.lost or self.revealed or not any(self.cat_ok.values())) else opening + searching
        if cheap: jobs = [("chase",) + e for _, e, first in turns if first][:1] or [("search",) + e for e in paging if top is None and self.allowance() > 1 and min(opens, key=lambda p: self.price(p) or 0, default=None) in doubt and any(v <= Util.stems(self.keywords(*e)) for v in named)][:1] + [j for j in opening if j[1] not in doubt] or searching[:1] or opening
        done = 0
        for job in jobs[:self.allowance()]:
            if self.busy(): break
            if job[0] == "search" and self.dry(*job[1:]): continue
            if job[0] == "open": self.open_item(job[1])
            elif job[0] == "chase": self.chase(*job[1:])
            else: self.seek(*job[1:])
            done += 1
        return done
    def target(self, needs=None):
        model = next((s for s in self.searched if self.search_tools[s["tool"]].get("text") in s["args"]), None)
        for need in ([n["value"] for n in self.unsearched()] + self.linked() if needs is None else needs) if model else []:
            entry, before = self.search_entry({"tool": model["tool"], "args": {**{k: v for k, v in model["args"].items() if self.moded(model["tool"], k)}, self.search_tools[model["tool"]]["text"]: self.wording(need)}}), self.step
            if entry and not self.busy() and not self.dry(*entry) and (self.seek(*entry) or self.step > before): return 1
        return 0
    def linked(self):
        said, lead = [Util.stems(self.keywords(s["tool"], s["args"])) for s in self.searched], " ".join(r["values"][0] for r in self.reqs if r["values"] and not r["exclude"] and r["min"] is None and r["max"] is None)
        words = [w for l in self.links if l["item"] == self.at for a in self.within(l["on"], self.passing)[:1] for known in [((l["quote"], a) in self.fitted or self.tie(l, a, [], [a])) and self.fitted.get((l["quote"], a)) or {}] if known.get("applies") for w in known.get("words", [])]
        return [w for w in dict.fromkeys(words + [w + " " + lead for w in words if lead]) if not any(Util.stems(w) <= t for t in said)]
    def fitting(self, pids):
        links = [l for l in self.links if l["item"] == self.at]
        groups = {l["on"]: self.within(l["on"], self.passing)[:JUDGE_BATCH // 2] for l in links}
        known = {l["on"] for l in links if any(self.fitted.get((l["quote"], a, p), 0) >= 1 for a in groups[l["on"]] for p in pids)}
        for link in [l for l in links if pids and groups[l["on"]] and l["on"] not in known]: self.suits(link, groups[link["on"]][0], pids, groups[link["on"]])
        return [p for p in pids if all(any(map(all, [v for v in ([self.fitted.get((l["quote"], a, p)) for l in links if l["on"] == on] for a in groups[on]) if None not in v] or [[on not in known]])) for on in groups)]
    def written(self, number, unit):
        ways = [not gap for p in dict.fromkeys(p for ids in self.search_rows for p in ids) for n, gap in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)([\s-]?)%s(?![a-z])" % unit, " ".join(v for v in self.products.get(p, {}).values() if isinstance(v, str)).lower()) if Util.number(n) != number] if re.fullmatch(r"[a-z]{2,}", unit) else []
        return ("%g" % number + unit if 2 * sum(ways) >= len(ways) else "%g %s" % (number, unit)) if ways else None
    def wording(self, need):
        return need + " " + self.product if Util.stems(need) - Util.stems(self.product) else self.product
    def uncovered(self, accepted):
        if not accepted: return True
        if self.objective == "any" and (len(accepted) > 1 or any(s.get("low") is None for i, s in enumerate(self.items) if i != self.at)): return False
        return bool(self.unsearched() or self.pages() or self.routes() or self.objective == "cheapest" and self.cheaper(self.best()) or [p for p in self.prefer(self.fresh()) if self.under(p, self.best())])
    def covered(self):
        return len(shelf := self.opened()) > 1 and self.keys(shelf[-1:]) <= self.keys(shelf[:-1])
    def settle(self, mode=0):
        for _ in range(LOOP):
            accepted = self.fitting(self.judge())
            if mode: self.stage()
            if self.busy() or mode < 2 and (accepted or not mode and self.covered()) or not self.uncovered(accepted): return
            if (before := (self.step, len(self.details))) and not self.explore(accepted) and not self.drill(self.best()) or (self.step, len(self.details)) == before: return
    def drill(self, top):
        turns, before = self.pages(), self.step
        fresh = lambda: [(p,) for p in self.prefer([p for p in self.fresh() if self.under(p, top)])[:self.allowance()]]
        for go, entries in ((self.run_search, lambda: [e for wide, e, _ in turns if not wide]), (self.target, lambda: [()]), (self.seek, lambda: self.routes()), (self.open_item, fresh), (self.run_search, lambda: [e for wide, e, _ in turns if wide])):
            for entry in entries():
                if self.busy(): return int(self.step > before)
                go(*entry)
                if self.step > before and go != self.open_item: return 1
            if self.step > before: return self.step - before
        return 0


class Judge:
    def judge(self):
        self.drain("notices")
        self.rescale()
        cats = sorted({c for c in map(self.category, self.products) if c})[:40]
        if set(cats) <= set(self.cat_ok): cats = []
        elif any(Util.same(c, self.product) for c in cats): self.cat_ok, cats = {c: Util.same(c, self.product) for c in cats}, []
        best, like = self.objective == "best" and self.best_by, ""
        numbered = {str(i + 1): r for i, r in enumerate(self.reqs)}
        unclear, coded, marks = {}, {}, {p: self.mark(p) for p in self.details}
        self.sort([p for p in self.details if self.eligible(p)], numbered, marks, unclear, coded)
        if best:
            found = {p: [(Util.figure(v), v) for k, v in self.attributes(p).items() if Util.same(k, self.best_by)] for p in self.details}
            found = {p: f[0] for p, f in found.items() if len(f) == 1 and f[0][0]}
            units = [f[1] for f, _ in found.values()]
            unit = max(units, key=units.count) if units else None
            like = next((v for f, v in found.values() if f[1] == unit), "")
            self.measure.update({marks[p]: f[0] for p, (f, _) in found.items() if f[1] == unit})
        measure_for = [p for p in self.details if best and marks[p] not in self.measure and self.eligible(p)]
        todo = list(dict.fromkeys(list(unclear) + measure_for))
        extra = []
        try:
            for pid in [p for p in self.fresh() if p in self.cached and not self.foreign(p)] if todo or cats else []:
                if len(todo) and len(todo) % JUDGE_BATCH == 0: break
                self.details[pid] = self.cached[pid]
                marks[pid] = self.mark(pid)
                extra.append(pid)
                self.sort([pid], numbered, marks, unclear, coded)
                if best and marks[pid] not in self.measure: measure_for.append(pid)
                if pid in unclear or pid in measure_for: todo.append(pid)
            self.examine(todo, cats, numbered, marks, unclear, coded, measure_for, like)
        finally:
            for pid in extra: self.details.pop(pid, None)
        for (pid, key), verdict in coded.items(): self.judged.setdefault((*marks[pid], Util.sign(numbered[key])), verdict)
        self.inherit()
        return self.passing()
    def sort(self, pids, numbered, marks, unclear, coded):
        for pid in pids:
            for key, req in numbered.items():
                if (*marks[pid], Util.sign(req)) in self.judged: continue
                verdict = self.decide(pid, req)
                if verdict is not None and self.notes(pid): coded[(pid, key)], verdict = verdict, None
                if verdict is None: unclear.setdefault(pid, []).append(key)
                else: self.judged[(*marks[pid], Util.sign(req))], self.decided["code"] = verdict, self.decided["code"] + 1
    def examine(self, todo, cats, numbered, marks, unclear, coded, measure_for, like):
        best, sized = self.objective == "best" and self.best_by, []
        for start in range(0, max(len(todo), 1 if cats else 0), JUDGE_BATCH):
            chunk = todo[start:start + JUDGE_BATCH]
            data = {"product": self.product, "requirements": numbered, "categories": cats if start == 0 else [], "products": {p: {"name": self.field(p, "title"), "attributes": self.attributes(p), "notes": self.notes(p), "check": unclear.get(p, [])} for p in chunk}}
            if best: data.update(measure=self.best_by, measure_like=like, measure_for=[p for p in chunk if p in measure_for])
            out = self.model(Prompts.VERIFY, data, 3000, DEEPSEEK)
            if out is None: break
            if data["categories"]:
                chosen = set(Util.listed(out.get("categories", out.get("category")))) & set(cats)
                self.cat_ok.update({c: c in chosen if chosen else None for c in cats})
            results = out.get("products") if isinstance(out.get("products"), dict) else {}
            for pid in chunk:
                result = results.get(pid) if isinstance(results.get(pid), dict) else {}
                if best and pid in measure_for and pid in results: self.measure[marks[pid]] = Util.number(result.get("measure"))
                for key in unclear.get(pid, []):
                    if not isinstance(result.get(key), dict): continue
                    sign, source = Util.sign(numbered[key]), result[key].get("source")
                    borrowed = isinstance(source, dict) and bool(Util.listed(source.get("seller")) or Util.described(source))
                    if borrowed: self.sources[(*marks[pid], sign)] = {k: source.get(k) for k in ("seller", "product", "facts")}
                    code, quote = coded.get((pid, key)), Util.norm(result[key].get("quote") or "")
                    named = any(Util.holds(quote, w) for w in numbered[key]["values"]) or not numbered[key]["values"] and re.search(r"\d", quote)
                    denied = code is True and result[key].get("pass") is False and bool(named) and quote in Util.norm(" ".join(self.notes(pid))) and quote not in self.shown(pid)
                    self.judged[(*marks[pid], sign)] = None if borrowed else False if denied else code if code is not None else self.check(pid, numbered[key], self.agree(marks[pid], numbered[key], result[key]))
                    self.judged[(*marks[pid], "~" + sign)] = None if borrowed or code is not None else self.quoted(pid, numbered[key], result[key])
                    self.decided["model"] += 1
                    text = next((str(v) for v in [result[key].get("quote")] + [v for k, v in self.attributes(pid).items() if Util.flat(k) == Util.flat(result[key].get("key") or "")] if re.search(r"\d", str(v or ""))), None)
                    if text and result[key].get("key") and not borrowed and code is None and (numbered[key]["min"] is not None or numbered[key]["max"] is not None): sized.append(((*marks[pid], sign), text, numbered[key]))
        asked = [s for s in sized if (Util.figure(s[1]) or (0, "x"))[1]]
        out = self.model(Prompts.UNITS, {"values": [{"value": v, "unit": r["unit"]} for _, v, r in asked]}, 20 + 12 * len(asked), DEEPSEEK) if asked else None
        numbers = {**{m: Util.figure(v)[0] for m, v, _ in sized if Util.figure(v) and not Util.figure(v)[1]}, **dict(zip([m for m, _, _ in asked], map(Util.number, out["numbers"] if isinstance(out, dict) and isinstance(out.get("numbers"), list) and len(out["numbers"]) == len(asked) else [])))}
        for mark, _, req in sized:
            if numbers.get(mark) is not None: self.judged[mark] = (req["min"] is None or round(numbers[mark], 6) >= req["min"]) and (req["max"] is None or round(numbers[mark], 6) <= req["max"])
    def notes(self, pid, record=None):
        skip = {self.idk} | {self.fields.get(k) for k in ("title", "price", "stock")}
        rows = [r for r in (self.details.get(pid, {}) if record is None else record, self.products.get(pid, {})) if isinstance(r, dict)]
        return list(dict.fromkeys(Util.clip(v, 400) for r in rows for shown in [skip | set(self.attributes(pid, r))] for k, v in r.items() if isinstance(v, str) and v.strip() and k not in shown))
    def inherit(self):
        waiting = lambda p: [r for r in self.reqs if (*self.mark(p), Util.sign(r)) in self.sources and self.verdict(p, r) is None]
        pending = [p for p in self.details if self.eligible(p) and waiting(p) and all(self.verdict(p, r) is not False for r in self.reqs)]
        for pid in self.rank(pending):
            top = self.passing()[:1]
            if top and (self.objective == "any" or self.rank([pid] + top)[0] != pid) or self.busy(): return
            for req in waiting(pid):
                key = (*self.mark(pid), Util.sign(req))
                values = self.lookup(self.sources[key], req["attribute"])
                if values is None: return
                self.judged[key] = self.borrow(req, values)
                if self.judged[key] is False: break
    def lookup(self, target, attribute):
        key = json.dumps([target, Util.norm(attribute)], sort_keys=True, default=str)
        if key not in self.inherited:
            if self.busy() or not self.searcher(): return None
            self.inherited[key] = self.aside(lambda job: Util.listed(self.follow({"listing": target, "attribute": attribute}, job).get("value")), set(self.details))
        return self.inherited[key]
    def borrow(self, req, values):
        found = [Util.meets(v, req, self.product) for v in values]
        return None if not values else (True in found) is not req["exclude"] if True in found or None not in found else self.rejudge(req, {req["attribute"]: ", ".join(values)})
    def rejudge(self, req, attributes, notes=(), name=DEEPSEEK):
        out = self.model(Prompts.VERIFY, {"product": self.product, "requirements": {"1": req}, "categories": [], "products": {"source": {"name": "", "attributes": attributes, **({"notes": notes} if notes else {}), "check": ["1"]}}}, 300, name) or {}
        found = [r.get("1") for r in [out.get("products")] if isinstance(r, dict) for r in [r.get("source")] if isinstance(r, dict)]
        return found[0]["pass"] if found and isinstance(found[0], dict) and isinstance(found[0].get("pass"), bool) else None
    def agree(self, mark, req, result):
        key, quote, sign = Util.flat(result.get("key") or ""), Util.norm(result.get("quote") or ""), Util.sign(req)
        if not isinstance(result.get("pass"), bool) or not key or not quote or req["min"] is not None or req["max"] is not None: return result
        ev = (json.dumps([key, quote]), None, sign)
        first, self.judged[(*mark, "@" + sign)] = self.judged.setdefault(ev, result["pass"]), ev[0]
        if (first is not result["pass"] or key == "notes" and first and not any(Util.holds(quote, w) for w in req["values"])) and (ev[0], "", sign) not in self.judged:
            self.judged[(ev[0], "", sign)] = verdict = self.rejudge(req, {} if key == "notes" else {result["key"]: result["quote"]}, [result["quote"]] * (key == "notes"), SOL)
            for held in [k[:2] for k, v in self.judged.items() if verdict is not None and k[2] == "@" + sign and v == ev[0] and k[:2] != mark and k[0] in self.details and k[:2] == self.mark(k[0])]: self.judged[(*held, sign)] = self.check(held[0], req, {**result, "pass": verdict})
            first = self.judged[ev] = first if verdict is None else verdict
        return {**result, "pass": first}
    def passing(self):
        return self.rank([p for p in self.details if self.passes(p)])
    def rank(self, pids):
        level = {p: self.proof(p) for p in pids}
        if self.objective == "best" and self.best_by:
            measure = {p: self.measure.get(self.mark(p)) for p in level}
            return sorted(level, key=lambda p: (level[p][:1] > (1,), measure[p] is None, -(measure[p] or 0) if self.best_high else measure[p] or 0, level[p], self.price(p) or 0))
        if self.objective == "cheapest": return sorted(level, key=lambda p: (self.price(p) is None, self.price(p) or 0, level[p]))
        return sorted(level, key=lambda p: (level[p], self.price(p) is None, self.price(p) or 0))
    def passes(self, pid):
        return pid in self.details and self.eligible(pid) and all(self.verdict(pid, r) is True for r in self.reqs)
    def attributes(self, pid, record=None):
        drop = {self.idk} | {self.fields.get(k) for k in ("title", "description", "price", "stock")}
        out = {k: Util.clip(v, 100) for k, v in Util.attrs(self.details.get(pid, {}) if record is None else record, {}).items() if k not in drop}
        if self.category(pid): out.setdefault("category", self.category(pid))
        if isinstance(self.field(pid, "brand"), str) and self.field(pid, "brand").strip(): out.setdefault("brand", self.field(pid, "brand"))
        return out
    def sheet(self, pid):
        record = self.details.get(pid) or self.cached.get(pid) or self.products.get(pid) or {}
        return {"name": self.field(pid, "title"), "attributes": self.attributes(pid, record), "notes": self.notes(pid, record)}
    def own(self, pid, name):
        named, full = {w for w in Util.stems(name) if len(w) > 2}, Util.stems(name) | Util.stems(self.product)
        words = lambda k: Util.spelled(k, full)
        found = self.attributes(pid).items()
        split = [(Util.keyname(k), v) for k, v in found]
        keys = [(Util.norm(k), v) for k, v in found if named <= words(k)]
        return [(k, v) for k, v in split if words(k) == Util.stems(name)] or [(k, v) for k, v in split if named <= words(k) <= full] or [(k, v) for k, v in keys if all(len(w) <= 2 for w in words(k) - full)] or [(k, v) for k, v in keys if len({w for w in words(k) - full if len(w) > 2}) <= 2]
    def decide(self, pid, req):
        own = self.own(pid, req["attribute"])
        if req["exclude"] and self.stated(pid, req, not own): return False
        if not req["exclude"] and (ranged := [Util.ranged(" ".join([self.shown(pid)] + self.notes(pid)), w) for w in req["values"]]) and (True in ranged or None not in ranged): return True in ranged
        if not own: return None
        found = [Util.meets(v, req, self.product) for _, v in own]
        if req["exclude"]: return False if True in found else None if None in found or not all(Util.figure(v) for _, v in own) else True
        if True in found: return False if False in found else None if any(f is None and not {Util.norm(x) for x in v.split(",")} <= {Util.norm(x) for (_, w), g in zip(own, found) if g for x in w.split(",")} for (_, v), f in zip(own, found)) or self.contradicts(pid, req) else True
        return None if None in found else False
    def check(self, pid, req, result):
        own, key, quote = self.own(pid, req["attribute"]), Util.flat(result.get("key") or ""), Util.norm(result.get("quote") or "")
        values = [v for _, v in own] if own and req["values"] and not req["exclude"] and key not in {Util.flat(k) for k, _ in own} | {"notes", "category"} and not Util.stems(result.get("key") or "") & Util.stems(req["attribute"]) else [v for k, v in own + list(self.attributes(pid).items()) if Util.flat(k) == key] or [quote] * (key == "notes" and bool(quote) and quote in Util.norm(" ".join(self.notes(pid))))
        if not own and not req["exclude"] and not (values and Util.stems(result.get("key") or "") & Util.stems(req["attribute"])) and self.stated(pid, req): return True
        texts = [v for _, v in own] or values
        if req["exclude"]: return result.get("pass") is True and bool(texts) if result.get("pass") is not False or not texts or any(Util.meets(v, req, self.product) for v in texts) else (self.judged[mark] if (mark := (json.dumps(texts), "?", Util.sign(req))) in self.judged else self.judged.setdefault(mark, self.rejudge(req, {req["attribute"]: ", ".join(texts)}, (), SOL))) is True
        items, name = [x for v in values for x in [v] + v.split(",")], req["attribute"] + " " + self.product
        if result.get("pass") is not True or not values or req["values"] and not any(re.search(r"\d", w) or Util.related(v, w, req["attribute"]) for v in values for w in req["values"]): return False
        shape = 0 if not req["values"] or any(Util.alike(x, w, name) or Util.holds(x, w) and Util.variant(x, w, name) < 2 for x in items + [quote] * (quote in self.shown(pid)) for w in req["values"]) else max(Util.variant(x, w, name) for x in items for w in req["values"])
        return shape == 0 or (set(Util.norm(" ".join(values)).split()) - set(Util.norm(" ".join(req["values"]) + " " + name).split()) <= set(Util.norm(self.query).split()) if shape == 1 else False if any(Util.holds(x, w) for x in items for w in req["values"]) else None)
    def shown(self, pid):
        return Util.norm(" ".join([str(self.field(pid, "title") or "")] + [k + " " + v for k, v in self.attributes(pid).items()]))
    def quoted(self, pid, req, result):
        quote, bound = Util.norm(result.get("quote") or ""), req["min"] is not None or req["max"] is not None
        if result.get("loose") is not True or not quote or quote not in self.shown(pid) or bound and not re.search(r"\d", quote): return None
        return True if req["exclude"] or bound or any(re.search(r"\d", w) or Util.related(quote, w, "") for w in req["values"]) else None
    def stated(self, pid, req, keyed=True):
        return any(Util.spans([*map(Util.singular, Util.norm(self.field(pid, "title") or "").split())], " ".join(map(Util.singular, Util.norm(w).split()))) for w in req["values"]) or keyed and any(Util.flat(v) == Util.flat(w) for v in self.attributes(pid).values() for w in req["values"] if len(Util.flat(w)) > 3)
    def grade(self, value, wanted, name):
        (a, b), words = map(Util.figure, (value, wanted)), lambda text, form=Util.root: [form(w) for w in (w.strip(".,;:!?()[]'\"") for w in str(text).lower().split()) if w]
        return 0 if words(value, str) == words(wanted, str) else 1 if a and b and a[1] != b[1] or words(value) == words(wanted) or [w for w in words(value) if w not in words(name)] == [w for w in words(wanted) if w not in words(name)] != [] else 2
    def proof(self, pid):
        full, levels = Util.stems(self.product), []
        for req in self.reqs:
            names = Util.stems(req["attribute"]) | full
            plain = [v for k, v in self.own(pid, req["attribute"]) if Util.spelled(k, names) <= names]
            if req["exclude"]: levels.append(0 if plain else 2)
            elif req["min"] is not None or req["max"] is not None: levels.append(min([1 - bool(f and (not req["unit"] or f[1] == Util.flat(req["unit"]))) for v in plain for f in [Util.figure(v)] if f or re.search(r"\d", v)], default=2))
            else: levels.append(min([self.grade(x, w, req["attribute"] + " " + self.product) for v in plain for x in [v] + v.split(",") for w in req["values"]], default=2))
        return tuple(sorted(levels, reverse=True))
    def contradicts(self, pid, req):
        title = self.field(pid, "title") or ""
        if req["exclude"] or req["min"] is not None or req["max"] is not None or any(Util.holds(title, w) for w in req["values"]): return False
        return any(len(Util.flat(x)) > 2 and Util.holds(title, x) and not any(Util.holds(x, w) or Util.holds(w, x) or Util.alike(x, w, req["attribute"]) for w in req["values"]) for x in {Util.norm(x) for p in self.details if p != pid for _, v in self.own(p, req["attribute"]) for x in str(v).split(",")})
    def loose(self, pid, req):
        if req["exclude"]: return False
        own = self.own(pid, req["attribute"])
        if req["min"] is not None or req["max"] is not None: return bool(own) and self.verdict(pid, req) is not False and all(Util.meets(v, req, self.product) is None for _, v in own)
        text = " ".join([str(self.field(pid, "title") or ""), str(self.field(pid, "description") or "")] + list(self.attributes(pid).values()))
        return not own and any(Util.holds(text, w) for w in req["values"]) and not self.contradicts(pid, req)
    def nearest(self):
        near = lambda pid, req: self.verdict(pid, req) is True or (*self.mark(pid), Util.sign(req)) not in self.sources and (self.ruled(pid, "~" + Util.sign(req)) is True or self.loose(pid, req))
        return self.rank([p for p in self.details if self.eligible(p) and all(near(p, r) for r in self.reqs)])

class Ask:
    def consult(self):
        sent = 0
        if "message" in self.roles and self.allowed != 0: self.restock()
        for _ in range(LOOP if self.plans else 0):
            if self.left() <= 0 or self.late() or self.hard - self.step <= self.reserve(): break
            if not self.pick() and not (sent and self.restock()) or not self.ask(): break
            sent += 1
        self.consulted = True
    def pick(self):
        heads = [(self.worth(index), -index) for index, plan in self.plans.items() if plan]
        if heads: self.focus(-max(heads)[-1])
        return bool(heads)
    def worth(self, index):
        odds, covered = AGAIN if self.items[index]["revealed"] else HIDDEN, sum(a["share"] for a in self.items[index]["asked"] if a["share"] is not None)
        return odds * self.plans[index][0][0] / (1 - odds * (1 - min(covered, 1)))
    def ask(self):
        at, sent = self.at, None
        if "message" in self.roles and self.left() > 0 and self.hard - self.step > self.reserve():
            if not self.plans.get(at): self.plans[at] = self.chart()
            if self.pick() and (sent := self.send()): self.heed(self.asked[-1:])
            if self.at != at: self.focus(at)
        return sent
    def restock(self):
        pending, wide = list(range(len(self.items))), set()
        for index in pending:
            self.focus(index)
            if any(map(self.foreign, self.fresh())): self.judge()
        for _ in range(LOOP):
            if not pending: break
            for index in list(pending):
                self.focus(index)
                done, drilled = self.gather(1, index in wide)
                if drilled: wide.add(index)
                if done: pending.remove(index)
        for index in range(len(self.items)):
            self.focus(index)
            self.judge()
            self.plans[index] = self.choose(*self.shelf())
        return any(self.plans.values())
    def heed(self, asked):
        lines = [{"question": a["question"], "reply": a["reply"]} for a in asked]
        if lines and self.recheck(lines, "replies to the agent's questions") is None: self.pending += lines
    def send(self):
        plan = self.plans[self.at]
        pick = next((n for n, (_, prop) in enumerate(plan) if (prop is None and "budget" not in {a["property"] for s in self.items for a in s["asked"]} or prop and not self.taken(prop)) and self.permitted(self.phrase(prop))), None)
        if pick is None or self.over("ask") and (self.recover("ask", "ask the shopper one more question about one property though the messages planned for asking are used", json.dumps(self.talk()[-1:], ensure_ascii=False)) or {}).get("action") not in ("retry", "call"): return None
        entry = plan.pop(pick)
        text, tool = self.phrase(entry[1]), self.roles["message"]["tool"]
        obs, fix = self.act("message", text=text), {}
        if not (self.dead or self.finished or isinstance(obs, dict) and obs.get("error") == "budget_exceeded" or obs is not None and not self.last["fault"]):
            fix = self.recover("ask", "send the shopper this question and read the reply (need, no preference, confused, reference, budget or other): " + text, "the call was not sent" if obs is None else json.dumps(obs, ensure_ascii=False), [{"tool": tool, "args": Util.fill(self.roles["message"]["args"], None, None, text)}]) or {}
            call = fix.get("call") if isinstance(fix.get("call"), dict) else {}
            if fix.get("action") in ("retry", "call") and isinstance(call.get("args"), dict):
                obs = self.act("message", tool=call["tool"] if call.get("tool") in self.tools and call["tool"] not in {self.roles[r]["tool"] for r in PLAN if r in self.roles} else tool, args=call["args"])
                text = next((v for v in call["args"].values() if isinstance(v, str) and v.strip()), text)
        if obs is None: return None
        self.asked.append({"property": entry[1] or "budget", "share": entry[0] if entry[1] else None, "item": self.product, "question": text, "reply": self.last.get("lines") or Util.clip(obs, 300)})
        if isinstance(obs, dict) and obs.get("error") == "budget_exceeded" or fix.get("action") == "finish": self.allowed = self.questions = self.sent()
        return None if fix.get("action") == "finish" else entry
    def phrase(self, prop):
        line = " for the " + self.product if self.product else ""
        if prop is None: return "What is your budget" + (line if len(self.items) == 1 else " for the whole order") + "?"
        return "Which " + Util.singular(prop) + " would you like" + line + "?"
    def spread(self):
        return sum(o["listings"] * min(len(o["values"]), EXAMPLES) for o in self.shelf()[0] if len(o["values"]) > 1)
    def chart(self):
        if any(map(self.foreign, self.fresh())): self.judge()
        self.gather(LOOP, False)
        self.judge()
        return self.choose(*self.shelf())
    def gather(self, quota, wide):
        for _ in range(LOOP):
            fresh = self.prefer([p for p in self.fresh() if not self.foreign(p)]) or (wide and self.costly())
            paid = not (fresh and fresh[0] in self.cached)
            if paid and quota <= 0: return False, wide
            if self.late() or paid and (self.covered() or self.room() <= 0) or not fresh and wide: break
            quota -= paid
            self.open_item(fresh[0]) if fresh else self.drill(self.best())
            wide = wide or not fresh
            if fresh and fresh[0] in self.details and fresh[0] not in self.opened(): break
        return True, wide
    def costly(self):
        return sorted([p for p in self.products if p not in self.details and self.cat_ok.get(self.category(p)) and not self.foreign(p) and p not in self.gone and p not in self.vetoed and (not self.in_stock or self.stocked(p) is not False) and self.price(p) is not None and self.cap() is not None and self.price(p) * self.qty > self.cap()], key=self.price)
    def opened(self, pids=None):
        return self.prefer([p for p in (self.details if pids is None else pids) if self.cat_ok.get(self.category(p)) is not False and not self.foreign(p)])
    def foreign(self, pid):
        return self.cat_ok.get(self.category(pid)) is not True and any(i != self.at and s["cat_ok"].get(self.category(pid)) is True for i, s in enumerate(self.items))
    def keys(self, pids):
        return {Util.keyname(k) for p in pids for k in Util.attrs(self.details[p], {})}
    def shelf(self):
        shown = {k for p in self.products if p not in self.details for k in self.products[p]} | {self.idk} | set(self.fields.values())
        opened = shelf if len(shelf := self.opened()) > 1 else self.opened({**self.details, **self.cached})
        fit = [p for p in opened if all(self.verdict(p, r) is not False for r in self.reqs)]
        known = lambda p: Util.attrs({k: v for k, v in (self.details[p] if p in self.details else self.cached[p]).items() if k not in shown or isinstance(v, dict)}, {})
        facts = {p: {Util.keyname(k): Util.norm(v) for k, v in known(p).items() if Util.norm(v)} for p in (fit if len(fit) > 1 else opened)[:SHELF]}
        keys, groups = {key: {p: found[key] for p, found in facts.items() if key in found} for row in facts.values() for key in row}, []
        for key, seen in sorted(keys.items(), key=lambda kv: -len(kv[1])):
            if 2 * sum(sum(1 for o, w in facts[p].items() if o != key and len(w) > 2 and Util.flat(w) != Util.flat(v) and Util.flat(w) in Util.flat(v)) > 1 for p, v in seen.items()) >= len(seen): continue
            for group in groups:
                same = [p for p in seen if p in group["seen"] and Util.flat(seen[p]) == Util.flat(group["seen"][p])]
                if 3 * len(same) >= 2 * len([p for p in seen if p in group["seen"]]) and (any(Util.overlaps(key, n) for n in group["names"]) or len({seen[p] for p in same}) > 1):
                    group["names"].append(key)
                    group["seen"] = {**seen, **group["seen"]}
                    break
            else:
                groups.append({"names": [key], "seen": dict(seen)})
        claim, words, picks = self.claimed(), set(Util.norm(self.product).split()), []
        for group in groups:
            values = [Util.flat(v) for v in group["seen"].values()]
            label = max([n for n in group["names"] if len(n.split()) <= NAME_WORDS] or group["names"], key=lambda n: (sum(Util.stems(o) < Util.stems(n) and not Util.same(o, n) for o in group["names"]), len(keys[n]), -len(n.split()), -len(n)))
            if any(len(v) > 2 and v in words for v in group["seen"].values()) or sum(map(len, values)) > FREE_TEXT * len(values) or len(label.split()) > NAME_WORDS or Util.stems(label) <= Util.stems(self.product) or any(self.taken(n, claim) for n in group["names"]): continue
            share = (len(values) + 1) / (len(facts) + 2)
            picks.append({"property": label, "listings": len(values), "values": list(dict.fromkeys(group["seen"].values()))[:EXAMPLES], "share": round(share, 2)})
        return sorted(picks, key=lambda o: -o["listings"])[:OPTIONS], len(facts)
    def claimed(self):
        seen, owned, books = {}, set(), {**self.cached, **self.details}
        for pid in shelf if len(shelf := self.opened()) > 1 else self.opened(books):
            for key, value in Util.attrs(books[pid], {}).items(): seen.setdefault(Util.keyname(key), {})[pid] = Util.norm(value)
        for req in self.reqs:
            near, held = {k for k in seen if Util.stems(req["attribute"]) and Util.stems(req["attribute"]) <= Util.stems(k)}, {k: (sum(any(Util.holds(v, w) and Util.holds(w, v) for w in req["values"]) for v in vs.values()), sum(any(Util.holds(v, w) for w in req["values"]) for v in vs.values())) for k, vs in seen.items()}
            owned |= {k for k in near if held[k][1]} or {k for k, h in held.items() if h[1] and h == max(held.values())} or near
        return seen, owned | {k for k, vs in seen.items() for o in owned if o != k and (both := vs.keys() & seen[o].keys()) and all(Util.flat(vs[p]) == Util.flat(seen[o][p]) for p in both)}
    def taken(self, name, claim=None):
        seen, owned = claim or self.claimed()
        told, own, asked = [self.best_by] + [r["attribute"] for r in self.reqs if name not in seen], lambda n: Util.stems(n) - Util.stems(self.product) or Util.stems(n), sorted(self.known) + [a["property"] for a in self.asked if a["item"] == self.product]
        return name in owned or any(Util.stems(t) and Util.stems(t) <= Util.stems(name) for t in told if t) or any(own(t) == own(name) for t in asked)
    def ties(self):
        return [Util.norm(q).split() for q in self.basket_rules + [l["quote"] for l in self.links]]
    def tied(self, name):
        for words in self.ties():
            names = [(c, d, i) for i, s in enumerate(self.items) for c, d in Util.spans(words, s["product"])]
            for a, b in Util.spans(words, name):
                if (near := sorted((c - b if b <= c else a - d, i) for c, d, i in names)) and near[0][1] == self.at and not any(c < b and a < d for c, d, _ in names): return True
        return False
    def choose(self, options, size):
        others = [s["product"] for i, s in enumerate(self.items) if i != self.at]
        data = {"request": self.query, "product": self.product, "other_products": others, "requirements": self.reqs, "asked": [a["property"] for a in self.asked], "candidate_pool": options}
        outs = [out for out in (self.model(prompt, data, 700) for prompt in (Prompts.QUESTION, Prompts.QUESTION.replace(Prompts.LINKS, ""))) if out] or [{}]
        stated = lambda h: isinstance(h, dict) and any(Util.spans(words, h.get("words") or "") for words in self.ties())
        pool, unseen, lift = [o["property"] for o in options], 1 / (size + 2), {o["property"]: o["share"] * (VARIED if len(o["values"]) > 1 else 1) for o in options}
        lines, weights, scored, claim = [Util.stems(p) for p in others + [self.product]], {"hinted": 5, "core": 2, "more": 1, "minor": 1, "skip": 0.1}, {}, self.claimed()
        fair = lambda n: (n in pool or n and len(n.split()) <= NAME_WORDS and not self.taken(n, claim)) and not self.tied(n)
        for out in outs:
            tiers, seen = {k: [Util.norm((h.get("property") or "") if isinstance(h, dict) else h) for h in out.get(k) if isinstance(h, (str, dict)) and not stated(h)] if isinstance(out.get(k), list) else [] for k in ("hinted", "core", "minor", "more", "skip")}, {}
            tiers["minor"] += [n for n in pool if not any(n in v for v in tiers.values())]
            for kind, weight in weights.items():
                for name in [n for n in dict.fromkeys(tiers[kind]) if fair(n) and not any(Util.stems(n) <= line for line in lines)]: seen.setdefault(Util.flat(Util.singular(name)), (weight, name))
            for key, (weight, name) in seen.items(): scored.setdefault(key, [0, name])[0] += weight / len(outs)
        return sorted(((weight * lift.get(name, unseen), name) for weight, name in scored.values()), key=lambda x: -x[0]) + [(weights["minor"] * unseen, None)] * (self.budget is None and self.total is None and not any(s["budget"] for s in self.items))


class Recheck:
    def drain(self, kind):
        lines, self.pending = self.pending, []
        found = self.recheck(lines, kind) if lines else set()
        if found is None: self.pending, found = lines + self.pending, set()
        if kind == "notices": self.shifted |= found
        return found
    def recheck(self, lines, kind):
        data = {"requirements": self.reqs, "budget": self.budget if self.total is None else self.total, "budget_unit": self.unit, "quantity": self.qty, "cart": [Util.clip(self.products.get(p, {"id": p}), 80) for p in self.cart], "questions_asked": [a["property"] for a in self.asked], "kind": kind, "new_lines": lines}
        if len(self.items) > 1: data.update(current_product=self.product, products_in_order=[s["product"] for s in self.items])
        out = self.model(Prompts.CHANGE, data, 600)
        if out is None: return None
        changes = out.get("changes") if isinstance(out.get("changes"), list) else []
        said, page = set(Util.norm(json.dumps(lines, ensure_ascii=False)).split()), [Util.norm(t) for t in lines if kind == "notices" and isinstance(t, str) and t.split(": ", 1)[0] in self.products]
        self.hearing = [Util.norm(t) for t in self.heard if Util.norm(t) in Util.norm(json.dumps(lines, ensure_ascii=False))]
        stated = lambda c: bool(Util.norm(c.get("quote") or "")) and len(set(Util.norm(c["quote"]).split()) - said) * 5 <= len(set(Util.norm(c["quote"]).split()))
        found = {k for k in (self.route(c) for c in changes if isinstance(c, dict) and stated(c) and (c.get("type") in ("out_of_stock", "price", "other") or not any(Util.norm(c["quote"]) in t for t in page))) if k}
        if found <= {"other"}: self.heard.update(line for line in lines if isinstance(line, str))
        return found
    def route(self, change):
        product, listing = str(change.get("product") or ""), change.get("listing") if isinstance(change.get("listing"), dict) else {}
        named = [i for i, s in enumerate(self.items) if Util.stems(product) == Util.stems(s["product"])] or [i for i, s in enumerate(self.items) if Util.overlaps(product, s["product"])]
        if change.get("type") == "need" and Util.norm(product) and not named and len(self.items) > 1 and not any(Util.overlaps(product, str(self.field(p, "title") or "")) for p in self.cart): return None
        if listing and len(self.items) > 1:
            words, cut = Util.norm(change.get("quote") or "").split(), Util.norm(listing.get("product") or "").split()
            last = max([i for i in range(len(words)) if cut and words[i:i + len(cut)] == cut] or [len(words)])
            rest = " ".join(words[:last] + words[last + len(cut):])
            said = [i for i, s in enumerate(self.items) if Util.holds(rest, s["product"])]
            named = said if len(said) == 1 else named
        owner = [i for i, s in enumerate(self.items) if change.get("id") in s["details"]]
        index = (owner or ([self.at] if self.at in named else named) or [self.at])[0]
        done = self.apply(change) if index == self.at else self.within(index, lambda: self.apply(change))
        if done: self.touched.add(index)
        return done
    def apply(self, change):
        kind, attribute = change.get("type"), str(change.get("attribute") or "").strip()
        words, cut = Util.norm(change.get("quote") or "").split(), Util.norm(attribute).split()
        keys = {frozenset(Util.stems(Util.keyname(k))) for p in self.details for k in self.attributes(p)}
        attribute = next((" ".join(words[i - n:i + len(cut)]) for i in range(len(words)) if cut and words[i:i + len(cut)] == cut for n in range(i, 0, -1) if frozenset(Util.stems(" ".join(words[i - n:i + len(cut)]))) in keys), attribute)
        match =[r for r in self.reqs if Util.norm(attribute) and (Util.overlaps(attribute, r["attribute"]) or Util.flat(attribute) in Util.flat(r["attribute"]) or Util.flat(r["attribute"]) in Util.flat(attribute))]
        amount, new, held = Util.number(change.get("number")), Util.requirement(change), self.budget if self.total is None else self.total
        listing = change.get("listing") if isinstance(change.get("listing"), dict) else None
        kind = "reference" if kind == "need" and listing and change.get("relation") else kind
        quote = Util.norm(change.get("quote") or "")
        source = next((t for t in self.hearing + [Util.norm(t) for t in self.heard] if quote and " %s " % quote in " %s " % t), None)
        if kind == "reference" and isinstance(listing.get("facts"), dict):
            bare = [str(k).strip() for k, v in listing["facts"].items() if str(k).strip() and not Util.listed(v)]
            attribute = attribute if not bare or Util.grounded(attribute, (source or quote).split()) else bare[0]
            listing = {**listing, "facts": {k: v for k, v in listing["facts"].items() if Util.listed(v)}}
        here = [a["property"] for a in self.asked if source and source in Util.norm(json.dumps(a["reply"], ensure_ascii=False))] or [a["property"] for a in self.asked]
        anchor = (self.at, source)
        mine = [(name, sign, facts) for a, name, sign, facts in self.anchors if source and a == anchor]
        told = {v for _, _, facts in mine for v in facts}
        same = [sign for name, sign, _ in mine if Util.overlaps(name, attribute)]
        stray = source is not None and not Util.grounded(attribute, source.split()) and not any(Util.overlaps(attribute, a) for a in here)
        if kind in ("need", "changed") and new: new = {**new, "values": Util.joined(new["values"], source or quote)}
        if kind == "need" and (None in same or stray or new and new["values"] and all(Util.flat(v) in told for v in new["values"])): return None
        if kind in ("need", "changed") and new:
            if any(Util.sign({**r, "attribute": ""}) == Util.sign({**new, "attribute": ""}) and Util.overlaps(r["attribute"], new["attribute"]) for r in self.reqs): return None
            gone = match if kind == "changed" else [r for r in self.reqs if Util.stems(r["attribute"]) == Util.stems(attribute) and r["exclude"] == new["exclude"]]
            self.reqs = [r for r in self.reqs if r not in gone] + [new]
            self.anchors.append((anchor, attribute, Util.sign(new), ()))
            self.known.add(attribute)
            self.revealed.append(new)
        elif kind == "reference" and attribute and listing:
            facts = {Util.flat(v) for v in Util.described(listing)}
            same += [sign for _, sign, _ in mine if sign and set(map(Util.flat, json.loads(sign)[1])) & facts]
            self.known.add(attribute)
            self.reqs = [r for r in self.reqs if Util.sign(r) not in same]
            self.anchors.append((anchor, attribute, None, facts))
            made = self.pursue([{**change, "attribute": attribute, "listing": listing, "item": self.at}])
            if not made and not same: return None
            self.revealed += made
            kind = "need"
        elif kind == "no_preference" and attribute and attribute not in self.known: self.known.add(attribute)
        elif kind == "dropped" and match: self.reqs = [r for r in self.reqs if r not in match]
        elif kind == "budget" and amount and amount > 0 and amount != held:
            kind = "raised" if held is not None and amount > held else kind
            if self.total is None: self.budget = amount
            else: self.total = amount
        elif kind == "quantity" and amount and amount == int(amount) and amount >= 1 and amount != self.qty:
            if change.get("per_unit") is True and self.total is None and self.budget and not self.per_unit: self.budget, self.per_unit = self.budget / self.qty, True
            self.qty = int(amount)
        elif kind == "out_of_stock":
            if not self.sold_out(change.get("id")): return None
        elif kind in ("price", "other"): return kind
        else: return None
        self.version += 1
        self.changes += 1
        live = {Util.sign(r) for s in self.items for r in s["reqs"]}
        self.judged = {k: v for k, v in self.judged.items() if k[2].lstrip("~") in live}
        return kind
    def sold_out(self, pid=None):
        hit = [pid] if pid in self.cart or pid in self.products else [p for p in self.cart if p in self.details] or (list(self.cart) if len(self.cart) == 1 else [])
        fresh = set(hit) - self.gone
        self.gone |= fresh
        for line in self.items: line["lost"].update(p for p in fresh if p in line["details"])
        return bool(fresh) or not hit
    def repair(self, found, lines):
        at, touched, self.touched = self.at, self.touched, set()
        add = self.roles.get("add")
        self.mend = 1 + (bool(add) and (1 if Util.slots(add["args"]) else sum(len(self.chunks(s["qty"])) for s in self.items) + ("remove" in self.roles)))
        for index in range(len(self.items)):
            self.focus(index)
            top, held = self.judge(), [p for p in lines if p in self.details]
            wider = bool(found & WIDEN - ({"raised"} if self.objective == "cheapest" else set())) and index in touched and self.objective != "any"
            if wider or [p for p in held if p not in top]: self.reopen(top, held + list(self.lost), wider)
        self.focus(at)
        self.ask()
    def reopen(self, top, kin, wider):
        cheap, seen = self.objective == "cheapest", list(dict.fromkeys(p for rows in self.search_rows for p in rows))
        name = lambda p: Util.norm(str(self.field(p, "title") or ""))
        for _ in range(LOOP):
            if top and not cheap or self.late(): break
            twins, bar = {name(p) for p in top[:1] + kin} - {""}, self.price(top[0]) if top else None
            todo = sorted([p for p in seen if p not in self.details and self.eligible(p) and (not cheap or self.under(p, bar))], key=lambda p: (name(p) not in twins, self.cat_ok.get(self.category(p)) is not True, self.price(p) or 0 if cheap else 0))
            pair = [p for p in todo if name(p) in twins]
            spare = self.hard - self.step - self.mend * (1 + bool(top and not pair))
            if not todo or spare <= 0: break
            for pid in (pair or todo)[:max(1, spare // 2)]: self.open_item(pid, final=True)
            top = self.judge()
        if wider or not self.judge():
            if self.unsearched(): self.first_search()
            self.settle(2)
    def reading(self, value):
        if value is None: return None
        key = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        if key not in self.readings:
            seen = {v if isinstance(v, str) else json.dumps(v, sort_keys=True) for v in (self.field(p, "stock") for p in self.products) if v is not None and not isinstance(v, (bool, int, float))}
            todo = sorted(seen - set(self.readings) | {key})
            out = self.model(Prompts.STOCK, {"key": self.fields.get("stock"), "values": todo}, 40 + 20 * len(todo)) or {}
            self.readings.update({v: out.get(v) if isinstance(out.get(v), bool) else None for v in todo})
        return self.readings[key]


class Finish:
    def finish(self):
        tried = set()
        for _ in range(LOOP):
            if self.finished or self.dead or self.step >= self.hard: return
            pick = self.choice()[:1]
            if not pick and self.vetoed:
                self.vetoed, self.version = set(), self.version + 1
                continue
            if not pick and not self.lean:
                self.lean = True
                continue
            asked, before = len(self.asked), self.step
            result = self.buy(pick[0]) if pick else False
            if result is not False and self.step > before:
                self.settle(2)
                continue
            if (self.questions is None or asked < self.questions) and self.room() > 0:
                self.ask()
                if len(self.asked) > asked: continue
            if pick and result is None and self.short is not None:
                if self.reach(): continue
                self.focus(self.short)
                self.lean = True
                if self.widen(): continue
            elif not pick and self.widen(): continue
            if not self.closing:
                at = self.at
                rest = next((i for i in range(len(self.items)) if i not in self.widened and self.focus(i) is None and not self.choice()), None)
                if rest is None: self.focus(at)
                self.closing = rest is None
                continue
            if not pick:
                if self.rescue(True): continue
                tried.add(self.at)
                if next((i for i in range(len(self.items)) if i not in tried and self.focus(i) is None and self.choice()), None) is not None: continue
                self.focus(min(tried))
            lines = {p: q for p, q in self.cart.items() if p not in self.gone and self.stocked(p) is not False} or pick and self.basket(pick[0])
            if not lines: return
            obs = self.order(lines)
            if obs is False: continue
            if obs is None or self.finished or self.dead: return
            self.refusal(next(iter(lines)), lines)
    def rescue(self, final):
        job, before, words = self.searcher(), self.step, Util.norm(self.product).split()
        if not job or self.hard - self.step <= self.reserve(True) or not final and self.busy(): return False
        schema, known = self.tools[job["tool"]]["parameters"], set() if final else set(self.yields())
        for text in (" ".join(words[i:]) for i in range(len(words))):
            args = Util.fit(schema, {**job["args"], job["key"]: text})
            if args is None or (self.at, text) in self.rescued or self.family(job["tool"], args) in known: continue
            self.rescued.add((self.at, text))
            if final: self.act("search", tool=job["tool"], args=args, final=True)
            else: self.run_search(job["tool"], args)
            if self.step > before: return True
        return False
    def salvage(self):
        self.gating = True
        try:
            if not (self.finished or self.dead): self.finish()
        except Exception as error:
            self.trace.append({"error": repr(error)[:300]})
    def endgame(self):
        self.gating, tried = True, []
        try:
            for left in range(min(self.hard - self.step, 2 * len(self.items) + 2), 0, -1):
                if self.finished or self.dead or self.step >= self.hard or self.elapsed() >= self.deadline(): return
                want, seen = [{"product": s["product"], "quantity": s["qty"], "requirements": s["reqs"], "candidates": [{"id": p, "name": self.field(p, "title"), "price": self.price(p)} for p in self.within(i, self.partial)[:4]]} for i, s in enumerate(self.items)], [r for r in self.raws if r["role"] in READS and r["result"]]
                goal = "Place the order within the next %d calls without messaging the shopper: for each product take the listing among its candidates or the results that fits it best (an imperfect fit is better than no order); search only while more than two calls are left and a product has no listing of its kind there; the last two calls put every product into the cart at its quantity and buy with the tool that orders." % left
                if self.amend("final", goal, {"order": want, "cart": self.cart, "results": [json.dumps(r, ensure_ascii=False, default=str)[:9000 // len(seen)] for r in seen]}, tried) is None: break
            if self.cart and "add" in self.roles and not (self.finished or self.dead): self.act("checkout", final=True)
        except Exception as error:
            self.trace.append({"error": repr(error)[:300]})
    def amend(self, step, goal, raw, tried):
        out = self.recover(step, goal, json.dumps(raw, ensure_ascii=False, default=str)[:12000], tried) or {}
        call = out.get("call") if out.get("action") == "call" and isinstance(out.get("call"), dict) else {}
        if call.get("tool") not in self.tools: return None
        role, args = next((r for r, e in self.roles.items() if e["tool"] == call["tool"]), "search" if call["tool"] in self.search_tools else "other"), call.get("args") if isinstance(call.get("args"), dict) else {}
        obs = self.act(role, tool=call["tool"], args={**args, **{k: v for k, v in self.roles[role]["args"].items() if "<" not in json.dumps(v)}} if role == "checkout" else args, final=True)
        tried.append({"call": call, "result": self.raws[-1]["result"] if obs is not None else "not sent"})
        if role in ("add", "remove", "review") and (held := self.held(obs, role != "review")) is not None: self.cart = held
        return obs
    def choice(self):
        keep = self.assigned.get(self.at)
        if keep in self.cart and keep not in self.gone and keep not in self.vetoed and (self.fixed() or self.settled(keep)): return [keep]
        return [p for p in self.judge() or self.nearest() or (self.partial() if self.closing else []) if not self.foreign(p)]
    def fixed(self):
        return bool(add := self.roles.get("add")) and "remove" not in self.roles and not (Util.slots(add["args"]) and self.replaces)
    def partial(self):
        hits, fails, seen = self.evidence(), lambda p: [r for r in self.reqs if self.verdict(p, r) is False], set(self.details) | {p for rows in self.search_rows for p in rows}
        words = {w for w in Util.stems(self.product) if len(w) > 2}
        named = lambda p: self.cat_ok.get(self.category(p)) or words and 2 * len(words & Util.stems(json.dumps(self.products[p], ensure_ascii=False, default=str))) >= len(words)
        known = [p for p in self.products if p not in self.gone and (not self.in_stock or self.stocked(p) is not False) and self.cat_ok.get(self.category(p)) is not False]
        pool = [p for p in known if p in seen] or [p for p in known if named(p)]
        return sorted(self.rank(pool) or pool, key=lambda p: (not self.eligible(p), self.cat_ok.get(self.category(p)) is not True, p not in self.details, len(fails(p)), -hits.get(p, 0), sum(bool(self.own(p, r["attribute"])) for r in fails(p))))
    def widen(self):
        if self.at in self.widened or self.widened.add(self.at) or self.hard - self.step <= self.reserve(True) or self.elapsed() >= self.deadline(): return False
        before, job, swap, have = self.step, self.searcher(), "add" in self.roles and self.assigned.get(self.at) in self.cart and len(self.chunks(self.qty)) + ("remove" in self.roles), self.judge()
        for pid in [] if have else self.prefer([p for p in self.fresh(True) if not self.foreign(p)]):
            if self.dead or self.hard - self.step <= self.reserve(True) + swap + 1 or self.late() or pid not in self.cached and self.judge(): break
            self.open_item(pid, final=True)
        if not have and self.judge(): return True
        entry = next(filter(None, (self.search_entry({"tool": job["tool"], "args": {**job["args"], job["key"]: self.wording(need)}}) for need in [""] + [n["value"] for n in self.unsearched()])), None) if job else None
        if entry and self.busy(): self.act("search", tool=entry[0], args=entry[1])
        elif entry: self.seek(*entry)
        self.settle(2)
        return self.step > before
    def basket(self, pid):
        at, options, self.short, self.unfit = self.at, {}, None, None
        tied = {l[k] for l in self.links for k in ("item", "on")}
        for index in range(len(self.items)):
            options[index] = [pid] * (index == at) + [p for p in self.within(index, self.choice) if p != pid] * (index != at or index in tied)
            if not options[index] and not self.closing:
                self.short = index
                return self.focus(at)
        self.focus(at)
        for link in [l for l in self.links if options[l["item"]] and options[l["on"]]]: self.suits(link, options[link["on"]][0], options[link["item"]], options[link["on"]])
        chosen = self.combine(options, self.links)
        if (weak := [l for l in self.links if chosen and self.fitted.get((l["quote"], chosen.get(l["on"]), chosen.get(l["item"]))) is True]) and not self.closing or chosen is None:
            link = next(iter(weak), None) or min(self.links, key=lambda l: sum(len(self.suits(l, a, options[l["item"]], options[l["on"]])) for a in options[l["on"]]), default=None)
            if not self.closing or link and self.fitted.get(("reached", link["quote"])) != self.step:
                self.short, self.unfit = (link["item"], (link, options[link["on"]])) if link else (max(options), None)
                return None
        self.assigned = dict(chosen or self.combine(options, []) or {})
        for code in {self.fitted.get(("rest", p)) for i, p in self.assigned.items() if any(l["item"] == i for l in self.links)} - {None}:
            if ("looked", code) not in self.fitted and self.room() > 0 and not any(Util.flat(code) in Util.flat(str(self.field(q, "title") or "")) for q in self.products if q not in self.assigned.values()) and (job := self.searcher()) and (args := Util.fit(self.tools[job["tool"]]["parameters"], {**job["args"], job["key"]: code})): self.fitted[("looked", code)] = self.aside(lambda job: self.run_search(job["tool"], args))
        return {p: sum(self.items[i]["qty"] for i, q in self.assigned.items() if q == p) for p in self.assigned.values()}
    def combine(self, options, links):
        cost = lambda i, p: (self.price(p) or 0) * self.items[i]["qty"]
        limit, best = None if self.total is None else self.total * self.scale + 1e-9, [None, None]
        cheap = set(options) if (own := {i for i in options if self.items[i]["objective"] == "cheapest"}) and (self.total is not None or self.links) else own
        links = [l for l in links if options[l["item"]] and options[l["on"]]]
        depth = dict.fromkeys(options, 0)
        for _, l in itertools.product(options, links): depth[l["item"]] = max(depth[l["item"]], min(depth[l["on"]] + 1, len(options)))
        tied = {l[k] for l in links for k in ("item", "on")}
        order = sorted((i for i in options if options[i]), key=lambda i: (i not in tied, depth[i], i))
        floor = [min(cost(i, p) for p in options[i]) for i in order]
        score = lambda spent, saved, weak: (limit is not None and spent > limit, weak[0] * (not cheap), spent if limit is not None and spent > limit else saved, weak[1])
        fits = lambda l, chosen: self.suits(l, chosen[l["on"]], options[l["item"]], options[l["on"]]) and self.fitted.get((l["quote"], chosen[l["on"]], chosen[l["item"]]), 0)
        def walk(k, chosen, spent, saved, weak):
            if best[0] is not None and not score(spent + sum(floor[k:]), saved + sum(f for i, f in zip(order[k:], floor[k:]) if i in cheap), weak) < best[0]: return
            if k == len(order): best[:] = [score(spent, saved, weak), dict(chosen)]
            for p in options[order[k]] if k < len(order) else []:
                if p in chosen.values(): continue
                chosen[order[k]] = p
                ties = [fits(l, chosen) for l in links if order[k] in (l["item"], l["on"]) and l["item"] in chosen and l["on"] in chosen]
                if all(ties): walk(k + 1, chosen, spent + cost(order[k], p), saved + cost(order[k], p) * (order[k] in cheap), (weak[0] + (("rest", p) in self.fitted), weak[1] + ties.count(True)))
                del chosen[order[k]]
        walk(0, {}, 0, 0, (0, 0))
        return best[1]
    def suits(self, link, anchor, pids, peers=(), sure=False):
        quote, pids = link["quote"], list(dict.fromkeys(pids))
        for chunk in ([p for p in pids[start:start + SHELF] if (quote, anchor, p) not in self.fitted] for start in range(0, len(pids), SHELF)):
            if chunk and not self.tie(link, anchor, chunk, peers): break
        return [p for p in pids if self.fitted.get((quote, anchor, p), 0) >= 1 + sure]
    def tie(self, link, anchor, chunk, peers):
        quote, batch = link["quote"], [anchor] + [a for a in dict.fromkeys(peers) if a != anchor and (not self.fitted.get((link["quote"], a), {}).get("fresh") or any((link["quote"], a, p) not in self.fitted for p in chunk))][:JUDGE_BATCH // 2 - 1]
        data = {"rule": quote, "product": self.items[link["item"]]["product"], "chosen": (chosen := {f"A{i}": {"product": self.items[link["on"]]["product"], **self.sheet(a), "continued": [self.sheet(p) for p in self.fitted.get((quote, a), {}).get("pages", [])]} for i, a in enumerate(batch, 1)}), "listings": {str(i): self.sheet(p) for i, p in enumerate(chunk, 1)}}
        out = None if self.late(1) else self.model(Prompts.FIT, data, 600 + 150 * len(batch) + 40 * len(chunk), SOL)
        if out is None: return self.fitted.update({(quote, a, p): 0 for a in batch for p in chunk})
        self.fitted.update({("rest", p): Util.clip(code, 60) for key, p in zip(data["listings"], chunk) if isinstance(code := (out.get("rest") if isinstance(out.get("rest"), dict) else {}).get(key), str) and len(Util.flat(code)) > 1 and Util.norm(code) in Util.norm(json.dumps(data["listings"][key], ensure_ascii=False))})
        for name, a in zip(chosen, batch):
            said, old = out.get(name) if isinstance(out.get(name), dict) else {}, self.fitted.get((quote, a), {"pages": [], "followed": set()})
            codes = [c for text in [Util.norm(json.dumps(chosen[name], ensure_ascii=False, default=str)).split()] for c in dict.fromkeys(Util.listed(said.get("codes"))) if len(Util.flat(c)) > 1 and Util.spans(text, c)]
            more = {k: m.get(k) for k in ("for", "id", "seller", "product", "facts")} if isinstance(m := said.get("more"), dict) and (Util.ident(m.get("id")) or Util.listed(m.get("seller")) or Util.described(m)) else None
            self.fitted[(quote, a)] = {**old, "fresh": True, "applies": said.get("applies") is not False, "codes": codes, "words": [Util.clip(w, 80) for w in Util.listed(said.get("words")) + Util.listed(said.get("needs"))], "more": more if more and json.dumps(more, sort_keys=True) not in old["followed"] else None}
            self.fitted.update({(quote, a, p): 2 if said.get("applies") is not False and len(Util.norm(quotes.get(key) or "")) > 1 and Util.norm(quotes[key]) in Util.norm(json.dumps(data["listings"][key], ensure_ascii=False)) else said.get("applies") is False for quotes in [said.get("fit") if isinstance(said.get("fit"), dict) else {}] for key, p in zip(data["listings"], chunk)})
        return True
    def extend(self, link, anchor):
        known, job = self.fitted.get((link["quote"], anchor)), self.searcher()
        if not (more := known and known["more"]) or not job or self.busy(): return False
        known["more"], known["followed"] = None, known["followed"] | {key := json.dumps(more, sort_keys=True)}
        if ("page", key) not in self.fitted: self.fitted[("page", key)] = self.aside(lambda job: self.find(more, str(more.get("for") or link["quote"]), job, list(dict.fromkeys([anchor] + known["pages"] + [p for p in self.products if self.category(p) and self.category(p) == self.category(anchor)])))[0], urgent=True)
        if (page := self.fitted[("page", key)]) is None or page in known["pages"]: return False
        known["pages"], known["fresh"] = known["pages"] + [page], False
        for stale in [k for k in self.fitted if len(k) == 3 and k[:2] == (link["quote"], anchor)]: del self.fitted[stale]
        return True
    def hunt(self, link, anchor, peers):
        skip, rows = set(peers) | {p for i, s in enumerate(self.items) if i != link["item"] for p in s["details"]}, lambda: self.prefer([p for p in self.pool(None) if p not in self.details and p not in skip and self.live(p) and not self.foreign(p) and not (self.category(p) and self.category(p) == self.category(anchor))])
        for _ in range(LOOP):
            fit, known = self.suits(link, anchor, rows(), peers, True), self.fitted.get((link["quote"], anchor)) or {}
            if fit or self.busy() or not known.get("applies", True) or not self.target(known.get("words", []) + known.get("codes", [])) and not self.extend(link, anchor): return fit
        return []
    def reach(self):
        if not self.unfit: return False
        (link, anchors), at, before, tried, done = self.unfit, self.at, self.step, set(), lambda anchor: self.busy() or any(self.suits(link, anchor, self.passing(), anchors, True))
        self.focus(link["item"])
        for anchor in anchors:
            while not done(anchor) and (fit := [p for p in self.hunt(link, anchor, anchors) if p not in tried]):
                for pid in sorted(fit, key=lambda p: self.price(p) or 0) if self.objective == "cheapest" else fit:
                    if done(anchor): break
                    tried.add(pid)
                    self.open_item(pid)
                    self.judge()
            if done(anchor): break
        self.focus(at)
        if self.step == before: self.fitted[("reached", link["quote"])] = self.step
        return self.step > before
    def worse(self):
        pid, top = self.assigned.get(self.at), self.judge()[:1]
        return pid in self.cart and (not self.eligible(pid) or bool(top) and (not self.passes(pid) or self.objective == "cheapest" and (self.price(top[0]) or 0) < (self.price(pid) or 0)))
    def stage(self):
        if "add" not in self.roles or self.fixed() and not all(self.each(self.nearest)) or self.cart and not any(p in self.gone or p in self.vetoed for p in self.cart) and not any(self.each(self.worse)): return False
        stop, before, kept = self.stop, self.step, self.closing
        self.closing = kept or not self.fixed()
        pick = self.choice()[:1]
        lines = pick and self.basket(pick[0])
        self.closing = kept
        if lines and lines != self.cart and not self.late() and self.hard - self.reserve() - self.unasked() - self.step >= self.moves(lines) and not self.audit(lines, True) and self.place(lines): self.moved(pick[0], lines)
        self.stop = stop
        return self.step > before
    def buy(self, pid):
        lines = self.basket(pid)
        if not lines or self.audit(lines, self.closing): return
        need, cart = 1 + self.moves(lines), "add" in self.roles and "review" in self.roles
        if self.hard - self.step < need: return False
        if self.tries and cart and self.hard - self.step > need: self.look()
        if not self.place(lines) or self.moved(pid, lines): return
        if not self.tries and cart and self.hard - self.step > 1 and (self.blind or not any(map(self.unread, lines))):
            self.look()
            if self.moved(pid, lines) or self.cart != lines and self.hard - self.step > self.moves(lines) and not self.place(lines): return
        if self.order(lines) is False: return
        self.tries += 1
        if not self.finished and not self.dead: self.refusal(pid, lines)
    def verify(self, lines):
        for pid in sorted([p for p in lines if self.unread(p)], key=lambda p: self.read_at.get(p, -1)):
            for role in self.unread(pid):
                if self.hard - self.step <= self.reserve(True) or self.late(): return
                self.act(role, pid=pid, final=True)
                self.drain("notices")
    def unread(self, pid):
        return [r for r in ("details", "view") if r in self.roles and (r == "details" or self.dual) and pid not in self.read_by.setdefault(r, set())] or ([] if self.seen(pid) else ["view" if "view" in self.roles else "details"])
    def refusal(self, pid, lines):
        mark, found = len(self.raws), self.drain("the order was refused")
        single, hit, role = bool(found) and found <= {"out_of_stock"}, False, "view" if "view" in self.roles else "details" if "details" in self.roles else None
        for line in lines if not found or found & {"out_of_stock", "price", "other"} else []:
            if single and hit: break
            if line not in self.gone and role and self.hard - self.step > 1:
                self.act(role, pid=line, final=True)
                self.drain("notices")
            if self.stocked(line) is False and self.in_stock:
                hit = hit or line not in self.gone
                self.gone.add(line)
                for item in [s for s in self.items if line in s["details"]]: item["lost"].add(line)
        if not found and not hit or self.over("final"):
            self.blind, self.trusted = True, False
            self.amend("final", "The order of these lines (listing id: quantity) was refused: %s. Find out why and fix it so that the order can be placed." % json.dumps(lines), self.raws[mark - 1:mark], self.raws[mark:])
        key = json.dumps(sorted(lines.items()))
        self.refused[key] = self.refused.get(key, 0) + 1
        if self.refused[key] > 1 and not found - {"price", "other"}: self.gone.add(pid)
        found, self.shifted = found | self.shifted, set()
        self.repair(found, lines)
    def moved(self, pid, lines):
        shifts, self.shifted = self.drain("notices") | self.shifted, set()
        if shifts: self.repair(shifts, lines)
        else: self.ask()
        ranked = self.choice()
        return ranked[:1] != [pid] and (pid in ranked or not self.eligible(pid) or bool(shifts & SHIFTS)) or self.basket(pid) != lines
    def chunks(self, qty, role="add"):
        entry = self.roles[role]
        if "<QTY>" not in json.dumps(entry["args"]): return [1] * qty
        top = Util.most(entry["args"], self.tools[entry["tool"]]["parameters"])
        top = min(([int(top)] if top and top >= 1 else []) + [entry[k] for k in ("most",) if k in entry], default=qty)
        return [qty] if self.sets and not Util.slots(entry["args"]) else [top] * (qty // top) + [qty % top] * bool(qty % top)
    def split(self, lines, role="add"):
        return [(p, n) for p, q in lines.items() for n in self.chunks(q, role)]
    def calls(self, lines):
        add, have = self.roles.get("add"), self.cart
        if not add: return []
        lines = {p: min(q, add["held"]) if "held" in add else q for p, q in lines.items()}
        if Util.slots(add["args"]):
            if self.replaces: return [("add", None, dict(lines))] * (have != lines)
            send, size = {p: q for p, q in lines.items() if have.get(p) != q}, add.get("lines") or len(lines)
            return [("remove", p, q) for p, q in have.items() if q != lines.get(p)] + [("add", None, dict(list(send.items())[i:i + size])) for i in range(0, len(send), size)]
        out = [("remove", p, q) for p, q in have.items() if p not in lines]
        for p, q in lines.items():
            held = have.get(p, 0)
            if held == q: continue
            if self.sets: out.append(("add", p, q))
            elif held and q > held and self.sets is False: out += [("add", p, n) for n in self.chunks(q - held)]
            elif held and self.sets is None and "remove" not in self.roles: out.append(("add", p, q))
            else: out += [("remove", p, held)] * bool(held) + [("add", p, n) for n in self.chunks(q)]
        return out
    def moves(self, lines):
        return len(self.calls(lines))
    def place(self, lines):
        mark, hit = len(self.raws), True
        for _ in range(3):
            plan = self.calls(lines)
            if any(role not in self.roles for role, _, _ in plan) and not self.remapped:
                self.remapped, kept, known = True, dict(self.roles), (self.replaces, self.sets, dict(self.search_tools))
                self.map_tools()
                self.roles, self.replaces, self.search_tools = {**self.roles, **kept}, known[0], known[2]
                self.sets = self.sets if known[1] is None else known[1]
                plan = self.calls(lines)
            plan = [c for c in plan if c[0] in self.roles]
            if not plan: return True
            if len(plan) >= self.hard - self.step or not self.gate(lines, "add"): return False
            if (hit := next((h for h in (self.change(*c) for c in plan) if not h), True)) is None: break
        if set(lines) & self.gone or not self.calls(lines) or not self.over("cart"): return hit is not None
        for _ in self.calls(lines):
            if self.amend("cart", "Make the cart hold exactly these lines (listing id: quantity): %s" % json.dumps(lines), self.raws[-1:], self.raws[mark:]) is None or not self.calls(lines): break
        return not self.calls(lines)
    def change(self, role, pid, qty):
        entry, before = self.roles[role], dict(self.cart)
        sent = list(qty) if pid is None else [pid]
        obs = self.act(role, tool=entry["tool"], args=Util.fill(entry["args"], None, None, None, lines=self.split(qty)), final=True) if pid is None else self.act(role, pid=pid, qty=qty, final=True)
        self.placed.update(dict.fromkeys(sent, self.step))
        if obs is None or self.last["fault"]:
            self.blind, self.trusted = True, False
            if role == "add": return self.shrink(entry, sent, qty, before, obs)
            if entry.get("trial"): self.roles.pop(role)
            return None
        if pid is None: expect = dict(qty) if self.replaces else {**before, **qty}
        elif role == "add": expect = {**before, pid: qty if self.sets else before.get(pid, 0) + qty}
        else: expect = {p: q for p, q in before.items() if p != pid}
        seen = self.held(obs, True)
        if seen is None or not seen and expect or role == "add" and not set(seen) & (set(before) | set(sent)):
            self.cart, self.blind, self.trusted = expect, True, self.trusted and not isinstance(obs, str)
            return True
        if role == "add" and not all(p in seen for p in sent):
            self.gone.update(p for p in sent if p not in seen)
            self.cart, self.blind = seen, False
            return None
        if role == "add" and pid and before.get(pid) and seen[pid] in (qty, before[pid] + qty): self.sets = seen[pid] == qty
        if pid is None and set(before) - set(qty): self.replaces = not set(before) - set(qty) & set(seen)
        if role == "remove" and seen.get(pid, 0) >= before.get(pid, 0) > 0: self.roles.pop(role)
        entry.pop("trial", None)
        self.cart, self.blind = seen, False
        return seen == expect
    def shrink(self, entry, sent, qty, before, obs):
        named = [p for p in sent if p in json.dumps(obs)]
        units = sum(qty.values()) if isinstance(qty, dict) else qty
        if named or obs is None or units <= 1 and not (len(sent) == 1 and before.get(sent[0])):
            self.gone.update(named or sent)
            return None
        if isinstance(qty, dict) and len(qty) > 1: entry["lines"] = len(qty) // 2
        elif units > 1: entry["most"] = units // 2
        else: entry["held"] = min(entry.get("held", before[sent[0]]), before[sent[0]])
        return False
    def look(self):
        if self.looked == self.step: return
        held = self.held(obs := self.act("review", final=True))
        if held is not None:
            self.trusted = self.trusted or self.blind and held == self.cart
            self.cart, self.looked, self.blind = held, self.step, False
            self.priceby(obs)
    def priceby(self, obs):
        if self.fields.get("price") or not self.cart: return
        rows = [r for r in Util.dicts(obs) if any(str(v) in self.cart for v in r.values())]
        totals = {Util.number(v) for d in Util.dicts(obs) if all(d is not r for r in rows) for v in d.values() if not isinstance(v, (bool, dict, list))} - {None}
        for key in {k for r in rows for k, v in r.items() if Util.number(v) and not isinstance(v, bool)}:
            if all(key in r for r in rows) and any(key in p for p in self.products.values()) and sum(Util.number(r[key]) * self.cart.get(next(str(v) for v in r.values() if str(v) in self.cart), 1) for r in rows) in totals:
                self.fields["price"] = key
                return
    def held(self, obs, merge=False):
        if obs is None: return None
        known, skip = set(self.products) | set(self.cart), {self.fields.get("price"), self.fields.get("stock")}
        names = {w for r in ("add", "remove") if r in self.roles for d in Util.dicts(self.roles[r]["args"]) for k, v in d.items() if v == "<QTY>" for w in Util.stems(Util.keyname(k))}
        out = {}
        for row in Util.dicts(obs):
            pid = next((str(v) for v in row.values() if Util.ident(v) and str(v) in known), None)
            if pid is None: continue
            counts = {k: v for k, v in row.items() if isinstance(v, int) and not isinstance(v, bool) and k not in skip and self.products.get(pid, {}).get(k) != v}
            named = [v for k, v in counts.items() if Util.stems(Util.keyname(k)) & names]
            out[pid] = out.get(pid, 0) + (named[0] if len(named) == 1 else next(iter(counts.values())) if len(counts) == 1 else self.cart.get(pid, 1))
        whole = any(isinstance(v, list) for d in Util.dicts(obs) for v in d.values())
        if out and merge and not whole: out = {**self.cart, **out}
        if out or whole: return {p: q for p, q in out.items() if q > 0}
        return None
    def audit(self, lines, light=False):
        if light and not self.basket_rules: return []
        key = json.dumps([sorted(lines.items()), light])
        if key in self.audited: return self.audited[key]
        at, rows, facts = self.at, [], {}
        for index, pid in self.assigned.items():
            self.focus(index)
            if pid not in self.details: continue
            reqs = {} if light else {str(n + 1): r for n, r in enumerate(self.reqs) if (*self.mark(pid), Util.sign(r)) not in self.sources and not (self.decide(pid, r) is True and not self.notes(pid))}
            listing = {**self.sheet(pid), "price": self.price(pid)}
            facts[index] = (pid, reqs, Util.norm(json.dumps(listing, ensure_ascii=False, default=str)))
            rows.append({"line": index, "product": self.product, "quantity": self.qty, "requirements": reqs, "listing": listing})
        self.focus(at)
        if not any(r["requirements"] for r in rows) and not self.basket_rules or self.late(1): return self.audited.setdefault(key, [])
        data = {"request": self.query, "lines": rows, "total": sum((self.price(p) or 0) * q for p, q in lines.items())}
        if self.basket_rules: data["basket_rules"] = {str(n + 1): r for n, r in enumerate(self.basket_rules)}
        out = self.model(Prompts.ORDER, data, 900, SOL) or {}
        failed = [p for p in (facts[n][0] for e in Util.items(out.get("rules")) if isinstance(e, dict) and str(e.get("rule")) in data.get("basket_rules", {}) and e.get("pass") is False for n in Util.items(e.get("replace")) if n in facts) if self.spare(p)]
        self.veto(failed)
        for entry in [e for e in Util.items(out.get("lines")) if isinstance(e, dict) and e.get("line") in facts]:
            pid, reqs, text = facts[entry["line"]]
            self.focus(entry["line"])
            kept = lambda quote, req, own=None: Util.norm(quote) in self.shown(pid) and (self.decide(pid, req) is True or not req["exclude"] and Util.meets(quote, req, self.product) is True) or bool(own) and not any(Util.norm(v) in Util.norm(quote) for _, v in own)
            broken = [reqs[str(f.get("requirement"))] for f in Util.items(entry.get("fails")) if isinstance(f, dict) and str(f.get("requirement")) in reqs and len(Util.norm(f.get("quote") or "")) > 1 and Util.norm(f.get("quote")) in text and not kept(f["quote"], reqs[str(f.get("requirement"))], self.own(pid, reqs[str(f.get("requirement"))]["attribute"]))]
            if broken:
                self.judged.update({(*self.mark(pid), Util.sign(r)): False for r in broken})
                failed.append(pid)
        self.focus(at)
        self.audited[key] = list(dict.fromkeys(failed))
        self.trace.append({"audit": self.audited[key]})
        return self.audited[key]
    def order(self, lines):
        before, add = self.version, "add" in self.roles
        if add and self.blind and not self.trusted and "review" in self.roles and self.hard - self.step > 1: self.look()
        if add and self.cart != lines and self.moves(lines) < self.hard - self.step and not self.place(lines) and self.version != before: return False
        lines = self.cart or lines if add else lines
        self.verify(lines)
        if self.version != before or not self.gate(lines): return False
        if add:
            obs = self.act("checkout", text=self.reason(lines), final=True)
            self.blind = self.blind or not self.finished
            return obs
        entry, (pid, qty) = self.roles["checkout"], next(iter(lines.items()))
        self.cart = dict(lines)
        return self.act("checkout", tool=entry["tool"], args=Util.fill(entry["args"], pid, qty, self.reason(lines), lines=self.split(lines, "checkout")), final=True)
    def reason(self, lines):
        bill, rows = self.receipt(lines), [(p, lines[p], self.items[i]) for i, p in sorted(self.assigned.items()) if p in lines]
        short = ["%s: %s" % (p, ", ".join(m)) for p, q, s in rows for m in [[r["attribute"] for r in s["reqs"] if self.verdict(p, r) is not True] + ["stock"] * (self.stocked(p) is False) + ["budget"] * (s["budget"] is not None and (self.price(p) or 0) * q > s["budget"] * self.scale * (q if s["per_unit"] else 1))] if m] + ["total: over budget"] * (self.total is not None and bill["total"] > self.total * self.scale)
        return ("No clean fit found; the closest listings do not show every need (%s)." % "; ".join(short) if short else "Every stated need is met.") + " Ordered %s; total %s." % ("; ".join("%s x %s (%s) at %s each" % (r["count"], r["id"], Util.clip(r["listing"], 40), r["price_each"]) for r in bill["lines"]), bill["total"])


class Rules:
    def owing(self, rule):
        return self.state(self.cart) not in rule["done"] if "checkout" in rule["before"] else not rule["done"]
    def reserved(self, skip=()):
        moments = {}
        for r in [r for r in self.rules if r not in skip and self.owing(r)]: moments[frozenset(r["before"])] = max(moments.get(frozenset(r["before"]), 0), r["messages"])
        return int("message" in self.roles) * sum(moments.values())
    def owed(self):
        return self.reserved() + sum(r["calls"] * (len(self.items) if r["per_line"] else 1) for r in self.rules if self.owing(r))
    def acts(self, plan=False):
        return {k: v for k, v in ROLE.items() if (k == "search" or k in self.roles) and (plan or k not in PLAN)}
    def usage(self):
        entry = self.roles.get("status")
        obs = self.act("status") if entry else None
        if obs is None: return
        out = self.model(Prompts.USAGE, {"tool": Util.clip(self.tools[entry["tool"]]["description"], 400), "result": Util.clip(obs, 400), "messages_sent_by_agent": self.sent(), "actions_taken_by_agent": self.step}, 90) or {}
        left, steps, money = (Util.number(out.get(k)) for k in ("messages_left", "steps_left", "money_left"))
        if left is not None and left >= 0:
            self.allowed = self.sent() + int(left)
            self.questions = max(0, self.allowed - self.reserved())
        if steps is not None and steps >= 0: self.hard = min(self.max_steps, self.step + int(steps))
        if money and money > 0 and not any([self.total] + [s["budget"] for s in self.items]):
            if len(self.items) > 1: self.total = money
            else: self.budget = money
    def permitted(self, text):
        rules = [r["quote"] for r in self.rules if "message" in r["before"] and not r["messages"] + r["calls"]]
        if not rules: return True
        found = [s["product"] for i, s in enumerate(self.items) if (any(map(self.passes, list(self.details))) if i == self.at else s.get("low") is not None)]
        key = (text, tuple(found), self.sent())
        if key not in self.permits:
            out = self.model(Prompts.SEND, {"instructions": rules, "next_message": text, "messages_sent": self.talk() + self.said, "products_with_a_fitting_listing": found, "products_in_order": [s["product"] for s in self.items]}, 60) or {}
            self.permits[key] = out.get("send") is not False
        return self.permits[key]
    def gate(self, lines, moment="checkout"):
        if self.gating: return True
        self.gating = True
        try:
            for _ in range(LOOP):
                todo = self.unmet(lines, moment)
                if not todo: return True
                before = self.version
                if not self.carry(*todo, lines): return True
                if self.version != before: return False
            return True
        finally:
            self.gating = False
    def unmet(self, lines, moment):
        key = self.state(lines, moment)
        rules = [r for r in self.rules if r["messages"] + r["calls"] and key not in r["done"] and (moment in r["before"] or moment == "add" and "remove" in r["before"] or moment == "checkout")]
        return self.weigh_rules(rules, lines, moment) if rules else None
    def state(self, lines, moment="checkout"):
        return json.dumps([sorted(lines.items()), [self.price(p) for p in sorted(lines)], self.changes, moment == "checkout"], default=str)
    def can_tell(self):
        return "message" in self.roles and (self.allowed is None or self.allowed > self.sent()) and self.hard - self.step > 1 and not self.late()
    def carry(self, rules, steps, left, mark, lines):
        done, start, talk = False, self.version, any(r["messages"] for r in rules)
        for step in steps:
            role, target, text = step["action"], str(step.get("target") or ""), str(step.get("text") or "").strip()
            if role in ("message", "approve"):
                if not talk: continue
                obs = self.act(role, text=text, final=True) if text and (left is None or left > 0) and self.can_tell() else None
                if obs is None: break
                left = None if left is None else left - 1
                reply = self.last.get("lines") or Util.clip(obs, 300)
                if isinstance(obs, dict) and obs.get("error") == "budget_exceeded":
                    self.allowed = self.questions = self.sent()
                    break
                self.said.append({"rules": [r["quote"] for r in rules], "message": text, "reply": reply, "step": self.step})
                before = self.version
                self.recheck([{"question": text, "reply": reply}], "replies to the agent's messages")
                if step.get("reply") is True:
                    if self.version == before: self.answer(rules, text, reply, mark, lines)
                    return True
            elif role == "review":
                if self.hard - self.step <= 1: break
                self.look()
            elif role in self.roles:
                pids = list(lines) if target == "each" else [target] if target in self.products else []
                for pid in [p for p in pids if role not in READS or not self.seen(p)] if "<ID>" in json.dumps(self.roles[role]["args"]) else [None]:
                    if self.hard - self.step <= 1: break
                    self.act(role, pid=pid, final=True)
                self.drain("notices")
                if self.version != start: return True
            done = True
        for rule in rules if done else []: rule["done"].add(mark)
        return done
    def answer(self, rules, text, reply, mark, lines):
        out = (self.model(Prompts.ACCEPT, {"message": text, "reply": reply, "basket": [{"line": i, "product": s["product"], "listing": self.field(self.assigned.get(i), "title")} for i, s in enumerate(self.items) if self.assigned.get(i) in lines]}, 120) if lines else None) or {}
        for rule in rules if out.get("accepted") is not False else []: rule["done"].add(mark)
        if out.get("accepted") is not False: return
        rejected = [self.assigned[n] for n in Util.items(out.get("rejected")) if self.assigned.get(n) in lines]
        if rejected and self.veto(rejected): return
        ask = "Which item in the basket should I change, and how?"
        obs = self.act("message", text=ask, final=True) if self.can_tell() else None
        if obs is None: return
        reply = self.last.get("lines") or Util.clip(obs, 300)
        self.said.append({"rules": [], "message": ask, "reply": reply, "step": self.step})
        before = self.version
        self.recheck([{"question": ask, "reply": reply}], "replies to the agent's messages")
        for rule in rules if self.version == before else []: rule["done"].add(mark)
    def veto(self, pids):
        fresh = {p for p in pids if p not in self.vetoed and self.spare(p)}
        if not fresh: return False
        self.vetoed |= fresh
        self.version += 1
        self.changes += 1
        return True
    def spare(self, pid):
        return bool([p for i, q in self.assigned.items() if q == pid for p in self.within(i, lambda: [p for p in self.passing() if p != pid])])
    def seen(self, pid):
        return self.read_at.get(pid, -1) > self.placed.get(pid, -1)
    def receipt(self, lines):
        rows = [{"id": p, "count": q, "listing": Util.clip(str(self.field(p, "title") or p), 90), "price_each": self.price(p), "stock": self.field(p, "stock"), "viewed_after_last_cart_change": self.seen(p)} for p, q in lines.items()]
        return {"lines": rows, "total": sum((self.price(p) or 0) * q for p, q in lines.items())}
    def weigh_rules(self, rules, lines, moment):
        mark = self.state(lines, moment)
        key = (mark, moment, len(self.said), self.sent(), tuple(r["quote"] for r in rules))
        if key not in self.verdicts:
            acts, left, last = self.acts(), None if self.allowed is None else self.allowed - self.sent() - self.reserved([r for r in rules if moment == "checkout" or "checkout" not in r["before"]]), max(self.placed.values(), default=0)
            told = [{"message": m["message"], "reply": m["reply"], "for_instructions": m["rules"], "after_last_cart_change": m["step"] > last} for m in self.said]
            data = {"next_action": moment, "next_action_does": ROLE.get(moment, ""), "instructions": {str(n + 1): r["quote"] for n, r in enumerate(rules)}, "basket": self.receipt(lines),
                    "recent_actions": [" ".join([t["role"], Util.clip(str(next(iter(t["args"].values()), "")), 60)]) for t in self.trace[-40:] if "role" in t and t.get("args") is not None], "messages": told + self.talk(), "actions_it_can_take": acts, "messages_left": left, "steps_left": self.hard - self.step}
            out = self.model(Prompts.GATE, data, 700) or {}
            quiet = [str(n + 1) for n, r in enumerate(rules) if r["messages"] and not any(r["quote"] in m["rules"] and m["step"] > last for m in self.said)]
            if out and set(quiet) - set(map(str, Util.items(out.get("unmet")))): out = self.model(Prompts.GATE, {**data, "not_yet_told": quiet}, 700) or out
            steps = [x for x in Util.items(out.get("steps")) if isinstance(x, dict) and x.get("action") in acts]
            unmet = [r for n, r in enumerate(rules) if str(n + 1) in map(str, Util.items(out.get("unmet")))] or rules
            for rule in rules if out else []:
                if not steps or rule not in unmet: rule["done"].add(mark)
            self.verdicts[key] = (unmet, steps, left, mark) if steps else None
        return self.verdicts[key]
    def settled(self, pid):
        return any(r["done"] and r["messages"] for r in self.rules) and not self.can_tell() and self.eligible(pid) and self.passes(pid)


class Agent(Core, Read, Probe, Search, Judge, Ask, Recheck, Finish, Rules):
    def run(self):
        if self.finished: return
        if not self.read(): return
        count, cited = len(self.items), set(map(self.owner, self.refs))
        for place, index in enumerate(sorted(range(count), key=lambda i: i in cited)):
            if index in cited:
                self.focus(index)
                self.pursue(self.refs)
                self.rescale()
                cited = set()
            self.focus(index, count - place + (count > 1))
            self.first_search()
            self.judge()
            while not any(self.cat_ok.values()) and self.rescue(False): self.judge()
            if not any(self.cat_ok.values()): self.explore(self.judge())
        weight, order = [1 + len(s["reqs"]) for s in self.items], self.lines_by(lambda: (any(map(self.passes, list(self.details))), -self.spread()))
        room = self.room()
        for index in order:
            self.focus(index, 2 * sum(weight) / weight[index], room)
            self.settle()
        self.consult()
        gain = lambda: (bool(self.passing()), self.objective != "cheapest" or not self.cheaper(self.best()), not (self.pages() or self.unsearched()))
        for mode, share in ((1, 2), (2, 1)):
            self.focus(self.at)
            self.stage()
            room, weight = self.room(), [1 + len(s["reqs"]) for s in self.items]
            for index in self.lines_by(gain):
                self.focus(index, share * sum(weight) / weight[index], room)
                self.settle(mode)
        self.finish()
    def lines_by(self, key):
        score = self.each(key)
        return sorted(range(len(score)), key=lambda i: (score[i], i))

def agent_main(problem_data):
    agent = Agent(problem_data)
    try:
        agent.run()
    except Exception as error:
        agent.trace.append({"error": repr(error)[:300]})
        agent.salvage()
    finally:
        agent.endgame()
        agent.trace.append({"steps": agent.step, "model_calls": agent.cost["model"][0], "seconds": round(agent.elapsed(), 1), "finished": agent.finished, "judged": agent.decided, "version": agent.version, "asked": [a["property"] for s in agent.items for a in s["asked"]], "items": [[s["product"], s["qty"]] for s in agent.items]})
    return agent.trace
