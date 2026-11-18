"""A buying agent for one shopper in an ORO store world (a fork of the public KYO agent).

The light model drives the store through its own tools. The code holds the ledger (requested items, needs, budget and
the listings seen), plans the questions for the shopper and blocks wasted steps; the heavy model reads the request once,
then checks the cart before an order and the evidence before ending without a purchase.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

WALKER_MODEL = "anthropic/claude-haiku-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"

TURN_CAP = 42
CALL_CAP = 60
SOFT_BUDGET_USD = 0.055
HARD_BUDGET_USD = 0.078
REFUSED_ROOM_USD = 0.015
SURVEY_PER_ITEM = 6         # W2: listings read per item before the question plan (4 with three or more items)
VALUE_SEARCH_STEPS = 8      # W1: store steps the value search after the interview may take
# waits before resending a failed model call; an episode has 300 s in all, so at most 45 s go to waiting
RETRY_PAUSES = (4, 10, 20, 35)
CUT_RETRY_TOKENS = 4000
RETRY_PAUSE_CAP = 45
PREVIEW_CHARS = 450
LOOP_SECONDS = 240          # the store gives an episode 300 s: the model loop stops here and code finishes the episode (T1: was 225)
LATE_SECONDS = 262          # past this, the ending skips the optional step gate (a model call)
monotonic = time.monotonic  # (a clock the local harness can replace)
REPLY_CHAR_CAP = 15000
CHAT_CHAR_CAP = 150000
STEP_GATE_ROUNDS = 3             # store calls the process gate may make before one checkout or ending
EXTRA_PAGES = 6              # further search pages Kyo reads itself for a cheapest, most or least pick
CHEAPER_READS = 3           # cheaper rows of a cart line's kind whose attributes Kyo reads before a cheapest checkout

# one attempt per call (think() resends): a stuck call costs 50 s, not minutes, inside the 300 s
_model_link = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_store_link = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


def _lines(*rows: str) -> str:
    """A prompt kept as its rows."""
    return "\n".join(rows)


def _say(*sentences: str) -> str:
    """One prompt row kept as its sentences."""
    return " ".join(sentences)


WALKER_GUIDE = _lines(
    _say(
        "You are Kyo, a careful buyer working for one shopper in an online store.",
        "You can act only by calling the store's tools.",
        "The episode is graded 1 or 0: it earns 1 only when the finish is exactly right.",
        "A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.",
    ),
    "",
    "HOW THE STORE WORKS",
    _say(
        "- Every tool call is one step and steps are limited; thinking costs nothing.",
        "Tool names, argument names and result shapes are particular to this store: read each tool's description and schema and use the exact argument names and allowed values.",
    ),
    _say(
        "- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page.",
        "Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them.",
        "Keyword queries work best short and value-first (the wanted value plus the product noun).",
        "A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.",
    ),
    _say(
        "- Search rows are summaries.",
        "Stores split a listing's record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either.",
        "The ledger names the tool that shows attributes once Kyo knows it.",
        "Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need.",
        "Never open a listing again when nothing has changed since you read it.",
    ),
    _say(
        "- Questions to the shopper are handled for you by Kyo's interviewer, so you never message the shopper yourself.",
        "Its answers arrive in the conversation and are added to the ledger.",
    ),
    _say(
        "- The request can also ask for steps along the way: an approval, a message, something to check or read first.",
        "The ledger lists them.",
        "Kyo carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store's tools (product questions go through Kyo's interviewer).",
    ),
    "",
    "WHAT COUNTS AS A FIT",
    "- The kind of product is decided by the listing's category label: only the label that matches the requested item as a whole counts, never a neighbouring one.",
    _say(
        "- A requirement on a property is met when the listing's attribute for that property states the value.",
        "Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough.",
        "Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value's words counts (\"deep teal\" meets teal, \"extra roomy fit\" meets roomy).",
        "A value with different words does not count (an abbreviation or code is not the word it stands for, and \"teal green\" is not \"sea green\").",
        "Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.",
    ),
    _say(
        "- An attribute whose key names the requirement's property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say.",
        "A key for a different property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement.",
        "Kyo adds [evidence] lines after opened listings with the keys whose names match a requirement.",
    ),
    _say(
        "- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts.",
        "A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.",
    ),
    "- When the category label itself states a requirement's value (a season or audience in the label), that requirement is met by the category.",
    _say(
        "- A requirement can point at another listing instead of giving a value (\"the same X as the listing sold by <shop> with <property>\", \"within N of the X on the page of ...\").",
        "Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value.",
        "The order counts only when every page of that chain was opened.",
        "When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.",
    ),
    "- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.",
    _say(
        "- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code that both pages carry).",
        "When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.",
    ),
    _say(
        "- Respect exclusions and closed lists.",
        "Wishes (\"ideally\", \"would be nice\") are tie-breakers only.",
    ),
    "- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.",
    "",
    "CHOOSING",
    _say(
        "- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit.",
        "Find the whole fitting set (page through it: result rows are often in no price order, and Kyo reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits.",
        "When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.",
    ),
    "- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.",
    "- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.",
    _say(
        "- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on.",
        "Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.",
    ),
    "- A request for several different items needs one cart line per item, each at its requested quantity.",
    "",
    "CHANGES WHILE YOU WORK",
    _say(
        "- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out.",
        "They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool's result shows it, and the ledger lists the requirements in force.",
        "The episode goes on after a refusal: apply every change, fix the cart and check out again.",
    ),
    "- Re-read a chosen listing's record shortly before checking out, because price and stock can change silently.",
    "",
    "FINISHING",
    "- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.",
    _say(
        "- End without a purchase only after a real search finds no listing that meets everything.",
        "Running short of steps proves nothing of the kind: then check out the best cart you have.",
        "Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.",
    ),
    _say(
        "- Every reply must call at least one tool.",
        "Keep your text short: what you know, then the next move.",
    ),
)

BRIEF_PROMPT = _lines(
    _say(
        "You prepare a shopping task for an agent.",
        "Read the shopper's request and the store's tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:",
    ),
    "",
    _say(
        "\"items\": a list with one entry per distinct product the shopper wants bought, each {\"name\": the kind of product in the request's words, \"shelf\": the store's category label for it when a tool lists labels (copy it exactly), else \"\", \"quantity\": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), \"musts\": a list of firm requirements on that product, each \"property: constraint\" with the exact value, limit, exclusion or closed list.",
        "Name the property with the request's own word for it (\"meant for a lakeside occasion\" gives \"occasion: lakeside\", \"a 12-inch blade\" gives \"blade length:",
        "12 inch\"); do not swap it for a neighbouring property.",
        "A must whose value is read from another listing is ONE must holding the whole description of that listing (\"rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid\"): words that describe that other listing (its shop, its kind, \"with <property> <value> and <property> <value>\" after it) stay inside that one must and are never musts of this product}.",
        "Never put price, budget, stock or quantity into musts.",
        "The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season, an audience or a use named with the product) into musts: those belong to \"name\".",
        "A product the request names in one phrase joined by \"and\" (one \"I want ...\" with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.",
    ),
    "\"budget\": the most the shopper will pay in total as a number in the request's own unit, or null.",
    "\"budget_unit\": that unit in words (for example \"cents\"), or null.",
    _say(
        "\"budget_scope\":",
        "\"total\" when the limit covers the whole order, \"each\" when it is per item, null when there is no limit.",
    ),
    "\"in_stock_only\": true unless the request clearly accepts items that are not available.",
    _say(
        "\"pick\":",
        "\"cheapest\" when the request asks for the lowest price or smallest total, \"most\" or \"least\" when it asks for the extreme of some measure, else \"any\".",
        "Remarks that money is tight are not a pick rule.",
    ),
    "\"pick_measure\": what is maximised or minimised for \"most\"/\"least\", else \"\".",
    "\"messages\": how many messages the shopper allows the agent to send (integer), or null when not stated.",
    "\"steps\": how many steps, moves or tool calls the request allows (integer), or null.",
    "\"lookups\": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].",
    "\"rules\": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.",
    "\"wishes\": a list of things the shopper would merely like.",
    "\"ask_first\": a list of things the request explicitly says to check with the shopper before choosing (empty when none).",
    _say(
        "\"process\": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {\"step\": what must be done, in at most 15 words, \"when\":",
        "\"choose\" (before the products are chosen), \"order\" (right before checking out) or \"end\" (just before ending without a purchase), \"messages\": how many messages to the shopper it takes (0 when none)};",
        "[] when the request asks for no such step.",
        "Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.",
    ),
    "\"quiet_decline\": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.",
    _say(
        "\"roles\": an object naming the exact tool for each job, or null when no tool does it:",
        "\"search\", \"open\" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), \"offer\" (shows one listing's current offer, such as its price and stock, when that is a different tool from \"open\"), \"compare\" (several listings in full), \"ask\" (sends a message to the shopper), \"cart\" (changes the cart), \"cart_view\" (shows the cart), \"buy\" (checks out), \"refuse\" (ends without a purchase).",
    ),
    "\"buy_args\": the fixed arguments that make the \"buy\" tool check out (for a tool with a mode argument, that mode), else {}.",
    "\"refuse_args\": the fixed arguments that make the \"refuse\" tool end without a purchase, else {}.",
    "",
)

ASK_PLAN_PROMPT = _lines(
    _say(
        "You choose the questions a shopping agent sends to a shopper.",
        "Shoppers hold a few firm preferences they did not write down, each about one property of one requested product.",
        "A question reveals a preference only when it names that exact property, and each question is answered only about the first property it names, so every question costs one property.",
    ),
    "",
    _say(
        "You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry the key and example values.",
        "An unwritten preference may be on a property that no listed key names.",
    ),
    "",
    "Choose the properties in this order:",
    _say(
        "0.",
        "What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.",
    ),
    _say(
        "1.",
        "Hinted properties.",
        "The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product's looks or the setting it belongs in, the person meant to use or wear it, or what is inside it.",
        "For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.",
    ),
    "Name a hinted property by a listed key of that item when one fits: a hint about who will wear it points at gender; a hint about its looks, its character or the setting it belongs in points at the key that defines how that kind of product looks (its style, pattern, construction or the like), and at colour only when the story itself speaks of colours, shades or matching a colour.",
    _say(
        "2.",
        "Then the listed keys, ranked: favour the properties that set one listing of that kind of product apart from another (often its material, colour, type or style, its form, fit or construction, its capacity or a defining dimension, whichever that kind of product is bought by) and keys carried by most of the item's listings; a plain size is seldom the unwritten preference unless the story points at it; avoid keys seen on only one or two of them.",
    ),
    _say(
        "Skip a property the request already settles for that item, a second name for a property already chosen, a key that lists purchase variants (a mix of sizes, colours and pack counts), identifiers, warranty and other logistics.",
        "With several items, put every item's hinted properties first, then take turns between the items.",
    ),
    "\"label\" is the property's plain name: for a listed key, all of the key's words in plain spelling (\"lid colour\" for lidColour, \"rim finish\" for rim_finish; never shortened to \"colour\" or \"finish\"); only the property's name, no example values, no brackets, no second property.",
    _say(
        "Choose up to COUNT, best first.",
        "Return ONE JSON object and nothing else: {\"ask\":",
        "[{\"item\": item number, \"label\":",
        "\"...\", \"hint\":",
        "\"at most 8 of the story's words it comes from, or an empty string\"}]}",
    ),
)

ANSWER_PROMPT = _lines(
    _say(
        "A shopping agent asked a shopper some questions.",
        "Read each answer and return ONE JSON object and nothing else:",
    ),
    _say(
        "{\"needs\":",
        "[{\"item\": item number, \"need\":",
        "\"property: constraint\", \"points\": true when the value is to be read from another listing, else false}], \"drop\":",
        "[{\"item\": item number, \"property\":",
        "\"a requirement the shopper now gives up\"}], \"free\":",
        "[{\"item\": item number, \"property\":",
        "\"...\"}], \"crowded\":",
        "[numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], \"budget\": new total limit as a number or null}",
    ),
    _say(
        "- \"needs\": every firm preference an answer states, with the exact value or limit and its unit (\"at least 30 cm\", \"oval\", \"not wicker\").",
        "Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need.",
        "Copy the value word for word as the answer says it, keeping its nouns and units (\"at least 3 shelves\", not \"at least 3\").",
    ),
    _say(
        "- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with \"points\": true.",
        "Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it.",
        "The properties describing the other product are not needs of this item.",
    ),
    "- \"free\": the property of each question the shopper said they do not mind about; when a question names several properties and the answer does not say which one it is about, give no \"free\" entry for it.",
    "- Folksy or roundabout wording that names a value still states that value.",
)

NOTICE_PROMPT = _lines(
    _say(
        "While a shopping agent works, the store showed this note (it may come from the shopper or the store).",
        "Return ONE JSON object and nothing else:",
    ),
    _say(
        "{\"kind\":",
        "\"change\" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), \"error\" when it only reports that a call was malformed or not accepted, \"other\" for anything else, \"budget\": the new total limit as a number in the request's unit, or null, \"add\":",
        "[{\"item\": item number, \"need\":",
        "\"property: constraint\"}], \"drop\":",
        "[{\"item\": item number, \"need\":",
        "\"the requirement no longer wanted\"}], \"gone\":",
        "[listing ids said to be unavailable], \"sold_out\": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), \"price_moved\": true when it says a price changed, \"summary\":",
        "\"one short sentence\"}",
    ),
    _say(
        "A replaced requirement is a drop of the old one plus an add of the new one.",
        "Report every change in the note.",
    ),
)

CART_CHECK_PROMPT = _lines(
    _say(
        "You are the last check before a shopping agent checks out.",
        "Decide from the records shown whether this cart is exactly right.",
    ),
    _say(
        "Reading rules: a requirement is met when the listing's attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value's words counts, such as \"deep teal\" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the listing must be the requested kind of product, but a store category is often a wider shelf than the product's name (a footwear shelf holds water shoes and boat shoes, a memory shelf holds laptop RAM): a listing on such a shelf is that kind when its title, description or a type attribute says so, or when it meets every stated requirement and nothing in its record says it is another kind; a rule of being unlike another listing's value counts as met unless the records show that same value.",
        "Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget.",
        "The newest record of a listing wins.",
    ),
    "When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.",
    _say(
        "Return ONE JSON object and nothing else: {\"ok\": true or false, \"problems\":",
        "[\"short, concrete problem\", ...], \"better\":",
        "[listing ids that should replace a cart line, best first]}.",
        "Use ok=false only for a problem the records clearly show or a requirement no record addresses.",
    ),
)

FIT_RULES = "the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value's words counts, such as \"deep teal\" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body's colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the listing must be the requested kind of product, but a store category is often a wider shelf than the product's name (a footwear shelf holds water shoes and boat shoes, a memory shelf holds laptop RAM): a listing on such a shelf is that kind when its title, description or a type attribute says so, or when it meets every stated requirement and nothing in its record says it is another kind; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit."

DECLINE_REVIEW_PROMPT = _say(
    "A shopping agent wants to end without buying because it thinks nothing fits.",
    "Check the listing records shown against the requirements, with these reading rules: ",
) + FIT_RULES + _lines(
    "",
    "Records marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.",
    _say(
        "Return ONE JSON object and nothing else: {\"fits\": {\"item number\":",
        "[listing ids that appear to meet every requirement, cheapest first, at most 4]}, \"open\":",
        "[ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], \"reason\":",
        "\"one sentence\"}",
    ),
)

REF_SPLIT_PROMPT = _say(
    "A shopper's request was read into firm requirements on a product.",
    "Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with \"with ...\") are sometimes read by mistake as requirements on the product being bought.",
    "Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing.",
    "Return ONE JSON object and nothing else: {\"describe_other\":",
    "[those requirements copied exactly; empty when every one is a real requirement on the product]}",
)

CHAIN_PROMPT = _lines(
    _say(
        "A shopping agent is about to check out.",
        "Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property.",
        "The order counts only when the agent has opened in full every listing along each such chain.",
        "Follow each requirement's chain from the requirement's own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page;",
        "ROW: seen only as a search result row, its page not read).",
    ),
    _say(
        "Return ONE JSON object and nothing else: {\"chains\":",
        "[{\"need\": the requirement copied exactly, \"links\":",
        "[{\"id\": the listing id when a record below is that link, else \"\", \"about\": how the requirement or the previous page describes that listing}], \"done\": true when the last link is an OPENED page that states the property's value itself, else false, \"value\": the property's value as stated on that last page, or \"\"}]}",
    ),
    "List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.",
)

STEP_GATE_PROMPT = _lines(
    _say(
        "A shopping agent is about to STAGE.",
        "The shopper's request asks for the steps below along the way.",
        "Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now.",
        "A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product's properties do not do a step.",
    ),
    _say(
        "When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments.",
        "A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need.",
        "Give no call when every due step is done or when the messages or steps left cannot do the missing one.",
    ),
    _say(
        "Return ONE JSON object and nothing else: {\"steps\":",
        "[{\"step\": the step, \"done\": true or false, \"lines\":",
        "[the RECORD line numbers that show it done]}], \"call\": {\"tool\":",
        "\"...\", \"args\": {...}} or null, \"hold\": what a reply asks to change before going on, or \"\", \"needs\":",
        "[{\"item\": item number, \"need\":",
        "\"property: constraint\"} for each firm requirement such a reply states]}",
    ),
)

SOLD_SWAP_PROMPT = _say(
    "A listing in a shopping agent's cart has just sold out.",
    "Find replacements for that cart line among the other listing records shown, with these reading rules: ",
) + FIT_RULES + _lines(
    "",
    _say(
        "Return ONE JSON object and nothing else: {\"fits\":",
        "[{\"id\": listing id, \"price\": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line's item, at most 6], \"reason\":",
        "\"one sentence\"}",
    ),
)

RESULT_SHAPE_PROMPT = _say(
    "A store tool returned the result below.",
    "Name its parts by the exact keys the result and the tool's parameters use.",
    "Return ONE JSON object and nothing else: {\"rows\": the key of the list of listings, \"count\": the key holding how many listings matched in all (not how many are shown), or null, \"next\": the key holding the value that asks for the following page, or null, \"next_arg\": the tool parameter that takes that value, or null, \"price\": the key, inside a row, of the listing's price, or null, \"stock\": the key, inside a row, that says whether the listing can be bought now, or null}",
)



_RA_COMPACT = {"ensure_ascii": False, "separators": (",", ":")}
_RA_FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.S)
_RA_CAMEL = re.compile(r"([a-z0-9])([A-Z])")
_RA_SEPARATORS = re.compile(r"[\s_.\-]+")
_RA_SIGNED_NUMBER = re.compile(r"\s*-?\d+(\.\d+)?\s*")
_RA_WHOLE_NUMBER = re.compile(r"\s*\d+(\.0+)?\s*")
_RA_DIGIT = re.compile(r"\d")
_RA_EXPECTED_STEP = re.compile(r"expected (\d+)")
_RA_NO_CHANGE = object()      # marker: a schema kind offers no conversion for a value
_RA_RESEND = (402, 408, 425, 429)
_RA_REJECTED = (400, 401, 402, 403, 404, 422)
_RA_CLOSED_WORDS = ("finish", "quarantin", "finaliz")


def _to_json(value: Any) -> str:
    try:
        text = json.dumps(value, **_RA_COMPACT)
    except (TypeError, ValueError):
        text = str(value)
    return text


def _trim_value(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    deeper = depth + 1
    if isinstance(value, dict):
        clipped = {}
        for key, inner in value.items():
            clipped[key] = _trim_value(inner, limit, deeper)
        return clipped
    if isinstance(value, list):
        return list(map(lambda inner: _trim_value(inner, limit, deeper), value))
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value


def _ra_load_dict(text: str) -> tuple[bool, dict[str, Any] | None]:
    """(whether the text parsed as JSON, the parsed value when it is an object)."""
    try:
        parsed = json.loads(text)
    except ValueError:
        return False, None
    if not isinstance(parsed, dict):
        parsed = None
    return True, parsed


def _parse_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = _RA_FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    parsed, found = _ra_load_dict(body)
    if parsed:
        return found
    # fall back to the outermost braces inside surrounding prose
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    return _ra_load_dict(body[start:end + 1])[1]


_M4_PART = re.compile(r" \| (?=[A-Za-z_][A-Za-z0-9_]*: )")
_M4_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_M4_FACT = re.compile(r"; (?=[A-Za-z_][\w ]{0,40}=)")


def _m4_text_record(text: Any) -> dict[str, Any] | None:
    """M4: "key: value | key: value | ..." (three or more parts, one-word keys) as a record; None for any other text."""
    if not isinstance(text, str) or " | " not in text:
        return None
    parts = _M4_PART.split(text)
    if len(parts) < 3:
        return None
    record: dict[str, Any] = {}
    for part in parts:
        key, sep, value = part.partition(": ")
        if not sep or not _M4_KEY.fullmatch(key):
            return None
        value = value.strip()
        if value in ("None", "True", "False"):
            record[key] = None if value == "None" else value == "True"
        elif re.fullmatch(r"-?\d+", value):
            record[key] = int(value)
        elif re.match(r"[A-Za-z_][\w ]{0,40}=", value) and all(re.fullmatch(r"[A-Za-z_][\w ]{0,40}=[^;]*", bit) for bit in _M4_FACT.split(value)):
            # every "; " part a "name=value" pair: a dict of facts
            record[key] = {bit.partition("=")[0].strip(): bit.partition("=")[2].strip() for bit in _M4_FACT.split(value)}
        else:
            record[key] = value
    return record


_S2_SINGLE = re.compile(
    r"\b(?:one|a single|1)\s+(?:question|thing|topic|subject|query|queries|property|point|prompt|detail|item|matter)s?"
    r"\s+(?:at a time|per (?:message|turn|text|reply|go))"
    r"|\b(?:questions?|messages?|ask(?:ing)?|texts?)\b[^.?!]{0,40}\b(?:one at a time|one by one|separately|singly)"
    r"|\bone at a time\b[^.?!]{0,30}\b(?:questions?|ask|messages?)"
    r"|\b(?:don'?t|do not|never|please no|no)\s+(?:bundle|stack|pile|combine|lump|cram)[^.?!]{0,30}\bquestions?",
    re.I)


def _plain_words(key: str) -> str:
    spaced = _RA_CAMEL.sub(r"\1 \2", str(key))
    return _RA_SEPARATORS.sub(" ", spaced).strip().lower()


def _m2_words(text: str) -> set[str]:
    """M2: the words of a property name, spelling-blind ("Colours" = "color"), small words dropped."""
    words = set()
    for word in re.findall(r"[a-z0-9]+", _plain_words(text)):
        if word in ("of", "the", "a", "an", "for", "and", "or", "to", "in", "on", "with", "fa"):
            continue
        word = word.replace("colour", "color")
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") else word)
    return words


def _ra_singular(word: str) -> str:
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


def _word_stems(text: str) -> frozenset[str]:
    return frozenset(map(_ra_singular, _plain_words(text).split()))


def _names_property(prop: frozenset[str], key: frozenset[str]) -> bool:
    if prop and key:
        if prop == key or prop <= key:
            return True
        return len(key) >= 2 and key <= prop
    return False


def _schema_types(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if isinstance(declared, str):
        return [declared]
    return list(filter(lambda entry: isinstance(entry, str), declared or []))


def _ra_is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _ra_is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


_RA_KIND_TESTS = {
    "string": lambda value: isinstance(value, str),
    "integer": _ra_is_integer,
    "number": _ra_is_number,
    "boolean": lambda value: isinstance(value, bool),
    "array": lambda value: isinstance(value, list),
    "object": lambda value: isinstance(value, dict),
    "null": lambda value: value is None,
}


def _matches_type(value: Any, kind: str) -> bool:
    test = _RA_KIND_TESTS.get(kind)
    if test is None:
        return True
    return test(value)


def _ra_fits_some(value: Any, kinds: list[str]) -> bool:
    for kind in kinds:
        if _matches_type(value, kind):
            return True
    return False


def _ra_convert(value: Any, kind: str) -> Any:
    """The value turned into one schema kind, or _RA_NO_CHANGE when that kind has no conversion for it."""
    if kind == "integer" and isinstance(value, float) and value.is_integer():
        return int(value)
    if kind in ("integer", "number") and isinstance(value, str) and _RA_SIGNED_NUMBER.fullmatch(value):
        number = float(value)
        if kind == "integer" or number.is_integer():
            return int(number)
        return number
    if kind == "string" and _ra_is_number(value):
        return str(value)
    if kind == "boolean" and isinstance(value, str):
        flag = value.strip().lower()
        if flag in ("true", "false"):
            return flag == "true"
    if kind == "array" and value is not None and not isinstance(value, list):
        return [value]
    return _RA_NO_CHANGE


def _ra_pick_option(value: Any, options: list[Any], where: str, faults: list[str]) -> Any:
    if value in options:
        return value
    if isinstance(value, str):
        wanted = value.strip().lower()
        same = next((o for o in options if isinstance(o, str) and o.lower() == wanted), None)
        if same is not None:
            value = same
    if value not in options:
        faults.append(f"{where}: allowed values are {options}")
    return value


def _ra_bound_scalar(value: Any, schema: dict[str, Any], where: str, faults: list[str]) -> Any:
    if isinstance(value, str):
        longest, shortest = schema.get("maxLength"), schema.get("minLength")
        if isinstance(longest, int) and len(value) > longest:
            value = value[:longest]
        if isinstance(shortest, int) and len(value) < shortest:
            faults.append(f"{where} is too short")
    elif _ra_is_number(value):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            faults.append(f"{where} is below {low}")
        if isinstance(high, (int, float)) and value > high:
            faults.append(f"{where} is above {high}")
    return value


def _ra_shape_list(value: list[Any], schema: dict[str, Any], where: str, faults: list[str]) -> list[Any]:
    most, least = schema.get("maxItems"), schema.get("minItems")
    if isinstance(most, int) and len(value) > most:
        faults.append(f"{where} allows at most {most} entries")
    if isinstance(least, int) and len(value) < least:
        faults.append(f"{where} needs at least {least} entries")
    inner = schema.get("items")
    if not isinstance(inner, dict):
        return value
    shaped = []
    for index, entry in enumerate(value):
        shaped.append(_conform(entry, inner, f"{where}[{index}]", faults))
    return shaped


def _ra_shape_dict(value: dict[str, Any], schema: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
    declared = schema.get("properties")
    props = declared if isinstance(declared, dict) else {}
    if schema.get("additionalProperties") is False:
        for name in value:
            if name not in props:
                faults.append(f"{where}.{name} is not an allowed argument")
    for name in schema.get("required") or []:
        if name not in value:
            faults.append(f"{where}.{name} is required")
    shaped = {}
    for key, entry in value.items():
        if key in props:
            entry = _conform(entry, props[key], f"{where}.{key}", faults)
        shaped[key] = entry
    return shaped


def _conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Nudge a value toward its JSON schema and record in faults whatever stays wrong."""
    if not isinstance(schema, dict):
        return value
    kinds = _schema_types(schema)
    if kinds and not _ra_fits_some(value, kinds):
        for kind in kinds:
            converted = _ra_convert(value, kind)
            if converted is not _RA_NO_CHANGE:
                value = converted
                break
        if not _ra_fits_some(value, kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list):
        value = _ra_pick_option(value, options, where, faults)
    value = _ra_bound_scalar(value, schema, where, faults)
    if isinstance(value, list):
        return _ra_shape_list(value, schema, where, faults)
    if isinstance(value, dict):
        return _ra_shape_dict(value, schema, where, faults)
    return value


_V1_FILLER = frozenset("a an the of to for and or with in on at by be is it its that this not no any very just exactly about around".split())
_V1_RULE_VERB = re.compile(r"\b(?:has to|have to|must|should|got to|gotta|needs? to|needs? it to|wants? (?:it )?to be)\b", re.I)
_V1_NUMBERS = {w: str(n) for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
_V1_POINTER = re.compile(r"\b(?:page|listing|same as|other than|works alongside)\b", re.I)
_V1_LINK = re.compile(r"\b(?:fit|fits|fitting|compatib\w*|match\w*|works? with)\b", re.I)


def _v1_words(text: str) -> set[str]:
    spaced = re.sub(r"(?<=[0-9])(?=[a-z])|(?<=[a-z])(?=[0-9])", " ", str(text).lower())
    return {_V1_NUMBERS.get(w, _ra_singular(w)) for w in re.sub(r"[^a-z0-9]+", " ", spaced).split()} - _V1_FILLER


def _v1_stated(need: str, request: str, name: str = "") -> bool:
    """V1: the request states this must. A sentence holds most words of the value and names the property, or one clause
    holds them with a rule verb (has to / must / got to ...); a value the item's own name carries (a v-neck top) and a
    link to another product or listing are always stated."""
    prop, _, value = need.partition(":")
    # a fit with another product is a link the listings must show, never a story wish
    if _V1_POINTER.search(value) or _V1_LINK.search(need):
        return True
    # "leather, not synthetic": the leading value is what has to be stated
    value = re.split(r",|;|\bnot\b|\brather than\b", value)[0]
    value_words = _v1_words(value)
    if not value_words or value_words <= _v1_words(name):
        return True
    prop_words = _v1_words(prop)
    floor = max(1, -(-3 * len(value_words) // 5))
    for sentence in re.split(r"(?<=[.!?])\s+", request):
        words = _v1_words(sentence)
        if len(value_words & words) >= floor and prop_words & words:
            return True
        # clause by clause for the verb: "a ring for my right hand, and it needs to be easy to wear" states no hand rule
        for clause in re.split(r"[,;]\s+(?=(?:and|but|so|as|because)\b)", sentence):
            if len(value_words & _v1_words(clause)) >= floor and _V1_RULE_VERB.search(clause):
                return True
    return False


_V2_TEMPLATE = re.compile(r"\bthe ([a-z0-9][\w'/ -]{0,40}?) (?:has to|have to|must|needs? to|should|got to|is to)\b(.*?)"
                          r"(?=,? and the [a-z0-9][\w'/ -]{0,40}? (?:has to|have to|must|needs? to|should|got to)\b|[.;!?](?:\s|$)|$)", re.I | re.S)


def _v2_stated(need: str, request: str, name: str = "") -> bool:
    """V2: with the requirement template in the request, a must is stated only by a template clause that names its
    property or carries its value; without any template clause the V1 test decides."""
    prop, _, value = need.partition(":")
    if _V1_POINTER.search(value) or _V1_LINK.search(need):
        return True
    clauses = [(m.group(1), m.group(2)) for m in _V2_TEMPLATE.finditer(request)]
    if not clauses:
        return _v1_stated(need, request, name)
    value = re.split(r",|;|\bnot\b|\brather than\b", value)[0]
    value_words = _v1_words(value)
    if not value_words or value_words <= _v1_words(name):
        return True
    prop_words = _v1_words(prop)
    floor = max(1, -(-3 * len(value_words) // 5))
    for subject, rest in clauses:
        subject_words = _v1_words(subject)
        if prop_words & subject_words or (len(value_words & _v1_words(rest)) >= floor and len(subject_words) <= 4):
            return True
    return False


_V5_APPROVAL = re.compile(r"\b(?:yes|approv\w*|confirm\w*|okay|ok\b|o\.k\.|permission|sign(?:s|ed)? off|go-ahead|go ahead"
                          r"|check(?:s|ed)? with me|run (?:it|that|them) (?:by|past) me|agree\w*|consent)\b", re.I)
_V3_LINKING = re.compile(r"^(?:(?:has |have )?(?:got )?(?:to )?(?:match(?:es|ing)?|equals?|equal to|be|being|is|as|landing on|lands? on"
                         r"|come in as|comes in as|locked to|stays? locked to|set to|stand(?:s)? firm at)\s+)+", re.I)


def _v3_clean(need: str) -> str:
    """V3: "style: match contemporary" -> "style: contemporary"; a pointer at another listing is left alone."""
    prop, colon, value = need.partition(":")
    if not colon:
        return need
    rest = _V3_LINKING.sub("", value.strip())
    if not rest or rest == value.strip() or _V1_POINTER.search(rest) or re.search(r"\b(?:whatever|wherever|page|listing)\b", rest, re.I):
        return need
    return f"{prop}: {rest}"


_W5_NON_PROPERTY = re.compile(r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*"
                              r"|videos?|urls?|links?|images?|photos?|condition)\b", re.I)
_C1_POINTER = re.compile(r"\b(?:refer|take|taken|go by|look (?:at|to)|use the|use its|from the|comes? from|instead of this"
                         r"|rather than|wherever|on the page for|matching|mates up|same as|unlike|other than)\b", re.I)
_C1_STOP = {"the", "and", "with", "for", "from", "that", "this", "one", "listing", "listings", "page", "item", "items", "its",
            "his", "her", "whatever", "wherever", "ends", "sending", "you", "exactly", "anything", "else", "rather", "than",
            "instead", "refer", "take", "taken", "look", "use", "go", "by", "of", "on", "in", "to", "is", "are", "has", "have",
            "made", "comes", "come", "matching", "match", "matches", "same", "like", "type", "mates", "up", "kind", "product",
            "store", "shop", "sold", "seller", "piece", "version", "versions", "not", "definitely", "unlike", "other", "any"}
_W1_LIMIT_WORDS = re.compile(r"\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below"
                             r"|above|less|more|between|within|up to|or)\b", re.I)


def _w_roots(text: str) -> set[str]:
    """Content words of a label with a plural ending taken off."""
    roots = set()
    for word in _plain_words(text).split():
        if re.search("(?:ch|sh|ss|x|z)es$", word):
            word = word[:-2]
        elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        if len(word) >= 3 and word not in ("and", "for", "the", "with"):
            roots.add(word)
    return roots


_P1_GENDER = re.compile(r"\b(?:gender|department|sex|target audience)\b", re.I)
_P1_SOLD_BY = re.compile(r"\b(?:men'?s|women'?s|ladies|unisex|boys'?|girls'?|male|female)\b", re.I)


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if not isinstance(value, (str, dict)) or not value:
        return []
    return [value]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else None
    if isinstance(value, str) and _RA_WHOLE_NUMBER.fullmatch(value):
        return int(float(value))
    return None


def _looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    core = value.strip()
    if " " in core or not 3 <= len(core) <= 64:
        return False
    return named or _RA_DIGIT.search(value) is not None


def _ra_id_score(rows: list[dict[str, Any]], key: Any) -> tuple[bool, float] | None:
    """How well a key serves as the row id (named for it, mean length), or None when it cannot serve."""
    named = "id" in str(key).lower()
    raw = [row.get(key) for row in rows]
    if not all(_looks_like_id(entry, named) for entry in raw):
        return None
    values = [entry.strip() for entry in raw]
    if len(rows) > 1 and len(set(values)) < len(values):
        return None
    return named, sum(map(len, values)) / len(values)


def _guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    """Pick the row key that works as an id: spaceless strings (with a digit unless the key says id), distinct, longest."""
    chosen, top = None, None
    for key in rows[0]:
        score = _ra_id_score(rows, key)
        if score is not None and (top is None or score > top):
            chosen, top = key, score
    return chosen


def _ra_holds_id(record: Any, listing_id: str) -> bool:
    if not isinstance(record, dict):
        return False
    return any(isinstance(field, str) and str(field).strip() == listing_id for field in record.values())


def _mentions_id(node: Any, listing_id: str, depth: int = 0) -> bool:
    """True when some list inside the result has a record with this id, as a cart view does while the line is in it."""
    if depth > 5:
        return False
    if isinstance(node, dict):
        children = node.values()
    elif isinstance(node, list):
        if any(_ra_holds_id(record, listing_id) for record in node):
            return True
        children = node
    else:
        return False
    return any(_mentions_id(child, listing_id, depth + 1) for child in children)


def _ra_flat_fields(block: dict[str, Any]) -> dict[str, Any]:
    kept = {}
    for key, field in block.items():
        if isinstance(field, str) or _ra_is_number(field):
            kept[key] = field
    return kept


def _attribute_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    richest = None
    for block in node.values():
        if not isinstance(block, dict) or len(block) < 2:
            continue
        fields = _ra_flat_fields(block)
        if len(fields) < 2:
            continue
        if richest is None or len(fields) > len(richest):
            richest = fields
    return richest


def _ra_usable_tool(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    spec = entry.get("function")
    return isinstance(spec, dict) and isinstance(spec.get("name"), str)


def _ra_transient(status_text: Any) -> bool:
    """Failures worth sending again: no credit, rate limits, timeouts, server errors and no status at all."""
    if status_text is None or status_text in _RA_RESEND:
        return True
    return isinstance(status_text, int) and status_text >= 500


def _ra_cache_marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copies of the messages with a prompt-cache mark on the first and the last text message."""
    marked = [dict(m) for m in messages]
    for entry in (marked[0], marked[-1]):
        text = entry.get("content")
        if isinstance(text, str) and text:
            entry["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
    return marked


class Wanted:
    def __init__(self, name: str, shelf_name: str, count_wanted: int, firm_needs: list[str]):
        self.name, self.shelf_name, self.count_wanted, self.firm_needs = name, shelf_name, count_wanted, firm_needs
        self.told: list[str] = []
        self.given_up: list[str] = []
        self.open_props: list[str] = []


class Buyer:
    def __init__(self, problem: dict[str, Any]):
        setting = problem["environment"]
        self.env_binding = dict(setting["binding"])
        self.policy = setting["policy_view"]
        self._ra_take_view()
        self._ra_zero_counters()
        self._ra_empty_ledger()

    def _ra_take_view(self) -> None:
        policy = self.policy
        self.request_text = str(policy.get("query") or "")
        self.tools = list(filter(_ra_usable_tool, policy.get("tools") or []))
        self.tool_schemas = {}
        for tool in self.tools:
            spec = tool["function"]
            params = spec.get("parameters")
            if not isinstance(params, dict):
                params = {"type": "object", "properties": {}}
            self.tool_schemas[spec["name"]] = params
        limit = policy.get("max_steps")
        steps = _as_int(limit)
        if steps is None and isinstance(limit, float):
            steps = int(limit)
        self.step_cap = steps if (steps or 0) > 0 else 30
        self.is_over = bool(policy.get("finished"))

    def _ra_zero_counters(self) -> None:
        self.steps_used = self.paused = self.empty_replies = self.model_calls = self.turns = 0
        self.started_at = monotonic()
        self.loosened = self.loosen_seen = 0
        self.message_cap = self.messages_sent = self.step_reserve = self.gate_messages = 0
        self.epoch = self.checkout_holds = self.decline_holds = self.checkouts = self.refusals = 0
        self.chain_cache = self.chain_holds = self.side_ok_size = self.side_fail_size = self.pages_done = 0
        self.spend_usd = 0.0
        self.cart_changed_at = self.cart_viewed_at = self.refusal_step = -1
        self.end_refused_at = -99
        self.repeat_skips, self.stuck = 0, False
        self.aborted = self.out_of_credit = self.interviewed = self.cart_used = self.reviewed = False
        self.note_said_sold = self.search_pushed = self.late_hold = self.cart_completed = self.wider_done = False
        self.budget: Any = None
        self.cart_snapshot: Any = None
        self.confirmed_id: str | None = None
        self.id_field: str | None = None
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.cart_reply = ""

    def _ra_empty_ledger(self) -> None:
        # what the episode has learnt so far: brief, items, listings, notes and the cart
        self.plan_brief: dict[str, Any] = {}
        self.jobs: dict[str, str] = {}
        self.items: list[Wanted] = []
        self.process_steps: list[dict[str, Any]] = []
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.opened_records: dict[str, dict[str, Any]] = {}
        self.seen_ids: list[str] = []
        self.note_texts: list[str] = []
        self.cart_qty: dict[str, int] = {}
        self.offer_read_at: dict[str, int] = {}
        self.sold_at: dict[str, int] = {}
        self.review_items: dict[str, list[str]] = {}
        self.cart_verdicts: dict[str, bool] = {}
        self.shows_attrs: dict[str, bool] = {}
        self.ref_text: dict[str, str] = {}
        self.tool_fields: dict[str, set[str]] = {}
        self.view_fields: dict[str, tuple[int, set[str]]] = {}
        self.read_memo: dict[str, tuple[int, int, str]] = {}
        self.result_shapes: dict[str, dict[str, Any] | None] = {}
        self.log: list[dict[str, Any]] = []
        self.record_lines: list[str] = []
        self.record_steps: list[int] = []
        self.review_leads: list[str] = []
        self.best_cart: list[str] = []
        self.pending_notes: list[str] = []
        self.recent_calls: list[str] = []
        self.dialog: list[dict[str, Any]] = []
        self.ref_needs: list[tuple[int, str]] = []
        self.extra_pages: list[Any] = []
        self.gate_passed: set[tuple[str, str]] = set()
        self.notes_heard: set[str] = set()
        self.checked_ids: set[str] = set()
        self.unbuyable: set[str] = set()
        self.sold_phrases: set[str] = set()
        self.id_params: set[tuple[str, str]] = set()
        self.given_up: set[str] = set()
        self.opened: set[str] = set()
        self.cheaper_seen: set[str] = set()
        self.searched: list[dict[str, Any]] = []

    # ---------- plumbing ----------
    def seconds_used(self) -> float:
        return monotonic() - self.started_at

    def steps_left(self) -> int:
        return self.step_cap - self.steps_used

    def messages_left(self) -> int:
        """Messages left for the interviewer once those the order or ending steps still need are held back."""
        held_back = self.step_reserve - self.gate_messages
        if held_back < 0:
            held_back = 0
        return self.message_cap - self.messages_sent - held_back

    def cart_stamp(self) -> str:
        lines = sorted(self.cart_qty.items())
        return _to_json(lines)

    def page_viewers(self) -> set[str]:
        """Tools that count as opening a listing's page: the offer view if the store keeps one beside the attributes
        view (attributes alone are not the page), otherwise the single-listing view; plus the side-by-side view."""
        found = set()
        for tool in (self.jobs.get("offer") or self.jobs.get("open"), self.jobs.get("compare")):
            if tool:
                found.add(tool)
        return found

    def attribute_viewer(self) -> str | None:
        """Single-listing tool for attributes: the first that has shown some, failing that the brief's attributes view."""
        for tool, showed in self.shows_attrs.items():
            if showed and self.id_param(tool):
                if tool:
                    return tool
                break
        tool = self.jobs.get("open")
        if self.shows_attrs.get(tool or "") is False:
            return None
        return tool if self.id_param(tool) else None

    def side_view_has_attrs(self) -> bool | None:
        """True or False once the side-by-side view has shown whether it carries attributes; None until then."""
        side = self.jobs.get("compare") or ""
        return self.shows_attrs.get(side)

    def side_view_bare(self) -> bool:
        """Whether attributes cannot be read from the side-by-side view: it showed none, or it is untried while a separate
        attributes view exists (one that showed attributes, or one apart from the offer view)."""
        seen = self.side_view_has_attrs()
        if seen is not None:
            return seen is False
        if self.shows_attrs.get(self.attribute_viewer() or "") is True:
            return True
        return bool(self.jobs.get("offer"))

    def steps_due(self, stage: str) -> bool:
        # some step belongs to this stage and the gate has not yet passed it for the current cart
        if all(p["when"] != stage for p in self.process_steps):
            return False
        return (stage, f"{self.cart_stamp()}#{self.epoch}") not in self.gate_passed

    def reserve_steps(self) -> int:
        """Steps held for the finish: one for the checkout, a second while an order step still needs a message."""
        talks = any(p["messages"] > 0 for p in self.process_steps if p["when"] == "order")
        if talks and self.messages_sent < self.message_cap and self.steps_due("order"):
            return 2
        return 1

    def converse(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.model_calls >= CALL_CAP:
                break
            self.model_calls += 1
            reply, status_text = self._ra_with_retries(model, messages, max_tokens, tools)
            if reply is not None:
                return reply
            if self.key_spent(status_text):
                self.out_of_credit = True
                break
        return None

    def key_spent(self, status_text: Any) -> bool:
        """B1: the run key's limit is used up (403) or the funding account is empty (402 payment); a 402 for in-flight
        reservations clears as other episodes' calls finish, so it does not end model work."""
        body = str(getattr(self, "fail_body", "") or "").lower()
        if status_text == 403:
            return "limit" in body
        if status_text != 402:
            return False
        return not ("in_flight" in body or "in-flight" in body or "in flight" in body)

    def _ra_with_retries(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """Send one model call, resending it after a pause while the failure is transient and waiting time is left."""
        attempt = 0
        while True:
            reply, status_text = self.model_call(model, messages, max_tokens, tools)
            if reply is not None or not _ra_transient(status_text) or self.key_spent(status_text):
                return reply, status_text
            # a timeout has already used the client's 50 s, so it gets one resend at most
            allowed = 1 if status_text in (None, 408) else len(RETRY_PAUSES)
            if attempt >= allowed or self.paused + RETRY_PAUSES[attempt] > RETRY_PAUSE_CAP:
                return None, status_text
            wait = RETRY_PAUSES[attempt]
            pause = wait * (0.8 + 0.4 * random.random())
            self.paused += wait
            attempt += 1
            time.sleep(pause)

    def model_call(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """Run one inference request; give back the reply message and 200, or None with the failure status (0: empty reply)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = _ra_cache_marked(messages)
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body.update(tools=tools, tool_choice="auto")
        result = _model_link.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            return self._ra_read_choice(model, data)
        error = result.error or {}
        status_text, detail = error.get("status"), error.get("body")
        inner = data.get("error") if isinstance(data, dict) else None
        if isinstance(inner, dict):
            # the provider's failure came back inside a 200 reply, with its own code
            status_text, detail = _as_int(inner.get("code")), inner.get("message")
        self.fail_body = str(detail or "")[:400]
        self.log.append({"kind": "llm_fail", "model": model, "status": status_text, "body": str(detail or "")[:200]})
        return None, status_text

    def _ra_read_choice(self, model: str, data: dict[str, Any]) -> tuple[dict[str, Any] | None, int]:
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.spend_usd += float(spent)
        self.finish_why = data["choices"][0].get("finish_reason")
        reply = data["choices"][0].get("message") or {}
        if not (reply.get("content") or reply.get("tool_calls")):
            self.log.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        return reply, 200

    def json_from_model(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        for model in models:
            self.finish_why = None
            reply = self.converse([{"role": "user", "content": prompt}], [model], max_tokens)
            parsed = _parse_object(reply.get("content") if reply else None)
            if parsed is None and self.finish_why == "length" and max_tokens < CUT_RETRY_TOKENS and not (self.aborted or self.out_of_credit):
                # B3: the reasoning used the whole limit; the same model answers again with room to finish
                self.log.append({"kind": "cut_retry", "model": model, "tokens": max_tokens})
                reply = self.converse([{"role": "user", "content": prompt}], [model], CUT_RETRY_TOKENS)
                parsed = _parse_object(reply.get("content") if reply else None)
            if parsed is not None:
                return parsed
            if self.aborted or self.out_of_credit or self.model_calls >= CALL_CAP:
                break
        return None

    def store_call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = {**self.env_binding, "step": self.steps_used + 1, "action": {"name": name, "args": args}}
        for _attempt in range(4):
            # the step number follows any correction the store asked for on the previous attempt
            payload["step"] = self.steps_used + 1
            result = _store_link.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                return self._ra_settle(name, args, result.data)
            if not self._ra_try_again(result):
                return None
        return None

    def _ra_settle(self, name: str, args: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
        reported = data.get("step")
        self.steps_used = reported if isinstance(reported, int) else self.steps_used + 1
        if data.get("finished"):
            self.is_over = True
        seen = data.get("observation")
        failed = isinstance(seen, dict) and seen.get("error")
        if name in self.page_viewers() and not failed:
            # remember the listings whose whole page was opened (a search row does not count)
            for value in args.values():
                for entry in value if isinstance(value, list) else [value]:
                    if isinstance(entry, str):
                        self.opened.add(entry.strip())
        return data

    def _ra_try_again(self, result: Any) -> bool:
        """Log a failed store call; True when it is worth sending again."""
        error = result.error or {}
        status_text = error.get("status")
        body = str(error.get("body") or "")
        self.log.append({"kind": "env_fail", "status": status_text, "body": body[:200]})
        if status_text == 409:
            return self._ra_step_clash(body)
        if status_text in _RA_REJECTED:
            if status_text == 402:
                self.aborted = True
            return False
        return True

    def _ra_step_clash(self, body: str) -> bool:
        """A 409: the episode is over (False), or the step number needs correcting before a resend (True)."""
        low = body.lower()
        if any(word in low for word in _RA_CLOSED_WORDS):
            self.is_over = True
            return False
        expected = _RA_EXPECTED_STEP.search(body)
        if expected:
            self.steps_used = int(expected.group(1)) - 1
            return True
        if "reuse" in low or "different action" in low:
            self.steps_used += 1
            return True
        return False

    # ---------- reading the request ----------

    @staticmethod
    def _rb_number(value: Any) -> int | float | None:
        """A plain number (a numeric string becomes a float); None for anything else, booleans included."""
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return value if Buyer._rb_is_num(value) else None

    @staticmethod
    def _rb_is_num(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @staticmethod
    def _rb_plain_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool)

    def _rb_brief_prompt(self) -> str:
        catalogue = []
        for tool in self.tools:
            spec = tool["function"]
            catalogue.append("- {}: {} | parameters: {}".format(
                spec["name"], spec.get("description", ""), _to_json(spec.get("parameters") or {})))
        pieces = [BRIEF_PROMPT, "\nTOOLS:\n", "\n".join(catalogue), "\n\nREQUEST:\n", self.request_text]
        opening = self.policy.get("opening")
        if opening is not None:
            pieces.append("\n\nOPENING NOTE FROM THE STORE:\n")
            pieces.append(_to_json(_trim_value(opening, 1500)))
        return "".join(pieces)

    @staticmethod
    def _rb_brief_thin(plan_brief: dict[str, Any]) -> bool:
        """The reading names no item, or no tool for the purchase."""
        if not _as_list(plan_brief.get("items")):
            return True
        jobs = plan_brief.get("roles")
        return not isinstance(jobs, dict) or not jobs.get("buy")

    def _rb_split_modes(self, plan_brief: dict[str, Any]) -> None:
        """One tool that both buys and ends without buying needs the arguments telling the two apart."""
        buy = self.jobs.get("buy")
        if not buy or self.jobs.get("refuse") != buy:
            return
        on, off = plan_brief["buy_args"], plan_brief["refuse_args"]
        if on and off and on != off:
            return
        question_text = ("The store tool " + buy + " both checks out and ends without a purchase, depending on its arguments. Its schema: "
                    + _to_json(self.tool_schemas.get(buy))
                    + "\nReturn ONE JSON object and nothing else: {\"buy_args\": the fixed arguments that check out, "
                    "\"refuse_args\": the fixed arguments that end without a purchase} (only the arguments that choose the mode).")
        answer = self.json_from_model(question_text, [WALKER_MODEL, JUDGE_MODEL], 600) or {}
        on, off = answer.get("buy_args"), answer.get("refuse_args")
        if not isinstance(on, dict) or not isinstance(off, dict) or on == off:
            return
        plan_brief["buy_args"] = answer["buy_args"]
        plan_brief["refuse_args"] = answer["refuse_args"]

    @staticmethod
    def _rb_item_from(entry: dict[str, Any]) -> Wanted:
        count_wanted = _as_int(entry.get("quantity")) or 0
        if count_wanted <= 0:
            count_wanted = 1
        firm_needs = []
        for must in _as_list(entry.get("musts")):
            if must:
                firm_needs.append(str(must))
        return Wanted(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""), count_wanted, firm_needs)

    def _rb_owner(self, text: str) -> int | None:
        """The number of the first item holding this must, or None."""
        for number, item in enumerate(self.items, 1):
            if text in item.firm_needs:
                return number
        return None

    def _rb_loose_must(self, text: str) -> str:
        """A lookup copied loosely stands for the must on the same property; unchanged when there is none."""
        prop = _plain_words(text.split(":")[0])
        for item in self.items:
            for must in item.firm_needs:
                if _plain_words(must.split(":")[0]) == prop:
                    return must
        return text

    def _rb_take_lookups(self, plan_brief: dict[str, Any]) -> None:
        for lookup in _as_list(plan_brief.get("lookups")):
            text = str(lookup).strip()
            owner = self._rb_owner(text)
            if owner is None:
                text = self._rb_loose_must(text)
                owner = self._rb_owner(text) or 1
            if not text:
                continue
            pair = (owner, text)
            if pair not in self.ref_needs:
                self.ref_needs.append(pair)

    def _rb_take_limits(self, plan_brief: dict[str, Any]) -> None:
        budget = self._rb_number(plan_brief.get("budget"))
        if budget is not None and budget > 0:
            self.budget = budget
        messages = _as_int(plan_brief.get("messages"))
        if messages is None or messages < 0:
            if self.jobs.get("ask"):
                self.message_cap = 3
        else:
            self.message_cap = messages
        steps = _as_int(plan_brief.get("steps"))
        if steps is None:
            return
        if 10 <= steps < self.step_cap:
            self.step_cap = steps

    def _rb_take_process(self, plan_brief: dict[str, Any]) -> None:
        # V5: without approval wording in the request, a step is said in the ending call's own text, never in a message
        approval = bool(_V5_APPROVAL.search(self.request_text or ""))
        for entry in _as_list(plan_brief.get("process")):
            if not isinstance(entry, dict):
                continue
            steps_used = str(entry.get("step") or "").strip()
            if not steps_used:
                continue
            moment = str(entry.get("when") or "").strip().lower()
            if moment not in ("choose", "order", "end"):
                moment = "order"
            if not approval and self.request_text:
                rules = [str(r) for r in _as_list(plan_brief.get("rules")) if r]
                rules.append(f"the request's '{steps_used}' is said in the text of the checkout or ending call itself, not in a message to the shopper"
                             + ("; choose the listing yourself by the pick rule (the shopper does not pick)" if moment == "choose" else ""))
                plan_brief["rules"] = rules
                self.log.append({"kind": "step_in_text", "step": steps_used[:120], "when": moment})
                continue
            self.process_steps.append({"step": steps_used, "when": moment, "messages": max(0, _as_int(entry.get("messages")) or 0)})
        if not self.jobs.get("ask"):
            return
        # keep back the messages the order and ending steps take, at least one for each of them that needs any
        checkout_holds = []
        for entry in self.process_steps:
            if entry["when"] in ("order", "end") and entry["messages"]:
                checkout_holds.append(entry["messages"])
        self.step_reserve = min(max(checkout_holds), self.message_cap) if checkout_holds else 0

    def _v1_soften(self, plan_brief: dict[str, Any]) -> None:
        """V1: a must only the story suggests becomes a tie-break wish for the checks and the decline audit."""
        if not self.request_text:
            return
        wishes = [str(w) for w in _as_list(plan_brief.get("wishes")) if w]
        moved = []
        for item in self.items:
            kept = [need for need in item.firm_needs if _v2_stated(need, self.request_text, item.name)]
            for need in item.firm_needs:
                if need not in kept:
                    wishes.append(f"{item.name}: {need}")
                    moved.append(need)
            item.brief_needs = list(item.firm_needs)
            item.firm_needs = kept
        if moved:
            plan_brief["wishes"] = wishes
            self.log.append({"kind": "soft_musts", "moved": moved})

    def make_brief(self) -> None:
        prompt = self._rb_brief_prompt()
        # the light model reads first; the heavy one gets the request only when that reading lacks items or a buy tool
        # (4000 tokens leaves the light model room to think: a cut-off reply once pushed a tenth of race 6 to the heavy one)
        plan_brief = self.json_from_model(prompt, [WALKER_MODEL], 4000) or {}
        if not self.out_of_credit and self._rb_brief_thin(plan_brief):
            plan_brief = self.json_from_model(prompt, [JUDGE_MODEL, WALKER_MODEL], 2500) or plan_brief
        self.plan_brief = plan_brief
        named = plan_brief.get("roles")
        self.jobs = {}
        if isinstance(named, dict):
            for role, tool in named.items():
                if isinstance(tool, str) and tool in self.tool_schemas:
                    self.jobs[role] = tool
        for field in ("buy_args", "refuse_args"):
            if not isinstance(plan_brief.get(field), dict):
                plan_brief[field] = {}
        self._rb_split_modes(plan_brief)
        for entry in _as_list(plan_brief.get("items")):
            if isinstance(entry, dict):
                self.items.append(self._rb_item_from(entry))
        if not self.items:
            self.items = [Wanted("the requested product", "", 1, [])]
        self._v1_soften(plan_brief)
        self._rb_take_lookups(plan_brief)
        self.split_refs()
        self._rb_take_limits(plan_brief)
        self._rb_take_process(plan_brief)

    def _rb_unmix_one(self, number: int, item: Wanted) -> None:
        pointers = []
        for n, need in self.ref_needs:
            if n == number and need in item.firm_needs:
                pointers.append(need)
        rest = [m for m in item.firm_needs if m not in pointers]
        if not (pointers and rest):
            return
        lines = [REF_SPLIT_PROMPT, "", "REQUEST:", self.request_text, "", f"PRODUCT: {item.name}", "REQUIREMENTS POINTING AT ANOTHER LISTING:"]
        lines.extend(pointers)
        lines.append("OTHER REQUIREMENTS:")
        lines.extend(rest)
        reply = self.json_from_model("\n".join(lines), [JUDGE_MODEL, WALKER_MODEL], 500) or {}
        other = {str(x).strip() for x in _as_list(reply.get("describe_other"))}
        moved = [m for m in rest if m in other]
        if not moved:
            return
        anchor = pointers[0]
        merged = anchor + " (that listing: " + "; ".join(moved) + ")"
        kept = []
        for must in item.firm_needs:
            if must in moved:
                continue
            kept.append(merged if must == anchor else must)
        item.firm_needs = kept
        ref_needs = []
        for n, need in self.ref_needs:
            ref_needs.append((n, merged if n == number and need == anchor else need))
        self.ref_needs = ref_needs
        self.log.append({"kind": "unmix", "item": number, "moved": moved})

    def split_refs(self) -> None:
        """Move musts that only describe the listing a lookup points at (misread as musts of the product) into that lookup."""
        for number, item in enumerate(self.items, 1):
            self._rb_unmix_one(number, item)

    def _rb_need_text(self, need: str) -> str:
        if need not in self.ref_text:
            return need
        return f"{need} (the page it points at gives: {self.ref_text[need]})"

    def _rb_item_line(self, number: int, item: Wanted, many: bool) -> str:
        needs = [self._rb_need_text(m) for m in item.firm_needs + item.told if m not in item.given_up]
        line = f"item {number}: {item.name}" if many else f"item: {item.name}"
        if item.shelf_name:
            line += f" (store label: {item.shelf_name})"
        line += f", quantity {item.count_wanted}; requirements: "
        line += "; ".join(needs) if needs else "none beyond the kind of product"
        if item.open_props:
            line += "; shopper does not mind: " + ", ".join(item.open_props)
        return line

    @staticmethod
    def _rb_pick_line(plan_brief: dict[str, Any]) -> str:
        pick = plan_brief.get("pick")
        if pick == "cheapest":
            return "pick: the cheapest listing that fits (lowest price)"
        if pick not in ("most", "least"):
            return "pick: any listing that fits everything"
        return f"pick: the {pick} {plan_brief.get('pick_measure') or ''} among fitting listings".replace("  ", " ")

    def _rb_source_line(self) -> str | None:
        """Where listing attributes are read, and which views show a listing's page."""
        facts = self.attribute_viewer()
        if not facts:
            return None
        offer = self.jobs.get("offer")
        if self.shows_attrs.get(facts) is not True and not offer:
            return None
        views = self.page_viewers()
        bare = sorted(t for t in views | {offer or ""} if t and self.shows_attrs.get(t) is False)
        pages = sorted(t for t in views if t != facts)
        line = f"listing attributes come from {facts}"
        if bare:
            line += "; " + ", ".join(bare) + " show none"
        if pages:
            line += "; a listing's page (what a chain of pages or a page check means) is shown by " + " or ".join(pages)
        return line

    def ledger_text(self) -> str:
        many = len(self.items) > 1
        parts = [self._rb_item_line(n, item, many) for n, item in enumerate(self.items, 1)]
        plan_brief = self.plan_brief
        if self.budget is not None:
            scope = "in total"
            if plan_brief.get("budget_scope") == "each":
                scope = "for each item"
            parts.append(f"budget: at most {self.budget} {plan_brief.get('budget_unit') or ''} {scope}".replace("  ", " "))
        parts.append(self._rb_pick_line(plan_brief))
        if plan_brief.get("in_stock_only", True) is not False:
            parts.append("in stock only")
        parts.extend("wish (tie-breaker only): {}".format(w) for w in _as_list(plan_brief.get("wishes")))
        parts.extend("rule from the request: {}".format(r) for r in _as_list(plan_brief.get("rules")))
        if plan_brief.get("quiet_decline") is True:
            parts.append("if nothing fits: end without naming or suggesting any listing")
        moments = {"choose": "before choosing (one message showing the options is enough; the reply is the shopper's answer, and when it names no listing, choose by the pick rule yourself and check out; never wait for a pick or end without a purchase because none came)",
                   "order": "at the order (Kyo does it when you check out)",
                   "end": "at an ending without a purchase (Kyo does it then)"}
        for entry in self.process_steps:
            parts.append(f"step the request asks for, {moments[entry['when']]}: {entry['step']}")
        source = self._rb_source_line()
        if source is not None:
            parts.append(source)
        return "\n".join(f"- {p}" for p in parts)

    def _rb_pace_hint(self, steps_left: int) -> str:
        if steps_left <= self.reserve_steps() + 1:
            return (" FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when"
                    " your search showed that an item has no fitting listing (running out of steps does not show that).")
        idle = self.checkouts == 0
        if steps_left <= 6 or (idle and steps_left <= len(self.items) + 5):
            return " Few steps left: set the cart to the chosen lines and check out now."
        if not idle:
            return ""
        if steps_left <= 12:
            return " Steps are running low: settle on your picks, fill the cart and check out within the next few steps."
        if not self.cart_used and self.steps_used >= self.step_cap // 2:
            return (" Half the steps are used and the cart is empty: cart each item's fitting listing now"
                    " (the store may still sell one out).")
        return ""

    def status_text(self) -> str:
        steps_left = self.steps_left()
        out = [f"[Kyo] step {self.steps_used} of {self.step_cap} used, {steps_left} left.", self._rb_pace_hint(steps_left)]
        current = set(need for _, need in self.needs_in_force())
        open_refs = [need for _, need in self.ref_needs if need in current and need not in self.ref_text]
        if open_refs and not self.checkouts:
            out.append("\n[resolve first: open the listing each of these points at, and every listing its page refers to;"
                       " the last page gives the value] " + " | ".join(open_refs[:3]))
        if self.review_leads:
            out.append("\n[review candidates] " + ", ".join(self.review_leads))
        if self.note_texts:
            out.append("\n[changes so far] " + " | ".join(self.note_texts[-4:]))
        out.append("\n[ledger]\n")
        out.append(self.ledger_text())
        return "".join(out)

    # ---------- listings ----------
    def _rb_own_id(self, node: dict[str, Any]) -> str | None:
        key = _guess_id_key([node])
        if not key:
            return None
        value = node[key].strip()
        if "id" in str(key).lower() or value in self.rows:
            return value
        return None

    def listing_id(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        """The id of a listing record, by the id key learned from result rows; None when the node is not one."""
        if isinstance(node, dict) and len(node) >= 3:
            key = self.id_field
            if key and _looks_like_id(node.get(key), True):
                return node[key].strip()
            if in_list or depth == 0 or any(isinstance(v, dict) for v in node.values()):
                return self._rb_own_id(node)
        return None

    def note_rows(self, node: list[Any]) -> bool:
        rows = []
        for entry in node:
            if isinstance(entry, dict) and len(entry) >= 3:
                rows.append(entry)
        if not rows or 2 * len(rows) < len(node):
            return False
        key = self.id_field
        if key and all(_looks_like_id(r.get(key), True) for r in rows):
            return True
        learned = _guess_id_key(rows)
        if not learned:
            return False
        self.id_field = learned
        return True

    @staticmethod
    def _rb_flipped(old: dict[str, Any], new: dict[str, Any], before: bool, after: bool) -> bool:
        for key, value in new.items():
            if old.get(key) is before and value is after:
                return True
        return False

    def _rb_take_record(self, listing_id: str, node: dict[str, Any]) -> None:
        if listing_id not in self.rows:
            self.rows[listing_id] = {}
            self.seen_ids.append(listing_id)
        record = self.rows[listing_id]
        # a flag of the listing's own record going from true to false means it can no longer be bought
        if self._rb_flipped(record, node, True, False):
            self.unbuyable.add(listing_id)
        elif self._rb_flipped(record, node, False, True):
            self.unbuyable.discard(listing_id)
        record.update(node)
        if _attribute_block(node):
            self.opened_records[listing_id] = dict(record)
            return
        if listing_id in self.opened_records:
            self.opened_records[listing_id].update(node)

    def _rb_walk(self, node: Any, depth: int, in_list: bool = False) -> None:
        if depth > 6:
            return
        if isinstance(node, list):
            # M4: a list of "key: value | ..." strings is a list of rows
            node = [(_m4_text_record(child) or child) if isinstance(child, str) else child for child in node]
            rowish = self.note_rows(node)
            for child in node:
                self._rb_walk(child, depth + 1, rowish)
            return
        if isinstance(node, str):
            record = _m4_text_record(node)
            if record:
                self._rb_walk(record, depth, in_list)
            return
        if not isinstance(node, dict):
            return
        listing_id = self.listing_id(node, in_list, depth)
        if listing_id:
            self._rb_take_record(listing_id, node)
            return
        for child in node.values():
            self._rb_walk(child, depth + 1)

    def take_in(self, observation: Any) -> None:
        self._rb_walk(observation, 0)

    def shelf_labels(self) -> set[str]:
        """Every enum option in the tool schemas (lowercased), six levels deep."""
        options: set[str] = set()
        pending = [(schema, 0) for schema in self.tool_schemas.values()]
        while pending:
            schema, depth = pending.pop()
            if depth > 6 or not isinstance(schema, dict):
                continue
            choices = schema.get("enum") or []
            options.update(c.strip().lower() for c in choices if isinstance(c, str))
            pending.extend((v, depth + 1) for v in schema.values() if isinstance(v, dict))
        return options

    def shelf_field(self) -> str | None:
        tally: dict[str, list[Any]] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if not isinstance(value, str) or not value or len(value) > 40 or re.search(r"\d{3}", value):
                    continue
                entry = tally.setdefault(key, [set(), 0])
                entry[0].add(value)
                entry[1] += 1
        floor = 0.8 * max(1, len(self.rows))
        known = self.shelf_labels()
        ranked = []
        for key, (values, seen) in tally.items():
            if seen < floor or len(values) > 40:
                continue
            size = len(values)
            listed = len([v for v in values if v.strip().lower() in known]) / size
            mean = sum(map(len, values)) / size
            ranked.append(((-listed, size, -seen, -mean), key))
        if not ranked:
            return None
        return min(ranked, key=lambda pair: pair[0])[1]

    @staticmethod
    def _rb_long_stems(text: str) -> set[str]:
        return {w for w in _word_stems(text) if len(w) >= 4}

    def _rb_wanted(self) -> list[tuple[str, str, set[str]]]:
        wanted = []
        for item in self.items:
            shelf_name = item.shelf_name or ""
            wanted.append((_plain_words(shelf_name), _plain_words(item.name), self._rb_long_stems(shelf_name) | self._rb_long_stems(item.name)))
        return wanted

    def _rb_fits_wanted(self, label: str, wanted: list[tuple[str, str, set[str]]]) -> bool:
        plain = _plain_words(label)
        stems = self._rb_long_stems(label)
        return any(plain == shelf_name or plain == name or stems & long for shelf_name, name, long in wanted)

    def _rb_fact_groups(self, cat_key: str | None) -> dict[str, dict[str, Any]]:
        """Per category: how many listings with read attributes, and per attribute name its count, values and per-listing value."""
        groups: dict[str, dict[str, Any]] = {}
        for listing_id in self.seen_ids:
            record = self.opened_records.get(listing_id)
            if not record:
                continue
            label = "listings"
            if cat_key:
                label = str(record.get(cat_key) or "listings")
            if label not in groups:
                groups[label] = {"n": 0, "keys": {}}
            groups[label]["n"] += 1
            slots = groups[label]["keys"]
            names: set[str] = set()
            for key, value in (_attribute_block(record) or {}).items():
                name = _plain_words(key)
                if name and name not in names:
                    names.add(name)
                    self._rb_note_value(slots, name, listing_id, str(value).strip().lower())
        return groups

    @staticmethod
    def _rb_note_value(slots: dict[str, dict[str, Any]], name: str, listing_id: str, text: str) -> None:
        if name not in slots:
            slots[name] = {"n": 0, "values": [], "per": {}}
        slot = slots[name]
        slot["n"] += 1
        slot["per"][listing_id] = text
        if text not in slot["values"]:
            slot["values"].append(text)

    @staticmethod
    def _rb_readable(values: list[str]) -> bool:
        """Short values, not mostly ' / ' lists, and not a set of long all-distinct texts."""
        size = len(values)
        mean = sum(len(v) for v in values) / size
        if mean > 40:
            return False
        if 2 * len([v for v in values if " / " in v]) >= size:
            return False
        if not (size >= 4 and mean > 12 and len(set(values)) == size):
            return True
        # P3: distinct texts that share a leading word (a maker: "qualcomm snapdragon ...") are a component's values
        heads = [v.split()[0] for v in values if v.split()]
        return len(set(heads)) < len(heads)

    def _rb_kept_keys(self, keys: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        kept: list[tuple[str, dict[str, Any]]] = []
        for name, slot in sorted(keys.items(), key=lambda kv: kv[1]["n"], reverse=True):
            values = list(slot["per"].values())
            if not values or not self._rb_readable(values):
                continue
            # a longer name carrying the very same values as one already kept adds nothing
            if any(name.startswith(other + " ") and slot["per"] == spec["per"] for other, spec in kept):
                continue
            kept.append((name, slot))
        return kept

    def attribute_menu(self) -> list[tuple[str, str, int, list[str]]]:
        """Rows of (category, attribute name, listings carrying it, up to four sample values), most carried first."""
        groups = self._rb_fact_groups(self.shelf_field())
        wanted = self._rb_wanted()
        hits = [label for label in groups if self._rb_fits_wanted(label, wanted)]
        if hits:
            groups = {label: groups[label] for label in groups if label in hits}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: kv[1]["n"], reverse=True):
            for name, slot in self._rb_kept_keys(group["keys"])[:14]:
                example_values = [v[:40] for v in slot["values"][:4]]
                table.append((label, name, slot["n"], example_values))
        return table

    def rows_by_item(self) -> list[list[str]]:
        """Listing ids by category, the categories matching a requested item kept when any do; largest first."""
        cat_key = self.shelf_field()
        groups: dict[str, list[str]] = {}
        for listing_id in self.seen_ids:
            label = ""
            if cat_key:
                label = str(self.rows[listing_id].get(cat_key) or "")
            if label not in groups:
                groups[label] = []
            groups[label].append(listing_id)
        wanted = self._rb_wanted()
        pool = [ids for label, ids in groups.items() if self._rb_fits_wanted(label, wanted)]
        if not pool:
            pool = list(groups.values())
        pool.sort(key=len, reverse=True)
        return pool[:max(1, len(self.items))]

    def id_param(self, tool: str | None) -> str | None:
        """The one free-text argument a tool takes (its only required one), or None."""
        if not tool:
            return None
        schema = self.tool_schemas.get(tool) or {}
        texts = []
        for key, spec in (schema.get("properties") or {}).items():
            if isinstance(spec, dict) and "string" in _schema_types(spec) and "enum" not in spec:
                texts.append(key)
        required = schema.get("required") or []
        target = next(filter(lambda k: k in texts, required), None)
        if not target:
            target = texts[0] if len(texts) == 1 else None
        if not target:
            return None
        for key in required:
            if key != target:
                return None
        return target

    def price_for(self, listing_id: str) -> float | None:
        """The price the listing's newest record shows, under the price key named for result rows."""
        if not self.price_field:
            return None
        value = (self.rows.get(listing_id) or {}).get(self.price_field)
        if isinstance(value, dict):
            value = next(filter(self._rb_is_num, value.values()), None)
        number = self._rb_number(value)
        return None if number is None else float(number)

    def can_buy(self, listing_id: str) -> bool:
        """Not sold out by any note or refusal, and its stock flag (under the named key) does not say no."""
        if self.is_gone(listing_id):
            return False
        if not self.stock_field:
            return True
        flag = (self.rows.get(listing_id) or {}).get(self.stock_field)
        if flag is False:
            return False
        if not self._rb_is_num(flag):
            return True
        return not flag <= 0

    def _rb_rows_of(self, seq: list[Any], out: list[tuple[str, dict[str, Any]]]) -> None:
        for row in seq:
            if not isinstance(row, dict):
                continue
            listing_id = self.listing_id(row, True)
            if listing_id:
                out.append((listing_id, row))

    def listing_records(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """Listing records in a result: the result itself when it is one, else the rows of its lists, two levels down."""
        found: list[tuple[str, dict[str, Any]]] = []
        if isinstance(observation, list):
            self._rb_rows_of(observation, found)
            return found
        if not isinstance(observation, dict):
            return found
        listing_id = self.listing_id(observation)
        if listing_id:
            return [(listing_id, observation)]
        for value in observation.values():
            if isinstance(value, list):
                self._rb_rows_of(value, found)
                continue
            if isinstance(value, dict):
                for inner in value.values():
                    if isinstance(inner, list):
                        self._rb_rows_of(inner, found)
        return found

    def mark_viewed(self, name: str, observation: Any) -> None:
        """Record the fields each view tool shows and the fields each listing has shown since the last change, so a
        view that would show nothing new is not run again."""
        records = self.listing_records(observation)
        if not records:
            return
        by_tool = self.tool_fields.setdefault(name, set())
        for listing_id, record in records:
            fields = set(record)
            by_tool.update(fields)
            stamp, have = self.view_fields.get(listing_id, (-1, set()))
            if stamp == self.epoch:
                fields = have | fields
            self.view_fields[listing_id] = (self.epoch, fields)

    def already_shown_all(self, name: str, listing_id: str) -> bool:
        fields = self.tool_fields.get(name)
        if not fields or listing_id in self.sold_at:
            return False
        stamp, have = self.view_fields.get(listing_id, (-1, set()))
        return stamp == self.epoch and fields <= have

    # ---------- result pages ----------
    @staticmethod
    def _rb_key_in(key: Any, where: dict[str, Any]) -> str | None:
        if isinstance(key, str) and key in where:
            return key
        return None

    def _rb_shape_prompt(self, name: str, observation: dict[str, Any], lists: list[str]) -> str:
        sample = {}
        for key, value in observation.items():
            sample[key] = value[:3] if key in lists else value
        schema = _to_json(self.tool_schemas.get(name) or {})
        body = _to_json(_trim_value(sample, 120))[:4000]
        return f"{RESULT_SHAPE_PROMPT}\n\nTOOL: {name} | parameters: {schema}\n\nRESULT (first rows only):\n{body}"

    def learn_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        """The keys of a tool's row-list result for the rows, the match count, the next-page value (with the parameter
        that takes it), a row's price and its can-be-bought flag: the light model names them once per tool, checked here."""
        if name in self.result_shapes:
            return self.result_shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = []
        for key, value in observation.items():
            if isinstance(value, list) and len(value) >= 2 and all(isinstance(r, dict) for r in value):
                lists.append(key)
        if not lists:
            return None
        self.result_shapes[name] = None
        got = self.json_from_model(self._rb_shape_prompt(name, observation, lists), [WALKER_MODEL, JUDGE_MODEL], 1500) or {}
        rows = got.get("rows")
        if rows not in lists or not rows:
            return None
        first = observation[rows][0]
        props = (self.tool_schemas.get(name) or {}).get("properties") or {}
        count = self._rb_key_in(got.get("count"), observation)
        if count and not self._rb_plain_int(observation[count]):
            count = None
        shape = {"rows": rows, "count": count,
                 "next": self._rb_key_in(got.get("next"), observation),
                 "next_arg": self._rb_key_in(got.get("next_arg"), props),
                 "price": self._rb_key_in(got.get("price"), first),
                 "stock": self._rb_key_in(got.get("stock"), first)}
        self.result_shapes[name] = shape
        if not self.price_field:
            self.price_field = shape["price"]
        if not self.stock_field:
            self.stock_field = shape["stock"]
        self.log.append({"kind": "shape", "tool": name, "shape": shape})
        return shape

    def page_spec(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matches reported, rows shown, the next-page parameter, its value or None on the last page), read by the
        tool's named result shape; None when the result lacks those parts."""
        shape = self.learn_shape(name, observation)
        if not shape or not all((shape["count"], shape["next"], shape["next_arg"])) or not isinstance(observation, dict):
            return None
        rows = observation.get(shape["rows"])
        if not isinstance(rows, list) or not rows:
            return None
        count = observation.get(shape["count"])
        if not self._rb_plain_int(count):
            return None
        token = observation.get(shape["next"])
        if token is None or (isinstance(token, (str, int)) and not isinstance(token, bool)):
            return count, len(rows), shape["next_arg"], token
        return None

    def sorted_by_price(self, observation: Any) -> bool:
        prices = [self.price_for(listing_id) for listing_id, _ in self.listing_records(observation)]
        if len(prices) < 3 or None in prices:
            return False
        for low, high in zip(prices, prices[1:]):
            if not low <= high:
                return False
        return True

    @staticmethod
    def _rb_paging_itself(given: Any) -> bool:
        """The agent already asked for a later page."""
        if isinstance(given, int) and not isinstance(given, bool):
            return given not in (None, 0, 1)
        return bool(str(given or "").strip())

    def _rb_read_pages(self, name: str, args: dict[str, Any], param: str, value: Any, pages: int) -> list[str]:
        texts: list[str] = []
        while len(texts) < pages and value is not None:
            faults: list[str] = []
            call = _conform({**args, param: value}, self.tool_schemas.get(name), "args", faults)
            if faults:
                break
            reply = self.store_call(name, call)
            if reply is None or self.is_over:
                break
            self.pages_done += 1
            page = reply.get("observation")
            self.take_reply(name, page)
            self.extra_pages.append(page)
            shown = self.render_reply(page)
            self.read_memo[name + _to_json(call)] = (self.epoch, self.steps_used, shown)
            texts.append(f"{param}={_to_json(value)}: {shown}")
            nxt = self.page_spec(name, page)
            value = nxt[3] if nxt else None
        return texts

    def read_more_pages(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search under a cheapest, most or least pick that matched more listings than it showed: Kyo reads a short
        remainder itself (one step a page, no model turn), and names a long one so the agent can narrow the search."""
        self.extra_pages = []
        pick = self.plan_brief.get("pick")
        if pick not in ("cheapest", "most", "least"):
            return ""
        plan = self.page_spec(name, observation)
        if not plan:
            return ""
        count, shown, param, value = plan
        # nothing to do on the last page, or while the agent pages on its own
        if value is None or count <= shown or self._rb_paging_itself(args.get(param)):
            return ""
        if pick == "cheapest" and self.sorted_by_price(observation):
            return ""
        pages = (count - 1) // shown
        if pages > 2 and self.pages_done + 2 <= EXTRA_PAGES and self.steps_left() - self.reserve_steps() - 2 >= 12:
            pages = 2   # W3: read two more pages of a long result, then say how much of it is still unseen
        if pages > 2 or self.pages_done + pages > EXTRA_PAGES or self.steps_left() - self.reserve_steps() - pages < 12:
            return ("\n[Kyo] This search matched " + str(count) + " listings and showed " + str(shown)
                    + ", in no price order. The pick counts every fitting listing: search with narrower words"
                    " (the requirement's value) or page on with " + f"{param}={_to_json(value)}.")
        texts = self._rb_read_pages(name, args, param, value, pages)
        self.log.append({"kind": "pages", "tool": name, "read": len(texts)})
        if not texts:
            return ""
        seen_rows = shown * (len(texts) + 1)
        unseen = ""
        if count > seen_rows:
            unseen = (f"\n[Kyo] {count} listings matched and {seen_rows} rows of them have been read; narrow the search with the"
                      " requirement's value or page on to see the rest.")
        return "\n[Kyo read the next pages of this search itself, one step a page]\n" + "\n".join(texts) + unseen

    def _w3_cap_ignored(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """W3: a search given a price ceiling whose rows still cost more than it: the store ignored that argument."""
        props = (self.tool_schemas.get(name) or {}).get("properties") or {}
        if not self.price_field:
            return ""

        def about_price(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search("price|cost|amount|budget|spend|money", key + " " + str(spec.get("description") or ""), re.I))
        ceilings = []
        for key, value in args.items():
            if key not in props or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if re.search("max|most|under|below|limit|cap|upto|up_to", key, re.I) and about_price(key):
                ceilings.append((value, key))
        if not ceilings:
            return ""
        ceiling, key = min(ceilings)
        dearer = [i for i, _ in self.listing_records(observation) if (self.price_for(i) or 0) > ceiling]
        if not dearer:
            return ""
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get("enum"), list)}
        other = {k: [o for o in props[k]["enum"] if o != v] for k, v in modes.items()}
        hint = ""
        if any(other.values()):
            hint = ("; list the category with " + ", ".join(f"{k}={_to_json(o[0])}" for k, o in other.items() if o)
                    + f" and the same {key} to see only cheaper listings")
        return f"\n[Kyo] This search did not apply {key}={ceiling:g}: {len(dearer)} of its rows cost more{hint}."

    def cheapest_note(self, observations: list[Any]) -> str:
        """For a cheapest pick: the buyable rows of a search result and the pages Kyo read on, cheapest first, since
        stores do not list them in price order."""
        if not self.price_field or self.plan_brief.get("pick") != "cheapest":
            return ""
        idents = set()
        for seen in observations:
            idents.update(listing_id for listing_id, _ in self.listing_records(seen))
        priced = []
        for listing_id in idents:
            if not self.can_buy(listing_id):
                continue
            price = self.price_for(listing_id)
            if price is not None:
                priced.append((price, listing_id))
        if len(priced) < 2:
            return ""
        priced.sort()
        listing = ", ".join("{} ({:g})".format(listing_id, price) for price, listing_id in priced[:6])
        return "\n[Kyo] buyable rows here, cheapest first: " + listing

    def hold_for_wider(self) -> str:
        """Once, before a cheapest checkout with steps to spare: keyword search reads titles, so a search for a
        requirement's value misses fitting listings whose titles use other words."""
        if self.wider_done:
            return ""
        if self.plan_brief.get("pick") != "cheapest" or self.steps_left() < 14 + 2 * len(self.items):
            return ""
        self.wider_done = True
        lines = []
        for listing_id in self.cart_listing_ids():
            price = self.price_for(listing_id)
            if price is not None:
                lines.append("{} at {:g}".format(listing_id, price))
        cart = " (" + "; ".join(lines) + ")" if lines else ""
        return ("Not run yet (no step used). " + str(self.steps_left()) + " steps are left and the pick is the cheapest."
                " Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the"
                " fitting listings, and rows come in no price order. Before checking out, run a few more searches for each"
                " item under other names for the product (its plain one-word name, synonyms, words from fitting titles you"
                " have seen) and look only at buyable rows priced below the cart's line" + cart
                + "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")


    # ---------- reading attributes before a judgement ----------
    def _rc_slot(self, raw: Any) -> int:
        """Item number named by a model answer, clamped into the item list (1 when missing)."""
        return min(max(1, _as_int(raw) or 1), len(self.items))

    def _rc_merge_facts(self, tool: str, seen_facts: bool) -> None:
        """Remember whether this view tool has ever shown an attribute block."""
        if not self.shows_attrs.get(tool, False):
            self.shows_attrs[tool] = seen_facts

    def _rc_open_one(self, tool: str, target: str, listing_id: str) -> str | None:
        """One attributes call for `ident`; the text for the judge, or None when no call went through."""
        problems: list[str] = []
        call_args = _conform({target: listing_id}, self.tool_schemas.get(tool), "args", problems)
        if problems:
            return None
        result = self.store_call(tool, call_args)
        if result is None:
            return None
        policy = result.get("observation")
        self.take_reply(tool, policy)
        self.mark_viewed(tool, policy)
        if isinstance(policy, dict) and self.listing_id(policy):
            self._rc_merge_facts(tool, bool(_attribute_block(policy)))
        return f"{listing_id}: {self.render_reply(policy)[:1500]}"

    def read_attributes(self, ids: list[str], keep: int) -> list[str]:
        """Open listings that are about to be judged but were never shown in full, leaving `keep` steps spare."""
        tool = self.attribute_viewer()
        target = self.id_param(tool)
        if not (tool and target):
            return []
        unread: list[str] = []
        for listing_id in ids:
            if listing_id not in self.rows or listing_id in self.opened_records or self.is_gone(listing_id):
                continue
            unread.append(listing_id)
        gathered: list[str] = []
        for listing_id in unread:
            out_of_room = self.steps_left() <= keep or self.is_over
            if out_of_room or self.shows_attrs.get(tool) is False:
                break
            text = self._rc_open_one(tool, target, listing_id)
            if text is None:
                break
            gathered.append(text)
        if gathered:
            self.log.append({"kind": "fill_facts", "n": len(gathered)})
        return gathered

    def _rc_kind(self, listing_id: str, key: str) -> str:
        row = self.rows.get(listing_id) or {}
        return str(row.get(key) or "").strip().lower()

    def _rc_below(self, kind: str, ceiling: float, cart: list[str], key: str) -> list[str]:
        """Seen, buyable, not yet opened rows of `kind` that cost less than `ceiling`, cheapest first."""
        found: list[str] = []
        for listing_id in self.seen_ids:
            if listing_id in cart or listing_id in self.opened_records or not self.can_buy(listing_id):
                continue
            if self._rc_kind(listing_id, key) != kind:
                continue
            spend_usd = self.price_for(listing_id)
            if spend_usd is None or not spend_usd < ceiling:
                continue
            found.append(listing_id)
        return sorted(found, key=lambda listing_id: self.price_for(listing_id) or 0.0)

    def read_cheaper(self) -> list[str]:
        """With a cheapest pick, open (cheapest first) a few buyable rows of a cart line's own kind that cost
        less than that line and were never read, so the final check can compare them with the cart."""
        if self.plan_brief.get("pick") != "cheapest":
            return []
        mark = self.cart_stamp() + f"#{self.epoch}"
        key = self.shelf_field()
        if not key or mark in self.cheaper_seen:
            return []
        self.cheaper_seen.add(mark)
        cart = self.cart_listing_ids()
        chosen: list[str] = []
        for line in cart:
            ceiling = self.price_for(line)
            kind = self._rc_kind(line, key)
            if not kind or ceiling is None:
                continue
            fresh = [listing_id for listing_id in self._rc_below(kind, ceiling, cart, key)[:2] if listing_id not in chosen]
            chosen.extend(fresh)
        return self.read_attributes(chosen[:CHEAPER_READS], self.reserve_steps() + 4)

    def take_reply(self, name: str, observation: Any) -> str:
        """Take in the result of one of Kyo's own calls and read any loose note in it; returns what changed."""
        self.take_in(observation)
        lines = self.note_lines(observation)
        if not lines:
            return ""
        meaning = self.apply_note(lines)
        if meaning:
            self.pending_notes.append("(seen on %s) %s" % (name, meaning))
        return meaning

    # ---------- opening more listings ----------
    def _rc_array_slot(self, props: dict[str, Any]) -> str | None:
        for key, spec in props.items():
            if isinstance(spec, dict) and "array" in _schema_types(spec):
                return key
        return None

    def _rc_spread(self, cap: int) -> list[str]:
        """Unopened ids shared out over the item groups, at most `cap` in all."""
        groups = self.rows_by_item()
        each = max(1, cap // max(1, len(groups)))
        spread: list[str] = []
        for group in groups:
            spread.extend([i for i in group if i not in self.opened_records][:each])
        return spread[:cap]

    def _rc_compare_call(self, tool: str, schema: dict[str, Any], target: str, ids: list[str]) -> dict[str, Any] | None:
        problems: list[str] = []
        call_args = _conform({target: ids}, schema, "args", problems)
        if problems:
            return None
        return self.store_call(tool, call_args)

    def open_more(self) -> None:
        """One side-by-side view of more listings, so the attribute key counts rest on more than a few of them."""
        tool = self.jobs.get("compare")
        schema = self.tool_schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = self._rc_array_slot(props)
        usable = bool(tool) and bool(target) and all(k == target for k in schema.get("required") or [])
        # with no attributes in the side-by-side view the planner would see no keys: open listings one at a time
        if not usable or self.side_view_bare():
            self.open_more_singly()
            return
        if self.steps_left() <= 12:
            return
        limit = props[target].get("maxItems")
        if not isinstance(limit, int):
            limit = 4
        ids = self._rc_spread(limit)
        if len(ids) < 2:
            return
        opened_before = len(self.opened_records)
        result = self._rc_compare_call(tool, schema, target, ids)
        if result is None:
            return
        policy = result.get("observation")
        self.take_reply(tool, policy)
        self.mark_viewed(tool, policy)
        records = self.listing_records(policy)
        if records:
            self._rc_merge_facts(tool, any(_attribute_block(rec) for _, rec in records))
        self.log.append(dict(kind="widen", ids=ids, opened=len(self.opened_records) - opened_before))
        if records and self.shows_attrs.get(tool) is False:
            # attributes are missing from this view: fall back to single listings
            self.open_more_singly()
            return
        nothing_new = len(self.opened_records) == opened_before
        if not nothing_new or len(ids) <= 2 or self.steps_left() <= 12:
            return
        retry = self._rc_compare_call(tool, schema, target, ids[:2])
        if retry is not None:
            self.take_reply(tool, retry.get("observation"))

    def open_more_singly(self) -> None:
        """No side-by-side view with attributes: open a few more listings one by one."""
        tool = self.attribute_viewer()
        target = self.id_param(tool)
        if not (tool and target):
            return
        by_price = self.plan_brief.get("pick") == "cheapest"

        def spend_usd(listing_id: str) -> float:
            price = self.price_for(listing_id)
            return float("inf") if price is None else price

        # W2: up to SURVEY_PER_ITEM per item group; rows naming more requirement words lead (cheapest first for a cheapest pick)
        words = set()
        for _, need in self.needs_in_force():
            words.update(re.findall("[a-z0-9]{3,}", need.split(":", 1)[-1].lower()))
        per_item = SURVEY_PER_ITEM if len(self.items) <= 2 else 4

        def named(listing_id: str) -> int:
            return len(words & set(re.findall("[a-z0-9]{3,}", _to_json(self.rows.get(listing_id) or {}).lower())))
        lanes: list[list[str]] = []
        heads: list[int] = []
        for group in self.rows_by_item():
            opened = sum(1 for i in group if i in self.opened_records)
            candidates = [i for i in group if i not in self.opened_records and self.can_buy(i)]
            if by_price:
                candidates.sort(key=spend_usd)
            candidates.sort(key=named, reverse=True)
            lanes.append(candidates[:max(0, per_item - opened)])
            heads.append(max(0, 4 - opened))
        first: list[str] = []
        for depth in range(4):
            for lane, head in zip(lanes, heads):
                if depth < min(head, len(lane)):
                    first.append(lane[depth])
        later: list[str] = []
        for depth in range(per_item):
            for lane, head in zip(lanes, heads):
                if head <= depth < len(lane):
                    later.append(lane[depth])
        cap = max(5, 2 * len(lanes) + 2)
        # the first reads keep 22 steps for the rest of the episode, the later ones 30
        read = self.read_attributes(first[:cap][:8], 22)
        self.read_attributes((first[cap:] + later)[:max(0, 10 - len(read))], 30)

    # ---------- questions to the shopper ----------
    def time_to_ask(self) -> bool:
        blocked = self.interviewed or not self.jobs.get("ask") or self.messages_left() <= 0
        if blocked or self.steps_left() <= 8:
            return False
        return len(self.opened_records) > 3 * len(self.items) or self.steps_used >= max(8, self.step_cap // 4)

    def _rc_text_slot(self, schema: dict[str, Any], props: dict[str, Any]) -> str | None:
        free_text: list[str] = []
        for key, spec in props.items():
            if isinstance(spec, dict) and "enum" not in spec and "string" in _schema_types(spec):
                free_text.append(key)
        wanted = next((key for key in schema.get("required") or [] if key in free_text), None)
        if wanted:
            return wanted
        return free_text[0] if free_text else None

    def ask_arguments(self, text: str) -> dict[str, Any] | None:
        schema = self.tool_schemas.get(self.jobs.get("ask") or "") or {}
        props = schema.get("properties") or {}
        slot = self._rc_text_slot(schema, props)
        if not slot:
            return None
        call: dict[str, Any] = {slot: text}
        for key in schema.get("required") or []:
            if key in call:
                continue
            choices = (props.get(key) or {}).get("enum")
            if not (isinstance(choices, list) and choices):
                return None
            call[key] = choices[0]
        problems: list[str] = []
        call = _conform(call, schema, "args", problems)
        if problems:
            return None
        return call

    def _rc_plan_prompt(self, count: int) -> str:
        """The question planner's prompt: request, items with what is settled, wanted questions, keys seen."""
        rows = [f"item {n}: {item.name} - settled by the request: " + ("; ".join(getattr(item, "brief_needs", item.firm_needs)) or "nothing")
                for n, item in enumerate(self.items, 1)]
        keys: list[str] = []
        current = None
        for position, (category, key, carried, values) in enumerate(self.attribute_menu(), 1):
            if category != current:
                current = category
                keys.append(f"category {category!r}:")
            keys.append("  {}. {} (on {}): {}".format(position, key, carried, " | ".join(values)))
        parts = [ASK_PLAN_PROMPT.replace("COUNT", str(count + 1)), "\n\nREQUEST:\n", self.request_text, "\n\nITEMS:\n", "\n".join(rows)]
        first = []
        for wanted in _as_list(self.plan_brief.get("ask_first")):
            if str(wanted).strip():
                first.append(str(wanted))
        if first:
            parts += ["\n\nASK FIRST (the request asks to be asked about these):\n", "\n".join(first)]
        parts += ["\n\nATTRIBUTE KEYS SEEN:\n", "\n".join(keys) or "(no full listing records yet)"]
        return "".join(parts)

    def _p2_fill(self, picks: list[tuple[int, str]], count: int) -> list[tuple[int, str]]:
        """P2: a single-item plan shorter than the messages takes the most carried open keys of the opened listings."""
        if len(picks) >= count or len(self.items) != 1:
            return picks
        settled = " ".join(getattr(self.items[0], "brief_needs", self.items[0].firm_needs)).lower()
        chosen = {_plain_words(label) for _, label in picks}
        added = []
        for _, name, carried, _ in self.attribute_menu():
            if len(picks) + len(added) >= count:
                break
            if carried < 2 or name in chosen or name in settled or _P1_GENDER.search(name) or _W5_NON_PROPERTY.search(name):
                continue
            chosen.add(name)
            added.append((1, name))
        if added:
            self.log.append({"kind": "plan_fill", "added": [name for _, name in added]})
        return picks + added

    def _p1_gender_keyed(self) -> bool:
        """P1: an opened listing carries a key naming gender, or a seen listing is sold by gender (men's, unisex ...);
        with no opened records at all, the planner is trusted."""
        records = [self.opened_records.get(listing_id) for listing_id in self.seen_ids]
        records = [record for record in records if record]
        if not records:
            return True
        if any(_P1_GENDER.search(str(key)) for record in records for key in (_attribute_block(record) or {})):
            return True
        return any(_P1_SOLD_BY.search(_to_json(self.rows.get(listing_id) or {})) for listing_id in self.seen_ids)

    _rc_joiners = re.compile(r"\s+(?:and|or)\s+|[/,;(]")

    def _rc_first_property(self, raw: Any) -> str:
        """A label that names two properties keeps only the first."""
        first = self._rc_joiners.split(str(raw or ""))[0]
        return first.strip().rstrip("?.")

    def _rc_plan_picks(self, plan: dict[str, Any]) -> list[tuple[int, str]]:
        """(item number, property) pairs from the plan, one property each and no repeats per item."""
        picks: list[tuple[int, str]] = []
        for entry in _as_list(plan.get("ask")):
            if not isinstance(entry, dict):
                continue
            number = self._rc_slot(entry.get("item"))
            label = self._rc_first_property(entry.get("label"))
            if not label or len(label.split()) > 5:
                continue
            if _W5_NON_PROPERTY.search(label):
                self.log.append({"kind": "nonprop_skip", "label": label})
                continue
            if _P1_GENDER.search(label) and not self._p1_gender_keyed():
                self.log.append({"kind": "gender_skip", "item": number})
                continue
            key = _plain_words(label)
            if any(n == number and _plain_words(other) == key for n, other in picks):
                continue
            picks.append((number, label))
        return picks

    def _rc_crowded(self, notes_heard: dict[str, Any], batch: list[tuple[int, str, str]]) -> list[tuple[int, str, str]]:
        out: list[tuple[int, str, str]] = []
        for raw in _as_list(notes_heard.get("crowded")):
            position = _as_int(raw)
            if position and 1 <= position <= len(batch):
                out.append(batch[position - 1])
        return out

    def _rc_label_of(self, head: list[tuple[int, str]], number: int, text: str) -> str | None:
        """The planned property whose question for item `number` was exactly `text`."""
        for n, lab in head:
            if n != number:
                continue
            if self.question_text(n, lab) == text:
                return lab
        return None

    def _rc_follow_up(self, head: list[tuple[int, str]], ordered: list[tuple[int, str]],
                      crowded: list[tuple[int, str, str]]) -> tuple[int, str, str] | None:
        """The last message: a sharper re-ask (with sample values) of an answer that wanted one question, else
        the next pick of the plan."""
        if crowded:
            # X1: a single-property question answered "one at a time" is a confused reply; a re-ask reveals nothing
            self.log.append({"kind": "crowded_skip", "q": [text for _, text, _ in crowded]})
        for number, text, _ in crowded if any(" and " in text.split("?")[0] for _, text, _ in crowded) else []:
            label = self._rc_label_of(head, number, text)
            if not label:
                continue
            values = self.example_values(number, label)
            if len(values) >= 2:
                return number, label, self.question_text(number, label, values)
        if len(ordered) <= len(head):
            return None
        number, label = ordered[len(head)]
        return number, label, self.question_text(number, label)

    def _m1_interview(self, count: int, order: list[int], keys: dict[int, list[str]]) -> str:
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        told_before = [len(item.told) for item in self.items]
        queue: dict[int, list[str]] = {n: list(keys[n]) for n in order}
        done: dict[int, set[str]] = {n: set() for n in order}
        misses: dict[int, int] = {n: 0 for n in order}
        spent: dict[int, int] = {n: 0 for n in order}
        revealed: set[int] = set()
        # S2: a request asking for one question at a time gets one property per message from the start
        width = 1 if _S2_SINGLE.search(self.request_text or "") else 8
        if width == 1:
            self.log.append({"kind": "s2_single"})
        blank_lists = 0
        for _ in range(count):
            if self.messages_left() <= 0:
                break
            pick = self._m2_next(order, queue, done, revealed, spent, width)
            if not pick:
                break
            number, chosen = pick
            text = self._m1_text(number, chosen)
            before = [len(item.told) for item in self.items]
            got = self.send_questions([(number, chosen[0], text)])
            if not got:
                break
            sent.extend(got)
            spent[number] += 1
            heard = self.read_answers(got, (WALKER_MODEL, JUDGE_MODEL))
            self.apply_answers(heard, told)
            revealed |= {n for n, item in enumerate(self.items, 1) if len(item.told) > before[n - 1]}
            if width > 1 and len(chosen) > 1:
                # S3b: two list replies in a row that state no need (a vague "don't mind" for the whole list counts) also end the lists
                freed = [entry for entry in _as_list(heard.get("free")) if isinstance(entry, dict)]
                told_now = len(self.items[number - 1].told) > before[number - 1]
                blank_lists = 0 if told_now else blank_lists + 1
                if blank_lists >= 2 and not self._rc_crowded(heard, got):
                    width = 1
                    self.log.append({"kind": "s3_single"})
            if width > 1 and self._rc_crowded(heard, got):
                # a refused list asked nothing: its item starts again with one property at a time, its best one first
                width = 1
                self.log.append({"kind": "m1_single"})
                continue
            new = self.items[number - 1].told[before[number - 1]:]
            hit = self._m2_settle(number, chosen, new, queue, done, misses)
            self.log.append({"kind": "m1_ask", "item": number, "keys": chosen, "hit": hit, "next": queue[number][:4],
                             "revealed": sorted(revealed)})
        if not sent:
            return ""
        if told:
            self.epoch += 1
        fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > told_before[n - 1]]
        found = self._w1_value_search(fresh) if fresh else ""
        walked = self._c1_walk_all()
        summary = self._rc_interview_summary(sent, told)
        return summary + ("\n" + found if found else "") + ("\n" + walked if walked else "")

    def _rc_interview_summary(self, sent: list[tuple[int, str, str]], told: list[str]) -> str:
        exchange = "\n".join(f"Q: {q}\nA: {a}" for _, q, a in sent)
        found = "; ".join(told) if told else "none"
        text = ("Kyo's interviewer asked the shopper (each question used one step):\n" + exchange
                + "\nNew firm requirements from these answers: " + found
                + ". Apply them to every pick (search again with the new values where needed).")
        linked = []
        for _, need in self.ref_needs:
            if any(need in t for t in told):
                linked.append(need)
        if linked:
            text += (" These take their value from another listing: " + " | ".join(linked) + ". Find that listing, open its page in full,"
                     " and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.")
        if self.pending_notes:
            text += "\n[notes seen meanwhile] " + " | ".join(self.pending_notes)
            self.pending_notes = []
        return text

    def _m1_item_keys(self, picks: list[tuple[int, str]]) -> dict[int, list[str]]:
        """M1: per item, the plan's properties for it first, then the most carried open keys of its own category (a plain
        size last)."""
        menu = self.attribute_menu()
        wanted = self._rb_wanted()
        gender_ok = self._p1_gender_keyed()
        out: dict[int, list[str]] = {}
        for number, item in enumerate(self.items, 1):
            settled = " ".join(getattr(item, "brief_needs", item.firm_needs)).lower()
            keys: list[str] = []
            seen: set[str] = set()
            for n, label in picks:
                key = _plain_words(label)
                if n == number and key and key not in seen:
                    seen.add(key)
                    keys.append(label)
            own = [wanted[number - 1]] if number - 1 < len(wanted) else []
            sizes: list[str] = []
            # M3: keys two or more listings carry first, then those only one carries
            for floor in (2, 1):
                for category, name, carried, _ in menu:
                    if len(self.items) > 1 and not self._rb_fits_wanted(category, own):
                        continue
                    if not name or name in seen or carried < floor or name in settled or len(name.split()) > 4:
                        continue
                    if _W5_NON_PROPERTY.search(name) or (_P1_GENDER.search(name) and not gender_ok):
                        continue
                    seen.add(name)
                    # K1: plain size and brand keys almost never hide a need (size 0/890, brand 3%, races 13-16)
                    (sizes if re.fullmatch(r"sizes?|brands?", name) else keys).append(name)
            # K1: the plan's own size or brand picks also go after the other keys
            dead = [k for k in keys if re.fullmatch(r"sizes?|brands?", _plain_words(k))]
            out[number] = [k for k in keys if k not in dead] + dead + sizes
        return out

    def _m1_text(self, number: int, keys: list[str]) -> str:
        if len(keys) == 1:
            return self.question_text(number, keys[0])
        return f"What {', '.join(keys[:-1])} or {keys[-1]} would you like for the {self.items[number - 1].name}?"

    def _m2_next(self, order: list[int], queue: dict[int, list[str]], done: dict[int, set[str]], revealed: set[int],
                 spent: dict[int, int], width: int) -> tuple[int, list[str]] | None:
        """M2: the next message: items with no reveal before revealed ones, fewer messages first, then plan order; the
        item's queue without ended properties and properties an answer already told."""
        for number in sorted(order, key=lambda n: (n in revealed, spent[n], order.index(n))):
            told = [_m2_words(t.split(":", 1)[0]) for t in self.items[number - 1].told]
            left = [k for k in queue[number] if k not in done[number] and not any(
                _m2_words(k) and (_m2_words(k) <= t or (len(t) >= 2 and t <= _m2_words(k))) for t in told)]
            if left:
                return number, left[:width]
        return None

    @staticmethod
    def _m2_hit(chosen: list[str], new: list[str]) -> int:
        """Index of the earliest listed property a revealed need is about (most shared words); -1 when none maps."""
        hits = []
        for need in new:
            prop = _m2_words(need.split(":", 1)[0])
            scores = [len(_m2_words(k) & prop) for k in chosen]
            if scores and max(scores) > 0:
                hits.append(scores.index(max(scores)))
        return min(hits) if hits else -1

    def _m2_settle(self, number: int, chosen: list[str], new: list[str], queue: dict[int, list[str]],
                   done: dict[int, set[str]], misses: dict[int, int]) -> int:
        """M2: end what the answer settled and put the item's queue in its next order; returns the answered index."""
        if len(chosen) == 1:
            done[number].add(chosen[0])
            return 0 if new else -1
        hit = self._m2_hit(chosen, new)
        if hit >= 0:
            done[number].update(chosen[:hit + 1])
            ahead = chosen[hit + 1:]
            misses[number] = 0
        else:
            done[number].add(chosen[0])
            ahead = chosen[:0:-1]
            misses[number] += 1
        rest = [k for k in queue[number] if k not in chosen]
        queue[number] = rest + ahead if hit < 0 and misses[number] % 2 == 0 else ahead + rest
        return hit

    @staticmethod
    def _m1_next(order: list[int], keys: dict[int, list[str]], asked: dict[int, list[list[str]]], revealed: set[int],
                 width: int) -> tuple[int, list[str]] | None:
        """The next message: items never asked, in plan order; then an item with no reveal (its list reversed once,
        then its next keys); then a revealed item's next keys."""
        def fresh(n: int) -> list[str]:
            used = {k for lst in asked[n] for k in lst}
            return [k for k in keys[n] if k not in used][:width]
        for n in order:
            if not asked[n]:
                return n, keys[n][:width]
        for n in order:
            if n in revealed:
                continue
            last = asked[n][-1]
            if width > 1 and len(asked[n]) == 1 and len(last) >= 2:
                return n, list(reversed(last))
            if fresh(n):
                return n, fresh(n)
        for n in order:
            if n in revealed and fresh(n):
                return n, fresh(n)
        return None

    def run_interview(self) -> str:
        self.interviewed = True
        count = self.messages_left()
        if count <= 0:
            return ""
        self.open_more()
        # the plan holds one pick beyond the message budget: the last message takes it when no answer needs a re-ask
        # M5: at least eight picks, the most one M1 list names
        plan = self.json_from_model(self._rc_plan_prompt(max(count, 7)), [JUDGE_MODEL, WALKER_MODEL], 1800) or {}
        picks = self._rc_plan_picks(plan)
        self.log.append(dict(kind="plan", ask=_trim_value(plan.get("ask"), 120)))
        # M1: each message lists several properties of one item; each answer is read as it comes
        keys = self._m1_item_keys(picks)
        order = [n for n in dict.fromkeys([n for n, _ in picks] + list(range(1, len(self.items) + 1))) if keys.get(n)]
        if order:
            return self._m1_interview(count, order, keys)
        ordered = self._p2_fill(picks, count)
        # every message but the last goes out at once; the last one is decided after reading those answers
        head = ordered[:max(1, count - 1)]
        opening = []
        for n, lab in head:
            opening.append((n, lab, self.question_text(n, lab)))
        batch = self.send_questions(opening)
        if not batch:
            return ""
        sent = list(batch)
        told: list[str] = []
        told_before = [len(item.told) for item in self.items]
        notes_heard = self.read_answers(batch)
        self.apply_answers(notes_heard, told)
        extra = None
        if count >= 2 and self.messages_left() > 0:
            extra = self._rc_follow_up(head, ordered, self._rc_crowded(notes_heard, batch))
        if extra:
            more = self.send_questions([extra])
            if more:
                sent.extend(more)
                # a single answer is read by the light model
                self.apply_answers(self.read_answers(more, (WALKER_MODEL, JUDGE_MODEL)), told)
        if told:
            self.epoch += 1
        fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > told_before[n - 1]]
        found = self._w1_value_search(fresh) if fresh else ""
        walked = self._c1_walk_all()
        summary = self._rc_interview_summary(sent, told)
        return summary + ("\n" + found if found else "") + ("\n" + walked if walked else "")

    def _c1_seller_key(self) -> str | None:
        counts: dict[str, int] = {}
        for row in list(self.rows.values())[:200]:
            for key, value in row.items():
                if isinstance(value, str) and 2 < len(value) <= 40 and re.search("seller|shop|store|vendor|merchant|maker|brand", str(key), re.I):
                    counts[key] = counts.get(key, 0) + 1
        return max(counts, key=lambda k: (re.search("seller|shop|store|vendor|merchant", k, re.I) is not None, counts[k])) if counts else None

    def _c1_parse(self, text: str, sellers: set[str]) -> tuple[str, set[str]]:
        """The seller a pointer names and the words of the details it gives about that listing."""
        seller = ""
        for word in re.findall(r"\b([A-Z][a-z]{3,}[a-z]*)(?:'s)?\b", text):
            if word.lower() in sellers:
                seller = word
                break
        if not seller:
            hit = re.search(r"\b(?:by|from|for)\s+([A-Z][a-z]{3,})\b|\b([A-Z][a-z]{3,})'s\b", text)
            seller = (hit.group(1) or hit.group(2)) if hit else ""
        after = text[text.find(seller) + len(seller):] if seller else text
        after = re.split(r"\bor wherever\b|\s[-\u2013\u2014]\s*(?:or|anything)\b", after)[0]
        details = {w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", after.lower()) if (len(w) >= 3 or w.isdigit()) and w not in _C1_STOP}
        return seller, details

    @staticmethod
    def _c1_text(record: dict[str, Any]) -> str:
        leaves: list[str] = []
        stack: list[Any] = [record]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                    else:
                        leaves.append(f"{key} {value}")
            elif isinstance(node, list):
                stack.extend(node)
            else:
                leaves.append(str(node))
        return re.sub(r"(?<=\d),(?=\d{3})", "", " ".join(leaves).lower())

    def _c1_score(self, record: dict[str, Any], details: set[str]) -> tuple[int, int]:
        """How many named details the listing's own attribute values, title and self-description show (not the free
        description, where decoys mention them too); ties go to the whole page."""
        own = " ".join(f"{k} {v}" for k, v in (_attribute_block(record) or {}).items())
        for key, value in record.items():
            if isinstance(value, str) and len(value) <= 160 and not re.search("descr", str(key), re.I):
                own += " " + value
            elif isinstance(value, str) and re.search("note|remark", str(key), re.I):
                own += " " + " ".join(x for x in re.split(r"(?<=[.!?])\s+", value) if not _C1_POINTER.search(x) and not re.search(r"\bversions?\b", x, re.I))
        own = re.sub(r"(?<=\d),(?=\d{3})", "", own.lower())
        tokens = lambda text: set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text))
        return len(details & tokens(own)), len(details & tokens(self._c1_text(record)))

    def _c1_misses(self, ids: list[str], details: set[str]) -> dict[str, int]:
        """Per candidate: attribute keys where another candidate's value shows a named detail and its own does not."""
        blocks = {i: {_plain_words(str(k)): str(v).lower() for k, v in (_attribute_block(self.opened_records.get(i) or self.rows.get(i) or {}) or {}).items()} for i in ids}
        hit = lambda value: bool(details & set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", value)))
        keyed = {k for b in blocks.values() for k, v in b.items() if hit(v)}
        return {i: sum(1 for k in keyed if k in b and not hit(b[k])) for i, b in blocks.items()}

    def _c1_next_pointer(self, record: dict[str, Any], prop: str) -> str:
        """A sentence of the page that sends this property to another listing, or ''."""
        words = {w for w in re.findall("[a-z]+", prop.lower().replace("colour", "color")) if len(w) > 2}
        for key, value in record.items():
            if not isinstance(value, str) or len(value) < 20:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", value):
                low = sentence.lower().replace("colour", "color")
                if _C1_POINTER.search(sentence) and words and words <= set(re.findall("[a-z]+", low)) and re.search(r"[A-Z][a-z]{3,}", sentence[1:]):
                    return sentence
        return ""

    def _c1_value(self, record: dict[str, Any], prop: str) -> str:
        target = _plain_words(prop.replace("colour", "color"))
        for key, value in (_attribute_block(record) or {}).items():
            name = _plain_words(str(key).replace("colour", "color"))
            if name == target or (target and (target in name.split() or name in target.split())):
                return str(value)
        return ""

    def _c1_walk(self, need: str) -> tuple[str, list[str]]:
        """Walk one chain; the last page's value for the property (or '') and the ids opened along it."""
        prop, _, pointer = need.partition(":")
        seller_key = self._c1_seller_key()
        tool = self.attribute_viewer()
        target = self.id_param(tool)
        if not (seller_key and tool and target):
            return "", []
        path: list[str] = []
        text = pointer
        for _ in range(5):
            if self.steps_left() < 18 or self.is_over:
                break
            sellers = {str(r.get(seller_key) or "").strip().lower() for r in self.rows.values()} - {""}
            seller, details = self._c1_parse(text, sellers)
            if not seller:
                break

            def of_seller() -> list[str]:
                return [i for i, r in self.rows.items() if str(r.get(seller_key) or "").strip().lower() == seller.lower() and i not in path]

            def score(listing_id: str) -> tuple[int, int]:
                record = self.opened_records.get(listing_id) or self.rows.get(listing_id) or {}
                return self._c1_score(record, details)
            pool = of_seller()
            if not any(score(i)[0] >= max(1, len(details) // 2) for i in pool if i in self.opened_records):
                shape = self._w1_search_shape()
                if shape:
                    search, base, field = shape
                    words = " ".join([seller] + sorted(details, key=lambda w: (not w.isdigit(), -len(w)))[:4])
                    problems: list[str] = []
                    call = _conform({**base, field: words}, self.tool_schemas.get(search), "args", problems)
                    if not problems:
                        reply = self.store_call(search, call)
                        if reply is not None:
                            self.take_reply(search, reply.get("observation"))
                pool = of_seller()
            misses = self._c1_misses(pool, details)
            pool.sort(key=lambda i: (score(i)[0] - misses.get(i, 0), score(i)[1]), reverse=True)
            for listing_id in [i for i in pool if i not in self.opened_records][:3]:
                if self.steps_left() < 18:
                    break
                self._rc_open_one(tool, target, listing_id)
            opened = [i for i in pool if i in self.opened_records]
            if not opened:
                break
            misses = self._c1_misses(opened, details)
            best = max(opened, key=lambda i: (score(i)[0] - misses.get(i, 0), score(i)[1]))
            if score(best)[0] < max(1, len(details) // 2):
                break
            path.append(best)
            record = self.opened_records.get(best) or {}
            onward = self._c1_next_pointer(record, prop)
            if onward:
                text = onward
                continue
            return self._c1_value(record, prop), path
        return "", path

    def _c1_walk_all(self) -> str:
        """C1: walk every requirement still read from another listing; returns the note for the walker."""
        if self.steps_left() < 30:
            return ""
        live = {need for _, need in self.needs_in_force()}
        lines = []
        for _, need in list(self.ref_needs):
            if need in self.ref_text or need not in live or self.steps_left() < 30:
                continue
            value, path = self._c1_walk(need)
            self.log.append({"kind": "chain_walk", "need": need[:120], "path": path, "value": value})
            if value:
                self.ref_text[need] = value
                self.epoch += 1
                lines.append(f"{need.split(':', 1)[0].strip()}: the chain {' -> '.join(path)} ends on {path[-1]}, whose page gives {value!r}")
            elif path:
                lines.append(f"{need.split(':', 1)[0].strip()}: Kyo followed the chain through {' -> '.join(path)}; read the last page's note to go on")
        if not lines:
            return ""
        return ("[Kyo walked the chains these requirements point along, opening every page] " + " | ".join(lines)
                + ". Apply the value as the requirement says (the same value, a matching one, or one unlike it).")

    def _w_owner(self, label: str) -> int:
        """The requested item a category label belongs to (0 when none): exact shelf, exact name, else most shared words."""
        if not label:
            return 0
        plain, roots = _plain_words(label), _w_roots(label)
        best, top = 0, 0.0
        for number, item in enumerate(self.items, 1):
            if plain == _plain_words(item.shelf_name or ""):
                score = 3.0
            elif plain == _plain_words(item.name):
                score = 2.0
            else:
                mine = _w_roots(item.shelf_name or "") | _w_roots(item.name)
                score = len(roots & mine) / max(1, len(roots | mine))
            if score > top:
                best, top = number, score
        return best

    def _w1_search_shape(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool, the fixed arguments of the agent's last worded search, and its free-text field."""
        tool = self.jobs.get("search")
        schema = self.tool_schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = schema.get("required") or []
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _schema_types(v) and "enum" not in v]
        queries = [k for k in texts if re.search("(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword", k, re.I)]
        if not tool or not queries:
            return None
        cat_key = self.shelf_field()
        shelves = {_plain_words(item.shelf_name or "") for item in self.items} | {_plain_words(item.name) for item in self.items}
        if cat_key:
            shelves |= {_plain_words(str(row.get(cat_key) or "")) for row in self.rows.values()}
        shelves.discard("")
        for args in reversed(self.searched):
            fields = [k for k in queries if isinstance(args.get(k), str) and re.search("[a-z]", args[k], re.I)]
            if not fields:
                continue
            field = fields[0]
            keep: dict[str, Any] = {}
            for key, value in args.items():
                if key == field or not (isinstance((props.get(key) or {}).get("enum"), list) or key in required):
                    continue
                # a category filter would hide the other kinds a fresh value search may need; keep it only when required
                if key not in required and isinstance(value, str) and _plain_words(value) in shelves:
                    continue
                keep[key] = value
            return tool, keep, field
        return None

    def _w1_values(self, number: int) -> list[str]:
        """Plain stated values of an item's requirements in force (no limits, exclusions or pointers)."""
        pointed = [need for _, need in self.ref_needs]
        values = []
        for n, need in self.needs_in_force():
            if n != number or need in pointed or ":" not in need:
                continue
            value = need.split(":", 1)[1].strip()
            if value and not _W1_LIMIT_WORDS.search(value):
                values.append(value)
        return values

    def _w1_value_search(self, numbers: list[int]) -> str:
        """W1: search each item that just got a requirement with its values together, and open the first fitting rows."""
        shape = self._w1_search_shape()
        if not shape or self.steps_left() < 22:
            return ""
        tool, base, field = shape
        cat_key = self.shelf_field()
        room = VALUE_SEARCH_STEPS
        lines: list[str] = []
        for number in numbers:
            item = self.items[number - 1]
            values = self._w1_values(number)
            heard = {t.split(":", 1)[1].strip() for t in item.told if ":" in t}
            if not any(v in heard for v in values) or room <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in heard]
            tries = [values]
            if len(values) > 1:
                drop = stated[-1] if stated else 0
                tries.append(values[:drop] + values[drop + 1:])
            for words in tries:
                if not words:
                    continue
                problems: list[str] = []
                call = _conform({**base, field: " ".join(words)}, self.tool_schemas.get(tool), "args", problems)
                if problems or self.is_over:
                    break
                reply = self.store_call(tool, call)
                room -= 1
                if reply is None:
                    break
                observation = reply.get("observation")
                self.take_reply(tool, observation)
                found = [i for i, _ in self.listing_records(observation)]
                ceiling = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.price_for(i) for i in found]
                if ceiling is not None and prices and all(p is not None and p > 10 * ceiling for p in prices):
                    ceiling = None   # the budget is in another unit than the prices
                fitting = [i for i in found if self.can_buy(i)
                           and (not cat_key or self._w_owner(str((self.rows.get(i) or {}).get(cat_key) or "")) == number)
                           and (ceiling is None or (self.price_for(i) or 0) <= ceiling)]
                lines.append(f"{_to_json(call.get(field))}: " + (", ".join(fitting[:6]) or "no row of that kind"))
                if fitting:
                    opened = self.read_attributes([i for i in fitting if i not in self.opened_records][:min(3, room)], self.reserve_steps() + 16)
                    room -= len(opened)
                    if opened:
                        lines.append("opened: " + " | ".join(opened)[:2400])
                    break
        if not lines:
            return ""
        self.log.append({"kind": "value_search", "seen": [line[:200] for line in lines]})
        return "[Kyo searched each such item's requirement values together and read the first fitting-kind rows] " + "\n".join(lines)

    def question_text(self, number: int, label: str, example_values: list[str] | None = None) -> str:
        stem = f"What {label} would you like for the {self.items[number - 1].name}"
        if not example_values:
            return stem + "?"
        return stem + f", for example {example_values[0]} or {example_values[1]}?"

    def example_values(self, number: int, label: str) -> list[str]:
        """Two values the listings carry for this property, showing the shopper which property is meant."""
        key = _plain_words(label)
        settled = " ".join(self.items[number - 1].firm_needs).lower()
        values = next((vals for _, name, _, vals in self.attribute_menu() if name == key), None)
        picked: list[str] = []
        for value in values or []:
            if len(picked) == 2:
                break
            if value and value not in settled and len(value) <= 30:
                picked.append(value)
        return picked

    def _rc_record_answer(self, number: int, text: str, answer: Any) -> tuple[int, str, str]:
        self.messages_sent += 1
        words = self.words_of(answer)
        self.record_lines.append("step {}: message to the shopper {!r} -> reply {!r} (cart then: {})".format(self.steps_used, text, words[:300], self.cart_stamp()))
        self.record_steps.append(self.steps_used)
        self.log.append(dict(kind="ask", q=text, a=words[:300]))
        return number, text, words

    def _rc_cannot_message(self, floor: int) -> bool:
        if self.is_over or self.steps_left() <= floor:
            return True
        return self.messages_sent >= self.message_cap

    def send_questions(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.jobs["ask"]
        delivered: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self._rc_cannot_message(floor):
                break
            call_args = self.ask_arguments(text)
            if not call_args:
                break
            result = self.store_call(tool, call_args)
            if result is None:
                break
            answer = result.get("observation")
            delivered.append(self._rc_record_answer(number, text, answer))
            if isinstance(answer, dict) and answer.get("error"):
                break
        return delivered

    @staticmethod
    def _rc_positive(value: Any) -> bool:
        """A real positive number (booleans do not count)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return value > 0

    def _rc_heard_needs(self, entries: list[Any], told: list[str]) -> None:
        several = len(self.items) > 1
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            need = _v3_clean(str(entry.get("need") or "").strip())
            if not need:
                continue
            number = self._rc_slot(entry.get("item"))
            item = self.items[number - 1]
            if need not in item.told:
                item.told.append(need)
                told.append(f"item {number}: {need}" if several else need)
            pair = (number, need)
            if entry.get("points") is True and pair not in self.ref_needs:
                self.ref_needs.append(pair)

    @staticmethod
    def _rc_live_needs(item: Wanted) -> list[str]:
        """An item's requirements (stated or told) that are not dropped."""
        return [need for need in item.firm_needs + item.told if need not in item.given_up]

    def _rc_heard_drops(self, entries: list[Any], told: list[str]) -> None:
        for entry in entries:
            raw = str(entry.get("property") or "") if isinstance(entry, dict) else ""
            if not raw.strip():
                continue
            item = self.items[self._rc_slot(entry.get("item")) - 1]
            sold_at = _plain_words(raw.split(":")[0])
            matched = []
            for need in self._rc_live_needs(item):
                if _plain_words(need.split(":")[0]) == sold_at:
                    matched.append(need)
            item.given_up.extend(matched)
            told.extend(f"dropped {need}" for need in matched)

    def apply_answers(self, notes_heard: dict[str, Any], told: list[str]) -> None:
        """Put what the answers said into the ledger; new requirements also go to `told` for the summary."""
        self._rc_heard_needs(_as_list(notes_heard.get("needs")), told)
        self._rc_heard_drops(_as_list(notes_heard.get("drop")), told)
        for entry in _as_list(notes_heard.get("free")):
            if not isinstance(entry, dict) or not entry.get("property"):
                continue
            self.items[self._rc_slot(entry.get("item")) - 1].open_props.append(str(entry["property"]))
        if self._rc_positive(notes_heard.get("budget")):
            self.budget = notes_heard["budget"]

    def read_answers(self, sent: list[tuple[int, str, str]], models: tuple[str, str] = (JUDGE_MODEL, WALKER_MODEL)) -> dict[str, Any]:
        item_lines = []
        for n, it in enumerate(self.items, 1):
            item_lines.append(f"item {n}: {it.name}")
        qa_lines = []
        for i, (n, q, a) in enumerate(sent, 1):
            qa_lines.append(f"{i}. (about item {n}) Q: {q}\nA: {a}")
        prompt = "".join([ANSWER_PROMPT, "\n\nITEMS:\n", "\n".join(item_lines), "\n\nQUESTIONS AND ANSWERS:\n", "\n".join(qa_lines)])
        return self.json_from_model(prompt, list(models), 1500) or {}

    # ---------- the process gate before an order or an ending ----------
    def _rc_gate_prompt(self, stage: str, steps: list[dict[str, Any]], tools: list[str]) -> str:
        ordering = stage == "order"
        record = "(nothing yet)"
        if self.record_lines:
            record = "\n".join("%d. %s" % (i, line) for i, line in enumerate(self.record_lines, 1))
        parts = [STEP_GATE_PROMPT.replace("STAGE", "place the order" if ordering else "end without a purchase"),
                 "\n\nREQUEST:\n", self.request_text,
                 "\n\nSTEPS DUE NOW:\n", "\n".join(f"- {p['step']}" for p in steps),
                 "\n\nLEDGER:\n", self.ledger_text(),
                 "\n\nCART NOW: ", self.cart_summary(),
                 "\n\nRECORD (messages and step calls so far):\n", record]
        if self.note_texts:
            parts += ["\n\nSTORE NOTES SEEN:\n", "\n".join(self.note_texts[-4:])]
        what = "order" if ordering else "ending"
        parts.append(f"\n\nMESSAGES LEFT: {max(0, self.message_cap - self.messages_sent)}   STEPS LEFT: {self.steps_left()} (the {what} itself takes one)")
        parts += ["\n\nTOOLS:\n", "\n".join(tools)]
        return "".join(parts)

    def _rc_gate_unproven(self, stage: str, steps: list[dict[str, Any]], got: dict[str, Any]) -> list[dict[str, Any]]:
        """Steps the model calls done without pointing at a record line that did them. At the order, a step that
        takes a message concerns the final cart, so its line must be later than the last cart change."""
        talk = False
        if stage == "order":
            talk = any(p["messages"] > 0 for p in steps)
        if talk:
            recent_calls = self.record_steps[:len(self.record_lines)]
            proof = {i for i, made_at in enumerate(recent_calls, 1) if made_at > self.cart_changed_at}
        else:
            proof = set(range(1, len(self.record_lines) + 1))
        missing = []
        for entry in _as_list(got.get("steps")):
            if not isinstance(entry, dict):
                continue
            cited = {_as_int(n) for n in _as_list(entry.get("lines"))}
            if entry.get("done") is not True or not cited & proof:
                missing.append(entry)
        return missing

    def _rc_gate_nudge(self, unproven: list[dict[str, Any]]) -> None:
        self.log.append({"kind": "gate_unproven", "steps": _trim_value(unproven, 120)})
        named = "; ".join([str(e.get("step") or "") for e in unproven])[:200]
        left_msgs = self.message_cap - self.messages_sent
        self.record_lines.append(f"(Kyo: no line made since the cart last changed at step {self.cart_changed_at} shows " + named
                             + f" done, and {left_msgs} message(s) and {self.steps_left()} steps are left: give the call that does it)")
        self.record_steps.append(-1)

    def _rc_gate_store(self, name: str, args: dict[str, Any], said: list[str]) -> bool:
        """Make a non-message call the gate asked for; False when it did not go through."""
        result = self.store_call(name, args)
        if result is None:
            return False
        policy = result.get("observation")
        self.take_in(policy)
        call_text, view_text = _to_json(args), self.render_reply(policy)
        self.record_lines.append("step {}: {} {} -> {} (cart then: {})".format(self.steps_used, name, call_text, view_text[:300], self.cart_stamp()))
        self.record_steps.append(self.steps_used)
        said.append("{} {} -> {}".format(name, call_text, view_text[:400]))
        return True

    def _rc_gate_message(self, args: dict[str, Any], said: list[str]) -> bool:
        text = ""
        for value in args.values():
            if isinstance(value, str):
                text = value
                break
        batch = self.send_questions([(1, "", text)], 1)
        if not batch:
            return False
        self.gate_messages += 1
        said.append(f"Q: {text}\nA: {batch[0][2]}")
        return True

    def _rc_gate_reply(self, got: dict[str, Any], told: list[str]) -> str:
        self.apply_answers({"needs": got.get("needs")}, told)
        return str(got.get("hold") or "").strip()

    @staticmethod
    def _rc_did_steps(said: list[str]) -> str:
        return "Before going on, Kyo did the steps the request asks for:\n" + "\n".join(said)

    def step_gate(self, stage: str) -> tuple[bool, str]:
        """Just before a checkout ("order") or an ending with no purchase ("end"): carry out the steps the request
        wants at that point through store calls the model picks (a shopper message, a page, any store tool) until
        the model sees none missing for the current cart. Returns (go on?, text for the agent)."""
        steps = [p for p in self.process_steps if p["when"] == stage]
        if not steps or self.is_over or not self.steps_due(stage):
            return True, ""
        ending = {self.jobs.get("buy"), self.jobs.get("refuse")} - {None}
        tools = []
        for spec in self.tools:
            fn = spec["function"]
            if fn["name"] in ending:
                continue
            tools.append("- {}: {} | parameters: {}".format(fn["name"], fn.get("description", ""), _to_json(fn.get("parameters") or {})))
        asker = self.jobs.get("ask")
        said: list[str] = []
        nudged = False
        for _ in range(STEP_GATE_ROUNDS):
            # the light model decides, the heavy one after a nudge
            models = [JUDGE_MODEL, WALKER_MODEL] if nudged else [WALKER_MODEL, JUDGE_MODEL]
            got = self.json_from_model(self._rc_gate_prompt(stage, steps, tools), models, 2000)
            if got is None:
                break
            # only a reply to the gate's own call is read here (earlier answers belong to the interviewer)
            told: list[str] = []
            hold = self._rc_gate_reply(got, told) if said else ""
            call = got.get("call")
            if not isinstance(call, dict):
                call = None
            self.log.append(dict(kind="gate", stage=stage, call=_trim_value(call, 200), steps=_trim_value(got.get("steps"), 120),
                                   hold=hold[:200], told=told))
            if told or hold:
                self.epoch += 1
                prefix = self._rc_did_steps(said) + "\n" if said else ""
                changes = told + [hold] if hold else told
                return False, prefix + "The shopper's reply changes the task: " + "; ".join(changes) + ". Fix the cart for it, then check out again."
            name = str((call or {}).get("tool") or "")
            unproven = self._rc_gate_unproven(stage, steps, got)
            if (not call and unproven and not (said or nudged) and asker
                    and self.messages_sent < self.message_cap and self.steps_left() >= 2):
                nudged = True
                self._rc_gate_nudge(unproven)
                continue
            if not call:
                self.gate_passed.add((stage, "{}#{}".format(self.cart_stamp(), self.epoch)))
                break
            if name not in self.tool_schemas or name in ending:
                break
            is_message = name == asker
            if self.steps_left() < 2 or (is_message and self.messages_sent >= self.message_cap):
                break
            problems: list[str] = []
            raw = call.get("args")
            args = _conform(raw if isinstance(raw, dict) else {}, self.tool_schemas[name], "args", problems)
            if problems:
                break
            went = self._rc_gate_message(args, said) if is_message else self._rc_gate_store(name, args, said)
            if not went:
                break
        if not said:
            return True, ""
        return True, self._rc_did_steps(said)

    def words_of(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []
        stack: list[tuple[Any, int]] = [(observation, 0)]
        # depth-first, children in their own order, nothing below depth 4
        while stack:
            node, depth = stack.pop()
            if depth > 4:
                continue
            if isinstance(node, str):
                if len(node.split()) >= 2:
                    found.append(node)
                continue
            if isinstance(node, dict):
                children = list(node.values())
            elif isinstance(node, list):
                children = list(node)
            else:
                continue
            stack.extend((child, depth + 1) for child in reversed(children))
        if not found:
            return _to_json(observation)[:400]
        return " ".join(found)[:800]

    # ---------- notes inside tool results ----------
    def _rc_loose_in(self, value: Any) -> list[str]:
        """Sentence-like strings carried by one top-level field of a result."""
        if isinstance(value, str) and len(value.split()) >= 5:
            return [value]
        if isinstance(value, list):
            only_text = bool(value) and all(isinstance(v, str) for v in value)
            return [v for v in value if len(v.split()) >= 5] if only_text else []
        if isinstance(value, dict) and not self.listing_id(value, False, 1):
            return [v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6]
        return []

    def note_lines(self, observation: Any) -> list[str]:
        if not isinstance(observation, dict) or self.listing_id(observation):
            return []
        fresh: list[str] = []
        for value in observation.values():
            fresh.extend(line for line in self._rc_loose_in(value) if line not in self.notes_heard)
        return fresh

    def _rc_note_drop(self, entry: dict[str, Any]) -> str:
        item = self.items[self._rc_slot(entry.get("item")) - 1]
        target = str(entry["need"]).strip()
        live = self._rc_live_needs(item)
        exact = [m for m in live if m == target]
        if exact:
            item.given_up.extend(exact)
        else:
            stem = _plain_words(target.split(":")[0])
            near = next((m for m in live if _plain_words(m.split(":")[0]) == stem), None)
            item.given_up.append(target if near is None else near)
        return f"dropped {target}"

    def _rc_note_add(self, entry: dict[str, Any]) -> str:
        item = self.items[self._rc_slot(entry.get("item")) - 1]
        need = _v3_clean(str(entry["need"]).strip())
        if need in item.given_up:
            item.given_up.remove(need)
        if need not in item.firm_needs + item.told:
            item.told.append(need)
        return f"added {need}"

    @staticmethod
    def _rc_need_entries(raw: Any) -> list[dict[str, Any]]:
        return [e for e in _as_list(raw) if isinstance(e, dict) and str(e.get("need") or "").strip()]

    def _rc_new_version(self, note: str) -> None:
        """A note changed the task: bump the version, forget recent calls, keep the note for the gate."""
        self.epoch += 1
        self.recent_calls = []
        self.note_texts.append(note)

    def apply_note(self, lines: list[str]) -> str:
        self.notes_heard.update(lines)
        prompt = "".join([NOTICE_PROMPT, "\n\nITEMS AND REQUIREMENTS NOW:\n", self.ledger_text(), "\n\nTHE NOTE:\n", "\n".join(lines)])
        got = self.json_from_model(prompt, [WALKER_MODEL, BACKUP_MODEL], 1200) or {}
        sold_at = []
        for g in _as_list(got.get("gone")):
            if isinstance(g, (str, int)):
                sold_at.append(str(g))
        self.unbuyable.update(g for g in sold_at if g in self.rows)
        self.note_said_sold = bool(sold_at) or got.get("sold_out") is True
        if len(lines) == 1 and self.note_said_sold:
            # remember a note only when it says nothing but the sell-out: the same words later mean the same
            self.sold_phrases.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed: list[str] = []
        budget = got.get("budget")
        if self._rc_positive(budget) and budget != self.budget:
            if isinstance(self.budget, (int, float)) and budget > self.budget:
                self.loosened += 1
            self.budget = budget
            changed.append(f"budget now {budget}")
        drops = self._rc_need_entries(got.get("drop"))
        adds = self._rc_need_entries(got.get("add"))
        changed.extend(self._rc_note_drop(e) for e in drops)
        self.loosened += len(drops)
        changed.extend(self._rc_note_add(e) for e in adds)
        if sold_at:
            changed.append("unavailable: " + ", ".join(sold_at))
        if got.get("price_moved") is True:
            changed.append("a price changed")
        budget_moved = any(c.startswith("budget now") for c in changed)
        if self.plan_brief.get("pick") == "cheapest" and (drops or sold_at or self.note_said_sold or budget_moved):
            # W4: the fitting set changed, so the cheapest pick is open again
            self.wider_done = False
            self.cheaper_seen = set()
        summary = str(got.get("summary") or "").strip()
        if changed or summary:
            self._rc_new_version(summary or "; ".join(changed))
        if not changed:
            return summary.strip()
        return (summary + " (" + "; ".join(changed) + ")").strip()


    def needs_in_force(self) -> list[tuple[int, str]]:
        """(item number, requirement) for every requirement not dropped yet, items counted from 1."""
        return [(position, want)
                for position, entry in enumerate(self.items, 1)
                for want in [*entry.firm_needs, *entry.told]
                if want not in entry.given_up]

    def _rd_fact_ids(self, observation):
        """Ids of the listings an observation carries an attribute block for, in depth-first order (six levels deep)."""
        found = []
        stack = [(observation, 0)]
        while stack:
            node, level = stack.pop()
            if level > 6:
                continue
            if isinstance(node, list):
                children = node
            elif isinstance(node, dict):
                name = self.listing_id(node, False, level)
                if name and name not in found and _attribute_block(node):
                    found.append(name)
                    continue
                children = list(node.values())
            else:
                continue
            stack.extend((child, level + 1) for child in reversed(children))
        return found

    def _rd_evidence_line(self, name, props):
        """The keys of one listing's attribute block that name each requirement's property."""
        block = _attribute_block(self.opened_records.get(name) or {}) or {}
        pieces = []
        for label, prop in props:
            stem = _word_stems(prop)
            matched = [f"{key}={str(value)[:50]}" for key, value in block.items() if _names_property(stem, _word_stems(key))][:3]
            if matched:
                pieces.append(f"{label}{prop} -> {', '.join(matched)}")
        return "; ".join(pieces)

    def evidence_lines(self, observation: Any) -> str:
        found = self._rd_fact_ids(observation)
        needs = self.needs_in_force()
        if not (found and needs):
            return ""
        several = len(self.items) > 1
        props = [(f"(item {position}) " if several else "", want.split(":", 1)[0].strip()) for position, want in needs[:8]]
        rows = []
        for name in found[:12]:
            line = self._rd_evidence_line(name, props)
            if line:
                rows.append(f"{name}: {line}")
        if not rows:
            return ""
        return ("[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] "
                + " | ".join(rows))

    # ---------- checks ----------
    @staticmethod
    def _rd_snip(record, depth, cap):
        """A record dumped with its values clipped to `depth`, cut to `cap` characters."""
        return _to_json(_trim_value(record, depth))[:cap]

    def _rd_note(self, kind, **fields):
        self.log.append({"kind": kind, **fields})

    def _rd_preamble(self, lead):
        """Opening sections shared by the review prompts: the instructions, the request and the requirements."""
        return [lead, "REQUEST:\n" + self.request_text, "REQUIREMENTS IN FORCE:\n" + self.ledger_text()]

    def _rd_ask(self, sections, models, budget):
        return self.json_from_model("\n\n".join(sections), models, budget)

    def _rd_newest(self, cap, render, order=None):
        """Rendered listings, newest first unless `order` is given, skipping those `render` gives None for; at most `cap`."""
        shown = []
        for listing_id in self.seen_ids[::-1] if order is None else order:
            text = render(listing_id)
            if text is None:
                continue
            shown.append(text)
            if len(shown) >= cap:
                break
        return shown

    def _rd_tagged(self, page, listing_id, wide, narrow):
        """An opened page in full, else the listing's search row in short."""
        if page:
            return "OPENED " + self._rd_snip(page, *wide)
        return "ROW " + self._rd_snip(self.rows.get(listing_id), *narrow)

    def _rd_view_fresh(self):
        """Whether the last cart view was taken after the last cart change."""
        return self.cart_snapshot is not None and self.cart_viewed_at >= self.cart_changed_at

    def _rd_record(self, listing_id):
        return self.opened_records.get(listing_id) or self.rows.get(listing_id)

    def cart_summary(self) -> str:
        if not self._rd_view_fresh():
            # an outdated view could list lines removed since; describe the cart from the calls Kyo made instead
            return "lines by Kyo's record of the cart calls (listing: units): " + _to_json(self.cart_qty)
        return self._rd_snip(self.cart_snapshot, 300, 3000)

    def cart_listing_ids(self) -> list[str]:
        if not self._rd_view_fresh():
            return list(self.cart_qty)
        shown = _to_json(self.cart_snapshot)
        return list(filter(lambda listing_id: listing_id in shown, self.rows))

    def record_text(self, listing_id: str) -> str:
        found = self._rd_record(listing_id)
        return self._rd_snip(found, 400, 3000) if found else "(no record seen)"

    def _rd_cheap_rank(self, listing_id):
        price = self.price_for(listing_id)
        return listing_id not in self.opened_records, float("inf") if price is None else price

    def other_records(self, skip: list[str], limit: int = 30) -> str:
        newest = self.seen_ids[::-1]
        if self.plan_brief.get("pick") == "cheapest":
            # read listings ahead of rows, and within each group the lowest price first
            newest.sort(key=self._rd_cheap_rank)

        def render(listing_id):
            if listing_id in skip or self.is_gone(listing_id):
                return None
            return self._rd_snip(self._rd_record(listing_id), 160, 900)

        return "\n".join(self._rd_newest(limit, render, newest))

    def verdict_key(self, ids: list[str]) -> str:
        return f"{'|'.join(sorted(ids))}#{self.epoch}"

    def _rd_check_prompt(self, ids, planned):
        sections = self._rd_preamble(CART_CHECK_PROMPT)
        if planned:
            amounts = "; ".join(f"item {position}: quantity {entry.count_wanted}" for position, entry in enumerate(self.items, 1))
            sections.append("CART (planned, judge these listings as if they were carted at the requested quantities: "
                            + amounts + "):\n" + ", ".join(ids))
        else:
            sections.append("CART:\n" + self.cart_summary())
        described = []
        for listing_id in ids:
            flag = "SOLD OUT (the store says it can no longer be bought). " if self.is_gone(listing_id) else ""
            described.append(f"{listing_id}: {flag}{self.record_text(listing_id)}")
        sections.append("RECORDS OF THE CART'S LISTINGS:\n" + "\n".join(described))
        if self.plan_brief.get("pick") in ("cheapest", "most", "least"):
            sections.append("OTHER LISTINGS SEEN (newest first):\n" + self.other_records(ids, 15))
        numeric = self._n1_needs()
        if numeric:
            # N1: the figures behind every requirement that holds a number, line by line
            sections.append(
                "NUMBER CHECK (required in your JSON): these requirements hold a number: "
                + "; ".join(f"item {n}: {need}" for n, need in numeric[:8])
                + ". Add the key \"numbers\": a list with one entry per cart line and each such requirement of that line's item,"
                " {\"line\": listing id, \"need\": the requirement copied, \"stated\": the value the line's records state for it"
                " (attributes, title or description) with its unit, or null when they state none, \"in_need_unit\": that value"
                " converted to the requirement's unit, or null, \"meets\": true or false (null only when stated is null)}."
                " A requirement measured against another listing (\"within N of\", \"no more than\" its value) uses that listing's"
                " figure as the records show it. A line that fails a number or states none makes the cart wrong.")
        return sections

    def _n1_needs(self) -> list[tuple[int, str]]:
        """N1: requirements in force whose text holds a figure."""
        out = []
        for n, need in self.needs_in_force():
            value = need.split(":", 1)[-1]
            pointing = re.search(r"\b(?:same as|identical|unlike|like the|pointed|page of|listing it|sold by|defers?|partner)\b", value, re.I)
            if pointing and not re.search(r"\bwithin\s+[\d.]+", value, re.I):
                continue  # a value read from another listing: the numbers describe that listing, not a limit
            if re.search(r"\d", value):
                out.append((n, need))
        return out

    def _n1_apply(self, got: Any) -> Any:
        """N1: a line that fails a number, or states none, turns the check's verdict into a problem."""
        if not isinstance(got, dict):
            return got
        faults = []
        for entry in _as_list(got.get("numbers")):
            if not isinstance(entry, dict):
                continue
            line, need = str(entry.get("line") or ""), str(entry.get("need") or "")
            if entry.get("meets") is False:
                faults.append(f"{line} does not meet {need} (it states {entry.get('stated')})")
            elif entry.get("stated") in (None, "", "null") and entry.get("meets") is not True:
                faults.append(f"{line} states no value for {need}")
        if faults:
            self._rd_note("n1_faults", faults=faults[:6])
            got = dict(got)
            got["ok"] = False
            got["problems"] = faults[:4] + [str(p) for p in _as_list(got.get("problems"))][:2]
        return got

    def _rd_check_advice(self, ids, got):
        self.review_leads = [lead for lead in self.review_leads if lead not in ids]
        problems = list(map(str, _as_list(got.get("problems"))))[:5]
        better = []
        for entry in _as_list(got.get("better")):
            name = str(entry)
            if name in self.rows and not self.is_gone(name):
                better.append(name)
        said = ["A careful review says this cart is not right yet: " + "; ".join(problems or ["see the requirements"])]
        if better[:4]:
            said.append(". Listings that look better: " + ", ".join(better[:4]))
        said.append(". Fix the cart (or show the review is wrong by re-reading the records), then check out.")
        return "".join(said)

    def _rd_claim_check(self, ids, limit):
        """The check key for `ids` when this cart has not been checked yet and spending allows it (marked as checked)."""
        key = self.verdict_key(ids)
        if ids and self.spend_usd <= limit and key not in self.checked_ids:
            self.checked_ids.add(key)
            return key
        return None

    def judge_cart(self, ids: list[str] | None = None, limit: float = SOFT_BUDGET_USD) -> str:
        planned = ids is not None
        if not planned:
            ids = self.cart_listing_ids()
        key = self._rd_claim_check(ids, limit)
        if key is None:
            return ""
        got = self._n1_apply(self._rd_cascade_check(self._rd_check_prompt(ids, planned), ids))
        self._rd_note("check", cart=ids, got=got)
        passed = bool(got) and got.get("ok") is not False
        self.cart_verdicts[key] = passed
        if passed or not got:
            return ""
        return self._rd_check_advice(ids, got)

    def _rd_cascade_check(self, sections, ids):
        """B2: the walker model checks first; its pass stands unless the pick rule could prefer another shown listing; an
        objection or no answer goes to the judge model with the same prompt."""
        light = self._rd_ask(sections, [WALKER_MODEL], 1500)
        if isinstance(light, dict) and light.get("ok") is True and not self._rd_rule_could_prefer(ids):
            self._rd_note("check_light", cart=ids)
            return light
        if self.aborted or self.out_of_credit:
            return light
        return self._rd_ask(sections, [JUDGE_MODEL, WALKER_MODEL], 1500)

    def _rd_rule_could_prefer(self, ids):
        pick = self.plan_brief.get("pick")
        if pick not in ("cheapest", "most", "least"):
            return False
        if pick != "cheapest":
            return True
        newest = self.seen_ids[::-1]
        newest.sort(key=self._rd_cheap_rank)
        shown = [i for i in newest if i not in ids and not self.is_gone(i)][:15]
        shelf_key = self.shelf_field()
        for line in ids:
            price = self.price_for(line)
            if price is None:
                return True
            shelf = (self.rows.get(line) or {}).get(shelf_key) if shelf_key else None
            for other in shown:
                cheaper = self.price_for(other)
                if cheaper is None or cheaper >= price:
                    continue
                if shelf_key is None or shelf is None or (self.rows.get(other) or {}).get(shelf_key) == shelf:
                    return True
        return False

    def _rd_fit(self, tool, raw):
        """`raw` bent to the tool's schema, or None when the schema rejects it."""
        faults = []
        args = _conform(raw, self.tool_schemas.get(tool), "args", faults)
        return None if faults else args

    def _rd_call(self, tool, raw):
        """Fit `raw` to the tool's schema and run it. Gives (None, None) when the schema rejects the arguments, and
        (arguments, None) when the store answered nothing or the episode is over."""
        args = self._rd_fit(tool, raw)
        if args is None:
            return None, None
        reply = self.store_call(tool, args)
        return args, (None if reply is None or self.is_over else reply)

    def _rd_take(self, tool, reply):
        """Feeds a store reply's observation to the record keeping and hands it back."""
        observation = reply.get("observation")
        self.take_reply(tool, observation)
        return observation

    def _rd_chain_need(self, chain, pending):
        named = str(chain.get("need") or "").strip()
        if named in pending and named:
            return named
        return pending[0] if len(pending) == 1 else None

    def _rd_missing_link(self, need, links, ids):
        """A chain whose next listing nobody has found: what to look for."""
        about = next((str(link.get("about") or "") for link, listing_id in zip(links, ids) if listing_id not in self.rows), "")
        return f"{need} -> find {about or 'the listing it points at'}"

    def _rd_trail_chains(self, got, pending):
        """Reads the model's chains: stores finished values, and returns the known links still unopened and the
        links nobody has found yet."""
        reach, lost = [], []
        for chain in _as_list(got.get("chains")):
            need = self._rd_chain_need(chain, pending) if isinstance(chain, dict) else None
            if not need:
                continue
            links = list(filter(lambda link: isinstance(link, dict), _as_list(chain.get("links"))))
            ids = [str(link.get("id") or "").strip() for link in links]
            value = str(chain.get("value") or "").strip()
            if value and links and chain.get("done") is True and all(i in self.opened for i in ids):
                self.ref_text[need] = value
                self.epoch += 1
            elif any(i in self.rows and i not in self.opened for i in ids):
                fresh = [i for i in ids if i in self.rows and i not in self.opened and i not in reach]
                reach.extend(fresh)
            else:
                lost.append(self._rd_missing_link(need, links, ids))
        return reach, lost

    def _rd_trail_open(self, tool, target, idents, seen):
        for listing_id in idents:
            reply = self._rd_call(tool, {target: listing_id})[1]
            if reply is None:
                return
            seen.append(listing_id + ": " + self.render_reply(self._rd_take(tool, reply))[:1500])

    def _rd_trail_text(self, seen):
        parts = []
        if seen:
            parts.append("[Kyo opened the pages a requirement points at]\n" + "\n".join(seen) + "\n")
        if self.ref_text:
            values = "; ".join(f"{need} -> {value}" for need, value in self.ref_text.items())
            parts.append(f"[values read from the pages requirements point at] {values}\n")
        return "".join(parts)

    def _rd_trail_round(self, pending):
        """One look at the chains: asks the model and records what it read. Returns (links to open, links unfound)."""
        def render(listing_id):
            return self._rd_tagged(self._rd_record(listing_id) if listing_id in self.opened else None, listing_id, (300, 1400), (80, 300))

        got = self._rd_ask([CHAIN_PROMPT, "REQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n" + "\n".join(pending),
                            "RECORDS (newest first):\n" + "\n".join(self._rd_newest(60, render))], [WALKER_MODEL, JUDGE_MODEL], 1800) or {}
        self._rd_note("trail", pending=pending, got=_trim_value(got, 150))
        return self._rd_trail_chains(got, pending)

    def follow_chains(self) -> tuple[str, bool]:
        """Ahead of a checkout, every requirement whose value lives on another listing needs each page of its chain
        opened. Kyo opens the links it can name; a link still unfound holds the checkout back, at most twice.
        Gives back the text of what was seen and whether the checkout should wait."""
        live = {need for _, need in self.needs_in_force()}
        pending = [need for _, need in self.ref_needs if need not in self.ref_text and need in live]
        # links are pages: the offer view when it stands apart from the attributes view, else the page view
        tool = self.offer_viewer()
        target = self.id_param(tool)
        seen, lost = [], []
        while pending and self.spend_usd < HARD_BUDGET_USD and self.chain_cache < 4:
            self.chain_cache += 1
            reach, lost = self._rd_trail_round(pending)
            pending = [p for p in pending if p not in self.ref_text]
            if self.is_over or not (reach and tool and target) or self.steps_left() < len(reach) + 4:
                break
            self._rd_trail_open(tool, target, reach[:max(0, min(3, 6 - len(seen)))], seen)
        text = self._rd_trail_text(seen)
        hold = bool(lost) and self.chain_holds < 2 and self.steps_left() > 5
        if hold:
            self.chain_holds += 1
            text += ("The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: "
                     + " | ".join(lost[:3]) + ". Search for it by its shop, brand or words, open it, apply the value its page gives, then check out.")
        return text, hold

    def offer_viewer(self) -> str | None:
        """Where a listing's live offer is re-read: the store's offer view if there is one, otherwise the listing view."""
        return self.jobs.get("offer") or self.jobs.get("open")

    @staticmethod
    def _rd_moved(before, after):
        """Plain fields present in both records whose value differs."""
        return [key for key, value in after.items()
                if not isinstance(value, (dict, list)) and key in before and before[key] != value]

    def _rd_reread_one(self, tool, target, listing_id):
        """Re-reads one line's offer: None when the call could not be made, else a note of what changed ("" for nothing)."""
        earlier = dict(self.rows.get(listing_id) or {})
        reply = self._rd_call(tool, {target: listing_id})[1]
        if reply is None:
            return None
        self.offer_read_at[listing_id] = self.steps_used
        seen = self._rd_take(tool, reply)
        # diff the stored record so a re-read wrapped in a note still reveals what changed
        moved = self._rd_moved(earlier, self.rows.get(listing_id) or {})
        if not moved:
            return ""
        return f"{listing_id} changed ({', '.join(moved[:4])}): {self._rd_snip(seen, 300, 900)}"

    def recheck_offers(self) -> str:
        """Right before checkout, re-read every cart line through the offer view, since stock and price may shift silently."""
        tool = self.offer_viewer()
        target = self.id_param(tool)
        if not (tool and target):
            return ""
        since = max(self.steps_used - 2, self.refusal_step)
        stale = list(filter(lambda i: i not in self.sold_at and self.offer_read_at.get(i, -99) < since, self.cart_listing_ids()))
        if not stale or len(stale) + 3 > self.steps_left():
            return ""
        changes = []
        for listing_id in stale:
            change = self._rd_reread_one(tool, target, listing_id)
            if change is None:
                break
            if change:
                changes.append(change)
        self._rd_note("reread", ids=stale, changed=len(changes))
        if changes:
            self.epoch += 1
        return "\n".join(changes)

    def is_gone(self, listing_id: str) -> bool:
        """Gone by a note or a refusal that named the listing, or by its own stock flag turning from true to false."""
        return listing_id in self.sold_at or listing_id in self.unbuyable

    def _rd_swap_advice(self, picks):
        if not picks:
            return " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."
        first, names = picks[0], ", ".join(picks)
        if self.steps_left() <= 6:
            rest = ", ".join(picks[1:]) or "none"
            return f" Few steps are left: in this turn add {first} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {rest}."
        if self.plan_brief.get("pick") == "cheapest":
            return f" Listings already read that appear to fit that item instead, cheapest first: {names}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {first}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        return f" Listings already read that appear to fit that item instead: {names}. In this turn take the sold-out line out of the cart and add {first} at the requested quantity, then check out."

    def swap_note(self) -> str:
        """For cart lines now recorded as sold out (each named once), suggest replacements among listings already read."""
        lost = [listing_id for listing_id in self.cart_listing_ids() if self.is_gone(listing_id) and listing_id not in self.sold_at]
        if not lost:
            return ""
        self.sold_at.update(dict.fromkeys(lost, self.steps_used))
        self.review_leads = [lead for lead in self.review_leads if lead not in lost]
        self.epoch += 1
        picks = self.find_swaps(lost)
        head = f"[Kyo] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        return head + self._rd_swap_advice(picks)

    def locate_sold(self) -> str:
        """A refusal reported a sold-out pick without naming it. With one cart line that line is the one; with several,
        the offer view re-reads them so the records tell which. Returns what that re-read showed."""
        lines = [listing_id for listing_id in self.cart_listing_ids() if listing_id not in self.sold_at]
        if not lines or any(map(self.is_gone, lines)):
            return ""
        if len(lines) > 1:
            return self.recheck_offers()
        self.unbuyable.add(lines[0])
        return ""

    @staticmethod
    def _rd_fit_price(entry):
        """(listing id, price or None) from one entry of the model's fit list."""
        if isinstance(entry, dict):
            return str(entry.get("id")), entry.get("price")
        return str(entry), None

    def find_swaps(self, lost: list[str]) -> list[str]:
        if self.spend_usd > SOFT_BUDGET_USD:
            return []

        def render(listing_id):
            if listing_id in lost or listing_id in self.sold_at or self.is_gone(listing_id):
                return None
            return self._rd_snip(self._rd_record(listing_id), 200, 1200)

        shown = self._rd_newest(25, render)
        if not shown:
            return []
        sections = self._rd_preamble(SOLD_SWAP_PROMPT)
        sections.append("SOLD-OUT CART LINE:\n" + "\n".join(f"{listing_id}: {self.record_text(listing_id)}" for listing_id in lost))
        sections.append("CART NOW:\n" + self.cart_summary())
        sections.append("OTHER LISTING RECORDS SEEN (newest first):\n" + "\n".join(shown))
        got = self._rd_ask(sections, [JUDGE_MODEL, WALKER_MODEL], 1200) or {}
        excluded = set(self.cart_listing_ids())
        fits = []
        for entry in _as_list(got.get("fits")):
            listing_id, price = self._rd_fit_price(entry)
            if listing_id not in self.rows or listing_id in lost or listing_id in self.sold_at or listing_id in excluded:
                continue
            excluded.add(listing_id)
            numeric = isinstance(price, (int, float)) and not isinstance(price, bool)
            fits.append((float(price) if numeric else float("inf"), listing_id))
        # the model picks fitting listings well but sorts them by price poorly, so sort here
        ordered = sorted(fits, key=lambda fit: fit[0]) if self.plan_brief.get("pick") == "cheapest" else fits
        self._rd_note("swap", lost=lost, fits=fits)
        return [listing_id for _, listing_id in ordered][:4]

    def _rd_audit_round(self):
        got = self.review_call()
        return got, self.review_fits(got)

    def review_decline(self, force: bool = False) -> str:
        if self.reviewed or (self.spend_usd > SOFT_BUDGET_USD and not force) or self.spend_usd > HARD_BUDGET_USD:
            return ""
        self.reviewed = True
        got, named = self._rd_audit_round()
        if not named:
            # listings seen only as rows may fit as well: open the likely ones (one step) and review again
            candidates = [str(e) for e in _as_list((got or {}).get("open"))]
            candidates = [i for i in candidates if i in self.rows and i not in self.opened and not self.is_gone(i)]
            if not candidates or not self.open_listed(candidates):
                return ""
            got, named = self._rd_audit_round()
            if not named:
                return ""
        self.review_items = named
        self.review_leads = [listing_id for group in named.values() for listing_id in group]
        listed = "; ".join(f"item {number}: {', '.join(group)}" for number, group in named.items())
        reason = got.get("reason")
        return ("A careful review thinks these listings may fit: " + listed + (f" ({reason})" if reason else "")
                + ". Open them and buy what truly fits; end without a purchase only if each one fails.")

    def review_call(self) -> dict[str, Any] | None:
        def render(listing_id):
            if self.is_gone(listing_id):
                return None
            return self._rd_tagged(self.opened_records.get(listing_id), listing_id, (200, 1200), (120, 500))

        shown = self._rd_newest(30, render)
        if not shown:
            return None
        sections = self._rd_preamble(DECLINE_REVIEW_PROMPT)
        sections.append("LISTING RECORDS SEEN (newest first):\n" + "\n".join(shown))
        got = self._rd_ask(sections, [JUDGE_MODEL, WALKER_MODEL], 1500)
        self._rd_note("audit", got=got)
        return got

    def review_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {}
        for number, entries in fits.items():
            usable = list(filter(lambda i: i in self.rows and not self.is_gone(i), map(str, _as_list(entries))))
            named[str(number)] = usable[:4]
        return dict(pair for pair in named.items() if pair[1])

    def _rd_open_plan(self, rows):
        joiner = self.jobs.get("compare")
        single = self.attribute_viewer()
        many = self.side_list_param()
        key = self.id_param(single)
        if joiner and many and not self.side_view_bare():
            return [(joiner, {many: rows[: self.side_capacity()]})]
        if not single or not key:
            return None
        return [(single, {key: listing_id}) for listing_id in rows[:2]]

    def open_listed(self, rows: list[str]) -> bool:
        """Open listings known only from search rows: all at once in the side-by-side view if possible, else the first two singly."""
        plan = self._rd_open_plan(rows)
        if plan is None or self.steps_left() < len(plan) + 5:
            return False
        count = len(self.opened_records)
        for name, raw in plan:
            args, reply = self._rd_call(name, raw)
            if args is None:
                return False
            if reply is None:
                break
            self._rd_take(name, reply)
        self._rd_note("audit_open", rows=rows, opened=len(self.opened_records) - count)
        return count < len(self.opened_records)

    @staticmethod
    def _rd_first(spec, wanted, plain=False):
        """First property whose schema allows the `wanted` type (and, with `plain`, carries no enum)."""
        return next((key for key, sub in spec.items()
                     if isinstance(sub, dict) and wanted in _schema_types(sub) and not (plain and "enum" in sub)), None)

    def _rd_array_calls(self, props, lines):
        """A cart tool taking a list of line objects: one call carrying every line. (False, None) when the tool has no such list."""
        for key, spec in props.items():
            if not isinstance(spec, dict) or "array" not in _schema_types(spec):
                continue
            shape = spec.get("items")
            if not isinstance(shape, dict) or not isinstance(shape.get("properties"), dict):
                continue
            id_field = self._rd_first(shape["properties"], "string", True)
            count_key = self._rd_first(shape["properties"], "integer")
            if not id_field:
                return True, None
            entries = []
            for listing_id, units in lines.items():
                entry = {id_field: listing_id}
                if count_key:
                    entry[count_key] = units
                entries.append(entry)
            return True, [{key: entries}]
        return False, None

    def _rd_cart_target(self, name, schema, props):
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _schema_types(s) and "enum" not in s]
        # first usable of: a learned id argument of this tool, a required text field, the only text field
        choices = (next((key for tool, key in self.id_params if tool == name and key in texts), None),
                   next((key for key in schema.get("required") or [] if key in texts), None),
                   texts[0] if len(texts) == 1 else None)
        return next(filter(None, choices), None)

    @staticmethod
    def _rd_mode_key(props):
        """A property choosing between at least two actions (add / remove)."""
        for key, spec in props.items():
            if isinstance(spec, dict) and isinstance(spec.get("enum"), list) and len(spec["enum"]) >= 2:
                return key
        return None

    def _rd_removal(self, props, target, count, mode, sets, listing_id):
        """The call that takes `ident` off the cart, or None when the schema offers no way."""
        have = self.cart_qty.get(listing_id, 1)
        call = {target: listing_id}
        if mode:
            call[mode] = props[mode]["enum"][1]
            if count:
                call[count] = have
            return call
        if not count:
            return None
        if sets:
            call[count] = 0
            return call
        floor = props[count].get("minimum")
        if isinstance(floor, int) and floor <= -have:
            call[count] = -have
            return call
        return None

    def cart_plan(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """Cart-tool arguments, worked out from its schema, that leave the cart holding `lines` and nothing of `drop`;
        None when the schema does not tell how."""
        name = self.jobs.get("cart") or ""
        schema = self.tool_schemas.get(name) or {}
        props = schema.get("properties") or {}
        listed, result = self._rd_array_calls(props, lines)
        if listed:
            return result
        target = self._rd_cart_target(name, schema, props)
        if not target:
            return None
        count, mode, sets = self._rd_first(props, "integer"), self._rd_mode_key(props), self.cart_sets_count(name)
        model_calls = []
        for listing_id in drop:
            call = self._rd_removal(props, target, count, mode, sets, listing_id)
            if call is None:
                return None
            model_calls.append(call)
        carted = self.cart_listing_ids()
        for listing_id, want in lines.items():
            have = self.cart_qty.get(listing_id, 0) if listing_id in carted else 0
            if want == have:
                continue
            if have and not sets:
                return None
            call = {target: listing_id}
            if mode:
                call[mode] = props[mode]["enum"][0]
            if count:
                call[count] = want
            model_calls.append(call)
        return model_calls

    def _rd_kind(self, cat_key, listing_id):
        if not cat_key:
            return listing_id
        return str((self.rows.get(listing_id) or {}).get(cat_key) or "").strip().lower()

    def _rd_reviewed_plan(self, cat_key):
        """One reviewed listing per item (none sharing a category), at that item's quantity."""
        plan, kinds = {}, set()
        for number, ids in self.review_items.items():
            index = _as_int(number)
            if not index or not 1 <= index <= len(self.items):
                continue
            pick = next(filter(lambda i: not self.is_gone(i), ids), None)
            if not pick:
                continue
            kind = self._rd_kind(cat_key, pick)
            if kind not in kinds:
                plan[pick] = self.items[index - 1].count_wanted
                kinds.add(kind)
        return plan

    def _rd_apply_calls(self, cart, model_calls):
        """Runs the cart calls in order; None as soon as one cannot be made or the episode ends."""
        done = []
        for raw in model_calls:
            args, reply = self._rd_call(cart, raw)
            if reply is None:
                return None
            self.record_cart(cart, args)
            self._rd_take(cart, reply)
            done.append(_to_json(args))
        return done

    def _rd_verdict(self, lines):
        """Checks the planned lines (up to the hard spend cap) and gives the check's verdict on them."""
        self.judge_cart(list(lines), HARD_BUDGET_USD)
        return self.cart_verdicts.get(self.verdict_key(list(lines)))

    def _rd_retry_ready(self, lines):
        """Before the second checkout try: re-read the lines and check them afresh."""
        self.recheck_offers()
        passed = self._rd_verdict(lines)
        if self.is_over or self.steps_left() < 1:
            return False
        return bool(passed) and not any(map(self.is_gone, lines))

    def _rd_buy_once(self, buy, args, lines, drop, replies):
        """One checkout attempt, its replies added to `replies`. True when the store refused it and a retry may follow."""
        go, said = self.step_gate("order")
        if said:
            replies.append(said)
        if not go:
            return False
        reply = self.store_call(buy, args)
        self._rd_note("completed", lines=lines, drop=drop, reply=_trim_value(reply, 200))
        if reply is None:
            return False
        self.checkouts += 1
        ended = reply.get("finished")
        if not ended:
            self.refusal_step = self.steps_used
        replies.append(self.render_reply(self._rd_take(buy, reply)))
        return not (ended or self.is_over)

    def _rd_split_cart(self, plan, cat_key):
        """Current cart lines outside the plan: kept (other items, still buyable) or dropped."""
        planned_kinds = {self._rd_kind(cat_key, p) for p in plan}
        keep, drop = {}, []
        for listing_id in self.cart_listing_ids():
            if listing_id in plan:
                continue
            if self.is_gone(listing_id) or self._rd_kind(cat_key, listing_id) in planned_kinds:
                drop.append(listing_id)
            else:
                keep[listing_id] = self.cart_qty.get(listing_id, 1)
        return keep, drop

    def finish_reviewed_cart(self) -> str:
        """The agent is about to end without buying though the independent check confirmed a reviewed listing: Kyo
        carts the reviewed lines (one per item, keeping other items' lines) and checks out once a fresh check passes."""
        cart, buy = self.jobs.get("cart"), self.jobs.get("buy")
        if not (cart and buy) or self.cart_completed or not self.confirmed_id or self.is_gone(self.confirmed_id):
            return ""
        self.cart_completed = True
        cat_key = self.shelf_field()
        plan = self._rd_reviewed_plan(cat_key)
        if self.confirmed_id not in plan:
            return ""
        keep, drop = self._rd_split_cart(plan, cat_key)
        if len(plan) + len({self._rd_kind(cat_key, i) for i in keep}) < len(self.items):
            return ""
        lines = dict(keep)
        lines.update(plan)
        model_calls = self.cart_plan(lines, drop)
        if not model_calls or self.steps_left() < len(model_calls) + 1:
            return ""
        # carting needs no model call; an unchecked cart gets one check, allowed up to the hard spend cap
        self.read_attributes(list(lines), len(model_calls) + 3)
        if not self._rd_verdict(lines):
            return ""
        done = self._rd_apply_calls(cart, model_calls)
        if done is None:
            return ""
        args = self._rd_fit(buy, dict(self.plan_brief.get("buy_args") or {}))
        if args is None:
            return ""
        # a refused checkout (price, budget or stock moved) is retried once, after a re-read and a new check
        replies = []
        if self._rd_buy_once(buy, args, lines, drop, replies) and self._rd_retry_ready(lines):
            self._rd_buy_once(buy, args, lines, drop, replies)
        if not replies:
            return ""
        return ("Not run (Kyo acted instead): the independent check confirmed " + self.confirmed_id + ", so Kyo set the cart ("
                + "; ".join(done) + ") and checked out. Store reply: " + " | then: ".join(replies))

    # ---------- tool use ----------

    def _re_role_call(self, role: str, name: str, args: dict[str, Any]) -> bool:
        """Whether `name` is the tool that plays `role`, called with every argument the brief fixes for that role."""
        if self.jobs.get(role) != name:
            return False
        pinned = self.plan_brief.get(role + "_args")
        if not isinstance(pinned, dict):
            return True
        for key, wanted in pinned.items():
            if args.get(key) != wanted:
                return False
        return True

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        return self._re_role_call("buy", name, args)

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        return self._re_role_call("refuse", name, args)

    def _re_endgame_tools(self) -> set[Any] | None:
        """The only tools worth a step when few are left (None: no narrowing). The closer the end, the shorter the list."""
        finishers = {self.jobs.get("buy"), self.jobs.get("refuse")}
        finishers.discard(None)
        if not finishers:
            return None
        steps_left, step_reserve = self.steps_left(), self.reserve_steps()
        if steps_left <= step_reserve:
            return finishers
        cart_tools = [self.jobs.get("cart"), self.jobs.get("cart_view")]
        if steps_left <= step_reserve + 3:
            return finishers.union(cart_tools)
        if self.checkouts == 0 and steps_left <= len(self.items) + 5:
            cart_tools.append(self.jobs.get("offer"))
            return finishers.union(cart_tools)
        return None

    def tool_specs(self) -> list[dict[str, Any]]:
        interviewer = self.jobs.get("ask")
        tools: list[dict[str, Any]] = []
        for tool in self.tools:
            if tool["function"]["name"] != interviewer:
                tools.append(tool)
        allowed = self._re_endgame_tools()
        if allowed is None:
            return tools
        narrowed = [tool for tool in tools if tool["function"]["name"] in allowed]
        return narrowed if narrowed else tools

    @staticmethod
    def _re_free_text(spec: Any) -> bool:
        return isinstance(spec, dict) and "enum" not in spec and "string" in _schema_types(spec)

    def quiet_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fill the ending call's free-text fields with a neutral decline, so no listing is named while declining."""
        schema = self.tool_schemas.get(name) or {}
        needed = schema.get("required") or []
        always = self.plan_brief.get("quiet_decline") is not False
        result = dict(args)
        for key, spec in (schema.get("properties") or {}).items():
            if not self._re_free_text(spec):
                continue
            current = result.get(key)
            if current is None and key not in needed:
                continue
            said = str(current or "")
            if always or not said or any(listing_id in said for listing_id in self.rows):
                result[key] = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."[: spec.get("maxLength", 200)]
        return result

    def _re_row_starts(self, row: dict[str, Any], prefix: str) -> bool:
        for field, value in row.items():
            if field == self.id_field:
                continue
            if isinstance(value, str) and value.strip().lower().startswith(prefix):
                return True
        return False

    def _re_listing_id(self, value: Any) -> tuple[Any, str]:
        """Map a title-like value onto the one seen listing whose text starts with it; an id or short text passes as is."""
        if not isinstance(value, str):
            return value, ""
        bare = value.strip()
        if bare in self.rows:
            return bare, ""
        if _looks_like_id(value, True) or len(bare) < 8:
            return value, ""
        prefix = bare.lower()
        hits: list[str] = []
        for listing_id, row in self.rows.items():
            if self._re_row_starts(row, prefix):
                hits.append(listing_id)
        if not hits:
            return value, f"{value!r} is not a listing id; pass the id from the result rows (their {self.id_field} field)"
        if len(hits) > 1:
            return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
        return hits[0], ""

    def _re_id_argument(self, name: str) -> str | None:
        """The single free argument of a listing tool, when that argument is known to carry listing ids."""
        listing_tools = set()
        for role in ("cart", "open", "offer", "compare"):
            tool = self.jobs.get(role)
            if tool is not None:
                listing_tools.add(tool)
        if not self.id_field or name not in listing_tools:
            return None
        props = (self.tool_schemas.get(name) or {}).get("properties") or {}
        open_props: list[str] = []
        for key, spec in props.items():
            if not isinstance(spec, dict) or "enum" in spec:
                continue
            kinds = _schema_types(spec)
            if "string" in kinds or "array" in kinds:
                open_props.append(key)
        if len(open_props) != 1:
            return None
        # the cart always takes ids; any other tool only once a call of it was seen carrying one there
        if name == self.jobs.get("cart") or (name, open_props[0]) in self.id_params:
            return open_props[0]
        return None

    def fix_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Swap a listing title for its id where the tool expects ids, or explain the problem without spending a step."""
        key = self._re_id_argument(name)
        if key is None:
            return args, ""
        if key not in args:
            return dict(args), ""
        given = args[key]
        if isinstance(given, list):
            mapped = []
            for item in given:
                value, problem = self._re_listing_id(item)
                if problem:
                    return args, problem
                mapped.append(value)
            return {**args, key: mapped}, ""
        value, problem = self._re_listing_id(given)
        if problem:
            return args, problem
        return {**args, key: value}, ""

    def cart_sets_count(self, name: str) -> bool:
        """True for a cart tool that sets a line's count (0 drops the line): a count field allowing 0 and no mode field."""
        props = self.tool_schemas.get(name, {}).get("properties") or {}
        specs = [spec for spec in props.values() if isinstance(spec, dict)]
        for spec in specs:
            modes = spec.get("enum")
            if isinstance(modes, list) and len(modes) >= 2:
                return False
        for spec in specs:
            if spec.get("minimum") == 0 and "integer" in _schema_types(spec):
                return True
        return False

    @staticmethod
    def _re_first_count(values: Any, default: int | None) -> int | None:
        for value in values:
            if isinstance(value, str):
                continue
            count = _as_int(value)
            if count is not None:
                return count
        return default

    @staticmethod
    def _re_removes(spec: dict[str, Any], args: dict[str, Any]) -> bool:
        # a mode field set to anything but its first option takes lines off
        flags = []
        for key, value in args.items():
            modes = (spec.get(key) or {}).get("enum")
            flags.append(isinstance(modes, list) and len(modes) >= 2 and value in modes[1:])
        return any(flags)

    def _re_named_lines(self, spec: dict[str, Any], args: dict[str, Any]) -> tuple[list[tuple[str, int | None]], bool]:
        """(listing, count) pairs a cart call names, and whether it lays out the whole cart (a list of line objects)."""
        loose = self._re_first_count(args.values(), None)
        found: list[tuple[str, int | None]] = []
        complete = False
        for key, value in args.items():
            if isinstance(value, str):
                if value in self.rows:
                    found.append((value, loose))
                continue
            if not isinstance(value, list):
                continue
            shape = (spec.get(key) or {}).get("items") or {}
            if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                complete = True
            for entry in value:
                if isinstance(entry, str):
                    if entry in self.rows:
                        found.append((entry, loose))
                elif isinstance(entry, dict):
                    complete = True
                    listing_id = next((v for v in entry.values() if isinstance(v, str) and v in self.rows), None)
                    if listing_id:
                        found.append((listing_id, self._re_first_count(entry.values(), 1)))
        return found, complete

    @staticmethod
    def _re_line_count(have: int, count: int | None, removal: bool, sets: bool) -> int:
        if removal:
            return have - count if count else 0
        units = 1 if count is None else count
        return units if sets else have + units

    def record_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_used = True
        self.cart_changed_at = self.steps_used
        spec = self.tool_schemas.get(name, {}).get("properties") or {}
        removal = self._re_removes(spec, args)
        lines, complete = self._re_named_lines(spec, args)
        if complete:
            fresh: dict[str, int] = {}
            for listing_id, count in lines:
                if count and count > 0:
                    fresh[listing_id] = count
            self.cart_qty = fresh
            return
        sets = self.cart_sets_count(name)
        for listing_id, count in lines:
            have = self._re_line_count(self.cart_qty.get(listing_id, 0), count, removal, sets)
            if have <= 0:
                self.cart_qty.pop(listing_id, None)
            else:
                self.cart_qty[listing_id] = have

    def render_reply(self, observation: Any) -> str:
        text = _to_json(_trim_value(observation, PREVIEW_CHARS))
        if len(text) <= REPLY_CHAR_CAP:
            return text
        return text[:REPLY_CHAR_CAP] + "...(cut)"

    def _re_compare_schema(self) -> dict[str, Any]:
        return self.tool_schemas.get(self.jobs.get("compare") or "") or {}

    def side_list_param(self) -> str | None:
        schema = self._re_compare_schema()
        target = None
        for key, spec in (schema.get("properties") or {}).items():
            if isinstance(spec, dict) and "array" in _schema_types(spec):
                target = key
                break
        if not target:
            return None
        for key in schema.get("required") or []:
            if key != target:
                return None
        return target

    def side_capacity(self) -> int:
        """Listings one side-by-side call can take: the schema's cap if it states one, else what the store proved to handle."""
        props = self._re_compare_schema().get("properties") or {}
        spec = props.get(self.side_list_param() or "") or {}
        cap = spec.get("maxItems")
        if isinstance(cap, int):
            return cap
        if not self.side_fail_size:
            return max(4, self.side_ok_size)
        return max(2, min(self.side_fail_size - 1, max(self.side_ok_size, 2)))

    def _re_lone_open(self, call: dict[str, Any], opener: str, target: str) -> str | None:
        """The listing a call opens, when it is a bare open of one seen listing that is still on sale."""
        fn = call.get("function") or {}
        raw = fn.get("arguments")
        parsed = _parse_object(raw) if isinstance(raw, str) else None
        if fn.get("name") != opener or not parsed or set(parsed) != {target}:
            return None
        listing_id = str(parsed.get(target) or "").strip()
        if listing_id not in self.rows or listing_id in self.sold_at:
            return None
        return listing_id

    def merge_opens(self, model_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Merge a turn's single-listing opens into one side-by-side call, spending one step where they would spend several."""
        opener = self.jobs.get("open")
        joiner = self.jobs.get("compare")
        target = self.id_param(opener)
        many = self.side_list_param()
        if not (opener and joiner and target and many) or opener == joiner:
            return model_calls
        # the merged view must show what the opens would have shown (the attributes)
        if self.side_view_bare():
            return model_calls
        picks: list[tuple[int, str]] = []
        for position, call in enumerate(model_calls):
            listing_id = self._re_lone_open(call, opener, target)
            if listing_id is not None:
                picks.append((position, listing_id))
        picks = picks[: self.side_capacity()]
        if len(picks) < 2:
            return model_calls
        idents = [listing_id for _, listing_id in picks]
        merged = {position for position, _ in picks}
        lead = picks[0][0]
        joined: list[dict[str, Any]] = []
        for position, call in enumerate(model_calls):
            if position == lead:
                joined.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _to_json({many: idents})}})
            elif position in merged:
                joined.append({"id": call["id"], "joined": True})
            else:
                joined.append(call)
        self.log.append({"kind": "joined", "ids": list(idents)})
        return joined

    # ---------- one tool call: guards, the store, then bookkeeping ----------
    def execute(self, model_calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        batch = self.merge_opens(model_calls)
        answers: list[tuple[str, str]] = []
        for call in batch:
            answers.append((call["id"], self._re_handle(call, len(batch))))
        return answers

    def _re_handle(self, call: dict[str, Any], batch_size: int) -> str:
        """Run one tool call of the model's turn and return the text it gets back. Each guard that answers
        instead of the store costs no step; the guards run in a fixed order."""
        allowed = {tool["function"]["name"] for tool in self.tool_specs()}
        if call.get("joined"):
            return "Shown in the side-by-side result above: Kyo opened these listings together in one step."
        fn = call.get("function") or {}
        name = fn.get("name")
        stop = self._re_tool_refusal(name, allowed)
        if stop:
            return stop
        args, stop = self._re_call_args(name, fn.get("arguments"))
        if stop:
            return stop
        many = self.side_list_param() if name == self.jobs.get("compare") else None
        args, note_texts = self._re_cap_compare(args, many)
        target = self.id_param(name)
        stop = self._re_sold_target(name, args, target)
        if stop:
            return stop
        stop = self._re_nothing_to_remove(name, args)
        if stop:
            return stop
        ending = self.is_checkout(name, args) or self.is_decline(name, args)
        stop = self._re_finish_timing(ending, batch_size)
        if stop:
            return stop
        mark = name + _to_json(args)
        readers = {self.jobs.get(role) for role in ("search", "open", "offer", "compare")} - {None}
        stop = self._re_repeat_refusal(name, args, mark, ending, readers, target)
        if stop:
            return stop
        if many and isinstance(args.get(many), list) and args[many]:
            args, extra, stop = self._re_skip_shown(name, args, many)
            if stop:
                return stop
            if extra:
                note_texts += extra
                mark = name + _to_json(args)
        if ending and self.steps_left() > 2:
            stop = self._re_checkout_hold() if self.is_checkout(name, args) else self._re_decline_review()
            if stop:
                return stop
        if self.is_decline(name, args):
            stop = self._re_late_decline_hold() or self._re_search_push()
            if stop:
                return stop
            done = self.finish_reviewed_cart()
            if done:
                return done + (" [episode finished]" if self.is_over else "")
            args = self.quiet_args(name, args)
        if ending and self.steps_left() >= 2:
            go, said = self.step_gate("order" if self.is_checkout(name, args) else "end")
            if not go:
                return self._re_with_cart("Not run yet. " + said)
            if said:
                note_texts = "\n[Kyo] " + said + note_texts
        reply = self.store_call(name, args)
        if reply is None:
            return "The call failed (network or server). You may try again."
        return self._re_after_store(name, args, reply, many, mark, readers, ending, note_texts)

    @staticmethod
    def _re_skipped(reason: str) -> str:
        # a guard answered instead of the store: no step was spent
        return "Not run (no step used): " + reason

    @staticmethod
    def _re_held(reason: str) -> str:
        # a finishing call held back for now, without spending a step
        return "Not run yet (no step used). " + reason

    def _re_with_cart(self, text: str) -> str:
        return text + "\nCart now: " + self.cart_summary()

    @staticmethod
    def _re_tool_names(allowed: set[str]) -> str:
        return ", ".join(sorted(allowed))

    def _re_tool_refusal(self, name: Any, allowed: set[str]) -> str | None:
        if self.is_over:
            return "Not run: the episode is over."
        if name == self.jobs.get("ask"):
            return self._re_skipped("Kyo's interviewer handles every message to the shopper.")
        if name not in self.tool_schemas:
            return self._re_skipped(f"unknown tool. Tools: {self._re_tool_names(allowed)}.")
        if name not in allowed:
            return self._re_skipped("too few steps are left for that; use: " + self._re_tool_names(allowed) + ".")
        return None

    def _re_call_args(self, name: str, raw: Any) -> tuple[Any, str]:
        """The call's arguments parsed, fitted to the tool's schema and with titles turned into ids; or why that failed."""
        if isinstance(raw, dict):
            parsed = raw
        elif (raw or "").strip():
            parsed = _parse_object(raw)
            if parsed is None:
                return None, self._re_skipped("the arguments were not one JSON object.")
        else:
            parsed = {}
        faults: list[str] = []
        shaped = _conform(parsed, self.tool_schemas[name], "args", faults)
        if faults:
            return None, self._re_skipped("; ".join(faults[:6]))
        fixed, wrong = self.fix_ids(name, shaped)
        if wrong:
            return None, self._re_skipped(wrong + ".")
        return fixed, ""

    def _re_cap_compare(self, args: dict[str, Any], many: str | None) -> tuple[dict[str, Any], str]:
        """Cut a side-by-side call down to what the store showed it can take, with a note naming the rest."""
        if not many or not self.side_fail_size:
            return args, ""
        sent = args.get(many)
        if not isinstance(sent, list):
            return args, ""
        room = self.side_capacity()
        if len(sent) <= room:
            return args, ""
        rest = ", ".join(map(str, sent[room:]))
        note = f"\n[Kyo] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {rest}."
        return {**args, many: sent[:room]}, note

    def _re_sold_target(self, name: str, args: dict[str, Any], target: str | None) -> str | None:
        if not target or name not in (self.jobs.get("open"), self.jobs.get("offer")):
            return None
        key = str(args.get(target))
        if key not in self.sold_at:
            return None
        return self._re_skipped(f"{args[target]} sold out at step {self.sold_at[key]} and will not come back. Replace its cart line with another fitting listing.")

    def _re_nothing_to_remove(self, name: str, args: dict[str, Any]) -> str | None:
        """A negative cart count for a listing the cart does not hold has nothing to take off."""
        if name != self.jobs.get("cart"):
            return None
        negative = False
        for value in args.values():
            if not isinstance(value, str) and (_as_int(value) or 0) < 0:
                negative = True
                break
        if not negative:
            return None
        policy = _to_json(self.cart_snapshot) if self.cart_viewed_at >= self.cart_changed_at else ""
        for value in args.values():
            if not isinstance(value, str) or value not in self.rows:
                continue
            if value in self.cart_qty or value in policy or value in self.cart_reply:
                continue
            return self._re_skipped(f"{value} is not in the cart, so there is nothing to take off. Cart now: " + self.cart_summary())
        return None

    def _re_finish_timing(self, ending: bool, batch_size: int) -> str | None:
        if ending:
            if batch_size > 1:
                return self._re_skipped("send the finishing call on its own, after reading your other results.")
            return None
        if self.steps_left() <= self.reserve_steps() and (self.jobs.get("buy") or self.jobs.get("refuse")):
            return self._re_skipped("the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase.")
        return None

    def _re_repeat_refusal(self, name: str, args: dict[str, Any], mark: str, ending: bool, readers: set[str], target: str | None) -> str | None:
        """Answer a call whose result is already known: the same call just made, a cached read, or a fully shown listing."""
        if not ending and name != self.jobs.get("cart") and mark in self.recent_calls:
            # L1: the same call again and again changes nothing; the fifth repeat ends the loop
            self.repeat_skips += 1
            if self.repeat_skips >= 5:
                self.stuck = True
                self.log.append({"kind": "stuck", "call": mark[:120]})
            if self.repeat_skips >= 3:
                return self._re_skipped(f"you have sent this same call {self.repeat_skips} times and nothing changed; it will not change. "
                                        "Do something else now: cart the listing that fits and check out, search with other words, or end without a purchase.")
            return self._re_skipped("you just made this same call and nothing changed; its result is above.")
        self.repeat_skips = 0
        if name not in readers:
            return None
        cached = self.read_memo.get(mark)
        if cached and cached[0] == self.epoch:
            return self._re_skipped(f"the same call at step {cached[1]} returned this, and nothing has changed since:\n" + cached[2][:3000])
        if not target or name == self.jobs.get("compare"):
            return None
        if not self.already_shown_all(name, str(args.get(target))):
            return None
        return self._re_skipped(f"{args[target]} has shown everything this view gives since the last change. Its record: " + self.record_text(str(args[target])))

    def _re_skip_shown(self, name: str, args: dict[str, Any], many: str) -> tuple[dict[str, Any], str, str]:
        """Drop the listings of a side-by-side call that this view already showed in full: (args, note, refusal)."""
        listed = args[many]
        unseen = [listing_id for listing_id in listed if not self.already_shown_all(name, str(listing_id))]
        if not unseen:
            records = "\n".join(f"{listing_id}: {self.record_text(str(listing_id))}" for listing_id in listed)
            return args, "", self._re_skipped("these listings have shown everything this view gives since the last change. Their records:\n" + records[:4000])
        if len(unseen) == len(listed):
            return args, "", ""
        skipped = ", ".join(str(listing_id) for listing_id in listed if listing_id not in unseen)
        return {**args, many: unseen}, f"\n[Kyo] Left out {skipped}: shown in full already, nothing changed since.", ""

    def _re_look_at_cart(self) -> str:
        """Spend one step on the cart view when the cart changed since it was last seen; returns the change it shows."""
        viewer = self.jobs.get("cart_view")
        if not self.cart_used or self.cart_viewed_at >= self.cart_changed_at or not viewer or self.steps_left() <= 3:
            return ""
        faults: list[str] = []
        view_args = _conform({}, self.tool_schemas.get(viewer), "args", faults)
        if faults:
            return ""
        policy = self.store_call(viewer, view_args)
        if policy is None:
            return ""
        self.cart_snapshot = policy.get("observation")
        self.cart_viewed_at = self.steps_used
        return self.take_reply(viewer, self.cart_snapshot)

    def _re_checkout_hold(self) -> str | None:
        """Reasons to hold a checkout back while steps remain; None lets it go on."""
        wider = self.hold_for_wider()
        if wider:
            return self._re_with_cart(wider)
        reread = self.recheck_offers()
        change = self._re_look_at_cart()
        if change:
            self.pending_notes = []
            return self._re_held(self._re_with_cart("Looking at the cart showed a change: " + change) + "\nApply the change, fix the cart if needed, then check out.")
        swap = self.swap_note()
        # a sold-out line holds the checkout at most twice, so a wrong cart record cannot stall the episode
        checkout_holds = [] if self.checkout_holds >= 2 else [listing_id for listing_id in self.cart_listing_ids() if listing_id in self.sold_at]
        if swap or checkout_holds:
            if not swap:
                self.checkout_holds += 1
            reason = swap or f"{', '.join(checkout_holds)} in the cart is sold out: replace that line first."
            return self._re_held(self._re_with_cart(reason))
        chain, hold = self.follow_chains()
        if hold:
            return self._re_held(self._re_with_cart(chain))
        # the check weighs attributes: read the cart lines' own first (and a few cheaper rows' for a cheapest pick)
        read = self.read_attributes(self.cart_listing_ids(), self.reserve_steps() + 3) + self.read_cheaper()
        verdict = self.judge_cart()
        if verdict:
            seen = ""
            if read:
                seen = "Kyo read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n"
            return self._re_with_cart("Not run yet. " + chain + seen + verdict)
        if reread and self.spend_usd > SOFT_BUDGET_USD:
            return self._re_held("Kyo re-read the cart's listings just now:\n" + reread
                                 + "\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.")
        return None

    def _re_decline_review(self) -> str | None:
        """Reasons to hold an early ending without a purchase: the review's objections, then listings that seem to fit."""
        loosened = self._loosened_review()
        if loosened is not None:
            return loosened
        if self.steps_left() <= 5:
            return None
        review = self.review_decline()
        if review:
            return self._re_held(review)
        if not self.review_leads or self.steps_left() <= 4:
            return None
        if self.decline_holds == 0:
            self.decline_holds = 1
            self.log.append({"kind": "refuse_blocked", "leads": list(self.review_leads)})
            return self._re_held("The review found listings that appear to fit: " + ", ".join(self.review_leads)
                                 + ". Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.")
        if self.decline_holds != 1:
            return None
        self.decline_holds = 2
        first = self.review_leads[0]
        self.judge_cart([first], HARD_BUDGET_USD)
        if not self.cart_verdicts.get(self.verdict_key([first])) or first not in self.review_leads:
            return None
        self.confirmed_id = first
        self.log.append({"kind": "refuse_blocked", "confirmed": first})
        return self._re_held(f"An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement.")

    def _loosened_review(self) -> str | None:
        """Once per loosening: a note that dropped a requirement or raised the budget means the store now expects a
        purchase, so the review runs again on the requirements in force before an ending without a purchase."""
        if self.loosened <= self.loosen_seen or self.steps_left() <= 3:
            return None
        self.loosen_seen = self.loosened
        self.reviewed, self.review_leads, self.decline_holds = False, [], 0
        review = self.review_decline(force=True)
        self.log.append({"kind": "loosened_review", "found": bool(review)})
        if review:
            return self._re_held("The shopper loosened the request, so a purchase is now expected. " + review)
        if self.steps_left() <= 4:
            return None
        needs = "; ".join(f"item {n}: {m}" for n, m in self.needs_in_force())
        return self._re_held(f"The shopper loosened the request since you last searched (requirements in force: {needs}; budget {self.budget}), "
                             "which means the store now expects a purchase. Search again with only the requirements in force (short value-first "
                             "queries, the category browse), open the likely listings, cart the one that meets them and check out.")

    def _re_late_decline_hold(self) -> str | None:
        """Once, near the end: a decline that would throw away a cart already built for every item is turned back."""
        if self.steps_left() > 5 or self.late_hold:
            return None
        lines = [listing_id for listing_id in self.best_cart if not self.is_gone(listing_id)]
        fresh = self.cart_snapshot is not None and self.cart_viewed_at >= self.cart_changed_at
        missing = [listing_id for listing_id in lines if listing_id not in self.cart_listing_ids() or (listing_id in self.given_up and not fresh)]
        if len(lines) < len(self.items) or len(missing) + 1 > self.steps_left():
            return None
        self.late_hold = True
        self.log.append({"kind": "late_decline_held", "lines": lines, "missing": missing})
        if missing:
            redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out."
        else:
            redo = "Check out the cart as it is."
        return self._re_held("Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. "
                             + redo + " End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.")

    def _re_search_push(self) -> str | None:
        """Once, with many steps left and no leads: ask for a wider search before ending without a purchase."""
        if self.review_leads or self.search_pushed or self.steps_left() <= self.step_cap // 3:
            return None
        self.search_pushed = True
        return self._re_held(f"{self.steps_left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that.")

    def _re_learn_compare_room(self, args: dict[str, Any], many: str | None, observation: Any) -> str:
        """Learn from a side-by-side reply how many listings the view handles; returns a note when it showed none."""
        if not many:
            return ""
        sent = args.get(many)
        if not isinstance(sent, list) or len(sent) <= 1:
            return ""
        shown = _to_json(observation)
        for listing_id in sent:
            if str(listing_id) in shown:
                self.side_ok_size = max(self.side_ok_size, len(sent))
                return ""
        self.side_fail_size = min(self.side_fail_size or 999, len(sent))
        return f"\n[Kyo] The side-by-side view showed nothing for {len(sent)} listings: send at most {self.side_capacity()} per call."

    def _re_note_id_args(self, name: str, args: dict[str, Any]) -> None:
        # remember which arguments carried listing ids (each value, or every element of a list value, a seen listing)
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(item, str) and item in self.rows for item in values):
                self.id_params.add((name, key))

    def _re_other_view(self, name: str) -> str | None:
        """Another single-listing tool not yet seen to lack attributes, to stand in as the open view."""
        excluded = (self.jobs.get("cart"), self.jobs.get("buy"), self.jobs.get("refuse"), self.jobs.get("ask"), self.jobs.get("search"))
        for tool in self.tool_schemas:
            if tool == name or not self.id_param(tool):
                continue
            if self.shows_attrs.get(tool) is not False and tool not in excluded:
                return tool
        return None

    def _re_learn_views(self, name: str, observation: Any) -> None:
        """After a single-listing view: note whether it shows attributes and settle which tools serve as open and offer."""
        excluded = (self.jobs.get("cart"), self.jobs.get("buy"), self.jobs.get("refuse"), self.jobs.get("ask"))
        if not self.id_param(name) or name in excluded or not isinstance(observation, dict):
            return
        listing = self.listing_id(observation)
        if not listing:
            return
        self.shows_attrs[name] = self.shows_attrs.get(name, False) or bool(_attribute_block(observation))
        if name == self.jobs.get("open") and not self.shows_attrs[name]:
            other = self._re_other_view(name)
            if other:
                self.jobs["open"] = other
        if not self.shows_attrs[name] and name != self.jobs.get("open") and not self.jobs.get("offer"):
            self.jobs["offer"] = name
        if name == self.offer_viewer():
            self.offer_read_at[listing] = self.steps_used

    def _re_after_cart_call(self, name: str, args: dict[str, Any], observation: Any) -> None:
        self.record_cart(name, args)
        self.cart_reply = _to_json(observation)
        self.recent_calls = []
        # a zero or negative count drops the named lines (unless the reply still lists them); the fullest cart so far
        # is kept, to be offered back when a late decline would throw it away
        first = self._re_first_count(args.values(), None)
        if first is not None:
            for value in args.values():
                if not isinstance(value, str) or value not in self.rows:
                    continue
                if first > 0:
                    self.given_up.discard(value)
                elif not _mentions_id(observation, value):
                    self.given_up.add(value)
        lines = [listing_id for listing_id in self.cart_listing_ids() if not self.is_gone(listing_id) and listing_id not in self.given_up]
        if len(lines) >= len(self.best_cart):
            self.best_cart = lines

    def _re_zero_kept(self, args: dict[str, Any], observation: Any) -> bool:
        """A cart call with a count of 0 whose reply still lists the line: this cart tool adds counts."""
        zero = False
        for value in args.values():
            if not isinstance(value, str) and _as_int(value) == 0:
                zero = True
                break
        if not zero:
            return False
        for value in args.values():
            if isinstance(value, str) and value in self.rows and _mentions_id(observation, value):
                return True
        return False

    def _re_names_sold(self, observation: Any) -> bool:
        if self.note_said_sold:
            return True
        if not self.sold_phrases:
            return False
        said = self.words_of(observation)
        for line in self.sold_phrases:
            if line in said:
                return True
        return False

    def _re_after_store(self, name: str, args: dict[str, Any], reply: dict[str, Any], many: str | None, mark: str,
                        readers: set[str], ending: bool, note_texts: str) -> str:
        """Book what a store reply shows, then build the text the model reads for it."""
        observation = reply.get("observation")
        note_texts += self._re_learn_compare_room(args, many, observation)
        self.recent_calls = (self.recent_calls + [mark])[-3:]
        self._re_note_id_args(name, args)
        self.take_in(observation)
        if name == self.jobs.get("compare"):
            rows = self.listing_records(observation)
            if rows:
                self.shows_attrs[name] = self.shows_attrs.get(name, False) or any(_attribute_block(row) for _, row in rows)
        if name in readers - {self.jobs.get("search")}:
            self.mark_viewed(name, observation)
        # only search results come in pages: every other role's tool is plain
        plain = {tool for job, tool in self.jobs.items() if job != "search"}
        searching = name not in plain and not ending
        paged = self.read_more_pages(name, args, observation) if searching else ""
        self._re_learn_views(name, observation)
        if name == self.jobs.get("cart"):
            self._re_after_cart_call(name, args, observation)
        if name == self.jobs.get("cart_view"):
            self.cart_snapshot = observation
            self.cart_viewed_at = self.steps_used
        buying = self.is_checkout(name, args)
        is_over = reply.get("finished")
        if ending and not is_over:
            self.end_refused_at = self.steps_used
        if buying:
            self.checkouts += 1
            if not is_over:
                self.refusal_step = self.steps_used
        text = self.render_reply(observation) + paged + note_texts
        if searching:
            text += self.cheapest_note([observation] + self.extra_pages) + self._w3_cap_ignored(name, args, observation)
            if name == self.jobs.get("search") and self.listing_records(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
        base = text
        proof = self.evidence_lines(observation)
        if proof:
            text += "\n" + proof
        if name == self.jobs.get("cart") and self._re_zero_kept(args, observation):
            text += "\n[Kyo] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off."
        self.note_said_sold = False
        loose = self.note_lines(observation)
        if loose:
            meaning = self.apply_note(loose)
            if meaning:
                text += "\n[note read] " + meaning + ". Apply this change to your picks before checking out."
        if buying and not is_over and self._re_names_sold(observation):
            seen = self.locate_sold()
            if seen:
                text += "\n[Kyo re-read the cart's lines after the refusal]\n" + seen
        swap = self.swap_note()
        if swap:
            text += "\n" + swap
        if self.pending_notes:
            text += "\n[notes seen on Kyo's own calls] " + " | ".join(self.pending_notes)
            self.pending_notes = []
        if name in readers:
            self.read_memo[mark] = (self.epoch, self.steps_used, base)
        if is_over:
            text += " [episode finished]"
        return text

    # ---------- ending the episode ----------
    def _re_forced_checkout(self, buy: str, again: bool) -> str | None:
        """Check out the held cart with no turn left for the model. A string ends `finish` with it; None goes on to the decline."""
        faults: list[str] = []
        args = _conform(dict(self.plan_brief.get("buy_args") or {}), self.tool_schemas.get(buy), "args", faults)
        if faults:
            return None
        # with no steps left to act on it, a reply that adds a requirement does not stop this checkout
        if self.steps_left() >= 2 and self.seconds_used() < LATE_SECONDS:
            self.step_gate("order")
        if self.is_over or self.steps_left() <= 0:
            return ""
        reply = self.store_call(buy, args)
        self.log.append({"kind": "forced_buy", "reply": _trim_value(reply, 200)})
        if self.is_over or self.steps_left() <= 0:
            return ""
        if reply is None or not again:
            return None
        self.checkouts += 1
        self.refusal_step = self.steps_used
        self.take_reply(buy, reply.get("observation"))
        return "the checkout was refused: " + self.render_reply(reply.get("observation"))

    def _re_forced_decline(self, refuse: str) -> str:
        faults: list[str] = []
        args = _conform(self.quiet_args(refuse, dict(self.plan_brief.get("refuse_args") or {})), self.tool_schemas.get(refuse), "args", faults)
        if faults:
            return ""
        if self.steps_left() >= 2 and self.seconds_used() < LATE_SECONDS:
            self.step_gate("end")
        if self.is_over or self.steps_left() <= 0:
            return ""
        reply = self.store_call(refuse, args)
        self.log.append({"kind": "forced_refuse", "reply": _trim_value(reply, 200)})
        if reply is None or self.is_over:
            return ""
        self.take_reply(refuse, reply.get("observation"))
        return "ending without a purchase was refused: " + self.render_reply(reply.get("observation"))

    def wrap_up(self, again: bool = False) -> str:
        """Close the episode from code: check out a held cart, otherwise end without a purchase. With `again`, a refused
        checkout comes back (as the store's reply) for the model to handle instead of being followed by a decline."""
        if self.is_over or self.aborted or self.steps_left() <= 0:
            return ""
        buy = self.jobs.get("buy")
        refuse = self.jobs.get("refuse")
        fresh = self.cart_snapshot is not None and self.cart_viewed_at >= self.cart_changed_at
        holding = bool(self.cart_listing_ids()) if fresh else bool(self.cart_qty)
        if buy and (holding or (self.cart_used and not fresh and self.steps_left() >= 2)):
            outcome = self._re_forced_checkout(buy, again)
            if outcome is not None:
                return outcome
        if not refuse:
            return ""
        return self._re_forced_decline(refuse)

    def shrink_chat(self) -> None:
        """Keep the chat under the history limit by shortening the older half of the long tool results."""
        size = 0
        for message in self.dialog:
            size += len(_to_json(message.get("content")))
        if size <= CHAT_CHAR_CAP:
            return
        results = [position for position, message in enumerate(self.dialog) if message.get("role") == "tool"]
        for position in results[: max(1, len(results) // 2)]:
            content = self.dialog[position].get("content") or ""
            if len(content) <= 600:
                continue
            named: list[str] = []
            met: set[str] = set()
            for listing_id in re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content):
                if listing_id in met:
                    continue
                met.add(listing_id)
                if listing_id in self.rows:
                    named.append(listing_id)
                    if len(named) == 20:
                        break
            self.dialog[position]["content"] = content[:400] + f"...(older result shortened; listings in it: {', '.join(named)})"

    def run(self) -> list[dict[str, Any]]:
        if self.is_over:
            return self.log
        self.make_brief()
        self.log.append({"kind": "brief", "brief": self.plan_brief})
        opening = self.policy.get("opening")
        parts = ["SHOPPER'S REQUEST:\n", self.request_text]
        if opening is not None:
            self.take_in(opening)
            parts.append("\n\nWHAT THE STORE SHOWS FIRST:\n")
            parts.append(self.render_reply(opening))
        parts.append("\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n")
        parts.append(self.status_text())
        self.dialog = [{"role": "system", "content": WALKER_GUIDE}, {"role": "user", "content": "".join(parts)}]
        self.loop(TURN_CAP, HARD_BUDGET_USD)
        if not self.is_over:
            self._re_second_round()
        self.log.append({"kind": "end", "finished": self.is_over, "steps": self.steps_used, "cost": round(self.spend_usd, 5), "calls": self.model_calls})
        return self.log

    def _re_second_round(self) -> None:
        turned = self.wrap_up(again=True)
        # the store refused the ending (the shopper changed something, a line sold out): one short extra round handles it
        if turned and not self.is_over and not self.aborted and not self.out_of_credit and self.steps_left() > 1 and self.seconds_used() < LOOP_SECONDS:
            self.dialog.append({"role": "user", "content": "[Kyo] Kyo tried to end the episode and " + turned + "\nApply what the store says, then finish.\n" + self.status_text()})
            self.loop(self.turns + 10, HARD_BUDGET_USD + 0.01)
        if not self.is_over:
            self.wrap_up()

    # ---------- the model loop ----------
    def _re_may_continue(self, loop_cap: int, spend_cap: float) -> bool:
        if self.is_over or self.aborted or self.steps_left() <= 0 or self.stuck:
            return False
        # S1: a refused ending carries news the model must act on, so the spend cap leaves room for it
        if self.steps_used - self.end_refused_at <= 12:
            spend_cap += REFUSED_ROOM_USD
        return self.turns < loop_cap and self.spend_usd < spend_cap and self.seconds_used() < LOOP_SECONDS

    def _re_tool_calls(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """The message's tool calls in a uniform shape: string id, name, and arguments dumped as one JSON object."""
        model_calls: list[dict[str, Any]] = []
        for number, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if not isinstance(fn, dict):
                fn = {}
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                parsed = raw
            elif isinstance(raw, str) and raw.strip():
                parsed = _parse_object(raw)
            else:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            model_calls.append({"id": str(call.get("id") or f"k{self.turns}_{number}"), "type": "function",
                          "function": {"name": str(fn.get("name") or ""), "arguments": _to_json(parsed)}})
        return model_calls

    @staticmethod
    def _re_text_of(content: Any) -> str:
        if isinstance(content, str):
            return content
        return _to_json(content) if content else ""

    def _re_ask_first(self, model_calls: list[dict[str, Any]], content: str) -> bool:
        """Before the first commitment (cart, checkout, ending) the shopper is interviewed; the calls are not run."""
        if self.interviewed or not self.jobs.get("ask") or self.messages_left() <= 0:
            return False
        committing = {self.jobs.get("cart"), self.jobs.get("buy"), self.jobs.get("refuse")} - {None}
        if not any(call["function"]["name"] in committing for call in model_calls) or self.steps_left() <= 8:
            return False
        self.dialog.append({"role": "assistant", "content": content or None, "tool_calls": model_calls})
        for call in model_calls:
            self.dialog.append({"role": "tool", "tool_call_id": call["id"], "content": self._re_skipped("the shopper is asked first.")})
        summary = self.run_interview()
        self.dialog.append({"role": "user", "content": (summary or "The interviewer had nothing to ask.") + "\n" + self.status_text()})
        return True

    def _re_play(self, model_calls: list[dict[str, Any]], content: str) -> None:
        self.dialog.append({"role": "assistant", "content": content or None, "tool_calls": model_calls})
        self.log.append({"kind": "turn", "text": content[:300], "calls": [call["function"] for call in model_calls]})
        replies = self.execute(model_calls)
        if not replies:
            return
        for cid, text in replies[:-1]:
            self.dialog.append({"role": "tool", "tool_call_id": cid, "content": text})
        cid, text = replies[-1]
        # the step count rides on the turn's last result
        self.dialog.append({"role": "tool", "tool_call_id": cid, "content": text + "\n" + self.status_text()})

    def loop(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while self._re_may_continue(loop_cap, spend_cap):
            if self.time_to_ask():
                summary = self.run_interview()
                if summary:
                    self.dialog.append({"role": "user", "content": summary + "\n" + self.status_text()})
                continue
            self.shrink_chat()
            self.turns += 1
            message = self.converse(self.dialog, [WALKER_MODEL, BACKUP_MODEL], 3000, self.tool_specs())
            if message is None:
                self.empty_replies += 1
                if self.out_of_credit or self.model_calls >= CALL_CAP or self.empty_replies > 2:
                    break
                continue
            model_calls = self._re_tool_calls(message)
            content = self._re_text_of(message.get("content"))
            if model_calls:
                idle = 0
                if not self._re_ask_first(model_calls, content):
                    self._re_play(model_calls, content)
                continue
            idle += 1
            self.dialog.append({"role": "assistant", "content": content or "(no call)"})
            self.dialog.append({"role": "user", "content": "Call one of the tools to continue.\n" + self.status_text()})
            if idle >= 3:
                break



def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Buyer(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.run()
    except Exception as exc:
        episode.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.is_over:
                episode.wrap_up()
        except Exception as again:
            episode.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.log


__all__ = ["agent_main"]
