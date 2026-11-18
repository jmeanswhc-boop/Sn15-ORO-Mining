from __future__ import annotations

import itertools
import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST = "anthropic/claude-haiku-5.5"
DEEP = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"

MIN_CALLS = 70
SPEND_EASE = 0.030
SPEND_TIGHT = 0.048
SPEND_END = 0.068
SPEND_STOP = 0.078
TASK_SECONDS = 250.0
WRAP_SECONDS = 60.0
LAST_SECONDS = 28.0
RETRY_WAITS = (4, 10, 20, 35)
RETRY_TOTAL = 45
LIST_MIN = 3
LIST_WIDTH = 5
LIST_STEP = 3
SURVEY_ONE = 6
SURVEY_MANY = 4
SURVEY_TOTAL = 10
JUDGE_BATCH = 6
HUNT_ROUNDS = 14
VALUE_SEARCHES = 3
VALUE_WORDS = 3
HIT_PAGES = 4
WALK_REACH = 10
HUNT_READS = 4
PAGE_LIMIT = 9
COMBO_LIMIT = 400
MENU_KEYS = 14
CHASE_ROUNDS = 9
SHOP_PAGES = 3
SHOP_OPENS = 3
CHASE_STEPS = 12
ORDER_ROUNDS = 4
FINISH_TRIES = 3
RESCUE_TURNS = 30
JSON_ENDING = "\n\nReply now with only the JSON object, as plain text."
JSON_AGAIN = "\n\nYou have no tools and nothing more will be shown. Reply now with only the JSON object as plain text, no other words."

_mind_line = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_shop_line = ProxyClient(timeout=50, max_retries=3, retry_delay=1)

READ_GUIDE = """# Request reader - this step: turn a shopper's request and a store's tool list into one compact JSON object
Return only the object, with these keys.

# Products
"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the words the request uses when it says what to buy (from a named range or category, those words, not a word from the story around it), "shelf": the store's category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "store": the shop this product itself must be bought from when the request says it is sold by or bought from that shop (the name alone as written), else ""; a make, brand or vehicle the product is for or made by is not a shop and stays under its own property; such a shop also goes into "musts" as "shop: name" and into "search_values"; a shop named only to describe another listing is not this product's store, "cap": the most this one product may cost, as written, or null, "search_values": a list holding, for each must whose value is a plain word or name, that value's own shortest words for a store search, with the alternatives of a closed list joined by "|" (a must "finish: brushed or matte" gives "brushed|matte"); a must that is a limit, a range, an exclusion or a value read from another listing adds nothing to this list, which may be empty, "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Keep the request's own word for the value, not a gloss of it ("the single pack type: one can or one bottle" gives "pack type: single"). Copy that word letter for letter even when it looks misspelled, run together or like nothing you know ("in the finish brushedmattish" gives "finish: brushedmattish"): never correct, split or shorten it, and "search_values" keeps it whole too. The value stops where it stops: words that only finish the sentence around it ("get me a lamp in teal sorted" gives "color: teal") stay out. A word the shopper calls you by (the way one says "mate" or "pal") is no requirement, and a brand is a must only when the request says it is the brand. Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: same as the rim colour of the kettle Ovarno sells with a glass lid"), keeping the request's own relation word (same as, matching, unlike, near) and adding none: words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it, when it must arrive) is context, not a must, unless it states a property's value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store's category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
- "lookups": one entry per must whose value is read from another listing, else []: {"must": the must copied exactly, "store": the shop that other listing is found at, the name alone as written, or "", "section": the store's category label for that other listing's kind of product when a tool lists labels (copied exactly), else "", "values": [the own words of each value described for that other listing, short, for a store search]}.
- "fits": for each item that must suit another requested item, {"part": its item number, "main": the item number it must suit}, else [].
- "ifs": conditional rules on the chosen products, each a short sentence in the request's words, else [].

# Money and choice
- "budget": the total the shopper will pay, written exactly as the figure appears in the request, never converted; null when absent. "budget_scope": "total", "each", or null.
- "in_stock_only": true unless the request plainly accepts unavailable listings.
- "pick": "cheapest" for lowest price or smallest total, "most" or "least" for the extreme of a measure, else "any". A remark that money is tight is not a pick. "pick_measure": the measure for most or least, else "".

# Limits and procedure
- "messages": how many messages the shopper allows, or null. "steps": how many steps or tool calls the request allows, or null.
- "ask_first": what the request says to check with the shopper before choosing, else [].
- "process": steps demanded besides finding and buying: {"step": at most 15 words, "when": "order" or "end", "messages": messages it takes}. Requirements, limits and the no-fit ending are not steps.
- "no_fit": the request's sentences about what to do when nothing fits, copied word for word, or "".

# Store tools: describe how each job is called, using exact tool and argument names from the schemas; null for a job no tool does
- "search": {"tool": free-word search, "words": its text argument, "page": its paging argument or null, "fixed": {any other argument the tool requires for a free-word search: the value to send}, else {}}.
- "browse": {"tool": a tool listing a section or filter when one exists, "section": argument, "cap": price ceiling argument or null, "maker": brand argument or null, "page": paging argument or null, "fixed": {any other argument the tool requires for listing a section: the value to send}, else {}}. A search tool that also takes these is named in both jobs.
- "page": {"tool": shows one listing's text, offer and remarks, "id": its id argument}.
- "facts": {"tool": shows one listing's attributes when that is a separate tool from "page", "id": argument}, else null.
- "many": {"tool": shows several listings at once, "ids": its list argument}, else null.
- "ask": {"tool": messages the shopper, "text": argument}.
- "cart": for a tool replacing the whole cart: {"tool", "style": "whole", "lines": its list argument, "line_id": id key inside a line, "line_qty": quantity key inside a line or null}. For a tool changing one line: {"tool", "style": "line", "id": id argument, "qty": quantity argument or null, "mode": mode argument or null, "add": the mode value adding, "remove": the mode value removing}.
- "view": {"tool": shows the cart}, else null.
- "order": {"tool": checks out, "args": the fixed arguments that make it order, else {}}.
- "decline": {"tool": ends without an order, "args": the fixed arguments making it decline, else {}, "note": its free-text argument or null}."""

SHAPE_GUIDE = """# Result reader - this step: name the parts of a store tool's result by their exact keys
Return only: {"rows": the key of the list of listings, or null, "count": the key with how many matched in all (not how many shown), or null, "next": the key with the value asking for the next page, or null, "id": the key inside a row of the listing's id, "title": its title key or null, "section": its department or category key or null, "price": its price key or null, "stock": the key saying whether it can be bought now or null, "maker": its brand key or null, "shop": its seller key or null}"""

FILL_GUIDE = """# Call writer - this step: write the arguments for one store tool from its schema, to do the job described
Return only: {"args": {...}}"""

PLAN_GUIDE = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'

HEAR_GUIDE = """# Answer reader - this step: read a shopper's answer to a shopping agent's question
Return only: {"needs": [{"item": item number, "need": "property: constraint", "bare": "the value's own shortest words for a store search, empty for a limit, a range or a value read from another listing", "points": true when the value is to be read from another listing, "store": for such a value, the shop that listing is found at, the name alone, else "", "section": that listing's kind of product as the answer names it, else "", "values": [the own words of each value described for that listing]}], "drop": [{"item": item number, "property": "a requirement now given up"}], "budget": a new total limit written as the answer writes it, or null, "one_at_a_time": true when the shopper asks for one question at a time instead of answering}
- "needs": every firm preference stated, with the exact value or limit and unit. Name it by the property the answer itself names, even if another was asked. Copy the value word for word, keeping nouns and units. Roundabout wording that names a value still states it, and so does a reply phrased as a question ("is it clear that X means Y for me?").
- An answer may give the value by pointing at another listing (like, unlike, near the value on some shop's product). That is one need with "points": true, whose value is the answer's words from the comparison to the end of the description, unshortened. Properties describing the other product are not needs of this item.
- A reply saying the shopper does not mind, did not follow, or wants one thing at a time states no need."""

NOTE_GUIDE = """# Note reader - this step: a store result carried this note, from the shopper or the store
Return only: {"kind": "change" when it changes the task (budget, requirements, a price, availability, a refused order), "error" when it only reports a malformed or rejected call, else "other", "budget": the new total limit written as the note writes it, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when a chosen listing is gone even if unnamed, "price_moved": true when a price changed}
- A replaced requirement is a drop of the old plus an add of the new. Report every change."""

CHASE_GUIDE = """# Chain follower - this step: find the value a requirement takes from another listing
A requirement can point at a listing described by shop, kind and a few properties. That listing's page can in turn send the reader to a further listing. An order counts only when every page along the chain was opened in full, and the last page gives the value.

You have no tools here: the agent runs the action you write.

# Each turn, give one action
- {"do": "search", "shop": "the shop name alone, when the listing is described by its shop; the agent lists that shop's rows and opens the likeliest ones; name the same shop again to open more of them", "values": ["each value the wanted listing is described by"], "words": "only when no shop is known: one or two of the described values' own words", "section": "the store category holding that kind of product, copied from STORE CATEGORIES, or empty"} to find a listing.
- {"do": "open", "id": "listing id"} to open a page. A row is not an open page.
- {"do": "done", "value": "the property's value as the last page states it", "rule_value": "the value given by a rule that page states for this requirement, or empty", "trail": [the ids of every page on the chain, in order]} when the last page was opened and states the value itself, and no page sends you further.
- {"do": "stuck", "about": "what could not be found"} when nothing more can be tried.

# Reading pages
- Pick the listing whose shop, kind and every described value match. Other listings of the same shop are decoys. A row seldom shows every value, so keep naming the shop or opening its rows until a page shows the described values, rather than searching other words.
- Work forward: once a page matching the description sends you to a further listing, look only for that listing and do not go back to earlier shops.
- Read each page's remarks: words sending you to another listing mean the chain goes on, whatever other values the page shows.
- The value belongs to the property the requirement names, on the last page. An "unlike" or "near" requirement still needs that page's value.
Return only the JSON object for your one action."""

READING_RULES = """Reading rules: the listing's attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, case, plural, spelling variants and unit conversions count, a value containing all of the required value's words counts, a different word does not count even when it means nearly the same, limits include the edge); an options key lists what can be chosen at purchase and states nothing about the property; a requirement the request ties to a rule the store states is met by the value that rule gives; a key for a different property, even a related one (a key about one part of the product says nothing about another part), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing's value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit. A listing's kind is decided by its category label alone: filed under the category of the requested kind, it is that kind whatever its title, description or type attribute calls it. The story around a request (who it is for, their age or size, the occasion, when it must arrive) is not a requirement unless listed as one."""

JUDGE_GUIDE = """# Listing judge - this step: test each listing record against one product's requirements
""" + READING_RULES + """
- Budget and stock are handled elsewhere: do not judge them.
- "pass": every requirement is met by something the record states. Give, per requirement, the exact words of the record that show it: the value of the key named after that property when the record has one. "fail": the record states something that breaks a requirement. "unclear": the record leaves a requirement open.
- "misses": how many requirements are not met.
- "see_also": when the record's remarks send the reader to another listing for more about this product, those words, else "".
- "measure": when a MEASURE is named, a plain number: the figure the record states, converted to the smallest metric unit of its kind so that figures from different records compare; else null.
Return only: {"results": [{"id": listing id, "verdict": "pass", "fail" or "unclear", "quotes": ["..."], "misses": number, "see_also": "", "measure": null}]}"""

IDEAS_GUIDE = """# Search ideas - this step: nothing read so far fits this product; propose new searches
Listing titles and sections often use other words than the request, and keyword search usually reads titles. Prefer short queries: the product's plain name, a required value with the noun, words seen in near-fit titles. Do not repeat a search already tried.
Return only: {"searches": ["up to 3 short searches"]}"""

PAIR_GUIDE = """# Pair judge - this step: decide which listing combinations satisfy the links between requested products
You get the links (one item must suit another), conditional rules and the candidates per item. A link holds only when the records connect the two: a code or name both carry, or a statement on one covering the other.
Return only: {"sets": [{"lines": {"item number": listing id}, "reason": "short"}]}, listing every valid combination you can find, at most 8, or an empty list when none is valid."""

CHECK_GUIDE = """# Last check - this step: decide from the records whether this cart is exactly right before checkout
""" + READING_RULES + """
- Each requested item needs its own line at the requested quantity, with nothing extra. Conditional rules apply to the listings chosen. Sold-out listings do not fit. The newest record wins. Prices and the budget were checked by code and share one unit: do not judge them.
- When one item must suit another, their records must connect them.
- A value the records leave open is not a problem: this cart was the closest the store offers. Only a stated value that breaks a requirement, a wrong quantity or a sold-out listing is.
Return only: {"ok": true or false, "item": the item number at fault or null, "problem": "short, concrete"}."""

CLOSE_GUIDE = """# Closing decision - this step: no complete fit was found; decide how to end, following the request's own words
You get what the request says when nothing fits, the requirements in force, and per item the listings judged, with what each misses.
- If those words tell the agent to buy the nearest listing, answer "order" with the ids to buy, chosen by their own rule.
- If they tell the agent to report and offer a listing, answer "decline", choose the listing by their own rule (counting requirements missed, stock and budget included, and any allowance stated) and name its id in the remark.
- If they only say to report that nothing was found, forbid naming listings or offering a substitute, or say nothing on what to do when nothing fits (words about sold-out listings are not about this): those words apply only when nothing can serve, and a shopper expects an order otherwise. Answer "order" with one listing per item, each on sale, with no stated value breaking a requirement and at least one requirement shown met (verdict "unclear", misses below that item's number of requirements, fewest misses first), keeping the total within the budget. Answer "decline" and name none when some item has no such listing, or no such listings can keep the total within the budget.
Return only: {"action": "order" or "decline", "ids": [ids], "remark": "one or two plain sentences"}"""

GATE_GUIDE = """# Procedure gate - this step: an agent is about to STAGE, and the request demands the steps below
Decide from the numbered RECORD whether each step due now is done for the cart as it stands. For a step about the cart or order, the record line must come after the cart last changed and its reply must grant it. Questions on product properties do not do a step.
When one is missing, write the one short message to the shopper that does it, about that step alone. Give no message when all are done.
Return only: {"message": "..." or null, "hold": what a reply asks to change first, or ""}"""

RESCUE_GUIDE = """# Shopping agent Narmada - this step: finish the errand with the store's tools directly
The usual routine could not read this store. Read each schema and use exact names. Each call costs a step. A correct order holds every requested product at the requested quantity, meeting every requirement, inside the budget. A required value must be stated on the listing. Find, read, cart one line per product, then check out in a call of its own. End without ordering only after real searching found no fit. Every reply must call a tool."""


def as_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def shorten(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: shorten(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [shorten(v, limit, depth + 1) for v in value]
    return value


def parse_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fenced = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    body = fenced.group(1) if fenced else body
    for attempt in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body and "}" in body else ""):
        if not attempt:
            continue
        try:
            got = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(got, dict):
            return got
    return None


def spaced(key: Any) -> str:
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", split).strip().lower()


def word_set(text: Any) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in spaced(text).split())


def names_same(prop: frozenset[str], key: frozenset[str]) -> bool:
    return bool(prop and key) and (prop <= key or (len(key) >= 2 and key <= prop))


def need_prop(need: str) -> str:
    return str(need).split(":", 1)[0].strip()


def need_value(need: str) -> str:
    parts = str(need).split(":", 1)
    return parts[1].strip() if len(parts) > 1 else parts[0].strip()


def listed(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [value] if isinstance(value, (str, dict)) and value else []


def whole(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def number_of(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        clean = value.replace(",", "").strip()
        if re.fullmatch(r"\d+(\.\d+)?", clean):
            return float(clean)
    return None


def measure_of(value: Any) -> float | None:
    found = re.search(r"\d[\d,]*(?:\.\d+)?", str(value)) if value is not None and not isinstance(value, bool) else None
    return float(found.group().replace(",", "")) if found else None


def figures_in(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]


def written_figure(value: Any, source: str) -> float | None:
    amount = number_of(value)
    if amount is None and isinstance(value, str):
        written = figures_in(value)
        amount = written[0] if len(written) == 1 else None
    if amount is None or amount <= 0:
        return None
    seen = figures_in(source)
    if not seen or any(abs(x - amount) < 1e-6 for x in seen):
        return amount
    for scale in (100.0, 0.01):
        if any(abs(x - amount * scale) < 1e-6 for x in seen):
            return amount * scale
    return amount


def kinds_of(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def has_kind(value: Any, kind: str) -> bool:
    checks = {"string": lambda v: isinstance(v, str), "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
              "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool), "boolean": lambda v: isinstance(v, bool),
              "array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict), "null": lambda v: v is None}
    test = checks.get(kind)
    return test(value) if test else True


def recast(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            amount = float(value)
            return int(amount) if kind == "integer" or amount.is_integer() else amount
        if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if kind == "array" and not isinstance(value, list) and value is not None:
            return [value]
    return value


def fit_schema(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = kinds_of(schema)
    if kinds and not any(has_kind(value, k) for k in kinds):
        value = recast(value, kinds)
        if not any(has_kind(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        same = [o for o in options if isinstance(o, str) and isinstance(value, str) and o.lower() == value.strip().lower()]
        if same:
            value = same[0]
        else:
            faults.append(f"{where}: allowed values are {options}")
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [fit_schema(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            faults.extend(f"{where}.{n} is not an allowed argument" for n in value if n not in props)
        faults.extend(f"{where}.{n} is required" for n in schema.get("required") or [] if n not in value)
        value = {k: fit_schema(v, props[k], f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
    return value


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or any(c.isdigit() for c in value)


def find_id_key(rows: list[dict[str, Any]]) -> str | None:
    best: tuple[tuple[bool, float], str] | None = None
    for key in rows[0]:
        named = "id" in str(key).lower()
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(v, named) for v in values):
            continue
        clean = [str(v).strip() for v in values]
        if len(rows) > 1 and len(set(clean)) < len(clean):
            continue
        score = (named, sum(len(v) for v in clean) / len(clean))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def holds_id(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(holds_id(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(holds_id(v, ident, depth + 1) for v in node.values())
    return False


def fact_block(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best: dict[str, Any] | None = None
    for value in node.values():
        block: dict[str, Any] = {}
        if isinstance(value, dict):
            block = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
        elif isinstance(value, list) and len(value) >= 2 and all(isinstance(r, dict) and len(r) == 2 for r in value):
            for row in value:
                left, right = list(row.values())
                if isinstance(left, str) and isinstance(right, (str, int, float)) and not isinstance(right, bool):
                    block[left] = right
        if len(block) >= 2 and (best is None or len(block) > len(best)):
            best = block
    return best


def free_text_keys(schema: dict[str, Any]) -> list[str]:
    props = schema.get("properties") or {}
    return [k for k, v in props.items() if isinstance(v, dict) and "string" in kinds_of(v) and "enum" not in v]


def said_words(observation: Any) -> str:
    if isinstance(observation, str):
        return observation
    found: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 4:
            return
        if isinstance(node, str) and len(node.split()) >= 2:
            found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)

    walk(observation, 0)
    return " ".join(found)[:800] if found else as_text(observation)[:400]


def cut_words(text: str, limit: int | None) -> str:
    clean = " ".join(str(text).split())
    if not limit or len(clean) <= limit:
        return clean
    kept = ""
    for word in clean.split():
        if len(kept) + len(word) + 1 > limit:
            break
        kept = f"{kept} {word}".strip()
    return kept or clean[:limit]


class Clock:
    def __init__(self) -> None:
        self.start = time.monotonic()

    def left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.start)


class Minds:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]]) -> None:
        self.clock = clock
        self.trace = trace
        self.cost = 0.0
        self.calls = 0
        self.call_cap = MIN_CALLS
        self.waited = 0
        self.no_credit = False
        self.late = False

    def spent_out(self) -> bool:
        return self.no_credit or self.late or self.calls >= self.call_cap or self.cost >= SPEND_STOP

    def checkers(self) -> list[str]:
        return [DEEP, FAST] if self.cost < SPEND_EASE else [FAST, DEEP]

    def send(self, seat: str, messages: list[dict[str, Any]], tokens: int, tools: list[dict[str, Any]] | None, plain: bool) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if seat.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            for index in (0, len(sent) - 1):
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": seat, "messages": sent, "max_tokens": tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if plain:
            data = _mind_line.post("/inference/chat/completions", body)
            problem: dict[str, Any] = {} if data else {"status": None}
        else:
            result = _mind_line.post_verbose("/inference/chat/completions", body)
            data, problem = result.data, result.error or {}
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.cost += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.trace.append({"kind": "mind_empty", "seat": seat})
            return None, 0
        status = problem.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            status = whole(data["error"].get("code"))
            problem = {"body": data["error"].get("message")}
        self.trace.append({"kind": "mind_fail", "seat": seat, "status": status, "body": str(problem.get("body") or "")[:200]})
        return None, status

    def think(self, messages: list[dict[str, Any]], seats: list[str], tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for seat in seats:
            if self.calls >= self.call_cap or self.cost >= SPEND_STOP:
                return None
            if self.clock.left() < LAST_SECONDS:
                self.late = True
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.send(seat, messages, tokens, tools, tries > 0)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                allowed = 1 if status in (None, 408) else len(RETRY_WAITS)
                if not passing or tries >= allowed or self.waited + RETRY_WAITS[tries] > RETRY_TOTAL:
                    break
                if self.clock.left() - RETRY_WAITS[tries] < LAST_SECONDS:
                    self.late = True
                    break
                self.waited += RETRY_WAITS[tries]
                time.sleep(RETRY_WAITS[tries] * (0.8 + 0.4 * random.random()))
                tries += 1
            if status == 402:
                self.no_credit = True
                return None
        return None

    def answer(self, prompt: str, seats: list[str], tokens: int = 1800, fixed: str = "") -> dict[str, Any] | None:
        lead = [{"role": "system", "content": fixed}] if fixed else []
        for seat in seats:
            for ending in (JSON_ENDING, JSON_AGAIN):
                message = self.think(lead + [{"role": "user", "content": prompt + ending}], [seat], tokens)
                text = (message or {}).get("content")
                got = parse_object(text)
                if got is not None:
                    return got
                if self.spent_out():
                    return None
                if not str(text or "").strip():
                    break
        return None


class Shop:
    def __init__(self, binding: dict[str, Any], limit: int, finished: bool, trace: list[dict[str, Any]]) -> None:
        self.binding = binding
        self.limit = limit
        self.finished = finished
        self.trace = trace
        self.step = 0
        self.broke = False

    def left(self) -> int:
        return self.limit - self.step

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(4):
            result = _shop_line.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
                if data.get("finished"):
                    self.finished = True
                return data
            problem = result.error or {}
            status = problem.get("status")
            body = str(problem.get("body") or "")
            self.trace.append({"kind": "shop_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "finaliz" in low:
                    self.finished = True
                    return None
                expected = re.search(r"expected (\d+)", body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                elif "reuse" in low or "different action" in low:
                    self.step += 1
                else:
                    return None
                payload["step"] = self.step + 1
                continue
            if status in (400, 401, 402, 403, 404, 422):
                self.broke = status == 402
                return None
        return None


def has_fault(observation: Any) -> bool:
    if not isinstance(observation, dict):
        return False
    return any(("error" in str(k).lower() or "fault" in str(k).lower()) and v for k, v in observation.items())


def row_lists(node: Any, depth: int = 0) -> list[list[dict[str, Any]]]:
    found: list[list[dict[str, Any]]] = []
    if depth > 3:
        return found
    if isinstance(node, list):
        rows = [r for r in node if isinstance(r, dict)]
        if rows and len(rows) == len(node):
            found.append(rows)
    elif isinstance(node, dict):
        for value in node.values():
            found += row_lists(value, depth + 1)
    return found


def core_of(node: Any, depth: int = 0) -> Any:
    if depth > 3 or not isinstance(node, dict):
        return node
    if any(isinstance(v, list) and v and all(isinstance(r, dict) for r in v) for v in node.values()):
        return node
    inner = [v for v in node.values() if isinstance(v, dict)]
    scalars = [v for v in node.values() if (isinstance(v, (int, float)) and not isinstance(v, bool)) or looks_like_id(v)]
    return core_of(inner[0], depth + 1) if len(inner) == 1 and len(node) <= 3 and not scalars else node


def plain_text(record: Any) -> str:
    return re.sub(r"[^a-z0-9.]+", " ", as_text(record).lower()).strip()


def quote_holds(quote: Any, text: str) -> bool:
    words = plain_text(quote).split()
    return bool(words) and sum(1 for w in words if w in text) * 4 >= len(words) * 3


def key_twins(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
    a, b = set(name.split()), set(other.split())
    if not a or not b or not a & b:
        return False

    def agree(x: str, y: str) -> bool:
        wx, wy = re.findall(r"[a-z0-9]+", x.lower()), re.findall(r"[a-z0-9]+", y.lower())
        short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
        return x == y or (bool(short) and set(short) <= set(long) and len(long) <= 2 * len(short) + 2)

    both = [i for i in spec["per"] if i in slot["per"] and spec["per"][i].strip() and slot["per"][i].strip()]
    same = sum(1 for i in both if agree(spec["per"][i], slot["per"][i]))
    if not (a <= b or b <= a):
        return len(both) >= 2 and same == len(both)
    if both:
        return same * 2 > len(both)
    return bool((set(spec["values"]) & set(slot["values"])) - {""})


class Errand:
    def __init__(self, problem: dict[str, Any]) -> None:
        env = problem["environment"]
        self.view = env["policy_view"]
        self.request = str(self.view.get("query") or "")
        self.tools = [t for t in self.view.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.tools}
        limit = whole(self.view.get("max_steps")) or 30
        self.trace: list[dict[str, Any]] = []
        self.clock = Clock()
        self.minds = Minds(self.clock, self.trace)
        self.shop = Shop(dict(env["binding"]), limit if limit > 0 else 30, bool(self.view.get("finished")), self.trace)
        self.brief: dict[str, Any] = {}
        self.jobs: dict[str, dict[str, Any]] = {}
        self.items: list[dict[str, Any]] = []
        self.links: list[tuple[int, int]] = []
        self.budget: float | None = None
        self.msg_limit = 0
        self.one_only = False
        self.many_room = 0
        self.ref_stores: dict[str, dict[str, Any]] = {}
        self.asked_labels: list[tuple[int, str]] = []
        self.budget_hunted = False
        self.stop_at = 0
        self.check_failed = False
        self.walked: set[int] = set()
        self.hits: dict[int, dict[str, int]] = {}
        self.hit_done: set[tuple[int, str]] = set()
        self.hit_ends: dict[int, list[tuple[str, int, Any]]] = {}
        self.search_seen: dict[str, tuple[str, Any]] = {}
        self.msg_used = 0
        self.msg_kept = 0
        self.process: list[dict[str, Any]] = []
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.book: dict[str, dict[str, Any]] = {}
        self.order_seen: list[str] = []
        self.opened: set[str] = set()
        self.keys: dict[str, str | None] = {}
        self.shapes: dict[str, dict[str, Any]] = {}
        self.searched: set[str] = set()
        self.version = 0
        self.heard: set[str] = set()
        self.gone: set[str] = set()
        self.verdicts: dict[tuple[int, str, int], dict[str, Any]] = {}
        self.cart: dict[str, int] = {}
        self.cart_step = -1
        self.dialogue: list[tuple[int, str]] = []
        self.checked: set[str] = set()
        self.refused = 0
        self.plan_order: list[int] = []
        self.pages_read: set[str] = set()

    def left(self) -> int:
        return self.shop.left()

    def spare(self) -> int:
        return self.left() - (len(self.items) + 5)

    def room(self) -> bool:
        return self.spare() > 0 and self.shop.step < self.stop_at and self.time_ok()

    def time_ok(self) -> bool:
        return self.clock.left() > WRAP_SECONDS and not self.minds.spent_out()

    def note_store(self, need: str, entry: dict[str, Any]) -> None:
        store = str(entry.get("store") or "").strip()
        if store:
            self.ref_stores[need] = {"store": store, "section": str(entry.get("section") or "").strip(), "values": [str(v).strip() for v in listed(entry.get("values")) if str(v).strip()]}

    def schema_of(self, job: str) -> dict[str, Any]:
        return self.schemas.get((self.jobs.get(job) or {}).get("tool") or "") or {}

    def read_request(self) -> None:
        catalog = "\n".join(f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {as_text(t['function'].get('parameters') or {})}" for t in self.tools)
        fixed = READ_GUIDE + "\n\nTOOLS:\n" + catalog
        prompt = "REQUEST:\n" + self.request
        if self.view.get("opening") is not None:
            prompt += "\n\nOPENING NOTE FROM THE STORE:\n" + as_text(shorten(self.view.get("opening"), 1500))
        brief = self.minds.answer(prompt, [DEEP, FAST], 4000, fixed) or {}
        self.brief = brief
        self.settle_jobs()
        if not self.minds.no_credit and (not listed(brief.get("items")) or "order" not in self.jobs or "page" not in self.jobs):
            brief = self.minds.answer(prompt, [DEEP, FAST], 3500, fixed) or brief
            self.brief = brief
            self.settle_jobs()
        for entry in listed(brief.get("items")):
            if isinstance(entry, dict):
                count = whole(entry.get("quantity"))
                self.items.append({"name": str(entry.get("name") or "the product"), "shelf": str(entry.get("shelf") or ""), "qty": count if count and count > 0 else 1,
                                   "musts": [str(m) for m in listed(entry.get("musts")) if m], "store": str(entry.get("store") or "").strip(), "words": [str(w).strip() for w in listed(entry.get("search_values")) if str(w).strip()], "told": [], "dropped": [], "cap": written_figure(entry.get("cap"), self.request), "pick": None, "round": 0})
        if not self.items:
            self.items = [{"name": "the requested product", "shelf": "", "qty": 1, "musts": [], "store": "", "words": [], "told": [], "dropped": [], "cap": None, "pick": None, "round": 0}]
        for entry in listed(brief.get("fits")):
            if isinstance(entry, dict):
                part, main = whole(entry.get("part")), whole(entry.get("main"))
                if part and main and part != main and part <= len(self.items) and main <= len(self.items):
                    self.links.append((part, main))
        for lookup in listed(brief.get("lookups")):
            entry = lookup if isinstance(lookup, dict) else {"must": lookup}
            text = str(entry.get("must") or "").strip()
            number = next((n for n, item in enumerate(self.items, 1) if text in item["musts"]), 0)
            if not number:
                said = word_set(plain_text(text))
                near = [(len(said & word_set(plain_text(must))), n, must) for n, item in enumerate(self.items, 1) for must in item["musts"] if word_set(need_prop(must)) == word_set(need_prop(text))]
                if near:
                    _, number, text = max(near, key=lambda x: x[0])
            if number:
                self.refs.append((number, text))
                self.note_store(text, entry)
        self.budget = written_figure(brief.get("budget"), self.request)
        limit = whole(brief.get("messages"))
        self.msg_limit = limit if limit is not None and limit >= 0 else (3 if "ask" in self.jobs else 0)
        steps = whole(brief.get("steps"))
        if steps is not None and 10 <= steps < self.shop.limit:
            self.shop.limit = steps
        for entry in listed(brief.get("process")):
            if isinstance(entry, dict) and str(entry.get("step") or "").strip():
                when = "end" if str(entry.get("when") or "").strip().lower() == "end" else "order"
                self.process.append({"step": str(entry["step"]).strip(), "when": when, "messages": max(0, whole(entry.get("messages")) or 0)})
        self.msg_kept = min(max([p["messages"] for p in self.process] or [0]), self.msg_limit)
        self.minds.call_cap = max(MIN_CALLS, self.shop.limit + 30)

    def settle_jobs(self) -> None:
        self.jobs = {}
        for job in ("search", "browse", "page", "facts", "many", "ask", "cart", "view", "order", "decline"):
            entry = self.brief.get(job)
            if not isinstance(entry, dict) or entry.get("tool") not in self.schemas:
                continue
            props = self.schemas[entry["tool"]].get("properties") or {}
            clean = {"tool": entry["tool"]}
            for key, value in entry.items():
                if key == "tool":
                    continue
                if key in ("words", "page", "section", "cap", "maker", "id", "ids", "text", "note", "qty", "mode", "lines"):
                    clean[key] = value if value in props else None
                else:
                    clean[key] = value
            self.jobs[job] = clean
        search = self.jobs.get("search")
        if search:
            props = self.schemas[search["tool"]].get("properties") or {}
            if not search.get("words"):
                search["words"] = next(iter(free_text_keys(self.schemas[search["tool"]])), None)
            others = [k for k in props if k != search.get("words") and not (self.jobs.get("browse") or {}).get("tool") == search["tool"]]
            if not search.get("page") and len(others) == 1:
                search["page"] = others[0]

    def fill_call(self, job: str, tool: str, purpose: str, problem: str = "") -> dict[str, Any] | None:
        prompt = FILL_GUIDE + f"\n\nTOOL: {tool}\nSCHEMA: {as_text(self.schemas.get(tool))}\nJOB: {purpose}"
        if problem:
            prompt += f"\nLAST ATTEMPT WAS REFUSED: {problem[:400]}"
        got = self.minds.answer(prompt, [FAST, DEEP], 700) or {}
        return got["args"] if isinstance(got.get("args"), dict) else None

    def do(self, tool: str | None, args: dict[str, Any]) -> Any:
        if not tool or self.shop.finished or self.left() <= 0:
            return None
        faults: list[str] = []
        shaped = fit_schema(args, self.schemas.get(tool), "args", faults)
        if faults:
            self.trace.append({"kind": "bad_args", "tool": tool, "faults": faults[:4]})
            return None
        reply = self.shop.call(tool, shaped)
        if reply is None:
            return None
        seen = reply.get("observation")
        self.trace.append({"kind": "call", "tool": tool, "args": shorten(shaped, 80), "step": self.shop.step, "seen": as_text(shorten(seen, 60))[:300]})
        self.absorb(tool, seen)
        self.read_notes(seen)
        return seen if seen is not None else {}

    def learn_shape(self, tool: str, seen: Any) -> dict[str, Any]:
        if tool in self.shapes:
            return self.shapes[tool]
        core = core_of(seen)
        shape: dict[str, Any] = {}
        lists = [k for k, v in core.items() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)] if isinstance(core, dict) else []
        if not lists:
            return shape
        sample = {k: (v[:2] if k in lists else v) for k, v in core.items()}
        prompt = SHAPE_GUIDE + "\n\nTOOL: " + tool + " | parameters: " + as_text(self.schemas.get(tool) or {}) + "\n\nRESULT (first rows only):\n" + as_text(shorten(sample, 120))[:4000]
        got = self.minds.answer(prompt, [FAST, DEEP], 900) or {}
        rows_key = got.get("rows") if got.get("rows") in lists else lists[0]
        first = core[rows_key][0]
        shape = {"rows": rows_key, "count": got.get("count") if got.get("count") in core else None, "next": got.get("next") if got.get("next") in core else None}
        for key in ("id", "title", "section", "price", "stock", "maker", "shop"):
            value = got.get(key)
            if isinstance(value, str) and value in first and not self.keys.get(key):
                self.keys[key] = value
        if not self.keys.get("id"):
            self.keys["id"] = find_id_key(core[rows_key])
        self.shapes[tool] = shape
        self.trace.append({"kind": "shape", "tool": tool, "shape": shape, "keys": dict(self.keys)})
        return shape

    def id_of(self, record: Any) -> str | None:
        key = self.keys.get("id")
        if not key or not isinstance(record, dict):
            return None
        value = record.get(key)
        return value.strip() if looks_like_id(value, True) else None

    def keep_record(self, record: dict[str, Any], full: bool) -> str | None:
        ident = self.id_of(record)
        if not ident:
            return None
        if ident not in self.book:
            self.book[ident] = {}
            self.order_seen.append(ident)
        self.book[ident].update(record)
        stock = self.keys.get("stock")
        if stock and record.get(stock) is False:
            self.gone.add(ident)
        if full:
            self.opened.add(ident)
        return ident

    def absorb(self, tool: str, seen: Any) -> None:
        if tool not in self.shapes:
            self.learn_shape(tool, seen)
        full = tool in {(self.jobs.get(j) or {}).get("tool") for j in ("page", "facts", "many")}
        bare_many = tool == (self.jobs.get("many") or {}).get("tool")
        core = core_of(seen)
        for rows in row_lists(core):
            for row in rows:
                self.keep_record(row, full and (not bare_many or bool(fact_block(row))))
        if isinstance(core, dict):
            self.keep_record(core, full and not bare_many)

    def read_notes(self, seen: Any) -> None:
        lines: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 3:
                return
            if isinstance(node, dict):
                if self.id_of(node):
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
            elif isinstance(node, str) and len(node.split()) >= 5:
                lines.append(node)

        walk(seen, 0)
        fresh = [line for line in lines if line not in self.heard]
        if fresh and self.not_a_reply(fresh):
            self.heard.update(fresh)
            self.take_note(fresh)

    def not_a_reply(self, lines: list[str]) -> bool:
        return not self.speaking or len(lines) > 1

    def take_note(self, lines: list[str]) -> None:
        ledger = "\n".join(f"item {n}: {it['name']}: " + "; ".join(self.needs_of(n)) for n, it in enumerate(self.items, 1))
        got = self.minds.answer(NOTE_GUIDE + f"\n\nITEMS AND REQUIREMENTS NOW:\n{ledger}\nBUDGET: {self.budget}\n\nCART: {as_text(self.cart)}\n\nTHE NOTE:\n" + "\n".join(lines), [FAST, BACKUP, DEEP], 900) or {}
        self.trace.append({"kind": "note", "lines": [l[:200] for l in lines], "got": shorten(got, 120)})
        if got.get("kind") not in (None, "change"):
            return
        changed = False
        budget = written_figure(got.get("budget"), " ".join(lines))
        if budget is not None and budget != self.budget:
            self.budget = budget
            changed = True
        for entry in listed(got.get("drop")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                item = self.item_at(entry.get("item"))
                target = spaced(need_prop(entry["need"]))
                hit = [m for m in item["musts"] + item["told"] if m not in item["dropped"] and spaced(need_prop(m)) == target]
                item["dropped"].extend(hit)
                changed = changed or bool(hit)
        for entry in listed(got.get("add")):
            if isinstance(entry, dict) and str(entry.get("need") or "").strip():
                item = self.item_at(entry.get("item"))
                need = str(entry["need"]).strip()
                if need not in item["musts"] + item["told"]:
                    item["told"].append(need)
                    item["pick"] = None
                    changed = True
        gone = [str(g) for g in listed(got.get("gone")) if str(g) in self.book]
        self.gone.update(gone)
        if got.get("sold_out") is True and not gone:
            self.reread_cart = True
        if got.get("price_moved") is True:
            self.reread_cart = True
        for item in self.items:
            if item["pick"] and (item["pick"] in self.gone or changed):
                item["pick"] = None
        if changed or gone:
            self.version += 1

    reread_cart = False
    speaking = False

    def item_at(self, value: Any) -> dict[str, Any]:
        return self.items[min(max(1, whole(value) or 1), len(self.items)) - 1]

    def needs_of(self, number: int) -> list[str]:
        item = self.items[number - 1]
        out = []
        for need in item["musts"] + item["told"]:
            if need in item["dropped"]:
                continue
            out.append(f"{need} (the value on the page it points at: {self.ref_values[need]})" if need in self.ref_values else need)
        return out

    def field(self, ident: str, key: str) -> Any:
        name = self.keys.get(key)
        return (self.book.get(ident) or {}).get(name) if name else None

    def price_of(self, ident: str) -> float | None:
        value = self.field(ident, "price")
        if isinstance(value, dict):
            value = next((v for v in value.values() if number_of(v) is not None), None)
        return number_of(value)

    def on_sale(self, ident: str) -> bool:
        if ident in self.gone:
            return False
        value = self.field(ident, "stock")
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def affordable(self, ident: str, number: int) -> bool:
        item = self.items[number - 1]
        price = self.price_of(ident)
        if price is None:
            return True
        cost = price * item["qty"]
        if item["cap"] is not None and price > item["cap"] + 1e-6:
            return False
        return self.budget is None or (price if self.brief.get("budget_scope") == "each" else cost) <= self.budget + 1e-6

    def label_score(self, label: Any, number: int) -> int:
        item = self.items[number - 1]
        mine = word_set(item["shelf"]) | word_set(item["name"])
        theirs = word_set(label)
        if not theirs:
            return 0
        if spaced(label) in (spaced(item["shelf"]), spaced(item["name"])):
            return 3
        if theirs <= mine:
            return 2
        return 1 if any(a.startswith(b) or b.startswith(a) for a in theirs for b in mine if len(a) >= 4 and len(b) >= 4) else 0

    def kin_rows(self, number: int) -> list[str]:
        if not self.keys.get("section"):
            return list(self.order_seen)
        scores = {i: self.label_score(self.field(i, "section"), number) for i in self.order_seen}
        best = max(scores.values(), default=0)
        return [i for i in self.order_seen if scores[i] == best] if best else list(self.order_seen)

    def attrs_of(self, ident: str) -> dict[str, Any]:
        return fact_block(self.book.get(ident) or {}) or {}

    def read_listing(self, ident: str, both: bool = False) -> bool:
        facts, page = self.jobs.get("facts"), self.jobs.get("page")
        first = facts if facts and facts.get("id") else page
        if not first or not first.get("id"):
            return False
        seen = self.do(first["tool"], {first["id"]: ident})
        if seen is None:
            return False
        if first is page:
            self.pages_read.add(ident)
        if both and facts and page and page.get("id") and page["tool"] != facts["tool"] and self.spare() > 0:
            self.open_page(ident)
        return True

    def open_page(self, ident: str) -> Any:
        page = self.jobs.get("page")
        self.pages_read.add(ident)
        return self.do(page["tool"], {page["id"]: ident}) if page and page.get("id") else None

    def read_some(self, ids: list[str], keep: int = 0) -> None:
        todo = [i for i in ids if i not in self.opened]
        many = self.jobs.get("many")
        if many and many.get("ids") and len(todo) >= 2 and self.many_room != 1:
            if not self.many_room:
                spec = (self.schema_of("many").get("properties") or {}).get(many["ids"]) or {}
                self.many_room = max(2, spec["maxItems"]) if isinstance(spec.get("maxItems"), int) else 4
            while todo and self.many_room > 1:
                if self.spare() <= 0 or self.left() <= keep:
                    return
                part = todo[: self.many_room]
                seen = self.do(many["tool"], {many["ids"]: part})
                if seen is None:
                    return
                got = {self.id_of(r) for rows in row_lists(core_of(seen)) for r in rows}
                if not got & set(part):
                    self.many_room //= 2
                    continue
                if not self.opened & set(part):
                    self.many_room = 1
                    break
                todo = [i for i in todo if i not in self.opened]
            if not todo:
                return
        for ident in todo:
            if self.spare() <= 0 or self.left() <= keep or self.shop.finished or not self.read_listing(ident):
                return

    def run_search(self, words: str, section: str = "", cap: float | None = None, start: Any = None, within: str = "") -> tuple[str, Any] | None:
        browse, search = self.jobs.get("browse"), self.jobs.get("search")
        if section and browse and browse.get("section"):
            args: dict[str, Any] = {browse["section"]: self.shelf_label(section) or section}
            if cap is not None and browse.get("cap"):
                args[browse["cap"]] = cap
            if start is not None and browse.get("page"):
                args[browse["page"]] = start
            tool, name = browse["tool"], "browse"
        elif words and search and search.get("words"):
            limit = ((self.schema_of("search").get("properties") or {}).get(search["words"]) or {}).get("maxLength")
            args = {search["words"]: cut_words(words, limit if isinstance(limit, int) else None)}
            if start is not None and search.get("page"):
                args[search["page"]] = start
            tool, name = search["tool"], "search"
            label = self.shelf_label(within)
            if label and browse and browse.get("tool") == tool and browse.get("section"):
                args[browse["section"]] = label
        else:
            return None
        args = self.settled_args(name, tool, args)
        mark = tool + as_text(args)
        if mark in self.searched:
            return self.search_seen.get(mark)
        self.searched.add(mark)
        seen = self.do(tool, args)
        if seen is None:
            return None
        self.search_seen[mark] = (tool, seen)
        return tool, seen

    def shelf_label(self, label: str) -> str:
        browse = self.jobs.get("browse") or {}
        if not label or not browse.get("section"):
            return ""
        allowed = ((self.schema_of("browse").get("properties") or {}).get(browse["section"]) or {}).get("enum")
        if not allowed:
            return label
        return next((str(x) for x in allowed if spaced(str(x)) == spaced(label)), "")

    def rows_total(self, found: tuple[str, Any] | None) -> int | None:
        if not found:
            return None
        count = (self.shapes.get(found[0]) or {}).get("count")
        core = core_of(found[1])
        return number_of(core.get(count)) if count and isinstance(core, dict) else None

    def walk_shelf(self, number: int) -> bool:
        if number in self.walked:
            return False
        self.walked.add(number)
        item = self.items[number - 1]
        browse = self.jobs.get("browse") or {}
        section = item["shelf"] if browse.get("section") and browse.get("page") else ""
        words = "" if section else item["name"]
        before = len(self.book)
        found = self.run_search(words, section=section)
        total, rows = self.rows_total(found), len(self.rows_from(found[1])) if found else 0
        if total and rows and total > WALK_REACH * PAGE_LIMIT * rows:
            return len(self.book) > before
        for _ in range(PAGE_LIMIT - 1):
            value = self.next_page(found[0], found[1]) if found else None
            if value is None or not self.room() or self.stop_at - self.shop.step <= HUNT_READS:
                break
            found = self.run_search(words, section=section, start=value)
        return len(self.book) > before

    def value_hits(self, number: int, text: str, weight: int = 1) -> bool:
        item = self.items[number - 1]
        browse, search = self.jobs.get("browse") or {}, self.jobs.get("search") or {}
        narrow = bool(self.shelf_label(item["shelf"])) and browse.get("tool") == search.get("tool") and browse.get("section")
        found = self.run_search(text if narrow else f"{text} {item['name']}", within=item["shelf"])
        ids = self.rows_from(found[1]) if found else []
        if (number, text) not in self.hit_done:
            self.hit_done.add((number, text))
            self.count_hits(number, ids, weight)
            if ids:
                self.hit_ends.setdefault(number, []).append((text, weight, found))
        return bool(ids)

    def count_hits(self, number: int, ids: list[str], weight: int) -> None:
        hits = self.hits.setdefault(number, {})
        for ident in ids:
            hits[ident] = hits.get(ident, 0) + weight

    def page_hits(self, number: int, limit: int = 0) -> None:
        item = self.items[number - 1]
        browse, search = self.jobs.get("browse") or {}, self.jobs.get("search") or {}
        narrow = bool(self.shelf_label(item["shelf"])) and browse.get("tool") == search.get("tool") and browse.get("section")
        ends = self.hit_ends.get(number, [])
        top = max((weight for _, weight, _ in ends), default=0)
        kept: list[tuple[str, int, Any]] = []
        for text, weight, found in ends:
            pages = limit or (PAGE_LIMIT * 2 if self.brief.get("pick") == "cheapest" else HIT_PAGES)
            for _ in range(pages - 1):
                value = self.next_page(found[0], found[1]) if found else None
                if value is None or (weight < top and not limit) or not self.room() or self.stop_at - self.shop.step <= HUNT_READS:
                    break
                found = self.run_search(text if narrow else f"{text} {item['name']}", within=item["shelf"], start=value)
                if found:
                    self.count_hits(number, self.rows_from(found[1]), weight)
            kept.append((text, weight, found))
        self.hit_ends[number] = kept

    def search_values(self, number: int) -> bool:
        groups = [w.split("|") for w in reversed(self.items[number - 1]["words"])][:VALUE_WORDS]
        if not groups:
            return False
        got = False
        for combo in list(itertools.product(*groups))[:VALUE_SEARCHES]:
            if self.left() > 8:
                got = self.value_hits(number, " ".join(combo), len(groups)) or got
        if got or len(groups) == 1:
            return got
        for group in groups[:VALUE_SEARCHES]:
            if self.left() > 8:
                got = self.value_hits(number, group[0]) or got
        return got

    def settled_args(self, name: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs[name]
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") or {}
        fixed = job.get("fixed") if isinstance(job.get("fixed"), dict) else {}
        full = {**{k: v for k, v in fixed.items() if k in props}, **args}
        missing = [k for k in schema.get("required") or [] if k not in full]
        if missing:
            got = self.fill_call(name, tool, f"{name} the store with these arguments already chosen: {as_text(args)}. Give all the arguments.") or {}
            kept = {k: got[k] for k in missing if k in got}
            job["fixed"] = {**fixed, **kept}
            full.update(kept)
        return full

    def rows_from(self, seen: Any) -> list[str]:
        out: list[str] = []
        for rows in row_lists(core_of(seen)):
            out += [i for i in (self.id_of(r) for r in rows) if i]
        return out

    def next_page(self, tool: str, seen: Any) -> Any:
        shape = self.shapes.get(tool) or {}
        core = core_of(seen)
        if not shape.get("next") or not isinstance(core, dict):
            return None
        value = core.get(shape["next"])
        return value if isinstance(value, (str, int)) and not isinstance(value, bool) and value != "" else None

    def learn_shelf(self, number: int) -> None:
        item = self.items[number - 1]
        if item["shelf"] or not self.keys.get("section"):
            return
        counts: dict[str, int] = {}
        for ident in self.order_seen:
            label = str(self.field(ident, "section") or "")
            if self.label_score(label, number):
                counts[label] = counts.get(label, 0) + self.label_score(label, number)
        if counts:
            item["shelf"] = max(counts, key=lambda k: counts[k])

    def survey(self) -> None:
        for number, item in enumerate(self.items, 1):
            if self.spare() <= 0:
                return
            if not self.search_values(number):
                self.run_search(item["name"], within=item["shelf"])
            self.learn_shelf(number)
            if item["shelf"] and self.spare() > 0 and len(self.hits.get(number, {})) < 4:
                self.run_search("", section=item["shelf"])
        wanted = {w for n in range(1, len(self.items) + 1) for need in self.needs_of(n) for w in re.findall(r"[a-z0-9]{3,}", need_value(need).lower())}
        each = SURVEY_ONE if len(self.items) <= 2 else SURVEY_MANY
        lanes: list[list[str]] = []
        heads: list[int] = []
        for number in range(1, len(self.items) + 1):
            kin = self.kin_rows(number)
            have = sum(1 for i in kin if i in self.opened)
            pool = [i for i in kin if i not in self.opened and self.on_sale(i)]
            if self.brief.get("pick") == "cheapest":
                pool.sort(key=lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf"))
            pool.sort(key=lambda i: (-self.hits.get(number, {}).get(i, 0), -len(wanted & set(re.findall(r"[a-z0-9]{3,}", as_text(self.book[i]).lower())))))
            lanes.append(pool[: max(0, each - have)])
            heads.append(max(0, 4 - have))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        room = max(5, 2 * len(lanes) + 2)
        opening = first[:room][:8]
        self.read_some(opening, 22)
        self.read_some((first[room:] + more)[: max(0, SURVEY_TOTAL - len(opening))], 30)

    def owner_of(self, label: str) -> int:
        best, top = 0, 0.0
        words = spaced(label)
        long = {w for w in word_set(label) if len(w) >= 4}
        for number, item in enumerate(self.items, 1):
            if label and words == spaced(item["shelf"]):
                score = 3.0
            elif label and words == spaced(item["name"]):
                score = 2.0
            else:
                mine = {w for w in word_set(item["shelf"]) | word_set(item["name"]) if len(w) >= 4}
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = number, score
        return best

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.order_seen:
            block = self.attrs_of(ident)
            if not block or ident not in self.opened:
                continue
            label = str(self.field(ident, "section") or "listings")
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = spaced(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        matched = {label: g for label, g in groups.items() if self.owner_of(label)}
        table: list[tuple[str, str, int, list[str]]] = []
        for label, group in sorted((matched or groups).items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: (-kv[1]["n"], len(kv[0]))):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                recur = sum(len(set(re.findall(r"[a-z]{3,}", v)) & set(re.findall(r"[a-z]{3,}", " ".join(o for o in values if o is not v)))) > 0 for v in values)
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and recur * 2 < len(values):
                    continue
                place = next((k for k, (other, spec) in enumerate(kept) if key_twins(other, spec, name, slot)), None)
                if place is not None:
                    other, twin = kept[place]
                    for ident, text in slot["per"].items():
                        twin["per"].setdefault(ident, text)
                        if text not in twin["values"]:
                            twin["values"].append(text)
                    twin["n"] = len(twin["per"])
                    kept[place] = (name if len(name.split()) > len(other.split()) else other, twin)
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]["n"])
            table += [(label, name, slot["n"], [v[:40] for v in slot["values"][:4]]) for name, slot in kept[:MENU_KEYS]]
        return table

    def not_stated(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_of(label)
        if not number:
            return ""
        for must in self.items[number - 1]["musts"]:
            prop, _, value = must.partition(":")
            words = set(re.findall(r"[a-z0-9]{3,}", value.lower()))
            shown = set(re.findall(r"[a-z0-9]{3,}", " ".join(values).lower()))
            if prop.strip() and spaced(prop) != name and word_set(prop) & word_set(name) and words and not words <= shown:
                return f" [not the stated {prop.strip()}: another property]"
        return ""

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        settled = [word_set(need_prop(m)) for m in self.items[number - 1]["musts"]]
        rows = [(label, key, carried, values) for label, key, carried, values in table if self.owner_of(label) == number]
        if len(self.items) == 1 and not rows:
            rows = list(table)
        measured = lambda values: sum(1 for v in values if re.match(r"^\s*[\d.,/x -]+\s*[a-z]{0,6}\s*$", v)) * 2 > len(values)
        rows.sort(key=lambda r: (-r[2], measured(r[3]), -len(r[3])))
        return [key for _, key, _, _ in rows if not any(names_same(prop, word_set(key)) for prop in settled)]

    def plan_questions(self, count: int) -> dict[int, list[str]]:
        table = self.key_table()
        settled = [f"item {n}: {it['name']} - settled by the request: " + ("; ".join(it["musts"]) or "nothing") for n, it in enumerate(self.items, 1)]
        listing: list[str] = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values) + self.not_stated(label, name, values))
        prompt = PLAN_GUIDE.replace("COUNT", str(max(1, count * self.list_width()) + len(self.items) + 2)) + "\n\nREQUEST:\n" + self.request + "\n\nITEMS:\n" + "\n".join(settled)
        first = [str(x) for x in listed(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the request asks to be asked about these):\n" + "\n".join(first)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full listing records yet)")
        plan = self.minds.answer(prompt, [DEEP, FAST], 3500) or {}
        queues: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        for pick in listed(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, whole(pick.get("item")) or 1), len(self.items))
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if all(spaced(label) != spaced(other) for other in queues[number]):
                queues[number].append(label)
            if str(pick.get("hint") or "").strip() and number not in self.plan_order:
                self.plan_order.append(number)
        for number, queue in queues.items():
            for key in self.lane_keys(number, table):
                if all(spaced(key) != spaced(other) for other in queue):
                    queue.append(key)
        self.trace.append({"kind": "plan", "ask": shorten(plan.get("ask"), 120), "queues": {n: v[:8] for n, v in queues.items()}})
        return queues

    def next_questions(self, queues: dict[int, list[str]], found: dict[int, int], spent: dict[int, int]) -> tuple[int, list[str]] | None:
        first = {n: i for i, n in enumerate(self.plan_order)}
        scarce = self.msg_limit <= len(queues)
        mains, parts = {m for _, m in self.links}, {p for p, _ in self.links}
        role = lambda n: (0 if n in mains else 2 if n in parts else 1) if scarce else 1
        for number, queue in queues.items():
            told = [word_set(need_prop(t)) for t in self.items[number - 1]["told"]]
            asked = [word_set(label) for n, label in self.asked_labels if n == number]
            queue[:] = [q for q in queue if not any(t and t <= word_set(q) for t in told) and not any(word_set(q) <= a for a in asked)]
        for number in sorted(queues, key=lambda n: (found.get(n, 0) > 0 and role(n) != 0, spent.get(n, 0) + (role(n) == 2), role(n), first.get(n, len(first) + n))):
            if queues[number]:
                return number, queues[number][: self.list_width()]
        return None

    def list_width(self) -> int:
        return LIST_WIDTH if len(self.items) > 1 and self.msg_limit >= LIST_MIN and not self.one_only else 1

    def question_for(self, number: int, labels: list[str]) -> str:
        name = self.items[number - 1]["name"]
        if len(labels) == 1:
            return f"What {labels[0]} would you like for the {name}?"
        return f"For the {name}, what {', '.join(labels[:-1])} or {labels[-1]} would you like?"

    def say(self, text: str) -> tuple[str, bool] | None:
        job = self.jobs.get("ask")
        if not job or self.msg_used >= self.msg_limit or self.left() <= 2:
            return None
        target = job.get("text") or next(iter(free_text_keys(self.schema_of("ask"))), None)
        if not target:
            return None
        self.dialogue.append((self.shop.step + 1, ""))
        self.speaking = True
        seen = self.do(job["tool"], {target: text})
        self.speaking = False
        if seen is None:
            self.dialogue.pop()
            return None
        self.msg_used += 1
        words = said_words(seen)
        self.dialogue[-1] = (self.shop.step, f"{text} -> {words[:300]}")
        self.trace.append({"kind": "ask", "q": text, "a": words[:300]})
        return words, has_fault(seen)

    def interview(self) -> None:
        count = self.msg_limit - self.msg_used - self.msg_kept
        if count <= 0 or "ask" not in self.jobs or not self.time_ok():
            return
        queues = self.plan_questions(count)
        found: dict[int, int] = {}
        spent: dict[int, int] = {}
        hunts: list[tuple[int, str]] = []
        while self.msg_limit - self.msg_used - self.msg_kept > 0 and self.time_ok() and self.left() > 8:
            choice = self.next_questions(queues, found, spent)
            if choice is None:
                break
            number, labels = choice
            question = self.question_for(number, labels)
            result = self.say(question)
            if result is None:
                break
            answer, refused = result
            if refused:
                self.msg_limit = self.msg_used
                break
            spent[number] = spent.get(number, 0) + 1
            before = sum(len(it["told"]) for it in self.items)
            prompt = HEAR_GUIDE + "\n\nITEMS:\n" + "\n".join(f"item {n}: {it['name']}" for n, it in enumerate(self.items, 1)) + f"\n\nQ (about item {number}): {question}\nA: {answer}"
            heard = self.minds.answer(prompt, [FAST, DEEP], 900) or {}
            asked_words = {w for w in word_set(plain_text(" ".join(labels) + " " + self.items[number - 1]["name"])) if len(w) >= 4}
            if not listed(heard.get("needs")) and heard.get("one_at_a_time") is not True and asked_words & word_set(plain_text(answer)):
                heard = self.minds.answer(prompt, [DEEP], 900) or heard
            if len(labels) > 1 and heard.get("one_at_a_time") is True:
                self.one_only = True
                continue
            hunts += self.take_answer(heard, answer)
            if len(labels) == 1:
                self.asked_labels.append((number, labels[0]))
            queues[number] = queues[number][min(LIST_STEP, len(labels)):]
            if sum(len(it["told"]) for it in self.items) > before:
                found[number] = found.get(number, 0) + 1
                self.version += 1
                settled = word_set(need_prop(self.items[number - 1]["told"][-1]))
                queues[number] = [q for q in queues[number] if not (settled and settled <= word_set(q))]
        for number in list(dict.fromkeys(n for n, bare in hunts if bare))[:3]:
            if self.spare() > 6:
                self.value_search(number, 3)

    def take_answer(self, heard: dict[str, Any], source: str) -> list[tuple[int, str]]:
        hunts = []
        for entry in listed(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, whole(entry.get("item")) or 1), len(self.items))
            item = self.items[number - 1]
            need = str(entry["need"]).strip()
            twins = [t for t in item["told"] if need_value(t).lower() == need_value(need).lower() and t != need]
            said = word_set(plain_text(source))
            if twins and word_set(plain_text(need_prop(need))) <= said:
                item["told"][:] = [t for t in item["told"] if t not in twins]
            if need not in item["told"] and not (twins and not word_set(plain_text(need_prop(need))) <= said):
                item["told"].append(need)
            if entry.get("points") is True:
                self.refs.append((number, need))
                self.note_store(need, entry)
            else:
                bare = str(entry.get("bare") or "").strip()
                hunts.append((number, bare))
                if bare and bare not in item["words"]:
                    item["words"].append(bare)
        for entry in listed(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                item = self.item_at(entry.get("item"))
                gone = spaced(need_prop(entry["property"]))
                named = word_set(gone) <= word_set(source)
                if named:
                    item["dropped"].extend(m for m in item["musts"] + item["told"] if spaced(need_prop(m)) == gone and m not in item["dropped"])
        budget = written_figure(heard.get("budget"), source)
        if budget is not None:
            self.budget = budget
        return hunts

    def value_search(self, number: int, reads: int) -> None:
        if not self.search_values(number):
            return
        hits = self.hits.get(number, {})
        cost = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        fresh = [i for i in self.kin_rows(number) if hits.get(i) and i not in self.opened and self.on_sale(i) and self.affordable(i, number)]
        self.read_some(sorted(fresh, key=lambda i: (-hits[i], cost(i)))[:reads], 16)

    def chase_all(self) -> None:
        live = {n for number in range(1, len(self.items) + 1) for n in self.items[number - 1]["musts"] + self.items[number - 1]["told"]}
        for number, need in list(dict.fromkeys(self.refs)):
            if need in live and need not in self.ref_values and self.time_ok() and self.spare() > 0:
                self.chase_one(number, need)

    def shop_walk(self, shop: str, section: str, hints: str, shops: dict[str, list[str]]) -> list[str]:
        if shop not in shops:
            name = word_set(plain_text(shop))
            found = self.run_search(shop, within=section)
            ids = self.rows_from(found[1]) if found else []
            for _ in range(SHOP_PAGES - 1):
                value = self.next_page(found[0], found[1]) if found else None
                if value is None or self.spare() <= 0 or not ids or not all(name <= word_set(plain_text(self.book[i])) for i in ids[-3:] if i in self.book):
                    break
                found = self.run_search(shop, within=section, start=value)
                ids += self.rows_from(found[1]) if found else []
            shops[shop] = [i for i in dict.fromkeys(ids) if i in self.book and name <= word_set(plain_text(self.book[i]))]
        wanted = word_set(plain_text(hints))
        rows = shops[shop]
        ranked = sorted((i for i in rows if i not in self.chase_seen), key=lambda i: -len(wanted & word_set(plain_text(self.book[i]))))
        for ident in ranked[:SHOP_OPENS]:
            if self.spare() <= 0:
                break
            self.chase_seen.add(ident)
            self.open_page(ident)
        return rows

    def chase_one(self, number: int, need: str) -> None:
        page = self.jobs.get("page")
        if not page or not page.get("id"):
            return
        walk: list[str] = []
        recent = list(self.order_seen[-20:])
        strikes = 0
        spot = self.ref_stores.get(need)
        shops: dict[str, list[str]] = {}
        if spot:
            ids = self.shop_walk(spot["store"], spot["section"], need + " " + " ".join(spot["values"]), shops)
            recent = ids + [i for i in recent if i not in ids]
            walk.append(f"searched the shop {spot['store']!r} and opened its likeliest rows: {len(ids)} rows")
        stop = self.shop.step + max(CHASE_STEPS, self.spare() // 2)
        for _ in range(CHASE_ROUNDS):
            if self.spare() <= 0 or not self.time_ok() or strikes >= 3 or self.shop.step >= stop:
                return
            pages = ["OPENED " + as_text(shorten(self.book[i], 300))[:1400] for i in reversed(self.order_seen) if i in self.opened and i in self.chase_seen][:8]
            rows = ["ROW " + as_text(shorten(self.book[i], 80))[:300] for i in reversed(recent) if i not in self.chase_seen][:25]
            shelves = ((self.schema_of("browse").get("properties") or {}).get((self.jobs.get("browse") or {}).get("section") or "") or {}).get("enum") or []
            prompt = (CHASE_GUIDE + f"\n\nREQUIREMENT (item {number}: {self.items[number - 1]['name']}):\n{need}\n\nREQUEST:\n{self.request}\n\nSTORE CATEGORIES: " + (", ".join(str(x) for x in shelves) or "(not listed)") + "\n\nWHAT YOU DID SO FAR:\n" + ("\n".join(walk) or "(nothing yet)")
                      + "\n\nPAGES OPENED (newest first):\n" + ("\n".join(pages) or "(none)") + "\n\nROWS SEEN (newest first):\n" + ("\n".join(rows) or "(none)"))
            got = self.minds.answer(prompt, [FAST, DEEP], 800) or {}
            action = str(got.get("do") or "")
            self.trace.append({"kind": "chase", "need": need[:120], "got": shorten(got, 150)})
            if action == "open":
                ident = str(got.get("id") or "").strip()
                if ident not in self.book or ident in self.chase_seen:
                    strikes += 1
                    walk.append(f"open {ident}: unknown or already open")
                    continue
                self.chase_seen.add(ident)
                if self.open_page(ident) is None:
                    return
                walk.append(f"opened {ident}")
            elif action == "search" and str(got.get("shop") or "").strip():
                shop = str(got.get("shop")).strip()
                fresh = [i for i in shops.get(shop, []) if i not in self.chase_seen]
                if shop in shops and not fresh:
                    strikes += 1
                    walk.append(f"every row of the shop {shop!r} is already open; do not name it again")
                    continue
                ids = self.shop_walk(shop, str(got.get("section") or ""), " ".join(str(v) for v in listed(got.get("values"))) or need, shops)
                recent = ids + [i for i in recent if i not in ids]
                walk.append(f"searched the shop {shop!r} and opened its likeliest rows: {len(ids)} rows")
                strikes += 0 if ids else 1
            elif action == "search":
                words = " ".join(str(got.get("words") or "").split())
                found = self.run_search(words, within=str(got.get("section") or "")) if words else None
                ids = self.rows_from(found[1]) if found else []
                recent = ids + [i for i in recent if i not in ids]
                walk.append(f"searched {words!r}: {len(ids)} rows")
                strikes += 0 if ids else 1
            elif action == "done":
                trail = [str(i).strip() for i in listed(got.get("trail"))]
                value = str(got.get("value") or "").strip()
                unread = [i for i in trail if i in self.book and i not in self.chase_seen]
                if unread:
                    for ident in unread[:3]:
                        self.chase_seen.add(ident)
                        self.open_page(ident)
                    walk.append("opened the rest of the trail: " + ", ".join(unread[:3]))
                    continue
                if value and trail:
                    rule = str(got.get("rule_value") or "").strip()
                    self.ref_values[need] = f"{value}; a rule on that page gives: {rule}" if rule else value
                    self.version += 1
                    return
                strikes += 1
            elif action == "stuck":
                return
            else:
                strikes += 1

    chase_seen: set[str] = set()

    def judge(self, number: int, ids: list[str]) -> None:
        todo = [i for i in ids if (number, i, self.needs_mark(number)) not in self.verdicts]
        needs = self.needs_of(number)
        item = self.items[number - 1]
        measure = str(self.brief.get("pick_measure") or "") if self.brief.get("pick") in ("most", "least") else ""
        for start in range(0, len(todo), JUDGE_BATCH):
            if not self.time_ok():
                return
            batch = todo[start: start + JUDGE_BATCH]
            head = f"PRODUCT: {item['name']}\nREQUIREMENTS:\n" + ("\n".join(f"- {n}" for n in needs) or "- none beyond the kind of product")
            if measure:
                head += f"\nMEASURE: {measure}"
            records = "\n".join(as_text(shorten(self.book[i], 350))[:2200] for i in batch)
            got = self.minds.answer(JUDGE_GUIDE + "\n\n" + head + "\n\nRECORDS:\n" + records, [FAST, DEEP], 3200) or {}
            seen: set[str] = set()
            for entry in listed(got.get("results")):
                if not isinstance(entry, dict) or str(entry.get("id")) not in batch:
                    continue
                ident = str(entry["id"])
                seen.add(ident)
                verdict = str(entry.get("verdict") or "unclear")
                text = plain_text(self.book[ident])
                quotes = listed(entry.get("quotes"))
                if verdict == "pass" and needs and (not quotes or not all(quote_holds(q, text) for q in quotes)):
                    verdict = "unclear"
                if verdict == "pass" and self.skips_key(ident, needs, quotes):
                    verdict = "fail"
                if verdict == "pass" and self.key_differs(ident, needs):
                    verdict = "unclear"
                self.verdicts[(number, ident, self.needs_mark(number))] = {"verdict": verdict, "misses": whole(entry.get("misses")) or 0, "see_also": str(entry.get("see_also") or ""), "measure": measure_of(entry.get("measure"))}
            self.trace.append({"kind": "judge", "item": number, "ids": batch, "got": shorten(got.get("results"), 100)})

    def named_values(self, ident: str, need: str) -> tuple[list[str], str]:
        prop = word_set(need_prop(need))
        held = [(key, value) for key, value in self.attrs_of(ident).items() if prop and prop == word_set(spaced(key))]
        marks = [m for key, value in held for m in (str(key).lower(), spaced(key), str(value).strip().lower())]
        return marks, plain_text(" ".join(str(value) for _, value in held))

    def skips_key(self, ident: str, needs: list[str], quotes: list[Any]) -> bool:
        cited = " ".join(str(q) for q in quotes).lower()
        for need in needs:
            marks, held = self.named_values(ident, need)
            wanted = word_set(plain_text(need_value(need)))
            if marks and "not" not in wanted and not wanted <= word_set(held) and not any(mark in cited for mark in marks):
                return True
        return False

    def key_hits(self, ident: str, number: int) -> int:
        count = 0
        for need in self.needs_of(number):
            marks, held = self.named_values(ident, need)
            count += bool(marks) and word_set(plain_text(need_value(need))) <= word_set(held)
        return count

    def key_differs(self, ident: str, needs: list[str]) -> bool:
        for need in needs:
            marks, held = self.named_values(ident, need)
            value = need_value(need)
            wanted = {w for w in word_set(plain_text(value)) if len(w) >= 3}
            if marks and wanted and "not" not in wanted and not re.search(r"\d", value) and not wanted & word_set(held):
                return True
        return False

    def results(self, number: int, verdict: str) -> list[str]:
        out = []
        for ident in self.kin_rows(number):
            entry = self.verdicts.get((number, ident, self.needs_mark(number)))
            if entry and entry["verdict"] == verdict and self.on_sale(ident) and self.affordable(ident, number):
                out.append(ident)
        return out

    def rank(self, number: int, ids: list[str]) -> list[str]:
        pick = self.brief.get("pick")
        if pick not in ("most", "least"):
            cost = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
            if pick == "cheapest":
                return sorted(ids, key=cost)
            return sorted(ids, key=lambda i: (-self.key_hits(i, number), cost(i)))
        if pick in ("most", "least"):
            value = lambda i: (self.verdicts.get((number, i, self.needs_mark(number))) or {}).get("measure")
            known = [i for i in ids if value(i) is not None]
            return sorted(known, key=lambda i: value(i), reverse=pick == "most") + [i for i in ids if i not in known]
        return ids

    def near_fits(self, number: int) -> list[str]:
        mark = self.needs_mark(number)
        hits = self.hits.get(number, {})
        cost = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        needs = len(self.needs_of(number))
        some_met = [i for i in self.results(number, "unclear") if self.verdicts[(number, i, mark)]["misses"] < needs]
        return sorted(some_met, key=lambda i: (self.verdicts[(number, i, mark)]["misses"], -hits.get(i, 0), cost(i)))

    def needs_mark(self, number: int) -> str:
        return as_text(self.needs_of(number))

    def price_top(self, number: int) -> float | None:
        item = self.items[number - 1]
        limits = [x for x in (item["cap"], self.budget) if x is not None]
        if limits:
            return min(limits)
        prices = [self.price_of(i) for i in self.kin_rows(number) if self.price_of(i) is not None]
        return max(prices) if prices else None

    def read_cheapest(self, number: int, under: float) -> bool:
        while self.room():
            pool = [i for i in self.kin_rows(number) if self.on_sale(i) and self.affordable(i, number)]
            self.judge(number, [i for i in pool if i in self.opened])
            passes = self.results(number, "pass")
            best = min([self.price_of(i) for i in passes if self.price_of(i) is not None] or [float("inf")])
            limit = min(best, under)
            cost = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
            unread = sorted((i for i in pool if i not in self.opened and cost(i) < limit), key=cost)
            if not unread:
                return bool(passes) and best <= under
            self.read_some(unread[:HUNT_READS])
        return False

    def price_ladder(self, number: int) -> None:
        item = self.items[number - 1]
        top = self.price_top(number)
        for share in (0.2, 0.4, 0.7, 1.0):
            if not self.room() or not top:
                return
            cap = float(int(top * share))
            self.run_search("", section=item["shelf"], cap=cap)
            if self.read_cheapest(number, cap):
                return

    def hunt(self, number: int, last: bool = False) -> None:
        item = self.items[number - 1]
        self.stop_at = self.shop.step + (self.spare() if last else max(HUNT_READS + 2, self.spare() // (len(self.items) - number + 1)))
        extreme = self.brief.get("pick") in ("most", "least")
        if self.room():
            self.search_values(number)
            self.page_hits(number)
        if last and self.room():
            for words in reversed(item["words"][-VALUE_WORDS:]):
                self.value_hits(number, words.split("|")[0], 1)
            self.page_hits(number, PAGE_LIMIT)
        hits = self.hits.setdefault(number, {})
        browse = self.jobs.get("browse") or {}
        if not extreme and not hits and browse.get("cap") and item["shelf"]:
            self.price_ladder(number)
        cost = lambda i: self.price_of(i) if self.price_of(i) is not None else float("inf")
        walked = last
        paged = False
        tried: list[str] = []
        for _ in range(HUNT_ROUNDS):
            if not self.room():
                break
            pool = [i for i in self.kin_rows(number) if self.on_sale(i) and self.affordable(i, number)]
            self.judge(number, [i for i in pool if i in self.opened])
            passes = self.results(number, "pass")
            best = min([cost(i) for i in passes] or [float("inf")])
            unread = sorted((i for i in pool if i not in self.opened and (extreme or cost(i) < best)), key=lambda i: (-self.row_fit(number, i), -hits.get(i, 0), cost(i)))
            if unread and (hits.get(unread[0]) or self.row_fit(number, unread[0]) or walked or (passes and not hits)):
                self.read_some(unread[:HUNT_READS])
                continue
            if passes and hits and not extreme and self.brief.get("pick") != "cheapest":
                break
            if not paged:
                paged = True
                self.page_hits(number, PAGE_LIMIT * 2)
                continue
            if not walked:
                walked = True
                self.walk_shelf(number)
                continue
            if (passes and self.brief.get("pick") != "cheapest") or not self.more_rows(number, tried) or not [i for i in self.kin_rows(number) if i not in self.opened and self.on_sale(i) and self.affordable(i, number)]:
                break
        self.judge(number, [i for i in self.kin_rows(number) if self.on_sale(i) and i in self.opened])

    def row_fit(self, number: int, ident: str) -> int:
        shown = word_set(plain_text(self.book.get(ident)))
        count = 0
        for need in self.needs_of(number):
            options = [word_set(part) for part in re.split(r"\s+or\s+|\||,", need_value(need))]
            count += any(words and words <= shown for words in options)
        return count

    def more_rows(self, number: int, tried: list[str]) -> bool:
        item = self.items[number - 1]
        before = len(self.book)
        prompt = IDEAS_GUIDE + f"\n\nPRODUCT: {item['name']}\nREQUIREMENTS: " + "; ".join(self.needs_of(number)) + "\nTITLES SEEN: " + "; ".join(str(self.field(i, "title"))[:70] for i in self.kin_rows(number)[:12]) + "\nTRIED: " + "; ".join(tried)
        got = self.minds.answer(prompt, [FAST, DEEP], 400) or {}
        for words in [str(w) for w in listed(got.get("searches"))][:3]:
            tried.append(words)
            if self.spare() > 0:
                self.run_search(words, within=item["shelf"])
        return len(self.book) > before

    def choose_all(self) -> bool:
        for number, item in enumerate(self.items, 1):
            if item["pick"] and item["pick"] not in self.gone and (number, item["pick"], self.needs_mark(number)) in self.verdicts:
                continue
            self.hunt(number)
            ranked = self.rank(number, self.results(number, "pass")) or self.near_fits(number)
            item["pick"] = ranked[0] if ranked else None
        for number, item in enumerate(self.items, 1):
            if not item["pick"] and self.spare() > 0:
                self.hunt(number, last=True)
                ranked = self.rank(number, self.results(number, "pass")) or self.near_fits(number)
                item["pick"] = ranked[0] if ranked else None
        if not all(item["pick"] for item in self.items):
            return False
        if self.fit_basket():
            return True
        if self.budget is None or self.budget_hunted:
            return False
        self.budget_hunted = True
        for number in range(1, len(self.items) + 1):
            self.hunt(number)
        return self.fit_basket()

    def fit_basket(self) -> bool:
        pools = []
        for number in range(1, len(self.items) + 1):
            passes = self.rank(number, self.results(number, "pass")) or self.near_fits(number)
            pools.append(passes[:8] or [self.items[number - 1]["pick"]])
        allowed: list[tuple[str, ...]] | None = None
        ifs = listed(self.brief.get("ifs"))
        if (self.links or ifs) and self.time_ok():
            lines = [f"item {p} must suit item {m}" for p, m in self.links] + [f"rule: {r}" for r in ifs]
            shown = "\n".join(f"item {n}: {self.items[n - 1]['name']}\n" + "\n".join(as_text(shorten(self.book[i], 200))[:1000] for i in pools[n - 1]) for n in range(1, len(self.items) + 1))
            got = self.minds.answer(PAIR_GUIDE + "\n\nLINKS AND RULES:\n" + "\n".join(lines) + "\n\nCANDIDATES:\n" + shown, self.minds.checkers(), 1500) or {}
            self.trace.append({"kind": "pair", "got": shorten(got, 100)})
            sets = []
            for entry in listed(got.get("sets")):
                lines_map = entry.get("lines") if isinstance(entry, dict) else None
                if isinstance(lines_map, dict):
                    chosen = tuple(str(lines_map.get(str(n)) or "") for n in range(1, len(self.items) + 1))
                    if all(c in pools[n] for n, c in enumerate(chosen)):
                        sets.append(chosen)
            allowed = sets or None
        combos = allowed or list(itertools.islice(itertools.product(*pools), COMBO_LIMIT))
        total = lambda c: sum((self.price_of(i) or 0.0) * self.items[n]["qty"] for n, i in enumerate(c))
        whole_budget = self.brief.get("budget_scope") != "each"
        good = [c for c in combos if len(set(c)) == len(c) and (self.budget is None or not whole_budget or total(c) <= self.budget + 1e-6)]
        if not good:
            return False
        best = min(good, key=total) if self.brief.get("pick") not in ("most", "least") else good[0]
        for item, ident in zip(self.items, best):
            item["pick"] = ident
        return True

    def set_cart(self) -> bool:
        wanted = {item["pick"]: item["qty"] for item in self.items}
        if wanted == self.cart and self.cart_step >= 0:
            return True
        job = self.jobs.get("cart")
        if not job:
            return False
        ok = False
        if job.get("style") == "whole" and job.get("lines") and job.get("line_id"):
            line = lambda i, q: {job["line_id"]: i, **({job["line_qty"]: q} if job.get("line_qty") else {})}
            seen = self.do(job["tool"], {job["lines"]: [line(i, q) for i, q in wanted.items()]})
            ok = seen is not None and not has_fault(seen)
            problem = as_text(seen)
        elif job.get("id"):
            ok, problem = self.set_lines(job, wanted)
        else:
            problem = "no usable form"
        if not ok and self.left() > 2 and not self.shop.finished:
            args = self.fill_call("cart", job["tool"], "Make the cart hold exactly these lines (listing id: units): " + as_text(wanted) + ". Lines now: " + as_text(self.cart), problem)
            if args is not None:
                seen = self.do(job["tool"], args)
                ok = seen is not None and not has_fault(seen)
        if ok:
            self.cart = wanted
            self.cart_step = self.shop.step
        return ok

    def set_lines(self, job: dict[str, Any], wanted: dict[str, int]) -> tuple[bool, str]:
        for ident, have in list(self.cart.items()):
            if wanted.get(ident) != have:
                args: dict[str, Any] = {job["id"]: ident}
                if job.get("mode"):
                    args[job["mode"]] = job.get("remove")
                if job.get("qty"):
                    props = (self.schema_of("cart").get("properties") or {}).get(job["qty"]) or {}
                    args[job["qty"]] = have if job.get("mode") else (0 if props.get("minimum") == 0 else -have)
                seen = self.do(job["tool"], args)
                if seen is None or has_fault(seen):
                    return False, as_text(seen)
                self.cart.pop(ident, None)
        for ident, qty in wanted.items():
            if ident in self.cart:
                continue
            args = {job["id"]: ident}
            if job.get("mode"):
                args[job["mode"]] = job.get("add")
            if job.get("qty"):
                args[job["qty"]] = qty
            seen = self.do(job["tool"], args)
            if seen is None or has_fault(seen):
                return False, as_text(seen)
            self.cart[ident] = qty
        return True, ""

    def run_gate(self, stage: str) -> bool:
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or "ask" not in self.jobs:
            return True
        record = "\n".join(f"{n}. step {s}: {t}" for n, (s, t) in enumerate(self.dialogue, 1))
        prompt = GATE_GUIDE.replace("STAGE", "place the order" if stage == "order" else "end without an order") + "\n\nREQUEST:\n" + self.request + "\n\nSTEPS DUE:\n" + "\n".join(p["step"] for p in steps) + f"\n\nCART (last changed at step {self.cart_step}): {as_text(self.cart)}\n\nRECORD:\n{record or '(nothing yet)'}"
        got = self.minds.answer(prompt, [FAST, DEEP], 500) or {}
        message = str(got.get("message") or "").strip()
        if message and self.say(message) is not None:
            return not str(got.get("hold") or "").strip()
        return True

    def final_check(self) -> dict[str, Any]:
        mark = as_text(sorted(self.cart.items())) + f"#{self.version}"
        if mark in self.checked or self.minds.cost > SPEND_TIGHT:
            return {}
        self.checked.add(mark)
        needs = "\n".join(f"item {n}: {it['name']}, quantity {it['qty']}: " + ("; ".join(self.needs_of(n)) or "no further requirements") for n, it in enumerate(self.items, 1))
        rules = "\n".join(f"rule: {r}" for r in listed(self.brief.get("ifs"))) + "".join(f"\nitem {p} must suit item {m}" for p, m in self.links)
        prompt = CHECK_GUIDE + "\n\nREQUEST:\n" + self.request + "\n\nREQUIREMENTS:\n" + needs + "\n" + rules + "\n\nCART:\n" + "\n".join(f"{i} x{q}: " + as_text(shorten(self.book.get(i), 400))[:2500] for i, q in self.cart.items())
        got = self.minds.answer(prompt, self.minds.checkers(), 700) or {}
        self.trace.append({"kind": "check", "got": got})
        return got if got.get("ok") is False else {}

    def refresh_cart(self) -> None:
        for ident in list(self.cart):
            if self.left() <= 2:
                return
            if self.reread_cart or ident not in self.pages_read:
                self.open_page(ident)
        self.reread_cart = False

    def try_order(self) -> bool:
        job = self.jobs.get("order")
        if not job:
            return False
        version = self.version
        self.refresh_cart()
        if self.shop.finished or self.version != version or any(i in self.gone for i in self.cart):
            return self.shop.finished
        fault = self.final_check() if self.left() > 2 * len(self.cart) + 3 else {}
        if fault:
            number = whole(fault.get("item")) or 1
            if 1 <= number <= len(self.items):
                bad = self.items[number - 1]["pick"]
                self.verdicts[(number, bad, self.needs_mark(number))] = {"verdict": "fail", "misses": 1, "see_also": "", "measure": None}
                self.items[number - 1]["pick"] = None
            self.check_failed = True
            return False
        self.check_failed = False
        if not self.run_gate("order"):
            return False
        before = self.shop.step
        seen = self.do(job["tool"], dict(job.get("args") or {}))
        self.trace.append({"kind": "order", "seen": shorten(seen, 200), "finished": self.shop.finished})
        if not self.shop.finished and self.shop.step > before:
            self.refused += 1
        return self.shop.finished

    def decline(self, remark: str) -> None:
        job = self.jobs.get("decline")
        if not job or self.shop.finished:
            return
        self.run_gate("end")
        args = dict(job.get("args") or {})
        if job.get("note"):
            limit = ((self.schema_of("decline").get("properties") or {}).get(job["note"]) or {}).get("maxLength")
            args[job["note"]] = remark[: limit if isinstance(limit, int) else 500]
        self.do(job["tool"], args)
        self.trace.append({"kind": "decline", "remark": remark[:300], "finished": self.shop.finished})

    def order_near_fits(self) -> bool:
        for _ in range(FINISH_TRIES):
            picks = []
            for number in range(1, len(self.items) + 1):
                ranked = self.rank(number, self.results(number, "pass")) or self.near_fits(number)
                if not ranked:
                    return False
                picks.append(ranked[0])
            total = sum((self.price_of(i) or 0.0) * it["qty"] for i, it in zip(picks, self.items))
            if len(set(picks)) < len(picks) or (self.budget is not None and self.brief.get("budget_scope") != "each" and total > self.budget + 1e-6):
                return False
            for item, ident in zip(self.items, picks):
                item["pick"] = ident
            self.trace.append({"kind": "near", "picks": picks})
            wanted = {item["pick"]: item["qty"] for item in self.items}
            changes = len(set(wanted.items()) ^ set(self.cart.items()))
            if self.left() <= changes + 1:
                if self.cart and self.left() > 0:
                    self.do((self.jobs.get("order") or {}).get("tool"), dict((self.jobs.get("order") or {}).get("args") or {}))
                return self.shop.finished
            if not self.set_cart():
                return False
            refused = self.refused
            if self.try_order():
                return True
            if self.shop.finished or not (self.check_failed or self.refused > refused):
                return False
        return False

    def out_of_reach(self) -> bool:
        for number, item in enumerate(self.items, 1):
            shown = " ".join(plain_text(self.book[i]) for i in self.kin_rows(number))
            for must in item["musts"]:
                value = need_value(must)
                words = [w for w in word_set(plain_text(value)) if len(w) >= 4]
                if must not in item["dropped"] and must not in self.ref_values and not any(n == must for _, n in self.refs) and words and not re.search(r"\d", value) and not any(w in shown for w in words):
                    self.trace.append({"kind": "reach", "item": number, "absent": must})
                    return True
            mark = self.needs_mark(number)
            kin = self.kin_rows(number)
            found = [i for i in kin if (self.verdicts.get((number, i, mark)) or {}).get("verdict") == "pass"]
            usable = [i for i in kin if self.on_sale(i) and self.affordable(i, number) and (i in found or (i not in self.opened and self.row_fit(number, i)))]
            if found and not usable:
                self.trace.append({"kind": "reach", "item": number, "only": found[:4]})
                return True
        return False

    def close_out(self) -> None:
        if self.shop.finished:
            return
        if self.out_of_reach():
            self.decline("Sorry, nothing in the store can meet this request, so I am not ordering.")
            if self.shop.finished or self.left() <= 0:
                return
        lines = []
        for number, item in enumerate(self.items, 1):
            judged = [(i, self.verdicts[(number, i, self.needs_mark(number))]) for i in self.kin_rows(number) if (number, i, self.needs_mark(number)) in self.verdicts]
            lines.append(f"item {number}: {item['name']}, needs: " + "; ".join(self.needs_of(number)))
            lines += [f"  {i} price {self.price_of(i)} {'on sale' if self.on_sale(i) else 'sold out'}: {v['verdict']}, misses {v['misses']}" for i, v in sorted(judged, key=lambda x: (x[1]["verdict"] == "fail", x[1]["misses"], self.price_of(x[0]) or 0.0))[:8]]
        if self.order_near_fits():
            return
        got: dict[str, Any] = {}
        if self.time_ok():
            prompt = CLOSE_GUIDE + "\n\nWHAT THE REQUEST SAYS WHEN NOTHING FITS:\n" + (str(self.brief.get("no_fit") or "").strip() or "(nothing)") + f"\n\nBUDGET: {self.budget}\n\n" + "\n".join(lines)
            got = self.minds.answer(prompt, [DEEP, FAST], 500) or {}
        self.trace.append({"kind": "close", "got": got})
        if got.get("action") == "order":
            ids = [str(i) for i in listed(got.get("ids")) if str(i) in self.book]
            total = sum((self.price_of(i) or 0.0) * it["qty"] for i, it in zip(ids, self.items))
            if len(ids) == len(self.items) and (self.budget is None or self.brief.get("budget_scope") == "each" or total <= self.budget + 1e-6):
                for item, ident in zip(self.items, ids):
                    item["pick"] = ident
                if self.set_cart() and self.try_order():
                    return
        self.decline(str(got.get("remark") or "Sorry, nothing in the store meets all of your requirements, so I am not ordering."))

    def order_rounds(self) -> None:
        for _ in range(ORDER_ROUNDS):
            if self.shop.finished or self.left() <= 2:
                return
            if self.time_ok() and self.spare() > 0:
                self.chase_all()
            if not self.choose_all() or not self.set_cart() or self.try_order():
                return

    def complete(self) -> None:
        for _ in range(FINISH_TRIES):
            self.order_rounds()
            if self.shop.finished or self.left() <= 0:
                return
            ready = all(item["pick"] for item in self.items) and self.cart and self.refused == 0 and not self.check_failed
            if ready and self.jobs.get("order"):
                self.do(self.jobs["order"]["tool"], dict(self.jobs["order"].get("args") or {}))
            if not self.shop.finished:
                self.close_out()
            if self.shop.finished or self.left() <= 0:
                return

    def rescue(self) -> None:
        ledger = "\n".join(f"item {n}: {it['name']} x{it['qty']}: " + "; ".join(self.needs_of(n)) for n, it in enumerate(self.items, 1))
        chat: list[dict[str, Any]] = [{"role": "system", "content": RESCUE_GUIDE}, {"role": "user", "content": f"REQUEST:\n{self.request}\n\nAS READ:\n{ledger}\nBUDGET: {self.budget}\nSteps left: {self.left()}"}]
        for turn in range(RESCUE_TURNS):
            if self.shop.finished or self.left() <= 0 or not self.time_ok():
                return
            message = self.minds.think(chat, [FAST, BACKUP, DEEP], 2000, self.tools)
            calls = (message or {}).get("tool_calls") or []
            if not calls:
                return
            chat.append({"role": "assistant", "content": (message or {}).get("content") or None, "tool_calls": calls})
            for call in calls:
                fn = call.get("function") or {}
                raw = fn.get("arguments")
                args = raw if isinstance(raw, dict) else (parse_object(raw) or {})
                reply = self.shop.call(str(fn.get("name") or ""), args)
                seen = (reply or {}).get("observation")
                chat.append({"role": "tool", "tool_call_id": call.get("id") or f"r{turn}", "content": as_text(shorten(seen, 400))[:9000] + f"\nSteps left: {self.left()}"})

    def run(self) -> list[dict[str, Any]]:
        if self.shop.finished:
            return self.trace
        self.chase_seen = set()
        self.read_request()
        self.trace.append({"kind": "brief", "brief": self.brief})
        if self.view.get("opening") is not None:
            self.absorb("opening", self.view.get("opening"))
        usable = "cart" in self.jobs and "order" in self.jobs and ("search" in self.jobs or "browse" in self.jobs)
        if usable:
            self.survey()
            usable = bool(self.book) and bool(self.keys.get("id"))
        if not usable:
            self.trace.append({"kind": "rescue", "jobs": sorted(self.jobs), "rows": len(self.book)})
            self.rescue()
        else:
            self.interview()
            self.complete()
        self.trace.append({"kind": "end", "finished": self.shop.finished, "steps": self.shop.step, "cost": round(self.minds.cost, 5), "calls": self.minds.calls})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        errand = Errand(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return errand.run()
    except Exception as exc:
        errand.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not errand.shop.finished and errand.items:
                errand.close_out()
        except Exception as again:
            errand.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return errand.trace


__all__ = ["agent_main"]
