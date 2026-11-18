"""11-halo: a buyer for one shopper in an ORO store world.

A light model walks the store with the world's own tools. Code keeps the books (the items and their needs, every
listing seen, the cart), asks the shopper every question itself (a list of one item's likely properties per message,
rotated after each reply, single questions once the shopper asks for them), and holds an order or a no-purchase
ending until its checks pass. Listings sent as plain "key: value | key: value" text are read like structured ones.
"""
from __future__ import annotations

import json
import random
import re
import time

from src.agent.proxy_client import ProxyClient

M_LIGHT = "anthropic/claude-haiku-5.5"
M_HEAVY = "anthropic/claude-sonnet-5.5"
M_SPARE = "openai/gpt-6-luna"


class K:
    """Every number the agent uses, by topic. Never instantiated."""

    # model use per task
    turns, calls, extra_turns = 42, 60, 10
    soft, hard, heavy_room, extra_spend = 0.055, 0.078, 0.02, 0.01
    waits, wait_total = (4, 10, 20, 35), 45
    # per million tokens: (prompt, cached prompt, completion), used only when a reply carries no cost
    price = {M_HEAVY: (2.0, 0.2, 10.0), M_LIGHT: (0.124, 0.011, 0.485), M_SPARE: (0.124, 0.0124, 0.671)}
    # text sizes
    cut_text, cut_result, history = 300, 8000, 80000
    # clock, seconds from the task's start
    task_s, margin_s, floor_s = 300.0, 20.0, 6.0
    ask_by, question_by, wind_down, last_turn, empty_room, late_by = 165.0, 210.0, 200.0, 255.0, 12.0, 190.0
    # steps
    late_steps, gate_rounds, auto_pages, cheaper_opens, widen_each = 8, 3, 6, 3, 6
    # the interview
    scarce_msgs, list_size, guard_keys, guard_gap, table_keys = 2, 8, 3, 0.34, 14


R_CAP_ARG = r"(?:^|_)(?:cap|max|maximum|ceiling|under|up_?to)(?:_|$)"
R_CAP_NOT = r"result|row|page|count|size|num|item|step"
R_SHELF_ARG = r"section|department|categor|label|shelf|aisle|(?<!age)(?:^|_)group(?:_name|_label)?$"
R_LIST_MODE = r"field|filter|facet|brows|categor|list|structured"
R_AMOUNT = r"\d[\d.,]*\s*[a-z\"']{1,12}"
R_FLOOR = r"\b(?:at least|no (?:less|fewer) than|min(?:imum)?)\s+(" + R_AMOUNT + r")\s*$"
R_QUERY = r"(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword"
R_NOT_PROP = (r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?"
              r"|images?|photos?|condition)\b")
R_CATCH_ALL = (r"^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?"
               r"|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b")
R_LIMIT_WORDS = (r"\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within"
                 r"|up to|or)\b")
R_FIT_CODE = r"[A-Z][A-Za-z]{0,5} ?\d{3,6}"
R_PART_CODE = r"[A-Z]{1,4}\d{3,6}"
R_SHOP_KEY = r"seller|shop|store|vendor|brand|maker|merchant"
# words that make a requirement a limit or an exclusion (never read as a stated value)
R_HEDGE = re.compile(r"not|no|least|most|under|over|unlike|different|min|max|minimum|maximum")
# joining words of a pointer description that pick out no listing
R_FILLER = re.compile(r"the|and|with|from|that|comes|listing|page|its|for")


# ----------------------------------------------------------------------------------------------- text and values

def jdump(value):
    """Compact JSON text of a value (its str() when it is not JSON-able)."""
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def shorten(value, size, level=0):
    """A copy of a JSON-like value with every string cut to `size` characters (8 levels deep at most)."""
    if level > 7:
        return "..."
    if type(value) is str:
        return value if len(value) <= size else value[:size] + "..."
    if type(value) is dict:
        return dict((k, shorten(x, size, level + 1)) for k, x in value.items())
    if type(value) is list:
        return list(shorten(x, size, level + 1) for x in value)
    return value


def json_object(text):
    """The JSON object a model reply holds: the whole reply, a fenced block, or the outermost braces."""
    if type(text) is not str:
        return None
    body = text.strip()
    fence = re.match(r"^```[A-Za-z]*\s*(.*?)\s*```$", body, re.S)
    body = fence.group(1) if fence else body
    chunks = [body]
    a, b = body.find("{"), body.rfind("}")
    if a >= 0 and b > a:
        chunks.append(body[a:b + 1])
    for chunk in chunks:
        try:
            parsed = json.loads(chunk)
        except ValueError:
            continue
        return parsed if type(parsed) is dict else None
    return None


def spaced(key):
    """A key in plain lower-case words: camelCase split, underscores, dots and dashes made spaces."""
    split = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return " ".join(re.split(r"[\s_.\-]+", split)).strip().lower()


def snake(key):
    return "_".join(spaced(key).split())


def _drop_s(word):
    return word[:-1] if len(word) > 3 and word[-1] == "s" else word


def stems(text):
    """Words with a plural -s dropped from words longer than 3 letters."""
    return frozenset(map(_drop_s, spaced(text).split()))


def roots(text):
    """Content words of 3+ letters, -es/-s plurals folded, a few joining words left out."""
    kept = set()
    for word in spaced(text).split():
        if re.search(r"(?:ch|sh|ss|x|z)es$", word):
            word = word[:-2]
        elif not word.endswith("ss"):
            word = _drop_s(word)
        if len(word) > 2 and word not in {"and", "for", "the", "with"}:
            kept.add(word)
    return kept


def folded(text):
    """A property name's words with plural -s and British -our folded ("colours" = "color")."""
    words = []
    for word in spaced(text).split():
        if not word.endswith("ss"):
            word = _drop_s(word)
        words.append(word[:-3] + "or" if word.endswith("our") else word)
    return frozenset(words)


def names_same(prop, key):
    """Whether a key names a property: the property's words inside the key, or a key of 2+ words inside it."""
    if prop and key:
        return prop.issubset(key) or (len(key) > 1 and key.issubset(prop))
    return False


def word_set(text, size=3):
    return set(re.findall(r"[a-z0-9]{%d,}" % size, str(text).lower()))


def as_list(value):
    if type(value) is list:
        return value
    if isinstance(value, (str, dict)) and len(value):
        return [value]
    return []


def as_int(value):
    if value is True or value is False:
        return None
    if type(value) is int:
        return value
    if type(value) is float:
        return int(value) if value.is_integer() else None
    if type(value) is str and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def is_number(value):
    return type(value) in (int, float)


def as_number(value):
    if is_number(value):
        return float(value)
    if type(value) is str and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
        return float(value)
    return None


def text_at(entry, key):
    return str(entry.get(key) or "").strip() if type(entry) is dict else ""


def idish(value, named=False):
    """A value that can be a listing id: 3-64 characters, no space, and a digit unless its key is named as an id."""
    if type(value) is not str:
        return False
    text = value.strip()
    if " " in text or len(text) < 3 or len(text) > 64:
        return False
    return named or re.search(r"\d", text) is not None


def id_key_of(rows):
    """The key that identifies these rows: id-like and distinct in every row; keys named id first, then longer."""
    best, best_rank = None, None
    for key in rows[0]:
        named = "id" in str(key).lower()
        texts = []
        for row in rows:
            got = row.get(key)
            if not idish(got, named):
                break
            texts.append(got.strip())
        else:
            if len(rows) > 1 and len(set(texts)) != len(texts):
                continue
            rank = (named, sum(len(t) for t in texts) / len(texts))
            if best_rank is None or rank > best_rank:
                best, best_rank = key, rank
    return best


def carries_id(node, ident, level=0):
    """Whether some dict inside a list in `node` holds `ident` as one of its text values."""
    if level > 5:
        return False
    if type(node) is list:
        for x in node:
            if type(x) is dict and ident in (str(v).strip() for v in x.values() if type(v) is str):
                return True
        children = node
    elif type(node) is dict:
        children = list(node.values())
    else:
        return False
    return any(carries_id(x, ident, level + 1) for x in children)


def facts_in(node):
    """The record's attribute block: its nested dict with the most (2+) plain values."""
    if type(node) is not dict:
        return None
    best = None
    for inner in filter(lambda x: type(x) is dict, node.values()):
        plain = dict((k, v) for k, v in inner.items() if type(v) in (str, int, float))
        if len(plain) > 1 and (best is None or len(best) < len(plain)):
            best = plain
    return best


# --------------------------------------------------------------------------------- listings written as one line

_FIELD = re.compile(r"\s*([A-Za-z_][A-Za-z0-9_ \-]{0,30}?)\s*:\s(.*)\Z", re.S)


def _plain_value(text):
    word = text.strip()
    if word in ("True", "true"):
        return True
    if word in ("False", "false"):
        return False
    if word in ("None", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", word):
        return int(word)
    if re.fullmatch(r"-?\d+\.\d+", word):
        return float(word)
    return word


def _fact_pairs(text):
    """'a=1; b=two; c=x' as {'a': '1', 'b': 'two', 'c': 'x'} (2+ pairs, most parts holding '='), else None."""
    if "=" not in text or ";" not in text:
        return None
    pairs, last, loose = {}, None, 0
    for part in text.split(";"):
        key, sign, value = part.partition("=")
        if sign and key.strip() and len(key.strip()) <= 40:
            last = key.strip()
            pairs[last] = value.strip() if last not in pairs else pairs[last] + ", " + value.strip()
        elif last is not None:
            pairs[last] += ";" + part
            loose += 1
        else:
            loose += 1
    return pairs if len(pairs) >= 2 and loose * 2 < len(pairs) + loose else None


def line_record(text):
    """A listing sent as one line of 'key: value | key: value | ...' text, as a dict (3+ fields, one of them
    id-like); a field holding 'k=v; k=v' becomes a nested attribute dict. None for any other text."""
    if type(text) is not str or text.count(" | ") < 2:
        return None
    raw, last = {}, None
    for chunk in text.split(" | "):
        found = _FIELD.match(chunk)
        if found and found.group(1).strip() and found.group(1).strip() not in raw:
            last = found.group(1).strip()
            raw[last] = found.group(2)
        elif last is None:
            return None
        else:
            raw[last] += " | " + chunk
    if len(raw) < 3:
        return None
    record = {}
    for key, value in raw.items():
        record[key] = _fact_pairs(value) or _plain_value(value)
    for key, value in record.items():
        if type(value) is str and idish(value, "id" in key.lower() or "ref" in key.lower()):
            return record
    return None


def unpack(node, level=0):
    """A copy of an observation with listings written as text lines turned into dicts: a list of such lines, and a
    dict whose only text value is one such line (that dict becomes the record)."""
    if level > 6:
        return node
    if type(node) is str:
        return line_record(node) or node
    if type(node) is list:
        return [unpack(x, level + 1) for x in node]
    if type(node) is dict:
        if len(node) == 1:
            only = next(iter(node.values()))
            got = line_record(only) if type(only) is str else None
            if got:
                return got
        return dict((k, unpack(v, level + 1)) for k, v in node.items())
    return node


# ------------------------------------------------------------------------------------------- tool argument shapes

def kinds_of(schema):
    kind = schema.get("type")
    if type(kind) is str:
        return [kind]
    return list(filter(lambda k: type(k) is str, kind or []))


_KIND_TEST = {
    "string": lambda v: type(v) is str, "integer": lambda v: type(v) is int, "number": is_number,
    "boolean": lambda v: type(v) is bool, "array": lambda v: type(v) is list, "object": lambda v: type(v) is dict,
    "null": lambda v: v is None,
}


def _fits(value, kinds):
    return any(_KIND_TEST.get(k, lambda v: True)(value) for k in kinds)


def _coerce(value, kinds):
    """The first plain conversion of `value` that one of the schema's kinds accepts, else the value unchanged."""
    for kind in kinds:
        if kind == "integer" and type(value) is float and value.is_integer():
            return int(value)
        if kind in ("integer", "number") and type(value) is str and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
            number = float(value)
            return int(number) if kind == "integer" or number.is_integer() else number
        if kind == "string" and is_number(value):
            return str(value)
        if kind == "boolean" and type(value) is str and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        if kind == "array" and value is not None and type(value) is not list:
            return [value]
    return value


def conform(value, schema, where, faults):
    """`value` made to fit a JSON schema where a plain conversion can (numbers, strings, booleans, one-item arrays,
    enum case, a length cap); every remaining mismatch is described in `faults`."""
    if type(schema) is not dict:
        return value
    kinds = kinds_of(schema)
    if kinds and not _fits(value, kinds):
        value = _coerce(value, kinds)
        if not _fits(value, kinds):
            faults.append("%s should be %s" % (where, "/".join(kinds)))
            return value
    allowed = schema.get("enum")
    if type(allowed) is list and value not in allowed:
        if type(value) is str:
            low = value.strip().lower()
            value = next(filter(lambda o: type(o) is str and o.lower() == low, allowed), value)
        if value not in allowed:
            faults.append("%s: allowed values are %s" % (where, allowed))
    limits = schema.get
    if type(value) is str:
        if type(limits("maxLength")) is int and len(value) > limits("maxLength"):
            value = value[:limits("maxLength")]
        if type(limits("minLength")) is int and len(value) < limits("minLength"):
            faults.append(where + " is too short")
    elif is_number(value):
        if is_number(limits("minimum")) and value < limits("minimum"):
            faults.append("%s is below %s" % (where, limits("minimum")))
        if is_number(limits("maximum")) and value > limits("maximum"):
            faults.append("%s is above %s" % (where, limits("maximum")))
    elif type(value) is list:
        if type(limits("maxItems")) is int and len(value) > limits("maxItems"):
            faults.append("%s allows at most %s entries" % (where, limits("maxItems")))
        if type(limits("minItems")) is int and len(value) < limits("minItems"):
            faults.append("%s needs at least %s entries" % (where, limits("minItems")))
        if type(limits("items")) is dict:
            value = [conform(x, limits("items"), "%s[%d]" % (where, n), faults) for n, x in enumerate(value)]
    elif type(value) is dict:
        props = limits("properties") if type(limits("properties")) is dict else {}
        if limits("additionalProperties") is False:
            for key in value:
                if key not in props:
                    faults.append("%s.%s is not an allowed argument" % (where, key))
        for key in limits("required") or []:
            if key not in value:
                faults.append("%s.%s is required" % (where, key))
        value = dict((k, conform(x, props[k], where + "." + k, faults) if k in props else x) for k, x in value.items())
    return value


def props_of(schema):
    props = schema.get("properties") if type(schema) is dict else None
    return props if type(props) is dict else {}


class Params:
    """A tool's parameters sorted by kind."""

    def __init__(self, schema):
        self.schema = schema if type(schema) is dict else {}
        self.props = props_of(self.schema)
        self.required = list(self.schema.get("required") or [])
        self.texts, self.ints, self.arrays, self.enums = [], [], [], {}
        for name, spec in self.props.items():
            if type(spec) is not dict:
                continue
            kinds = kinds_of(spec)
            options = spec.get("enum")
            if "string" in kinds and "enum" not in spec:
                self.texts.append(name)
            if "integer" in kinds:
                self.ints.append(name)
            if "array" in kinds:
                self.arrays.append(name)
            if type(options) is list and len(options) > 1:
                self.enums[name] = options

    def mode(self):
        return next(iter(self.enums), None)

    def count(self):
        return self.ints[0] if self.ints else None


def free_texts(schema):
    """Parameters taking free text (string, no enum)."""
    return Params(schema).texts


# ------------------------------------------------------------------------------------- accessory and device codes

_RX_CODE = re.compile(r"\b" + R_FIT_CODE + r"\b")
_RX_OWN = re.compile(r"[\[(]\s*(?:part\s+)?(" + R_PART_CODE + r")\s*[\])]", re.I)
_RX_MODEL = re.compile(r"\bmodel\s*(?:no\.?|number|#)\s*[:.]?\s*(" + R_FIT_CODE + r")\b", re.I)
_RX_ALIAS = re.compile(r"\b(?:also sold as|also known as|sold also as)\s+([^.;]+)", re.I)
_RX_WORKS = re.compile(r"\b(?<!not )(?<!never )(?:works with|compatible with|fits)\s*:?\s+([^.;]+)", re.I)
_RX_ALSO_TAIL = re.compile(r"\bpart\s+" + R_PART_CODE + r"\s+also\s+$", re.I)
_RX_ALSO = re.compile(r"\bpart\s+(" + R_PART_CODE + r")\s+also fits\s*:?\s*([^.;]+)", re.I)
_RX_MORE = re.compile(r"\b(?:more|details|see|read)\s+(?:on|at|under|about)?\s*(?:the\s+)?part\s+(" + R_PART_CODE + ")", re.I)


def code_list(text):
    return [code_key(c) for c in _RX_CODE2.findall(str(text or ""))]


def fit_marks(text):
    """Codes a listing's text gives: its own part code, the model codes it is sold as, the codes and series it works
    with (minus the models it names as exceptions), notes of other parts' fits, and a part it sends the reader to."""
    text = str(text or "")
    sold_as, works, series, excepted = [], [], [], []
    for rx in (_RX_MODEL, _RX_ALIAS):
        for hit in rx.finditer(text):
            sold_as.extend(code_list(hit.group(1)))
    for hit in _RX_WORKS.finditer(text):
        if _RX_ALSO_TAIL.search(text[max(0, hit.start() - 40):hit.start()]) is not None:
            continue
        clause = hit.group(1)
        pieces = re.split(r"\bexcept\b", clause, maxsplit=1, flags=re.I)
        head, tail = pieces[0], (pieces[1] if len(pieces) > 1 else "")
        series.extend(code_key(s) for s in _RX_SERIES.findall(head))
        works.extend(c for c in code_list(_RX_SERIES.sub(" ", head)))
        excepted.extend(code_list(tail))
    own, more = _RX_OWN.search(text), _RX_MORE.search(text)
    return {"own": own.group(1).upper() if own else "", "codes": sold_as, "works": works, "series": series,
            "except": excepted, "also": [(h.group(1).upper(), code_list(h.group(2))) for h in _RX_ALSO.finditer(text)],
            "more": more.group(1).upper() if more else ""}

def compose(head, *sections):
    """A prompt: the instruction text, then each (TITLE, body) section as a blank line, 'TITLE:' and the body."""
    pieces = [head]
    for title, body in sections:
        pieces += ["\n\n", title, ":\n", body]
    return "".join(pieces)

def code_key(code):
    """A model code compared without case, spaces or hyphens ("MOZ-2120" = "moz2120")."""
    return re.sub(r"[\s\-]", "", str(code)).lower()

def fits_code(mark, extra, code):
    """Whether a part fits a device code: listed by the part (or by a note on the part's own code), or inside a series
    the part names and not one of its exceptions."""
    key = code_key(code)
    if key in mark.get("except", ()):
        return False
    if key in mark.get("works", ()) or key in extra:
        return True
    return any(key.startswith(s) and len(key) == len(s) + 2 for s in mark.get("series", ()))


# U3: model codes with a hyphen ("MOZ-2120"), "every ROT31xx model" series, and model numbers read with either
_RX_CODE2 = re.compile(r"\b[A-Z][A-Za-z]{0,5}[ -]?\d{3,6}\b")
_RX_SERIES = re.compile(r"\bevery\s+([A-Z][A-Za-z]{0,5}[ -]?\d{1,4})xx\b", re.I)
_RX_MODEL = re.compile(r"\bmodel\s*(?:no\.?|number|#)\s*[:.]?\s*([A-Z][A-Za-z]{0,5}[ -]?\d{3,6})\b", re.I)


# ------------------------------------------------------------------------------------------------------ prompts

P_WALK = """You shop on behalf of one shopper in an online store, and the only way to act is to call the store's tools. The episode scores 1 or 0, and it scores 1 only for an exactly right finish: either an order holding every requested item at its requested quantity, each one meeting every requirement in force at checkout, within the budget; or, when the store really has nothing that meets everything, an ending without a purchase made the way the request asks.

THE STORE
- Each tool call uses one step and steps are limited, while thinking is free. Tools, argument names and result shapes differ from store to store: read every tool's description and schema, and use the exact argument names and allowed values.
- Searches can be narrow: a category filter, a keyword mode that may read only the first words, a few rows per page, a page token or offset. When the answer depends on seeing all matches, compare the reported total with the rows seen and page on with the exact token or offset the store returns. Keyword queries work best short, with the wanted value first and then the product noun. A filtered search with no rows usually means a wrong filter value or parameter: retry with the label written exactly as result rows show it, or with another filter parameter.
- Result rows are summaries. Stores split a listing's record in different ways: one view can show its attributes (properties), another only its offer and description, and a side-by-side view can carry either. The ledger names the tool that shows attributes once the code knows it. Judge every requirement on a property from those attributes, so read them for each listing you might buy; one call that shows several listings saves steps when it shows what you need. Do not open a listing again when nothing has changed since you read it.
- The code's interviewer handles every question to the shopper; you never message the shopper. Its answers come into the conversation and into the ledger.
- The request can also ask for steps along the way (an approval, a message, something to check or read first). The ledger lists them. The steps due right before the order or the ending are done by the code when you call the checkout or the ending tool, so never wait for them and never end without a purchase because of them; do the others with the store's tools (questions about products go through the interviewer).

WHAT FITS
- The listing's category label decides the kind of product: only the label matching the requested item as a whole counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value containing the required value's words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). Different words do not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Convert units before comparing numbers; limits include the edge value; a range must lie wholly inside a limit.
- An attribute whose key names the requirement's property (that name or a plain synonym) decides: when it says something else, the listing fails, whatever other keys or the title say. A key for another property, even a related one (a handle colour is not the body's colour), decides nothing about the requirement. The code adds [evidence] lines after opened listings, naming the keys whose names match a requirement.
- When the listing has no attribute for that property, the wanted value stated under another attribute (a list of styles or features that includes it, say) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.
- When the category label itself states a requirement's value (a season or an audience in the label), the category meets that requirement.
- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-listing or side-by-side tool (a search row is not enough), and if the page points onward to another listing for the property, locate and open that one in the same manner; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing's value, the rule excludes only listings sharing that exact value; never end without a purchase because of it alone.
- When a listing you are about to buy sends you to another page for more about it, look at that page before buying.
- When one requested item must fit another (an accessory for a device), choose the two together from the listings' own words (a list of what it fits, a model or series code both pages carry). When no accessory meeting its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.
- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") only break ties.
- Never buy an item the store shows as sold out. The total (price times quantity, summed over all lines) may not exceed the budget; equal is fine.

CHOOSING
- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: rows often come in no price order, and the code reads the next pages of a short result itself), check candidates from the lowest price up, and take the first that fully fits. When a search takes a maximum price, use it: list the category below your current best pick's price so only cheaper candidates are left.
- When the request asks for the most or least of a measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.
- Otherwise any listing that meets everything is right: do not spend steps hunting for a better one.
- Cart each item as soon as you have opened a listing meeting every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.
- A request for several different items needs one cart line per item, each at its requested quantity.

CHANGES
- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear in tool results, often as a refusal when you try to check out; a note counts whichever tool's result shows it, and the ledger lists the requirements in force. The episode continues after a refusal: apply every change, fix the cart and check out again.
- Read a chosen listing's record again shortly before checking out, since price and stock can change silently.

FINISHING
- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.
- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request's wording for that ending, and when a remark is needed, keep it short and name no listing.
- Every reply must call at least one tool. Keep your text short: what you know, then the next move."""

P_BRIEF = """You set up a shopping task for an agent. Read the shopper's request and the store's tools, and return ONE JSON object (compact, no indentation, nothing else) with these keys:

"items": one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request's own words, "shelf": the store's category label for it when a tool lists labels (copied exactly), or the category the request itself names for it ("a coat in rainwear" gives "rainwear"), else "", "quantity": units of that one listing (an integer, 1 when not stated; a pair or a set sold as one listing counts as 1), "musts": the firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list}.
  How to write musts:
  - Name the property with the request's own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); never swap it for a neighbouring property.
  - A must whose value is read from another listing is ONE must holding that listing's whole description ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words describing that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product.
  - Never put price, budget, stock or quantity into musts.
  - The story around the request (who will use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property's value or limit outright. Words that are part of what the product is called (a season or an audience named with the product) stay in "name", never in musts.
  - A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "a bag, and it is for a wedding", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store's category label already names it. A particular person's event, a person or a setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context.
  - A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.
"budget": the most the shopper will pay in total, as a number in the request's own unit, or null.
"budget_unit": that unit in words (for example "cents"), or null.
"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.
"in_stock_only": true unless the request clearly accepts items that are not available.
"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.
"pick_measure": what is maximised or minimised for "most"/"least", else "".
"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.
"steps": how many steps, moves or tool calls the request allows (integer), or null.
"lookups": the musts (copied exactly) whose value must first be read from another listing the request points at, else [].
"rules": every instruction the request gives about how to work or how to end, each in at most 12 words.
"wishes": things the shopper would merely like.
"ask_first": things the request explicitly says to check with the shopper before choosing ([] when none).
"process": the steps the request requires besides finding and buying the right items (getting the shopper's approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when there is no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.
"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.
"fits": {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.
"roles": the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing's attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing's current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).
"buy_args": the fixed arguments that make the "buy" tool check out (the mode, for a tool with a mode argument), else {}.
"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.
"""

P_MODES = ("The store tool {tool} both checks out and ends without a purchase, depending on its arguments. Its schema: "
           "{schema}\nReturn ONE JSON object and nothing else: {{\"buy_args\": the fixed arguments that check out, "
           "\"refuse_args\": the fixed arguments that end without a purchase}} (only the arguments that choose the "
           "mode).")

P_PLAN = """You pick the questions a shopping agent will put to a shopper. The shopper holds a few firm preferences that were never written down, each about one property of one requested product. A question draws such a preference out only when it names that exact property. A property the shopper has no view on gets "don't mind"; one the shopper never thinks about gets a puzzled reply; so every property named uses part of a message.

You get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item's category, with how many of those listings carry each key and some example values. An unwritten preference can sit on a property that no listed key names.

Rank properties in this order:
0. Whatever the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it concerns.
1. Hinted properties. The request's story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, how the product looks or the setting it belongs in, the person meant to use or wear it, or what goes inside it. For each such sentence, name the one property it points at and the item it is about, in story order. Use a listed key of that item when one fits the hint (a hint about who wears it: the gender or audience key; about its looks: the listed colour, pattern or finish key that most of its listings carry). A hinted property that no listing of the item carries comes after the item's three most carried keys.
2. Then the listed keys: first the properties a buyer of that kind of product chooses between (often its type or style, form or fit, size, capacity or dimensions, material, colour or pattern, whichever that kind of product is bought by) and the keys most of the item's listings carry; once five or more listings were read, avoid keys seen on only one or two of them.
Leave out: a property the request already settles for that item (a requirement names it; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is a different property from that requirement whatever words its name shares with it. Rank each item on its own: its hinted properties first, then its listed keys.
"label" is the property's plain name: for a listed key, all of the key's words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the name, no example values, no brackets, no second property.
Pick up to COUNT per item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story's words it comes from, or an empty string"}]}, with each item's picks in rank order"""

P_HEAR = """A shopping agent asked a shopper some questions. Read every answer and return ONE JSON object and nothing else:
{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants them one at a time; not answers that only say they did not follow it or want simpler words], "budget": a new total limit as a number, or null}
- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name it by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3"). The value is only what a listing could state, from the wanted value or its limit word to its end, without the framing the answer puts around it ("oval" for "matching oval", "settled, with oval" or "oval and nothing else"; "at least 30 cm" for "equal to at least 30 cm").
- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop's product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer's own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.
- "free": the property of each question the shopper said they do not mind about.
- Folksy or roundabout wording that names a value still states that value. Wording that refuses to bargain over a value (it will not be traded, given up or changed) states that value as wanted; write "not X" only when the answer rejects X itself."""

P_NOTE = """While a shopping agent works, the store showed the note below (it may come from the shopper or the store). Return ONE JSON object and nothing else:
{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports a malformed or refused call, "other" for anything else, "budget": the new total limit as a number in the request's unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}
A replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note."""

P_RULES = ("a listing's attribute for a property decides when its key names that same property (any key spelling, one "
           "element of a list is enough, synonyms and unit conversions count, a value containing the required "
           "value's words counts, such as \"deep teal\" for teal, limits include the edge); a key for another "
           "property, even a related one (a handle colour is not the body's colour), decides nothing; when no "
           "attribute names that property, the wanted value stated under another attribute or in the title or "
           "description counts (a bare figure that does not say what it measures counts for nothing); a requirement "
           "the category label itself states is met by that category; the category label must match the item; a "
           "rule of being unlike another listing's value counts as met unless the records show that same value; "
           "out-of-stock listings and listings that break the budget do not fit.")

P_CHECK = """You are the last check before a shopping agent checks out. From the records shown, decide whether this cart is exactly right.
How to read records: """ + P_RULES + """ A number of another kind does not count (a weight is not a capacity). Each requested item must be a separate cart line at the requested quantity, nothing else may be in the cart, nothing may be shown out of stock, and the total of price times quantity over all lines must be within the budget. A listing's newest record wins.
When the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart's choice on that rule, the cart is wrong.
Return ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Say ok=false only for a problem the records clearly show or a requirement no record addresses."""

P_AUDIT = """A shopping agent wants to end without buying because it believes nothing fits. Check the listing records shown against the requirements. How to read records: """ + P_RULES + """
Records marked ROW were seen only as a search result row: their page (the attributes) was not read, so they can be neither passed nor failed on a property the row does not show.
Return ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}"""

P_UNMIX = """A shopper's request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words picking out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes misread as requirements on the product being bought. Using the request's wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}"""

P_TRAIL = """A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing's page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement's chain from its own description, link by link, through the pages' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page unread).
Return ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property's value itself (also when the requirement asks for a value other than or unlike that one: the page's value is what it is measured against), else false, "value": the property's value as stated on that last page, or ""}]}
A record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop's listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop's name and the description's values, as a few search words.
List each chain's links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown."""

P_GATE = """A shopping agent is about to STAGE. The shopper's request asks for the steps below along the way. From the numbered RECORD of what the agent has done, decide whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions about a product's properties do not do a step.
When a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.
Return ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}"""

P_SWAP = ("A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the "
          "other listing records shown. How to read records: " + P_RULES + "\nReturn ONE JSON object and nothing "
          "else: {\"fits\": [{\"id\": listing id, \"price\": its price as a number as the record shows it} for every "
          "listing that appears to meet every requirement of the sold-out line's item, at most 6], \"reason\": \"one "
          "sentence\"}")

P_SHAPE = ("A store tool returned the result below. Name its parts by the exact keys the result and the tool's "
           "parameters use. Return ONE JSON object and nothing else: {\"rows\": the key of the list of listings, "
           "\"count\": the key holding how many listings matched in all (not how many are shown), or null, \"next\": "
           "the key holding the value that asks for the following page, or null, \"next_arg\": the tool parameter "
           "that takes that value, or null, \"price\": the key, inside a row, of the listing's price, or null, "
           "\"stock\": the key, inside a row, that says whether the listing can be bought now, or null}")

QUIET_LINE = "Sorry, nothing in the store meets all of your requirements, so I am not buying anything."


# --------------------------------------------------------------------------------------- model and store traffic

def _failure_status(got):
    """(status, body) of a failed proxy reply: an error carried inside a 200 body wins over the HTTP error."""
    data, error = got.data, got.error or {}
    inner = data.get("error") if type(data) is dict else None
    if type(inner) is dict:
        return as_int(inner.get("code")), str(inner.get("message") or "")
    return error.get("status"), str(error.get("body") or "")


class Wire:
    """Every request the episode sends: model calls (with cache marks, spend, retries and a clock-aware timeout) and
    store steps (with the step counter kept in sync)."""

    store = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
    models = {}

    def __init__(self, episode):
        self.ep = episode
        self.spent, self.calls, self.waited = 0.0, 0, 0
        self.no_credit = self.broke = False

    # -- model side
    def client(self):
        """A model client whose timeout fits the time left: 50 s, or less in whole 5 s steps."""
        room = self.ep.room()
        limit = 50 if room >= 50 else max(5, 5 * (int(room) // 5))
        if limit not in Wire.models:
            Wire.models[limit] = ProxyClient(timeout=limit, max_retries=1, retry_delay=1)
        return Wire.models[limit]

    def post(self, path, body):
        return self.client().post_verbose(path, body)

    @staticmethod
    def marked(model, messages):
        """Anthropic requests get a cache mark on the first and the newest message."""
        if len(messages) < 2 or not model.startswith("anthropic/"):
            return messages
        out = list(map(dict, messages))
        for spot in sorted({0, len(out) - 1}):
            text = out[spot].get("content")
            if text and type(text) is str:
                out[spot]["content"] = [dict(type="text", text=text, cache_control=dict(type="ephemeral"))]
        return out

    def bill(self, model, usage):
        """Add a reply's cost: its own figure, else its tokens at the model's list price."""
        if type(usage) is not dict:
            return
        paid = usage.get("cost")
        if is_number(paid) and paid > 0:
            self.spent += float(paid)
            return
        rate = K.price.get(model)
        if rate:
            sent = as_int(usage.get("prompt_tokens")) or 0
            extra = usage.get("prompt_tokens_details")
            cached = min(sent, (as_int(extra.get("cached_tokens")) or 0) if type(extra) is dict else 0)
            made = as_int(usage.get("completion_tokens")) or 0
            self.spent += (rate[0] * (sent - cached) + rate[1] * cached + rate[2] * made) / 1e6

    def once(self, model, messages, room, tools):
        body = dict(model=model, messages=self.marked(model, messages), max_tokens=room, temperature=0)
        if tools:
            body.update(tools=tools, tool_choice="auto")
        got = self.post("/inference/chat/completions", body)
        data = got.data
        choices = data.get("choices") if type(data) is dict else None
        if choices:
            self.bill(model, data.get("usage"))
            message = choices[0].get("message") or {}
            if message.get("tool_calls") or message.get("content"):
                return message, 200
            self.ep.note("llm_fail", model=model, status=0)
            return None, 0
        status, text = _failure_status(got)
        self.ep.note("llm_fail", model=model, status=status, body=text[:200])
        return None, status

    def chat(self, messages, models, room, tools=None):
        """The first model reply (text or tool calls) from the models in order; transient failures are retried after
        jittered waits while the episode's wait allowance and clock permit."""
        for model in models:
            if self.calls >= K.calls:
                return None
            if self.ep.room() < K.floor_s:
                self.ep.note("late_call_skipped", at=round(self.ep.clock(), 1))
                return None
            self.calls += 1
            tries, status = 0, None
            while True:
                message, status = self.once(model, messages, room, tools)
                if message is not None:
                    return message
                if not self.again(status, tries):
                    break
                pause = K.waits[tries]
                self.waited += pause
                tries += 1
                time.sleep(pause * random.uniform(0.8, 1.2))
            if status == 402:
                self.no_credit = True
                return None
        return None

    def again(self, status, tries):
        """Whether a failed model request is worth one more try after the next wait."""
        if status is not None and status not in (402, 408, 425, 429) and not (is_number(status) and status >= 500):
            return False
        if tries >= (1 if status in (None, 408) else len(K.waits)):
            return False
        pause = K.waits[tries]
        return self.waited + pause <= K.wait_total and self.ep.room() - 1.2 * pause >= K.floor_s

    def ask(self, prompt, models, room=1800):
        """A JSON job: the first model whose reply holds a JSON object."""
        for model in models:
            reply = self.chat([dict(role="user", content=prompt)], [model], room)
            got = json_object((reply or {}).get("content"))
            if got is not None:
                return got
            if self.no_credit or self.broke or self.calls >= K.calls:
                return None
        return None

    # -- store side
    def step(self, name, args):
        """One store call; keeps the step counter in line with the store's and marks the episode finished."""
        ep = self.ep
        packet = dict(ep.binding, action=dict(name=name, args=args))
        tries = 4
        while tries:
            tries -= 1
            packet["step"] = ep.step + 1
            got = Wire.store.post_verbose("/environment/call", packet)
            reply = got.data
            if type(reply) is dict:
                ep.step = reply["step"] if type(reply.get("step")) is int else ep.step + 1
                ep.finished = ep.finished or bool(reply.get("finished"))
                seen = reply.get("observation")
                failed = type(seen) is dict and seen.get("error")
                if not failed and name in ep.page_tools():
                    for value in args.values():
                        for x in (value if type(value) is list else [value]):
                            if type(x) is str:
                                ep.books.opened.add(x.strip())
                return reply
            error = got.error or {}
            status, text = error.get("status"), str(error.get("body") or "")
            ep.note("env_fail", status=status, body=text[:200])
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
            if status != 409:
                continue
            low = text.lower()
            if re.search(r"finish|quarantin|finaliz", low):
                ep.finished = True
                return None
            expected = re.search(r"expected (\d+)", text)
            if expected:
                ep.step = int(expected.group(1)) - 1
            elif "reuse" in low or "different action" in low:
                ep.step += 1
            else:
                return None
        return None


# ------------------------------------------------------------------------------------------- the listing ledger

class Books:
    """Every listing the episode has seen: search rows, full records (with an attribute block), what each view shows,
    which listings were opened, sold out or reported gone, and where a row keeps its id, label, price and stock."""

    def __init__(self, schemas):
        self.rows, self.full, self.order = {}, {}, []
        self.id_key = self.price_key = self.stock_key = None
        self.opened, self.unavailable, self.gone = set(), set(), {}
        self.tool_keys, self.view_keys = {}, {}
        self.edits, self._label_cache = 0, (-1, None)
        self.enum_labels = set()
        pending = [(s, 0) for s in schemas.values()]
        while pending:
            node, level = pending.pop()
            if type(node) is dict and level <= 6:
                for option in node.get("enum") or []:
                    if type(option) is str:
                        self.enum_labels.add(option.strip().lower())
                pending += [(v, level + 1) for v in node.values() if type(v) is dict]

    # -- ids
    def ident(self, node, in_list=False, level=0):
        """The listing id a dict carries, when the dict is a listing record."""
        if type(node) is not dict or len(node) <= 2:
            return None
        if self.id_key is not None and idish(node.get(self.id_key), True):
            return node[self.id_key].strip()
        if level > 0 and not in_list and not [v for v in node.values() if type(v) is dict]:
            return None
        key = id_key_of([node])
        if key is not None and ("id" in str(key).lower() or node[key].strip() in self.rows):
            return node[key].strip()
        return None

    def learn_rows(self, nodes):
        """Whether a list holds listing rows; learns the id key from them the first time."""
        rows = [n for n in nodes if type(n) is dict and len(n) > 2]
        if not rows or 2 * len(rows) < len(nodes):
            return False
        if self.id_key and all(idish(r.get(self.id_key), True) for r in rows):
            return True
        key = id_key_of(rows)
        self.id_key = key or self.id_key
        return key is not None

    def keep(self, ident, node):
        """Merge one record into the ledger (stock flags flipping mark it unavailable or available again)."""
        if ident not in self.rows:
            self.order.append(ident)
            self.rows[ident] = {}
        row = self.rows[ident]
        self.edits += 1
        flips = [(row.get(k), v) for k, v in node.items() if type(v) is bool and type(row.get(k)) is bool]
        if (True, False) in flips:
            self.unavailable.add(ident)
        elif (False, True) in flips:
            self.unavailable.discard(ident)
        row.update(node)
        if facts_in(node):
            self.full[ident] = dict(row)
        elif ident in self.full:
            self.full[ident].update(node)

    def absorb(self, seen):
        """Take every listing record out of an (unpacked) observation."""
        stack = [(seen, 0, False)]
        while stack:
            node, level, listed = stack.pop()
            if level > 6:
                continue
            if type(node) is list:
                rows = self.learn_rows(node)
                stack += [(x, level + 1, rows) for x in reversed(node)]
            elif type(node) is dict:
                ident = self.ident(node, listed, level)
                if ident:
                    self.keep(ident, node)
                else:
                    stack += [(x, level + 1, False) for x in reversed(list(node.values()))]

    def records_in(self, seen):
        """(id, record) for the listing records at the top of an (unpacked) observation or in its lists."""
        def rows_of(rows):
            out = []
            for r in rows:
                ident = self.ident(r, True) if type(r) is dict else None
                if ident:
                    out.append((ident, r))
            return out
        if type(seen) is list:
            return rows_of(seen)
        if type(seen) is not dict:
            return []
        top = self.ident(seen)
        if top:
            return [(top, seen)]
        found = []
        for value in seen.values():
            if type(value) is list:
                found += rows_of(value)
            elif type(value) is dict:
                for inner in value.values():
                    if type(inner) is list:
                        found += rows_of(inner)
        return found

    # -- labels, prices, stock
    def label_key(self):
        """The row key that holds the category label (cached until a record changes)."""
        if self._label_cache[0] != self.edits:
            self._label_cache = (self.edits, self._find_label_key())
        return self._label_cache[1]

    def _find_label_key(self):
        seen, count = {}, {}
        for row in self.rows.values():
            for key, value in row.items():
                if type(value) is str and 0 < len(value) <= 40 and re.search(r"\d{3}", value) is None:
                    seen.setdefault(key, set()).add(value)
                    count[key] = count.get(key, 0) + 1
        floor = 0.8 * max(1, len(self.rows))
        ranked = []
        for key, values in seen.items():
            if count[key] >= floor and len(values) <= 40:
                listed = len([v for v in values if v.strip().lower() in self.enum_labels]) / len(values)
                mean = sum(len(v) for v in values) / len(values)
                ranked.append(((-listed, len(values), -count[key], -mean), key))
        return min(ranked, key=lambda pair: pair[0])[1] if ranked else None

    def label(self, ident):
        key = self.label_key()
        if not key:
            return ""
        return str(self.rows.get(ident, {}).get(key) or "")

    def price(self, ident):
        if not self.price_key:
            return None
        value = self.rows.get(ident, {}).get(self.price_key)
        if type(value) is dict:
            value = next(filter(is_number, value.values()), None)
        return as_number(value)

    def sold_out(self, ident):
        return ident in self.gone or ident in self.unavailable

    def buyable(self, ident):
        if self.sold_out(ident):
            return False
        stock = self.rows.get(ident, {}).get(self.stock_key) if self.stock_key else None
        if stock is False:
            return False
        return not (is_number(stock) and stock <= 0)

    def record(self, ident, size=400, total=3000):
        found = self.full.get(ident) or self.rows.get(ident)
        if not found:
            return "(no record seen)"
        return jdump(shorten(found, size))[:total]

    # -- views
    def note_view(self, tool, seen, version):
        for ident, record in self.records_in(seen):
            self.tool_keys.setdefault(tool, set()).update(record.keys())
            when, have = self.view_keys.get(ident, (-1, set()))
            have = set(have) if when == version else set()
            self.view_keys[ident] = (version, have | set(record.keys()))

    def shown_already(self, tool, ident, version):
        keys = self.tool_keys.get(tool)
        when, have = self.view_keys.get(ident, (-1, set()))
        if not keys or when != version or ident in self.gone:
            return False
        return keys.issubset(have)


# --------------------------------------------------------------------------------------------- items and the task

class Want:
    """One requested item: its name, store label, quantity, the request's musts and what the shopper told since."""

    def __init__(self, name, shelf, qty, musts):
        self.name, self.shelf, self.qty, self.musts = name, shelf, qty, musts
        self.told, self.dropped, self.free = [], [], []

    def live(self):
        return [m for m in self.musts + self.told if m not in self.dropped]


def prop_of(need):
    return need.split(":", 1)[0].strip()


def value_of(need):
    head, colon, tail = need.partition(":")
    return tail.strip() if colon else ""


def long_words(text):
    return set(w for w in stems(text) if len(w) > 3)


class TaskSide:
    """Reading the request into items, limits, roles and process steps; who owns a store label; the text the walker
    is shown about the task each turn."""

    # -- counts and limits
    def left(self):
        return self.max_steps - self.step

    def clock(self):
        return time.monotonic() - self.started

    def room(self):
        return K.task_s - K.margin_s - self.clock()

    def msgs_free(self):
        kept = max(0, self.reserve - self.gate_msgs)
        return self.msg_limit - self.msg_used - kept

    def judges(self, line=K.soft):
        """Models a judgement may use at the spend so far: heavy then light while cheap, light alone up to hard."""
        spent = self.wire.spent
        if spent > K.hard:
            return []
        if spent <= line and spent + K.heavy_room <= K.hard:
            return [M_HEAVY, M_LIGHT]
        return [M_LIGHT]

    def due(self, stage):
        if (stage, self.cart_mark()) in self.gate_ok:
            return False
        return any(p["when"] == stage for p in self.process)

    def tail(self):
        if self.msg_used < self.msg_limit and any(p["when"] == "order" and p["messages"] > 0 for p in self.process):
            return 2 if self.due("order") else 1
        return 1

    def cart_mark(self):
        return "%s#%d" % (self.cart_state(), self.version)

    def page_tools(self):
        return set(filter(None, (self.roles.get("offer") or self.roles.get("open"), self.roles.get("compare"))))

    def params(self, tool):
        if tool not in self.param_cache:
            self.param_cache[tool] = Params(self.schemas.get(tool or "") or {})
        return self.param_cache[tool]

    def single_target(self, tool):
        """The one free-text argument a single-listing tool takes, when every required argument is that one."""
        if not tool:
            return None
        p = self.params(tool)
        target = next(filter(lambda k: k in p.texts, p.required), None)
        if target is None and len(p.texts) == 1:
            target = p.texts[0]
        if target is None or [k for k in p.required if k != target]:
            return None
        return target

    def facts_tool(self):
        for tool, shows in self.facts_from.items():
            if shows and self.single_target(tool):
                return tool
        tool = self.roles.get("open")
        if self.facts_from.get(tool or "") is False or not self.single_target(tool):
            return None
        return tool

    def compare_bare(self):
        shown = self.facts_from.get(self.roles.get("compare") or "")
        if shown is not None:
            return shown is False
        return self.facts_from.get(self.facts_tool() or "") is True or bool(self.roles.get("offer"))

    # -- reading the request
    def tool_lines(self, skip=frozenset()):
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            if fn["name"] not in skip:
                lines.append("- %s: %s | parameters: %s" % (fn["name"], fn.get("description", ""), jdump(fn.get("parameters") or {})))
        return lines

    @staticmethod
    def brief_ok(brief):
        roles = brief.get("roles")
        return type(roles) is dict and bool(roles.get("buy")) and bool(as_list(brief.get("items")))

    @staticmethod
    def shelves_clash(brief):
        names = [spaced(e.get("shelf") or "") for e in as_list(brief.get("items")) if type(e) is dict]
        names = [n for n in names if n]
        return len(set(names)) < len(names)

    def read_request(self):
        parts = [P_BRIEF, "\nTOOLS:\n", "\n".join(self.tool_lines()), "\n\nREQUEST:\n", self.request]
        if self.view.get("opening") is not None:
            parts += ["\n\nOPENING NOTE FROM THE STORE:\n", jdump(shorten(self.view.get("opening"), 1500))]
        prompt = "".join(parts)
        ask = self.wire.ask
        brief = ask(prompt, [M_LIGHT], 4000) or {}
        if self.wire.no_credit:
            pass
        elif not self.brief_ok(brief):
            brief = ask(prompt, [M_HEAVY, M_LIGHT], 2500) or brief
        elif self.shelves_clash(brief):
            second = ask(prompt, [M_HEAVY], 2500) or {}
            brief = second if self.brief_ok(second) else brief
        self.brief = brief
        roles = brief.get("roles")
        for job, tool in (roles.items() if type(roles) is dict else ()):
            if type(tool) is str and tool in self.schemas:
                self.roles[job] = tool
        brief["buy_args"] = brief.get("buy_args") if type(brief.get("buy_args")) is dict else {}
        brief["refuse_args"] = brief.get("refuse_args") if type(brief.get("refuse_args")) is dict else {}
        self.split_modes()
        self.take_items()
        self.take_limits()

    def split_modes(self):
        """When one tool both checks out and declines and the brief cannot tell its modes apart, ask for them."""
        tool = self.roles.get("buy")
        yes, no = self.brief["buy_args"], self.brief["refuse_args"]
        if not tool or self.roles.get("refuse") != tool or (yes and no and yes != no):
            return
        got = self.wire.ask(P_MODES.format(tool=tool, schema=jdump(self.schemas.get(tool))), [M_LIGHT, M_HEAVY], 1500) or {}
        yes, no = got.get("buy_args"), got.get("refuse_args")
        if type(yes) is dict and type(no) is dict and yes != no:
            self.brief["buy_args"], self.brief["refuse_args"] = yes, no

    def take_items(self):
        renumber, old = {}, 0
        for entry in as_list(self.brief.get("items")):
            if type(entry) is not dict:
                continue
            old += 1
            qty = as_int(entry.get("quantity"))
            want = Want(str(entry.get("name") or "the product"), str(entry.get("shelf") or ""),
                        qty if qty and qty > 0 else 1, [str(m) for m in as_list(entry.get("musts")) if m])
            twin = next((w for w in self.items if self.same_shelf(w, want)), None)
            if twin:
                twin.musts += [m for m in want.musts if m not in twin.musts]
                renumber[old] = self.items.index(twin) + 1
                self.note("items_merged", into=twin.name, dropped=want.name)
            else:
                self.items.append(want)
                renumber[old] = len(self.items)
        self.items = self.items or [Want("the requested product", "", 1, [])]
        count = len(self.items)
        for pair in as_list(self.brief.get("fits")):
            part = renumber.get(as_int(pair.get("item")) or 0) if type(pair) is dict else None
            main = renumber.get(as_int(pair.get("with")) or 0) if type(pair) is dict else None
            if part and main and part != main and max(part, main) <= count and (part, main) not in self.bound:
                self.bound.append((part, main))
        for raw in as_list(self.brief.get("lookups")):
            text = str(raw).strip()
            if not text:
                continue
            if all(text not in w.musts for w in self.items):
                same = [m for w in self.items for m in w.musts if spaced(prop_of(m)) == spaced(prop_of(text))]
                text = same[0] if same else text
            owner = 1
            for number, want in enumerate(self.items, 1):
                if text in want.musts:
                    owner = number
                    break
            if (owner, text) not in self.refs:
                self.refs.append((owner, text))
        self.unmix()

    @staticmethod
    def same_shelf(a, b):
        """Two brief items for one store label whose musts never give one property two values."""
        if not b.shelf.strip() or spaced(b.shelf) != spaced(a.shelf):
            return False
        for x in a.musts:
            for y in b.musts:
                if spaced(prop_of(x)) == spaced(prop_of(y)) and spaced(value_of(x)) != spaced(value_of(y)):
                    return False
        return True

    def unmix(self):
        """Musts that only describe the listing a pointer must points at move inside that pointer must."""
        for number in range(1, len(self.items) + 1):
            want = self.items[number - 1]
            pointing = []
            for owner, need in self.refs:
                if owner == number and need in want.musts:
                    pointing.append(need)
            others = list(filter(lambda m: m not in pointing, want.musts))
            if len(pointing) == 0 or len(others) == 0:
                continue
            reply = self.wire.ask(compose(P_UNMIX, ("REQUEST", self.request), ("PRODUCT", want.name),
                                          ("REQUIREMENTS POINTING AT ANOTHER LISTING", "\n".join(pointing)),
                                          ("OTHER REQUIREMENTS", "\n".join(others))), [M_HEAVY, M_LIGHT], 1500) or {}
            said = [str(x).strip() for x in as_list(reply.get("describe_other"))]
            moved = [m for m in others if m in said]
            if len(moved) == 0:
                continue
            head = pointing[0]
            joined = head + " (that listing: " + "; ".join(moved) + ")"
            kept = []
            for must in want.musts:
                if must in moved:
                    continue
                kept.append(joined if must == head else must)
            want.musts = kept
            fixed = []
            for owner, need in self.refs:
                fixed.append((owner, joined if owner == number and need == head else need))
            self.refs = fixed
            self.note("unmix", item=number, moved=moved)

    def take_limits(self):
        brief = self.brief
        raw = brief.get("budget")
        amount = None if raw is True or raw is False else as_number(raw)
        if amount and amount > 0:
            self.budget = raw if type(raw) is int else amount
        limit = as_int(brief.get("messages"))
        if limit is None or limit < 0:
            limit = 3 if self.roles.get("ask") else self.msg_limit
        self.msg_limit = limit
        steps = as_int(brief.get("steps"))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in as_list(brief.get("process")):
            what = text_at(entry, "step")
            if not what:
                continue
            when = text_at(entry, "when").lower()
            self.process.append(dict(step=what, when=when if when in ("choose", "order", "end") else "order",
                                     messages=max(0, as_int(entry.get("messages")) or 0)))
        talking = [p["messages"] for p in self.process if p["messages"] and p["when"] != "choose"]
        if self.roles.get("ask") and talking:
            self.reserve = min(self.msg_limit, max(talking))

    # -- which item a store label belongs to
    def owner_of(self, label):
        if not label:
            return 0
        if label in self.adopted:
            return self.adopted[label]
        plain, mine = spaced(label), roots(label)
        best, top = 0, 0.0
        for number, want in enumerate(self.items, 1):
            if plain == spaced(want.shelf or ""):
                score = 3.0
            elif plain == spaced(want.name):
                score = 2.0
            else:
                theirs = roots(want.name) | roots(want.shelf or "")
                score = len(mine & theirs) / max(1, len(mine | theirs))
            if score > top:
                best, top = number, score
        return best

    def owns(self, number, label):
        return label != "" and self.owner_of(label) == number

    def adopt_labels(self):
        """With 2+ items: labels of read listings that no item owns go to the one item owning no label."""
        if len(self.items) < 2 or not self.books.label_key():
            return
        labels = set(self.books.label(i) for i in self.books.full)
        labels.discard("")
        strays = [x for x in labels if self.owner_of(x) == 0]
        if not strays:
            return
        holders = set(self.owner_of(x) for x in labels if x not in self.adopted)
        orphans = [n for n in range(1, len(self.items) + 1) if n not in holders]
        if len(orphans) == 1:
            for x in strays:
                self.adopted[x] = orphans[0]
            self.note("adopted", item=orphans[0], labels=strays)

    def item_ids(self, number):
        if self.books.label_key() is None:
            return []
        return [i for i in self.books.order if self.owns(number, self.books.label(i))]

    def home_label(self, number):
        """The item's own label (spaced): the request's shelf or name when a listing carries it, else the label most
        of its read listings (or, with none read, its rows) carry."""
        if self.books.label_key() is None:
            return ""
        want = self.items[number - 1]
        for pool in (self.books.full, self.books.rows):
            tally = {}
            for ident in pool:
                label = self.books.label(ident)
                if self.owns(number, label):
                    tally[spaced(label)] = tally.get(spaced(label), 0) + 1
            for name in (spaced(want.shelf), spaced(want.name)):
                if name in tally:
                    return name
            if tally:
                return max(tally, key=tally.get)
        return ""

    def needs_now(self):
        out = []
        for number, want in enumerate(self.items, 1):
            out.extend((number, need) for need in want.live())
        return out

    def pointer_needs(self):
        return [need for _, need in self.refs]

    def described(self, number, need):
        """A must in the shopper's own words that no attribute key of the item's read listings could state."""
        lookups = [str(x) for x in as_list(self.brief.get("lookups"))]
        prop, _, value = need.partition(":")
        if need in self.pointer_needs() or need in lookups or not prop.strip():
            return False
        if len(re.findall(r"[a-z0-9]+", value.lower())) < 4 or re.search(r"\b(?:listing|page|shown|other than|unlike|same as)\b", value, re.I):
            return False
        want = self.items[number - 1]
        for other in self.items:
            if other is not want and long_words(other.name) & stems(value):
                return False
        labelled = self.books.label_key() is not None
        read, keys = 0, set()
        for ident, record in self.books.full.items():
            block = facts_in(record)
            if (labelled and not self.owns(number, self.books.label(ident))) or not block:
                continue
            read += 1
            keys.update(spaced(k) for k in block)
        if read < 3:
            return False
        return not any(stems(k) & stems(prop) for k in keys)

    # -- what the walker is told
    def item_line(self, number, want, many):
        live = want.live()
        loose = [m for m in live if self.described(number, m)]
        firm = []
        for need in live:
            if need in loose:
                continue
            if need in self.ref_values:
                need += " (the page it points at gives: %s)" % self.ref_values[need]
            firm.append(need)
        head = ("item %d: " % number if many else "item: ") + want.name
        if want.shelf:
            head += " (store label: %s)" % want.shelf
        text = "%s, quantity %d; requirements: %s" % (head, want.qty, "; ".join(firm) or "none beyond the kind of product")
        if loose:
            text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + "; ".join(loose)
        return text

    def ledger(self):
        many = len(self.items) > 1
        lines = [self.item_line(n, w, many) for n, w in enumerate(self.items, 1)]
        brief = self.brief
        if self.budget is not None:
            scope = "for each item" if brief.get("budget_scope") == "each" else "in total"
            lines.append(" ".join(("budget: at most %s %s %s" % (self.budget, brief.get("budget_unit") or "", scope)).split()))
        pick = brief.get("pick")
        if pick == "cheapest":
            lines.append("pick: the cheapest listing that fits (lowest price)")
        elif pick in ("most", "least"):
            lines.append(" ".join(("pick: the %s %s among fitting listings" % (pick, brief.get("pick_measure") or "")).split()))
        else:
            lines.append("pick: any listing that fits everything")
        if brief.get("in_stock_only", True) is not False:
            lines.append("in stock only")
        lines.extend("wish (tie-breaker only): %s" % w for w in as_list(brief.get("wishes")))
        lines.extend("rule from the request: %s" % r for r in as_list(brief.get("rules")))
        if brief.get("quiet_decline") is True:
            lines.append("if nothing fits: end without naming or suggesting any listing")
        timing = dict(choose="before choosing", order="at the order (the code does it when you check out)",
                      end="at an ending without a purchase (the code does it then)")
        lines.extend("step the request asks for, %s: %s" % (timing[p["when"]], p["step"]) for p in self.process)
        source = self.source_line()
        if source:
            lines.append(source)
        return "\n".join("- " + x for x in lines)

    def source_line(self):
        facts = self.facts_tool()
        if not facts or not (self.facts_from.get(facts) is True or self.roles.get("offer")):
            return ""
        views = self.page_tools()
        bare = sorted(t for t in views | {self.roles.get("offer") or ""} if t and self.facts_from.get(t) is False)
        pages = sorted(views - {facts})
        text = "listing attributes come from " + facts
        if bare:
            text += "; %s show none" % ", ".join(bare)
        if pages:
            text += "; a listing's page (what a chain of pages or a page check means) is shown by " + " or ".join(pages)
        return text

    def status(self):
        left, tail = self.left(), self.tail()
        out = ["[code] step %d of %d used, %d left." % (self.step, self.max_steps, left)]
        empty = self.buys == 0
        if left <= tail + 1:
            out.append(" FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only "
                       "when your search showed that an item has no fitting listing (running out of steps does not show that).")
        elif left <= 6 or (empty and left <= len(self.items) + 5):
            out.append(" Few steps left: set the cart to the chosen lines and check out now.")
        elif empty and left <= 12:
            out.append(" Steps are running low: settle on your picks, fill the cart and check out within the next few steps.")
        elif empty and not self.cart_touched and 2 * self.step >= self.max_steps - self.max_steps % 2:
            out.append(" Half the steps are used and the cart is empty: cart each item's fitting listing now (the store "
                       "may still sell one out).")
        if left > tail + 1 and self.clock() >= K.wind_down:
            out.append(" TIME IS NEARLY UP: set the cart to the chosen lines and check out within the next two turns; an "
                       "episode that runs out of time earns nothing.")
        live = set(need for _, need in self.needs_now())
        unresolved = [need for need in self.pointer_needs() if need in live and need not in self.ref_values]
        if unresolved and empty:
            out.append("\n[resolve first: open the listing each of these points at, and every listing its page refers to; "
                       "the last page gives the value] " + " | ".join(unresolved[:3]))
        if self.leads:
            out.append("\n[review candidates] " + ", ".join(self.leads))
        if self.bound and empty:
            plan = self.fit_plan()
            if plan:
                out.append("\n" + plan)
        if empty:
            out.append(self.seen_fits())
        if self.notes:
            out.append("\n[changes so far] " + " | ".join(self.notes[-4:]))
        out.append("\n[ledger]\n" + self.ledger())
        return "".join(out)

    def seen_fits(self):
        """E5: per item still without a cart line, the read buyable listings whose attributes state every requirement."""
        books = self.books
        if books.label_key() is None:
            return ""
        held = set(self.cart_ids())
        parts = []
        for number in range(1, len(self.items) + 1):
            if any(self.owns(number, books.label(i)) for i in held):
                continue
            fits = [i for i in self.item_ids(number) if i in books.full and books.buyable(i) and self.states_needs(i, number)]
            if fits:
                tag = "item %d (%s)" % (number, self.items[number - 1].name)
                parts.append("%s: %s" % (tag, ", ".join(fits[:3])))
        if not parts:
            return ""
        return ("\n[fits seen, not carted: their attributes state every requirement word for word; cart one per item now "
                "(cheapest pick: compare prices first)] " + " | ".join(parts))


# ------------------------------------------------------------------------------------------ reading the store

HEDGED_VALUE = re.compile(r"\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below"
                          r"|above|less|more|between|within|up to|or)\b", re.I)
PRICED_WORDS = re.compile(r"price|cost|amount|budget|spend|money", re.I)
CAP_WORDS = re.compile(r"max|most|under|below|limit|cap|upto|up_to", re.I)


class ShopSide:
    """Observations: records, store notes, result shapes and paging, the code's own reads and searches, and the lines
    the code adds to a result for the walker."""

    def show(self, seen):
        """A result as the walker sees it: listings written as text lines shown as records, long strings cut."""
        text = jdump(shorten(unpack(seen), K.cut_text))
        if len(text) > K.cut_result:
            text = text[:K.cut_result] + "...(cut)"
        return text

    def observe(self, tool, seen):
        """Absorb a result of the code's own call; a store note in it is read and queued for the walker."""
        flat = unpack(seen)
        self.books.absorb(flat)
        lines = self.loose_lines(flat)
        if not lines:
            return ""
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append("(seen on %s) %s" % (tool, meaning))
        return meaning

    def loose_lines(self, flat):
        """Free text a result carries outside listing records (5+ words), not read before."""
        if type(flat) is not dict or self.books.ident(flat):
            return []
        found = []
        for value in flat.values():
            if type(value) is str:
                if len(value.split()) > 4:
                    found.append(value)
            elif type(value) is list:
                if value and all(type(v) is str for v in value):
                    found.extend(v for v in value if len(v.split()) > 4)
            elif type(value) is dict and not self.books.ident(value, False, 1):
                found.extend(v for v in value.values() if type(v) is str and len(v.split()) > 5)
        return [x for x in found if x not in self.heard]

    def take_note(self, lines):
        """Read a store note into budget, requirement and stock changes; returns its meaning for the walker."""
        self.heard.update(lines)
        answer = self.wire.ask(compose(P_NOTE, ("ITEMS AND REQUIREMENTS NOW", self.ledger()), ("THE NOTE", "\n".join(lines))),
                               [M_LIGHT, M_SPARE], 2500) or {}
        rows = self.books.rows
        gone = [str(g) for g in as_list(answer.get("gone")) if type(g) in (str, int)]
        self.books.unavailable.update(filter(lambda g: g in rows, gone))
        self.note_sold = len(gone) > 0 or answer.get("sold_out") is True
        self.note_named = len([g for g in gone if g in rows]) > 0
        if len(lines) == 1 and self.note_sold:
            self.sold_lines.update(lines)
        if answer.get("kind") not in (None, "change"):
            return ""
        changes = self.apply_changes(answer)
        if gone:
            changes.append("unavailable: " + ", ".join(gone))
        if answer.get("price_moved") is True:
            changes.append("a price changed")
        summary = text_at(answer, "summary")
        cheaper_again = gone or self.note_sold or any(c.startswith(("budget now", "dropped ")) for c in changes)
        if self.brief.get("pick") == "cheapest" and cheaper_again:
            self.wider_held = False
            self.cheaper_marks = set()
        if changes or summary:
            self.version, self.recent = self.version + 1, []
            self.notes.append(summary if summary else "; ".join(changes))
        tail = " (%s)" % "; ".join(changes) if changes else ""
        return (summary + tail).strip()

    def item_no(self, value):
        number = as_int(value) or 1
        return min(len(self.items), max(1, number))

    # -- result shapes and paging
    def shape_of(self, tool, flat):
        """Where a search result keeps its rows, total, next-page value, and a row's price and stock (one model
        call per tool, the first time a list of 2+ rows comes back)."""
        if tool in self.shapes or type(flat) is not dict:
            return self.shapes.get(tool)
        lists = []
        for key, value in flat.items():
            if type(value) is list and len(value) >= 2 and not [r for r in value if type(r) is not dict]:
                lists.append(key)
        if not lists:
            return None
        self.shapes[tool] = None
        sample = {}
        for key, value in flat.items():
            sample[key] = value[:3] if key in lists else value
        head = "%s\n\nTOOL: %s | parameters: %s" % (P_SHAPE, tool, jdump(self.schemas.get(tool) or {}))
        answer = self.wire.ask(compose(head, ("RESULT (first rows only)", jdump(shorten(sample, 120))[:4000])), [M_LIGHT, M_HEAVY], 2500) or {}
        if answer.get("rows") not in lists:
            return None
        rows = answer["rows"]
        where = dict(count=flat, next=flat, next_arg=self.params(tool).props, price=flat[rows][0], stock=flat[rows][0])
        shape = dict(rows=rows)
        for field, place in where.items():
            key = answer.get(field)
            shape[field] = key if type(key) is str and key in place else None
        if shape["count"] is not None and type(flat[shape["count"]]) is not int:
            shape["count"] = None
        self.shapes[tool] = shape
        if self.books.price_key is None:
            self.books.price_key = shape["price"]
        if self.books.stock_key is None:
            self.books.stock_key = shape["stock"]
        self.note("shape", tool=tool, shape=shape)
        return shape

    def paging(self, tool, flat):
        """(total, rows shown, page parameter, next value) of a paged result, else None."""
        shape = self.shape_of(tool, flat)
        if type(flat) is not dict or not shape or not (shape["count"] and shape["next"] and shape["next_arg"]):
            return None
        rows, total, value = flat.get(shape["rows"]), flat.get(shape["count"]), flat.get(shape["next"])
        if type(rows) is not list or not rows or type(total) is not int:
            return None
        if value is not None and type(value) not in (str, int):
            return None
        return total, len(rows), shape["next_arg"], value

    def price_ordered(self, flat):
        prices = [self.books.price(i) for i, _ in self.books.records_in(flat)]
        if len(prices) < 3 or None in prices:
            return False
        return prices == sorted(prices)

    def more_pages(self, tool, args, flat):
        """Under a cheapest/most/least pick, read the next pages of a short result (one step each)."""
        self.paged = []
        pick = self.brief.get("pick")
        plan = self.paging(tool, flat) if pick in ("cheapest", "most", "least") else None
        if plan is None:
            return ""
        total, shown, param, value = plan
        given = args.get(param)
        paged_on = given not in (0, 1) if type(given) is int else len(str(given or "").strip()) > 0
        if paged_on or value is None or shown >= total or (pick == "cheapest" and self.price_ordered(flat)):
            return ""
        pages = self.page_budget(total, shown)
        if not pages:
            return ("\n[code] This search matched %d listings and showed %d, in no price order. The pick counts every "
                    "fitting listing: search with narrower words (the requirement's value) or page on with %s=%s."
                    % (total, shown, param, jdump(value)))
        read = []
        for _ in range(pages):
            reply, sent = self.use(tool, dict(args, **{param: value}))
            if self.finished or sent is None or reply is None:
                break
            seen = reply.get("observation")
            self.pages_read += 1
            self.paged.append(unpack(seen))
            body = self.show(seen)
            self.read_cache[tool + jdump(sent)] = (self.version, self.step, body)
            read.append("%s=%s: %s" % (param, jdump(value), body))
            after = self.paging(tool, self.paged[-1])
            value = None if after is None else after[3]
            if value is None:
                break
        self.note("pages", tool=tool, read=len(read))
        if len(read) == 0:
            return ""
        tail = ""
        if value is not None:
            tail = ("\n[code] %d listings matched and the code has read %d rows of them; narrow the search with the "
                    "requirement's value or page on to see the rest." % (total, shown * (1 + len(read))))
        return "\n[the code read the next pages of this search itself, one step a page]\n" + "\n".join(read) + tail

    def cap_ignored(self, tool, args, flat):
        """A price cap the search did not apply, with the enum switch that would list the category under it."""
        params = self.params(tool).props
        caps = []
        for key, value in args.items():
            spec = params.get(key)
            if not is_number(value) or spec is None or not CAP_WORDS.search(key):
                continue
            about = str(spec.get("description") or "") if type(spec) is dict else ""
            if PRICED_WORDS.search(key + " " + about):
                caps.append(key)
        if not caps or not self.books.price_key:
            return ""
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = len([i for i, _ in self.books.records_in(flat) if (self.books.price(i) or 0) > cap])
        if not over:
            return ""
        switches = []
        for key, value in args.items():
            options = (params.get(key) or {}).get("enum") if type(value) is str else None
            if type(options) is list:
                switches.append((key, [o for o in options if o != value]))
        hint = ""
        if any(o for _, o in switches):
            hint = "; list the category with %s and the same %s to see only cheaper listings" % (
                ", ".join("%s=%s" % (k, jdump(o[0])) for k, o in switches if o), caps[0])
        return "\n[code] This search did not apply %s=%g: %d of its rows cost more%s." % (caps[0], cap, over, hint)

    def cheapest_rows(self, flats):
        books = self.books
        if self.brief.get("pick") != "cheapest" or books.price_key is None:
            return ""
        best = {}
        for flat in flats:
            for ident, _ in books.records_in(flat):
                price = books.price(ident)
                if price is not None and books.buyable(ident):
                    best[ident] = price
        if len(best) <= 1:
            return ""
        order = sorted(best, key=lambda i: (best[i], i))[:6]
        return "\n[code] buyable rows here, cheapest first: " + ", ".join("%s (%g)" % (i, best[i]) for i in order)

    def evidence(self, flat):
        """For each fully read listing in a result: its attribute keys whose names hold a requirement's property, and
        whether its fit text covers a model code a requirement names."""
        books, found = self.books, []
        frontier = [(flat, 0)]
        while frontier:
            node, level = frontier.pop(0)
            if level > 6 or type(node) not in (dict, list):
                continue
            ident = books.ident(node, False, level) if type(node) is dict else None
            if ident and facts_in(node):
                found.append((ident, level))
                continue
            children = node.values() if type(node) is dict else node
            frontier[0:0] = [(c, level + 1) for c in children]
        ids = list(dict.fromkeys(i for i, _ in found))[:12]
        needs = self.needs_now()[:8]
        if not (ids and needs):
            return ""
        tagged = len(self.items) > 1
        index = self.fit_index() if any(self.needed_codes(n) for n in range(1, len(self.items) + 1)) else None
        lines = []
        for ident in ids:
            block = facts_in(books.full.get(ident) or {}) or {}
            bits = []
            for number, need in needs:
                prop = prop_of(need)
                keys = [k for k in block if names_same(stems(prop), stems(k))][:3]
                if keys:
                    lead = "(item %d) " % number if tagged else ""
                    bits.append(lead + prop + " -> " + ", ".join("%s=%s" % (k, str(block[k])[:50]) for k in keys))
            if index is not None:
                mark = index["marks"].get(ident) or {}
                if mark.get("works") or mark.get("series"):
                    for number in self.owners(ident):
                        for code in self.needed_codes(number):
                            bits.append("fits %s by its fit text: %s" % (code.upper(), "yes" if self.part_fits(ident, index, [code]) else "NO"))
            if bits:
                lines.append("%s: %s" % (ident, "; ".join(bits)))
        if not lines:
            return ""
        return ("[evidence: keys whose names contain a requirement's property; one naming a different property (another "
                "part or aspect) decides nothing, and a key with another name may still hold the same property] " +
                " | ".join(lines))

    # -- the code's own reads
    def read_facts(self, ids, keep):
        """Open unread listings with the attribute tool, one step each, keeping `keep` steps."""
        tool = self.facts_tool()
        target = self.single_target(tool)
        if not (tool and target):
            return []
        books, texts = self.books, []
        todo = [i for i in ids if i in books.rows and i not in books.full and not books.sold_out(i)]
        for ident in todo:
            if self.finished or self.left() <= keep or self.facts_from.get(tool) is False:
                break
            reply, sent = self.use(tool, {target: ident})
            if sent is None or reply is None:
                break
            seen = reply.get("observation")
            flat = unpack(seen)
            books.note_view(tool, flat, self.version)
            if type(flat) is dict and books.ident(flat):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(facts_in(flat))
            texts.append("%s: %s" % (ident, self.show(seen)[:1500]))
        if texts:
            self.note("fill_facts", n=len(texts))
        return texts

    def shelf_words(self):
        """Item names, item shelves and every seen label (spaced): words a value search must not keep fixed."""
        label_key = self.books.label_key()
        names = set(spaced(w.shelf or "") for w in self.items) | set(spaced(w.name) for w in self.items)
        if label_key:
            names |= set(spaced(str(r.get(label_key) or "")) for r in self.books.rows.values())
        return names

    def search_plan(self):
        """(tool, fixed arguments, words field) copied from the walker's latest words search."""
        tool = self.roles.get("search")
        p = self.params(tool) if tool else None
        if p is None:
            return None
        queries = [k for k in p.texts if re.search(R_QUERY, snake(k), re.I)]
        label_key = self.books.label_key()
        labels = set()
        for row in (self.books.rows.values() if label_key else ()):
            labels.add(str(row.get(label_key) or "").strip().lower())
        labels.discard("")
        history = self.searched

        def usable(k):
            if queries:
                return True
            used = [str(a[k]) for a in history if a.get(k)]
            multi = [a for a in history if len(str(a.get(k)).split()) >= 2]
            listed = [a for a in history if str(a.get(k) or "").strip().lower() in labels]
            return len(set(used)) > 1 and len(multi) > 0 and len(listed) == 0
        for args in reversed(history):
            fields = [k for k in (queries or p.texts) if type(args.get(k)) is str and re.search(r"[a-z]", args[k], re.I) and usable(k)]
            if fields:
                return tool, self.fixed_args(p, args, fields[0]), fields[0]
        return None

    def value_plan(self):
        """The search a value search copies; when every search so far listed a category without words, the tool's one
        query field with its words mode set. None when the schema leaves a doubt."""
        found = self.search_plan()
        tool = self.roles.get("search")
        if found is not None or tool is None or len(self.searched) == 0:
            return found
        p = self.params(tool)
        queries = [k for k in p.texts if re.search(R_QUERY, snake(k), re.I)]
        if len(queries) != 1:
            return None
        last = self.searched[-1]
        modes = []
        for key in p.enums:
            options = p.enums[key]
            reading = list(filter(lambda o: type(o) is str and re.search(R_QUERY, o, re.I), options))
            if len(reading) == 1:
                modes.append((key, reading[0]))
                continue
            flipped = [o for o in options if o != last.get(key)]
            if key in p.required and len(options) == 2 and len(flipped) == 1:
                modes.append((key, flipped[0]))
        if len(modes) > 1 or (len(p.enums) > 0 and len(modes) == 0):
            return None
        base = {}
        for key in p.required:
            if key in last and key != queries[0]:
                base[key] = last[key]
        for key, option in modes:
            base[key] = option
        return tool, base, queries[0]

    def value_words(self, number):
        """(search words, told by the shopper) for each of the item's requirement values a words search can match;
        an answer's lower limit with a unit searches its amount."""
        told = self.items[number - 1].told
        pointers = self.pointer_needs()
        out = []
        for owner, need in self.needs_now():
            if owner != number or ":" not in need or need in pointers:
                continue
            value = value_of(need)
            floor = re.search(R_FLOOR, value, re.I)
            if floor:
                if need in told:
                    out.append((floor.group(1).strip(), True))
            elif value and not HEDGED_VALUE.search(value):
                out.append((value, need in told))
        return out

    def value_tries(self, number):
        pairs = self.value_words(number)
        fresh = [v for v, told in pairs if told]
        if not fresh:
            return []
        lead = [v for v in fresh if len(v.split()) == 1 or re.fullmatch(R_AMOUNT, v, re.I)]
        values = lead + [v for v, _ in pairs if v not in lead]
        if len(values) > 1:
            stated = [n for n, v in enumerate(values) if v not in fresh]
            cut = stated[-1] if stated else 0
            return [values, values[:cut] + values[cut + 1:]]
        if lead:
            return [[values[0], self.items[number - 1].name], values]
        return [values]

    def value_search(self, numbers):
        """After new needs: search each such item's values together and open the first rows of its kind."""
        plan = self.value_plan()
        if plan is None or self.left() < 22:
            return ""
        tool, base, field = plan
        books = self.books
        seen, allowance = [], 8
        for number in numbers:
            tries = self.value_tries(number) if allowance > 2 else []
            for words in tries[:2]:
                if not words:
                    continue
                query = dict(base)
                query[field] = " ".join(words)
                if self.finished or self.fit_args(tool, query) is None:
                    break
                reply, sent = self.use(tool, query)
                allowance -= 1
                if reply is None:
                    break
                found = [i for i, _ in books.records_in(unpack(reply.get("observation")))]
                cap = self.budget if is_number(self.budget) else None
                prices = [books.price(i) for i in found]
                if cap is not None and prices and all(x is not None and x > 10 * cap for x in prices):
                    cap = None
                labelled = books.label_key() is not None
                rows = []
                for ident in found:
                    if not books.buyable(ident):
                        continue
                    if labelled and not self.owns(number, books.label(ident)):
                        continue
                    if cap is not None and (books.price(ident) or 0) > cap:
                        continue
                    rows.append(ident)
                home = self.home_label(number)
                if labelled and home:
                    rows.sort(key=lambda i: spaced(books.label(i)) != home)
                seen.append("%s: %s" % (jdump(sent.get(field)), ", ".join(rows[:6]) or "no row of that kind"))
                if rows:
                    opened = self.read_facts([i for i in rows if i not in books.full][:min(3, allowance)], self.tail() + 16)
                    allowance -= len(opened)
                    if opened:
                        seen.append("opened: " + " | ".join(opened)[:2400])
                    break
        if not seen:
            return ""
        self.note("value_search", seen=[s[:200] for s in seen])
        return "[the code searched each such item's requirement values together and read the first fitting-kind rows] " + "\n".join(seen)

    # -- cheaper candidates under a cheapest pick
    def wider_hold(self):
        if self.wider_held or self.brief.get("pick") != "cheapest" or self.left() < 14 + 2 * len(self.items):
            return ""
        self.wider_held = True
        priced = ["%s at %g" % (i, self.books.price(i)) for i in self.cart_ids() if self.books.price(i) is not None]
        below = " (%s)" % "; ".join(priced) if priced else ""
        return ("Not run yet (no step used). %d steps are left and the pick is the cheapest. Searches may read only part "
                "of a listing, so a requirement's value finds only some of the fitting listings, and rows come in no price "
                "order. Before checking out, run a few more searches for each item under other names for the product (its "
                "plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows "
                "priced below the cart's line%s; read the attributes of the cheapest such rows, swap in any that fits, "
                "then check out." % (self.left(), below))

    def cap_plan(self):
        """The listing tool taking a shelf and a maximum price, with the mode an earlier shelf listing used (else the
        one option naming a listing); None when no tool or mode is clear."""
        for tool in self.schemas:
            p = self.params(tool)
            caps = []
            for key, spec in p.props.items():
                if type(spec) is dict and {"integer", "number"} & set(kinds_of(spec)) and re.search(R_CAP_ARG, snake(key), re.I) \
                        and not re.search(R_CAP_NOT, key, re.I):
                    caps.append(key)
            shelves = [k for k in p.texts if re.search(R_SHELF_ARG, snake(k), re.I)]
            if len(caps) != 1 or len(shelves) != 1:
                continue
            shelf, base = shelves[0], {}
            for key, options in p.enums.items():
                choice = None
                for done in self.searched[::-1]:
                    if shelf not in done or key not in done:
                        continue
                    worded = False
                    for k, v in done.items():
                        enum_param = type(p.props.get(k)) is dict and "enum" in p.props[k]
                        if k != shelf and not enum_param and type(v) is str and v.strip() and re.search(R_QUERY, snake(k), re.I):
                            worded = True
                    if not worded:
                        choice = done[key]
                        break
                if choice is None:
                    listing = [o for o in options if type(o) is str and re.search(R_LIST_MODE, o, re.I)]
                    if len(listing) != 1:
                        return None
                    choice = listing[0]
                base[key] = choice
            return tool, base, shelf, caps[0]
        return None

    def list_below(self, cart):
        plan = self.cap_plan()
        books = self.books
        if plan is None or books.label_key() is None:
            return
        tool, base, shelf, cap = plan
        for line in cart:
            price, label = books.price(line), books.label(line)
            if price is None or price < 2 or not label or (line, price) in self.capped:
                continue
            if self.finished or self.left() <= self.tail() + 5:
                return
            self.capped.add((line, price))
            query = dict(base)
            query[shelf] = label
            query[cap] = int(price) - 1 if float(price).is_integer() else int(price)
            reply, sent = self.use(tool, query)
            if sent is None or reply is None:
                return
            self.note("list_below", line=line, price=price)
            page = unpack(reply.get("observation"))
            more = self.paging(tool, page)
            if not more or more[3] is None or self.left() <= self.tail() + 5:
                continue
            if more[1] < more[0] <= 2 * more[1] and not self.price_ordered(page):
                following = dict(sent)
                following[more[2]] = more[3]
                self.use(tool, following)

    def cheaper_reads(self):
        """Under a cheapest pick, once per cart version: list each line's label below its price, then read up to two
        cheaper unread rows of the same label per line."""
        books = self.books
        if self.brief.get("pick") != "cheapest" or books.label_key() is None:
            return []
        mark = self.cart_mark()
        if mark in self.cheaper_marks:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        self.list_below(cart)
        picks = []
        for line in cart:
            price, kind = books.price(line), books.label(line).strip().lower()
            if price is None or not kind:
                continue
            pool = []
            for ident in books.order:
                if ident in cart or ident in books.full or not books.buyable(ident):
                    continue
                other = books.price(ident)
                if books.label(ident).strip().lower() == kind and other is not None and other < price:
                    pool.append(ident)
            pool.sort(key=lambda i: books.price(i) or 0.0)
            for ident in pool[:2]:
                if ident not in picks:
                    picks.append(ident)
        return self.read_facts(picks[:min(6, K.cheaper_opens * max(1, len(cart)))], self.tail() + 4)

    def show_links(self, ids, keep):
        """Search the part code a cart line sends the reader to (once per code) and read the page found; multi-line
        carts only. The result feeds the ledger."""
        plan = self.value_plan()
        if not plan or max(len(ids), len(self.items)) < 2:
            return
        tool, base, field = plan
        for ident in ids:
            index = self.fit_index()["marks"]
            code = index.get(ident, {}).get("more")
            owned = set(mark.get("own") for mark in index.values())
            if not code or code in owned or code in self.link_tried:
                continue
            if self.left() <= keep or self.finished:
                return
            self.link_tried.add(code)
            reply, sent = self.use(tool, dict(base, **{field: code}))
            if reply is None or sent is None:
                return
            self.note("show_link", line=ident, code=code)
            index = self.fit_index()["marks"]
            self.read_facts([i for i in index if index[i].get("own") == code], max(self.tail() + 6, keep))

    def apply_changes(self, answer):
        """Budget, dropped and added requirements a store note reports; returns their descriptions."""
        said = []
        budget = answer.get("budget")
        if is_number(budget) and budget > 0 and self.budget != budget:
            self.budget = budget
            said.append("budget now %s" % budget)
        for entry in [e for e in as_list(answer.get("drop")) if text_at(e, "need")]:
            want, target = self.items[self.item_no(entry.get("item")) - 1], text_at(entry, "need")
            live = want.live()
            same = [m for m in live if m == target] or [m for m in live if spaced(prop_of(m)) == spaced(prop_of(target))][:1]
            want.dropped.extend(same if same else [target])
            said.append("dropped " + target)
        for entry in [e for e in as_list(answer.get("add")) if text_at(e, "need")]:
            want, need = self.items[self.item_no(entry.get("item")) - 1], text_at(entry, "need")
            while need in want.dropped:
                want.dropped.remove(need)
                break
            if need not in want.musts + want.told:
                want.told.append(need)
            said.append("added " + need)
        return said

    def page_budget(self, total, shown):
        """How many next pages the code reads itself: 0 when it should only tell the walker how to page."""
        spare = self.left() - self.tail()
        wanted = -((shown - total) // shown)
        if wanted > 2 and K.auto_pages >= self.pages_read + 2 and spare > 13:
            wanted = 2
        if wanted > 2 or wanted + self.pages_read > K.auto_pages or spare - wanted < 12:
            return 0
        return wanted

    def fixed_args(self, p, args, field):
        """The arguments of a past search a new words search keeps: required ones and enum modes, but no shelf text that
        names an item or a seen label; each enum takes the value most words searches used."""
        shelves = self.shelf_words()
        longs = [long_words(s) for s in shelves if s]
        keep = {}
        for key, value in args.items():
            enum_list = type((p.props.get(key) or {}).get("enum")) is list
            if key == field or not (enum_list or key in p.required):
                continue
            plain = key in p.required or type(value) is not str
            if plain or (spaced(value) not in shelves and not any(long_words(value) & s for s in longs)):
                keep[key] = value
        worded = [a for a in self.searched if type(a.get(field)) is str and re.search(r"[a-z]", a[field], re.I)]
        for key in [k for k in keep if type((p.props.get(k) or {}).get("enum")) is list]:
            seen = [jdump(a[key]) for a in worded if key in a]
            if seen:
                top = max(seen, key=seen.count)
                keep[key] = [a[key] for a in worded if key in a and jdump(a[key]) == top][0]
        return keep


# ----------------------------------------------------------------------------------------------- the interview

MEASURED = re.compile(r"^\s*[\d.,/x -]+\s*[a-z]{0,6}\s*$")


def respell(text):
    """A label's words with a plural -s (after 3+ letters) and British -our folded, for matching table keys."""
    return re.sub(r"our\b", "or", re.sub(r"(?<=[a-z]{3})s\b", "", spaced(text)))


class AskSide:
    """Questions to the shopper: when, which properties (a table of the keys read listings carry, a planner call and
    a code guard), lists of one item's properties rotated after each reply, hearing the replies, the late question."""

    def ready_to_ask(self):
        if self.asked or not self.roles.get("ask") or self.msgs_free() <= 0:
            return False
        if self.clock() > K.ask_by or self.left() <= 8:
            return False
        if len(self.books.full) > 3 * len(self.items):
            return True
        return self.step >= max(8, self.max_steps // 4)

    def message_args(self, text):
        p = self.params(self.roles.get("ask"))
        target = next((k for k in p.required if k in p.texts), None) or (p.texts[0] if p.texts else None)
        if target is None:
            return None
        args = {target: text}
        for key in p.required:
            if key == target:
                continue
            options = (p.props.get(key) or {}).get("enum")
            if type(options) is not list or not options:
                return None
            args[key] = options[0]
        return self.fit_args(self.roles.get("ask"), args)

    # -- reading more listings before the questions
    def compare_target(self):
        p = self.params(self.roles.get("compare"))
        target = p.arrays[0] if p.arrays else None
        if target is None or [k for k in p.required if k != target]:
            return None
        return target

    def compare_room(self):
        spec = self.params(self.roles.get("compare")).props.get(self.compare_target() or "") or {}
        if type(spec.get("maxItems")) is int:
            return spec["maxItems"]
        if not self.compare_bad:
            return max(4, self.compare_ok)
        return max(2, min(self.compare_bad - 1, max(2, self.compare_ok)))

    def wanted_names(self):
        return [(spaced(w.shelf or ""), spaced(w.name), long_words(w.shelf or "") | long_words(w.name)) for w in self.items]

    def label_is_ours(self, label, wanted):
        plain, longs = spaced(label), long_words(label)
        for shelf, name, words in wanted:
            if plain == shelf or plain == name or longs & words:
                return True
        return bool(self.owner_of(label))

    def item_groups(self):
        """Ids of seen listings grouped by label: the groups the items own (else all), largest first, one per item."""
        groups = {}
        for ident in self.books.order:
            groups.setdefault(self.books.label(ident), []).append(ident)
        wanted = self.wanted_names()
        ours = [ids for label, ids in groups.items() if self.label_is_ours(label, wanted)]
        pool = ours if ours else list(groups.values())
        pool.sort(key=len, reverse=True)
        return pool[:max(1, len(self.items))]

    def widen(self):
        tool, target = self.roles.get("compare"), self.compare_target()
        if not (tool and target) or self.compare_bare():
            return self.widen_singly()
        if self.left() <= 12:
            return None
        spec = self.params(tool).props[target]
        cap = spec["maxItems"] if type(spec.get("maxItems")) is int else 4
        groups = self.item_groups()
        share = max(1, cap // max(1, len(groups)))
        picks = []
        for ids in groups:
            picks += [i for i in ids if i not in self.books.full][:share]
        picks = picks[:cap]
        if len(picks) < 2 or self.fit_args(tool, {target: picks}) is None:
            return None
        before = len(self.books.full)
        reply, _ = self.use(tool, {target: picks})
        if reply is None:
            return None
        flat = unpack(reply.get("observation"))
        self.books.note_view(tool, flat, self.version)
        rows = self.books.records_in(flat)
        if rows:
            self.facts_from[tool] = self.facts_from.get(tool, False) or any(facts_in(r) for _, r in rows)
        self.note("widen", ids=picks, opened=len(self.books.full) - before)
        if rows and self.facts_from.get(tool) is False:
            return self.widen_singly()
        if before == len(self.books.full) and len(picks) > 2 and self.left() > 12:
            self.use(tool, {target: picks[:2]})
        return None

    def widen_singly(self):
        tool = self.facts_tool()
        if not (tool and self.single_target(tool)):
            return None
        books = self.books
        wanted = set()
        for _, need in self.needs_now():
            wanted |= word_set(need.split(":", 1)[-1])
        each = K.widen_each if len(self.items) < 3 else 4
        cheapest = self.brief.get("pick") == "cheapest"
        lanes, heads = [], []
        for ids in self.item_groups():
            read = len([i for i in ids if i in books.full])
            pool = [i for i in ids if i not in books.full and books.buyable(i)]
            if cheapest:
                pool.sort(key=lambda i: float("inf") if books.price(i) is None else books.price(i))
            pool.sort(key=lambda i: -len(wanted & word_set(jdump(books.rows.get(i) or {}))))
            lanes.append(pool[:max(0, each - read)])
            heads.append(max(0, 4 - read))
        first, more = [], []
        for k in range(each):
            for lane, head in zip(lanes, heads):
                if k < len(lane):
                    (first if k < min(head, 4) else more).append(lane[k])
        cap = max(5, 2 * len(lanes) + 2)
        done = self.read_facts(first[:min(cap, 8)], 22)
        self.read_facts((first[cap:] + more)[:max(0, 10 - len(done))], 30)
        return None

    # -- the property table
    @staticmethod
    def twins(name, held, other, slot):
        """Whether two keys of one label are one property: shared words and agreeing values on shared listings."""
        a, b = set(name.split()), set(other.split())
        if not a & b:
            return False

        def agree(x, y):
            if x == y:
                return True
            wx, wy = re.findall(r"[a-z0-9]+", x.lower()), re.findall(r"[a-z0-9]+", y.lower())
            short, long = sorted((wx, wy), key=len)
            return len(short) > 0 and set(short) <= set(long) and len(long) <= 2 * len(short) + 2
        mine, theirs = held["per"], slot["per"]
        both = [i for i in mine if i in theirs and mine[i].strip() and theirs[i].strip()]
        same = len([i for i in both if agree(mine[i], theirs[i])])
        if not (a <= b or b <= a):
            return len(both) > 1 and same == len(both)
        if both:
            return 2 * same > len(both)
        return bool(set(held["values"]) & set(slot["values"]) - {""})

    def key_groups(self):
        """Per label: how many read listings, and per key the listings carrying it, its values, and per listing value."""
        books = self.books
        label_key = books.label_key()
        groups = {}
        for ident in books.order:
            record = books.full.get(ident)
            if not record:
                continue
            label = str(record.get(label_key) or "listings") if label_key else "listings"
            group = groups.setdefault(label, dict(n=0, keys={}))
            group["n"] += 1
            counted = set()
            for key, value in (facts_in(record) or {}).items():
                name = spaced(key)
                if name and name not in counted:
                    counted.add(name)
                    slot = group["keys"].setdefault(name, dict(n=0, values=[], per={}))
                    text = str(value).strip().lower()
                    slot["n"] += 1
                    slot["per"][ident] = text
                    if text not in slot["values"]:
                        slot["values"].append(text)
        return groups

    def key_table(self):
        """(label, key, listings carrying it, sample values) for the attribute keys of read listings, per label the
        items own, best carried first (14 per label)."""
        groups = self.key_groups()
        wanted = self.wanted_names()
        chosen = [label for label in groups if self.label_is_ours(label, wanted)] or list(groups)
        chosen.sort(key=lambda x: groups[x]["n"], reverse=True)
        table = []
        for label in chosen:
            keys = groups[label]["keys"]
            merged = []
            for name in sorted(keys, key=lambda k: (-keys[k]["n"], len(k))):
                slot = keys[name]
                values = list(slot["per"].values())
                if not values or self.unaskable(values):
                    continue
                into = next((held for other, held in merged if self.twins(other, held, name, slot)), None)
                if into is None:
                    merged.append((name, slot))
                    continue
                for ident in slot["per"]:
                    text = slot["per"][ident]
                    if ident not in into["per"]:
                        into["per"][ident] = text
                    if text not in into["values"]:
                        into["values"].append(text)
                into["n"] = len(into["per"])
            merged.sort(key=lambda pair: pair[1]["n"], reverse=True)
            table.extend((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]) for name, slot in merged[:K.table_keys])
        return table

    @staticmethod
    def unaskable(values):
        """Long texts, option lists, and id-like keys (all values distinct and long, few shared words)."""
        mean = sum(len(v) for v in values) / len(values)
        if mean > 40 or 2 * len([v for v in values if " / " in v]) >= len(values):
            return True
        if len(values) < 4 or mean <= 12 or len(set(values)) < len(values):
            return False
        words = [set(re.findall(r"[a-z]{3,}", v)) for v in values]
        shared = 0
        for n, v in enumerate(values):
            rest = set(re.findall(r"[a-z]{3,}", " ".join(o for o in values if o is not v)))
            shared += 1 if words[n] & rest else 0
        return 2 * shared < len(values)

    def not_stated(self, label, name, values):
        number = self.owner_of(label)
        if not number:
            return ""
        shown = word_set(" ".join(values))
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(":")
            words = word_set(value)
            if re.search(R_LIMIT_WORDS, value, re.I) or not words or not prop.strip():
                continue
            if spaced(prop) != name and stems(prop) & stems(name) and not words <= shown:
                return " [not the stated %s: another property]" % prop.strip()
        return ""

    def read_count(self, number):
        if self.books.label_key() is None:
            return 0
        return len([i for i, r in self.books.full.items() if self.owns(number, self.books.label(i)) and facts_in(r)])

    def lane_keys(self, number, table):
        """The item's table keys, most carried first (measurements after other keys of equal count), without keys a
        must settles or that name ids, options or media; catch-all keys last."""
        settled = [stems(prop_of(m)) for m in self.items[number - 1].musts]
        mine = [(key, n, len(vals)) for label, key, n, vals in table if self.owner_of(label) == number]
        if not mine and len(self.items) == 1:
            mine = [(key, n, len(vals)) for _, key, n, vals in table]
        sized = {}
        for _, key, _, vals in table:
            sized[key] = 2 * len([v for v in vals if MEASURED.match(v)]) > len(vals)
        mine.sort(key=lambda r: (-r[1], sized.get(r[0], False), -r[2]))
        keys = []
        for key, _, _ in mine:
            if re.search(R_NOT_PROP, key) or any(names_same(s, stems(key)) for s in settled):
                continue
            keys.append(key)
        plain = [k for k in keys if not re.search(R_CATCH_ALL, k)]
        return plain + [k for k in keys if k not in plain]

    # -- the plan
    def plan_prompt(self, count, table):
        lines, last = [], None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                lines.append("category %r:" % label)
                last = label
            tag = " [several features in one: ask last]" if re.search(R_CATCH_ALL, name) else ""
            lines.append("  %d. %s (on %d): %s%s%s" % (index, name, carried, " | ".join(values), self.not_stated(label, name, values), tag))
        items = ["item %d: %s - settled by the request: %s" % (n, w.name, "; ".join(w.musts) or "nothing")
                 for n, w in enumerate(self.items, 1)]
        out = [P_PLAN.replace("COUNT", str(max(1, count))), "\n\nREQUEST:\n", self.request, "\n\nITEMS:\n", "\n".join(items)]
        first = [str(x) for x in as_list(self.brief.get("ask_first")) if str(x).strip()]
        if first:
            out += ["\n\nASK FIRST (the request asks to be asked about these):\n", "\n".join(first)]
        out += ["\n\nATTRIBUTE KEYS SEEN:\n", "\n".join(lines) or "(no full listing records yet)"]
        return "".join(out)

    def plan_lanes(self, count):
        self.adopt_labels()
        table = self.key_table()
        answer = self.wire.ask(self.plan_prompt(count, table), [M_HEAVY, M_LIGHT], 3500) or {}
        picked = dict((n, []) for n in range(1, len(self.items) + 1))
        hinted = dict((n, set()) for n in picked)
        for pick in as_list(answer.get("ask")):
            if type(pick) is not dict:
                continue
            number = self.item_no(pick.get("item") or 1)
            label = re.split(r"\s+(?:and|or)\s+|[/,;(]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if spaced(label) not in map(spaced, picked[number]):
                picked[number].append(label)
            if text_at(pick, "hint"):
                hinted[number].add(spaced(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        lanes = dict((n, self.guarded_lane(n, picked[n], hinted[n], table)) for n in picked)
        self.note("plan", ask=shorten(answer.get("ask"), 120), lanes=dict((n, v[:8]) for n, v in lanes.items()))
        return lanes

    def guarded_lane(self, number, picks, hinted, table):
        """The planner's labels for one item, with its most carried keys held in the first places (unless the planner's
        picks there are story hints or carried nearly as widely), then every other key of its table; catch-all last."""
        keys = self.lane_keys(number, table)
        denominator = float(max(1, self.read_count(number)))
        share = {}
        for row in table:
            if self.owner_of(row[0]) == number:
                share[row[1]] = max(row[2] / denominator, share.get(row[1], 0.0))
        by_spelling = {}
        for key in share:
            by_spelling.setdefault(respell(key), key)
        own_name = lambda label: by_spelling.get(respell(label), label)

        def reach(label):
            plain = spaced(label)
            if plain in share:
                return share[plain]
            best = 0.0
            for key, value in share.items():
                if names_same(stems(label), stems(key)) or names_same(stems(key), stems(label)):
                    best = max(best, value)
            return best
        hints = {spaced(own_name(h)) for h in hinted}
        lane, seen = [], set()
        for pick in map(own_name, picks):
            if spaced(pick) not in seen:
                seen.add(spaced(pick))
                lane.append(pick)
        guards = [k for k in keys if re.search(R_CATCH_ALL, k) is None][:K.guard_keys]
        for key in guards:
            if spaced(key) in [spaced(x) for x in lane[:3]]:
                continue
            weak = next((n for n, x in enumerate(lane[:3]) if spaced(x) not in hints and reach(x) + K.guard_gap <= share.get(key, 0.0)), None)
            if weak is None and len(lane) > 2:
                continue
            trial = [x for x in lane if spaced(x) != spaced(key)]
            if weak is None:
                trial.append(key)
            elif len(self.items) == 1:
                trial.insert(min(weak, len(trial)), key)
            else:
                trial.insert(min(3, len(trial)), trial[weak])
                trial[weak] = key
            front = {spaced(y) for y in trial[:3]}
            if not any(spaced(x) in hints and spaced(x) not in front for x in lane[:3]):
                lane = trial
        for key in keys:
            if all(spaced(key) != spaced(x) for x in lane):
                lane.append(key)
        broad = [x for x in lane if re.search(R_CATCH_ALL, spaced(x))]
        return [x for x in lane if x not in broad] + broad

    def open_lanes(self, count):
        self.lanes = self.plan_lanes(count)
        self.lane_found = dict.fromkeys(self.lanes, 0)
        self.lane_spent = dict.fromkeys(self.lanes, 0)
        self.lane_used = dict((n, set()) for n in self.lanes)

    # -- the questions
    def pointer_note(self, told, whole):
        pointed = [need for need in self.pointer_needs() if any(need in t for t in told)]
        if not pointed:
            return ""
        text = (" These take their value from another listing: " + " | ".join(pointed) + ". Find that listing, open its "
                "page in full, and open every listing its page refers to; the last page gives the value")
        return text + (", and the order counts only after all of them are opened." if whole else ".")

    def interview(self):
        self.asked = True
        if self.msgs_free() <= 0:
            return ""
        self.widen()
        left = self.msg_limit - self.msg_used
        self.open_lanes(max(left, K.list_size) if self.listing else left)
        self.follow_up = []
        sent, told = [], []
        while self.msgs_free() > 0 and self.left() > 6 and self.clock() < K.question_by and not self.finished:
            batch = self.ask_next(told)
            if not batch:
                break
            sent.extend(batch)
        hit = [n for n in self.lane_found if self.lane_found[n] > 0]
        searched = self.value_search(hit) if hit else ""
        if told:
            self.version += 1
        out = ["The code's interviewer asked the shopper (each question used one step):\n",
               "\n".join("Q: %s\nA: %s" % (q, a) for _, q, a in sent),
               "\nNew firm requirements from these answers: ", "; ".join(told) or "none",
               ". Apply them to every pick (search again with the new values where needed).", self.pointer_note(told, True)]
        if searched:
            out.append("\n" + searched)
        if self.queued:
            out.append("\n[notes seen meanwhile] " + " | ".join(self.queued))
            self.queued = []
        return "".join(out)

    def late_question(self):
        """Before a checkout, one more planned question with a message only an ending without a purchase would have
        needed. Returns the walker's summary when the answer adds a requirement (the buy is then held), else ''."""
        if self.late_asked or self.msg_used >= self.msg_limit or not self.roles.get("ask"):
            return ""
        for p in self.process:
            if p["when"] == "order" and p["messages"] > 0:
                return ""
        if self.clock() > K.late_by or self.left() < K.late_steps or not self.judges():
            return ""
        self.late_asked = True
        if not self.lanes:
            self.open_lanes(K.list_size if self.listing else 1)
        told, budget = [], self.budget
        batch = self.ask_next(told, K.late_steps - 2)
        if not batch:
            self.note("late_ask", skipped="no property left to ask or no message sent")
            return ""
        if budget != self.budget:
            told.append("budget now %s" % self.budget)
        number, question, answer = batch[0]
        self.note("late_ask", q=question, a=answer[:200], told=told)
        if not told:
            return ""
        self.version += 1
        text = ("The code asked the shopper one more question before checking out (one step):\nQ: %s\nA: %s\nNew firm "
                "requirements from this answer: %s. The cart was not checked out: check each line against them (search "
                "again with the new values where needed), fix the cart, then check out." % (question, answer, "; ".join(told)))
        text += self.pointer_note(told, False)
        searched = self.value_search([number])
        return text + "\n" + searched if searched else text

    def told_about(self, number, label):
        """Whether an answer already gave a need on this property of the item (spelling and plurals folded; a told
        one-word name hides no longer label)."""
        mine = folded(label)
        for need in self.items[number - 1].told:
            if names_same(mine, folded(prop_of(need))):
                return True
        return False

    def next_labels(self):
        """The item to ask about next and its next labels."""
        rank = dict((n, k) for k, n in enumerate(self.plan_order))
        scarce = not self.listing and self.msg_limit <= K.scarce_msgs

        def order(n):
            return (self.lane_found.get(n, 0) > 0, 0 if scarce else self.lane_spent.get(n, 0), rank.get(n, len(rank) + n))
        size = K.list_size if self.listing else 1
        for number in sorted(self.lanes, key=order):
            used = self.lane_used.get(number, set())
            fresh = [x for x in self.lanes[number] if spaced(x) not in used and not self.told_about(number, x)]
            if fresh:
                return number, fresh[:size]
        return 0, []

    def name_parts(self, number):
        """The products an item name joins with "and" or commas when each is a plural noun, else []."""
        name = self.items[number - 1].name
        if re.search(r"\band\b", name, re.I) is None:
            return []
        parts = list(filter(None, (p.strip() for p in re.split(r"\s*(?:,|\band\b)\s*", name, flags=re.I))))
        if len(parts) < 2:
            return []
        return parts if all(re.search(r"[a-z]{3}(?<![su])s$", p, re.I) for p in parts) else []

    def item_said(self, number):
        """How a question names the item: a name joining plural nouns with "and" is asked with "or"; after a one at a
        time reply, the part whose head noun most of the item's rows name outside their label."""
        name = self.items[number - 1].name
        parts = self.name_parts(number)
        if parts and number not in self.one_product:
            return re.sub(r"\band\b", "or", name, flags=re.I)
        if not parts:
            return name
        books = self.books
        label_key = books.label_key()
        score = dict.fromkeys(parts, 0)
        heads = dict((p, roots(p.split()[-1])) for p in parts)
        for ident, row in books.rows.items():
            if label_key and not self.owns(number, books.label(ident)):
                continue
            bag = roots(" ".join(str(v) for k, v in row.items() if type(v) is str and k != label_key))
            for p in parts:
                if heads[p] & bag:
                    score[p] += 1
        best = parts[0]
        for p in parts[1:]:
            if score[p] > score[best]:
                best = p
        return best

    def samples(self, number, label):
        want = spaced(label)
        settled = " ".join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.key_table():
            if name == want:
                return [v for v in values if v and len(v) <= 30 and v not in settled][:2]
        return []

    def wording(self, number, labels, with_samples=False):
        said = self.item_said(number)
        if len(labels) > 1:
            return "What %s or %s would you like for the %s?" % (", ".join(labels[:-1]), labels[-1], said)
        examples = self.samples(number, labels[0]) if with_samples else []
        tail = ", for example %s or %s?" % (examples[0], examples[1]) if len(examples) > 1 else "?"
        return "What %s would you like for the %s%s" % (labels[0], said, tail)

    def fitting(self, number, labels, with_samples):
        """The longest leading part of `labels` whose question the message tool takes whole."""
        size = len(labels)
        while size:
            text = self.wording(number, labels[:size], with_samples)
            args = self.message_args(text)
            if args and text in args.values():
                return labels[:size], text
            size -= 1
        return [], ""

    def ask_next(self, told, floor=6):
        """Send the next question, read its answer into the items' needs (each new one appended to `told`) and settle
        the labels it named. Returns the sent batch, or []."""
        chosen = self.pick_question()
        batch = self.send([(chosen[0], chosen[1][0], chosen[2])], floor) if chosen else []
        if not batch:
            return []
        number, fitted, _, with_samples = chosen
        self.lane_spent[number] = 1 + self.lane_spent.get(number, 0)
        counts = dict((id(w), len(w.told)) for w in self.items)
        heard = self.hear(batch, [M_LIGHT, M_HEAVY])
        if len(fitted) != 1:
            heard.pop("free", None)
        self.take_heard(heard, told)
        new = []
        for n, w in enumerate(self.items, 1):
            if len(w.told) > counts[id(w)]:
                new.extend(w.told[counts[id(w)]:])
                self.lane_found[n] = 1 + self.lane_found.get(n, 0)
        crowded = any(as_int(x) == 1 for x in as_list(heard.get("crowded")))
        self.settle(number, fitted, with_samples, new, crowded)
        return batch

    def settle(self, number, labels, with_samples, new, crowded):
        """Mark what a reply showed about the labels its question named (see ALGORITHM.md section 9, Settle)."""
        used = self.lane_used.setdefault(number, set())
        if len(labels) > 1:
            self.settle_list(number, labels, used, new, crowded)
            return
        label = labels[0]
        used.add(spaced(label))
        if new or not crowded:
            return
        if number not in self.one_product and self.name_parts(number):
            self.one_product.add(number)
            self.note("one_product", item=number, said=self.item_said(number))
            self.follow_up.insert(0, (number, labels, with_samples))
            return
        head = label.split()[-1]
        if len(label.split()) > 1 and spaced(head) not in used:
            self.follow_up.insert(0, (number, [head], False))
        elif not with_samples and len(self.samples(number, label)) > 1:
            self.follow_up.insert(0, (number, labels, True))

    def settle_list(self, number, labels, used, new, crowded):
        if crowded and not new:
            self.listing = False
            self.note("lists_off", item=number)
            self.follow_up.insert(0, (number, labels[:1], False))
            return
        hit = self.answered_at(labels, new) if new else -1
        if hit < 0:
            ahead, done = labels[:0:-1], labels[:1]
            misses = self.lane_misses.get(number, 0) + 1
        else:
            ahead, done, misses = labels[hit + 1:], labels[:hit + 1], 0
        used.update(spaced(x) for x in done)
        self.lane_misses[number] = misses
        rest = [x for x in self.lanes.get(number) or [] if x not in labels]
        self.lanes[number] = rest + ahead if misses and misses % 2 == 0 else ahead + rest
        self.note("list_settled", item=number, done=labels[:hit + 1], next=ahead[:4])

    def answered_at(self, labels, new):
        """Index of the earliest label a list's reply answered (by property words, else by the label's sample values),
        -1 when no new need maps to a label."""
        samples = {}
        for _, key, _, values in self.key_table():
            bag = samples.setdefault(spaced(key), set())
            for v in values:
                bag |= roots(v)
        best = len(labels)
        for need in new:
            prop, _, value = need.partition(":")
            scores = [len(folded(x) & folded(prop)) for x in labels]
            if not any(scores):
                scores = [len(samples.get(spaced(x), set()) & roots(value)) for x in labels]
            top = max(scores or [0])
            if top > 0:
                best = min(best, scores.index(top))
        return best if best < len(labels) else -1

    def send(self, questions, floor=6):
        tool = self.roles["ask"]
        sent = []
        for number, _, text in questions:
            if self.finished or self.msg_used >= self.msg_limit or self.left() <= floor:
                break
            args = self.message_args(text)
            reply = self.wire.step(tool, args) if args else None
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get("observation")
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.done_log.append("step %d: message to the shopper %r -> reply %r (cart then: %s)" % (
                self.step, text, words[:300], self.cart_state()))
            self.done_steps.append(self.step)
            self.note("ask", q=text, a=words[:300])
            if type(answer) is dict and answer.get("error"):
                break
        return sent

    @staticmethod
    def spoken(seen):
        if type(seen) is str:
            return seen
        found, stack = [], [(seen, 0)]
        while stack:
            node, level = stack.pop()
            if level > 4:
                continue
            if type(node) is str:
                if len(node.split()) > 1:
                    found.append(node)
            elif type(node) is dict:
                stack += [(x, level + 1) for x in reversed(list(node.values()))]
            elif type(node) is list:
                stack += [(x, level + 1) for x in reversed(node)]
        return " ".join(found)[:800] if found else jdump(seen)[:400]

    def hear(self, sent, models):
        listing = ["item %d: %s" % (n, w.name) for n, w in enumerate(self.items, 1)]
        qa = []
        for k, (n, q, a) in enumerate(sent, 1):
            qa.append("%d. (about item %d) Q: %s\nA: %s" % (k, n, q, a))
        reply = self.wire.ask(compose(P_HEAR, ("ITEMS", "\n".join(listing)), ("QUESTIONS AND ANSWERS", "\n".join(qa))), models, 2500)
        return reply if reply else {}

    def take_heard(self, heard, told):
        multi = len(self.items) != 1
        for entry in filter(lambda e: text_at(e, "need"), as_list(heard.get("needs"))):
            number = self.item_no(entry.get("item"))
            need, want = text_at(entry, "need"), self.items[number - 1]
            if need not in want.told:
                want.told.append(need)
                told.append(("item %d: " % number if multi else "") + need)
            if entry.get("points") is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in as_list(heard.get("drop")):
            gone = spaced(prop_of(text_at(entry, "property")))
            want = self.items[self.item_no(entry.get("item")) - 1] if gone else None
            if want is None:
                continue
            hit = list(filter(lambda m: spaced(prop_of(m)) == gone, want.live()))
            want.dropped.extend(hit)
            told.extend(["dropped " + m for m in hit])
        for entry in as_list(heard.get("free")):
            if type(entry) is dict and entry.get("property"):
                self.items[self.item_no(entry.get("item")) - 1].free.append(str(entry["property"]))
        if is_number(heard.get("budget")) and heard["budget"] > 0:
            self.budget = heard["budget"]

    def pick_question(self):
        """(item, labels, question text, with samples) of the next message, or None when nothing is left to ask: a
        pending follow-up first, else the item's next planned labels; labels whose question the tool cannot take
        are set aside."""
        tries = K.list_size
        while tries > 0:
            tries -= 1
            self.follow_up = [f for f in self.follow_up if not self.told_about(f[0], f[1][0])]
            number, labels, with_samples = self.follow_up.pop(0) if self.follow_up else self.next_labels() + (False,)
            if not labels:
                return None
            fitted, text = self.fitting(number, labels, with_samples)
            if with_samples and not fitted:
                fitted, text = self.fitting(number, labels, False)
            if fitted:
                return number, fitted, text, with_samples
            self.lane_used.setdefault(number, set()).add(spaced(labels[0]))
            self.note("ask_too_long", item=number, label=labels[0])
        return None


# ------------------------------------------------------------------------------------------------------ the cart

UNIT_KEY = re.compile(r"\b(?:qty|quantity|count|units?)\b")


def plain_int(value):
    """An integer argument value (never a text)."""
    return None if type(value) is str else as_int(value)


class CartSide:
    """The cart as the code knows it (from cart calls and cart views), the calls that set it, and the code's own
    line checks before an order."""

    def cart_state(self):
        return jdump(sorted(self.cart_lines.items()))

    def cart_fresh(self):
        return self.cart_view is not None and self.cart_at <= self.cart_seen_at

    def cart_text(self):
        if not self.cart_fresh():
            return "lines by the code's record of the cart calls (listing: units): " + jdump(self.cart_lines)
        return jdump(shorten(self.cart_view, 300))[:3000]

    def cart_ids(self):
        if not self.cart_fresh():
            return list(self.cart_lines)
        text = jdump(self.cart_view)
        return [i for i in self.books.rows if i in text]

    def held_units(self, ident):
        if self.cart_fresh():
            pending = [self.cart_view]
            while pending:
                node = pending.pop()
                if type(node) is list:
                    pending.extend(node)
                elif type(node) is dict:
                    texts = [v.strip() for v in node.values() if type(v) is str]
                    if ident in texts:
                        units = [plain_int(v) for k, v in node.items() if plain_int(v) is not None and UNIT_KEY.search(spaced(k))]
                        if len(units) == 1:
                            return units[0]
                    pending.extend(v for v in node.values() if type(v) in (dict, list))
        return self.cart_lines.get(ident, 1)

    def sets_count(self, tool):
        """Whether the cart tool sets a line's units (an integer with minimum 0 and no mode enum)."""
        p = self.params(tool)
        if p.enums:
            return False
        return any(p.props[k].get("minimum") == 0 for k in p.ints)

    def removal_in(self, p, args):
        return any(k in p.enums and v in p.enums[k][1:] for k, v in args.items())

    def note_cart(self, tool, args):
        """Update the cart record from a cart call's arguments."""
        self.cart_touched, self.cart_at = True, self.step
        p, rows = self.params(tool), self.books.rows
        loose = None
        for value in args.values():
            if plain_int(value) is not None:
                loose = plain_int(value)
                break
        lines, whole = [], False
        for key in args:
            value = args[key]
            if type(value) is str and value in rows:
                lines.append((value, loose))
            if type(value) is not list:
                continue
            shape = (p.props.get(key) or {}).get("items") or {}
            whole = whole or (not value and type(shape) is dict and ("properties" in shape or shape.get("type") == "object"))
            for entry in value:
                if type(entry) is str:
                    if entry in rows:
                        lines.append((entry, loose))
                    continue
                if type(entry) is not dict:
                    continue
                whole = True
                ids = [v for v in entry.values() if type(v) is str and v in rows]
                counts = [plain_int(v) for v in entry.values() if plain_int(v) is not None]
                if ids:
                    lines.append((ids[0], counts[0] if counts else 1))
        if whole:
            self.cart_lines = {}
            for ident, units in lines:
                if units and units > 0:
                    self.cart_lines[ident] = units
            return
        removing, setting = self.removal_in(p, args), self.sets_count(tool)
        for ident, units in lines:
            have = self.cart_lines.get(ident, 0)
            if not removing:
                step = 1 if units is None else units
                have = step if setting else have + step
            elif p.ints:
                have -= units if units else 1
            else:
                have = have - units if units else 0
            if have > 0:
                self.cart_lines[ident] = have
            elif ident in self.cart_lines:
                del self.cart_lines[ident]

    def whole_removal(self, tool, args):
        """A removal of a held line sent without a count names every unit held."""
        p = self.params(tool)
        count = p.count()
        ident = next((v for v in args.values() if type(v) is str and v in self.cart_lines), None)
        if count is None or count in args or ident is None or not self.removal_in(p, args):
            return args
        out = dict(args)
        out[count] = min(self.cart_lines[ident], p.props[count].get("maximum", self.cart_lines[ident]))
        return out

    def id_slot(self, tool, p):
        """The cart argument that takes a listing id."""
        for used_tool, key in self.id_args:
            if used_tool == tool and key in p.texts:
                return key
        for key in p.required:
            if key in p.texts:
                return key
        return p.texts[0] if len(p.texts) == 1 else None

    def cart_calls(self, lines, drop):
        """The cart calls that leave exactly `lines` (id: units) after taking `drop` off; None when the tool's shape
        does not allow it."""
        tool = self.roles.get("cart") or ""
        p = self.params(tool)
        listed = [k for k in p.arrays if type(p.props[k].get("items")) is dict and type(p.props[k]["items"].get("properties")) is dict]
        if listed:
            sub = Params({"properties": p.props[listed[0]]["items"]["properties"]})
            if len(sub.texts) == 0:
                return None
            unit = sub.count()
            return [{listed[0]: [dict([(sub.texts[0], i)] + ([(unit, n)] if unit else [])) for i, n in lines.items()]}]
        slot = self.id_slot(tool, p)
        if not slot:
            return None
        count, mode, setting = p.count(), p.mode(), self.sets_count(tool)
        floor = p.props[count].get("minimum") if count else None
        out = []
        for ident in drop:
            units = self.cart_lines.get(ident, 1)
            if mode:
                out.append(dict([(slot, ident), (mode, p.enums[mode][1])] + ([(count, units)] if count else [])))
            elif count and (setting or (type(floor) is int and floor <= -units)):
                out.append({slot: ident, count: 0 if setting else -units})
            else:
                return None
        held = set(self.cart_ids())
        for ident in lines:
            have = self.cart_lines.get(ident, 0) if ident in held else 0
            want = lines[ident]
            if want == have:
                continue
            if have and not setting:
                return None
            pairs = [(slot, ident)]
            if mode:
                pairs.append((mode, p.enums[mode][0]))
            if count:
                pairs.append((count, want))
            out.append(dict(pairs))
        return out

    def run_cart(self, calls, record=True):
        """Send cart calls; returns their argument texts (stops at the first failure)."""
        tool = self.roles.get("cart") or ""
        sent = []
        for call in calls:
            shaped = self.fit_args(tool, call)
            if shaped is None:
                break
            reply = self.wire.step(tool, shaped)
            if reply is None or self.finished:
                break
            self.note_cart(tool, shaped)
            if record:
                self.observe(tool, reply.get("observation"))
            sent.append(jdump(shaped))
            self.cart_said = jdump(reply.get("observation"))
        return sent

    # -- line checks
    def held_lines(self):
        return [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]

    def owners(self, ident):
        label = self.books.label(ident)
        return [n for n in range(1, len(self.items) + 1) if self.owns(n, label)]

    def missing_items(self):
        """Items with no line in a cart holding fewer lines than items (items no seen label belongs to not counted)."""
        ids = self.held_lines()
        if self.books.label_key() is None or len(ids) >= len(self.items):
            return []
        seen = set(n for i in self.books.rows for n in self.owners(i))
        held = set(n for i in ids for n in self.owners(i))
        return [n for n in range(1, len(self.items) + 1) if n in seen and n not in held]

    def states_needs(self, ident, number):
        """Whether a read listing's attributes state each of the item's requirement values word for word (limits and
        exclusions count as not stated)."""
        block = facts_in(self.books.full.get(ident) or {}) or {}

        def words(text):
            return set(spaced(re.sub(r"[,/;|]", " ", spaced(text))).split())
        for owner, need in self.needs_now():
            if owner != number:
                continue
            prop, _, value = need.partition(":")
            want = words(value)
            if not want or any(R_HEDGE.fullmatch(w) for w in want):
                return False
            if not any(names_same(folded(prop), folded(k)) and want <= words(str(v)) for k, v in block.items()):
                return False
        return True

    def fill_short_cart(self):
        """Before a forced order: add, for each item without a line, the cheapest read listing that states its
        requirements. False when an item stays uncovered."""
        books = self.books
        tied = set()
        for pair in self.bound:
            tied.update(pair)
        for number in self.missing_items():
            if number in tied:
                return False
            want = self.items[number - 1]
            pool = [i for i in self.item_ids(number) if i in books.full and i not in self.cart_lines and books.buyable(i)]
            pool = [i for i in pool if books.price(i) is not None and self.states_needs(i, number)]
            if not pool:
                return False
            pick = min(pool, key=lambda i: books.price(i))
            if is_number(self.budget):
                each = self.brief.get("budget_scope") == "each"
                carried = 0.0
                if not each:
                    carried = sum((books.price(i) or 0.0) * (self.held_units(i) or 0) for i in self.cart_ids())
                if books.price(pick) * (1 if each else want.qty) + carried > self.budget + 1e-9:
                    return False
            target = {}
            for ident in self.cart_ids():
                if ident in self.cart_lines:
                    target[ident] = self.cart_lines[ident]
            target[pick] = want.qty
            calls = self.cart_calls(target, [])
            if not calls or self.left() <= len(calls) or len(self.run_cart(calls, record=False)) != len(calls):
                return False
            self.note("filled_line", item=number, line=pick)
        return True

    def fuller_label(self, number, label):
        """A seen label naming the item in full when the line's own label holds only part of the item's name."""
        want = self.items[number - 1]
        mine = roots(want.name)
        if len(mine) < 2 or mine <= roots(label) or self.books.label_key() is None:
            return ""
        if want.shelf and roots(label) == roots(want.shelf):
            return ""
        for seen in sorted(set(self.books.label(i) for i in self.books.rows)):
            if seen and mine <= roots(seen):
                return seen
        return ""

    def line_faults(self):
        self.adopt_labels()
        books = self.books
        ids = self.held_lines()
        if not ids or books.label_key() is None:
            return []
        known = set(n for i in books.rows for n in self.owners(i))
        owner = dict((i, self.owners(i)) for i in ids)
        faults = []
        for n, want in enumerate(self.items, 1):
            if n not in known:
                continue
            mine = [i for i in ids if n in owner[i]]
            if not mine:
                faults.append("item %d (%s) has no line in the cart" % (n, want.name))
            elif len([i for i in mine if owner[i] == [n]]) > 1:
                faults.append("item %d (%s) has %d lines (%s); keep one" % (n, want.name, len(mine), ", ".join(mine)))
            for i in mine:
                units = self.held_units(i)
                if owner[i] == [n] and units != want.qty:
                    faults.append("%s is held at %s unit(s); item %d (%s) wants %d" % (i, units, n, want.name, want.qty))
                fuller = self.fuller_label(n, books.label(i))
                if fuller:
                    faults.append("%s is from %s, but the store has the label %s, which names item %d (%s) in full: take "
                                  "the line from there" % (i, books.label(i), fuller, n, want.name))
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append("%s (%s) is not a kind the shopper asked for: take it out" % (i, books.label(i) or "unknown kind"))
        faults.extend("%s is sold out" % i for i in ids if books.sold_out(i))
        prices = dict((i, books.price(i)) for i in ids)
        if is_number(self.budget) and None not in prices.values():
            cap = self.budget
            if self.brief.get("budget_scope") == "each":
                for i in ids:
                    if cap < prices[i] <= 10 * cap:
                        faults.append("%s costs %g, over the %g limit for each item" % (i, prices[i], cap))
            else:
                total = sum(prices[i] * self.held_units(i) for i in ids)
                if cap + 1e-9 < total <= 10 * cap:
                    faults.append("the total is %g, over the budget of %g" % (total, cap))
        return faults + self.code_faults()

    def count_fixes(self):
        """(line, units held, units wanted) for single lines of an item held at the wrong count."""
        if self.books.label_key() is None:
            return []
        held = self.cart_ids()
        wrong = []
        for ident in held:
            units = self.held_units(ident) or 0
            owners = self.owners(ident)
            if units <= 0 or len(owners) != 1 or self.books.sold_out(ident):
                continue
            number = owners[0]
            mates = [i for i in held if (self.cart_lines.get(i, 1) or 0) > 0 and self.owns(number, self.books.label(i))]
            if len(mates) == 1 and units != self.items[number - 1].qty:
                wrong.append((ident, units, self.items[number - 1].qty))
        return wrong

    def fix_counts(self, wrong):
        tool = self.roles.get("cart") or ""
        p = self.params(tool)
        slot, count, mode = self.id_slot(tool, p), p.count(), p.mode()
        if not tool or slot is None or count is None:
            return []
        setting = self.sets_count(tool)
        if not setting and mode is None:
            return []
        plan = []
        for ident, have, want in wrong:
            if setting:
                plan.append({slot: ident, count: want})
                continue
            choices = [str(o) for o in p.enums[mode]]
            off = next(filter(lambda o: re.search(r"remov|delet|drop|clear|take", o, re.I), choices), choices[1])
            on = next(filter(lambda o: o != off, choices), choices[0])
            plan += [{slot: ident, mode: off, count: have}, {slot: ident, mode: on, count: want}]
        if self.left() < len(plan) + 3:
            return []
        report = []
        for call in plan:
            reply, sent = self.use(tool, call, absorb=False)
            if sent is None or reply is None or self.finished:
                break
            seen = reply.get("observation")
            self.note_cart(tool, sent)
            self.cart_said = jdump(seen)
            self.observe(tool, seen)
            report.append(jdump(sent) + " -> " + self.show(seen)[:300])
        self.note("count_fix", calls=report)
        return report

    # -- accessories and devices
    def fit_index(self):
        marks, notes = {}, {}
        books = self.books
        for ident in books.order:
            record = books.full.get(ident) or books.rows.get(ident) or {}
            flat = [str(v) for v in record.values() if type(v) is str]
            inner = ["%s %s" % (k, v) for block in record.values() if type(block) is dict for k, v in block.items() if type(v) is str]
            marks[ident] = fit_marks(" ".join(flat) + " " + ". ".join(inner))
            for part, codes in marks[ident]["also"]:
                notes.setdefault(part, set()).update(codes)
        return dict(marks=marks, notes=notes)

    @staticmethod
    def fit_set(ident, index):
        mark = index["marks"].get(ident) or {}
        found = set(mark.get("works") or ())
        found |= set(s + "xx" for s in mark.get("series") or ())
        own = mark.get("own")
        return found | index["notes"].get(own, set()) if own else found

    def fit_faults(self):
        if not self.bound:
            return []
        index, cart, faults = self.fit_index(), self.cart_ids(), []
        for part, main in self.bound:
            mains, parts = set(self.item_ids(main)), set(self.item_ids(part))
            devices = [i for i in cart if i in mains]
            pieces = [i for i in cart if i in parts]
            if len(devices) != 1 or not pieces:
                continue
            device = devices[0]
            codes = set(index["marks"].get(device, {}).get("codes") or ())
            if not codes:
                continue
            for line in pieces:
                fits = self.fit_set(line, index)
                if not fits or self.part_fits(line, index, codes):
                    continue
                good = [i for i in self.item_ids(part) if i not in cart and not self.books.sold_out(i) and self.part_fits(i, index, codes)]
                text = "%s (%s) works with %s, not with %s (%s)" % (line, self.items[part - 1].name, ", ".join(sorted(fits)[:4]),
                                                                    device, ", ".join(sorted(codes)))
                if good:
                    text += "; seen parts of that kind that work with it: " + ", ".join(good[:4])
                else:
                    text += "; no seen part of that kind works with it, so look for one or switch the device: " + self.fit_plan(short=True)
                faults.append(text)
        return faults

    def fit_plan(self, short=False):
        if not self.bound:
            return ""
        index, lines = self.fit_index(), []
        for main in sorted(set(m for _, m in self.bound)):
            tally = {}
            for part in [p for p, m in self.bound if m == main]:
                for ident in self.item_ids(part):
                    if self.books.sold_out(ident):
                        continue
                    for code in self.fit_set(ident, index):
                        counts = tally.setdefault(code, {})
                        counts[part] = counts.get(part, 0) + 1
            best = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            bits = []
            for code, counts in best:
                devices = [i for i in self.item_ids(main) if code in (index["marks"].get(i, {}).get("codes") or ())][:2]
                where = " (device seen: %s)" % ", ".join(devices) if devices else " (device not seen yet)"
                bits.append("%s: %s%s" % (code, ", ".join("%d %s" % (n, self.items[p - 1].name) for p, n in sorted(counts.items())), where))
            if bits:
                lines.append("codes the %s candidates' parts name -> %s" % (self.items[main - 1].name, "; ".join(bits)))
        if not lines:
            return ""
        joined = " | ".join(lines)
        if short:
            return joined
        return ("[fit plan: a device fits a part when the part's text names the device's model code or another name it is "
                "sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + joined)

    def part_fits(self, ident, index, codes):
        mark = index["marks"].get(ident) or {}
        extra = index["notes"].get(mark.get("own"), set()) if mark.get("own") else set()
        return any(fits_code(mark, extra, c) for c in codes)

    def needed_codes(self, number):
        """Model codes the item's requirements name (a device the part must fit)."""
        found = []
        for owner, need in self.needs_now():
            if owner == number and re.search(r"fit|compatib|model|device|work", prop_of(need), re.I):
                found += [c for c in code_list(value_of(need)) if c not in found]
        return found

    def code_faults(self):
        """U3: cart lines whose fit text excludes or omits the model code their item's requirement names."""
        index, faults = self.fit_index(), []
        for ident in self.cart_ids():
            mark = index["marks"].get(ident) or {}
            if not (mark.get("works") or mark.get("series")):
                continue
            for number in self.owners(ident):
                for code in self.needed_codes(number):
                    if not self.part_fits(ident, index, [code]):
                        faults.append("%s does not fit %s by its own fit text (fits: %s%s)" % (
                            ident, code.upper(), ", ".join(sorted(self.fit_set(ident, index)))[:120],
                            "; except " + ", ".join(mark.get("except")) if mark.get("except") else ""))
        return faults


# ------------------------------------------------------------------------------------- judgements before an ending

def leaves_of(node):
    """Every plain value inside a JSON-like node, as text."""
    out, stack = [], [node]
    while stack:
        top = stack.pop()
        if type(top) is dict:
            stack.extend(top.values())
        elif type(top) is list:
            stack.extend(top)
        else:
            out.append(str(top))
    return out


def whole_word(word, text):
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])", text) is not None


class CheckSide:
    """Model-backed checks: the cart check, the decline audit, pointer chains, sold-out replacements, the process
    gate; and the code completing a cart a check confirmed."""

    def others(self, skip, limit=30):
        books = self.books
        order = books.order[::-1]
        if self.brief.get("pick") == "cheapest":
            def rank(ident):
                price = books.price(ident)
                return (ident not in books.full, float("inf") if price is None else price)
            order = sorted(order, key=rank)
        picked = [i for i in order if i not in skip and not books.sold_out(i)][:limit]
        return "\n".join(jdump(shorten(books.full.get(i) or books.rows.get(i), 160))[:900] for i in picked)

    @staticmethod
    def check_key(ids, version):
        return "%s#%d" % ("|".join(sorted(ids)), version)

    def check_cart(self, ids=None, line=K.soft):
        """The cart check (or a check of a planned set of lines). '' when it passes or cannot run; else the
        problems and better listings for the walker."""
        planned = ids is not None
        ids = ids if planned else self.cart_ids()
        key = self.check_key(ids, self.version)
        if len(ids) == 0 or key in self.checked or len(self.judges(line)) == 0:
            return ""
        self.checked.add(key)
        prompt = self.check_prompt(ids, planned)
        judges = self.judges(line)
        verdict = None
        light_first = len(judges) > 1 and not planned and not self.audited and self.brief.get("pick") not in ("cheapest", "most", "least")
        if light_first:
            verdict = self.wire.ask(prompt, [M_LIGHT], 3000)
            if verdict:
                judges = [M_HEAVY] if verdict.get("ok") is False and M_HEAVY in self.judges(line) else []
        if judges and not self.wire.no_credit:
            verdict = self.wire.ask(prompt, judges, 3000) or verdict
        self.note("check", cart=ids, got=verdict)
        ok = verdict is not None and verdict != {} and verdict.get("ok") is not False
        self.verdicts[key] = ok
        if ok or not verdict:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        books = self.books
        problems = [str(p) for p in as_list(verdict.get("problems"))][:5]
        better = []
        for b in as_list(verdict.get("better")):
            if str(b) in books.rows and not books.sold_out(str(b)) and len(better) < 4:
                better.append(str(b))
        out = "A careful review says this cart is not right yet: " + "; ".join(problems if problems else ["see the requirements"])
        if better:
            out += ". Listings that look better: " + ", ".join(better)
        return out + ". Fix the cart (or show the review is wrong by re-reading the records), then check out."

    def passed(self, ids):
        return self.verdicts.get(self.check_key(ids, self.version)) is True

    # -- pointer chains
    def shop_names(self, ident):
        names = []
        for key, value in self.books.rows.get(ident, {}).items():
            if type(value) is str and 2 < len(value.strip()) <= 40 and re.search(R_SHOP_KEY, str(key), re.I):
                names.append(value.strip())
        return names

    def named_in(self, about, ident):
        low = about.lower()
        return any(whole_word(n.lower(), low) for n in self.shop_names(ident))

    def find_link(self, about, words):
        """Rows that may be a chain's missing link: an opened page of the named shop already showing the values ends the
        search; else one search of the words (remembered), rows of that shop of the described kind."""
        books = self.books
        opened_shop = [i for i in books.opened if self.named_in(about, i)]
        unopened_shop = [i for i in books.rows if i not in books.opened and self.named_in(about, i)]
        named = set()
        for ident in unopened_shop + opened_shop:
            for name in self.shop_names(ident):
                named |= word_set(name)
        tokens = re.findall(r"[a-z0-9]+", (words or about).lower())
        values = set(w for w in tokens if (len(w) > 2 or w.isdigit()) and not R_FILLER.fullmatch(w) and w != "of") - named
        for ident in opened_shop:
            text = " ".join(leaves_of(books.full.get(ident) or books.rows.get(ident) or {})).lower()
            if values and all(whole_word(v, text) for v in values):
                self.note("link_known", id=ident)
                return []

        def shop_rows(ids):
            return [i for i in ids if i not in books.opened and self.named_in(about, i)]
        memo = " ".join(words.lower().split())
        found = []
        plan = self.search_plan()
        if memo and memo in self.link_seen:
            self.link_searches += 1
            found = shop_rows(self.link_seen[memo])
        elif words and plan:
            tool, base, field = plan
            query = dict(base)
            query[field] = words
            shaped = self.fit_args(tool, query)
            if shaped is not None:
                self.link_searches += 1
                reply, _ = self.use(tool, shaped)
                if reply is not None:
                    self.link_seen[memo] = [i for i, _ in books.records_in(unpack(reply.get("observation")))]
                    found = shop_rows(self.link_seen[memo])
                    self.note("link_search", words=words[:80], rows=found[:4])
        kind = set(w for w in word_set(words or about) if not R_FILLER.fullmatch(w)) - named

        def of_kind(ids):
            return [i for i in ids if not kind or kind & word_set(jdump(books.rows.get(i) or {}))]
        return (of_kind(found) or of_kind(shop_rows(books.order[::-1])))[:2]

    def chain_records(self):
        books = self.books
        opened = []
        unread = []
        for ident in books.order[::-1]:
            record = books.full.get(ident) or books.rows.get(ident)
            if record and ident in books.opened:
                if len(opened) < 45:
                    opened.append("OPENED %s" % jdump(shorten(record, 300))[:1400])
                continue
            unread.append("ROW %s" % jdump(shorten(books.rows.get(ident), 80))[:300])
        return opened + unread[:max(0, 60 - len(opened))]

    def trail(self):
        """Follow pointer requirements to the page that gives their value, opening known links; (text, hold)."""
        live = set(need for _, need in self.needs_now())
        pending = [need for need in self.pointer_needs() if need in live and need not in self.ref_values]
        tool = self.offer_tool()
        target = self.single_target(tool)
        opened_text, lost = [], []
        while pending and self.trails < (4 if self.left() <= 15 else 6) and self.wire.spent < K.hard:
            self.trails += 1
            values, reach, gaps = self.read_chains(pending)
            for need, value in values.items():
                self.ref_values[need] = value
                self.version += 1
            lost = ["%s -> find %s%s" % (need, about, " (search: %s)" % words if words else "") for need, about, words in gaps]
            pending = [need for need in pending if need not in self.ref_values]
            if not reach and gaps and tool and target and self.link_searches < 3 and self.left() > 12:
                reach = self.find_link(gaps[0][1], gaps[0][2])
                latest = self.trace[-1] if self.trace else {}
                if latest.get("kind") == "link_known":
                    opened_text.append(latest["id"] + ": an opened page of the shop this chain names that shows the values "
                                       "its description gives; read the value from it")
                    lost = lost[1:]
            if not (reach and tool and target) or self.finished or self.left() - 4 < len(reach):
                break
            for ident in reach[:max(0, min(3, 6 - len(opened_text)))]:
                reply, sent = self.use(tool, {target: ident})
                if self.finished or sent is None or reply is None:
                    break
                opened_text.append(ident + ": " + self.show(reply.get("observation"))[:1500])
        pieces = []
        if opened_text:
            pieces.append("[the code opened the pages a requirement points at]\n" + "\n".join(opened_text) + "\n")
        if self.ref_values:
            pieces.append("[values read from the pages requirements point at] " +
                          "; ".join(need + " -> " + value for need, value in self.ref_values.items()) + "\n")
        hold = False
        if lost and self.left() > 5 and self.trail_blocks < (2 if self.left() <= 15 else 3):
            hold = True
            self.trail_blocks += 1
            pieces.append("The order counts only after every page a requirement points at has been opened in full (a search "
                          "row is not enough), following each page that refers to a further listing. Still to find and "
                          "open: " + " | ".join(lost[:3]) + ". Search for it by its shop's name together with the values its "
                          "description gives (the shop's other listings of that kind are not it), open the row of that shop "
                          "whose page shows those values, apply the value its page gives, then check out.")
        return "".join(pieces), hold

    # -- stock and replacements
    def offer_tool(self):
        return self.roles.get("offer") or self.roles.get("open")

    def reread_offers(self):
        """Read again the cart lines whose offer was last seen before (step - 2) or the last refusal."""
        tool = self.offer_tool()
        target = self.single_target(tool) if tool else None
        if target is None:
            return ""
        cutoff = max(self.refused_at, self.step - 2)
        stale = []
        for ident in self.cart_ids():
            if ident not in self.books.gone and self.offer_seen.get(ident, -99) < cutoff:
                stale.append(ident)
        if len(stale) == 0 or self.left() - 3 < len(stale):
            return ""
        notes = []
        for ident in stale:
            old = dict(self.books.rows.get(ident) or {})
            reply, sent = self.use(tool, {target: ident}, absorb=False)
            if reply is None or sent is None or self.finished:
                break
            self.offer_seen[ident] = self.step
            seen = reply.get("observation")
            self.observe(tool, seen)
            new = self.books.rows.get(ident) or {}
            moved = [k for k in new if k in old and old[k] != new[k] and type(new[k]) is not dict and type(new[k]) is not list]
            if moved:
                notes.append("%s changed (%s): %s" % (ident, ", ".join(moved[:4]), jdump(shorten(unpack(seen), 300))[:900]))
        self.note("reread", ids=stale, changed=len(notes))
        self.version += 1 if notes else 0
        return "\n".join(notes)

    def find_sold(self):
        """After a sold-out note naming no line: a one-line cart's line is the one; else re-read the lines."""
        ids = [i for i in self.cart_ids() if i not in self.books.gone]
        if not ids or any(self.books.sold_out(i) for i in ids):
            return ""
        if len(ids) > 1:
            return self.reread_offers()
        self.books.unavailable.add(ids[0])
        return ""

    def replacements(self, lost):
        if not self.judges():
            return []
        books = self.books
        pool = [i for i in books.order[::-1] if i not in lost and not books.sold_out(i)][:25]
        if not pool:
            return []
        records = [jdump(shorten(books.full.get(i) or books.rows.get(i), 200))[:1200] for i in pool]
        prompt = "".join([P_SWAP, "\n\nREQUEST:\n", self.request, "\n\nREQUIREMENTS IN FORCE:\n", self.ledger(),
                          "\n\nSOLD-OUT CART LINE:\n", "\n".join("%s: %s" % (i, books.record(i)) for i in lost),
                          "\n\nCART NOW:\n", self.cart_text(), "\n\nOTHER LISTING RECORDS SEEN (newest first):\n", "\n".join(records)])
        answer = self.wire.ask(prompt, list(reversed(self.judges())), 3500) or {}
        held = self.cart_ids()
        fits, taken = [], set()
        for entry in as_list(answer.get("fits")):
            ident = str(entry.get("id")) if type(entry) is dict else str(entry)
            price = entry.get("price") if type(entry) is dict else None
            if ident in taken or ident in lost or ident in books.gone or ident in held or ident not in books.rows:
                continue
            taken.add(ident)
            fits.append((float(price) if is_number(price) else float("inf"), ident))
        if self.brief.get("pick") == "cheapest":
            ranked = [i for _, i in sorted(fits, key=lambda f: f[0])]
        else:
            ranked = [i for _, i in fits]
        self.note("swap", lost=lost, fits=fits)
        return ranked[:4]

    def swap_hint(self):
        books = self.books
        lost = [i for i in self.cart_ids() if i not in books.gone and books.sold_out(i)]
        if len(lost) == 0:
            return ""
        books.gone.update(dict.fromkeys(lost, self.step))
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.replacements(lost)
        head = "[code] Sold out now: " + ", ".join(lost) + " (the store shows it cannot be bought). It will not come back, so do not open it again."
        if len(picks) == 0:
            return head + " Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits."
        first, rest = picks[0], picks[1:]
        if self.left() < 7:
            body = (" Few steps are left: in this turn add %s at the requested quantity and remove the sold-out line (two "
                    "calls), then check out. Other listings already read that appear to fit: %s." % (first, ", ".join(rest) or "none"))
        elif self.brief.get("pick") == "cheapest":
            body = (" Listings already read that appear to fit that item instead, cheapest first: %s. The pick rule is the "
                    "cheapest, so unless you have already seen the whole fitting set, run one search under %s's price (the "
                    "search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting "
                    "listing." % (", ".join(picks), first))
        else:
            body = (" Listings already read that appear to fit that item instead: %s. In this turn take the sold-out line "
                    "out of the cart and add %s at the requested quantity, then check out." % (", ".join(picks), first))
        return head + body

    # -- before a decline
    def audit_fits(self, answer):
        fits = (answer or {}).get("fits")
        out = {}
        if type(fits) is not dict:
            return out
        for key, value in fits.items():
            good = [str(i) for i in as_list(value) if str(i) in self.books.rows and not self.books.sold_out(str(i))][:4]
            if good:
                out[str(key)] = good
        return out

    def audit_call(self):
        books = self.books
        records = []
        for ident in reversed(books.order):
            if len(records) >= 30:
                break
            if books.sold_out(ident):
                continue
            full = books.full.get(ident)
            records.append(("OPENED " + jdump(shorten(full, 200))[:1200]) if full else ("ROW " + jdump(shorten(books.rows.get(ident), 120))[:500]))
        if not records:
            return None
        prompt = compose(P_AUDIT, ("REQUEST", self.request), ("REQUIREMENTS IN FORCE", self.ledger()),
                         ("LISTING RECORDS SEEN (newest first)", "\n".join(records)))
        judges = self.judges()
        answer = self.wire.ask(prompt, [M_LIGHT] if len(judges) else [], 4000)
        if not self.audit_fits(answer) and M_HEAVY in judges and not self.wire.no_credit:
            second = self.wire.ask(prompt, [M_HEAVY], 4000)
            answer = second if second else answer
        self.note("audit", got=answer)
        return answer

    def open_rows(self, rows):
        joiner, many = self.roles.get("compare"), self.compare_target()
        single = self.facts_tool()
        slot = self.single_target(single)
        if joiner and many and not self.compare_bare():
            calls = [(joiner, {many: rows[:self.compare_room()]})]
        elif single and slot:
            calls = [(single, {slot: ident}) for ident in rows[:2]]
        else:
            return False
        if self.left() - 5 < len(calls):
            return False
        start = len(self.books.full)
        for tool, args in calls:
            reply, sent = self.use(tool, args)
            if sent is None:
                return False
            if self.finished or reply is None:
                break
        gained = len(self.books.full) - start
        self.note("audit_open", rows=rows, opened=gained)
        return gained > 0

    def audit(self):
        if self.audited or not self.judges():
            return ""
        self.audited = True
        books = self.books
        answer = self.audit_call()
        named = self.audit_fits(answer)
        if not named:
            worth = []
            for raw in as_list((answer or {}).get("open")):
                ident = str(raw)
                if ident in books.rows and ident not in books.opened and not books.sold_out(ident):
                    worth.append(ident)
            if worth and self.open_rows(worth):
                answer = self.audit_call()
                named = self.audit_fits(answer)
            if not named:
                return ""
        self.lead_items = named
        self.leads = sum(named.values(), [])
        reason = (answer or {}).get("reason")
        parts = ["item %s: %s" % (key, ", ".join(ids)) for key, ids in named.items()]
        return ("A careful review thinks these listings may fit: " + "; ".join(parts) + (" (%s)" % reason if reason else "")
                + ". Open them and buy what truly fits; end without a purchase only if each one fails.")

    def complete_cart(self):
        """With a lead a check confirmed: set the cart to the leads (one per item kind, other kinds kept), check the
        planned cart, then check out (twice at most)."""
        books, buy = self.books, self.roles.get("buy")
        if self.completed or not self.confirmed or not buy or not self.roles.get("cart") or books.sold_out(self.confirmed):
            return ""
        self.completed = True
        labelled = books.label_key() is not None
        kind = (lambda i: books.label(i).strip().lower()) if labelled else (lambda i: i)
        plan, kinds = self.lead_plan(kind)
        if self.confirmed not in plan:
            return ""
        current = self.cart_ids()
        keep = dict((i, self.cart_lines.get(i, 1)) for i in current
                    if i not in plan and not books.sold_out(i) and kind(i) not in kinds)
        drop = [i for i in current if i not in plan and i not in keep]
        if len(self.items) > len(plan) + len(set(kind(i) for i in keep)):
            return ""
        lines = dict(keep, **plan) if all(type(k) is str for k in plan) else {**keep, **plan}
        calls = self.cart_calls(lines, drop)
        if not calls or self.left() <= len(calls):
            return ""
        ids = list(lines)
        self.read_facts(ids, 3 + len(calls))
        self.show_links(ids, 3 + len(calls))
        self.check_cart(ids, K.hard)
        if not self.passed(ids):
            return ""
        done = self.run_cart(calls)
        args = self.fit_args(buy, dict(self.brief.get("buy_args") or {})) if len(done) == len(calls) else None
        if args is None:
            return ""
        replies = self.buy_twice(ids, args, lines, drop)
        if not replies:
            return ""
        return ("Not run (the code acted instead): the independent check confirmed %s, so the code set the cart (%s) and "
                "checked out. Store reply: %s" % (self.confirmed, "; ".join(done), " | then: ".join(replies)))

    # -- the request's process steps
    def gate_prompt(self, stage, steps):
        ending = set(filter(None, (self.roles.get("buy"), self.roles.get("refuse"))))
        record = "\n".join("%d. %s" % (n, x) for n, x in enumerate(self.done_log, 1)) or "(nothing yet)"
        notes = "\n\nSTORE NOTES SEEN:\n" + "\n".join(self.notes[-4:]) if self.notes else ""
        doing = "place the order" if stage == "order" else "end without a purchase"
        return "".join([P_GATE.replace("STAGE", doing), "\n\nREQUEST:\n", self.request, "\n\nSTEPS DUE NOW:\n",
                        "\n".join("- " + p["step"] for p in steps), "\n\nLEDGER:\n", self.ledger(), "\n\nCART NOW: ",
                        self.cart_text(), "\n\nRECORD (messages and step calls so far):\n", record, notes,
                        "\n\nMESSAGES LEFT: %d   STEPS LEFT: %d (the %s itself takes one)" % (
                            max(0, self.msg_limit - self.msg_used), self.left(), "order" if stage == "order" else "ending"),
                        "\n\nTOOLS:\n", "\n".join(self.tool_lines(ending))])

    def gate(self, stage):
        """Do the process steps due before an order or an ending; (go on, text of what was done)."""
        steps = [p for p in self.process if p["when"] == stage]
        if not steps or self.finished or not self.due(stage):
            return True, ""
        state = dict(said=[], nudged=False)
        rounds = K.gate_rounds
        while rounds:
            rounds -= 1
            outcome = self.gate_round(stage, steps, state)
            if outcome in ("go", "stop"):
                break
            if outcome != "again":
                return False, outcome
        done = state["said"]
        return True, "Before going on, the code did the steps the request asks for:\n" + "\n".join(done) if done else ""

    def check_prompt(self, ids, planned):
        books = self.books
        if planned:
            units = "; ".join("item %d: quantity %d" % (n, w.qty) for n, w in enumerate(self.items, 1))
            cart = ("CART (planned, judge these listings as if they were carted at the requested quantities: %s)" % units, ", ".join(ids))
        else:
            cart = ("CART", self.cart_text())
        records = []
        for ident in ids:
            flag = "SOLD OUT (the store says it can no longer be bought). " if books.sold_out(ident) else ""
            records.append(ident + ": " + flag + books.record(ident))
        sections = [("REQUEST", self.request), ("REQUIREMENTS IN FORCE", self.ledger()), cart,
                    ("RECORDS OF THE CART'S LISTINGS", "\n".join(records))]
        if self.brief.get("pick") in ("cheapest", "most", "least"):
            sections.append(("OTHER LISTINGS SEEN (newest first)", self.others(ids, 15)))
        return compose(P_CHECK, *sections)

    def gate_proof(self, talk):
        """Record lines that can prove a step: all of them, or for a step needing a message at the order only those
        made after the cart last changed."""
        if not talk:
            return set(range(1, len(self.done_log) + 1))
        after = [n + 1 for n, at in enumerate(self.done_steps) if at > self.cart_at]
        return set(n for n in after if n <= len(self.done_log))

    def gate_round(self, stage, steps, state):
        """One round of the process gate. Returns 'go', 'stop', 'again' or a text that holds the ending."""
        order = [M_LIGHT, M_HEAVY]
        if state["nudged"]:
            order.reverse()
        answer = self.wire.ask(self.gate_prompt(stage, steps), order, 3500)
        if answer is None:
            return "stop"
        said = state["said"]
        told = []
        hold = ""
        if len(said):
            hold = text_at(answer, "hold")
            self.take_heard({"needs": answer.get("needs")}, told)
        call = answer.get("call")
        call = call if type(call) is dict else None
        self.note("gate", stage=stage, call=shorten(call, 200), steps=shorten(answer.get("steps"), 120), hold=hold[:200], told=told)
        if hold or told:
            self.version += 1
            asks = told + [hold] if hold else told
            lead = ("Before going on, the code did the steps the request asks for:\n" + "\n".join(said) + "\n") if said else ""
            return lead + "The shopper's reply changes the task: " + "; ".join(asks) + ". Fix the cart for it, then check out again."
        proof = self.gate_proof(stage == "order" and any(p["messages"] for p in steps))
        weak = []
        for entry in as_list(answer.get("steps")):
            if type(entry) is not dict:
                continue
            lines = set(as_int(x) or 0 for x in as_list(entry.get("lines")))
            if entry.get("done") is not True or not (lines & proof):
                weak.append(entry)
        if call is None:
            room = self.roles.get("ask") and self.msg_used < self.msg_limit and self.left() > 1
            if weak and room and not said and not state["nudged"]:
                state["nudged"] = True
                self.note("gate_unproven", steps=shorten(weak, 120))
                what = "; ".join(str(e.get("step") or "") for e in weak)[:200]
                self.done_log.append("(code: no line made since the cart last changed at step %d shows %s done, and %d "
                                     "message(s) and %d steps are left: give the call that does it)"
                                     % (self.cart_at, what, self.msg_limit - self.msg_used, self.left()))
                self.done_steps.append(-1)
                return "again"
            self.gate_ok.add((stage, self.cart_mark()))
            return "go"
        return self.gate_call(call, said)

    def gate_call(self, call, said):
        """Carry out the one call the gate asked for; 'again' after it ran, 'go' when it cannot run."""
        name = str(call.get("tool") or "")
        ends = (self.roles.get("buy"), self.roles.get("refuse"))
        asking = name == self.roles.get("ask")
        if name in ends or name not in self.schemas or self.left() < 2 or (asking and self.msg_limit <= self.msg_used):
            return "go"
        args = self.fit_args(name, call["args"] if type(call.get("args")) is dict else {})
        if args is None:
            return "go"
        if asking:
            words = next(filter(lambda v: type(v) is str, args.values()), "")
            batch = self.send([(1, "", words)], 1)
            if not batch:
                return "go"
            self.gate_msgs += 1
            said.append("Q: " + words + "\nA: " + batch[0][2])
            return "again"
        reply = self.wire.step(name, args)
        if reply is None:
            return "go"
        seen = reply.get("observation")
        self.books.absorb(unpack(seen))
        text = self.show(seen)
        self.done_log.append("step %d: %s %s -> %s (cart then: %s)" % (self.step, name, jdump(args), text[:300], self.cart_state()))
        self.done_steps.append(self.step)
        said.append(name + " " + jdump(args) + " -> " + text[:400])
        return "again"

    def read_chains(self, pending):
        """One TRAIL call: (values found, known links to open, links to search for)."""
        books = self.books
        models = [M_LIGHT]
        if K.hard >= self.wire.spent + K.heavy_room:
            models.append(M_HEAVY)
        answer = self.wire.ask(compose(P_TRAIL, ("REQUIREMENTS WITH A VALUE ON ANOTHER LISTING", "\n".join(pending)),
                                       ("RECORDS (newest first)", "\n".join(self.chain_records()))), models, 4500) or {}
        self.note("trail", pending=pending, got=shorten(answer, 150))
        values, reach, gaps = {}, [], []
        for chain in filter(lambda c: type(c) is dict, as_list(answer.get("chains"))):
            asked = text_at(chain, "need")
            if asked in pending:
                need = asked
            elif len(pending) == 1:
                need = pending[0]
            else:
                continue
            links = list(filter(lambda x: type(x) is dict, as_list(chain.get("links"))))
            ids = list(map(lambda x: text_at(x, "id"), links))
            stated = text_at(chain, "value")
            if links and set(ids) <= books.opened:
                if (chain.get("done") is True and stated) or not text_at(links[-1], "find"):
                    values[need] = stated or "as stated on " + ids[-1]
                    continue
            unopened = [i for i in ids if i in books.rows and i not in books.opened]
            if unopened:
                reach.extend(i for i in unopened if i not in reach)
                continue
            gap = {}
            for link, ident in zip(links, ids):
                if ident not in books.rows:
                    gap = link
                    break
            gaps.append((need, text_at(gap, "about") or "the listing it points at", text_at(gap, "find")))
        return values, reach, gaps

    def lead_plan(self, kind):
        """Cart lines the confirmed leads give: the first unsold lead of each item, one per item kind."""
        plan, kinds = {}, set()
        for number, ids in self.lead_items.items():
            digits = re.search(r"\d+", str(number))
            index = int(digits.group()) if digits else (1 if len(self.items) == 1 else 0)
            unsold = [i for i in ids if not self.books.sold_out(i)]
            if not unsold or not 1 <= index <= len(self.items) or kind(unsold[0]) in kinds:
                continue
            plan[unsold[0]] = self.items[index - 1].qty
            kinds.add(kind(unsold[0]))
        return plan, kinds

    def buy_twice(self, ids, args, lines, drop):
        """Gate and check out the planned cart; a refused first try is re-read and re-checked once."""
        replies = []
        for second in (False, True):
            if second:
                self.reread_offers()
                self.check_cart(ids, K.hard)
                stop = self.finished or self.left() < 1 or not self.passed(ids)
                if stop or any(self.books.sold_out(i) for i in ids):
                    break
            go, said = self.gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.wire.step(self.roles["buy"], args)
            self.note("completed", lines=lines, drop=drop, reply=shorten(reply, 200))
            if reply is None:
                break
            self.buys += 1
            ended = bool(reply.get("finished"))
            if not ended:
                self.refused_at = self.step
            self.observe(self.roles["buy"], reply.get("observation"))
            replies.append(self.show(reply.get("observation")))
            if ended or self.finished:
                break
        return replies


# ------------------------------------------------------------------------------------------------ the walker

TURNED = "Not run (no step used): "
HELD = "Not run yet (no step used). "


class WalkSide:
    """The light model's walk: the tools it is offered, the filter every call it makes passes through (turn-downs,
    holds before an ending, the code's own work around a call), the forced finish, and the loop."""

    # -- the code's own calls
    def fit_args(self, tool, args):
        """Arguments shaped to the tool's schema, or None when they cannot be made to fit."""
        faults = []
        shaped = conform(args, self.schemas.get(tool), "args", faults)
        return shaped if not faults else None

    def use(self, tool, args, absorb=True):
        """Send one call of the code's own: (store reply or None, the arguments sent or None)."""
        shaped = self.fit_args(tool, args)
        if shaped is None:
            return None, None
        reply = self.wire.step(tool, shaped)
        if reply is not None and absorb:
            self.observe(tool, reply.get("observation"))
        return reply, shaped

    def matches(self, name, args, job):
        fixed = self.brief.get(job + "_args")
        fixed = fixed if type(fixed) is dict else {}
        if name != self.roles.get(job):
            return False
        return all(args.get(k) == v for k, v in fixed.items())

    def is_buy(self, name, args):
        return self.matches(name, args, "buy")

    def is_refuse(self, name, args):
        return self.matches(name, args, "refuse")

    def offered(self):
        """The walker's tools: never the message tool; near the end only the endings (and the cart tools)."""
        roles, left, tail = self.roles, self.left(), self.tail()
        ends = set(filter(None, (roles.get("buy"), roles.get("refuse"))))
        cart_tools = {roles.get("cart"), roles.get("cart_view")}
        if left <= tail:
            narrow = ends
        elif left <= tail + 3:
            narrow = ends | cart_tools
        elif self.buys == 0 and left <= len(self.items) + 5:
            narrow = ends | cart_tools | {roles.get("offer")}
        else:
            narrow = None
        tools = [t for t in self.tools if t["function"]["name"] != roles.get("ask")]
        if narrow is None or not ends:
            return tools
        return [t for t in tools if t["function"]["name"] in narrow] or tools

    def quiet(self, name, args):
        """A no-purchase call's free texts replaced by a short line when they name a listing or must stay quiet."""
        p = self.params(name)
        out = dict(args)
        loud = self.brief.get("quiet_decline") is False
        for key in p.texts:
            if out.get(key) is None and key not in p.required:
                continue
            said = str(out.get(key) or "")
            if said and loud and not any(i in said for i in self.books.rows):
                continue
            out[key] = QUIET_LINE[:p.props[key].get("maxLength", 200)]
        return out

    def can_refuse(self):
        tool = self.roles.get("refuse")
        return bool(tool) and self.fit_args(tool, self.quiet(tool, dict(self.brief.get("refuse_args") or {}))) is not None

    def fix_ids(self, name, args):
        """A listing's text (a unique row-text prefix) passed where an id belongs is swapped for the id; anything else
        that is not an id is refused with the reason."""
        books, roles = self.books, self.roles
        if not books.id_key or name not in set(filter(None, (roles.get(r) for r in ("cart", "open", "offer", "compare")))):
            return args, ""
        p = self.params(name)
        slots = [k for k in p.props if (k in p.texts or k in p.arrays) and "enum" not in p.props[k]]
        if len(slots) != 1:
            return args, ""
        slot = slots[0]
        if (name != roles.get("cart") and (name, slot) not in self.id_args) or slot not in args:
            return args, ""
        given = args[slot]
        values = given if type(given) is list else [given]
        fixed = []
        for value in values:
            if type(value) is str and value.strip() in books.rows:
                fixed.append(value.strip())
                continue
            if type(value) is not str or len(value.strip()) < 8 or idish(value, True):
                fixed.append(value)
                continue
            low = value.strip().lower()
            hits = []
            for ident, row in books.rows.items():
                for key, text in row.items():
                    if key != books.id_key and type(text) is str and text.strip().lower().startswith(low):
                        hits.append(ident)
                        break
            if len(hits) != 1:
                if hits:
                    return args, "several listings carry the text %r (%s); pass one listing id" % (value, ", ".join(hits[:6]))
                return args, "%r is not a listing id; pass the id from the result rows (their %s field)" % (value, books.id_key)
            fixed.append(hits[0])
        out = dict(args)
        out[slot] = fixed if type(given) is list else fixed[0]
        return out, ""

    def join_opens(self, calls):
        """Several single opens in one reply become one side-by-side call (when that view shows attributes)."""
        opener, joiner = self.roles.get("open"), self.roles.get("compare")
        target, many = self.single_target(opener), self.compare_target()
        if None in (opener, joiner, target, many) or opener == joiner or self.compare_bare():
            return calls
        spots, ids = [], []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            if fn.get("name") != opener or type(fn.get("arguments")) is not str:
                continue
            args = json_object(fn["arguments"])
            ident = str((args or {}).get(target) or "").strip()
            if args and list(args) == [target] and ident in self.books.rows and ident not in self.books.gone:
                spots.append(index)
                ids.append(ident)
        room = self.compare_room()
        spots, ids = spots[:room], ids[:room]
        if len(spots) < 2:
            return calls
        out = list(calls)
        for index in spots[1:]:
            out[index] = dict(id=calls[index]["id"], joined=True)
        out[spots[0]] = dict(id=calls[spots[0]]["id"], type="function", function=dict(name=joiner, arguments=jdump({many: ids})))
        self.note("joined", ids=ids)
        return out

    # -- one walker call
    def run_calls(self, calls):
        calls = self.join_opens(calls)
        return [(call["id"], self.one_call(call, len(calls))) for call in calls]

    def one_call(self, call, count):
        if call.get("joined"):
            return "Shown in the side-by-side result above: the code opened these listings together in one step."
        ready = self.vet(call, count)
        if type(ready) is str:
            return ready
        name, args, mark, many, extra = ready
        ending = self.is_buy(name, args) or self.is_refuse(name, args)
        if not ending:
            return self.perform(name, args, mark, many, extra, False)
        verdict = self.hold_ending(name, args)
        if type(verdict) is str:
            return verdict
        args = verdict
        if self.left() >= 2:
            go, said = self.gate("order" if self.is_buy(name, args) else "end")
            if not go:
                return "Not run yet. %s\nCart now: %s" % (said, self.cart_text())
            if said:
                extra = "\n[code] " + said + extra
        return self.perform(name, args, mark, many, extra, True)

    def vet(self, call, count):
        """A walker call made ready to run (name, args, repeat mark, compare argument, note), or the text of why it
        is not run (no step used)."""
        fn = call.get("function") or {}
        name, raw = fn.get("name"), fn.get("arguments")
        allowed = sorted(t["function"]["name"] for t in self.offered())
        roles = self.roles
        if self.finished:
            return "Not run: the episode is over."
        if name == roles.get("ask"):
            return TURNED + "the code's interviewer handles every message to the shopper."
        if name not in self.schemas:
            return TURNED + "unknown tool. Tools: %s." % ", ".join(allowed)
        if name not in allowed:
            return TURNED + "too few steps are left for that; use: %s." % ", ".join(allowed)
        if type(raw) is dict:
            args = raw
        elif (raw or "").strip():
            args = json_object(raw)
        else:
            args = {}
        if args is None:
            return TURNED + "the arguments were not one JSON object."
        faults = []
        args = conform(args, self.schemas[name], "args", faults)
        if faults:
            return TURNED + "; ".join(faults[:6])
        args, wrong = self.fix_ids(name, args)
        if wrong:
            return TURNED + wrong + "."
        is_cart = name == roles.get("cart")
        if is_cart:
            args = self.whole_removal(name, args)
        extra = ""
        many = self.compare_target() if name == roles.get("compare") else None
        listed = args.get(many) if many else None
        room = self.compare_room()
        if type(listed) is list and self.compare_bad and len(listed) > room:
            extra = ("\n[code] Sent the first %d listings only: this store shows at most %d side by side. Not shown yet: %s."
                     % (room, room, ", ".join(map(str, listed[room:]))))
            args = dict(args)
            args[many] = listed[:room]
        target = self.single_target(name)
        gone = self.books.gone
        if target and name in (roles.get("open"), roles.get("offer")) and str(args.get(target)) in gone:
            return TURNED + "%s sold out at step %s and will not come back. Replace its cart line with another fitting listing." % (
                args[target], gone[str(args[target])])
        if is_cart and any((as_int(v) or 0) < 0 for v in args.values() if type(v) is not str):
            view = jdump(self.cart_view) if self.cart_seen_at >= self.cart_at else ""
            for value in args.values():
                if type(value) is str and value in self.books.rows and value not in self.cart_lines and value not in view \
                        and value not in self.cart_said:
                    return TURNED + "%s is not in the cart, so there is nothing to take off. Cart now: %s" % (value, self.cart_text())
        ending = self.is_buy(name, args) or self.is_refuse(name, args)
        if ending and count > 1:
            return TURNED + "send the finishing call on its own, after reading your other results."
        if not ending and self.left() <= self.tail() and (roles.get("buy") or roles.get("refuse")):
            return TURNED + ("the last steps are kept for checking out (with the shopper's approval of the basket when it is "
                             "due) or ending without a purchase.")
        mark = name + jdump(args)
        if not (ending or is_cart) and mark in self.recent:
            return TURNED + "you just made this same call and nothing changed; its result is above."
        readers = self.readers()
        if name in readers:
            cached = self.read_cache.get(mark)
            if cached and cached[0] == self.version:
                return TURNED + "the same call at step %s returned this, and nothing has changed since:\n%s" % (cached[1], cached[2][:3000])
            if target and name != roles.get("compare") and self.books.shown_already(name, str(args.get(target)), self.version):
                return TURNED + "%s has shown everything this view gives since the last change. Its record: %s" % (
                    args[target], self.books.record(str(args[target])))
        listed = args.get(many) if many else None
        if type(listed) is list and listed:
            fresh = [i for i in listed if not self.books.shown_already(name, str(i), self.version)]
            if not fresh:
                shown = "\n".join("%s: %s" % (i, self.books.record(str(i))) for i in listed)
                return TURNED + "these listings have shown everything this view gives since the last change. Their records:\n" + shown[:4000]
            if len(fresh) < len(listed):
                extra += "\n[code] Left out %s: shown in full already, nothing changed since." % ", ".join(str(i) for i in listed if i not in fresh)
                args = dict(args)
                args[many] = fresh
                mark = name + jdump(args)
        return name, args, mark, many, extra

    def readers(self):
        return set(filter(None, (self.roles.get(r) for r in ("search", "open", "offer", "compare"))))

    def hold_ending(self, name, args):
        """The code's work before a checkout or a no-purchase ending: a text when the ending is held (no step used,
        or the code finished the episode itself), else the arguments to send."""
        buying, left = self.is_buy(name, args), self.left()
        held = ""
        if buying and left > 2:
            held = self.before_buy()
        elif not buying and left > 5:
            held = self.before_refuse()
        if held:
            return held
        if buying:
            late = self.late_question()
            return "Not run yet. %s\nCart now: %s" % (late, self.cart_text()) if late else args
        if left <= 5 and not self.late_held:
            lines = [i for i in self.built if not self.books.sold_out(i)]
            fresh = self.cart_fresh()
            missing = [i for i in lines if i not in self.cart_ids() or (not fresh and i in self.dropped)]
            if len(lines) >= len(self.items) and len(missing) < left:
                self.late_held = True
                self.note("late_decline_held", lines=lines, missing=missing)
                redo = "Put %s back in the cart at the requested quantities, then check out." % ", ".join(missing) if missing \
                    else "Check out the cart as it is."
                return (HELD + "Ending without a purchase earns nothing when the store has a fitting listing, and running "
                        "short of steps does not show that it has none. " + redo + " End without a purchase only if your "
                        "search showed that some requested item has no listing that meets its requirements; then send the "
                        "ending call again.")
        if not self.leads and not self.pushed and left > self.max_steps // 3:
            self.pushed = True
            return (HELD + "%d steps are still left, so search in ways you have not tried before ending: browse the "
                    "category of each requested item and page through it with the token or offset the store returns, try "
                    "single-word value queries and the property's other names, and open the listings that might fit. End "
                    "without a purchase only after that." % left)
        done = self.complete_cart()
        if done:
            return done + (" [episode finished]" if self.finished else "")
        return self.quiet(name, args)

    def before_buy(self):
        """Holds before a checkout, in order (see ALGORITHM.md section 8); '' lets it go ahead."""
        def with_cart(text):
            return "%s\nCart now: %s" % (text, self.cart_text())
        wider = self.wider_hold()
        if wider:
            return with_cart(wider)
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        viewer = self.roles.get("cart_view")
        if viewer and self.cart_touched and self.cart_at > self.cart_seen_at and self.left() > 3:
            reply, _ = self.use(viewer, {}, absorb=False)
            if reply is not None:
                self.cart_view, self.cart_seen_at = reply.get("observation"), self.step
                change = self.observe(viewer, self.cart_view)
                if change:
                    self.queued = []
                    return (HELD + "Looking at the cart showed a change: " + change + "\nCart now: " + self.cart_text() +
                            "\nApply the change, fix the cart if needed, then check out.")
        swap = self.swap_hint()
        stale = [i for i in self.cart_ids() if i in self.books.gone] if self.held < 2 else []
        if swap:
            return with_cart(HELD + swap)
        if stale:
            self.held += 1
            return with_cart(HELD + "%s in the cart is sold out: replace that line first." % ", ".join(stale))
        chain, hold = self.trail()
        if hold:
            return with_cart(HELD + chain)
        if self.fit_holds < 2 and self.left() > self.tail() + 3:
            misfit = self.fit_faults()
            if misfit:
                self.fit_holds += 1
                self.note("fit_hold", faults=misfit[:4])
                return with_cart(HELD + "A part in the cart does not work with the device in the cart: " +
                                 "; ".join(misfit[:3]) + ". Swap the part (or the device), then check out.")
        faults = self.line_faults() if self.line_holds < 2 and self.left() > self.tail() + 2 else []
        if faults and all(" is held at " in f for f in faults):
            fixed = self.fix_counts(self.count_fixes())
            if fixed:
                faults = self.line_faults()
                if not faults:
                    self.line_holds += 1
                    return with_cart(HELD + "The code set the cart's quantities itself: " + "; ".join(fixed) +
                                     ". If the store refused any of these, fix those lines; then check out.")
        if faults:
            self.line_holds += 1
            self.note("line_hold", faults=faults[:6])
            return with_cart(HELD + "The cart does not match the request yet: " + "; ".join(faults[:6]) +
                             ". Fix those lines (the cart tool's own way to add or take off units), then check out.")
        lines = self.cart_ids()
        read = self.read_facts(lines, self.tail() + (1 if len(lines) > 1 else 3))
        self.show_links(lines, self.tail() + 1)
        read = read + self.cheaper_reads()
        verdict = self.check_cart()
        if verdict:
            seen = "The code read these listings' attributes for the check (one step each):\n" + "\n".join(read) + "\n" if read else ""
            return with_cart("Not run yet. " + chain + seen + verdict)
        if reread and self.wire.spent > K.soft:
            return (HELD + "The code re-read the cart's listings just now:\n" + reread + "\nCheck out again when every line "
                    "still shows stock and a price inside the budget; otherwise replace that line first.")
        return ""

    def before_refuse(self):
        review = self.audit()
        if review:
            return HELD + review
        if not self.leads or self.left() <= 4:
            return ""
        if self.refuse_blocks == 0:
            self.refuse_blocks = 1
            self.note("refuse_blocked", leads=list(self.leads))
            return (HELD + "The review found listings that appear to fit: " + ", ".join(self.leads) + ". Open them if "
                    "needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a "
                    "purchase only if each of them clearly fails a requirement; then say which one in your text and send "
                    "the ending call again.")
        if self.refuse_blocks == 1:
            self.refuse_blocks = 2
            first = self.leads[0]
            self.check_cart([first], K.hard)
            if first in self.leads and self.passed([first]):
                self.confirmed = first
                self.note("refuse_blocked", confirmed=first)
                return (HELD + "An independent check of %s's records says it meets every requirement. Add it to the cart "
                        "at the requested quantity and check out; end without a purchase only if its record clearly breaks "
                        "a requirement." % first)
        return ""

    def perform(self, name, args, mark, many, extra, ending):
        """Run a walker call and build the text it gets back."""
        reply = self.wire.step(name, args)
        if reply is None:
            return "The call failed (network or server). You may try again."
        roles, books = self.roles, self.books
        seen = reply.get("observation")
        flat = unpack(seen)
        listed = args.get(many) if many else None
        if type(listed) is list and len(listed) >= 2:
            dumped = jdump(flat)
            if [i for i in listed if str(i) in dumped]:
                self.compare_ok = max(len(listed), self.compare_ok)
            else:
                self.compare_bad = min(len(listed), self.compare_bad or 999)
                extra += "\n[code] The side-by-side view showed nothing for %d listings: send at most %d per call." % (
                    len(listed), self.compare_room())
        self.recent = self.recent[-2:] + [mark]
        for key in args:
            group = args[key] if type(args[key]) is list else [args[key]]
            if group and not [x for x in group if type(x) is not str or x not in books.rows]:
                self.id_args.add((name, key))
        books.absorb(flat)
        if roles.get("compare") == name:
            rows = books.records_in(flat)
            if rows:
                self.facts_from[name] = self.facts_from.get(name, False) or any(facts_in(r) for _, r in rows)
        if name in self.readers() and roles.get("search") != name:
            books.note_view(name, flat, self.version)
        searching = not ending and name not in [tool for job, tool in roles.items() if job != "search"]
        paged = self.more_pages(name, args, flat) if searching else ""
        self.learn_views(name, flat)
        if roles.get("cart") == name:
            self.after_cart(name, args, seen, flat)
        if roles.get("search") == name and books.records_in(flat):
            self.searched = self.searched[-7:] + [dict(args)]
        if roles.get("cart_view") == name:
            self.cart_view = seen
            self.cart_seen_at = self.step
        if self.is_buy(name, args):
            self.buys += 1
            self.refused_at = self.refused_at if reply.get("finished") else self.step
        return self.reply_text(name, args, reply, flat, paged + extra, searching, mark)

    def after_cart(self, name, args, seen, flat):
        self.note_cart(name, args)
        self.cart_said, self.recent = jdump(seen), []
        counts = [as_int(v) for v in args.values() if type(v) is not str and as_int(v) is not None]
        if counts:
            for value in args.values():
                if type(value) is not str or value not in self.books.rows:
                    continue
                if counts[0] > 0:
                    self.dropped.discard(value)
                elif not carries_id(flat, value):
                    self.dropped.add(value)
        lines = [i for i in self.cart_ids() if i not in self.dropped and not self.books.sold_out(i)]
        if len(lines) >= len(self.built):
            self.built = lines

    def reply_text(self, name, args, reply, flat, extra, searching, mark):
        seen = reply.get("observation")
        pieces = [self.show(seen), extra]
        if searching:
            pieces += [self.cheapest_rows([flat] + self.paged), self.cap_ignored(name, args, flat)]
        base = "".join(pieces)
        proof = self.evidence(flat)
        if proof:
            pieces.append("\n" + proof)
        if name == self.roles.get("cart"):
            zeroed = [v for v in args.values() if type(v) is not str and as_int(v) == 0]
            kept = [v for v in args.values() if type(v) is str and v in self.books.rows and carries_id(flat, v)]
            if zeroed and kept:
                pieces.append("\n[code] A count of 0 left that line in the cart: this tool adds the count to the line, so "
                              "a negative count takes units off.")
        self.note_sold = False
        lines = self.loose_lines(flat)
        meaning = self.take_note(lines) if lines else ""
        if meaning:
            pieces.append("\n[note read] %s. Apply this change to your picks before checking out." % meaning)
        buying = self.is_buy(name, args)
        refused = buying and not reply.get("finished")
        if self.cart_ids() and not self.finished:
            told_again = refused and any(x in self.spoken(seen) for x in self.sold_lines)
            unnamed = self.note_sold and (refused or not (buying or self.note_named))
            if told_again or unnamed:
                found = self.find_sold()
                if found:
                    pieces.append("\n[the code re-read the cart's lines after the refusal]\n" + found)
        self.note_sold = False
        swap = self.swap_hint()
        if swap:
            pieces.append("\n" + swap)
        if self.queued:
            pieces.append("\n[notes seen on the code's own calls] " + " | ".join(self.queued))
            self.queued = []
        if name in self.readers():
            self.read_cache[mark] = (self.version, self.step, base)
        if reply.get("finished"):
            pieces.append(" [episode finished]")
        return "".join(pieces)

    def learn_views(self, name, flat):
        """A single-listing view teaches whether it shows attributes; the open and offer roles follow."""
        roles = self.roles
        busy = (roles.get("cart"), roles.get("buy"), roles.get("refuse"), roles.get("ask"))
        if name in busy or not self.single_target(name) or type(flat) is not dict:
            return
        listing = self.books.ident(flat)
        if not listing:
            return
        shows = self.facts_from.get(name, False) or bool(facts_in(flat))
        self.facts_from[name] = shows
        if not shows and name == roles.get("open"):
            for tool in self.schemas:
                if tool != name and tool not in busy + (roles.get("search"),) and self.facts_from.get(tool) is not False \
                        and self.single_target(tool):
                    roles["open"] = tool
                    break
        if not shows and name != roles.get("open") and not roles.get("offer"):
            roles["offer"] = name
        if name == self.offer_tool():
            self.offer_seen[listing] = self.step

    # -- ending the episode
    def finish(self, again=False):
        """End the episode with the cart held (checked out), else without a purchase; the store's refusal is returned."""
        if self.left() <= 0 or self.finished or self.wire.broke:
            return ""
        fresh = self.cart_fresh()
        holding = len(self.cart_ids()) > 0 if fresh else len(self.cart_lines) > 0
        complete = True
        if len(self.items) > 1 and holding:
            complete = self.fill_short_cart() or not self.can_refuse()
            if complete and len(self.cart_ids()) > 1:
                self.read_facts(self.cart_ids(), self.tail())
                if self.room() > 5 and self.left() > 2:
                    self.show_links(self.cart_ids(), self.tail())
            elif not complete:
                self.note("short_cart_refused", lines=self.cart_ids())
        if self.left() <= 0 or self.finished:
            return ""
        wants_buy = holding or (self.cart_touched and not fresh and self.left() > 1)
        if complete and wants_buy and self.roles.get("buy"):
            args = self.fit_args(self.roles["buy"], dict(self.brief.get("buy_args") or {}))
            if args is not None:
                said = self.end_call("buy", args, again)
                if said or self.finished or self.left() < 1:
                    return said
        tool = self.roles.get("refuse")
        args = self.fit_args(tool, self.quiet(tool, dict(self.brief.get("refuse_args") or {}))) if tool else None
        return self.end_call("refuse", args, again) if args is not None else ""

    def trim(self):
        """Keep the chat under the history size: older tool results cut to a head plus the listing ids they held."""
        size = 0
        for message in self.chat:
            size += len(jdump(message.get("content")))
        if size <= K.history:
            return
        spots = [n for n in range(len(self.chat)) if self.chat[n].get("role") == "tool"]
        keep = self.books.rows
        for spot in spots[:max(1, len(spots) // 2)]:
            body = self.chat[spot].get("content") or ""
            if len(body) <= 600:
                continue
            ids = []
            for found in re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', body):
                if found in keep and found not in ids:
                    ids.append(found)
            self.chat[spot]["content"] = body[:400] + "...(older result shortened; listings in it: %s)" % ", ".join(ids[:20])

    def tool_calls_of(self, message):
        calls = []
        for number, call in enumerate(message.get("tool_calls") or []):
            if type(call) is not dict:
                continue
            fn = call.get("function") if type(call.get("function")) is dict else {}
            raw = fn.get("arguments")
            if type(raw) is dict:
                parsed = raw
            elif type(raw) is str and raw.strip():
                parsed = json_object(raw)
            else:
                parsed = {}
            calls.append(dict(id=str(call.get("id") or "k%d_%d" % (self.turns, number)), type="function",
                              function=dict(name=str(fn.get("name") or ""), arguments=jdump(parsed if type(parsed) is dict else {}))))
        return calls

    def keep_walking(self, turns, line):
        if self.finished or self.wire.broke or self.left() <= 0 or self.turns >= turns or self.wire.spent >= line:
            return False
        return self.clock() < K.last_turn or (not self.cart_ids() and self.room() > K.empty_room)

    def walk(self, turns, line):
        idle = 0
        while self.keep_walking(turns, line):
            if self.ready_to_ask():
                summary = self.interview()
                if summary:
                    self.chat.append(dict(role="user", content="%s\n%s" % (summary, self.status())))
                continue
            self.trim()
            self.turns += 1
            message = self.wire.chat(self.chat, [M_LIGHT, M_SPARE], 3000, self.offered())
            if message is None:
                self.misses += 1
                stop = self.wire.no_credit or self.wire.calls >= K.calls
                if stop or self.misses > 2:
                    break
                continue
            said = message.get("content")
            if type(said) is not str:
                said = jdump(said) if said else ""
            calls = self.tool_calls_of(message)
            if len(calls) == 0:
                idle += 1
                self.chat.append(dict(role="assistant", content=said if said else "(no call)"))
                self.chat.append(dict(role="user", content="Call one of the tools to continue.\n" + self.status()))
                if idle > 2:
                    break
                continue
            idle = 0
            self.chat.append(dict(role="assistant", content=said or None, tool_calls=calls))
            if self.ask_before_commit(calls):
                continue
            self.note("turn", text=said[:300], calls=[c["function"] for c in calls])
            answers = self.run_calls(calls)
            for spot, (cid, text) in enumerate(answers, 1):
                if spot == len(answers):
                    text = text + "\n" + self.status()
                self.chat.append(dict(role="tool", tool_call_id=cid, content=text))

    def run(self):
        if self.finished:
            return self.trace
        self.read_request()
        self.note("brief", brief=self.brief)
        opening = self.view.get("opening")
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            self.books.absorb(unpack(opening))
            intro += "\n\nWHAT THE STORE SHOWS FIRST:\n" + self.show(opening)
        intro += ("\n\nStart by searching for each item in its category and opening several listings in full, so the "
                  "interviewer can see which properties these listings carry.\n\n" + self.status())
        self.chat = [dict(role="system", content=P_WALK), dict(role="user", content=intro)]
        self.walk(K.turns, K.hard)
        if not self.finished:
            refused = self.finish(again=True)
            if refused and not (self.finished or self.wire.broke or self.wire.no_credit) and self.left() > 1:
                self.chat.append(dict(role="user", content="[code] The code tried to end the episode and " + refused +
                                      "\nApply what the store says, then finish.\n" + self.status()))
                self.walk(self.turns + K.extra_turns, K.hard + K.extra_spend)
            if not self.finished:
                self.finish()
        self.note("end", finished=self.finished, steps=self.step, cost=round(self.wire.spent, 5), calls=self.wire.calls)
        return self.trace

    def end_call(self, job, args, again):
        """Send the forced checkout or no-purchase call (the gate first); text of the store's refusal, else ''."""
        tool = self.roles[job]
        if self.left() > 1:
            self.gate("order" if job == "buy" else "end")
        if self.finished or self.left() < 1:
            return ""
        reply = self.wire.step(tool, args)
        self.note("forced_" + job, reply=shorten(reply, 200))
        if job == "buy":
            if self.finished or self.left() < 1 or not again or reply is None:
                return ""
            self.buys += 1
            self.refused_at = self.step
        elif reply is None or self.finished:
            return ""
        self.observe(tool, reply.get("observation"))
        lead = "the checkout was refused: " if job == "buy" else "ending without a purchase was refused: "
        return lead + self.show(reply.get("observation"))

    def ask_before_commit(self, calls):
        """A cart, checkout or no-purchase call before the interview ran is not run: the shopper is asked first."""
        roles = self.roles
        if self.asked or not roles.get("ask") or self.msgs_free() <= 0 or self.left() <= 8 or self.clock() > K.ask_by:
            return False
        commits = [roles.get("cart"), roles.get("buy"), roles.get("refuse")]
        if not [c for c in calls if c["function"]["name"] in commits and c["function"]["name"]]:
            return False
        for c in calls:
            self.chat.append(dict(role="tool", tool_call_id=c["id"], content=TURNED + "the shopper is asked first."))
        summary = self.interview() or "The interviewer had nothing to ask."
        self.chat.append(dict(role="user", content=summary + "\n" + self.status()))
        return True


# ----------------------------------------------------------------------------------------------- the episode

class Episode(TaskSide, ShopSide, AskSide, CartSide, CheckSide, WalkSide):
    """One task from start to finish; the sides above hold its behaviour, this class holds its state."""

    def __init__(self, problem):
        world = problem["environment"]
        view = world["policy_view"]
        self.binding, self.view = dict(world["binding"]), view
        self.request = str(view.get("query") or "")
        self.tools = []
        for tool in view.get("tools") or []:
            fn = tool.get("function") if type(tool) is dict else None
            if type(fn) is dict and type(fn.get("name")) is str:
                self.tools.append(tool)
        self.schemas = {}
        for tool in self.tools:
            spec = tool["function"].get("parameters")
            self.schemas[tool["function"]["name"]] = spec if type(spec) is dict else dict(type="object", properties={})
        cap = view.get("max_steps")
        steps = as_int(cap)
        if steps is None and is_number(cap):
            steps = int(cap)
        self.max_steps = steps if steps and steps > 0 else 30
        self.started, self.step, self.finished = time.monotonic(), 0, bool(view.get("finished"))
        self.trace = []
        self.wire, self.books, self.param_cache = Wire(self), Books(self.schemas), {}
        # the task
        self.brief, self.roles, self.items, self.budget = {}, {}, [], None
        self.msg_limit = self.msg_used = self.reserve = self.gate_msgs = 0
        self.process, self.refs, self.ref_values, self.bound, self.adopted = [], [], {}, [], {}
        self.version, self.notes, self.heard, self.queued = 0, [], set(), []
        self.note_sold = self.note_named = False
        self.sold_lines = set()
        # the walk
        self.chat, self.recent, self.searched, self.paged = [], [], [], []
        self.turns = self.misses = self.pages_read = self.compare_ok = self.compare_bad = 0
        self.read_cache, self.facts_from, self.shapes, self.id_args = {}, {}, {}, set()
        # the interview
        self.asked = self.late_asked = False
        self.listing = True
        self.lanes, self.lane_found, self.lane_spent, self.lane_used, self.lane_misses = {}, {}, {}, {}, {}
        self.plan_order, self.follow_up, self.one_product = [], [], set()
        self.done_log, self.done_steps, self.gate_ok = [], [], set()
        # the cart and the endings
        self.cart_lines, self.cart_view, self.cart_said = {}, None, ""
        self.cart_touched = False
        self.cart_at = self.cart_seen_at = self.refused_at = -1
        self.built, self.dropped, self.buys, self.offer_seen = [], set(), 0, {}
        self.checked, self.verdicts, self.leads, self.lead_items, self.confirmed = set(), {}, [], {}, None
        self.audited = self.completed = self.pushed = self.late_held = self.wider_held = False
        self.refuse_blocks = self.held = self.line_holds = self.fit_holds = 0
        self.trails = self.trail_blocks = self.link_searches = 0
        self.link_tried, self.link_seen, self.capped, self.cheaper_marks = set(), {}, set(), set()
        self.cart_read_held, self.bare_done = set(), set()

    def note(self, kind, **fields):
        entry = dict(kind=kind)
        entry.update(fields)
        self.trace.append(entry)


def _crash(error):
    return dict(kind="crash", error=("%s: %s" % (type(error).__name__, error))[:400])


def agent_main(problem_data):
    episode = None
    try:
        episode = Episode(problem_data)
        return episode.run()
    except Exception as error:
        if episode is None:
            return [_crash(error)]
        episode.trace.append(_crash(error))
    try:
        if not episode.finished:
            episode.finish()
    except Exception as error:
        episode.trace.append(_crash(error))
    return episode.trace


__all__ = ["agent_main"]
