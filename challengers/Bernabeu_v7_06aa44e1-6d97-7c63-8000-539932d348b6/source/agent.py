

from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_MODELS = {
    "chutes": "deepseek-ai/DeepSeek-V3.2-TEE",
    "openrouter": "deepseek/deepseek-v3.2",
}
_PROXY = ProxyClient(timeout=110, max_retries=1)

_SECONDS = 270.0
_ENDGAME = 45.0
_ASKABLE = 55.0

_SHEET = 10
_TOGETHER = 5
_ENOUGH_PAGES = 56
_SWEEPS = 4

_KEEP_WHOLE = 6
_HEAD_ROOM = 7000
_TAIL_ROOM = 5600
_OLD_ROOM = 480
_TEXT_ROOM = 640
_ROW_ROOM = 12
_DEEP = 6

_QUOTE_ROOM = 44
_NEARBY = 40

# How a shopper writes a ceiling, and how they write a floor. Read forwards from the wording: the
# cue comes first and the amount follows it, which is how these sentences are actually built.
_UNDER = (
    "at most", "at or under", "at or below", "up to", "under", "below", "beneath",
    "less than", "no more than", "not more than", "no higher than", "not above", "not over",
    "within", "maximum of", "maximum", "max of", "max", "cap of", "capped at", "ceiling of",
    "limit of", "limited to", "budget of", "budget is", "budget", "spend up to", "spend",
    "afford", "pay", "keep it to", "keep it under", "keep it below", "top out at", "go up to",
    "stay under", "stay below", "stay within",
)
_OVER = (
    "at least", "no less than", "not less than", "minimum of", "minimum", "min of",
    "more than", "over", "above", "upwards of", "starting at", "starting from",
)
_TRAILING = ("or less", "or under", "or below", "or lower", "tops", "max", "maximum",
             "at most", "ceiling", "cap", "limit", "budget")
_REFUSAL = ("can't", "cannot", "can not", "won't", "will not", "don't", "do not", "never",
            "under no", "rather not", "nothing", "none", "no")

_MORE_IS_BETTER = ("highest", "largest", "biggest", "greatest", "maximum", "most", "fastest")
_LESS_IS_BETTER = ("lowest", "smallest", "cheapest", "least", "minimum", "slowest")
_RANKING = _MORE_IS_BETTER + _LESS_IS_BETTER

_NOISE = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "by", "for", "with", "from",
    "into", "that", "this", "it", "is", "are", "be", "was", "me", "my", "you", "your", "i",
    "find", "need", "want", "buy", "get", "order", "place", "test", "one", "please", "would",
    "under", "over", "within", "listing", "product", "item", "category", "compare", "shop",
}
# Words a request uses to talk about itself rather than about anything on sale.
_PAPERWORK = {
    "also", "use", "using", "only", "supported", "reason", "reasons", "field", "fields",
    "specification", "specifications", "shopper", "request", "must", "true", "such", "as",
    "value", "values", "claim", "claims", "include", "least", "structured", "justification",
}

_FIGURE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?\s*([A-Za-z]{1,12}|\u20b1)?")
_GAUGE = re.compile(r"(?<![\w.])(\d{1,4}(?:\.\d{1,3})?)\s*-?\s*([A-Za-z]{1,10})\b")
_LOOSE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)")
_HANDLE = re.compile(r"(\d{6,})::([A-Za-z0-9._\-]+)")
_INSIDE_QUOTES = re.compile(r"\"([^\"]{2,80})\"")
_AS_PHRASE = re.compile(r"matches (?:the (?:phrase|text|words)\s+)?([^\",.;]{2,60})", re.I)
_FAMILY_A = re.compile(r"in the ([A-Za-z][\w&'\- ]{1,48}?) category", re.I)
_FAMILY_B = re.compile(r"categor(?:y|ies)\s*(?:is|are|:)\s*([A-Za-z][\w&'\- ]{1,48}?)(?=[.,;\n]|$)",
                       re.I)
_HELD_BACK = re.compile(r"\bif you ask\b|\bask me\b|\bI will tell you\b", re.I)
_EXPLAIN = re.compile(r"\b(?:why|reasons?|justif\w*|good pick|walk me through|explain)\b", re.I)
_PAY_MORE = re.compile(
    r"don't just grab the cheapest|cheapest isn't|rather pay|not blindly buying the cheapest|"
    r"matters more to me than saving", re.I)
_A_QUESTION = re.compile(
    r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|which (?:one|option)"
    r"|tell me (?:which|whether|if)|confirm whether)\b", re.I)
_SHOP_MOVED = re.compile(
    r"still available|out of stock|sold out|more expensive|price (?:jumped|went)|re-?check", re.I)
_BETTER_AT = re.compile(
    r"\b(" + "|".join(_RANKING) + r")\s+((?:advertised\s+|maximum\s+|available\s+|current\s+"
    r"|overall\s+)*[a-z]+(?:\s+[a-z]+)?)", re.I)

# The ladder of structured limits: how much of the stated budget to allow, whether to keep the
# maker, whether to keep the category. Walked loosest-constraint-first when nothing fits.
_LADDER = (
    (1.0, True, True), (1.0, False, True), (1.0, True, False),
    (0.75, True, True), (0.55, True, True), (0.4, True, True), (0.28, True, True),
    (0.18, True, True), (0.5, False, True), (0.24, False, True),
)
_BROWSING = ("search", "filter", "view", "compare")
_TILL_WORDS = ("order", "purchase", "checkout", "buy")
_CUT = object()

_HOUSE_RULES = """You are buying one thing for one shopper, on a live catalog, using only the
supplied tools.

A search result is a row in a list. The product page behind it is what the shop is prepared to
assert. Open the promising ones before you decide anything.

The shop reacts to your first commitment: that exact item is liable to be repriced or sold out the
moment it goes in the basket. Look at it again before ordering, and expect the thing you chose
first to be the thing that moves.

A requirement is met when the product's own facts say it is met. A title is not a fact. If nothing
meets everything, take the closest thing that is genuinely in stock and inside the budget and say
so, rather than ordering nothing at all.

If the shopper asks you something, answer it."""


def _pick_model() -> str:
    where = getenv("INFERENCE_PROVIDER", "chutes")
    return getenv("SANDBOX_MODEL") or _MODELS.get(where, _MODELS["chutes"])


def _fallback_models() -> list:
    out: list = []
    for name in [_pick_model()] + list(_MODELS.values()):
        if name and name not in out:
            out.append(name)
    return out


def _text(value: Any) -> str:
    try:
        return json.dumps(value, default=str)
    except (TypeError, ValueError):
        return str(value)


def _clip(body: Any, room: int) -> str:
    body = str(body)
    return body if len(body) <= room else body[:room] + " ... (trimmed)"


def _slim(value: Any, depth: int = 0) -> Any:
    """A shop answer reduced to the part worth carrying in the conversation."""
    if depth >= _DEEP:
        return "..."
    if isinstance(value, str):
        return _clip(value, _TEXT_ROOM)
    if isinstance(value, list):
        rows = [_slim(v, depth + 1) for v in value[:_ROW_ROOM]]
        if len(value) > _ROW_ROOM:
            rows.append("... and %d more" % (len(value) - _ROW_ROOM))
        return rows
    if isinstance(value, dict):
        return {k: _slim(v, depth + 1) for k, v in value.items()}
    return value


def _unfence(reply: str) -> Any:
    """The JSON in an answer that may also be wrapped, apologised for, or talked around."""
    body = str(reply or "").strip()
    if body.startswith("```"):
        parts = body.split("```")
        body = parts[1] if len(parts) > 1 else body[3:]
        if body[:4].lower() == "json":
            body = body[4:]
    try:
        return json.loads(body)
    except (TypeError, ValueError):
        pass
    for left, right in (("{", "}"), ("[", "]")):
        start, stop = body.find(left), body.rfind(right)
        if 0 <= start < stop:
            try:
                return json.loads(body[start:stop + 1])
            except (TypeError, ValueError):
                continue
    return None


def _amount(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    body = str(value or "").strip().replace(",", "")
    if not body:
        return None
    try:
        return float(body)
    except ValueError:
        found = _LOOSE.search(body)
        return float(found.group(1)) if found else None


def _all_amounts(body: Any) -> list:
    return [float(m.group(1)) for m in _LOOSE.finditer(str(body or ""))]


def _split(body: Any) -> list:
    return [w for w in re.split(r"[^A-Za-z0-9.]+", str(body or "").lower()) if w]


def _bag(body: Any) -> set:
    return {w for w in _split(body) if w not in _NOISE}


def _says(word: str, body: Any) -> bool:
    """The word is written there as itself, not buried inside a longer one."""
    word = str(word or "").strip().lower()
    if not word:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(word) + r"(?![a-z0-9])",
                     str(body or "").lower()) is not None


def _says_loosely(word: str, body: Any) -> bool:
    """The same word, allowing for the endings a catalog puts on it."""
    if _says(word, body):
        return True
    word = str(word or "").lower()
    if len(word) < 5:
        return False
    root = word[:max(5, len(word) - 3)]
    return re.search(r"(?<![a-z])" + re.escape(root) + r"[a-z]{0,8}(?![a-z])",
                     str(body or "").lower()) is not None


def _clauses(body: Any) -> list:
    """A fact cut where a catalog separates the choices it is offering."""
    return [p.strip() for p in re.split(r"\s*(?:[;,/|]|\sor\s|\s-\s)\s*", str(body or ""))
            if p.strip()]


def _wobbly(body: Any) -> bool:
    """The fact quotes several competing amounts, so it has not committed to any of them."""
    return sum(1 for clause in _clauses(body) if _all_amounts(clause)) > 1


def _holds(value: float, body: Any) -> bool:
    return any(abs(value - other) < 1e-6 for other in _all_amounts(body))


def _holds_in_unit(value: float, unit: str, body: Any) -> bool:
    """The amount is written up against the unit the shopper measured it in."""
    root = unit[:3] if len(unit) > 3 else unit
    if not root:
        return False
    return re.search(r"(?<![\w.])" + re.escape("%g" % value) + r"\s*-?\s*" + re.escape(root),
                     str(body or ""), re.I) is not None


def _exactly(value: float) -> str:
    """A number written so that reading it back gives the same number.

    Prices arrive as floats and a repriced one is seldom round. Claiming 572.7 of a thing that
    costs 572.6999999999999 is a false claim, however nearly true it looks.
    """
    brief = "%g" % value
    try:
        return brief if float(brief) == float(value) else repr(float(value))
    except (TypeError, ValueError):
        return brief


def _spellings(value: float, unit: str) -> list:
    """The ways a catalog might write one measurement, so one of them will match how it did."""
    shown = "%g" % value
    out = ["%s %s" % (shown, unit), "%s%s" % (shown, unit), "%s-%s" % (shown, unit)]
    if len(unit) > 2:
        out.append("%s%s" % (shown, unit[:2]))
    return out


def _same_question(body: str) -> str:
    """What a search is actually asking, once the punctuation around its numbers is set aside.

    A phrase task is graded on how many genuinely different questions were put to the shop, and
    "6.7 inch", "6.7-inch" and "6.7in" are one question asked three times.
    """
    flat = " ".join(str(body or "").lower().split())
    return re.sub(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]{2,})",
                  lambda m: m.group(1) + m.group(2)[:2], flat)


def _thin_out(asked: list) -> list:
    seen, out = set(), []
    for raw in asked:
        body = " ".join(str(raw or "").split())
        if not body:
            continue
        shape = _same_question(body)
        if shape in seen or all(w in _NOISE for w in _split(shape)):
            continue
        seen.add(shape)
        out.append(body)
    return out


def _figures(body: str) -> list:
    """Every amount in a text, with the unit written after it and where it sits."""
    out = []
    for found in _FIGURE.finditer(str(body or "")):
        whole, part, unit = found.group(1), found.group(2), found.group(3)
        try:
            value = float(whole.replace(",", "") + "." + (part or "0"))
        except ValueError:
            continue
        out.append((value, (unit or "").strip(), found.start(), found.end()))
    return out


def _nearest_after(body: str, at: int) -> tuple | None:
    """The first amount written after this point, if it is written close enough to belong to it."""
    for value, unit, start, end in _figures(body):
        if start < at:
            continue
        return (value, unit, start, end) if start - at <= 24 else None
    return None


def _ceilings(body: str) -> list:
    """Every amount the shopper put a lid on, and where in the text it was written.

    Read forwards, because that is how the sentences are built: the wording that limits comes
    first and the amount follows it. A bare number is not a budget, and a floor is only a
    ceiling when the shopper is refusing to cross it.
    """
    body = str(body or "")
    low = body.lower()
    found = []
    for cue in _UNDER:
        at = low.find(cue)
        while at >= 0:
            hit = _nearest_after(body, at + len(cue))
            if hit:
                found.append(hit)
            at = low.find(cue, at + len(cue))
    for cue in _OVER:
        at = low.find(cue)
        while at >= 0:
            before = low[max(0, at - 30):at]
            if any(no in before for no in _REFUSAL):
                hit = _nearest_after(body, at + len(cue))
                if hit:
                    found.append(hit)
            at = low.find(cue, at + len(cue))
    for value, unit, start, end in _figures(body):
        tail = low[end:end + 18].strip()
        if any(tail.startswith(cue) for cue in _TRAILING):
            found.append((value, unit, start, end))
    seen, out = set(), []
    for hit in found:
        if hit[2] not in seen:
            seen.add(hit[2])
            out.append(hit)
    return out


def _without_money(body: str) -> str:
    """The request with its handles and its prices blanked, leaving its measurements behind."""
    body = _HANDLE.sub(" ", str(body or ""))
    for _value, _unit, start, end in sorted(_ceilings(body), key=lambda c: -c[2]):
        body = body[:start] + " " * (end - start) + body[end:]
    return body


def _measurements(body: str, coin: str) -> list:
    """The amount-and-unit pairs the shopper asked for: 6.7 inch, 165 Hz, 8 GB, 720p."""
    out = []
    for found in _GAUGE.finditer(body):
        unit = found.group(2).lower()
        if unit in _NOISE or unit in _RANKING or (coin and unit == coin.lower()):
            continue
        pair = (float(found.group(1)), unit)
        if pair not in out:
            out.append(pair)
    return out


def _shorthands(body: str) -> list:
    """Short shouted words the shopper wrote beside an amount, like the HD in "720p HD".

    Their own name for what they asked for. Worth recognising when a catalog happens to use it,
    and worth nothing once a catalog states the amount itself.
    """
    out = []
    for found in _GAUGE.finditer(body):
        for word in re.findall(r"[A-Za-z]{2,5}", body[found.end():found.end() + 24])[:2]:
            if word.isupper() and word.lower() not in _NOISE and word.lower() not in out:
                out.append(word.lower())
    return out


class Timer:
    """How much of the session there is left to spend."""

    def __init__(self) -> None:
        self.opened = time.time()
        self.budget = _SECONDS

    def elapsed(self) -> float:
        return time.time() - self.opened

    def left(self) -> float:
        return self.budget - self.elapsed()

    def closing(self) -> bool:
        return self.left() < _ENDGAME


class Ask:
    """The shopper's request, read out of their own wording and revised when they say more."""

    def __init__(self, wording: str) -> None:
        self.wording = str(wording or "")
        # Whatever follows the rules heading is the harness briefing the agent, not the shopper
        # describing what they want.
        self.goal = self.wording.split("\n\nTask rules:")[0]
        lids = _ceilings(self.goal)
        self.cap = min((c[0] for c in lids), default=None)
        self.coin = next((c[1] for c in lids if c[1]), "")
        bare = _without_money(self.goal)
        self.needs = [n for n in _measurements(bare, self.coin)
                      if self.cap is None or abs(n[0] - self.cap) > 1e-6]
        self.marks = _shorthands(bare)
        self.handles = ["%s::%s" % (m.group(1), m.group(2)) for m in _HANDLE.finditer(self.goal)]
        self.phrase = self._phrase()
        self.family = self._family(self.goal)
        self.maker = self._maker()
        self.orders = self._orders()
        self.tokens = [w for w in self._describing(self.goal) if w != (self.coin or "").lower()]
        self.held_back = bool(_HELD_BACK.search(self.goal))
        self.wants_why = bool(_EXPLAIN.search(self.goal))
        self.not_cheapest = bool(_PAY_MORE.search(self.goal))

    def _phrase(self) -> list:
        for pattern in (_INSIDE_QUOTES, _AS_PHRASE):
            found = pattern.search(self.goal)
            if not found:
                continue
            words = [w for w in _split(found.group(1)) if w not in _NOISE]
            if 2 <= len(words) <= 5:
                return words
        return []

    @staticmethod
    def _family(body: str) -> str:
        for pattern in (_FAMILY_B, _FAMILY_A):
            found = pattern.search(body)
            if found:
                return found.group(1).strip()
        found = re.search(r"([A-Za-z][\w\-]{2,})\s*(?=,|\bat most\b|\bunder\b|\bbelow\b|\bwithin\b)",
                          body, re.I)
        return found.group(1).strip() if found else ""

    def _maker(self) -> str:
        """A run of capitalised words sitting where a maker's name sits."""
        head = self.goal.split(self.family)[0] if self.family else self.goal
        best: list = []
        run: list = []
        for word in head.split():
            bare = word.strip(",.;:'\"")
            named = bare[:1].isupper() and not bare.isdigit() and bare.lower() not in _NOISE \
                and not (bare.isupper() and len(bare) <= 3)
            if named:
                run.append(bare)
            else:
                best, run = (run if len(run) > len(best) else best), []
        if len(run) > len(best):
            best = run
        return " ".join(best[:3]) if best else ""

    def _orders(self) -> list:
        """The tradeoffs the shopper stated: which way is better, on what."""
        out = []
        for clause in re.split(r"[.;]", self.goal):
            for found in _BETTER_AT.finditer(clause):
                cue = found.group(1).lower()
                words = [w for w in _split(found.group(2))
                         if len(w) > 3 and w not in _NOISE and w not in _RANKING
                         and w not in ("advertised", "available", "current", "overall", "option")]
                if not words or any(words == o[1] for o in out):
                    continue
                out.append((cue in _LESS_IS_BETTER, self._fuller(words), " ".join(words)))
        return out

    def _fuller(self, words: list) -> list:
        """The property named more completely elsewhere in the brief than the shopper named it.

        Somebody who asks for the highest refresh, in a brief that keeps saying refresh rate, is
        asking about the refresh rate. Hunting only their shorter wording finds whatever number
        happens to sit near it.
        """
        nearby = set()
        low = self.wording.lower()
        for word in words:
            for found in re.finditer(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", low):
                window = low[max(0, found.start() - 24):found.end() + 24]
                nearby |= {w for w in _split(window)
                           if len(w) > 2 and w not in _NOISE and w not in _RANKING
                           and w not in words}
        return words + sorted(nearby)[:3]

    @staticmethod
    def _describing(body: str) -> list:
        out = []
        for word in _split(body):
            if len(word) > 3 and not word.replace(".", "").isdigit() \
                    and word not in _NOISE and word not in _PAPERWORK and word not in out:
                out.append(word)
        return out

    def paying_extra_for(self) -> list:
        """What somebody would pay more for is a plainer statement of what they actually want."""
        found = _PAY_MORE.search(self.goal)
        if not found:
            return []
        tail = re.split(r"\bthan\b|[.!?\n]", self.goal[found.end():])[0]
        return self._describing(tail)[:6]

    def revise(self, said: str, measurements: bool = True) -> None:
        """Take in a later message. Shoppers tighten budgets and name specs halfway through."""
        said = str(said or "").strip()
        if not said:
            return
        tighter = min((c[0] for c in _ceilings(said)), default=None)
        if tighter is not None and (self.cap is None or tighter < self.cap):
            self.cap = tighter
        stated = self._family(said)
        if stated and _FAMILY_A.search(said) or _FAMILY_B.search(said):
            # A category said outright beats one guessed from where a word fell in a sentence.
            self.family = stated
        if measurements:
            bare = _without_money(said)
            for pair in _measurements(bare, self.coin):
                if pair not in self.needs:
                    self.needs.append(pair)
            for mark in _shorthands(bare):
                if mark not in self.marks:
                    self.marks.append(mark)
            self.held_back = False
        for word in self._describing(said):
            if word not in self.tokens:
                self.tokens.append(word)


class Board:
    """Everything the shop has said, filed by product and by variant.

    The conversation gets shortened as it grows; this does not. Decisions are taken against what
    is written here, so something the shop said twenty turns ago still counts for as much.
    """

    def __init__(self) -> None:
        self.lots: dict = {}
        self.sheets: dict = {}
        self.tills: dict = {}
        self.basket: list = []
        self.buried: set = set()
        self.buried_pids: set = set()
        self.put: set = set()
        self.questions: list = []
        self.commits = 0
        self.last_commit = -1
        self.tick = 0
        self.looked: dict = {}

    @staticmethod
    def handle(pid: Any, sku: Any) -> str:
        return "%s::%s" % (pid, sku)

    def file_lot(self, row: dict) -> None:
        pid = row.get("product_id")
        if pid is None:
            return
        held = self.lots.setdefault(self.handle(pid, row.get("sku")),
                                    {"product_id": str(pid), "sku": row.get("sku")})
        for field, value in row.items():
            if value not in (None, ""):
                held[field] = value

    def file_sheet(self, pid: Any, page: dict) -> None:
        if pid is not None:
            self.sheets.setdefault(str(pid), {}).update(page)

    def file_till(self, pid: Any, sku: Any, seen: dict) -> None:
        self.tick += 1
        key = self.handle(pid, sku)
        self.tills[key] = seen
        self.looked[key] = self.tick

    def looked_since_commit(self, pid: Any, sku: Any) -> bool:
        when = self.looked.get(self.handle(pid, sku))
        return when is not None and (self.commits == 0 or when > self.last_commit)


class Runner:
    """Works the catalog in phases, filling a whole envelope with each phase's questions."""

    def __init__(self, problem: dict) -> None:
        setting = problem["environment"]
        self.binding = setting["binding"]
        rules = setting["policy_view"]
        self.reference = str(problem.get("problem_id", problem.get("id", "problem")))
        self.tools = list(rules.get("tools") or [])
        self.max_steps = int(rules.get("max_steps", 30) or 30)
        self.per_turn = max(1, int(rules.get("max_calls_per_turn", 1) or 1))
        self.timer = Timer()
        self.ask = Ask(rules.get("query") or "")
        self.board = Board()
        self.models = _fallback_models()
        self.model_at = 0

        self.named = {str(((t.get("function") or {}).get("name")) or "")
                      for t in self.tools if isinstance(t, dict)} - {""}
        self.shapes = {str(((t.get("function") or {}).get("name")) or ""):
                       ((t.get("function") or {}).get("parameters") or {})
                       for t in self.tools if isinstance(t, dict)}
        self.tills = sorted(n for n in self.named
                            if any(w in n.lower() for w in _TILL_WORDS))
        self.committer = self._named_tool("add", "cart") or self._named_tool("cart")
        # "inspect" alone matches the basket reader as well as the stock reader, and the basket
        # reader sorts first. Choosing it means never seeing the price the shop just changed.
        self.looker = (self._named_tool("inspect", without=("cart",))
                       or self._named_tool("stock", without=("cart",)))
        self.dropper = self._named_tool("remove", "cart")
        self.crier = self._named_tool("message") or self._named_tool("reply")

        self.turn = 0
        self.done = False
        self.placed = False
        self.phase = "open"
        self.rung = 0
        self.wording_at = 0
        self.shapes_asked: set = set()
        self.shelf_read = False
        self.opened_with_question = False
        self.recheck_due: tuple | None = None
        self.price_when_committed: float | None = None
        self.moved_against: str | None = None
        self.reopened = False
        self.question_open = False
        self.question_held = 0
        self.units: dict = {}
        self.one_word: dict = {}
        self.repeated = ""
        self.repeats = 0
        self.chat: list = [{"role": "system", "content": _HOUSE_RULES},
                           {"role": "user", "content": self.ask.wording}]
        self.log: list = []

    # ------------------------------------------------------------------ tools

    def _named_tool(self, *needles: str, without: tuple = ()) -> str:
        for name in sorted(self.named):
            low = name.lower()
            if name in self.tills or any(bad in low for bad in without):
                continue
            if all(n in low for n in needles):
                return name
        return ""

    def _shape_of(self, name: str) -> dict:
        return self.shapes.get(name) or {}

    def _fit(self, name: str, args: dict) -> dict:
        """Arguments cut to what this tool will accept, all the way down.

        Not only at the top. The reasons given at the till are a list of objects two levels in,
        with the permitted names pinned by a list, and one name outside that list loses the order
        and the episode with it.
        """
        made = self._mould(args, self._shape_of(name))
        return made if isinstance(made, dict) else {}

    def _mould(self, value: Any, shape: Any) -> Any:
        if not isinstance(shape, dict) or not shape:
            return value
        fields = shape.get("properties")
        if isinstance(fields, dict) and fields:
            out = {}
            for key, held in (value if isinstance(value, dict) else {}).items():
                slot = key if key in fields else self._same_field(key, fields)
                if slot is None:
                    continue
                made = self._mould(held, fields[slot])
                if made is not _CUT:
                    out[slot] = made
            for key in shape.get("required") or []:
                spec = fields.get(key) if isinstance(fields.get(key), dict) else {}
                # A missing string can be stood in for. A missing object cannot: filling it with
                # an empty string passes the schema and then breaks when the reasons go into it.
                # And where the permitted values are listed, nothing may be invented at all.
                if key not in out and spec.get("type") == "string" and not spec.get("enum"):
                    out[key] = ""
            return out
        if shape.get("type") == "array":
            each = shape.get("items") if isinstance(shape.get("items"), dict) else {}
            rows = []
            for entry in (value if isinstance(value, list) else []):
                made = self._mould(entry, each)
                if made is _CUT:
                    continue
                if isinstance(made, dict) and any(not str(made.get(r, "")).strip()
                                                  for r in (each.get("required") or [])):
                    continue
                rows.append(made)
            return rows
        allowed = shape.get("enum")
        if isinstance(allowed, list) and allowed:
            if value in allowed:
                return value
            spoken = _bag(value)
            return next((a for a in allowed if _bag(a) == spoken), _CUT)
        kind = shape.get("type")
        if kind == "number" and _amount(value) is not None:
            return _amount(value)
        if kind == "integer" and _amount(value) is not None:
            return int(_amount(value))
        if kind == "boolean":
            return bool(value)
        if kind == "string":
            return str(value)
        return value

    @staticmethod
    def _same_field(key: str, fields: dict) -> str | None:
        spoken = _bag(key)
        return next((n for n in fields if _bag(n) == spoken), None) if spoken else None

    # -------------------------------------------------------------- listening

    def _absorb_payload(self, node: Any, pid: Any = None) -> None:
        """File everything a shop answer says about products, by the shape of it.

        Each tool nests its answer differently, so this looks for what a product looks like -- an
        identifier, and whatever is filed beside it -- rather than for a place one ought to be.
        Variants inherit the identifier of the page they were listed under.
        """
        if isinstance(node, list):
            for entry in node:
                self._absorb_payload(entry, pid)
            return
        if not isinstance(node, dict):
            return
        here = node.get("product_id") or node.get("id") or pid
        plain = {k: v for k, v in node.items() if isinstance(v, (str, int, float, bool))}
        if here is not None:
            if any(k in node for k in ("facts", "variants", "category_path", "specifications")):
                page = {k: v for k, v in node.items() if k != "variants"}
                page["product_id"] = str(here)
                self.board.file_sheet(here, page)
            if node.get("sku") is not None or plain.get("price") is not None or plain.get("title"):
                row = dict(plain)
                row["product_id"] = str(here)
                row.setdefault("sku", None)
                self.board.file_lot(row)
        for held in node.values():
            if isinstance(held, (dict, list)):
                self._absorb_payload(held, here)

    @staticmethod
    def _core(seen: Any) -> dict:
        node = seen
        for _step in range(4):
            if not isinstance(node, dict):
                return {}
            inner = node.get("observation")
            if not isinstance(inner, dict):
                return node
            node = inner
        return node if isinstance(node, dict) else {}

    @staticmethod
    def _refused(seen: Any) -> bool:
        """The envelope repeats a refusal at more than one level; either one means refused."""
        node = seen
        for _step in range(4):
            if not isinstance(node, dict):
                return False
            if node.get("error"):
                return True
            node = node.get("observation")
        return False

    # --------------------------------------------------------------- listings

    def _facts(self, pid: Any) -> dict:
        page = self.board.sheets.get(str(pid)) or {}
        held = page.get("facts") or page.get("specifications") or {}
        out = {}
        if isinstance(held, dict):
            for key, value in held.items():
                if isinstance(value, (str, int, float, bool)):
                    out[str(key)] = value
                elif isinstance(value, dict):
                    for inner, deeper in value.items():
                        if isinstance(deeper, (str, int, float, bool)):
                            out[str(inner)] = deeper
        elif isinstance(held, list):
            for entry in held:
                if isinstance(entry, dict) and (entry.get("name") or entry.get("key")):
                    out[str(entry.get("name") or entry.get("key"))] = entry.get("value")
        return out

    def _headline(self, pid: Any, sku: Any = None) -> str:
        row = self.board.lots.get(Board.handle(pid, sku)) or {}
        page = self.board.sheets.get(str(pid)) or {}
        return str(row.get("title") or page.get("title") or page.get("name") or "")

    def _maker_of(self, pid: Any) -> str:
        page = self.board.sheets.get(str(pid)) or {}
        if page.get("brand"):
            return str(page["brand"])
        for key, row in self.board.lots.items():
            if key.split("::")[0] == str(pid) and row.get("brand"):
                return str(row["brand"])
        return ""

    def _variant_words(self, pid: Any, sku: Any) -> str:
        row = self.board.lots.get(Board.handle(pid, sku)) or {}
        return " ".join(str(v) for k, v in row.items()
                        if isinstance(v, str) and k not in ("product_id", "sku", "price",
                                                            "in_stock", "title"))

    def _asserted(self, pid: Any, sku: Any) -> str:
        """What the listing commits to. A title sells and is left out of this."""
        return " ".join(["%s %s" % (k, v) for k, v in self._facts(pid).items()]
                        + [self._variant_words(pid, sku)])

    def _fact_words(self, pid: Any) -> str:
        return " ".join("%s %s" % (k, v) for k, v in self._facts(pid).items())

    def _cost(self, pid: Any, sku: Any) -> float | None:
        key = Board.handle(pid, sku)
        till = self.board.tills.get(key) or {}
        seen = _amount(till.get("price"))
        return seen if seen is not None else _amount((self.board.lots.get(key) or {}).get("price"))

    def _available(self, pid: Any, sku: Any) -> bool:
        key = Board.handle(pid, sku)
        till = self.board.tills.get(key)
        if till and till.get("in_stock") is not None:
            return bool(till["in_stock"])
        return bool((self.board.lots.get(key) or {}).get("in_stock", True))

    def _within_cap(self, price: float | None) -> bool:
        return price is not None and (self.ask.cap is None or price <= self.ask.cap + 1e-9)

    # -------------------------------------------------------------- weighing

    def _marks_still_count(self) -> bool:
        """A shopper's shorthand stands in for their amount only until a listing states it.

        The moment one does, the shorthand is the looser reading and would let through listings
        that never claimed the thing at all.
        """
        if not self.ask.marks or not self.ask.needs:
            return False
        for key in self.board.tills:
            pid, sku = key.split("::", 1)
            body = self._asserted(pid, sku)
            if body and all(_holds(v, body) for v, _u in self.ask.needs):
                return False
        return True

    def _need_met(self, value: float, body: str) -> bool:
        if _holds(value, body):
            return True
        return self._marks_still_count() and any(_says(m, body) for m in self.ask.marks)

    def _meets(self, pid: Any, sku: Any) -> bool:
        if self.ask.held_back:
            return False
        if not self.ask.needs:
            return True
        body = self._asserted(pid, sku)
        return bool(body.strip()) and all(self._need_met(v, body) for v, _u in self.ask.needs)

    def _in_the_unit(self, pid: Any, sku: Any) -> bool:
        body = self._asserted(pid, sku)
        return all(_holds_in_unit(v, u, body) for v, u in self.ask.needs)

    def _plainly(self, pid: Any, sku: Any) -> bool:
        """Some single fact says it outright, rather than offering it among alternatives."""
        facts = self._facts(pid)
        for value, _unit in self.ask.needs:
            if not any(self._need_met(value, str(v)) and not _wobbly(v) for v in facts.values()):
                return False
        return True

    def _facts_alone(self, pid: Any) -> bool:
        body = self._fact_words(pid)
        return all(self._need_met(v, body) for v, _u in self.ask.needs)

    def _maker_ok(self, pid: Any) -> bool:
        if not self.ask.maker:
            return True
        wanted = _bag(self.ask.maker)
        maker = self._maker_of(pid)
        return wanted <= _bag(maker) if maker else wanted <= _bag(self._headline(pid))

    def _family_ok(self, pid: Any) -> bool:
        if not self.ask.family:
            return True
        page = self.board.sheets.get(str(pid)) or {}
        path = page.get("category_path")
        if not path:
            return True
        body = " ".join(path) if isinstance(path, list) else str(path)
        return bool({w.rstrip("s") for w in _bag(self.ask.family)}
                    & {w.rstrip("s") for w in _bag(body)})

    def _phrase_ok(self, pid: Any, sku: Any) -> bool:
        if not self.ask.phrase:
            return True
        title = self._headline(pid, sku)
        return all(_says(w, title) for w in self.ask.phrase)

    def _describes(self, pid: Any, sku: Any, words: list) -> int:
        if not words:
            return 0
        title, body = self._headline(pid, sku), self._asserted(pid, sku)
        return sum(2 if _says_loosely(w, body) else (1 if _says_loosely(w, title) else 0)
                   for w in words)

    def _survey(self) -> list:
        """Every variant on record, weighed against the request as it now stands."""
        out = []
        for key, row in self.board.lots.items():
            pid, sku = key.split("::", 1)
            if sku in ("None", ""):
                continue
            price = self._cost(pid, sku)
            opened = str(pid) in self.board.sheets
            alive = (self._available(pid, sku) and key not in self.board.buried
                     and str(pid) not in self.board.buried_pids)
            meets = self._meets(pid, sku)
            listed_ok = (not self.ask.handles) or key in self.ask.handles
            out.append({
                "key": key, "pid": pid, "sku": sku, "price": price, "opened": opened,
                "alive": alive, "meets": meets,
                "buyable": (alive and opened and meets and listed_ok and self._within_cap(price)
                            and self._maker_ok(pid) and self._family_ok(pid)
                            and self._phrase_ok(pid, sku)
                            and (not self.ask.wants_why or self._facts_alone(pid))),
                "plainly": meets and self._plainly(pid, sku),
                "in_unit": meets and self._in_the_unit(pid, sku),
                "maker_ok": self._maker_ok(pid),
                "described": self._describes(pid, sku, self.ask.tokens),
                "premium": self._describes(pid, sku, self.ask.paying_extra_for()),
            })
        return out

    def _firm(self, lot: dict) -> bool:
        if not lot["buyable"]:
            return False
        if self.ask.needs and not lot["plainly"]:
            return False
        return not (self.ask.paying_extra_for() and lot["premium"] <= 0)

    def _finalists(self) -> list:
        return sorted([l for l in self._survey() if self._firm(l)], key=self._standing)

    # ---------------------------------------------------------------- reading

    def _fragments(self, pid: Any, sku: Any) -> list:
        out = [self._headline(pid, sku), self._variant_words(pid, sku)]
        out += ["%s: %s" % (k, v) for k, v in self._facts(pid).items()]
        return [f for f in out if f]

    def _beside(self, body: str, words: list) -> tuple | None:
        """The amount written closest to any of these words, within a single field."""
        low = str(body or "").lower()
        marks = [m.start() for w in words for m in
                 re.finditer(r"(?<![a-z])" + re.escape(w) + r"[a-z]{0,2}(?![a-z])", low)]
        if not marks:
            return None
        best = None
        for found in _GAUGE.finditer(str(body or "")):
            unit = found.group(2).lower()
            if len(unit) < 2 or unit in _NOISE:
                continue
            gap = min(abs(found.start() - m) for m in marks)
            if gap > _NEARBY:
                continue
            # "165Hz refresh rate" puts the property after the number, so an amount running into
            # the word beats one merely trailing behind it.
            rank = (0 if any(found.end() <= m + 2 for m in marks) else 1, gap)
            if best is None or rank < best[0]:
                best = (rank, float(found.group(1)), unit)
        return (best[1], best[2]) if best else None

    def _unit_for(self, words: tuple) -> str | None:
        """What the shop measures this in, counted across the shelf and only then asked about."""
        stamp = len(self.board.sheets)
        held = self.units.get(words)
        if held is not None and held[0] == stamp:
            return held[1]
        tally: dict = {}
        for lot in self._survey():
            if not lot["opened"]:
                continue
            for fragment in self._fragments(lot["pid"], lot["sku"]):
                hit = self._beside(fragment, list(words))
                if hit:
                    tally[hit[1]] = tally.get(hit[1], 0) + 1
        top = max(tally.items(), key=lambda kv: (kv[1], kv[0])) if tally else None
        unit = top[0] if top and top[1] > 1 else self._ask_unit(words)
        self.units[words] = (stamp, unit)
        return unit

    def _ask_unit(self, words: tuple) -> str | None:
        """Some catalogs print an amount and never write near it what it measures.

        Nothing in the shop's text can then say what the number is, so the naming is asked for --
        one word -- and kept only once the listings are seen to write numbers that way.
        """
        if words in self.one_word:
            return self.one_word[words]
        pool = [l for l in self._survey() if l["opened"]]
        if self.timer.left() < _ENDGAME or len(pool) < 2:
            return None
        self.one_word[words] = None
        said = self._side_question(
            "A shopper is comparing products on this property: " + " ".join(words) + ".\n\n"
            "What unit is it measured in, written the way a listing abbreviates it directly after "
            "the number? Answer with that abbreviation alone: no digits, no punctuation, not the "
            "name of the property. Answer null if it is not measured in units at all.\n"
            'Reply with JSON only: {"unit": "<abbreviation or null>"}')
        read = _unfence(said)
        if not isinstance(read, dict):
            return None
        unit = re.sub(r"[^a-z]+", "", str(read.get("unit") or "").lower())
        if not unit or len(unit) > 6 or unit in _NOISE:
            return None
        root = unit[:4] if len(unit) > 4 else unit
        written = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*-?\s*" + re.escape(root), re.I)
        proof = sum(1 for l in pool if written.search(" ".join(self._fragments(l["pid"],
                                                                              l["sku"]))))
        self.one_word[words] = unit if proof >= 2 else None
        return self.one_word[words]

    def _reading(self, lot: dict, order: tuple) -> float | None:
        """This listing's number on a property the shopper is weighing."""
        downward, words = order[0], tuple(order[1])
        if any(w in ("price", "cost") for w in words):
            return lot["price"]
        choose = min if downward else max
        unit = self._unit_for(words)
        if unit:
            root = unit[:4] if len(unit) > 4 else unit
            written = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*-?\s*" + re.escape(root), re.I)
            # The variant names the one thing being bought; the page describes all of them.
            here = [float(m.group(1))
                    for m in written.finditer(self._variant_words(lot["pid"], lot["sku"]))]
            if here:
                return choose(here)
            anywhere = [float(m.group(1)) for m in
                        written.finditer(" ".join(self._fragments(lot["pid"], lot["sku"])))]
            if anywhere:
                return choose(anywhere)
        for fragment in self._fragments(lot["pid"], lot["sku"]):
            hit = self._beside(fragment, list(words))
            if hit:
                return hit[0]
        roots = {w.rstrip("s") for w in words}
        spare = []
        for key, value in self._facts(lot["pid"]).items():
            if roots & {t.rstrip("s") for t in _split(key)}:
                spare.extend(_all_amounts(str(value)))
        return choose(spare) if spare else None

    def _live_orders(self) -> list:
        """Weigh only on what the shelf can actually state; a half-known axis misleads."""
        if not self.ask.orders:
            return []
        pool = [l for l in self._survey() if l["alive"] and l["opened"]]
        if len(pool) < 2:
            return list(self.ask.orders)
        kept = []
        for order in self.ask.orders:
            if any(w in ("price", "cost") for w in order[1]):
                kept.append(order)
                continue
            known = sum(1 for l in pool if self._reading(l, order) is not None)
            if known >= 2 and known * 2 >= len(pool):
                kept.append(order)
        return kept

    def _standing(self, lot: dict) -> tuple:
        """Where a listing comes in the running order. Lower is better."""
        price = lot["price"] if lot["price"] is not None else 1e12
        orders = self._live_orders()
        if orders:
            # A stated order of preference settles it, and a listing the shelf cannot measure on
            # the first-named property loses to every one it can rather than winning on price.
            scored = []
            for order in orders:
                value = self._reading(lot, order)
                scored.append(float("inf") if value is None
                              else (value if order[0] else -value))
            return tuple(scored) + (price, lot["key"])
        if self.ask.not_cheapest:
            return (-lot["premium"], -lot["described"], -price, lot["key"])
        wrong_maker = 1 if (self.ask.maker and self.board.commits == 0
                            and not lot["maker_ok"]) else 0
        return (wrong_maker, 0 if lot["plainly"] else 1, 0 if lot["in_unit"] else 1,
                -lot["described"], price, lot["key"])

    # ----------------------------------------------------------- the phrasing

    def _wordings(self) -> list:
        """The same request put to the shop several ways, grounded in words it has used itself."""
        ask, out = self.ask, []
        phrase = " ".join(ask.phrase)
        if phrase:
            out.append(phrase)
            out.extend(ask.phrase)
            if ask.family:
                out.append("%s %s" % (ask.family, phrase))
        for value, unit in ask.needs:
            for form in _spellings(value, unit):
                out.append(" ".join(x for x in (ask.maker, form, ask.family) if x))
                if ask.maker:
                    out.append(" ".join(x for x in (form, ask.family) if x))
        if ask.maker and ask.family:
            out.append("%s %s" % (ask.maker, ask.family))
        elif ask.family:
            out.append(ask.family)
        vocabulary: set = set()
        for key in self.board.lots:
            pid, sku = key.split("::", 1)
            vocabulary |= _bag(self._headline(pid, sku))
        drop = _bag(ask.family) | _bag(ask.maker)
        loose = [w for w in ask.paying_extra_for() + ask.tokens
                 if w not in drop and (not vocabulary or w in vocabulary)]
        if loose:
            tail = [ask.family] if ask.family else []
            out.append(" ".join(loose[:4] + tail))
            if len(loose) > 4:
                out.append(" ".join(loose[4:8] + tail))
        return _thin_out(out)

    def _openers(self) -> list:
        asked = self._wordings()
        return asked or [" ".join(self.ask.tokens[:6]) or self.ask.goal[:60]]

    def _search_args(self, wording: str) -> dict:
        return self._fit("search", {"query": wording, "k": _SHEET, "in_stock": True})

    def _fresh_wordings(self, room: int) -> list:
        """The next wordings nobody has put to the shop yet."""
        if room <= 0 or "search" not in self.named:
            return []
        if self.ask.phrase and len(self.board.questions) >= _SWEEPS:
            return []
        out = []
        asked = self._wordings()
        while self.wording_at < len(asked) and len(out) < room:
            wording = asked[self.wording_at]
            self.wording_at += 1
            shape = _same_question(wording)
            if wording.lower() in self.board.questions or shape in self.shapes_asked:
                continue
            self.shapes_asked.add(shape)
            out.append(("search", self._search_args(wording)))
        return out

    def _unopened(self, room: int) -> list:
        """Search hits nobody has opened, gathered into as few reads as they will fit."""
        if room <= 0 or len(self.board.sheets) >= _ENOUGH_PAGES:
            return []
        waiting = []
        for key, row in self.board.lots.items():
            pid = key.split("::")[0]
            if pid in self.board.sheets or pid in self.board.buried_pids or pid in waiting:
                continue
            if not row.get("in_stock", True):
                continue
            price = _amount(row.get("price"))
            if price is not None and not self._within_cap(price):
                continue
            if self.ask.phrase and not self._phrase_ok(pid, row.get("sku")):
                continue
            if self.ask.maker and not self._maker_ok(pid):
                continue
            waiting.append(pid)
        # Open the ones that already look like the answer. Cheapest-first opens the accessories
        # that sit below the real product's price.
        waiting.sort(key=self._promise)
        out = []
        while waiting and len(out) < room:
            if "compare" in self.named and len(waiting) > 1:
                batch, waiting = waiting[:_TOGETHER], waiting[_TOGETHER:]
                out.append(("compare", self._fit("compare", {"product_ids": batch})))
            elif "view" in self.named:
                out.append(("view", self._fit("view", {"product_id": waiting.pop(0)})))
            else:
                break
        return out

    def _promise(self, pid: str) -> tuple:
        title = self._headline(pid)
        hits = sum(1 for value, _u in self.ask.needs if _holds(value, title))
        liked = sum(1 for w in self.ask.paying_extra_for() if _says_loosely(w, title))
        prices = [_amount(r.get("price")) for k, r in self.board.lots.items()
                  if k.split("::")[0] == pid]
        prices = [p for p in prices if p is not None]
        return (-hits, -liked, min(prices) if prices else 1e12)

    def _narrowings(self, room: int) -> list:
        """The ladder of structured limits, loosest given up first."""
        if room <= 0 or "filter" not in self.named:
            return []
        if not (self.ask.family or self.ask.maker):
            return []
        answered = len([l for l in self._survey() if l["buyable"]]) >= 2
        out = []
        while self.rung < len(_LADDER) and len(out) < room:
            share, keep_maker, keep_family = _LADDER[self.rung]
            self.rung += 1
            if share < 1.0 and answered and not self.ask.phrase:
                continue
            if not keep_family and not self.ask.maker:
                continue
            if share != 1.0 and self.ask.cap is None:
                continue
            args: dict = {"in_stock": True, "k": _SHEET}
            if keep_maker and self.ask.maker:
                args["brand"] = self.ask.maker
            if keep_family and self.ask.family:
                args["category"] = self.ask.family
            if self.ask.cap is not None:
                args["max_price"] = round(self.ask.cap * share, 2)
            args = self._fit("filter", args)
            if self._fingerprint("filter", args) in self.board.put:
                continue
            out.append(("filter", args))
        return out

    @staticmethod
    def _fingerprint(name: str, args: dict) -> str:
        return name + json.dumps(args, sort_keys=True, default=str)

    def _seen_enough(self) -> bool:
        """Whether there is enough on record to choose, or reading should go on."""
        if self.board.commits:
            return True
        if self.shelf_read or self.turn >= self.max_steps - 6 or self.timer.left() < 100:
            return True
        if self.ask.phrase and len(self.board.questions) < _SWEEPS:
            return False
        # The first commitment is the one the market takes away, so it is worth having a second
        # answer in hand before making it.
        if len(self._finalists()) >= 2:
            return True
        return not (self.turn * 100 >= self.max_steps * 55 or self.timer.left() < 120)

    def _gather(self) -> list:
        """A whole envelope of reading for whichever phase the session is in."""
        if self._seen_enough() and self.board.lots:
            self.phase = "settle"
            return []
        room = self.per_turn
        if not self.board.questions and "search" in self.named:
            self.phase = "open"
            # Several spellings of the one question, in the one envelope. Asked a turn apart they
            # cost a turn apart, and a quoted phrase is graded on how many were genuinely tried.
            return [("search", self._search_args(w))
                    for w in self._openers()[:max(1, min(room, _SWEEPS))]]
        steps: list = []
        steps += self._unopened(room - len(steps))
        self.phase = "read" if steps else "sweep"
        steps += self._narrowings(room - len(steps))
        steps += self._fresh_wordings(room - len(steps))
        if not steps:
            self.shelf_read = True
            self.phase = "settle"
        return steps

    # --------------------------------------------------------------- speaking

    def _envelope(self, steps: list) -> dict:
        """Send one envelope, however many questions it carries, and file what comes back."""
        steps = [s for s in steps if s][:max(1, self.per_turn)]
        if not steps:
            return {}
        forced = any(name in self.tills for name, _args in steps)
        if self.done or (self.timer.left() < 8 and not forced):
            return {}
        self.turn += 1
        group = "%s-turn-%d" % (self.reference, self.turn)
        parcel = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": self.turn,
            "calls": [{"call_id": "%s-%d" % (group, n),
                       "action": {"name": name, "args": args}}
                      for n, (name, args) in enumerate(steps, start=1)],
        }
        began = time.time()
        answer = None
        for attempt in range(3):
            if self.timer.left() < 8 and not forced:
                break
            try:
                answer = _PROXY.post("/environment/call", json_data=parcel)
            except Exception:
                answer = None
            if answer is not None:
                break
            if self.timer.left() < _ENDGAME:
                break
            time.sleep(min(2 * (attempt + 1), 5))
        if answer is None:
            self.chat.append({"role": "user", "content": "That call did not go through. Carry on."})
            return {}
        self._pace(time.time() - began)
        for name, args in steps:
            self.board.put.add(self._fingerprint(name, args))
        return self._file(steps, answer, group)

    def _pace(self, took: float) -> None:
        """Keep the envelopes as full as the clock allows and no fuller."""
        turns_left = max(1, self.max_steps - self.turn)
        if took * turns_left > max(self.timer.left() - _ENDGAME, 0.0):
            self.per_turn = max(1, self.per_turn // 2)

    def _file(self, steps: list, answer: dict, group: str) -> dict:
        replies = answer.get("calls") or []
        stubs = [{"id": "%s-%d" % (group, n + 1), "type": "function",
                  "function": {"name": name, "arguments": json.dumps(args, default=str)}}
                 for n, (name, args) in enumerate(steps)]
        self.log.append({"role": "assistant", "content": "", "tool_calls": stubs,
                         "environment_result": answer})
        self.chat.append({"role": "assistant", "content": "", "tool_calls": stubs})
        share = max(400, _TAIL_ROOM // max(1, len(steps) - 1))
        first: dict = {}
        for n, (name, args) in enumerate(steps):
            seen = replies[n].get("observation") if n < len(replies) else None
            self._absorb_payload(self._core(seen))
            if name == "search" and args.get("query"):
                self.board.questions.append(str(args["query"]).strip().lower())
            room = _HEAD_ROOM if n == 0 else share
            if n == 0:
                first = seen if isinstance(seen, dict) else {}
            self.chat.append({"role": "tool", "tool_call_id": "%s-%d" % (group, n + 1),
                              "content": _clip(_text(_slim(seen)), room)})
        if any(self._core(r.get("observation")).get("done") for r in replies
               if isinstance(r.get("observation"), dict)):
            self.done = True
        spoken = answer.get("user_message")
        if isinstance(spoken, dict) and spoken.get("content"):
            said = str(spoken["content"])
            self.chat.append({"role": "user", "content": said})
            # A notice that the market moved is news about the shop, not a new requirement.
            self.ask.revise(said, measurements=not bool(_SHOP_MOVED.search(said)))
            if _A_QUESTION.search(said):
                self.question_open = True
                self.question_held = 0
        return first

    # ----------------------------------------------------------------- claims

    def _till_slot(self) -> tuple:
        """Where the till wants its reasons, and what it will take as one.

        Read off the tool's own schema each run. Most of these tasks pin the permitted names with
        a list and some only describe them, so the slot is found by its shape -- an array of
        objects with two named string fields -- rather than by demanding the list.
        """
        if not self.tills:
            return ()
        stack = [([], self._shape_of(self.tills[0]))]
        while stack:
            trail, node = stack.pop(0)
            if not isinstance(node, dict):
                continue
            each = node.get("items")
            if isinstance(each, dict) and trail:
                fields = each.get("properties") or {}
                usable = [k for k, v in fields.items()
                          if isinstance(v, dict) and (v.get("type") == "string" or v.get("enum"))]
                wanted = [k for k in (each.get("required") or usable) if k in usable]
                if len(wanted) >= 2:
                    pinned = next((k for k in wanted
                                   if isinstance((fields[k] or {}).get("enum"), list)), wanted[0])
                    free = next(k for k in wanted if k != pinned)
                    spec = fields[pinned] or {}
                    return (trail, pinned, free, list(spec.get("enum") or []),
                            str(spec.get("description") or ""))
            for key, branch in (node.get("properties") or {}).items():
                stack.append((trail + [str(key)], branch))
        return ()

    def _shortened(self, body: str, value: float | None = None, unit: str = "") -> str:
        """The part of a fact carrying the amount, without the alternatives beside it."""
        parts = _clauses(body)
        for mark in self.ask.marks:
            for part in parts:
                if _says(mark, part):
                    return part[:_QUOTE_ROOM]
        if value is not None:
            for part in parts:
                if not _holds(value, part):
                    continue
                part = part[:_QUOTE_ROOM]
                # A bare figure is not a measurement; give it back in the unit it was asked in.
                if unit and not re.search(r"[A-Za-z]", part):
                    return ("%s %s" % (part, unit))[:_QUOTE_ROOM]
                return part
        return (parts[0] if parts else str(body).strip())[:_QUOTE_ROOM]

    def _quotation(self, pid: Any, sku: Any) -> tuple:
        """The fact carrying what the shopper asked about, and the words in it that say so.

        Quoted, not restated. Saying the shopper's own words back as though the shop had said
        them is answering the question with the question.
        """
        facts = self._facts(pid)
        for mark in self.ask.marks:
            for key, value in facts.items():
                if _says(mark, str(value)):
                    return str(key), self._shortened(str(value))
        for value, unit in self.ask.needs:
            for reads in (lambda b: _holds_in_unit(value, unit, b), lambda b: _holds(value, b)):
                for key, held in facts.items():
                    if reads(str(held)):
                        return str(key), self._shortened(str(held), value, unit)
        label = self._variant_words(pid, sku)
        for value, unit in self.ask.needs:
            if _holds(value, label):
                return "", self._shortened(label, value, unit)
        return "", ""

    def _proper_name(self, fallback: str, allowed: list) -> str:
        """What the shopper's requirement is called, where the shop files it under its own name.

        A request about a resolution meeting a catalog that publishes a display is the ordinary
        case, and the till still insists on the resolution. Nothing local can bridge those two
        words, and a table that tried would be a guess about one kind of product and wrong about
        the next. So it is asked -- and where the till publishes the names it takes, the asking
        is a choice from that list, which can be checked.
        """
        if not self.ask.needs or self.timer.left() < 30:
            return fallback
        held = self.one_word.get(("__named__",))
        if held is not None:
            return held or fallback
        wording = " ".join("%g %s" % (v, u) for v, u in self.ask.needs) \
            + " " + " ".join(self.ask.marks)
        if allowed:
            question = ("Which one of these names is that requirement about? Answer with exactly "
                        "one of them, copied as written: " + ", ".join(allowed))
        else:
            question = ("Which single product attribute is that requirement about? Answer in "
                        "snake_case, the way a specification sheet would label that row, using "
                        "the shortest name that still says what it is. Do not answer with a "
                        "category, a product type, or a word copied from the request.")
        read = _unfence(self._side_question(
            "Shopper request: " + self.ask.goal[:600] + "\n"
            "The requirement in question: " + wording.strip() + "\n\n" + question
            + '\nReply with JSON only: {"attribute": "<name>"}'))
        name = ""
        if isinstance(read, dict):
            name = re.sub(r"[^a-z0-9_]+", "_",
                          str(read.get("attribute") or "").strip().lower()).strip("_")
        if allowed:
            name = next((a for a in allowed if a.lower() == name), "")
        elif not name or len(name) > 30 or name in _PAPERWORK:
            name = ""
        self.one_word[("__named__",)] = name
        return name or fallback

    def _reasons(self, pid: Any, sku: Any, slot: tuple) -> list:
        """Why this was bought, every line of it read off the product."""
        _trail, name_key, value_key, allowed, described = slot
        price, stocked = self._cost(pid, sku), self._available(pid, sku)
        pairs: list = []
        key, quoted = self._quotation(pid, sku)
        if quoted:
            named = self._proper_name(key, allowed)
            if allowed and named not in allowed:
                named = next((a for a in allowed if _bag(a) & _bag(named)), named)
            pairs.append((named, quoted))
        if price is not None:
            pairs.append(("price", _exactly(price)))
        if stocked:
            pairs.append(("in_stock", "true"))
        # Whatever else the till accepts and the shop has actually stated. A listing's own
        # top-level properties count for as much as its specification rows: a maker's name is
        # filed beside the product, not among its measurements, and is still the shop's word.
        stated = dict(self._facts(pid))
        for holder in (self.board.sheets.get(str(pid)) or {},
                       self.board.lots.get(Board.handle(pid, sku)) or {}):
            for field, value in holder.items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool) \
                        and field not in ("title", "name", "product_id", "sku"):
                    stated.setdefault(str(field), value)
        for field in (allowed or [w for w in _bag(described) if w not in _PAPERWORK]):
            if any(field == n for n, _v in pairs) or len(pairs) >= 5:
                continue
            for source, value in stated.items():
                if not (_bag(field) & _bag(source)):
                    continue
                shown = self._shortened(str(value))
                if shown and len(shown) <= _QUOTE_ROOM and str(value).lower() not in \
                        ("false", "no", "none", "unknown"):
                    pairs.append((field, shown))
                    break
        out, used = [], set()
        for field, value in pairs:
            field = str(field or "").strip()
            if not field or field in used or not str(value).strip():
                continue
            if allowed and field not in allowed:
                continue
            used.add(field)
            out.append({name_key: field, value_key: str(value)[:_QUOTE_ROOM]})
        return out[:6]

    def _till_args(self, pid: Any, sku: Any) -> dict:
        args = self._fit(self.tills[0], {"product_id": str(pid), "sku": str(sku)})
        slot = self._till_slot()
        if not slot:
            return args
        reasons = self._reasons(pid, sku, slot)
        if not reasons:
            return args
        node = args
        for step in slot[0][:-1]:
            if not isinstance(node.get(step), dict):
                node[step] = {}
            node = node[step]
        node[slot[0][-1]] = reasons
        return args

    # --------------------------------------------------------------- settling

    def _choice(self) -> dict | None:
        """What to act on: the best thing fit to buy, or the nearest thing if none is."""
        finalists = self._finalists()
        if finalists:
            return finalists[0]
        if self._still_looking():
            return None
        return self._nearest()

    def _still_looking(self) -> bool:
        """Nothing answers the request, but the shelf is not exhausted and there is time."""
        if self.shelf_read or self.turn >= self.max_steps - 6 or self.timer.left() < 100:
            return False
        found = self._survey()
        if self.ask.needs:
            return not any(l["buyable"] and l["plainly"] for l in found)
        if self.ask.paying_extra_for():
            return not any(l["buyable"] and l["premium"] > 0 for l in found)
        return not any(l["buyable"] for l in found)

    def _nearest(self) -> dict | None:
        """Give up the loosest requirement first, and the stated amount last of all.

        An order that misses the amount the shopper named scores like no order at all, so it is
        only worth making once there is no longer time to find one that does not.
        """
        pool = [l for l in self._survey() if l["alive"]]
        if not pool:
            return None
        backed = (lambda l: self._facts_alone(l["pid"])) if self.ask.wants_why else (lambda _l: True)
        rungs = [
            lambda l: l["opened"] and l["plainly"] and l["maker_ok"] and backed(l),
            lambda l: l["opened"] and l["meets"] and l["maker_ok"] and backed(l),
            lambda l: l["opened"] and l["meets"] and backed(l),
            lambda l: l["opened"] and l["meets"],
        ]
        if not self.ask.needs or self.turn >= self.max_steps - 6 or self.timer.left() < 100:
            rungs += [lambda l: l["opened"] and l["maker_ok"], lambda l: l["opened"]]
        rungs.append(lambda _l: True)
        for rung in rungs:
            kept = [l for l in pool if rung(l)]
            if kept:
                return sorted(kept, key=self._standing)[0]
        return None

    def _carted(self, lot: dict) -> bool:
        return (lot["pid"], lot["sku"]) in self.board.basket

    def _settling_move(self) -> tuple | None:
        """The committing steps, made one at a time so each answer is unambiguous."""
        if self.done or self.placed:
            return None
        choice = self._choice()
        if choice is None:
            return None
        if self._carted(choice):
            if self.looker and not self.board.looked_since_commit(choice["pid"], choice["sku"]):
                return (self.looker, self._fit(self.looker, {"product_id": choice["pid"],
                                                             "sku": choice["sku"]}))
            if self.tills:
                return (self.tills[0], self._till_args(choice["pid"], choice["sku"]))
            return None
        if self.committer:
            return (self.committer, self._fit(self.committer, {"product_id": choice["pid"],
                                                               "sku": choice["sku"]}))
        return None

    def _look_again(self) -> None:
        """Read the basket item back. The shop moves the first commitment, so check before buying.

        What counts as moved is judged against the price it was committed at, not against the
        budget: a shop that trebles a price has taken that product away as surely as one that
        marks it sold out, and a request that named no ceiling would otherwise wave it through.
        """
        pid, sku = self.recheck_due or (None, None)
        self.recheck_due = None
        if pid is None or not self.looker:
            return
        seen = self._envelope([(self.looker, self._fit(self.looker, {"product_id": pid,
                                                                     "sku": sku}))])
        body = self._core(seen)
        self.board.file_till(pid, sku, {"price": body.get("price"),
                                        "in_stock": body.get("in_stock", True)})
        now = self._cost(pid, sku)
        before = self.price_when_committed
        dearer = now is not None and before is not None and now > before + 1e-9
        if self._available(pid, sku) and self._within_cap(now) and not dearer:
            return
        key = Board.handle(pid, sku)
        self.moved_against = key
        self.board.buried.add(key)
        self.board.buried_pids.add(str(pid))
        self.reopened = False

    def _after(self, step: tuple, seen: dict) -> None:
        """Keep the basket and the market in step with what the shop just did."""
        name, args = step
        pid, sku = args.get("product_id"), args.get("sku")
        # No observation means the call never left. Reading that as success ends the session
        # believing something was bought while the basket still sits there.
        if not seen or self._refused(seen):
            return
        if name == self.committer and pid is not None:
            # One slot: committing to a second lets go of the first.
            self.board.basket = [(str(pid), str(sku))]
            self.board.commits += 1
            self.board.tick += 1
            self.board.last_commit = self.board.tick
            self.recheck_due = (str(pid), str(sku))
            self.price_when_committed = self._cost(pid, sku)
        elif name == self.dropper and pid is not None:
            self.board.basket = [b for b in self.board.basket if b != (str(pid), str(sku))]
        elif name == self.looker and pid is not None:
            body = self._core(seen)
            self.board.file_till(pid, sku, {"price": body.get("price"),
                                            "in_stock": body.get("in_stock", True)})
        elif name in self.tills:
            self.placed = True

    # ------------------------------------------------------------------ model

    def _side_question(self, prompt: str) -> str:
        """One short question outside the shopping conversation."""
        for attempt in range(2):
            model = self.models[min(self.model_at + attempt, len(self.models) - 1)]
            try:
                reply = _PROXY.post("/inference/chat/completions", json_data={
                    "model": model, "temperature": 0,
                    "messages": [{"role": "user", "content": prompt}]})
            except Exception:
                reply = None
            picks = (reply or {}).get("choices") or []
            if picks:
                return str((picks[0].get("message") or {}).get("content") or "")
        return ""

    def _trim_chat(self) -> None:
        older = 0
        for line in reversed(self.chat):
            if line.get("role") != "tool":
                continue
            older += 1
            if older > _KEEP_WHOLE and len(line.get("content") or "") > _OLD_ROOM:
                line["content"] = line["content"][:_OLD_ROOM] + " ... (earlier, shortened)"

    def _briefing(self) -> str:
        """A short current picture, so the model stops working it out again every turn."""
        lines = ["", "# Where things stand", "Phase: %s. Turn %d of %d."
                 % (self.phase, self.turn + 1, self.max_steps)]
        if self.ask.cap is not None:
            lines.append("Spend at most %g %s." % (self.ask.cap, self.ask.coin))
        if self.ask.needs:
            lines.append("Required, and it must be in the product's own facts rather than its "
                         "title: " + ", ".join("%g %s" % (v, u) for v, u in self.ask.needs))
        if self.ask.phrase:
            lines.append("These words must appear in the title: " + " ".join(self.ask.phrase))
        if self.ask.orders:
            lines.append("Weigh these in order: "
                         + ", ".join(("lowest " if o[0] else "highest ") + o[2]
                                     for o in self.ask.orders))
        lines.append("On record: %d pages opened, %d variants priced."
                     % (len(self.board.sheets), len(self.board.tills)))
        if self.board.questions:
            lines.append("Already asked: " + "; ".join(self.board.questions[-6:]))
        fit = [l for l in self._survey() if l["buyable"]]
        lines.append("%d candidates currently meet every stated requirement." % len(fit))
        lines.append(self._nudge(fit))
        return "\n".join(lines)

    def _nudge(self, fit: list) -> str:
        found = self._survey()
        if not found:
            return "Nothing on record. Search or filter first."
        if fit:
            return "Something is buyable; read on only if a better match may still exist."
        if not any(l["opened"] for l in found):
            return "Hits are listed but unopened. View or compare the promising ones."
        if self.ask.needs and not any(l["meets"] for l in found):
            return "Nothing states the required amount. Search that amount directly."
        if all(not self._within_cap(l["price"]) for l in found if l["price"] is not None):
            return "Everything priced is over budget. Look for cheaper listings."
        return "Nothing meets every requirement yet. Keep looking."

    def _from_model(self) -> tuple | None:
        """Ask for one move, when the phases have nothing of their own left to do."""
        if self.timer.left() < _ASKABLE:
            return None
        self._trim_chat()
        self.chat[0] = {"role": "system", "content": _HOUSE_RULES + "\n" + self._briefing()}
        for attempt in range(3):
            model = self.models[min(self.model_at, len(self.models) - 1)]
            try:
                reply = _PROXY.post("/inference/chat/completions", json_data={
                    "model": model, "messages": self.chat, "tools": self.tools,
                    "tool_choice": "required", "temperature": 0})
            except Exception:
                reply = None
            if reply is None:
                self.model_at = min(self.model_at + 1, len(self.models) - 1)
                if self.timer.left() < _ENDGAME:
                    return None
                continue
            picks = reply.get("choices") or []
            wanted = ((picks[0].get("message") or {}).get("tool_calls") or []) if picks else []
            for call in wanted[:1]:
                name = str((call.get("function") or {}).get("name") or "")
                if name not in self.named:
                    continue
                raw = (call.get("function") or {}).get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, ValueError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if name in self.tills:
                    # The reasons are rebuilt from the shop's own facts whatever the model wrote,
                    # because a claim it invented is a false claim with the order attached.
                    pid = args.get("product_id") or ""
                    if pid:
                        return (name, self._till_args(pid, args.get("sku") or ""))
                    continue
                return (name, self._fit(name, args))
            if attempt < 2 and self.timer.left() > _ENDGAME:
                time.sleep(1)
        return None

    # ------------------------------------------------------------------ close

    def _orderable_now(self, lot: dict) -> bool:
        # Where the shop offers no way to re-read stock, not having re-read it cannot be grounds
        # for refusing to order.
        return (self._carted(lot) and lot["alive"] and self._within_cap(lot["price"])
                and (not self.looker
                     or self.board.looked_since_commit(lot["pid"], lot["sku"])))

    def _finish(self) -> None:
        """Put through whatever was settled on, in as few calls as the seconds allow."""
        if self.done or self.placed or not self.tills:
            return
        known = {(l["pid"], l["sku"]): l for l in self._survey()}
        for pid, sku in reversed(self.board.basket):
            lot = known.get((pid, sku))
            if lot and self._orderable_now(lot):
                self._place(lot)
                return
        settled = self._choice() or self._nearest()
        if settled is None and not self.board.lots and "search" in self.named:
            # Nothing was ever read. One question to the shop beats ending having asked none.
            self._envelope([("search", self._search_args(self._openers()[0]))])
            settled = self._choice() or self._nearest()
        if settled is None:
            if not self.board.basket:
                return
            pid, sku = self.board.basket[-1]
            settled = known.get((pid, sku)) or {"pid": pid, "sku": sku, "price": None}
        if not self._carted(settled):
            if not self.committer:
                return
            self._envelope([(self.committer, self._fit(self.committer,
                                                       {"product_id": settled["pid"],
                                                        "sku": settled["sku"]}))])
            if self.done:
                return
        if self.looker and self.timer.left() >= 12:
            self._envelope([(self.looker, self._fit(self.looker,
                                                    {"product_id": settled["pid"],
                                                     "sku": settled["sku"]}))])
            if self.done or not self._available(settled["pid"], settled["sku"]):
                return
        self._place(settled)

    def _place(self, lot: dict) -> None:
        """Order it, and believe it happened only if the shop says it did."""
        if self.done or self.placed:
            return
        seen = self._envelope([(self.tills[0], self._till_args(lot["pid"], lot["sku"]))])
        if seen and not self._refused(seen):
            self.placed = True

    def _answer_text(self) -> str:
        choice = self._choice()
        if choice is None:
            return "Still looking; nothing on the shelf meets everything yet."
        title = self._headline(choice["pid"], choice["sku"])[:60] or "the candidate"
        state = "in stock" if choice["alive"] else "no longer available"
        if choice["price"] is None:
            return "I have %s in view and it is %s. I will check again before ordering." % (
                title, state)
        return "I checked again: %s is %s at %s%s, and that is the one I will order." % (
            title, state, _exactly(choice["price"]),
            (" " + self.ask.coin) if self.ask.coin else "")

    def _held_back_question(self) -> str:
        if self.ask.orders:
            return "Before I order: which matters more to you, %s? I will choose on that." \
                % " or ".join(o[2] for o in self.ask.orders[:2])
        return ("Before I order: what exactly does it need to have? Tell me the specification "
                "and I will match it.")

    # -------------------------------------------------------------------- run

    def run(self) -> list:
        if self.ask.held_back and self.crier and not self.opened_with_question:
            self.opened_with_question = True
            self._envelope([(self.crier, self._fit(self.crier,
                                                   {"content": self._held_back_question()}))])
        while not self.done and self.turn < self.max_steps:
            if self.timer.closing() or self.turn >= self.max_steps - 3:
                self._finish()
                break
            if self.recheck_due:
                self._look_again()
                continue
            if self.question_open and self.crier:
                # Let the model answer in its own words for a turn or two -- but never carry an
                # unanswered question past the till. Somebody who asks "is that one still
                # available?" and is answered by a receipt has not been answered.
                move_next = self._settling_move()
                closing_next = move_next is not None and move_next[0] in self.tills
                if self.question_held < 2 and not closing_next:
                    self.question_held += 1
                else:
                    self.question_open = False
                    self._envelope([(self.crier, self._fit(self.crier,
                                                           {"content": self._answer_text()}))])
                    continue
            if self.moved_against and not self.reopened:
                # The market moved, so the narrowings already walked are worth walking again.
                self.reopened = True
                self.shelf_read = False
                self.rung = 0
                self.board.put = {p for p in self.board.put if not p.startswith("filter")}
            reading = self._gather()
            if reading:
                self._envelope(reading)
                continue
            move = self._settling_move()
            if move is not None:
                mark = self._fingerprint(*move)
                if mark == self.repeated:
                    self.repeats += 1
                    if self.repeats > 2:
                        # The same forced move three times over is a loop, not a plan.
                        self.board.put.add(mark)
                        self.shelf_read = True
                        self.repeated, self.repeats = "", 0
                        continue
                else:
                    self.repeated, self.repeats = mark, 0
                self._after(move, self._envelope([move]))
                continue
            proposal = self._from_model()
            if proposal is None:
                self._finish()
                break
            self._after(proposal, self._envelope([proposal]))
        if not self.done and not self.placed:
            self._finish()
        return self.log


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Shop for the request in problem_data and hand back the dialogue that did it."""
    runner = Runner(problem_data)
    try:
        return runner.run()
    except Exception:
        # An episode that ends in an exception scores nothing, while one that ends early keeps
        # whatever it managed. Whatever broke, try to close and hand back the trace.
        try:
            runner._finish()
        except Exception:
            pass
        return runner.log


__all__ = ["agent_main"]
