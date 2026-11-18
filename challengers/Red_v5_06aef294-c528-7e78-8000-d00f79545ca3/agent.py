from __future__ import annotations

import collections
import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

_MAIN_MODEL = "anthropic/claude-haiku-5.5"
_proxy = ProxyClient(timeout=120, max_retries=2)

_TIME_LIMIT = 300.0
_ASK_FORM = re.compile(r"^Do you have a preference on [^?.!\n]{2,90}\?$")
_MEETS_RULE = (
    "A requirement is met only when the listing states the required value -- in its facts, or in its title when its "
    "facts have no key for that property (a fact stating another value decides) --: its whole phrase, alone, with other "
    "words around it, or as one option of a list; a number must match or satisfy the bound, with units converted. A "
    "spelling slip in the store's value (one letter added, dropped or changed in an otherwise identical word) still "
    "states it. A "
    "listing that does not state it does not meet it, unless no listing of the kind states it. A listing's kind is its "
    "category field, and whether it fits another item comes only from the store's own data, never from general "
    "product knowledge: the same value under a key both listings carry, or the other's model or series code in the "
    "listing's own works-with list; a limit one listing states (a maximum or minimum) must hold for the other.")
import re

_NUM_RE = re.compile(r"(\d+(?:\.\d+)?)(?:\s*/\s*(\d+(?:\.\d+)?))?(?:\s*([a-z]+|\"))?(?![a-z])", re.I)
_LINK = re.compile(r"^(x|to|by|and|or)$", re.I)
_JOIN_RE = re.compile(r"\s*(?:x|\xd7|-|~|to|by)\s*$", re.I)
_SLASH = r"(?<!\d)/|/(?!\d)"
_SEP = re.compile(r"\s*(?:,|;|" + _SLASH + r"|\||\band\b|&|(?<=\s)\+(?=\s))\s*", re.I)
_OPT = re.compile(r"\s*(?:,|\bor\b|" + _SLASH + r")\s*", re.I)


def _pre(s):
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(s).lower())


def _vc_norm(s):
    t = re.sub(r"[^a-z0-9.~+ ]+", " ", _pre(s))
    t = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _vc_tok(s):
    out = []
    for t in _vc_norm(s).split():
        t = t.strip('.')
        if len(t) > 3 and t.endswith('s') and not t.endswith('ss'):
            t = t[:-1]
        if t:
            out.append(t)
    return out


def _has_phrase(req, val):
    r, v = _vc_tok(req), _vc_tok(val)
    if not r:
        return False
    return any(v[i:i + len(r)] == r for i in range(len(v) - len(r) + 1))


def _close(a, b):
    if a == b or abs(len(a) - len(b)) > 1 or min(len(a), len(b)) < 5:
        return a == b
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def _near_phrase(req, val):
    r, v = _vc_tok(req), _vc_tok(val)
    return bool(r) and any(len(v[i:i + len(r)]) == len(r) and all(_close(x, y) for x, y in zip(r, v[i:i + len(r)]))
                           for i in range(len(v) - len(r) + 1))


def _nums(s):
    t = _pre(s)
    ms = list(_NUM_RE.finditer(t))
    units = [('' if _LINK.match(m.group(3) or '') else (m.group(3) or '').lower()) for m in ms]
    for i in range(len(ms) - 2, -1, -1):
        if not units[i] and units[i + 1] and _JOIN_RE.fullmatch(t[ms[i].end(2) if ms[i].group(2) else ms[i].end(1):ms[i + 1].start()]):
            units[i] = units[i + 1]
    out = []
    for m, u in zip(ms, units):
        d = float(m.group(2)) if m.group(2) else 1.0
        n = float(m.group(1)) / (d or 1.0)
        out.append((n, u or None, n, 1.0))
    return out


def _same_unit(a, b):
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    return a == b or a + 's' == b or b + 's' == a or a + 'es' == b or b + 'es' == a


def _joined_span(req, val):
    rj, v = ''.join(_vc_tok(req)), _vc_tok(val)
    return len(rj) >= 4 and any(''.join(v[i:j]) == rj for i in range(len(v)) for j in range(i + 1, min(len(v), i + 4) + 1))


def _unit_only(req):
    ns = _nums(req)
    return len(ns) == 1 and all(re.fullmatch(r"[\d.]+", t) or _same_unit(t, ns[0][1]) for t in _vc_norm(req).split())


def _num_equal(rq, vn):
    same = [x for x in vn if rq[1] and x[1] and _same_unit(x[1], rq[1])]
    if any(abs(x[0] - rq[0]) < 1e-9 for x in same):
        return 'yes'
    if same:
        return 'no'
    if any(abs(x[0] - rq[0]) < 1e-9 for x in vn) or (rq[1] and any(x[1] for x in vn)):
        return 'unknown'
    return "no"


def _between(req, rnums, bounds):
    t = _pre(req)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:[a-z\"]+\s*)?(?:to|-|~|and)\s*(\d+(?:\.\d+)?)", t)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        ia = next((i for i, x in enumerate(rnums) if abs(x[0] - a) < 1e-9), None)
        ib = next((i for i, x in enumerate(rnums) if abs(x[0] - b) < 1e-9 and i != ia), None)
        if ia is not None and ib is not None:
            return min(bounds[ia], bounds[ib]), max(bounds[ia], bounds[ib])
    w = re.search(r"within\s+(\d+(?:\.\d+)?)\s*\S*?\s*of\s+(\d+(?:\.\d+)?)", t)
    if w and len(bounds) >= 2:
        d, c = bounds[0], bounds[1]
        return c - d, c + d
    return min(bounds), max(bounds)


def _bound(req, op, rnums, vn):
    bounds = [x[0] for x in rnums]
    vals = [x[0] for x in vn]
    if op == 'between':
        lo, hi = _between(req, rnums, bounds)
        return [lo - 1e-9 <= v <= hi + 1e-9 for v in vals]
    if op == 'ge':
        return [v >= bounds[0] - 1e-9 for v in vals]
    return [v <= bounds[0] + 1e-9 for v in vals]


_OP = [('between', re.compile(r"\bbetween\b|\bwithin\b|\bfrom\b.*\bto\b", re.I)),
       ('ge', re.compile(r"at least|or more|or above|and up|minimum|\bmin\b|no less than|not less than|>=", re.I)),
       ('le', re.compile(r"at most|or less|or lower|or below|up to|maximum|\bmax\b|no more than|not more than|under|below|<=", re.I))]


def _op(req):
    for name, rx in _OP:
        if rx.search(req):
            return name
    return 'eq'


def _is_range_label(s):
    return bool(re.search(r"\d\s*(?:[a-z]+\s*)?(?:~|-|to)\s*\d", s.lower())) or bool(re.search(r"\d+\s*(?:and up|\+)", s.lower()))


def _vc_check(required, value):
    if value is None or str(value).strip() == '':
        return 'unknown'
    req = str(required).strip()
    op = _op(req)
    rnums = _nums(req)
    if rnums and op != 'eq':
        rdim = next((x[1] for x in rnums if x[1]), None)
        vn = [x for x in _nums(value) if rdim is None or x[1] is None or _same_unit(x[1], rdim)]
        if not vn:
            return 'unknown'
        return 'yes' if all(_bound(req, op, rnums, vn)) else 'no'
    options = [o for o in _OPT.split(req) if o.strip()]
    if len(options) > 1:
        res = [_vc_check(o, value) for o in options]
        return 'yes' if 'yes' in res else ('no' if all(r == 'no' for r in res) else 'unknown')
    parts = [p for p in _SEP.split(str(value)) if p.strip()] + [str(value)]
    if _is_range_label(req):
        rn = _vc_norm(req).replace(' ', '')
        return 'yes' if any(_vc_norm(p).replace(' ', '') == rn for p in parts) else 'no'
    if any(_has_phrase(req, p) for p in parts):
        return 'yes'
    if len(rnums) == 1 and _unit_only(req):
        return _num_equal(rnums[0], _nums(value))
    if not rnums and (any(_near_phrase(req, p) for p in parts)
                      or any(_has_phrase(p, req) for p in parts if _vc_tok(p))
                      or any(_joined_span(req, p) or _joined_span(p, req) for p in parts if _vc_tok(p))):
        return 'unknown'
    return 'no'


def _vc_check_text(required, text):
    req = str(required).strip()
    op, rnums = _op(req), _nums(req)
    if rnums and op != 'eq':
        rdim = next((x[1] for x in rnums if x[1]), None)
        if rdim is None:
            return 'unknown'
        tn = _nums(text)
        vn = [x for x in tn if _same_unit(x[1], rdim)]
        if not vn:
            return 'unknown' if any(x[1] for x in tn) else 'no'
        return 'yes' if any(_bound(req, op, rnums, [x])[0] for x in vn) else 'no'
    return _vc_check(required, text)


def _vc_check_fields(required, values):
    res = [_vc_check(required, v) for v in values]
    if 'yes' in res:
        return 'yes'
    if res and all(r == 'no' for r in res):
        return 'no'
    return 'unknown'


def _model() -> str:
    return _MAIN_MODEL


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


_ID_KEY = re.compile(r"^(id|ids|ref|.*_id|.*id_code|.*idcode|.*_ref|product_id.*|item_id.*|.*id_number)$", re.I)


def _cart_failed(obs: Any) -> bool:
    if isinstance(obs, str):
        return "error" in obs.lower()
    if isinstance(obs, dict):
        return bool(obs.get("error")) or any(v is False for v in obs.values() if isinstance(v, bool))
    return False


_UNITS_KEY = re.compile(r"(qty|quantity|units|count|pieces)", re.I)


def _row_units(obs: Any) -> dict[str, int]:
    out: dict[str, int] = {}

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for r in x:
                if isinstance(r, dict):
                    rid = next((v for k, v in r.items() if isinstance(v, str) and _ID_KEY.match(k)), None)
                    n = next((v for k, v in r.items() if isinstance(v, int) and not isinstance(v, bool)
                              and _UNITS_KEY.search(k) and not _PRICE_KEY.search(k)), None)
                    if rid and n is not None:
                        out[rid] = n
                walk(r)

    walk(obs)
    return out


def _removing(args: dict[str, Any], removes: dict[str, Any] | None) -> bool:
    return bool(removes) and all(str(args.get(k, "")).strip().lower() == str(v).strip().lower() for k, v in removes.items())


def _track_basket(held: dict[str, int], seen: dict[str, Any], args: dict[str, Any], obs: Any,
                  removes: dict[str, Any] | None = None) -> None:
    args = {k: v for k, v in args.items() if k != "notes"}
    inner = obs
    if isinstance(obs, dict) and len(obs) == 2 and _notes(obs):
        inner = next((v for v in obs.values() if not isinstance(v, list)), obs)
    rows = _row_ids(inner)
    if rows or (isinstance(inner, dict) and any(isinstance(v, list) and not v for v in inner.values())):
        units = _row_units(inner)
        kept = {i: units.get(i, held.get(i, 1)) for i in rows}
        held.clear()
        held.update(kept)
        return
    if _cart_failed(inner):
        return
    ids = [v for v in args.values() if isinstance(v, str) and v in seen]
    if len(ids) != 1:
        return
    pid = ids[0]
    words = [v for v in args.values() if isinstance(v, str) and v != pid]
    nums = [int(v) for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    n = nums[0] if nums else 1
    if _removing(args, removes):
        left = held.get(pid, 0) - n
    elif not words and nums:
        left = n
    else:
        left = held.get(pid, 0) + n
    if left > 0:
        held[pid] = left
    else:
        held.pop(pid, None)


def _cart_op(args: dict[str, Any], seen: dict[str, Any], removes: dict[str, Any] | None = None) -> tuple[str, str, int] | None:
    args = {k: v for k, v in args.items() if k != "notes"}
    ids = [v for v in args.values() if isinstance(v, str) and v in seen]
    if len(ids) != 1:
        return None
    pid = ids[0]
    words = [v for v in args.values() if isinstance(v, str) and v != pid]
    nums = [int(v) for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    n = nums[0] if nums else 1
    if _removing(args, removes):
        return pid, "remove", n
    if not words and nums:
        return pid, ("remove" if n <= 0 else "set"), n
    return pid, "add", n


_CAT_FIELD = re.compile(r"^(category|department|shelf|group|group_path|aisle|section)$", re.I)


def _page_rows(obs: Any) -> list[tuple[str, float, dict[str, Any]]]:
    out: list[tuple[str, float, dict[str, Any]]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if x and all(isinstance(r, dict) for r in x):
                for r in x:
                    rid = next((v for k, v in r.items() if isinstance(v, str) and _ID_KEY.match(k)), None)
                    if rid and not any(_STOCK_KEY.search(k) and v is False for k, v in r.items()):
                        out.append((rid, _row_price(r), r))
            else:
                for v in x:
                    walk(v)

    walk(obs)
    return out


def _cheaper_rows(run: Any, tool: dict[str, Any], pid: str) -> list[tuple[float, str, dict[str, Any]]]:
    d = run.seen.get(pid) or {}
    price = _row_price(d)
    shelf = next((v for k, v in d.items() if isinstance(v, str) and _CAT_FIELD.match(k)), None)
    if price == float("inf") or not shelf or price <= 0:
        return []
    found: dict[str, tuple[float, dict[str, Any]]] = {}
    cap = price - (1 if tool["int"] else 0.01)
    for _ in range(3):
        if run.left() < 8 or cap <= 0:
            break
        args: dict[str, Any] = {tool["shelf"]: shelf, tool["cap"]: int(cap) if tool["int"] else round(cap, 2)}
        args.update(tool.get("extra") or {})
        try:
            obs = run.env(tool["name"], args)
        except Exception:
            break
        page = [(i, p, r) for i, p, r in _page_rows(obs) if p < price and i != pid]
        if not page:
            break
        for i, p, r in page:
            found[i] = (p, r)
        cap = min(p for _, p, _ in page) - (1 if tool["int"] else 0.01)
    return sorted(((p, i, r) for i, (p, r) in found.items() if i not in run.opened), key=lambda x: x[0])[:6]


def _row_ids(obs: Any) -> list[str]:
    out: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            if x and all(isinstance(r, dict) for r in x):
                for r in x:
                    for k, v in r.items():
                        if isinstance(v, str) and _ID_KEY.match(k) and v not in out:
                            out.append(v)
                            break
            for v in x:
                walk(v)
    walk(obs)
    return out


_NEXT_KEY = re.compile(r"(next|follow|continue|cursor|page_token)", re.I)


def _small_page(obs: Any) -> tuple[str, Any, int] | None:
    if not isinstance(obs, dict):
        return None
    for d in [obs] + [v for v in obs.values() if isinstance(v, dict)]:
        lists = [v for v in d.values() if isinstance(v, list) and v and all(isinstance(r, dict) for r in v)]
        keys = [(k, v) for k, v in d.items() if _NEXT_KEY.search(k) and v not in (None, False, "", 0)]
        if lists and keys and len(lists[0]) <= 12:
            return keys[0][0], keys[0][1], len(lists[0])
    return None


_FINAL_RE = re.compile(r"FINAL VALUE:\s*\**\s*([^\n]+)", re.I)


def _derived_value(rep: str) -> str | None:
    m = _FINAL_RE.findall(rep or "")
    if not m:
        return None
    v = m[-1].strip().strip("*").strip()
    if not v or "not found" in v.lower():
        return None
    v = v.split(":")[-1].strip().strip("*'\". ")
    return v if 0 < len(v) <= 40 else None


def _notes(obs: Any) -> list[str]:
    if not isinstance(obs, dict) or len(obs) != 2:
        return []
    for v in obs.values():
        if isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            return [x for x in v if x.strip()]
    return []


_DROP_TASK = ("A shopper sent this note while their order was being prepared: \"{note}\". Does the note let go of (drop) "
              "one of their requirements, so that it no longer applies? Reply with JSON only: {{\"drops\": true or false, "
              "\"property\": \"<the property dropped, or null>\"}}.")


def _dropped(note: str) -> str | None:
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You read shopper notes. Reply with one JSON object."},
                                        {"role": "user", "content": _DROP_TASK.format(note=note)}])
    except Exception:
        return None
    if isinstance(data, dict) and data.get("drops") is True:
        return str(data.get("property") or "a requirement").strip() or "a requirement"
    return None


def _multi_fail(args: dict[str, Any], obs: Any) -> int:
    n = max((len(v) for v in args.values() if isinstance(v, list) and all(isinstance(x, str) for x in v)), default=0)
    if n < 3 or _row_ids(obs) or len(json.dumps(obs, default=str)) > 400:
        return 0
    return n


_MULTI_NOTE = ("That call with {n} ids returned no listing: this store's tool may take fewer ids at once. Call it again "
               "with at most 2 ids per call (or read them one by one); do not repeat the {n}-id call.")


def _over_cap(run: Any, name: str, args: dict[str, Any]) -> bool:
    cap = getattr(run, "id_cap", {}).get(name)
    if not cap:
        return False
    return any(isinstance(v, list) and len(v) > cap for v in args.values())


_CAP_NOTE = "Not run (no step used): this tool takes at most 2 ids per call here; call it with 2 ids at a time."


def _seen_listings(obs: Any, out: dict[str, dict[str, Any]]) -> None:
    if isinstance(obs, dict):
        idv = next((v for k, v in obs.items() if isinstance(v, str) and _ID_KEY.match(k)), None)
        if idv:
            d = out.setdefault(idv, {})
            for k, v in obs.items():
                if isinstance(v, (str, int, float, bool)) or v is None:
                    d[k] = v[:600] if isinstance(v, str) else v
                elif isinstance(v, dict) and v and all(isinstance(x, (str, int, float, bool)) or x is None
                                                       for x in v.values()):
                    d[k] = {kk: vv for kk, vv in list(v.items())[:40]}
        for v in obs.values():
            if isinstance(v, (dict, list)):
                _seen_listings(v, out)
    elif isinstance(obs, list):
        for v in obs:
            _seen_listings(v, out)


_DECLINE_REVIEW_TASK = (
    "A shopping assistant wants to decline the request below because it believes no listing fits. Check the listings it "
    "has seen. Judge every stated requirement on the listing's own details (fact tables first; "
    + _MEETS_RULE + " A key with a related name holding the same value "
    "counts; a missing key is not a contradiction when the title or page states the value). For each item, find the "
    "cheapest in-stock listing of the item's kind that meets every stated, looked-up and revealed requirement within the "
    "budget. Reply with JSON only: {{\"buy\": [{{\"id\": \"<listing id>\", \"why\": \"<short reason>\"}}, ...]}} "
    "with one listing per item when every item has one, or {{\"decline\": true, \"why\": \"<the item that has "
    "none>\"}}.\n\nREQUEST AND FINDINGS:\n{context}\n\nLISTINGS SEEN (id | field: value | ... | facts: key: value; ...):\n{listings}")


def _review_row(i: str, d: dict[str, Any]) -> str:
    parts = [i]
    for k, v in d.items():
        if isinstance(v, dict) or v is None or (isinstance(v, str) and (not v.strip() or v == i)):
            continue
        sv = str(v)
        parts.append(f"{k}: {sv[:300] + '...' if len(sv) > 300 else sv}")
    facts: dict[tuple[str, str], str] = {}
    for v in d.values():
        if isinstance(v, dict):
            for kk, vv in v.items():
                if vv is None or (isinstance(vv, str) and not vv.strip()):
                    continue
                key = " ".join(_vc_tok(str(kk).replace("_", " ").replace("-", " "))) or str(kk)
                facts.setdefault((key, str(vv).strip().lower()), f"{kk}: {vv}")
    line = " | ".join(parts)
    if facts:
        line += " | facts: " + "; ".join(facts.values())
    return line[:3000]


def _unread_by_item(run: Any, names: list[str], only: set[int] | None = None, per_item: int = 4) -> str:
    groups: dict[int, list[tuple[str, dict[str, Any]]]] = {}
    for i, r in run.seen.items():
        if i in run.opened or len(r) <= 2:
            continue
        idx = _line_item(_flat_facts(r), names) if names else None
        if idx is None or (only is not None and idx not in only):
            continue
        groups.setdefault(idx, []).append((i, r))
    parts = []
    for idx in sorted(groups):
        rows = sorted(groups[idx], key=lambda x: _row_price(x[1]))[:per_item]
        parts.append(f"for the {names[idx]}: " + "; ".join(_review_row(i, r) for i, r in rows))
    return " || ".join(parts)


def _review_rows(listings: dict[str, dict[str, Any]], names: list[str], budget: int = 60000) -> list[str]:
    groups: dict[Any, list[tuple[str, dict[str, Any]]]] = {}
    for i, d in listings.items():
        if len(d) <= 2:
            continue
        idx = _line_item(_flat_facts(d), names) if names else None
        groups.setdefault(idx, []).append((i, d))
    queues = []
    for lst in groups.values():
        lst.sort(key=lambda x: (not any(isinstance(v, dict) for v in x[1].values()), _row_price(x[1])))
        queues.append(lst)
    rows: list[str] = []
    used = 0
    while used < budget and any(queues):
        for q in queues:
            if q and used < budget:
                i, d = q.pop(0)
                r = _review_row(i, d)
                rows.append(r)
                used += len(r)
    return rows


def _decline_review(context: str, listings: dict[str, dict[str, Any]],
                    names: list[str] | None = None) -> list[tuple[str, str]] | None:
    rows = _review_rows(listings, names or [])
    if not rows:
        return None
    try:
        data, _ = _chat_json(_model(), [
            {"role": "system", "content": "You check shopping decisions. Reply with one JSON object."},
            {"role": "user", "content": _DECLINE_REVIEW_TASK.format(context=context[-9000:], listings="\n".join(rows))}],
            max_tokens=12000)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("decline") or not data.get("buy"):
        return None
    out = [(str(x.get("id")), str(x.get("why") or "")) for x in data.get("buy") or []
           if isinstance(x, dict) and str(x.get("id")) in listings]
    return out or None


_CLEAN_TASK = (
    "Below are a shopping request, the reports of looking up listings it (or a shopper reply) refers to, and a draft "
    "list of the requirements of the items to buy. Correct the draft: (1) remove every value that only identifies a "
    "listing along a reference chain -- the seller or brand, and the properties named to pick out a listing ('<seller>'s "
    "<kind> with <property> <value>'), for the first listing and for every listing the chain passes through; (2) for a "
    "requirement that comes from a chain, keep only the chain's final value, and when the last page says which value "
    "goes together with its own, use that other value; (3) keep every requirement the request states for the item to "
    "buy and every value the shopper's replies require. Reply with JSON only: {{\"requirements\": \"item: property "
    "value; item: property value\"}}.\n\n{context}\n\nDRAFT REQUIREMENTS: {draft}")


def _clean_requirements(context: str, draft: str) -> str:
    try:
        data, _ = _chat_json(_model(), [
            {"role": "system", "content": "You check shopping requirement lists. Reply with one JSON object."},
            {"role": "user", "content": _CLEAN_TASK.format(context=context[-12000:], draft=draft)}])
    except Exception:
        return draft
    if isinstance(data, dict) and isinstance(data.get("requirements"), str) and data["requirements"].strip():
        return data["requirements"].strip()
    return draft


def _arg_ids(args: dict[str, Any]) -> set[str]:
    vals: set[str] = set()
    for v in args.values():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, str):
                vals.add(x)
    return vals


def _clip(value: Any, limit: int = 600) -> Any:
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, list):
        return [_clip(v, limit) for v in value]
    if isinstance(value, dict):
        return {k: _clip(v, limit) for k, v in value.items()}
    return value


def _compact(text: str, limit: int = 1500) -> str:
    try:
        obj = json.loads(text)
    except Exception:
        return text[:300] + "..."

    def walk(x: Any) -> Any:
        if isinstance(x, str):
            return x if len(x) <= 60 else "..."
        if isinstance(x, list):
            return [walk(v) for v in x]
        if isinstance(x, dict):
            return {k: walk(v) for k, v in x.items()}
        return x
    out = json.dumps(walk(obj), ensure_ascii=False)
    return out if len(out) <= limit else out[:limit] + "..."


def _json_of(text: str) -> Any:
    try:
        body = text[text.index("{"): text.rindex("}") + 1]
    except ValueError:
        return None
    try:
        return json.loads(body)
    except Exception:
        pass
    try:
        return json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', body))
    except Exception:
        return None


def _chat_json(model: str, messages: list[dict[str, Any]], max_tokens: int = 4000) -> tuple[Any, str]:
    text = ""
    for _ in range(2):
        msg = _chat(model, messages, max_tokens=max_tokens)
        text = msg.get("content") or ""
        data = _json_of(text)
        if data is not None:
            return data, text
    return None, text


def _with_cache(messages: list[dict[str, Any]], idxs: list[int | None]) -> list[dict[str, Any]]:
    out = list(messages)
    marked: set[int] = set()
    for i in sorted({i for i in idxs if isinstance(i, int) and 0 <= i < len(messages)}):
        j = i
        while j >= 0 and j not in marked and not (isinstance(out[j].get("content"), str) and out[j]["content"]):
            j -= 1
        if j < 0 or j in marked or len(marked) >= 4:
            continue
        out[j] = {**out[j], "content": [{"type": "text", "text": out[j]["content"], "cache_control": {"type": "ephemeral"}}]}
        marked.add(j)
    return out


def _chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
          tool_choice: str = "auto", max_tokens: int | None = None, cache_at: list[int | None] | None = None) -> dict[str, Any]:
    if cache_at:
        messages = _with_cache(messages, cache_at)
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": 8000}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if max_tokens:
        body["max_tokens"] = max_tokens
    resp = _proxy.post("/inference/chat/completions", json_data=body)
    if resp is None:
        raise RuntimeError("inference request failed")
    choice = resp["choices"][0]
    msg = choice.get("message") or {}
    if (choice.get("finish_reason") == "content_filter" and not (msg.get("content") or msg.get("tool_calls"))):
        body["temperature"] = 0.2
        again = _proxy.post("/inference/chat/completions", json_data=body)
        if again is not None:
            return again["choices"][0].get("message") or {}
    return msg


_MODULES: list[dict[str, Any]] = [
    {"name": "spread_items", "flags": {"multi_item"}, "stage": "planner",
     "text": "Several items: every item needs its own questions. A question without an item reaches only the items "
             "whose listings have that property, so before a second property of any item, ask each OTHER item about "
             "a property its own listings have (name that item); the item with the fewest stated requirements comes "
             "first."},
    {"name": "facts_view_keys", "flags": {"page_and_facts"}, "stage": "browse",
     "text": "This store has separate page and facts views: read the FACTS view of candidates for their properties and "
             "KEYS; only the keys of the facts table are datasheet keys."},
    {"name": "value_only_search", "flags": set(), "stage": "buy",
     "text": "Search also with the requirement values alone (one or two value words, without the item's kind word) "
             "with the store's text search, and read the next pages of those results too: a listing whose title does "
             "not say the value is often found only this way."},
    {"name": "value_only_first_query", "flags": {"search_modes"}, "stage": "summary",
     "text": "Make the FIRST query just the two most distinctive requirement values (no item kind word), the "
             "shopper's revealed value first, e.g. 'X Y'."},
    {"name": "category_not_title", "flags": set(), "stage": "buy",
     "text": "A row's kind is its category field, never its title words: read every cheaper row of the right category "
             "even when its title names something else."},
    {"name": "lookup_start_page", "flags": {"lookup"}, "stage": "browse",
     "text": "Lookups: the referenced listing is the seller's row of that kind whose details (facts or page notes) show "
             "the identifying property values named for it; the same value under a related key counts (e.g. 'blade "
             "count: 3' for 'number of blades 3'). Read the seller's rows of that kind until one shows them all; if none does, "
             "take the one that shows the most. Follow only that page's pointer."},
    {"name": "lookup_other_kind", "flags": {"lookup_other"}, "stage": "planner",
     "text": "A looked-up listing of another kind than the item to buy only supplies a value: the properties that "
             "identify it are not properties of the item to buy, so never ask them for that item."},
    {"name": "lookup_identifiers", "flags": {"lookup_same"}, "stage": "planner",
     "text": "Lookups: never ask the property a lookup reads (its value comes from the pages). The properties that "
             "identify a looked-up listing in the checklist (its seller aside) are not requirements of the item to "
             "buy, which makes them likely hidden requirements of it: ask them early, one per message, at most two of "
             "them per item and none once that item has revealed a requirement; then go on to the other items."},
]


def _situation(plan: dict[str, Any] | None, kinds: dict[str, Any], tools_all: list[dict[str, Any]]) -> set[str]:
    flags: set[str] = set()
    items = [i for i in (plan or {}).get("items") or [] if isinstance(i, dict)]
    if len(items) > 1:
        flags.add("multi_item")
    if any(i.get("lookups") for i in items):
        flags.add("lookup")
    for i in items:
        kind = _cat_words(str(i.get("name") or ""))
        for lk in i.get("lookups") or []:
            if isinstance(lk, dict):
                same = bool(kind) and kind <= _cat_words(str(lk.get("listing") or ""))
                flags.add("lookup_same" if same else "lookup_other")
    if (plan or {}).get("confirm"):
        flags.add("confirm")
    browse = {(t.get("function") or {}).get("name") for t in kinds.get("browse") or []}
    roles = kinds.get("roles") or {}
    if kinds.get("sweep"):
        flags.add("price_filter")
    for t in tools_all:
        fn = t.get("function") or {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        if fn.get("name") not in browse:
            continue
        if any((p or {}).get("type") == "array" for p in props.values()):
            flags.add("compare_tool")
        if roles.get(fn.get("name")) == "search" and any(len((p or {}).get("enum") or []) == 2 for p in props.values()):
            flags.add("search_modes")
    if len(kinds.get("open") or []) >= 2:
        flags.add("page_and_facts")
    if not kinds.get("ask"):
        flags.add("no_messages")
    return flags


def _extra(flags: set[str], stage: str) -> str:
    return "\n".join(m["text"] for m in _MODULES if m["stage"] == stage and set(m["flags"]) <= flags)


_ROLE_TASK = (
    "Below are the tools of an online store's shopping session: name, description and arguments. Give each tool "
    "exactly one role: 'ask' (sends a message to the shopper), 'order' (places the order for the basket and ends the "
    "session), 'decline' (turns the request down and ends the session), 'finish' (ends the session, ordering or "
    "declining as an argument chooses), 'basket_change' (adds, removes or sets the units of a basket line), "
    "'basket_view' (shows what the basket holds), 'search' (finds listings), 'open' (shows one listing given its id: "
    "its page, details or facts), 'compare' (shows several listings side by side), 'other' (anything else). Also "
    "give: for each "
    "'finish' tool, the argument values that order and those that decline; for each 'basket_change' tool, the "
    "argument values that take units out of a line (an empty object when a unit count of 0 does that); for a "
    "'search' tool that can list one category under a price cap, the argument that caps the price, the argument "
    "that names the category and the other argument values that kind of listing needs (null when no tool does). "
    "Reply with JSON only: {{\"roles\": {{\"<tool>\": \"<role>\"}}, \"finish_args\": {{\"<tool>\": {{\"order\": "
    "{{\"<argument>\": \"<value>\"}}, \"decline\": {{\"<argument>\": \"<value>\"}}}}}}, \"remove_args\": {{\"<tool>\": "
    "{{\"<argument>\": \"<value>\"}}}}, \"price_listing\": {{\"tool\": \"<tool>\", \"price_cap\": \"<argument>\", "
    "\"category\": \"<argument>\", \"with\": {{\"<argument>\": \"<value>\"}}}} or null}}.\n\nTOOLS:\n{tools}")
_ROLES = ("ask", "order", "decline", "finish", "basket_change", "basket_view", "search", "open", "compare", "other")


def _tool_roles(tools: list[dict[str, Any]]) -> dict[str, Any]:
    lines = []
    for t in tools:
        fn = t.get("function") or {}
        props = (fn.get("parameters") or {}).get("properties") or {}
        args = {k: ((v or {}).get("enum") or (v or {}).get("type")) for k, v in props.items() if k != "notes"}
        lines.append(f"- {fn.get('name')}: {fn.get('description') or ''} | arguments: {json.dumps(args)}")
    try:
        data, _ = _chat_json(_model(), [
            {"role": "system", "content": "You read tool lists. Reply with one JSON object."},
            {"role": "user", "content": _ROLE_TASK.format(tools="\n".join(lines))}], max_tokens=2000)
    except Exception:
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("roles"), dict):
        return {}
    names = {(t.get("function") or {}).get("name") for t in tools}
    roles = {k: v for k, v in data["roles"].items() if k in names and v in _ROLES}
    return {"roles": roles,
            "finish": {k: v for k, v in (data.get("finish_args") or {}).items() if isinstance(v, dict) and roles.get(k) == "finish"},
            "remove": {k: v for k, v in (data.get("remove_args") or {}).items() if isinstance(v, dict) and roles.get(k) == "basket_change"},
            "price": data.get("price_listing") if isinstance(data.get("price_listing"), dict) else None}


def _schema_values(fn: dict[str, Any], given: Any) -> dict[str, Any] | None:
    props = (fn.get("parameters") or {}).get("properties") or {}
    if not isinstance(given, dict):
        return None
    for k, v in given.items():
        spec = props.get(k)
        if spec is None or ((spec or {}).get("enum") and v not in spec["enum"]):
            return None
    return dict(given)


def _price_listing(fns: dict[str, dict[str, Any]], given: Any, browse: set[str]) -> dict[str, Any] | None:
    if not isinstance(given, dict) or given.get("tool") not in browse:
        return None
    fn = fns[given["tool"]]
    props = (fn.get("parameters") or {}).get("properties") or {}
    cap, shelf = given.get("price_cap"), given.get("category")
    if (props.get(cap) or {}).get("type") not in ("integer", "number") or cap == shelf:
        return None
    if shelf not in props or (props.get(shelf) or {}).get("type", "string") != "string":
        return None
    extra = _schema_values(fn, given.get("with") or {})
    if extra is None or cap in extra or shelf in extra:
        return None
    return {"name": given["tool"], "cap": cap, "int": props[cap].get("type") == "integer", "shelf": shelf, "extra": extra}


def _finish_values(fn: dict[str, Any], given: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    props = (fn.get("parameters") or {}).get("properties") or {}
    out: dict[str, dict[str, Any]] = {}
    for side in ("order", "decline"):
        vals = given.get(side)
        if not isinstance(vals, dict) or not vals:
            return None
        for k, v in vals.items():
            enum = (props.get(k) or {}).get("enum")
            if not enum or v not in enum:
                return None
        out[side] = dict(vals)
    return out if out["order"] != out["decline"] else None


def _tool_kinds(tools: list[dict[str, Any]]) -> dict[str, Any]:
    read = _tool_roles(tools)
    roles, fargs, rargs = read.get("roles") or {}, read.get("finish") or {}, read.get("remove") or {}
    fns = {(t.get("function") or {}).get("name"): (t.get("function") or {}) for t in tools}
    ask_name, ask_arg = "", ""
    for name, fn in fns.items():
        props = {k: v for k, v in ((fn.get("parameters") or {}).get("properties") or {}).items() if k != "notes"}
        if roles.get(name) == "ask" and len(props) == 1:
            (arg, spec), = props.items()
            if (spec or {}).get("type", "string") == "string":
                ask_name, ask_arg = name, arg
                break
    close, cart, view, order, decline, compare = set(), set(), set(), set(), set(), set()
    opens: list[str] = []
    finish_vals: dict[str, dict[str, dict[str, Any]]] = {}
    removes: dict[str, dict[str, Any]] = {}
    for name, fn in fns.items():
        props = {k: v for k, v in ((fn.get("parameters") or {}).get("properties") or {}).items() if k != "notes"}
        if name == ask_name:
            continue
        role = roles.get(name)
        if role == "order":
            close.add(name)
            order.add(name)
        elif role == "decline":
            close.add(name)
            decline.add(name)
        elif role == "finish":
            close.add(name)
            if (fv := _finish_values(fn, fargs.get(name) or {})):
                finish_vals[name] = fv
        elif role == "basket_change" and props:
            cart.add(name)
            removes[name] = _schema_values(fn, rargs.get(name) or {}) or {}
        elif role in ("basket_change", "basket_view"):
            view.add(name)
        elif role == "open" and len(props) == 1 and (next(iter(props.values())) or {}).get("type", "string") == "string":
            opens.append(name)
        elif role == "compare":
            compare.add(name)
    browse = [t for t in tools if (t.get("function") or {}).get("name") not in close | cart | {ask_name}]
    browse_names = {(t.get("function") or {}).get("name") for t in browse}
    return {"ask": ask_name, "ask_arg": ask_arg, "close": close, "cart": cart, "view": view, "browse": browse,
            "order": order, "decline": decline, "finish_vals": finish_vals, "tools": tools, "open": opens,
            "compare": compare, "removes": removes, "roles": roles,
            "sweep": _price_listing(fns, read.get("price"), browse_names - view)}


class _Run:
    def __init__(self, problem_data: dict[str, Any]):
        environment = problem_data["environment"]
        self.binding = environment["binding"]
        self.policy = environment["policy_view"]
        self.max_steps = int(self.policy["max_steps"])
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dialogue: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.id_cap: dict[str, int] = {}
        self.seen: dict[str, dict[str, Any]] = {}
        self.views: dict[str, set[str]] = {}
        self.opened: set[str] = set()
        self.cart_names: set[str] = set()
        self.held: dict[str, int] = {}
        self.read_names: set[str] = set()
        self.removes: dict[str, dict[str, Any]] = {}
        self.state_names: set[str] = set()
        self.reads: dict[str, Any] = {}

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def env(self, name: str, args: dict[str, Any], tool_call: dict[str, Any] | None = None, content: str = "") -> Any:
        key = name + json.dumps({k: v for k, v in args.items() if k != "notes"}, sort_keys=True)
        if name in self.read_names and key in self.reads:
            return self.reads[key]
        self.step += 1
        result = _proxy.post("/environment/call", json_data={**self.binding, "step": self.step,
                                                              "action": {"name": name, "args": args}})
        if result is None:
            raise RuntimeError("environment call failed")
        self.finished = bool(result["finished"])
        call = tool_call or {"id": f"code-{self.step}", "type": "function",
                             "function": {"name": name, "arguments": json.dumps(args)}}
        self.dialogue.append({"role": "assistant", "content": content, "tool_call": call, "environment_result": result})
        if name in self.state_names or _notes(result["observation"]):
            self.reads.clear()
        elif name in self.read_names:
            self.reads[key] = result["observation"]
        try:
            _seen_listings(result["observation"], self.seen)
            if name in self.cart_names:
                _track_basket(self.held, self.seen, args, result["observation"], self.removes.get(name))
            self.opened |= _arg_ids(args)
            for v in args.values():
                if isinstance(v, str):
                    self.views.setdefault(v, set()).add(name)
        except Exception:
            pass
        return result["observation"]


_BRIEF_TASK = (
    "Turn the shopper's request below into a checklist for a shopping assistant, as JSON only:\n"
    "{\"items\": [{\"name\": \"<what to buy, in the request's words, without the quantity>\", \"quantity\": <n>,\n"
    "  \"must\": [\"<property>: <value>\", ...],\n"
    "  \"lookups\": [{\"listing\": \"<the other listing: seller or brand, kind of item, identifying properties>\",\n"
    "               \"read\": \"<property to read there>\", \"relation\": \"<how the item to buy must relate to it>\"}],\n"
    "  \"hints\": [\"<property of this item the story may hint at>\"]}],\n"
    " \"budget\": \"<amount, and whether per item or for everything>\", \"exclude\": \"<e.g. sold out>\",\n"
    " \"messages\": <message limit or null>, \"steps\": <step limit or null>, \"other\": \"<any other rule>\",\n"
    " \"if_none_fits\": \"<what the request says to do when no listing meets every requirement, in its words, or null>\",\n"
    " \"confirm\": <true only when the request explicitly asks to be told, listed or confirmed what will arrive, the basket "
    "or the total, or asks a direct question about them (a rule or limit written as a question, such as a budget, a "
    "message or a step limit, is a rule, not a question to answer); wishes about planning, "
    "storage, timing or use are hints, not such a request: then false>}\n"
    "Rules: item names in a request are usually the store's category names and can contain '&' or commas (e.g. "
    "'pots & pans', 'cups, mugs & saucers'): keep such a name whole as ONE item, and a quantity before "
    "it ('2 pairs of X') belongs to that whole item. 'must' holds only requirements the listing to buy must itself meet, with the request's exact values; a rule "
    "that the item must fit or be compatible with something the request names directly (e.g. 'it must work with the "
    "device I own, model X') is a 'must' entry ('compatible with: X'), not a lookup, and a conditional rule ('X unless Y') is a "
    "'must' entry with its condition. A "
    "lookup is a value that has to be read from another listing; the properties that identify that other listing "
    "describe it only and are NOT requirements of the item to buy. That listing may say the value is on a further "
    "listing; do not assume that a value that fits or goes with another is the same value: the pages say which "
    "value goes with which. "
    "Hints: details of the story (use, place, people, feel, timing) that may point to a hidden requirement.")


def _checklist(request: str) -> dict[str, Any] | None:
    try:
        msg = _chat(_model(), [{"role": "system", "content": "You prepare shopping checklists. Reply with one JSON object."},
                               {"role": "user", "content": f"SHOPPER REQUEST:\n{request}\n\n{_BRIEF_TASK}"}])
        data = _json_of(msg.get("content") or "")
        if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
            return data
    except Exception:
        pass
    return None


def _lookup_task(n: int, name: Any, lk: dict[str, Any]) -> str:
    return (f"For item {n + 1} ({name}), find this listing: {lk.get('listing')}, and read its "
            f"'{lk.get('read')}'. Work in hops. Each hop: (a) search the seller or brand name plus ONE "
            "identifying value word (2-3 words, e.g. '<seller> <value>'); (b) read ALL of that seller's rows of "
            "the right kind in ONE side-by-side call (rows do not show properties); (c) open the PAGE view of the "
            "listing that has the identifying properties (notes and pointers are on the page; when the store has "
            "separate page and facts views, open the page even if you read its facts) and read what it says. If it "
            "says the value is to be taken from another "
            "listing, do the next hop for that listing. Stop when a page states the value; if it says which "
            "value goes together with its own, the item to buy needs that other value. If a filter returns "
            "nothing, do not retry it. If searches of the seller plus a value do not show the seller's row with those "
            "values, search the seller name alone and read the next pages of that search (a seller can have many "
            "rows). Report: the chain (id, seller, what each page says) and a last line "
            f"'FINAL VALUE: <property>: <value the item to buy must have>' ({lk.get('relation')}), or "
            "'FINAL VALUE: not found' with the next listing to look for.")


_RL_TASK = (
    "Some shopper replies below may say that a requirement's value is to be read from another listing, named by its "
    "seller or brand, its kind and some of its properties. List each such reference as JSON only: "
    "{\"lookups\": [{\"item\": \"<item the requirement is for>\", \"listing\": \"<the other listing: seller or brand, "
    "kind, identifying properties>\", \"read\": \"<property to read>\", \"relation\": \"<how the item must relate>\"}]} "
    "(an empty list when there is none).")


def _reply_lookups(talk: list[tuple[str, str]]) -> list[dict[str, Any]]:
    if not talk:
        return []
    convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk)
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You read shopper replies. Reply with one JSON object."},
                                        {"role": "user", "content": f"{convo}\n\n{_RL_TASK}"}])
        data = data or {}
        return [x for x in data.get("lookups", []) if isinstance(x, dict) and x.get("listing")][:3]
    except Exception:
        return []


_IDS_TASK = ("Here is how a shopper describes another listing: \"{listing}\". List the property names it uses to "
             "identify that listing (e.g. the property name in '<property> <value>'), leaving out the seller, the brand, the kind "
             "of item and pointers to other pages. Reply with JSON only: {{\"properties\": [\"<property>\", ...]}}.")


def _identifiers(listing: str) -> list[str]:
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
                                        {"role": "user", "content": _IDS_TASK.format(listing=listing)}])
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [str(x).strip() for x in data.get("properties") or [] if isinstance(x, str) and str(x).strip()][:4]


_STORY_TASK = (
    "Hidden requirements are listing properties the shopper did not state; the story often hints at one indirectly "
    "(a remark about where the item will go, what it has to suit or how it will be used can point to one property of "
    "the item). For each item, name at most ONE property "
    "that a phrase of the story clearly hints at: it must be a key of that item's listings in the FACT KEYS list above, "
    "not a property the request states, and you must quote the exact words of the request that hint at it. Give none "
    "for an item when no phrase clearly points to one property. Reply with JSON only: {\"hints\": [{\"item\": "
    "\"<item name as in the checklist>\", \"property\": \"<key>\", \"phrase\": \"<exact words>\"}]}")


def _fact_keys(seen: dict[str, dict[str, Any]], names: list[str], n_vals: int = 3) -> str:
    per: list[dict[str, dict[str, list[str]]]] = [{} for _ in names]
    for d in seen.values():
        tabs = [v for v in d.values() if isinstance(v, dict)]
        if not tabs:
            continue
        idx = _line_item(_flat_facts(d), names) if names else None
        if idx is None:
            continue
        for tab in tabs:
            for k, v in tab.items():
                if v is None or isinstance(v, bool) or not str(v).strip() or _PROTOCOL_KEY.search(str(k)):
                    continue
                key = " ".join(_vc_tok(str(k).replace("_", " ").replace("-", " "))) or str(k)
                e = per[idx].setdefault(key, {"names": [], "vals": []})
                if k not in e["names"]:
                    e["names"].append(k)
                sv = str(v).strip()[:40]
                if sv not in e["vals"] and len(e["vals"]) < n_vals:
                    e["vals"].append(sv)
    holders: dict[str, list[int]] = {}
    for i in range(len(names)):
        for key in per[i]:
            holders.setdefault(key, []).append(i)
    def mark(i: int, key: str, e: dict[str, list[str]]) -> str:
        tags = []
        if len({v.lower() for v in e["vals"]}) > 1:
            tags.append("varies")
        others = [j + 1 for j in holders.get(key, []) if j != i]
        if others:
            tags.append("shared with item " + ", ".join(map(str, others)))
        return f"{' / '.join(e['names'][:2])} ({', '.join(e['vals'])})" + (f" [{'; '.join(tags)}]" if tags else "")
    lines = [f"item {i + 1} ({nm}): " + "; ".join(mark(i, key, e) for key, e in per[i].items())
             for i, nm in enumerate(names) if per[i]]
    return ("FACT KEYS (the exact keys of the listings read, from the store's fact tables, with values seen; 'varies' = "
            "the values differ among the item's listings; 'shared' = another requested item's listings have the key):\n"
            + "\n".join(lines)) if lines else ""


def _story_props(context: str, request: str, names: list[str]) -> list[tuple[str, str]]:
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
                                        {"role": "user", "content": f"{context}\n\n{_STORY_TASK}"}])
    except Exception:
        return []
    out: list[tuple[str, str]] = []
    low = re.sub(r"\s+", " ", request.lower())
    for h in (data or {}).get("hints") or [] if isinstance(data, dict) else []:
        if not isinstance(h, dict):
            continue
        item, prop, phrase = str(h.get("item") or ""), str(h.get("property") or "").strip(), str(h.get("phrase") or "")
        phrase = re.sub(r"\s+", " ", phrase.lower()).strip(" .,'\"")
        if not prop or len(phrase) < 6 or phrase not in low:
            continue
        nm = next((n for n in names if _norm(n) == _norm(item) or _norm(n) in _norm(item) or _norm(item) in _norm(n)), "")
        if nm or len(names) == 1:
            out.append((nm or names[0], prop))
    return out


def _cat_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2 and w not in ("and", "the")}


_ASKS_TASK = ("Does the shopper's request below ask a direct question that needs an answer about the purchase, the "
              "basket or the total? Polite requests phrased as questions "
              "('Could you ...?', 'Can you ...?') do not count. Reply with JSON only: {\"answer\": \"yes\" or \"no\"}.")


def _asks_question(request: str) -> bool:
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
                                        {"role": "user", "content": f"REQUEST:\n{request}\n\n{_ASKS_TASK}"}])
    except Exception:
        return False
    return isinstance(data, dict) and str(data.get("answer", "")).strip().lower().startswith("y")


_QTY_TASK = ("How many units of '{name}' does the request below ask for? Use only a number the request states for this "
             "item (in digits or in words); who will use it or how many people there are does not count; if no "
             "number is stated, the answer is 1. Reply with JSON only: {{\"quantity\": <n>}}.")


def _stated_quantity(request: str, name: str) -> int | None:
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
                                        {"role": "user", "content": f"REQUEST:\n{request}\n\n" + _QTY_TASK.format(name=name)}])
        return int(data.get("quantity")) if isinstance(data, dict) and data.get("quantity") is not None else None
    except Exception:
        return None


def _item_text(item: dict[str, Any]) -> str:
    must = "; ".join(str(m) for m in item.get("must") or []) or "none stated"
    return f"{item.get('name')} x {item.get('quantity') or 1} (requirements: {must})"


_BROWSE_SYS = (
    "You research an online store for a shopping assistant, using only the supplied tools. Search with short queries "
    "of 2-3 words (the store reads only the first words), e.g. a seller or brand name or the item's kind word plus "
    "one distinctive value. Search results are rows without properties: read the details of the promising rows, "
    "cheapest of the right kind first (rows show prices; a cheaper row is not skipped for its title), "
    "several side by side in one call when the store has a tool for that. Never read a listing twice. A result page "
    "may give a cursor for the next page. When the store has separate tools for a listing's page and for its facts, "
    "notes and pointers to other listings are on the page view, and requirements are judged on the facts view: a value "
    "missing from a page, title or description is no evidence, so read the facts view before excluding a listing. Judge properties on the details (facts), not on titles. A fit or "
    "compatibility list that names a model family or series (e.g. 'X 35xx') covers the models in it (X 3520). A requirement is met "
    "only when the details state that value itself; a related, stronger or weaker value is a different value (e.g. "
    "'X-free' does not meet 'low X'), and a value that inserts words inside the phrase ('X Y Z' for 'X Z') does not "
    "state it. A listing's notes "
    "(annotations) are store information too: when a page says its value goes together with another value, the item "
    "to buy needs that other value. When your task is "
    "done, or when you are told to stop, reply in plain text without a tool call: that text is your report.")


def _browse(run: _Run, kinds: dict[str, Any], context: str, task: str, allowance: int, deadline: float,
            extra: str = "", model: str | None = None) -> str:
    msgs: list[dict[str, Any]] = [
        {"role": "system", "content": _BROWSE_SYS + (f"\n{extra}" if extra else "")},
        {"role": "user", "content": f"{context}\n\nYOUR TASK: {task}\nYou may make at most {allowance} tool calls."}]
    tools = kinds["browse"]
    used = 0
    roles = kinds.get("roles") or {}
    searches, read_any, rows_seen = 0, False, False
    for _ in range(allowance + 2):
        if run.finished:
            return ""
        if used >= allowance or run.elapsed() > deadline:
            break
        msg = _chat(model or _model(), msgs, tools, cache_at=[1, len(msgs) - 1])
        calls = msg.get("tool_calls") or []
        if not calls:
            return (msg.get("content") or "").strip()
        msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for tc in calls:
            if used >= allowance:
                obs: Any = {"error": "tool-call allowance used: write your report now"}
            else:
                if _over_cap(run, tc["function"]["name"], _arguments(tc)):
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"error": _CAP_NOTE})})
                    continue
                role = roles.get(tc["function"]["name"])
                if role == "search" and searches >= 2 and rows_seen and not read_any:
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({
                        "error": "Not run (no step used): you have searched twice; read the details of the promising "
                                 "rows of those results first (side by side when the store has a tool for that), then "
                                 "search again only if none of them fits."})})
                    continue
                searches += role == "search"
                read_any = read_any or role in ("open", "compare")
                try:
                    obs = run.env(tc["function"]["name"], _arguments(tc), tc, msg.get("content") or "")
                except ValueError as exc:
                    obs = {"error": str(exc)}
                used += 1
                if roles.get(tc["function"]["name"]) == "search" and _row_ids(obs):
                    rows_seen = True
                try:
                    mf = _multi_fail(_arguments(tc), obs)
                except Exception:
                    mf = 0
                if mf:
                    run.id_cap[tc["function"]["name"]] = 2
                    obs = {"result": obs, "note": _MULTI_NOTE.format(n=mf)}
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(_clip(obs))})
    msgs.append({"role": "user", "content": "Stop now and write your report in plain text."})
    try:
        return (_chat(model or _model(), msgs, tools, tool_choice="none", cache_at=[1, len(msgs) - 1]).get("content") or "").strip()
    except Exception:
        return ""


_Q_TASK = (
    "You choose the next question to a shopper, to uncover hidden requirements on listing properties. Each question "
    "asks about ONE property of ONE item: name that item as in the checklist. A reply that gives a value is a "
    "requirement; a reply with no preference, or one saying the shopper did not understand, means that property is not "
    "a requirement for that item: never ask it again. A reply asking you to ask one thing at a time, to a question about a single property, means that property IS a "
    "requirement for one or more items: ask it next naming each item whose listings have it, one per message, and "
    "keep going through ALL those items even after one of them revealed a requirement. Choose "
    "in this order: (1) properties the story hints at; (1b) a property the request states for one item, asked for "
    "each OTHER item whose listings have it; (2) a key the FACT KEYS list marks as shared with another item; (3) for "
    "each item still without a requirement, among its keys marked 'varies', the one a buyer of that kind of item, "
    "given what the request already states for it, would most likely insist on; (4) other properties of the items "
    "still without a requirement. Avoid condition, warranty, options, and properties the request or a lookup already "
    "fixes. Ask only properties whose keys the FACT KEYS list or the request show; when a story hint names "
    "several possible properties, ask the one the store's listings have; never ask a bare "
    "'type'. Use each property key exactly as the store spells it (underscores may become spaces), never a paraphrase "
    "(never a shortened or reworded name), as "
    "the store shows them. Reply with JSON only: {\"property\": \"<property key>\", \"item\": \"<item name as in "
    "the checklist>\"} or {\"done\": true} when nothing useful is left.")


_KIND_TASK = (
    "A shopping assistant asked a shopper the question below. Classify the shopper's reply as one of: 'requirement' "
    "(it states a value the item must have, or that it must be the same as another listing), 'no_preference', "
    "'not_understood' (it asks to repeat, simplify or explain), 'one_at_a_time' (it asks for the questions to come one "
    "at a time). Also name the item the reply is about, using exactly one of these names, "
    "or null: {names}. For a 'requirement', also give the property as the reply names it (not the question's word), the relation ('must be', 'at most', 'at least', "
    "'not', or 'same as' when the value is to be read from another listing) and the value. Read the whole sentence "
    "and decide which values the shopper accepts. First copy the words of the reply that say which values are "
    "accepted ('accepted'). When only one value is accepted, the relation is 'must be' with that value ('same as' when "
    "it is to be read from another listing). When every value except one is accepted, the relation is 'not' with that "
    "one value, also when that value is to be read from another listing. Mind the words that exclude a value: a "
    "phrase meaning 'every value except V' is a 'not V' wish when the shopper wants it, and a 'must be V' requirement "
    "when the shopper turns it down. "
    "Reply with JSON only: {{\"kind\": \"...\", \"item\": ..., \"property\": ..., \"accepted\": \"...\", \"relation\": ..., "
    "\"value\": ...}}")


def _reply_kind(question: str, reply: str, names: list[str]) -> tuple[str, str, str]:
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You classify shopper replies. Reply with one JSON object."},
                                        {"role": "user", "content": f"QUESTION: {question}\nREPLY: {reply}\n\n"
                                                                    + _KIND_TASK.format(names=json.dumps(names))}])
    except Exception:
        return "", "", ""
    if not isinstance(data, dict):
        return "", "", ""
    kind = str(data.get("kind") or "").strip().lower()
    item = str(data.get("item") or "").strip()
    reading = ""
    if kind == "requirement" and data.get("property") and data.get("value"):
        reading = f"{item or 'the item'}: {data.get('property')} {data.get('relation') or 'must be'} {data.get('value')}"
    return kind, item, reading


def _norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def _norm_prop(prop: str, names: list[str]) -> str:
    nouns = {w.rstrip("s") for n in names for w in re.findall(r"[a-z]+", n.lower()) if len(w) > 2}
    words = [w for w in re.findall(r"[a-z0-9]+", (prop or "").lower())]
    kept = [w for w in words if w.rstrip("s") not in nouns] or words
    return "".join(kept)


def _next_question(context: str, talk: list[tuple[str, str]], left: int, avoid: str = "",
                   extra: str = "") -> tuple[str, str] | None:
    convo = "\n".join(f"Q{i + 1}: {q}\nA{i + 1}: {a}" for i, (q, a) in enumerate(talk)) or "(no questions yet)"
    msgs = [{"role": "system", "content": "You plan questions for a shopping assistant. Reply with one JSON object."},
            {"role": "user", "content": f"{context}\n\nQUESTIONS AND REPLIES SO FAR:\n{convo}\n\n"
                                        f"Messages left: {left}.\n\n{_Q_TASK}" + (f"\n{extra}" if extra else "")
                                        + (f"\n\n{avoid}" if avoid else "")}]
    data: Any = None
    for _ in range(2):
        try:
            data, text = _chat_json(_model(), msgs)
            msg = {"content": text}
        except Exception:
            return None
        if isinstance(data, dict) and data.get("done") and left > 0:
            msgs += [{"role": "assistant", "content": msg.get("content") or ""},
                     {"role": "user", "content": "Messages are still left and an unused message is a wasted chance: "
                                                 "do not return done; pick the most promising remaining question."}]
            continue
        break
    if not isinstance(data, dict) or data.get("done"):
        return None
    item, prop = str(data.get("item") or "").strip(), str(data.get("property") or "").strip()
    if item.lower() in ("null", "none"):
        item = ""
    if not prop:
        return None
    return item, prop


_QUERY_TASK = ("Write one store search query per item (up to 4) that finds listings meeting its requirements. Each query has "
               "exactly 3 words: the word that names the item's kind (e.g. 'kettles', 'lamps', 'jackets') as ONE word, "
               "plus the most distinctive single word of TWO requirement values together (always include the newest "
               "revealed requirement; a brand or dealer name the requirement names counts as a value). Keep the word that names a "
               "value itself; leave out words that only qualify it (degree, negation, comparison), property names, and "
               "conditions on price, budget, stock or delivery.")


def _queries(context: str, talk: list[tuple[str, str]], extra: str = "", model: str | None = None) -> tuple[str, list[str]]:
    convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk) or "(none)"
    task = ("First write every requirement value of each item as 'item: property value; item: property value': the "
            "stated ones (a conditional one with its condition, as 'property value (only if <condition>)'), the FINAL value of every lookup (follow what the pages say; when the last page says its value "
            "goes together with another value, the item to buy needs that other value; the properties that identify a "
            "looked-up listing are not requirements of the item to buy), and every value the shopper's replies require, "
            "taken literally (keep bounds and comparisons exactly: 'at most N' stays 'at most N', never 'N') even when it looks like a placeholder (a reply that wants a placeholder value under a key requires "
            "that value; ignore replies that give no preference or did not understand). " + (f"{extra} " if extra else "") + "Then: " + _QUERY_TASK + " Return JSON only: {\"requirements\": \"...\", \"queries\": [\"...\"]}.")
    try:
        data, _ = _chat_json(model or _model(), [{"role": "system", "content": "You write search queries for an online store. Reply with one JSON object and nothing else."},
                                        {"role": "user", "content": f"{context}\n\nQUESTIONS AND SHOPPER REPLIES:\n{convo}\n\n{task}"}])
        data = data or {}
        qs = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:4]
        return str(data.get("requirements") or "").strip(), qs
    except Exception:
        return "", []


_BUY_SYS = (
    "You are a shopping assistant in an online store. Buy what the shopper asks for: every stated requirement, every "
    "lookup result and every requirement the shopper's replies revealed must hold, and among the listings that fit "
    "choose the best one (the cheapest in-stock listing within the budget unless the shopper says otherwise). Decline "
    "only when no listing can fit.\n"
    "The order is all or nothing: never order only part of the requested items. If some item has no listing that "
    "meets its STATED requirements (the request's own rules and the budget) after a real search, empty the basket and "
    "decline the whole request. If the stated requirements are met and only a lookup-derived or revealed value cannot "
    "be confirmed, buy the best fitting listing instead of declining.\n"
    + _MEETS_RULE + " Search results show titles only, and a title that does not mention a required value says "
    "nothing about it: after each search, read the details of its in-stock, in-budget rows of the right kind, the "
    "cheapest first (the best fit is the cheapest one that fits; never skip a cheaper row because its title does not "
    "mention a requirement), side by side when the store has a tool for that, before you judge them or search again. A listing's notes (annotations) are store information: when a page says its value goes "
    "together with another value, the item to buy needs that other value. Buy every item: if no listing meets every "
    "requirement of an item after reading every right-kind, in-budget row your searches showed and at least three "
    "searches aimed at the missing requirement (for a 'not X' rule, "
    "search the item's kind word with other common values of that property), take the one that meets the most rather "
    "than leaving it out; this applies only to values derived from lookups: never buy a listing that lacks a value "
    "the shopper's replies revealed. Before "
    "adding a listing, check its details against every FINAL REQUIREMENT yourself (the research reports' verdicts may "
    "be wrong). The store may also check a requirement on OTHER basket lines that have the same property key (e.g. if "
    "one item must have value V under key K, another item's listing with key K and another value can break it; a "
    "different key such as 'item-K' does not). So when you can choose, take for the other items a listing "
    "that lacks that key or has the same value there; but never decline or leave an item out for this reason. Among "
    "listings that fit equally, prefer the one with fewer of those keys, then the one with fewer properties. A listing of another category never counts, even when it states the "
    "values; if no right-kind listing fits after a real search, decline, unless the request says what to do when "
    "nothing fits: then do that instead (order the listing it describes). The best fit is the cheapest one, so for every item "
    "list the item's shelf with the store's filter, naming the category exactly as listings show it (e.g. "
    "'Desks', not 'desk'), with a LOW price cap first, about a quarter of the budget, so that only a few rows remain "
    "(the filter shows a few rows per page and does not sort by price); if none fits, double the cap; read the "
    "candidates before choosing (text search alone misses cheap fits). Where the store has separate page and facts "
    "views, judge requirements on the facts view: a value missing from a page, title or description is no evidence, so "
    "never exclude a listing (or decline) before reading its facts view. A listing's "
    "category must be the requested kind of item. When a listing you "
    "choose has a fine-print note (annotation), search its title words once to see same-title twins from other sellers, "
    "and prefer an equal-price twin with no note that shows each required value under every spelling of that key the "
    "twins use (e.g. both 'x y' and 'x_y'); this spelling preference is only between same-title equal-price twins: "
    "between different listings one spelling of the key is enough, and the cheaper fit wins. When the store has separate page and facts views, open BOTH views of "
    "every listing you order, in this episode, before finishing (the store checks that). Before finishing, if steps allow, read the carted listings once more "
    "to check they are still in stock. Search with short queries (2-3 words; try the suggested searches first); rows do not show "
    "properties, so read the promising rows' details, several side by side when the store has a tool for that; never "
    "read a listing twice; when a page shows the right kind but no fit, read its next page. Then put the chosen "
    "listings in the basket with the requested quantities, each item's whole quantity on ONE listing line (do not "
    "split it over several listings), and finish with the order. If no listing within the budget states a required "
    "value in its details, a listing whose title or page states it counts (prefer one whose other required values are "
    "in its details). If the store refuses, read the "
    "notice, fix the basket and try again: after a refused finish or a notice, re-open every carted listing's details "
    "(stock and price), replace any line that is sold out with the next cheapest fit (look at the cheapest rows of your result pages "
    "again rather than choosing from memory), then finish. Any tool result can carry notices from the shopper or the store (a new "
    "budget, an item sold out or repriced, a requirement dropped): read them every time and update the requirements "
    "(a new budget replaces the old one) before you finish.\n"
    "Every tool call has a 'notes' argument: write there 'REQUIREMENTS: <item>: <value>; ...' with every requirement "
    "value, then 'NOTES: ...' with the listings read, whether each fits, and what you will do next.\n"
    "Every reply must be exactly one call to one of the supplied tools; each tool call is one environment step. Never "
    "answer in plain text. Continue until a result reports finished=true.")


_REVIEW_TASK = (
    "Below are a shopping assistant's notes and the latest store results. For every listing that is in the basket "
    "(or about to be ordered) and whose details have a fine-print note (an annotation or remark), write one search "
    "query of 3-4 distinctive product words taken from its title (skip promotional words), to find same-title twins "
    "from other sellers. Reply with JSON only: "
    "{\"queries\": [\"...\"]} (an empty list when no basket listing has a note).")


def _review(messages: list[dict[str, Any]]) -> list[str]:
    recent = [m for m in messages if m.get("role") in ("tool", "assistant")][-14:]
    text = []
    for m in recent:
        if m["role"] == "tool":
            text.append("RESULT: " + str(m.get("content"))[:2500])
        else:
            for tc in m.get("tool_calls") or []:
                text.append("CALL: " + str(tc["function"].get("name")) + " " + str(tc["function"].get("arguments"))[:800])
    try:
        data, _ = _chat_json(_model(), [{"role": "system", "content": "You review a shopping basket. Reply with one JSON object."},
                                        {"role": "user", "content": "\n".join(text) + "\n\n" + _REVIEW_TASK}])
        data = data or {}
        return [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:3]
    except Exception:
        return []


def _is_decline(kinds: dict[str, Any], name: str, args: dict[str, Any]) -> bool:
    if name in kinds.get("decline", ()):
        return True
    if name in kinds.get("order", ()):
        return False
    fv = (kinds.get("finish_vals") or {}).get(name)
    if fv and args:
        return all(str(args.get(k, "")).strip().lower() == str(v).strip().lower() for k, v in fv["decline"].items())
    return False


def _fill_required(fn: dict[str, Any], args: dict[str, Any], order: bool) -> dict[str, Any]:
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    for k in params.get("required") or []:
        if k not in args and (props.get(k) or {}).get("type", "string") == "string" and not (props.get(k) or {}).get("enum"):
            args[k] = "Ordering the basket as it stands." if order else "No listing fits every requirement."
    return args


def _close_call(kinds: dict[str, Any], order: bool) -> tuple[str, dict[str, Any]] | None:
    fns = {(t.get("function") or {}).get("name"): (t.get("function") or {}) for t in kinds.get("tools") or []}
    for name in sorted(kinds.get("order" if order else "decline") or ()):
        if name in fns:
            return name, _fill_required(fns[name], {}, order)
    for name, fv in sorted((kinds.get("finish_vals") or {}).items()):
        if name in fns:
            return name, _fill_required(fns[name], dict(fv["order" if order else "decline"]), order)
    return None


_NOTICE_GATE = (
    "Not finished yet (no step used). These notices arrived after the basket was planned: {notices}. Handle each one "
    "before finishing: a new budget replaces the old one (a higher budget brings back listings you left out as too "
    "dear, so reconsider them; a lower one may put the basket over it); a dropped requirement no longer applies, so "
    "the cheapest in-stock listing of the kind that meets the remaining requirements may be another one: search the "
    "item's category for it; a pick that is sold out, taken or unavailable must be removed and replaced by the next "
    "cheapest fit; a repriced pick must be checked again against the budget and the other fits. First re-open the "
    "details of {lines} to see their stock and price now, then fix the basket and finish.")


_PRICE_KEY = re.compile(r"(price|cost|amount)", re.I)


def _row_price(row: dict[str, Any]) -> float:
    for k, v in row.items():
        if _PRICE_KEY.search(k) and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return float("inf")


_REQ_STRUCT_TASK = (
    "Below are the items of a shopping request and its final requirements. List every requirement that a listing's own "
    "facts must state, one entry each, as JSON only: {{\"reqs\": [{{\"item\": \"<item name as written>\", "
    "\"property\": \"<property name as written>\", \"value\": \"<the required value with its words and any bound, e.g. "
    "'at most 30 cm', 'steel', 'a or b'>\"}}]}}. Leave out the budget, price, stock, quantity, delivery, every 'not' rule, "
    "conditional rules, compatibility or fit rules, and anything that is not one property's value.\n\nITEMS: {items}\n"
    "FINAL REQUIREMENTS: {reqs}")
_PROTOCOL_KEY = re.compile(r"(^id$|_id$|title|name|description|about|overview|write|seller|dealer|vendor|brand|maker|"
                           r"price|cost|amount|stock|ready|avail|orderable|category|department|shelf|remark|note|"
                           r"annotation|fine|image|url|rating|review|option)", re.I)
_STOCK_KEY = re.compile(r"(stock|avail|ready|orderable)", re.I)


def _struct_reqs(req_list: str, names: list[str]) -> list[dict[str, str]]:
    try:
        data, _ = _chat_json(_model(), [
            {"role": "system", "content": "You read shopping requirements. Reply with one JSON object."},
            {"role": "user", "content": _REQ_STRUCT_TASK.format(items="; ".join(names), reqs=req_list)}])
    except Exception:
        return []
    out = []
    for r in (data or {}).get("reqs") or [] if isinstance(data, dict) else []:
        if isinstance(r, dict) and r.get("property") and r.get("value"):
            out.append({k: str(r.get(k) or "") for k in ("item", "property", "value")})
    return out


def _flat_facts(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out.update({kk: vv for kk, vv in v.items() if isinstance(vv, (str, int, float, bool))})
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
    return out


def _prop_values(prop: str, facts: dict[str, Any]) -> list[str]:
    pl = _vc_tok(prop)
    pt = set(pl)
    vals = []
    for k, v in facts.items():
        if _PROTOCOL_KEY.search(k) or isinstance(v, bool):
            continue
        kl = _vc_tok(k.replace("_", " ").replace("-", " "))
        if kl and pl and (set(kl) == pt or (len(kl) == len(pl) and all(_close(a, b) for a, b in zip(kl, pl)))):
            vals.append(str(v))
    return vals


def _item_index(text: str, names: list[str]) -> int | None:
    if len(names) == 1:
        return 0
    tt = _vc_tok(text)
    for i, n in enumerate(names):
        if tt and _vc_tok(n) == tt:
            return i
    best = sorted(((len(set(_vc_tok(n)) & set(tt)), i) for i, n in enumerate(names)), reverse=True)
    return best[0][1] if best and best[0][0] > 0 and (len(best) < 2 or best[0][0] > best[1][0]) else None


def _line_item(facts: dict[str, Any], names: list[str]) -> int | None:
    cat = " ".join(str(v) for k, v in facts.items() if re.search(r"(category|department|shelf|group)", k, re.I))
    return _item_index(cat, names)


def _basket_problems(cart_ids: list[str], seen: dict[str, dict[str, Any]], sreqs: list[dict[str, str]],
                     names: list[str], held: dict[str, int] | None = None, qtys: list[int] | None = None) -> list[str]:
    problems = []
    for pid in cart_ids[:8]:
        facts = _flat_facts(seen.get(pid) or {})
        if not facts:
            continue
        if any(_STOCK_KEY.search(k) and v is False for k, v in facts.items()):
            problems.append(f"{pid} shows out of stock")
        idx = _line_item(facts, names)
        if idx is None:
            continue
        if held and qtys and idx < len(qtys) and held.get(pid) and qtys[idx] and held[pid] != qtys[idx]:
            problems.append(f"{pid}: check its quantity: the request, as read, asks for {qtys[idx]} of the {names[idx]} "
                            f"and the basket holds {held[pid]} (a requested quantity counts listing units, a pack listing "
                            f"is one unit; keep {held[pid]} only if the request itself asks for {held[pid]})")
        mine = [r for r in sreqs if _item_index(r.get("item", ""), names) == idx]
        text = " ".join(str(v) for k, v in facts.items() if isinstance(v, str))
        for r in mine:
            vals = _prop_values(r["property"], facts)
            if vals:
                if _vc_check_fields(r["value"], vals) == "no":
                    shown = "; ".join(sorted(set(vals)))[:120]
                    problems.append(f"{pid}: {r['property']} must be '{r['value']}', its facts say '{shown}'")
            elif _vc_check_text(r["value"], text) == "no":
                problems.append(f"{pid} does not state {r['property']} '{r['value']}'")
    lines = {pid: _line_item(_flat_facts(seen.get(pid) or {}), names) for pid in cart_ids[:8]}
    if len(names) > 1 and lines and all(i is not None for i in lines.values()):
        for j, nm in enumerate(names):
            if j not in lines.values():
                twice = [p for p, i in lines.items() if list(lines.values()).count(i) > 1]
                problems.append(f"no basket line is of the kind '{nm}' (by its category; a listing of another category "
                                f"does not count)" + (f": {', '.join(twice)} share the kind of one other item" if twice else ""))
    return problems


def _drop_hits(dp: str, r: dict[str, str]) -> bool:
    stop = {"the", "a", "an", "of", "on", "for", "its", "any", "requirement", "requirements", "item", "items"}
    dt = {w for w in _vc_tok(dp) if w not in stop}
    pt, vt = set(_vc_tok(r.get("property", ""))), set(_vc_tok(r.get("value", "")))
    return bool(dt) and bool(pt) and (dt <= pt or pt <= dt or dt <= vt)


def _drop_valid(dp: str, req_text: str) -> bool:
    stop = {"the", "a", "an", "of", "on", "for", "its", "any", "requirement", "requirements", "item", "items"}
    words = [w for w in _vc_tok(dp) if w not in stop]
    have = set(_vc_tok(req_text))
    return 0 < len(words) <= 4 and all(w in have for w in words)


def _sentences(notes: list[str]) -> list[str]:
    return [n for n in notes if len(n) >= 15 and " " in n.strip()]


def _buy(run: _Run, kinds: dict[str, Any], tools_all: list[dict[str, Any]], context: str, allow_ask: bool,
         msg_cap: int = 0, summary: bool = False, n_lines: int = 0, extra: str = "",
         seen: set[str] | None = None, cheaper_sweep: bool = False, sreqs: list[dict[str, str]] | None = None,
         names: list[str] | None = None, qtys: list[int] | None = None, fallback: str = "") -> None:
    fallback_held = 0
    tools = json.loads(json.dumps([t for t in tools_all if allow_ask or summary or (t.get("function") or {}).get("name") != kinds["ask"]]))
    summary_sent = False
    decline_held = False
    decline_reviewed = False
    cart_ids: list[str] = list(run.held)
    new_notices: list[str] = []
    reread: set[str] = set()
    gate_seen: set[str] = set(seen or ())
    browse_names = {(t.get("function") or {}).get("name") for t in kinds.get("browse") or []}
    detail = list(kinds.get("open") or [])
    views_held: set[tuple[str, str]] = set()
    checked: set[str] = set()
    sweep = kinds.get("sweep")
    swept: set[str] = set()
    read_ids: set[str] = set()
    seen_notes: set[str] = set(seen or ())
    paged_queries: set[str] = set()
    for tool in tools:
        params = tool.setdefault("function", {}).setdefault("parameters", {"type": "object", "properties": {}})
        params.setdefault("properties", {})["notes"] = {"type": "string", "description": "Your working notes: REQUIREMENTS line and NOTES line."}
        params["required"] = list(params.get("required") or []) + ["notes"]
    messages: list[dict[str, Any]] = [{"role": "system", "content": _BUY_SYS + (f"\n{extra}" if extra else "")},
                                      {"role": "user", "content": context}]
    warned = step_warned = held = decide_warned = refused_noted = False
    pending: list[str] = []
    start_step = run.step
    sent_msgs = 0
    idle = 0
    forced = False
    while not run.finished and run.step < run.max_steps:
        if run.elapsed() > 270 and not forced:
            forced = True
            call = _close_call(kinds, order=bool(run.held))
            if call is not None:
                run.env(call[0], call[1])
                if run.finished:
                    return
        if not warned and (run.elapsed() > 200 or run.step - start_step >= 25):
            warned = True
            messages.append({"role": "user", "content": "Decide now: put the best fitting listings you have read in the basket and finish with the order."})
        if not decide_warned and n_lines and run.left() <= n_lines + 9 and not warned:
            decide_warned = True
            messages.append({"role": "user", "content": f"{run.left()} steps are left: add your chosen listings to the "
                             "basket now (searching more leaves no steps for the basket check), then finish."})
        if not step_warned and run.left() <= 4:
            step_warned = True
            messages.append({"role": "user", "content": "Only 4 steps are left: put the best fitting listings in the basket now (if they are not there yet) and finish with the order."})
        tool_idx = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        old = set(tool_idx[:-8])
        sent = [({**m, "content": _compact(str(m.get("content")))} if i in old and len(str(m.get("content"))) > 300 else m)
                for i, m in enumerate(messages)]
        msg = _chat(_model(), sent, tools, cache_at=[1, max(old) if old else None, len(sent) - 1])
        content = msg.get("content") or ""
        calls = msg.get("tool_calls") or []
        if not calls:
            idle += 1
            if idle > 2:
                call = _close_call(kinds, order=bool(run.held))
                if call is None:
                    raise RuntimeError("model returned no tool call before the episode finished")
                run.env(call[0], call[1])
                return
            messages += [{"role": "assistant", "content": content},
                         {"role": "user", "content": "Choose one of the supplied tools to continue."}]
            continue
        idle = 0
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for tc in calls:
            name = tc["function"].get("name")
            try:
                args = {k: v for k, v in _arguments(tc).items() if k != "notes"}
            except ValueError as exc:
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"error": str(exc)})})
                continue
            if run.finished:
                obs: Any = {"error": "the episode has finished"}
            elif (name in kinds["close"] and sreqs is not None and names and cart_ids
                  and not _is_decline(kinds, name, args) and run.left() >= 3 and run.elapsed() < 255
                  and (found := [p for p in _basket_problems(cart_ids, run.seen, sreqs, names, run.held, qtys)
                                 if p not in checked])):
                checked.update(found)
                obs = {"error": "Not finished yet (no step used): a check of each basket line against the FINAL "
                                "REQUIREMENTS found: " + " | ".join(found) + ". Replace such a line with the cheapest "
                                "listing whose facts meet every requirement of its item (search for one if you have not "
                                "read it); keep a line only when no listing of its kind states the value after a real "
                                "search. Then finish with the order."}
            elif (name in kinds["close"] and new_notices and run.left() >= 3 and run.elapsed() < 250
                  and (_is_decline(kinds, name, args) or not cart_ids or not set(cart_ids) <= reread)):
                obs = {"error": _NOTICE_GATE.format(notices=" | ".join(f"'{n}'" for n in new_notices[-4:]),
                                                    lines=", ".join(cart_ids[:8]) if cart_ids else "the listings you meant to buy")}
                new_notices = []
            elif (name in kinds["close"] and len(detail) >= 2 and cart_ids and not _is_decline(kinds, name, args)
                  and (missing := [(i, d) for i in cart_ids[:8] for d in detail
                                   if d not in run.views.get(i, set()) and (i, d) not in views_held])
                  and len(missing) <= run.left() - 1 and run.elapsed() < 255):
                views_held.update(missing)
                obs = {"error": "Not finished yet (no step used): the store checks that every ordered listing was "
                                "opened in each of its detail views in this episode. Still missing: "
                                + "; ".join(f"{i} with {d}" for i, d in missing)
                                + ". Make these calls now (one each), then finish with the order."}
            elif (name in kinds["cart"] and sweep and (cop := _cart_op(args, run.seen, (kinds.get("removes") or {}).get(name))) and cop[1] in ("add", "set")
                  and cop[0] not in swept and cop[0] not in run.held and len(swept) < n_lines + 2
                  and run.left() >= 10 + n_lines and run.elapsed() < 220
                  and (swept.add(cop[0]) or True)
                  and (cheaper := _cheaper_rows(run, sweep, cop[0]))):
                obs = {"error": f"Not added yet: the store's price filter shows cheaper in-stock rows of the same shelf "
                                f"that you have not opened (cheapest first): "
                                + "; ".join(f"{i} {p:g} {str(r.get(next((k for k in r if re.search('title|name', k, re.I)), ''), ''))[:60]}"
                                            for p, i, r in cheaper)
                                + f". The best fit is the cheapest listing that meets every requirement: read these rows' "
                                f"details, cheapest first (side by side when the store has a tool for that), before adding "
                                f"{cop[0]}; add {cop[0]} only if none of them fits."}
            elif (name in kinds["close"] and fallback and fallback_held < 2 and _is_decline(kinds, name, args)
                  and run.left() >= 3):
                fallback_held += 1
                obs = {"error": "Not declined (no step used): the request says what to do when no listing meets every "
                                f"requirement: '{fallback[:300]}'. Do that instead of declining: put the listing it "
                                "describes in the basket (the quantities the request asks for) and finish with the order."}
            elif (name in browse_names and n_lines and not (_arg_ids(args) & set(cart_ids))
                  and run.left() <= n_lines + 2 + (1 if summary and not summary_sent else 0) + 4):
                obs = {"error": "Not run (no step used): only the steps to add your picks, finish and recover from a "
                                "refused finish are left. Add the best fitting listings you have read now and finish."}
            elif (name in kinds["close"] and _is_decline(kinds, name, args) and not decline_held
                  and run.left() >= 5 and run.elapsed() < 235):
                decline_held = True
                unread = _unread_by_item(run, names or [])
                obs = {"error": "Not declined yet (no step used). "
                                + ("Rows your result lists showed that you never opened, per item, cheapest first: "
                                   + unread + ". Open those of the item that has no fit yet (side by side when the "
                                   "store has a tool for that) before anything else. " if unread else "")
                                + "A decline fails unless some item has no listing of "
                                "its kind at all. First search the bare required value words with the item's kind "
                                "word (e.g. '<kind> <value>') and read the next pages; a value of several words (e.g. "
                                "'a/b blend') gets one search per distinctive word ('<kind> a', '<kind> b'); read the FACTS view of every "
                                "candidate you excluded only from its page, title or description; a listing whose title or page "
                                "states a required value counts when none states it in its details within the budget. "
                                "Only requirements that the request states, the lookups give or the shopper's replies "
                                "reveal count: a remark about where or how the item will be used is not a requirement, "
                                "and a property the shopper had no preference on accepts any value. "
                                "If every item has a listing, buy the best fitting basket within the budget instead."}
            elif (name in kinds["close"] and _is_decline(kinds, name, args) and decline_held
                  and not decline_reviewed and run.left() >= 4 and run.elapsed() < 245):
                decline_reviewed = True
                fits = _decline_review(context, run.seen, names)
                fit_items = {_line_item(_flat_facts(run.seen.get(i) or {}), names) for i, _ in fits or []} if names else set()
                missing = {n for n in range(len(names or [])) if n not in fit_items} if fits and names and len(names) > 1 else set()
                if fits and missing:
                    rows = _unread_by_item(run, names or [], only=missing)
                    obs = {"error": "Not declined yet (no step used). A check of the listings you have seen found fits for "
                                    "some items: " + "; ".join(f"{i} ({w})" for i, w in fits) + ". None is confirmed "
                                    "yet for: " + ", ".join(names[n] for n in sorted(missing)) + "."
                                    + (" Their rows you never opened, cheapest first: " + rows + "." if rows else "")
                                    + " Open those before deciding; decline only if such an item has no fitting "
                                    "listing of its kind."}
                elif fits:
                    obs = {"error": "Not declined (no step used). A check of the listings you have seen found fits: "
                                    + "; ".join(f"{i} ({w})" for i, w in fits)
                                    + ". Open their details if you have not, put them in the basket (the quantities "
                                    "the request asks for) and finish with the order."}
                else:
                    obs = run.env(name, args, tc, content)
            elif (summary and not summary_sent and name in kinds["close"] and not _is_decline(kinds, name, args)
                  and run.left() >= 2):
                obs = {"error": "Not finished yet (no step used): the shopper asked to be told what will arrive. Send "
                                "ONE message now with every basket line (item, quantity, price) and the total against "
                                "the budget, then finish with the order."}
                summary = summary and run.left() >= 2
            elif name == kinds["ask"] and summary and not summary_sent and not allow_ask:
                summary_sent = True
                obs = run.env(name, args, tc, content)
            elif name == kinds["ask"] and summary and summary_sent and not allow_ask:
                obs = {"error": "Message not sent (no step used): the summary was already sent; finish with the order."}
            elif name in kinds["close"] and not held and run.left() >= 4 and run.elapsed() < 230:
                held = True
                twin_qs = _review(messages)
                twins = (" (2) Run these searches now, one after the other: " + " | ".join(twin_qs) + "; read the "
                         "equal-price same-title twins of the basket listings side by side, and keep or swap to the "
                         "twin that has no fine-print note AND shows each required value under every spelling of that "
                         "key the twins use (e.g. both 'x y' and 'x_y').") if twin_qs else ""
                obs = {"error": "This is a check, not a refusal (no step used). Before finishing: (0) view the basket "
                                "and make sure it holds exactly the requested items and quantities, no extra or "
                                "sold-out lines; (1) every line's "
                                "details state each FINAL REQUIREMENT of its item;" + twins + " (3) each line's "
                                "category is the requested kind and it is in stock; (3b) re-read the notices in the "
                                "latest results: the basket total must fit the newest budget; (4) when the store has separate "
                                "page and facts views, every ordered listing has been opened in BOTH views in this "
                                "episode: open the missing ones now. "
                                + ("(5) with the store's price filter, list each item's category capped just below "
                                   "its line's price and read the cheaper rows of the right kind side by side; a "
                                   "cheaper row that meets every FINAL REQUIREMENT replaces the line. "
                                   if cheaper_sweep and run.left() >= n_lines + 8 else "")
                                + "Swap a line only for a listing you "
                                "have read that fits better; then finish with the order. Do not decline because of "
                                "this check."}
            elif name == kinds["ask"] and sent_msgs >= msg_cap:
                obs = {"error": "Message not sent (no step used): the message limit is used up."}
            elif name == kinds["ask"] and not _ASK_FORM.match(str(args.get(kinds["ask_arg"], "")).strip()):
                obs = {"error": "Message not sent (no step used). Send exactly one short question in this form and "
                                "nothing else: 'Do you have a preference on <property> for the <item>?'"}
            else:
                if name == kinds["ask"]:
                    sent_msgs += 1
                if name in browse_names and _over_cap(run, name, args):
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps({"error": _CAP_NOTE})})
                    continue
                read_ids |= _arg_ids(args) if name in browse_names else set()
                obs = run.env(name, args, tc, content)
                if name in browse_names:
                    reread |= _arg_ids(args)
                elif name != kinds["ask"] and name not in kinds["close"]:
                    cart_ids = list(run.held)
                if name in browse_names and _multi_fail(args, obs):
                    run.id_cap[name] = 2
                    pending.append(_MULTI_NOTE.format(n=_multi_fail(args, obs)))
                if name in browse_names:
                    rows = _row_ids(obs)
                    if len(rows) >= 2:
                        unread = [r for r in rows if r not in read_ids]
                        if unread:
                            pending.append("Rows of this result not yet read: " + ", ".join(unread[:8]) + ". Read the "
                                           "cheapest ones of the right category before choosing or searching again.")
                for note in _notes(obs):
                    if note in seen_notes:
                        continue
                    seen_notes.add(note)
                    dp = _dropped(note)
                    if dp and not _drop_valid(dp, context):
                        dp = None
                    if dp and sreqs:
                        sreqs = [r for r in sreqs if not _drop_hits(dp, r)]
                    if dp:
                        pending.append(f"The shopper's note '{note}' is the shopper's own message (this store "
                                       "delivers the shopper's messages beside tool results). It drops the requirement "
                                       f"on {dp}: that requirement no longer applies. Only the stated, looked-up and "
                                       "revealed requirements that remain count (a remark about where or how the item "
                                       "is used is not one), so the item's best listing is now the cheapest in-stock "
                                       "listing of its kind within the budget that meets the remaining requirements: "
                                       "the cheapest one may not be among the rows "
                                       "you have seen: search the item's category for its cheapest in-stock rows (with "
                                       "the store's price filter and a low price cap when it has one), read them, "
                                       "re-check the stock of the carted line, and swap the basket line.")
                if name in browse_names:
                    sp = _small_page(obs)
                    qkey = json.dumps({k: v for k, v in sorted(args.items()) if not _NEXT_KEY.search(k)
                                       and not re.search(r"page|token|notes", k, re.I)}, sort_keys=True)
                    if sp and (qkey in paged_queries or len(paged_queries) >= 3 or run.left() <= n_lines + 10):
                        sp = None
                    if sp:
                        paged_queries.add(qkey)
                        pending.append(f"This result shows only {sp[2]} rows and has a next page ({sp[0]}: "
                                       f"{json.dumps(sp[1])}). A cheaper fit is often on a later page: read the next "
                                       "pages of the search that names the requirement value (pass that token), up to "
                                       "3 pages, checking the right-category rows of each page, before choosing.")
                if name in kinds["close"] and not run.finished:
                    obs = {"store_reply": obs, "episode": "still open: the store did not accept this finish; read its "
                                                         "note, update the requirements and the basket, then finish again"}
                if name in kinds["close"] and not run.finished and not refused_noted:
                    refused_noted = True
                    if _is_decline(kinds, name, args):
                        pending.append("The decline was refused, so the episode is still open. A notice in that reply "
                                       "overrides the request (a new budget replaces the old one; a dropped requirement no "
                                       "longer applies): search again for the items with the updated requirements, and "
                                       "if every item has a fitting listing, put them in the basket and finish with the "
                                       "order.")
                    else:
                        pending.append("The finish was refused. A notice in that reply overrides the request (a new "
                                       "budget replaces the old one; a dropped requirement no longer applies). Re-open "
                                       "every carted listing's details; if a line is sold out, replace it with the "
                                       "cheapest remaining fit you have already read (a fit at the same price counts), "
                                       "then finish again.")
            fresh = [n for n in _sentences(_notes(obs)) if n not in gate_seen]
            if fresh:
                new_notices += fresh
                reread = set()
                gate_seen.update(fresh)
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": json.dumps(_clip(obs))})
        if pending:
            messages.append({"role": "user", "content": " ".join(pending)})
            pending.clear()


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    run = _Run(problem_data)
    policy = run.policy
    request = policy["query"]
    if policy.get("opening") is not None:
        request += "\n\n" + json.dumps(policy["opening"])
    tools_all = policy["tools"]
    kinds = _tool_kinds(tools_all)
    run.cart_names = set(kinds["cart"]) | set(kinds["view"])
    run.state_names = set(kinds["cart"]) | set(kinds["close"])
    run.removes = dict(kinds.get("removes") or {})
    run.read_names = {(t.get("function") or {}).get("name") for t in kinds["browse"]
                      if (t.get("function") or {}).get("name") not in run.cart_names | {kinds["ask"]}
                      and any(k != "notes" for k in ((t.get("function") or {}).get("parameters") or {}).get("properties") or {})}

    plan = _checklist(request)
    m_lim = re.search(r"(\d+)\s+(?:more\s+)?(?:messages?|questions?)\b", request, re.I)
    stated_limit = int(m_lim.group(1)) if m_lim else 0
    if plan is None:
        _buy(run, kinds, tools_all, f"SHOPPER REQUEST:\n{request}", allow_ask=bool(kinds["ask"]), msg_cap=stated_limit or 3)
        return run.dialogue
    items = [i for i in plan["items"] if isinstance(i, dict)]
    merged: list[dict[str, Any]] = []
    for it in items:
        same = next((m for m in merged if _norm(str(m.get("name"))) == _norm(str(it.get("name")))), None)
        if same is None:
            merged.append(it)
        else:
            same["quantity"] = it.get("quantity") or same.get("quantity")
            same["must"] = list(dict.fromkeys(list(same.get("must") or []) + list(it.get("must") or [])))
            same["lookups"] = list(same.get("lookups") or []) + list(it.get("lookups") or [])
            same["hints"] = list(same.get("hints") or []) + list(it.get("hints") or [])
    if len(merged) != len(items):
        items = merged
        plan["items"] = merged
    groups: dict[str, list[dict[str, Any]]] = {}
    category_items: list[str] = []
    for it in items:
        key = "|".join(sorted(_norm(str(m)) for m in it.get("must") or []))
        if key:
            groups.setdefault(key, []).append(it)
    for key, grp in groups.items():
        if len(grp) >= 3 and not any(g.get("lookups") for g in grp):
            first = grp[0]
            words = [str(g.get("name") or "").split() for g in grp]
            lead = words[0][0] if all(w and w[0].lower() == words[0][0].lower() for w in words) and all(len(w) > 1 for w in words) else ""
            parts = [" ".join(w[1:]) if lead else " ".join(w) for w in words]
            first["name"] = ((lead + " ") if lead else "") + ", ".join(parts[:-1]) + " & " + parts[-1]
            category_items.append(str(first["name"]))
            first["quantity"] = max(int(g.get("quantity") or 1) for g in grp)
            items = [i for i in items if i is first or i not in grp]
            plan["items"] = items
    for it in items:
        try:
            q = int(it.get("quantity") or 1)
        except (TypeError, ValueError):
            q = 1
        if q > 1:
            sq = _stated_quantity(request, str(it.get("name") or ""))
            if sq and sq != q:
                it["quantity"] = sq
    for it in items:
        looks = " ".join(re.sub(r"[^a-z0-9]+", " ", str(lk.get("listing") or "").lower())
                         for lk in it.get("lookups") or [] if isinstance(lk, dict))
        if looks:
            it["must"] = [m for m in it.get("must") or []
                          if re.sub(r"[^a-z0-9]+", " ", str(m).lower()).strip() not in f" {looks} "]
    flags = _situation(plan, kinds, tools_all)
    try:
        msg_limit = int(plan.get("messages")) if plan.get("messages") is not None else stated_limit or 3
    except (TypeError, ValueError):
        msg_limit = 3
    if stated_limit:
        msg_limit = min(msg_limit, stated_limit)
    if not kinds["ask"]:
        msg_limit = 0
    confirm = bool(plan.get("confirm")) and msg_limit > 0
    q_limit = msg_limit - (1 if confirm else 0)
    base = (f"SHOPPER REQUEST:\n{request}\n\nCHECKLIST:\n" + "\n".join(f"ITEM {n + 1}: {_item_text(i)}" for n, i in enumerate(items))
            + f"\nBUDGET: {plan.get('budget')}\nEXCLUDE: {plan.get('exclude')}\nOTHER: {plan.get('other')}"
            + "".join(f"\nONE ITEM: '{c}' is a single store category: buy exactly ONE listing of it (any one of these "
                      "kinds), never one listing per kind." for c in category_items))

    reserve_buy = max(10, 3 * len(items) + 4)
    q_steps = min(msg_limit, 6)
    browse_budget = max(0, run.max_steps - reserve_buy - q_steps)
    deadline = 130.0

    lookup_reports: list[str] = []
    not_reqs: list[str] = []
    derived: dict[int, list[str]] = {}
    for n, item in enumerate(items):
        for lk in item.get("lookups") or []:
            if not isinstance(lk, dict) or run.finished:
                continue
            allowance = min(8, browse_budget)
            if allowance < 2:
                break
            task = _lookup_task(n, item.get("name"), lk)
            before = run.step
            rep = _browse(run, kinds, base, task, allowance, deadline, _extra(flags, "browse"))
            browse_budget -= run.step - before
            if ("FINAL VALUE" not in rep.upper() or "NOT FOUND" in rep.upper()) and browse_budget >= 4 and not run.finished:
                before = run.step
                more = _browse(run, kinds, base + f"\n\nREPORT SO FAR:\n{rep}", "Continue this lookup from the report "
                               "so far (do not repeat its reads). " + task, min(5, browse_budget), deadline,
                               _extra(flags, "browse"))
                browse_budget -= run.step - before
                rep = rep + "\n\nCONTINUED:\n" + more
                if (("FINAL VALUE" not in more.upper() or "NOT FOUND" in more.upper()) and "next listing" in more.lower()
                        and browse_budget >= 6 and not run.finished):
                    before = run.step
                    more2 = _browse(run, kinds, base + f"\n\nREPORT SO FAR:\n{rep}", "Continue this lookup from the "
                                    "report so far (do not repeat its reads). " + task, min(5, browse_budget - 2), deadline,
                                    _extra(flags, "browse"))
                    browse_budget -= run.step - before
                    rep = rep + "\n\nCONTINUED:\n" + more2
            lookup_reports.append(f"LOOKUP for item {n + 1} ({lk.get('listing')} -> {lk.get('read')}):\n{rep}")
            dv = _derived_value(rep)
            if dv:
                derived.setdefault(n, []).append(dv)
            not_reqs.append(f"item {n + 1}: {lk.get('listing')} (only identifies the listing to read '{lk.get('read')}' from)")
    look = "\n\n".join(lookup_reports)

    research: list[str] = []
    per_item = max(2, min(4, browse_budget // max(1, len(items))))
    for n, item in enumerate(items):
        if run.finished or browse_budget < 2 or run.elapsed() > deadline:
            break
        task = (f"Item {n + 1}: {_item_text(item)}. Find in-stock listings of this kind that meet its requirements "
                "(and the lookup results above, if any): at most 2 searches of 2-3 words (the item's kind word plus one "
                "distinctive requirement value), then READ the details of the promising rows (side by side when the "
                "store has a tool for that); do not end without reading details. Report: up to 6 CANDIDATES (id, "
                "price, in stock, the facts that matter, which requirements each meets), and KEYS: the keys of the "
                "listings' property/facts table (their datasheet) with the values seen; never record fields such as "
                "title, name, description, seller, shop, maker, price or stock.")
        if derived.get(n):
            dv = derived[n][0]
            if item.get("must"):
                task += (f" The lookup gave this item the value '{dv}': make your FIRST search one stated requirement "
                         f"value of the item plus that value plus the item's kind word (e.g. '<stated value> {dv} "
                         f"{item.get('name')}') and read its rows before any other search.")
            else:
                task += (f" The lookup gave this item the value '{dv}' and the item has no other stated requirement: "
                         f"make your FIRST search that value alone ('{dv}'), read its next pages too (cheaper rows come "
                         "later), and never add words that only identified the looked-up listings.")
        before = run.step
        rep = _browse(run, kinds, base + (f"\n\n{look}" if look else ""), task, min(per_item, browse_budget), deadline,
                      _extra(flags, "browse"))
        browse_budget -= run.step - before
        research.append(f"RESEARCH for item {n + 1} ({item.get('name')}):\n{rep}")
    reports = "\n\n".join(r for r in [look] + research if r)

    talk: list[tuple[str, str]] = []
    queue: list[tuple[str, str]] = []
    asked: set[tuple[str, str]] = set()
    revealed: list[str] = []
    q_notes: set[str] = set()
    dropped: list[str] = []
    planned: set[str] = set()
    reasked: set[str] = set()
    bare_form = False
    answer_checked = False
    asked_list: list[tuple[str, str]] = []
    names = [str(i.get("name") or "") for i in items]
    for nm, it in zip(names, items):
        for m in it.get("must") or []:
            key = str(m).split(":")[0].strip()
            if key:
                asked.add((_norm(nm), _norm_prop(key, names)))
                if len(items) == 1:
                    asked.add((_norm(""), _norm_prop(key, names)))
    hints = "; ".join(f"item {n + 1} ({i.get('name')}): {', '.join(str(h) for h in i.get('hints') or [])}"
                      for n, i in enumerate(items) if i.get("hints"))
    fact_keys = _fact_keys(run.seen, names)
    qctx = base + f"\nSTORY HINTS: {hints or 'none'}\n\n{reports}" + (f"\n\n{fact_keys}" if fact_keys else "")
    for nm, p in _story_props(qctx, request, names):
        if (_norm(nm), _norm_prop(p, names)) not in asked and all(_norm_prop(p, names) != _norm_prop(q, names) for _, q in queue):
            queue.append((nm, p))
    for n, it in enumerate(items):
        kind = _cat_words(str(it.get("name") or ""))
        for lk in it.get("lookups") or []:
            if not isinstance(lk, dict) or not kind or not (kind <= _cat_words(str(lk.get("listing") or ""))):
                continue
            ids = [p for p in _identifiers(str(lk.get("listing") or ""))
                   if _norm_prop(p, names) != _norm_prop(str(lk.get("read") or ""), names)]
            for p in ids[:2]:
                if (_norm(names[n]), _norm_prop(p, names)) not in asked and (names[n], p) not in queue:
                    queue.append((names[n], p))
    while (not run.finished and kinds["ask"] and len(talk) < q_limit and run.left() > reserve_buy - 2
           and run.elapsed() < 170):
        if (revealed and not confirm and len(talk) == q_limit - 1 and q_limit >= 2 and not answer_checked):
            answer_checked = True
            if _asks_question(request):
                confirm = True
                break
        queue = [(i, p) for i, p in queue if (_norm(i), _norm_prop(p, names)) not in asked]
        bare = bare_form
        if queue:
            item_name, prop = queue.pop(0)
            bare = bare or (not item_name and len(items) == 1)
            planned.add(_norm(item_name))
        else:
            pick = None
            for attempt in range(2):
                avoid = ("ALREADY ASKED, do not ask again: " + "; ".join(f"{p} ({i or 'no item'})" for i, p in asked_list)) if attempt else ""
                uncovered = sorted((it for nm, it in zip(names, items) if _norm(nm) not in planned),
                                   key=lambda it: len(it.get("must") or []))
                if len(items) > 1 and talk and uncovered:
                    cover = (f"NEXT QUESTION: it must be about the item '{uncovered[0].get('name')}' (give it as the "
                             "item): a property its OWN listings have (see its research report and KEYS).")
                    avoid = cover + ("\n" + avoid if avoid else "")
                pick = _next_question(qctx, talk, q_limit - len(talk), avoid, _extra(flags, "planner"))
                if pick is None:
                    break
                item_name, prop = pick
                if len(items) == 1:
                    item_name = names[0]
                elif item_name:
                    item_name = next((n for n in names if _norm(n) == _norm(item_name)), None) or next(
                        (n for n in names if _norm(n) and (_norm(n) in _norm(item_name) or _norm(item_name) in _norm(n))),
                        item_name)
                elif uncovered:
                    item_name = str(uncovered[0].get("name") or "")
                if item_name:
                    planned.update(_norm(nm) for nm in names
                                   if _norm(nm) == _norm(item_name) or _norm(nm) in _norm(item_name) or _norm(item_name) in _norm(nm))
                if not prop:
                    pick = None
                    break
                if len(items) > 1 and not item_name:
                    pick = None
                    continue
                if (_norm(item_name), _norm_prop(prop, names)) not in asked:
                    break
                pick = None
            if pick is None:
                break
        if len(items) == 1 and not bare:
            item_name = str(items[0].get("name") or "")
        question = (f"Do you have a preference on {prop} for the {item_name}?" if item_name
                    else f"Do you have a preference on {prop}?")
        obs = run.env(kinds["ask"], {kinds["ask_arg"]: question})
        if isinstance(obs, dict) and isinstance(obs.get("value"), str) and len(obs) == 1:
            reply = obs["value"]
        else:
            reply = json.dumps(obs, ensure_ascii=False)
        talk.append((question, reply))
        asked.add((_norm(item_name), _norm_prop(prop, names)))
        asked_list.append((item_name, prop))
        for note in _notes(obs):
            if note not in q_notes:
                q_notes.add(note)
                dp = _dropped(note)
                if dp and not _drop_valid(dp, base + " " + " ".join(revealed)):
                    dp = None
                if dp:
                    dropped.append(f"{dp} (the shopper's note: '{note}')")
        kind, _about, reading = _reply_kind(question, reply, names)
        if reading:
            revealed.append(reading)
        if kind == "one_at_a_time":
            flags.add("one_at_a_time")
            if len(items) == 1 and item_name and _norm_prop(prop, names) not in reasked:
                reasked.add(_norm_prop(prop, names))
                queue.insert(0, ("", prop))
                bare_form = True
        if not (isinstance(obs, dict) and len(obs) == 1):
            flags.add("notice")

    if run.finished:
        return run.dialogue
    reply_reports: list[str] = []
    convo_txt = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk)
    for lk in _reply_lookups(talk):
        if run.finished or run.left() < reserve_buy + 4 or run.elapsed() > 185:
            break
        rtask = _lookup_task(0, lk.get("item"), lk).replace("For item 1 (", "For the item (")
        rep = _browse(run, kinds, base + f"\n\nQUESTIONS AND SHOPPER REPLIES:\n{convo_txt}", rtask,
                      min(8, run.left() - reserve_buy), 195.0, _extra(flags, "browse"))
        if (("FINAL VALUE" not in rep.upper() or "NOT FOUND" in rep.upper()) and run.left() >= reserve_buy + 4
                and not run.finished and run.elapsed() < 200):
            more = _browse(run, kinds, base + f"\n\nREPORT SO FAR:\n{rep}", "Continue this lookup from the report so "
                           "far (do not repeat its reads). " + rtask, min(6, run.left() - reserve_buy), 215.0,
                           _extra(flags, "browse"))
            rep = rep + "\n\nCONTINUED:\n" + more
        reply_reports.append(f"LOOKUP from a shopper reply ({lk.get('item')}: {lk.get('listing')} -> {lk.get('read')}):\n{rep}")
        not_reqs.append(f"{lk.get('item')}: {lk.get('listing')} (only identifies the listing to read '{lk.get('read')}' from)")
    if reply_reports:
        look = "\n\n".join(([look] if look else []) + reply_reports)
        reports = "\n\n".join(([reports] if reports else []) + reply_reports)
    nr = ("\n\nNOT REQUIREMENTS (seller, brand and properties that only identify a looked-up listing; never list them "
          "as requirements of the item to buy):\n" + "\n".join(not_reqs)) if not_reqs else ""
    rv = ("\n\nREVEALED REQUIREMENTS (exact readings of the shopper's replies):\n" + "\n".join(revealed)) if revealed else ""
    if dropped:
        rv += ("\n\nDROPPED REQUIREMENTS (the shopper's notes let go of these; they no longer apply and any value "
               "fits):\n" + "\n".join(dropped))
    req_list, qs = _queries(base + (f"\n\n{look}" if look else "") + nr + rv, talk, _extra(flags, "summary"),
                            )
    if look and req_list:
        convo_q = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk)
        req_list = _clean_requirements(base + f"\n\n{look}\n\nQUESTIONS AND SHOPPER REPLIES:\n{convo_q}" + nr + rv,
                                       req_list)
    convo = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk) or "(no questions were asked)"
    picks: list[str] = []
    if len(items) > 1 and req_list and not run.finished:
        spare = run.left() - (2 * len(items) + 6)
        per = min(5, spare // len(items)) if spare > 0 else 0
        for n, it in enumerate(items):
            if per < 2 or run.elapsed() > 150 or run.finished:
                break
            task = (f"Item {n + 1} only: {it.get('name')} x {it.get('quantity') or 1}. Its requirements are the "
                    f"'{it.get('name')}' entries of these FINAL REQUIREMENTS: {req_list}. Find the cheapest in-stock "
                    "listing of this kind within the budget that meets every one of them: search with its requirement "
                    "values, read the details of the cheapest rows of the right kind side by side, and read the next "
                    "page of the best search. Report one line 'CHOSEN: <id> (<price>)' with the details that meet each "
                    "requirement, or 'CHOSEN: none' with the closest listing and what it lacks.")
            rep_n = _browse(run, kinds, base + f"\n\n{reports}\n\nQUESTIONS AND SHOPPER REPLIES:\n{convo}" + rv + nr,
                            task, per, 170.0, _extra(flags, "browse"))
            picks.append(f"ITEM {n + 1} ({it.get('name')}): {rep_n}")
    context = (base + f"\n\n{reports}\n\nQUESTIONS AND SHOPPER REPLIES:\n{convo}" + rv + nr
               + ("\n\nPER-ITEM SELECTION (one focused search per item; use these choices unless a check shows they "
                  "fail):\n" + "\n\n".join(picks) if picks else "")
               + (f"\n\nFINAL REQUIREMENTS (from the request, the lookups and the replies): {req_list}" if req_list else "")
               + (f"\nSUGGESTED SEARCHES: {' | '.join(qs)}\nYour FIRST call is the search '{qs[0]}', exactly as written; "
                  "then read its cheapest rows of the right kind." if qs else "")
               + f"\n\nSteps left: {run.left()}. Now buy: confirm the best fitting listing of every item (search more "
                 "if a revealed requirement is not met by the reported candidates), put them in the basket and finish "
                 "with the order.")
    names = [str(it.get("name") or "") for it in items]
    qtys = [int(it.get("quantity") or 1) if str(it.get("quantity") or "1").isdigit() else 1 for it in items]
    sreqs = _struct_reqs(req_list, names) if req_list and not run.finished else []
    fallback = str(plan.get("if_none_fits") or "").strip()
    if fallback.lower() in ("null", "none", "no", "n/a"):
        fallback = ""
    _buy(run, kinds, tools_all, context, allow_ask=False, summary=confirm, n_lines=len(items), extra=_extra(flags, "buy"),
         cheaper_sweep="price_filter" in flags,
         seen=q_notes, sreqs=sreqs, names=names, qtys=qtys, fallback=fallback)
    return run.dialogue


__all__ = ["agent_main"]
