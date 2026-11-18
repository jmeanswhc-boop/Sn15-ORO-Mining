from __future__ import annotations

import json
import math
import re
from typing import Any

from src.agent.proxy_client import ProxyClient

FAST = "anthropic/claude-haiku-5.5"
SPARE = "openai/gpt-6-luna"
CHEAP = "deepseek/deepseek-v4-flash-0731"
SPEND_STOP = 0.05
CALL_CAP = 40
OBS_CHARS = 6000

_env = ProxyClient(timeout=60, max_retries=3, retry_delay=1)
_llm = ProxyClient(timeout=60, max_retries=1, retry_delay=1)


def words(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text or "").lower())


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        w = w[:-3] + "y"
    elif len(w) > 4 and w.endswith("es") and w[-3] in "sxz":
        w = w[:-2]
    elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 5 and w.endswith("our"):
        w = w[:-3] + "or"
    if len(w) > 6 and re.search(r"is(e|er|ed|ing|ation)$", w):
        w = re.sub(r"is(e|er|ed|ing|ation)$", r"iz\1", w)
    return w


def roots(text: Any) -> set[str]:
    return {stem(w) for w in words(text) if len(w) > 1}


def keyname(k: Any) -> str:
    return " ".join(words(str(k).replace("_", " ").replace("-", " ")))


def num(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        for x in v.values():
            n = num(x)
            if n is not None:
                return n
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", str(v or "").replace(",", ""))
    return float(m.group()) if m else None


def jdump(o: Any, limit: int = OBS_CHARS) -> str:
    s = json.dumps(o, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + "…"


def parse_json(text: str) -> Any:
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else "", t[t.find("["): t.rfind("]") + 1] if "[" in t else ""):
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            try:
                return json.loads(re.sub(r",\s*([}\]])", r"\1", cand))
            except Exception:
                continue
    return None


class Wire:
    def __init__(self, env: dict):
        self.binding = dict(env.get("binding") or {})
        self.step = 0
        self.finished = bool((env.get("policy_view") or {}).get("finished"))
        self.log: list[dict] = []
        self.dialogue: list[dict] = []

    def call(self, name: str, args: dict) -> Any:
        if self.finished:
            return {"error": "finished"}
        step = self.step + 1
        body = {**self.binding, "step": step, "action": {"name": name, "args": args}}
        res = None
        for _ in range(3):
            res = _env.post("/environment/call", json_data=body)
            if res is not None:
                break
        if res is None:
            self.step = step
            obs: Any = {"error": "no response"}
        else:
            self.step = int(res.get("step") or step)
            self.finished = bool(res.get("finished"))
            obs = res.get("observation")
        self.log.append({"step": step, "tool": name, "args": args, "obs": obs})
        self.dialogue.append({"role": "assistant", "tool_call": {"name": name, "arguments": args}, "environment_result": res})
        return obs


class Mind:
    def __init__(self) -> None:
        self.spent = 0.0
        self.calls = 0
        self.notes: list[dict] = []

    def ok(self) -> bool:
        return self.spent < SPEND_STOP and self.calls < CALL_CAP

    def ask(self, system: str, user: str, max_tokens: int = 700, models: tuple = (FAST, SPARE), think: bool = False) -> Any:
        for model in models:
            if not self.ok():
                return None
            self.calls += 1
            room = max(500, max_tokens)
            text = ""
            for _ in range(2):
                body = {"model": model, "temperature": 0, "max_tokens": room,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
                if not think:
                    body["reasoning"] = {"effort": "low"} if model.startswith("anthropic/") else {"enabled": False}
                data = _llm.post("/inference/chat/completions", json_data=body)
                if not data:
                    break
                usage = data.get("usage") or {}
                if isinstance(usage.get("cost"), (int, float)):
                    self.spent += float(usage["cost"])
                try:
                    choice = data["choices"][0]
                    text = choice["message"].get("content") or ""
                except Exception:
                    text = ""
                    break
                if choice.get("finish_reason") != "length" or not self.ok():
                    break
                room *= 2
            if not text:
                continue
            out = parse_json(text)
            if out is not None:
                self.notes.append({"model": model, "chars": len(user)})
                return out
        return None


ROLE_HINTS = {
    "message": r"shopper|customer|buyer|client|user|message|ask|chat|write|talk|say|note",
    "order": r"check ?out|purchase|buy|order|settle|place|commit|final",
    "decline": r"without (?:a )?(?:buy|purchas)|no purchase|declin|give ?up|abstain|refus|pass ?_?(?:on|order)|close out|stand ?_?down|nothing|end\w*\W+(?:\w+\W+){0,4}without",
    "view": r"cart|basket|bag|trolley|selection|contents|review|show|\bmy\b|\bmine\b",
    "status": r"status|usage|remaining|left|report|session|budget",
    "detail": r"detail|propert|attribute|spec|record|page|open|describ|inspect|examin|datasheet|sheet|particular|trait|fact|view|read",
    "search": r"search|find|look|locat|query|seek|scan|browse|discover|list(?!ing)|filter|hunt|catalog|sift|screen",
}
ID_ARG = r"(?:^|_)(id|ids|ref|sku|code|item|product|listing|target|which|offer|article)s?(?:_|$)|id$|Id$|ref$|Ref$"
QTY_ARG = r"qty|quant|units|count|number|how_?many|amount|num|piece"
PAGE_ARG = r"page|offset|start|skip|from|cursor|token|after|continue|resume|begin|row|index|next"
TEXT_ARG = r"query|text|terms|phrase|keyword|words|(?:^|_)q(?:_|$)|^q[A-Z]|search|find|free|needle|wanted|input"
SHELF_ARG = r"categor|department|group|section|shelf|aisle|path|collection|family|range|class|label"
MAKER_ARG = r"brand|maker|manufactur|make|vendor|marque"
CAP_ARG = r"max|cap|ceiling|limit|upto|under|budget|price"
ADD_WORDS = r"add|include|append|put|hold|insert|plus|increase|set"
REMOVE_WORDS = r"remove|drop|delete|take|release|minus|decrease|clear|out|discard|exclude|cancel|subtract|reduce|withdraw|eject|unset|rid|pull|less"
ORDER_WORDS = r"order|buy|purchase|commit|checkout|confirm|place|accept|yes"
DECLINE_WORDS = r"declin|pass|none|abstain|skip|refus|no|cancel|give"

BIND_SYS = """You are wiring a shopping agent to an online store whose tool names, argument names, option values and
descriptions are new each time. Read every tool (name, description, JSON schema) and say which tool does each job and
which argument carries what. Judge by meaning and by the schema (types, enums, arrays of objects), not by familiar words.
Jobs:
- search_text: find listings by words. search_shelf: list a category/department/group (maybe with a brand and a price
  ceiling). One tool may do both, chosen by an option argument: then put that argument and the option value in "fixed".
- detail: open one listing by its id (facts/properties table = "facts"; page with text/seller/price = "page"; both = "all").
- compare: several listings' details at once (an array of ids).
- message: send the shopper a message or question.
- cart_add / cart_remove: change the cart one listing at a time (id, quantity, and an action option if there is one:
  give the option value that adds, and the one that removes, in "fixed").
- cart_set: replace the cart with a list of line objects; give the argument and the field names inside one line.
- cart_view: show the cart. status: show remaining steps/messages/budget.
- order: buy the cart and end. decline: end without buying (its free-text reason argument, if any). One finishing tool with
  an option may do both: give each option value in "fixed".
Return ONE JSON object only:
{"search_text":{"tool":name,"text":arg,"page":arg|null,"page_kind":"offset"|"page"|"token"|null,"fixed":{arg:value}}|null,
 "search_shelf":{"tool":name,"shelf":arg,"maker":arg|null,"cap":arg|null,"page":arg|null,"page_kind":...,"fixed":{arg:value}}|null,
 "detail":[{"tool":name,"id":arg,"gives":"facts"|"page"|"all"}],
 "compare":{"tool":name,"ids":arg,"max":int}|null,
 "message":{"tool":name,"text":arg}|null,
 "cart_add":{"tool":name,"id":arg,"qty":arg|null,"fixed":{arg:value}}|null,
 "cart_remove":{"tool":name,"id":arg,"qty":arg|null,"fixed":{arg:value}}|null,
 "cart_set":{"tool":name,"lines":arg,"line_id":field,"line_qty":field}|null,
 "cart_view":{"tool":name}|null,
 "order":{"tool":name,"fixed":{arg:value}},
 "decline":{"tool":name,"reason":arg|null,"fixed":{arg:value}},
 "status":{"tool":name}|null}
Use only tool names, argument names and option values that appear in the schemas. "page_kind": offset = a row number to
start from, page = a page number, token = a value the previous result returns."""


class Contract:
    def __init__(self, tools: list[dict], mind: Mind):
        self.tools: dict[str, dict] = {}
        for t in tools if isinstance(tools, list) else []:
            if not isinstance(t, dict):
                continue
            f = t.get("function") if isinstance(t.get("function"), dict) else t
            name = f.get("name")
            if name:
                self.tools[name] = {"name": name, "desc": str(f.get("description") or ""),
                                    "props": dict(((f.get("parameters") or {}).get("properties") or {})),
                                    "req": list(((f.get("parameters") or {}).get("required") or []))}
        self.mind = mind
        self.weak: set[str] = set()
        self.roles: dict[str, Any] = self.heuristic()
        self.refine()
        st, ss = self.roles.get("search_text"), self.roles.get("search_shelf")
        if isinstance(st, dict) and isinstance(ss, dict) and st.get("tool") == ss.get("tool") and ss.get("shelf") \
                and ss["shelf"] in (self.tools.get(st["tool"]) or {}).get("props", {}) and not st.get("shelf"):
            st["shelf"] = ss["shelf"]

    def bound(self) -> set[str]:
        out = set()
        for v in self.roles.values():
            for x in (v if isinstance(v, list) else [v]):
                if isinstance(x, dict) and x.get("tool"):
                    out.add(x["tool"])
        return out

    def extras(self) -> list[str]:
        b = self.bound()
        return [n for n in self.tools if n not in b or n in self.weak]

    def unbind(self, name: str) -> None:
        for r, v in list(self.roles.items()):
            if isinstance(v, list):
                self.roles[r] = [x for x in v if not (isinstance(x, dict) and x.get("tool") == name)]
            elif isinstance(v, dict) and v.get("tool") == name and r in ("detail", "compare", "cart_view", "status", "message"):
                self.roles[r] = None
        self.weak.discard(name)

    def text_of(self, t: dict) -> str:
        return (t["name"].replace("_", " ") + " " + re.sub(r"([a-z])([A-Z])", r"\1 \2", t["name"]) + " " + t["desc"]).lower()

    def args_matching(self, t: dict, pat: str, types: tuple = (), free: bool = False) -> list[str]:
        out = []
        for a, s in t["props"].items():
            if free and s.get("enum"):
                continue
            ty = s.get("type")
            tys = set(ty) if isinstance(ty, list) else {ty}
            if types and not (tys & set(types)):
                continue
            snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", a).lower()
            if re.search(pat, a, re.I) or re.search(pat, snake, re.I) or re.search(pat, str(s.get("description") or ""), re.I):
                out.append(a)
        return out

    def enum_arg(self, t: dict, pat: str) -> tuple[str, str] | None:
        for a, s in t["props"].items():
            for v in s.get("enum") or []:
                if re.search(pat, str(v), re.I):
                    return a, v
        return None

    def heuristic(self) -> dict:
        R: dict[str, Any] = {"detail": []}
        score = {}
        for n, t in self.tools.items():
            txt = self.text_of(t)
            score[n] = {r: len(re.findall(p, txt)) for r, p in ROLE_HINTS.items()}
        for n, t in self.tools.items():
            props = t["props"]
            nprops = len(props)
            strs = [a for a, s in props.items() if "string" in (s.get("type") if isinstance(s.get("type"), list) else [s.get("type")])]
            arrs = [a for a, s in props.items() if s.get("type") == "array"]
            sc = score[n]
            obj_arr = [a for a in arrs if (props[a].get("items") or {}).get("type") == "object"]
            if obj_arr:
                item = props[obj_arr[0]]["items"].get("properties") or {}
                lid = next((k for k in item if re.search(ID_ARG, k, re.I)), None) or next((k for k, s in item.items() if s.get("type") == "string"), None)
                lq = next((k for k in item if re.search(QTY_ARG, k, re.I)), None) or next((k for k, s in item.items() if s.get("type") == "integer"), None)
                R["cart_set"] = {"tool": n, "lines": obj_arr[0], "line_id": lid, "line_qty": lq}
                continue
            if arrs and not obj_arr:
                R.setdefault("compare", {"tool": n, "ids": arrs[0], "max": int(props[arrs[0]].get("maxItems") or 4)})
                continue
            if nprops == 0:
                if sc["view"] and not (sc["order"] > sc["view"]):
                    R.setdefault("cart_view", {"tool": n})
                elif sc["decline"] > 0 and sc["decline"] >= sc["order"]:
                    R.setdefault("decline", {"tool": n, "reason": None, "fixed": {}})
                elif sc["order"]:
                    R.setdefault("order", {"tool": n, "fixed": {}})
                elif sc["status"]:
                    R.setdefault("status", {"tool": n})
                continue
            eo, ed = self.enum_arg(t, ORDER_WORDS), self.enum_arg(t, DECLINE_WORDS)
            if eo and ed and eo[0] == ed[0] and eo[1] != ed[1] and not self.args_matching(t, ID_ARG):
                reason = next((a for a in strs if a != eo[0]), None)
                R["order"] = {"tool": n, "fixed": {eo[0]: eo[1]}}
                R["decline"] = {"tool": n, "reason": reason, "fixed": {ed[0]: ed[1]}}
                continue
            ids = self.args_matching(t, ID_ARG, ("string", "integer"))
            if sc["decline"] and sc["decline"] > sc["search"] and not ids and len(strs) == nprops and nprops <= 2 and re.search(ROLE_HINTS["decline"], t["desc"].lower() + " " + t["name"].lower()):
                R.setdefault("decline", {"tool": n, "reason": strs[0] if strs else None, "fixed": {}})
                continue
            qtys = self.args_matching(t, QTY_ARG, ("integer", "number"))
            ea, er = self.enum_arg(t, ADD_WORDS), self.enum_arg(t, REMOVE_WORDS)
            for a, sp in t["props"].items():
                ev = [str(v) for v in sp.get("enum") or []]
                if len(ev) == 2 and ids and a not in ids:
                    if ea and not er and ea[0] == a:
                        er = (a, next(v for v in ev if v != ea[1]))
                    elif er and not ea and er[0] == a:
                        ea = (a, next(v for v in ev if v != er[1]))
                    elif not ea and not er and (qtys or re.search(r"cart|basket|bag|trolley|selection", self.text_of(t), re.I)):
                        ea, er = (a, ev[0]), (a, ev[1])
            cartish = bool(re.search(r"cart|basket|bag|trolley|selection|hold|stash|keep|add|put|pick", self.text_of(t)))
            if ids and (qtys or ea or er or (cartish and sc["view"] and sc["detail"] <= sc["view"])) and not sc["search"] > 1:
                if ea and er and ea[0] == er[0]:
                    R["cart_add"] = {"tool": n, "id": ids[0], "qty": qtys[0] if qtys else None, "fixed": {ea[0]: ea[1]}}
                    R["cart_remove"] = {"tool": n, "id": ids[0], "qty": qtys[0] if qtys else None, "fixed": {er[0]: er[1]}}
                elif re.search(REMOVE_WORDS, t["name"], re.I):
                    R.setdefault("cart_remove", {"tool": n, "id": ids[0], "qty": qtys[0] if qtys else None, "fixed": {}})
                else:
                    R.setdefault("cart_add", {"tool": n, "id": ids[0], "qty": qtys[0] if qtys else None, "fixed": {}})
                continue
            listingish = sc["detail"] > 0 or bool(re.search(r"listing|item|product|offer|article|sku|goods|page|catalog", self.text_of(t)))
            if len(strs) == 1 and nprops == 1 and (ids or sc["detail"] > sc["message"]) and not sc["message"] > sc["detail"] and listingish:
                gives = "facts" if re.search(r"propert|attribute|spec|datasheet|particular|trait|fact|technical", self.text_of(t)) else "all"
                if re.search(r"page|text|seller|description|open", t["desc"], re.I) and gives == "facts" and re.search(r"page", t["desc"], re.I):
                    gives = "page"
                R["detail"].append({"tool": n, "id": strs[0], "gives": gives})
                continue
            if len(strs) == 1 and nprops == 1:
                R.setdefault("message", {"tool": n, "text": strs[0]})
                continue
            if len(strs) <= 2 and nprops <= 2 and sc["decline"] and not sc["search"]:
                R.setdefault("decline", {"tool": n, "reason": strs[0] if strs else None, "fixed": {}})
                continue
            pages = [a for a, s in props.items() if re.search(PAGE_ARG, a, re.I)]
            page = pages[0] if pages else None
            pk = None
            if page:
                ty = props[page].get("type")
                tys = set(ty) if isinstance(ty, list) else {ty}
                if "string" in tys or re.search(r"token|cursor|continue|resume|after", page, re.I):
                    pk = "token"
                elif re.search(r"page|sheet|pg", page, re.I) and not re.search(r"row|offset|start|skip|from|begin", page, re.I):
                    pk = "page"
                else:
                    pk = "offset"
            text = next((a for a in self.args_matching(t, TEXT_ARG, ("string",), True) if a != page), None)
            shelf = next((a for a in self.args_matching(t, SHELF_ARG, ("string",), True) if a not in (text, page)), None)
            maker = next((a for a in self.args_matching(t, MAKER_ARG, ("string",), True) if a not in (text, shelf, page)), None)
            cap = next((a for a in self.args_matching(t, CAP_ARG, ("number", "integer")) if a != page), None)
            if page is None:
                for a, sp in props.items():
                    if a in (text, shelf, maker, cap) or sp.get("enum"):
                        continue
                    ty = sp.get("type"); tys = set(ty) if isinstance(ty, list) else {ty}
                    if tys & {"integer", "number"}:
                        page, pk = a, ("page" if sp.get("minimum") == 1 else "offset")
                        break
                    if "string" in tys and text and shelf:
                        page, pk = a, "token"
                        break
            mode = None
            for a, s in props.items():
                if s.get("enum") and a not in (shelf, text):
                    mode = a
            fixed_t, fixed_s = {}, {}
            if mode:
                en = props[mode]["enum"]
                vt = next((v for v in en if re.search(r"type|word|text|key|free|search|query|term", str(v), re.I)), en[0])
                vs = next((v for v in en if v != vt), en[-1])
                fixed_t, fixed_s = {mode: vt}, {mode: vs}
            if text:
                R.setdefault("search_text", {"tool": n, "text": text, "page": page, "page_kind": pk, "fixed": fixed_t})
            if shelf:
                if "search_shelf" not in R or text is None:
                    R["search_shelf"] = {"tool": n, "shelf": shelf, "maker": maker, "cap": cap, "page": page, "page_kind": pk, "fixed": fixed_s}
        return R

    def valid(self, R: dict) -> bool:
        return bool(R.get("order") and R.get("decline") and (R.get("search_text") or R.get("search_shelf"))
                    and (R.get("cart_add") or R.get("cart_set")))

    def complete(self, R: dict) -> bool:
        return self.valid(R) and bool(R.get("message")) and bool(R.get("detail"))

    def refine(self) -> None:
        listing = [{"name": t["name"], "description": t["desc"][:400], "parameters": t["props"], "required": t["req"]} for t in self.tools.values()]
        got = self.mind.ask(BIND_SYS, "TOOLS:\n" + jdump(listing, 12000), 2500, think=True)
        if not isinstance(got, dict):
            return
        merged = dict(self.roles)
        self.model_roles = set()
        for role in ("search_text", "search_shelf", "detail", "compare", "message", "cart_add", "cart_remove", "cart_set",
                     "cart_view", "order", "decline", "status"):
            spec = got.get(role)
            if role == "detail":
                good = [d for d in (spec if isinstance(spec, list) else [spec] if isinstance(spec, dict) else [])
                        if isinstance(d, dict) and self.exists(d.get("tool"), [d.get("id")])]
                if good:
                    mine = [dict(d, gives=d.get("gives") if d.get("gives") in ("facts", "page", "all") else "all") for d in good]
                    names = {d["tool"] for d in mine}
                    merged["detail"] = mine + [d for d in (self.roles.get("detail") or []) if d.get("tool") not in names]
                    merged["detail"].sort(key=lambda d: {"facts": 0, "all": 1, "page": 2}.get(d.get("gives"), 1))
                    self.model_roles.add(role)
                continue
            if isinstance(spec, dict) and self.checks(role, spec):
                prior = self.roles.get(role)
                if isinstance(prior, dict) and prior.get("tool") == spec.get("tool"):
                    for k, v in prior.items():
                        if spec.get(k) in (None, "", {}) and v not in (None, "", {}):
                            spec[k] = v
                merged[role] = spec
                self.model_roles.add(role)
        groups = {"search": ("search_text", "search_shelf"), "detail": ("detail",), "compare": ("compare",), "message": ("message",),
                  "cart": ("cart_add", "cart_remove", "cart_set"), "view": ("cart_view",), "finish": ("order", "decline"), "status": ("status",)}
        def tools_of(R: dict, roles: tuple) -> set:
            out = set()
            for r in roles:
                v = R.get(r)
                for x in (v if isinstance(v, list) else [v]):
                    if isinstance(x, dict) and x.get("tool"):
                        out.add(x["tool"])
            return out
        for g, roles in groups.items():
            mine = tools_of(merged, roles)
            others = set().union(*[tools_of(merged, rr) for gg, rr in groups.items() if gg != g])
            if mine & others:
                for r in roles:
                    if r in self.model_roles:
                        merged[r] = self.roles.get(r)
                        self.model_roles.discard(r)
        if self.valid(merged) or not self.valid(self.roles):
            self.roles = merged
        def tools_named(v: Any) -> set:
            return {x["tool"] for x in (v if isinstance(v, list) else [v]) if isinstance(x, dict) and isinstance(x.get("tool"), str)}
        named = set().union(*[tools_named(v) for v in got.values()]) if got else set()
        for r in ("detail", "compare", "cart_view", "status", "message"):
            mine = tools_named(got.get(r)) & set(self.tools)
            if not mine:
                continue
            for x in (self.roles.get(r) if isinstance(self.roles.get(r), list) else [self.roles.get(r)]):
                if isinstance(x, dict) and x.get("tool") in self.tools and x["tool"] not in mine and x["tool"] not in named:
                    self.weak.add(x["tool"])

    def checks(self, role: str, spec: dict) -> bool:
        t = self.tools.get(spec.get("tool"))
        if not t:
            return False
        names = self.arg_names(role, spec)
        if role == "cart_set":
            items = (t["props"].get(spec.get("lines")) or {}).get("items") or {}
            inner = items.get("properties") or {}
            if spec.get("lines") not in t["props"] or (inner and (spec.get("line_id") not in inner or (spec.get("line_qty") and spec.get("line_qty") not in inner))):
                return False
            names = [spec.get("lines")]
        if any(a is not None and a not in t["props"] for a in names):
            return False
        fixed = spec.get("fixed") or {}
        if not isinstance(fixed, dict):
            return False
        for a, v in fixed.items():
            sp = t["props"].get(a)
            if sp is None or (sp.get("enum") and v not in sp["enum"]):
                return False
        if role in ("search_text",) and not spec.get("text"):
            return False
        if role in ("cart_add", "cart_remove") and not spec.get("id"):
            return False
        return True

    @staticmethod
    def arg_names(role: str, spec: dict) -> list:
        keys = {"search_text": ["text", "page"], "search_shelf": ["shelf", "maker", "cap", "page"], "compare": ["ids"], "message": ["text"],
                "cart_add": ["id", "qty"], "cart_remove": ["id", "qty"], "cart_set": ["lines"], "decline": ["reason"]}.get(role, [])
        out = [spec.get(k) for k in keys if spec.get(k)]
        out += list((spec.get("fixed") or {}).keys())
        return out

    def exists(self, tool: Any, args: list) -> bool:
        t = self.tools.get(tool)
        if not t:
            return False
        for a in args:
            if a and a not in t["props"]:
                return False
        return True

    def page_value(self, spec: dict, index: int, token: Any, size: int) -> Any:
        kind = spec.get("page_kind")
        if kind == "page":
            return index + 1
        if kind == "token":
            return token
        return index * size if token is None else token

    @staticmethod
    def fit_value(v: Any, sp: dict) -> Any:
        if not isinstance(sp, dict):
            return v
        ty = sp.get("type")
        tys = set(ty) if isinstance(ty, list) else {ty} if ty else set()
        if sp.get("enum"):
            en = sp["enum"]
            if v in en:
                return v
            low = {str(x).lower(): x for x in en}
            if str(v).lower() in low:
                return low[str(v).lower()]
            near = [x for x in en if str(v).lower() in str(x).lower() or str(x).lower() in str(v).lower()]
            return near[0] if len(near) == 1 else v
        if "array" in tys:
            items = sp.get("items") or {}
            vs = v if isinstance(v, list) else [v]
            vs = [Contract.fit_value(x, items) for x in vs]
            if sp.get("maxItems"):
                vs = vs[:int(sp["maxItems"])]
            return vs
        if "object" in tys and isinstance(v, dict):
            props = sp.get("properties") or {}
            out = {k: Contract.fit_value(x, props.get(k, {})) for k, x in v.items() if not props or k in props or sp.get("additionalProperties") is not False}
            return out
        if tys & {"integer"} and not isinstance(v, bool):
            n = num(v)
            if n is not None:
                n = int(round(n))
                if sp.get("minimum") is not None:
                    n = max(n, int(sp["minimum"]))
                if sp.get("maximum") is not None:
                    n = min(n, int(sp["maximum"]))
                return n
        if tys & {"number"} and not isinstance(v, bool):
            n = num(v)
            if n is not None:
                return n
        if tys & {"string"} and not isinstance(v, str) and v is not None and not isinstance(v, (dict, list)):
            v = str(v)
        if isinstance(v, str) and sp.get("maxLength"):
            v = v[:int(sp["maxLength"])]
        return v

    def conform(self, t: dict, args: dict) -> dict:
        props = t.get("props") or {}
        out = {}
        for k, v in args.items():
            if k not in props and props:
                continue
            out[k] = self.fit_value(v, props.get(k) or {})
        return out

    def learn_fix(self, role: str, sent: dict, fixed: dict) -> None:
        spec = self.roles.get(role)
        spec = spec[0] if isinstance(spec, list) and spec else spec
        if not isinstance(spec, dict):
            return
        ren = {}
        for k, v in sent.items():
            if k not in fixed:
                k2 = next((x for x, y in fixed.items() if x not in sent and y == v), None)
                if k2:
                    ren[k] = k2
        for k, v in fixed.items():
            if k not in sent and k not in ren.values() and isinstance(v, (str, int, float, bool)):
                spec.setdefault("fixed", {})[k] = v
        if ren:
            spec["rename"] = {**(spec.get("rename") or {}), **ren}

    def build(self, role: str, **kw: Any) -> tuple[str, dict] | None:
        spec = self.roles.get(role)
        if not spec:
            return None
        if role == "detail":
            spec = spec[0] if isinstance(spec, list) and spec else None
            if not spec:
                return None
        args = dict(spec.get("fixed") or {})
        t = self.tools[spec["tool"]]
        if role == "search_text":
            args[spec["text"]] = kw["text"][:int(t["props"].get(spec["text"], {}).get("maxLength") or 200)]
            if kw.get("shelf") and spec.get("shelf") and spec["shelf"] in t["props"]:
                args[spec["shelf"]] = kw["shelf"]
        elif role == "search_shelf":
            args[spec["shelf"]] = kw["shelf"]
            if kw.get("cap") is not None and spec.get("cap"):
                args[spec["cap"]] = kw["cap"]
            if kw.get("maker") and spec.get("maker"):
                args[spec["maker"]] = kw["maker"]
        elif role in ("detail", "detail_alt"):
            args[spec["id"]] = kw["id"]
        elif role == "compare":
            args[spec["ids"]] = list(kw["ids"])
        elif role == "message":
            args[spec["text"]] = kw["text"][:int(t["props"].get(spec["text"], {}).get("maxLength") or 400)]
        elif role in ("cart_add", "cart_remove"):
            args[spec["id"]] = kw["id"]
            if spec.get("qty") and kw.get("qty"):
                args[spec["qty"]] = int(kw["qty"])
        elif role == "cart_set":
            args[spec["lines"]] = [{spec["line_id"]: i, spec["line_qty"]: int(q)} for i, q in kw["lines"]]
        elif role == "decline":
            if spec.get("reason") and kw.get("reason") is not None:
                args[spec["reason"]] = kw["reason"][:int(t["props"].get(spec["reason"], {}).get("maxLength") or 400)]
        if role in ("search_text", "search_shelf") and kw.get("page") is not None and spec.get("page"):
            args[spec["page"]] = kw["page"]
        for k, k2 in (spec.get("rename") or {}).items():
            if k in args:
                args[k2] = args.pop(k)
        args = self.conform(t, args)
        words_for = {"cart_add": ADD_WORDS, "cart_remove": REMOVE_WORDS, "order": ORDER_WORDS, "decline": DECLINE_WORDS}
        for a in t.get("req") or []:
            if a in args:
                continue
            sp = t["props"].get(a) or {}
            ev = sp.get("enum") or []
            if ev:
                pat = words_for.get(role)
                args[a] = next((v for v in ev if pat and re.search(pat, str(v), re.I)), ev[0])
            elif sp.get("type") in ("integer", "number"):
                args[a] = max(1, int(sp.get("minimum") or 1))
        return spec["tool"], args


class Shapes:
    def __init__(self) -> None:
        self.id_key: str | None = None
        self.keys: dict[str, str | None] = {}

    @staticmethod
    def dict_lists(o: Any, path: tuple = ()) -> list[tuple[tuple, list]]:
        out = []
        if isinstance(o, list) and o and all(isinstance(x, dict) for x in o):
            out.append((path, o))
            for n, x in enumerate(o[:1]):
                out += Shapes.dict_lists(x, path + (n,))
        elif isinstance(o, dict):
            for k, v in o.items():
                out += Shapes.dict_lists(v, path + (k,))
        return out

    @staticmethod
    def idlike(v: Any) -> bool:
        return isinstance(v, str) and 3 <= len(v) <= 64 and " " not in v.strip() and bool(re.search(r"\d", v))

    def learn_rows(self, rows: list[dict]) -> None:
        if len(rows) < 1:
            return
        keys = set().union(*[set(r) for r in rows])
        best, best_s = None, -1.0
        for k in keys:
            vals = [r.get(k) for r in rows]
            if not all(self.idlike(v) for v in vals):
                continue
            uniq = len(set(vals)) == len(vals)
            s = (2 if uniq else 0) + (1.5 if re.search(r"id|ref|sku|code|pid", k, re.I) else 0) - 0.01 * max(len(v) for v in vals)
            if s > best_s:
                best, best_s = k, s
        if best and (self.id_key is None or best_s >= 2):
            self.id_key = best
        self.learn_fields(rows)

    def learn_fields(self, rows: list[dict]) -> None:
        keys = set().union(*[set(r) for r in rows]) - {self.id_key}
        def vals(k): return [r.get(k) for r in rows if r.get(k) is not None]
        price = [k for k in keys if vals(k) and all(num(v) is not None and not isinstance(v, (bool, str)) for v in vals(k))]
        price.sort(key=lambda k: (not re.search(r"price|cost|amount|cents", k, re.I), k))
        if price and "price" not in self.keys:
            self.keys["price"] = price[0]
        stock = [k for k in keys if vals(k) and all(isinstance(v, bool) for v in vals(k))]
        if stock:
            self.keys.setdefault("stock", stock[0])
        strs = [k for k in keys if vals(k) and all(isinstance(v, str) for v in vals(k))]
        if strs:
            avg = {k: sum(len(v) for v in vals(k)) / len(vals(k)) for k in strs}
            distinct = {k: len(set(vals(k))) for k in strs}
            titles = [k for k in strs if avg[k] >= 12 and distinct[k] >= max(1, len(rows) * 0.8)]
            if titles:
                self.keys.setdefault("title", min(titles, key=lambda k: (not re.search(r"title|name|head|label", k, re.I), abs(avg[k] - 60))))
            longs = [k for k in strs if k != self.keys.get("title") and avg[k] >= 40]
            if longs:
                self.keys.setdefault("text", max(longs, key=lambda k: avg[k]))
            shelf = [k for k in strs if re.search(r"categ|depart|group|shelf|section|aisle|path", k, re.I)]
            if shelf:
                self.keys.setdefault("shelf", shelf[0])
            seller = [k for k in strs if re.search(r"seller|merchant|shop|store|vendor|sold", k, re.I)]
            if seller:
                self.keys.setdefault("seller", seller[0])
            brand = [k for k in strs if re.search(r"brand|maker|make|manufact", k, re.I)]
            if brand:
                self.keys.setdefault("brand", brand[0])

    def rows_in(self, obs: Any) -> list[dict]:
        lists = self.dict_lists(obs)
        best: list[dict] = []
        for path, L in lists:
            if self.id_key and all(self.id_key in r for r in L):
                if len(L) > len(best):
                    best = L
        if best:
            return best
        for path, L in lists:
            if any(any(self.idlike(v) for v in r.values()) for r in L) and len(L) > len(best):
                best = L
        if best:
            self.learn_rows(best)
        return best

    def ident(self, rec: dict) -> str | None:
        if self.id_key and self.idlike(rec.get(self.id_key)):
            return rec[self.id_key]
        return None

    @staticmethod
    def facts_in(rec: dict) -> dict:
        best: dict = {}
        for k, v in rec.items():
            if isinstance(v, dict) and len(v) > len(best) and all(not isinstance(x, (dict, list)) or isinstance(x, list) for x in v.values()):
                if not any(isinstance(x, (int, float)) and len(v) <= 3 for x in v.values()):
                    best = v
            elif isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                pairs = {}
                for x in v:
                    ks = list(x.keys())
                    if len(ks) == 2:
                        a, b = x[ks[0]], x[ks[1]]
                        if isinstance(a, str):
                            pairs[a] = b
                if len(pairs) > len(best):
                    best = pairs
        return best


def next_marker(obs: Any, rows_key_path: Any = None) -> Any:
    if not isinstance(obs, dict):
        return "absent"
    for k, v in obs.items():
        if isinstance(v, dict):
            r = next_marker(v)
            if r != "absent":
                return r
    for k, v in obs.items():
        if re.search(r"next|continue|more|follow|token|cursor|resume|after", k, re.I) and not isinstance(v, (list, dict)):
            return v
    return "absent"


def find_key(obs: Any, key: str) -> Any:
    if isinstance(obs, dict):
        if key in obs and not isinstance(obs[key], (dict, list)):
            return obs[key]
        for v in obs.values():
            r = find_key(v, key)
            if r != "absent":
                return r
    return "absent"


def total_in(obs: Any) -> int | None:
    if not isinstance(obs, dict):
        return None
    for k, v in obs.items():
        if isinstance(v, dict):
            r = total_in(v)
            if r is not None:
                return r
        if isinstance(v, int) and not isinstance(v, bool) and re.search(r"total|count|match|found|num|hits", k, re.I):
            return v
    return None


def notes_in(obs: Any) -> tuple[Any, list[str]]:
    if isinstance(obs, dict) and len(obs) == 2:
        ks = list(obs)
        for a, b in ((ks[0], ks[1]), (ks[1], ks[0])):
            if isinstance(obs[b], list) and obs[b] and all(isinstance(x, str) for x in obs[b]):
                return obs[a], list(obs[b])
    return obs, []


def refusal_in(obs: Any) -> list[str]:
    if isinstance(obs, dict):
        for k, v in obs.items():
            if isinstance(v, list) and v and all(isinstance(x, str) for x in v) and len(obs) <= 2:
                return list(v)
            if isinstance(v, str) and re.search(r"refus|cannot|can't|not allowed|rejected|blocked", v, re.I) and len(obs) <= 2:
                return [v]
    return []


def error_in(obs: Any) -> str | None:
    if isinstance(obs, dict) and len(obs) <= 3:
        for k, v in obs.items():
            if isinstance(v, str) and re.search(r"error|issue|problem|fault|invalid|unknown|missing|unexpected|not found|no such|must|exhausted|used up", k + " " + v, re.I):
                if re.search(r"err|issue|problem|fault|detail|message|reason", k, re.I) or len(obs) == 1:
                    return v
    if isinstance(obs, str) and re.search(r"error|invalid|unknown", obs, re.I):
        return obs
    return None


class Episode:
    def __init__(self, problem: dict):
        env = problem.get("environment") or {}
        self.policy = env.get("policy_view") or {}
        self.request = str(self.policy.get("query") or "")
        if self.policy.get("opening") is not None:
            self.request += "\n\n" + jdump(self.policy.get("opening"), 3000)
        self.max_steps = int(self.policy.get("max_steps") or 40)
        self.wire = Wire(env)
        self.mind = Mind()
        self.contract = Contract(self.policy.get("tools") or [], self.mind)
        self.shapes = Shapes()
        self.listings: dict[str, dict] = {}
        self.facts: dict[str, dict] = {}
        self.notices: list[str] = []
        self.cart: dict[str, int] = {}
        self.cart_failed = False
        self.shaped: set[str] = set()
        self.opened: set[str] = set()
        self.gave_facts: dict[str, int] = {}
        self.repaired: set[str] = set()
        self.last_total: int | None = None
        self.intent: str | None = None
        self.last_reason = "Nothing in the store meets every requirement, so I am not buying anything."
        self.extra_plan: dict[str, dict] = {}
        self.extra_used: dict[str, int] = {}
        self.extra_ok: dict[str, bool] = {}
        try:
            self.plan_extras()
        except Exception:
            self.extra_plan = {}

    def act(self, role: str, **kw: Any) -> Any:
        b = self.contract.build(role, **kw)
        if not b:
            return None
        obs = self.wire.call(*b)
        obs, notes = notes_in(obs)
        if notes:
            self.notices += notes
        err = error_in(obs)
        if err and re.search(ARG_ERR, err + " " + jdump(obs, 400)) and self.steps_left() > 2 and role not in self.repaired:
            fixed = self.repair(role, b[0], b[1], jdump(obs, 600))
            if fixed is not None:
                obs2 = self.wire.call(b[0], fixed)
                obs2, notes = notes_in(obs2)
                self.notices += notes
                if not error_in(obs2):
                    self.contract.learn_fix(role, b[1], fixed)
                    return obs2
                obs = obs2
            self.repaired.add(role)
        return obs

    def repair(self, role: str, tool: str, args: dict, error: str) -> dict | None:
        t = self.contract.tools.get(tool) or {}
        schema = {"name": tool, "description": t.get("desc", "")[:400], "parameters": t.get("props"), "required": t.get("req")}
        got = self.mind.ask(FIX_SYS, f"ROLE: {role}\nTOOL: {jdump(schema, 3000)}\nARGUMENTS SENT: {jdump(args, 1500)}\nSTORE SAID: {error}", 600)
        fixed = (got or {}).get("arguments") if isinstance(got, dict) else None
        return fixed if isinstance(fixed, dict) and fixed != args else None

    def steps_left(self) -> int:
        return self.max_steps - self.wire.step

    def read_shape(self, tool: str, obs: Any, override: bool = True) -> None:
        if tool in self.shaped or not isinstance(obs, (dict, list)) or not self.mind.ok():
            return
        self.shaped.add(tool)
        lists = Shapes.dict_lists(obs)
        rows = max((L for p, L in lists), key=len, default=[])
        sample = obs
        if rows:
            def trim(o: Any) -> Any:
                if isinstance(o, list) and o and all(isinstance(x, dict) for x in o) and o is rows:
                    return o[:3]
                if isinstance(o, dict):
                    return {k: trim(v) for k, v in o.items()}
                return o
            sample = trim(obs)
        t = self.contract.tools.get(tool) or {}
        got = self.mind.ask(SHAPE_SYS, f"TOOL: {tool}: {t.get('desc', '')[:200]} | parameters: {jdump(t.get('props'), 1500)}\nRESULT:\n{jdump(sample, 5000)}", 600)
        if not isinstance(got, dict):
            return
        recs = rows or ([obs] if isinstance(obs, dict) else [])
        keys = set().union(*[set(r) for r in recs]) if recs else set()
        def vals(k): return [r.get(k) for r in recs if r.get(k) is not None]
        k = got.get("id")
        if k in keys and vals(k) and all(Shapes.idlike(v) for v in vals(k)) and len(set(vals(k))) == len(vals(k)) and (override or not self.shapes.id_key):
            self.shapes.id_key = k
        for f in ("title", "price", "stock", "shelf", "brand", "seller", "text"):
            k = got.get(f)
            if k not in keys or not vals(k):
                continue
            v = vals(k)
            if f == "price" and not all(num(x) is not None and not isinstance(x, bool) for x in v):
                continue
            if f == "stock" and not all(isinstance(x, (bool, int, str)) for x in v):
                continue
            if override:
                self.shapes.keys[f] = k
            else:
                self.shapes.keys.setdefault(f, k)
        top = obs if isinstance(obs, dict) else {}
        flat = {}
        def walk(o: Any) -> None:
            if isinstance(o, dict):
                for kk, vv in o.items():
                    if not isinstance(vv, (dict, list)):
                        flat.setdefault(kk, vv)
                    else:
                        walk(vv)
        walk(top)
        for f in ("next", "total"):
            k = got.get(f)
            if k in flat and (f != "total" or isinstance(flat[k], int)):
                self.shapes.keys[f + "@" + tool] = k

    def absorb_rows(self, obs: Any) -> list[str]:
        rows = self.shapes.rows_in(obs)
        out = []
        for r in rows:
            i = self.shapes.ident(r)
            if not i:
                continue
            rec = self.listings.setdefault(i, {})
            rec.update(r)
            out.append(i)
        return out

    def search(self, role: str, pages: int = 1, **kw: Any) -> list[str]:
        spec = self.contract.roles.get(role)
        if not spec:
            return []
        found: list[str] = []
        token: Any = None
        size = 0
        for p in range(pages):
            if self.steps_left() <= 4:
                break
            page = None
            if p > 0:
                if spec.get("page") is None:
                    break
                page = self.contract.page_value(spec, p, token, size)
                if page is None:
                    break
            obs = self.act(role, page=page, **kw)
            self.read_shape(spec["tool"], obs)
            ids = self.absorb_rows(obs)
            if not ids:
                break
            size = size or len(ids)
            found += [i for i in ids if i not in found]
            nk = self.shapes.keys.get("next@" + spec["tool"])
            mk = find_key(obs, nk) if nk else next_marker(obs)
            if mk in (None, False, "", 0) and mk != "absent":
                break
            if mk != "absent":
                token = mk
            elif spec.get("page_kind") == "offset":
                token = len(found)
            tk = self.shapes.keys.get("total@" + spec["tool"])
            tot = find_key(obs, tk) if tk else total_in(obs)
            tot = tot if isinstance(tot, int) and not isinstance(tot, bool) else total_in(obs)
            self.last_total = tot
            if tot is not None and len(found) >= tot:
                break
        return found

    def read(self, i: str, fresh: bool = False) -> dict:
        self.opened.add(i)
        specs = list(self.contract.roles.get("detail") or [])
        specs.sort(key=lambda d: -self.gave_facts.get(d["tool"], 0))
        for spec in specs:
            t = self.contract.tools.get(spec["tool"]) or {}
            obs = self.wire.call(spec["tool"], self.contract.conform(t, {**(spec.get("fixed") or {}), spec["id"]: i}))
            obs, notes = notes_in(obs)
            self.notices += notes
            if error_in(obs) or not isinstance(obs, dict):
                continue
            fx = Shapes.facts_in(obs)
            if not fx and str(i) not in jdump(obs, 20000) and len(self.contract.roles.get("detail") or []) > 1:
                self.contract.roles["detail"] = [d for d in self.contract.roles["detail"] if d.get("tool") != spec["tool"]]
                continue
            self.read_shape(spec["tool"], obs, override=False)
            rec = self.listings.setdefault(i, {})
            self.gave_facts[spec["tool"]] = self.gave_facts.get(spec["tool"], 0) + (1 if fx else -1)
            if fx:
                self.facts[i] = {**self.facts.get(i, {}), **fx}
            rec.update({k: v for k, v in obs.items() if not isinstance(v, (dict, list)) or k == self.shapes.keys.get("price")})
            if not fresh and (spec.get("gives") != "facts" or fx):
                break
        return self.listings.get(i, {})

    def read_many(self, ids: list[str]) -> list[str]:
        ids = [i for i in dict.fromkeys(ids) if i not in self.facts and i not in self.opened]
        spec = self.contract.roles.get("compare")
        if spec and len(ids) > 1 and self.steps_left() > 1:
            chunk = ids[:max(1, int(spec.get("max") or 4))]
            b = self.contract.build("compare", ids=chunk)
            obs = self.wire.call(*b) if b else None
            obs, notes = notes_in(obs)
            self.notices += notes
            done = []
            recs: list[tuple[str, dict]] = []
            if not error_in(obs):
                for r in self.shapes.rows_in(obs):
                    i = self.shapes.ident(r)
                    if i in chunk:
                        recs.append((i, r))
                if not recs and isinstance(obs, dict):
                    for box in [obs] + [v for v in obs.values() if isinstance(v, dict)]:
                        recs += [(i, box[i]) for i in chunk if isinstance(box.get(i), dict)]
            for i, r in recs:
                fx = Shapes.facts_in(r)
                rec = self.listings.setdefault(i, {})
                rec.update({k: v for k, v in r.items() if not isinstance(v, (dict, list))})
                if fx:
                    self.facts[i] = {**self.facts.get(i, {}), **fx}
                    done.append(i)
            if done:
                return done
            self.contract.roles["compare"] = None
        if ids and self.steps_left() > 1:
            self.read(ids[0])
            return [ids[0]]
        return []

    def per_step(self) -> int:
        spec = self.contract.roles.get("compare")
        return max(1, int(spec.get("max") or 4)) if spec else 1

    def price(self, i: str) -> float | None:
        k = self.shapes.keys.get("price")
        return num(self.listings.get(i, {}).get(k)) if k else None

    def in_stock(self, i: str) -> bool:
        k = self.shapes.keys.get("stock")
        v = self.listings.get(i, {}).get(k) if k else None
        if v is None or v is True:
            return True
        if v is False or (isinstance(v, (int, float)) and v <= 0):
            return False
        return not (isinstance(v, str) and re.search(r"\b(?:out|sold|none|unavailable|no|zero|false)\b", v, re.I))

    def title(self, i: str) -> str:
        k = self.shapes.keys.get("title")
        return str(self.listings.get(i, {}).get(k) or "") if k else ""

    def absorb_cart(self, obs: Any) -> None:
        pk = self.shapes.keys.get("price")
        for path, L in Shapes.dict_lists(obs):
            for r in L:
                i = self.shapes.ident(r)
                if i and i in self.listings and pk and pk in r:
                    self.listings[i][pk] = r[pk]

    def put_cart(self, lines: list[tuple[str, int]]) -> Any:
        failed = False
        if self.contract.roles.get("cart_set"):
            obs = self.act("cart_set", lines=lines)
            failed = bool(error_in(obs))
            self.absorb_cart(obs)
        else:
            for i, q in list(self.cart.items()):
                if i not in dict(lines):
                    failed |= bool(error_in(self.act("cart_remove", id=i, qty=q)))
            obs = None
            for i, q in lines:
                have = self.cart.get(i, 0)
                if have != q:
                    if have:
                        failed |= bool(error_in(self.act("cart_remove", id=i, qty=have)))
                    obs = self.act("cart_add", id=i, qty=q)
                    failed |= bool(error_in(obs))
        self.cart = dict(lines)
        if failed:
            self.cart_failed = True
            self.sync_cart()
        return obs

    def sync_cart(self) -> None:
        if not self.contract.roles.get("cart_view") or self.steps_left() <= 2:
            if self.cart_failed:
                self.cart = {}
            return
        view = self.act("cart_view")
        text = jdump(view, 20000)
        if self.cart and not error_in(view) and not any(str(i) in text for i in self.cart) and not Shapes.dict_lists(view) \
                and not re.search(r"\[\s*\]|:\s*0(?:\.0+)?\b", text):
            self.contract.roles["cart_view"] = None
            return
        self.absorb_cart(view)
        held = {self.shapes.ident(r) for path, rows in Shapes.dict_lists(view) for r in rows}
        self.cart = {i: q for i, q in self.cart.items() if i in held}

    def plan_extras(self) -> None:
        names = self.contract.extras()
        if not names:
            return
        def guessed(n: str) -> dict:
            for r, v in self.contract.roles.items():
                for x in (v if isinstance(v, list) else [v]):
                    if n in self.contract.weak and isinstance(x, dict) and x.get("tool") == n:
                        return {"currently_guessed_job": r}
            return {}
        listing = [{"name": n, **guessed(n), "description": self.contract.tools[n]["desc"][:500], "parameters": self.contract.tools[n]["props"],
                    "required": self.contract.tools[n]["req"]} for n in names[:8]]
        got = self.mind.ask(EXTRA_SYS, "REQUEST:\n" + self.request[:2500] + "\n\nTOOLS:\n" + jdump(listing, 6000), 900)
        for x in (got or {}).get("tools") or [] if isinstance(got, dict) else []:
            if not isinstance(x, dict) or x.get("name") not in names:
                continue
            w = str(x.get("when") or "").lower()
            when = next((k for k in ("never", "before_order", "each_pick", "start") if w.startswith(k) or w == k), None) or \
                ("never" if re.search(r"\bnever\b|\bnot (?:call|use)\b|unrelated", w) else
                 "before_order" if re.search(r"before (?:check|order|buy|purchas|plac)|check(?:ing)? ?out", w) else
                 "each_pick" if re.search(r"each (?:listing|item|product|pick)|per listing|every listing", w) else
                 "start" if re.search(r"before shopping|at the start|first|once before", w) else None)
            if w.startswith("known") or w == "known":
                self.contract.weak.discard(x["name"])
                continue
            if x["name"] in self.contract.weak and when in ("never", "start", "each_pick", "before_order"):
                self.contract.unbind(x["name"])
            if when in ("start", "each_pick", "before_order"):
                self.extra_plan[x["name"]] = {"when": when, "args": x.get("args") if isinstance(x.get("args"), dict) else None}

    def extras_when(self, when: str) -> list[str]:
        return [n for n, p in self.extra_plan.items() if p["when"] == when]

    def extra_args(self, name: str, ids: list[str] | None = None, said: str = "") -> dict:
        t = self.contract.tools[name]
        ids = list(ids or self.cart or [])
        tpl = (self.extra_plan.get(name) or {}).get("args")
        if not t["props"]:
            tpl = {}
        elif tpl is None or (said and not tpl):
            ctx = {"cart": [{"id": i, "qty": self.cart.get(i, 1), "title": self.title(i)[:80]} for i in ids]}
            got = self.mind.ask(FILL_SYS, f"REQUEST:\n{self.request[:2000]}\n\nTOOL: {jdump({'name': name, 'description': t['desc'][:500], 'parameters': t['props'], 'required': t['req']}, 3000)}"
                                f"\nCART: {jdump(ctx, 1500)}\nSTORE SAID: {said[:800]}", 600)
            tpl = (got or {}).get("arguments") if isinstance(got, dict) else None
            tpl = tpl if isinstance(tpl, dict) else {}
        def fill(v: Any) -> Any:
            if isinstance(v, str):
                if v == "<LISTING_IDS>":
                    return ids
                if v == "<LISTING_ID>":
                    return ids[0] if ids else v
                if v == "<QTY>":
                    return self.cart.get(ids[0], 1) if ids else 1
                return v.replace("<LISTING_ID>", ids[0]) if ids and "<LISTING_ID>" in v else v
            if isinstance(v, list):
                return [fill(x) for x in v]
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            return v
        args = self.contract.conform(t, fill(tpl))
        for a in list(args):
            sp = t["props"].get(a) or {}
            if isinstance(args[a], str) and re.fullmatch(r"<[^<>]*>", args[a]):
                del args[a]
            elif sp.get("enum") and args[a] not in sp["enum"]:
                args[a] = sp["enum"][0]
        for a in t.get("req") or []:
            if a in args:
                continue
            sp = t["props"].get(a) or {}
            if sp.get("enum"):
                args[a] = sp["enum"][0]
            elif sp.get("type") in ("integer", "number"):
                args[a] = max(1, int(sp.get("minimum") or 1))
            elif sp.get("type") == "array":
                args[a] = ids
            elif re.search(ID_ARG, a, re.I) and ids:
                args[a] = ids[0]
        return args

    def call_extra(self, name: str, ids: list[str] | None = None, said: str = "") -> Any:
        if self.steps_left() <= 1 or self.wire.finished:
            return None
        self.extra_used[name] = self.extra_used.get(name, 0) + 1
        obs = self.wire.call(name, self.extra_args(name, ids, said))
        obs, notes = notes_in(obs)
        self.notices += notes
        return obs

    def start_extras(self) -> None:
        for n in self.extras_when("start"):
            if self.steps_left() <= 8:
                break
            obs = self.call_extra(n)
            if obs is not None and not error_in(obs):
                self.request += f"\n\nThe store's {n} says: " + jdump(obs, 1500)

    def check_picks(self, ids: list[str]) -> set[str]:
        bad = set()
        tools = self.extras_when("each_pick")
        for i in ids:
            if i in self.extra_ok or not tools:
                continue
            self.extra_ok[i] = True
            for n in tools:
                if self.steps_left() <= 4:
                    return bad
                obs = self.call_extra(n, [i])
                if obs is None or error_in(obs):
                    continue
                got = self.mind.ask(CHECK_SYS, f"REQUEST:\n{self.request[:1500]}\n\nLISTING: {i} {self.title(i)[:100]}\nTOOL: {n}: "
                                    f"{self.contract.tools[n]['desc'][:300]}\nANSWER: {jdump(obs, 1500)}", 300)
                if isinstance(got, dict) and got.get("ok") is False:
                    self.extra_ok[i] = False
                    bad.add(i)
                    break
        return bad

    def pre_order(self) -> None:
        for n in self.extras_when("before_order"):
            if not self.extra_used.get(n) and self.steps_left() > 2:
                self.call_extra(n, list(self.cart))

    def unblock(self, obs: Any) -> bool:
        if self.wire.finished or self.steps_left() <= 2:
            return False
        text = jdump(obs, 3000)
        roles = self.contract.roles
        side = {(roles.get(r) or {}).get("tool") for r in ("cart_view", "status") if isinstance(roles.get(r), dict)}
        for n in list(self.contract.tools):
            if n in side or n in self.contract.extras():
                if self.extra_used.get(n, 0) >= 2:
                    continue
                parts = re.findall(r"[A-Z]?[a-z]+|[0-9]+|[A-Z]+(?![a-z])", n)
                spelled = r"(?<![A-Za-z0-9])" + r"[\s_-]*".join(map(re.escape, parts)) + r"(?![A-Za-z0-9])" if len(parts) > 1 else None
                if re.search(r"(?<![A-Za-z0-9_])" + re.escape(n) + r"(?![A-Za-z0-9_])", text) or (spelled and re.search(spelled, text, re.I)):
                    self.call_extra(n, list(self.cart), said=text)
                    return True
        return False

    def rescue(self, obs: Any) -> bool:
        unused = [n for n in self.contract.extras() if not self.extra_used.get(n)]
        if not unused or self.wire.finished or self.steps_left() <= 2 or getattr(self, "rescued", False):
            return False
        self.rescued = True
        listing = [{"name": n, "description": self.contract.tools[n]["desc"][:400], "parameters": self.contract.tools[n]["props"],
                    "required": self.contract.tools[n]["req"]} for n in unused[:8]]
        got = self.mind.ask(RESCUE_SYS, f"REQUEST:\n{self.request[:1500]}\n\nCART: {jdump({i: self.title(i)[:60] for i in self.cart}, 800)}"
                            f"\nSTORE SAID: {jdump(obs, 1200)}\nTOOLS:\n{jdump(listing, 4000)}", 500)
        name = (got or {}).get("tool") if isinstance(got, dict) else None
        if name not in unused:
            return False
        if isinstance(got.get("arguments"), dict):
            self.extra_plan[name] = {"when": "before_order", "args": got["arguments"]}
        self.call_extra(name, list(self.cart), said=jdump(obs, 800))
        return True

    def finish_order(self) -> Any:
        self.intent = "order"
        self.pre_order()
        obs = self.act("order")
        if not self.wire.finished and (self.unblock(obs) or ((error_in(obs) or refusal_in(obs)) and self.rescue(obs))):
            if not self.wire.finished and self.steps_left() > 0:
                obs = self.act("order")
        return obs

    def finish_decline(self, reason: str) -> Any:
        self.intent, self.last_reason = "decline", reason
        obs = self.act("decline", reason=reason)
        if not self.wire.finished:
            self.unblock(obs)
        return obs


def agent_main(problem_data: dict) -> list[dict]:
    try:
        ep = Episode(problem_data if isinstance(problem_data, dict) else {})
    except Exception:
        ep = Episode({"environment": (problem_data or {}).get("environment") if isinstance(problem_data, dict) else {}})
    brain = None
    try:
        brain = Brain(ep)
        brain.run()
    except Exception as exc:
        ep.mind.notes.append({"error": repr(exc)[:300]})
    for _ in range(3):
        if ep.wire.finished or ep.steps_left() <= 0:
            break
        try:
            if ep.intent == "order" or (ep.intent is None and ep.cart):
                ep.finish_order()
            elif ep.intent == "decline":
                ep.finish_decline(ep.last_reason)
            if not ep.wire.finished and ep.steps_left() > 0 and ep.intent != "order":
                ep.finish_decline(ep.last_reason)
        except Exception:
            break
    return ep.wire.dialogue




BRIEF_SYS = """You read a shopper's request for a shopping agent. Return ONE JSON object only:
{"lines":[{"item":"what to buy, in the request's own words","kinds":["2-4 short search nouns for it, most specific first"],
  "qty":int,"musts":[{"property":"<the property, named the way a store labels it, e.g. material, colour, size, type>","value":"<the request's own word(s) for the value, without filler such as -shaped, -coloured, style, type around them>","also":["other wordings the request itself uses for that same value"],"relation":"is"|"at_least"|"at_most"|"not"|"any_of"}],
  "seller":null|"<seller the request names for this item>","brand":null|"<brand>","fits_line":null|int},...],
 "budget":number|null,"budget_each":bool,"cheapest":bool,"messages":int|null,"steps":int|null,"in_stock_only":bool,
 "decline_rule":"name_closest"|"no_listing_ids"|"free","confirm_before_order":bool,
 "avoid":[{"property":"...","value":"..."}],"hints":["properties the story points at without giving a value"]}
Rules: one line per distinct product to buy (a department name joined by "and" or "&" is ONE product). "fits_line" is the index
of the line this product must work with or fit, when the request says so and that other product is also being bought
here (a device the shopper already owns is not a line: put it as a must, property "compatibility"). "budget" is the number the request gives as the
spending limit, in the request's unit. "messages" is how many messages the agent may send the shopper; "steps" how many
actions/moves/steps/calls it may take. "decline_rule": name_closest when the request asks to name the closest/nearest option
if nothing fits; no_listing_ids when it forbids naming listings or substitutes in that message; otherwise free.
Never replace the request's value word with a different word of your own (keep "lidless", not "none"). "qty" counts
units to buy ("a pair of" shoes or jeans is 1); a pack size or count inside one listing is a must, not qty.
When the shopper adds right next to a property that they will verify that part on their own (e.g. "meant for a lakeside
picnic, and I'll be confirming that bit personally"), that property is a must even if it reads like a use (occasion: lakeside
picnic). Without such a
remark, events, people and reasons in the story are not musts. People in the story are not extra products.
"confirm_before_order" only when the request asks you to check back with the shopper before buying; the shopper checking
the item themselves is not that. Story details and why the shopper wants something are not musts. That a product must fit, suit or work with another product
of the request is "fits_line", never a must. Copy values in the request's words."""

CUE_SYS = """A shopping agent may ask the shopper a few questions about the products. The shopper holds back firm preferences
on some properties and states one only when a question names that property. For each product you get the property names
its listings carry. Pick the properties that the request's story or wording points at without giving a value (an event,
a use, a person, a look, a place, a worry, a fit), most clearly pointed-at first. Leave out properties the request already
fixes with a value. Pick nothing when the request gives no such pointer; do not guess from general likelihood. Use the
exact property names given. Also, for each product, list up to 6 other properties a shopper typically decides between
for it that are missing from its list (short names, as a store would label them). Return ONE JSON object only:
{"cues":{"<product index>":["property",...]},"more":{"<product index>":["property",...]}}"""

HEAR_SYS = """A shopping agent asked a shopper a question. Read the reply. Return ONE JSON object only:
{"kind":"need"|"no_preference"|"confused"|"one_at_a_time"|"other","needs":[{"product":int|null,"property":"...","relation":"is"|"at_least"|"at_most"|"not"|"any_of","value":"...",
  "from_listing":null|{"kind":"...","seller":"...","details":[["property","value"],...]}}]}
"need" when the reply states a firm preference; give the property in the shopper's words, the value word for word (keep units),
and read the relation by meaning: when the shopper turns down everything other than a value, the need is that value ("is");
use "not" only when the shopper rejects that one value itself;
and the product index from the list. "from_listing" when the value must be copied from another listing's page that the reply
describes. "no_preference" when the shopper does not mind; "confused" when they did not understand; "one_at_a_time" when
they ask for one question at a time."""

NOTE_SYS = """While shopping, the store or shopper sent these notes. Return ONE JSON object only:
{"changes":[{"kind":"sold_out"|"price_changed"|"budget"|"requirement"|"dropped"|"other","listing":null|"<id or title words>","budget":null|number,
  "property":null|"...","value":null|"..."}]}
"sold_out": a listing is gone or has none left; "price_changed": a listing costs more or less now; "budget": the spending
limit changed (give the new number); "requirement": the shopper now wants a property to have a (new) value; "dropped": the
shopper no longer cares about a property (forget it, skip it, drop it, not needed any more)."""

JUDGE_SYS = """Decide whether each listing meets each requirement, using only the listing's FACTS table (its attributes); the title
and description do not count. A property listed under another spelling or a clear synonym of its name counts; a key about a
different property, or about one part of the product when the requirement is about the whole, decides nothing; numbers
compare with their units; if the facts say nothing about it, answer "unknown". Return ONE JSON object only:
{"verdicts":{"<listing id>":{"<requirement number>":"yes"|"no"|"unknown"}}}"""

FIT_SYS = """Read these listing texts and extract compatibility facts. Return ONE JSON object only:
{"listings":{"<id>":{"is_model":["model codes or names this listing IS"],"fits":["model codes or names it fits/works with; keep
wildcards like 57xx and write exceptions as 'except <code>'"],"part_code":null|"<its part code>",
"also_fits":[{"part":"<part code>","models":["..."]}]}}}
Copy codes exactly. Leave lists empty when the text says nothing."""

AVAIL = r"availab|on hand|in stock|stock|deliver|ship|online|website|listed|listing"
ARG_ERR = r"invalid|missing|required|expected|unknown (?:argument|field|property)|additional propert|must be|not allowed|schema|type"
FIX_SYS = """A shopping agent called a store tool and the store rejected the arguments. Using the tool's parameter schema and the
store's message, return corrected arguments that do the same job. Keep every value the agent meant to send; only fix names,
shapes, types or missing required parts. Return ONE JSON object only: {"arguments":{...}}"""

SHAPE_SYS = """Read one result of an online store's tool. Name where things are, using the exact keys of the result (keys
inside one listing row for the row fields). Return ONE JSON object only:
{"id":key|null,"title":key|null,"price":key|null,"stock":key|null,"shelf":key|null,"brand":key|null,"seller":key|null,
 "text":key|null,"next":key|null,"total":key|null}
"id": the listing identifier; "price": the price number; "stock": whether it can be bought now; "shelf": its category or
department; "text": its longest description text; "next": the key whose value asks for the following page (null if none);
"total": the key holding how many listings matched (null if none)."""

EXTRA_SYS = """A shopping agent already knows how to search the store, open listings, message the shopper, change the cart,
check out and end without buying. The store also offers the tools below, which do none of those jobs. Read each tool's name,
description and schema, and the shopper's request, and decide when the agent should call it:
- "start": once before shopping, because it tells the agent about this shopper, their needs, or the store's rules for this task;
- "each_pick": once for each listing the agent is about to buy, because it checks whether that one listing suits, fits, is
  available or is allowed;
- "before_order": once just before checking out, because the store needs this step done (confirm, reserve, choose, accept)
  before an order goes through;
- "never": unrelated to this request, only decorative, or it would change or cancel something the shopper asked for;
- "known": it actually does one of the agent's known jobs (searches listings, opens or shows one listing's details or
  page, compares listings, messages the shopper, shows the cart, reports remaining steps or messages).
In "args" give the values to send: copy values from the request where the tool needs them, use option values exactly as the
schema lists them, write "<LISTING_ID>" where one listing's id goes, "<LISTING_IDS>" for the list of listings being bought
and "<QTY>" for a quantity. Return ONE JSON object only: {"tools":[{"name":"...","when":"...","args":{...}}]}"""

FILL_SYS = """A shopping agent must call a store tool. Using the tool's schema, the shopper's request, the cart and what the store
said, give the arguments to send. Use option values exactly as the schema lists them and listing ids exactly as given.
Return ONE JSON object only: {"arguments":{...}}"""

RESCUE_SYS = """A shopping agent tried to check out and the store did not accept it. Read what the store said and the tools the
agent has not used yet. If calling one of them is the missing step that lets the order go through (a confirmation,
reservation, choice or acceptance the store asks for), name it and the arguments to send (option values exactly as the
schema lists them); otherwise answer null (for example when an item sold out or a price changed).
Return ONE JSON object only: {"tool":"<name>"|null,"arguments":{...}}"""

CHECK_SYS = """A shopping agent asked a store tool about one listing it is about to buy for the shopper's request. Read the
tool's answer. "ok" is false only when the answer clearly says this listing cannot be bought, does not fit or suit what the
request needs, or is not allowed; an error about the call itself, or an answer that says nothing either way, is ok.
Return ONE JSON object only: {"ok":true|false,"why":"..."}"""

NOT_ASK = r"warrant|\bcondition\b|\bship|deliver|origin|\breturn|guarantee|seller|\bsku|\bcreat|\blisted|upload|^options?$|variant|model number|\bean\b|\bupc\b|\basin\b|\bprice|\bstock\b|\bid$|^id |image|\burl\b|\bratings?\b|\breview|margin|short des"


class Need:
    def __init__(self, prop: str, value: Any, rel: str = "is", source: str = "stated", ref: dict | None = None):
        self.prop, self.value, self.rel, self.source, self.ref = str(prop or ""), value, rel or "is", source, ref
        self.alts: list[str] = []

    def label(self) -> str:
        r = {"is": "", "at_least": "at least ", "at_most": "at most ", "not": "not ", "any_of": "one of "}.get(self.rel, "")
        return f"{self.prop}: {r}{self.value}"


class Line:
    def __init__(self, n: int, d: dict):
        self.n = n
        self.item = str(d.get("item") or "item")
        self.kinds = [str(k) for k in (d.get("kinds") or []) if k][:4] or [self.item]
        try:
            self.qty = max(1, int(d.get("qty") or 1))
        except Exception:
            self.qty = 1
        self.needs: list[Need] = []
        for m in d.get("musts") or []:
            if isinstance(m, dict) and m.get("property") and m.get("value") not in (None, ""):
                val, rel = str(m["value"]), m.get("relation") or "is"
                neg = re.match(r"\s*(?:no|non|without|not)[\s-]+(.+)", val, re.I)
                if neg and rel in ("is", "not"):
                    val, rel = neg.group(1), "not"
                nd = Need(m["property"], val, rel)
                nd.alts = [str(a) for a in (m.get("also") or []) if isinstance(a, (str, int, float)) and str(a).strip()][:3]
                self.needs.append(nd)
        self.seller = d.get("seller")
        self.brand = d.get("brand")
        self.needs = [x for x in self.needs if not re.search(AVAIL, f"{x.prop} {x.value}", re.I)]
        if self.brand and not any(roots(self.brand) <= roots(x.value) for x in self.needs):
            self.needs.append(Need("brand", self.brand))
        if self.seller:
            self.needs.append(Need("seller", self.seller))
        fl = d.get("fits_line")
        self.fits_line = fl if isinstance(fl, int) and fl != n else None
        self.shelf: str | None = None
        self.cands: list[str] = []
        self.asked: list[str] = []
        self.found = 0
        self.msgs = 0
        self.vhits: dict[str, int] = {}
        self.vpos: dict[str, int] = {}
        self.vq: set[str] = set()
        self.order: list[str] = []
        self.cues: list[str] = []
        self.exhausted = False


class Brain:
    def __init__(self, ep: Episode):
        self.ep = ep
        self.mind = ep.mind
        self.lines: list[Line] = []
        self.budget: float | None = None
        self.budget_each = False
        self.cheapest = False
        self.msg_cap = 0
        self.msgs = 0
        self.decline_rule = "free"
        self.confirm = False
        self.avoid: list[Need] = []
        self.hints: list[str] = []
        self.verdict: dict[tuple, str] = {}
        self.fit: dict[str, dict] = {}
        self.dead: set[str] = set()
        self.vq: dict[str, list[str]] = {}
        self.paged: dict[str, bool] = {}
        self.scope_ok: bool | None = None
        self.carted = False
        self.polished = False

    def brief(self) -> None:
        ep = self.ep
        d = self.mind.ask(BRIEF_SYS, "REQUEST:\n" + ep.request, 1200) or {}
        lines = d.get("lines") if isinstance(d.get("lines"), list) else []
        if not lines:
            lines = [{"item": " ".join(words(ep.request)[:6]), "kinds": [], "qty": 1, "musts": []}]
        self.lines = [Line(n, x) for n, x in enumerate(lines[:5]) if isinstance(x, dict)]
        text = ep.request.replace(",", "")
        self.budget = num(d.get("budget")) if d.get("budget") is not None else None
        m = re.findall(r"(\d+(?:\.\d+)?)\s*(?:cts|cents?|¢)\b", text, re.I)
        if m and (self.budget is None or str(int(self.budget)) not in [x.split(".")[0] for x in m]):
            self.budget = float(m[0]) if len(set(m)) == 1 else self.budget
        self.budget_each = bool(d.get("budget_each"))
        self.cheapest = bool(d.get("cheapest"))
        mm = re.findall(r"(\d+)\s+(?:more\s+)?(?:messages?|questions?|notes?)\b", text, re.I)
        self.msg_cap = int(d["messages"]) if isinstance(d.get("messages"), int) else (int(mm[0]) if mm else 2)
        if mm and len(set(mm)) == 1 and int(mm[0]) != self.msg_cap:
            self.msg_cap = int(mm[0])
        if not self.ep.contract.roles.get("message"):
            self.msg_cap = 0
        st = re.findall(r"(\d+)\s+(?:moves|steps|actions|tool calls|calls|turns)\b", text, re.I)
        steps = d.get("steps") if isinstance(d.get("steps"), int) else (int(st[0]) if st else None)
        if steps:
            ep.max_steps = min(ep.max_steps, int(steps))
        self.decline_rule = d.get("decline_rule") if d.get("decline_rule") in ("name_closest", "no_listing_ids", "free") else "free"
        self.confirm = bool(d.get("confirm_before_order"))
        for a in d.get("avoid") or []:
            if isinstance(a, dict) and a.get("property") and a.get("value") and \
                    not re.search(AVAIL + r"|substitut|alternativ|replacement|detour|anything else", f"{a['property']} {a['value']}", re.I):
                self.avoid.append(Need(a["property"], a["value"], "not"))
        self.hints = [str(h) for h in (d.get("hints") or []) if h][:6]

    def row_shelf(self, i: str) -> str | None:
        k = self.ep.shapes.keys.get("shelf")
        v = self.ep.listings.get(i, {}).get(k) if k else None
        return str(v) if v is not None else None

    def kind_roots(self, L: Line) -> set[str]:
        return roots(" ".join(L.kinds + [L.item]))

    def census(self, L: Line, budget_steps: int) -> None:
        ep = self.ep
        start = ep.wire.step
        seen: list[str] = []
        queries = [L.kinds[0]] + ([L.item] if roots(L.item) != roots(L.kinds[0]) else []) + L.kinds[1:2]
        for q in queries:
            if ep.wire.step - start >= max(2, budget_steps // 3) or ep.steps_left() < 8:
                break
            if not ep.contract.roles.get("search_text"):
                break
            got = ep.search("search_text", pages=1, text=q)
            seen += [i for i in got if i not in seen]
            if self.home(L, seen):
                break
        L.shelf = self.home(L, seen) or L.shelf
        twin = next((K for K in self.lines[:L.n] if K.shelf and K.shelf == L.shelf), None)
        if twin is not None:
            L.cands = [i for i in dict.fromkeys(twin.cands + seen) if self.belongs(L, i)]
            return
        if L.shelf is None and ep.contract.roles.get("search_shelf") and L.kinds:
            got = ep.search("search_shelf", pages=1, shelf=L.kinds[0])
            seen += [i for i in got if i not in seen]
            L.shelf = self.home(L, seen) or (self.row_shelf(got[0]) if got else None)
        if L.shelf and ep.contract.roles.get("search_shelf"):
            left = max(1, budget_steps - (ep.wire.step - start))
            cap = None
            if self.budget and ep.contract.roles["search_shelf"].get("cap"):
                cap = self.budget if self.budget_each else self.budget / max(1, L.qty)
            got = ep.search("search_shelf", pages=left, shelf=L.shelf, cap=cap)
            seen += [i for i in got if i not in seen]
            L.exhausted = len(got) < left * 6
            more = 0
            while not any(ep.in_stock(i) and self.affordable(L, i) for i in got) and got and more < 6 and ep.steps_left() > self.reserve() + 8:
                more += 1
                spec = ep.contract.roles["search_shelf"]
                got = ep.search("search_shelf", pages=left + 3 * more, shelf=L.shelf, cap=cap)
                seen += [i for i in got if i not in seen]
        L.cands = [i for i in seen if self.belongs(L, i)]
        if ep.steps_left() >= 8:
            self.value_search(L, max(1, min(3, budget_steps - (ep.wire.step - start))))

    def home(self, L: Line, ids: list[str]) -> str | None:
        kr = self.kind_roots(L)
        vote: dict[str, float] = {}
        for i in ids:
            sh = self.row_shelf(i)
            if not sh:
                continue
            ov = len(kr & roots(sh)) * 2 + len(kr & roots(self.ep.title(i)))
            vote[sh] = vote.get(sh, 0.0) + 0.2 + ov
        if not vote:
            return None
        best = max(vote, key=lambda s: vote[s])
        return best if vote[best] >= 1.0 else None

    def belongs(self, L: Line, i: str) -> bool:
        sh = self.row_shelf(i)
        if L.shelf and sh is not None:
            return sh == L.shelf
        return bool(self.kind_roots(L) & roots(self.ep.title(i) + " " + (sh or "")))

    def record_text(self, i: str) -> str:
        rec = self.ep.listings.get(i, {})
        parts = [str(v) for k, v in rec.items() if isinstance(v, str)]
        return " ".join(parts)

    def fact_keys(self, i: str) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for k, v in (self.ep.facts.get(i) or {}).items():
            vals = v if isinstance(v, list) else re.split(r"\s*[,;/]\s*", str(v))
            out.setdefault(keyname(k), []).extend(str(x) for x in vals if str(x).strip())
        rec = self.ep.listings.get(i, {})
        for role in ("brand", "seller"):
            k = self.ep.shapes.keys.get(role)
            if k and rec.get(k) and role not in out:
                out[role] = [str(rec[k])]
        return out

    def pwords(self, L: Line | None) -> set[str]:
        if L is None:
            return set()
        return roots(L.shelf) if L.shelf else roots(words(L.item)[-1:] if words(L.item) else "")

    def bare(self, text: Any, L: Line | None) -> set[str]:
        r = roots(text)
        b = r - self.pwords(L)
        return b if b else r

    @staticmethod
    def same_key(a: str, b: str) -> bool:
        return keyname(a) == keyname(b) or (bool(roots(a)) and roots(a) == roots(b) and len(words(a)) == len(words(b)))

    @staticmethod
    def key_match(prop: str, key: str) -> bool:
        p, k = roots(prop), roots(key)
        if not p or not k:
            return False
        return keyname(prop) == key or p <= k or (k <= p and len(k) >= 1 and len(p) <= len(k) + 1)

    def keys_for(self, i: str, need: Need, L: Line | None = None) -> list[str]:
        fk = self.fact_keys(i)
        rk = {keyname(x) for x in (self.ep.facts.get(i) or {})}
        real = [k for k in fk if k not in ("brand", "seller") or k in rk]
        keys = [k for k in real if self.key_match(need.prop, k)]
        exact = [k for k in keys if self.same_key(need.prop, k) or (L is not None and self.bare(k, L) == self.bare(need.prop, L))]
        if exact:
            return exact
        if L is not None and self.line_has_key(L, need, exact=True):
            return []
        if keys:
            return keys
        if L is not None and self.line_has_key(L, need):
            return []
        if L is not None and need.rel in ("is", "any_of") and roots(need.value):
            loc = [k for k in self.value_keys(L, need) if k in real]
            if loc:
                return loc
        return [k for k in fk if k in ("brand", "seller") and self.key_match(need.prop, k)]

    def remap(self, L: Line) -> None:
        for n in L.needs:
            if n.source != "stated" or n.rel not in ("is", "any_of") or not roots(n.value):
                continue
            opts = [str(n.value)] + n.alts
            under: dict[str, int] = {}
            named_hit = named_seen = False
            for i in L.cands:
                fx = self.ep.facts.get(i)
                if not fx:
                    continue
                for k, vals in self.fact_keys(i).items():
                    if k not in {keyname(x) for x in fx}:
                        continue
                    if self.same_key(n.prop, k) or self.bare(k, L) == self.bare(n.prop, L):
                        named_seen = True
                    if any(roots(o) and (roots(o) == roots(v) or roots(o) == roots(v) - (self.pwords(L) - roots(o))) for o in opts for v in vals):
                        if self.same_key(n.prop, k) or self.bare(k, L) == self.bare(n.prop, L):
                            named_hit = True
                        under[k] = under.get(k, 0) + 1
            if under and not named_hit and (not named_seen or max(under.values()) >= 2):
                best = max(under, key=lambda k: (under[k], -len(k)))
                n.prop = best
                self.verdict = {k: v for k, v in self.verdict.items() if k[1] != n.label()}
        self._lk, self._vk = {}, {}

    def line_has_key(self, L: Line, need: Need, exact: bool = False) -> bool:
        cache = getattr(self, "_lk", {})
        self._lk = cache
        ck = (L.n, need.prop, exact, len(self.ep.facts))
        if ck not in cache:
            want = keyname(need.prop)
            cache[ck] = any(self.same_key(want, keyname(k)) if exact else self.key_match(need.prop, keyname(k))
                            for j in L.cands for k in (self.ep.facts.get(j) or {}))
        return cache[ck]

    def value_keys(self, L: Line, need: Need) -> list[str]:
        cache = getattr(self, "_vk", {})
        self._vk = cache
        ck = (L.n, need.label(), len(self.ep.facts))
        if ck in cache:
            return cache[ck]
        cnt: dict[str, int] = {}
        want = roots(need.value)
        for j in L.cands:
            for k, vals in self.fact_keys(j).items():
                if k in ("brand", "seller") and k not in {keyname(x) for x in (self.ep.facts.get(j) or {})}:
                    continue
                if any(want <= roots(v) or (roots(v) and roots(v) <= want) for v in vals):
                    cnt[k] = cnt.get(k, 0) + 1
        out = sorted(cnt, key=lambda k: -cnt[k])[:1]
        cache[ck] = out
        return out

    @staticmethod
    def same_number(want: str, got: str) -> bool | None:
        a, b = num(want), num(got)
        if a is None or b is None or not re.search(r"\d", str(want)):
            return None
        if abs(a - b) > 1e-6 * max(1.0, abs(a)):
            return False
        ua = re.sub(r"[\d.\s-]+", " ", str(want).lower()).split()
        ub = re.sub(r"[\d.\s-]+", " ", str(got).lower()).split()
        if ua and ub and not (ua[0].startswith(ub[0][:1]) or ub[0].startswith(ua[0][:1])):
            return False
        if len(re.findall(r"\d+(?:\.\d+)?", str(want))) != len(re.findall(r"\d+(?:\.\d+)?", str(got))):
            return False
        if {stem(w) for w in ua[1:]} != {stem(w) for w in ub[1:]}:
            return None
        return True

    def check(self, i: str, need: Need, L: Line | None = None) -> str:
        v0 = self.verdict.get((i, need.label()))
        if v0:
            return v0
        fk = self.fact_keys(i)
        keys = self.keys_for(i, need, L)
        vals = [v for k in keys for v in fk[k]]
        want = need.value
        named = any(self.same_key(need.prop, k) or (L is not None and self.bare(k, L) == self.bare(need.prop, L)) for k in keys)
        loose = bool(keys) and not named and any(self.key_match(need.prop, k) for k in keys)
        if need.rel in ("at_least", "at_most"):
            w = num(want)
            got = [num(v) for v in vals]
            got = [g for g in got if g is not None]
            if w is None or not got:
                return "unknown" if not vals else "no"
            ok = max(got) >= w if need.rel == "at_least" else min(got) <= w
            return "yes" if ok else "no"
        opts = ([want] if need.rel != "any_of" else re.split(r"\s*(?:,|\bor\b|/)\s*", str(want))) + need.alts
        kw = set().union(*[roots(k) for k in keys]) | roots(need.prop) if keys else roots(need.prop)
        exact = any(roots(o) and (roots(o) == roots(v) or roots(o) == roots(v) - (self.pwords(L) - roots(o))
                                  or roots(o) == roots(v) - (kw - roots(o))) for o in opts for v in vals) or \
            any(self.same_number(str(o), str(v)) for o in opts for v in vals)
        part = any(roots(o) and roots(v) and (roots(o) <= roots(v) or roots(v) <= roots(o)) for o in opts for v in vals
                   if self.same_number(str(o), str(v)) is not False)
        within = any(roots(o) and roots(o) <= roots(v) and (len(roots(o)) > 1 or any(c.isdigit() for c in str(o)) or len(roots(v)) <= 3)
                     for o in opts for v in vals)
        if need.rel == "not":
            if vals:
                return "no" if (exact or part) else "yes"
            seen = roots(self.ep.title(i) + " " + " ".join(str(v) for vs in fk.values() for v in vs))
            return "no" if any(roots(o) and roots(o) <= seen for o in opts) else "near"
        real_keys = {keyname(x) for x in (self.ep.facts.get(i) or {})}
        elsewhere = any(roots(o) and roots(o) == roots(v) for o in opts for k, vs in fk.items()
                        if k not in keys and (k in real_keys or k in ("brand", "seller")) for v in vs)
        if vals:
            if exact and (loose or not named):
                return "near"
            if exact:
                return "yes"
            if within and named:
                return "near"
            if part or within:
                return "part"
            if need.source == "stated" and elsewhere and not named:
                return "near"
            if any(roots(o) and roots(o) == roots(v) for o in opts for k, vs in fk.items()
                   if k not in keys and k in {keyname(x) for x in (self.ep.facts.get(i) or {})} and self.key_match(need.prop, k) for v in vs):
                return "near"
            return "no"
        for k, vs in fk.items():
            if any(roots(o) and roots(o) == roots(k) for o in opts) and any(re.fullmatch(r"\s*(yes|true|y|1)\s*", str(v), re.I) for v in vs):
                return "near"
        if need.rel in ("is", "any_of") and not elsewhere:
            rel = [v for k, vs in fk.items() if k in real_keys and k not in ("options",) and self.key_match(need.prop, k) for v in vs]
            if rel and not any(roots(o) and (roots(o) <= roots(v) or roots(v) <= roots(o)) for o in opts for v in rel):
                return "no"
        if need.rel in ("is", "any_of") and elsewhere:
            return "near" if need.source == "stated" else "part"
        txt = roots(self.ep.title(i) + " " + self.record_text(i))
        if any(roots(o) and roots(o) <= txt for o in opts):
            return "text"
        return "unknown"

    LEVEL = {"yes": 0, "near": 1, "text": 2, "part": 3, "unknown": 4, "no": 9}

    def status(self, L: Line, i: str) -> tuple[int, int, int]:
        no = unk = weak = 0
        skip = self.unverifiable(L)
        for need in L.needs + self.avoid:
            if need.label() in skip:
                continue
            v = self.check(i, need, L)
            if v == "no":
                no += 1
            elif v == "unknown":
                unk += 1
            elif v in ("near", "text", "part"):
                weak += 1
        return no, unk, weak

    def unverifiable(self, L: Line) -> set[str]:
        key = (L.n, len(self.ep.facts), len(L.needs))
        cache = getattr(self, "_unv", {})
        self._unv = cache
        if key not in cache:
            read = [i for i in L.cands if i in self.ep.facts]
            out = set()
            if len(read) >= 4:
                for n in L.needs:
                    if n.rel != "not" and all(self.check(i, n, L) == "unknown" for i in read):
                        out.add(n.label())
            cache[key] = out
        return cache[key]

    def quality(self, L: Line, i: str) -> tuple:
        skip = self.unverifiable(L)
        lv = [self.LEVEL[self.check(i, n, L)] for n in L.needs + self.avoid if n.label() not in skip] or [0]
        return (max(lv), sum(lv))

    def affordable(self, L: Line, i: str) -> bool:
        p = self.ep.price(i)
        if p is None or self.budget is None:
            return True
        return p * L.qty <= self.budget + 1e-6 if not self.budget_each else p <= self.budget + 1e-6

    def row_score(self, L: Line, i: str) -> tuple:
        txt = roots(self.ep.title(i) + " " + self.record_text(i))
        ev = sum(1 for n in L.needs if n.rel in ("is", "any_of") and roots(n.value) and roots(n.value) <= txt)
        ok = self.ep.in_stock(i) and self.affordable(L, i) and i not in self.dead
        p = self.ep.price(i) or 0
        fit = 0
        if L.fits_line is not None and L.fits_line < len(self.lines):
            A = self.lines[L.fits_line]
            fit = int(any(self.fits_code(i, a) for a in A.cands if a in self.ep.facts and self.quality(A, a)[0] <= self.LEVEL["near"]))
        elif self.parts_of(L):
            fit = self.support(L, i)
        return (not ok, -fit, -L.vhits.get(i, 0), -ev, p if self.cheapest else 0, L.vpos.get(i, 99), p)

    def shelf_word(self, L: Line) -> str:
        src = L.shelf or L.kinds[0]
        ws = [w for w in words(src) if len(w) > 3]
        return ws[0] if ws else (words(src) or [""])[0]

    def value_search(self, L: Line, room: int) -> None:
        ep = self.ep
        if not ep.contract.roles.get("search_text"):
            return
        vals = []
        for n in L.needs:
            v = str(n.value or "")
            if n.rel in ("is", "any_of") and 0 < len(v) <= 30 and n.prop not in ("seller",):
                vals.append(" ".join(words(v)[:3]))
        vals = [v for v in dict.fromkeys(vals) if v]
        if not vals:
            return
        scoped = bool((ep.contract.roles.get("search_text") or {}).get("shelf") and L.shelf) and self.scope_ok is not False
        noun = "" if scoped else self.shelf_word(L)
        combos = [vals]
        if len(vals) > 1:
            combos += [[v for v in vals if v != x] for x in vals]
            combos += [[v] for v in vals]
        start = ep.wire.step
        for combo in combos:
            if ep.wire.step - start >= room or ep.steps_left() <= self.reserve() + 2:
                break
            q = " ".join(combo + ([noun] if noun else []))
            if q in self.vq:
                for pos, i in enumerate(self.vq[q]):
                    if self.belongs(L, i) and i not in L.cands:
                        L.cands.append(i)
                continue
            kw = {"shelf": L.shelf} if scoped else {}
            got = ep.search("search_text", pages=3 if combo is combos[0] else 1, text=q, **kw)
            if scoped and self.scope_ok is None and got:
                shelves = [self.row_shelf(i) for i in got if self.row_shelf(i)]
                self.scope_ok = not shelves or sum(1 for x in shelves if x == L.shelf) >= 0.8 * len(shelves)
                if not self.scope_ok:
                    scoped, noun = False, self.shelf_word(L)
                    q = " ".join(combo + [noun])
                    got = ep.search("search_text", pages=3 if combo is combos[0] else 1, text=q)
            if not [i for i in got if self.belongs(L, i)] and noun and ep.steps_left() > self.reserve() + 3:
                got = ep.search("search_text", pages=2, text=" ".join(combo))
            self.vq[q] = got
            for pos, i in enumerate(got):
                if self.belongs(L, i):
                    if i not in L.cands:
                        L.cands.append(i)
                    if len(combo) > L.vhits.get(i, 0):
                        L.vhits[i], L.vpos[i] = len(combo), pos
                    elif len(combo) == L.vhits.get(i, 0):
                        L.vpos[i] = min(L.vpos.get(i, pos), pos)
            full = [i for i in L.cands if L.vhits.get(i, 0) >= len(vals) and ep.in_stock(i) and self.affordable(L, i)]
            if len(full) >= 3:
                break

    def read_some(self, L: Line, n: int) -> int:
        ep = self.ep
        done = 0
        for _ in range(n):
            if ep.steps_left() <= self.reserve():
                break
            todo = [i for i in sorted(L.cands, key=lambda i: self.row_score(L, i)) if i not in ep.facts and i not in ep.opened and i not in self.dead]
            if not todo:
                break
            got = ep.read_many(todo[:ep.per_step()])
            if not got:
                break
            done += len(got)
        return done

    def reserve(self) -> int:
        return 4 + max(0, self.msg_cap - self.msgs) + (1 if self.confirm else 0) + len(self.ep.extras_when("before_order")) + \
            (len(self.lines) * len(self.ep.extras_when("each_pick")) if not self.carted else 0)

    def menu(self, L: Line) -> list[str]:
        cnt: dict[str, int] = {}
        vals: dict[str, set] = {}
        per: dict[str, dict[str, str]] = {}
        for i in L.cands:
            real = {keyname(k) for k in (self.ep.facts.get(i) or {})}
            for k, v in self.fact_keys(i).items():
                if k not in real or re.search(NOT_ASK, k) or len(k) > 40:
                    continue
                cnt[k] = cnt.get(k, 0) + 1
                vals.setdefault(k, set()).update(x.lower() for x in v[:3])
                per.setdefault(i, {})[k] = " ".join(sorted(str(x).lower() for x in v[:3]))
        def gone(n: Need, k: str) -> bool:
            if self.same_key(n.prop, k) or roots(n.prop) == roots(k):
                return True
            if not (self.key_match(n.prop, k) or self.key_match(k, keyname(n.prop))):
                return False
            rv = roots(n.value)
            located = [j for j in cnt if rv and any(rv <= roots(x) or (roots(x) and roots(x) <= rv) for x in vals[j])]
            return not located or k in located
        keys = [k for k in cnt if not any(gone(n, k) for n in L.needs)
                and not any(self.key_match(f, k) or self.key_match(k, keyname(f)) for f in L.asked)]
        keys.sort(key=lambda k: (-cnt[k], -len(vals[k]), k))
        kept: list[str] = []
        for k in keys:
            def agree(j: str) -> bool:
                both = [d for d in per.values() if j in d and k in d]
                return len(both) >= 2 and sum(d[j] == d[k] for d in both) >= 0.8 * len(both)
            if not any(roots(j) & roots(k) and agree(j) for j in kept):
                kept.append(k)
        keys = kept
        self._menu_info = getattr(self, "_menu_info", {})
        self._menu_info[L.n] = {k: (cnt[k], sorted(vals[k])[:4]) for k in keys[:25]}
        return keys[:25]

    def rank_keys(self) -> None:
        info = {}
        for L in self.lines:
            L.order = self.menu(L)
            L.cues = []
            info[str(L.n)] = {"product": L.item, "properties": L.order}
        got = self.mind.ask(CUE_SYS, "REQUEST:\n" + self.ep.request + "\n\nPRODUCTS:\n" + jdump(info, 8000), 1500, think=True)
        cues = (got or {}).get("cues") if isinstance(got, dict) else None
        if isinstance(cues, dict):
            for L in self.lines:
                c = cues.get(str(L.n)) or cues.get(L.n)
                if isinstance(c, list):
                    L.cues = [k for k in dict.fromkeys(c) if isinstance(k, str) and k in L.order][:1]
                    L.order = L.cues + [k for k in L.order if k not in L.cues]
            more = (got or {}).get("more") if isinstance(got, dict) else None
            for L in self.lines:
                extra = (more or {}).get(str(L.n)) or (more or {}).get(L.n) if isinstance(more, dict) else None
                if isinstance(extra, list) and len(L.order) < 6:
                    for k in extra[:6]:
                        if isinstance(k, str) and k.strip() and not re.search(NOT_ASK, keyname(k)) and \
                                not any(self.key_match(k, x) or self.key_match(x, keyname(k)) for x in L.order + [n.prop for n in L.needs]):
                            L.order.append(keyname(k))

    def pick_line(self) -> Line | None:
        pool = [L for L in self.lines if any(k not in L.asked for k in L.order)]
        if not pool:
            return None
        return min(pool, key=lambda L: (L.found > 0, L.msgs, L.n))

    def interview(self) -> None:
        ep = self.ep
        queue: dict[int, list[list[str]]] = {L.n: [] for L in self.lines}
        refills = 0
        while self.msgs < self.msg_cap - (1 if self.confirm else 0) and ep.steps_left() > 6:
            L = self.pick_line()
            if L is None and refills < 2 and ep.steps_left() > self.reserve() + 8:
                refills += 1
                for M in self.lines:
                    self.read_some(M, 2)
                    fresh = [k for k in self.menu(M) if k not in M.order]
                    M.order += fresh
                L = self.pick_line()
            if L is None:
                break
            keys = [k for k in (queue[L.n].pop(0) if queue[L.n] else []) if k not in L.asked]
            if not keys:
                free = [k for k in L.order if k not in L.asked]
                if not free:
                    L.asked += [k for k in L.order]
                    continue
                keys = [free[0]]
                nxt = next((k for k in free[1:] if not (roots(k) & roots(free[0]))), None)
                if nxt:
                    keys.append(nxt)
            noun = self.noun(L)
            if len(keys) == 1:
                text = f"What {keys[0]} would you like for the {noun}?"
            else:
                text = f"What {keys[0]} or {keys[1]} would you like for the {noun}?"
            obs = ep.act("message", text=text)
            self.msgs += 1
            L.msgs += 1
            reply = self.reply_text(obs)
            if error_in(obs) and not reply:
                break
            kind, got = self.hear(text, reply, L)
            if kind == "one_at_a_time" and len(keys) > 1:
                queue[L.n] = [[k] for k in keys] + queue[L.n]
                continue
            L.asked += [k for k in keys if k not in L.asked]
            L.asked += [k for k in L.order if k not in L.asked and any(roots(k) <= roots(x) for x in keys)]

    def noun(self, L: Line) -> str:
        return (L.shelf or L.kinds[0] or L.item).lower()

    @staticmethod
    def reply_text(obs: Any) -> str:
        if isinstance(obs, str):
            return obs
        if isinstance(obs, dict):
            strs = [v for v in obs.values() if isinstance(v, str)]
            return max(strs, key=len) if strs else jdump(obs, 800)
        return jdump(obs, 800)

    def hear(self, question: str, reply: str, L: Line) -> tuple[str, list]:
        prods = {str(x.n): x.item for x in self.lines}
        got = self.mind.ask(HEAR_SYS, f"PRODUCTS: {jdump(prods)}\nQUESTION: {question}\nREPLY: {reply}", 500) or {}
        kind = got.get("kind") if got.get("kind") in ("need", "no_preference", "confused", "one_at_a_time", "other") else "other"
        added: list[Need] = []
        if kind != "need":
            return kind, added
        for n in got.get("needs") or []:
            if not isinstance(n, dict) or not n.get("property"):
                continue
            idx = n.get("product")
            T = self.lines[idx] if isinstance(idx, int) and 0 <= idx < len(self.lines) else L
            ref = n.get("from_listing") if isinstance(n.get("from_listing"), dict) else None
            need = Need(n["property"], n.get("value"), n.get("relation") or "is", "revealed", ref)
            added.append(need)
            if ref:
                v = self.trail(T, need)
                if v is None:
                    continue
                need.value, need.rel = v, "is"
            if need.value in (None, ""):
                continue
            if any(self.key_match(need.prop, keyname(o.prop)) and roots(o.value) == roots(need.value) for o in T.needs):
                T.found += 1
                continue
            T.needs.append(need)
            T.found += 1
        return ("need" if added else "other"), added

    def trail(self, L: Line, need: Need, depth: int = 0) -> Any:
        ep = self.ep
        ref = need.ref or {}
        seller = str(ref.get("seller") or "").lower()
        details = [d for d in (ref.get("details") or []) if isinstance(d, list) and len(d) == 2]
        kind = str(ref.get("kind") or L.kinds[0])
        sk = ep.shapes.keys.get("seller") or ""
        def by_seller(i: str) -> bool:
            return not seller or seller in str(ep.listings[i].get(sk, "")).lower() or seller in self.record_text(i).lower()
        def fit(i: str) -> int:
            txt = roots(ep.title(i) + " " + self.record_text(i) + " " + " ".join(str(v) for vs in self.fact_keys(i).values() for v in vs))
            return sum(1 for d in details if roots(d[1]) and roots(d[1]) <= txt)
        pool = [i for i in ep.listings if by_seller(i)]
        if (not pool or max(fit(i) for i in pool) < len(details)) and ep.contract.roles.get("search_text") and ep.steps_left() > self.reserve() + 3:
            for q in (" ".join([kind] + [str(d[1]) for d in details[:3]]), " ".join([str(d[1]) for d in details[:2]] + [seller])):
                got = ep.search("search_text", pages=1, text=q[:60])
                pool += [i for i in got if i not in pool and by_seller(i)]
                if pool and max(fit(i) for i in pool) >= len(details):
                    break
        pool.sort(key=lambda i: -fit(i))
        best = None
        for i in pool[:2]:
            if ep.steps_left() <= self.reserve() + 1:
                break
            if not self.paged.get(i):
                ep.read(i, fresh=True)
                self.paged[i] = True
            score = fit(i) + sum(1 for d in details if self.check(i, Need(d[0], d[1])) in ("yes", "near", "text"))
            if best is None or score > best[0]:
                best = (score, i)
        if not best:
            return None
        i = best[1]
        text = self.record_text(i)
        m = re.search(r"([^.]*\b" + re.escape(words(need.prop)[0] if words(need.prop) else "x") + r"\b[^.]*\b(?:take|look at|see|from)\b[^.]*)", text, re.I)
        if m and depth < 2 and re.search(r"take|look at|see|taken from", m.group(1), re.I):
            nxt = self.mind.ask(HEAR_SYS, f"PRODUCTS: {jdump({'0': L.item})}\nQUESTION: where does {need.prop} come from?\nREPLY: {m.group(1)}", 400) or {}
            for n in nxt.get("needs") or []:
                if isinstance(n, dict) and isinstance(n.get("from_listing"), dict):
                    return self.trail(L, Need(need.prop, None, "is", "reference", n["from_listing"]), depth + 1)
        fk = self.fact_keys(i)
        keys = [k for k in fk if self.key_match(need.prop, k)]
        return fk[keys[0]][0] if keys and fk[keys[0]] else None

    def parse_fits(self) -> None:
        ids = [i for L in self.lines for i in L.cands if i in self.ep.facts or i in self.ep.listings]
        ids = [i for i in ids if i not in self.fit and re.search(r"\d", self.record_text(i))][:24]
        if not ids:
            return
        recs = {i: (self.ep.title(i) + " | " + self.record_text(i))[:500] for i in ids}
        got = self.mind.ask(FIT_SYS, jdump(recs, 12000), 1500) or {}
        for i, v in ((got.get("listings") or {}) if isinstance(got, dict) else {}).items():
            if isinstance(v, dict):
                self.fit[i] = v

    FIT_WORDS = r"\b(?:works? with|compatible with|fits?|goes with|suits|for use with|designed for|made for|pairs? with)\b"
    CODE_RX = r"\b(?=[\w-]*\d)(?=[\w-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b"
    ALIAS_RX = r"\b[A-Z][A-Za-z]+ \d{2,5}\b"

    @staticmethod
    def cnorm(c: str) -> str:
        return re.sub(r"[^a-z0-9]", "", c.lower())

    def fit_facts(self, i: str) -> dict:
        text = self.ep.title(i) + " . " + self.record_text(i)
        cache = getattr(self, "_ff", {})
        self._ff = cache
        if cache.get(i, (None,))[0] == len(text):
            return cache[i][1]
        ids, strong, claims, pointers, excl = set(), set(), set(), [], set()
        text = re.sub(r"\b(no|nr|ref|art|approx|incl)\.\s", r"\1 ", text, flags=re.I)
        for sent in re.split(r"(?<=[.;!?|])\s+|\n", text):
            m = re.search(self.FIT_WORDS, sent, re.I)
            if m:
                after = sent[m.end():]
                cut = re.search(r"\b(?:except|excluding|but not|not for|not with|other than)\b", after, re.I)
                if cut:
                    excl |= {self.cnorm(c) for c in re.findall(self.CODE_RX, after[cut.end():])}
                    after = after[:cut.start()]
                got = {self.cnorm(c) for c in re.findall(self.CODE_RX, after)} | {self.cnorm(a) for a in re.findall(self.ALIAS_RX, after)}
                before = [self.cnorm(c) for c in re.findall(self.CODE_RX, sent[:m.start()])]
                if re.search(r"\bpart\b", sent[:m.start()], re.I) and before and re.search(r"\balso\b", sent, re.I):
                    pointers.append((set(before), got))
                else:
                    claims |= got
                continue
            codes = {self.cnorm(c) for c in re.findall(self.CODE_RX, sent)}
            ids |= codes
            if re.search(r"\b(?:model|unit|code|ref|part|sku|no\.)", sent, re.I):
                strong |= codes
            for a in re.findall(r"also sold as ([^.;]+)", sent, re.I):
                strong |= {self.cnorm(x) for x in re.findall(self.ALIAS_RX, a)} | {self.cnorm(a)}
        out = {"ids": strong or ids, "claims": {c for c in claims if len(c) >= 4}, "pointers": pointers, "except": excl}
        cache[i] = (len(text), out)
        return out

    def fits_code(self, part: str, anchor: str) -> bool | None:
        A, P = self.fit_facts(anchor), self.fit_facts(part)
        names = {c for c in A["ids"] if len(c) >= 4}
        if not names:
            return None
        if names & P["except"] or any(self.code_match(c, n) for c in P["except"] if re.search(r"\dx|xx", c) for n in names):
            return False
        if names & P["claims"] or any(self.code_match(c, n) for c in P["claims"] if re.search(r"\dx|xx", c) for n in names):
            return True
        for j in self.ep.listings:
            for parts, anchors in self.fit_facts(j)["pointers"] if j in self.ep.facts or j == part else []:
                if parts & P["ids"] and names & anchors:
                    return True
        return False if P["claims"] else None

    @staticmethod
    def code_match(pattern: str, code: str) -> bool:
        pat = re.sub(r"\s+", "", pattern.lower())
        c = re.sub(r"\s+", "", code.lower())
        if pat == c:
            return True
        rx = re.escape(pat).replace("x", r"\w")
        return bool(re.fullmatch(rx, c)) and "x" in pat

    def fits(self, part: str, anchor: str) -> bool | None:
        by_code = self.fits_code(part, anchor)
        if by_code is not None:
            return by_code
        a, p = self.fit.get(anchor) or {}, self.fit.get(part) or {}
        names = [str(x) for x in (a.get("is_model") or [])]
        if not names:
            return None
        claims = [str(x) for x in (p.get("fits") or [])]
        pc = str(p.get("part_code") or "")
        for other in self.fit.values():
            for af in other.get("also_fits") or []:
                if isinstance(af, dict) and pc and str(af.get("part") or "").lower() == pc.lower():
                    claims += [str(x) for x in af.get("models") or []]
        if not claims:
            return None
        ok = False
        for c in claims:
            exc = re.findall(r"except\s+([\w-]+)", c, re.I)
            base = re.sub(r",?\s*except.*", "", c, flags=re.I).strip()
            for n in names:
                if self.code_match(base, n) or (base and n.lower() in base.lower()) or (n and base.lower() in n.lower() and len(base) > 3):
                    if not any(self.code_match(e, n) for e in exc):
                        ok = True
        return ok

    def candidates(self, L: Line, strict: bool = True) -> list[str]:
        out = []
        for i in L.cands:
            if i in self.dead or i not in self.ep.facts and strict:
                continue
            if not self.ep.in_stock(i) or not self.affordable(L, i):
                continue
            no, unk, txt = self.status(L, i)
            if no == 0:
                out.append(i)
        def key(i: str) -> tuple:
            return self.quality(L, i) + (self.ep.price(i) or 0,)
        out.sort(key=key)
        if self.cheapest and out:
            best = key(out[0])[0]
            sure = [i for i in out if key(i)[0] == best]
            out = sorted(sure, key=lambda i: self.ep.price(i) or 0) + [i for i in out if i not in sure]
        return out

    def choose(self) -> list[tuple[str, int]] | None:
        pools = [self.candidates(L) or self.candidates(L, strict=False) for L in self.lines]
        if any(not p for p in pools):
            return None
        tiered = []
        for L, pool in zip(self.lines, pools):
            best = min(self.quality(L, i)[0] for i in pool)
            tiered.append([i for i in pool if self.quality(L, i)[0] == best] + [i for i in pool if self.quality(L, i)[0] != best])
        pools = tiered
        for k, L in enumerate(self.lines):
            if self.parts_of(L):
                pools[k] = sorted(pools[k], key=lambda i, L=L: (self.quality(L, i)[0], -self.support(L, i)))
        first = [[i for i in pool if self.quality(L, i)[0] == self.quality(L, pool[0])[0]] for L, pool in zip(self.lines, pools)]
        anchor_of = {L.n: L.fits_line for L in self.lines if L.fits_line is not None}
        best = None
        def rec(k: int, picked: list[str], total: float) -> None:
            nonlocal best
            if best is not None and self.cheapest is False and len(picked) == len(self.lines):
                return
            if k == len(self.lines):
                if best is None or total < best[0]:
                    best = (total, list(picked))
                return
            a = anchor_of.get(k)
            for i in pools[k][:16 if a is not None or self.parts_of(self.lines[k]) else 8]:
                if i in picked:
                    continue
                if a is not None and a < k:
                    f = self.fits(i, picked[a])
                    if f is False or (f is None and any(self.fits_code(j, picked[a]) for j in pools[k][:16])):
                        continue
                cost = (self.ep.price(i) or 0) * self.lines[k].qty
                if self.budget is not None and not self.budget_each and total + cost > self.budget + 1e-6:
                    continue
                rec(k + 1, picked + [i], total + cost)
        full_pools = pools
        pools = first
        rec(0, [], 0.0)
        if best is None:
            pools = full_pools
            rec(0, [], 0.0)
        if best is None:
            return None
        return [(i, self.lines[n].qty) for n, i in enumerate(best[1])]

    def weak(self, pick: list[tuple[str, int]] | None) -> bool:
        return pick is None or self.rank_pick(pick)[0] >= self.LEVEL["part"]

    def settled(self, pick: list[tuple[str, int]]) -> bool:
        return all(self.quality(L, i)[0] == 0 for L, (i, q) in zip(self.lines, pick))

    def rank_pick(self, pick: list[tuple[str, int]]) -> tuple:
        worst = max(self.quality(L, i)[0] for L, (i, q) in zip(self.lines, pick))
        total = sum(self.quality(L, i)[1] for L, (i, q) in zip(self.lines, pick))
        cost = sum((self.ep.price(i) or 0) * q for i, q in pick)
        return (worst, total, cost if self.cheapest else 0)

    def closest(self) -> str | None:
        L = self.lines[0]
        ep = self.ep
        def good() -> list[str]:
            return [i for i in L.cands if i in ep.facts and i not in self.dead and ep.in_stock(i)
                    and self.quality(L, i)[0] <= self.LEVEL["near"]]
        tries = 0
        while not good() and tries < 3 and ep.steps_left() > self.reserve() + 1:
            rows = sorted([i for i in L.cands if i not in ep.facts and i not in ep.opened and i not in self.dead and ep.in_stock(i)],
                          key=lambda i: (-L.vhits.get(i, 0), L.vpos.get(i, 99), ep.price(i) or 0))
            if not rows or not ep.read_many(rows[:ep.per_step()]):
                break
            tries += 1
        pool = good()
        if pool:
            return min(pool, key=lambda i: (self.quality(L, i), ep.price(i) or 0))
        pool = [i for i in L.cands if i in ep.facts]
        if not pool:
            return None
        return min(pool, key=lambda i: (self.status(L, i)[0], self.quality(L, i), ep.price(i) or 0))

    def absorb_notes(self, notes: list[str]) -> None:
        if not notes:
            return
        got = self.mind.ask(NOTE_SYS, "CART: " + jdump({i: self.ep.title(i) for i in self.ep.cart}) + "\nNOTES:\n" + "\n".join(notes), 500) or {}
        for c in got.get("changes") or []:
            if not isinstance(c, dict):
                continue
            kind = c.get("kind")
            target = self.match_listing(str(c.get("listing") or ""))
            if kind == "sold_out":
                for i in ([target] if target else list(self.ep.cart)):
                    self.dead.add(i)
            elif kind == "price_changed":
                for i in ([target] if target else list(self.ep.cart)):
                    before = self.ep.price(i)
                    self.ep.read(i, fresh=True)
                    if self.ep.contract.roles.get("cart_view"):
                        self.ep.absorb_cart(self.ep.act("cart_view"))
                    if self.ep.price(i) == before:
                        self.dead.add(i)
            elif kind == "budget" and num(c.get("budget")) is not None:
                self.budget = num(c.get("budget"))
            elif kind == "dropped" and c.get("property"):
                for L in self.lines:
                    L.needs = [n for n in L.needs if not (self.key_match(c["property"], keyname(n.prop)) or self.key_match(n.prop, keyname(c["property"])))]
                self.avoid = [n for n in self.avoid if not (self.key_match(c["property"], keyname(n.prop)) or self.key_match(n.prop, keyname(c["property"])))]
            elif kind == "requirement" and c.get("property") and c.get("value"):
                for L in self.lines:
                    L.needs = [n for n in L.needs if not self.key_match(c["property"], keyname(n.prop))]
                self.lines[0].needs.append(Need(c["property"], c["value"], "is", "changed"))

    def match_listing(self, s: str) -> str | None:
        if not s:
            return None
        if s in self.ep.listings:
            return s
        r = roots(s)
        best = None
        for i in self.ep.cart or self.ep.listings:
            ov = len(r & roots(self.ep.title(i)))
            if ov and (best is None or ov > best[0]):
                best = (ov, i)
        return best[1] if best else None

    def run(self) -> None:
        ep = self.ep
        ep.start_extras()
        self.brief()
        n = len(self.lines)
        avail = max(6, ep.max_steps - self.reserve())
        for L in self.lines:
            self.census(L, max(3, int(avail * 0.3 / n)))
        self.merge_lines()
        for L in self.lines:
            if L.shelf:
                joint = bool(re.search(r"&|,|\band\b", L.shelf))
                sh = roots(L.shelf)
                L.needs = [x for x in L.needs if not (x.rel == "is" and x.source == "stated" and roots(x.value) and roots(x.value) <= sh and
                                                      (not joint or len(roots(x.value)) >= len(sh) - 1))]
        n = len(self.lines)
        for L in self.lines:
            self.read_some(L, 4 if n == 1 else 3)
            self.remap(L)
        for L in self.lines:
            if len(self.menu(L)) < 3 and ep.steps_left() > self.reserve() + 12:
                self.read_some(L, 3)
                self.remap(L)
        self.rank_keys()
        self.interview()
        self.msg_cap = self.msgs + (1 if self.confirm and self.msgs < self.msg_cap else 0)
        if ep.notices:
            self.absorb_notes(ep.notices)
            ep.notices = []
        if any(L.fits_line is not None for L in self.lines):
            self.parse_fits()
        for k, L in enumerate(self.lines):
            if self.need_reads(L):
                self.value_search(L, max(1, min(4, (ep.steps_left() - self.reserve()) // (3 * max(1, n - k)))))
        for L in self.lines:
            self.remap(L)
        if any(L.fits_line is not None for L in self.lines):
            self.verify_all(keep=self.recovery() + 8)
            self.fit_rows()
        self.verify_all()
        for L in self.lines:
            self.remap(L)
        if self.cheapest:
            for k, L in enumerate(self.lines):
                self.sweep_cheaper(L, max(1, (ep.steps_left() - self.reserve()) // max(1, n - k)))
                self.cap_sweep(L)
        if any(L.fits_line is not None for L in self.lines):
            self.parse_fits()
        for L in self.lines:
            self.judge_llm(L)
        self.finish()

    def merge_lines(self) -> None:
        keep: list[Line] = []
        for L in self.lines:
            twin = next((K for K in keep if K.shelf and K.shelf == L.shelf and K.fits_line == L.fits_line
                         and roots(K.item) & roots(K.shelf) and roots(L.item) & roots(L.shelf)
                         and not any(x.fits_line in (K.n, L.n) for x in self.lines)), None)
            if twin is None:
                keep.append(L)
                continue
            for n in L.needs:
                if not any(n.label() == m.label() for m in twin.needs):
                    twin.needs.append(n)
            twin.cands += [i for i in L.cands if i not in twin.cands]
            twin.qty = max(twin.qty, L.qty)
        if len(keep) < len(self.lines):
            old = {L.n: k for k, L in enumerate(keep)}
            for k, L in enumerate(keep):
                L.n = k
                L.fits_line = old.get(L.fits_line) if L.fits_line is not None else None
            self.lines = keep

    def judge_llm(self, L: Line) -> None:
        top = self.candidates(L, strict=True)[:6]
        needs = L.needs + self.avoid
        todo = [i for i in top if any(self.check(i, n, L) in ("unknown", "part", "text") for n in needs)]
        if not todo or not needs or not self.mind.ok():
            return
        reqs = {str(k + 1): n.label() for k, n in enumerate(needs)}
        recs = {i: {"title": self.ep.title(i), "facts": self.ep.facts.get(i)} for i in todo}
        got = self.mind.ask(JUDGE_SYS, "REQUIREMENTS:\n" + jdump(reqs) + "\n\nLISTINGS:\n" + jdump(recs, 10000), 900) or {}
        for i, vs in ((got.get("verdicts") or {}) if isinstance(got, dict) else {}).items():
            if i not in recs or not isinstance(vs, dict):
                continue
            for k, v in vs.items():
                n = needs[int(k) - 1] if str(k).isdigit() and 0 < int(k) <= len(needs) else None
                if n is not None and v in ("yes", "no", "unknown"):
                    code = self.check(i, n, L)
                    if code in ("unknown", "text") and v == "yes":
                        self.verdict[(i, n.label())] = "near"
                    elif code == "unknown" and v == "no":
                        self.verdict[(i, n.label())] = "no"
                    elif code in ("part", "text") and v == "no":
                        self.verdict[(i, n.label())] = "no"

    def need_reads(self, L: Line) -> bool:
        c = self.candidates(L)
        return not c or self.status(L, c[0])[1] > 0 or self.status(L, c[0])[2] > 0

    def parts_of(self, L: Line) -> list[Line]:
        return [M for M in self.lines if M.fits_line == L.n and M is not L]

    def support(self, A: Line, a: str) -> int:
        cache = getattr(self, "_sup", {})
        self._sup = cache
        key = (A.n, a, sum(len(M.cands) for M in self.parts_of(A)), len(self.ep.facts), len(self.dead), self.budget)
        if key not in cache:
            lines = rows = 0
            left = None if self.budget is None or self.budget_each else self.budget - (self.ep.price(a) or 0) * A.qty
            def usable(M: Line, j: str) -> bool:
                if j in self.dead or not self.ep.in_stock(j):
                    return False
                if left is not None and (self.ep.price(j) or 0) * M.qty > left + 1e-6:
                    return False
                return j not in self.ep.facts or self.quality(M, j)[0] <= self.LEVEL["near"]
            for M in self.parts_of(A):
                n = sum(1 for j in M.cands if usable(M, j) and self.fits_code(j, a))
                lines += n > 0
                rows += min(n, 3)
            cache[key] = 10 * lines + min(rows, 9)
        return cache[key]

    def verified(self, L: Line) -> list[str]:
        out = [i for i in self.candidates(L) if self.quality(L, i)[0] == 0]
        parts = self.parts_of(L)
        if parts and out:
            out = [i for i in out if self.support(L, i) // 10 == len(parts)]
        if L.fits_line is not None and L.fits_line < len(self.lines) and out:
            A = self.lines[L.fits_line]
            anchors = [a for a in self.candidates(A) if self.quality(A, a)[0] <= self.LEVEL["near"]]
            out = [i for i in out if any(self.fits(i, a) is True for a in anchors)]
        return out

    def cap_sweep(self, L: Line) -> None:
        ep = self.ep
        spec = ep.contract.roles.get("search_shelf") or {}
        if not (self.cheapest and L.shelf and spec.get("cap")):
            return
        want = set().union(*[roots(n.value) for n in L.needs if n.rel in ("is", "any_of")] or [set()]) - self.pwords(L)
        best = [i for i in self.candidates(L) if self.quality(L, i)[0] <= self.LEVEL["near"]]
        if not best:
            return
        top = min(ep.price(i) or 0 for i in best)
        cap = top - 0.01
        for _ in range(3):
            if cap <= 0 or ep.steps_left() <= self.reserve() + self.recovery() + 3:
                break
            got = ep.search("search_shelf", pages=3, shelf=L.shelf, cap=cap)
            total = ep.last_total
            rows = [i for i in got if self.belongs(L, i) and i not in self.dead and ep.in_stock(i) and (ep.price(i) or 0) < top]
            L.cands += [i for i in rows if i not in L.cands]
            todo = sorted([i for i in rows if i not in ep.facts and i not in ep.opened],
                          key=lambda i: (-len(want & roots(ep.title(i) + " " + self.record_text(i))), ep.price(i) or 0))
            found = None
            for i in todo:
                if ep.steps_left() <= self.reserve() + self.recovery() + 2:
                    break
                if not want & roots(ep.title(i) + " " + self.record_text(i)) and found is None and len(todo) > 8:
                    break
                ep.read_many([i])
                if self.quality(L, i)[0] <= self.LEVEL["near"] and self.status(L, i)[0] == 0:
                    found = i
                    break
            if found:
                top = ep.price(found) or top
                cap = top - 0.01
                continue
            if total is not None and total <= len(got):
                break
            cap = cap / 2

    def sweep_cheaper(self, L: Line, room: int) -> None:
        ep = self.ep
        v = self.verified(L)
        if not v:
            return
        cap = ep.price(v[0]) or 0
        rows = [i for i in L.cands if i not in ep.facts and i not in ep.opened and i not in self.dead and ep.in_stock(i) and (ep.price(i) or 0) < cap]
        rows.sort(key=lambda i: self.row_score(L, i))
        for i in rows[:room]:
            if ep.steps_left() <= self.reserve() + self.recovery():
                break
            ep.read(i)

    def final_room(self) -> int:
        return 2 + 2 * len(self.lines)

    def recovery(self) -> int:
        return 0 if self.carted else 2 + 2 * len(self.lines)

    def fit_rows(self) -> None:
        ep = self.ep
        spec = ep.contract.roles.get("search_shelf") or {}
        for M in self.lines:
            if M.fits_line is None or M.fits_line >= len(self.lines) or not M.shelf or ep.steps_left() <= self.reserve() + self.recovery() + 4:
                continue
            A = self.lines[M.fits_line]
            anchors = sorted([a for a in A.cands if a in ep.facts and self.quality(A, a)[0] <= self.LEVEL["near"]],
                             key=lambda a: -self.support(A, a))[:4]
            if not anchors:
                continue
            if any(self.fits_code(j, a) for a in anchors for j in M.cands if self.affordable(M, j) and ep.in_stock(j) and j not in self.dead):
                continue
            left = (self.budget - min(ep.price(a) or 0 for a in anchors)) / max(1, M.qty) if self.budget else None
            got = ep.search("search_shelf", pages=3, shelf=M.shelf, cap=left if spec.get("cap") else None)
            M.cands += [j for j in got if j not in M.cands and self.belongs(M, j)]

    def verify_all(self, keep: int | None = None) -> None:
        ep = self.ep
        keep = (3 if self.cheapest else 0) + self.recovery() if keep is None else keep
        while ep.steps_left() > self.reserve() + keep:
            todo = [L for L in self.lines if not self.verified(L)]
            if not todo:
                break
            if not sum(self.read_some(L, 1) for L in todo):
                break

    def read_cheaper(self, L: Line) -> int:
        ep = self.ep
        if ep.steps_left() <= self.reserve() + 2:
            return 0
        pool = lambda: [i for i in L.cands if i not in ep.facts and i not in ep.opened and i not in self.dead and ep.in_stock(i) and self.affordable(L, i)]
        if not pool():
            self.more_rows(L)
        rows = sorted(pool(), key=lambda i: (-L.vhits.get(i, 0), ep.price(i) or 0))
        return len(ep.read_many(rows[:ep.per_step()])) if rows and ep.steps_left() > self.reserve() + 2 else 0

    def more_rows(self, L: Line) -> None:
        ep = self.ep
        vals = [" ".join(words(str(n.value))[:3]) for n in L.needs if n.rel in ("is", "any_of") and 0 < len(str(n.value)) <= 30]
        before = len(L.cands)
        if vals and ep.contract.roles.get("search_text") and ep.steps_left() > self.reserve() + 4:
            got = ep.search("search_text", pages=3, text=" ".join(list(dict.fromkeys(vals)) + [self.shelf_word(L)]))
            L.cands += [i for i in got if i not in L.cands and self.belongs(L, i)]
        if len(L.cands) == before and L.shelf and ep.contract.roles.get("search_shelf") and ep.steps_left() > self.reserve() + 4:
            got = ep.search("search_shelf", pages=3, shelf=L.shelf)
            L.cands += [i for i in got if i not in L.cands and self.belongs(L, i)]

    def verify(self, L: Line, room: int) -> None:
        ep = self.ep
        start = ep.wire.step
        if self.verified(L):
            return
        open_needs = [n for n in L.needs if n.rel in ("is", "any_of") and len(str(n.value)) < 40 and n.source != "stated"]
        for n in open_needs[:2]:
            if ep.wire.step - start >= room // 2 or not ep.contract.roles.get("search_text"):
                break
            got = ep.search("search_text", pages=1, text=f"{n.value} {L.kinds[0]}")
            L.cands += [i for i in got if i not in L.cands and self.belongs(L, i)]
        while not self.verified(L) and ep.wire.step - start < room and ep.steps_left() > self.reserve():
            if not self.read_some(L, 2):
                break

    def finish(self) -> None:
        ep = self.ep
        for attempt in range(4):
            if ep.wire.finished:
                return
            self.absorb_notes(ep.notices)
            ep.notices = []
            pick = self.choose()
            if pick is None:
                for L in self.lines:
                    if not self.candidates(L) and ep.steps_left() > self.reserve() + 2:
                        self.read_some(L, max(1, ep.steps_left() - self.reserve() - 1))
                pick = self.choose()
            while self.weak(pick) and ep.steps_left() > self.reserve() + 2:
                if pick is None:
                    got = sum(self.read_cheaper(L) for L in self.lines)
                else:
                    got = 0
                    for L, (i, q) in zip(self.lines, pick):
                        if self.quality(L, i)[0] >= self.LEVEL["part"]:
                            if not [j for j in L.cands if j not in ep.facts and j not in ep.opened and j not in self.dead]:
                                self.more_rows(L)
                            got += self.read_some(L, 1)
                if not got:
                    break
                pick = self.choose()
            if self.weak(pick):
                pick = None
            if pick is None:
                close = self.closest() if self.decline_rule == "name_closest" else None
                if self.decline_rule == "name_closest" and close:
                    reason = f"Nothing in the store meets every requirement. The closest listing is {close} ({ep.title(close)[:80]})."
                else:
                    reason = "Nothing in the store meets every requirement, so I am not buying anything."
                obs = ep.finish_decline(reason)
                ref = refusal_in(obs)
                if ref:
                    ep.notices += ref
                    continue
                return
            if self.carted and not self.settled(pick) and ep.steps_left() > self.reserve() + self.final_room():
                for L in self.lines:
                    if not self.verified(L) and not [i for i in L.cands if i not in ep.facts and i not in ep.opened and i not in self.dead]:
                        self.more_rows(L)
                self.verify_all(keep=self.final_room())
                pick = self.choose() or pick
            bad = ep.check_picks([i for i, q in pick])
            if bad:
                self.dead |= bad
                continue
            pages = [d for d in (ep.contract.roles.get("detail") or []) if d.get("gives") in ("page", "all")]
            unseen = [i for i, q in pick if not self.paged.get(i)]
            if pages and unseen and not self.carted and ep.steps_left() > self.reserve() + len(unseen) + 3:
                for i in unseen:
                    ep.read(i, fresh=True)
                    self.paged[i] = True
                pick = self.choose() or pick
            if dict(pick) != ep.cart:
                edits = 1 if ep.contract.roles.get("cart_set") else sum(1 + (1 if ep.cart.get(i) else 0) for i, q in pick if ep.cart.get(i) != q) + \
                    sum(1 for i in ep.cart if i not in dict(pick))
                if ep.steps_left() < edits + 1:
                    if ep.cart and not any(i in self.dead for i in ep.cart):
                        ep.finish_order()
                    if not ep.wire.finished:
                        ep.finish_decline("Nothing in the store meets every requirement, so I am not buying anything.")
                    return
                ep.put_cart(pick)
            first = not self.carted
            self.carted = True
            if ep.notices:
                continue
            if first and not self.polished and (self.cheapest or not self.settled(pick)):
                self.polished = True
                self.verify_all(keep=self.final_room())
                if self.cheapest:
                    for L in self.lines:
                        self.sweep_cheaper(L, max(0, ep.steps_left() - self.reserve() - self.final_room()))
                better = self.choose()
                if better and ep.check_picks([i for i, q in better]):
                    self.dead |= {i for i, q in better if ep.extra_ok.get(i) is False}
                    better = None
                if better and dict(better) != ep.cart and self.rank_pick(better) < self.rank_pick(pick):
                    ep.put_cart(better)
                    pick = better
                    if ep.notices:
                        continue
            if self.confirm and self.msgs < self.msg_cap:
                summary = "; ".join(f"{q} x {ep.title(i)[:50]}" for i, q in pick)
                ep.act("message", text=f"My final basket is: {summary}. I will check it out now.")
                self.msgs += 1
                if ep.notices:
                    continue
            obs = ep.finish_order()
            if ep.wire.finished:
                return
            ref = refusal_in(obs)
            ep.notices += ref or ([error_in(obs)] if error_in(obs) else [])
            if not ref:
                ep.sync_cart()
        if not ep.wire.finished:
            if ep.cart:
                ep.finish_order()
            if not ep.wire.finished:
                ep.finish_decline("I could not complete this request.")
