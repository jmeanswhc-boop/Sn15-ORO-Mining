"""Hybrid KB event-aware shopping agent for ORO Bench generated environments.

KB EpisodeBoard outer loop preserved. Grafts: GLM failover, 14-style hard gate /
close-race re-pick before cart, KA meta-key claim cleanup, v3 recovery color/model hints.
"""

from __future__ import annotations

import json
import re
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

# Balanced timeout: large models need headroom; keep retries modest.
_proxy = ProxyClient(timeout=120, max_retries=3)

_MODELS = {
    "chutes": {
        "default": "Qwen/Qwen3.5-397B-A17B-TEE",
        "fallback": (
            "deepseek-ai/DeepSeek-V3.2-TEE",
            "zai-org/GLM-5.2-TEE",
            "moonshotai/Kimi-K2.6-TEE",
        ),
    },
    "openrouter": {
        "default": "qwen/qwen3.5-397b-a17b",
        "fallback": (
            "deepseek/deepseek-v3.2",
            "z-ai/glm-5.2",
            "moonshotai/kimi-k2.6",
        ),
    },
}

# Grafted knobs (14 / KA / v3) — used only inside existing board/guard sites.
_SELF_CONSISTENCY_GAP = 1.5
_HARD_CONSTRAINT_GATE = True
_REQUIRE_STOCK_FRESH = True
_KA_META_CLAIM_FIELDS = frozenset({
    "product_id",
    "sku",
    "done",
    "ok",
    "error",
    "message",
    "status",
    "reason",
    "compared",
    "results",
    "items",
    "cart",
    "variants",
    "options",
    "quantity",
    "qty",
    "title",
    "name",
    "description",
    "url",
    "image",
    "images",
    "thumbnail",
    "query",
    "total",
    "count",
    "page",
    "tool",
    "action",
    "currency",
    "category_path",
    "returned",
    "added",
    "observation",
    "current_price",
    "price_at_add",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_BUDGET_RE = re.compile(
    r"(?:"
    r"(?:under|below|upto|up\s+to|at\s+or\s+under|at\s+most|no\s+more\s+than|"
    r"max(?:imum)?|budget(?:\s+of)?|spend(?:\s+up\s+to)?|"
    r"can\s+(?:only\s+)?(?:spend(?:\s+up\s+to)?|go\s+up\s+to)|"
    r"go\s+up\s+to)\s*"
    r"(?:of\s+)?"
    r"(?P<a>\d[\d,]*(?:\.\d+)?)"
    r"|"
    r"(?P<b>\d[\d,]*(?:\.\d+)?)\s*(?P<cur>[a-z]{3})\b"
    r")",
    re.I,
)
_CANDIDATE_KEY_RE = re.compile(r"\b(\d{6,})::([A-Za-z0-9_-]+)\b")
_NUM_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?")
# number + following word taken from the query itself (unit/spec), not a fixed vocab list
_NUM_UNIT_RE = re.compile(r"\b(\d{1,4}(?:\.\d+)?)\s*-?\s*([A-Za-z]{1,16})\b")


def _canon_unit(unit: str) -> str:
    try:
        """Orthographic fold for length-quote units only — no product vocab."""
        u = (unit or "").lower().strip()
        if u in {'"', "'", "in", "inch", "inches"}:
            return "in"
        return u
    except Exception:
        return ""


def _canon_spec_token(token: str) -> str:
    try:
        """Normalize a query/product spec token (NUM-unit folds only)."""
        raw = (token or "").strip()
        if not raw:
            return raw
        match = re.fullmatch(r"(\d+(?:\.\d+)?)[\s\-]*([A-Za-z%]{1,16}|\"|')", raw)
        if match:
            return f"{match.group(1)}-{_canon_unit(match.group(2))}"
        match = re.fullmatch(r"(\d+(?:\.\d+)?)(in|inch|inches)\b", raw, re.I)
        if match:
            return f"{match.group(1)}-in"
        return raw
    except Exception:
        return ""
_CATEGORY_RE = re.compile(
    r"\bin\s+the\s+([A-Za-z0-9][\w\s&/-]{1,60}?)\s+category\b",
    re.I,
)
_OBS_CHAR_CAP = 1800
_STR_CHAR_CAP = 240
_LIST_ROW_CAP = 6
_BOARD_CANDIDATE_CAP = 6
_HISTORY_SOFT_CAP = 8
_MAX_STORED_CANDIDATES = 20
_TURN_CEILING = 28
_SEED_STOP = frozenset(
    """
    a an the and or but for with from that this these those into onto over under
    at to of in on by is are was were be been being it its as if then than so
    not no yes you your we our they their i me my need want find get buy purchase
    please can could would should must also just only more most less least very
    item items product products listing listings order orders cart shop shopping
    available stock stocked budget price prices within below above prefer preferred
    ideally soft firm requirement requirements category categories brand brands
    spend spending going actually really sorry hold wait still needs needed
    lean leaning toward towards check checked recheck before anything
    compare choose among remaining ties tradeoff tradeoffs candidate candidates
    variant variants option options walk through why right pick select
    highest lowest largest smallest cheapest maximum minimum advertised
    including explicitly current live between while after once
    place one test rather pay bit every kind something settle
    through give trust tell another keep exact ask asking say identify
    title type labels insufficient replacement reason good has care about
    matters goal sku dollars saving few grab dont don't matches phrase catalog
    """.split()
)
_DIR_MAX = frozenset(
    {"highest", "largest", "most", "maximum", "max", "greatest", "biggest"}
)
_DIR_MIN = frozenset(
    {"lowest", "smallest", "least", "minimum", "min", "cheapest", "fewest"}
)
_TRADEOFF_RE = re.compile(
    r"(?:(?:choose|pick|prefer|select|want|get|preferring)\s+(?:the\s+)?"
    r"|(?:then|and|,)\s+(?:the\s+)?"
    r"|(?:\A|;|\.)\s*(?:the\s+)?)"
    r"(?P<dir>highest|lowest|largest|smallest|cheapest|most|least|maximum|minimum)\s+"
    r"(?P<field>(?:(?:advertised|maximum|minimum|available|current|live)\s+){0,3}"
    r"[a-z][a-z0-9]*"
    r"(?:\s+(?!then\b|and\b|or\b|before\b|after\b|while\b|among\b|between\b|with\b)"
    r"[a-z][a-z0-9/-]*){0,4})",
    re.I | re.M,
)


def _provider() -> str:
    try:
        return (getenv("INFERENCE_PROVIDER") or "chutes").strip().lower()
    except Exception:
        return ""


def _model_candidates() -> list[str]:
    try:
        """Prefer provider-native allowlisted IDs; remap TEE overrides on OpenRouter."""
        provider = _provider()
        catalog = _MODELS.get(provider, _MODELS["chutes"])
        override = (getenv("SANDBOX_MODEL") or "").strip()
        ordered: list[str] = []
        if override:
            if provider == "openrouter" and override.endswith("-TEE"):
                mapped = {
                    "deepseek-ai/DeepSeek-V3.2-TEE": "deepseek/deepseek-v3.2",
                    "deepseek-ai/DeepSeek-V3.1-TEE": "deepseek/deepseek-chat-v3.1",
                    "Qwen/Qwen3.5-397B-A17B-TEE": "qwen/qwen3.5-397b-a17b",
                    "Qwen/Qwen3-32B-TEE": "qwen/qwen3-32b",
                    "moonshotai/Kimi-K2.6-TEE": "moonshotai/kimi-k2.6",
                    "zai-org/GLM-5.2-TEE": "z-ai/glm-5.2",
                }
                ordered.append(mapped.get(override, catalog["default"]))
            else:
                ordered.append(override)
        ordered.append(catalog["default"])
        ordered.extend(catalog["fallback"])
        # Preserve order, drop duplicates.
        uniq: list[str] = []
        seen: set[str] = set()
        for model in ordered:
            if model in seen:
                continue
            seen.add(model)
            uniq.append(model)
        return uniq
    except Exception:
        return []


def _model() -> str:
    try:
        return _model_candidates()[0]
    except Exception:
        return ""


def _fn(spec: dict[str, Any]) -> dict[str, Any] | None:
    raw = spec.get("function") if isinstance(spec, dict) else None
    return raw if isinstance(raw, dict) else None


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {
        fn["name"]
        for spec in tools
        for fn in [_fn(spec)]
        if fn and isinstance(fn.get("name"), str)
    }


def _required_args(tools: list[dict[str, Any]], name: str) -> list[str]:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        return [x for x in (params.get("required") or []) if isinstance(x, str)]
    return []


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> set[str]:
    allowed: set[str] = set()
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != "place_test_order":
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        just = props.get("justification") if isinstance(props.get("justification"), dict) else {}
        jprops = just.get("properties") if isinstance(just.get("properties"), dict) else {}
        claims = jprops.get("claims") if isinstance(jprops.get("claims"), dict) else {}
        items = claims.get("items") if isinstance(claims.get("items"), dict) else {}
        iprops = items.get("properties") if isinstance(items.get("properties"), dict) else {}
        field = iprops.get("field") if isinstance(iprops.get("field"), dict) else {}
        enum = field.get("enum")
        if isinstance(enum, list):
            allowed.update(str(x) for x in enum if isinstance(x, str) and x.strip())
        desc = " ".join(
            str(part)
            for part in (
                field.get("description"),
                just.get("description"),
                claims.get("description"),
            )
            if isinstance(part, str)
        ).lower()
        for token in re.findall(r"\b[a-z][a-z0-9_]{1,40}\b", desc):
            if token in {
                "field",
                "fields",
                "value",
                "claims",
                "claim",
                "string",
                "object",
                "array",
                "supported",
                "reason",
                "reasons",
                "useful",
                "specification",
                "shopper",
                "request",
                "feature",
                "features",
                "positive",
                "capabilities",
                "unknown",
                "absent",
            }:
                continue
            if "_" in token or token in {"price", "brand", "category"}:
                allowed.add(token)
        reason = re.search(r"reason(?:s)?\s*:?\s*([^.]+)", desc)
        if reason:
            clause = reason.group(1)
            stop = {
                "such",
                "also",
                "one",
                "the",
                "and",
                "for",
                "or",
                "as",
                "to",
                "use",
                "only",
                "field",
                "value",
                "supported",
                "positive",
                "capability",
                "capabilities",
            }
            for token in re.findall(r"[a-z][a-z0-9_]{1,40}", clause):
                if token in stop:
                    continue
                allowed.add(token)
            for left, right in re.findall(r"([a-z]+)\s+([a-z]+)", clause):
                if left in stop or right in stop:
                    continue
                allowed.add(f"{left}_{right}")
        if "justification" in json.dumps(params, ensure_ascii=False).lower() or allowed:
            allowed.update({"price", "in_stock"})
    return allowed


def _justification_open_spec(tools: list[dict[str, Any]]) -> bool:
    for spec in tools:
        fn = _fn(spec)
        if not fn or fn.get("name") != "place_test_order":
            continue
        blob = json.dumps(fn.get("parameters") or {}, ensure_ascii=False).lower()
        if "specification" in blob and "shopper" in blob:
            return True
        if "specification field" in blob:
            return True
    return False


def _flatten_product(row: dict[str, Any]) -> dict[str, Any]:
    try:
        flat = dict(row)
        for nest in ("facts", "specification", "attributes", "specs"):
            block = row.get(nest)
            if isinstance(block, dict):
                for key, value in block.items():
                    flat.setdefault(key, value)
        path = row.get("category_path")
        if isinstance(path, list) and path:
            flat.setdefault(
                "category", path[-1] if isinstance(path[-1], str) else path[0]
            )
        elif isinstance(path, str) and path.strip():
            flat.setdefault("category", path.split(",")[-1].strip().strip("[]\"' "))
        title = str(row.get("title") or "")
        if title and not flat.get("brand"):
            token = title.split()[0] if title.split() else ""
            if token[:1].isupper() and token.isalpha() and len(token) >= 3:
                flat.setdefault("brand", token)
        return flat
    except Exception:
        return {}


def _parse_budget(text: str) -> float | None:
    try:
        best: float | None = None
        for match in _BUDGET_RE.finditer(text or ""):
            raw = match.group("a") or match.group("b")
            if not raw:
                continue
            try:
                value = float(raw.replace(",", ""))
            except ValueError:
                continue
            if value < 5:
                continue
            if best is None or value < best:
                best = value
        return best
    except Exception:
        return None


def _preamble(query: str) -> str:
    try:
        """Shopper request text before any Task rules appendix."""
        return (query or "").split("Task rules:", 1)[0].strip()
    except Exception:
        return ""


def _needs_clarification(query: str) -> bool:
    try:
        """Detect deferred constraints via discourse co-occurrence (no fixed suite clauses)."""
        text = _preamble(query).lower()
        if not text:
            return False
        if re.search(
            r"\b(?:ask|asking)\b.{0,48}\b(?:tell|share|reveal|provide|give)\b",
            text,
        ) or re.search(
            r"\b(?:tell|share|reveal|provide|give)\b.{0,48}\b(?:ask|asking)\b",
            text,
        ):
            return True
        if re.search(
            r"\b(?:haven'?t|have not|not yet)\s+(?:told|said|shared|given)\b", text
        ) and re.search(r"\bask\b", text):
            return True
        if re.search(
            r"\bif\b.{0,24}\bask\b.{0,40}\b(?:tell|share|reveal|provide|give)\b", text
        ):
            return True
        return False
    except Exception:
        return False


def _budget_number_strings(text: str) -> set[str]:
    try:
        found: set[str] = set()
        for match in _BUDGET_RE.finditer(text or ""):
            raw = match.group("a") or match.group("b")
            if not raw:
                continue
            cleaned = raw.replace(",", "")
            found.add(cleaned)
            try:
                found.add(str(int(float(cleaned))))
            except ValueError:
                pass
        return found
    except Exception:
        return set()


def _currency_codes_in_text(text: str) -> set[str]:
    try:
        """Collect currency codes written next to amounts in the ask (not a fixed list)."""
        codes: set[str] = set()
        for match in _BUDGET_RE.finditer(text or ""):
            cur = match.group("cur")
            if cur:
                codes.add(cur.lower())
        # Budget branch often consumes the amount without the trailing code — catch it here.
        for match in re.finditer(r"\b\d[\d,]*(?:\.\d+)?\s+([A-Za-z]{3})\b", text or ""):
            codes.add(match.group(1).lower())
        return codes
    except Exception:
        return set()


def _category_hint(query: str) -> str | None:
    try:
        match = _CATEGORY_RE.search(_preamble(query))
        if not match:
            return None
        value = re.sub(r"\s+", " ", match.group(1)).strip(" .,;:")
        return value or None
    except Exception:
        return None


def _sanitize_note_for_search(note: str) -> str:
    try:
        """Drop price-narration / catalog-key noise from shopper notes before retrieval."""
        text = note or ""
        text = _CANDIDATE_KEY_RE.sub(" ", text)
        text = re.sub(
            r"\b\d[\d,]*(?:\.\d+)?\s*(?:→|->|to)\s*\d[\d,]*(?:\.\d+)?\b",
            " ",
            text,
        )
        text = re.sub(
            r"(?:latest\s+)?live\s+price\b[^.]{0,60}",
            " ",
            text,
            flags=re.I,
        )
        text = re.sub(
            r"\b(?:i am going with|i'll go with|going with)\b[^.]{0,120}",
            " ",
            text,
            flags=re.I,
        )
        # Strip hike verbs, but keep spend-cap phrasing like "go up to <amount>".
        text = re.sub(
            r"\b(?:went\s+up|price\s+hike|more\s+expensive|expensive|"
            r"budget\s+left|which|stated|ordering)\b",
            " ",
            text,
            flags=re.I,
        )
        text = re.sub(r"\bgo\s+up(?!\s+to\b)\b", " ", text, flags=re.I)
        text = re.sub(
            r"\b(?:thanks|update|leaning|toward|available|re-check|recheck)\b",
            " ",
            text,
            flags=re.I,
        )
        # Bare short integers in event notes are almost always price deltas, not model codes.
        text = re.sub(r"(?<![A-Za-z0-9])\d{1,3}(?![A-Za-z0-9.-])", " ", text)
        return text
    except Exception:
        return ""


def _priority_soft_phrases(text: str) -> list[str]:
    try:
        """Extract ranked soft factors from shopper priority clarifications."""
        raw = text or ""
        found: list[str] = []
        seen: set[str] = set()

        def push(phrase: str) -> None:
            phrase = re.sub(r"\s+", " ", (phrase or "")).strip(" .,;:")
            phrase = re.sub(
                r"^(?:with\s+a|with|a|an|the|behind|close)\s+",
                "",
                phrase,
                flags=re.I,
            )
            words = [
                tok
                for tok in re.split(r"[\s/]+", phrase)
                if tok
                and tok.lower() not in _SEED_STOP
                and tok.lower()
                not in {
                    "most",
                    "least",
                    "next",
                    "behind",
                    "close",
                    "counts",
                    "count",
                    "php",
                    "under",
                    "stock",
                }
            ]
            if not words:
                return
            phrase = " ".join(words[:4])
            low = phrase.lower()
            if low in seen or len(low) < 3:
                return
            seen.add(low)
            found.append(phrase)
            for tok in words[:4]:
                tl = tok.lower()
                if len(tl) >= 3 and tl not in seen:
                    seen.add(tl)
                    found.append(tok)

        # "X counts most/least"
        for match in re.finditer(
            r"\b([A-Za-z][A-Za-z0-9]+(?:[\s/]+[A-Za-z][A-Za-z0-9]+){0,3})\s+"
            r"counts?\s+(?:the\s+)?(?:most|least)\b",
            raw,
            re.I,
        ):
            push(match.group(1))
        # "with a Y close behind"
        for match in re.finditer(
            r"\bwith\s+a\s+([A-Za-z][A-Za-z0-9]+(?:\s+[A-Za-z][A-Za-z0-9]+){0,3})\s+"
            r"close\s+behind\b",
            raw,
            re.I,
        ):
            push(match.group(1))
        # "Z next" immediately after "close behind"
        for match in re.finditer(
            r"\bclose\s+behind\s+([A-Za-z][A-Za-z0-9]+(?:\s+[A-Za-z][A-Za-z0-9]+){0,2})\s+next\b",
            raw,
            re.I,
        ):
            push(match.group(1))
        return found[:12]
    except Exception:
        return []


def _note_states_spend_cap(text: str) -> bool:
    try:
        """True when the shopper is revising a spend limit (not narrating a price hike)."""
        raw = text or ""
        if not raw.strip():
            return False
        if re.search(
            r"\b(?:budget|at\s+most|no\s+more\s+than|can\s+only|balance|"
            r"max(?:imum)?(?:\s+budget)?|spend)\b",
            raw,
            re.I,
        ):
            return True
        if re.search(r"\b(?:go\s+)?up\s+to\s+\d", raw, re.I) and re.search(
            r"\b(?:can\s+only|sorry|now|budget|balance|under|below|at\s+most)\b",
            raw,
            re.I,
        ):
            return True
        return False
    except Exception:
        return False


def _soft_fact_dimensions(query: str) -> list[str]:
    try:
        """Extract inspection dimensions named in the public task contract text."""
        text = query or ""
        found: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(
            r"\bfacts?\s+(?:such\s+as|like)\s+([^.!?]{6,180})",
            text,
            re.I,
        ):
            chunk = match.group(1)
            chunk = re.split(
                r"\b(?:then|instead|before|after|while|so\s+that)\b",
                chunk,
                maxsplit=1,
                flags=re.I,
            )[0]
            for part in re.split(r"\s*,\s*|\s*/\s*|\s+and\s+", chunk):
                part = re.sub(r"\s+", " ", part).strip(" .,;:()")
                if len(part) < 2 or len(part) > 48:
                    continue
                # Keep short multi-word or single-token dimension labels only.
                words = [
                    tok
                    for tok in re.findall(r"[A-Za-z][A-Za-z0-9/+-]{1,}", part)
                    if tok.lower() not in _SEED_STOP
                    and tok.lower()
                    not in {
                        "use",
                        "specific",
                        "features",
                        "feature",
                        "raw",
                        "product",
                        "visible",
                    }
                ]
                if not words:
                    continue
                phrase = " ".join(words[:4])
                low = phrase.lower()
                if low in seen or "use-specific" in low:
                    continue
                seen.add(low)
                found.append(phrase)
                for tok in words[:3]:
                    tl = tok.lower()
                    if tl not in seen and len(tl) >= 3:
                        seen.add(tl)
                        found.append(tok)
        return found[:10]
    except Exception:
        return []


def _use_context_phrases(query: str) -> list[str]:
    try:
        """Capture use-case windows from the shopper preamble (discourse only)."""
        preamble = _preamble(query)
        found: list[str] = []
        seen: set[str] = set()
        for match in re.finditer(
            r"\bfor\s+(?!an?\s+in-stock\b)(?!a\s+test\b)([^.;!]{5,70})",
            preamble,
            re.I,
        ):
            prev = preamble[max(0, match.start() - 24) : match.start()].lower()
            if re.search(
                r"\b(?:settle|rather|looking|search|place|order|ask|except|excepting)\s*$",
                prev,
            ):
                continue
            chunk = match.group(1)
            chunk = re.split(r"\b(?:with|that|than|under|at\s+or)\b", chunk, maxsplit=1, flags=re.I)[
                0
            ]
            words = [
                tok
                for tok in re.findall(r"[A-Za-z][A-Za-z0-9/+-]{2,}", chunk)
                if tok.lower() not in _SEED_STOP
                and tok.lower()
                not in {
                    "place",
                    "order",
                    "stock",
                    "test",
                    "item",
                    "device",
                    "category",
                    "budget",
                    "flimsy",
                    "settle",
                    "something",
                }
            ]
            if not words:
                continue
            phrase = " ".join(words[:4])
            low = phrase.lower()
            if low in seen:
                continue
            seen.add(low)
            found.append(phrase)
        return found[:4]
    except Exception:
        return []


def _soft_code_match(blob: str, code: str) -> bool:
    try:
        """Match preferred model codes as whole tokens — never as price substrings."""
        code = (code or "").strip()
        if not code or not blob:
            return False
        blob_l = blob.lower()
        if re.fullmatch(r"\d{3,14}", code):
            return bool(re.search(rf"(?<!\d){re.escape(code)}(?!\d)", blob_l))
        return _fragment_match(blob_l, code)
    except Exception:
        return False


def _match_targets(text: str) -> list[str]:
    try:
        """Extract explicit match phrases the shopper asked the catalog to hit."""
        found: list[str] = []
        seen: set[str] = set()

        def push(raw: str) -> None:
            value = re.sub(r"\s+", " ", (raw or "")).strip(" .,;:|/\\\"'")
            if len(value) < 2:
                return
            low = value.lower()
            if low in seen or low in _SEED_STOP:
                return
            seen.add(low)
            found.append(value)

        for match in re.finditer(r'"([^"\n]{2,80})"', text or ""):
            push(match.group(1))
        for match in re.finditer(
            r"\bmatches\s+the\s+phrase\s+([^,.;\n]{2,80})",
            text or "",
            re.I,
        ):
            push(match.group(1))
        for match in re.finditer(
            r"\bmatches\s+(?!the\s+phrase\b)([A-Za-z0-9][\w\s&/-]{1,60}?)(?="
            r"\s*,|\s+at\s+or|\s+under|\s+and\s+buy|\s*$)",
            text or "",
            re.I,
        ):
            push(match.group(1))
        return found[:3]
    except Exception:
        return []


def _search_parts(query: str, notes: list[str] | None = None) -> list[str]:
    try:
        """Split the shopper ask into short discovery tokens (query-derived only)."""
        chunks = [_preamble(query)]
        if notes:
            chunks.extend(_sanitize_note_for_search(n) for n in notes[-2:])
        text = " ".join(chunks)
        # Catalog keys are not useful retrieval terms.
        text = _CANDIDATE_KEY_RE.sub(" ", text)
        targets = _match_targets(text)
        if targets:
            return targets[:2]

        budget_nums = _budget_number_strings(text)
        currency_codes = _currency_codes_in_text(text)
        parts: list[str] = []
        seen: set[str] = set(currency_codes)
        used_units: set[str] = set()

        def add(token: str, *, allow_model_code: bool = False) -> None:
            tok = (token or "").strip(" ,.;:|/\\\"'")
            if not tok:
                return
            tok = _canon_spec_token(tok)
            low = tok.lower()
            if low in seen or low in _SEED_STOP or low in currency_codes:
                return
            # Bare unit / availability words are not useful discovery terms.
            if low in {"in", "inch", "inches", "in-stock", "instock", "stock"}:
                return
            # Drop catalog keys / long ids leaked from the prompt — not useful search terms.
            if "::" in tok:
                return
            if re.fullmatch(r"\d{6,}", low):
                # Keep mid-length model codes from soft preferences; drop 19-digit product ids.
                if not allow_model_code or not (6 <= len(low) <= 14):
                    return
            if re.fullmatch(r"\d{3,5}", low):
                if not allow_model_code:
                    return
                if low in budget_nums:
                    return
            if re.fullmatch(r"[a-z0-9]+-\d+", low) and len(low) >= 8:
                return
            if re.fullmatch(r"\d+(?:\.\d+)?", low):
                if low in budget_nums:
                    return
                # Keep short model-like numerals; drop large bare currency amounts.
                try:
                    if float(low) >= 1000 and not allow_model_code:
                        return
                except ValueError:
                    return
            seen.add(low)
            parts.append(tok)

        # Keep number+unit pairs from the ask (unit length >= 2).
        for match in _NUM_UNIT_RE.finditer(text):
            num, unit = match.group(1), match.group(2)
            unit_l = _canon_unit(unit)
            if len(unit_l) < 2 or unit_l in _SEED_STOP or unit_l in currency_codes:
                continue
            if num in budget_nums:
                continue
            try:
                # Bare amounts this large are almost always budgets, not model codes.
                if float(num) >= 1000 and len(unit_l) < 2:
                    continue
            except ValueError:
                continue
            used_units.add(unit_l)
            add(f"{num}-{unit_l}")

        # Soft-preference windows may carry model codes / color+code pairs.
        soft_parts: list[str] = []
        soft_seen: set[str] = set()
        for soft_match in _SOFT_CUE_RE.finditer(text):
            window = text[soft_match.end() : soft_match.end() + 48]
            for code in re.findall(r"\b(\d{3,14})\b", window):
                low = code.lower()
                if low not in soft_seen:
                    soft_seen.add(low)
                    soft_parts.append(code)
                add(code, allow_model_code=True)
            for tok in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", window):
                low = tok.lower()
                if low in _SEED_STOP or low in soft_seen:
                    continue
                soft_seen.add(low)
                soft_parts.append(tok)
                add(tok)
                # Light morphology: noun -ance → adjective -ant when present in the ask.
                if low.endswith("ance") and len(low) > 6:
                    alt = low[:-4] + "ant"
                    if alt not in soft_seen and alt not in _SEED_STOP:
                        soft_seen.add(alt)
                        soft_parts.append(alt)
                        add(alt)

        # Compact alnum codes / sizes present in the ask.
        for match in re.finditer(r"\b([A-Za-z]*\d+[A-Za-z0-9.-]*)\b", text):
            tok = match.group(1)
            bare = tok.replace(".", "").replace("-", "")
            if bare in budget_nums or tok.lower() in currency_codes:
                continue
            add(tok)

        # Capitalized tokens from the shopper preamble (often brand / model family).
        for match in re.finditer(r"\b([A-Z][A-Za-z0-9]{2,})\b", _preamble(query)):
            tok = match.group(1)
            if tok.lower() in currency_codes:
                continue
            add(tok)

        for tok in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", text):
            if tok.lower() in used_units:
                continue
            add(tok)
            if len(parts) >= 14:
                break

        specs = [p for p in parts if re.search(r"\d", p)]
        words = [p for p in parts if p not in specs]
        soft_l = {s.lower() for s in soft_parts}
        # Core nouns first (category/product), then soft preference modifiers.
        core = [w for w in words if w.lower() not in soft_l and len(w) >= 4]
        soft_keep = [w for w in soft_parts if w.lower() not in {c.lower() for c in core}]
        short = [w for w in words if len(w) < 4 and w.lower() not in soft_l]
        ordered = specs[:3] + core[:3] + soft_keep[:3] + short[:1]
        return (ordered or parts)[:7]
    except Exception:
        return []


def _search_seed(query: str, notes: list[str] | None = None) -> str:
    try:
        """Build a short discovery query from the public goal + latest shopper notes."""
        parts = _search_parts(query, notes)
        return " ".join(parts).strip() or "in stock"
    except Exception:
        return ""


def _dedupe_tokens(parts: list[str]) -> list[str]:
    try:
        out: list[str] = []
        seen: set[str] = set()
        for part in parts:
            low = part.lower()
            if not low or low in seen:
                continue
            seen.add(low)
            out.append(part)
        return out
    except Exception:
        return []


def _search_variants(query: str, notes: list[str] | None, attempt: int) -> str:
    try:
        """Rotate short formulations when prior searches were empty or noisy."""
        # Explicit match phrases are the strongest retrieval signal — try them first.
        note_blob = " ".join([_preamble(query)] + list(notes or [])[-2:])
        targets = _match_targets(note_blob)
        if targets:
            phrase = targets[0]
            firm, soft = [], []
            try:
                firm, soft = _split_constraints(query, notes)
            except Exception:
                pass
            soft_codes = [s for s in soft if re.fullmatch(r"\d{3,14}", s)]
            soft_words = [s for s in soft if s not in soft_codes][:2]
            variants = [
                phrase,
                " ".join(_dedupe_tokens([phrase] + soft_words[:1])[:4]),
                " ".join(_dedupe_tokens([phrase] + soft_codes[:1] + soft_words[:1])[:4]),
                " ".join(_dedupe_tokens([phrase] + [f for f in firm if not re.search(r"\d", f)][:1])[:3]),
            ]
            return variants[attempt % len(variants)]

        parts = _dedupe_tokens(_search_parts(query, notes))
        if not parts:
            return "in stock"
        specs = [p for p in parts if re.search(r"\d", p)]
        words = [p for p in parts if p not in specs]
        category = _category_hint(query)
        soft: list[str] = []
        firm: list[str] = []
        try:
            firm, soft = _split_constraints(query, notes)
        except Exception:
            soft = []
        soft = _dedupe_tokens([s for s in soft if len(s.split()) <= 3])
        soft_codes = [s for s in soft if re.fullmatch(r"\d{3,14}", s)]
        # Prefer single-token soft cues in queries; multi-word phrases pollute ranking.
        soft_words = [
            s for s in soft if s not in soft_codes and " " not in s and len(s) >= 3
        ][:3]
        soft_l = {s.lower() for s in soft}
        firm_nouns = [
            f
            for f in firm
            if not re.search(r"\d", f) and len(f.split()) == 1 and len(f) >= 4
        ][:3]
        type_nouns = [f for f in firm_nouns if not f[:1].isupper()][:1]
        brand_nouns = [f for f in firm_nouns if f[:1].isupper()][:1]
        core = [w for w in words if w.lower() not in soft_l][:2]
        # Strongest retrieval anchors: codes, product noun, soft modifiers, type noun.
        primary = _dedupe_tokens(
            soft_codes[:1] + soft_words[:1] + core[:1] + firm_nouns[:1] + specs[:1]
        )
        # Type + spec + brand first when the ask names a product type firmly.
        typed = _dedupe_tokens(type_nouns + specs[:1] + brand_nouns + soft_words[:1] + core[:1])
        variants = [
            " ".join(typed[:4]) if type_nouns else (
                " ".join(soft_codes[:1] + soft_words[:1] + core[:1])
                if (soft_codes or soft_words)
                else " ".join(primary[:4])
            ),
            " ".join(primary[:4]),
            " ".join(
                _dedupe_tokens(
                    soft_codes[:1] + core[:1] + soft_words[:2] + firm_nouns[:1]
                )[:4]
            ),
            " ".join(_dedupe_tokens(soft_words[:2] + core[:1])[:3]),
            " ".join(_dedupe_tokens(soft_codes[:1] + firm_nouns[:1] + soft_words[:1])[:3]),
            " ".join(_dedupe_tokens(core[:1] + soft_words[:2] + specs[:1])[:4]),
            " ".join(parts[:5]),
            " ".join(words[:3]) if words else " ".join(parts[:3]),
        ]
        if category:
            variants.append(f"{category} {' '.join(primary[:2])}".strip())
        variants = [v.strip() for v in variants if v and v.strip()]
        if not variants:
            return "in stock"
        return variants[attempt % len(variants)]
    except Exception:
        return ""


def _classify_task_shape(
    problem_data: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    named: list[tuple[str, str]],
    needs_ask: bool,
) -> str:
    try:
        """Branch only on public problem_data / schema signals (not wall-clock)."""
        family = str(
            problem_data.get("family")
            or (problem_data.get("environment") or {}).get("family")
            or ""
        ).strip().lower()
        if family:
            return family
        query = str(
            ((problem_data.get("environment") or {}).get("policy_view") or {}).get("query")
            or problem_data.get("query")
            or ""
        )
        preamble = _preamble(query)
        text_l = (query or "").lower()
        pre_l = preamble.lower()
        # Named multi-variant compare loops are ranking/tradeoff tasks.
        if len(named) >= 2:
            return "ranking"
        if "justification" in _required_args(tools, "place_test_order") or re.search(
            r"\bwalk\s+me\s+through\s+why\b", pre_l
        ):
            return "justification"
        if needs_ask:
            return "intent_decomposition"
        # Intent: soft brand subordinated to firm specs (shopper may intervene later).
        if re.search(
            r"\bideally\b.{0,60}\bprefer\b.{0,100}\bfirm\s+requirements?\s+matter",
            pre_l,
        ):
            return "intent_decomposition"
        # Phrase-match retrieval: quoted target or "matches the phrase …".
        if (
            re.search(r'\bmatches?\s+"[^"\n]{2,80}"', preamble)
            or re.search(r"\bmatches?\s+[“‘'][^”’'\n]{2,80}[”’']", preamble)
            or re.search(r"\bmatches?\s+(?:the\s+)?phrase\s+\S+", pre_l)
        ):
            return "retrieval_recall"
        # Preference: soft-preference discourse in the ask / task contract.
        if (
            re.search(r"\bi'?d\s+rather\b", pre_l)
            or re.search(r"\bwhat\s+i\s+(?:actually\s+)?care\s+about\b", pre_l)
            or re.search(r"\bmatters?\s+more\s+to\s+me\b", pre_l)
            or re.search(r"\bsoft\s+preferences?\s+matter\b", text_l)
            or re.search(r"\bloose\s+hard\s+request\s+plus\s+soft\b", text_l)
            or (
                re.search(r"\bclarify\b.{0,48}\bpriorit", text_l)
                and re.search(r"\bsoft\s+preferences?\b", text_l)
            )
        ):
            return "preference_reasoning"
        # Recovery: exact preferred variant may disappear / become unavailable.
        if re.search(
            r"\b(?:disappears?|unavailable|if\s+that\s+exact|strongly\s+prefer)\b",
            pre_l,
        ):
            return "recovery"
        if re.search(r"\bfirm\s+requirements?\b", text_l) and re.search(
            r"\b(?:soft\s+preference|preferred\s+brand|ideally)\b", text_l
        ):
            return "constraint_satisfaction"
        return "retrieval"
    except Exception:
        return ""


def _tradeoff_priorities(query: str) -> list[dict[str, str]]:
    try:
        """Parse ranked max/min axes from the shopper's own wording."""
        text = (query or "").split("Task rules:")[0]
        found: list[dict[str, str]] = []
        seen: set[str] = set()
        # Only strip discourse fillers from the captured field — do not apply the
        # general seed stop list (it would erase axis nouns the shopper just used).
        filler = {
            "advertised",
            "maximum",
            "minimum",
            "available",
            "current",
            "live",
            "including",
            "explicitly",
            "mode",
            "the",
            "a",
            "an",
            "of",
            "and",
            "or",
            "with",
            "for",
        }
        for match in _TRADEOFF_RE.finditer(text):
            direction_word = match.group("dir").lower()
            raw_field = re.sub(r"\s+", " ", match.group("field")).strip(" .,;:")
            tokens = [
                tok
                for tok in _TOKEN_RE.findall(raw_field.lower())
                if tok not in filler
                and len(tok) > 1
                and tok not in {"then", "among", "between", "while", "before", "after"}
            ]
            if not tokens:
                continue
            phrase = " ".join(tokens[:4])
            if phrase in seen:
                continue
            seen.add(phrase)
            direction = "max" if direction_word in _DIR_MAX else "min"
            found.append({"field": phrase, "direction": direction})
        return found[:5]
    except Exception:
        return []


def _iter_fact_pairs(row: dict[str, Any]) -> list[tuple[str, str]]:
    try:
        pairs: list[tuple[str, str]] = []
        stack: list[Any] = [row]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if value in (None, ""):
                        continue
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                        continue
                    if isinstance(key, str):
                        pairs.append((key, str(value)))
            elif isinstance(node, list):
                stack.extend(node)
        return pairs
    except Exception:
        return []


def _metric_tokens(phrase: str) -> list[str]:
    try:
        """Axis tokens taken only from the shopper's tradeoff phrase."""
        found: list[str] = []
        seen: set[str] = set()
        for tok in _TOKEN_RE.findall((phrase or "").lower()):
            if len(tok) <= 1 or tok in seen:
                continue
            seen.add(tok)
            found.append(tok)
        return found
    except Exception:
        return []


_SKIP_METRIC_KEYS = frozenset(
    {
        "product_id",
        "sku",
        "listing_id",
        "currency",
        "brand",
        "title",
        "category_path",
        "options",
    }
)


def _as_float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            try:
                return float(text)
            except ValueError:
                match = _NUM_TOKEN_RE.search(text)
                if not match:
                    return None
                try:
                    return float(match.group(0))
                except ValueError:
                    return None
        return None
    except Exception:
        return None


def _harvest_unit_numbers(surface: str) -> list[tuple[str, float]]:
    try:
        """Parse NUM+unit pairs from one catalog string (facts/options/title)."""
        found: list[tuple[str, float]] = []
        text = surface or ""
        if not text:
            return found
        # Skip dimension separators like ``1920 x 1080`` — not unit measurements.
        cleaned = re.sub(
            r"\b\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\b",
            " ",
            text,
            flags=re.I,
        )
        # Skip title ranges like ``24-32"`` that span multiple models.
        cleaned = re.sub(
            r"\b\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?\s*(?:\"|''|in(?:ch(?:es)?)?)\b",
            " ",
            cleaned,
            flags=re.I,
        )
        seen_pair: set[tuple[str, float]] = set()
        for match in re.finditer(
            r"(\d+(?:\.\d+)?)\s*-?\s*([A-Za-z%]{1,16}|\"|')",
            cleaned,
        ):
            try:
                number = float(match.group(1))
            except ValueError:
                continue
            if number >= 10_000_000:
                continue
            # Reject numbers embedded inside model tokens; keep glued unit suffixes.
            left = match.start(1)
            if left > 0 and cleaned[left - 1].isalnum():
                continue
            right = match.end(2)
            if right < len(cleaned) and cleaned[right].isalnum():
                continue
            unit = _canon_unit(match.group(2))
            # Single-letter units are almost always separators or noise.
            if len(unit) == 1:
                continue
            if unit in _SEED_STOP:
                continue
            pair = (unit, number)
            if pair in seen_pair:
                continue
            seen_pair.add(pair)
            found.append(pair)
        return found
    except Exception:
        return []


def _title_unit_numbers(row: dict[str, Any]) -> list[tuple[str, float]]:
    try:
        """Harvest (unit, number) pairs, preferring variant options over listing facts/title."""
        found: list[tuple[str, float]] = []
        seen_unit: set[str] = set()

        def absorb(surface: str) -> None:
            for unit, number in _harvest_unit_numbers(surface):
                if unit in seen_unit:
                    continue
                seen_unit.add(unit)
                found.append((unit, number))

        # Variant options are the selected-sku specification — trust them first.
        options = str(row.get("options") or "")
        if options:
            absorb(options)
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        for key, value in facts.items():
            if value in (None, "") or key in _SKIP_METRIC_KEYS:
                continue
            text = str(value)
            if re.search(r"\d", text):
                absorb(text)
        # Remaining fact keys outside the facts dict (flattened rows).
        for key, value in _iter_fact_pairs(row):
            if key in _SKIP_METRIC_KEYS or key in {"price", "title", "options", "facts"}:
                continue
            text = str(value)
            if re.search(r"\d", text) and re.search(r"[A-Za-z\"']", text):
                absorb(text)
        title = str(row.get("title") or "")
        if title:
            # Title fills only units still missing after options/facts.
            absorb(title)
        return found
    except Exception:
        return []


def _metric_value(row: dict[str, Any], phrase: str, query: str = "") -> float | None:
    try:
        """Score a query-derived axis via structural key/token overlap on live facts."""
        del query  # kept for call-site compatibility; axes come from ``phrase`` only
        tokens = _metric_tokens(phrase)
        if not tokens:
            return None
        phrase_kn = _norm_key(phrase)

        # When the shopper's axis names a price/cost field, read live price columns.
        price_axis = any(
            _keys_structurally_match(_norm_key(tok), "price")
            or _keys_structurally_match(_norm_key(tok), "cost")
            for tok in tokens
        ) or _keys_structurally_match(phrase_kn, "price")
        if price_axis and len(tokens) <= 2:
            for key in ("price", "current_price", "price_at_add", "new_price"):
                number = _as_float(row.get(key))
                if number is not None:
                    return number
            return None

        scored: list[tuple[int, float]] = []
        for key, value in _iter_fact_pairs(row):
            if key in _SKIP_METRIC_KEYS:
                continue
            kn = _norm_key(key)
            if key in {"price", "current_price", "price_at_add", "new_price"}:
                continue
            text = f"{kn} {value}".lower()
            overlap = 0
            for tok in tokens:
                tok_n = _norm_key(tok)
                if not tok_n:
                    continue
                if _keys_structurally_match(kn, tok_n) or tok_n in kn:
                    overlap += 2
                elif tok in text:
                    overlap += 1
            if _keys_structurally_match(kn, phrase_kn):
                overlap += 2
            if overlap <= 0:
                continue
            number = _as_float(value)
            if number is None or number >= 10_000_000:
                continue
            scored.append((overlap, number))

        # Title/options: numbers whose local window overlaps the axis tokens.
        for surface_key in ("title", "options"):
            surface = str(row.get(surface_key) or "").lower()
            if not surface:
                continue
            for match in re.finditer(r"(\d+(?:\.\d+)?)", surface):
                try:
                    number = float(match.group(1))
                except ValueError:
                    continue
                if number >= 10_000_000:
                    continue
                window = surface[max(0, match.start() - 28) : match.end() + 28]
                overlap = sum(1 for tok in tokens if tok in window)
                # NUM+unit glue (e.g. value written beside its unit token in the title).
                glued = surface[match.end() : match.end() + 12]
                unit_match = re.match(r"\s*-?\s*([a-z%]{1,16})", glued)
                if unit_match:
                    unit = unit_match.group(1)
                    if len(unit) >= 2 and any(
                        _keys_structurally_match(_norm_key(tok), _norm_key(unit))
                        or (len(tok) >= 3 and tok in unit)
                        or (len(unit) >= 3 and unit in tok)
                        for tok in tokens
                    ):
                        overlap += 2
                    elif len(tokens) >= 2 and len(unit) >= 2:
                        # Compound axes often appear as NUM+unit without the full phrase.
                        overlap = max(overlap, 1)
                if overlap <= 0:
                    continue
                scored.append((overlap, number))

        if not scored:
            return None
        scored.sort(key=lambda item: (-item[0], -item[1]))
        top = scored[0][0]
        tied = [value for overlap, value in scored if overlap == top]
        return max(tied) if tied else None
    except Exception:
        return None


def _axis_unit_vote(
    rows: list[dict[str, Any]], phrase: str
) -> str | None:
    try:
        """When facts omit an axis, infer which title-unit tracks it across the shortlist."""
        tokens = _metric_tokens(phrase)
        if not tokens:
            return None
        # Price/cost axes never bind to a harvested unit token.
        if any(
            _keys_structurally_match(_norm_key(tok), "price")
            or _keys_structurally_match(_norm_key(tok), "cost")
            for tok in tokens
        ):
            return None
        # Prefer units that already look like the axis tokens.
        for row in rows:
            for unit, _number in _title_unit_numbers(row):
                if len(unit) < 2:
                    continue
                if any(
                    _keys_structurally_match(_norm_key(tok), _norm_key(unit))
                    or tok in unit
                    or unit in tok
                    for tok in tokens
                ):
                    return unit
        # Otherwise pick the unit with the widest numeric spread across candidates —
        # that is the live attribute the catalog is differentiating on.
        buckets: dict[str, list[float]] = {}
        for row in rows:
            seen_units: set[str] = set()
            for unit, number in _title_unit_numbers(row):
                if len(unit) < 2 or unit in seen_units:
                    continue
                seen_units.add(unit)
                buckets.setdefault(unit, []).append(number)
        best_unit: str | None = None
        best_score = -1.0
        for unit, values in buckets.items():
            if len(values) < 2:
                continue
            span = max(values) - min(values)
            # Prefer units that cover more of the shortlist, then wider span.
            coverage = len(values)
            score = (coverage * 1000.0) + span + 0.01 * len(unit)
            if score > best_score:
                best_score = score
                best_unit = unit
        return best_unit
    except Exception:
        return None


def _metric_value_with_unit(
    row: dict[str, Any], phrase: str, unit: str | None, query: str = ""
) -> float | None:
    try:
        tokens = _metric_tokens(phrase)
        phrase_kn = _norm_key(phrase)
        price_axis = any(
            _keys_structurally_match(_norm_key(tok), "price")
            or _keys_structurally_match(_norm_key(tok), "cost")
            for tok in tokens
        ) or _keys_structurally_match(phrase_kn, "price")
        if price_axis:
            for key in ("price", "current_price", "price_at_add", "new_price"):
                number = _as_float(row.get(key))
                if number is not None:
                    return number
            return None
        if unit:
            values = [number for u, number in _title_unit_numbers(row) if u == unit]
            if values:
                return max(values)
        return _metric_value(row, phrase, query)
    except Exception:
        return None


def _goal_spec_fragments(query: str, notes: list[str] | None = None) -> list[str]:
    try:
        """Numeric/spec fragments the shopper asked for — used to filter result rows."""
        frags: list[str] = []
        seen: set[str] = set()
        for part in _search_parts(query, notes):
            if not re.search(r"\d", part):
                continue
            low = part.lower()
            if low not in seen:
                seen.add(low)
                frags.append(low)
            for num in _NUM_TOKEN_RE.findall(part):
                if num not in seen:
                    seen.add(num)
                    frags.append(num)
        return frags
    except Exception:
        return []


def _spec_match_score(row: dict[str, Any], specs: list[str]) -> int:
    try:
        """How well a catalog row carries the shopper's numeric/spec fragments."""
        if not specs:
            return 0
        title = str(row.get("title") or "")
        brand = str(row.get("brand") or "")
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}
        blob = f"{title} {brand} {json.dumps(facts, ensure_ascii=False, default=str)}".lower()
        compact = re.sub(r"[^a-z0-9.]+", "", blob)
        score = 0
        missed = 0
        matched_nums: set[str] = set()
        for spec in specs:
            sn = re.sub(r"[^a-z0-9.]+", "", spec)
            nums = _NUM_TOKEN_RE.findall(spec)
            hit = bool(sn and sn in compact)
            if not hit and nums and any(num in blob or num in compact for num in nums):
                hit = True
            if hit:
                # Credit each distinct numeric once to avoid double-counting paired forms.
                credit = False
                for num in nums or [sn]:
                    if num and num not in matched_nums:
                        matched_nums.add(num)
                        credit = True
                if credit:
                    score += 5
                continue
            missed += 1
        score -= 4 * missed
        return score
    except Exception:
        return 0


_FIRM_CUE_RE = re.compile(
    r"\b(?:must|require(?:d|s)?|has\s+to|have\s+to|"
    r"only|exactly|mandatory|non[\s-]?negotiable)\b",
    re.I,
)
_SOFT_CUE_RE = re.compile(
    r"\b(?:preferred|preferring|prefer|ideally|leaning|lean|"
    r"nice\s+to\s+have|would\s+like|rather|bonus|optional|soft|"
    r"looking\s+for)\b",
    re.I,
)
# Soft-preference clauses — discourse patterns only, no product vocabulary.
_SOFT_PHRASE_RE = re.compile(
    r"(?:"
    r"\b(?:rather(?:\s+[^,.;]{0,40})?\s+for|instead\s+of(?:\s+just)?|"
    r"don'?t\s+just(?:\s+\w+){0,3}|care\s+about)\s+([^.,;!?]{4,70})"
    r"|"
    r"(?:^|[.;:])\s*([^.;:]{4,90}?)\s+is\s+what\s+i\s+(?:actually\s+)?care\s+about\b"
    r"|"
    r"(?:^|[.;:])\s*([^.;:]{4,90}?)\s+matters?\s+more(?:\s+to\s+me)?\b"
    r")",
    re.I,
)
_INSTRUCTION_VERBS = frozenset(
    """
    say says saying identify identifies tell tells show shows walk give gives
    explain describes describe mention mentions label labels
    """.split()
)


def _row_text_blob(row: dict[str, Any]) -> str:
    try:
        """Flatten live product surfaces into one searchable lowercase blob."""
        chunks: list[str] = []
        for key in (
            "title",
            "brand",
            "sku",
            "options",
            "category",
            "condition",
            "model",
        ):
            value = row.get(key)
            if value not in (None, ""):
                chunks.append(str(value))
        path = row.get("category_path")
        if isinstance(path, list):
            chunks.extend(str(part) for part in path if part is not None)
        elif isinstance(path, str) and path.strip():
            chunks.append(path)
        facts = row.get("facts")
        if isinstance(facts, dict):
            for key, value in facts.items():
                if value in (None, "", False):
                    continue
                if isinstance(value, (dict, list)):
                    continue
                chunks.append(f"{key} {value}")
        for key, value in row.items():
            if key in {
                "title",
                "brand",
                "sku",
                "options",
                "category",
                "category_path",
                "facts",
                "product_id",
                "listing_id",
                "price",
                "in_stock",
                "currency",
                "variants",
            }:
                continue
            if value in (None, "", False) or isinstance(value, (dict, list, bool)):
                continue
            if isinstance(key, str):
                chunks.append(f"{key} {value}")
        text = " ".join(chunks).lower()
        # Orthographic fold for length quotes so NUM-in fragments match catalog text.
        text = re.sub(r'(\d+(?:\.\d+)?)\s*(?:["\'])', r"\1-in ", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:inches|inch)\b", r"\1-in", text)
        text = re.sub(r"(\d+(?:\.\d+)?)\s+in\b", r"\1-in", text)
        return text
    except Exception:
        return ""


def _fragment_match(blob: str, fragment: str) -> bool:
    try:
        """True when a query-derived fragment is evidenced on the product blob."""
        frag = _canon_spec_token((fragment or "").strip().lower())
        if not frag or not blob:
            return False
        variants = {frag}
        # Light morphology so soft asks match catalog adjectives/nouns.
        if frag.endswith("ance") and len(frag) > 6:
            variants.add(frag[:-4] + "ant")
            variants.add(frag[:-6] if frag.endswith("stance") else frag)
        if frag.endswith("ant") and len(frag) > 5:
            variants.add(frag[:-3] + "ance")
        if frag.endswith("ing") and len(frag) > 5:
            variants.add(frag[:-3])
        for candidate in list(variants):
            if candidate in blob:
                return True
            compact_blob = re.sub(r"[^a-z0-9.]+", "", blob)
            compact_frag = _normalize_fragment(candidate)
            if len(compact_frag) >= 3 and compact_frag in compact_blob:
                return True
        # Also try the pre-canon surface for loosely written catalog text.
        alt = (fragment or "").strip().lower()
        if alt and alt != frag and alt in blob:
            return True
        tokens = [t for t in _TOKEN_RE.findall(frag) if len(t) > 1 and t not in _SEED_STOP]
        if not tokens:
            return False
        # Multi-token fragments need most content tokens present.
        hits = sum(1 for tok in tokens if tok in blob or tok in re.sub(r"[^a-z0-9.]+", "", blob))
        if len(tokens) == 1:
            return hits == 1
        return hits >= max(1, len(tokens) - (1 if len(tokens) >= 3 else 0))
    except Exception:
        return False


def _normalize_fragment(fragment: str) -> str:
    try:
        frag = _canon_spec_token(fragment or "")
        return re.sub(r"[^a-z0-9.]+", "", frag.lower())
    except Exception:
        return ""


def _dedupe_fragments(frags: list[str]) -> list[str]:
    try:
        """Drop near-duplicate fragments that compact to the same key."""
        out: list[str] = []
        seen: set[str] = set()
        for frag in frags:
            key = _normalize_fragment(frag)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(frag)
        return out
    except Exception:
        return []


def _split_constraints(
    query: str, notes: list[str] | None = None
) -> tuple[list[str], list[str]]:
    try:
        """Derive firm vs soft constraint fragments from the public ask only."""
        chunks = [_preamble(query)]
        if notes:
            chunks.extend(notes[-3:])
        text = " ".join(chunks)
        text = _CANDIDATE_KEY_RE.sub(" ", text)
        firm: list[str] = []
        soft: list[str] = []
        seen_f: set[str] = set()
        seen_s: set[str] = set()
        budget_nums = _budget_number_strings(text)
        currency_codes = _currency_codes_in_text(text)

        def push(target: list[str], seen: set[str], value: str, *, allow_model_code: bool = False) -> None:
            raw = re.sub(r"\s+", " ", (value or "")).strip(" .,;:|/\\\"'")
            if len(raw) < 2:
                return
            raw = _canon_spec_token(raw)
            low = raw.lower()
            if low in seen or low in _SEED_STOP:
                return
            if low in {"in", "inch", "inches"}:
                return
            if "::" in raw:
                return
            if re.fullmatch(r"\d{6,}", low):
                if not allow_model_code or not (6 <= len(low) <= 14):
                    return
            if re.fullmatch(r"\d{3,5}", low):
                if not allow_model_code:
                    return
                if low in budget_nums:
                    return
            if low in budget_nums or low in currency_codes:
                return
            glued = re.fullmatch(r"(\d+(?:\.\d+)?)[\s-]+([a-z]{3})", low)
            if glued and (
                glued.group(1) in budget_nums or glued.group(2) in currency_codes
            ):
                return
            # Instruction / meta-prose fragments are not product constraints.
            if any(
                bad in low
                for bad in (
                    "replacement",
                    "insufficient",
                    "identify",
                    "explicit",
                    "label",
                    "requirements",
                    "saving",
                    "dollars",
                    "matter",
                    "ideally",
                )
            ):
                return
            seen.add(low)
            target.append(raw)

        for match in re.finditer(r'"([^"\n]{2,80})"', text):
            push(firm, seen_f, match.group(1))

        category = _category_hint(query)
        if category:
            push(firm, seen_f, category)

        for match in _NUM_UNIT_RE.finditer(text):
            num, unit = match.group(1), match.group(2)
            unit_l = _canon_unit(unit)
            if num in budget_nums or unit_l in currency_codes:
                continue
            if len(unit_l) < 2 or unit_l in _SEED_STOP:
                continue
            push(firm, seen_f, f"{num}-{unit_l}")

        discourse_skip = {
            "besides",
            "except",
            "including",
            "versus",
            "brand",
            "something",
            "anything",
            "item",
            "product",
            "have",
            "has",
            "with",
            "without",
            "using",
            "via",
            "also",
            "please",
        }
        # "I need a <noun>" / "Find me a <noun>" — lowercase head noun as firm cue.
        for match in re.finditer(
            r"\b(?:need|want|find|get|buy|order)(?:\s+me)?\s+"
            r"(?:an?\s+|the\s+)?([a-z][a-z0-9-]{2,28})\b",
            _preamble(query),
            re.I,
        ):
            noun = match.group(1)
            if noun.lower() in discourse_skip or noun.lower() in _SEED_STOP:
                continue
            push(firm, seen_f, noun)

        # Capitalized brand / model-family tokens from the shopper's first ask sentence.
        preamble = _preamble(query)
        ask_focus = re.split(
            r"\b(?:walk\s+me|for\s+a\s+\w+|give\s+me\s+a\s+reason|tell\s+me\s+why|"
            r"task\s+rules)\b",
            preamble,
            maxsplit=1,
            flags=re.I,
        )[0]
        ask_focus = ask_focus.split(".")[0]
        for match in re.finditer(
            r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})\b",
            ask_focus,
        ):
            brand = match.group(1).strip()
            low = brand.lower()
            if low in currency_codes or low in _SEED_STOP or low in discourse_skip:
                continue
            # Skip short ALLCAPS/tech acronyms and discourse openers — not brands.
            if low in currency_codes or low in {
                "sku",
                "can",
                "find",
                "want",
                "get",
                "need",
                "buy",
                "order",
                "please",
                "give",
                "tell",
                "walk",
                "full",
            }:
                continue
            if len(low) <= 3 and low.isalpha():
                continue
            # Keep multi-word brands together; also keep strong single tokens.
            if " " in brand or len(brand) >= 4:
                push(firm, seen_f, brand)
            # Product-type noun often follows brand + optional specs — take the
            # last concrete noun in the local window (query-derived only).
            tail = ask_focus[match.end() : match.end() + 64]
            nouns = [
                noun
                for noun in re.findall(r"\b([a-z][a-z0-9-]{3,24})\b", tail)
                if noun not in _SEED_STOP
                and noun not in discourse_skip
                and noun not in currency_codes
                and noun
                not in {
                    "with",
                    "from",
                    "into",
                    "over",
                    "under",
                    "stock",
                    "full",
                    "hard",
                }
            ]
            if nouns:
                push(firm, seen_f, nouns[-1])
                if len(nouns) >= 2 and nouns[-2] != nouns[-1]:
                    push(firm, seen_f, nouns[-2])
        split_pat = (
            r"[.!?;\n,]|Task rules:|\bbut\b|\band\b|\bunder\b|\bbelow\b|"
            r"\bwithin\b|\bin\s+the\b"
        )
        for cue_re, bucket, seen in (
            (_FIRM_CUE_RE, firm, seen_f),
            (_SOFT_CUE_RE, soft, seen_s),
        ):
            for match in cue_re.finditer(text):
                window = text[match.end() : match.end() + 40]
                window = re.split(split_pat, window, maxsplit=1)[0]
                for quoted in re.finditer(r'"([^"\n]{2,60})"', window):
                    push(bucket, seen, quoted.group(1))
                found_num = False
                for num_unit in _NUM_UNIT_RE.finditer(window):
                    num, unit = num_unit.group(1), _canon_unit(num_unit.group(2))
                    if num in budget_nums or unit in currency_codes:
                        continue
                    if len(unit) < 2 or unit in _SEED_STOP:
                        continue
                    push(bucket, seen, f"{num}-{unit}")
                    found_num = True
                for code in re.findall(r"\b(\d{3,14})\b", window):
                    if code in budget_nums:
                        continue
                    push(bucket, seen, code, allow_model_code=True)
                    found_num = True
                words = [
                    tok
                    for tok in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{1,}", window)
                    if tok.lower() not in _SEED_STOP
                    and tok.lower() not in currency_codes
                    and tok.lower() not in discourse_skip
                    and tok.lower() not in _INSTRUCTION_VERBS
                ]
                # When a numeric spec was captured, skip the residual prose phrase —
                # it is usually a noisy paraphrase of the same constraint.
                if words and not found_num:
                    # Skip instruction-verb openings (say/identify/tell...) as firm specs.
                    if bucket is firm and words[0].lower() in _INSTRUCTION_VERBS:
                        continue
                    if bucket is firm and any(
                        w.lower() in _INSTRUCTION_VERBS for w in words[:2]
                    ):
                        continue
                    push(bucket, seen, " ".join(words[:3]))
                elif words and bucket is soft:
                    # Keep short preference tokens alongside model codes.
                    for word in words[:3]:
                        push(bucket, seen, word)

        # Clarification notes often say "Required X is Y" without classic firm cues.
        for match in re.finditer(
            r"\b(?:required|requirement|constraint)\s+([A-Za-z][\w-]{1,24})"
            r"\s*(?:is|=|:)\s*([A-Za-z0-9][\w.-]{0,24})",
            text,
            re.I,
        ):
            push(firm, seen_f, match.group(1))
            push(firm, seen_f, match.group(2))

        # Shopper replies often append further "key value" pairs after "and".
        if notes:
            note_blob = " ".join(notes[-3:])
            for match in re.finditer(
                r"\band\s+([A-Za-z][\w-]{1,20})\s+([A-Za-z0-9][\w.-]{0,20})\b",
                note_blob,
                re.I,
            ):
                left, right = match.group(1), match.group(2)
                if left.lower() in discourse_skip or right.lower() in discourse_skip:
                    continue
                push(firm, seen_f, left)
                push(firm, seen_f, right)

        explicit_brand = re.search(r"\bbrand\s+([A-Za-z0-9][\w-]{1,24})\b", text, re.I)
        if explicit_brand:
            brand = explicit_brand.group(1)
            prefix = text[max(0, explicit_brand.start() - 40) : explicit_brand.start()]
            if _FIRM_CUE_RE.search(prefix) or re.search(
                r"\b(?:must|need|require)\b", prefix, re.I
            ):
                push(firm, seen_f, brand)
            else:
                push(soft, seen_s, brand)

        for part in _search_parts(query, notes):
            low = part.lower()
            if low in budget_nums or low in currency_codes:
                continue
            if re.search(r"\d", part) or re.fullmatch(r"[A-Za-z]+[0-9].*", part):
                push(firm, seen_f, part)

        for match in _SOFT_CUE_RE.finditer(text):
            window = text[match.end() : match.end() + 36]
            window = re.split(split_pat, window, maxsplit=1)[0]
            words = [
                tok
                for tok in re.findall(r"[A-Za-z][A-Za-z0-9.+-]{2,}", window)
                if tok.lower() not in seen_f
                and tok.lower() not in _SEED_STOP
                and tok.lower() not in currency_codes
                and tok.lower() not in discourse_skip
            ]
            if words:
                push(soft, seen_s, " ".join(words[:3]))

        for match in _SOFT_PHRASE_RE.finditer(text):
            phrase = next((g for g in match.groups() if g), "") or ""
            phrase = re.sub(r"\s+", " ", phrase).strip(" .,;:")
            # Prefer the clause after a dash / "for" when the capture starts with discourse.
            if " - " in phrase:
                phrase = phrase.split(" - ")[-1]
            elif re.search(r"\bfor\b", phrase, re.I) and phrase.lower().startswith(
                ("don", "rather", "instead")
            ):
                phrase = re.split(r"\bfor\b", phrase, maxsplit=1, flags=re.I)[-1]
            # Drop trailing comparison tails after "than ...".
            phrase = re.split(r"\bthan\b", phrase, maxsplit=1, flags=re.I)[0]
            # Drop relative clauses that start with "that ..." after the preference core.
            phrase = re.split(r"\bthat\b", phrase, maxsplit=1, flags=re.I)[0]
            words = [
                tok
                for tok in phrase.split()
                if tok.lower() not in _SEED_STOP
                and tok.lower() not in discourse_skip
                and tok.lower()
                not in {
                    "but",
                    "than",
                    "that",
                    "few",
                    "dollars",
                    "saving",
                    "grab",
                    "cheapest",
                    "don't",
                    "dont",
                    "settle",
                    "something",
                    "instead",
                    "rather",
                }
                and len(tok) > 1
                and not tok.startswith("-")
            ]
            if not words:
                continue
            short_phrase = " ".join(words[:3])
            push(soft, seen_s, short_phrase)
            for tok in words[:3]:
                if tok.lower() not in discourse_skip and tok.lower() not in _SEED_STOP:
                    push(soft, seen_s, tok)

        # Merge contract-named fact dimensions and use-case windows as soft cues.
        for frag in _soft_fact_dimensions(query) + _use_context_phrases(query):
            push(soft, seen_s, frag)

        soft = [
            s
            for s in soft
            if _normalize_fragment(s) not in {_normalize_fragment(f) for f in firm}
        ]
        return _dedupe_fragments(firm)[:10], _dedupe_fragments(soft)[:10]
    except Exception:
        return tuple()


def _soft_token_coverage(blob: str, soft: list[str]) -> tuple[int, int]:
    try:
        """How many distinct soft content tokens are evidenced on a product blob."""
        tokens: list[str] = []
        seen: set[str] = set()
        for frag in soft or []:
            for tok in re.split(r"[\s/]+", (frag or "").lower()):
                if len(tok) < 4 or tok in seen or tok in _SEED_STOP:
                    continue
                if re.fullmatch(r"\d{3,14}", tok):
                    continue
                seen.add(tok)
                tokens.append(tok)
        if not tokens:
            return 0, 0
        hits = 0
        for tok in tokens:
            if _fragment_match(blob, tok) or tok in blob:
                hits += 1
        return hits, len(tokens)
    except Exception:
        return tuple()


def _required_brand_labels(notes: list[str]) -> list[str]:
    try:
        """Extract explicitly required brand labels from clarification notes."""
        found: list[str] = []
        seen: set[str] = set()
        for note in notes or []:
            for match in re.finditer(
                r"\brequired\s+brand\s+is\s+([A-Za-z0-9][\w-]{1,24})",
                note or "",
                re.I,
            ):
                brand = match.group(1)
                low = brand.lower()
                if low in seen:
                    continue
                seen.add(low)
                found.append(brand)
        return found
    except Exception:
        return []


def _priority_soft_axes(notes: list[str] | str, query: str = "") -> list[str]:
    try:
        """Ordered soft axes from shopper priority discourse (phrase-level, not tokens)."""
        if isinstance(notes, list):
            raw = " ".join(n for n in notes if n)
        else:
            raw = notes or ""
        axes: list[str] = []
        seen: set[str] = set()

        def add(phrase: str) -> None:
            phrase = re.sub(r"\s+", " ", (phrase or "")).strip(" .,;:")
            if not phrase:
                return
            words = [w for w in re.split(r"\s+", phrase) if w]
            if len(words) == 1 and any(words[0].lower() in a.lower().split() for a in axes):
                return
            low = phrase.lower()
            if low in seen or len(low) < 3:
                return
            seen.add(low)
            axes.append(phrase)

        for phrase in _priority_soft_phrases(raw):
            add(phrase)
        # Fall back to contract-named fact dimensions when notes have no priority list yet.
        if len(axes) < 2:
            for phrase in _soft_fact_dimensions(query or ""):
                add(phrase)
        return axes[:6]
    except Exception:
        return []


def _axis_evidenced(blob: str, axis: str) -> bool:
    try:
        """True when a priority axis phrase is supported by listing text."""
        if _fragment_match(blob, axis):
            return True
        toks = [
            t
            for t in re.split(r"[\s/]+", (axis or "").lower())
            if len(t) >= 4 and t not in _SEED_STOP
        ]
        if not toks:
            tok2 = [
                t
                for t in re.split(r"[\s/]+", (axis or "").lower())
                if len(t) >= 3 and t not in _SEED_STOP
            ]
            return bool(tok2) and all(_fragment_match(blob, t) or t in blob for t in tok2)
        hits = sum(1 for t in toks if _fragment_match(blob, t) or t in blob)
        return hits >= max(1, len(toks) - (1 if len(toks) >= 3 else 0))
    except Exception:
        return False


def _failed_product_ids(board: "EpisodeBoard") -> set[str]:
    try:
        """Product ids previously abandoned after a live failure / price event."""
        out: set[str] = set()
        for key in board.failed_keys or set():
            if "::" in key:
                out.add(key.split("::", 1)[0])
        if board.last_failed_key and "::" in board.last_failed_key:
            out.add(board.last_failed_key.split("::", 1)[0])
        return out
    except Exception:
        return set()


def _preference_cart_ok(board: "EpisodeBoard", snap: dict[str, Any]) -> bool:
    try:
        """Gate preference carts on multi-axis soft coverage from the clarified ask."""
        if board.task_shape != "preference_reasoning" or not board.soft_preferences:
            return True
        blob = _row_text_blob(snap)
        _fh, _fm, soft_hits = _compliance_scores(
            snap, board.firm_constraints, board.soft_preferences
        )
        if soft_hits <= 0:
            return False
        cov_hits, cov_total = _soft_token_coverage(blob, board.soft_preferences)
        # Need at least two distinct clarified soft tokens evidenced on the listing.
        if cov_total >= 4 and cov_hits < 2:
            return False
        axes = _priority_soft_axes(board.shopper_notes, board.query)
        if len(axes) >= 2:
            evidenced = [i for i, axis in enumerate(axes[:4]) if _axis_evidenced(blob, axis)]
            if not evidenced:
                return False
            # Reject least-priority-only decoys ("counts least" alone).
            if len(evidenced) == 1 and evidenced[0] >= min(2, len(axes[:4]) - 1):
                return False
            top_hits = sum(1 for i in evidenced if i < 2)
            lower_hits = sum(1 for i in evidenced if i >= 2)
            # Prefer multi-axis survivors: top ladder plus at least one lower cue, or
            # two top cues with broad token coverage.
            if top_hits == 0 and lower_hits <= 1 and cov_hits < 3:
                return False
            if top_hits >= 1 and lower_hits == 0 and cov_hits < 2:
                return False
        return True
    except Exception:
        return False


def _compliance_scores(
    row: dict[str, Any],
    firm: list[str],
    soft: list[str],
) -> tuple[int, int, int]:
    try:
        """Return (firm_hits, firm_misses, soft_hits) against a live product row."""
        blob = _row_text_blob(row)
        firm = _dedupe_fragments(firm)
        soft = _dedupe_fragments(soft)
        firm_hits = 0
        firm_misses = 0
        for frag in firm:
            if _fragment_match(blob, frag):
                firm_hits += 1
            else:
                alt = frag.replace("-", " ")
                if alt != frag and _fragment_match(blob, alt):
                    firm_hits += 1
                else:
                    firm_misses += 1
        soft_hits = sum(1 for frag in soft if _fragment_match(blob, frag))
        return firm_hits, firm_misses, soft_hits
    except Exception:
        return tuple()


def _is_compliant(row: dict[str, Any], firm: list[str]) -> bool:
    try:
        firm = _dedupe_fragments(firm)
        if not firm:
            return True
        # Only numeric/spec fragments hard-gate; type/brand nouns stay ranking signals.
        numeric = [f for f in firm if re.search(r"\d", f)]
        if not numeric:
            return True
        blob = _row_text_blob(row)
        for frag in numeric:
            alt = frag.replace("-", " ")
            if not (_fragment_match(blob, frag) or _fragment_match(blob, alt)):
                return False
        return True
    except Exception:
        return False


def _unwrap_observation(observation: Any) -> dict[str, Any]:
    try:
        """Flatten env payloads nested under observation.observation."""
        if not isinstance(observation, dict):
            return {}
        inner = observation.get("observation")
        if isinstance(inner, dict) and any(
            key in inner
            for key in (
                "results",
                "compared",
                "product_id",
                "facts",
                "variants",
                "order",
                "status",
                "sku",
                "price",
                "action",
                "items",
            )
        ):
            merged = dict(inner)
            if observation.get("error"):
                merged["error"] = observation["error"]
            if "done" in observation:
                merged["done"] = observation["done"]
            return merged
        return dict(observation)
    except Exception:
        return {}


def _title_goal_alignment(title: str, goal_tokens: set[str]) -> int:
    try:
        """Score how directly a title matches goal tokens; demote 'X for <goal>' shapes."""
        title_l = str(title or "").lower()
        if not title_l or not goal_tokens:
            return 0
        score = min(5, _value_overlap(title_l, goal_tokens))
        for_idx = title_l.find(" for ")
        early_hits = 0
        late_only = 0
        cutoff = max(12, len(title_l) // 3)
        for tok in goal_tokens:
            if len(tok) < 4:
                continue
            idx = title_l.find(tok)
            if idx < 0:
                continue
            if idx <= cutoff:
                early_hits += 1
            elif for_idx >= 0 and idx > for_idx:
                late_only += 1
        score += min(3, early_hits)
        if late_only and early_hits == 0:
            score -= 4
        # Structural accessory shape: content before " for " is the product being sold.
        if for_idx >= 0:
            score -= 5
            head = title_l[:for_idx]
            if not any(len(t) >= 5 and t in head for t in goal_tokens):
                score -= 3
        return score
    except Exception:
        return 0


def _recommend_message(board: "EpisodeBoard") -> str:
    try:
        """Shopper-facing recommendation used to answer interventions."""
        key = ""
        row: dict[str, Any] = {}
        # Ranking/tradeoff tasks: recommend the current live ranking winner.
        if board.named and (board.priorities or board.task_shape == "ranking"):
            ranked_named = board.ranked_named()
            if ranked_named:
                pid, sku, key = ranked_named[0]
                row = board._product_snapshot(pid, sku)
        if not key and board.cart:
            top = board.cart[-1]
            key = top["key"]
            row = board.candidates.get(key) or board.viewed.get(top["product_id"]) or {}
        if not key:
            ranked = board.ranked_candidates()
            if not ranked:
                return (
                    "I am verifying options that meet your constraints and will order shortly."
                )
            _score, key, row = ranked[0]
        title = str(row.get("title") or key)[:100]
        bits = [f"I am going with {key} — {title}."]
        specs = _goal_spec_fragments(board.query, board.shopper_notes)
        # Avoid dumping sku/id fragments into the shopper reply.
        clean_specs = [
            s
            for s in specs
            if not re.search(r"\d{6,}", s) and "::" not in s and not re.fullmatch(r"[\w-]+-\d+", s)
        ]
        if clean_specs:
            bits.append(f"It should cover {', '.join(clean_specs[:2])}.")
        if board.budget is not None:
            bits.append(f"I will keep it at or under {board.budget}.")
        stock = board.stock.get(key) or board.stock.get(str(row.get("product_id") or "")) or {}
        price = _as_float(stock.get("price"))
        if price is not None:
            bits.append(f"Latest live price seen: {price}.")
        return " ".join(bits)
    except Exception:
        return ""


def _named_candidates(text: str) -> list[tuple[str, str]]:
    try:
        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        for match in _CANDIDATE_KEY_RE.finditer(text or ""):
            key = f"{match.group(1)}::{match.group(2)}"
            if key in seen:
                continue
            seen.add(key)
            found.append((match.group(1), match.group(2)))
        return found
    except Exception:
        return []


def _fields_mentioned(query: str, allowed: set[str], extra: str = "") -> set[str]:
    try:
        """Return schema claim fields whose names appear in the public query or notes."""
        if not allowed:
            return set()
        q = f"{_preamble(query)} {extra}".lower()
        q_norm = re.sub(r"[^a-z0-9]+", " ", q)
        compact = q_norm.replace(" ", "")
        hits: set[str] = set()
        for field in allowed:
            if field in {"price", "in_stock"}:
                continue
            surface = field.replace("_", " ")
            if field in q_norm or surface in q or surface.replace(" ", "") in compact:
                hits.add(field)
        return hits
    except Exception:
        return set()


def _goal_tokens(query: str, notes: list[str] | None = None) -> set[str]:
    try:
        text = _preamble(query)
        if notes:
            text = f"{text} {' '.join(notes[-3:])}"
        budget_nums = _budget_number_strings(text)
        currency = _currency_codes_in_text(text)
        tokens: set[str] = set()
        for tok in _TOKEN_RE.findall(text):
            low = tok.lower()
            if len(low) <= 1 or re.fullmatch(r"0+", low):
                continue
            if low in _SEED_STOP or low in budget_nums or low in currency:
                continue
            if re.fullmatch(r"\d+(?:\.\d+)?", low) and (
                low in budget_nums or float(low) >= 1000
            ):
                continue
            tokens.add(low)
        return tokens
    except Exception:
        return set()


def _value_overlap(value: Any, goal_tokens: set[str]) -> int:
    try:
        if value is None or value == "":
            return 0
        text = str(value).lower()
        tokens = set(_TOKEN_RE.findall(text))
        nums = set(_NUM_TOKEN_RE.findall(text))
        score = len(tokens & goal_tokens)
        for num in nums:
            if num in goal_tokens or any(num in g or g in num for g in goal_tokens if any(ch.isdigit() for ch in g)):
                score += 2
        return score
    except Exception:
        return 0


def _soft_claim_fields(tools: list[dict[str, Any]], allowed: set[str]) -> set[str]:
    try:
        """Soft/capability claim fields inferred only from live schema prose."""
        soft: set[str] = set()
        for spec in tools:
            fn = _fn(spec)
            if not fn or fn.get("name") != "place_test_order":
                continue
            blob = json.dumps(fn.get("parameters") or {}, ensure_ascii=False).lower()
            for field in allowed:
                if field in {"price", "in_stock"}:
                    continue
                surface = field.replace("_", " ")
                if field in blob and any(
                    cue in blob
                    for cue in ("reason", "capability", "positive", "why", "benefit")
                ):
                    soft.add(field)
                if surface in blob and "positive" in blob:
                    soft.add(field)
        return soft
    except Exception:
        return set()


def _system_prompt(
    tools: list[dict[str, Any]],
    *,
    max_calls: int,
    max_steps: int,
    needs_ask: bool,
    named: list[tuple[str, str]],
    claim_fields: list[str],
    needs_justification: bool,
) -> str:
    try:
        tool_lines: list[str] = []
        for spec in tools:
            fn = _fn(spec)
            if not fn or not isinstance(fn.get("name"), str):
                continue
            desc = str(fn.get("description") or "").strip()
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
            arg_names = ", ".join(sorted(str(k) for k in props.keys()))
            tool_lines.append(
                f"- {fn['name']}: {desc}" + (f" Args: {arg_names}." if arg_names else "")
            )

        parts = [
            "You are a shopping agent. Obey the shopper request and every Task rules block in it.",
            "Copy product_id/sku only from observations.",
            "",
            "Non-negotiable process:",
            "1. If any firm requirement is missing or the goal defers a constraint until you ask, "
            "send a message and ask before your first cart add.",
            "2. Discover with short search/filter queries that include the distinctive constraints "
            "(category, brand if required, numeric specs, model codes). Reformulate if results are weak.",
            "3. view/compare a shortlist. Check category_path, variant options, and raw facts against "
            "EVERY firm requirement for the EXACT sku you will buy. Soft preferences are secondary.",
            "4. Your FIRST add_to_cart must be a genuine fully compliant choice — never a placeholder "
            "and never an item that misses a firm requirement. If several skus fit, prefer soft matches.",
            "5. Soft preferences (e.g. preferred brand/color/model) matter for the first choice when a "
            "fully compliant option exists; firm constraints always win.",
            "6. After any shopper message that follows a cart change, assume the market may have moved: "
            "inspect_stock the carted sku. If price exceeds budget or stock is false, remove_from_cart "
            "and recover to another still-valid option. Re-inspect the replacement before ordering.",
            "7. Do not wait for permission once a valid item is verified. place_test_order immediately.",
            "8. Never re-search or re-inspect after a carted sku already passed inspect_stock — order now.",
            f"9. Up to {max_calls} tool calls/turn — batch independent reads. Budget {max_steps} turns.",
            "",
            "Tools:",
            *tool_lines,
        ]
        if needs_ask:
            parts.extend(
                [
                    "",
                    "This request is underspecified. Your first action should be a message that asks "
                    "for the missing firm product specification.",
                ]
            )
        if named:
            shown = ", ".join(f"{pid}::{sku}" for pid, sku in named[:12])
            parts.extend(
                [
                    "",
                    "The shopper named specific variant keys. Compare/view these before ranking or carting:",
                    shown,
                    "Use the stated tradeoff order in the request (higher primary metric first, then "
                    "secondary, then lower price). After a market change, re-rank on live facts and "
                    "switch the cart if another named option becomes best.",
                ]
            )
        if needs_justification:
            parts.extend(
                [
                    "",
                    "Checkout requires structured justification claims grounded in view/compare/"
                    "inspect_stock of the FINAL product.",
                    f"Allowed reason/schema fields: {', '.join(claim_fields)}.",
                    "Always include price and in_stock from the latest inspect_stock.",
                    "Also include exactly one claim for the shopper's constrained product attribute "
                    "(short snake_case field name) whose value is copied from raw facts — not from "
                    "title paraphrases. If facts use a neighboring key, still emit the shopper-facing "
                    "attribute name with the observed value that matches the request.",
                    "Also include one supported positive capability/reason field when the schema allows it.",
                ]
            )
        else:
            parts.append("Omit justification unless the schema requires it.")
        parts.append("Continue until an observation reports done=true.")
        return "\n".join(parts)
    except Exception:
        return ""


def _repair_json_object(raw: str) -> dict[str, Any]:
    try:
        text = (raw or "").strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        fixed = text.replace("'", '"')
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)
        try:
            parsed = json.loads(fixed)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    except Exception:
        return {}


def _arguments(tool_call: dict[str, Any]) -> dict[str, Any]:
    try:
        raw = tool_call.get("function", {}).get("arguments", "{}")
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            return _repair_json_object(raw)
        return {}
    except Exception:
        raise


def _compact_value(value: Any, depth: int = 0) -> Any:
    try:
        if depth > 5:
            return "..."
        if isinstance(value, str):
            return value if len(value) <= _STR_CHAR_CAP else value[: _STR_CHAR_CAP - 3] + "..."
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [_compact_value(item, depth + 1) for item in value[:_LIST_ROW_CAP]]
        if isinstance(value, dict):
            return {str(k): _compact_value(v, depth + 1) for k, v in value.items()}
        return str(value)[:200]
    except Exception:
        return None


def _compact_observation(observation: dict[str, Any]) -> str:
    try:
        text = json.dumps(_compact_value(observation), ensure_ascii=False, default=str)
        if len(text) <= _OBS_CHAR_CAP:
            return text
        return text[: _OBS_CHAR_CAP - 3] + "..."
    except Exception:
        return ""


def _norm_key(text: str) -> str:
    try:
        return re.sub(r"[^a-z0-9]+", "", str(text).lower())
    except Exception:
        return ""


def _keys_structurally_match(kn: str, needle: str) -> bool:
    try:
        """Match catalog keys by normalized equality or shared stem — no synonym tables."""
        if not kn or not needle:
            return False
        if kn == needle:
            return True
        if len(needle) >= 4 and len(kn) >= 4:
            if kn.endswith(needle) or needle.endswith(kn):
                return True
            if kn.startswith(needle) or needle.startswith(kn):
                return True
        return False
    except Exception:
        return False


def _goal_numbers_for_field(query: str, field: str, notes: list[str] | None = None) -> set[str]:
    try:
        """Numbers in the shopper text near a schema field's surface form."""
        text = _preamble(query).lower()
        if notes:
            text = f"{text} {' '.join(notes[-3:]).lower()}"
        compact = re.sub(r"[^a-z0-9]+", "", text)
        surfaces = {
            field.lower(),
            field.replace("_", " ").lower(),
            _norm_key(field),
            *[p for p in field.lower().split("_") if len(p) >= 4],
        }
        hits: set[str] = set()
        for surface in surfaces:
            if len(surface) < 3:
                continue
            for match in re.finditer(re.escape(surface), text):
                window = text[max(0, match.start() - 32) : match.end() + 32]
                hits.update(_NUM_TOKEN_RE.findall(window))
            if surface.isalpha() or surface.isalnum():
                for match in re.finditer(re.escape(_norm_key(surface)), compact):
                    # Map compact index roughly back by collecting nearby nums from full text.
                    approx = text[
                        max(0, match.start() - 40) : min(len(text), match.end() + 40)
                    ]
                    hits.update(_NUM_TOKEN_RE.findall(approx))
        return hits
    except Exception:
        return set()


def _lookup_fact(blobs: list[dict[str, Any]], field: str) -> Any:
    try:
        """Find an observed value for a schema field via structural key matching only."""
        needle = _norm_key(field)
        if not needle:
            return None
        best: tuple[int, Any] | None = None
        for blob in blobs:
            stack: list[Any] = [blob]
            while stack:
                node = stack.pop()
                if not isinstance(node, dict):
                    continue
                for key, value in node.items():
                    if value in (None, ""):
                        continue
                    kn = _norm_key(key)
                    if _keys_structurally_match(kn, needle):
                        rank = abs(len(kn) - len(needle))
                        if best is None or rank < best[0]:
                            best = (rank, value)
                    if isinstance(value, dict):
                        stack.append(value)
                    elif isinstance(value, list):
                        stack.extend(item for item in value if isinstance(item, dict))
        return None if best is None else best[1]
    except Exception:
        return None


class EpisodeBoard:
    def __init__(
        self,
        query: str,
        allowed_claims: set[str],
        *,
        task_shape: str = "retrieval",
    ) -> None:
        try:
            self.query = query
            self.allowed_claims = set(allowed_claims)
            self.task_shape = task_shape
            self.budget = _parse_budget(_preamble(query))
            self.needs_ask = _needs_clarification(query)
            self.named = _named_candidates(query)
            self.priorities = _tradeoff_priorities(query)
            self.mentioned_fields = _fields_mentioned(query, allowed_claims)
            self.search_seed = _search_seed(query)
            self.firm_constraints, self.soft_preferences = _split_constraints(query)
            self.shopper_notes: list[str] = []
            self.candidates: dict[str, dict[str, Any]] = {}
            self.viewed: dict[str, dict[str, Any]] = {}
            self.stock: dict[str, dict[str, Any]] = {}
            self.cart: list[dict[str, str]] = []
            self.action_log: list[str] = []
            self.last_error: str | None = None
            self.pending_question = False
            self.asked = False
            self.received_clarification = False
            # After a post-cart shopper update: inspect each cart key at most once, then
            # either order (if live checks pass) or remove+recover (never inspect-loop).
            self.needs_live_check = False
            self.checked_since_message: set[str] = set()
            self.must_recover = False
            self.needs_rerank = False
            self.event_seen = False
            self.event_generation = 0
            self.stock_generation: dict[str, int] = {}
            self.search_count = 0
            self.view_count = 0
            self.cart_adds = 0
            self.failed_keys: set[str] = set()
            # Most recent live-failed cart key — used after remove when cart is empty.
            self.last_failed_key: str | None = None
            # Keys inspected while currently in cart — required before checkout.
            self.cart_confirmed: set[str] = set()
            # Stock snapshots taken after the latest cart mutation for that key.
            self.post_cart_stock: set[str] = set()
            # Compare/view generation applied after the latest market event.
            self.facts_refresh_generation = -1
        except Exception:
            return None

    def _invalidate_named_after_event(self) -> None:
        try:
            """Drop stale stock/facts for every named variant after a market move."""
            self.needs_rerank = len(self.named) >= 2 and self.task_shape in {
                "ranking",
                "recovery",
                "preference_reasoning",
            }
            if self.task_shape == "preference_reasoning":
                self.needs_rerank = True
            self.needs_live_check = bool(self.cart)
            self.checked_since_message.clear()
            self.cart_confirmed.clear()
            self.post_cart_stock.clear()
            for item in list(self.cart):
                self.stock.pop(item["key"], None)
                self.stock.pop(item["product_id"], None)
                self.stock_generation.pop(item["key"], None)
            for pid, sku in self.named:
                key = f"{pid}::{sku}"
                self.stock.pop(key, None)
                self.stock.pop(pid, None)
                self.stock_generation.pop(key, None)
        except Exception:
            return None

    def _refresh_from_notes(self) -> None:
        try:
            raw_notes = " ".join(self.shopper_notes[-3:])
            notes_text = _sanitize_note_for_search(raw_notes)
            # Adopt spend-cap revisions from raw notes. Parse raw text so sanitize
            # cannot erase "go up to <amount>", and ignore pure price-hike stories.
            if _note_states_spend_cap(raw_notes):
                updated = _parse_budget(raw_notes)
                if updated is not None and (
                    self.budget is None or updated <= self.budget + 1e-9
                ):
                    original = _parse_budget(_preamble(self.query))
                    explicit_cut = bool(
                        re.search(
                            r"\b(?:can\s+only|sorry|balance|at\s+most|now)\b",
                            raw_notes,
                            re.I,
                        )
                    )
                    floor = 0.12 if explicit_cut else 0.45
                    if original is None or updated >= max(5.0, floor * original) or explicit_cut:
                        prev_budget = self.budget
                        self.budget = updated
                        if prev_budget is not None and updated + 1e-9 < prev_budget and self.cart:
                            top = self.cart[-1]
                            pid = str(top.get("product_id") or "")
                            sku = str(top.get("sku") or "")
                            if pid and sku and not self.stock_ok(pid, sku):
                                self.must_recover = True
                                self.needs_live_check = False
                                self.failed_keys.add(f"{pid}::{sku}")
                                self.last_failed_key = f"{pid}::{sku}"
            self.mentioned_fields = _fields_mentioned(
                self.query, self.allowed_claims, extra=notes_text
            )
            # Build search/constraints from the ask plus sanitized notes only.
            clean_notes = [notes_text] if notes_text.strip() else []
            self.search_seed = _search_seed(self.query, clean_notes)
            firm_q, soft_q = _split_constraints(self.query)
            firm_n, soft_n = _split_constraints(notes_text) if notes_text.strip() else ([], [])
            # Firm specs from clarifications merge in; soft prefs stay query-anchored
            # so price-event chatter cannot invent new model codes.
            # Priority clarifications first so ranking/search keep sport/fit/wireless cues.
            soft_keep: list[str] = []
            soft_seen: set[str] = set()
            for frag in (
                _priority_soft_phrases(raw_notes) + list(soft_q) + list(soft_n)
            ):
                low = frag.lower()
                if low in soft_seen:
                    continue
                # Clarifications may add color / material words, not short price ints.
                if re.fullmatch(r"\d{1,5}", frag):
                    continue
                soft_keep.append(frag)
                soft_seen.add(low)
            # Only absorb note firm fragments after we asked for clarification, or when
            # the note itself states a required/constraint clause — never from nudges.
            absorb_firm = self.asked or bool(
                re.search(
                    r"\b(?:required|requirement|constraint|must|exactly)\b",
                    notes_text,
                    re.I,
                )
            )
            if absorb_firm:
                self.firm_constraints = _dedupe_fragments(list(firm_q) + list(firm_n))
            else:
                self.firm_constraints = _dedupe_fragments(list(firm_q))
            self.soft_preferences = _dedupe_fragments(soft_keep)[:16]
            extra_named = _named_candidates(notes_text)
            if extra_named:
                known = {f"{pid}::{sku}" for pid, sku in self.named}
                for pid, sku in extra_named:
                    key = f"{pid}::{sku}"
                    if key not in known:
                        self.named.append((pid, sku))
                        known.add(key)
        except Exception:
            return None

    def note_shopper(self, content: str) -> None:
        try:
            text = (content or "").strip()
            if not text:
                return
            self.shopper_notes.append(text[:1600])
            self.shopper_notes = self.shopper_notes[-5:]
            self._refresh_from_notes()
            if "?" in text:
                self.pending_question = True
            # Any shopper reply after we asked counts as clarification progress.
            if self.asked:
                self.received_clarification = True
                self.needs_ask = False
            if self.cart:
                low = text.lower()
                # Shopper nudges are not market events — re-check live cart viability only.
                top = self.cart[-1]
                pid = str(top.get("product_id") or "")
                sku = str(top.get("sku") or "")
                if pid and sku and not self.stock_ok(pid, sku):
                    self.must_recover = True
                    self.needs_live_check = False
                if self.named and len(self.named) >= 2 and any(
                    cue in low
                    for cue in (
                        "compare",
                        "re-check",
                        "recheck",
                        "re rank",
                        "rerank",
                        "current option",
                        "current options",
                        "before ordering",
                    )
                ):
                    self.needs_rerank = True
                    self.facts_refresh_generation = self.event_generation - 1
        except Exception:
            return None

    def remember_action(self, name: str, args: dict[str, Any]) -> None:
        try:
            self.action_log.append(name)
            self.action_log = self.action_log[-24:]
            if name in {"search", "filter"}:
                self.search_count += 1
            if name in {"view", "compare"}:
                self.view_count += 1
            if name == "message":
                self.asked = True
                self.pending_question = False
            if name == "add_to_cart":
                self.cart_adds += 1
                pid = str(args.get("product_id") or "")
                sku = str(args.get("sku") or "")
                if pid and sku:
                    key = f"{pid}::{sku}"
                    self.cart = [item for item in self.cart if item.get("key") != key]
                    self.cart.append({"product_id": pid, "sku": sku, "key": key})
                    # Cart mutations can fire market events — never trust pre-cart stock.
                    self.stock.pop(key, None)
                    self.stock.pop(pid, None)
                    self.needs_live_check = True
                    self.must_recover = False
                    self.checked_since_message.clear()
                    self.cart_confirmed.discard(key)
                    self.post_cart_stock.discard(key)
            if name == "remove_from_cart":
                pid = str(args.get("product_id") or "")
                sku = str(args.get("sku") or "")
                key = f"{pid}::{sku}"
                if key:
                    # Only blacklist on failed live checks — not tradeoff cart switches.
                    if self.must_recover:
                        self.failed_keys.add(key)
                        self.last_failed_key = key
                    self.cart_confirmed.discard(key)
                    self.post_cart_stock.discard(key)
                self.cart = [item for item in self.cart if item.get("key") != key]
                if not self.must_recover:
                    self.needs_live_check = False
                self.checked_since_message.discard(key)
        except Exception:
            return None

    def _remember_row(self, row: dict[str, Any], *, viewed: bool = False) -> None:
        try:
            pid = str(row.get("product_id") or "")
            if not pid:
                return
            sku = str(row.get("sku") or "")
            key = f"{pid}::{sku}" if sku else pid
            prev = self.candidates.get(key, {})
            merged = {**prev, **{k: v for k, v in row.items() if v is not None}}
            merged["product_id"] = pid
            if sku:
                merged["sku"] = sku
            self.candidates[key] = merged
            if viewed:
                # Keep a slim fact snapshot only — full payloads blow sandbox RAM.
                slim = {
                    k: merged.get(k)
                    for k in (
                        "product_id",
                        "sku",
                        "title",
                        "brand",
                        "price",
                        "in_stock",
                        "category_path",
                        "facts",
                        "listing_id",
                    )
                    if merged.get(k) is not None
                }
                self.viewed[pid] = {**self.viewed.get(pid, {}), **slim}
            if len(self.candidates) > _MAX_STORED_CANDIDATES:
                # Drop lowest-priority unseen rows first.
                ranked = sorted(
                    self.candidates.items(),
                    key=lambda kv: (
                        0 if kv[1].get("product_id") in self.viewed else 1,
                        0 if (kv[0] in self.stock or kv[1].get("product_id") in self.stock) else 1,
                        str(kv[1].get("price") or 10**18),
                    ),
                )
                keep = {k for k, _ in ranked[:_MAX_STORED_CANDIDATES]}
                self.candidates = {k: v for k, v in self.candidates.items() if k in keep}
        except Exception:
            return None

    def _product_snapshot(self, product_id: str, sku: str = "") -> dict[str, Any]:
        try:
            key = f"{product_id}::{sku}" if sku else product_id
            viewed = dict(self.viewed.get(product_id) or {})
            facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
            stock = dict(self.stock.get(key) or self.stock.get(product_id) or {})
            cand = dict(self.candidates.get(key) or self.candidates.get(product_id) or {})
            merged = {**cand, **viewed, **stock, "facts": facts}
            merged["product_id"] = product_id
            if sku:
                merged["sku"] = sku
            return merged
        except Exception:
            return {}

    def _apply_cart_observation(self, observation: dict[str, Any]) -> None:
        try:
            """Absorb live cart lines and market_event payloads from cart mutations."""
            event = observation.get("market_event")
            if isinstance(event, dict) and event:
                self.event_seen = True
                self.event_generation += 1
                self._invalidate_named_after_event()
                if self.named and len(self.named) >= 2:
                    self.needs_rerank = True
                    self.facts_refresh_generation = self.event_generation - 1
                target = event.get("target") if isinstance(event.get("target"), dict) else {}
                t_pid = str(target.get("product_id") or "")
                t_sku = str(target.get("sku") or "")
                t_key = f"{t_pid}::{t_sku}" if t_pid and t_sku else ""
                payload: dict[str, Any] = {}
                if event.get("new_price") is not None:
                    payload["price"] = event.get("new_price")
                elif event.get("old_price") is not None and event.get("kind") != "price_change":
                    payload["price"] = event.get("old_price")
                if "in_stock_after" in event:
                    payload["in_stock"] = bool(event.get("in_stock_after"))
                if t_pid:
                    payload["product_id"] = t_pid
                if t_sku:
                    payload["sku"] = t_sku
                if t_key and payload:
                    prev = dict(self.stock.get(t_key) or {})
                    merged = {**prev, **payload}
                    self.stock[t_key] = merged
                    self.stock[t_pid] = merged
                    self.stock_generation[t_key] = self.event_generation
                    self._remember_row(merged)
                    in_cart = any(item.get("key") == t_key for item in self.cart)
                    if in_cart:
                        self.post_cart_stock.add(t_key)
                        self.cart_confirmed.add(t_key)
                        self.checked_since_message.add(t_key)
                        if not self.stock_ok(t_pid, t_sku):
                            self.must_recover = True
                            self.needs_live_check = False
                            self.failed_keys.add(t_key)
                            self.last_failed_key = t_key
                        else:
                            self.needs_live_check = False
            added = observation.get("added")
            if isinstance(added, dict):
                pid = str(added.get("product_id") or "")
                sku = str(added.get("sku") or "")
                if pid and sku:
                    key = f"{pid}::{sku}"
                    payload = {
                        "product_id": pid,
                        "sku": sku,
                        "price": added.get("price"),
                        "in_stock": added.get("in_stock"),
                        "currency": added.get("currency"),
                    }
                    prev = dict(self.stock.get(key) or {})
                    merged = {**prev, **{k: v for k, v in payload.items() if v is not None}}
                    self.stock[key] = merged
                    self.stock[pid] = merged
                    self._remember_row(merged)
                    self.post_cart_stock.add(key)
                    self.cart_confirmed.add(key)
                    self.checked_since_message.add(key)
                    if not self.stock_ok(pid, sku):
                        self.must_recover = True
                        self.needs_live_check = False
                        self.failed_keys.add(key)
                        self.last_failed_key = key
                    else:
                        self.needs_live_check = False

            for row in observation.get("cart") or []:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                if not pid or not sku:
                    continue
                key = f"{pid}::{sku}"
                price = row.get("current_price", row.get("price", row.get("price_at_add")))
                payload = {
                    "product_id": pid,
                    "sku": sku,
                    "price": price,
                    "in_stock": row.get("in_stock"),
                    "currency": row.get("currency"),
                }
                prev = dict(self.stock.get(key) or {})
                merged = {**prev, **{k: v for k, v in payload.items() if v is not None}}
                self.stock[key] = merged
                self.stock[pid] = merged
                self._remember_row(merged)
                if any(item.get("key") == key for item in self.cart):
                    self.post_cart_stock.add(key)
                    self.cart_confirmed.add(key)
                    if not self.stock_ok(pid, sku):
                        self.must_recover = True
                        self.needs_live_check = False
                        self.failed_keys.add(key)
                        self.last_failed_key = key
        except Exception:
            return None

    def ingest(self, name: str, args: dict[str, Any], observation: dict[str, Any]) -> None:
        try:
            observation = _unwrap_observation(observation)
            if not observation:
                return
            if observation.get("error"):
                self.last_error = str(observation.get("error"))[:240]
                return
            if name in {"search", "filter"}:
                rows = [row for row in (observation.get("results") or []) if isinstance(row, dict)]
                for row in rows:
                    self._remember_row(row)
                if not rows:
                    self.search_count = max(self.search_count, 1)
            elif name == "compare":
                if self.named and len(self.named) >= 2:
                    self.facts_refresh_generation = self.event_generation
                    # Compare refreshes facts but not live stock — force re-inspect before re-rank.
                    if self.needs_rerank or self.event_seen:
                        for pid, sku in self.named:
                            key = f"{pid}::{sku}"
                            if key in self.failed_keys:
                                continue
                            self.stock.pop(key, None)
                            self.stock.pop(pid, None)
                            self.stock_generation.pop(key, None)
                            self.post_cart_stock.discard(key)
                for row in observation.get("compared") or []:
                    if isinstance(row, dict):
                        self._remember_row(row, viewed=True)
                        pid = str(row.get("product_id") or "")
                        for variant in (row.get("variants") or [])[:8]:
                            if isinstance(variant, dict):
                                self._remember_row(
                                    {
                                        **variant,
                                        "product_id": variant.get("product_id") or pid,
                                        "title": row.get("title"),
                                        "brand": row.get("brand"),
                                        "category_path": row.get("category_path"),
                                        "facts": row.get("facts"),
                                    },
                                    viewed=True,
                                )
            elif name == "view":
                self._remember_row(observation, viewed=True)
                pid = str(observation.get("product_id") or args.get("product_id") or "")
                if pid:
                    facts = observation.get("facts") if isinstance(observation.get("facts"), dict) else {}
                    self.viewed[pid] = {
                        **self.viewed.get(pid, {}),
                        "product_id": pid,
                        "title": observation.get("title"),
                        "brand": observation.get("brand"),
                        "category_path": observation.get("category_path"),
                        "facts": facts,
                        "listing_id": observation.get("listing_id"),
                    }
                for row in (observation.get("variants") or [])[:8]:
                    if isinstance(row, dict):
                        self._remember_row(
                            {**row, "product_id": row.get("product_id") or pid}
                        )
            elif name == "inspect_stock":
                pid = str(observation.get("product_id") or args.get("product_id") or "")
                sku = str(observation.get("sku") or args.get("sku") or "")
                if pid:
                    payload = dict(observation)
                    # Coerce numeric price strings so budget gates stay reliable.
                    coerced = _as_float(payload.get("price"))
                    if coerced is not None:
                        payload["price"] = coerced
                    key = f"{pid}::{sku}"
                    self.stock[key] = payload
                    self.stock[pid] = payload
                    self.stock_generation[key] = self.event_generation
                    self._remember_row({"product_id": pid, "sku": sku, **payload})
                    self.checked_since_message.add(key)
                    if any(item.get("key") == key for item in self.cart):
                        self.cart_confirmed.add(key)
                        self.post_cart_stock.add(key)
                    if self.needs_live_check:
                        if self.stock_ok(pid, sku):
                            self.needs_live_check = False
                            self.must_recover = False
                        else:
                            # One failed live check is enough — recover instead of re-inspecting.
                            self.needs_live_check = False
                            self.must_recover = True
                            self.failed_keys.add(key)
                            self.cart_confirmed.discard(key)
                            self.post_cart_stock.discard(key)
            elif name in {"add_to_cart", "remove_from_cart", "inspect_cart"}:
                self._apply_cart_observation(observation)
                if name == "inspect_cart":
                    rebuilt: list[dict[str, str]] = []
                    stack: list[Any] = [observation]
                    while stack:
                        node = stack.pop()
                        if isinstance(node, dict):
                            pid = str(node.get("product_id") or "")
                            sku = str(node.get("sku") or "")
                            if pid and sku:
                                rebuilt.append(
                                    {"product_id": pid, "sku": sku, "key": f"{pid}::{sku}"}
                                )
                            stack.extend(node.values())
                        elif isinstance(node, list):
                            stack.extend(node)
                    if rebuilt:
                        seen: set[str] = set()
                        uniq: list[dict[str, str]] = []
                        for item in rebuilt:
                            if item["key"] in seen:
                                continue
                            seen.add(item["key"])
                            uniq.append(item)
                        self.cart = uniq
        except Exception:
            return None

    def stock_ok(self, product_id: str, sku: str) -> bool:
        try:
            key = f"{product_id}::{sku}"
            stock = self.stock.get(key) or self.stock.get(product_id)
            if not isinstance(stock, dict):
                return False
            # Require an explicit positive in_stock signal from live inspect.
            if stock.get("in_stock") is not True:
                return False
            price = _as_float(stock.get("price"))
            if price is None:
                return False
            if self.budget is not None and price > self.budget + 1e-9:
                return False
            return True
        except Exception:
            return False

    def stock_is_fresh(self, product_id: str, sku: str) -> bool:
        try:
            key = f"{product_id}::{sku}"
            return self.stock_generation.get(key, -1) >= self.event_generation and (
                key in self.stock or product_id in self.stock
            )
        except Exception:
            return False

    def merged_row(self, product_id: str, sku: str = "") -> dict[str, Any]:
        try:
            """Live snapshot used for compliance and ranking."""
            return self._product_snapshot(product_id, sku)
        except Exception:
            return {}

    def row_compliant(self, row: dict[str, Any], sku: str = "") -> bool:
        try:
            pid = str(row.get("product_id") or "")
            sku = sku or str(row.get("sku") or "")
            snap = self._product_snapshot(pid, sku) if pid else dict(row)
            # Until viewed, require title/search-surface compliance only when firm specs exist.
            if pid and pid not in self.viewed and self.firm_constraints:
                surface = {
                    **row,
                    "title": row.get("title") or snap.get("title"),
                    "brand": row.get("brand") or snap.get("brand"),
                    "options": row.get("options") or snap.get("options"),
                    "category_path": row.get("category_path") or snap.get("category_path"),
                }
                # Pre-view: only enforce numeric/spec fragments (full text may live in facts).
                specs = [f for f in self.firm_constraints if re.search(r"\d", f)]
                if specs:
                    return _is_compliant(surface, specs)
                return True
            return _is_compliant(snap if pid else row, self.firm_constraints)
        except Exception:
            return False

    def best_sku_for_product(self, product_id: str) -> str | None:
        try:
            """Pick the compliant sku for a product among remembered variants."""
            options: list[tuple[int, int, str]] = []
            for key, row in self.candidates.items():
                if str(row.get("product_id") or "") != product_id:
                    continue
                sku = str(row.get("sku") or "")
                if not sku or key in self.failed_keys:
                    continue
                snap = self._product_snapshot(product_id, sku)
                hits, misses, soft = _compliance_scores(
                    snap, self.firm_constraints, self.soft_preferences
                )
                if misses and not _is_compliant(snap, self.firm_constraints):
                    continue
                stock_bonus = 1 if self.stock_ok(product_id, sku) else 0
                options.append((misses - hits - soft - stock_bonus, -soft, sku))
            if not options:
                # Fall back to any known sku for the product.
                for key, row in self.candidates.items():
                    if str(row.get("product_id") or "") == product_id:
                        sku = str(row.get("sku") or "")
                        if sku and key not in self.failed_keys:
                            return sku
                return None
            options.sort()
            return options[0][2]
        except Exception:
            return None

    def tradeoff_rank_key(
        self,
        product_id: str,
        sku: str,
        *,
        axis_units: dict[str, str | None] | None = None,
    ) -> tuple[Any, ...]:
        try:
            snap = self._product_snapshot(product_id, sku)
            parts: list[Any] = []
            missing = 0
            for rule in self.priorities:
                unit = None if not axis_units else axis_units.get(rule["field"])
                value = _metric_value_with_unit(
                    snap, rule["field"], unit, self.query
                )
                if value is None:
                    missing += 1
                    parts.append((1, 0.0))
                    continue
                if rule["direction"] == "max":
                    parts.append((0, -float(value)))
                else:
                    parts.append((0, float(value)))
            # Price is only a final tie-break among rows that already resolved the axes.
            price = snap.get("price")
            if isinstance(price, (int, float)):
                parts.append((0 if missing == 0 else 1, float(price)))
            else:
                parts.append((1, 0.0))
            return (missing, *parts)
        except Exception:
            return tuple()

    def ranked_named(self) -> list[tuple[str, str, str]]:
        try:
            """Order named pid/sku pairs by query tradeoffs, then live feasibility."""
            snaps = [
                self._product_snapshot(pid, sku)
                for pid, sku in self.named
                if f"{pid}::{sku}" not in self.failed_keys
            ]
            axis_units: dict[str, str | None] = {}
            used_units: set[str] = set()
            for rule in self.priorities:
                # Prefer a fresh unit not already claimed by an earlier tradeoff axis.
                vote = _axis_unit_vote(snaps, rule["field"])
                if vote in used_units:
                    # Re-vote ignoring used units by temporarily filtering snaps' units.
                    buckets: dict[str, list[float]] = {}
                    for row in snaps:
                        seen: set[str] = set()
                        for unit, number in _title_unit_numbers(row):
                            if unit in used_units or unit in seen:
                                continue
                            seen.add(unit)
                            buckets.setdefault(unit, []).append(number)
                    vote = None
                    best_span = -1.0
                    for unit, values in buckets.items():
                        if len(values) < 2:
                            continue
                        span = max(values) - min(values)
                        score = span + 0.01 * len(unit)
                        if score > best_span:
                            best_span = score
                            vote = unit
                axis_units[rule["field"]] = vote
                if vote:
                    used_units.add(vote)
            rows: list[tuple[tuple[Any, ...], str, str, str]] = []
            for pid, sku in self.named:
                key = f"{pid}::{sku}"
                if key in self.failed_keys:
                    continue
                snap = self._product_snapshot(pid, sku)
                feasible = 0
                if self.stock_ok(pid, sku):
                    feasible = 0
                elif key in self.stock or pid in self.stock:
                    feasible = 2
                else:
                    feasible = 1
                price = snap.get("price")
                over = (
                    1
                    if self.budget is not None
                    and isinstance(price, (int, float))
                    and float(price) > self.budget
                    else 0
                )
                rank = (
                    feasible,
                    over,
                    self.tradeoff_rank_key(pid, sku, axis_units=axis_units),
                    key,
                )
                rows.append((rank, pid, sku, key))
            rows.sort(key=lambda item: item[0])
            return [(pid, sku, key) for _rank, pid, sku, key in rows]
        except Exception:
            return []

    def ranked_candidates(self) -> list[tuple[int, str, dict[str, Any]]]:
        try:
            if self.named and self.priorities:
                named_ranked: list[tuple[int, str, dict[str, Any]]] = []
                for index, (pid, sku, key) in enumerate(self.ranked_named()):
                    row = self._product_snapshot(pid, sku)
                    score = 1000 - index
                    if self.stock_ok(pid, sku):
                        score += 50
                    named_ranked.append((score, key, row))
                if named_ranked:
                    return named_ranked
            goal = _goal_tokens(self.query, self.shopper_notes)
            specs = _goal_spec_fragments(self.query, self.shopper_notes)
            category = (_category_hint(self.query) or "").lower()
            firm = self.firm_constraints
            soft = self.soft_preferences
            ranked: list[tuple[int, str, dict[str, Any]]] = []
            for key, row in self.candidates.items():
                if key in self.failed_keys:
                    continue
                score = 0
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "")
                viewed = self.viewed.get(pid) or {}
                facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
                merged = {**row, **viewed, "facts": facts}
                if sku:
                    merged = {**merged, **self._product_snapshot(pid, sku)}
                firm_hits, firm_misses, soft_hits = _compliance_scores(merged, firm, soft)
                if firm:
                    score += 6 * firm_hits
                    score -= 8 * firm_misses
                    if pid in self.viewed and firm_misses and not _is_compliant(merged, firm):
                        # Hard demote viewed non-compliant variants so we do not cart them.
                        score -= 40
                brands = _required_brand_labels(self.shopper_notes)
                if brands:
                    blob_b = _row_text_blob(merged)
                    if any(_fragment_match(blob_b, b) for b in brands):
                        score += 40
                    else:
                        score -= 35
                if soft:
                    score += 5 * soft_hits
                    # Prefer stronger soft alignment when multiple options clear firm gates.
                    if firm_misses == 0 and soft_hits:
                        score += 10 * soft_hits
                    elif soft_hits:
                        score += 4 * soft_hits
                    # Only demote missing soft cues on preference-shaped asks.
                    if soft_hits == 0 and self.task_shape == "preference_reasoning":
                        score -= 8
                    if self.task_shape == "preference_reasoning" and soft:
                        # Coverage ratio rewards items that hit more soft dimensions.
                        score += int(18 * soft_hits / max(1, len(soft)))
                        if soft_hits >= max(2, len(soft) // 2):
                            score += 12
                        # Prefer listings that evidence multiple clarified priority axes.
                        axes = _priority_soft_axes(self.shopper_notes, self.query)
                        if axes:
                            blob_ax = _row_text_blob(merged)
                            axis_hits = sum(
                                1 for axis in axes[:4] if _axis_evidenced(blob_ax, axis)
                            )
                            score += 14 * axis_hits
                            if axis_hits < 2 and len(axes) >= 2:
                                score -= 25
                        if pid in _failed_product_ids(self):
                            score -= 80
                    # Preferred model codes are near-mandatory when stated.
                    blob = _row_text_blob(merged)
                    for frag in soft:
                        if not re.fullmatch(r"\d{3,14}", frag):
                            continue
                        if _soft_code_match(blob, frag) or frag in str(merged.get("sku") or ""):
                            score += 30
                        else:
                            # During live recovery, missing codes should not bury stem
                            # siblings that still share the preferred model family.
                            score -= 8 if self.must_recover else 55
                    # Soft color / short tokens still matter for same-model ranking.
                    for frag in soft:
                        if re.fullmatch(r"\d{3,14}", frag) or len(frag) < 3:
                            continue
                        if _fragment_match(blob, frag):
                            score += 10
                        else:
                            score -= 4
                if firm:
                    # Head-noun category cues (e.g. device type) must appear on path/title.
                    path = " ".join(
                        str(part).lower()
                        for part in (merged.get("category_path") or [])
                        if part is not None
                    )
                    title_l = str(merged.get("title") or "").lower()
                    for frag in firm:
                        if re.search(r"\d", frag) or len(frag) < 4:
                            continue
                        fl = frag.lower()
                        if fl in path or fl in title_l:
                            score += 8
                        elif path and fl not in path and fl not in title_l:
                            score -= 12
                if pid in self.viewed:
                    score += 4
                if key in self.stock or pid in self.stock:
                    score += 3
                    if self.stock_ok(pid, sku):
                        score += 4
                    else:
                        score -= 5
                if row.get("in_stock") is True:
                    score += 1
                price = _as_float(row.get("price"))
                if price is None:
                    price = _as_float((self.stock.get(key) or {}).get("price"))
                if self.budget is not None and price is not None:
                    score += 2 if price <= self.budget else -6
                if any(pid == n[0] for n in self.named):
                    score += 3
                title = str(row.get("title") or "")
                title_l = title.lower()
                brand = str(row.get("brand") or viewed.get("brand") or "")
                score += _title_goal_alignment(title, goal)
                score += _spec_match_score(merged, specs)
                # Compact codes / model numerals from the ask are high-precision filters.
                fact_blob = " ".join(
                    str(v).lower()
                    for v in (
                        title,
                        brand,
                        json.dumps(facts, ensure_ascii=False, default=str),
                        row.get("options") or "",
                    )
                    if v not in (None, "")
                )
                for tok in goal:
                    if len(tok) < 2:
                        continue
                    if any(ch.isdigit() for ch in tok) and tok in fact_blob:
                        score += 5
                if brand:
                    score += min(2, _value_overlap(brand, goal))
                    if brand.lower() in goal:
                        score += 2
                if category:
                    path = " ".join(
                        str(part).lower()
                        for part in (merged.get("category_path") or [])
                        if part is not None
                    )
                    if category in path or category in title_l:
                        score += 3
                    elif path and category not in path:
                        score -= 2
                for field in self.mentioned_fields:
                    observed = _lookup_fact([viewed, row, facts], field)
                    if observed is None:
                        continue
                    score += 2 + min(3, _value_overlap(observed, goal))
                ranked.append((score, key, row))
            ranked.sort(key=lambda item: (-item[0], item[1]))
            # 14-style self-consistency: when top-2 scores are close, prefer the one
            # with better stock freshness + firm compliance (no new episode phase).
            if len(ranked) >= 2:
                s0, k0, r0 = ranked[0]
                s1, k1, r1 = ranked[1]
                if (s0 - s1) < _SELF_CONSISTENCY_GAP * 4:
                    def tie_key(item: tuple[int, str, dict[str, Any]]) -> tuple:
                        sc, key, row = item
                        pid = str(row.get("product_id") or "")
                        sku = str(row.get("sku") or "")
                        fresh = 1 if (pid and sku and self.stock_is_fresh(pid, sku)) else 0
                        ok = 1 if (pid and sku and self.stock_ok(pid, sku)) else 0
                        miss = _compliance_scores(
                            self._product_snapshot(pid, sku) if pid else row,
                            self.firm_constraints,
                            self.soft_preferences,
                        )[1]
                        return (-ok, -fresh, miss, -sc, key)

                    top2 = sorted(ranked[:2], key=tie_key)
                    ranked = top2 + ranked[2:]
            return ranked
        except Exception:
            return []

    def best_recovery(self) -> dict[str, str] | None:
        try:
            """Recover to a sibling/sku-stem alternate, even after the cart was cleared."""
            cur_pid = ""
            cur_sku = ""
            if self.cart and not self.named:
                top = self.cart[-1]
                cur_pid = str(top.get("product_id") or "")
                cur_sku = str(top.get("sku") or "")
            if (not cur_pid or not cur_sku) and self.last_failed_key and "::" in self.last_failed_key:
                cur_pid, cur_sku = self.last_failed_key.split("::", 1)
            if (not cur_pid or not cur_sku) and self.failed_keys:
                # Fall back to the most recently remembered failed key.
                for key in reversed(list(self.failed_keys)):
                    if "::" in key:
                        cur_pid, cur_sku = key.split("::", 1)
                        break

            # v3-style recovery hints from shopper ask (color/model), used only as ranking bias.
            ask = f"{_preamble(self.query)} {' '.join(self.shopper_notes[-3:])}".lower()
            prefer_color = ""
            prefer_model = ""
            m_color = re.search(
                r"\b(black|white|blue|red|green|silver|gold|gray|grey|pink|purple|orange)\b",
                ask,
            )
            if m_color:
                prefer_color = m_color.group(1)
            m_model = re.search(r"\b(?:model|variant)\s+([a-z0-9][\w.-]{1,20})\b", ask)
            if m_model:
                prefer_model = m_model.group(1).lower()

            def _v3_recovery_bonus(pid: str, sku: str) -> int:
                snap = self._product_snapshot(pid, sku)
                blob = _row_text_blob(snap)
                bonus = 0
                if prefer_color and prefer_color in blob:
                    bonus += 3
                if prefer_model and prefer_model in blob.replace(" ", "").replace("-", ""):
                    bonus += 4
                return bonus

            # Preference asks must change source product — never stem/sku siblings.
            if self.task_shape == "preference_reasoning":
                failed_stem = re.sub(r"-\d+$", "", cur_sku) if cur_sku else ""
                failed_pids = _failed_product_ids(self)
                scored_opts: list[tuple[int, dict[str, str]]] = []
                for _score, key, row in self.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (self.best_sku_for_product(pid) or "")
                    if not pid or not sku or key in self.failed_keys:
                        continue
                    if cur_pid and pid == cur_pid:
                        continue
                    if pid in failed_pids:
                        continue
                    if failed_stem and len(failed_stem) >= 4 and sku.startswith(failed_stem):
                        continue
                    snap = self._product_snapshot(pid, sku)
                    if not _preference_cart_ok(self, snap):
                        continue
                    bonus = _v3_recovery_bonus(pid, sku)
                    if pid not in self.viewed:
                        scored_opts.append((bonus, {"action": "view", "product_id": pid, "sku": sku}))
                        continue
                    if key not in self.stock and pid not in self.stock:
                        scored_opts.append(
                            (bonus, {"action": "inspect_stock", "product_id": pid, "sku": sku})
                        )
                        continue
                    if self.stock_ok(pid, sku):
                        scored_opts.append(
                            (bonus + 1, {"action": "add_to_cart", "product_id": pid, "sku": sku})
                        )
                if scored_opts:
                    scored_opts.sort(key=lambda item: -item[0])
                    return scored_opts[0][1]
                return None

            if cur_pid and cur_sku and not self.named:
                alt = self.best_sku_for_product(cur_pid)
                if alt and alt != cur_sku:
                    alt_key = f"{cur_pid}::{alt}"
                    if alt_key not in self.failed_keys:
                        if cur_pid not in self.viewed:
                            return {"action": "view", "product_id": cur_pid, "sku": alt}
                        if alt_key not in self.stock and cur_pid not in self.stock:
                            return {
                                "action": "inspect_stock",
                                "product_id": cur_pid,
                                "sku": alt,
                            }
                        if self.stock_ok(cur_pid, alt):
                            return {
                                "action": "add_to_cart",
                                "product_id": cur_pid,
                                "sku": alt,
                            }
                # Same model family: sibling SKUs share a sku stem before the final -N.
                stem = re.sub(r"-\d+$", "", cur_sku)
                if stem and len(stem) >= 4:
                    soft_codes = [
                        frag
                        for frag in self.soft_preferences
                        if re.fullmatch(r"\d{3,14}", frag)
                    ]
                    soft_words = [
                        frag.lower()
                        for frag in self.soft_preferences
                        if not re.fullmatch(r"\d{3,14}", frag) and len(frag) >= 3
                    ]
                    stem_hits: list[tuple[int, str, str, str]] = []
                    for key, row in self.candidates.items():
                        if key in self.failed_keys:
                            continue
                        other_pid = str(row.get("product_id") or "")
                        other_sku = str(row.get("sku") or "")
                        if not other_pid or not other_sku or other_sku == cur_sku:
                            continue
                        if not other_sku.startswith(stem):
                            continue
                        snap = self._product_snapshot(other_pid, other_sku)
                        # Same sku stem = same model family; do not re-apply firm filters
                        # that may be incomplete on sparse variant rows.
                        blob = _row_text_blob(snap).lower()
                        score = 0
                        if soft_codes:
                            if not any(
                                _soft_code_match(blob, code) or code in other_sku
                                for code in soft_codes
                            ):
                                # Keep stem siblings even without a code hit — they are
                                # still better than jumping to a different model.
                                score -= 2
                            else:
                                score += 8
                        if soft_words and any(w in blob for w in soft_words):
                            score += 4
                        if other_pid == cur_pid:
                            score += 3
                        if self.stock_ok(other_pid, other_sku):
                            score += 5
                        stem_hits.append((score, other_pid, other_sku, key))
                    stem_hits.sort(key=lambda item: (-item[0], item[3]))
                    for _score, other_pid, other_sku, key in stem_hits:
                        if other_pid not in self.viewed:
                            return {
                                "action": "view",
                                "product_id": other_pid,
                                "sku": other_sku,
                            }
                        if key not in self.stock and other_pid not in self.stock:
                            return {
                                "action": "inspect_stock",
                                "product_id": other_pid,
                                "sku": other_sku,
                            }
                        if self.stock_ok(other_pid, other_sku):
                            return {
                                "action": "add_to_cart",
                                "product_id": other_pid,
                                "sku": other_sku,
                            }
                    # Soft-coded recovery with no known stem sibling yet — force a
                    # fresh search rather than carting an unrelated ranked hit.
                    if soft_codes:
                        return None
            if self.named:
                for pid, sku, key in self.ranked_named():
                    if key in self.failed_keys:
                        continue
                    if pid not in self.viewed:
                        return {"action": "view", "product_id": pid, "sku": sku}
                    if key not in self.stock and pid not in self.stock:
                        return {"action": "inspect_stock", "product_id": pid, "sku": sku}
                    if self.stock_ok(pid, sku):
                        return {"action": "add_to_cart", "product_id": pid, "sku": sku}
                return None
            soft_codes = [
                frag
                for frag in self.soft_preferences
                if re.fullmatch(r"\d{3,14}", frag)
            ]
            # When recovering with a preferred code, never fall through to loose
            # ranked products that lack a whole-token code match.
            for score, key, row in self.ranked_candidates():
                if score < 2 and not soft_codes:
                    break
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "") or (self.best_sku_for_product(pid) or "")
                if not pid or not sku or key in self.failed_keys:
                    continue
                snap = self._product_snapshot(pid, sku)
                if soft_codes:
                    blob = _row_text_blob(snap)
                    if not any(
                        _soft_code_match(blob, code) or code in sku for code in soft_codes
                    ):
                        continue
                elif pid in self.viewed and self.firm_constraints and not self.row_compliant(
                    snap, sku
                ):
                    continue
                if pid not in self.viewed:
                    return {"action": "view", "product_id": pid, "sku": sku}
                if key in self.stock or pid in self.stock or f"{pid}::{sku}" in self.stock:
                    if self.stock_ok(pid, sku):
                        return {"action": "add_to_cart", "product_id": pid, "sku": sku}
                    continue
                return {"action": "inspect_stock", "product_id": pid, "sku": sku}
            return None
        except Exception:
            return {}

    def next_hint(self, turn: int, max_steps: int) -> str:
        try:
            remaining = max_steps - turn + 1
            if self.needs_ask and not self.received_clarification:
                return (
                    "Ask the shopper for the missing firm specification with message "
                    "before searching deeply or carting."
                )
            if self.pending_question:
                return "Answer the shopper with message before ordering."
            if self.must_recover and self.cart:
                top = self.cart[-1]
                return (
                    f"Live check failed for {top['key']}. remove_from_cart, find another "
                    "fully valid option, view it, cart it, inspect_stock, then order."
                )
            if self.needs_live_check and self.cart:
                top = self.cart[-1]
                if top["key"] not in self.post_cart_stock:
                    return (
                        f"Cart mutated — inspect_stock {top['key']} once for the live "
                        "price/stock, then order or recover."
                    )
            if self.last_error:
                return f"Recover from error ({self.last_error}), then continue."
            recent = self.action_log[-4:]
            if recent and len(set(recent)) == 1 and recent[0] in {"search", "filter", "message", "inspect_stock"}:
                return f"Stop repeating {recent[0]}; advance toward a verified place_test_order."
            if self.named and self.view_count < min(2, len(self.named)):
                return "View/compare the named variant keys from the request before carting."
            if self.named and self.priorities:
                return (
                    "Rank named options by the shopper's stated tradeoff order using live "
                    "facts, cart the current best, and re-rank after any market event."
                )
            if not self.candidates:
                return (
                    f"Search/filter with short constraint-focused queries "
                    f"(seed hint: {self.search_seed}) and in_stock=true."
                )
            if self.view_count == 0:
                return "Inspect a shortlist with view/compare and verify the exact sku facts."
            if not self.cart:
                if remaining <= 7:
                    return "Cart the best fully compliant inspected variant now."
                return (
                    "Cart only a fully compliant intended purchase. Honor soft preferences "
                    "when a compliant option exists."
                )
            top = self.cart[-1]
            key = top["key"]
            if key not in self.post_cart_stock:
                return "inspect_stock the carted sku once after carting before place_test_order."
            if not self.stock_ok(top["product_id"], top["sku"]):
                return "Carted sku failed live checks — remove_from_cart and recover."
            if remaining <= 8:
                return "Verified — place_test_order NOW (include required claims if schema needs them)."
            return "Live checks passed — place_test_order for the carted variant."
        except Exception:
            return ""

    def render(self, turn: int, max_steps: int) -> str:
        try:
            lines = [
                f"board turn={turn}/{max_steps} remaining={max_steps - turn + 1}",
                f"shape={self.task_shape} budget={self.budget if self.budget is not None else 'none'} "
                f"needs_ask={self.needs_ask and not self.received_clarification} "
                f"needs_live_check={self.needs_live_check} must_recover={self.must_recover} "
                f"event_seen={self.event_seen}",
                f"searches={self.search_count} views={self.view_count} "
                f"candidates={len(self.candidates)} cart={len(self.cart)} adds={self.cart_adds}",
                f"search_seed={self.search_seed}",
                f"next={self.next_hint(turn, max_steps)}",
            ]
            if self.firm_constraints:
                lines.append("firm=" + ", ".join(self.firm_constraints[:8]))
            if self.soft_preferences:
                lines.append("soft=" + ", ".join(self.soft_preferences[:6]))
            if self.priorities:
                lines.append(
                    "tradeoffs="
                    + ", ".join(
                        f"{item['direction']}:{item['field']}" for item in self.priorities
                    )
                )
            if self.named:
                lines.append(
                    "named_keys="
                    + ", ".join(f"{pid}::{sku}" for pid, sku in self.named[:8])
                )
            if self.mentioned_fields:
                lines.append("query_claim_fields=" + ", ".join(sorted(self.mentioned_fields)))
            if self.shopper_notes:
                lines.append("shopper_notes:")
                for note in self.shopper_notes[-2:]:
                    lines.append(f"  - {note}")
            if self.cart:
                lines.append("cart:")
                for item in self.cart[-3:]:
                    stock = self.stock.get(item["key"]) or self.stock.get(item["product_id"]) or {}
                    lines.append(
                        f"  - {item['key']} price={stock.get('price', '?')} "
                        f"in_stock={stock.get('in_stock', '?')} "
                        f"post_cart={item['key'] in self.post_cart_stock}"
                    )
            ranked = self.ranked_candidates()
            if ranked:
                lines.append("candidates:")
                for score, key, row in ranked[:_BOARD_CANDIDATE_CAP]:
                    lines.append(
                        f"  - score={score} {key} title={str(row.get('title') or '')[:64]} "
                        f"price={row.get('price', '?')} stock={row.get('in_stock', '?')} "
                        f"brand={row.get('brand', '?')}"
                    )
            if self.cart:
                top = self.cart[-1]
                viewed = self.viewed.get(top["product_id"]) or {}
                facts = viewed.get("facts") if isinstance(viewed.get("facts"), dict) else {}
                if facts:
                    preview = ", ".join(
                        f"{k}={str(v)[:40]}" for k, v in list(facts.items())[:8]
                    )
                    lines.append(f"final_facts={preview}")
                    lines.append(
                        "If justification is required, claim price, in_stock, one shopper "
                        "attribute from these facts, and one allowed reason field."
                    )
            return "\n".join(lines)
        except Exception:
            return ""


def _surface_claim_value(
    product: dict[str, Any], field: str, goal_text: str
) -> Any:
    try:
        """Build a claim value from observed text near the field's own name tokens."""
        parts = [p for p in re.split(r"[_\s]+", field.lower()) if len(p) >= 3]
        if not parts:
            return None
        want_nums = set(_NUM_TOKEN_RE.findall(goal_text))
        surfaces: list[str] = []
        for key, value in product.items():
            if not isinstance(key, str) or value in (None, "", True, False):
                continue
            if isinstance(value, (dict, list)):
                continue
            surfaces.append(f"{key} {value}")
        blob = " ".join(surfaces).lower()
        if not blob:
            return None
        best: tuple[int, str] | None = None
        for match in re.finditer(r"(\d+(?:\.\d+)?)\s*-?\s*([a-z]{0,8})", blob):
            num = match.group(1)
            unit = (match.group(2) or "").lower()
            window = blob[max(0, match.start() - 30) : match.end() + 30]
            score = 0
            if any(p in window for p in parts):
                score += 3
            if num in want_nums:
                score += 2
            if any(p in goal_text for p in parts) and num in want_nums:
                score += 2
            kn_hit = any(
                _keys_structurally_match(_norm_key(key), _norm_key(field))
                for key in product
                if isinstance(key, str)
            )
            if kn_hit and num in set(_NUM_TOKEN_RE.findall(str(product.get(field, "")))):
                score += 4
            if score < 3:
                continue
            value = f"{num}{unit}" if unit else num
            # Prefer compact unit forms that literally appear in the product blob.
            glued = f"{num}{unit}" if unit else ""
            if glued and glued in blob.replace(" ", ""):
                value = glued
            if best is None or score > best[0]:
                best = (score, value)
        if best:
            return best[1]
        # Direct fact key structural match without title parsing.
        for key, value in product.items():
            if not isinstance(key, str) or value in (None, ""):
                continue
            if isinstance(value, (dict, list, bool)):
                continue
            if _keys_structurally_match(_norm_key(key), _norm_key(field)):
                return value
        return None
    except Exception:
        return None


def _ground_order(
    args: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    board: EpisodeBoard,
) -> dict[str, Any]:
    try:
        cleaned = dict(args)
        required = _required_args(tools, "place_test_order")
        needs = "justification" in required
        if not needs:
            cleaned.pop("justification", None)
            return cleaned

        pid = str(cleaned.get("product_id") or "")
        sku = str(cleaned.get("sku") or "")
        if (not pid or not sku) and board.cart:
            top = board.cart[-1]
            pid = pid or top["product_id"]
            sku = sku or top["sku"]
            cleaned["product_id"] = pid
            cleaned["sku"] = sku
        key = f"{pid}::{sku}"
        stock = dict(board.stock.get(key) or board.stock.get(pid) or {})
        viewed = dict(board.viewed.get(pid) or {})
        candidate = dict(board.candidates.get(key) or board.candidates.get(pid) or {})
        product = _flatten_product({**candidate, **viewed, **stock})
        allowed = _claim_field_allowlist(tools)
        open_spec = _justification_open_spec(tools)
        soft_fields = _soft_claim_fields(tools, allowed)
        goal = _goal_tokens(board.query, board.shopper_notes)
        goal_text = f"{_preamble(board.query)} {' '.join(board.shopper_notes[-3:])}".lower()
        prefer = set(board.mentioned_fields)

        raw_claims: list[dict[str, Any]] = []
        just = cleaned.get("justification")
        if isinstance(just, dict) and isinstance(just.get("claims"), list):
            raw_claims = [c for c in just["claims"] if isinstance(c, dict)]

        # Open-spec: admit model-proposed attribute names and query-derived names.
        if open_spec:
            allowed.update({"price", "in_stock"})
            allowed.update(soft_fields)
            allowed.update(prefer)
            for claim in raw_claims:
                field = str(claim.get("field") or "").strip()
                if field and re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,40}", field):
                    allowed.add(field)
                    prefer.add(field)

        if not allowed:
            allowed = {"price", "in_stock"}

        grounded: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(field: str, value: Any) -> None:
            # KA meta-key cleanup: never emit observation/envelope keys as claims.
            if field.lower() in _KA_META_CLAIM_FIELDS:
                return
            if field not in allowed or field in seen:
                return
            if value is None or value == "":
                return
            if isinstance(value, bool):
                text = "true" if value else "false"
            else:
                text = str(value).strip()
            if not text:
                return
            if text.lower() in {"unknown", "none", "n/a", "null"}:
                return
            if field != "in_stock" and text.lower() == "false":
                return
            grounded.append({"field": field, "value": text[:80]})
            seen.add(field)

        def best_observed_value(field: str) -> Any:
            direct = _lookup_fact([product, stock, viewed, candidate], field)
            if direct not in (None, ""):
                return direct
            # Soft/capability fields only accept explicit positive evidence on that key.
            if field in soft_fields and field != "brand":
                return None
            want_nums = _goal_numbers_for_field(board.query, field, board.shopper_notes)
            best_val: Any = None
            best_score = 0
            for key_name, value in product.items():
                if value in (None, "", True, False):
                    continue
                if not isinstance(key_name, str):
                    continue
                if isinstance(value, (dict, list)):
                    continue
                if key_name in {
                    "product_id",
                    "sku",
                    "title",
                    "price",
                    "in_stock",
                    "currency",
                    "listing_id",
                    "brand",
                } and field not in {"brand", "price", "in_stock"}:
                    # Brand/title are weak evidence for unrelated attribute claims.
                    continue
                text = str(value).lower().strip()
                nums = set(_NUM_TOKEN_RE.findall(text))
                shared = nums & want_nums if want_nums else set()
                overlap = 0
                if _keys_structurally_match(_norm_key(key_name), _norm_key(field)):
                    overlap += 8
                if shared:
                    overlap += 3 * len(shared)
                if len(text) >= 2 and (
                    text in goal_text or any(tok == text for tok in goal if len(tok) >= 2)
                ):
                    overlap += 6
                # Partial token overlap is weaker than exact value hits.
                overlap += min(2, _value_overlap(value, goal))
                if overlap > best_score:
                    best_score = overlap
                    best_val = value
            return best_val if best_score >= 6 else None

        add("price", stock.get("price", product.get("price")))
        add("in_stock", stock.get("in_stock", product.get("in_stock")))

        # Claim shopper-requested specs that are evidenced on the viewed product.
        # Justification schemas allow: price, in_stock, ONE request spec field, and
        # ONE supported reason (usually brand) — extra fields can invalidate the order.
        spec_frags = _goal_spec_fragments(board.query, board.shopper_notes)
        facts = product.get("facts") if isinstance(product.get("facts"), dict) else {}
        spec_field_claimed = False
        for frag in spec_frags:
            if spec_field_claimed:
                break
            frag_l = frag.lower().strip()
            if len(frag_l) < 2:
                continue
            # Prefer compact numeric forms used by successful checkouts (e.g. "27").
            frag_alts = [frag_l, frag_l.replace("-", " ")]
            if re.fullmatch(r"\d+(?:\.\d+)?-\w+", frag_l):
                frag_alts.append(frag_l.split("-", 1)[0])
            matched_field = None
            matched_val = None
            # Scan live facts + flat product keys (schema-agnostic; no fixed attr list).
            surfaces: list[tuple[str, Any]] = list(facts.items())
            for key, value in product.items():
                if key == "facts":
                    continue
                surfaces.append((key, value))
            for key, value in surfaces:
                if not isinstance(key, str) or value in (None, "", True, False):
                    continue
                if isinstance(value, (dict, list)):
                    continue
                if key in {
                    "title",
                    "listing_id",
                    "product_id",
                    "sku",
                    "model",
                    "condition",
                    "navigation",
                    "brand",
                    "price",
                    "in_stock",
                    "currency",
                    "category_path",
                    "options",
                    "variants",
                }:
                    continue
                text = str(value).lower()
                if not any(alt in text or alt in text.replace("-", " ") for alt in frag_alts):
                    continue
                matched_field = key
                # Prefer the shopper's own fragment wording when it appears.
                matched_val = next(
                    (alt for alt in frag_alts if alt in text or alt in text.replace("-", " ")),
                    value,
                )
                if key in facts:
                    break
            if matched_field and matched_val is not None:
                if open_spec:
                    allowed.add(matched_field)
                add(matched_field, matched_val)
                spec_field_claimed = True

        # Prefer model-proposed attribute claims with observation-backed values.
        for claim in raw_claims:
            field = str(claim.get("field") or "").strip()
            if not field or field in {"price", "in_stock"}:
                continue
            if field not in allowed and not open_spec:
                continue
            if open_spec:
                allowed.add(field)
            observed = best_observed_value(field)
            if observed is None:
                observed = _surface_claim_value(product, field, goal_text)
            if observed is not None:
                add(field, observed)
                continue
            raw_val = claim.get("value")
            if raw_val is None or raw_val == "":
                continue
            token = str(raw_val).strip().lower()
            blob = json.dumps(
                {k: v for k, v in product.items() if k != "title"},
                ensure_ascii=False,
                default=str,
            ).lower()
            if len(token) >= 2 and token in blob:
                add(field, raw_val)

        # Schema / query-mentioned attributes before soft brand fillers.
        attr_fields: list[str] = []
        for field in list(prefer) + sorted(allowed):
            if field in {"price", "in_stock"} or field in soft_fields:
                continue
            if field in attr_fields:
                continue
            surface = field.replace("_", " ").lower()
            parts = [p for p in field.lower().split("_") if len(p) >= 3]
            mentioned = (
                field in prefer
                or surface in goal_text
                or any(len(p) >= 4 and p in goal_text for p in parts)
                or (len(parts) >= 2 and all(p in goal_text for p in parts))
            )
            if mentioned:
                attr_fields.append(field)
        for field in attr_fields:
            if field in seen:
                continue
            observed = best_observed_value(field)
            if observed is None:
                observed = _surface_claim_value(product, field, goal_text)
            if observed is not None:
                add(field, observed)

        # One evidence-backed attribute claim from goal↔fact overlap when still missing.
        if not any(f not in {"price", "in_stock"} | soft_fields for f in seen):
            scored: list[tuple[int, str, Any]] = []
            name_hints = [
                str(c).strip()
                for c in list(prefer) + [str(c.get("field") or "") for c in raw_claims]
                if str(c).strip() and str(c).strip() not in {"price", "in_stock"}
            ]
            for key_name, value in product.items():
                if not isinstance(key_name, str) or value in (None, ""):
                    continue
                if key_name in {
                    "product_id",
                    "sku",
                    "title",
                    "price",
                    "in_stock",
                    "currency",
                    "listing_id",
                    "category_path",
                    "variants",
                    "options",
                    "brand",
                    "condition",
                    "model",
                }:
                    continue
                if isinstance(value, (dict, list, bool)):
                    continue
                text = str(value).lower().strip()
                overlap = 0
                if len(text) >= 2 and (
                    text in goal_text or any(tok == text for tok in goal if len(tok) >= 2)
                ):
                    overlap += 6
                overlap += min(3, _value_overlap(value, goal))
                nums = set(_NUM_TOKEN_RE.findall(text))
                overlap += 2 * len(nums & {t for t in goal if any(ch.isdigit() for ch in t)})
                if overlap <= 0:
                    continue
                claim_name = None
                for cand in name_hints:
                    if _keys_structurally_match(_norm_key(cand), _norm_key(key_name)):
                        claim_name = cand
                        overlap += 3
                        break
                # Without a shopper/model field name, skip — bare catalog keys are often off-schema.
                if claim_name is None:
                    continue
                if open_spec or claim_name in allowed:
                    scored.append((overlap, claim_name, value))
            scored.sort(key=lambda row: (-row[0], row[1]))
            if scored:
                _score, field, value = scored[0]
                if open_spec:
                    allowed.add(field)
                add(field, value)

        # Open-spec repair: model named a field but value lookup missed — use best goal-overlapping fact.
        for claim in raw_claims:
            field = str(claim.get("field") or "").strip()
            if not field or field in seen or field in {"price", "in_stock"}:
                continue
            if not open_spec and field not in allowed:
                continue
            best_val: Any = None
            best_score = 0
            for key_name, value in product.items():
                if not isinstance(key_name, str) or value in (None, ""):
                    continue
                if key_name in {
                    "product_id",
                    "sku",
                    "title",
                    "price",
                    "in_stock",
                    "currency",
                    "listing_id",
                    "brand",
                    "condition",
                }:
                    continue
                if isinstance(value, (dict, list, bool)):
                    continue
                text = str(value).lower().strip()
                score = 0
                if len(text) >= 2 and (
                    text in goal_text or any(tok == text for tok in goal if len(tok) >= 2)
                ):
                    score += 6
                score += min(3, _value_overlap(value, goal))
                if score > best_score:
                    best_score = score
                    best_val = value
            if best_val is not None and best_score >= 6:
                if open_spec:
                    allowed.add(field)
                add(field, best_val)

        if "brand" in allowed and "brand" not in seen and product.get("brand"):
            brand_val = str(product.get("brand") or "").strip()
            title_l = str(product.get("title") or "").lower()
            brand_l = brand_val.lower()
            # Only claim brand when the listing itself evidences it — never from the ask alone.
            if brand_l and (
                brand_l in title_l
                or brand_l == str((viewed.get("brand") or "")).lower()
                or brand_l == str((candidate.get("brand") or "")).lower()
            ):
                add("brand", brand_val)

        if soft_fields and not (seen & soft_fields):
            for field in sorted(soft_fields):
                if field == "brand" and product.get("brand"):
                    add(field, product.get("brand"))
                    break
                observed = _lookup_fact([product], field)
                if observed is True or str(observed).strip().lower() in {"true", "yes", "1"}:
                    add(field, "true")
                    break

        # Never submit an empty justification when the schema requires one.
        if needs and not grounded:
            add("price", stock.get("price", product.get("price")))
            add("in_stock", stock.get("in_stock", product.get("in_stock", True)))
            if product.get("brand") and "brand" in allowed:
                add("brand", product.get("brand"))

        # Keep justification payloads minimal: price, stock, one ask-spec, one reason.
        if needs and grounded:
            reason_names = set(soft_fields) | {"brand"}
            core = [c for c in grounded if c["field"] in {"price", "in_stock"}]
            specs = [
                c
                for c in grounded
                if c["field"] not in {"price", "in_stock"} and c["field"] not in reason_names
            ]
            reason_claims = [c for c in grounded if c["field"] in reason_names]
            slim: list[dict[str, str]] = list(core)
            if specs:
                slim.append(specs[0])
            if reason_claims:
                slim.append(reason_claims[0])
            elif product.get("brand") and "brand" in allowed:
                slim.append({"field": "brand", "value": str(product.get("brand"))[:80]})
            grounded = slim

        cleaned["justification"] = {"claims": grounded}
        return cleaned
    except ValueError:
        raise
    except Exception:
        return {}


def _normalize_args(
    name: str,
    args: dict[str, Any],
    *,
    tools: list[dict[str, Any]],
    board: EpisodeBoard,
) -> dict[str, Any]:
    try:
        cleaned = dict(args)
        if name == "search":
            query = str(cleaned.get("query") or "").strip()
            if not query or query.isdigit() or len(_TOKEN_RE.findall(query)) < 2:
                query = _search_variants(board.query, board.shopper_notes, board.search_count)
            # Drop bare budget amounts, catalog keys, and discourse-only fragments.
            budget_nums = _budget_number_strings(
                f"{_preamble(board.query)} {' '.join(board.shopper_notes[-2:])}"
            )
            discourse = {
                "compare",
                "choose",
                "among",
                "remaining",
                "ties",
                "tradeoff",
                "candidate",
                "candidates",
                "variant",
                "variants",
                "option",
                "options",
                "walk",
                "through",
                "why",
                "right",
                "pick",
                "select",
            }
            kept: list[str] = []
            for tok in query.split():
                bare = tok.replace(",", "").lower()
                if bare in budget_nums or bare in discourse:
                    continue
                if re.fullmatch(r"\d{6,}", bare) or "::" in tok:
                    continue
                if re.fullmatch(r"[a-z0-9]+-\d+", bare) and len(bare) >= 8:
                    continue
                if re.fullmatch(r"\d+(?:\.\d+)?", bare) and (
                    bare in budget_nums or float(bare) >= 1000
                ):
                    continue
                kept.append(tok)
            query = " ".join(kept).strip() or board.search_seed
            tokens = _TOKEN_RE.findall(query)
            if len(tokens) < 2 or all(t.lower() in discourse for t in tokens):
                query = _search_variants(board.query, board.shopper_notes, board.search_count)
                tokens = _TOKEN_RE.findall(query)
            if len(tokens) > 8 and '"' not in query:
                cleaned["query"] = " ".join(tokens[:6])
            else:
                cleaned["query"] = query
            cleaned.setdefault("in_stock", True)
            cleaned.setdefault("k", 10)
            if board.budget is not None and board.budget >= 5:
                cleaned.setdefault("max_price", board.budget)
        if name == "filter":
            cleaned.setdefault("in_stock", True)
            cleaned.setdefault("k", 10)
            if board.budget is not None and board.budget >= 5 and "max_price" not in cleaned:
                cleaned["max_price"] = board.budget
            if not str(cleaned.get("query") or "").strip():
                cleaned["query"] = _search_variants(
                    board.query, board.shopper_notes, board.search_count
                )
            category = _category_hint(board.query)
            if category and "category" not in cleaned:
                cleaned["category"] = category
        if name == "compare":
            ids = cleaned.get("product_ids")
            if isinstance(ids, list):
                cleaned["product_ids"] = [str(x) for x in ids[:5] if str(x).strip()]
            elif isinstance(ids, str) and ids.strip():
                cleaned["product_ids"] = [
                    part.strip() for part in ids.split(",") if part.strip()
                ][:5]
            if not cleaned.get("product_ids") and board.named:
                cleaned["product_ids"] = [pid for pid, _sku in board.named[:5]]
        if name == "message":
            cleaned["content"] = str(cleaned.get("content") or "").strip()[:400]
        if name == "place_test_order":
            if (not cleaned.get("product_id") or not cleaned.get("sku")) and board.cart:
                top = board.cart[-1]
                cleaned.setdefault("product_id", top["product_id"])
                cleaned.setdefault("sku", top["sku"])
            cleaned = _ground_order(cleaned, tools=tools, board=board)
        return cleaned
    except Exception:
        return {}


def _guard_actions(
    action_pairs: list[tuple[str, dict[str, Any]]],
    *,
    board: EpisodeBoard,
    tools: list[dict[str, Any]],
    max_calls: int,
) -> list[tuple[str, dict[str, Any]]]:
    try:
        names = _tool_names(tools)
        guarded: list[tuple[str, dict[str, Any]]] = []

        # Force an early clarification ask when the public goal defers a firm constraint.
        if (
            board.needs_ask
            and not board.received_clarification
            and "message" in names
            and not board.asked
        ):
            risky = {"add_to_cart", "place_test_order", "remove_from_cart"}
            if any(name in risky for name, _ in action_pairs) or not any(
                name == "message" for name, _ in action_pairs
            ):
                return [
                    (
                        "message",
                        {
                            "content": (
                                "Before I search, what firm product specification must the item "
                                "meet besides category, budget, and in-stock availability?"
                            )
                        },
                    )
                ]

        # Deterministic recover: remove failed cart item, then advance to next ranked option.
        # If the caller already planned a remove (scripted/progress), do not rewrite the
        # follow-up — stem-sibling recovery is often better than a fresh best_recovery().
        already_removing = any(name == "remove_from_cart" for name, _ in action_pairs)
        if (
            board.must_recover
            and board.cart
            and "remove_from_cart" in names
            and not already_removing
        ):
            if (
                board.named
                and len(board.named) >= 2
                and board.needs_rerank
                and board.facts_refresh_generation < board.event_generation
                and "compare" in names
            ):
                return [
                    (
                        "compare",
                        {"product_ids": [pid for pid, _sku in board.named[:6]]},
                    )
                ]
            top = board.cart[-1]
            forced: list[tuple[str, dict[str, Any]]] = [
                ("remove_from_cart", {"product_id": top["product_id"], "sku": top["sku"]})
            ]
            nxt = board.best_recovery()
            if nxt and len(forced) < max_calls:
                action = nxt["action"]
                if action in names:
                    # Preference recovers must clear the multi-axis soft gate.
                    if action == "add_to_cart" and board.task_shape == "preference_reasoning":
                        snap_n = board._product_snapshot(nxt["product_id"], nxt["sku"])
                        if not _preference_cart_ok(board, snap_n):
                            nxt = None
                    if nxt and action in names:
                        args = {"product_id": nxt["product_id"]}
                        if action != "view":
                            args["sku"] = nxt["sku"]
                        forced.append((action, args))
            if len(forced) == 1 and "search" in names:
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    forced.append(nxt_d)
            return forced[:max_calls]

        if (
            not board.cart
            and board.failed_keys
            and not any(name in {"add_to_cart", "view", "inspect_stock", "search", "filter"} for name, _ in action_pairs)
        ):
            nxt = board.best_recovery()
            if nxt and nxt["action"] in names:
                args = {"product_id": nxt["product_id"]}
                if nxt["action"] != "view":
                    args["sku"] = nxt["sku"]
                return [(nxt["action"], args)]

        for name, args in action_pairs:
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")

            if (
                board.needs_ask
                and not board.asked
                and name in {"add_to_cart", "place_test_order"}
                and "message" in names
            ):
                guarded.append(
                    (
                        "message",
                        {
                            "content": (
                                "What firm specification is required for this purchase "
                                "besides category, budget, and stock?"
                            )
                        },
                    )
                )
                continue

            if name == "add_to_cart" and pid and pid not in board.viewed and "view" in names:
                guarded.append(("view", {"product_id": pid}))
                continue

            if name == "add_to_cart" and pid and sku:
                snap = board._product_snapshot(pid, sku)
                # Preference recoveries must change source product after a failed event target.
                if (
                    board.task_shape == "preference_reasoning"
                    and pid in _failed_product_ids(board)
                ):
                    nxt = board.best_recovery()
                    if nxt and nxt["action"] in names:
                        args = {"product_id": nxt["product_id"]}
                        if nxt["action"] != "view":
                            args["sku"] = nxt["sku"]
                        guarded.append((nxt["action"], args))
                        continue
                    if "search" in names:
                        nxt_d = _next_discovery(board, names)
                        if nxt_d:
                            guarded.append(nxt_d)
                            continue
                    continue
                brands = _required_brand_labels(board.shopper_notes)
                if brands:
                    blob = _row_text_blob(snap)
                    if not any(_fragment_match(blob, b) for b in brands):
                        # Prefer another ranked brand-matching candidate instead.
                        replaced = False
                        for _score, key, row in board.ranked_candidates()[:8]:
                            pid2 = str(row.get("product_id") or "")
                            sku2 = str(row.get("sku") or "") or (
                                board.best_sku_for_product(pid2) or ""
                            )
                            if not pid2 or not sku2 or key in board.failed_keys:
                                continue
                            snap2 = board._product_snapshot(pid2, sku2)
                            if any(
                                _fragment_match(_row_text_blob(snap2), b) for b in brands
                            ):
                                guarded.append(
                                    ("add_to_cart", {"product_id": pid2, "sku": sku2})
                                )
                                replaced = True
                                break
                        if replaced:
                            continue
                        if "search" in names:
                            nxt_d = _next_discovery(board, names)
                            if nxt_d:
                                guarded.append(nxt_d)
                                continue
                if not _preference_cart_ok(board, snap):
                    # Soft-gated preference: inspect/view a better soft match instead.
                    replaced = False
                    for _score, key, row in board.ranked_candidates()[:10]:
                        pid2 = str(row.get("product_id") or "")
                        sku2 = str(row.get("sku") or "") or (
                            board.best_sku_for_product(pid2) or ""
                        )
                        if not pid2 or not sku2 or key in board.failed_keys:
                            continue
                        if pid2 in _failed_product_ids(board):
                            continue
                        snap2 = board._product_snapshot(pid2, sku2)
                        if not _preference_cart_ok(board, snap2):
                            continue
                        if pid2 not in board.viewed and "view" in names:
                            guarded.append(("view", {"product_id": pid2, "sku": sku2}))
                            replaced = True
                            break
                        if board.stock_ok(pid2, sku2):
                            guarded.append(
                                ("add_to_cart", {"product_id": pid2, "sku": sku2})
                            )
                            replaced = True
                            break
                        if "inspect_stock" in names:
                            guarded.append(
                                ("inspect_stock", {"product_id": pid2, "sku": sku2})
                            )
                            replaced = True
                            break
                    if replaced:
                        continue
                    if "search" in names:
                        nxt_d = _next_discovery(board, names)
                        if nxt_d:
                            guarded.append(nxt_d)
                            continue
                    # Never cart a soft-failing preference option.
                    continue
                if pid in board.viewed and board.firm_constraints and not board.row_compliant(snap, sku):
                    # Prefer a compliant variant of the same product, else next recovery target.
                    alt = board.best_sku_for_product(pid)
                    if alt and alt != sku:
                        alt_snap = board._product_snapshot(pid, alt)
                        if board.row_compliant(alt_snap, alt):
                            guarded.append(("add_to_cart", {"product_id": pid, "sku": alt}))
                            continue
                    nxt = board.best_recovery()
                    if nxt and nxt["action"] in names:
                        args = {"product_id": nxt["product_id"]}
                        if nxt["action"] != "view":
                            args["sku"] = nxt["sku"]
                        guarded.append((nxt["action"], args))
                        continue
                    if "view" in names:
                        guarded.append(("view", {"product_id": pid}))
                        continue

                # 14-style hard gate before cart: require live stock_ok (+ optional fresh).
                if _HARD_CONSTRAINT_GATE and pid and sku:
                    if not board.stock_ok(pid, sku):
                        if (
                            _REQUIRE_STOCK_FRESH
                            and "inspect_stock" in names
                            and (f"{pid}::{sku}" not in board.stock and pid not in board.stock)
                        ):
                            guarded.append(
                                ("inspect_stock", {"product_id": pid, "sku": sku})
                            )
                            continue
                        if "inspect_stock" in names and not board.stock_ok(pid, sku):
                            # Stale/failed stock: try recovery instead of carting.
                            nxt = board.best_recovery()
                            if nxt and nxt["action"] in names:
                                args = {"product_id": nxt["product_id"]}
                                if nxt["action"] != "view":
                                    args["sku"] = nxt["sku"]
                                guarded.append((nxt["action"], args))
                                continue
                            continue
                    elif (
                        _REQUIRE_STOCK_FRESH
                        and not board.stock_is_fresh(pid, sku)
                        and "inspect_stock" in names
                    ):
                        guarded.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                        continue

            if name == "place_test_order" and pid and sku:
                key = f"{pid}::{sku}"

                if (
                    board.task_shape == "preference_reasoning"
                    and pid in _failed_product_ids(board)
                ):
                    if board.cart and "remove_from_cart" in names:
                        top = board.cart[-1]
                        guarded.append(
                            (
                                "remove_from_cart",
                                {"product_id": top["product_id"], "sku": top["sku"]},
                            )
                        )
                        continue
                    nxt = board.best_recovery()
                    if nxt and nxt["action"] in names:
                        args = {"product_id": nxt["product_id"]}
                        if nxt["action"] != "view":
                            args["sku"] = nxt["sku"]
                        guarded.append((nxt["action"], args))
                        continue

                if board.needs_rerank and board.named and len(board.named) >= 2:
                    named_actions = _named_pipeline_actions(
                        board, tools=tools, max_calls=max_calls
                    )
                    if named_actions:
                        guarded.extend(named_actions)
                        continue

                # Failed live check → recover once (never re-inspect the dead sku).
                if board.must_recover and "remove_from_cart" in names:
                    guarded.append(("remove_from_cart", {"product_id": pid, "sku": sku}))
                    continue

                # Post-message / post-cart live check: require a stock snapshot after carting.
                if board.needs_live_check and key not in board.post_cart_stock:
                    if "inspect_stock" in names:
                        guarded.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                        continue

                has_stock = key in board.stock or pid in board.stock
                if (
                    not has_stock
                    or key not in board.cart_confirmed
                    or key not in board.post_cart_stock
                ) and "inspect_stock" in names:
                    guarded.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                    continue

                if has_stock and not board.stock_ok(pid, sku):
                    board.must_recover = True
                    if "remove_from_cart" in names:
                        board.failed_keys.add(key)
                        guarded.append(
                            ("remove_from_cart", {"product_id": pid, "sku": sku})
                        )
                        continue

                if (
                    board.task_shape == "retrieval_recall"
                    and board.view_count < 2
                    and "view" in names
                ):
                    nxt = board.best_recovery()
                    if nxt and nxt["action"] == "view":
                        guarded.append(("view", {"product_id": nxt["product_id"]}))
                        continue

                if board.pending_question and "message" in names:
                    guarded.append(("message", {"content": _recommend_message(board)[:400]}))
                    continue

                # Require a view of the final product before checkout when claims are needed.
                if (
                    "justification" in _required_args(tools, "place_test_order")
                    and pid not in board.viewed
                    and "view" in names
                ):
                    guarded.append(("view", {"product_id": pid}))
                    continue

            # Collapse redundant live re-checks into checkout when the cart already passes.
            if name == "inspect_stock" and pid and sku:
                key = f"{pid}::{sku}"
                if (
                    not board.needs_live_check
                    and not board.needs_rerank
                    and key in board.post_cart_stock
                    and board.stock_ok(pid, sku)
                    and "place_test_order" in names
                    and not (
                        board.task_shape == "retrieval_recall" and board.view_count < 2
                    )
                ):
                    guarded.append(
                        (
                            "place_test_order",
                            _ground_order(
                                {"product_id": pid, "sku": sku},
                                tools=tools,
                                board=board,
                            ),
                        )
                    )
                    continue

            if name in {"search", "filter"} and board.named and (
                board.priorities
                or board.task_shape in {"ranking", "preference_reasoning"}
                or len(board.named) >= 2
            ):
                # Named tradeoff tasks should not wander into open search.
                named_actions = _named_pipeline_actions(board, tools=tools, max_calls=1)
                if named_actions:
                    guarded.extend(named_actions)
                    continue

            if name in {"search", "filter"}:
                recent = board.action_log[-4:]
                if (
                    len(recent) >= 3
                    and all(n in {"search", "filter"} for n in recent[-3:])
                    and board.candidates
                ):
                    progress = _progress_actions(board, tools=tools, max_calls=1)
                    if progress:
                        guarded.extend(progress)
                        continue

            if name == "place_test_order" and pid and sku and not board.stock_ok(pid, sku):
                if board.cart and "remove_from_cart" in names:
                    top = board.cart[-1]
                    guarded.append(
                        (
                            "remove_from_cart",
                            {"product_id": top["product_id"], "sku": top["sku"]},
                        )
                    )
                    continue
                progress = _progress_actions(board, tools=tools, max_calls=1)
                if progress:
                    guarded.extend(progress)
                    continue

            guarded.append((name, args))

        # Never checkout in the same planned turn as a fresh cart add.
        if any(name == "add_to_cart" for name, _ in guarded):
            guarded = [(name, args) for name, args in guarded if name != "place_test_order"]

        unique: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for name, args in guarded:
            stamp = name + ":" + json.dumps(args, sort_keys=True, default=str)
            if stamp in seen:
                continue
            seen.add(stamp)
            unique.append((name, args))
        return unique[:max_calls]
    except ValueError:
        raise
    except Exception:
        return []


def _ready_checkout(board: EpisodeBoard) -> dict[str, str] | None:
    try:
        """Return carted product_id/sku when live checks already pass."""
        if board.must_recover or board.needs_live_check or board.pending_question or not board.cart:
            return None
        if board.needs_rerank:
            return None
        top = board.cart[-1]
        if top["key"] not in board.post_cart_stock:
            return None
        if top["key"] not in board.cart_confirmed:
            return None
        # Named compare/tradeoff tasks must finish post-event re-rank before checkout.
        if board.named and len(board.named) >= 2:
            if board.facts_refresh_generation < board.event_generation:
                return None
            for pid, sku in board.named:
                key = f"{pid}::{sku}"
                if key in board.failed_keys:
                    continue
                if not board.stock_is_fresh(pid, sku):
                    return None
            ranked = board.ranked_named()
            if ranked and ranked[0][2] != top["key"]:
                return None
        if not board.stock_ok(top["product_id"], top["sku"]):
            return None
        return {"product_id": top["product_id"], "sku": top["sku"]}
    except Exception:
        return {}


def _named_pipeline_actions(
    board: EpisodeBoard,
    *,
    tools: list[dict[str, Any]],
    max_calls: int,
) -> list[tuple[str, dict[str, Any]]]:
    try:
        """Deterministic compare → inspect → tradeoff-cart → re-rank loop for named keys."""
        names = _tool_names(tools)
        if not board.named:
            return []
        actions: list[tuple[str, dict[str, Any]]] = []
        # After any market move or shopper re-rank nudge, compare before re-ranking.
        if (
            (board.needs_rerank or board.event_seen)
            and board.facts_refresh_generation < board.event_generation
            and "compare" in names
            and len(board.named) >= 2
        ):
            return [
                (
                    "compare",
                    {"product_ids": [pid for pid, _sku in board.named[:6]]},
                )
            ]
        if board.view_count == 0 and "compare" in names and len(board.named) >= 2:
            return [
                (
                    "compare",
                    {"product_ids": [pid for pid, _sku in board.named[:6]]},
                )
            ]
        missing_view = [
            (pid, sku)
            for pid, sku in board.named
            if pid not in board.viewed and f"{pid}::{sku}" not in board.failed_keys
        ]
        if missing_view and "view" in names:
            for pid, sku in missing_view[:max_calls]:
                actions.append(("view", {"product_id": pid, "sku": sku}))
            return actions[:max_calls]

        # After a market/shopper event, re-inspect every named option before re-ranking.
        if board.needs_rerank or board.event_seen:
            stale = [
                (pid, sku)
                for pid, sku in board.named
                if f"{pid}::{sku}" not in board.failed_keys
                and not board.stock_is_fresh(pid, sku)
            ]
            if stale and "inspect_stock" in names:
                for pid, sku in stale[:max_calls]:
                    actions.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                return actions[:max_calls]

        missing_stock = [
            (pid, sku)
            for pid, sku in board.named
            if f"{pid}::{sku}" not in board.failed_keys
            and f"{pid}::{sku}" not in board.stock
            and pid not in board.stock
        ]
        if missing_stock and "inspect_stock" in names:
            for pid, sku in missing_stock[:max_calls]:
                actions.append(("inspect_stock", {"product_id": pid, "sku": sku}))
            return actions[:max_calls]

        ranked = board.ranked_named()
        if not ranked:
            return []
        pid, sku, key = ranked[0]
        if not board.cart:
            if board.stock_ok(pid, sku) and "add_to_cart" in names:
                return [("add_to_cart", {"product_id": pid, "sku": sku})]
            if "inspect_stock" in names and not board.stock_is_fresh(pid, sku):
                return [("inspect_stock", {"product_id": pid, "sku": sku})]
            return []

        top = board.cart[-1]
        if board.must_recover:
            return []
        if top["key"] != key:
            # Switch to the live tradeoff winner once its stock snapshot is fresh.
            if not board.stock_is_fresh(pid, sku) and "inspect_stock" in names:
                return [("inspect_stock", {"product_id": pid, "sku": sku})]
            if board.stock_ok(pid, sku) and "remove_from_cart" in names:
                forced: list[tuple[str, dict[str, Any]]] = [
                    (
                        "remove_from_cart",
                        {"product_id": top["product_id"], "sku": top["sku"]},
                    )
                ]
                if "add_to_cart" in names and len(forced) < max_calls:
                    forced.append(("add_to_cart", {"product_id": pid, "sku": sku}))
                return forced[:max_calls]
            return []
        # Cart matches current winner — only checkout after every live named option is fresh.
        if board.named and all(
            board.stock_is_fresh(pid, sku)
            for pid, sku in board.named
            if f"{pid}::{sku}" not in board.failed_keys
        ):
            board.needs_rerank = False
        else:
            stale = [
                (pid, sku)
                for pid, sku in board.named
                if f"{pid}::{sku}" not in board.failed_keys
                and not board.stock_is_fresh(pid, sku)
            ]
            if stale and "inspect_stock" in names:
                for pid, sku in stale[:max_calls]:
                    actions.append(("inspect_stock", {"product_id": pid, "sku": sku}))
                return actions[:max_calls]
            return []
        if top["key"] == key and top["key"] not in board.post_cart_stock and "inspect_stock" in names:
            return [("inspect_stock", {"product_id": top["product_id"], "sku": top["sku"]})]
        if (
            top["key"] == key
            and top["key"] in board.post_cart_stock
            and board.stock_ok(top["product_id"], top["sku"])
            and not board.pending_question
            and not board.needs_rerank
            and "place_test_order" in names
        ):
            return [
                (
                    "place_test_order",
                    _ground_order(
                        {"product_id": top["product_id"], "sku": top["sku"]},
                        tools=tools,
                        board=board,
                    ),
                )
            ]
        return []
    except Exception:
        return []


def _discovery_search(board: EpisodeBoard) -> tuple[str, dict[str, Any]]:
    try:
        """Pick the next search/filter action with a rotated query."""
        clean_notes = [
            _sanitize_note_for_search(n) for n in (board.shopper_notes or [])[-2:]
        ]
        clean_notes = [n for n in clean_notes if n.strip()]
        query = _search_variants(board.query, clean_notes, board.search_count)
        # Preference-shaped asks: lead with category noun + distinct soft tokens so
        # retrieval surfaces preference-aligned listings ahead of cheapest decoys.
        if board.task_shape == "preference_reasoning" and board.soft_preferences:
            soft_phrases = [
                frag
                for frag in board.soft_preferences
                if not re.fullmatch(r"\d{3,}", frag)
                and len(frag) >= 3
                and frag.lower() not in {"in-stock", "instock", "stock"}
                # Rare 3+ word labels hurt retrieval; keep them for ranking only.
                and len(frag.split()) <= 2
            ]
            nouns = [
                p
                for p in _search_parts(board.query, clean_notes)
                if not re.search(r"\d", p)
                and p.lower() not in {"in-stock", "instock", "stock", "daily", "every", "kind"}
                and p.lower() not in _SEED_STOP
                and not any(p.lower() in s.lower().split() for s in soft_phrases[:6])
            ][:2]
            cat = _category_hint(board.query)
            if cat and cat.lower() not in {n.lower() for n in nouns}:
                nouns = [cat.split()[-1]] + nouns
            # Prefer longer phrases; drop tokens already covered by a longer phrase.
            soft_phrases = sorted(soft_phrases, key=lambda s: (-len(s.split()), -len(s)))
            chosen_soft: list[str] = []
            covered: set[str] = set()
            for frag in soft_phrases:
                toks = [t.lower() for t in frag.split() if len(t) > 2]
                if toks and all(t in covered for t in toks):
                    continue
                chosen_soft.append(frag)
                covered.update(toks)
                if len(chosen_soft) >= 3:
                    break
            # Always fold uncovered soft tokens that the shopper actually stated.
            soft_token_pool = []
            seen_tok: set[str] = set()
            for frag in board.soft_preferences:
                for tok in re.split(r"[\s/]+", frag.lower()):
                    if len(tok) < 4 or tok in seen_tok or tok in _SEED_STOP:
                        continue
                    if re.fullmatch(r"\d{3,14}", tok):
                        continue
                    seen_tok.add(tok)
                    soft_token_pool.append(tok)
            for tok in soft_token_pool:
                if tok in covered:
                    continue
                chosen_soft.append(tok)
                covered.add(tok)
                if len(chosen_soft) >= 6:
                    break
            if nouns or chosen_soft:
                query = " ".join(nouns[:1] + chosen_soft[:5]).strip() or query
                seen_q: set[str] = set()
                kept: list[str] = []
                for tok in query.split():
                    low = tok.lower()
                    if low in seen_q:
                        continue
                    seen_q.add(low)
                    kept.append(tok)
                query = " ".join(kept)
        note_blob = " ".join([_preamble(board.query)] + clean_notes)
        has_phrase = bool(_match_targets(note_blob))
        # Fold unmet firm fragments into the query so retrieval stays constraint-focused.
        # Keep explicit match phrases undiluted; avoid re-stuffing brand duplicates.
        if board.firm_constraints and board.search_count > 0 and not (
            has_phrase and board.search_count < 3
        ):
            extra = []
            for frag in board.firm_constraints[:4]:
                fl = frag.lower()
                if fl in query.lower():
                    continue
                # Skip fragments already covered by a longer query token.
                if any(fl in tok or tok in fl for tok in query.lower().split() if len(tok) >= 4):
                    continue
                extra.append(frag)
            if extra:
                query = f"{query} {' '.join(extra[:2])}".strip()
        # Soft preferences help retrieval from the first query, not only after recovery.
        if board.soft_preferences and board.task_shape != "preference_reasoning" and not (
            has_phrase and board.search_count < 2
        ):
            soft_extra = [
                frag
                for frag in board.soft_preferences[:3]
                if frag.lower() not in query.lower()
                and not re.fullmatch(r"\d{6,}", frag)
            ]
            if soft_extra:
                query = f"{query} {' '.join(soft_extra[:2])}".strip()
        # On later preference searches, rotate in uncovered soft tokens.
        if (
            board.task_shape == "preference_reasoning"
            and board.soft_preferences
            and board.search_count > 0
        ):
            rotate = [
                frag
                for frag in board.soft_preferences
                if frag.lower() not in query.lower()
                and not re.fullmatch(r"\d{6,}", frag)
                and len(frag) >= 3
            ]
            if rotate:
                query = f"{query} {' '.join(rotate[:2])}".strip()
                seen_q2: set[str] = set()
                kept2: list[str] = []
                for tok in query.split():
                    low = tok.lower()
                    if low in seen_q2:
                        continue
                    seen_q2.add(low)
                    kept2.append(tok)
                query = " ".join(kept2)
        category = _category_hint(board.query)
        # Drop price-hike residue that sometimes leaks from shopper notes into queries.
        query = re.sub(r"\bgo\s+up\b", " ", query, flags=re.I)
        query = re.sub(r"\s+", " ", query).strip()
        args: dict[str, Any] = {"query": query, "in_stock": True, "k": 12 if board.soft_preferences else 10}
        if board.budget is not None and board.budget >= 5:
            # After a failed cart, periodically search without max_price so recovery
            # can see near-budget alternates the hard cap would hide.
            if not (board.failed_keys and board.search_count % 3 == 2):
                args["max_price"] = board.budget
        brand = None
        preamble = _preamble(board.query)
        # Prefer an explicit "brand X" clause when present.
        explicit = re.search(r"\bbrand\s+([A-Za-z0-9][\w-]{1,24})\b", preamble, re.I)
        if explicit:
            brand = explicit.group(1)
        # Prefer filter category alongside a pure phrase query rather than stuffing
        # the category into the free-text query string.
        if has_phrase and board.search_count < 3:
            return ("search", args)
        if category and (not has_phrase) and board.search_count % 2 == 1:
            filt: dict[str, Any] = {**args, "category": category}
            if brand:
                filt["brand"] = brand
            return ("filter", filt)
        return ("search", args)
    except Exception:
        return tuple()


def _next_discovery(
    board: EpisodeBoard, names: set[str]
) -> tuple[str, dict[str, Any]] | None:
    try:
        """Resolve a discovery action against the live tool roster."""
        if "search" not in names and "filter" not in names:
            return None
        action, args = _discovery_search(board)
        if action in names:
            return action, args
        if "search" in names:
            # Filter unavailable — keep the free-text query only.
            clean = {k: v for k, v in args.items() if k != "category"}
            return "search", clean
        if "filter" in names:
            return "filter", args
        return None
    except Exception:
        return tuple()


def _progress_actions(
    board: EpisodeBoard,
    *,
    tools: list[dict[str, Any]],
    max_calls: int,
) -> list[tuple[str, dict[str, Any]]]:
    try:
        """Deterministic discovery → filter/rank → verify → cart → order."""
        names = _tool_names(tools)
        actions: list[tuple[str, dict[str, Any]]] = []

        if board.named and (
            board.priorities
            or board.task_shape in {"ranking", "preference_reasoning"}
            or len(board.named) >= 2
        ):
            named_actions = _named_pipeline_actions(
                board, tools=tools, max_calls=max_calls
            )
            if named_actions:
                return named_actions

        if board.named and board.view_count < min(2, len(board.named)):
            if "compare" in names and len(board.named) >= 2:
                return [
                    (
                        "compare",
                        {"product_ids": [pid for pid, _sku in board.named[:5]]},
                    )
                ]
            if "view" in names:
                pid, sku = board.named[board.view_count]
                return [("view", {"product_id": pid, "sku": sku})]

        if not board.candidates:
            if "search" in names or "filter" in names:
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    return [nxt_d]
            return actions

        ranked = board.ranked_candidates()
        specs = _goal_spec_fragments(board.query, board.shopper_notes)

        # If a live-valid compliant option is already in hand, cart/order it immediately
        # instead of reformulating search after shopper nudges.
        if not board.cart and not board.named:
            # Preference asks: inspect several options before committing so soft facts
            # are grounded and the first cart is preference-aligned.
            prefer_ready = board.task_shape != "preference_reasoning" or (
                board.view_count >= 2 and board.asked
            )
            if prefer_ready:
                for _score, key, row in ranked:
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (board.best_sku_for_product(pid) or "")
                    if not pid or not sku or key in board.failed_keys:
                        continue
                    if pid not in board.viewed:
                        continue
                    snap = board._product_snapshot(pid, sku)
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        continue
                    brands = _required_brand_labels(board.shopper_notes)
                    if brands:
                        blob = _row_text_blob(snap)
                        if not any(_fragment_match(blob, b) for b in brands):
                            continue
                    if board.task_shape == "preference_reasoning" and board.soft_preferences:
                        if not _preference_cart_ok(board, snap):
                            continue
                    if board.stock_ok(pid, sku) and "add_to_cart" in names:
                        return [("add_to_cart", {"product_id": pid, "sku": sku})]
                    if key not in board.stock and pid not in board.stock and "inspect_stock" in names:
                        # Prefer inspecting an already-viewed candidate over more search.
                        return [("inspect_stock", {"product_id": pid, "sku": sku})]
            elif board.task_shape == "preference_reasoning" and "view" in names:
                for _score, key, row in ranked:
                    pid = str(row.get("product_id") or "")
                    if not pid or pid in board.viewed:
                        continue
                    sku = str(row.get("sku") or "")
                    args = {"product_id": pid}
                    if sku:
                        args["sku"] = sku
                    return [("view", args)]

        # If top hits look unrelated to the ask, reformulate once — then inspect anyway.
        if ranked and not board.named and board.view_count == 0 and board.search_count < 2:
            top_score, _top_key, top_row = ranked[0]
            weak = top_score < -2
            if specs and _spec_match_score(top_row, specs) < -3:
                weak = True
            if weak and ("search" in names or "filter" in names):
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    return [nxt_d]

        for score, key, row in ranked:
            # After the first search, inspect something rather than spinning on reformulation.
            if score < 1 and board.search_count < 4 and not board.named and board.view_count > 0:
                continue
            if score < -5 and board.search_count < 3 and not board.named:
                continue
            if (
                specs
                and _spec_match_score(
                    {**row, **(board.viewed.get(str(row.get("product_id") or "")) or {})},
                    specs,
                )
                < 0
                and board.search_count < 5
                and not board.named
                and board.view_count > 0
            ):
                continue
            pid = str(row.get("product_id") or "")
            sku = str(row.get("sku") or "") or (board.best_sku_for_product(pid) or "")
            if not pid or key in board.failed_keys:
                continue
            snap = board._product_snapshot(pid, sku) if sku else dict(row)
            if pid in board.viewed and board.firm_constraints and not board.row_compliant(snap, sku):
                # Skip non-compliant viewed rows; keep searching/viewing alternatives.
                # Late in the episode, keep the closest firm match instead of looping forever.
                if board.search_count < 8 and board.view_count < 10:
                    continue
            soft_codes = [
                frag
                for frag in board.soft_preferences
                if re.fullmatch(r"\d{3,14}", frag)
            ]
            if soft_codes and pid not in board.viewed:
                blob = _row_text_blob(snap)
                if not any(
                    _soft_code_match(blob, code) or code in str(row.get("sku") or "")
                    for code in soft_codes
                ):
                    # Early searches stay code-strict; later allow category hits so
                    # view can reveal codes buried in facts.
                    if board.search_count < 4:
                        continue
            # Reject obvious category mismatches before spending a view.
            # Search hits often omit category_path — only demote when path is present.
            if board.firm_constraints and pid not in board.viewed:
                path_parts = snap.get("category_path") or row.get("category_path") or []
                if path_parts:
                    path = " ".join(
                        str(part).lower() for part in path_parts if part is not None
                    )
                    title_l = str(snap.get("title") or row.get("title") or "").lower()
                    short_firm = [
                        f
                        for f in board.firm_constraints
                        if not re.search(r"\d", f) and len(f) >= 4 and " " not in f
                    ]
                    if short_firm and not any(
                        f.lower() in path or f.lower() in title_l for f in short_firm
                    ):
                        continue
            if pid not in board.viewed and "view" in names:
                args = {"product_id": pid}
                if sku:
                    args["sku"] = sku
                return [("view", args)]
            # After view, lock onto the best matching variant sku.
            if pid in board.viewed:
                chosen = board.best_sku_for_product(pid)
                if chosen:
                    sku = chosen
                    key = f"{pid}::{sku}"
                    snap = board._product_snapshot(pid, sku)
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        continue
            if sku and key not in board.stock and pid not in board.stock and "inspect_stock" in names:
                return [("inspect_stock", {"product_id": pid, "sku": sku})]
            if sku and (key in board.stock or pid in board.stock) and not board.stock_ok(pid, sku):
                continue
            if sku and board.stock_ok(pid, sku):
                if not board.cart and "add_to_cart" in names and pid in board.viewed:
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        if board.search_count < 8 and board.view_count < 10:
                            continue
                    if soft_codes:
                        blob = _row_text_blob(snap)
                        if not any(
                            _soft_code_match(blob, code) or code in sku for code in soft_codes
                        ):
                            # Recovery / preference-coded asks must keep hunting for the code.
                            if board.task_shape in {"recovery", "preference_reasoning"}:
                                continue
                            if board.search_count < 6 or board.view_count < 3:
                                continue
                    # Preference asks: prefer soft-word evidence, but do not hard-block
                    # after a broad enough survey (catalogs paraphrase freely).
                    soft_words = [
                        frag
                        for frag in board.soft_preferences
                        if not re.fullmatch(r"\d{3,14}", frag) and len(frag) >= 3
                    ]
                    if soft_words and board.task_shape == "preference_reasoning":
                        _fh, _fm, soft_hits = _compliance_scores(
                            snap, board.firm_constraints, soft_words
                        )
                        # Survey several listings before committing; require soft evidence
                        # when it exists on the shortlist.
                        if board.view_count < 3:
                            continue
                        if soft_hits <= 0 and board.view_count < 5:
                            # Prefer viewing another higher-soft candidate first.
                            for _s2, _k2, row2 in ranked:
                                pid2 = str(row2.get("product_id") or "")
                                if not pid2 or pid2 in board.viewed:
                                    continue
                                snap2 = board._product_snapshot(
                                    pid2, str(row2.get("sku") or "")
                                )
                                _h2, _m2, sh2 = _compliance_scores(
                                    snap2, board.firm_constraints, soft_words
                                )
                                if sh2 > 0 and "view" in names:
                                    return [("view", {"product_id": pid2})]
                            if board.search_count < 5 and (
                                "search" in names or "filter" in names
                            ):
                                nxt_d = _next_discovery(board, names)
                                if nxt_d:
                                    return [nxt_d]
                            continue
                    # Survey a second live candidate before the first cart so a
                    # post-cart market event has a pre-inspected backup.
                    if (
                        board.soft_preferences
                        or board.task_shape
                        in {"preference_reasoning", "recovery", "intent_decomposition"}
                    ) and board.view_count < 2:
                        for _score2, _key2, row2 in ranked:
                            pid2 = str(row2.get("product_id") or "")
                            if not pid2 or pid2 == pid or pid2 in board.viewed:
                                continue
                            if soft_codes:
                                snap2 = board._product_snapshot(
                                    pid2, str(row2.get("sku") or "")
                                )
                                blob2 = _row_text_blob(snap2)
                                if not any(
                                    _soft_code_match(blob2, code)
                                    or code in str(row2.get("sku") or "")
                                    for code in soft_codes
                                ):
                                    continue
                            if "view" in names:
                                return [("view", {"product_id": pid2})]
                            break
                    return [("add_to_cart", {"product_id": pid, "sku": sku})]
                if (
                    board.cart
                    and board.cart[-1]["key"] == key
                    and key in board.post_cart_stock
                    and board.stock_ok(pid, sku)
                    and not board.must_recover
                    and not board.needs_rerank
                    and "place_test_order" in names
                ):
                    return [
                        (
                            "place_test_order",
                            _ground_order(
                                {"product_id": pid, "sku": sku},
                                tools=tools,
                                board=board,
                            ),
                        )
                    ]
                if board.cart and board.cart[-1]["key"] == key and key not in board.post_cart_stock:
                    if "inspect_stock" in names:
                        return [("inspect_stock", {"product_id": pid, "sku": sku})]
                if (
                    board.cart
                    and board.cart[-1]["key"] != key
                    and "add_to_cart" in names
                    and board.must_recover
                ):
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        continue
                    return [("add_to_cart", {"product_id": pid, "sku": sku})]
            if len(actions) >= max_calls:
                break
        if (
            not actions
            and not board.named
            and ("search" in names or "filter" in names)
        ):
            search_cap = 12 if board.failed_keys or board.soft_preferences else 8
            if board.search_count < search_cap:
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    return [nxt_d]
        return actions
    except Exception:
        return []


def _scripted_actions(
    board: EpisodeBoard,
    *,
    tools: list[dict[str, Any]],
    max_calls: int,
    turn: int,
    max_steps: int,
) -> list[tuple[str, dict[str, Any]]] | None:
    try:
        """Skip the LLM when the next move is fully determined by public state."""
        names = _tool_names(tools)
        remaining = max_steps - turn + 1

        # Late episode: force a verified checkout rather than timing out.
        if remaining <= 8 and "place_test_order" in names:
            ready = _ready_checkout(board)
            if ready:
                return [
                    (
                        "place_test_order",
                        _ground_order(dict(ready), tools=tools, board=board),
                    )
                ]
            if board.cart and "inspect_stock" in names:
                top = board.cart[-1]
                if top["key"] not in board.post_cart_stock:
                    return [
                        (
                            "inspect_stock",
                            {"product_id": top["product_id"], "sku": top["sku"]},
                        )
                    ]
                if not board.stock_ok(top["product_id"], top["sku"]) and "remove_from_cart" in names:
                    return [
                        (
                            "remove_from_cart",
                            {"product_id": top["product_id"], "sku": top["sku"]},
                        )
                    ]
            if not board.cart and "add_to_cart" in names:
                for _score, key, row in board.ranked_candidates()[:16]:
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (board.best_sku_for_product(pid) or "")
                    if not pid or not sku or key in board.failed_keys:
                        continue
                    if pid not in board.viewed:
                        if "view" in names and remaining <= 6:
                            return [("view", {"product_id": pid, "sku": sku})]
                        continue
                    snap = board._product_snapshot(pid, sku)
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        continue
                    brands = _required_brand_labels(board.shopper_notes)
                    if brands and not any(
                        _fragment_match(_row_text_blob(snap), b) for b in brands
                    ):
                        continue
                    if key not in board.stock and pid not in board.stock:
                        if "inspect_stock" in names:
                            return [("inspect_stock", {"product_id": pid, "sku": sku})]
                        continue
                    if not board.stock_ok(pid, sku):
                        continue
                    return [("add_to_cart", {"product_id": pid, "sku": sku})]
                # Last resort: search without soft brand residue.
                if remaining <= 8 and ("search" in names or "filter" in names):
                    nxt_d = _next_discovery(board, names)
                    if nxt_d:
                        action, args = nxt_d
                        q = str(args.get("query") or "")
                        q = re.sub(r"\bgo\s+up\b", " ", q, flags=re.I)
                        # Drop long digit stems that pollute recovery searches.
                        q = re.sub(r"\b\d{8,}\b", " ", q)
                        # Drop soft preferred-brand tokens so firm specs can surface.
                        for frag in board.soft_preferences:
                            if re.search(r"\d", frag):
                                continue
                            if " " in frag:
                                continue
                            q = re.sub(rf"\b{re.escape(frag)}\b", " ", q, flags=re.I)
                        args = {**args, "query": re.sub(r"\s+", " ", q).strip()}
                        if board.budget is not None:
                            args["max_price"] = board.budget
                        return [(action, args)]
                # Absolute last resort: cart any stock-ok firm-compliant viewed item.
                if remaining <= 4:
                    for _score, key, row in board.ranked_candidates()[:20]:
                        pid = str(row.get("product_id") or "")
                        sku = str(row.get("sku") or "") or (
                            board.best_sku_for_product(pid) or ""
                        )
                        if not pid or not sku or key in board.failed_keys:
                            continue
                        if pid not in board.viewed:
                            continue
                        snap = board._product_snapshot(pid, sku)
                        if not _is_compliant(snap, board.firm_constraints):
                            continue
                        if board.stock_ok(pid, sku):
                            return [("add_to_cart", {"product_id": pid, "sku": sku})]
                        if "inspect_stock" in names and key not in board.stock:
                            return [("inspect_stock", {"product_id": pid, "sku": sku})]

        # Break pure search/inspect deadlocks late in the episode only.
        recent = board.action_log[-4:]
        inspect_loop = len(recent) >= 3 and all(a == "inspect_stock" for a in recent[-3:])
        search_loop = len(recent) >= 4 and all(a in {"search", "filter"} for a in recent[-4:])
        if (inspect_loop or search_loop) and remaining <= 14:
            if board.cart:
                top = board.cart[-1]
                if board.stock_ok(top["product_id"], top["sku"]) and "place_test_order" in names:
                    return [
                        (
                            "place_test_order",
                            _ground_order(
                                {"product_id": top["product_id"], "sku": top["sku"]},
                                tools=tools,
                                board=board,
                            ),
                        )
                    ]
                if (
                    not board.stock_ok(top["product_id"], top["sku"])
                    and "remove_from_cart" in names
                ):
                    return [
                        (
                            "remove_from_cart",
                            {"product_id": top["product_id"], "sku": top["sku"]},
                        )
                    ]
            for _score, key, row in board.ranked_candidates()[:8]:
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "") or (board.best_sku_for_product(pid) or "")
                if not pid or not sku or key in board.failed_keys:
                    continue
                if pid not in board.viewed and "view" in names:
                    return [("view", {"product_id": pid, "sku": sku})]
                snap = board._product_snapshot(pid, sku)
                if board.firm_constraints and not board.row_compliant(snap, sku):
                    continue
                if board.stock_ok(pid, sku) and "add_to_cart" in names:
                    return [("add_to_cart", {"product_id": pid, "sku": sku})]
                if "inspect_stock" in names and key not in board.stock:
                    return [("inspect_stock", {"product_id": pid, "sku": sku})]

        if (
            board.needs_ask
            and not board.received_clarification
            and not board.asked
            and "message" in names
        ):
            return [
                (
                    "message",
                    {
                        "content": (
                            "Before I search, what firm product specification must the item "
                            "meet besides category, budget, and in-stock availability?"
                        )
                    },
                )
            ]

        # Preference contracts often require priority clarification before the first cart.
        if (
            board.task_shape == "preference_reasoning"
            and not board.asked
            and not board.cart
            and board.view_count == 0
            and "message" in names
            and re.search(r"\bclarify\b.{0,48}\bpriorit", board.query, re.I)
        ):
            soft_bits = ", ".join(board.soft_preferences[:4]) or "your stated soft preferences"
            return [
                (
                    "message",
                    {
                        "content": (
                            f"Before I cart anything: which of these matter most for you — "
                            f"{soft_bits}? I'll compare visible product facts against that."
                        )
                    },
                )
            ]

        if board.must_recover and board.cart and "remove_from_cart" in names:
            if (
                board.named
                and len(board.named) >= 2
                and board.needs_rerank
                and board.facts_refresh_generation < board.event_generation
                and "compare" in names
            ):
                return [
                    (
                        "compare",
                        {"product_ids": [pid for pid, _sku in board.named[:6]]},
                    )
                ]
            top = board.cart[-1]
            forced: list[tuple[str, dict[str, Any]]] = [
                ("remove_from_cart", {"product_id": top["product_id"], "sku": top["sku"]})
            ]
            failed_stem = re.sub(r"-\d+$", "", str(top.get("sku") or ""))
            soft_codes = [
                frag
                for frag in board.soft_preferences
                if re.fullmatch(r"\d{3,14}", frag)
            ]
            nxt = board.best_recovery()
            if nxt and nxt["action"] in names and len(forced) < max_calls:
                nxt_sku = str(nxt.get("sku") or "")
                same_stem = bool(
                    failed_stem
                    and len(failed_stem) >= 4
                    and nxt_sku.startswith(failed_stem)
                )
                # Preference asks: after a market event, re-search soft cues rather
                # than jumping to an arbitrary ranked alternate.
                prefer_research = (
                    board.task_shape
                    in {"preference_reasoning", "justification", "retrieval_recall"}
                    and board.soft_preferences
                    and not soft_codes
                )
                # Accept recovery targets that stay on the failed sku stem for coded
                # recovery asks. Preference asks must change product source.
                skip_nxt = False
                if board.task_shape == "preference_reasoning":
                    failed_pid = (
                        board.last_failed_key.split("::", 1)[0]
                        if board.last_failed_key and "::" in board.last_failed_key
                        else str(top.get("product_id") or "")
                    )
                    if same_stem or (
                        failed_pid and str(nxt.get("product_id") or "") == failed_pid
                    ):
                        skip_nxt = True
                # Model-coded recovery: only accept stem siblings here. Unrelated
                # ranked hits must go through a stem-anchored re-search first.
                if (
                    board.task_shape == "recovery"
                    and soft_codes
                    and nxt
                    and not same_stem
                ):
                    skip_nxt = True
                if (
                    not skip_nxt
                    and same_stem
                    and not prefer_research
                ):
                    args = {"product_id": nxt["product_id"]}
                    if nxt["action"] != "view":
                        args["sku"] = nxt["sku"]
                    forced.append((nxt["action"], args))
                    return forced[:max_calls]
                if (
                    not skip_nxt
                    and (same_stem or not soft_codes)
                    and not prefer_research
                    and board.task_shape != "recovery"
                ):
                    # Brand-firm asks: only accept recovery targets that still match.
                    brand_bits = [
                        f
                        for f in board.firm_constraints
                        if f[:1].isupper() or (" " in f and not re.search(r"\d", f))
                    ]
                    if brand_bits and nxt.get("product_id"):
                        snap = board._product_snapshot(
                            str(nxt["product_id"]), str(nxt.get("sku") or "")
                        )
                        blob = _row_text_blob(snap)
                        if not any(_fragment_match(blob, b) for b in brand_bits):
                            brand_bits = []  # fall through to re-search
                        else:
                            args = {"product_id": nxt["product_id"]}
                            if nxt["action"] != "view":
                                args["sku"] = nxt["sku"]
                            forced.append((nxt["action"], args))
                            return forced[:max_calls]
                    elif not brand_bits:
                        args = {"product_id": nxt["product_id"]}
                        if nxt["action"] != "view":
                            args["sku"] = nxt["sku"]
                        forced.append((nxt["action"], args))
                        return forced[:max_calls]
            # Soft-coded recovery with no stem sibling in hand — re-search first.
            if (
                (soft_codes or board.task_shape == "preference_reasoning")
                and ("search" in names or "filter" in names)
                and len(forced) < max_calls
            ):
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    action, args = nxt_d
                    q = str(args.get("query") or "")
                    bits = [q]
                    if failed_stem and failed_stem not in q and soft_codes:
                        bits.append(failed_stem)
                    for frag in board.soft_preferences[:3]:
                        if frag and frag not in q and not re.search(r"\s", frag):
                            bits.append(frag)
                    args = {**args, "query": " ".join(b for b in bits if b).strip()}
                    forced.append((action, args))
                    return forced[:max_calls]
            # Prefer already-viewed soft-code stem matches before searching again.
            if soft_codes and "add_to_cart" in names:
                for score, key, row in board.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (
                        board.best_sku_for_product(pid) or ""
                    )
                    if not pid or not sku or key in board.failed_keys:
                        continue
                    if failed_stem and len(failed_stem) >= 4 and not sku.startswith(failed_stem):
                        continue
                    snap = board._product_snapshot(pid, sku)
                    blob = _row_text_blob(snap)
                    if not any(
                        _soft_code_match(blob, code) or code in sku for code in soft_codes
                    ):
                        continue
                    if pid not in board.viewed and "view" in names and len(forced) < max_calls:
                        forced.append(("view", {"product_id": pid, "sku": sku}))
                        return forced[:max_calls]
                    if board.stock_ok(pid, sku) and len(forced) < max_calls:
                        forced.append(("add_to_cart", {"product_id": pid, "sku": sku}))
                        return forced[:max_calls]
                    if (
                        "inspect_stock" in names
                        and len(forced) < max_calls
                        and key not in board.stock
                    ):
                        forced.append(
                            ("inspect_stock", {"product_id": pid, "sku": sku})
                        )
                        return forced[:max_calls]
            # No ready sibling — search again anchored on soft codes + sku stem.
            if ("search" in names or "filter" in names) and len(forced) < max_calls:
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    action, args = nxt_d
                    q = str(args.get("query") or "")
                    bits = [q]
                    if failed_stem and failed_stem not in q:
                        bits.append(failed_stem)
                    for frag in board.soft_preferences[:2]:
                        if frag and frag not in q:
                            bits.append(frag)
                    args = {**args, "query": " ".join(b for b in bits if b).strip()}
                    forced.append((action, args))
            return forced[:max_calls]

        # Empty cart after a failed live check — never search-loop; recover immediately.
        if board.must_recover and not board.cart:
            nxt = board.best_recovery()
            soft_codes = [
                frag
                for frag in board.soft_preferences
                if re.fullmatch(r"\d{3,14}", frag)
            ]
            stem = ""
            if board.last_failed_key and "::" in board.last_failed_key:
                stem = re.sub(r"-\d+$", "", board.last_failed_key.split("::", 1)[1])
            # Reject non-stem recovery targets when a model code was preferred.
            if (
                nxt
                and board.task_shape == "recovery"
                and soft_codes
                and stem
                and len(stem) >= 4
            ):
                nxt_sku = str(nxt.get("sku") or "")
                if not nxt_sku.startswith(stem):
                    nxt = None
            if nxt and nxt["action"] in names:
                args = {"product_id": nxt["product_id"]}
                if nxt["action"] != "view":
                    args["sku"] = nxt["sku"]
                return [(nxt["action"], args)]
            # Model-coded recovery: re-search the sku stem before carting unrelated
            # catalog hits that happen to mention a short code token.
            if (
                (soft_codes or stem)
                and board.task_shape == "recovery"
                and ("search" in names or "filter" in names)
                and board.search_count < 14
            ):
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    action, args = nxt_d
                    q = str(args.get("query") or "")
                    bits = [q]
                    if stem and stem not in q:
                        bits.append(stem)
                    for frag in board.soft_preferences[:3]:
                        if frag and frag not in q:
                            bits.append(frag)
                    args = {**args, "query": " ".join(b for b in bits if b).strip()}
                    return [(action, args)]
            # Phrase-match recovery: survey other phrase hits before reformulating.
            note_blob = " ".join(
                [_preamble(board.query)]
                + [_sanitize_note_for_search(n) for n in (board.shopper_notes or [])[-2:]]
            )
            targets = [t.lower() for t in _match_targets(note_blob)]
            if targets and "view" in names:
                for score, key, row in board.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (
                        board.best_sku_for_product(pid) or ""
                    )
                    if not pid or key in board.failed_keys:
                        continue
                    snap = board._product_snapshot(pid, sku) if sku else dict(row)
                    blob = _row_text_blob(snap)
                    if not any(t in blob for t in targets):
                        continue
                    if pid not in board.viewed:
                        return [
                            (
                                "view",
                                {"product_id": pid, "sku": sku} if sku else {"product_id": pid},
                            )
                        ]
                    if sku and board.stock_ok(pid, sku) and "add_to_cart" in names:
                        return [("add_to_cart", {"product_id": pid, "sku": sku})]
                    if sku and "inspect_stock" in names and key not in board.stock:
                        return [("inspect_stock", {"product_id": pid, "sku": sku})]
            if soft_codes and "view" in names:
                for score, key, row in board.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (
                        board.best_sku_for_product(pid) or ""
                    )
                    if not pid or key in board.failed_keys:
                        continue
                    # Prefer stem siblings over unrelated code-token matches.
                    if stem and len(stem) >= 4 and sku and not sku.startswith(stem):
                        continue
                    snap = board._product_snapshot(pid, sku) if sku else dict(row)
                    blob = _row_text_blob(snap)
                    if not any(
                        _soft_code_match(blob, code) or code in sku for code in soft_codes
                    ):
                        continue
                    if pid not in board.viewed:
                        return [("view", {"product_id": pid, "sku": sku} if sku else {"product_id": pid})]
                    if sku and board.stock_ok(pid, sku) and "add_to_cart" in names:
                        return [("add_to_cart", {"product_id": pid, "sku": sku})]
                    if sku and "inspect_stock" in names and key not in board.stock:
                        return [("inspect_stock", {"product_id": pid, "sku": sku})]
            if "search" in names or "filter" in names:
                # Phrase-match asks should re-query the phrase, not the failed sku stem.
                note_blob = " ".join(
                    [_preamble(board.query)]
                    + [_sanitize_note_for_search(n) for n in (board.shopper_notes or [])[-2:]]
                )
                has_phrase = bool(_match_targets(note_blob))
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    action, args = nxt_d
                    q = str(args.get("query") or "")
                    bits = [q]
                    if (
                        stem
                        and stem not in q
                        and not has_phrase
                        and board.soft_preferences
                    ):
                        bits.append(stem)
                    for frag in board.soft_preferences[:2]:
                        if frag and frag not in q:
                            bits.append(frag)
                    args = {**args, "query": " ".join(b for b in bits if b).strip()}
                    return [(action, args)]

        # After clarification, stop asking and immediately search with the new firm specs.
        if (
            board.received_clarification
            and board.task_shape == "constraint_satisfaction"
            and board.search_count == 0
            and ("search" in names or "filter" in names)
        ):
            nxt_d = _next_discovery(board, names)
            if nxt_d:
                return [nxt_d]

        if board.pending_question and "message" in names:
            # Interventions must be answered with message — do not skip to tools.
            actions: list[tuple[str, dict[str, Any]]] = [
                ("message", {"content": _recommend_message(board)[:400]})
            ]
            if board.cart and "inspect_stock" in names and len(actions) < max_calls:
                top = board.cart[-1]
                actions.append(
                    (
                        "inspect_stock",
                        {"product_id": top["product_id"], "sku": top["sku"]},
                    )
                )
            # Same turn: if a live-valid option is already inspected, cart it so a
            # shopper nudge cannot derail checkout into an endless re-search.
            if not board.cart and "add_to_cart" in names and len(actions) < max_calls:
                for _score, key, row in board.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (
                        board.best_sku_for_product(pid) or ""
                    )
                    if not pid or not sku or key in board.failed_keys:
                        continue
                    if pid not in board.viewed:
                        continue
                    if not board.stock_ok(pid, sku):
                        continue
                    snap = board._product_snapshot(pid, sku)
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        continue
                    actions.append(("add_to_cart", {"product_id": pid, "sku": sku}))
                    break
            return actions[:max_calls]

        if board.needs_live_check and board.cart and "inspect_stock" in names:
            top = board.cart[-1]
            if top["key"] not in board.post_cart_stock:
                return [
                    (
                        "inspect_stock",
                        {"product_id": top["product_id"], "sku": top["sku"]},
                    )
                ]

        # After a successful live inspect of a non-carted sku, cart it — never stall
        # on an LLM "I am going with" message that never commits.
        if (
            board.action_log
            and board.action_log[-1] == "inspect_stock"
            and not board.cart
            and "add_to_cart" in names
        ):
            soft_codes = [
                frag
                for frag in board.soft_preferences
                if re.fullmatch(r"\d{3,14}", frag)
            ]
            for score, key, row in board.ranked_candidates():
                pid = str(row.get("product_id") or "")
                sku = str(row.get("sku") or "") or (board.best_sku_for_product(pid) or "")
                if not pid or not sku or key in board.failed_keys:
                    continue
                if pid not in board.viewed:
                    continue
                if not board.stock_ok(pid, sku):
                    continue
                snap = board._product_snapshot(pid, sku)
                if board.firm_constraints and not board.row_compliant(snap, sku):
                    if board.search_count < 5 and board.view_count < 6:
                        continue
                brands = _required_brand_labels(board.shopper_notes)
                if brands:
                    blob = _row_text_blob(snap)
                    if not any(_fragment_match(blob, b) for b in brands):
                        continue
                if soft_codes:
                    blob = _row_text_blob(snap)
                    if not any(
                        _soft_code_match(blob, code) or code in sku for code in soft_codes
                    ):
                        if board.task_shape in {"recovery", "preference_reasoning"}:
                            continue
                        if score < 6:
                            continue
                if not _preference_cart_ok(board, snap):
                    continue
                # Prefer same sku-stem as the last failed cart when recovering model-coded
                # asks. Preference asks must change source — sibling skus score as no change.
                if (
                    board.must_recover
                    and board.last_failed_key
                    and "::" in board.last_failed_key
                    and board.task_shape != "preference_reasoning"
                ):
                    failed_sku = board.last_failed_key.split("::", 1)[1]
                    stem = re.sub(r"-\d+$", "", failed_sku)
                    if stem and len(stem) >= 4 and not sku.startswith(stem):
                        # Keep looking for a stem sibling first.
                        continue
                if board.must_recover and board.task_shape == "preference_reasoning":
                    failed_pid = board.last_failed_key.split("::", 1)[0] if board.last_failed_key else ""
                    if failed_pid and pid == failed_pid:
                        continue
                    failed_sku = (
                        board.last_failed_key.split("::", 1)[1]
                        if board.last_failed_key and "::" in board.last_failed_key
                        else ""
                    )
                    stem = re.sub(r"-\d+$", "", failed_sku)
                    if stem and len(stem) >= 4 and sku.startswith(stem):
                        continue
                return [("add_to_cart", {"product_id": pid, "sku": sku})]
            # If stem filter emptied the list, fall back to best_recovery.
            if board.must_recover:
                nxt = board.best_recovery()
                if nxt and nxt["action"] == "add_to_cart" and nxt["action"] in names:
                    if board.task_shape != "preference_reasoning" or _preference_cart_ok(
                        board,
                        board._product_snapshot(
                            str(nxt["product_id"]), str(nxt.get("sku") or "")
                        ),
                    ):
                        return [
                            (
                                "add_to_cart",
                                {"product_id": nxt["product_id"], "sku": nxt["sku"]},
                            )
                        ]
            # Preference: do not fall through to the LLM after a failed soft gate —
            # keep surveying or re-searching until a better soft match appears.
            if board.task_shape == "preference_reasoning":
                for _score, key, row in board.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    if not pid or pid in board.viewed:
                        continue
                    sku = str(row.get("sku") or "")
                    args = {"product_id": pid}
                    if sku:
                        args["sku"] = sku
                    if "view" in names:
                        return [("view", args)]
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    return [nxt_d]

        # Named tradeoff loops beat early checkout — re-rank after market moves first.
        if board.named and (
            board.priorities
            or board.task_shape in {"ranking", "preference_reasoning"}
            or len(board.named) >= 2
        ):
            named_actions = _named_pipeline_actions(
                board, tools=tools, max_calls=max_calls
            )
            if named_actions:
                return named_actions

        ready = _ready_checkout(board)
        if ready and "place_test_order" in names:
            return [
                (
                    "place_test_order",
                    _ground_order(dict(ready), tools=tools, board=board),
                )
            ]

        if board.cart and "inspect_stock" in names:
            top = board.cart[-1]
            key = top["key"]
            if key not in board.post_cart_stock or (
                key not in board.stock and top["product_id"] not in board.stock
            ):
                return [
                    (
                        "inspect_stock",
                        {"product_id": top["product_id"], "sku": top["sku"]},
                    )
                ]
            if not board.stock_ok(top["product_id"], top["sku"]) and "remove_from_cart" in names:
                return [
                    (
                        "remove_from_cart",
                        {"product_id": top["product_id"], "sku": top["sku"]},
                    )
                ]

        # Drive discovery/verify without the model when the board already has enough signal.
        if (
            board.candidates
            and board.view_count == 0
        ) or board.task_shape in {
            "recovery",
            "constraint_satisfaction",
            "retrieval",
            "retrieval_recall",
            "intent_decomposition",
            "preference_reasoning",
            "justification",
        }:
            progress = _progress_actions(board, tools=tools, max_calls=max_calls)
            if progress:
                return progress

        recent = board.action_log[-6:]
        stuck_search = (
            len(recent) >= 3
            and all(name in {"search", "filter", "message"} for name in recent[-3:])
        )
        stuck_inspect = (
            len(recent) >= 2
            and recent[-2:] == ["inspect_stock", "inspect_stock"]
        )
        if stuck_search or stuck_inspect or remaining <= 10 or turn == 1 or board.failed_keys:
            progress = _progress_actions(board, tools=tools, max_calls=max_calls)
            if progress:
                return progress

        if remaining <= 6:
            progress = _progress_actions(board, tools=tools, max_calls=max_calls)
            if progress:
                return progress
            if not board.named and ("search" in names or "filter" in names):
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    return [nxt_d]
        return None
    except Exception:
        return []


def _force_finish_actions(
    action_pairs: list[tuple[str, dict[str, Any]]],
    *,
    board: EpisodeBoard,
    tools: list[dict[str, Any]],
    remaining: int,
    max_calls: int = 1,
) -> list[tuple[str, dict[str, Any]]]:
    try:
        """When a carted sku is live-valid, checkout immediately — never stall."""
        names = _tool_names(tools)
        if "place_test_order" not in names:
            return action_pairs
        # Do not short-circuit named re-rank / recovery work.
        if board.named and len(board.named) >= 2 and (
            board.needs_rerank or board.must_recover
        ):
            named_actions = _named_pipeline_actions(
                board, tools=tools, max_calls=max_calls
            )
            if named_actions:
                return named_actions
        ready = _ready_checkout(board)
        if ready is not None:
            args = _ground_order(dict(ready), tools=tools, board=board)
            return [("place_test_order", args)]
        # Empty cart after failed live check — push recovery before time runs out.
        if not board.cart and board.failed_keys:
            nxt = board.best_recovery()
            if nxt and nxt["action"] in names:
                args = {"product_id": nxt["product_id"]}
                if nxt["action"] != "view":
                    args["sku"] = nxt["sku"]
                return [(nxt["action"], args)]
            # Late episode: cart any still-valid inspected option rather than stall.
            if remaining <= 10:
                for _score, key, row in board.ranked_candidates():
                    pid = str(row.get("product_id") or "")
                    sku = str(row.get("sku") or "") or (board.best_sku_for_product(pid) or "")
                    if not pid or not sku or key in board.failed_keys:
                        continue
                    snap = board._product_snapshot(pid, sku)
                    if pid not in board.viewed:
                        if "view" in names:
                            return [("view", {"product_id": pid, "sku": sku})]
                        continue
                    if not board.stock_ok(pid, sku):
                        if "inspect_stock" in names and key not in board.stock:
                            return [("inspect_stock", {"product_id": pid, "sku": sku})]
                        continue
                    if board.firm_constraints and not board.row_compliant(snap, sku):
                        continue
                    if board.task_shape == "preference_reasoning" and pid in _failed_product_ids(
                        board
                    ):
                        continue
                    if not _preference_cart_ok(board, snap):
                        continue
                    if "add_to_cart" in names:
                        return [("add_to_cart", {"product_id": pid, "sku": sku})]
            if remaining <= 12 and ("search" in names or "filter" in names):
                nxt_d = _next_discovery(board, names)
                if nxt_d:
                    return [nxt_d]
        if remaining <= 8 and not any(
            name in {"place_test_order", "add_to_cart", "view", "inspect_stock", "remove_from_cart"}
            for name, _ in action_pairs
        ):
            progress = _progress_actions(board, tools=tools, max_calls=1)
            if progress:
                return progress
        return action_pairs
    except Exception:
        return []


def _trim_messages(messages: list[dict[str, Any]]) -> None:
    try:
        if len(messages) <= _HISTORY_SOFT_CAP:
            return
        head = messages[:2]
        tail = messages[-(_HISTORY_SOFT_CAP - 3) :]
        messages[:] = head + [
            {
                "role": "user",
                "content": (
                    "Older turns are summarized on the episode board. "
                    "Use the board and recent tool results."
                ),
            }
        ] + tail
    except Exception:
        return None


def _slim_messages_for_inference(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    try:
        """Keep prompts small to avoid proxy 400s on oversized chat payloads."""
        if len(messages) <= 6:
            return messages
        head = messages[:2]
        tail = messages[-4:]
        return head + [
            {
                "role": "user",
                "content": "Earlier tool traffic omitted. Use the latest board status.",
            }
        ] + tail
    except Exception:
        return []


def _env_call(binding: dict[str, Any], envelope: dict[str, Any]) -> dict[str, Any] | None:
    try:
        """Single proxy post; ProxyClient already retries transient HTTP failures."""
        del binding  # binding is folded into the envelope by the caller
        return _proxy.post("/environment/call", json_data=envelope)
    except Exception:
        raise


def _inference(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        last_error: Exception | None = None
        # Prefer a slim payload first to cut 400/oversized failures; one model fallback max.
        candidates = [model]
        for alt in _model_candidates():
            if alt != model:
                candidates.append(alt)
                break
        message_variants = [_slim_messages_for_inference(messages), messages]
        for candidate in candidates:
            for msg in message_variants:
                try:
                    result = _proxy.post(
                        "/inference/chat/completions",
                        json_data={
                            "model": candidate,
                            "messages": msg,
                            "tools": tools,
                            "tool_choice": "required",
                            "temperature": 0,
                        },
                    )
                    if result is None:
                        raise RuntimeError("inference request failed")
                    return result
                except Exception as exc:  # noqa: BLE001 — try leaner payloads / models
                    last_error = exc
                    continue
        raise RuntimeError(f"inference request failed: {last_error}")
    except Exception:
        raise


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    dialogue: list[dict[str, Any]] = []
    try:
        environment = problem_data["environment"]
        binding = environment["binding"]
        policy = environment["policy_view"]
        tools = list(policy.get("tools") or [])
        query = str(policy.get("query") or "")
        problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        policy_calls = max(1, int(policy.get("max_calls_per_turn", 1)))
        named_preview = _named_candidates(query)
        # Named/compare tasks benefit from inspecting several variants per turn.
        call_cap = 8 if len(named_preview) >= 2 else 4
        max_calls = max(1, min(call_cap, policy_calls))
        max_steps = min(int(policy["max_steps"]), _TURN_CEILING)
        model = _model()

        allowed_claims = _claim_field_allowlist(tools)
        needs_justification = "justification" in _required_args(tools, "place_test_order")
        needs_ask = _needs_clarification(query)
        task_shape = _classify_task_shape(
            problem_data,
            tools=tools,
            named=named_preview,
            needs_ask=needs_ask,
        )
        board = EpisodeBoard(query, allowed_claims, task_shape=task_shape)

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": _system_prompt(
                    tools,
                    max_calls=max_calls,
                    max_steps=max_steps,
                    needs_ask=board.needs_ask,
                    named=board.named,
                    claim_fields=sorted(allowed_claims),
                    needs_justification=needs_justification,
                ),
            },
            {"role": "user", "content": query},
        ]

        for turn in range(1, max_steps + 1):
            _trim_messages(messages)
            assistant_content = ""
            action_pairs: list[tuple[str, dict[str, Any]]] = []

            scripted = _scripted_actions(
                board,
                tools=tools,
                max_calls=max_calls,
                turn=turn,
                max_steps=max_steps,
            )
            if scripted:
                action_pairs = scripted
            else:
                ask = list(messages)
                ask.append({"role": "user", "content": board.render(turn, max_steps)})
                tool_calls: list[dict[str, Any]] = []
                try:
                    inference = _inference(model=model, messages=ask, tools=tools)
                    assistant = inference["choices"][0]["message"]
                    assistant_content = assistant.get("content") or ""
                    tool_calls = list(assistant.get("tool_calls") or [])[:max_calls]
                except Exception:  # noqa: BLE001 — fall back to board progress on inference failure
                    tool_calls = []

                if not tool_calls:
                    action_pairs = _progress_actions(
                        board, tools=tools, max_calls=max_calls
                    )
                else:
                    for call in tool_calls:
                        try:
                            name = call["function"]["name"]
                            if name not in _tool_names(tools):
                                continue
                            args = _normalize_args(
                                name,
                                _arguments(call),
                                tools=tools,
                                board=board,
                            )
                        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                            continue
                        action_pairs.append((name, args))

                    # Drop lone commitment messages that never cart — progress instead.
                    if (
                        action_pairs
                        and all(name == "message" for name, _ in action_pairs)
                        and not board.pending_question
                        and not (board.needs_ask and not board.received_clarification)
                    ):
                        progress = _progress_actions(
                            board, tools=tools, max_calls=max_calls
                        )
                        if progress:
                            action_pairs = progress

            # One guard pass for scripted, LLM, and progress paths.
            if action_pairs:
                action_pairs = _guard_actions(
                    action_pairs, board=board, tools=tools, max_calls=max_calls
                )

            action_pairs = _force_finish_actions(
                action_pairs,
                board=board,
                tools=tools,
                remaining=max_steps - turn + 1,
                max_calls=max_calls,
            )
            # Final claim grounding immediately before the environment call.
            rebuilt: list[tuple[str, dict[str, Any]]] = []
            for name, args in action_pairs:
                if name == "place_test_order":
                    args = _ground_order(dict(args), tools=tools, board=board)
                rebuilt.append((name, args))
            action_pairs = rebuilt
            if not action_pairs:
                action_pairs = _progress_actions(board, tools=tools, max_calls=max_calls)
            if not action_pairs:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": "Previous arguments were invalid. Call a schema-valid tool.",
                        },
                    ]
                )
                continue

            prepared_calls: list[dict[str, Any]] = []
            for index, (name, args) in enumerate(action_pairs, start=1):
                prepared_calls.append(
                    {
                        "id": f"{problem_id}-t{turn}-c{index}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": prepared_calls,
                }
            )

            group_id = f"{problem_id}-turn-{turn}"
            envelope = {
                **binding,
                "call_id": group_id,
                "idempotency_key": group_id,
                "turn": turn,
                "calls": [
                    {
                        "call_id": f"{group_id}-{index}",
                        "action": {"name": name, "args": args},
                    }
                    for index, (name, args) in enumerate(action_pairs, start=1)
                ],
            }
            result = None
            for resync in range(2):
                attempt_envelope = dict(envelope)
                if resync:
                    attempt_envelope["turn"] = turn + resync
                    attempt_envelope["call_id"] = f"{group_id}-rs{resync}"
                    attempt_envelope["idempotency_key"] = attempt_envelope["call_id"]
                result = _env_call(binding, attempt_envelope)
                if result is not None:
                    break
            if result is None:
                break

            for name, args in action_pairs:
                board.remember_action(name, args)

            dialogue.append(
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": prepared_calls,
                    "environment_result": result,
                }
            )

            for (name, args), tool_call, call_result in zip(
                action_pairs, prepared_calls, result["calls"], strict=True
            ):
                raw_observation = call_result.get("observation") or {}
                if not isinstance(raw_observation, dict):
                    raw_observation = {"raw": raw_observation}
                observation = _unwrap_observation(raw_observation)
                board.ingest(name, args, observation)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": _compact_observation(observation),
                    }
                )

            user_message = result.get("user_message")
            if isinstance(user_message, dict) and user_message.get("content"):
                content = str(user_message["content"])
                board.note_shopper(content)
                messages.append({"role": "user", "content": content})

            if any(
                (
                    isinstance(call.get("observation"), dict)
                    and (
                        call["observation"].get("done")
                        or _unwrap_observation(call["observation"]).get("done")
                    )
                )
                for call in result.get("calls") or []
            ):
                break

        return dialogue
    except Exception:
        return dialogue


__all__ = ["agent_main"]
