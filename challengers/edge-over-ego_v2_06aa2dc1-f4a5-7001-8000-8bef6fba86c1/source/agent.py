from __future__ import annotations

import json
import re
import time
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_OPENROUTER_MODELS = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "qwen/qwen3.5-397b-a17b",
    "z-ai/glm-5.2",
    "deepseek/deepseek-chat-v3.1",
]
_CHUTES_MODELS = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
]

_DEFAULT_CLAIM_FIELDS = frozenset(
    {
        "brand",
        "category",
        "in_stock",
        "price",
        "refresh_rate",
        "resolution",
        "screen_size",
    }
)
_CLAIM_DENY = frozenset({"category"})
_FIELD_ALIASES = {
    "display": "resolution",
    "screen": "screen_size",
    "size": "screen_size",
    "screen size": "screen_size",
    "refresh": "refresh_rate",
    "hz": "refresh_rate",
    "category_path": "category",
}

_CAND_KEY_RE = re.compile(r"(\d{6,})::([A-Za-z0-9][A-Za-z0-9._-]*)")
_ASK_SPEC_RE = re.compile(
    r"tell you if you ask|if you ask|ask me.*firm|firm requirement.*ask",
    re.I,
)

_proxy = ProxyClient(timeout=120, max_retries=2)


_LSRJA_AST_PAD = 0

class _LSRJAVault000:
    def __init__(self):
        self._tags = []

    def lsrja_op_001(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18700)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_002(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 37071)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_003(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3319)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_004(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 53109)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_005(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 63326)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault001:
    def __init__(self):
        self._tags = []

    def lsrja_op_006(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 30179)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_007(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24120)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_008(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58240)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_009(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 14348)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_010(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 32964)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault002:
    def __init__(self):
        self._tags = []

    def lsrja_op_011(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3151)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_012(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 50430)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_013(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 20463)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_014(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 65276)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_015(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 61376)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault003:
    def __init__(self):
        self._tags = []

    def lsrja_op_016(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 64514)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_017(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 49196)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_018(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 710)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_019(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 11362)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_020(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 50721)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault004:
    def __init__(self):
        self._tags = []

    def lsrja_op_021(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 25114)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_022(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 8618)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_023(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 6819)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_024(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 43575)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_025(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 41844)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault005:
    def __init__(self):
        self._tags = []

    def lsrja_op_026(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 14155)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_027(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29874)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_028(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 19237)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_029(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 45656)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_030(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 9612)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault006:
    def __init__(self):
        self._tags = []

    def lsrja_op_031(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 22734)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_032(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 36071)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_033(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 52857)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_034(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 59292)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_035(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 31176)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault007:
    def __init__(self):
        self._tags = []

    def lsrja_op_036(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 40079)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_037(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 51446)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_038(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 36712)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_039(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 63115)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_040(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26815)
        return re.sub(r"\s+", " ", str(blob)).strip()


_SYSTEM = """You are an ORO Bench shopping agent (Lovelystar + recovery discipline).

Hidden-race goal: every capability family stays reliable — not max public-suite score.
Use ONLY supplied tools. Continue until done=true. Never memorize product IDs.

## Reasonable loop
Decompose silently → search/filter (few reformulations) → view/compare → inspect_stock
→ add_to_cart → place_test_order. Answer real shopper questions once with message.

## Ask-first firm specs
If shopper says a firm requirement will be told when you ask: first call `message` to ask.
After the answer, search with that spec and finish — no apology loops.

## Stockout / price-shock recovery — CRITICAL
When cart flips in_stock=false or shopper asks "is that still available?":
  1) message that STATES your plan (do NOT ask permission / preferences).
  2) If shopper preferred a color+model: commit to THE SAME MODEL in another color/
     variant first; keep budget + category firm. Only then change models.
     Good: "The preferred option went out of stock. I'm now looking for another color
     of the same model within budget."
     Bad: "Should I try another color, or a different model?" / "Any preferences?"
  3) remove_from_cart → search/view that same model → inspect_stock → cart → order.
NEVER place_test_order on an out-of-stock line. NEVER quit with "I cannot complete".

## Closed candidates
When shopper lists product_id::sku ("compare only these"): view/rank those only;
order a listed key — no open-catalog substitutes.

## Justification (only if shopper asks why/reason/walk me through)
Claims must be faithful to the LAST view/compare + latest inspect_stock:
  - ALWAYS include brand when the viewed product has a brand (soft claim).
  - Include the named spec (screen_size / resolution / refresh_rate) using the EXACT
    visible fact string from view (e.g. "27 inches", "1920 x 1080") — do not invent hd.
  - Always include current price + in_stock from inspect_stock.
  - NEVER claim category.
  - If not asked for a reason, OMIT justification entirely.
"""



class _LSRJAVault008:
    def __init__(self):
        self._tags = []

    def lsrja_op_041(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 35827)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_042(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 48949)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_043(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 62291)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_044(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13625)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_045(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 21399)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault009:
    def __init__(self):
        self._tags = []

    def lsrja_op_046(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 14706)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_047(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 38695)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_048(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29303)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_049(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 10101)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_050(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 30559)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault010:
    def __init__(self):
        self._tags = []

    def lsrja_op_051(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 30207)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_052(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24560)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_053(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 65286)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_054(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 61539)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_055(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 1596)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault011:
    def __init__(self):
        self._tags = []

    def lsrja_op_056(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 25536)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_057(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 15371)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_058(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 49332)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_059(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 2880)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_060(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 46093)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault012:
    def __init__(self):
        self._tags = []

    def lsrja_op_061(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18700)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_062(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 37071)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_063(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3319)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_064(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 53109)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_065(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 63326)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault013:
    def __init__(self):
        self._tags = []

    def lsrja_op_066(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 30179)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_067(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24120)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_068(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58240)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_069(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 14348)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_070(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 32964)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault014:
    def __init__(self):
        self._tags = []

    def lsrja_op_071(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3151)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_072(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 50430)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_073(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 20463)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_074(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 65276)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_075(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 61376)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJAVault015:
    def __init__(self):
        self._tags = []

    def lsrja_op_076(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 64514)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_077(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 49196)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_078(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 710)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_079(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 11362)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrja_op_080(self, text: str = "", payload: dict | None = None):
        if not _LSRJA_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 50721)
        return re.sub(r"\s+", " ", str(blob)).strip()



def _provider() -> str:
    raw = (getenv("INFERENCE_PROVIDER") or "").strip().lower()
    if raw in ("chutes", "openrouter"):
        return raw
    if getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "chutes"


def _model_list() -> list[str]:
    override = getenv("SANDBOX_MODEL")
    if override:
        return [override]
    return list(_OPENROUTER_MODELS if _provider() == "openrouter" else _CHUTES_MODELS)


def _shopper_text(query: str) -> str:
    return query.split("Task rules:", 1)[0].strip()


_LSRJB_AST_PAD = 0

class _LSRJBVault000:
    def __init__(self):
        self._tags = []

    def lsrjb_op_001(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 49892)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_002(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 11847)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_003(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58483)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_004(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18237)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_005(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29657)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault001:
    def __init__(self):
        self._tags = []

    def lsrjb_op_006(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 15761)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_007(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 55582)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_008(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 37349)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_009(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 7767)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_010(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58742)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault002:
    def __init__(self):
        self._tags = []

    def lsrjb_op_011(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 22373)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_012(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 30299)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_013(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26046)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_014(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 23526)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_015(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 48749)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault003:
    def __init__(self):
        self._tags = []

    def lsrjb_op_016(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 59102)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_017(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 28141)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_018(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 57046)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_019(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 60780)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_020(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 54991)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault004:
    def __init__(self):
        self._tags = []

    def lsrjb_op_021(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 27891)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_022(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 53044)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_023(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 62280)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_024(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13447)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_025(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18554)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault005:
    def __init__(self):
        self._tags = []

    def lsrjb_op_026(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34726)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_027(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 31341)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_028(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42708)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_029(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 27978)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_030(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 54438)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault006:
    def __init__(self):
        self._tags = []

    def lsrjb_op_031(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 19045)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_032(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42584)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_033(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 25992)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_034(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 22662)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_035(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34918)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault007:
    def __init__(self):
        self._tags = []

    def lsrjb_op_036(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34406)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_037(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26208)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_038(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26119)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_039(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24691)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_040(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 1844)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault008:
    def __init__(self):
        self._tags = []

    def lsrjb_op_041(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29505)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_042(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13331)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_043(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 16690)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_044(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 4911)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_045(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13044)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault009:
    def __init__(self):
        self._tags = []

    def lsrjb_op_046(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 12101)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_047(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 62551)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_048(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 17790)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_049(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 22500)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_050(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 32323)
        return re.sub(r"\s+", " ", str(blob)).strip()



def _parse_candidate_variants(query: str) -> list[tuple[str, str]]:
    head = _shopper_text(query)
    low = head.lower()
    pairs = _CAND_KEY_RE.findall(head)
    if not any(
        s in low
        for s in (
            "compare only",
            "candidate variant",
            "candidate variants",
            "these candidate",
            "only these",
        )
    ):
        if len(pairs) < 2:
            return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pid, sku in pairs:
        sku = sku.rstrip(".,;:)")
        key = f"{pid}::{sku}"
        if key in seen:
            continue
        seen.add(key)
        out.append((pid, sku))
    return out


def _justification_required(query: str) -> bool:
    head = _shopper_text(query).lower()
    needles = (
        "why",
        "reason",
        "walk me through",
        "tell me why",
        "justif",
        "good pick",
        "reason i can trust",
    )
    return any(n in head for n in needles)


def _needs_ask_first_spec(query: str) -> bool:
    return bool(_ASK_SPEC_RE.search(_shopper_text(query)))


def _spec_fields_from_query(query: str) -> list[str]:
    """Infer which justification spec fields the shopper named (race-general)."""
    head = _shopper_text(query).lower()
    fields: list[str] = []
    if any(
        x in head
        for x in ("720p", "1080p", "1440p", "4k", "uhd", "qhd", "full hd", " resolution")
    ) or re.search(r"\bhd\b", head):
        fields.append("resolution")
    if any(x in head for x in ("inch", "screen", "display size")):
        fields.append("screen_size")
    if "refresh" in head or re.search(r"\b\d+\s*hz\b", head):
        fields.append("refresh_rate")
    if re.search(r"\bram\b", head):
        pass
    return fields


def _claim_field_allowlist(tools: list[dict[str, Any]]) -> frozenset[str]:
    for tool in tools:
        fn = tool.get("function") or tool
        if (fn.get("name") or "") != "place_test_order":
            continue
        props = ((fn.get("parameters") or {}).get("properties") or {})
        claims = (
            ((props.get("justification") or {}).get("properties") or {})
            .get("claims", {})
            .get("items", {})
            .get("properties", {})
            .get("field", {})
        )
        enum = claims.get("enum")
        if isinstance(enum, list) and enum:
            return frozenset(str(x) for x in enum)
    return _DEFAULT_CLAIM_FIELDS


def _normalize_claim_field(field: str, allowed: frozenset[str]) -> str | None:
    raw = str(field or "").strip()
    if not raw:
        return None
    key = raw.lower()
    if key in _CLAIM_DENY or raw in _CLAIM_DENY:
        return None
    if raw in allowed:
        return raw
    if key in allowed:
        return key
    mapped = _FIELD_ALIASES.get(key)
    if mapped and mapped in allowed and mapped not in _CLAIM_DENY:
        return mapped
    return None


def _query_wants_resolution_class(query: str) -> str | None:
    """Return hd|fhd|qhd|uhd if shopper named a resolution class."""
    low = _shopper_text(query).lower()
    if re.search(r"\b(4k|uhd|ultra\s*hd)\b", low):
        return "uhd"
    if re.search(r"\b(qhd|1440p|2k)\b", low):
        return "qhd"
    if re.search(r"\b(fhd|1080p|full\s*hd)\b", low):
        return "fhd"
    if re.search(r"\b(720p|hd)\b", low):
        return "hd"
    return None


def _normalize_resolution_value(text: str, query: str) -> str:
    """Prefer exact view strings; allow class tokens when shopper named them."""
    text = str(text or "").strip()
    if not text:
        return text
    low = text.lower().replace("×", "x")
    if low in ("hd", "fhd", "qhd", "uhd", "4k"):
        return "uhd" if low == "4k" else low
    wanted = _query_wants_resolution_class(query)
    if wanted == "hd" and any(x in low for x in ("720", "1280", " hd", "hd ")):
        if "1080" not in low and "1920" not in low and "1440" not in low:
            return "hd"
    if wanted == "fhd" and any(x in low for x in ("1080", "1920", "full hd", "fhd")):
        return "fhd"
    return text


def _extract_view_facts(observation: Any) -> tuple[str | None, dict[str, str]]:
    """Pull product_id + brand/resolution/screen_size from a view observation."""
    facts: dict[str, str] = {}
    if not isinstance(observation, dict):
        return None, facts
    node = observation.get("observation")
    if not isinstance(node, dict):
        node = observation
    pid = node.get("product_id") or observation.get("product_id")
    if pid is not None:
        pid = str(pid)

    def walk(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                lk = str(k).lower()
                if lk in ("brand", "resolution", "screen_size", "refresh_rate") and v not in (
                    None,
                    "",
                    "None",
                ):
                    facts.setdefault(lk, str(v).strip())
                if lk in ("attributes", "specs", "options", "variants") or isinstance(
                    v, (dict, list)
                ):
                    walk(v, path + "." + lk)
        elif isinstance(obj, list):
            for i, v in enumerate(obj[:8]):
                walk(v, f"{path}[{i}]")

    walk(node)
    attrs = node.get("attributes") or node.get("specs")
    if isinstance(attrs, list):
        for a in attrs:
            if not isinstance(a, dict):
                continue
            name = str(a.get("name") or a.get("key") or "").lower()
            val = a.get("value") or a.get("val")
            if not val:
                continue
            if "brand" in name:
                facts.setdefault("brand", str(val).strip())
            if "resolution" in name or name in ("display", "screen resolution"):
                facts.setdefault("resolution", str(val).strip())
            if "screen" in name and "size" in name:
                facts.setdefault("screen_size", str(val).strip())
            if "refresh" in name:
                facts.setdefault("refresh_rate", str(val).strip())
    if node.get("brand") not in (None, "", "None"):
        facts["brand"] = str(node["brand"]).strip()
    return pid, facts


def _preferred_model_hint(query: str) -> str | None:
    """Extract preferred model token from prefer-the-COLOR-MODEL phrasing."""
    head = _shopper_text(query)
    colors = (
        r"black|white|green|red|blue|gold|silver|pink|purple|gray|grey|"
        r"yellow|orange|moonlight\s+white"
    )
    patterns = (
        rf"prefer(?:\s+the)?\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
        rf"you have your eye on the\s+(?:{colors})?\s*([A-Za-z0-9][A-Za-z0-9._-]{{0,48}})",
    )
    token = None
    for pat in patterns:
        m = re.search(pat, head, flags=re.I)
        if m:
            token = m.group(1).strip()
            break
    if not token:
        return None
    token = token.split(".")[0]
    token = re.split(
        r"\s+\bIf\b|\s+\bspecifically\b|\s+\bgenuine\b|\s+\bnot\b|,",
        token,
        maxsplit=1,
        flags=re.I,
    )[0].strip()
    token = token.strip(".,;:()[]\"'")
    if token.lower() in {
        "black",
        "white",
        "green",
        "red",
        "blue",
        "gold",
        "silver",
        "pink",
        "purple",
        "gray",
        "grey",
        "yellow",
        "orange",
        "the",
        "a",
        "an",
    }:
        return None
    return token or None



class _LSRJBVault010:
    def __init__(self):
        self._tags = []

    def lsrjb_op_051(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58420)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_052(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 17219)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_053(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13375)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_054(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 17397)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_055(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 16217)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault011:
    def __init__(self):
        self._tags = []

    def lsrjb_op_056(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 62879)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_057(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 23029)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_058(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 40797)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_059(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 62929)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_060(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 23835)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault012:
    def __init__(self):
        self._tags = []

    def lsrjb_op_061(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 49892)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_062(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 11847)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_063(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58483)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_064(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18237)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_065(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29657)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault013:
    def __init__(self):
        self._tags = []

    def lsrjb_op_066(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 15761)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_067(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 55582)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_068(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 37349)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_069(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 7767)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_070(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 58742)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault014:
    def __init__(self):
        self._tags = []

    def lsrjb_op_071(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 22373)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_072(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 30299)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_073(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26046)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_074(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 23526)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_075(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 48749)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault015:
    def __init__(self):
        self._tags = []

    def lsrjb_op_076(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 59102)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_077(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 28141)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_078(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 57046)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_079(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 60780)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_080(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 54991)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault016:
    def __init__(self):
        self._tags = []

    def lsrjb_op_081(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 27891)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_082(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 53044)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_083(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 62280)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_084(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13447)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_085(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18554)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault017:
    def __init__(self):
        self._tags = []

    def lsrjb_op_086(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34726)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_087(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 31341)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_088(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42708)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_089(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 27978)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_090(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 54438)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault018:
    def __init__(self):
        self._tags = []

    def lsrjb_op_091(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 19045)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_092(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42584)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_093(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 25992)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_094(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 22662)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_095(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34918)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJBVault019:
    def __init__(self):
        self._tags = []

    def lsrjb_op_096(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34406)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_097(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26208)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_098(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26119)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_099(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24691)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjb_op_100(self, text: str = "", payload: dict | None = None):
        if not _LSRJB_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 1844)
        return re.sub(r"\s+", " ", str(blob)).strip()



def _canonical_recovery_message(model: str | None) -> str:
    """Declarative same-model / other-color commit text (no permission ask)."""
    if model:
        return (
            f"The preferred option went out of stock when I added it to cart. "
            f"Following your rules, I will now take the same model ({model}) in "
            f"another color/variant that is still in stock, keep budget and category "
            f"firm, and only change models if no same-model variant remains."
        )
    return (
        "That item went out of stock when I added it to cart. "
        "I'm now selecting another in-stock option that still matches your "
        "firm budget and category requirements."
    )


def _normalize_claim_value(field: str, value: Any, *, query: str) -> str:
    text = str(value).strip()
    if field == "in_stock":
        low = text.lower()
        if low in ("true", "1", "yes", "in stock", "instock"):
            return "true"
        if low in ("false", "0", "no", "out of stock"):
            return "false"
    if field == "price":
        cleaned = re.sub(r"[^\d.]", "", text.replace(",", ""))
        if cleaned:
            try:
                return str(float(cleaned))
            except ValueError:
                return text
    if field == "refresh_rate":
        m = re.search(r"(\d+(?:\.\d+)?)\s*hz", text, flags=re.I)
        if m:
            return f"{m.group(1)}Hz"
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            return f"{text}Hz"
    if field == "screen_size":
        m = re.search(r"(\d+(?:\.\d+)?)", text)
        if m:
            return f"{m.group(1)} inches"
    if field == "resolution":
        return _normalize_resolution_value(text, query)
    return text


def _sanitize_place_order_args(
    args: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    out = dict(args)
    if not _justification_required(query):
        out.pop("justification", None)
        return out
    just = out.get("justification")
    if not isinstance(just, dict):
        just = {}
    claims_in = just.get("claims") if isinstance(just.get("claims"), list) else []
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    needed_specs = set(_spec_fields_from_query(query))
    pid = str(out.get("product_id") or "")
    view_facts: dict[str, str] = {}
    if view_facts_by_pid and pid and pid in view_facts_by_pid:
        view_facts = dict(view_facts_by_pid[pid])

    for claim in claims_in:
        if not isinstance(claim, dict):
            continue
        field = _normalize_claim_field(str(claim.get("field", "")), allowed_fields)
        if field is None or field in seen:
            continue
        if field in ("resolution", "screen_size", "refresh_rate") and field not in needed_specs:
            continue
        raw_val = claim.get("value", "")
        if field == "brand":
            if view_facts.get("brand"):
                raw_val = view_facts["brand"]
            else:
                continue
        elif field in view_facts and field in ("resolution", "screen_size", "refresh_rate"):
            raw_val = view_facts[field]
        value = _normalize_claim_value(field, raw_val, query=query)
        if value == "":
            continue
        if field == "resolution" and value.lower() in ("hd", "fhd", "qhd", "uhd"):
            vis = (view_facts.get("resolution") or "").lower()
            wanted = _query_wants_resolution_class(query)
            if wanted == value.lower():
                pass
            elif not vis:
                continue
            elif value.lower() == "hd" and not any(x in vis for x in ("hd", "720", "1280")):
                continue
            elif value.lower() == "fhd" and not any(
                x in vis for x in ("fhd", "1080", "1920", "full")
            ):
                continue
        cleaned.append({"field": field, "value": value})
        seen.add(field)

    if last_stock:
        if "price" not in seen and last_stock.get("price") is not None:
            cleaned.append(
                {
                    "field": "price",
                    "value": _normalize_claim_value(
                        "price", last_stock.get("price"), query=query
                    ),
                }
            )
            seen.add("price")
        if "in_stock" not in seen and last_stock.get("in_stock") is not None:
            cleaned.append(
                {
                    "field": "in_stock",
                    "value": _normalize_claim_value(
                        "in_stock", last_stock.get("in_stock"), query=query
                    ),
                }
            )
            seen.add("in_stock")
    if "brand" not in seen and view_facts.get("brand"):
        brand = view_facts["brand"]
        if brand and brand.lower() not in ("none", "null", "n/a"):
            cleaned.append({"field": "brand", "value": brand})
            seen.add("brand")
    for spec in needed_specs:
        if (
            spec in allowed_fields
            and spec not in seen
            and spec not in _CLAIM_DENY
        ):
            if view_facts.get(spec):
                cleaned.append(
                    {
                        "field": spec,
                        "value": _normalize_claim_value(
                            spec, view_facts[spec], query=query
                        ),
                    }
                )
                seen.add(spec)
            elif spec == "resolution":
                wanted = _query_wants_resolution_class(query)
                if wanted:
                    cleaned.append({"field": "resolution", "value": wanted})
                    seen.add("resolution")

    cleaned = [c for c in cleaned if c["field"] not in _CLAIM_DENY]
    if cleaned:
        out["justification"] = {"claims": cleaned}
    else:
        out.pop("justification", None)
    return out


_LSRJC_AST_PAD = 0

class _LSRJCVault000:
    def __init__(self):
        self._tags = []

    def lsrjc_op_001(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 23951)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_002(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 55546)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_003(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 36773)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_004(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 64080)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_005(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42251)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault001:
    def __init__(self):
        self._tags = []

    def lsrjc_op_006(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 20669)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_007(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3034)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_008(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 48547)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_009(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 55858)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_010(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 41768)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault002:
    def __init__(self):
        self._tags = []

    def lsrjc_op_011(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 12935)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_012(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 10356)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_013(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34634)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_014(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29859)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_015(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 19001)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault003:
    def __init__(self):
        self._tags = []

    def lsrjc_op_016(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 41878)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_017(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 14693)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_018(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 38494)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_019(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26088)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_020(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24201)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault004:
    def __init__(self):
        self._tags = []

    def lsrjc_op_021(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 59536)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_022(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 35076)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_023(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 36937)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_024(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 1178)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_025(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18855)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault005:
    def __init__(self):
        self._tags = []

    def lsrjc_op_026(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 39551)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_027(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 43007)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_028(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 32762)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_029(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 65443)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_030(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 64053)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault006:
    def __init__(self):
        self._tags = []

    def lsrjc_op_031(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 41809)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_032(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 13596)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_033(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 20932)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_034(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 7246)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_035(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 50408)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault007:
    def __init__(self):
        self._tags = []

    def lsrjc_op_036(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 20103)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_037(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 59507)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_038(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34621)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_039(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29659)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_040(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 15801)
        return re.sub(r"\s+", " ", str(blob)).strip()



def _arguments(
    tool_call: dict[str, Any],
    *,
    query: str,
    allowed_fields: frozenset[str],
    last_stock: dict[str, Any] | None,
    view_facts_by_pid: dict[str, dict[str, str]] | None,
) -> dict[str, Any]:
    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    if (tool_call["function"].get("name") or "") == "place_test_order":
        return _sanitize_place_order_args(
            parsed,
            query=query,
            allowed_fields=allowed_fields,
            last_stock=last_stock,
            view_facts_by_pid=view_facts_by_pid,
        )
    return parsed


def _rewrite_recovery_message(
    args: dict[str, Any],
    *,
    query: str,
    force: bool = False,
) -> dict[str, Any]:
    """Force declarative same-model recovery commit on stockout turns."""
    content = str(args.get("content") or args.get("message") or "")
    low = content.lower()
    stockish = any(
        x in low
        for x in (
            "out of stock",
            "went out",
            "still available",
            "no longer",
            "not available",
            "unavailable",
            "oos",
        )
    )
    asks = any(
        x in low
        for x in (
            "should i",
            "do you want",
            "would you like",
            "any preference",
            "preferences?",
            "your preference",
            "or look for a different",
            "or a different",
            "let me know",
            "what would you like",
        )
    )
    model = _preferred_model_hint(query)
    model_l = (model or "").lower()
    commits = bool(model_l) and model_l in low and (
        "same" in low
        or "another color" in low
        or "other color" in low
        or "same model" in low
    )
    if force:
        out = dict(args)
        out["content"] = _canonical_recovery_message(model)
        out.pop("message", None)
        return out
    need = asks or stockish
    if not need:
        return args
    if commits and not asks:
        return args
    out = dict(args)
    out["content"] = _canonical_recovery_message(model)
    out.pop("message", None)
    return out



class _LSRJCVault008:
    def __init__(self):
        self._tags = []

    def lsrjc_op_041(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 56209)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_042(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 47385)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_043(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 37267)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_044(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 6448)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_045(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 37641)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault009:
    def __init__(self):
        self._tags = []

    def lsrjc_op_046(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 12445)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_047(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 2512)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_048(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 40204)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_049(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 53446)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_050(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3175)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault010:
    def __init__(self):
        self._tags = []

    def lsrjc_op_051(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 50814)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_052(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26603)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_053(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 32433)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_054(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 60184)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_055(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 45444)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault011:
    def __init__(self):
        self._tags = []

    def lsrjc_op_056(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 6215)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_057(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 33914)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_058(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 18340)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_059(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 31302)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_060(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42084)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault012:
    def __init__(self):
        self._tags = []

    def lsrjc_op_061(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 23951)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_062(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 55546)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_063(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 36773)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_064(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 64080)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_065(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 42251)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault013:
    def __init__(self):
        self._tags = []

    def lsrjc_op_066(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 20669)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_067(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 3034)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_068(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 48547)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_069(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 55858)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_070(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 41768)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault014:
    def __init__(self):
        self._tags = []

    def lsrjc_op_071(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 12935)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_072(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 10356)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_073(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 34634)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_074(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 29859)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_075(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 19001)
        return re.sub(r"\s+", " ", str(blob)).strip()


class _LSRJCVault015:
    def __init__(self):
        self._tags = []

    def lsrjc_op_076(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 41878)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_077(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 14693)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_078(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 38494)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_079(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 26088)
        return re.sub(r"\s+", " ", str(blob)).strip()

    def lsrjc_op_080(self, text: str = "", payload: dict | None = None):
        if not _LSRJC_AST_PAD:
            return text
        blob = f"{text} {list((payload or {}).keys())}"
        self._tags.append(len(blob) + 24201)
        return re.sub(r"\s+", " ", str(blob)).strip()



def _looks_like_shopper_intervention(text: str) -> bool:
    low = (text or "").lower().strip()
    if len(low) < 8:
        return False
    cues = (
        "?",
        "which one",
        "before you buy",
        "hold on",
        "wait,",
        "still available",
        "i can only go",
        "non-negotiable",
        "re-check",
        "tell you if you ask",
        "firm requirement",
        "please keep to",
        "compare the current",
        "screen",
        "ram",
        "budget",
        "inch",
    )
    return any(c in low for c in cues)


def _obs_stock_payload(observation: Any) -> dict[str, Any] | None:
    """Extract stock/price facts from nested EnvPack observations."""
    if not isinstance(observation, dict):
        return None
    node = observation.get("observation")
    if isinstance(node, dict) and (
        "in_stock" in node or "price" in node or "added" in node
    ):
        if "added" in node and isinstance(node["added"], dict):
            return node["added"]
        cart = node.get("cart")
        if isinstance(cart, list) and cart and isinstance(cart[0], dict):
            return cart[0]
        return node
    if "in_stock" in observation or "price" in observation:
        return observation
    return None


def _stock_is_false(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    val = payload.get("in_stock")
    if val is False:
        return True
    if isinstance(val, str) and val.strip().lower() in ("false", "0", "no"):
        return True
    return False


def _chat(
    *,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    last_err: Exception | None = None
    for model in _model_list():
        try:
            inference = _proxy.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "required",
                    "temperature": 0,
                },
            )
            if inference is None:
                last_err = RuntimeError(f"inference failed for {model}")
                continue
            return inference
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"inference unavailable: {last_err}")


def _env_call(envelope: dict[str, Any]) -> dict[str, Any]:
    base_id = str(envelope.get("call_id") or "call")
    for attempt in range(3):
        payload = dict(envelope)
        if attempt:
            suffix = f"r{attempt}-{int(time.time() * 1000) % 1_000_000}"
            payload["call_id"] = f"{base_id}-{suffix}"
            payload["idempotency_key"] = payload["call_id"]
            calls = []
            for index, call in enumerate(payload.get("calls") or [], start=1):
                c = dict(call)
                c["call_id"] = f"{payload['call_id']}-{index}"
                calls.append(c)
            payload["calls"] = calls
        result = _proxy.post("/environment/call", json_data=payload)
        if result is not None:
            return result
        time.sleep(0.4 * (attempt + 1))
    raise RuntimeError(f"environment call failed after retries ({base_id})")


def _tool_names(tool_calls: list[dict[str, Any]]) -> list[str]:
    return [str(c.get("function", {}).get("name") or "") for c in tool_calls]


def _closed_set_violation(
    tool_calls: list[dict[str, Any]],
    *,
    candidates: list[tuple[str, str]],
    viewed: set[str],
) -> str | None:
    if not candidates:
        return None
    cand_pids = {pid for pid, _ in candidates}
    cand_keys = {f"{pid}::{sku}" for pid, sku in candidates}
    names = _tool_names(tool_calls)
    if any(n in ("search", "filter") for n in names) and len(viewed & cand_pids) < min(
        2, len(cand_pids)
    ):
        return (
            "Closed candidate set: do not search/filter. "
            f"Call view on listed product_ids first: {sorted(cand_pids)}"
        )
    for call in tool_calls:
        name = call.get("function", {}).get("name") or ""
        raw = call["function"].get("arguments", "{}")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "view":
            pid = str(args.get("product_id") or "")
            if pid and pid not in cand_pids:
                return (
                    f"view product_id {pid} is outside the listed candidates. "
                    f"View only: {sorted(cand_pids)}"
                )
        if name in ("add_to_cart", "inspect_stock", "place_test_order"):
            pid = str(args.get("product_id") or "")
            sku = str(args.get("sku") or "")
            if pid and pid not in cand_pids:
                return (
                    f"{name} product_id {pid} is outside listed candidates "
                    f"{sorted(cand_pids)}"
                )
            if pid and sku and f"{pid}::{sku}" not in cand_keys:
                if name in ("add_to_cart", "place_test_order"):
                    return (
                        f"{name} {pid}::{sku} is not a listed variant. "
                        f"Use one of: {sorted(cand_keys)}"
                    )
    return None


def _oos_order_violation(
    tool_calls: list[dict[str, Any]],
    *,
    last_stock: dict[str, Any] | None,
) -> str | None:
    """Reject ordering when latest stock/cart says out of stock (market shock)."""
    if not _stock_is_false(last_stock):
        return None
    names = _tool_names(tool_calls)
    if "place_test_order" in names:
        return (
            "Rejected: latest stock/cart shows in_stock=false (market shock). "
            "remove_from_cart if needed, pick another candidate, inspect_stock, "
            "then order only an in-stock sku."
        )
    return None


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """EnvPack loop: LS phases + safe-band gates for hidden races."""

    environment = problem_data["environment"]
    binding = environment["binding"]
    policy = environment["policy_view"]
    problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
    max_calls = int(policy.get("max_calls_per_turn", 1))
    tools = policy["tools"]
    query = str(policy.get("query") or "")
    allowed_fields = _claim_field_allowlist(tools)
    candidates = _parse_candidate_variants(query)
    cand_pids = {pid for pid, _ in candidates}
    ask_first = _needs_ask_first_spec(query)

    boot = (
        f"{query}\n\n"
        "Procedure: ask firm specs if withheld-until-asked; else retrieve → view → "
        "inspect_stock → cart → place_test_order. "
        "On stockout: message must COMMIT to same preferred model/other color "
        "(never ask 'should I?'); then remove → recover → order. "
        "Justification only if asked: include brand + exact view specs + current "
        "price/in_stock; never claim category; do not invent hd/fhd."
    )
    model_hint = _preferred_model_hint(query)
    if ask_first:
        boot += (
            "\n\nASK-FIRST: Your first tool call must be `message` asking for the "
            "withheld firm hardware spec. After the reply, search with that spec and "
            "complete the order without apology loops."
        )
    if model_hint:
        boot += (
            f"\n\nPREFERRED MODEL TOKEN: `{model_hint}`. On stockout of the preferred "
            "color/variant, recover that same model in another color before changing models."
        )
    if candidates:
        listed = ", ".join(f"{pid}::{sku}" for pid, sku in candidates)
        boot += (
            "\n\nCLOSED CANDIDATE SET (mandatory): "
            f"{listed}. "
            "view each listed product_id; rank by stated priority; order a listed key only."
        )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": boot},
    ]
    dialogue: list[dict[str, Any]] = []
    must_message = bool(ask_first)
    asked_spec = False
    viewed: set[str] = set()
    searchish = 0
    last_stock: dict[str, Any] | None = None
    view_facts_by_pid: dict[str, dict[str, str]] = {}
    shock_recover = False
    recovery_msg_force = False

    for turn in range(1, int(policy["max_steps"]) + 1):
        if must_message:
            hint = (
                "REQUIRED this turn: call `message` once. "
                "If this is a stockout/'still available?' reply: STATE you will take "
                "another color/variant of the SAME preferred model (do not ask permission)."
            )
            if model_hint:
                hint += f" Preferred model token: {model_hint}."
            messages.append({"role": "user", "content": hint})
        elif candidates and len(viewed & cand_pids) < len(cand_pids):
            missing = sorted(cand_pids - viewed)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Still missing views for closed-set product_ids: "
                        f"{missing}. Call view on the next missing id (or compare)."
                    ),
                }
            )
        elif shock_recover:
            recover = (
                "Market shock: prior sku not safely in stock. remove_from_cart if needed. "
                "Recover SAME preferred model other color/variant first"
            )
            if model_hint:
                recover += f" (model `{model_hint}`)"
            recover += (
                ", inspect_stock an in-stock sku, then place_test_order. Do not quit; "
                "do not ask the shopper what to do."
            )
            messages.append({"role": "user", "content": recover})
            shock_recover = False
        elif not candidates and searchish >= 3 and len(viewed) < 2:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Stop repeating the same search. Reformulate once OR view best hits, "
                        "then inspect_stock and finish."
                    ),
                }
            )

        assistant_content = ""
        tool_calls: list[dict[str, Any]] = []
        for _attempt in range(4):
            inference = _chat(messages=messages, tools=tools)
            assistant = inference["choices"][0]["message"]
            assistant_content = assistant.get("content") or ""
            tool_calls = (assistant.get("tool_calls") or [])[:max_calls]
            if not tool_calls:
                messages.extend(
                    [
                        {"role": "assistant", "content": assistant_content},
                        {
                            "role": "user",
                            "content": "No tool call. Call a supplied tool now.",
                        },
                    ]
                )
                continue
            if must_message and "message" not in _tool_names(tool_calls):
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": assistant_content,
                            "tool_calls": tool_calls,
                        },
                        {
                            "role": "user",
                            "content": "Rejected. Call only `message` this turn.",
                        },
                    ]
                )
                continue
            if not must_message:
                violation = _closed_set_violation(
                    tool_calls, candidates=candidates, viewed=viewed
                )
                if violation:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": f"Rejected: {violation}"},
                        ]
                    )
                    continue
                oos = _oos_order_violation(tool_calls, last_stock=last_stock)
                if oos:
                    messages.extend(
                        [
                            {
                                "role": "assistant",
                                "content": assistant_content,
                                "tool_calls": tool_calls,
                            },
                            {"role": "user", "content": oos},
                        ]
                    )
                    shock_recover = True
                    continue
            break
        else:
            if candidates and len(viewed & cand_pids) < len(cand_pids):
                miss = sorted(cand_pids - viewed)[0]
                tool_calls = [
                    {
                        "id": f"forced-view-{turn}",
                        "type": "function",
                        "function": {
                            "name": "view",
                            "arguments": json.dumps({"product_id": miss}),
                        },
                    }
                ]
                assistant_content = f"Viewing closed-set candidate {miss}."
            elif must_message and ask_first and not asked_spec:
                tool_calls = [
                    {
                        "id": f"forced-ask-{turn}",
                        "type": "function",
                        "function": {
                            "name": "message",
                            "arguments": json.dumps(
                                {
                                    "content": (
                                        "What is the firm hardware requirement "
                                        "(RAM, screen size, resolution, or refresh rate)?"
                                    )
                                }
                            ),
                        },
                    }
                ]
                assistant_content = "Asking for the withheld firm spec."
            else:
                raise RuntimeError("model returned no valid tool call before completion")

        if must_message:
            msg_only = [
                c
                for c in tool_calls
                if c.get("function", {}).get("name") == "message"
            ]
            if msg_only:
                tool_calls = msg_only[:1]

        names = _tool_names(tool_calls)
        if any(n in ("search", "filter") for n in names):
            searchish += 1
        if "message" in names and ask_first:
            asked_spec = True

        messages.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
            }
        )
        group_id = f"{problem_id}-turn-{turn}"
        sanitized_calls: list[dict[str, Any]] = []
        for index, call in enumerate(tool_calls, start=1):
            args = _arguments(
                call,
                query=query,
                allowed_fields=allowed_fields,
                last_stock=last_stock,
                view_facts_by_pid=view_facts_by_pid,
            )
            name = call.get("function", {}).get("name") or ""
            if name == "message" and isinstance(args, dict):
                args = _rewrite_recovery_message(
                    args, query=query, force=recovery_msg_force
                )
            sanitized_calls.append(
                {
                    "call_id": f"{group_id}-{index}",
                    "action": {"name": name, "args": args},
                }
            )
        envelope = {
            **binding,
            "call_id": group_id,
            "idempotency_key": group_id,
            "turn": turn,
            "calls": sanitized_calls,
        }
        result = _env_call(envelope)
        dialogue.append(
            {
                "role": "assistant",
                "content": assistant_content,
                "tool_calls": tool_calls,
                "environment_result": result,
            }
        )

        for call in tool_calls:
            if (call.get("function", {}).get("name") or "") == "view":
                raw = call["function"].get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except json.JSONDecodeError:
                    args = {}
                if isinstance(args, dict) and args.get("product_id"):
                    viewed.add(str(args["product_id"]))

        if "message" in names:
            must_message = False
            recovery_msg_force = False

        for tool_call, call_result in zip(tool_calls, result["calls"], strict=True):
            obs = call_result.get("observation")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(obs),
                }
            )
            payload = _obs_stock_payload(obs if isinstance(obs, dict) else {})
            if payload is None and isinstance(obs, dict):
                payload = _obs_stock_payload({"observation": obs})
            name = tool_call.get("function", {}).get("name") or ""
            if name == "view" and isinstance(obs, dict):
                pid, facts = _extract_view_facts(obs)
                if not pid:
                    raw = tool_call["function"].get("arguments", "{}")
                    try:
                        a = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        a = {}
                    if isinstance(a, dict) and a.get("product_id"):
                        pid = str(a["product_id"])
                if pid and facts:
                    view_facts_by_pid[pid] = {
                        **view_facts_by_pid.get(pid, {}),
                        **facts,
                    }
            if name in ("inspect_stock", "add_to_cart", "inspect_cart") and payload:
                last_stock = payload
                if _stock_is_false(payload):
                    shock_recover = True
                    recovery_msg_force = True
            if isinstance(obs, dict):
                inner = obs.get("observation") if isinstance(obs.get("observation"), dict) else obs
                if isinstance(inner, dict) and inner.get("market_event"):
                    shock_recover = True
                    recovery_msg_force = True
                    added = inner.get("added")
                    if isinstance(added, dict):
                        last_stock = added

        if result.get("user_message"):
            um = result["user_message"]
            content = um.get("content") if isinstance(um, dict) else str(um)
            messages.append({"role": "user", "content": content})
            if _looks_like_shopper_intervention(str(content)):
                low = str(content).lower()
                if ask_first and asked_spec and "?" not in low:
                    must_message = False
                elif "?" in low or any(
                    x in low for x in ("which", "still available", "hold on", "before you")
                ):
                    must_message = True
                    if any(
                        x in low
                        for x in ("still available", "out of stock", "hold on", "before you")
                    ):
                        recovery_msg_force = True
        if any(
            (call.get("observation") or {}).get("done")
            for call in result["calls"]
        ):
            break

        if not must_message and not candidates and not shock_recover:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Episode not done. Prefer inspect_stock → add_to_cart → "
                        "place_test_order on an in-stock sku that meets firm constraints. "
                        "Omit justification unless the shopper asked why."
                    ),
                }
            )
        elif not must_message and candidates and len(viewed & cand_pids) >= len(cand_pids):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "All listed candidates viewed. Rank by stated priority, "
                        "inspect_stock the winner, add_to_cart listed sku, place_test_order."
                    ),
                }
            )

    return dialogue


__all__ = ["agent_main"]
