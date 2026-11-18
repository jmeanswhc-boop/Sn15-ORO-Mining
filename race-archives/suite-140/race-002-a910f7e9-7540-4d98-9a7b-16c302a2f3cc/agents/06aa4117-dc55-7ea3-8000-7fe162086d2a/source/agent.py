"""Configurable ORO-Bench shopping agent.

Same controller as agent.py: the model explores, a deterministic buyer
commits. Every former constant — models, thresholds, family playbooks,
spec ontology, query grammar, tool/observation names — lives in
DEFAULT_POLICY and can be overlaid at runtime without editing this file.

Overlay sources, later wins:
  1. built-in DEFAULT_POLICY
  2. problem_data["agent_policy"] or environment["agent_policy"]
  3. AGENT_POLICY_JSON (JSON object)
  4. SANDBOX_MODEL / SANDBOX_MODELS / INFERENCE_PROVIDER / AGENT_* knobs
  5. Agent(..., policy={...}) constructor argument

Add a family, spec, qualifier, or identity rule by merging a dict. New
rank algorithms still need a rank_mode the controller already implements.
"""

from __future__ import annotations

import json
import re
import time
from copy import deepcopy
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient


_EPS = 1e-9
_NUM = r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?"

# ---------------------------------------------------------------------------
# Default policy. This is the whole formerly hard-coded surface.
# ---------------------------------------------------------------------------


def default_policy() -> dict:
    return {
        "runtime": {
            "problem_budget_s": 270.0,
            "obs_chars": 7000,
            "max_reprompts": 2,
            "proxy_timeout": 110.0,
            "proxy_retries": 2,
            "proxy_retry_delay": 1.0,
            "default_max_steps": 30,
            "temperature": 0,
            "compare_batch": 5,
            "search_k": 10,
            "forced_compares_cap": 8,
            "panic_remaining_s": 25.0,
            "panic_turns_left": 3,
            "roomy_turn_frac": 0.45,
            "roomy_clock_s": 140.0,
            "sealed_unseal_frac": 0.6,
            "wait_shopper_turns": 4,
            "pref_wait_frac": 0.6,
            "question_force_after": 2,
            "max_questions": 2,
            "max_cart_inspections": 1,
            "fallback_products": 2,
            "recall_early_margin": 2,
            "recall_early_searches": 8,
            "recall_min_searches": 2,
            "title_search_words": 8,
            "shortlist_n": 6,
            "note_clip": 300,
            "repeat_abort": 2,
            "inference_attempts": 3,
            "env_attempts": 4,
            "late_pick_turns": 8,
            "rank_identity_category": "smartphone",
            "default_currency": "PHP",
            "currencies": ["PHP", "USD", "EUR", "SGD", "MYR", "IDR", "THB", "VND"],
            "default_provider": "chutes",
            "core_rules": (
                "You are a careful shopping agent working a catalog through tools. Exactly one tool call "
                "per turn, at most {max_steps} turns, so be efficient. A deterministic controller checks "
                "every proposal and takes over the cart and the order once a candidate is fully verified, "
                "so your job is to surface and read candidates: search or filter, then view or compare so "
                "their raw facts, variant options and prices are on record. Never add an item to the cart "
                "as a placeholder - the first cart addition is graded. Prices and stock change while you "
                "shop; the controller re-checks the carted item and tells you when to move to a different "
                "product. Answer every direct question from the shopper with one short message before "
                "anything else, and never send progress or status updates. Never ask the shopper to raise "
                "the budget. Do not repeat a tool call with identical arguments; one compare of up to "
                "{compare_batch} products beats several view calls."
            ),
            "sealed_question": (
                "Before I search: which hardware specification is your firm requirement "
                "(screen size, resolution class, refresh rate, or memory), and what value do you need?"
            ),
            "resale_pattern": (
                r"customer (?:rating|review)|\bcompare \b|refurb|pre-owned|restored|renewed"
            ),
            "wish_stopwords": [
                "the", "and", "for", "with", "that", "this", "you", "your", "find", "need", "want",
                "order", "stock", "place", "under", "over", "one", "php", "test", "buy", "rather",
                "than", "pay", "bit", "more", "settle", "something", "its", "would", "like", "have",
            ],
        },
        "models": {
            "chutes": ["deepseek-ai/DeepSeek-V3.2-TEE", "Qwen/Qwen3-32B-TEE"],
            "openrouter": ["deepseek/deepseek-v3.2", "qwen/qwen3-32b"],
        },
        "tools": {
            "search": "search",
            "filter": "filter",
            "view": "view",
            "compare": "compare",
            "inspect_stock": "inspect_stock",
            "inspect_cart": "inspect_cart",
            "add_to_cart": "add_to_cart",
            "remove_from_cart": "remove_from_cart",
            "message": "message",
            "place_test_order": "place_test_order",
        },
        "obs": {
            "results": "results",
            "compared": "compared",
            "added": "added",
            "removed": "removed",
            "cart": "cart",
            "product_id": "product_id",
            "sku": "sku",
            "title": "title",
            "brand": "brand",
            "price": "price",
            "in_stock": "in_stock",
            "facts": "facts",
            "variants": "variants",
            "options": "options",
            "listing_id": "listing_id",
            "category_path": "category_path",
            "current_price": "current_price",
            "current_in_stock": "current_in_stock",
            "error": "error",
            "done": "done",
        },
        "grammar": {
            "budget": (
                r"(?:at or under|at or below|up to|under|below|no more than|not more than|"
                r"max(?:imum)? of|within|budget (?:is|of)|only go up to|can spend up to|at most)\s*"
                + _NUM
                + r"\s*(PHP|php|₱|pesos?)?"
            ),
            "key": r"(\d{6,})::([A-Za-z0-9._\-]+)",
            "quoted": r"\"([^\"]{2,80})\"",
            "phrase": r"matches (?:the phrase )?\"?([A-Za-z0-9][^\",]{1,80}?)\"?\s*,",
            "surface": r"surface at least (\d+) distinct",
            "pref_item": r"prefer the ([A-Za-z]+) ([A-Za-z0-9][A-Za-z0-9\-]*\d[A-Za-z0-9\-]*)\b",
            "pref_brand": (
                r"\bprefer(?:ably)?\s+(?:the\s+)?([A-Za-z][\w&.'\- ]{1,38}?)\s*"
                r"(?:,|\.|;|\bbut\b|\bwhen\b|\bif\b|$)"
            ),
            "category": r"in the ([A-Za-z][\w&'\- ]{1,48}?) category",
            "loose_category": (
                r"\b([a-z][a-z\-]{2,20})\s*(?:,|\s(?:at or (?:under|below)|at most|under|"
                r"below|within|with)\b)"
            ),
            "spec_word": (
                r"\b(?:full|ultra)\s+hd\b|\b(?:hd|fhd|qhd|uhd|2k|4k|\d+p)\b|"
                r"\d+(?:\.\d+)?\s*-?\s*(?:inch(?:es)?|in|hz|gb)\b"
            ),
            "reason": r"\b(?:why|reason|explain|justif)",
            "question": (
                r"\?|\b(let me know|would you (?:like|prefer)|do you want|should i|"
                r"which (?:one|option)|tell me (?:which|whether|if)|confirm whether)\b"
            ),
            "firm_requirement": r"firm requirement",
            "sealed_hint": r"firm requirement",
        },
        "grammar_flags": {
            "key": "",
        },
        "classifier": [
            {"family": "ranking", "when": "has_candidate_keys"},
            {"family": "recall", "when": "has_required_identities"},
            {"family": "recovery", "when": "has_pref_model"},
            {"family": "preference", "when": "rules_contain", "text": "listing_id"},
            {"family": "constraint", "when": "rules_contain", "text": "exact variant"},
            {"family": "justification", "when": "rules_contain", "text": "trustworthy reason"},
            {"family": "intent", "when": "rules_contain", "text": "nice-to-haves"},
            {"family": "recall", "when": "has_theme"},
            {"family": "justification", "when": "reason_in_goal"},
            {"family": "intent", "when": "has_pref_brand"},
            {"family": "constraint", "when": "spec_or_brand"},
            {"family": "preference", "when": "always"},
        ],
        "specs": [
            {"name": "screen_size", "fact_label": "size", "unit": "in", "extractor": "inch"},
            {"name": "refresh_rate", "fact_label": "refresh_rate", "unit": "hz", "extractor": "hz"},
            {"name": "resolution", "fact_label": "display", "unit": None, "extractor": "resolution"},
            {"name": "ram", "fact_label": "storage", "unit": "gb", "extractor": "memory"},
        ],
        "adopt_order": ["screen_size", "refresh_rate", "ram", "resolution"],
        "qualifiers": [
            {
                "name": "gaming",
                "goal_pattern": r"\bgaming\s+monitors?\b",
                "title_all": ["monitor"],
                "title_any": ["gaming"],
                "or_hz_at_least": 120,
            },
            {
                "name": "over_ear",
                "goal_pattern": r"\bover[\s-]?ear\b",
                "title_all": [],
                "title_any": ["over ear", "overear", "circumaural"],
                "token_pairs": [["over", "ear"]],
            },
        ],
        "identities": [
            {
                "category": "smartphone",
                "alias_pattern": r"(?:smart|cell|mobile|unlocked)?\s*phones?",
                "part_only": (
                    r"\bonly\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b|"
                    r"\b(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\s+only\b|"
                    r"^\s*replacement\s+(?:(?:lcd|oled|amoled)\s+)?(?:screen|display)\b"
                ),
                "not_device": (
                    r"\btablets?\b|\b(?:feature|basic)\s*phones?\b|"
                    r"\b(?:mystery|blind|lucky)\s*box(?:es)?\b|"
                    r"\b(?:accessor(?:y|ies)|screen\s*protectors?|phone\s*cases?|"
                    r"case\s*(?:cover\s*)?for|replacement\s*(?:screen|battery)|"
                    r"smart\s*watches?|earbuds?|headphones?)\b"
                ),
                "is_device": r"\bsmart\s*phones?\b",
                "os": (
                    r"\b(?:android(?:\s*\d+(?:\.\d+)*)?|ios|blackberry(?:\s*os)?|"
                    r"windows\s*phone|symbian|harmony\s*os|hyper\s*os|sailfish|"
                    r"ubuntu\s*touch|tizen)\b"
                ),
                "negated": (
                    r"\bnot\s+(?:an?\s+)?(?:smart\s*phones?|feature\s*phones?|"
                    r"basic\s*phones?|tablets?|accessor(?:y|ies))\b"
                ),
                "bundle": (
                    r"\b(?:(?:with\s+(?:free|bundled)|bundled\s+with|free)\s+|"
                    r"(?:with|includes?)\s+(?:an?\s+)?(?=(?:headphones?|earbuds?|"
                    r"phone\s*case|screen\s*protector|charger|accessor(?:y|ies))\b))"
                ),
                "kind_keys": ["phone_type", "device_type", "description_device_type"],
                "os_keys": ["device_operating_system", "description_operating_system"],
                "truncated_keys": ["phone_type", "device_type", "device_operating_system"],
                "compat_pattern": r"\bcompatib",
                "strict_families": ["constraint", "justification", "recovery"],
            }
        ],
        "extractors": {
            "inch": {"pattern": r"(\d+(?:\.\d+)?)\s*[-–]?\s*(?:(?:inch(?:es)?|in)\b|'{2}|[\"“”])"},
            "hz": {"pattern": r"(\d+(?:\.\d+)?)\s*hz"},
            "gb": {"pattern": r"(\d+(?:\.\d+)?)\s*(tb|gb)\b"},
            "pixels": {"pattern": r"(\d{3,5})\s*[x×]\s*(\d{3,5})"},
            "mem_pair": {"pattern": r"(?<!\d)(\d+(?:\.\d+)?)\s*(?:g(?:b)?)?\s*\+\s*(\d+(?:\.\d+)?)\s*g(?:b)?\b"},
            "mem_pair_hint": {"pattern": r"(?<!\d)\d+(?:\.\d+)?\s*(?:g(?:b)?)?\s*\+\s*\d+(?:\.\d+)?(?:\s*g(?:b)?)?(?!\d)"},
            "mem_named": {"pattern": r"(\d+(?:\.\d+)?)\s*gb\s*(?:ram|memory|unified memory)\b"},
            "quarter": {"pattern": r"\bqHD\b", "flags": ""},
        },
        "resolution_classes": [
            {"name": "4k", "long": 3840, "short": 2160, "words": r"\b(?:4k|uhd|ultra hd|2160p?)\b"},
            {"name": "qhd", "long": 2560, "short": 1440, "words": r"\b(?:qhd|1440p?|2k)\b"},
            {"name": "fhd", "long": 1920, "short": 1080, "words": r"\b(?:fhd|1080p?|full ?hd)\b"},
            {"name": "hd", "long": 1280, "short": 720, "words": r"\b(?:hd|720p?)\b"},
        ],
        "families": {
            "intent": {
                "playbook": (
                    "Firm requirements: the category, the named hardware specification, the budget and stock. A "
                    "named brand is only a preference: choose it when a fully compliant item of that brand exists, "
                    "otherwise any compliant brand. Use filter(category, brand, max_price, in_stock=true), then view "
                    "or compare to read the exact variant's specification from raw facts. Answer the shopper with "
                    "message; a budget the shopper lowers later applies to every later choice."
                ),
                "ask_sealed": False,
                "opening_question": None,
                "wait_shopper": True,
                "pref_brand_first": True,
                "rank_mode": "pref_brand",
                "kill_product_on_death": True,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": False,
                "require_named_keys": False,
                "require_exact_first": False,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": False,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": False,
                "model_picks": False,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": "Choose a different product that meets every firm requirement.",
            },
            "recall": {
                "playbook": (
                    "Run several differently worded searches (k=10, in_stock=true) built from the requested words and "
                    "close synonyms until enough distinct matching products have appeared; every requested word must "
                    "appear as a whole word of the title. Then view the strongest candidates so the controller can "
                    "verify one and buy it."
                ),
                "ask_sealed": False,
                "opening_question": None,
                "wait_shopper": False,
                "pref_brand_first": False,
                "rank_mode": "price",
                "kill_product_on_death": True,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": True,
                "require_named_keys": False,
                "require_exact_first": False,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": True,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": False,
                "model_picks": False,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": "Choose a different product that meets every firm requirement.",
            },
            "constraint": {
                "playbook": (
                    "The shopper hides one firm hardware requirement and answers it when asked; obey it exactly for "
                    "the specific variant, reading the variant's own option text before any product-wide value. Use "
                    "filter(category, brand, max_price, in_stock=true) and view candidates. If the carted variant "
                    "sells out, move to a different product that still meets every requirement."
                ),
                "ask_sealed": True,
                "opening_question": None,
                "wait_shopper": True,
                "pref_brand_first": False,
                "rank_mode": "price",
                "kill_product_on_death": True,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": False,
                "require_named_keys": False,
                "require_exact_first": False,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": False,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": False,
                "model_picks": False,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": (
                    "Choose a different product that still meets every requirement, not another variant of the same one."
                ),
            },
            "preference": {
                "playbook": (
                    "You already asked which preference matters most; use the answer. Search or filter the category "
                    "within budget, then compare candidates with different listing_id values, reading raw facts. The "
                    "first cart item is graded, so make it the best genuine match. If it is repriced above budget, "
                    "recover to a different listing_id."
                ),
                "ask_sealed": False,
                "opening_question": (
                    "Quick question so I choose well: which of your preferences counts most, and in what "
                    "order should I weigh the others? Let me know your priority."
                ),
                "wait_shopper": False,
                "pref_brand_first": False,
                "rank_mode": "priority",
                "kill_product_on_death": True,
                "recover_different_listing": True,
                "require_listing_diversity": 2,
                "require_identities": False,
                "require_named_keys": False,
                "require_exact_first": False,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": False,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": False,
                "model_picks": True,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": (
                    "Recover to a product with a different listing_id that still fits the shopper's priorities."
                ),
            },
            "ranking": {
                "playbook": (
                    "Only the named candidate variants count, and the controller ranks them for you: compare all of "
                    "them first so their facts are on the record. Highest advertised maximum refresh rate wins, then "
                    "the largest screen, then the lowest price, reading the selected variant's own value before a "
                    "product-wide one. After the market moves, the remaining candidates are re-ranked."
                ),
                "ask_sealed": False,
                "opening_question": None,
                "wait_shopper": False,
                "pref_brand_first": False,
                "rank_mode": "refresh_screen_price",
                "rank_fields": ["refresh_rate", "screen_size"],
                "kill_product_on_death": True,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": False,
                "require_named_keys": True,
                "require_exact_first": False,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": False,
                "require_measurable": ["refresh_rate", "screen_size"],
                "compare_named_first": True,
                "sibling_search": False,
                "model_picks": False,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": "The remaining named candidates are re-ranked; the controller takes the new best.",
            },
            "recovery": {
                "playbook": (
                    "Find the preferred model and color first: search the model code, then view it so every variant "
                    "and color is on the record. The first cart item must be exactly that model and color. If it "
                    "disappears, relax as little as possible - the same model in another color before any other "
                    "model - and never relax the category, budget or stock."
                ),
                "ask_sealed": False,
                "opening_question": None,
                "wait_shopper": False,
                "pref_brand_first": False,
                "rank_mode": "relax",
                "kill_product_on_death": False,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": False,
                "require_named_keys": False,
                "require_exact_first": True,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": False,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": True,
                "model_picks": False,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": (
                    "Recover with the smallest possible change: the same model in another color before any other model."
                ),
            },
            "justification": {
                "playbook": (
                    "Find a product whose own raw facts state the requested specification, in the requested category "
                    "and brand, in stock and within budget. View or compare it so its facts are on the record, then "
                    "the controller inspects it, carts it and orders it with claims taken only from that evidence."
                ),
                "ask_sealed": False,
                "opening_question": None,
                "wait_shopper": False,
                "pref_brand_first": False,
                "rank_mode": "price",
                "kill_product_on_death": True,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": False,
                "require_named_keys": False,
                "require_exact_first": False,
                "require_fact_match": True,
                "ban_resale_titles": True,
                "require_theme": False,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": False,
                "model_picks": False,
                "attach_claims": True,
                "claims_fallback_field": "resolution",
                "panic_strict": True,
                "recovery_guidance": "Choose a different product that meets every firm requirement.",
            },
            "generic": {
                "playbook": (
                    "Understand the firm requirements, filter or search the catalog, view candidates, and let the "
                    "controller verify and buy the best one."
                ),
                "ask_sealed": False,
                "opening_question": None,
                "wait_shopper": False,
                "pref_brand_first": False,
                "rank_mode": "price",
                "kill_product_on_death": True,
                "recover_different_listing": False,
                "require_listing_diversity": 0,
                "require_identities": False,
                "require_named_keys": False,
                "require_exact_first": False,
                "require_fact_match": False,
                "ban_resale_titles": False,
                "require_theme": False,
                "require_measurable": [],
                "compare_named_first": False,
                "sibling_search": False,
                "model_picks": False,
                "attach_claims": False,
                "claims_fallback_field": None,
                "panic_strict": False,
                "recovery_guidance": "Choose a different product that meets every firm requirement.",
            },
        },
    }


DEFAULT_POLICY = default_policy()


def _merge(base: Any, overlay: Any) -> Any:
    """Deep-merge overlay into base. Dicts merge; lists of named dicts merge by name; other lists replace."""
    if overlay is None:
        return deepcopy(base)
    if isinstance(base, dict) and isinstance(overlay, dict):
        out = dict(base)
        for key, val in overlay.items():
            if key == "_replace":
                continue
            if key in out and isinstance(out[key], (dict, list)) and isinstance(val, (dict, list)):
                out[key] = _merge(out[key], val)
            else:
                out[key] = deepcopy(val)
        if overlay.get("_replace") is True:
            return deepcopy(overlay)
        return out
    if isinstance(base, list) and isinstance(overlay, list):
        if overlay and all(isinstance(x, dict) and "name" in x for x in overlay) and all(
            isinstance(x, dict) and "name" in x for x in base
        ):
            by_name = {str(x["name"]): deepcopy(x) for x in base}
            order = [str(x["name"]) for x in base]
            for item in overlay:
                name = str(item["name"])
                if name in by_name:
                    by_name[name] = _merge(by_name[name], item)
                else:
                    by_name[name] = deepcopy(item)
                    order.append(name)
            return [by_name[n] for n in order]
        return deepcopy(overlay)
    return deepcopy(overlay)


def _flags(text: str | None) -> int:
    if text is None:
        text = "I"
    flags = 0
    for ch in text:
        flags |= getattr(re, ch, 0) if ch.isalpha() else 0
    return flags


def _rx(pattern: str, flags: str | None = "I") -> re.Pattern:
    return re.compile(pattern, _flags(flags))


def _env_overlay() -> dict:
    out: dict = {}
    blob = (getenv("AGENT_POLICY_JSON") or "").strip()
    if blob:
        try:
            parsed = json.loads(blob)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            out = _merge(out, parsed)
    runtime: dict = {}
    for env_key, path in (
        ("AGENT_BUDGET_S", ("runtime", "problem_budget_s", float)),
        ("AGENT_OBS_CHARS", ("runtime", "obs_chars", int)),
        ("AGENT_MAX_REPROMPTS", ("runtime", "max_reprompts", int)),
        ("AGENT_TEMPERATURE", ("runtime", "temperature", float)),
        ("AGENT_COMPARE_BATCH", ("runtime", "compare_batch", int)),
        ("AGENT_SEARCH_K", ("runtime", "search_k", int)),
        ("AGENT_DEFAULT_CURRENCY", ("runtime", "default_currency", str)),
    ):
        raw = getenv(env_key)
        if raw is None or raw == "":
            continue
        _, field, caster = path
        try:
            runtime[field] = caster(raw)
        except (TypeError, ValueError):
            continue
    if runtime:
        out = _merge(out, {"runtime": runtime})
    return out


def _problem_overlays(problem: dict | None) -> list:
    found = []
    env = (problem or {}).get("environment") if isinstance(problem, dict) else None
    env = env if isinstance(env, dict) else {}
    for src in (problem, env, env.get("policy_view") if isinstance(env.get("policy_view"), dict) else None):
        if not isinstance(src, dict):
            continue
        for key in ("agent_policy", "agent_config", "policy_overlay"):
            blob = src.get(key)
            if isinstance(blob, dict):
                found.append(blob)
            elif isinstance(blob, str) and blob.strip():
                try:
                    parsed = json.loads(blob)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    found.append(parsed)
    return found


def load_policy(problem: dict | None = None, extra: dict | None = None) -> "Policy":
    raw = default_policy()
    for overlay in _problem_overlays(problem):
        raw = _merge(raw, overlay)
    raw = _merge(raw, _env_overlay())
    if extra:
        raw = _merge(raw, extra)
    return Policy(raw)


class Policy:
    """Compiled, read-only view of the merged policy dict."""

    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.runtime = raw["runtime"]
        self.models = raw["models"]
        self.tools = raw["tools"]
        self.obs = raw["obs"]
        self.families = raw["families"]
        self.classifier = list(raw.get("classifier") or [])
        self.adopt_order = list(raw.get("adopt_order") or [])
        gflags = raw.get("grammar_flags") or {}
        self.rx = {}
        for name, pattern in (raw.get("grammar") or {}).items():
            try:
                self.rx[name] = _rx(pattern, gflags.get(name, "I"))
            except re.error:
                continue
        self.resale_re = _rx(self.runtime.get("resale_pattern") or r"refurb")
        self.extractors = {}
        for name, spec in (raw.get("extractors") or {}).items():
            try:
                self.extractors[name] = _rx(spec["pattern"], spec.get("flags", "I"))
            except (KeyError, re.error):
                continue
        self.res_classes = []
        for row in raw.get("resolution_classes") or []:
            item = dict(row)
            item["words_re"] = _rx(row["words"])
            self.res_classes.append(item)
        self.specs = [dict(s) for s in (raw.get("specs") or [])]
        self.spec_by_name = {s["name"]: s for s in self.specs}
        self.qualifiers = []
        for row in raw.get("qualifiers") or []:
            item = dict(row)
            item["goal_re"] = _rx(item["goal_pattern"])
            self.qualifiers.append(item)
        self.identities = []
        for row in raw.get("identities") or []:
            item = dict(row)
            for key in ("alias_pattern", "part_only", "not_device", "is_device", "os", "negated", "bundle", "compat_pattern"):
                if item.get(key):
                    item[key + "_re"] = _rx(item[key], "I")
            self.identities.append(item)

    def family(self, name: str | None) -> dict:
        fam = self.families.get(name or "") or self.families.get("generic") or {}
        return dict(fam)

    def tool(self, role: str) -> str:
        return self.tools.get(role) or role

    def field(self, name: str) -> str:
        return self.obs.get(name) or name

    def spec_names(self) -> list:
        return [s["name"] for s in self.specs]


# ---------------------------------------------------------------------------
# Spec / identity extractors (patterns come from Policy)
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _same_word(a: str, b: str) -> bool:
    x = re.sub(r"[^a-z0-9]+", "", (a or "").lower()).rstrip("s")
    y = re.sub(r"[^a-z0-9]+", "", (b or "").lower()).rstrip("s")
    return bool(x) and x == y


def _one(values: set) -> str | None:
    return next(iter(values)) if len(values) == 1 else None


def _inch_vals(policy: Policy, text: str) -> set:
    rx = policy.extractors.get("inch")
    if not rx:
        return set()
    return {"%gin" % float(m.group(1)) for m in rx.finditer(text or "")}


def _hz_vals(policy: Policy, text: str) -> set:
    rx = policy.extractors.get("hz")
    if not rx:
        return set()
    return {"%ghz" % float(m.group(1)) for m in rx.finditer(text or "")}


def _gb_vals(policy: Policy, text: str) -> set:
    rx = policy.extractors.get("gb")
    if not rx:
        return set()
    return {
        "%ggb" % (float(m.group(1)) * (1024 if m.group(2).lower() == "tb" else 1))
        for m in rx.finditer(text or "")
    }


def _pixel_class(policy: Policy, long_side: int, short_side: int) -> str | None:
    for row in policy.res_classes:
        if long_side >= int(row["long"]) and short_side >= int(row["short"]):
            return row["name"]
    return None


def _res_vals(policy: Policy, text: str) -> set:
    text = text or ""
    quarter = policy.extractors.get("quarter")
    if quarter and quarter.search(text):
        return set()
    low = text.lower()
    out = set()
    px = policy.extractors.get("pixels")
    if px:
        for m in px.finditer(low):
            pair = sorted((int(m.group(1)), int(m.group(2))), reverse=True)
            cls = _pixel_class(policy, pair[0], pair[1])
            if cls:
                out.add(cls)
    for row in policy.res_classes:
        words = row.get("words_re")
        blob = low
        if row["name"] == "hd":
            blob = re.sub(r"\b(?:full|ultra) hd\b", "", low)
        if words and words.search(blob):
            out.add(row["name"])
    return out


def _number_unit_vals(text: str, unit_pattern: str, unit: str) -> set:
    rx = _rx(r"(\d+(?:\.\d+)?)\s*(?:" + unit_pattern + r")")
    return {"%g" % float(m.group(1)) + unit for m in rx.finditer(text or "")}


def _spec_vals(policy: Policy, field: str, text: str) -> set:
    spec = policy.spec_by_name.get(field) or {}
    kind = spec.get("extractor")
    if kind == "inch":
        return _inch_vals(policy, text)
    if kind == "hz":
        return _hz_vals(policy, text)
    if kind == "resolution":
        return _res_vals(policy, text)
    if kind == "memory":
        return _gb_vals(policy, text)
    if kind == "number_unit" and spec.get("unit_pattern") and spec.get("unit"):
        return _number_unit_vals(text, spec["unit_pattern"], spec["unit"])
    return set()


def _adjacent_vals(policy: Policy, field: str, text: str) -> set:
    spec = policy.spec_by_name.get(field) or {}
    text = text or ""
    if spec.get("extractor") != "memory":
        return _spec_vals(policy, field, text)
    pair = policy.extractors.get("mem_pair")
    named = policy.extractors.get("mem_named")
    paired = {"%ggb" % float(a) for a, _b in (pair.findall(text) if pair else [])}
    return paired or {"%ggb" % float(n) for n in (named.findall(text) if named else [])}


def _mentions(policy: Policy, field: str, text: str) -> bool:
    spec = policy.spec_by_name.get(field) or {}
    text = text or ""
    kind = spec.get("extractor")
    if kind == "memory":
        hint = policy.extractors.get("mem_pair_hint")
        named = policy.extractors.get("mem_named")
        return bool((hint and hint.search(text)) or (named and named.search(text)))
    if kind == "inch":
        rx = policy.extractors.get("inch")
        return bool(rx and rx.search(text))
    if kind == "hz":
        rx = policy.extractors.get("hz")
        return bool(rx and rx.search(text))
    if kind == "resolution":
        px = policy.extractors.get("pixels")
        if px and px.search(text):
            return True
        return any(row["words_re"].search(text) for row in policy.res_classes)
    if kind == "number_unit" and spec.get("unit_pattern"):
        return bool(_rx(r"(\d+(?:\.\d+)?)\s*(?:" + spec["unit_pattern"] + r")").search(text))
    return bool(_spec_vals(policy, field, text))


def _wanted_value(policy: Policy, field: str | None, text: str) -> str | None:
    if not field:
        return None
    return _one(_spec_vals(policy, field, text or ""))


def _identity_text(rule: dict, text: str, title: bool = False) -> str:
    text = (rule.get("negated_re").sub("", text or "") if rule.get("negated_re") else (text or ""))
    bundle = rule.get("bundle_re")
    return bundle.split(text, maxsplit=1)[0] if title and bundle else text


def _device_identity(rule: dict, facts: dict, title: str, options: str) -> str:
    part = rule.get("part_only_re")
    if part and part.search(options or ""):
        return "no"
    facts = facts if isinstance(facts, dict) else {}
    raw_title = str(title or "")
    head = _identity_text(rule, raw_title, title=True)
    kind = " ".join(_identity_text(rule, str(facts.get(k) or "")) for k in (rule.get("kind_keys") or []))
    blob = head + " " + kind
    if rule.get("not_device_re") and rule["not_device_re"].search(blob):
        return "no"
    trunc = rule.get("truncated_keys") or (rule.get("kind_keys") or []) + (rule.get("os_keys") or [])
    if any("[truncated]" in str(facts.get(k) or "") for k in trunc):
        return "unclear"
    if rule.get("is_device_re") and rule["is_device_re"].search(blob):
        return "yes"
    compat = rule.get("compat_pattern_re")
    os_blob = " ".join(
        _identity_text(rule, str(facts.get(k) or ""))
        for k in (rule.get("os_keys") or [])
        if not (compat and compat.search(str(facts.get(k) or "")))
    )
    os_re = rule.get("os_re")
    if os_re and os_re.search(os_blob):
        return "yes"
    if os_re and os_re.search(head) and not (compat and compat.search(raw_title)):
        return "yes"
    return "unclear"


def _identity_for(policy: Policy, category: str | None) -> dict | None:
    for rule in policy.identities:
        if _same_word(category or "", rule.get("category") or ""):
            return rule
    return None


# ---------------------------------------------------------------------------
# Brief / Ledger
# ---------------------------------------------------------------------------


class Clock:
    def __init__(self, budget: float) -> None:
        self.start = time.monotonic()
        self.budget = budget

    def remaining(self) -> float:
        return self.budget - (time.monotonic() - self.start)


class Brief:
    """What the deterministic side needs, parsed from the public request text."""

    def __init__(self, query: str, policy: Policy) -> None:
        self.policy = policy
        self.raw = query or ""
        goal, _, rules = self.raw.partition("\n\nTask rules:")
        self.goal = goal.strip()
        self.rules = rules.lower()
        self.budget = self._budget(self.goal)
        rt = policy.runtime
        self.currency = rt.get("default_currency") or "PHP"
        m = re.search(r"\b([A-Z]{3})\b", self.goal)
        allowed = set(rt.get("currencies") or [])
        if m and m.group(1) in allowed:
            self.currency = m.group(1)
        key_rx = policy.rx.get("key")
        self.candidate_keys = [(pid, sku.rstrip(".,;:")) for pid, sku in (key_rx.findall(self.goal) if key_rx else [])]
        phrase_rx, quoted_rx = policy.rx.get("phrase"), policy.rx.get("quoted")
        phrase = phrase_rx.search(self.goal) if phrase_rx else None
        quoted = quoted_rx.findall(self.goal) if quoted_rx else []
        source = phrase.group(1) if phrase else (quoted[0] if quoted else "")
        self.theme = re.findall(r"[a-z0-9]+", source.lower())
        sm = policy.rx["surface"].search(self.rules) if policy.rx.get("surface") else None
        self.required_identities = int(sm.group(1)) if sm else 0
        pm = policy.rx["pref_item"].search(self.goal) if policy.rx.get("pref_item") else None
        self.pref_color = pm.group(1).lower() if pm else None
        self.pref_model = pm.group(2) if pm else None
        self.category = self._category(self.goal)
        self.brand = self._brand(self.goal, self.category)
        self.pref_brand = None if pm else self._pref_brand(self.goal)
        self.cat_qualifier = self._qualifier(self.goal)
        self.specs = self._specs(self.goal)
        self.spec_field, self.spec_value = self._spec()
        sealed = policy.rx.get("firm_requirement")
        self.spec_sealed = self.spec_field is None and bool(sealed and sealed.search(self.goal.lower()))
        self.family = self._family()

    def _family(self) -> str:
        testers = {
            "has_candidate_keys": lambda: bool(self.candidate_keys),
            "has_required_identities": lambda: bool(self.required_identities),
            "has_pref_model": lambda: bool(self.pref_model),
            "has_theme": lambda: bool(self.theme),
            "has_pref_brand": lambda: bool(self.pref_brand),
            "reason_in_goal": lambda: bool(self.policy.rx.get("reason") and self.policy.rx["reason"].search(self.goal)),
            "spec_or_brand": lambda: bool(self.spec_sealed or self.spec_field or self.brand),
            "always": lambda: True,
        }
        for step in self.policy.classifier:
            when = step.get("when")
            if when == "rules_contain":
                if str(step.get("text") or "") in self.rules:
                    return step["family"]
                continue
            test = testers.get(when)
            if test and test():
                return step["family"]
        return "generic"

    def _budget(self, text: str) -> float | None:
        rx = self.policy.rx.get("budget")
        if not rx:
            return None
        best = None
        for m in rx.finditer(text or ""):
            whole = m.group(1).replace(",", "")
            frac = m.group(2) or "0"
            try:
                val = float(whole + "." + frac)
            except ValueError:
                continue
            if best is None or val < best:
                best = val
        return best

    def _category(self, goal: str) -> str | None:
        rx = self.policy.rx.get("category")
        if rx:
            m = rx.search(goal or "")
            if m:
                return m.group(1).strip()
        loose = self.policy.rx.get("loose_category")
        if loose:
            m = loose.search(goal or "")
            return m.group(1) if m else None
        return None

    def _brand(self, goal: str, category: str | None) -> str | None:
        if not category:
            return None
        spec_word = self.policy.rx.get("spec_word")
        m = re.search(r"\b(?:an?|the)\s+([^,.;]{0,60}?)\b" + re.escape(category), goal or "", re.I)
        if not m:
            return None
        blob = spec_word.sub(" ", m.group(1)) if spec_word else m.group(1)
        words = [w for w in re.findall(r"[\w&.'\-]+", blob) if w]
        keep = [w for w in words if w[:1].isupper() and not any(c.isdigit() for c in w) and not _res_vals(self.policy, w)]
        return " ".join(keep) or None

    def _pref_brand(self, goal: str) -> str | None:
        rx = self.policy.rx.get("pref_brand")
        if not rx:
            return None
        m = rx.search(goal or "")
        if not m:
            return None
        name = m.group(1).strip(" .,")
        words = name.split()
        if not words or len(words) > 4 or not any(w[:1].isupper() for w in words):
            return None
        return name

    def _qualifier(self, goal: str) -> str | None:
        for row in self.policy.qualifiers:
            if row["goal_re"].search(goal or ""):
                return row["name"]
        return None

    def _specs(self, goal: str) -> dict:
        out = {}
        for spec in self.policy.specs:
            value = _wanted_value(self.policy, spec["name"], goal)
            if value:
                out[spec["name"]] = value
        return out

    def _spec(self) -> tuple:
        for field, value in self.specs.items():
            return field, value
        return None, None

    def adopt_spec(self, text: str) -> str | None:
        order = list(self.policy.adopt_order or [])
        for name in self.policy.spec_names():
            if name not in order:
                order.append(name)
        for field in order:
            value = _wanted_value(self.policy, field, text)
            if value:
                self.spec_field, self.spec_value = field, value
                self.specs[field] = value
                self.spec_sealed = False
                return field + " = " + value
        return None


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


class Ledger:
    """What the agent has observed for itself, keyed by product and variant."""

    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.cart: list = []
        self.checked: dict = {}
        self.checked_seq: dict = {}
        self.inspected: set = set()
        self.products: dict = {}
        self.rows: dict = {}
        self.options: dict = {}
        self.listing_ids: set = set()
        self.listing_of: dict = {}
        self.identities: set = set()
        self.seq = 0
        self.first_add_seq: int | None = None
        self.last_add_seq: int | None = None
        self.adds = 0
        self.event_target: str | None = None
        self.dead_keys: set = set()
        self.dead_products: set = set()
        self.calls_seen: set = set()

    def _f(self, name: str) -> str:
        return self.policy.field(name)

    @staticmethod
    def key(pid: Any, sku: Any) -> str:
        return str(pid) + "::" + str(sku or "")

    def record(self, name: str, args: dict, obs: dict, brief: Brief) -> None:
        self.seq += 1
        if not isinstance(obs, dict) or obs.get(self._f("error")):
            return
        tools = self.policy.tools
        if name in (tools["search"], tools["filter"]):
            for row in obs.get(self._f("results")) or []:
                if isinstance(row, dict):
                    self._note_row(row, brief)
        elif name == tools["view"]:
            self._note_product(obs, brief)
        elif name == tools["compare"]:
            for prod in obs.get(self._f("compared")) or []:
                if isinstance(prod, dict):
                    self._note_product(prod, brief)
        elif name == tools["inspect_stock"]:
            k = self.key(obs.get(self._f("product_id")), obs.get(self._f("sku")))
            self.checked[k] = {
                "in_stock": bool(obs.get(self._f("in_stock"))),
                "price": _num(obs.get(self._f("price"))),
            }
            self.checked_seq[k] = self.seq
            self.inspected.add(k)
        elif name == tools["add_to_cart"]:
            added = obs.get(self._f("added")) or {}
            pid, sku = added.get(self._f("product_id")), added.get(self._f("sku"))
            if pid and sku:
                self.adds += 1
                if (pid, sku) not in self.cart:
                    self.cart.append((pid, sku))
                if self.first_add_seq is None:
                    self.first_add_seq = self.seq
                self.last_add_seq = self.seq
            self._sync_cart(obs)
        elif name == tools["remove_from_cart"]:
            self._sync_cart(obs)
            removed = obs.get(self._f("removed")) or {}
            pair = (removed.get(self._f("product_id")), removed.get(self._f("sku")))
            if pair in self.cart:
                self.cart.remove(pair)
        elif name == tools["inspect_cart"]:
            self._sync_cart(obs)

    def _sync_cart(self, obs: dict) -> None:
        rows = obs.get(self._f("cart"))
        if not isinstance(rows, list):
            return
        pid_k, sku_k = self._f("product_id"), self._f("sku")
        pairs = [(r.get(pid_k), r.get(sku_k)) for r in rows if isinstance(r, dict) and r.get(pid_k)]
        if pairs or rows == []:
            self.cart = pairs
        for r in rows:
            if not isinstance(r, dict) or not r.get(pid_k):
                continue
            k = self.key(r.get(pid_k), r.get(sku_k))
            price = _num(r.get(self._f("current_price")))
            stock = r.get(self._f("current_in_stock"))
            if price is None and stock is None:
                continue
            prev = self.checked.get(k, {})
            self.checked[k] = {
                "in_stock": bool(stock) if stock is not None else prev.get("in_stock", True),
                "price": price if price is not None else prev.get("price"),
            }
            self.checked_seq[k] = self.seq

    def _note_row(self, row: dict, brief: Brief) -> None:
        pid = row.get(self._f("product_id"))
        if not pid:
            return
        title = str(row.get(self._f("title")) or "")
        if brief.theme:
            if all(t in _tokens(title) for t in brief.theme):
                self.identities.add(str(pid))
        else:
            self.identities.add(str(pid))
        if not row.get(self._f("sku")):
            return
        k = self.key(pid, row.get(self._f("sku")))
        self.rows[k] = row
        if row.get(self._f("price")) is not None:
            self.checked.setdefault(
                k,
                {
                    "in_stock": bool(row.get(self._f("in_stock"), True)),
                    "price": _num(row.get(self._f("price"))),
                },
            )

    def _note_product(self, prod: dict, brief: Brief) -> None:
        pid = prod.get(self._f("product_id"))
        if not pid:
            return
        self.products[str(pid)] = prod
        listing = prod.get(self._f("listing_id"))
        if listing:
            self.listing_ids.add(str(listing))
            self.listing_of[str(pid)] = str(listing)
        if brief.theme and all(t in _tokens(str(prod.get(self._f("title")) or "")) for t in brief.theme):
            self.identities.add(str(pid))
        for v in prod.get(self._f("variants")) or []:
            if not isinstance(v, dict) or not v.get(self._f("sku")):
                continue
            k = self.key(pid, v.get(self._f("sku")))
            self.checked[k] = {
                "in_stock": bool(v.get(self._f("in_stock"))),
                "price": _num(v.get(self._f("price"))),
            }
            self.checked_seq[k] = self.seq
            self.options[k] = str(v.get(self._f("options")) or "")

    def status(self, pid: str, sku: str) -> dict | None:
        return self.checked.get(self.key(pid, sku))


def _arguments(tool_call: dict) -> dict:
    raw = (tool_call.get("function") or {}).get("arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _trim(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 40] + ' ... (truncated by agent)"}'


class Agent:
    def __init__(self, problem: dict, policy: dict | None = None) -> None:
        self.cfg = load_policy(problem, policy)
        env = problem["environment"]
        self.binding = dict(env["binding"])
        self.policy = env["policy_view"]
        self.tools = list(self.policy.get("tools") or [])
        self.tool_names = {(t.get("function") or {}).get("name") for t in self.tools if isinstance(t, dict)}
        rt = self.cfg.runtime
        self.max_steps = int(self.policy.get("max_steps") or rt["default_max_steps"])
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn") or 1))
        self.problem_id = str(problem.get("problem_id") or problem.get("id") or "problem")
        self.brief = Brief(str(self.policy.get("query") or ""), self.cfg)
        self.budget = self.brief.budget
        self.ledger = Ledger(self.cfg)
        self.clock = Clock(float(rt["problem_budget_s"]))
        self.dialogue: list = []
        self.messages: list = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": self.policy.get("query") or ""},
        ]
        self.turn = 0
        self.asked_shopper = False
        self.shopper_question_pending = False
        self.shopper_updated = False
        self.rejected_streak = 0
        self.pending_envelope: dict | None = None
        self.search_calls = 0
        self.cart_inspections = 0
        self.pending_recheck: tuple | None = None
        self.forced_compares = 0
        self.hinted_exact = False
        self.tried: dict = {}
        self.questions_asked = 0
        self.searched_siblings = False
        self.hinted_brand = False
        self._wishes: set | None = None
        self.probes: set = set()
        self.searched_twins = False
        self.reads_done = False
        self.event_turn: int | None = None
        self.unreadable: set = set()
        self.question_waits = 0
        self.done = False
        self.model_index = 0
        self.proxy = ProxyClient(
            timeout=float(rt["proxy_timeout"]),
            max_retries=int(rt["proxy_retries"]),
            retry_delay=float(rt["proxy_retry_delay"]),
        )
        provider = (getenv("INFERENCE_PROVIDER") or "").strip().lower() or rt["default_provider"]
        roster = list(self.cfg.models.get(provider) or self.cfg.models.get(rt["default_provider"]) or [])
        pinned = [m for m in ((getenv("SANDBOX_MODEL") or "").strip(),) if m]
        extra = [m.strip() for m in (getenv("SANDBOX_MODELS") or "").split(",") if m.strip()]
        for model in reversed(extra + pinned):
            roster = [model] + [m for m in roster if m != model]
        self.roster = tuple(roster) or ("deepseek-ai/DeepSeek-V3.2-TEE",)

    @property
    def fp(self) -> dict:
        return self.cfg.family(self.brief.family)

    def update_policy(self, overlay: dict) -> None:
        """Merge a policy overlay mid-run (new specs, playbooks, thresholds)."""
        self.cfg = Policy(_merge(self.cfg.raw, overlay))
        self.brief.policy = self.cfg
        self.ledger.policy = self.cfg
        self.messages[0] = {"role": "system", "content": self._system_prompt()}

    def _t(self, role: str) -> str:
        return self.cfg.tool(role)

    def _has(self, role: str) -> bool:
        return self._t(role) in self.tool_names

    def _system_prompt(self) -> str:
        play = self.fp.get("playbook") or self.cfg.family("generic").get("playbook") or ""
        rt = self.cfg.runtime
        extra = ""
        if self.fp.get("require_identities") and self.brief.required_identities:
            extra = (
                " You must surface at least %d distinct matching products in search or filter "
                "results before ordering; the controller reports the count after each search."
                % self.brief.required_identities
            )
        rules = (
            str(rt["core_rules"])
            .replace("{max_steps}", str(self.max_steps))
            .replace("{compare_batch}", str(rt["compare_batch"]))
        )
        return rules + "\n\nFamily playbook: " + play + extra

    def _opening_question(self) -> str | None:
        if self.fp.get("ask_sealed") or self.brief.spec_sealed:
            return self.fp.get("opening_question") or self.cfg.runtime.get("sealed_question")
        return self.fp.get("opening_question")

    def _note(self, text: str) -> None:
        clip = int(self.cfg.runtime["note_clip"])
        self.messages.append({"role": "user", "content": "[note] " + text})
        self.dialogue.append({"role": "assistant", "content": "[note] " + text[:clip]})

    def _prod(self, pid: str) -> dict:
        return self.ledger.products.get(str(pid)) or {}

    def _facts(self, pid: str) -> dict:
        f = self._prod(pid).get(self.cfg.field("facts"))
        return f if isinstance(f, dict) else {}

    def _title(self, pid: str) -> str:
        title = str(self._prod(pid).get(self.cfg.field("title")) or "")
        if title:
            return title
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::"):
                return str(row.get(self.cfg.field("title")) or "")
        return ""

    def _brand_of(self, pid: str) -> str:
        brand = self._prod(pid).get(self.cfg.field("brand"))
        if brand:
            return str(brand)
        for key, row in self.ledger.rows.items():
            if key.startswith(str(pid) + "::") and row.get(self.cfg.field("brand")):
                return str(row[self.cfg.field("brand")])
        return ""

    def _cat_path(self, pid: str) -> list:
        path = self._prod(pid).get(self.cfg.field("category_path"))
        return list(path) if isinstance(path, list) else []

    def _options(self, pid: str, sku: str) -> str:
        return self.ledger.options.get(Ledger.key(pid, sku), "")

    def _strip_models(self, pid: str, title: str) -> str:
        for token in re.split(r"[;,]", str(self._facts(pid).get("model") or "")):
            token = token.strip()
            stem = re.split(r"[-_]", token)[0] if token else ""
            if (
                token
                and re.fullmatch(r"[a-z0-9]+(?:[-_][a-z0-9]+)+", token, re.I)
                and re.search(r"[a-z]", stem, re.I)
                and re.search(r"\d", stem)
                and not re.fullmatch(r"(?:[48]k|[ufq]?hd|fullhd|ultrahd|\d+p)\d*", stem, re.I)
            ):
                title = re.sub(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])", "", title, flags=re.I)
        return title

    def _fact_value(self, pid: str, field: str) -> str | None:
        spec = self.cfg.spec_by_name.get(field) or {}
        raw = self._facts(pid).get(spec.get("fact_label") or field)
        if not raw:
            return None
        text = str(raw).strip()
        found = _spec_vals(self.cfg, field, text)
        if not found and spec.get("extractor") != "resolution" and re.fullmatch(r"\d+(?:\.\d+)?", text) and spec.get("unit"):
            return "%g" % float(text) + spec["unit"]
        return _one(found)

    def _product_value(self, pid: str, field: str) -> str | None:
        spec = self.cfg.spec_by_name.get(field) or {}
        vals: set = set()
        conflict = False
        raw = self._facts(pid).get(spec.get("fact_label") or field)
        if raw:
            found = _spec_vals(self.cfg, field, str(raw))
            conflict = conflict or len(found) > 1
            if spec.get("extractor") == "resolution":
                quarter = self.cfg.extractors.get("quarter")
                if quarter and quarter.search(str(raw)):
                    conflict = True
            vals |= found
        title = self._title(pid)
        if spec.get("extractor") == "resolution":
            title = self._strip_models(pid, title)
            quarter = self.cfg.extractors.get("quarter")
            if quarter and quarter.search(title):
                conflict = True
        vals |= _adjacent_vals(self.cfg, field, title)
        return None if conflict else _one(vals)

    def _variant_value(self, pid: str, sku: str, field: str) -> str | None:
        opt = self._options(pid, sku)
        vals = _adjacent_vals(self.cfg, field, opt)
        if len(vals) > 1:
            return None
        if vals:
            return next(iter(vals))
        if _mentions(self.cfg, field, opt):
            return None
        return self._product_value(pid, field)

    def _measure(self, pid: str, sku: str, field: str) -> float | None:
        value = self._variant_value(pid, sku, field)
        spec = self.cfg.spec_by_name.get(field) or {}
        unit = spec.get("unit") or ""
        try:
            if not value:
                return None
            return float(value[: -len(unit)] if unit and value.endswith(unit) else value)
        except ValueError:
            return None

    @staticmethod
    def _stems(text: str) -> set:
        return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

    def _category_ok(self, pid: str) -> bool:
        want = self.brief.category
        path = self._cat_path(pid)
        if not want or not path:
            return True
        wanted = self._stems(want)
        labels = set()
        for label in path:
            labels |= self._stems(str(label))
        placed = bool(wanted & labels) or any(
            self._stems("".join(re.findall(r"[a-z]+", str(label).lower()))) == wanted for label in path
        )
        rule = _identity_for(self.cfg, want)
        if not placed and rule and rule.get("alias_pattern_re"):
            placed = any(rule["alias_pattern_re"].fullmatch(str(label).strip()) for label in path)
        if not placed:
            return False
        qualifier = self.brief.cat_qualifier
        if not qualifier:
            return True
        row = next((q for q in self.cfg.qualifiers if q["name"] == qualifier), None)
        if not row:
            return True
        title = self._title(pid)
        words = self._stems(title) | _tokens(title)
        blob = title.lower()
        if row.get("title_all") and not all(self._stems(w) <= words or w in blob for w in row["title_all"]):
            return False
        any_ok = not row.get("title_any")
        if row.get("title_any"):
            any_ok = any(w in blob or self._stems(w) <= words for w in row["title_any"])
        if row.get("token_pairs"):
            any_ok = any_ok or any(all(tok in words for tok in pair) for pair in row["token_pairs"])
        hz_need = row.get("or_hz_at_least")
        if hz_need is not None:
            fast = any(float(v[:-2]) >= float(hz_need) for v in _hz_vals(self.cfg, title))
            any_ok = any_ok or fast
        return any_ok

    def _brand_ok(self, pid: str) -> bool:
        want = self.brief.brand
        if not want:
            return True
        wanted = _tokens(want)
        return bool(wanted <= _tokens(self._brand_of(pid)) or wanted <= _tokens(self._title(pid)))

    def _device_ok(self, pid: str, sku: str) -> bool:
        rule = _identity_for(self.cfg, self.brief.category)
        if not rule:
            return True
        verdict = _device_identity(rule, self._facts(pid), self._title(pid), self._options(pid, sku))
        strict = self.brief.family in (rule.get("strict_families") or []) or self.fp.get("device_strict")
        return verdict == "yes" if strict else verdict != "no"

    def _resale_title(self, pid: str) -> bool:
        return bool(self.cfg.resale_re.search(self._title(pid)))

    def _relax(self, pid: str, sku: str) -> int:
        model, color = self.brief.pref_model, self.brief.pref_color
        if not model:
            return 0
        squash = lambda t: re.sub(r"[^a-z0-9]+", "", (t or "").lower())
        blob = squash(self._title(pid) + " " + " ".join(str(v) for v in self._facts(pid).values()))
        same_model = squash(model) in blob
        shade = (self._options(pid, sku) + " " + str(self._facts(pid).get("color") or "") + " " + self._title(pid)).lower()
        same_color = bool(color) and re.search(r"(?<![a-z])" + re.escape(color) + r"(?![a-z])", shade) is not None
        if same_model and same_color:
            return 0
        if same_model:
            return 1
        return 2 if same_color else 3

    def _assess(self, pid: str, sku: str) -> dict:
        fp = self.fp
        key = Ledger.key(pid, sku)
        st = self.ledger.checked.get(key) or {}
        price, stock = st.get("price"), st.get("in_stock")
        dead = key in self.ledger.dead_keys or (
            str(pid) in self.ledger.dead_products and fp.get("kill_product_on_death", True)
        )
        live = bool(stock) and price is not None and (self.budget is None or price <= self.budget + _EPS) and not dead
        viewed = str(pid) in self.ledger.products
        field, want = self.brief.spec_field, self.brief.spec_value
        spec = 2
        if field and want:
            got = self._variant_value(pid, sku, field)
            spec = 2 if got == want else (-1 if got else 0)
        ok = bool(live and viewed and spec == 2 and self._brand_ok(pid) and self._device_ok(pid, sku))
        cat_ok = self._category_ok(pid)
        ok = ok and cat_ok
        if fp.get("ban_resale_titles"):
            ok = ok and not self._resale_title(pid) and bool(self._brand_of(pid))
        if fp.get("require_fact_match") and field and want:
            ok = ok and self._fact_value(pid, field) == want and self._product_value(pid, field) == want
        if fp.get("recover_different_listing") and self.ledger.event_target:
            dead_listing = self.ledger.listing_of.get(self.ledger.event_target.split("::", 1)[0])
            if dead_listing and self.ledger.listing_of.get(str(pid)) == dead_listing:
                ok = False
        if fp.get("require_theme") and self.brief.theme:
            ok = ok and all(t in _tokens(self._title(pid)) for t in self.brief.theme)
        if fp.get("require_named_keys"):
            ok = ok and (str(pid), str(sku)) in self.brief.candidate_keys
        for meas in fp.get("require_measurable") or []:
            ok = ok and self._measure(pid, sku, meas) is not None
        return {
            "pid": str(pid), "sku": str(sku), "key": key, "price": price, "live": live,
            "viewed": viewed, "spec": spec, "cat_ok": cat_ok, "ok": ok,
            "relax": self._relax(pid, sku) if fp.get("rank_mode") == "relax" or fp.get("require_exact_first") else 0,
        }

    def _rank_of(self, c: dict, first: bool) -> tuple:
        fp = self.fp
        price = c["price"] if c["price"] is not None else 1e12
        mode = fp.get("rank_mode") or "price"
        if mode == "refresh_screen_price":
            keys = []
            for field in fp.get("rank_fields") or ["refresh_rate", "screen_size"]:
                keys.append(-(self._measure(c["pid"], c["sku"], field) or 0.0))
            return tuple(keys) + (price, c["key"])
        if mode == "relax":
            return (c["relax"], price, c["key"])
        extra = sum(
            1 for f, v in self.brief.specs.items()
            if f != self.brief.spec_field and self._variant_value(c["pid"], c["sku"], f) != v
        )
        identity_cat = self.cfg.runtime.get("rank_identity_category") or self.brief.category
        rule = _identity_for(self.cfg, identity_cat)
        proven = 1
        if rule:
            proven = 0 if _device_identity(rule, self._facts(c["pid"]), self._title(c["pid"]), self._options(c["pid"], c["sku"])) == "yes" else 1
        if mode == "pref_brand" and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            near = 0 if (wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))) else 1
            return (near, extra, proven, price, c["key"])
        return (extra, proven, price, c["key"])

    def _candidates(self) -> list:
        seen = set()
        out = []
        for key in list(self.ledger.checked):
            pid, _, sku = key.partition("::")
            if not sku or key in seen:
                continue
            seen.add(key)
            out.append(self._assess(pid, sku))
        return out

    def _target(self) -> dict | None:
        first = self.ledger.adds == 0
        fp = self.fp
        good = [c for c in self._candidates() if c["ok"]]
        if fp.get("model_picks"):
            carted = [c for c in good if (c["pid"], c["sku"]) in self.ledger.cart]
            if carted:
                return carted[0]
            late = int(self.cfg.runtime["late_pick_turns"])
            if good and self.turn >= self.max_steps - late:
                good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]),
                                         c["price"] if c["price"] is not None else 1e12))
                return good[0]
            return None
        if fp.get("pref_brand_first") and first and self.brief.pref_brand:
            wanted = _tokens(self.brief.pref_brand)
            named = [c for c in good if wanted <= _tokens(self._brand_of(c["pid"])) or wanted <= _tokens(self._title(c["pid"]))]
            if named:
                good = named
            elif self.turn < self.max_steps * float(self.cfg.runtime["pref_wait_frac"]):
                self._preferred_brand_hint()
                return None
        if fp.get("require_exact_first") and first:
            exact = [c for c in good if c["relax"] == 0]
            if exact:
                good = exact
            elif self.turn < self.max_steps * float(self.cfg.runtime["pref_wait_frac"]):
                self._exact_match_hint()
                return None
        if not good:
            return None
        good.sort(key=lambda c: self._rank_of(c, first))
        return good[0]

    def _answer_text(self) -> str:
        target = self._target() or (self._candidates() or [None])[0]
        if not target:
            return "I am still checking the options against your requirements; nothing is decided yet."
        pid, sku = target["pid"], target["sku"]
        parts = ["I am going with " + (self._title(pid) or pid)[:90]]
        field, want = self.brief.spec_field, self.brief.spec_value
        if field and want and self._variant_value(pid, sku, field) == want:
            parts.append("its listed " + field.replace("_", " ") + " is " + want)
        if target.get("price") is not None:
            parts.append("it is %.2f %s and in stock" % (target["price"], self.brief.currency))
        return ", ".join(parts) + "."

    def _read_more(self) -> tuple | None:
        if self.fp.get("require_identities") and len(self.ledger.identities) < self._recall_need():
            return None
        twin = self._twin_search()
        if twin is not None:
            return twin
        leads = self._unseen_leads()
        batch = int(self.cfg.runtime["compare_batch"])
        cap = int(self.cfg.runtime["forced_compares_cap"])
        if leads and self.forced_compares < cap and self.turn >= 2:
            self.forced_compares += 1
            if len(leads) > 1 and self._has("compare"):
                return (self._t("compare"), {"product_ids": leads[:batch]})
            return (self._t("view"), {"product_id": leads[0]})
        probe = (self.brief.brand or "", "%.2f" % (self.budget or 0.0), str(self.brief.spec_value))
        if self.turn < 2:
            return None
        if probe in self.probes:
            self.reads_done = True
            return None
        self.probes.add(probe)
        k = int(self.cfg.runtime["search_k"])
        if self._has("filter") and self.brief.category:
            args = {"category": self.brief.category, "in_stock": True, "k": k}
            if self.budget is not None:
                args["max_price"] = self.budget
            if self.brief.brand:
                args["brand"] = self.brief.brand
            return (self._t("filter"), args)
        if self._has("search"):
            words = [w for w in (self.brief.brand, self.brief.spec_value, self.brief.category) if w]
            if words:
                return (self._t("search"), {"query": " ".join(words), "in_stock": True, "k": k})
        self.reads_done = True
        return None

    def _twin_search(self) -> tuple | None:
        dead = self.ledger.event_target
        if self.searched_twins or not dead or not self._has("search"):
            return None
        title = self._title(dead.split("::", 1)[0])
        if not title:
            return None
        self.searched_twins = True
        self._note(
            "The item that died is one of several listings sharing a title; searching that title "
            "surfaces its other colors and sizes, which meet the same requirements."
        )
        n = int(self.cfg.runtime["title_search_words"])
        k = int(self.cfg.runtime["search_k"])
        return (self._t("search"), {"query": " ".join(title.split()[:n]), "k": k, "in_stock": True})

    def _preferred_brand_hint(self) -> None:
        if self.hinted_brand:
            return
        self.hinted_brand = True
        self._note(
            "No %s product that meets every firm requirement has been read yet. The shopper's first choice is "
            "graded on that brand, so filter by brand=%s within the budget and read those products before any "
            "other." % (self.brief.pref_brand, self.brief.pref_brand)
        )

    def _sibling_search(self) -> tuple | None:
        if self.searched_siblings:
            return None
        rows = [c for c in self._candidates() if c["live"] and c["viewed"] and c["relax"] == 0]
        if rows:
            return None
        near = [c for c in self._candidates() if c["viewed"] and c["relax"] == 1]
        if not near:
            return None
        title = self._title(near[0]["pid"])
        if not title:
            return None
        self.searched_siblings = True
        n = int(self.cfg.runtime["title_search_words"])
        k = int(self.cfg.runtime["search_k"])
        return (self._t("search"), {"query": " ".join(title.split()[:n]), "k": k, "in_stock": True})

    def _exact_match_hint(self) -> None:
        if self.hinted_exact or not self.brief.pref_model:
            return
        self.hinted_exact = True
        self._note(
            "The shopper's exact preference (%s%s) has not been found yet. Search the model code on its own "
            "and view the results so every color variant is on record; only that exact pairing may be the "
            "first cart item." % (self.brief.pref_model, ", " + self.brief.pref_color if self.brief.pref_color else "")
        )

    def _unseen_leads(self) -> list:
        leads = []
        for key, row in self.ledger.rows.items():
            pid, _, sku = key.partition("::")
            if not sku or str(pid) in self.ledger.products or str(pid) in self.ledger.dead_products:
                continue
            st = self.ledger.checked.get(key) or {}
            price = st.get("price")
            if not st.get("in_stock") or price is None or (self.budget is not None and price > self.budget + _EPS):
                continue
            title = str(row.get(self.cfg.field("title")) or "")
            if self.brief.theme and not all(t in _tokens(title) for t in self.brief.theme):
                continue
            if self.brief.pref_model and self.ledger.adds == 0 and not self.ledger.event_target:
                squashed = re.sub(r"[^a-z0-9]+", "", title.lower())
                if re.sub(r"[^a-z0-9]+", "", self.brief.pref_model.lower()) not in squashed:
                    continue
            if self.brief.brand:
                wanted = _tokens(self.brief.brand)
                if not (wanted <= _tokens(str(row.get(self.cfg.field("brand")) or "")) or wanted <= _tokens(title)):
                    continue
            leads.append((price, str(pid)))
        leads.sort()
        out = []
        for _price, pid in leads:
            if pid not in out:
                out.append(pid)
        return out

    def _shortlist_note(self) -> None:
        cands = self._candidates()
        if self.fp.get("model_picks") and not self.ledger.cart:
            good = [c for c in cands if c["ok"]]
            good.sort(key=lambda c: (-self._priority_hits(c["pid"], c["sku"]), c["price"] if c["price"] is not None else 1e12))
            if good:
                take = int(self.cfg.runtime["shortlist_n"])
                self._note(
                    "Products that meet every firm requirement so far: "
                    + "; ".join("%s::%s at %.2f" % (c["pid"], c["sku"], c["price"] or 0.0) for c in good[:take])
                    + ". Pick the one that best fits the shopper's stated priorities, inspect it, then add it."
                )
                return
        best = self._target()
        if best:
            self._note(
                "Verified candidate: %s::%s at %.2f. The controller will inspect, cart and order it; "
                "read other candidates only if one could rank better."
                % (best["pid"], best["sku"], best["price"] or 0.0)
            )
            return
        field, want = self.brief.spec_field, self.brief.spec_value
        blocked = [c for c in cands if c["viewed"] and c["live"] and not c["ok"]]
        if blocked and field:
            sample = blocked[:3]
            self._note(
                "None of the products read so far can be verified. For each, the evidence for %s must "
                "resolve to exactly %s from the variant's own option text or the product's raw facts: %s. "
                "Look at different products." % (field, want, ", ".join(c["pid"] + "::" + c["sku"] for c in sample))
            )
        elif blocked:
            self._note(
                "The products read so far fail a firm requirement (category, brand, stock, budget or the "
                "kind of device requested). Search for different products."
            )

    def _priority_hits(self, pid: str, sku: str) -> int:
        wants = self._wish_words()
        if not wants:
            return 0
        have = _tokens(
            " ".join(str(v) for v in self._facts(pid).values())
            + " "
            + self._title(pid)
            + " "
            + self._options(pid, sku)
        )
        return len(wants & have)

    def _wish_words(self) -> set:
        if self._wishes is None:
            said = self.brief.goal + " " + " ".join(
                str(m.get("content") or "") for m in self.messages if m.get("role") == "user"
            )
            drop = _tokens(self.brief.category or "") | set(self.cfg.runtime.get("wish_stopwords") or [])
            self._wishes = {w for w in _tokens(said) if len(w) > 3 and not w.isdigit() and w not in drop}
        return self._wishes

    def _budget_from_message(self, text: str) -> None:
        new = self.brief._budget(text)
        if new is None or (self.budget is not None and new >= self.budget):
            return
        self.budget = new
        self.shopper_updated = True
        self._note("Budget updated by the shopper: at most %.2f %s. Every later choice must fit it." % (new, self.brief.currency))
        for pid, sku in list(self.ledger.cart):
            st = self.ledger.status(pid, sku) or {}
            price = st.get("price")
            if price is not None and price > new + _EPS:
                self.ledger.dead_keys.add(Ledger.key(pid, sku))
                self._note(
                    "The carted item %s costs %.2f, above the new budget; the controller will replace it."
                    % (Ledger.key(pid, sku), price)
                )

    def _claims_for(self, pid: str, sku: str) -> list:
        out = []
        field, want = self.brief.spec_field, self.brief.spec_value
        fallback = self.fp.get("claims_fallback_field")
        if not field and fallback:
            field, want = fallback, self._fact_value(pid, fallback)
        if not field or not want:
            return []
        stated = dict(self.brief.specs) or {field: want}
        stated.setdefault(field, want)
        for name, value in stated.items():
            if self._fact_value(pid, name) == value and self._product_value(pid, name) == value:
                out.append({"field": name, "value": value})
        if not out:
            return []
        st = self.ledger.status(pid, sku) or {}
        if self.budget is not None:
            out.append({"field": "price", "value": "at most %.2f" % self.budget})
        elif st.get("price") is not None:
            out.append({"field": "price", "value": "%.2f" % st["price"]})
        out.append({"field": "in_stock", "value": "true"})
        brand = self._brand_of(pid)
        if brand:
            out.append({"field": "brand", "value": brand})
        return out

    def _order_args(self, pid: str, sku: str) -> dict:
        out = {"product_id": pid, "sku": sku}
        if self.fp.get("attach_claims"):
            claims = self._claims_for(pid, sku)
            if claims:
                out["justification"] = {"claims": claims}
        return out

    def _valid_now(self, pid: str, sku: str) -> tuple:
        st = self.ledger.status(pid, sku)
        k = Ledger.key(pid, sku)
        if st is None:
            return False, "inspect_stock the exact product_id and sku first"
        if (pid, sku) not in self.ledger.cart:
            return False, "add that exact variant to the cart first"
        if self.ledger.last_add_seq is not None and self.ledger.checked_seq.get(k, 0) <= self.ledger.last_add_seq:
            return False, "re-check the variant with inspect_stock after the cart addition"
        if not st.get("in_stock"):
            return False, "that variant is out of stock; choose a different product"
        price = st.get("price")
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return False, "that variant costs %.2f, above the budget of %.2f" % (price, self.budget)
        if k in self.ledger.dead_keys:
            return False, "that variant was invalidated by the market change; choose a different product"
        return True, ""

    def _recall_need(self) -> int:
        rt = self.cfg.runtime
        early = self.search_calls < int(rt["recall_early_searches"]) and self.turn < self.max_steps * 0.5
        return self.brief.required_identities + (int(rt["recall_early_margin"]) if early else 0)

    def _ready(self) -> bool:
        fp = self.fp
        rt = self.cfg.runtime
        if self.ledger.adds:
            return True
        if self.brief.spec_sealed:
            if self.turn < self.max_steps * float(rt["sealed_unseal_frac"]):
                return False
            self.brief.spec_sealed = False
            self._note("The firm requirement was never stated, so the remaining requirements decide the choice.")
        if fp.get("wait_shopper") and self.turn < int(rt["wait_shopper_turns"]) and not self.shopper_updated:
            return False
        need_listings = int(fp.get("require_listing_diversity") or 0)
        if need_listings and len(self.ledger.listing_ids) < need_listings:
            return False
        roomy = self.turn < self.max_steps * float(rt["roomy_turn_frac"]) and self.clock.remaining() > float(rt["roomy_clock_s"])
        if fp.get("require_identities") and roomy:
            if len(self.ledger.identities) < self._recall_need() or self.search_calls < int(rt["recall_min_searches"]):
                return False
        if roomy and not self.reads_done and len({c["pid"] for c in self._candidates() if c["ok"]}) < int(rt["fallback_products"]):
            return False
        return True

    def _fresh(self, pid: str, sku: str) -> bool:
        k = Ledger.key(pid, sku)
        if k not in self.ledger.inspected:
            return False
        return self.ledger.last_add_seq is None or self.ledger.checked_seq.get(k, 0) > self.ledger.last_add_seq

    def _plan(self) -> tuple | None:
        if self.done:
            return None
        rt = self.cfg.runtime
        if self.shopper_question_pending:
            if self.question_waits < int(rt["question_force_after"]) or not self._has("message"):
                self.question_waits += 1
                return None
            return (self._t("message"), {"content": self._answer_text()})
        fp = self.fp
        if fp.get("compare_named_first") and self._has("compare"):
            missing = [
                p for p, _s in self.brief.candidate_keys
                if p not in self.ledger.products and p not in self.unreadable
            ]
            if missing:
                return (self._t("compare"), {"product_ids": missing[: int(rt["compare_batch"])]})
        if fp.get("sibling_search") and not self.ledger.adds and self.brief.pref_model and self._has("search"):
            forced = self._sibling_search()
            if forced is not None:
                return forced
        target = self._target()
        if target is None or not self._ready():
            forced = self._read_more()
            if forced is not None:
                return forced
            if target is None:
                return None
        pid, sku = target["pid"], target["sku"]
        if (pid, sku) in self.ledger.cart:
            if not self._fresh(pid, sku):
                return (self._t("inspect_stock"), {"product_id": pid, "sku": sku})
            return (self._t("place_test_order"), self._order_args(pid, sku))
        if not self._ready():
            return None
        for cpid, csku in self.ledger.cart:
            if Ledger.key(cpid, csku) in self.ledger.dead_keys and self._has("remove_from_cart"):
                return (self._t("remove_from_cart"), {"product_id": cpid, "sku": csku})
        if not self._fresh(pid, sku):
            return (self._t("inspect_stock"), {"product_id": pid, "sku": sku})
        return (self._t("add_to_cart"), {"product_id": pid, "sku": sku})

    def _ask_model(self) -> dict | None:
        rt = self.cfg.runtime
        for attempt in range(int(rt["inference_attempts"])):
            model = self.roster[min(self.model_index, len(self.roster) - 1)]
            payload = {
                "model": model,
                "messages": self.messages,
                "tools": self.tools,
                "tool_choice": "required",
                "temperature": rt["temperature"],
            }
            resp = self.proxy.post("/inference/chat/completions", json_data=payload)
            if resp and isinstance(resp, dict) and resp.get("choices"):
                msg = resp["choices"][0].get("message") or {}
                calls = msg.get("tool_calls") or []
                if calls:
                    return {"content": msg.get("content") or "", "tool_calls": calls[: self.max_calls]}
                self.messages.append({"role": "assistant", "content": msg.get("content") or ""})
                self._note("No tool call was made. Choose exactly one of the supplied tools now.")
                continue
            if self.clock.remaining() > 30:
                time.sleep(min(2.0 * (attempt + 1), 5.0))
            self.model_index = min(self.model_index + 1, len(self.roster) - 1)
        return None

    def _send(self, envelope: dict) -> dict | None:
        for attempt in range(int(self.cfg.runtime["env_attempts"])):
            result = self.proxy.post("/environment/call", json_data=envelope)
            if isinstance(result, dict) and result.get("calls"):
                return result
            if self.clock.remaining() < 20:
                break
            time.sleep(min(2.0 * (attempt + 1), 6.0))
        return None

    def _act(self, name: str, args: dict, content: str = "", tool_call_id: str | None = None) -> dict | None:
        if self.pending_envelope is not None:
            pend = self.pending_envelope
            replay = self._send(pend["envelope"])
            if replay is None:
                self._note("The environment is not answering; the previous action is still pending and will be retried.")
                return None
            self.pending_envelope = None
            self.turn = pend["turn"]
            self._absorb(pend["name"], pend["args"], pend["content"], pend["call_id"], replay)
            if self.done:
                return None
        turn = self.turn + 1
        group = self.problem_id + "-turn-" + str(turn)
        envelope = {
            **self.binding,
            "call_id": group,
            "idempotency_key": group,
            "turn": turn,
            "calls": [{"call_id": group + "-1", "action": {"name": name, "args": args}}],
        }
        result = self._send(envelope)
        if result is None:
            self.pending_envelope = {
                "envelope": envelope, "turn": turn, "name": name, "args": args,
                "content": content, "call_id": tool_call_id or ("agent-" + str(turn)),
            }
        else:
            self.turn = turn
        return self._absorb(name, args, content, tool_call_id or ("agent-" + str(turn)), result)

    def _absorb(self, name: str, args: dict, content: str, call_id: str, result: dict | None) -> dict | None:
        tool_calls = [{"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]
        self.messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        self.dialogue.append(
            {"role": "assistant", "content": content, "tool_calls": tool_calls, "environment_result": result}
        )
        if not isinstance(result, dict):
            self.ledger.calls_seen.discard(name + json.dumps(args, sort_keys=True))
            self.messages.append(
                {"role": "tool", "tool_call_id": call_id,
                 "content": json.dumps({"error": "environment call failed; it will be retried automatically"})}
            )
            return None
        calls = result.get("calls") or []
        wrap = (calls[0].get("observation") if calls and isinstance(calls[0], dict) else None) or {}
        obs = wrap.get("observation") if isinstance(wrap.get("observation"), dict) else wrap
        if isinstance(obs, dict) and wrap.get("error") and not obs.get(self.cfg.field("error")):
            obs = dict(obs)
            obs[self.cfg.field("error")] = wrap.get("error")
        if wrap.get(self.cfg.field("done")):
            self.done = True
        self.ledger.record(name, args, obs, self.brief)
        self.messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": _trim(json.dumps(obs), int(self.cfg.runtime["obs_chars"]))}
        )
        user = result.get("user_message")
        if isinstance(user, dict) and user.get("content"):
            text = str(user["content"])
            self.messages.append({"role": "user", "content": text})
            self._budget_from_message(text)
            if self.fp.get("wait_shopper") or self.fp.get("ask_sealed"):
                if not self.brief.spec_value:
                    stated = self.brief.adopt_spec(text)
                    if stated:
                        self.shopper_updated = True
                        self._note("Firm requirement recorded: " + stated + ". Only a variant that resolves to it may be bought.")
            qrx = self.cfg.rx.get("question")
            if qrx and qrx.search(text):
                self.shopper_question_pending = True
                self._note("The shopper asked a question: answer it with message in your next turn, then continue.")
        if any(isinstance(c, dict) and (c.get("observation") or {}).get("done") for c in calls):
            self.done = True
        return obs

    def _post_add_recheck(self) -> None:
        pid, sku = self.pending_recheck
        self.pending_recheck = None
        obs = self._act(self._t("inspect_stock"), {"product_id": pid, "sku": sku})
        if not isinstance(obs, dict) or obs.get(self.cfg.field("error")):
            return
        stock = bool(obs.get(self.cfg.field("in_stock")))
        price = _num(obs.get(self.cfg.field("price")))
        over = self.budget is not None and price is not None and price > self.budget + _EPS
        if stock and not over:
            self._note(
                "Re-check after the cart addition: %s is in stock at %s, unchanged. The controller will order it."
                % (Ledger.key(pid, sku), price)
            )
            return
        self.ledger.event_target = Ledger.key(pid, sku)
        self.event_turn = self.turn
        self.ledger.dead_keys.add(Ledger.key(pid, sku))
        self.ledger.dead_products.add(str(pid))
        self.forced_compares = 0
        reason = "went out of stock" if not stock else "was repriced to %.2f, above the budget" % (price or 0.0)
        guidance = self.fp.get("recovery_guidance") or "Choose a different product that meets every firm requirement."
        self._note("Market change: the carted item %s %s. " % (Ledger.key(pid, sku), reason) + guidance)
        self._shortlist_note()

    def _finish_fast(self) -> None:
        for pid, sku in reversed(self.ledger.cart):
            ok, _why = self._valid_now(pid, sku)
            if ok and not self.done:
                self._act(self._t("place_test_order"), self._order_args(pid, sku))
                return
        if self.done or self.turn > self.max_steps - 3:
            return
        good = [c for c in self._candidates() if c["ok"]]
        if not good and not self.fp.get("panic_strict"):
            good = [
                c for c in self._candidates()
                if c["live"] and c["spec"] == 2 and self._brand_ok(c["pid"]) and c["cat_ok"]
                and (not self.brief.candidate_keys or (c["pid"], c["sku"]) in self.brief.candidate_keys)
            ]
        if not good:
            return
        good.sort(key=lambda c: self._rank_of(c, self.ledger.adds == 0))
        pid, sku = good[0]["pid"], good[0]["sku"]
        if (pid, sku) not in self.ledger.cart:
            obs = self._act(self._t("add_to_cart"), {"product_id": pid, "sku": sku})
            if self.done or not isinstance(obs, dict) or obs.get(self.cfg.field("error")):
                return
        obs = self._act(self._t("inspect_stock"), {"product_id": pid, "sku": sku})
        if self.done or not isinstance(obs, dict) or not obs.get(self.cfg.field("in_stock")):
            return
        price = _num(obs.get(self.cfg.field("price")))
        if self.budget is not None and price is not None and price > self.budget + _EPS:
            return
        self._act(self._t("place_test_order"), self._order_args(pid, sku))

    def run(self) -> list:
        question = self._opening_question()
        last_forced, repeats = None, 0
        rt = self.cfg.runtime
        while not self.done and self.turn < self.max_steps:
            if self.clock.remaining() < float(rt["panic_remaining_s"]) or self.turn >= self.max_steps - int(rt["panic_turns_left"]):
                self._finish_fast()
                break
            if question and not self.asked_shopper and self._has("message"):
                self.asked_shopper = True
                self._act(self._t("message"), {"content": question})
                continue
            if self.pending_recheck and self._has("inspect_stock"):
                self._post_add_recheck()
                continue
            plan = self._plan()
            if plan is not None:
                name, args = plan
                signature = name + json.dumps(args, sort_keys=True)
                repeats = repeats + 1 if signature == last_forced else 1
                last_forced = signature
                if repeats > int(rt["repeat_abort"]):
                    if name in (self._t("inspect_stock"), self._t("add_to_cart")):
                        self.ledger.dead_keys.add(Ledger.key(args.get("product_id"), args.get("sku")))
                    if name == self._t("compare"):
                        self.unreadable.update(str(p) for p in args.get("product_ids") or [])
                    if name == self._t("message"):
                        self.shopper_question_pending = False
                    last_forced, repeats = None, 0
                    self._note("That step changed nothing, so the controller moved on to other candidates.")
                    continue
                obs = self._act(name, args)
                if name == self._t("message"):
                    self.shopper_question_pending = False
                    self.question_waits = 0
                if name == self._t("add_to_cart") and isinstance(obs, dict) and not obs.get(self.cfg.field("error")):
                    added = obs.get(self.cfg.field("added")) or {}
                    if added.get(self.cfg.field("product_id")):
                        self.pending_recheck = (str(added[self.cfg.field("product_id")]), str(added[self.cfg.field("sku")]))
                continue
            last_forced, repeats = None, 0
            proposal = self._ask_model()
            if proposal is None:
                self._finish_fast()
                break
            call = proposal["tool_calls"][0]
            name = (call.get("function") or {}).get("name") or ""
            args = _arguments(call)
            content = proposal.get("content") or ""
            reprompts = 0
            while True:
                verdict = self._validate(name, args)
                if verdict is None:
                    break
                if reprompts >= int(rt["max_reprompts"]):
                    self.rejected_streak += 1
                    self._note("That call was not made: " + verdict + ". Choose a different, valid action.")
                    if self.rejected_streak >= 3:
                        self._shortlist_note()
                        self.rejected_streak = 0
                    verdict = "skip"
                    break
                reprompts += 1
                self.messages.append({"role": "assistant", "content": content, "tool_calls": [call]})
                self.messages.append(
                    {"role": "tool", "tool_call_id": call.get("id") or "x", "content": json.dumps({"rejected": verdict})}
                )
                self.dialogue.append({"role": "assistant", "content": "[rejected " + name + "] " + verdict[:200]})
                again = self._ask_model()
                if again is None:
                    break
                call = again["tool_calls"][0]
                name = (call.get("function") or {}).get("name") or ""
                args = _arguments(call)
                content = again.get("content") or ""
            if verdict == "skip":
                continue
            self.rejected_streak = 0
            if name not in self.tool_names:
                continue
            if name == self._t("place_test_order"):
                pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
                if not pid and self.ledger.cart:
                    pid, sku = self.ledger.cart[-1]
                args = self._order_args(pid, sku)
            if name == self._t("message"):
                if not self.shopper_question_pending:
                    self.questions_asked += 1
                self.shopper_question_pending = False
                self.question_waits = 0
            obs = self._act(name, args, content, call.get("id"))
            if name == self._t("add_to_cart") and isinstance(obs, dict) and not obs.get(self.cfg.field("error")):
                added = obs.get(self.cfg.field("added")) or {}
                if added.get(self.cfg.field("product_id")):
                    self.pending_recheck = (str(added[self.cfg.field("product_id")]), str(added[self.cfg.field("sku")]))
            if name in (self._t("search"), self._t("filter")):
                self.search_calls += 1
                if self.fp.get("require_identities") and self.brief.required_identities:
                    have, need = len(self.ledger.identities), self._recall_need()
                    self._note(
                        "Distinct matching products surfaced so far: %d of the required %d." % (have, need)
                        + ("" if have >= need else " Keep searching with different wording.")
                    )
            if name == self._t("inspect_cart"):
                self.cart_inspections += 1
            if name in (self._t("view"), self._t("compare")):
                self._shortlist_note()
        return self.dialogue

    def _validate(self, name: str, args: dict) -> str | None:
        fp = self.fp
        rt = self.cfg.runtime
        if name not in self.tool_names:
            return "unknown tool; choose one of " + str(sorted(n for n in self.tool_names if n))
        if name == self._t("place_test_order"):
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid and self.ledger.cart:
                pid, sku = self.ledger.cart[-1]
            ok, why = self._valid_now(pid, sku)
            if not ok:
                return "order refused: " + why
            if fp.get("require_identities") and len(self.ledger.identities) < self._recall_need():
                return (
                    "order refused: only %d distinct matching products have been surfaced; search with different "
                    "wording until at least %d appear" % (len(self.ledger.identities), self._recall_need())
                )
            need_listings = int(fp.get("require_listing_diversity") or 0)
            if need_listings and len(self.ledger.listing_ids) < need_listings:
                return "order refused: compare at least %d products with different listing_id values first" % need_listings
        if name == self._t("add_to_cart"):
            pid, sku = str(args.get("product_id") or ""), str(args.get("sku") or "")
            if not pid or not sku:
                return "add_to_cart needs both product_id and sku"
            c = self._assess(pid, sku)
            key = Ledger.key(pid, sku)
            if not c["live"]:
                return "that variant is out of stock, over budget, or already invalidated"
            if not c["viewed"]:
                return "view or compare that product first so its raw facts are on record"
            if key not in self.ledger.inspected:
                return "inspect_stock that exact variant first: its current price and stock decide whether it can be bought"
            if c["spec"] != 2 and self.brief.spec_field:
                return (
                    "that variant does not resolve to %s = %s from its own option text or raw facts"
                    % (self.brief.spec_field, self.brief.spec_value)
                )
            if not c["ok"]:
                return "that product fails a firm requirement (category, brand, or the kind of device requested)"
            need_listings = int(fp.get("require_listing_diversity") or 0)
            if need_listings and len(self.ledger.listing_ids) < need_listings:
                return "compare %d products with different listing_id values before the first cart addition" % need_listings
            if fp.get("require_exact_first") and not self.ledger.adds and self.brief.pref_model and self._relax(pid, sku):
                return (
                    "the first cart item must be the exact model and color the shopper named; the other colors are "
                    "separate listings sharing the same title, so search that title and read them first"
                )
            if not self._ready():
                return (
                    "hold: read one more product that also meets every requirement first, so there is a "
                    "fallback when the market moves right after the cart addition"
                )
        if name == self._t("message"):
            text = str(args.get("content") or "")
            if not text.strip():
                return "message content is required"
            if not self.shopper_question_pending and "?" not in text:
                return "do not send status updates; use message only to answer the shopper's question or to ask one"
            if not self.shopper_question_pending and self.questions_asked >= int(rt["max_questions"]):
                return (
                    "the shopper has already answered; decide from the evidence you have rather than asking again"
                )
        if name == self._t("inspect_cart") and self.cart_inspections >= int(rt["max_cart_inspections"]):
            return "the cart contents are already known from earlier observations"
        key = name + json.dumps(args, sort_keys=True)
        if key in self.ledger.calls_seen and name in (self._t("search"), self._t("filter"), self._t("view"), self._t("compare")):
            return "identical call already made; change the arguments"
        self.ledger.calls_seen.add(key)
        return None


def agent_main(problem_data: dict) -> list:
    try:
        return Agent(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": "agent error: " + type(exc).__name__ + ": " + str(exc)[:200]}]


__all__ = ["agent_main", "Agent", "Brief", "Policy", "load_policy", "default_policy", "DEFAULT_POLICY"]
