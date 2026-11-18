"""ORO Bench agent for 145_Valid — event-aware shopper on tool contract v5.

Adds a live catalog memory and a hard order gate: if price/stock of a previously
seen listing changes, or a shopper message reports a mid-episode update, the
agent must re-observe that listing before ``place_test_order`` is allowed to
fire. That closes the stateful ``event_closed`` gap the thin loop missed.

145_Valid's fail-fast proxy tuning and model chain (including GLM-5.2) are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

_INFER = ProxyClient(timeout=75, max_retries=1, retry_delay=1.0)
_WORLD = ProxyClient(timeout=20, max_retries=2, retry_delay=1.0)

SWEEPS = 3
NUDGES = 3

CHUTES = [
    "moonshotai/Kimi-K2.6-TEE",
    "zai-org/GLM-5.1-TEE",
    "zai-org/GLM-5.2-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
    "deepseek-ai/DeepSeek-R1-0528-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "Qwen/Qwen3-32B-TEE",
    "XiaomiMiMo/MiMo-V2-Flash-TEE",
    "google/gemma-4-31B-turbo-TEE",
]
OPENROUTER = [
    "moonshotai/kimi-k2.6",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-32b",
    "xiaomi/mimo-v2-flash",
    "google/gemma-4-31b-it",
]

GUIDE = (
    "You are a careful shopping agent in a generated environment. Keep every "
    "firm requirement (category, spec, budget, in-stock) for the whole episode. "
    "Use only the tools supplied for this task.\n"
    "If a later observation or shopper message shows that price or stock of an "
    "item you already looked at has changed, say so briefly (use a messaging "
    "tool if one is available), then re-check that item's current price and "
    "stock before ordering. Do not place an order on stale numbers. If the "
    "old pick is now over budget or out of stock, switch to another listing "
    "that still meets every firm requirement.\n"
    "Continue until an observation reports done=true."
)

_EVENT_HINTS = (
    "price",
    "stock",
    "sold",
    "unavailable",
    "changed",
    "updated",
    "no longer",
    "out of",
    "budget",
    "now costs",
    "repric",
)


def _models() -> list[str]:
    chain = OPENROUTER if getenv("INFERENCE_PROVIDER", "chutes") == "openrouter" else CHUTES
    pin = getenv("SANDBOX_MODEL")
    return [pin] + [m for m in chain if m != pin] if pin else chain


def _args(call: dict[str, Any]) -> dict[str, Any]:
    raw = (call.get("function") or {}).get("arguments", "{}")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _ask(messages: list[dict[str, Any]], tools: list[dict[str, Any]], cap: int):
    for _ in range(SWEEPS):
        for model in _models():
            try:
                resp = _INFER.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
            except Exception:
                continue
            if not resp or not resp.get("choices"):
                continue
            msg = resp["choices"][0]["message"]
            calls = (msg.get("tool_calls") or [])[:cap]
            if calls:
                return msg, calls
    return None, []


def _listings(blob: Any) -> list[dict[str, Any]]:
    if isinstance(blob, dict):
        if isinstance(blob.get("results"), list):
            return [x for x in blob["results"] if isinstance(x, dict)]
        if blob.get("product_id") is not None:
            return [blob]
        out = []
        for v in blob.get("variants") or []:
            if isinstance(v, dict) and blob.get("product_id"):
                row = dict(v)
                row["product_id"] = blob["product_id"]
                out.append(row)
        if out:
            return out
    if isinstance(blob, list):
        return [x for x in blob if isinstance(x, dict)]
    return []


@dataclass
class CatalogMemory:
    """Remembers last-seen price/stock so a later drift can be treated as an event."""

    snaps: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    event_open: bool = False
    fresh: set[str] = field(default_factory=set)

    def absorb(self, payload: Any) -> list[str]:
        notices: list[str] = []
        try:
            for item in _listings(payload):
                pid = str(item.get("product_id") or "")
                if not pid:
                    continue
                sku = str(item.get("sku") or "")
                key = f"{pid}::{sku}"
                price = item.get("price")
                stock = item.get("in_stock")
                if price is None and isinstance(item.get("variants"), list):
                    for v in item["variants"]:
                        if isinstance(v, dict) and v.get("price") is not None:
                            price = v.get("price")
                            stock = v.get("in_stock", stock)
                            sku = str(v.get("sku") or sku)
                            key = f"{pid}::{sku}"
                            break
                prev = self.snaps.get(key)
                if prev is not None and (price, stock) != prev and (price is not None or stock is not None):
                    self.event_open = True
                    notices.append(key)
                if price is not None or stock is not None:
                    self.snaps[key] = (price, stock)
                if self.event_open:
                    self.fresh.add(key)
                    self.fresh.add(f"{pid}::")
        except Exception:
            pass
        return notices

    def hear(self, text: str) -> bool:
        low = (text or "").lower()
        if any(tok in low for tok in _EVENT_HINTS):
            self.event_open = True
            return True
        return False

    def order_ok(self, product_id: Any, sku: Any) -> bool:
        if not self.event_open:
            return True
        pid = str(product_id or "")
        sku_s = str(sku or "")
        return f"{pid}::{sku_s}" in self.fresh or f"{pid}::" in self.fresh


class ConstraintSession:
    def __init__(self, problem_data: dict[str, Any]):
        env = problem_data["environment"]
        self.binding = env["binding"]
        self.policy = env["policy_view"]
        self.problem_id = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.max_calls = max(1, int(self.policy.get("max_calls_per_turn", 1)))
        self.max_steps = max(1, int(self.policy.get("max_steps", 20)))
        self.tools = self.policy.get("tools") or []
        self.memory = CatalogMemory()
        self.messages = [
            {"role": "system", "content": GUIDE},
            {"role": "user", "content": self.policy.get("query", "")},
        ]
        self.dialogue: list[dict[str, Any]] = []

    def _block_stale_order(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        kept = []
        blocked = False
        for call in calls:
            name = (call.get("function") or {}).get("name", "")
            if name == "place_test_order":
                a = _args(call)
                if not self.memory.order_ok(a.get("product_id"), a.get("sku")):
                    blocked = True
                    continue
            kept.append(call)
        if blocked:
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        "A mid-episode price or stock change was detected. "
                        "Re-check the intended listing with view or inspect_stock, "
                        "acknowledge the change if a messaging tool exists, and only "
                        "then place the order — or switch if it no longer qualifies."
                    ),
                }
            )
        return kept or calls

    def play(self) -> list[dict[str, Any]]:
        for turn in range(1, self.max_steps + 1):
            assistant, calls = None, []
            for _ in range(NUDGES):
                assistant, calls = _ask(self.messages, self.tools, self.max_calls)
                if calls:
                    break
                self.messages.append(
                    {
                        "role": "user",
                        "content": "The environment has not reported done=true. Choose a supplied tool.",
                    }
                )
            if not calls:
                self.dialogue.append(
                    {"role": "assistant", "content": "No model returned a tool call this turn."}
                )
                break

            calls = self._block_stale_order(calls)
            text = (assistant or {}).get("content") or ""
            self.messages.append({"role": "assistant", "content": text, "tool_calls": calls})
            tag = f"{self.problem_id}-turn-{turn}"
            envelope = {
                **self.binding,
                "call_id": tag,
                "idempotency_key": tag,
                "turn": turn,
                "calls": [
                    {
                        "call_id": f"{tag}-{i}",
                        "action": {
                            "name": (c.get("function") or {}).get("name", ""),
                            "args": _args(c),
                        },
                    }
                    for i, c in enumerate(calls, start=1)
                ],
            }
            result = None
            try:
                result = _WORLD.post("/environment/call", json_data=envelope)
            except Exception as exc:
                self.dialogue.append({"role": "assistant", "content": f"environment call failed: {exc}"})
            self.dialogue.append(
                {
                    "role": "assistant",
                    "content": text,
                    "tool_calls": calls,
                    "environment_result": result,
                }
            )
            if not result or "calls" not in result:
                break

            drifts = []
            for call, row in zip(calls, result["calls"]):
                obs = row.get("observation") or {}
                payload = obs.get("observation", obs)
                drifts.extend(self.memory.absorb(payload))
                name = (call.get("function") or {}).get("name", "")
                if name in ("view", "inspect_stock", "compare"):
                    self.memory.absorb(payload)
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(obs),
                    }
                )
            shopper = result.get("user_message")
            if shopper:
                body = shopper.get("content") if isinstance(shopper, dict) else str(shopper)
                heard = self.memory.hear(str(body))
                self.messages.append({"role": "user", "content": body})
                if heard:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Treat the shopper note as a live update. Re-verify current "
                                "price and stock before any order."
                            ),
                        }
                    )
            if drifts:
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Observed listing state changed for: "
                            + ", ".join(drifts[:4])
                            + ". Re-verify before ordering."
                        ),
                    }
                )
            if any((c.get("observation") or {}).get("done") for c in result["calls"]):
                break
        return self.dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return ConstraintSession(problem_data).play()
    except Exception as exc:
        return [{"role": "assistant", "content": f"Unhandled agent error: {type(exc).__name__}: {exc}"}]


__all__ = ["agent_main"]
