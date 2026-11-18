"""ORO Bench agent for 140_BILL — recovery-first shopper on tool contract v5.

140_BILL's longer inference budget and DeepSeek-first model order stay in place.
The loop is a small phase machine (browse → hold → event → close) so a mid-episode
price/stock shift cannot be ignored: the close phase refuses ``place_test_order``
until a post-event view/inspect_stock observation lands for the intended SKU.
"""

from __future__ import annotations

import json
from enum import Enum
from os import getenv
from typing import Any

from src.agent.proxy_client import ProxyClient

LLM = ProxyClient(timeout=180, max_retries=5, retry_delay=1.0)
ENV = ProxyClient(timeout=30, max_retries=5, retry_delay=1.0)

PASSES = 2
REPROMPTS = 3

CHUTES_ORDER = [
    "moonshotai/Kimi-K2.6-TEE",
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "deepseek-ai/DeepSeek-V3-0324-TEE",
    "deepseek-ai/DeepSeek-R1-0528-TEE",
    "zai-org/GLM-5.1-TEE",
    "Qwen/Qwen3.5-397B-A17B-TEE",
    "Qwen/Qwen3-32B-TEE",
    "XiaomiMiMo/MiMo-V2-Flash-TEE",
    "google/gemma-4-31B-turbo-TEE",
]
OR_ORDER = [
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-chat-v3.1",
    "deepseek/deepseek-chat-v3-0324",
    "deepseek/deepseek-r1-0528",
    "z-ai/glm-5.1",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3-32b",
    "xiaomi/mimo-v2-flash",
    "google/gemma-4-31b-it",
]

SYSTEM = (
    "You are a careful shopping agent operating inside a generated environment. "
    "Keep every explicit requirement — price, quantity, attributes, ordering, "
    "preference — for the rest of the episode. Use only the tools supplied here.\n"
    "State can change while you shop. After a cart action, read any shopper "
    "message before ordering. If price or availability moved, mention the change "
    "and confirm the current numbers with a tool. Do not repeat a failed call. "
    "Finish only when an observation reports done=true."
)


class Phase(str, Enum):
    BROWSE = "browse"
    HOLD = "hold"
    EVENT = "event"
    CLOSE = "close"


def _chain() -> list[str]:
    base = OR_ORDER if getenv("INFERENCE_PROVIDER", "chutes") == "openrouter" else CHUTES_ORDER
    pinned = getenv("SANDBOX_MODEL")
    if pinned:
        return [pinned, *[m for m in base if m != pinned]]
    return list(base)


def decode_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    blob = (tool_call.get("function") or {}).get("arguments", "{}")
    try:
        data = json.loads(blob) if isinstance(blob, str) else blob
    except (TypeError, ValueError):
        data = {}
    return data if isinstance(data, dict) else {}


def pick_tools(history: list[dict[str, Any]], tools: list[dict[str, Any]], limit: int):
    for _sweep in range(PASSES):
        for name in _chain():
            try:
                packed = LLM.post(
                    "/inference/chat/completions",
                    json_data={
                        "model": name,
                        "messages": history,
                        "tools": tools,
                        "tool_choice": "required",
                        "temperature": 0,
                    },
                )
            except Exception:
                continue
            if not packed or not packed.get("choices"):
                continue
            assistant = packed["choices"][0]["message"]
            chosen = (assistant.get("tool_calls") or [])[:limit]
            if chosen:
                return assistant, chosen
    return None, []


def _walk_items(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("product_id") is not None:
            found.append(node)
        for key in ("results", "products", "variants"):
            val = node.get(key)
            if isinstance(val, list):
                for row in val:
                    if isinstance(row, dict):
                        if "product_id" not in row and node.get("product_id"):
                            row = {**row, "product_id": node["product_id"]}
                        found.append(row)
    elif isinstance(node, list):
        for row in node:
            if isinstance(row, dict):
                found.append(row)
    return found


class ShopRunner:
    def __init__(self, problem_data: dict[str, Any]):
        environment = problem_data["environment"]
        self.bind = environment["binding"]
        self.view = environment["policy_view"]
        self.pid = str(problem_data.get("problem_id", problem_data.get("id", "problem")))
        self.cap = max(1, int(self.view.get("max_calls_per_turn", 1)))
        self.horizon = max(1, int(self.view.get("max_steps", 20)))
        self.toolbox = self.view.get("tools") or []
        self.phase = Phase.BROWSE
        self.last_seen: dict[str, dict[str, Any]] = {}
        self.confirmed: set[str] = set()
        self.history: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.view.get("query", "")},
        ]
        self.log: list[dict[str, Any]] = []

    def _mark(self, payload: Any) -> bool:
        moved = False
        try:
            for item in _walk_items(payload):
                product = str(item.get("product_id") or "")
                if not product:
                    continue
                sku = str(item.get("sku") or "")
                token = f"{product}|{sku}"
                snapshot = {
                    "price": item.get("price"),
                    "in_stock": item.get("in_stock"),
                }
                prior = self.last_seen.get(token)
                if prior and snapshot != prior and any(v is not None for v in snapshot.values()):
                    moved = True
                    self.phase = Phase.EVENT
                self.last_seen[token] = {**prior, **{k: v for k, v in snapshot.items() if v is not None}} if prior else snapshot
                if self.phase in (Phase.EVENT, Phase.CLOSE):
                    self.confirmed.add(token)
                    self.confirmed.add(f"{product}|")
        except Exception:
            return moved
        return moved

    def _shopper_shift(self, text: str) -> bool:
        low = text.lower()
        cues = ("price", "stock", "sold", "unavailable", "changed", "updated", "budget", "no longer", "out of")
        if any(c in low for c in cues):
            self.phase = Phase.EVENT
            return True
        return False

    def _ready(self, product_id: Any, sku: Any) -> bool:
        if self.phase not in (Phase.EVENT, Phase.CLOSE):
            return True
        product = str(product_id or "")
        sku_s = str(sku or "")
        return f"{product}|{sku_s}" in self.confirmed or f"{product}|" in self.confirmed

    def _filter(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.phase not in (Phase.EVENT, Phase.CLOSE):
            return calls
        surviving = []
        skipped = False
        for call in calls:
            fname = (call.get("function") or {}).get("name", "")
            if fname == "place_test_order":
                payload = decode_args(call)
                if not self._ready(payload.get("product_id"), payload.get("sku")):
                    skipped = True
                    continue
                self.phase = Phase.CLOSE
            surviving.append(call)
        if skipped:
            self.history.append(
                {
                    "role": "user",
                    "content": (
                        "Hold the order. Confirm current price and stock of the intended "
                        "item first (view or inspect_stock). If it fails the budget or "
                        "availability now, pick another qualifying listing."
                    ),
                }
            )
            return surviving or calls
        return surviving

    def run(self) -> list[dict[str, Any]]:
        step = 1
        while step <= self.horizon:
            assistant, calls = None, []
            tries = 0
            while tries < REPROMPTS and not calls:
                assistant, calls = pick_tools(self.history, self.toolbox, self.cap)
                if calls:
                    break
                self.history.append(
                    {
                        "role": "user",
                        "content": "Still not done. Issue one of the supplied tools.",
                    }
                )
                tries += 1
            if not calls:
                self.log.append({"role": "assistant", "content": "No model in the chain produced a tool call."})
                break

            calls = self._filter(calls)
            spoken = (assistant or {}).get("content") or ""
            self.history.append({"role": "assistant", "content": spoken, "tool_calls": calls})
            stamp = f"{self.pid}-turn-{step}"
            body = {
                **self.bind,
                "call_id": stamp,
                "idempotency_key": stamp,
                "turn": step,
                "calls": [
                    {
                        "call_id": f"{stamp}-{n}",
                        "action": {
                            "name": (item.get("function") or {}).get("name", ""),
                            "args": decode_args(item),
                        },
                    }
                    for n, item in enumerate(calls, start=1)
                ],
            }
            reply = None
            try:
                reply = ENV.post("/environment/call", json_data=body)
            except Exception as err:
                self.log.append({"role": "assistant", "content": f"environment call failed: {err}"})
            self.log.append(
                {
                    "role": "assistant",
                    "content": spoken,
                    "tool_calls": calls,
                    "environment_result": reply,
                }
            )
            if not reply or "calls" not in reply:
                break

            shifted = False
            for call, row in zip(calls, reply["calls"]):
                observation = row.get("observation") or {}
                inner = observation.get("observation", observation)
                shifted = self._mark(inner) or shifted
                fname = (call.get("function") or {}).get("name", "")
                if fname in ("add_to_cart", "remove_from_cart") and self.phase == Phase.BROWSE:
                    self.phase = Phase.HOLD
                self.history.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(observation),
                    }
                )
            extra = reply.get("user_message")
            if extra:
                content = extra["content"] if isinstance(extra, dict) else str(extra)
                noted = self._shopper_shift(str(content))
                self.history.append({"role": "user", "content": content})
                if noted:
                    self.history.append(
                        {
                            "role": "user",
                            "content": "Shopper update received. Re-read current state before closing.",
                        }
                    )
            if shifted:
                self.history.append(
                    {
                        "role": "user",
                        "content": "A listing you already saw changed price or stock. Re-check it.",
                    }
                )
            if any((row.get("observation") or {}).get("done") for row in reply["calls"]):
                break
            step += 1
        return self.log


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        return ShopRunner(problem_data).run()
    except Exception as exc:
        return [{"role": "assistant", "content": f"Unhandled agent error: {type(exc).__name__}: {exc}"}]


__all__ = ["agent_main"]
