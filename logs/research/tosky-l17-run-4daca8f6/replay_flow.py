"""Offline diagnostic execution against saved responses, with a virtual clock.

No model, environment server, or scorer is contacted. Tool actions and inference
stage sequence must match the recorded attempt; this produces no new score.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agents/tests"))
from test_tosky_l17 import load

RUN = ROOT / "LV_logs/local-4daca8f69b5c4cb28b9c14da6a468b7c"
DEST = Path(__file__).resolve().parent


def rows(path):
    return list(map(json.loads, path.read_text().splitlines()))


def heading(body):
    value = body["messages"][0]["content"]
    if isinstance(value, list):
        value = "".join(x.get("text", "") for x in value if isinstance(x, dict))
    return value.splitlines()[0]


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class RecordedWire:
    def __init__(self, recorded, clock):
        self.calls = recorded
        self.clock = clock
        self.zero = (recorded[0]["timestamp"] - recorded[0]["duration_ms"]) / 1000
        self.index = 0
        self.divergence = None
        self.agent = None
        self.terminal_snapshots = []

    def post_verbose(self, path, body):
        if self.index >= len(self.calls):
            self.divergence = "Extra call after recorded sequence"
            raise AssertionError(self.divergence)
        expected = self.calls[self.index]
        assert path == expected["path"], "Different API stage at index %d" % self.index
        if path == "/environment/call":
            assert body == expected["json_data"], "Different world action at index %d" % self.index
            agent = self.agent
            action = body.get("action") or {}
            if agent and action.get("name") in (agent.roles.get("buy"), agent.roles.get("refuse")):
                ids = agent.cart_ids()
                key = agent.check_key(ids)
                self.terminal_snapshots.append({"step": body["step"], "action": action,
                    "cart_ids": ids, "check_key": key, "checked": key in agent.checked,
                    "review_verdict": agent.verdicts.get(key),
                    "known_sold_out": [i for i in ids if agent.sold_out(i)],
                    "active_requirements": agent.needs_now(), "budget": agent.budget,
                    "in_recovery": agent.in_recovery, "spent": agent.cost,
                    "refused_at": agent.refused_at})
        else:
            assert body["model"] == expected["json_data"]["model"], "Different model at index %d" % self.index
            assert heading(body) == heading(expected["json_data"]), "Different inference role at index %d" % self.index
        self.index += 1
        self.clock.now = max(self.clock.now, expected["timestamp"] / 1000 - self.zero)
        return SimpleNamespace(data=copy.deepcopy(expected["response"]), error=None)


def replay(problem, recorded):
    ns = load()
    clock = Clock()
    ns["time"] = clock
    wire = RecordedWire(recorded["_shadow_proxy_calls"], clock)
    ns["_env"] = wire
    ns["_llm_within"] = lambda timeout: wire
    agent = ns["Cairn"](problem)
    wire.agent = agent
    original = agent.quote_inference
    denials = []

    def quote(body):
        result = original(body)
        if result is None:
            signature, messages = agent.token_prefix(body)
            previous = agent.token_prefixes.get(body.get("model"))
            without_effort = copy.deepcopy(body)
            without_effort.pop("reasoning", None)
            alternative = original(without_effort)
            cold_prompt = (len(ns["_dump"](body).encode("utf-8")) + 1024
                + 64 * len(body.get("messages") or []) + 128 * len(body.get("tools") or []))
            rates = [rate for rate in ns["INFERENCE_RATES"][body["model"]] if rate[0] <= cold_prompt]
            cold_reserved = 1.10 * (cold_prompt * max(rate[1] for rate in rates)
                                   + body["max_tokens"] * max(rate[2] for rate in rates))
            denials.append({"model": body.get("model"), "at": agent.clock(), "spent": agent.cost,
                "remaining_total": ns["EPISODE_USD"] - agent.cost,
                "remaining_regular": ns["EPISODE_USD"] - agent.cost - ns["FINISH_USD"],
                "reasoning": body.get("reasoning"),
                "body_bytes": len(ns["_dump"](body).encode("utf-8")),
                "cold_prompt_bound": cold_prompt, "cold_reserved_usd": cold_reserved,
                "previous_prompt_tokens": previous[2] if previous else None,
                "matching_previous_messages": bool(previous and messages[:len(previous[1])] == previous[1]),
                "matching_previous_signature": bool(previous and signature == previous[0]),
                "quote_without_effort": alternative,
                "in_recovery": agent.in_recovery})
        return result

    agent.quote_inference = quote
    error = None
    try:
        trace = agent.run()
    except Exception as exc:
        trace = agent.trace
        error = "%s: %s" % (type(exc).__name__, exc)
    return {"recorded_calls": len(wire.calls), "consumed_calls": wire.index,
            "complete_recorded_sequence": wire.index == len(wire.calls), "error": error,
            "crashes": [d for d in trace if d.get("kind") == "crash"],
            "final_step": agent.step, "finished": agent.finished, "cost": agent.cost,
            "budget_denials": denials,
            "terminal_snapshots": wire.terminal_snapshots,
            "last_events": trace[-8:]}


def main():
    summaries = json.loads((RUN / "summary.json").read_text())
    assert hashlib.sha256((ROOT / "agents/Tosky_l17.py").read_bytes()).hexdigest() == summaries["agent_sha256"]
    episodes = [row["episode"] for row in rows(RUN / "episode_results.jsonl")]
    problems = {row["environment"]["binding"]["session_id"]: row for row in rows(RUN / "problems.jsonl")}
    sandbox = {row["problem_id"]: row for row in rows(RUN / "sandbox/sandbox_output.jsonl")}
    wanted = set(sys.argv[1:]) or {"t-5a9d1682ae", "t-abf1c45838", "t-82a9890683"}
    results = {}
    for episode in episodes:
        if "all" not in wanted and episode["task_id"] not in wanted:
            continue
        problem = problems[episode["session_id"]]
        result = replay(problem, sandbox[problem["problem_id"]])
        results[episode["task_id"]] = result
        print(episode["task_id"], {k: result[k] for k in ("complete_recorded_sequence", "consumed_calls", "recorded_calls", "final_step", "finished", "error")})
    (DEST / "flow-replay.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
