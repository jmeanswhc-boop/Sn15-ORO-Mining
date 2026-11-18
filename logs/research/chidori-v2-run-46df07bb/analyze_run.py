"""Offline analysis of provided verification artifacts; never calls a service."""
import ast
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN_ID = "local-46df07bb11a34426b51f83c581c88575"
RUN = ROOT / "LV_logs" / RUN_ID
SOURCE = ROOT / "agents/ICDI_chidori_v2.py"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("chidori_run_analysis", SOURCE)
agent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(agent)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episodes(run):
    return {row["episode"]["task_id"]: row["episode"] for row in read_jsonl(run / "episode_results.jsonl")}


def requests(run, session):
    return [d for d in read_jsonl(run / "sandbox" / ("request_log_" + session + ".jsonl")) if d.get("kind") == "summary"]


def message_text(message):
    content = message.get("content")
    return "\n".join(p.get("text", "") for p in content if isinstance(p, dict)) if isinstance(content, list) else content or ""


def metrics(run):
    summary = json.loads((run / "summary.json").read_text())
    episode_map = episodes(run)
    logs = [d for e in episode_map.values() for d in requests(run, e["session_id"])]
    model_logs = [d for d in logs if d.get("path") == "/inference/chat/completions"]
    durations = [x["execution_time"] for x in read_jsonl(run / "sandbox/sandbox_output.jsonl")]
    costs = [d["response"]["usage"]["cost"] for d in model_logs if isinstance(d.get("response"), dict) and isinstance(d["response"].get("usage"), dict)]
    passed = sum(e["verdict"]["score"] == 1 for e in episode_map.values())
    return {"run_id": run.name, "agent_sha256": summary["agent_sha256"], "pack_sha256": summary["pack_sha256"],
            "passed": passed, "tasks": len(episode_map), "score": summary["aggregate_score"],
            "reported_agent_cost_usd": sum(costs), "cost_per_correct_usd": sum(costs) / passed if passed else None,
            "model_requests": len(model_logs), "failed_model_http_requests": sum(d["status_code"] != 200 for d in model_logs),
            "episode_seconds_median": statistics.median(durations), "episode_seconds_max": max(durations),
            "episode_seconds_sum_not_wall_time": sum(durations)}


summary = json.loads((RUN / "summary.json").read_text())
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == summary["agent_sha256"]
episode_map = episodes(RUN)
by_session = {e["session_id"]: e for e in episode_map.values()}
problem_map = {p["problem_id"]: p for p in read_jsonl(RUN / "problems.jsonl")}
tasks = []
models = collections.defaultdict(lambda: {"calls": 0, "cost": 0.0, "reservation_ratios": [], "byte_token_ratios": []})
empty_outputs = []
probes = []
control_probes = []
totals = collections.Counter()

with patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden")) as network:
    for sandbox in read_jsonl(RUN / "sandbox/sandbox_output.jsonl"):
        session = sandbox["problem_id"]
        episode = by_session[session]
        trace = sandbox["dialogue"]
        reqs = requests(RUN, session)
        mreqs = [d for d in reqs if d.get("path") == "/inference/chat/completions"]
        ereqs = [d for d in reqs if d.get("path") == "/environment/call"]
        budgets = [t for t in trace if t.get("kind") == "inference_budget"]
        stops = [t for t in trace if t.get("kind") == "local_inference_stop"]
        parsed = next(t for t in trace if t.get("kind") == "procedures_parsed")
        notes = []
        finishes = []
        for d in ereqs:
            req = d["json_data"]
            action = req["action"]
            observation = d["response"].get("observation")
            note_lines = []
            if isinstance(observation, dict):
                for key in ("rupimo", "refusal"):
                    value = observation.get(key)
                    if isinstance(value, list):
                        note_lines.extend(x for x in value if isinstance(x, str))
            if note_lines:
                notes.append({"step": req["step"], "tool": action["name"], "notes": note_lines})
            if action["name"] in ("settle_purchase", "give_up_task"):
                finishes.append({"step": req["step"], "tool": action["name"], "args": action["args"],
                                 "finished": d["response"].get("finished"), "notes": note_lines})
        record = {"task_id": episode["task_id"], "session_id": session, "passed": episode["verdict"]["score"] == 1,
                  "category": episode["verdict"].get("category") or "passed", "steps": episode["steps"],
                  "reported_agent_cost_usd": sum(d["response"]["usage"]["cost"] for d in mreqs),
                  "model_calls": len(mreqs), "sandbox_seconds": sandbox["execution_time"],
                  "budget_denials": sum(t.get("kind") == "budget_denied" for t in trace),
                  "local_stop": bool(stops), "first_local_stop_seconds": stops[0]["t"] if stops else None,
                  "procedure_count": parsed["count"], "procedure_parse_valid": parsed["valid"],
                  "forced_orders": sum(t.get("kind") == "forced_order" for t in trace),
                  "forced_declines": sum(t.get("kind") == "forced_decline" for t in trace),
                  "public_notices": notes, "finishes": finishes}
        tasks.append(record)
        totals["local_stops"] += bool(stops)
        totals["failed_with_local_stop"] += bool(stops) and not record["passed"]
        totals["passed_with_local_stop"] += bool(stops) and record["passed"]
        totals["budget_denials"] += record["budget_denials"]
        for b, d in zip(budgets, mreqs):
            assert b["model"] == d["json_data"]["model"]
            model = b["model"]
            cost = d["response"]["usage"]["cost"]
            data = models[model]
            data["calls"] += 1
            data["cost"] += cost
            if cost > 0:
                data["reservation_ratios"].append(b["reserved_usd"] / cost)
            body = d["json_data"]
            if message_text(body["messages"][0]).startswith(agent.PROCEDURE_PROMPT):
                totals["procedure_verification_requests"] += 1
            estimated = len(json.dumps(body, ensure_ascii=False).encode()) + 512 + 64 * len(body["messages"]) + 128 * len(body.get("tools") or [])
            data["byte_token_ratios"].append(estimated / d["response"]["usage"]["prompt_tokens"])
            choice = d["response"]["choices"][0]
            message = choice.get("message") or {}
            if not message.get("content") and not message.get("tool_calls"):
                empty_outputs.append({"task_id": episode["task_id"], "model": model, "cap": body["max_tokens"],
                                      "cost": cost, "finish_reason": choice.get("finish_reason"),
                                      "completion_tokens": d["response"]["usage"]["completion_tokens"],
                                      "reasoning_tokens": d["response"]["usage"].get("completion_tokens_details", {}).get("reasoning_tokens")})
        if stops:
            # Replay only public state and billing. No model reply or score is predicted.
            ep = agent.Episode(problem_map[session])
            brief = next(t["brief"] for t in trace if t.get("kind") == "brief")
            with patch.object(ep, "json_ask", return_value=json.loads(json.dumps(brief))):
                ep.read_brief()
            ep.inference_budget.load_prices(agent.BUDGET_PRICE_CONFIG)
            ep.inference_budget.actual = sum(agent.money_units(d["response"]["usage"]["cost"]) for d in mreqs)
            ep.calls = ep.inference_budget.attempts = len(mreqs)
            ep.no_credit = True
            ep.budget_locally_stopped = True
            ep.room = lambda: 100
            prompt = agent.NOTE_PROMPT + "\n\nITEMS AND REQUIREMENTS NOW:\n" + ep.ledger() + "\n\nTHE NOTICE:\n" + "\n".join(line for entry in notes for line in entry["notes"])
            body = {"model": agent.LIGHT, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2500, "temperature": 0}
            protected = ep.inference_budget.protected if ep.duties or not ep.procedures_valid else 0
            quote, reason = ep.inference_budget.quote(body, protected)
            client = SimpleNamespace(post_verbose=lambda *_: (_ for _ in ()).throw(AssertionError("Unexpected HTTP")))
            with patch.object(agent, "model_client", return_value=client) as factory:
                result = ep.think(body["messages"], [agent.LIGHT], 2500)
            probes.append({"task_id": episode["task_id"], "reported_agent_cost_usd": record["reported_agent_cost_usd"],
                           "remaining_allowance_usd": ep.inference_budget.remaining() / 1e9,
                           "ordinary_available_after_reserve_usd": ep.inference_budget.remaining(protected) / 1e9,
                           "compact_notice_quote_admitted": quote is not None,
                           "compact_notice_reserved_usd": quote["units"] / 1e9 if quote else None,
                           "quote_failure": reason, "think_returned_none": result is None,
                           "model_client_was_called": factory.called})
            if record["category"] == "changes missed":
                lines = [line for entry in finishes for line in entry["notes"]]
                if lines:
                    with patch.object(agent, "model_client", return_value=client) as factory:
                        text = ep.read_note(lines)
                    control_probes.append({"scenario": "notice remembered without interpretation after local stop",
                                           "task_id": episode["task_id"], "notice_added_to_heard": all(line in ep.heard for line in lines),
                                           "notice_interpreted_as_sold_out": ep.note_sold, "returned_text": text,
                                           "model_client_was_called": factory.called})
            if episode["task_id"] == "t-d4e2bc9e66":
                review = next(d for d in reversed(mreqs) if "RECORDS OF THE CARTED LISTINGS:" in message_text(d["json_data"]["messages"][0]))
                blocked_ep = agent.Episode(problem_map[session])
                with patch.object(blocked_ep, "json_ask", return_value=json.loads(json.dumps(brief))):
                    blocked_ep.read_brief()
                blocked_ep.inference_budget.load_prices(agent.BUDGET_PRICE_CONFIG)
                blocked_ep.inference_budget.actual = sum(agent.money_units(d["response"]["usage"]["cost"]) for d in mreqs)
                blocked_ep.calls = blocked_ep.inference_budget.attempts = len(mreqs)
                blocked_ep.room = lambda: 100
                expensive = dict(review["json_data"], model=agent.HEAVY)
                cheap_quote, _ = blocked_ep.inference_budget.quote(review["json_data"])
                with patch.object(agent, "model_client", return_value=client) as factory:
                    heavy_result = blocked_ep.think(expensive["messages"], [agent.HEAVY], 3000)
                    cheap_result = blocked_ep.think(review["json_data"]["messages"], [agent.LIGHT], 3000)
                assert blocked_ep.no_credit and cheap_quote is not None and heavy_result is None and cheap_result is None
                control_probes.append({"scenario": "single-model escalation denial prevents affordable light request",
                                       "task_id": episode["task_id"], "remaining_usd": blocked_ep.inference_budget.remaining() / 1e9,
                                       "cheap_request_reservation_usd": cheap_quote["units"] / 1e9,
                                       "no_credit_after_heavy_denial": blocked_ep.no_credit, "model_client_was_called": factory.called})
                checked = next(t for t in trace if t.get("kind") == "cart_check" and t.get("ok") is False)
                blocked_ep.cart_lines = {item: 1 for item in checked["ids"]}
                key = "|".join(sorted(checked["ids"])) + "#" + str(blocked_ep.version)
                blocked_ep.verdicts[key] = False
                forwarded = []
                def finish_mock(tool, args):
                    forwarded.append(tool)
                    blocked_ep.finished = True
                    return {"finished": True, "observation": {"accepted": True}}
                with patch.object(blocked_ep, "reread", return_value=""), patch.object(blocked_ep, "env", side_effect=finish_mock):
                    blocked_ep.finish()
                assert forwarded == [blocked_ep.roles["buy"]]
                control_probes.append({"scenario": "forced finish can order a cart with a recorded negative product review",
                                       "task_id": episode["task_id"], "product_review_verdict": False, "forwarded_tools": forwarded})
    assert not network.called

comparison = metrics(RUN)
comparison_runs = []
paired = {}
for old_id in ("local-f6809126cbb043e6a3b152a7582d982e", "local-0e47aedee480429ba19d94c1909dd7f3"):
    old_run = ROOT / "LV_logs" / old_id
    old = episodes(old_run)
    m = metrics(old_run)
    assert m["pack_sha256"] == comparison["pack_sha256"]
    assert set(old) == set(episode_map)
    matched = {"gains": [], "losses": [], "both_pass": [], "both_fail": []}
    for task, e in episode_map.items():
        assert old[task]["request_text"] == e["request_text"]
        a, b = old[task]["verdict"]["score"], e["verdict"]["score"]
        key = "gains" if b > a else "losses" if b < a else "both_pass" if b else "both_fail"
        matched[key].append(task)
    paired[old_id] = matched
    comparison_runs.append(m)

for m in models.values():
    m["median_reservation_to_reported_cost"] = statistics.median(m.pop("reservation_ratios"))
    m["median_byte_estimate_to_actual_prompt_tokens"] = statistics.median(m.pop("byte_token_ratios"))

analysis = {"run": comparison, "categories": dict(collections.Counter(t["category"] for t in tasks)),
            "procedure_parse_counts": dict(collections.Counter(t["procedure_count"] for t in tasks)),
            "budget_control_observations": dict(totals), "models": dict(models), "empty_outputs": empty_outputs,
            "compact_notice_admission_probes": probes, "control_flow_probes": control_probes,
            "same_pack_comparisons": comparison_runs,
            "paired_task_results": paired,
            "limits": ["Recorded model costs cover agent calls, not a reconciled shared simulator bill.",
                       "Matched tasks share pack and public request, but episode draws/model outputs are not controlled identical reruns.",
                       "Admission probes verify feasibility of a smaller request, not its answer or eventual task score."]}
(OUT / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
(OUT / "per-task.json").write_text(json.dumps(sorted(tasks, key=lambda t: t["task_id"]), indent=2) + "\n")
print(json.dumps({"score": comparison["score"], "categories": analysis["categories"],
                  "agent_cost": comparison["reported_agent_cost_usd"], "controls": dict(totals),
                  "stopped_tasks_with_affordable_compact_notice": sum(p["compact_notice_quote_admitted"] for p in probes),
                  "probe_count": len(probes), "models": dict(models)}, indent=2))
