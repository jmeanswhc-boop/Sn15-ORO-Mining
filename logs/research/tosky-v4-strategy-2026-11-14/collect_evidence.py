"""Recount existing public/local evidence; no network, inference, or agent imports."""
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
FOUNDATION = ROOT / "logs/research/tosky-v4-foundation-2026-11-14"
DONORS = ROOT / "logs/research/tosky-v2-2026-11-13"
INPUTS = {}


def read(path):
    data = path.read_bytes()
    INPUTS[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data.decode()


def load(path):
    return json.loads(read(path))


def jsonl(path):
    return [json.loads(line) for line in read(path).splitlines() if line.strip()]


def methods(path, names):
    source = read(path)
    tree = ast.parse(source)
    return {n.name: {"start": n.lineno, "end": n.end_lineno}
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names}


def local_evidence(run):
    base = ROOT / "LV_logs" / run
    summary = load(base / "summary.json")
    outputs = jsonl(base / "sandbox/sandbox_output.jsonl")
    counters = Counter()
    episodes_with = Counter()
    for row in outputs:
        kinds = Counter(d.get("kind", d.get("role")) for d in row["dialogue"])
        counters.update(kinds)
        episodes_with.update(kinds.keys())
    latest = {}
    for row in jsonl(base / "sandbox/inference_stats.jsonl"):
        latest[(row["problem_id"], row["instance_id"], row["execution_id"])] = row
    assert len(outputs) == len(latest) == summary["task_count"]
    assert summary["agent_sha256"] == INPUTS["agents/Tosky_v3.py"]
    return {"run": run, "source_sha256": summary["agent_sha256"],
            "task_count": summary["task_count"], "score": summary["aggregate_score"],
            "diagnostic_events": dict(counters), "episodes_with_event": dict(episodes_with),
            "typed_review_events": {k: counters[k] for k in
                                    ("review_boundary", "review_boundary_packet", "review_boundary_response")},
            "physical_calls": sum(r["inference_total"] for r in latest.values()),
            "failed_calls": sum(r["inference_failed"] for r in latest.values()),
            "missing_costs": sum(r["inference_cost_missing"] for r in latest.values()),
            "cost_usd": sum(r["inference_cost_usd"] for r in latest.values())}


def donor_evidence(name, controls):
    base = DONORS / "public-agents" / name
    source = base / "source/agent.py"
    provenance = load(base / "source/provenance.json")
    read(source)
    assert INPUTS[str(source.relative_to(ROOT))] == provenance["sha256"]
    included = [r for r in load(base / "runs.json")["data"]
                if r.get("is_included") and r["status"] == "SUCCESS"]
    qualifiers = []
    for run in included:
        if run["phase"] != "QUALIFYING":
            continue
        detail = load(base / "run-details" / (run["eval_run_id"] + ".json"))["data"]
        items = detail["items"]["items"]
        qualifiers.append({"run": run["eval_run_id"], "tasks": len(items),
                           "passed": sum(i["paid_reward"] for i in items),
                           "cost_usd": run["sandbox_metadata"]["inference_cost_usd"],
                           "world": detail["environment"]["catalog_epoch"]})
    targeted = [r for r in controls if r["agent"] == name]
    return {"name": name, "provenance": provenance, "qualifying": qualifiers,
            "qualifying_passed": sum(r["passed"] for r in qualifiers),
            "qualifying_tasks": sum(r["tasks"] for r in qualifiers),
            "targeted_passed": sum(r["paid_reward"] for r in targeted),
            "targeted_executions": len(targeted),
            "historical_races": [{"race": r["race_number"], "run": r["eval_run_id"],
                                  "score": r["score_components_summary"]["mean_aggregate_reward"],
                                  "cost_usd": r["sandbox_metadata"]["inference_cost_usd"],
                                  "inference_errors": r.get("inference_failure_count")}
                                 for r in included if r["phase"] == "RACE"]}


def question_evidence(name, folder, tasks):
    selected = []
    for path in sorted(folder.glob("*.json")):
        # Do not add every scanned feedback file to the input manifest.
        d = json.loads(path.read_text())["data"]
        if d["task_id"] not in tasks or d["phase"] != "QUALIFYING":
            continue
        load(path)
        questions = []
        for turn in d["interaction"]:
            for call in turn.get("calls") or []:
                if call["tool_name"] == "contact_customer":
                    questions.append({"step": turn["step_index"], "args": call["args"],
                                      "observation": call["observation"]})
        selected.append({"agent": name, "task": d["task_id"], "episode": d["episode_result_id"],
                         "passed": d["passed"], "failure_category": d.get("failure_category"),
                         "termination": d["termination_code"], "questions": questions})
    return selected


def main():
    pirate = ROOT / "champions/PIRATE_v1_06af6030-67a0-754e-8000-beacfd741adb/agent.py"
    v3 = ROOT / "agents/Tosky_v3.py"
    read(pirate)
    read(v3)
    assert INPUTS[str(pirate.relative_to(ROOT))] == "ea169847a7c61efc9957d1190e8751e134b6c7bbfb30089284facb979cc22e27"
    assert INPUTS[str(v3.relative_to(ROOT))] == "3b8b73ca3a2675b60fbbac5e86145af3fd3ece52ad69b69557305475b3dc5b50"
    local = [local_evidence(r) for r in ("local-71b9db485e8b423d986df21dacab6c2d",
                                       "local-cb574c03f70841abb0efec38dba1f2f0")]
    controls = load(DONORS / "public-controls-index.json")["controls"]
    donors = [donor_evidence(n, controls) for n in
              ("PRIMO", "Four_Leaf", "Hunter", "Nagini", "KING", "Abacus_r12")]
    public = load(FOUNDATION / "public-analysis.json")
    aggregates = [r for r in public["aggregate"] if r["agent"] == "pirate"]
    rows = [r for r in load(FOUNDATION / "public-task-results.json")
            if r["agent"] == "pirate" and r["phase"] == "RACE"]
    failures = Counter(r["failure"] or "passed" for r in rows)
    hard_tasks = {r["item_id"] for r in controls}
    quals = [r for r in load(FOUNDATION / "public-task-results.json")
             if r["agent"] == "pirate" and r["phase"] == "QUALIFYING" and r["task"] in hard_tasks]
    question_tasks = {"t-20fcb00ff0", "t-6fd2a792df", "t-fc0b16b797"}
    questions = question_evidence("PIRATE", FOUNDATION / "public/pirate/feedback", question_tasks)
    for name in ("PRIMO", "Four_Leaf"):
        questions.extend(question_evidence(name, DONORS / "public-agents" / name / "feedback", question_tasks))
    anchors = methods(pirate, {"think", "call_model", "act", "interview", "ask_next", "settle",
                               "take_heard", "gate", "take_note", "check_cart", "link_mismatch",
                               "scaled_same", "key_verdict", "bound_verdict", "price_swap",
                               "direct_order", "cheap_swap", "complete_cart", "run_calls", "finish"})
    probes = load(FOUNDATION / "offline-contract-probes.json")
    output = {"scope": "Existing released public records and supplied local runs; no new paid test.",
              "snapshot_note": "Race evidence uses 2026-11-14 00:06:31 America/New_York snapshot; donor archive collected 2026-11-13.",
              "pirate_aggregates": aggregates, "pirate_race_categories": dict(failures),
              "pirate_targeted_qualifier": {"tasks": len(quals), "passed": sum(r["reward"] for r in quals)},
              "local_v3": local, "donors": donors, "question_traces": questions,
              "pirate_code_anchors": anchors, "existing_offline_probes": probes,
              "limitations": ["Repeated validator runs are not independent task samples.",
                              "Historical donor races use different task worlds from PIRATE.",
                              "One published failure category does not identify every defect.",
                              "Question-response comparisons are observational, not controlled interventions.",
                              "Public PIRATE feedback lacks model request traces and per-task inference cost."]}
    (OUT / "strategy-evidence.json").write_text(json.dumps(output, indent=2) + "\n")
    (OUT / "input-hashes.json").write_text(json.dumps(INPUTS, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"pirate_race_categories": dict(failures),
                      "pirate_targeted_qualifier": output["pirate_targeted_qualifier"],
                      "typed_review_events": [r["typed_review_events"] for r in local],
                      "donors": [{k: r[k] for k in ("name", "qualifying_passed", "qualifying_tasks", "targeted_passed", "targeted_executions")} for r in donors],
                      "question_trace_count": len(questions), "input_files": len(INPUTS)}, indent=2))


if __name__ == "__main__":
    main()
