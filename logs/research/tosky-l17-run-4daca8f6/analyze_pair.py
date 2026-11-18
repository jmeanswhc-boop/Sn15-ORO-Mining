"""Read-only comparison of supplied practice runs; never invokes an agent or API."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
RUNS = {"before": "local-9f2ac835e808440197a18a9a65313580",
        "after": "local-4daca8f69b5c4cb28b9c14da6a468b7c"}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def first_inference(row):
    return next(c for c in row["_shadow_proxy_calls"] if c.get("path") == "/inference/chat/completions")


def load(run):
    path = ROOT / "LV_logs" / run
    summary = json.loads((path / "summary.json").read_text())
    problems = read(path / "problems.jsonl")
    pid = {p["environment"]["binding"]["session_id"]: p["problem_id"] for p in problems}
    sandbox = {r["problem_id"]: r for r in read(path / "sandbox/sandbox_output.jsonl")}
    episodes = [r["episode"] for r in read(path / "episode_results.jsonl")]
    return summary, {e["task_id"]: (e, sandbox[pid[e["session_id"]]]) for e in episodes}


def view(episode, row):
    cleaned = [{k: v for k, v in d.items() if k != "extra_info"} for d in row["dialogue"]]
    return {"score": episode["verdict"]["score"], "category": episode["verdict"].get("category"),
            "step_count": episode["step_count"], "execution_seconds": row["execution_time"],
            "usage": row["_shadow_inference_usage"], "dialogue": cleaned}


def timeline(episode, row):
    lines = ["TASK " + episode["task_id"], "QUERY " + episode["request_text"],
             "VERDICT " + json.dumps(episode["verdict"], ensure_ascii=False)]
    step = 0
    for call in row["_shadow_proxy_calls"]:
        body = call.get("json_data", {})
        response = call.get("response") or {}
        if call.get("path") == "/environment/call":
            step = body.get("step", step)
            lines += ["\nENVIRONMENT STEP " + str(step), json.dumps(body.get("action"), ensure_ascii=False),
                      "OBSERVATION " + json.dumps(response, ensure_ascii=False)]
        elif call.get("path") == "/inference/chat/completions":
            system = (body.get("messages") or [{}])[0].get("content", "")
            if isinstance(system, list):
                system = "".join(x.get("text", "") for x in system if isinstance(x, dict))
            lines.append("\nMODEL AFTER STEP %s %s %s" % (step, body.get("model"), str(system).splitlines()[0]))
            for choice in response.get("choices") or []:
                message = choice.get("message") or {}
                lines += [str(message.get("content") or ""), json.dumps(message.get("tool_calls") or [], ensure_ascii=False)]
    return "\n".join(lines) + "\n"


def main():
    loaded = {label: load(run) for label, run in RUNS.items()}
    previous, current = loaded["before"][1], loaded["after"][1]
    assert previous.keys() == current.keys()
    assert loaded["before"][0]["pack_sha256"] == loaded["after"][0]["pack_sha256"]
    tasks = {}
    transitions = Counter()
    totals = {}
    for label, (summary, joined) in loaded.items():
        use = Counter()
        kinds = Counter()
        for _, row in joined.values():
            for key in ("inference_total", "inference_failed", "inference_cost_usd", "prompt_tokens", "cached_tokens", "completion_tokens"):
                use[key] += row["_shadow_inference_usage"].get(key, 0)
            kinds.update(d.get("kind") for d in row["dialogue"])
        totals[label] = {"run": RUNS[label], "score": summary["aggregate_score"],
                         "agent_sha256": summary["agent_sha256"], "usage": dict(use),
                         "median_execution_seconds": statistics.median(row["execution_time"] for _, row in joined.values()),
                         "mean_execution_seconds": statistics.mean(row["execution_time"] for _, row in joined.values()),
                         "total_steps": sum(e["step_count"] for e, _ in joined.values()), "event_counts": dict(kinds)}
    for task in previous:
        a, b = previous[task], current[task]
        assert a[0]["request_text"] == b[0]["request_text"]
        transition = "%d -> %d" % (a[0]["verdict"]["score"], b[0]["verdict"]["score"])
        transitions[transition] += 1
        tasks[task] = {"request": a[0]["request_text"], "transition": transition,
                       "identical_first_inference_request": first_inference(a[1])["json_data"] == first_inference(b[1])["json_data"],
                       "before": view(*a), "after": view(*b)}
        if transition != "1 -> 1":
            for label, pair in (("before", a), ("after", b)):
                (DEST / (task + "-" + label + ".txt")).write_text(timeline(*pair))
    report = {"pack_sha256": loaded["after"][0]["pack_sha256"], "same_task_roster_and_queries": True,
              "first_inference_requests_identical": sum(t["identical_first_inference_request"] for t in tasks.values()),
              "transitions": dict(transitions), "totals": totals, "tasks": tasks,
              "current_source_sha256": hashlib.sha256((ROOT / "agents/Tosky_l17.py").read_bytes()).hexdigest()}
    (DEST / "paired-analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
