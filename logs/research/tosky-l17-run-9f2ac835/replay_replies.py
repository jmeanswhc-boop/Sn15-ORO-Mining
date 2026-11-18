"""Offline state-transition replay of recorded interpretations, not a benchmark rerun.

No new shopper replies or model outputs are generated. The original questions are
fed to each controller, even where the updated controller would ask differently.
"""
import copy
import hashlib
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "agents/tests"))
from test_tosky_l17 import field_text, load

DEST = Path(__file__).resolve().parent
RUN = ROOT / "LV_logs/local-9f2ac835e808440197a18a9a65313580"
BEFORE = Path(sys.argv[1]) if len(sys.argv) > 1 else DEST / "baseline-source.txt"
AFTER = ROOT / "agents/Tosky_l17.py"


def text(value):
    if isinstance(value, str):
        return value
    return "".join(x.get("text", "") for x in (value or []) if isinstance(x, dict))


def question_parts(question):
    match = re.fullmatch(r"What (.+) do you need for the (.+)\?", question)
    if not match:
        raise AssertionError("Unexpected question structure")
    properties = re.split(r", | or ", match[1])
    return properties, match[2]


def state(ns, brief, request, label, properties, order, parsed, reply):
    agent = ns["Cairn"]({"environment": {"binding": {"session_id": "offline-replay"},
                         "policy_view": {"query": request, "max_steps": 100, "tools": []}}})
    agent.brief = copy.deepcopy(brief)
    agent.roles = copy.deepcopy(brief.get("roles", {}))
    agent.budget = brief.get("budget")
    agent.msg_limit = brief.get("messages") or 3
    agent.items = [ns["Item"](row["name"], row.get("shelf", ""), row.get("quantity", 1),
                             copy.deepcopy(row.get("musts", []))) for row in brief["items"]]
    engine = ns["BodyAsk"](agent)
    engine.brief()
    assert len(engine.items) == 1, "Replay must handle multiple products explicitly"
    item = engine.items[0]
    item.label = label
    item.order = list(dict.fromkeys(order + properties))
    item.scores = {key: 3 for key in item.order}
    item.carry = {key: 1 for key in item.order}
    engine.think = lambda *args: copy.deepcopy(parsed)
    item.last = list(properties)
    item.asked = list(properties)
    item.questions = 1
    msg = {"kind": "list", "parts": [(item, properties)]}
    got = engine.hear(msg, engine.wording(msg), reply)
    following = engine.next_message()
    following_properties = [key for _, keys in following["parts"] for key in keys] if following else []
    return {"done": sorted(item.done), "open": item.open_props(), "next": following_properties,
            "needs": agent.needs_now(), "flexible": sorted(agent.items[0].free),
            "closed": getattr(item, "closed", False), "scope": got.get("indifference_scope"),
            "budget": agent.budget}


def main():
    old, new = load(BEFORE), load(AFTER)
    sandbox = [json.loads(line) for line in (RUN / "sandbox/sandbox_output.jsonl").read_text().splitlines()]
    problems = {row["problem_id"]: row for row in
                map(json.loads, (RUN / "problems.jsonl").read_text().splitlines())}
    counts = Counter()
    cases = []
    firm = []
    actual_pairs = []
    budget_replies = []
    totals = Counter()
    for row in sandbox:
        problem = problems[row["problem_id"]]
        usage = row["_shadow_inference_usage"]
        for key in ("inference_total", "inference_failed", "inference_cost_usd", "prompt_tokens", "cached_tokens", "completion_tokens"):
            totals[key] += usage.get(key, 0)
        brief = next(d["brief"] for d in row["dialogue"] if d.get("kind") == "brief")
        interview = next((d for d in row["dialogue"] if d.get("kind") == "sendo_ask"), {})
        order = [entry[0] for rank in interview.get("ranks", []) for entry in rank["order"]]
        recorded = []
        for call in row["_shadow_proxy_calls"]:
            messages = (call.get("json_data") or {}).get("messages", [])
            if not messages or not text(messages[0].get("content")).startswith("# Interpret the shopper reply"):
                continue
            prompt = text(messages[-1]["content"])
            question = field_text(prompt, "question")
            reply = field_text(prompt, "shopper_reply")
            content = call["response"]["choices"][0]["message"]["content"]
            parsed = new["_sd_json_in"](content, "kind")
            assert parsed, "Recorded interpretation must parse"
            properties, label = question_parts(question)
            assert new["SendoAsk"]._join(properties) == question[5:].split(" do you need for the ", 1)[0]
            before = state(old, brief, problem["query"], label, properties, order, parsed, reply)
            after = state(new, brief, problem["query"], label, properties, order, parsed, reply)
            counts["interpretations"] += 1
            counts[parsed["kind"]] += 1
            record = {"problem_id": row["problem_id"], "question": question, "reply": reply,
                      "parsed": parsed, "before": before, "after": after}
            cases.append(record)
            recorded.append((properties, parsed, reply))
            if parsed["kind"] == "indifferent":
                overlap_old = [p for p in properties if p in before["next"]]
                overlap_new = [p for p in properties if any(new["_sd_same_topic"](p, q) for q in after["next"])]
                counts["old_next_repeats_answered_question"] += bool(overlap_old)
                counts["new_next_repeats_answered_question"] += bool(overlap_new)
                if after["scope"] == "question":
                    assert not overlap_new, "A settled question topic recurred"
                else:
                    counts["named_indifference_cases"] += 1
                if re.search(r"budget|spend|total|working with|cap|ceiling|limit", reply, re.I):
                    budget_replies.append({"problem_id": row["problem_id"], "reply": reply,
                                           "recorded_interpretation": parsed,
                                           "budget_before": before["budget"], "budget_after": after["budget"]})
            if parsed.get("told"):
                assert before["needs"] == after["needs"], "Firm disclosure changed"
                firm.append({"problem_id": row["problem_id"], "told": parsed["told"], "preserved": True})
        for (props, parsed, reply), (following, _, _) in zip(recorded, recorded[1:]):
            if parsed["kind"] == "indifferent":
                overlap = [p for p in props if p in following]
                actual_pairs.append({"problem_id": row["problem_id"], "overlap": len(overlap),
                                     "previous_question_topics": len(props)})

    counts["question_scope_indifference_cases"] = counts["indifferent"] - counts["named_indifference_cases"]
    counts["settled_topic_repetition_in_question_scope_cases"] = sum(
        any(new["_sd_same_topic"](p, q) for p in question_parts(c["question"])[0] for q in c["after"]["next"])
        for c in cases if c["parsed"]["kind"] == "indifferent" and c["after"]["scope"] == "question")
    report = {"method": "Independent transitions under recorded model interpretations; no new inference, shopper replies or score.",
              "before_sha256": hashlib.sha256(BEFORE.read_bytes()).hexdigest(),
              "after_sha256": hashlib.sha256(AFTER.read_bytes()).hexdigest(),
              "counts": dict(counts), "firm_disclosures": firm,
              "actual_indifference_followups": {"pairs": len(actual_pairs),
                   "pairs_with_exact_topic_repetition": sum(x["overlap"] > 0 for x in actual_pairs),
                   "repeated_topic_mentions": sum(x["overlap"] for x in actual_pairs)},
              "budget_replies": budget_replies, "recorded_run_totals": dict(totals),
              "execution_seconds": {"min": min(r["execution_time"] for r in sandbox),
                   "median": statistics.median(r["execution_time"] for r in sandbox),
                   "max": max(r["execution_time"] for r in sandbox)},
              "limits": ["Question policy and model prompt change require a new paid run to measure accuracy.",
                   "Old model replies do not contain the new independent budget field; replay cannot validate new extraction.",
                   "Original questions are replayed separately; this is not a hypothetical full conversation or benchmark score."]}
    (DEST / "reply-replay-cases.json").write_text(json.dumps(cases, indent=2) + "\n")
    (DEST / "reply-replay-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
