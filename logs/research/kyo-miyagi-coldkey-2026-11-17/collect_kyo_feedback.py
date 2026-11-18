#!/usr/bin/env python3
"""Collect redacted public feedback aggregates for KYO v5 R7 vs v6 R8.

The public endpoints return full released trajectories. This script deliberately
persists only verdict categories and numeric diagnostics, never task text,
queries, shopper messages, tool arguments, observations, or interactions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from collect import atomic_json, get_json, now


ROOT = Path(__file__).resolve().parent
TARGETS = {
    "KYO_v5_race7": (
        "06ae242f-2942-79a9-8000-56cc25202293",
        7,
    ),
    "KYO_v6_race8": (
        "06ae3865-3cdc-7ce7-8000-220bd4158d7c",
        8,
    ),
}


def safe_feedback(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": data.get("passed"),
        "paid_reward": data.get("paid_reward"),
        "failure_category": data.get("failure_category"),
        "termination_code": data.get("termination_code"),
        "outcome": data.get("outcome"),
        "step_count": data.get("step_count"),
        "tool_call_count": data.get("tool_call_count"),
        "search_call_count": data.get("search_call_count"),
        "observable_issues": data.get("observable_issues") or [],
        "diagnostics_state": data.get("diagnostics_state"),
    }


def aggregate(items: list[dict[str, Any]], expected: int) -> dict[str, Any]:
    def average(key: str) -> float | None:
        values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
        return round(sum(values) / len(values), 6) if values else None

    return {
        "expected_items": expected,
        "feedback_items": len(items),
        "categories": dict(Counter(item.get("failure_category") or "passed" for item in items)),
        "terminations": dict(Counter(item.get("termination_code") or "unknown" for item in items)),
        "observable_issues": dict(
            Counter(issue for item in items for issue in item.get("observable_issues") or [])
        ),
        "mean_steps": average("step_count"),
        "mean_tool_calls": average("tool_call_count"),
        "mean_search_calls": average("search_call_count"),
    }


def main() -> None:
    progress_path = ROOT / "kyo-feedback-progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    report: dict[str, Any] = {"collected_at": now(), "privacy": "aggregate diagnostics only", "targets": {}}
    for label, (version_id, race_number) in TARGETS.items():
        runs = json.loads((ROOT / "api" / "runs" / f"{version_id}.json").read_text())
        selected = [
            run
            for run in runs
            if run.get("phase") == "RACE"
            and run.get("race_number") == race_number
            and run.get("status") == "SUCCESS"
            and run.get("is_included") is not False
            and isinstance((run.get("score_components_summary") or {}).get("mean_aggregate_reward"), (int, float))
        ]
        target_rows = []
        target_items: list[dict[str, Any]] = []
        for run in selected:
            run_id = run["eval_run_id"]
            detail = get_json(f"/evaluation-runs/{run_id}")
            atomic_json(ROOT / "api" / "run-details" / f"{run_id}.json", detail)
            episode_items = ((detail.get("items") or {}).get("items") or [])
            safe_items: list[dict[str, Any]] = []
            for index, item in enumerate(episode_items, 1):
                episode_id = item.get("episode_result_id")
                key = f"{run_id}:{episode_id}"
                if key not in progress:
                    feedback = get_json(f"/episode-results/{episode_id}/feedback")
                    progress[key] = safe_feedback(feedback)
                    atomic_json(progress_path, progress)
                safe_items.append(progress[key])
                if index % 10 == 0 or index == len(episode_items):
                    print(json.dumps({"target": label, "run": run_id, "progress": f"{index}/{len(episode_items)}"}), flush=True)
            target_items.extend(safe_items)
            target_rows.append(
                {
                    "eval_run_id": run_id,
                    "validator_score": run.get("validator_score"),
                    "inference_cost_usd": (run.get("sandbox_metadata") or {}).get("inference_cost_usd"),
                    "duration_seconds": (run.get("sandbox_metadata") or {}).get("duration_seconds"),
                    "inference_total": run.get("inference_total"),
                    "inference_failure_count": run.get("inference_failure_count"),
                    "aggregate": aggregate(safe_items, len(episode_items)),
                }
            )
        report["targets"][label] = {
            "agent_version_id": version_id,
            "race_number": race_number,
            "runs": target_rows,
            "combined": aggregate(target_items, sum(r["aggregate"]["expected_items"] for r in target_rows)),
        }
        atomic_json(ROOT / "kyo-feedback-summary.json", report)


if __name__ == "__main__":
    main()
