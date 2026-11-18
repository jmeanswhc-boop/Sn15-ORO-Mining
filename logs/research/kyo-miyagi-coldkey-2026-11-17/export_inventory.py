#!/usr/bin/env python3
"""Export the coldkey census as a flat, auditable CSV inventory.

Only public metadata is included for hidden versions. Source-derived columns are
populated exclusively when the corresponding public artifact was released and
downloaded.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def display_agent(row: dict | None) -> str:
    if not row:
        return ""
    return f"{row['agent_name']} v{row['version_number']} [{row['agent_version_id']}]"


def main() -> None:
    manifest = load_json("manifest.json")
    static = load_json("static-analysis.json")
    champion_comparisons = load_json("prior-champion-comparison.json")

    summaries = {
        row["agent_version_id"]: row for row in static["source_summaries"]
    }
    nearest = {
        row["agent_version_id"]: row for row in static["nearest_prior_sources"]
    }
    champions = {
        row["agent_version_id"]: row for row in champion_comparisons
    }

    fields = [
        "agent_name",
        "version_number",
        "agent_version_id",
        "miner_hotkey",
        "submitted_at",
        "code_available_at",
        "release_state",
        "download_state",
        "qualifier_score",
        "published_race_score",
        "run_count",
        "source_lines",
        "source_bytes",
        "definition_count",
        "source_sha256",
        "module_ast_sha256",
        "nearest_earlier_org_source",
        "nearest_earlier_org_composite",
        "nearest_earlier_org_same_hotkey",
        "nearest_public_prior_champion",
        "nearest_public_prior_champion_composite",
        "nearest_public_prior_champion_line_jaccard",
    ]

    rows = []
    for record in sorted(manifest["records"], key=lambda row: row["submitted_at"]):
        version_id = record["agent_version_id"]
        summary = summaries.get(version_id)

        internal = None
        internal_entry = nearest.get(version_id)
        if internal_entry and internal_entry["top_prior_matches"]:
            internal = internal_entry["top_prior_matches"][0]

        champion = None
        champion_entry = champions.get(version_id)
        if champion_entry and champion_entry["top_prior_champions"]:
            champion = champion_entry["top_prior_champions"][0]

        rows.append(
            {
                "agent_name": record["agent_name"],
                "version_number": record["version_number"],
                "agent_version_id": version_id,
                "miner_hotkey": record["miner_hotkey"],
                "submitted_at": record["submitted_at"],
                "code_available_at": record["code_available_at"],
                "release_state": record["release_state"],
                "download_state": record["download_state"],
                "qualifier_score": record["final_score"],
                "published_race_score": record["race_score"],
                "run_count": record["run_count"],
                "source_lines": summary["lines"] if summary else "",
                "source_bytes": summary["bytes"] if summary else "",
                "definition_count": summary["definition_count"] if summary else "",
                "source_sha256": summary["sha256"] if summary else "",
                "module_ast_sha256": summary["module_ast_sha256"] if summary else "",
                "nearest_earlier_org_source": display_agent(internal),
                "nearest_earlier_org_composite": (
                    internal["composite"] if internal else ""
                ),
                "nearest_earlier_org_same_hotkey": (
                    internal["same_hotkey"] if internal else ""
                ),
                "nearest_public_prior_champion": display_agent(champion),
                "nearest_public_prior_champion_composite": (
                    champion["composite"] if champion else ""
                ),
                "nearest_public_prior_champion_line_jaccard": (
                    champion["line_jaccard"] if champion else ""
                ),
            }
        )

    output = ROOT / "inventory.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
