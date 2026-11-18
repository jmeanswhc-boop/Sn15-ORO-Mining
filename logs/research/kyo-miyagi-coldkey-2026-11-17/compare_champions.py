#!/usr/bin/env python3
"""Compare organization sources to champions public before each submission."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from analyze import definition_map, jac, normal_lines


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def source_from(directory: Path, metadata: dict[str, Any]) -> Path | None:
    direct = directory / metadata.get("artifact_file", "agent.py")
    if direct.suffix == ".py" and direct.exists():
        return direct
    candidates = sorted((directory / "source").rglob("*.py"))
    return candidates[0] if len(candidates) == 1 else None


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    organization = []
    for record in manifest["records"]:
        if record.get("download_state") != "downloaded":
            continue
        path = ROOT / record["directory"] / "agent.py"
        text = path.read_text()
        defs = definition_map(ast.parse(text))
        organization.append(
            {
                "record": record,
                "lines": normal_lines(text),
                "named_defs": {(name, spec["named"]) for name, spec in defs.items()},
                "impl_defs": {spec["implementation"] for spec in defs.values()},
            }
        )

    champions = []
    for metadata_path in sorted((REPO / "champions").glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        detail = metadata.get("api_detail") or {}
        available = detail.get("code_available_at") or detail.get("released_at")
        path = source_from(metadata_path.parent, metadata)
        if not available or not path:
            continue
        text = path.read_text()
        defs = definition_map(ast.parse(text))
        champions.append(
            {
                "agent_version_id": detail.get("agent_version_id"),
                "agent_name": detail.get("agent_name"),
                "version_number": detail.get("version_number"),
                "code_available_at": available,
                "path": str(path.relative_to(REPO)),
                "lines": normal_lines(text),
                "named_defs": {(name, spec["named"]) for name, spec in defs.items()},
                "impl_defs": {spec["implementation"] for spec in defs.values()},
            }
        )

    output = []
    for item in organization:
        record = item["record"]
        matches = []
        for champion in champions:
            if champion["agent_version_id"] == record["agent_version_id"]:
                continue
            if champion["code_available_at"] > record["submitted_at"]:
                continue
            line_score = jac(item["lines"], champion["lines"])
            named_score = jac(item["named_defs"], champion["named_defs"])
            impl_score = jac(item["impl_defs"], champion["impl_defs"])
            composite = 0.55 * line_score + 0.30 * named_score + 0.15 * impl_score
            matches.append(
                {
                    key: champion[key]
                    for key in (
                        "agent_version_id",
                        "agent_name",
                        "version_number",
                        "code_available_at",
                        "path",
                    )
                }
                | {
                    "line_jaccard": round(line_score, 6),
                    "named_definition_jaccard": round(named_score, 6),
                    "implementation_jaccard": round(impl_score, 6),
                    "composite": round(composite, 6),
                }
            )
        matches.sort(key=lambda value: value["composite"], reverse=True)
        output.append(
            {
                "agent_version_id": record["agent_version_id"],
                "agent_name": record["agent_name"],
                "version_number": record["version_number"],
                "submitted_at": record["submitted_at"],
                "top_prior_champions": matches[:5],
            }
        )
    destination = ROOT / "prior-champion-comparison.json"
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    strong = [
        (item["agent_name"], item["version_number"], item["top_prior_champions"][0])
        for item in output
        if item["top_prior_champions"] and item["top_prior_champions"][0]["composite"] >= 0.35
    ]
    print(json.dumps({"organization_sources": len(output), "prior_champions": len(champions), "strong_matches": len(strong)}, indent=2))


if __name__ == "__main__":
    main()
