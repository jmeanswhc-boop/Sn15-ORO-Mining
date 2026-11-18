#!/usr/bin/env python3
"""Static and public-metadata analysis of the KYO/Miyagi corpus.

Downloaded agents are parsed as text/AST only and are never imported or run.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 6) if values else None


def normal_lines(text: str) -> set[str]:
    return {
        re.sub(r"\s+", " ", line.strip())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def jac(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def implementation_hash(node: ast.AST) -> str:
    clone = copy.deepcopy(node)
    if isinstance(clone, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        clone.name = "_"
    return digest(ast.dump(clone, include_attributes=False))


def named_hash(node: ast.AST) -> str:
    return digest(ast.dump(node, include_attributes=False))


def definition_map(tree: ast.Module) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = {
                "named": named_hash(node),
                "implementation": implementation_hash(node),
            }
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{member.name}"] = {
                        "named": named_hash(member),
                        "implementation": implementation_hash(member),
                    }
    return result


def string_constants(tree: ast.Module) -> dict[str, str]:
    result: dict[str, str] = {}
    for node in tree.body:
        name = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, value = node.target.id, node.value
        if name and isinstance(value, ast.Constant) and isinstance(value.value, str):
            result[name] = value.value
    return result


def run_summary(version_id: str) -> dict[str, Any]:
    path = ROOT / "api" / "runs" / f"{version_id}.json"
    runs = json.loads(path.read_text()) if path.exists() else []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("is_included") is False or run.get("status") != "SUCCESS":
            continue
        groups[run.get("phase") or "unknown"].append(run)

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [
            float(value)
            for run in items
            for value in [
                (run.get("score_components_summary") or {}).get("mean_aggregate_reward")
            ]
            if isinstance(value, (int, float))
        ]
        costs = [
            float(value)
            for run in items
            for value in [(run.get("sandbox_metadata") or {}).get("inference_cost_usd")]
            if isinstance(value, (int, float))
        ]
        durations = [
            float(value)
            for run in items
            for value in [(run.get("sandbox_metadata") or {}).get("duration_seconds")]
            if isinstance(value, (int, float))
        ]
        calls = [float(run["inference_total"]) for run in items if isinstance(run.get("inference_total"), int)]
        failures = [
            float(run["inference_failure_count"])
            for run in items
            if isinstance(run.get("inference_failure_count"), int)
        ]
        return {
            "runs": len(items),
            "score_mean": mean(scores),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
            "cost_mean_usd": mean(costs),
            "duration_mean_seconds": mean(durations),
            "inference_calls_mean": mean(calls),
            "inference_failures_mean": mean(failures),
            "race_numbers": sorted({r["race_number"] for r in items if r.get("race_number") is not None}),
        }

    result = {phase: summarize(items) for phase, items in groups.items()}
    race_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for run in groups.get("RACE", []):
        if run.get("race_number") is not None:
            race_groups[int(run["race_number"])].append(run)
    result["races"] = {str(number): summarize(items) for number, items in sorted(race_groups.items())}
    return result


def source_path(record: dict[str, Any]) -> Path:
    directory = ROOT / record["directory"]
    direct = directory / "agent.py"
    if direct.exists():
        return direct
    matches = sorted((directory / "source").rglob("*.py"))
    if len(matches) != 1:
        raise RuntimeError(f"cannot select source for {record['agent_version_id']}")
    return matches[0]


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    records = manifest["records"]
    released = [r for r in records if r.get("download_state") == "downloaded"]
    sources: list[dict[str, Any]] = []
    by_id = {r["agent_version_id"]: r for r in records}
    for record in released:
        path = source_path(record)
        text = path.read_text()
        tree = ast.parse(text)
        defs = definition_map(tree)
        constants = string_constants(tree)
        models = sorted(
            {
                value
                for value in re.findall(
                    r"[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.:/-]+)[\"']", text
                )
                if any(tag in value.lower() for tag in ("claude", "gpt", "gemini", "deepseek", "grok"))
            }
        )
        sources.append(
            {
                "agent_version_id": record["agent_version_id"],
                "agent_name": record["agent_name"],
                "version_number": record["version_number"],
                "miner_hotkey": record["miner_hotkey"],
                "submitted_at": record.get("submitted_at"),
                "final_score": record.get("final_score"),
                "race_score": record.get("race_score"),
                "path": str(path.relative_to(ROOT)),
                "sha256": record.get("archive_sha256"),
                "bytes": len(text.encode()),
                "lines": len(text.splitlines()),
                "module_ast_sha256": digest(ast.dump(tree, include_attributes=False)),
                "normal_lines": normal_lines(text),
                "definitions": defs,
                "constants": constants,
                "models": models,
                "runs": run_summary(record["agent_version_id"]),
            }
        )

    # Exact and AST-identical groups.
    exact_groups: dict[str, list[str]] = defaultdict(list)
    ast_groups: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        exact_groups[source["sha256"]].append(source["agent_version_id"])
        ast_groups[source["module_ast_sha256"]].append(source["agent_version_id"])

    # Nearest earlier source for each released version. This is similarity evidence,
    # not proof of authorship or the direction of copying.
    ancestry: list[dict[str, Any]] = []
    chronological = sorted(sources, key=lambda s: (s["submitted_at"] or "", s["agent_version_id"]))
    for index, current in enumerate(chronological):
        candidates = []
        current_named = {(name, value["named"]) for name, value in current["definitions"].items()}
        current_impl = {value["implementation"] for value in current["definitions"].values()}
        current_prompts = {digest(value) for value in current["constants"].values() if len(value) >= 200}
        for prior in chronological[:index]:
            prior_named = {(name, value["named"]) for name, value in prior["definitions"].items()}
            prior_impl = {value["implementation"] for value in prior["definitions"].values()}
            prior_prompts = {digest(value) for value in prior["constants"].values() if len(value) >= 200}
            line_score = jac(current["normal_lines"], prior["normal_lines"])
            named_score = jac(current_named, prior_named)
            impl_score = jac(current_impl, prior_impl)
            prompt_score = jac(current_prompts, prior_prompts)
            composite = 0.45 * line_score + 0.30 * named_score + 0.15 * impl_score + 0.10 * prompt_score
            candidates.append(
                {
                    "agent_version_id": prior["agent_version_id"],
                    "agent_name": prior["agent_name"],
                    "version_number": prior["version_number"],
                    "submitted_at": prior["submitted_at"],
                    "same_hotkey": prior["miner_hotkey"] == current["miner_hotkey"],
                    "line_jaccard": round(line_score, 6),
                    "named_definition_jaccard": round(named_score, 6),
                    "implementation_jaccard": round(impl_score, 6),
                    "prompt_jaccard": round(prompt_score, 6),
                    "composite": round(composite, 6),
                }
            )
        candidates.sort(key=lambda item: item["composite"], reverse=True)
        ancestry.append(
            {
                "agent_version_id": current["agent_version_id"],
                "agent_name": current["agent_name"],
                "version_number": current["version_number"],
                "top_prior_matches": candidates[:5],
            }
        )

    # KYO v1-v6 evolution and component provenance.
    kyo = {
        int(s["version_number"]): s
        for s in sources
        if s["agent_name"] == "KYO" and int(s["version_number"]) <= 6
    }
    kyo_steps = []
    for version in range(1, 7):
        current = kyo[version]
        step: dict[str, Any] = {
            "version": version,
            "agent_version_id": current["agent_version_id"],
            "submitted_at": current["submitted_at"],
            "final_score": current["final_score"],
            "race_score": current["race_score"],
            "lines": current["lines"],
            "bytes": current["bytes"],
            "definition_count": len(current["definitions"]),
            "prompt_sizes": {
                name: len(value)
                for name, value in current["constants"].items()
                if len(value) >= 200
            },
            "models": current["models"],
            "runs": current["runs"],
        }
        if version > 1:
            prior = kyo[version - 1]
            names = set(current["definitions"]) | set(prior["definitions"])
            step["delta_from_previous"] = {
                "added": sorted(set(current["definitions"]) - set(prior["definitions"])),
                "removed": sorted(set(prior["definitions"]) - set(current["definitions"])),
                "changed": sorted(
                    name
                    for name in set(current["definitions"]) & set(prior["definitions"])
                    if current["definitions"][name]["named"] != prior["definitions"][name]["named"]
                ),
                "unchanged": sum(
                    current["definitions"].get(name, {}).get("named")
                    == prior["definitions"].get(name, {}).get("named")
                    for name in names
                    if name in current["definitions"] and name in prior["definitions"]
                ),
                "line_jaccard": round(jac(current["normal_lines"], prior["normal_lines"]), 6),
            }
        kyo_steps.append(step)

    v5, v6 = kyo[5], kyo[6]
    changed_names = sorted(
        set(v6["definitions"]) - set(v5["definitions"])
        | {
            name
            for name in set(v6["definitions"]) & set(v5["definitions"])
            if v6["definitions"][name]["named"] != v5["definitions"][name]["named"]
        }
    )
    component_origins = []
    earlier = [s for s in chronological if (s["submitted_at"] or "") < (v6["submitted_at"] or "")]
    for name in changed_names:
        spec = v6["definitions"][name]
        matches = []
        for source in earlier:
            for other_name, other_spec in source["definitions"].items():
                if other_spec["implementation"] == spec["implementation"]:
                    matches.append(
                        {
                            "agent_version_id": source["agent_version_id"],
                            "agent_name": source["agent_name"],
                            "version_number": source["version_number"],
                            "submitted_at": source["submitted_at"],
                            "definition": other_name,
                            "same_name": other_name == name,
                        }
                    )
        matches.sort(key=lambda item: (item["submitted_at"] or "", item["agent_version_id"]))
        component_origins.append({"definition": name, "earliest_exact_implementation": matches[:5]})

    constant_delta = []
    for name in sorted(set(v5["constants"]) | set(v6["constants"])):
        before, after = v5["constants"].get(name), v6["constants"].get(name)
        if before == after:
            continue
        matches = []
        if after is not None:
            for source in earlier:
                for other_name, value in source["constants"].items():
                    if value == after:
                        matches.append(
                            {
                                "agent_version_id": source["agent_version_id"],
                                "agent_name": source["agent_name"],
                                "version_number": source["version_number"],
                                "submitted_at": source["submitted_at"],
                                "constant": other_name,
                            }
                        )
        matches.sort(key=lambda item: (item["submitted_at"] or "", item["agent_version_id"]))
        constant_delta.append(
            {
                "constant": name,
                "before_chars": len(before or ""),
                "after_chars": len(after or ""),
                "earliest_exact_value": matches[:5],
            }
        )

    day_counts = Counter((r.get("submitted_at") or "unknown")[:10] for r in records)
    hotkey_counts = Counter(r["miner_hotkey"] for r in records)
    name_counts = Counter(r["agent_name"] for r in records)
    release_counts = Counter(r.get("release_state") for r in records)
    payload = {
        "scope": {
            "coldkey": manifest["coldkey"],
            "suite_id": manifest["suite_id"],
            "versions_discovered": len(records),
            "released_sources_analyzed": len(sources),
            "hidden_sources_excluded": len(records) - len(sources),
            "release_counts": dict(release_counts),
            "hotkey_counts": dict(hotkey_counts),
            "name_counts": dict(name_counts),
            "submissions_by_day": dict(sorted(day_counts.items())),
            "first_submission": min(r["submitted_at"] for r in records if r.get("submitted_at")),
            "last_submission": max(r["submitted_at"] for r in records if r.get("submitted_at")),
            "ownership_note": manifest["ownership_note"],
        },
        "source_integrity": {
            "unique_archive_hashes": len(exact_groups),
            "exact_duplicate_groups": [ids for ids in exact_groups.values() if len(ids) > 1],
            "unique_module_ast_hashes": len(ast_groups),
            "ast_duplicate_groups": [ids for ids in ast_groups.values() if len(ids) > 1],
        },
        "source_summaries": [
            {key: value for key, value in source.items() if key not in {"normal_lines", "definitions", "constants"}}
            | {
                "definition_count": len(source["definitions"]),
                "prompt_sizes": {
                    name: len(value)
                    for name, value in source["constants"].items()
                    if len(value) >= 200
                },
            }
            for source in sources
        ],
        "nearest_prior_sources": ancestry,
        "kyo_v1_v6": kyo_steps,
        "kyo_v5_to_v6": {
            "changed_or_added_definitions": changed_names,
            "component_origins": component_origins,
            "constant_delta": constant_delta,
        },
    }
    (ROOT / "static-analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "versions": len(records),
                "released": len(sources),
                "hidden": len(records) - len(sources),
                "unique_source_hashes": len(exact_groups),
                "unique_ast_hashes": len(ast_groups),
                "kyo_v6_changed_or_added": len(changed_names),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
