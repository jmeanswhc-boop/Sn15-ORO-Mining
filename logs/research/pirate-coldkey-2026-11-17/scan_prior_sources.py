#!/usr/bin/env python3
"""Find the closest locally archived source public before SPIRIT v1."""

from __future__ import annotations

import ast
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
TARGETS = {
    "spirit1": (
        ROOT / "sources/SPIRIT_v1_06af33fd-4d66-73a7-8000-eaf571549005/agent.py",
        dt.datetime.fromisoformat("2026-11-10T16:01:25.842485"),
        ROOT / "prior-source-scan.json",
    ),
    "kungfu": (
        ROOT / "sources/KUNGFU_v1_06af49ca-fb47-73c3-8000-0a26e1097af4/agent.py",
        dt.datetime.fromisoformat("2026-11-11T16:49:52.507254"),
        ROOT / "prior-source-scan-kungfu.json",
    ),
}


def definitions(text: str) -> dict[str, str]:
    tree = ast.parse(text)
    result: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        elif isinstance(node, ast.ClassDef):
            result[node.name] = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = f"{node.name}.{child.name}"
                    result[name] = hashlib.sha256(ast.dump(child, include_attributes=False).encode()).hexdigest()
    return result


def released_at(metadata: dict) -> dt.datetime | None:
    detail = metadata.get("api_detail") or metadata
    value = detail.get("code_available_at") or detail.get("released_at")
    if not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", choices=TARGETS, nargs="?", default="spirit1")
    args = parser.parse_args()
    target, cutoff, output = TARGETS[args.target]
    target_text = target.read_text()
    target_defs = definitions(target_text)
    target_lines = {line.strip() for line in target_text.splitlines() if line.strip()}
    by_hash: dict[str, dict] = {}
    errors: list[dict] = []
    for parent in (PROJECT / "champions", PROJECT / "challengers"):
        for metadata_path in parent.glob("*/metadata.json"):
            source = metadata_path.parent / "agent.py"
            if not source.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
                release = released_at(metadata)
                if release is None or release > cutoff:
                    continue
                raw = source.read_bytes()
                source_hash = hashlib.sha256(raw).hexdigest()
                if source_hash in by_hash:
                    by_hash[source_hash]["paths"].append(str(source.relative_to(PROJECT)))
                    continue
                text = raw.decode()
                source_defs = definitions(text)
                common = set(target_defs) & set(source_defs)
                exact = {name for name in common if target_defs[name] == source_defs[name]}
                source_lines = {line.strip() for line in text.splitlines() if line.strip()}
                detail = metadata.get("api_detail") or metadata
                by_hash[source_hash] = {
                    "sha256": source_hash,
                    "paths": [str(source.relative_to(PROJECT))],
                    "release_at": release.isoformat(),
                    "agent_version_id": detail.get("agent_version_id"),
                    "agent_name": detail.get("agent_name"),
                    "version_number": detail.get("version_number"),
                    "bytes": len(raw),
                    "definitions": len(source_defs),
                    "common_definitions": len(common),
                    "exact_definitions": len(exact),
                    "target_exact_share": round(len(exact) / len(target_defs), 6),
                    "source_exact_share": round(len(exact) / max(1, len(source_defs)), 6),
                    "line_jaccard": round(
                        len(target_lines & source_lines) / max(1, len(target_lines | source_lines)), 6
                    ),
                }
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, SyntaxError) as error:
                errors.append({"path": str(source.relative_to(PROJECT)), "error": type(error).__name__})
    ranked = sorted(
        by_hash.values(),
        key=lambda row: (row["exact_definitions"], row["line_jaccard"], row["common_definitions"]),
        reverse=True,
    )
    report = {
        "cutoff": cutoff.isoformat(),
        "target": str(target.relative_to(PROJECT)),
        "target_definitions": len(target_defs),
        "unique_released_sources_scanned": len(by_hash),
        "errors": errors,
        "closest": ranked[:50],
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
