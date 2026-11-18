#!/usr/bin/env python3
"""Static, non-executing comparison of the released PIRATE-family sources."""

from __future__ import annotations

import ast
import collections
import difflib
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
SOURCES = {
    "CAIRN_v2": PROJECT / "champions/Cairn_v2_06aef3d7-bf0f-779c-8000-ee9b2a46ce76/agent.py",
    "GIYU_v17": PROJECT / "champions/Giyu_v17_06af0c4d-6ea3-7367-8000-bf906703c8c6/agent.py",
    "SPIRIT_v1": ROOT / "sources/SPIRIT_v1_06af33fd-4d66-73a7-8000-eaf571549005/agent.py",
    "SPIRIT_v2": ROOT / "sources/SPIRIT_v2_06af49bd-3f34-7b79-8000-eb63d5af6047/agent.py",
    "KUNGFU_v1": ROOT / "sources/KUNGFU_v1_06af49ca-fb47-73c3-8000-0a26e1097af4/agent.py",
    "PIRATE_v1": ROOT / "sources/PIRATE_v1_06af6030-67a0-754e-8000-beacfd741adb/agent.py",
}


def digest(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode()
    return hashlib.sha256(value).hexdigest()


def clean_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines() if line.strip()]


def definition_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    found: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = node
        elif isinstance(node, ast.ClassDef):
            found[node.name] = node
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}.{child.name}"] = child
    return found


def source_segment(lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])


def scalar_constants(tree: ast.Module) -> dict[str, Any]:
    constants: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        value = node.value
        if not isinstance(target, ast.Name):
            continue
        if isinstance(value, ast.Constant) and isinstance(value.value, (str, int, float, bool, type(None))):
            constants[target.id] = value.value
        elif isinstance(value, (ast.Tuple, ast.List)):
            try:
                constants[target.id] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
    return constants


def imports(tree: ast.Module) -> list[str]:
    result: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return result


def calls(tree: ast.Module) -> collections.Counter[str]:
    result: collections.Counter[str] = collections.Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            result[function.id] += 1
        elif isinstance(function, ast.Attribute):
            parts = [function.attr]
            value = function.value
            while isinstance(value, ast.Attribute):
                parts.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                parts.append(value.id)
            result[".".join(reversed(parts))] += 1
    return result


def describe(name: str, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode()
    lines = text.splitlines()
    tree = ast.parse(text)
    definitions = definition_nodes(tree)
    constants = scalar_constants(tree)
    prompts = {
        key: {"characters": len(value), "sha256": digest(value)}
        for key, value in constants.items()
        if isinstance(value, str) and len(value) >= 200
    }
    class_methods: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_methods[node.name] = [
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
    return {
        "name": name,
        "path": str(path.relative_to(PROJECT)),
        "bytes": len(raw),
        "lines": len(lines),
        "sha256": digest(raw),
        "tree": tree,
        "text": text,
        "source_lines": lines,
        "clean_lines": clean_lines(text),
        "definitions": definitions,
        "definition_hashes": {
            key: digest(ast.dump(node, include_attributes=False))
            for key, node in definitions.items()
        },
        "definition_lines": {
            key: getattr(node, "end_lineno", getattr(node, "lineno", 1)) - getattr(node, "lineno", 1) + 1
            for key, node in definitions.items()
        },
        "imports": imports(tree),
        "constants": constants,
        "prompts": prompts,
        "class_methods": class_methods,
        "calls": calls(tree),
        "ast_nodes": collections.Counter(type(node).__name__ for node in ast.walk(tree)),
    }


def ratio(a: Any, b: Any) -> float:
    return round(difflib.SequenceMatcher(None, a, b, autojunk=False).ratio(), 6)


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a_defs = a["definitions"]
    b_defs = b["definitions"]
    shared = sorted(set(a_defs) & set(b_defs))
    exact = [key for key in shared if a["definition_hashes"][key] == b["definition_hashes"][key]]
    changed: list[dict[str, Any]] = []
    for key in shared:
        if key in exact:
            continue
        left = source_segment(a["source_lines"], a_defs[key])
        right = source_segment(b["source_lines"], b_defs[key])
        changed.append({
            "definition": key,
            "similarity": ratio(clean_lines(left), clean_lines(right)),
            "a_lines": a["definition_lines"][key],
            "b_lines": b["definition_lines"][key],
        })
    changed.sort(key=lambda item: (item["similarity"], item["definition"]))
    a_line_set = set(a["clean_lines"])
    b_line_set = set(b["clean_lines"])
    prompt_names = sorted(set(a["prompts"]) | set(b["prompts"]))
    prompt_changes: list[dict[str, Any]] = []
    for key in prompt_names:
        left = a["constants"].get(key)
        right = b["constants"].get(key)
        if left == right:
            state = "same"
            similarity = 1.0
        elif not isinstance(left, str):
            state = "added"
            similarity = None
        elif not isinstance(right, str):
            state = "removed"
            similarity = None
        else:
            state = "changed"
            similarity = ratio(left.splitlines(), right.splitlines())
        prompt_changes.append({
            "name": key,
            "state": state,
            "similarity": similarity,
            "a_chars": len(left) if isinstance(left, str) else None,
            "b_chars": len(right) if isinstance(right, str) else None,
        })
    return {
        "a": a["name"],
        "b": b["name"],
        "clean_line_sequence_similarity": ratio(a["clean_lines"], b["clean_lines"]),
        "exact_clean_line_jaccard": round(
            len(a_line_set & b_line_set) / max(1, len(a_line_set | b_line_set)), 6
        ),
        "shared_definitions": len(shared),
        "exact_definitions": len(exact),
        "exact_definition_share": round(len(exact) / max(1, len(shared)), 6),
        "only_a_definitions": sorted(set(a_defs) - set(b_defs)),
        "only_b_definitions": sorted(set(b_defs) - set(a_defs)),
        "changed_definitions": changed,
        "prompt_changes": prompt_changes,
    }


def serialise(description: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in description.items() if key not in {
        "tree", "text", "source_lines", "clean_lines", "definitions", "constants", "calls", "ast_nodes"
    }}
    constants = description["constants"]
    result["configuration"] = {
        key: value
        for key, value in constants.items()
        if not isinstance(value, str) or len(value) < 200
    }
    result["call_counts"] = dict(description["calls"].most_common())
    result["ast_node_counts"] = dict(description["ast_nodes"].most_common())
    return result


def main() -> None:
    descriptions = {name: describe(name, path) for name, path in SOURCES.items()}
    comparisons: list[dict[str, Any]] = []
    names = list(descriptions)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            comparisons.append(compare(descriptions[left], descriptions[right]))
    report = {
        "generated_from_static_source_only": True,
        "sources": {name: serialise(value) for name, value in descriptions.items()},
        "comparisons": comparisons,
    }
    (ROOT / "static-analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
