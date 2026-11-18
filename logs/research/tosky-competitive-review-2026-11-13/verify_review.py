"""Independently verify this offline review against its local evidence."""
import ast
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
CONTROLS = ROOT.parent / "tosky-v2-2026-11-13"
SOURCE = PROJECT / "agents/TOSKY.py"
EXPECTED_HASH = "31b083e8ac95bc6863a4a01f9768a61879914fed4eb9116ea0f27b394d825a18"


def read(path):
    value = json.loads(path.read_text())
    return value.get("data", value) if isinstance(value, dict) else value


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


expected = read(CONTROLS / "six-task-public-outcomes.json")
totals = defaultdict(dict)
counts = defaultdict(dict)
records = []
for folder in sorted((CONTROLS / "public-agents").iterdir()):
    if not folder.is_dir():
        continue
    for path in sorted((folder / "feedback").glob("*.json")):
        record = read(path)
        task = record["task_id"]
        name = folder.name
        assert task in expected and name in expected[task], (task, name)
        totals[task][name] = totals[task].get(name, 0) + int(record["paid_reward"] == 1)
        counts[task][name] = counts[task].get(name, 0) + 1
        records.append(str(path.relative_to(PROJECT)))

assert len(records) == 72, len(records)
assert dict(totals) == expected, (dict(totals), expected)
assert all(n == 2 for row in counts.values() for n in row.values())
write("nonchampion-recount.json", {
    "feedback_records": len(records),
    "passing_executions_out_of_two": dict(totals),
    "execution_counts": dict(counts),
    "matches_previous_analysis": True,
    "source_paths": records,
})

raw = SOURCE.read_bytes()
source_hash = hashlib.sha256(raw).hexdigest()
assert source_hash == EXPECTED_HASH, source_hash
tree = ast.parse(raw.decode("utf-8"))
compile(tree, str(SOURCE), "exec")
assert any(isinstance(node, ast.FunctionDef) and node.name == "agent_main" for node in tree.body)
evidence = read(ROOT / "review-evidence.json")
assert evidence["source"]["sha256"] == source_hash
assert len(evidence["public_qualifier_matrix"]) == 30
assert sum(row["explicit_cheapest"] for row in evidence["public_qualifier_matrix"]) == 9
assert sum(phase["executions"] for agent in evidence["feedback_recount"].values() for phase in agent.values()) == 720
assert evidence["new_offline_probes"]["assertion_scenarios"] == 10
assert evidence["new_offline_probes"]["actual_network_calls"] == 0
assert len(evidence["public_limit_flows"]["flows"]) == 4
assert evidence["paid_calls"] == 0 and evidence["agent_modified"] is False

verification = {
    "verified_at": datetime.now(timezone.utc).isoformat(),
    "source_sha256": source_hash,
    "source_unchanged": True,
    "source_bytes": len(raw),
    "utf8_ast_compile_and_sync_entrypoint": "passed",
    "nonchampion_records_independently_recounted": 72,
    "nonchampion_recount_matches_prior_analysis": True,
    "main_comparison_records_recounted_by_review": 720,
    "public_tasks_reviewed": 30,
    "new_focused_offline_scenarios": 10,
    "full_controller_offline_flows": 4,
    "paid_calls": 0,
    "server_admission_or_score_tested": False,
    "live_metadata_refresh": "not executed; automatic approval review rejected scope",
}
write("verification.json", verification)
manifest = {
    "created_at": verification["verified_at"],
    "source_sha256": source_hash,
    "files": [
        {"name": path.name, "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(ROOT.iterdir())
        if path.is_file() and path.name != "manifest.json"
    ],
}
write("manifest.json", manifest)
print(json.dumps(verification, indent=2))
