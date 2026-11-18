#!/usr/bin/env python3
"""Collect released OROBench champion code and public problem records.

The collector intentionally uses only ORO's unauthenticated public API.  It
archives the union of OROBench race winners and public top-agent title holders,
plus every public qualifying suite and every recoverable race task roster.
Race prompts are not public; the API exposes task identifiers but returns a
null query in the miner-safe feedback projection.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


API_BASE = "https://api.oroagents.com"
USER_AGENT = "oro-public-research-archive/1.0"


class CollectionError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    write_bytes(path, json_bytes(value))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    data = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n").encode()
        for row in rows
    )
    write_bytes(path, data)


def request_bytes(
    path_or_url: str,
    *,
    payload: dict[str, Any] | None = None,
    attempts: int = 5,
) -> tuple[bytes, str]:
    url = path_or_url if path_or_url.startswith("http") else f"{API_BASE}{path_or_url}"
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "User-Agent": USER_AGENT,
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=data, headers=headers), timeout=90
            ) as response:
                body = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body, response.geturl()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == attempts:
                raise CollectionError(f"request failed after {attempts} attempts: {url}: {error}")
            time.sleep(min(2 ** (attempt - 1), 16))
    raise AssertionError("unreachable")


def request_json(path: str, *, payload: dict[str, Any] | None = None) -> Any:
    body, _ = request_bytes(path, payload=payload)
    try:
        return json.loads(body)
    except json.JSONDecodeError as error:
        raise CollectionError(f"invalid JSON from {path}: {error}") from error


def slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return result[:80] or "unnamed"


def all_races() -> list[dict[str, Any]]:
    races: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = request_json(f"/v1/public/races/history?limit=100&offset={offset}")
        batch = page["races"]
        races.extend(batch)
        offset += len(batch)
        if offset >= page["total"] or not batch:
            break
    return races


def safe_extract_zip(archive: Path, destination: Path) -> list[str]:
    extracted: list[str] = []
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            member = PurePosixPath(info.filename)
            if member.is_absolute() or ".." in member.parts:
                raise CollectionError(f"unsafe ZIP member in {archive}: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise CollectionError(f"symlink ZIP member in {archive}: {info.filename}")
            target = destination.joinpath(*member.parts)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted.append(member.as_posix())
    return extracted


def collect_source(
    champions_root: Path,
    version_id: str,
    roles: dict[str, Any],
) -> dict[str, Any]:
    detail = request_json(f"/v1/public/agent-versions/{version_id}")
    directory = champions_root / (
        f"{slug(detail['agent_name'])}_v{detail['version_number']}_{version_id}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    artifact = request_json(
        "/v1/public/artifacts/download-url",
        payload={"artifact_type": "AGENT_CODE", "agent_version_id": version_id},
    )
    archive_bytes, resolved_url = request_bytes(artifact["download_url"])
    archive_path = directory / "agent_code.zip"
    raw_path = directory / "agent.py"
    source_dir = directory / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    if archive_bytes.startswith(b"PK\x03\x04"):
        if raw_path.exists():
            raw_path.unlink()
        write_bytes(archive_path, archive_bytes)
        members = safe_extract_zip(archive_path, source_dir)
        python_files = sorted(member for member in members if member.endswith(".py"))
        artifact_file = archive_path.name
        artifact_format = "zip"
        syntax_valid = None
        has_agent_main = None
    else:
        try:
            source_text = archive_bytes.decode("utf-8")
            tree = ast.parse(source_text)
        except (UnicodeDecodeError, SyntaxError) as error:
            raise CollectionError(
                f"released source is neither ZIP nor valid Python: {error}"
            ) from error
        # Historical storage objects are named agent_code.zip but served with
        # gzip Content-Encoding; after standards-compliant decoding the public
        # artifact is the original Python source, not a ZIP container.
        if archive_path.exists():
            archive_path.unlink()
        write_bytes(raw_path, archive_bytes)
        write_bytes(source_dir / "agent.py", archive_bytes)
        members = ["agent.py"]
        python_files = ["agent.py"]
        artifact_file = raw_path.name
        artifact_format = "python"
        syntax_valid = True
        has_agent_main = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "agent_main"
            for node in tree.body
        )
    metadata = {
        "collected_at": utc_now(),
        "api_detail": detail,
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "archive_bytes": len(archive_bytes),
        "artifact_file": artifact_file,
        "artifact_format": artifact_format,
        "zip_members": members,
        "python_files": python_files,
        "syntax_valid": syntax_valid,
        "has_top_level_agent_main": has_agent_main,
        "roles": roles,
        "provenance": {
            "detail_endpoint": f"{API_BASE}/v1/public/agent-versions/{version_id}",
            "artifact_endpoint": f"{API_BASE}/v1/public/artifacts/download-url",
            "artifact_type": "AGENT_CODE",
            "presigned_url_redacted": True,
            "resolved_storage_host": urllib.parse.urlparse(resolved_url).netloc,
        },
    }
    # Never persist the temporary signed URL or its credentials.
    write_json(directory / "metadata.json", metadata)
    return {"directory": directory.name, **metadata}


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def generated_result_rows(response: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_run: dict[str, list[dict[str, Any]]] = {}
    for result in response.get("generated_results", []):
        identity = result.get("identity") or {}
        items_record = result.get("items") or {}
        run_id = identity.get("eval_run_id") or items_record.get("eval_run_id")
        if not run_id:
            continue
        by_run[run_id] = list(items_record.get("items") or [])
    return by_run


def checksums(
    root: Path, included: Iterable[Path], excluded: set[Path]
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    candidates: set[Path] = set()
    for entry in included:
        if entry.is_file():
            candidates.add(entry)
        elif entry.is_dir():
            candidates.update(item for item in entry.rglob("*") if item.is_file())
    for path in sorted(candidates):
        if path in excluded:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((digest, path.relative_to(root).as_posix()))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    champions_root = root / "champions"
    problems_root = root / "problems"
    champions_root.mkdir(parents=True, exist_ok=True)
    problems_root.mkdir(parents=True, exist_ok=True)
    collected_at = utc_now()

    suites = request_json("/v1/public/suites")
    envpack_suites = sorted(
        (suite for suite in suites if suite.get("execution_kind") == "envpack"),
        key=lambda item: item["suite_id"],
    )
    suite_ids = {suite["suite_id"] for suite in envpack_suites}
    races = [race for race in all_races() if race["suite_id"] in suite_ids]
    races.sort(key=lambda item: (item["suite_id"], item["race_number"]))
    top_history = request_json("/v1/public/top/history")
    title_entries = [
        entry for entry in top_history.get("entries", []) if entry["suite_id"] in suite_ids
    ]

    write_json(problems_root / "_api" / "suites.json", suites)
    write_json(problems_root / "_api" / "races-history.json", {"races": races})
    write_json(problems_root / "_api" / "top-history.json", top_history)

    winner_races: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for race in races:
        if race.get("winner_agent_version_id"):
            winner_races[race["winner_agent_version_id"]].append(race)
    title_roles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in title_entries:
        title_roles[entry["agent_version_id"]].append(entry)
    champion_ids = sorted(set(winner_races) | set(title_roles))

    source_records: dict[str, dict[str, Any]] = {}
    source_failures: list[dict[str, str]] = []
    for index, version_id in enumerate(champion_ids, 1):
        print(f"[{index}/{len(champion_ids)}] source {version_id}", flush=True)
        roles = {
            "race_wins": [
                {
                    "race_id": race["race_id"],
                    "suite_id": race["suite_id"],
                    "race_number": race["race_number"],
                    "winner_score": race.get("winner_score"),
                }
                for race in winner_races.get(version_id, [])
            ],
            "top_title_entries": title_roles.get(version_id, []),
        }
        try:
            source_records[version_id] = collect_source(
                champions_root, version_id, roles
            )
        except Exception as error:  # preserve the complete failure ledger
            source_failures.append({"agent_version_id": version_id, "error": str(error)})

    champion_catalog = []
    for version_id in champion_ids:
        record = source_records.get(version_id, {})
        detail = record.get("api_detail", {})
        champion_catalog.append(
            {
                "agent_version_id": version_id,
                "agent_name": detail.get("agent_name"),
                "version_number": detail.get("version_number"),
                "submitted_suite_id": detail.get("suite_id"),
                "race_win_count": len(winner_races.get(version_id, [])),
                "top_title_count": len(title_roles.get(version_id, [])),
                "archive_sha256": record.get("archive_sha256"),
                "directory": record.get("directory"),
                "download_status": "downloaded" if record else "failed",
            }
        )
    write_json(champions_root / "catalog.json", champion_catalog)
    write_csv(
        champions_root / "catalog.csv",
        list(champion_catalog[0]) if champion_catalog else ["agent_version_id"],
        champion_catalog,
    )
    write_json(champions_root / "download_failures.json", source_failures)

    qualifying_count = 0
    qualifying_unique: dict[str, dict[str, Any]] = {}
    suite_catalog: list[dict[str, Any]] = []
    for suite in envpack_suites:
        suite_id = suite["suite_id"]
        response = request_json(f"/v1/public/suites/{suite_id}/problems")
        destination = problems_root / "qualifiers" / f"suite-{suite_id}"
        write_json(destination / "api_response.json", response)
        tasks = response.get("qualifying_tasks") or []
        write_jsonl(destination / "problems.jsonl", tasks)
        qualifying_count += len(tasks)
        for task in tasks:
            qualifying_unique.setdefault(task["task_id"], task)
        suite_catalog.append(
            {
                **suite,
                "public_problem_count": len(response.get("problems") or []),
                "qualifying_task_count": len(tasks),
                "directory": destination.relative_to(root).as_posix(),
                "endpoint": f"{API_BASE}/v1/public/suites/{suite_id}/problems",
            }
        )
    write_json(problems_root / "qualifiers" / "catalog.json", suite_catalog)
    write_jsonl(
        problems_root / "qualifiers" / "all_unique_problems.jsonl",
        (qualifying_unique[key] for key in sorted(qualifying_unique)),
    )

    agent_race_data: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for version_id in sorted(winner_races):
        runs = request_json(f"/v1/public/agent-versions/{version_id}/runs")
        response = request_json(
            f"/v1/public/agent-versions/{version_id}/problems?phase=RACE"
        )
        destination = problems_root / "_api" / "winner-agent-versions" / version_id
        write_json(destination / "runs.json", runs)
        write_json(destination / "problems-race.json", response)
        agent_race_data[version_id] = (runs, response)

    race_catalog: list[dict[str, Any]] = []
    race_failures: list[dict[str, Any]] = []
    total_race_identifiers = 0
    unique_race_identifiers: dict[str, dict[str, Any]] = {}
    for race in races:
        race_id = race["race_id"]
        winner_id = race.get("winner_agent_version_id")
        race_detail = request_json(f"/v1/public/races/{race_id}")
        destination = problems_root / "races" / (
            f"suite-{race['suite_id']}_race-{race['race_number']:03d}_{race_id}"
        )
        write_json(destination / "race_detail.json", race_detail)
        roster: dict[str, dict[str, Any]] = {}
        source_run_ids: list[str] = []
        expected_counts: set[int] = set()
        if winner_id in agent_race_data:
            runs, response = agent_race_data[winner_id]
            run_to_race = {
                run["eval_run_id"]: run.get("race_id")
                for run in runs
                if run.get("phase") == "RACE"
            }
            by_run = generated_result_rows(response)
            for run_id, items in by_run.items():
                if run_to_race.get(run_id) != race_id:
                    continue
                source_run_ids.append(run_id)
                expected_counts.add(len(items))
                for item in items:
                    task_id = item.get("item_id")
                    if not task_id:
                        continue
                    row = roster.setdefault(
                        task_id,
                        {
                            "task_id": task_id,
                            "family": item.get("family"),
                            "query": None,
                            "query_visibility": "withheld_by_public_api",
                            "race_id": race_id,
                            "suite_id": race["suite_id"],
                            "race_number": race["race_number"],
                            "source_agent_version_id": winner_id,
                            "source_eval_run_ids": [],
                        },
                    )
                    row["source_eval_run_ids"].append(run_id)
        rows = [roster[key] for key in sorted(roster)]
        write_jsonl(destination / "problem_roster.jsonl", rows)
        total_race_identifiers += len(rows)
        for row in rows:
            unique_race_identifiers.setdefault(row["task_id"], row)
        metadata = {
            "race": race,
            "public_race_detail_endpoint": f"{API_BASE}/v1/public/races/{race_id}",
            "roster_source_endpoint": (
                f"{API_BASE}/v1/public/agent-versions/{winner_id}/problems?phase=RACE"
                if winner_id
                else None
            ),
            "source_eval_run_ids": sorted(source_run_ids),
            "expected_item_counts_observed": sorted(expected_counts),
            "recovered_task_identifier_count": len(rows),
            "full_prompt_count": 0,
            "limitation": (
                "ORO's public API releases race task identifiers and miner-safe episode "
                "feedback, but the original race query is null/withheld."
            ),
        }
        write_json(destination / "metadata.json", metadata)
        race_catalog.append(
            {
                "race_id": race_id,
                "suite_id": race["suite_id"],
                "race_number": race["race_number"],
                "status": race.get("status"),
                "winner_agent_version_id": winner_id,
                "recovered_task_identifier_count": len(rows),
                "source_run_count": len(source_run_ids),
                "roster_availability": (
                    "recovered"
                    if rows
                    else "not_applicable_cancelled"
                    if race.get("status") == "CANCELLED"
                    else "unavailable"
                ),
                "directory": destination.relative_to(root).as_posix(),
            }
        )
        if not rows and race.get("status") != "CANCELLED":
            race_failures.append(
                {"race_id": race_id, "winner_agent_version_id": winner_id, "error": "no public roster recovered"}
            )
    write_json(problems_root / "races" / "catalog.json", race_catalog)
    write_json(problems_root / "races" / "recovery_failures.json", race_failures)
    write_jsonl(
        problems_root / "races" / "all_unique_task_identifiers.jsonl",
        (unique_race_identifiers[key] for key in sorted(unique_race_identifiers)),
    )

    summary = {
        "snapshot_at": collected_at,
        "scope": "OROBench envpack suites only; legacy ShoppingBench excluded",
        "api_base": API_BASE,
        "envpack_suite_count": len(envpack_suites),
        "envpack_suite_ids": sorted(suite_ids),
        "race_count": len(races),
        "completed_race_count": sum(
            race.get("status") == "RACE_COMPLETE" for race in races
        ),
        "cancelled_race_count": sum(race.get("status") == "CANCELLED" for race in races),
        "unique_race_winner_versions": len(winner_races),
        "unique_top_title_versions": len(title_roles),
        "unique_champion_union_versions": len(champion_ids),
        "champion_sources_downloaded": len(source_records),
        "champion_source_failures": len(source_failures),
        "qualifying_problem_rows": qualifying_count,
        "unique_qualifying_task_ids": len(qualifying_unique),
        "race_task_identifier_rows": total_race_identifiers,
        "unique_race_task_ids": len(unique_race_identifiers),
        "races_with_recovered_rosters": sum(
            entry["recovered_task_identifier_count"] > 0 for entry in race_catalog
        ),
        "race_roster_failures": len(race_failures),
        "race_prompts_publicly_available": 0,
    }
    write_json(root / "orobench_archive_summary.json", summary)

    readme = f"""# OROBench public archive\n\nSnapshot: `{collected_at}`\n\nThis archive was collected only from ORO's unauthenticated public API. Legacy\nShoppingBench suites are excluded. The `champions/` corpus is the union of all\nreleased OROBench race-winner versions and all versions recorded as public top\nagent/title holders. Role mappings and source SHA-256 hashes are in\n`champions/catalog.json`.\n\n## Coverage\n\n- OROBench EnvPack suites: **{len(envpack_suites)}** ({', '.join(map(str, sorted(suite_ids)))})\n- Completed/cancelled race records: **{len(races)}**\n- Unique race-winning versions: **{len(winner_races)}**\n- Unique public top/title versions: **{len(title_roles)}**\n- Unique champion union: **{len(champion_ids)}**; downloaded: **{len(source_records)}**\n- Public qualifying rows: **{qualifying_count}** ({len(qualifying_unique)} unique task IDs)\n- Race rosters recovered: **{len(races) - len(race_failures)}/{len(races)}**\n- Race task-identifier rows: **{total_race_identifiers}** ({len(unique_race_identifiers)} unique IDs)\n\n## Important boundary\n\nPublic qualifying problems include full `goal_text` and are stored by suite in\n`problems/qualifiers/`. ORO keeps race prompts and sealed answers private. Its\nreleased race APIs expose task identifiers and result/feedback metadata, while\nthe feedback `query` field is null. Consequently, `problems/races/` contains\nevery recoverable public race roster with `query: null`; it does not claim that\nwithheld race prompts were downloaded. See each race's `metadata.json` and\n`problems/races/recovery_failures.json`.\n\nThe collector is `scripts/collect_orobench_archive.py`. Re-run it to refresh the\nsnapshot. Presigned storage URLs and temporary credentials are never persisted.\n`SHA256SUMS` covers every generated archive file except itself.\n"""
    completed_race_count = sum(
        race.get("status") == "RACE_COMPLETE" for race in races
    )
    cancelled_race_count = sum(race.get("status") == "CANCELLED" for race in races)
    recovered_race_count = sum(
        entry["recovered_task_identifier_count"] > 0 for entry in race_catalog
    )
    readme = readme.replace(
        f"- Completed/cancelled race records: **{len(races)}**",
        f"- Race records: **{len(races)}** ({completed_race_count} completed; "
        f"{cancelled_race_count} cancelled without a race task roster)",
    ).replace(
        f"- Race rosters recovered: **{len(races) - len(race_failures)}/{len(races)}**",
        f"- Completed-race rosters recovered: **{recovered_race_count}/{completed_race_count}**",
    )
    write_bytes(root / "OROBENCH_ARCHIVE_README.md", readme.encode())

    verification = {
        "snapshot_at": collected_at,
        "passed": not source_failures and not race_failures,
        "checks": {
            "all_champion_sources_downloaded": not source_failures,
            "all_race_rosters_recovered": not race_failures,
            "all_source_artifacts_recognized": all(
                record["artifact_format"] in {"python", "zip"}
                for record in source_records.values()
            ),
            "all_source_artifacts_contain_python": all(
                bool(record["python_files"]) for record in source_records.values()
            ),
            "all_python_sources_define_agent_main": all(
                record["artifact_format"] != "python"
                or record["has_top_level_agent_main"] is True
                for record in source_records.values()
            ),
            "all_qualifier_suites_have_tasks": all(
                entry["qualifying_task_count"] > 0 for entry in suite_catalog
            ),
        },
        "source_failures": source_failures,
        "race_failures": race_failures,
    }
    write_json(root / "orobench_archive_verification.json", verification)
    checksum_path = root / "SHA256SUMS"
    checksum_rows = checksums(
        root,
        {
            champions_root,
            problems_root,
            root / "OROBENCH_ARCHIVE_README.md",
            root / "orobench_archive_summary.json",
            root / "orobench_archive_verification.json",
            root / "scripts" / "collect_orobench_archive.py",
        },
        {
            checksum_path,
        },
    )
    write_bytes(
        checksum_path,
        "".join(f"{digest}  {name}\n" for digest, name in checksum_rows).encode(),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if verification["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
