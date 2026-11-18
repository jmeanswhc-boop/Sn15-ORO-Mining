#!/usr/bin/env python3
"""Archive public OROBench sources associated with the PIRATE coldkey.

This collector uses only ORO's unauthenticated public API.  It never stores
the temporary signed object-storage URLs returned by the artifact endpoint.
"""

from __future__ import annotations

import ast
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path(__file__).resolve().parent
API = "https://api.oroagents.com/v1/public"
COLDKEY = "5Hb2bcydcr2GyF5546thP2TE2kh1H4xfejWXw77uJGGAYfwz"
HOTKEYS = (
    "5CG5VEaWky1HqNQsJZgKXeqNyLsA3qrXAFtyaW7roWgpWkxp",
    "5EFZuxy9u3pjdGoBqn59NMLLynfTdbwRceW4Wo5phr6siPtu",
)
USER_AGENT = "oro-public-pirate-lineage-study/1.0"
MAX_BYTES = 2 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incoming")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(),
    )


def request(
    path_or_url: str,
    *,
    payload: dict[str, Any] | None = None,
    accept_json: bool = True,
) -> tuple[bytes, dict[str, str], str]:
    url = path_or_url if path_or_url.startswith("https://") else API + path_or_url
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Accept": "application/json" if accept_json else "application/octet-stream",
        "Accept-Encoding": "gzip" if accept_json else "identity",
        "User-Agent": USER_AGENT,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers), timeout=60
            ) as response:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("response exceeded 2 MiB")
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                if response_headers.get("content-encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw, response_headers, response.geturl()
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 4:
                raise
            time.sleep(min(2**attempt, 8))
    raise AssertionError("unreachable")


def get_json(path: str) -> Any:
    raw, _, _ = request(path)
    return json.loads(raw)


def post_json(path: str, payload: dict[str, Any]) -> Any:
    raw, _, _ = request(path, payload=payload)
    return json.loads(raw)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._") or "unnamed"


def safe_extract(raw: bytes, destination: Path) -> list[str]:
    members: list[str] = []
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        for entry in bundle.infolist():
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe ZIP member: {entry.filename}")
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"symlink ZIP member: {entry.filename}")
            target = destination.joinpath(*relative.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            members.append(relative.as_posix())
    return members


def collect_artifact(entry: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    version_id = entry["agent_version_id"]
    directory = ROOT / "sources" / (
        f"{slug(entry['agent_name'])}_v{entry['version_number']}_{version_id}"
    )
    result: dict[str, Any] = {
        "agent_version_id": version_id,
        "agent_name": entry["agent_name"],
        "version_number": entry["version_number"],
        "miner_hotkey": entry["miner_hotkey"],
        "release_state": status.get("release_state"),
        "code_available_at": status.get("code_available_at"),
        "download_state": "unknown",
    }
    try:
        artifact = post_json(
            "/artifacts/download-url",
            {"artifact_type": "AGENT_CODE", "agent_version_id": version_id},
        )
        signed_url = artifact["download_url"]
        parsed = urllib.parse.urlsplit(signed_url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("unsupported artifact URL")
        raw, _, resolved_url = request(signed_url, accept_json=False)
        digest = hashlib.sha256(raw).hexdigest()
        source_dir = directory / "source"
        if directory.exists():
            shutil.rmtree(directory)
        source_dir.mkdir(parents=True)
        if raw.startswith(b"PK\x03\x04"):
            atomic_bytes(directory / "agent_code.zip", raw)
            members = safe_extract(raw, source_dir)
            artifact_file = "agent_code.zip"
            syntax_valid = None
            agent_main = None
        else:
            text = raw.decode("utf-8")
            tree = ast.parse(text)
            atomic_bytes(directory / "agent.py", raw)
            atomic_bytes(source_dir / "agent.py", raw)
            members = ["agent.py"]
            artifact_file = "agent.py"
            syntax_valid = True
            agent_main = any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == "agent_main"
                for node in tree.body
            )
        metadata = {
            "collected_at": now(),
            "api_detail": status,
            "leaderboard_entry": entry,
            "archive_sha256": digest,
            "archive_bytes": len(raw),
            "artifact_file": artifact_file,
            "members": members,
            "syntax_valid": syntax_valid,
            "has_top_level_agent_main": agent_main,
            "provenance": {
                "status_endpoint": f"{API}/agent-versions/{version_id}/status",
                "artifact_endpoint": f"{API}/artifacts/download-url",
                "artifact_type": "AGENT_CODE",
                "presigned_url_redacted": True,
                "resolved_storage_host": urllib.parse.urlsplit(resolved_url).hostname,
            },
        }
        atomic_json(directory / "metadata.json", metadata)
        result.update(
            download_state="downloaded",
            archive_sha256=digest,
            archive_bytes=len(raw),
            directory=str(directory.relative_to(ROOT)),
        )
    except urllib.error.HTTPError as error:
        body = error.read(65536)
        try:
            detail = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            detail = {"http_status": error.code}
        error_code = detail.get("error_code") if isinstance(detail, dict) else None
        result.update(
            download_state=(
                "release_gated"
                if error_code in {"ARTIFACT_NOT_RELEASED", "RACE_IN_FLIGHT"}
                else "not_available"
            ),
            error_code=error_code,
            http_status=error.code,
        )
    return result


def main() -> None:
    collected_at = now()
    suites = get_json("/suites")
    atomic_json(ROOT / "api" / "suites.json", {"fetched_at": collected_at, "data": suites})
    entries: dict[str, dict[str, Any]] = {}
    leaderboard_snapshots: list[dict[str, Any]] = []
    for suite in suites:
        if suite.get("execution_kind") != "envpack":
            continue
        suite_id = suite["suite_id"]
        for hotkey in HOTKEYS:
            query = urllib.parse.urlencode(
                {"suite_id": suite_id, "q": hotkey, "limit": 100, "offset": 0}
            )
            page = get_json("/leaderboard?" + query)
            exact = [item for item in page.get("entries", []) if item.get("miner_hotkey") == hotkey]
            leaderboard_snapshots.append(
                {"suite_id": suite_id, "hotkey": hotkey, "total": len(exact), "entries": exact}
            )
            for item in exact:
                entries[item["agent_version_id"]] = item
            time.sleep(0.25)
    atomic_json(
        ROOT / "api" / "leaderboards.json",
        {"fetched_at": collected_at, "snapshots": leaderboard_snapshots},
    )

    records: list[dict[str, Any]] = []
    for version_id, entry in sorted(entries.items(), key=lambda item: item[1].get("eligible_at") or ""):
        status = get_json(f"/agent-versions/{version_id}/status")
        runs = get_json(f"/agent-versions/{version_id}/runs")
        atomic_json(ROOT / "api" / "status" / f"{version_id}.json", status)
        atomic_json(ROOT / "api" / "runs" / f"{version_id}.json", runs)
        record = collect_artifact(entry, status)
        record["submitted_at"] = status.get("submitted_at")
        record["final_score"] = status.get("final_score")
        record["race_score"] = status.get("race_score")
        record["runs"] = len(runs)
        records.append(record)
        atomic_json(ROOT / "manifest.json", {
            "collected_at": collected_at,
            "updated_at": now(),
            "coldkey": COLDKEY,
            "hotkeys": HOTKEYS,
            "versions_discovered": len(entries),
            "records": records,
        })
        print(json.dumps({key: record.get(key) for key in (
            "agent_name", "version_number", "agent_version_id", "download_state",
            "code_available_at", "archive_bytes"
        )}), flush=True)


if __name__ == "__main__":
    main()
