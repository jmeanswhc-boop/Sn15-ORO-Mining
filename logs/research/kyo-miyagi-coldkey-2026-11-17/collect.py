#!/usr/bin/env python3
"""Collect the public KYO/Miyagi coldkey corpus without executing agents.

Only suite 160 leaderboard entries whose hotkey is one of the ten hotkeys
observed in the coldkey's finalized-chain OwnedHotkeys value are admitted.
Agent code is requested only when the version status says RELEASED. Temporary
signed object-storage URLs are never written to disk.
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
SUITE_ID = 160
COLDKEY = "5E7QbBiYZWfW9X5ZTN6pBJFtbJignvKHS7ufvJc9RuziD7i1"
HOTKEYS = (
    "5F2GYd9TurPAaKshPn97xetfPzj2yUBoqNR9NG6YcAFu8jwH",
    "5GxwAarfPmctFUedJ897C658ysfXvZTWev6CkksxbGxWKj3J",
    "5E7hJqqWYARRrg3SJeQPyhmfyjvFy5VTSU548dx1DM8w9M5Z",
    "5Ct25h2R9uXD2zXMzYACXxbpZPr8XMm6X6LztDKm4rUZDLw7",
    "5EEj8R5aWwDp6XyX7D8KLirZdpZWqWj9LA3rAe5QYVbmA5yk",
    "5CLjFrXqwgCDFGetZaqoynuYddrtS7u5ohKXRPsnE2RPfmn7",
    "5HjEAFb7zXvknvS1wp7M9WuAquDdcxocbm9rduccc8YjEvKV",
    "5E7ToJVkfTVzUYAmpLbAH8WNkpLiFLk3onUKckVkXRVv7Uki",
    "5E5BFs2hLCRDuXUc8gDojja86AaKDReXYNa9rkM55mULcW9C",
    "5F6p6V8wnBSLL1FSBiHWL3NJr1HgQ4ZJKsfpGHvx7dzBk2KE",
)
USER_AGENT = "oro-public-kyo-miyagi-coldkey-study/1.0"
MAX_BYTES = 8 * 1024 * 1024
API_INTERVAL = 0.82
_next_api_call = 0.0


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
    global _next_api_call
    is_api = not path_or_url.startswith("https://") or path_or_url.startswith(API)
    url = path_or_url if path_or_url.startswith("https://") else API + path_or_url
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {
        "Accept": "application/json" if accept_json else "application/octet-stream",
        "Accept-Encoding": "gzip" if accept_json else "identity",
        "User-Agent": USER_AGENT,
    }
    if body is not None:
        headers["Content-Type"] = "application/json"
    for attempt in range(6):
        if is_api:
            delay = _next_api_call - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            _next_api_call = time.monotonic() + API_INTERVAL
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, data=body, headers=headers), timeout=60
            ) as response:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("response exceeded 8 MiB")
                response_headers = {key.lower(): value for key, value in response.headers.items()}
                if response_headers.get("content-encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw, response_headers, response.geturl()
        except urllib.error.HTTPError as error:
            if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else min(2**attempt, 16))
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 16))
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


def valid_existing(directory: Path, version_id: str) -> dict[str, Any] | None:
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("api_detail", {}).get("agent_version_id") != version_id:
        return None
    artifact = directory / metadata["artifact_file"]
    if not artifact.is_file():
        return None
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest != metadata.get("archive_sha256"):
        raise ValueError(f"existing artifact hash mismatch: {directory}")
    return metadata


def collect_artifact(
    entry: dict[str, Any], status: dict[str, Any]
) -> dict[str, Any]:
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
        "download_state": "release_gated",
    }
    if status.get("release_state") != "RELEASED":
        return result

    previous = valid_existing(directory, version_id)
    if previous is not None:
        result.update(
            download_state="downloaded",
            archive_sha256=previous["archive_sha256"],
            archive_bytes=previous["archive_bytes"],
            directory=str(directory.relative_to(ROOT)),
            reused=True,
        )
        return result

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
        temporary = directory.with_name(f".{directory.name}.incoming")
        if temporary.exists():
            shutil.rmtree(temporary)
        source_dir = temporary / "source"
        source_dir.mkdir(parents=True)
        if raw.startswith(b"PK\x03\x04"):
            atomic_bytes(temporary / "agent_code.zip", raw)
            members = safe_extract(raw, source_dir)
            artifact_file = "agent_code.zip"
            syntax_valid = None
            agent_main = None
        else:
            text = raw.decode("utf-8")
            tree = ast.parse(text)
            atomic_bytes(temporary / "agent.py", raw)
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
        atomic_json(temporary / "metadata.json", metadata)
        if directory.exists():
            raise ValueError(f"incomplete existing directory requires review: {directory}")
        temporary.replace(directory)
        result.update(
            download_state="downloaded",
            archive_sha256=digest,
            archive_bytes=len(raw),
            directory=str(directory.relative_to(ROOT)),
            reused=False,
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
                else "error"
            ),
            error_code=error_code,
            http_status=error.code,
        )
    return result


def main() -> None:
    collected_at = now()
    entries: dict[str, dict[str, Any]] = {}
    snapshots: list[dict[str, Any]] = []
    for hotkey in HOTKEYS:
        query = urllib.parse.urlencode(
            {"suite_id": SUITE_ID, "q": hotkey, "limit": 100, "offset": 0}
        )
        page = get_json("/leaderboard?" + query)
        exact = [item for item in page.get("entries", []) if item.get("miner_hotkey") == hotkey]
        if len(exact) != page.get("total"):
            raise RuntimeError(f"search for {hotkey} returned non-exact entries")
        snapshots.append(
            {"suite_id": SUITE_ID, "hotkey": hotkey, "total": len(exact), "entries": exact}
        )
        for item in exact:
            entries[item["agent_version_id"]] = item
        print(json.dumps({"hotkey": hotkey, "versions": len(exact)}), flush=True)
    atomic_json(
        ROOT / "api" / "leaderboards.json",
        {"fetched_at": collected_at, "suite_id": SUITE_ID, "snapshots": snapshots},
    )
    print(json.dumps({"versions_discovered": len(entries)}), flush=True)

    manifest_path = ROOT / "manifest.json"
    prior_records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        prior_records = {r["agent_version_id"]: r for r in old.get("records", [])}
    records = prior_records

    ordered = sorted(
        entries.items(), key=lambda item: item[1].get("eligible_at") or item[0]
    )
    for index, (version_id, entry) in enumerate(ordered, 1):
        try:
            status = get_json(f"/agent-versions/{version_id}/status")
            runs = get_json(f"/agent-versions/{version_id}/runs")
            atomic_json(ROOT / "api" / "status" / f"{version_id}.json", status)
            atomic_json(ROOT / "api" / "runs" / f"{version_id}.json", runs)
            record = collect_artifact(entry, status)
            record.update(
                submitted_at=status.get("submitted_at"),
                final_score=status.get("final_score"),
                race_score=status.get("race_score"),
                eliminated_at=status.get("eliminated_at"),
                run_count=len(runs) if isinstance(runs, list) else len(runs.get("items", [])),
            )
        except Exception as error:
            record = {
                "agent_version_id": version_id,
                "agent_name": entry.get("agent_name"),
                "version_number": entry.get("version_number"),
                "miner_hotkey": entry.get("miner_hotkey"),
                "download_state": "error",
                "error": f"{type(error).__name__}: {error}",
            }
        records[version_id] = record
        payload = {
            "collected_at": collected_at,
            "updated_at": now(),
            "suite_id": SUITE_ID,
            "coldkey": COLDKEY,
            "hotkeys": HOTKEYS,
            "ownership_note": "Current finalized-chain association; not submission-time proof.",
            "versions_discovered": len(entries),
            "records": [records[key] for key, _ in ordered if key in records],
        }
        atomic_json(manifest_path, payload)
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(ordered)}",
                    "agent_name": record.get("agent_name"),
                    "version_number": record.get("version_number"),
                    "download_state": record.get("download_state"),
                    "release_state": record.get("release_state"),
                    "error": record.get("error"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
