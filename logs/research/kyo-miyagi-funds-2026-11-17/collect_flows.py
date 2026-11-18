#!/usr/bin/env python3
"""Collect public on-chain fund flows for the KYO/Miyagi coldkey.

The collector is read-only and resumable. It uses the public OnFinality archive
endpoint, records only decoded chain data, and never handles wallet secrets.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from substrateinterface import SubstrateInterface
import websocket


ROOT = Path(__file__).resolve().parent
# The network-operated archive node is the authoritative fallback for historical
# reads.  Public endpoints are rate-limited, so every phase remains resumable.
ENDPOINT = "wss://archive.chain.opentensor.ai:443"
HTTP_ENDPOINT = "https://bittensor-finney.api.onfinality.io/public"
START_BLOCK = 9_350_000  # predates the coldkey's first nonzero state
CHUNK_BLOCKS = 10_000
NETUID = 15

TARGET = "5E7QbBiYZWfW9X5ZTN6pBJFtbJignvKHS7ufvJc9RuziD7i1"
HOTKEYS = [
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
]

# SN15 owner lineage, confirmed by on-chain coldkey swaps and current storage.
OWNER_COLDKEYS = {
    "original": "5GE9r7GMtDyfbsp5RKr2V8M5PaYJ7pgF9KBu6oBcRiYjZPCc",
    "intermediate": "5GMtrkkvybJiWpo5boj5oVfKmGVgcrghwuubAytsny5LzycW",
    "current": "5EHdase3QHY1FSU2KoxzpbnkzGhxBZEVcWU85zyjrxG62KTK",
}
OWNER_HOTKEY = "5DnqbBi4tYQZVR8DySJvmQ4gcEN94piu8hH6NiwnFzQxT5FW"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.incoming")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def connect() -> SubstrateInterface:
    return SubstrateInterface(
        url=ENDPOINT,
        ss58_format=42,
        ws_options={"timeout": 180},
    )


def retry(call, attempts: int = 5):
    last = None
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # network/indexer retries are checkpointed
            last = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(2 ** attempt, 12))
    raise last  # type: ignore[misc]


def http_rpc(method: str, params: list[Any]) -> Any:
    payload = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    ).encode()
    request = urllib.request.Request(
        HTTP_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read())
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result.get("result")


def http_rpc_batch(requests: list[tuple[str, list[Any]]]) -> list[Any]:
    payload = json.dumps(
        [
            {"jsonrpc": "2.0", "id": index, "method": method, "params": params}
            for index, (method, params) in enumerate(requests)
        ]
    ).encode()
    request = urllib.request.Request(
        HTTP_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        result = json.loads(response.read())
    ordered = sorted(result, key=lambda row: row["id"])
    if any(row.get("error") for row in ordered):
        raise RuntimeError([row.get("error") for row in ordered if row.get("error")])
    return [row.get("result") for row in ordered]


def ws_rpc_batch(requests: list[tuple[str, list[Any]]]) -> list[Any]:
    payload = [
        {"jsonrpc": "2.0", "id": index, "method": method, "params": params}
        for index, (method, params) in enumerate(requests)
    ]
    connection = websocket.create_connection(ENDPOINT, timeout=240)
    try:
        connection.send(json.dumps(payload))
        result = json.loads(connection.recv())
    finally:
        connection.close()
    if isinstance(result, dict):
        raise RuntimeError(result.get("error") or result)
    ordered = sorted(result, key=lambda row: row["id"])
    if any(row.get("error") for row in ordered):
        raise RuntimeError([row.get("error") for row in ordered if row.get("error")])
    return [row.get("result") for row in ordered]


def normalize(value: Any) -> Any:
    if hasattr(value, "value"):
        return normalize(value.value)
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, bytes):
        return "0x" + value.hex()
    if isinstance(value, Decimal):
        return str(value)
    return value


def contains(value: Any, address: str) -> bool:
    if value == address:
        return True
    if isinstance(value, dict):
        return any(contains(item, address) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(contains(item, address) for item in value)
    return False


def call_timestamp(block: dict[str, Any]) -> str | None:
    for extrinsic in block.get("extrinsics", []):
        value = normalize(extrinsic)
        call = value.get("call") or {}
        if call.get("call_module") != "Timestamp":
            continue
        args = call.get("call_args") or []
        for arg in args:
            if arg.get("name") == "now" and isinstance(arg.get("value"), int):
                stamp = datetime.fromtimestamp(arg["value"] / 1000, timezone.utc)
                return stamp.isoformat().replace("+00:00", "Z")
    return None


def safe_float_rao(value: Any) -> int:
    if not isinstance(value, dict):
        return 0
    mantissa = int(value.get("mantissa") or 0)
    exponent = int(value.get("exponent") or 0)
    number = Decimal(mantissa) * (Decimal(10) ** exponent)
    return int(number)


def decode_account_raw(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    data = bytes.fromhex(raw.removeprefix("0x"))
    if len(data) != 56:
        raise ValueError(f"unexpected System.Account length: {len(data)}")
    return {
        "nonce": int.from_bytes(data[0:4], "little"),
        "consumers": int.from_bytes(data[4:8], "little"),
        "providers": int.from_bytes(data[8:12], "little"),
        "sufficients": int.from_bytes(data[12:16], "little"),
        "data": {
            "free": int.from_bytes(data[16:24], "little"),
            "reserved": int.from_bytes(data[24:32], "little"),
            "frozen": int.from_bytes(data[32:40], "little"),
            "flags": int.from_bytes(data[40:56], "little"),
        },
    }


def decode_safe_float_raw(raw: str | None) -> dict[str, int]:
    if raw is None:
        return {"mantissa": 0, "exponent": 0}
    data = bytes.fromhex(raw.removeprefix("0x"))
    if len(data) != 24:
        raise ValueError(f"unexpected SafeFloat length: {len(data)}")
    return {
        "mantissa": int.from_bytes(data[0:16], "little", signed=True),
        "exponent": int.from_bytes(data[16:24], "little", signed=True),
    }


def query_change_chunks(
    chain: SubstrateInterface,
    keys: list[str],
    label: str,
    head_number: int,
) -> list[dict[str, Any]]:
    path = ROOT / f"{label}-changes.json"
    state = load_json(path, {"completed_through": START_BLOCK - 1, "changes": []})
    changes = {row["block_hash"] + ":" + row["storage_key"]: row for row in state["changes"]}
    completed_ranges = [tuple(pair) for pair in state.get("completed_ranges", [])]
    old_through = int(state.get("completed_through", START_BLOCK - 1))
    if old_through >= START_BLOCK and not completed_ranges:
        completed_ranges.append((START_BLOCK, old_through))

    ranges = []
    start = START_BLOCK
    while start <= head_number:
        end = min(start + CHUNK_BLOCKS - 1, head_number)
        if not any(done_start <= start and end <= done_end for done_start, done_end in completed_ranges):
            ranges.append((start, end))
        start = end + 1

    jobs = []
    for start, end in ranges:
        start_hash = retry(lambda start=start: chain.get_block_hash(start))
        end_hash = retry(lambda end=end: chain.get_block_hash(end))
        jobs.append((start, end, start_hash, end_hash))

    def fetch(job):
        start, end, start_hash, end_hash = job
        result = retry(
            lambda: http_rpc("state_queryStorage", [keys, start_hash, end_hash])
        )
        return start, end, result

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(fetch, job) for job in jobs]
        for future in as_completed(futures):
            start, end, result = future.result()
            for change_set in result:
                for storage_key, raw_value in change_set["changes"]:
                    row = {
                        "block_hash": change_set["block"],
                        "storage_key": storage_key,
                        "raw_value": raw_value,
                    }
                    changes[row["block_hash"] + ":" + storage_key] = row
            completed_ranges.append((start, end))
            state = {
                "collected_at": utcnow(),
                "endpoint": ENDPOINT,
                "start_block": START_BLOCK,
                "head_number": head_number,
                "completed_through": max(end for _, end in completed_ranges),
                "completed_ranges": sorted(completed_ranges),
                "changes": list(changes.values()),
            }
            atomic_json(path, state)
            print(f"{label}: blocks {start}-{end}; {len(changes)} changes", flush=True)
    return list(changes.values())


def decode_changes(
    chain: SubstrateInterface,
    changes: list[dict[str, Any]],
    key_objects: dict[str, Any],
    label: str,
) -> list[dict[str, Any]]:
    del chain, key_objects  # values use stable, explicit SCALE layouts in this range
    headers: dict[str, Any] = {}
    hashes = list(dict.fromkeys(row["block_hash"] for row in changes))
    for offset in range(0, len(hashes), 25):
        batch = hashes[offset : offset + 25]
        values = retry(
            lambda batch=batch: ws_rpc_batch(
                [("chain_getHeader", [block_hash]) for block_hash in batch]
            )
        )
        headers.update(zip(batch, values))
    output = []
    for row in changes:
        decoded = (
            decode_account_raw(row["raw_value"])
            if label == "target-account"
            else decode_safe_float_raw(row["raw_value"])
        )
        output.append(
            {
                **row,
                "block_number": int(headers[row["block_hash"]]["number"], 16),
                "decoded": decoded,
            }
        )
    output.sort(key=lambda item: (item["block_number"], item["storage_key"]))
    distinct = []
    previous: dict[str, str | None] = {}
    for row in output:
        key = row["storage_key"]
        if key in previous and previous[key] == row["raw_value"]:
            continue
        previous[key] = row["raw_value"]
        distinct.append(row)
    output = distinct
    atomic_json(ROOT / f"{label}-decoded.json", output)
    return output


def collect_block_details(
    chain: SubstrateInterface, account_changes: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    path = ROOT / "target-block-details.json"
    cached = load_json(path, [])
    by_hash = {row["block_hash"]: row for row in cached}
    hashes = sorted(
        {row["block_hash"] for row in account_changes},
        key=lambda block_hash: next(
            row["block_number"] for row in account_changes if row["block_hash"] == block_hash
        ),
    )

    watched = {TARGET, *OWNER_COLDKEYS.values(), OWNER_HOTKEY}
    jobs = [block_hash for block_hash in hashes if block_hash not in by_hash]
    local = threading.local()

    def worker(block_hash: str) -> dict[str, Any]:
        if not hasattr(local, "chain"):
            local.chain = connect()
        account_row = next(row for row in account_changes if row["block_hash"] == block_hash)
        block = retry(
            lambda: local.chain.get_block(block_hash=block_hash, include_author=False)
        )
        extrinsics = [normalize(item) for item in block.get("extrinsics", [])]
        events = [
            normalize(item)
            for item in retry(lambda: local.chain.get_events(block_hash=block_hash))
        ]

        signed_indices = {
            idx
            for idx, extrinsic in enumerate(extrinsics)
            if extrinsic.get("address") == TARGET
        }
        relevant_events = [
            event
            for event in events
            if event.get("extrinsic_idx") in signed_indices
            or any(contains(event, address) for address in watched)
        ]
        relevant_extrinsics = [
            {"extrinsic_index": idx, **extrinsic}
            for idx, extrinsic in enumerate(extrinsics)
            if idx in signed_indices
            or any(contains(extrinsic, address) for address in watched)
        ]
        return {
            "block_hash": block_hash,
            "block_number": account_row["block_number"],
            "time": call_timestamp(block),
            "account_state": account_row["decoded"],
            "extrinsics": relevant_extrinsics,
            "events": relevant_events,
        }

    completed = len(by_hash)
    # Archive providers meter historical work separately from ordinary RPCs.
    # Keep this conservative; checkpoints make a slower trace preferable to a
    # burst that is rejected halfway through.
    with ThreadPoolExecutor(max_workers=int(os.environ.get("DETAIL_WORKERS", "2"))) as pool:
        futures = [pool.submit(worker, block_hash) for block_hash in jobs]
        for future in as_completed(futures):
            row = future.result()
            by_hash[row["block_hash"]] = row
            completed += 1
            if completed % 5 == 0 or completed == len(hashes):
                ordered = sorted(by_hash.values(), key=lambda item: item["block_number"])
                atomic_json(path, ordered)
                print(f"details: {completed}/{len(hashes)} blocks", flush=True)
    return sorted(by_hash.values(), key=lambda row: row["block_number"])


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    chain = connect()
    head_hash = retry(chain.get_chain_finalised_head)
    head_number = retry(lambda: chain.get_block_number(head_hash))

    current_owner = retry(
        lambda: chain.query("SubtensorModule", "SubnetOwner", [NETUID], block_hash=head_hash).value
    )
    current_owner_hotkey = retry(
        lambda: chain.query(
            "SubtensorModule", "SubnetOwnerHotkey", [NETUID], block_hash=head_hash
        ).value
    )
    owned_hotkeys = retry(
        lambda: chain.query(
            "SubtensorModule", "OwnedHotkeys", [TARGET], block_hash=head_hash
        ).value
    )

    account_key = chain.create_storage_key("System", "Account", [TARGET])
    account_keys = {account_key.to_hex(): account_key}
    account_raw = query_change_chunks(
        chain, list(account_keys), "target-account", head_number
    )
    account_decoded = decode_changes(
        chain, account_raw, account_keys, "target-account"
    )

    alpha_keys = {}
    key_hotkey = {}
    for hotkey in HOTKEYS:
        key = chain.create_storage_key(
            "SubtensorModule", "AlphaV2", [hotkey, TARGET, NETUID]
        )
        alpha_keys[key.to_hex()] = key
        key_hotkey[key.to_hex()] = hotkey
    alpha_raw = query_change_chunks(
        chain, list(alpha_keys), "target-alpha", head_number
    )
    alpha_decoded = decode_changes(chain, alpha_raw, alpha_keys, "target-alpha")
    for row in alpha_decoded:
        row["hotkey"] = key_hotkey[row["storage_key"]]
        row["alpha_rao"] = safe_float_rao(row["decoded"])
    atomic_json(ROOT / "target-alpha-decoded.json", alpha_decoded)

    details = collect_block_details(chain, account_decoded)
    current_account = retry(
        lambda: chain.query("System", "Account", [TARGET], block_hash=head_hash).value
    )
    current_alpha = {
        hotkey: retry(
            lambda hotkey=hotkey: chain.query(
                "SubtensorModule", "AlphaV2", [hotkey, TARGET, NETUID], block_hash=head_hash
            ).value
        )
        for hotkey in HOTKEYS
    }
    atomic_json(
        ROOT / "snapshot.json",
        {
            "collected_at": utcnow(),
            "endpoint": ENDPOINT,
            "finalized_head": head_hash,
            "finalized_height": head_number,
            "start_block": START_BLOCK,
            "target_coldkey": TARGET,
            "target_hotkeys": HOTKEYS,
            "owned_hotkeys_at_head": owned_hotkeys,
            "owner_coldkey_lineage": OWNER_COLDKEYS,
            "owner_hotkey": OWNER_HOTKEY,
            "subnet_owner_at_head": current_owner,
            "subnet_owner_hotkey_at_head": current_owner_hotkey,
            "current_account": current_account,
            "current_alpha_v2": current_alpha,
            "account_change_count": len(account_decoded),
            "alpha_change_count": len(alpha_decoded),
            "detailed_block_count": len(details),
            "qualification": (
                "Public on-chain state only. Wallet relationships do not identify "
                "natural persons or prove common beneficial ownership."
            ),
        },
    )
    print(
        json.dumps(
            {
                "height": head_number,
                "account_changes": len(account_decoded),
                "alpha_changes": len(alpha_decoded),
                "detailed_blocks": len(details),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
