#!/usr/bin/env python3
"""Record the current finalized-chain KYO/Miyagi coldkey association."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from substrateinterface import SubstrateInterface


ROOT = Path(__file__).resolve().parent
COLDKEY = "5E7QbBiYZWfW9X5ZTN6pBJFtbJignvKHS7ufvJc9RuziD7i1"
ENDPOINT = "https://entrypoint-finney.opentensor.ai:443"


def main() -> None:
    chain = SubstrateInterface(url=ENDPOINT, ss58_format=42, ws_options={"timeout": 20})
    head = chain.get_chain_finalised_head()
    hotkeys = chain.query("SubtensorModule", "OwnedHotkeys", [COLDKEY], block_hash=head).value
    owners = {
        hotkey: chain.query("SubtensorModule", "Owner", [hotkey], block_hash=head).value
        for hotkey in hotkeys
    }
    payload = {
        "queried_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "finalized_head": head,
        "coldkey": COLDKEY,
        "owned_hotkeys": hotkeys,
        "owner_checks": owners,
        "all_owner_checks_match": all(owner == COLDKEY for owner in owners.values()),
        "source": "SubtensorModule.OwnedHotkeys and SubtensorModule.Owner at one finalized head",
        "qualification": "Current association only; not proof of ownership at submission time.",
    }
    destination = ROOT / "chain-ownership.json"
    temporary = destination.with_name(".chain-ownership.json.incoming")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps({"finalized_head": head, "owned_hotkeys": len(hotkeys), "all_match": payload["all_owner_checks_match"]}))


if __name__ == "__main__":
    main()
