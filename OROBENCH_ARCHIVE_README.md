# OROBench public archive

Snapshot: `2026-11-09T13:18:42.095071Z`

This archive was collected only from ORO's unauthenticated public API. Legacy
ShoppingBench suites are excluded. The `champions/` corpus is the union of all
released OROBench race-winner versions and all versions recorded as public top
agent/title holders. Role mappings and source SHA-256 hashes are in
`champions/catalog.json`.

## Coverage

- OROBench EnvPack suites: **11** (140, 142, 144, 146, 148, 150, 152, 154, 156, 158, 160)
- Race records: **58** (54 completed; 4 cancelled without a race task roster)
- Unique race-winning versions: **39**
- Unique public top/title versions: **36**
- Unique champion union: **40**; downloaded: **40**
- Public qualifying rows: **335** (155 unique task IDs)
- Completed-race rosters recovered: **54/54**
- Race task-identifier rows: **4860** (4534 unique IDs)

## Important boundary

Public qualifying problems include full `goal_text` and are stored by suite in
`problems/qualifiers/`. ORO keeps race prompts and sealed answers private. Its
released race APIs expose task identifiers and result/feedback metadata, while
the feedback `query` field is null. Consequently, `problems/races/` contains
every recoverable public race roster with `query: null`; it does not claim that
withheld race prompts were downloaded. See each race's `metadata.json` and
`problems/races/recovery_failures.json`.

The collector is `scripts/collect_orobench_archive.py`. Re-run it to refresh the
snapshot. Presigned storage URLs and temporary credentials are never persisted.
`SHA256SUMS` covers every generated archive file except itself.
