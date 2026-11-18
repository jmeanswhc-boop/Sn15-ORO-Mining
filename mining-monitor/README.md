# OROBench / Tosky monitor

The local dashboard frontend is intentionally an empty canvas while its new information architecture is defined. It does not request or render monitoring data. The existing collector and archive remain available behind the frontend and continue to read ORO's unauthenticated public API and finalized Finney chain
storage. No wallet files, API keys, inference calls, or submissions are used.
Legacy ShoppingBench suites are excluded.

## Run

From the repository root:

```bash
python3.11 -m venv mining-monitor/.venv
mining-monitor/.venv/bin/python -m pip install -r mining-monitor/requirements.txt
mining-monitor/.venv/bin/python mining-monitor/server.py --hotkey <PUBLIC_HOTKEY>
```

Open **http://127.0.0.1:8765**. The configured name defaults to Tosky. The fixed
hotkey persists locally; subsequent starts can omit `--hotkey`. Identity matches
the hotkey exactly, including older submissions with a different name. Another
miner's agent named Tosky is never treated as yours. Coldkeys are current
`SubtensorModule.Owner` values at a recorded finalized block, not proof of the
owner at submission time. If the chain is unavailable, unresolved ownership is
shown explicitly and retried separately from ORO monitoring.

Without the optional dependency, `python3 mining-monitor/server.py` still runs
all dashboard/archive functions; coldkeys remain unavailable. The default chain
endpoint is the official Finney HTTPS endpoint. Override with `--chain-url` if
necessary. `--port` selects another local port. `--no-worker` opens saved data
without upstream requests. The server binds only to loopback.

## Data capabilities (not currently rendered)

The presentation layer is blank. The following describes the preserved collector and archive capabilities available for the future dashboard:

- Champions: every discovered OROBench public top/title holder, submission time,
  qualifier score, title/overall score, coldkey/hotkey, and automatic code-download status.
  If only an older race roster exposes a qualifier score, its historical suite
  is labeled explicitly; it is never presented as a current-suite score.
- Champions can be searched by name, hotkey, or coldkey. Search and sorting
  cover the complete registry before pagination. Sort by overall/title score,
  public qualifier score, either key, or the date the agent first became champion;
  choose ascending or descending. Missing values stay last. Copy buttons copy
  the complete keys from the table or agent details, even when visually shortened.
- Open an agent to inspect its individual race scores, weighted/window scores,
  ranks, elimination records, and title dates. Raw race scores and overall scores
  remain distinct. Champion titles are distinct from merely winning a race.
- Tosky versions: submission timestamps, qualifier scores, global qualifier and raw-race leaderboard
  ranks with observation times, current reported state, elimination timestamps,
  and discovered race participation. Public evaluation runs expose their scores,
  inference costs, and duration when ORO publishes those fields.
- Tosky's **Score to beat** panel shows the incumbent's published Overall, the
  current additive challenge margin, and the strictly-greater Overall threshold
  from `public/top`. The margin is already included; it is not applied twice or
  treated as a percentage multiplier. Each version shows its current-race
  Overall gap when released, or an explicit pending/nonparticipating status.
  Qualifier scores and prior-race Overalls never fill a missing current score.
  When the two retained race deltas are known, a separate raw-score lead estimate
  uses `3 * margin + incumbent retained deltas - challenger retained deltas`.
  Pre-submission races contribute zero; missing older history remains unknown.
  Estimates use rounded public windows and the current margin, which may change
  before closing. They do not guarantee selection over another challenger.
  See [ORO scoring](https://oroagents.com/docs/miners/scoring).
- Archive: source states and inventory pass coverage. `Unavailable` is never
  converted to zero, and absence from a leaderboard never implies elimination.
  Search-result ranks are never used as global ranks. ORO only documents
  qualifying and race sorts; an unsupported overall sort can silently fall back
  to qualifier ordering. Weighted overall scores are kept separate from raw-race
  leaderboard positions.

## Automatic collection and backfill

The collector runs in the server, independently of browser tabs. Keep the server
running for continuous monitoring. Stopping it stops collection; restarting it
resumes from durable SQLite checkpoints. A page refresh only reloads saved data.
Upstream fast monitoring runs on a 30-second schedule **plus request latency**;
API requests are paced at least 0.8 seconds apart (at most 75 per minute across the shared client), with shared 429 backoff.
There is no guarantee of instantaneous delivery during outages or rate limits.

New champions take priority over historical downloads. A downloaded challenger
that becomes champion is copied into the champion registry without deleting its
original archive. The collector imports and hash-checks existing champion files,
then backfills titleholders and unsuccessful challengers from the union of:

- top history and current top;
- every OROBench suite's paginated qualifying leaderboard, plus the active
  raw-race leaderboard;
- race history, individual race rosters, and pending/running evaluations.

Existing public race snapshots seed the display, then are refreshed from ORO.
Live offset-paginated lists can shift during collection. Repeated scans deduplicate
by version UUID; completion of a pass is **not** a provable census of records
that ORO no longer exposes. The UI reports unfinished inventory/download work.
The initial full challenger backfill can take hours and significant disk space;
all collection is resumable and uses no paid model credits.

Sources with future release times are scheduled and periodically rechecked.
`ARTIFACT_NOT_RELEASED` is retained as release-gated, not permanently failed.
Available code is downloaded on a subsequent successful scheduled request;
release dates, backfill workload, network delays and API backoff affect latency.
Champion checks take precedence over this queue.

Files:

- `champions/<name>_v<version>_<UUID>/`: champion source and provenance; existing
  archive directories remain intact.
- `champions/registry.json`: current persistent titleholder index. Existing
  `champions/catalog.json` remains the earlier collector's snapshot.
- `challengers/<name>_v<version>_<UUID>/`: released non-titleholder source.
- `logs/orobench-monitor/registry.sqlite3`: configured hotkey, public records,
  race results, ownership cache, release/retry schedule, and scan checkpoints.

Sources are never executed or imported. Downloads record SHA-256 and provenance,
not presigned storage URLs. Archive extraction rejects traversal, symlinks,
duplicate names and excessive sizes. The local download route accepts only
version UUIDs registered in the database. No earlier agent code or evaluation
materials are deleted.

## Offline verification

```bash
python3 -m unittest discover -s mining-monitor -p 'test_*.py'
```

Tests mock upstream requests and exercise identity, persistence, pagination,
ranking scope, embargo/release retries, source integrity, archive extraction,
promotion, stale responses, API cache and global rate-limit backoff.

## Race directories and automatic result files

The **Race archives** view shows code-copy counts, indexed run lists, published
result files, release-gated files, unpublished artifacts, and errors for every
race. Open a race for its participants and file links. Collection is automatic;
download links are optional ways to inspect what is already saved.

Each race lives under:

```text
race-archives/suite-<suite>/race-<number>-<race-uuid>/
  race.json
  roster.json
  participants.csv
  manifest.json
  agents/<agent-version-uuid>/
    participant.json
    agent.py                  # or agent_code.zip, when released
    code_metadata.json
    source/
    runs.json                 # only validator runs for this race
    runs/<run-uuid>/
      run_detail.json
      feedback-<episode-uuid>.json
      log_bundle.zip          # only if ORO publishes it
      reasoning.bin           # only if ORO publishes it
  race-data/                  # qualifiers, public seed/variance records
```

The same agent version gets independent source copies and separate run data in
every race it enters. A three-race participant therefore appears in three race
directories. Network downloads are reused from the shared archive; race copies
are verified with SHA-256. Agent sources are never imported or executed, and
old released files are preserved even if the local Python parser rejects them.

`race-archives/index.json` summarizes the race archive. Its `_objects/` directory
caches resources shared across races; run and episode files are stored once,
directly inside their own race directory. The race manifest records checksums and public
release/retry state. Pending work persists in the existing SQLite database.

The worker archives released public code, race rosters/scores, qualifying rosters,
seed/variance records, run records, public log bundles and reasoning details,
and the miner-safe episode-feedback projection. Per-problem log downloads are
attempted only when the public result supplies a valid problem UUID. OROBench
episode IDs use the feedback route instead. Private world packs and private
episode/grading artifacts are not exposed by these public APIs.

Release is determined by ORO's responses and `code_available_at`, not by assuming
that five calendar days always implies availability. New race status transitions
refresh run discovery. Withheld or partial result rosters and unavailable
diagnostics stay retryable. An artifact 404 means unpublished, not downloaded;
it is rechecked later. No archive is labeled complete while enumerated downloads
are pending, result lists are partial, or participant code is missing.

Initial historical catch-up can require many hours for code and substantially
longer for individual episode feedback. The public API currently documents a
default allowance of 100 requests per minute per network identity; the shared
client stays below it and obeys `429`/`Retry-After`. A fresh installation is not
instantly complete. Use the live counts and per-race manifests to assess coverage.
The archive pauses artifact downloads if free space drops below 2 GiB.

## Appearance

Choose **Appearance** in the header. Light, dark and system-following modes are
available. Background, panels, text, accent, sidebar and border colors can be set
with a picker or six-digit hex value. Palettes are saved separately for light and
dark mode in this browser and survive reloads. **Reset current palette** restores
the selected mode's defaults. The dialog reports text/background contrast.

Theme logic can also be checked in a Node.js environment:

```bash
node --input-type=module -e "import('./mining-monitor/test_theme.mjs').then(async m => console.log(await m.runThemeTests()))"
```

## Progress, completion and priority

A progress bar on every view reports the historical backfill's processed indexed
items, downloaded resources, not-yet-public items and errors. Totals can grow as
run discovery exposes more files. Before the race-history scan finishes, the bar
is indeterminate. Future releases and unpublished files are identified separately;
transient diagnostics failures, partial results and download errors prevent a
successful completion indication.

Historical race membership is frozen after the first complete history scan. A
newly concluded race gets a persistent priority queue, including its winner's
code and fresh run discovery. Its eligible work preempts the historical queue
between files. The five-day code-release cohort is also prioritized, with ORO's
actual release state determining whether a download is permitted. Gated work
waits for its next retry while eligible historical work resumes. Completed files
and the historical scheduling cursor are retained across preemption and restarts.

On completion, a notification is stored by the server and displayed in the
dashboard. A dismissed/read notification is not repeated after browser reload.
No notification permission or external messaging service is needed.

Champion information shows only automatic-download status (for example,
**✓ Code downloaded**); it has no manual champion-code download button. The source
remains saved in the champion/race directories. Copy-key buttons use a familiar
overlapping-squares icon with accessible labels and checkmark feedback.

Dashboard logic checks in a Node.js environment:

```bash
node --input-type=module -e "import('./mining-monitor/test_dashboard.mjs').then(async m => console.log(await m.runDashboardTests()))"
```
