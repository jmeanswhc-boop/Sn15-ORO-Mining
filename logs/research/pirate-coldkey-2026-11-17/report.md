# PIRATE coldkey: public-source and race analysis

Research snapshot: 2026-11-17. API data was collected at 13:53-13:56 UTC;
chain ownership was checked at 14:05 UTC.

## Executive finding

The public evidence supports the central part of the proposed hypothesis:

- The PIRATE coldkey controls exactly two hotkeys at the finalized chain head
  queried for this report.
- Those hotkeys were used as two distinct submission lanes. One submitted
  `SPIRIT` v1-v6; the other submitted `KUNGFU` v1 and then `PIRATE` v1.
- The lanes were active in parallel. `SPIRIT` v2 and `KUNGFU` v1 were submitted
  3 minutes 39.717 seconds apart. `SPIRIT` v3 and `PIRATE` v1 were submitted
  5 minutes 41.248 seconds apart.
- All four currently released organization sources share the same Cairn-family
  architecture, entry point, major prompts, deterministic controller, and model
  policy. They are not eight independent architectures.
- They are also not one byte-identical agent repeatedly renamed. The source
  history resolves into two divergent branches: a broad SPIRIT branch based on
  Cairn v2 and a more targeted KUNGFU/PIRATE branch based on Giyu v17, itself a
  Cairn derivative.
- After its introduction, PIRATE v1 won four of five same-race comparisons
  against the SPIRIT lane. Including the earlier KUNGFU v1 precursor, the second
  branch won four of six. This is consistent with a parallel-variant strategy, but the
  public evidence cannot establish team size, private development chronology, or
  an explicit internal rule for selecting a survivor.

The most precise conclusion is therefore: **one architectural family, two
concurrent development branches, and two owned hotkey lanes**.

## Identity and chain ownership

Coldkey:

`5Hb2bcydcr2GyF5546thP2TE2kh1H4xfejWXw77uJGGAYfwz`

At finalized head
`0x1486f3f64548ef7a6ffa95952b34a38085255f92ab2e900734a02bdce920e0e7`,
`SubtensorModule.OwnedHotkeys` returned exactly:

1. `5CG5VEaWky1HqNQsJZgKXeqNyLsA3qrXAFtyaW7roWgpWkxp`
2. `5EFZuxy9u3pjdGoBqn59NMLLynfTdbwRceW4Wo5phr6siPtu`

The raw observation is preserved in [chain-ownership.json](chain-ownership.json).
This proves common coldkey ownership at the queried chain state. It does not by
itself identify the humans operating the keys.

## Complete public submission inventory

I queried every public envpack-suite leaderboard and matched the two hotkeys,
rather than searching only by agent name. Eight versions were found, all in
suite 160.

| Submitted UTC | Lane | Agent | Qualifier | Current race score | Runs | Source state at collection |
|---|---|---:|---:|---:|---:|---|
| Nov 10 16:01:25 | `5CG5...Wkxp` | SPIRIT v1 | 0.8500 | 0.3440 | 4 | downloaded |
| Nov 11 16:46:12 | `5CG5...Wkxp` | SPIRIT v2 | 0.8500 | 0.2344 | 4 | downloaded |
| Nov 11 16:49:52 | `5EFZ...iPtu` | KUNGFU v1 | 0.8334 | 0.2233 | 4 | downloaded |
| Nov 12 18:13:06 | `5CG5...Wkxp` | SPIRIT v3 | 0.8167 | 0.2087 | 4 | gated until Nov 17 18:13:06 |
| Nov 12 18:18:47 | `5EFZ...iPtu` | PIRATE v1 | 0.8334 | 0.4068 | 12 | downloaded; champion release |
| Nov 13 15:51:10 | `5CG5...Wkxp` | SPIRIT v4 | 0.8667 | 0.3286 | 4 | gated until Nov 18 15:51:10 |
| Nov 14 15:10:10 | `5CG5...Wkxp` | SPIRIT v5 | 0.8667 | 0.0618 | 6 | gated until Nov 19 15:10:10 |
| Nov 16 18:19:56 | `5CG5...Wkxp` | SPIRIT v6 | 0.8834 | 0.3953 | 4 | gated until Nov 21 18:19:56 |

“Current race score” is the live status value and is not the same thing as a
single race's raw score. The full IDs, exact timestamps, release states, hashes,
and paths are in [manifest.json](manifest.json).

### Release timing correction

The five-day timer applies to an ordinary version's submission timestamp, not
to the date on which some related agent became champion. PIRATE v1 became public
early when it gained the champion release on Nov 14. At collection time only
four of the eight sources were legitimately public. The other four artifact
requests returned `ARTIFACT_NOT_RELEASED`; those sources were not bypassed or
inferred.

The collector can be rerun after each listed timestamp and will fetch newly
released artifacts without changing its discovery rules:

```bash
python3 logs/research/pirate-coldkey-2026-11-17/collect.py
python3 logs/research/pirate-coldkey-2026-11-17/analyze.py
```

## Downloaded public sources

No downloaded source was executed. Each was stored verbatim and parsed only as
text/AST.

| Agent | File | Size | Lines | SHA-256 |
|---|---|---:|---:|---|
| SPIRIT v1 | [agent.py](sources/SPIRIT_v1_06af33fd-4d66-73a7-8000-eaf571549005/agent.py) | 381,848 B | 6,024 | `c069b78f...2c313` |
| SPIRIT v2 | [agent.py](sources/SPIRIT_v2_06af49bd-3f34-7b79-8000-eb63d5af6047/agent.py) | 433,560 B | 6,762 | `8bef2e57...6afb` |
| KUNGFU v1 | [agent.py](sources/KUNGFU_v1_06af49ca-fb47-73c3-8000-0a26e1097af4/agent.py) | 365,373 B | 5,866 | `a99e0037...6566` |
| PIRATE v1 | [agent.py](sources/PIRATE_v1_06af6030-67a0-754e-8000-beacfd741adb/agent.py) | 415,484 B | 6,553 | `ea169847...e27` |

The complete digests are in the manifest and per-source `metadata.json` files.

## Lineage: one family, two branches

The observed tree is:

```text
Cairn v2
├── SPIRIT v1
│   ├── SPIRIT v2
│   └── SPIRIT v3-v6 (source not public at snapshot time)
└── Giyu v17
    └── KUNGFU v1
        └── PIRATE v1
```

The tree distinguishes two kinds of evidence:

- The edges among released sources are supported by exact AST-definition
  preservation, whole-file similarity, changed-method structure, and timing.
- Cairn-to-Giyu is already visible in the public champion archive and explains
  why the two organization branches share so much code.
- For unreleased SPIRIT v3-v6, only names, hotkey, timing, scores, and run data
  are public. Their exact code ancestry must remain provisional until release.

### Time-valid prior-source search

To avoid attributing an agent to code that was not public when it was submitted,
the candidate corpus was filtered by each target's submission timestamp.

For SPIRIT v1, 568 unique sources that were public by Nov 10 16:01 were parsed.
Cairn v2 was the overwhelming nearest source:

| Candidate | Exact definitions | Shared definitions | Exact-line Jaccard |
|---|---:|---:|---:|
| Cairn v2 | 127 | 243 | 0.755106 |
| next candidate, Halo v2 | 17 | 24 | 0.513375 |

Cairn v2 had been public for 2 days 13:26:33 before SPIRIT v1. Giyu v17
was not yet public, so Giyu cannot be the publicly available parent of SPIRIT v1.

For KUNGFU v1, 681 unique sources that were public by Nov 11 16:49 were parsed.
Giyu v17 was the overwhelming nearest source:

| Candidate | Exact definitions | Shared definitions | Exact-line Jaccard |
|---|---:|---:|---:|
| Giyu v17 | 230 | 252 | 0.882103 |
| Cairn v2 | 134 | 243 | 0.801253 |
| next non-family candidate | 20 | 24 | 0.519861 |

Giyu v17 had been public for 12:33:59 before KUNGFU v1. This is strong evidence
that KUNGFU was adapted directly from the released Giyu v17. Similarity and
timing still cannot prove the operator's private act of copying.

One archived source (`Unstoppable` v1) has invalid Python syntax and could not be
AST-compared. This single failure is recorded in both scan outputs and is too
dissimilar at the text level to affect either leading match.

Detailed candidate rankings are preserved in
[prior-source-scan.json](prior-source-scan.json) and
[prior-source-scan-kungfu.json](prior-source-scan-kungfu.json).

### Released-source similarity

| Comparison | Sequence similarity | Exact-line Jaccard | Exact/shared definitions | Added | Removed | Changed |
|---|---:|---:|---:|---:|---:|---:|
| Cairn v2 -> SPIRIT v1 | 0.870661 | 0.755106 | 127/243 | 41 | 0 | 116 |
| SPIRIT v1 -> SPIRIT v2 | 0.935014 | 0.882566 | 255/284 | 31 | 0 | 29 |
| Giyu v17 -> KUNGFU v1 | 0.935040 | 0.882103 | 230/252 | 34 | 0 | 22 |
| KUNGFU v1 -> PIRATE v1 | 0.936683 | 0.886397 | 258/286 | 22 | 0 | 28 |
| SPIRIT v2 vs KUNGFU v1 | 0.765148 | 0.623140 | 184/244 | - | - | 60 |

Both direct organization updates are conservative in one important sense: they
retain all predecessor definitions and add/change behavior around the inherited
core. But the two lanes are genuinely divergent; SPIRIT v2 and KUNGFU v1 are
far less similar to each other than either is to its direct predecessor.

## Shared architecture and execution flow

Every released organization source defines the same central `Cairn` controller,
the same `Item` representation, and the same `agent_main` shape. The common
workflow is:

1. `agent_main(problem, ...)` constructs `Cairn(problem)` and invokes `run()`;
   an exception path logs the failure and tries to terminate safely.
2. `read_request()` gives the BRIEF prompt the user request and runtime tool
   schemas. It extracts products, quantities, hard constraints, budget and
   objective, message/tool limits, cross-listing lookups, required process, and
   maps semantic roles to the available environment tools.
3. `first_moves()` searches and opens catalog entries before the interview so
   that later questions use real store labels and observed property keys.
4. `interview()` uses PLAN/HEAR to ask within the shopper-message allowance,
   recording answers, indifference, dropped requirements, and budget changes.
5. A LIGHT/SPARE tool-calling planner proposes actions. The deterministic Cairn
   controller validates schemas, suppresses duplicate calls, pages results,
   normalizes observation shapes, and maintains product/cart/price/stock state.
6. Deterministic fit logic evaluates row and full-record evidence, category
   ownership, hard constraints, linked-reference requirements, price objectives,
   and compatibility.
7. NOTE ingests store or shopper changes and triggers cart repair.
8. CHECK/GATE/AUDIT/TRAIL paths review the cart and required procedure before
   buy/refuse termination.
9. Call, time, and estimated-spend admission rules select models or fallbacks.

The underlying design is hybrid: language models interpret and propose, while a
large deterministic state machine validates, repairs, and terminates.

All four released versions use the same advertised model policy and nominal
ceilings:

- LIGHT: `anthropic/claude-haiku-5.5`
- HEAVY: `anthropic/claude-sonnet-5.5`
- SPARE: `openai/gpt-6-luna`
- `LOOP_CALLS = 42`, `ALL_CALLS = 60`
- `SPEND_SOFT = 0.055`, `SPEND_HARD = 0.078` per task

## What each branch changed

### SPIRIT v1: broad coverage, chain handling, and adaptive resources

Relative to Cairn v2, SPIRIT v1 is the largest released organization update:
41 new definitions and 116 changed inherited definitions. Its additions group
into four themes:

1. **Reference-chain discovery:** `link_words`, `link_marks`, `open_links`,
   expanded `TRAIL`, and related methods try to locate and open listings named
   indirectly by another requirement.
2. **Terminal coverage:** `cover_items`, `cover_pick`, `held_fits`, and
   `buy_held` try to prevent an incomplete cart or premature refusal.
3. **Objective repair after changes:** `rehunt` and `repick` re-run cheapest or
   extreme-value selection when the world or request changes.
4. **Adaptive resource use:** `step_scale`, `loop_caps`, `call_cap`, and
   `spend_top` raise available work for longer tasks.

This is a breadth-first branch: more rescue paths and more heuristic coverage.
Its risk is the same breadth—more interacting terminal paths and a spend policy
that can expand with task limits.

### SPIRIT v2: targeted catalog-grounding additions

SPIRIT v2 keeps every SPIRIT v1 definition, adds 31, and changes 29. Major
additions are:

- per-item `max_price` extraction in BRIEF;
- property-key-grounded fit/miss checks (`keyed_fit`, `keyed_misses`);
- compatibility series pairing and ranking;
- stock rereading after a rejected checkout;
- delayed deep paging until the interview provides better constraints;
- deeper objective search once the target is known.

This is a targeted extension of the SPIRIT strategy, not a redesign.

### KUNGFU v1: evidence-grounded discovery and guarded completion

Relative to Giyu v17, KUNGFU adds 34 definitions and changes 22, retaining the
entire inherited definition set. Its core ideas are:

- tri-state, property-key-aware evidence through `key_verdict` and
  `bound_verdict`, including same-unit numeric bounds;
- prioritizing catalog keys that actually exist (`keys_first`);
- wider noun/need discovery and opening promising rows before declining;
- cart-size, cheapest-selection, missing-item, and decline guards.

Compared with SPIRIT, this is the narrower and more evidence-focused branch.

### PIRATE v1: ambiguity, changed intent, chains, and price repair

PIRATE preserves KUNGFU's definitions, adds 22, and changes 28. The additions
target precisely the harder interaction patterns visible in hidden shopping
tasks:

- `quoted_count` checks whether a number is actually tied to an item's quantity;
- `name_needs` recovers attributes that were folded into the product name;
- `drop_described`, `drop_folded`, and `dropped_by_value` remove stale
  name-derived constraints after the shopper changes their mind;
- `heard_key` grounds a shopper reply to the property being discussed;
- `LINK_CHECK`, `link_mismatch`, and `chain_catchup` strengthen dependent
  listing chains;
- `read_on`, `value_pages`, `price_swap`, and `price_fits` improve deep search
  and cheapest/budget repair;
- `direct_order` and a revised `decline_guard` add terminal recovery paths.

Two new prompts, COUNT_QUOTE and LINK_CHECK, are added; NOTE and PLAN are also
expanded. This is a targeted hard-task update, not simply a larger prompt.

### Static risks still present in PIRATE v1

Static review found two important risk areas; neither can be claimed to have
caused a particular race error without task traces:

1. `direct_order()` invokes `gate("order")` but does not branch on a false
   return before continuing its own path. `finish()` also has a route that can
   reach buy after a negative gate, and the gate's unavailable-model fallback
   can be permissive. The several terminal recovery paths are therefore not a
   single fail-closed decision point.
2. `scaled_same()` can treat numbers with incompatible units as equivalent in
   some shapes. A static probe such as `1000 mAh` versus `10 V` exposes the
   missing dimension check. That can contaminate numeric fit reasoning.

These are examples of the shared architecture's central trade-off: its many
deterministic repairs improve coverage, but overlapping repair paths make global
invariants harder to guarantee.

## Performance evidence

The following values are means of the included public validator runs for each
phase/race. `Calls` is inference requests per 90-task validator run; cost is the
validator-reported inference cost per full run.

| Agent | Race | Raw score | Cost USD | Seconds | Calls | Inference failures |
|---|---:|---:|---:|---:|---:|---:|
| SPIRIT v1 | 20 | 0.35000 | 4.7181 | 553.5 | 1,612.0 | 0.310% |
| SPIRIT v2 | 21 | 0.22775 | 4.8668 | 488.3 | 1,560.0 | 0.256% |
| KUNGFU v1 | 21 | 0.19445 | 3.9805 | 415.4 | 1,531.5 | 3.754% |
| SPIRIT v3 | 22 | 0.20000 | 4.4298 | 549.1 | 2,018.5 | 0.050% |
| PIRATE v1 | 22 | 0.22775 | 4.2093 | 533.4 | 1,972.5 | 0.456% |
| SPIRIT v4 | 23 | 0.30555 | 4.5659 | 438.9 | 1,743.0 | 0.115% |
| PIRATE v1 | 23 | 0.38335 | 4.0814 | 442.1 | 1,777.5 | 0.253% |
| SPIRIT v5 | 24 | 0.31665 | 4.6669 | 580.2 | 1,642.5 | 0.548% |
| PIRATE v1 | 24 | 0.34445 | 4.7148 | 564.8 | 1,644.5 | 0.395% |
| SPIRIT v5 | 25 | 0.05560 | 4.8001 | 556.8 | 1,894.5 | 0.501% |
| PIRATE v1 | 25 | 0.04440 | 4.7050 | 563.6 | 1,890.5 | 0.423% |
| SPIRIT v6 | 26 | 0.37780 | 4.1055 | 529.5 | 1,618.5 | 0.185% |
| PIRATE v1 | 26 | 0.39445 | 4.2467 | 526.4 | 1,680.0 | 0.327% |

Same-race comparisons remove the largest confounder—the hidden world changing
from day to day:

| Race | SPIRIT lane | KUNGFU/PIRATE lane | Score winner | Cost difference for second lane |
|---:|---:|---:|---|---:|
| 21 | 0.22775 | 0.19445 | SPIRIT by 0.03330 | -0.8863 USD |
| 22 | 0.20000 | 0.22775 | PIRATE by 0.02775 | -0.2205 USD |
| 23 | 0.30555 | 0.38335 | PIRATE by 0.07780 | -0.4845 USD |
| 24 | 0.31665 | 0.34445 | PIRATE by 0.02780 | +0.0479 USD |
| 25 | 0.05560 | 0.04440 | SPIRIT by 0.01120 | -0.0951 USD |
| 26 | 0.37780 | 0.39445 | PIRATE by 0.01665 | +0.1412 USD |

The second lane won four of five comparisons after PIRATE was introduced; the
table includes race 21 as the KUNGFU precursor comparison, where it lost. More
importantly, the same PIRATE source ranged from 0.0444 in race 25 to 0.39445 in
race 26. Hidden-race composition dominates enough that a cross-day score change
must not be attributed automatically to a code update.

PIRATE's public timeline was fast:

- first organization submission to PIRATE champion timestamp: 3 days 10:54:07;
- PIRATE v1 submission to champion timestamp: 1 day 08:36:46;
- PIRATE first raced in race 22 and reached the top in race 23.

## Assessment of the hypothesis

| Claim | Assessment | Evidence |
|---|---|---|
| One coldkey managed two hotkeys | Confirmed at queried finalized head | Direct chain storage query |
| Two hotkeys acted as separate submission lanes | Confirmed | All eight versions partition cleanly by name/lineage and hotkey |
| Multiple variants were developed in parallel | Strongly supported | Two pairs submitted 3:40 and 5:41 apart; sources form divergent branches |
| All agents used one underlying architecture | Confirmed at family/framework level for all released sources | Shared Cairn controller, entry point, prompts, models, caps, and high AST overlap |
| Every one of the eight versions has identical ancestry | Not yet provable | SPIRIT v3-v6 remain release-gated |
| Operators submitted variants and kept the survivor | Consistent with outcomes, not directly proved | PIRATE lane won most head-to-head races and became champion; no private selection record is public |
| A large team developed them | Unknown | Keys and code do not identify developer count |

## Reproducibility and evidence files

- [manifest.json](manifest.json): authoritative local inventory, release state,
  archive hashes, and paths.
- [api/leaderboards.json](api/leaderboards.json): matching live leaderboard rows.
- [api/status](api/status): raw public status responses for all eight versions.
- [api/runs](api/runs): raw public run responses used for the performance table.
- [static-analysis.json](static-analysis.json): AST/source inventory and all 15
  pairwise comparisons.
- [prior-source-scan.json](prior-source-scan.json): time-valid SPIRIT v1 parent
  search.
- [prior-source-scan-kungfu.json](prior-source-scan-kungfu.json): time-valid
  KUNGFU v1 parent search.
- [collect.py](collect.py), [analyze.py](analyze.py), and
  [scan_prior_sources.py](scan_prior_sources.py): reproducible collection and
  non-executing analysis scripts.

Official live endpoints used by the collector include the public suite list,
leaderboard, per-version status, runs, and released-artifact endpoints under
`https://api.oroagents.com/v1/public/`.

## Bottom line

PIRATE's rapid rise was not the result of eight unrelated bets. The public
record shows a disciplined two-lane experiment within one mature Cairn family:
one branch widened recovery and catalog coverage, while the other tightened
evidence handling and then added explicit defenses for ambiguity, changed
requirements, linked listings, and price repair. The second branch generally
won the direct hidden-race comparisons and became champion quickly.

The significant lesson is not merely “submit more agents.” The leverage came
from preserving a proven common controller while testing two meaningfully
different error theories in parallel. The remaining uncertainty is confined to
the unreleased SPIRIT v3-v6 implementations and to private organizational intent;
the collector is ready to close the source gap as each legal release time arrives.
