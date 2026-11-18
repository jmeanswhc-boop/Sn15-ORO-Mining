# KYO / Miyagi coldkey organization: public-corpus analysis

Snapshot: 2026-11-17 15:36–16:05 UTC  
Suite: OROBench-6 (`suite_id=160`)  
Coldkey: `5E7QbBiYZWfW9X5ZTN6pBJFtbJignvKHS7ufvJc9RuziD7i1`

## Executive conclusion

The evidence supports calling this a single **operational organization**: at the
finalized chain head queried for this study, the coldkey owned all ten hotkeys
that account for the 136 versions in the public ORO census. It does **not**
support treating all 136 versions as one code lineage.

The organization operated a high-throughput portfolio:

- 136 submitted versions under 38 names and ten hotkeys in 23 days;
- 113 released source artifacts downloaded and parsed; 23 release-gated
  versions represented by public metadata only;
- ten different initial agents submitted in a 67-minute launch wave;
- many parallel architectures, followed by internal recombination;
- fast adoption of newly released champions, often 13–19 hours after their
  source became public.

KYO_v6 was the first successful internal foundation. It was not a copied prior
champion: its closest eligible prior champion has negligible structural
similarity, while its source is an incremental continuation of KYO_v5. The
v5→v6 redesign targeted a specific hidden-test bottleneck: the agent did not
reliably understand unfamiliar store-tool result shapes, collect complete
product evidence, cover paginated/unsorted candidate sets, or avoid redundant
views before checkout. The redesign made schema discovery, evidence gathering,
pagination, price ordering, and observation caching first-class control-plane
functions without changing the model trio or increasing the spend ceilings.

That change coincided with a hidden-race mean rising from 0.1333 to 0.4056,
while the public qualifier mean stayed at 0.8167. The diagnostic category with
the largest observed improvement was `process_requirement_missed` (45.0% to
23.3% of episodes), followed by `request_not_met` (34.4% to 26.1%). The
cheapest-selection error rate did not improve. Because Race 7 and Race 8 used
different hidden tasks and worlds, these are strong directional observations,
not a controlled causal estimate.

After KYO_v6, the organization changed strategy. It continued using KYO_v6 as
an internal donor, but also incorporated the public architectures of Cloud,
Know Thyself, love, Halo, YesYesYes, Cairn, and Giyu. Miyagi_v2 is therefore
best understood as the endpoint of a later **Cairn → Giyu → Kogure/Miyagi**
branch. It is the same coldkey organization, but it is not technically a
KYO_v6 descendant in the narrow source-lineage sense.

## Scope and evidence rules

This study used four evidence classes:

1. **Public ORO metadata**: leaderboard entries, version status, run summaries,
   timestamps, scores, release state, hotkeys, and artifact availability.
2. **Released source artifacts**: downloaded only when ORO reported
   `release_state=RELEASED`; archive hashes were recorded and every Python
   source was parsed with `ast`.
3. **Current finalized-chain state**: `OwnedHotkeys` and `Owner` storage queries
   at one finalized block.
4. **Public champion archive**: prior-champion source was compared only when its
   `code_available_at` preceded the organization's submission time.

The analysis intentionally did not retrieve, infer, reconstruct, or persist
unreleased task content or hidden source. The 23 hidden versions have blank
source-derived fields in `inventory.csv`.

Interpretation labels used below:

- **Confirmed**: directly present in API, chain, or source evidence.
- **Source-derived**: reproducible static analysis of released code.
- **Inference**: the most economical explanation of the observed evidence, but
  not a statement about private human identity or intent.

## Corpus completeness

| Measure | Result |
|---|---:|
| Versions discovered | 136 |
| Released source artifacts downloaded | 113 |
| Release-gated sources excluded | 23 |
| Download/integrity errors | 0 |
| Hotkeys | 10 |
| Agent names | 38 |
| First submission | 2026-10-24 17:20:08 UTC |
| Last submission in snapshot | 2026-11-16 18:13:47 UTC |
| Unique released byte hashes | 113 / 113 |
| Unique normalized module-AST hashes | 113 / 113 |
| Corpus size on disk | about 63 MB |

There are no exact duplicate released sources, even after module-level AST
normalization. That does not imply 113 independent architectures: many are
edited descendants whose prompts, functions, and control flow changed enough
to produce unique hashes.

The complete per-version inventory is in `inventory.csv`. The 113 artifacts are
under `sources/`; `manifest.json` is the source-of-truth release manifest.

### Release-gated versions at the snapshot

Public metadata existed, but source was not downloaded, for:

`Miyagi v1/v3/v4`, `Sendo v1/v2/v3`, `Kogure v2/v3/v4`, `Maki v1/v2/v3/v4`,
`Hujima v1/v2/v3`, `Michui v3/v4`, `Hanagata v1/v2`, `Rukawa v2/v3`, and
`Akagi v2`.

Consequently, statements about those versions are performance/metadata
statements only. They are not source-lineage conclusions.

## Ownership and hotkey portfolio

At finalized block
`0xee930adc61086a496332a26651e72b1b194d5c5aab29227c9e881d522c0d0816`,
`SubtensorModule.OwnedHotkeys` returned exactly the ten hotkeys below, and each
individual `Owner` query returned the target coldkey.

| Hotkey | Versions | Released | Names used |
|---|---:|---:|---|
| `5F2GYd9TurPAaKshPn97xetfPzj2yUBoqNR9NG6YcAFu8jwH` | 15 | 14 | KYO, Akagi |
| `5GxwAarfPmctFUedJ897C658ysfXvZTWev6CkksxbGxWKj3J` | 17 | 15 | TERRY, Michui |
| `5E7hJqqWYARRrg3SJeQPyhmfyjvFy5VTSU548dx1DM8w9M5Z` | 14 | 11 | Missouri, Clutha, Narmada, Magdalena, Colorado, Volta, Kogure |
| `5Ct25h2R9uXD2zXMzYACXxbpZPr8XMm6X6LztDKm4rUZDLw7` | 15 | 13 | Shannon, Niger, Hanagata |
| `5EEj8R5aWwDp6XyX7D8KLirZdpZWqWj9LA3rAe5QYVbmA5yk` | 11 | 8 | Padeira, Prinoco, Hujima |
| `5CLjFrXqwgCDFGetZaqoynuYddrtS7u5ohKXRPsnE2RPfmn7` | 11 | 8 | Elaware, MAI, Lena, Mahanadi, Miyagi |
| `5HjEAFb7zXvknvS1wp7M9WuAquDdcxocbm9rduccc8YjEvKV` | 13 | 9 | Douro, Maki |
| `5E7ToJVkfTVzUYAmpLbAH8WNkpLiFLk3onUKckVkXRVv7Uki` | 20 | 17 | Gila, Zambezi, Yamuna, Leno, Kama, Tigris, Mekong, Sakuragi, Sendo |
| `5E5BFs2hLCRDuXUc8gDojja86AaKDReXYNa9rkM55mULcW9C` | 12 | 10 | Orange, Parana, Rukawa |
| `5F6p6V8wnBSLL1FSBiHWL3NJr1HgQ4ZJKsfpGHvx7dzBk2KE` | 8 | 8 | Euphrates, CRYSTAL |

This proves a current on-chain administrative relationship. It does not prove
that the same coldkey owned every hotkey at every historical submission block,
nor does it identify the number of people behind the keys.

## How the organization operated

### 1. It began as a diversified portfolio, not a single-agent campaign

The first ten submissions arrived between 17:20:08 and 18:27:44 UTC on
2026-10-24:

| Agent | Lines | Main architectural markers | Qualifier | Published race score |
|---|---:|---|---:|---:|
| KYO v1 | 1,456 | `Item`, `Kyo`; brief/plan/hear/check/audit pipeline | 0.8000 | 0.3977 |
| TERRY v1 | 1,180 | `Episode`; distinct prompt/control loop | 0.7833 | 0.3737 |
| Padeira v1 | 5,114 | large modular class system | 0.8000 | 0.3792 |
| Elaware v1 | 6,384 | `Episode`, `NeedList`; large alternate architecture | 0.8667 | 0.0140 |
| Shannon v1 | 2,056 | `Gatekeeper`, `Courier`, specialist roles | 0.6500 | 0.3545 |
| Douro v1 | 2,079 | `Clock`, `LLMDesk`, advisors, driver | 0.8500 | 0.1238 |
| Gila v1 | 2,136 | `Session`, `Shell` | 0.8333 | 0.1216 |
| Missouri v1 | 6,005 | large `Episode` architecture | 0.8667 | 0.1266 |
| Orange v1 | 3,328 | `Timer`, `Oracle`, multiple controllers | 0.7000 | 0.3422 |
| Euphrates v1 | 2,067 | `Timer`, `Models`, alternate orchestration | 0.6000 | 0.3533 |

Their pairwise structural similarity is low. The evidence is inconsistent with
"one base agent copied into ten names". The better model is a portfolio of
parallel hypotheses, probably intended to expose different failure modes and
increase the chance that at least one architecture survived hidden races.

### 2. It maintained unusually high experimental throughput

The coldkey submitted on every day in the snapshot: 10 versions on launch day,
then between 1 and 8 per day. The total is 136 versions over 24 calendar dates,
or 5.67 versions per day. Multiple hotkeys were active concurrently and names
were retired or replaced while the hotkeys persisted.

This resembles a population-based search process:

1. run diverse architectures in parallel;
2. observe hidden-race performance and public feedback;
3. continue a promising branch;
4. transplant successful components across hotkeys;
5. replace branches with newly released champion architectures when that has a
   better expected return than local refinement.

This is an inference from timing and source relationships, not a claim about
the team's private workflow.

### 3. Internal inheritance was selective

For the 112 released sources that had an earlier organization source to compare
against, only 53 had their nearest predecessor on the same hotkey. At composite
similarity at least 0.50, 54 had a strong earlier match; 48 were same-hotkey and
six crossed hotkeys. At a looser 0.40 graph threshold, the released corpus still
contains 46 connected components.

This means:

- hotkey identity is not a reliable architecture identifier;
- agent name is not a reliable architecture identifier;
- the organization both refined branches in place and moved code across keys;
- many submissions were rewrites or new external foundations rather than small
  mutations.

KYO_v6 later became one internal donor. Strong line-overlap descendants include
Shannon v11 (0.771), Parana v7 (0.750), Euphrates v4 (0.744), KYO v7 (0.742),
KYO v8 (0.726), and Euphrates v5 (0.720). These are similarity observations,
not proof that every changed component originated in KYO.

### 4. Public champion releases were exploited quickly

Each comparison below was time-gated: only champion code already public before
the organization submission was eligible.

| Organization submission | Nearest eligible champion | Release-to-submit lag | Composite | Line Jaccard |
|---|---|---:|---:|---:|
| Parana v1 | Cloud v1 | 19h 12m | 0.623 | 0.756 |
| Euphrates v4 | KYO v6 | 16h 26m | 0.665 | 0.744 |
| Prinoco v7 | Know Thyself v1 | 16h 00m | 0.721 | 0.822 |
| Shannon v11 | love v20 | 15h 31m | 0.879 | 0.918 |
| Douro v9 | Halo v2 | 14h 31m | 0.746 | 0.826 |
| Narmada v3 | YesYesYes v1 | 15h 32m | 0.435 | 0.715 |
| Kama v1 | Cairn v2 | 15h 41m | 0.515 | 0.811 |
| Kogure v1 | Giyu v17 | 13h 08m | 0.850 | 0.913 |
| Miyagi v2 | Giyu v17 | 2d 13h 16m | 0.668 | 0.720 |

The high similarities and short lags show a repeatable organizational method:
public champions were treated as high-information experimental results, then
adapted into one or more portfolio branches. This is materially stronger than
the claim that the organization merely "reviewed" prior champions.

## Portfolio performance

The stored run lists contain 694 run records: 420 race and 274 qualifying runs.
Among successful, included race records with a numerical score, there are 406
validator-run rows covering 132 versions and 24 race numbers.

| Run-level measure | Value |
|---|---:|
| Mean race score | 0.2217 |
| Median race score | 0.2167 |
| Mean inference cost | $4.2458 |
| Median inference cost | $4.4436 |
| Runs reporting more than $5 | 77 / 406 (19.0%) |
| Mean duration | 433.3 s |
| Mean inference calls | 2,144.1 |
| Mean inference failures | 17.8 |

These are **run-level**, not version-level, statistics. ORO commonly records two
validator executions for the same version/race, so those rows are validation
replicates rather than independent task sets. Older and newer races can also
differ in task distribution and runtime versions.

The important operational finding is that portfolio breadth did not eliminate
cost fragility. Almost one in five included successful race-run rows reported
more than $5, and several low-scoring versions suffered large inference-failure
counts. The organization learned under noisy infrastructure as well as task
difficulty.

## KYO v1–v6 chronology

| Version | Submitted (UTC) | Lines | Definitions | Qualifier mean | First hidden-race mean | Cost | Calls | Failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| v1 | Oct 24 17:20 | 1,456 | 59 | 0.8000 | 0.4111 (R3) | $3.7570 | 1,317.5 | 0.5 |
| v2 | Oct 25 18:37 | 1,688 | 67 | 0.7833 | 0.3778 (R4) | $4.6871 | 1,586.0 | 2.0 |
| v3 | Oct 26 18:45 | 1,855 | 72 | 0.7500 | 0.0333 (R5) | $0.6427 | 441.5 | 196.0 |
| v4 | Oct 27 17:29 | 2,187 | 86 | 0.7833 | 0.0111 (R6) | $5.0373 | 2,334.0 | 61.0 |
| v5 | Oct 28 18:46 | 2,332 | 91 | 0.8167 | 0.1333 (R7) | $4.7202 | 2,398.5 | 0.5 |
| v6 | Oct 29 17:45 | 2,647 | 108 | 0.8167 | 0.4056 (R8) | $4.5558 | 2,353.0 | 0.5 |

KYO_v6 was submitted 18h 27m after the later KYO_v5 Race 7 execution
completed. It became the public top agent at 2026-11-01 02:16:52 UTC, 2d 8h
31m after submission and 7d 8h 57m after KYO_v1. It is recorded as winner for
Races 10, 11, and 12.

The KYO line was cumulative: no named definitions were removed from v1 through
v6. The versions progressively added:

- **v2**: sold-out detection/replacement, offer rereads, ID resolution;
- **v3**: cart-state extraction, offer-tool identification, cart completion;
- **v4**: explicit process auditing, richer conversation handling, dependent
  listing trails, multi-record opening, model-call work separation;
- **v5**: process gate, cart fingerprinting, message reserve, endgame reserve;
- **v6**: dynamic result-shape reading, page traversal, cheapest-set coverage,
  facts-tool discovery, observation/view caching, cross-item compatibility,
  and pre-checkout evidence completion.

The source grew because the control plane became more explicit, not because the
organization switched to a more expensive model. KYO v1–v6 use the same model
trio: Claude Haiku 5.5, Claude Sonnet 5.5, and GPT-6 Luna.

## Why KYO_v6 was created

### The immediate problem

KYO_v3 and v4 demonstrated that a capable semantic policy is useless when the
inference path collapses: v3 averaged 196 inference failures; v4 crossed $5,
averaged 61 failures, and scored 0.0111. KYO_v5 stabilized inference, added a
process gate, and still scored only 0.1333 in Race 7. That isolated a behavioral
problem after infrastructure noise was largely removed.

The Race 7 public diagnostics identify the behavioral bottleneck:

| Category | KYO_v5 R7 | KYO_v6 R8 | Change |
|---|---:|---:|---:|
| Passed | 24/180 (13.3%) | 73/180 (40.6%) | +27.2 pp |
| Process requirement missed | 81/180 (45.0%) | 42/180 (23.3%) | -21.7 pp |
| Request not met | 62/180 (34.4%) | 47/180 (26.1%) | -8.3 pp |
| Not the best option | 13/180 (7.2%) | 14/180 (7.8%) | +0.6 pp |
| Mean environment steps | 38.64 | 33.54 | -5.09 |

Each row combines two 90-episode validator runs. These are different races, so
the table should be read as a before/after diagnostic signal, not an A/B test.

### The architectural diagnosis encoded in the v6 diff

KYO_v5 already had a good high-level decomposition:

```text
request + tool schemas
        ↓
structured brief and role assignment
        ↓
interview and requirement ledger
        ↓
search / open / compare / cart loop
        ↓
process gate and final evidence audit
        ↓
checkout or justified refusal
```

Its weakness was between the loop and the audit: the final checker could only
be as reliable as the evidence collected. In unfamiliar generated stores, tool
names, argument names, row-list keys, pagination tokens, price keys, stock keys,
and the separation between attributes and offers vary. An LLM policy could make
a locally plausible move while silently omitting a page, reading a summary as a
full record, reopening the same information, or comparing prices without the
full fitting set.

KYO_v6 moved these responsibilities out of free-form policy behavior and into a
stateful supervisory layer:

1. **Learn the store's result schema once.** `read_shape` and the `SHAPE`
   prompt identify row, count, continuation, price, and stock fields, then
   validate the proposed keys against the actual result and schema.
2. **Treat search coverage as state.** `paging`, `more_pages`, and
   `AUTO_PAGES` compare reported matches with observed rows and page through
   bounded result sets when an extreme-value decision requires full coverage.
3. **Separate summaries, offers, and facts.** `facts_tool`, `compare_facts`,
   `compare_bare`, and `fill_facts` determine which tool exposes attributes and
   fill missing records before judging a candidate.
4. **Make cheapest selection evidence-driven.** `price_of`, `price_sorted`,
   `cheapest_here`, `cheaper_rows`, and `wider_hold` search below the current
   cart price and open a bounded number of cheaper alternatives.
5. **Avoid paying twice for the same observation.** `note_view` and
   `shown_already` track which fields have been seen for each listing at the
   current state version; changed price/stock invalidates the cache.
6. **Improve dependent and cross-item reasoning.** The guide and `trail`
   require opening referenced listings rather than trusting search rows, and
   require jointly selecting compatible main products/accessories.
7. **Preserve semantic typing.** The guide explicitly rejects a bare number as
   evidence when its text does not identify the measured property, and the
   brief stops turning narrative context into a hard requirement without an
   explicit property value.

Seventeen named methods were added and twelve changed from v5. None of those 29
exact implementations, and none of the five changed/added prompt values, occurs
in any earlier released organization source. This supports the conclusion that
v6 was a genuine KYO-line redesign rather than a hidden transplant from one of
the other nine initial branches.

### The cost motivation

The v6 design also responds to cost and deadline pressure:

- model choice did not change;
- soft/hard spend controls were not raised;
- repeat views were suppressed;
- bounded automatic paging replaced extra model turns for small result sets;
- the final audit received more complete evidence with fewer environment steps;
- Haiku handled request parsing first, with Sonnet used only when essential
  structure was missing;
- the parser output allowance increased because truncated Haiku briefs had
  forced roughly a tenth of Race 6 briefs into the heavy fallback, according to
  the source comment.

The first v6 hidden run cost about $0.16 less than v5, used about 46 fewer model
calls, and took about 27 seconds less, while its score tripled. These are small
efficiency gains, but they refute the idea that v6 won merely by spending more.

### The real design objective

The best-supported interpretation is:

> KYO_v6 was built to turn an LLM shopper from a free-form tool user into a
> schema-adaptive, evidence-complete state machine whose expensive reasoning is
> reserved for ambiguity rather than routine store mechanics.

That is why it became a foundation. The reusable asset was not one prompt. It
was the supervisory structure that connected request interpretation, dynamic
tool semantics, a versioned evidence ledger, bounded exploration, and guarded
termination.

## What KYO_v6 did not solve

KYO_v6 was a successful foundation, not a complete solution:

- `not_the_best_option` did not improve in the v5/v6 diagnostic comparison;
- later race means varied sharply: R8 0.4056, R9 0.2167, R10 0.3944, R11
  0.3833, R12 0.1667, R13 0.1556, and R24 0.2056;
- R12/R13 again showed high failure counts and cost pressure;
- dynamic schema interpretation still depended on an LLM-generated mapping;
- bounded paging and three cheaper-candidate opens are compromises, not proofs
  of global optimality;
- the fixed brief schema could still lose nuance in ambiguous or changing
  customer intent.

These limitations explain why the organization later adopted other champion
foundations instead of indefinitely extending KYO.

## KYO versus Miyagi: same organization, different technical lineage

| Property | KYO_v6 | Miyagi_v2 |
|---|---:|---:|
| Submitted | 2026-10-29 17:45 UTC | 2026-11-13 17:32 UTC |
| Released | 2026-11-01 02:16 UTC | 2026-11-17 03:39 UTC |
| Lines | 2,647 | 5,954 |
| Definitions | 108 | 284 |
| Qualifier score | 0.8167 | 0.8333 |
| Published race score | 0.2753 | 0.4305 |
| Best listed raw race mean | 0.4056 (R8) | 0.4833 (R26) |
| Mean cost on that race | $4.5558 | $3.4946 |
| Mean inference calls | 2,353 | 1,602 |

Miyagi_v2's nearest earlier organization source is Kogure_v1 (composite 0.569,
line Jaccard 0.672). Its closest eligible public champion is Giyu_v17
(composite 0.668, line Jaccard 0.720). Kogure_v1 itself is extremely close to
Giyu_v17 (composite 0.850, line Jaccard 0.913), and Giyu is derived from the
Cairn family.

Therefore:

- **organizational lineage:** KYO and Miyagi are connected by the same current
  coldkey portfolio and research process;
- **code lineage:** Miyagi is primarily Giyu/Cairn-family code, with an internal
  Kogure/Miyagi adaptation path;
- **methodological lineage:** Miyagi inherits the organization's portfolio,
  rapid-release assimilation, and recombination method pioneered during the
  KYO period.

Saying “KYO, Cairn, Giyu, and Miyagi are all the same group” would go beyond the
evidence. What is confirmed is that the KYO/Miyagi coldkey organization later
used public Cairn/Giyu code. The evidence does not establish that Cairn's or
Giyu's original operators are the same people.

## Final assessment

The organization's competitive advantage was not a single magical agent. It
was an experimental system with three layers:

1. **breadth:** many concurrent architectures and hotkeys;
2. **selection:** hidden-race evidence used to retain or retire branches;
3. **assimilation:** successful internal and public external designs rapidly
   transplanted into the portfolio.

KYO_v6 mattered because it was the first internally evolved architecture to
show that a control-plane redesign could improve hidden performance without a
qualifier gain or a larger model budget. Miyagi mattered because the
organization later applied the same experimental discipline to a stronger
external foundation.

The central lesson is not “copy KYO_v6” or “copy Miyagi.” It is to reproduce the
scientific mechanism: isolate an observable failure class, change the smallest
control layer that governs it, preserve released-source provenance, and judge
the change on hidden operational behavior, cost, and reliability rather than
public qualifier score alone.

## Reproducibility map

- `manifest.json` — 136-version source/release manifest
- `inventory.csv` — one auditable row per version
- `sources/` — 113 released artifacts only
- `api/status/` and `api/runs/` — public metadata snapshots
- `static-analysis.json` — AST, prompt, model, run, and nearest-prior results
- `prior-champion-comparison.json` — time-gated external champion comparisons
- `kyo-feedback-summary.json` — redacted KYO v5/v6 diagnostic aggregates
- `chain-ownership.json` — finalized-state ownership evidence
- `collect.py`, `analyze.py`, `compare_champions.py`,
  `collect_kyo_feedback.py`, `collect_chain.py`, `export_inventory.py` —
  reproducible collectors and analyzers

Similarity is evidence of shared source structure, not proof of authorship.
Current chain ownership is evidence of current administration, not historical
ownership. Race-to-race score changes are observational because hidden tasks
and runtime conditions differ.
