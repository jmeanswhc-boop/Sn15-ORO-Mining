# ICDI_cn1: focused purchase-validation update

This candidate is based on the released Cairn v2 source, used with the user's
confirmation because the requested directory did not exist. The original archive
is intact. Only `ICDI_cn1.py` is the agent submission file; the tests, patch,
verification record, and this review are development artifacts.

## Requirement-preservation revision

The genuine local run `local-83164865b08e4e8f9176b3a69590c55f` scored 22/30
with 519 successful inference requests on the prior candidate. The user identified
`local-df6e57d22d0544a4acd18547aec4dc1d` as fake, so it is excluded from evidence.

This revision targets initial loss of explicit requirements and allowed product
alternatives. The existing extraction call now requests a quoted product phrase
and quoted explicit requirements. Valid source quotations preserve product
alternatives in the item name and add requirements to the normal mutable musts
list. Customer withdrawals continue through the existing dropped-requirement
logic. The implementation adds no inference calls.

Completed checkpoints include inspection of parsing, category ownership and
requirement construction, implementation in read_request, and 35 passing offline
tests including five new requirement-preservation regressions. Existing static
checks pass. These checks use mocked extraction and do not measure semantic
extraction accuracy, a new score, or server-side AST acceptance. The new review
and finalization behavior from the prior candidate still requires evaluation,
including the observed clock nontermination issue.

The analysis below describes the original purchase-validation revision and its
public-qualifier evidence. Its original test counts describe that earlier state.

## Evidence and conclusion

The analysis uses the 30 public qualifying requests for suite 160, Cairn v2's two
released qualifying result sets, all nine failed qualifying transcripts, and the
three passing transcripts for tasks that also failed. No race transcript, private
task, sealed answer, or grading implementation was used to design this change.
Comments about earlier races in the inherited source are not evaluation evidence
for this analysis.

The live public qualifying results show **26/30 and 25/30**, or **51/60 = 85%**.
All episodes completed; the recorded runs report no inference failures.

| Observed failure group | Episodes | Share of all 60 episodes | Decision |
| --- | ---: | ---: | --- |
| Purchases that do not meet the request | 4 | 6.67% | Target of this update |
| Undiscovered shopper needs | 3 | 5.00% | Interview behavior retained |
| Declines that do not meet the request | 2 | 3.33% | Search and decline behavior retained |

The four incorrect purchases recur on the same two requests in both validators:

- A request explicitly distinguishes the pack type. The purchased record supplies
  a unit count and an option description, while evidence for the requested pack
  type is absent. A related property is being treated as sufficient support.
- A request describes an optical capability reaching an endpoint. The purchased
  record gives a smaller range. Treating that range as merely below a ceiling
  loses the capability requirement's meaning.

These diagnoses are inferences from the original public wording, visible listing
records, and the reported `request_not_met` outcome. The released transcripts do
not include internal model reviews or detailed grading checks, so they do not
prove which internal decision caused each failure.

There is also a **directly reproduced implementation defect** in the same purchase
boundary. Baseline `check_cart()` adds a cart key to `checked` before inspecting
the review. It then returns an empty objection for every future check of that
key, even when its saved verdict is false. Baseline `finish()` can purchase the
rejected cart directly. The regression tests run the original code and reproduce
both bypasses before checking that ICDI_cn1 prevents them.

The selected weakness is therefore **purchase acceptance without a durable,
complete check of the evidence for the request**. It is the largest observed
failure group that can be addressed by one focused component. Recovering all four
targeted failures would represent 6.67 percentage points on this sample; that is
an opportunity ceiling, not a predicted or measured new score. The data cannot
prove that this is the largest possible improvement across unseen worlds.

## Plan and checkpoints

1. **Freeze the baseline and establish the evidence. Completed.** Verify the
   archived source hash, read the public requests, retrieve qualifying feedback,
   and classify all failures before choosing a component. Baseline SHA-256:
   `2344dcf9218708f75d4ff0dac8f37ea30c3d191a3df71daf1174c08b215ba500`.
2. **Prove feasibility before editing the decision flow. Completed.** Reproduce
   the repeated-check bypass with an explicit negative model verdict. Trace the
   normal, final-step, forced-finish, and recovery purchase paths. The agent
   already has the request, current requirements, and observed listing records;
   the correction requires no new tools or catalogue access.
3. **Replace the purchase review and close its bypasses. Completed.** Introduce
   numbered requirements and observed evidence, validate review coverage and
   citation provenance, cache the actual verdict against its inputs, and enforce
   that verdict at every purchase boundary. Preserve the rest of the strategy.
4. **Verify control flow, scope, and public submission rules. Completed locally.**
   Thirty offline tests pass. The AST comparison preserves 231 existing function
   or class definitions. The rule-oriented checks and their limits are below.
5. **Measure actual score and obtain platform acceptance. Pending external
   validation.** No OpenRouter key is configured here, and no submission was
   requested or made. A practice run would use a different world; it cannot be
   presented as a score on these qualifiers. A fresh qualifying evaluation and
   ORO's server-side analysis remain the acceptance gates.

## Component review

| Component | Logic and public-qualifier relevance | Finding and disposition |
| --- | --- | --- |
| Entry point and session setup | Copies the binding, reads dynamic tool schemas and the public step cap, then runs a single episode. | No completed-run crash pattern. Preserved. |
| Inference and resource limits | Light model handles ordinary turns; stronger models handle planning and reviews; retries, cost and elapsed-time limits bound work. | No recorded inference failures. Models and limits preserved; all three models verified on the current public OpenRouter allowlist. |
| Request decomposition | `read_request` extracts items, constraints, budget, choice rule, tool roles and process instructions. | Model paraphrases can weaken or overinterpret the original wording. The new terminal reviewer independently receives the original request; decomposition is preserved. |
| Schema shaping and identifiers | Coerces tool arguments, checks required fields and bounds, and resolves identifiers using seen rows. | Some failed transcripts contain recoverable cursor or comparison-size mistakes, but not a common terminal cause. Preserved. |
| Observation restoration | Restores text, tables, columnar rows and nested attributes; folds repeated key spellings. | Makes the catalogue available to later checks. Preserved exactly; review citations refer to the resulting observed records. |
| Ledger and revisions | Tracks explicit requirements, disclosed needs, drops, stock and budget updates. | The old ledger can downplay a poorly matched requirement. Numbered review coverage includes every current requirement, including those suppressed in the prose ledger. |
| Search and pagination | Learns result fields and cursors; uses ordinary searches, value searches, automatic pagination and narrower queries. | Nine public requests ask for the cheapest acceptable item. Search remains important, but no failed episode is categorized `not_the_best_option`. Preserved. |
| Catalogue exploration and cheapest search | Opens candidates, compares records, checks cheaper rows and searches below cart prices. | Broader changes would spend environment steps and affect successful tasks. Preserved. |
| Interview planning | Ranks observed property names, allocates the three-message allowance, and adapts multi-property questions to replies. | Three failures are categorized `needs_not_found`. A separate improvement opportunity; excluded from this update. |
| Reply interpretation | Reads shopper needs, dropped preferences and budget changes; skips some uninformative replies. | Preserved. `send` additionally retains the same already-received question/reply text for the purchase reviewer, with no added message or tool call. |
| Reference chains | Resolves requirements pointing to other observed listings and follows linked pages. | No demonstrated dominant failure in these public qualifiers. Preserved. |
| Compatibility logic | Parses device/part references and checks relationships among requested items. | The public requests concern single requested product selections. No evidence to justify changing compatibility. Preserved. |
| Cart accounting | Tracks quantities, stock, categories and removals; prepares repairs. | Existing mechanics preserved. Review cache now includes cart contents and the current records so an earlier verdict cannot survive a changed cart. |
| Changes and stock recovery | Reads notes, refreshes offers, identifies unavailable lines and finds replacements. | No `changes_missed` failure in this sample. Preserved. |
| Pre-purchase review | Previously requested a global `ok` and cached only that a check had occurred. | Replaced with complete per-requirement findings and evidence citations. Negative verdicts persist. This is the target. |
| Decline audit | Searches and independently reviews candidates before ending without a purchase. | Two unsuccessful declines remain a separate opportunity. The update does not turn a missing purchase review into a claim that no suitable listing exists. |
| Process gates | Performs task-requested steps around checkout or decline. | Preserved. The final purchase review runs after these gates, so newly disclosed needs invalidate prior approval. |
| Forced finish and recovery | Can set a recovered cart and terminate directly. | Purchase calls now use the same evidence check as ordinary checkout; the transport boundary provides a final guard. |

## Implemented behavior

The replacement reviewer must address each requested product and each live
requirement. It receives the original request, shopper replies, current ledger,
cart, observed listing fields, and the existing alternative-listing comparison.
The prompt distinguishes required capability coverage from an upper bound, and
requires support for the actual property rather than a related value.

The code checks that every requirement appears exactly once, every cited field
exists and belongs to the cart listing under review, one item's requirements are
not distributed across different products, every cart line is accounted for,
and approval is the literal Boolean `true`. Missing or contradictory evidence
cannot be hidden behind a global positive answer. These checks verify coverage
and provenance; **the model still judges semantic entailment**. A correct citation
alone is not mathematical proof that the product fits.

The cache stores the reviewed decision, not merely a checked flag. Its key covers
the request, items and live needs, replies, budget, cart, observed records,
unavailability and the alternatives used for an optimum comparison. Repeating an
unchanged rejected purchase stays blocked without spending another model call.
New relevant evidence or a changed task causes a fresh review. The existing
stronger-model fallback can resolve a weaker review's false alarm.

Normal successful reviews still use one model call in the common case. There are
no extra search or shopper calls. Review output can be longer; inference cost and
latency must be measured in a live run. Recovery purchases may require a second
review after changing the actual cart. An inference outage cannot authorize a
purchase and remains retryable. That behavior, and stricter semantic judgments,
can reduce performance if the reviewer is wrong or unavailable; unchanged search
logic alone is not proof of unchanged end-to-end score.

## Verification and submission boundary

Run the checks from the repository root:

```bash
python3 -m unittest discover -s agents/1110_icdi_cnb1 -p 'test_*.py' -v
```

Thirty tests pass, using invented shopping data and mocked inference. They cover
the baseline bypasses, coverage, citation provenance, cached decisions, state
invalidation, stronger-model fallback, every purchase route, an outage, message
accounting, and preservation of unrelated definitions and configuration.
`verification.json` records the source hash, exact scope, evidence counts, and
validation limits. `ICDI_cn1.patch` makes the change reviewable.

Local checks confirm valid UTF-8 Python, one synchronous `agent_main`, source
below 1 MB, no prohibited imports or file operations, and no overlap between
agent string constants and 143 observed public task/product identifiers. The new
logic adds no product, attribute, unit, or answer table and uses only episode
inputs and observations. It adds no clock-based strategy routing. The inherited
elapsed-time guards remain resource limits.

The public [code requirements](https://app.oroagents.com/docs/miners/code-requirements)
also require meaningful original functionality when building on a released agent.
This update adds a reachable purchase-evidence mechanism, not cosmetic renaming
or unused padding. Nevertheless, this is a focused Cairn fork and remains
structurally close to it. **Local AST checks cannot certify ORO's private
structural-similarity/integrity decision.** Neither submission acceptance nor an
improved qualifying score is claimed.

## Public sources and retained evidence

- [Cairn v2 qualifying results and requests](https://api.oroagents.com/v1/public/agent-versions/06aef3d7-bf0f-779c-8000-ee9b2a46ce76/problems?phase=QUALIFYING)
- [Public qualifying requests for suite 160](https://api.oroagents.com/v1/public/suites/160/problems)
- [Cairn v2 evaluation runs](https://api.oroagents.com/v1/public/agent-versions/06aef3d7-bf0f-779c-8000-ee9b2a46ce76/runs)
- [Public inference model allowlist](https://api.oroagents.com/v1/public/inference/models?provider=openrouter)
- Local raw feedback: `logs/research/icdi_cn1/qualifying/`. Individual feedback
  files are named by their public episode IDs and include `phase: QUALIFYING`.

The task-by-task table below refers to public request IDs for research provenance.
These IDs and their task content are not embedded in the submission source.

| # | Public task | Request focus | Run 1 | Run 2 | Failed-episode finding |
| ---: | --- | --- | --- | --- | --- |
| 1 | `t-0040fbe66b` | Router: specified voltage; cheapest | pass | pass | No failure in either run |
| 2 | `t-108fbf1075` | Cola: dietary requirement; cheapest | pass | pass | No failure in either run |
| 3 | `t-19e695ad40` | Sharing multipack: packaging requirement; cheapest | pass | pass | No failure in either run |
| 4 | `t-20fcb00ff0` | Dining set: seating feature | fail | fail | Undiscovered need |
| 5 | `t-275498f5da` | Glasses: frame construction | pass | pass | No failure in either run |
| 6 | `t-2f74bb051d` | Sneakers: toe style | pass | pass | No failure in either run |
| 7 | `t-3254025544` | Cola: single pack type | fail | fail | Unsupported purchase (target) |
| 8 | `t-32cbbe307e` | Sweatshirt/hoodie: material; cheapest | pass | pass | No failure in either run |
| 9 | `t-3553587b35` | Maternity bottoms: occasion | pass | pass | No failure in either run |
| 10 | `t-38edeb49f9` | Dining set: material; cheapest | pass | pass | No failure in either run |
| 11 | `t-43bc201724` | Candy: packaging | pass | pass | No failure in either run |
| 12 | `t-4818b794cc` | Dress: style and context | pass | pass | No failure in either run |
| 13 | `t-4c419a2ae3` | Shirt: neckline | pass | pass | No failure in either run |
| 14 | `t-52bddabbe7` | Television: minimum screen measure | pass | pass | No failure in either run |
| 15 | `t-56b3551396` | Sneakers: style and exclusion | pass | pass | No failure in either run |
| 16 | `t-58da11237f` | Router: voltage and exclusion; cheapest | pass | pass | No failure in either run |
| 17 | `t-6d9462687c` | Dining set: base | pass | pass | No failure in either run |
| 18 | `t-6fd2a792df` | Cola: dietary requirement and questions | fail | pass | Undiscovered need |
| 19 | `t-9bcd8e8a37` | Shirt: material | pass | pass | No failure in either run |
| 20 | `t-9dd842ad5e` | Seasonal footwear: pattern and exclusion | pass | pass | No failure in either run |
| 21 | `t-a38af7fbd2` | Sweater/cardigan: fit | pass | pass | No failure in either run |
| 22 | `t-a9bf915ffc` | Sweatshirt/hoodie: size and exclusion; cheapest | pass | pass | No failure in either run |
| 23 | `t-b47de9595e` | Sweater/cardigan: color; cheapest | pass | pass | No failure in either run |
| 24 | `t-b834e3e526` | Sweatshirt/hoodie: color | pass | pass | No failure in either run |
| 25 | `t-cfa0144fb0` | Mary janes: color and exclusion | pass | pass | No failure in either run |
| 26 | `t-dea5de2a48` | Preserves: pack type | pass | pass | No failure in either run |
| 27 | `t-e0ee1dc390` | Shirt: style | pass | pass | No failure in either run |
| 28 | `t-e622afe641` | Sweater/cardigan: material and arrival context | pass | fail | Incorrect decline |
| 29 | `t-fa3dff6e42` | Seasonal footwear: fastening and context | pass | fail | Incorrect decline |
| 30 | `t-fc0b16b797` | Glasses: capability and alternatives; cheapest | fail | fail | Unsupported purchase (target) |
