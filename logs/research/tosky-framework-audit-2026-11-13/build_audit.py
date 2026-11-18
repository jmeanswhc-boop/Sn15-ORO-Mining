"""Assemble source coverage and public benchmark context without modifying agents."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path.insert(0, str(PROJECT))


HELPERS = {
 'compact': 'Compact serialization is useful, but truncating the serialized JSON can cut a value or structural boundary. Consumers subsequently validate against the untruncated object, so displayed evidence and citeable evidence can disagree. Cache keys using truncated serialization can also collide.',
 'object_from_text': 'Accepts ordinary or fenced JSON and some surrounding prose. It validates only that the outer result is a dictionary, not each producer contract. It cannot distinguish a cut reply from an ordinary format fault because infer discards finish_reason.',
 'number': 'Decimal avoids ordinary binary floating-point arithmetic. However numeric NaN/Infinity and negative monetary amounts are accepted; commas are always removed, so a comma decimal is misread. Downstream comparisons may raise or use the wrong scale.',
 'single_number_in': 'Provides a useful fallback for a single observed numeral. Its caller treats an unparseable field as permission to accept an arbitrary model amount. It also inherits comma ambiguity and cannot resolve a multi-number price by itself.',
 'path_value': 'Works for ordinary nested keys and nonnegative array indices. A dot is always a separator, so a literal dotted key cannot be addressed as a leaf. Citing its parent object is possible but weakens field binding and is clipping-limited. Use runtime evidence IDs or an array of exact path segments.',
 'exact_scalar': 'Useful for detecting an opaque scalar in nested arguments. Any scalar occurrence is weaker than identity-field binding; an ID embedded in a comment field is not necessarily the action target or the owner of the cited fact.',
 'record_for_id': 'Locates a record by any matching scalar rather than a validated identity field. It can return a wrapper or echoed metadata. An identified record must also be the exclusive entity allowed to supply its eligibility facts.',
 'stock_value': 'Handles booleans, numeric quantities and some availability phrases without a product table. The phrase not in stock is interpreted as available; serialized Boolean strings are not resolved here. Numeric finiteness is unchecked.',
 'valid_schema': 'Checks core types, required keys, enums, additional properties, array sizes and numeric bounds; it correctly excludes Boolean integer quantities. String length/pattern, const, combinators and references are unsupported. The supplied 80 tool schemas include 80 maxLength and 40 minLength occurrences.',
 'agent_main': 'Provides the supported synchronous entrypoint and catches run exceptions without fabricating a grade. Construction occurs outside the try, so initialization errors escape. The exception path returns diagnostics without a separately budgeted recovery attempt.'}

METHODS = {
 '__init__': 'Separates requirements, obligations, candidates, scopes, cart state and several revisions. All state is per episode. There is no dollar allowance, uncertain-cost ledger, fatal-credit latch or elapsed-time deadline. The graph is represented primarily by mutable fields and free-text lists rather than dependency-bearing claims.',
 'left': 'Correctly derives the remaining world-step count. It does not represent remaining inference money, wall time or the complete cost of satisfying pending dependencies.',
 'role': 'Simple dynamic role lookup. Missing roles use an empty string; callers comparing an empty opening-tool name with an absent role can accidentally enter that role branch.',
 'tool_specs': 'Keeps planning grounded in the world-supplied tool schemas. An empty names list selects all schemas, although plan_action guards its own empty input. Repeatedly transmitting schemas for already bound deterministic actions adds cost.',
 'infer': 'Counts each wrapper-level physical attempt with ProxyClient max_retries=1 and has model fallback. It records reported cost but never admits or rejects requests by dollars, reserves no finalization budget, ignores 402 and finish_reason, and has no global deadline. A valid outer dictionary stops fallback even when its inner schema is unusable.',
 'execute': 'Preserves the complete opaque binding, consecutive step numbers and identical replay of an uncertain action. It stops new world calls after finished and blocks an unreconciled action. There is no timed/backoff policy for 429, and valid transport receipts are not necessarily successful actions.',
 'bootstrap': 'Builds a useful explicit contract and tool-role map. The returned structure lacks a full local schema check: a string hard field becomes characters, skipped items can leave noncontiguous indices, and duties have no source proof or explicit multiplicity. Malformed extraction falls back to a truncated request as one item, silently losing constraints.',
 'fallback_roles': 'Generic tool-verb matching is a reasonable schema fallback and is separate from prohibited product vocabulary tables. Requiring a tool to be unused can miss legitimate multifunction tools. It cannot replace semantic schema binding for unfamiliar tool descriptions.',
 'get_candidate': 'The item/opaque-ID pair keeps separate requested roles from sharing one mutable candidate object. Each object still mixes several observed pages and revisions; source-specific field ownership and validity are not first-class state.',
 'scope_key': 'Correctly removes the learned pagination argument to unite pages of a query. JSON key order is not canonical and serialization is clipped at 3000 characters, so equivalent queries can split and sufficiently long queries can collide.',
 'learn_shape': 'Reuses learned structural paths and validates some path existence and the pagination argument. Price, stock and ID keys are not validated against their claimed row fields. An invalid newly proposed shape can still be returned for immediate use. category_key is stored but does not enforce product type.',
 'absorb': 'Centralizes observation interpretation and delegates state updates. It calls a model after every action, including terminal receipts. A pass statement under the finished check does not return. Missing role names can match the empty opening tool. Parsed error text is logged without consistently preventing downstream state commits.',
 'absorb_search': 'Maintains query-local seen IDs, page count and stock/price observations. Updating a row does not invalidate candidate eligibility; missing new price fields retain old prices. Only 80 rows are ingested even when a null next token marks the scope complete. A completed filtered query does not prove complete relevant-catalog coverage.',
 'absorb_detail': 'Maintains tool-specific pages and invalidates evaluation after a new page. On the first page it stores the whole observation as detail even when record_for_id has isolated one product. With multiple views, tool names and per-view timestamps are removed from the displayed values, allowing old evidence to acquire a new global detail_step.',
 'absorb_cart': 'Requires an observed record and an integer quantity for each proposed line. It accepts a quantity if any numeric field equals it, including price, and does not verify that all observed lines were represented. It therefore cannot certify an exact cart from an incomplete model projection.',
 'explicit_empty_cart': 'Recognizes some empty cart containers, but it scans any top-level list with a broad key fragment. An empty recommendations/products list can be confused with the cart, while nested or unfamiliar empty-cart containers can be missed.',
 'absorb_reply': 'Keeps later shopper constraints in the pipeline and ties approvals to basket state. Any nonempty reply marks a question obligation answered, including confusion. Any nonapproval pending obligation can be completed by a message reply, including a read duty. The extracted free/refusal fields are not applied to their intended states.',
 'absorb_changes': 'Can refresh market offers and revise the budget. Empty text bypasses source containment. Budget units and exact changed prices are not applied; market changes do not revise candidate-evaluation dependencies. Requirement changes are reparsed with a second model call and a weakly typed delta.',
 'apply_contract_delta': 'Requires source support for additions and some budget changes. Drops are not source-checked, and additions run before drops: a same-property replacement erases the new need too. Quantity, objective, budget-unit, new-duty and indifference changes are not represented completely.',
 'observation_error': 'Recognizes only a truthy top-level error key. accepted:false, success:false, applied:false and textual errors are not sufficient to stop success-dependent commits.',
 'item_from_purpose': 'Extracts an item number from an internal purpose string. Procedures invoked without an item default to item one, so second-item page refreshes can fail to update their candidate. Structured item identity would avoid dependence on log wording.',
 'contract_summary': 'Offers a compact current view of items, budget, objective and relations. It omits budget_scope and the obligation states, and raw truncation can hide requirements. A summary is not a source-complete immutable decision snapshot.',
 'evaluate': 'Combines hard-requirement states with stock/price checks and rejects absent quotations. Product type is not a required result, exact free-text requirement copying is brittle, duplicate rows overwrite one another, and unknown/unavailable output is cached at the current revision. It cannot order evidence by per-view freshness and can accept arbitrary amounts from unparseable values.',
 'candidates_for': 'Filters by item/verdict and supplies a consistent starting order for simple cases. It sorts raw currency amounts and raw measurement values, so later pruning can use incomparable units before normalization.',
 'compatible_units': 'The runtime conversion cache avoids a fixed conversion registry. Missing units default to factor one; punctuation stripping can equate distinct symbol-only currencies. Failed inference is cached as incompatibility. Numeric factors are not checked for finiteness.',
 'basket_key': 'Tracks selected items, quantities, prices, request/market/cart revisions and observed cart lines. Including offer_step makes unchanged rereads revoke proofs; broad invalidation can create process loops. The 3000-character serialization limit can collapse distinct large baskets.',
 'basket_total': 'Performs Decimal arithmetic and applies runtime money-unit factors. It always requires prices, even with no monetary budget. The each mode constrains quantity-times-price, so a separate per-unit ceiling is not representable. Invalid and unknown combinations both return None.',
 'relation_ok': 'Checks model-returned quotations from opened pages and caches results. Quote count is not relation coverage: repeated proofs for one relation can omit another. Unknown/unavailable responses become False, and added reference pages are not cache dependencies.',
 'choose_basket': 'The bounded best-first combination search is useful for joint budget and relationship constraints. Its 600-node cutoff returns the same empty result as infeasibility. It forbids shared listing IDs across slots and never normalizes physical objective units. Each new relation combination can consume another model request.',
 'challenges': 'Makes some unresolved state visible for prompts and diagnostics. It is not the dependency scheduler: missing reference acquisition and format-only repair are absent, the list is truncated, and the run loop does not dispatch actions from these challenge objects.',
 'obligation_satisfied': 'Requires a positive recorded step and a matching basket for before-order duties. It does not verify the event type here, and all before-order duties are treated as cart-dependent, including one-time reads.',
 'reserve': 'Attempts to save world steps for cart work, refreshes and procedures. With no desired basket it reserves only one step. Extra-line removals, multi-action duties, reference acquisition, format recovery and inference time/money are not fully included.',
 'question_to_ask': 'Prioritizes mandatory questions and reserves approval messages. It hard-stops optional questions after two messages despite a larger budget, and does not reserve ordinary mandatory messages. Mutually exclusive ending branches are not modeled in the reservation.',
 'discover_questions': 'Uses request context and actual listing differences without a product-property registry. The cache key includes detail_step, so unchanged refreshes can repeatedly invoke discovery. It can keep proposing questions after the two-optional-message rule prevents using them.',
 'plan_action': 'Restricts the model to permitted tool names, validates arguments and checks target scalar occurrence. It permits two format attempts and has limited deterministic fallbacks. It still pays for planning known zero-argument actions and cannot bind the target to its actual identity argument field.',
 'perform': 'Combines planning, dispatch, observation and optional process proof. A successful transport receipt is treated as progress regardless of actual effect. A message can satisfy a pending read through absorb_reply. Repeated-action suppression lacks dependency revisions, while allowed repeats can loop on unchanged evidence.',
 'search_item': 'Uses a bounded number of alternate routes and avoids hardcoded taxonomies. It does not produce a proof that the broad relevant search space is covered, and the persistent action history can prevent a necessary same-query refresh after a material change.',
 'next_page': 'Preserves the initial query and changes its learned page argument without another planner call. False-like cursor values are treated as exhausted; pagination correctness depends on the learned shape and stale scope flags.',
 'open_candidate': 'Targets a selected listing and distinguishes detail from offer intent. It runs a model to build arguments even for known single-ID readers. Freshness is inferred from a later aggregate detail_step rather than source-specific current offer facts.',
 'next_unverified': 'Can avoid obviously noncompetitive raw-price alternatives. It skips every candidate that already has detail, including current unknowns, so evidence/format problems and newly available references have no recovery action here.',
 'unresolved_better': 'Identifies unread cheaper candidates and incomplete relevant scopes. It uses raw currency or physical amounts and is only an advisory run-loop condition; checkout_ready does not require it to be resolved.',
 'expected_cart': 'Builds a simple desired quantity map. Repeated listing IDs overwrite rather than aggregate, consistent with the optimizer forbidding shared IDs but incomplete for legitimate multi-role purchases of the same SKU.',
 'cart_ready': 'Requires equality with the recorded desired map and no dirty flag. The second return repeats the first condition. Its correctness is limited by the incomplete cart projection accepted upstream.',
 'checkout_ready': 'Provides a real gate for a full basket, current eligibility, stock, an observed offer, budget and due obligations. It is stronger than an unguarded forced checkout. It does not enforce the optimality frontier or dependencies on changed/reference observations.',
 'pending_process': 'Finds the next unsatisfied obligation at one of three stages. A stage acts as a dispatch phase even when the natural-language instruction only states a latest completion deadline, creating reference-before-order dependency cycles.',
 'do_process': 'Separates message duties from generic nonterminal actions. Read duties can still choose message tools and be completed by a reply; arbitrary provided procedural tools that lack a mapped role are excluded. There is no item field, explicit count/group or performed-action type contract.',
 'align_cart': 'Reads uncertain carts and verifies changes rather than assuming add means replace. It does not independently enforce before-selection duties. Repeated failed or misparsed receipts can cause unbounded same-cart cycles until another limit is hit.',
 'terminal': 'Checks readiness before ordering and some duties before declining. It has no infeasibility evidence requirement for decline. Terminal action argument planning and post-terminal observation parsing can consume inference money after the substantive decision is known.',
 'run': 'The explicit loop is auditable and has finite step/call/safety bounds. Its fixed order is not dependency-aware: only one pending candidate is evaluated, unknowns lack a repair route, before-order references wait for eligibility, failed before-selection duties can fall through to cart work, and best-effort decline conflates unresolved evidence with absence.'}

NESTED = {
 'Tosky.evaluate.evidence_value': 'Checks that a quotation occurs at a path in the row/detail value. It does not bind a first-comparison path to the selected product, disambiguate literal dotted keys, enforce numeric field meaning or determine source freshness. Truncating the citeable value at 2000 characters adds a second representation boundary.',
 'Tosky.choose_basket.rank': 'Ranks combinations by raw extreme metric and then normalized money, or by normalized money alone. The heap ordering is useful when each option dimension is monotone, but physical measure_unit is never used and objective aggregation across items is fixed rather than contract-typed.'}

DATACLASSES = {
 'Slot': 'Useful explicit requested-item state. Hard/soft constraints remain unsourced strings; there are no requirement IDs, withdrawal records, entity bindings or typed alternatives.',
 'Obligation': 'Useful deadline/kind/evidence-step state. It lacks source grounding, requested counts, grouping, item identity, earliest-start conditions and a cart-dependence flag.',
 'Candidate': 'Collects row, pages, price/stock, verdict and revisions. Evidence is reduced to quotation strings; per-field source/observation revisions and failed-interpretation diagnostics are not retained. measure_unit is written but never used in ranking.',
 'Scope': 'Collects query-local IDs, count and continuation. It lacks the predicate/canonical coverage relationship to the requested item and a catalog revision that can expire a previous completeness result.'}


def source_audit():
    path = PROJECT/'agents/TOSKY.py';raw = path.read_bytes();text = raw.decode('utf-8')
    start = time.monotonic();tree = ast.parse(text);parse_seconds = time.monotonic()-start
    entries = []
    def walk(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.ClassDef):walk(node.body, prefix+node.name+'.')
            elif isinstance(node, ast.FunctionDef):
                name = prefix+node.name
                note = NESTED.get(name) or METHODS.get(node.name) if prefix.startswith('Tosky.') else HELPERS.get(name)
                if name in NESTED:note = NESTED[name]
                assert note, name
                entries.append({'name': name, 'line': node.lineno, 'end_line': node.end_lineno, 'assessment': note})
                walk(node.body, name+'.')
    walk(tree.body)
    assert len(entries) == 61
    forbidden = {'eval','exec','__import__','open','write_text','write_bytes','unlink','chmod'}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports += [alias.name for alias in node.names]
        if isinstance(node,ast.ImportFrom):
            imports.append(node.module)
            assert node.module != 'os' or [n.name for n in node.names] == ['getenv']
        if isinstance(node,ast.Call):
            name = node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ''
            assert name not in forbidden,name
    assert set(imports) <= {'__future__','json','heapq','re','dataclasses','decimal','os','typing','src.agent.proxy_client'}
    assert len([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='agent_main']) == 1
    with patch('requests.sessions.Session.request',side_effect=AssertionError('No live HTTP')) as blocked:
        from src.agent.sandbox_executor import load_agent_from_file
        function=load_agent_from_file(str(path))
        assert function({'environment':{'binding':{'session_id':'offline'},'policy_view':{'query':'','tools':[], 'max_steps':0,'finished':True}}}) == []
        blocked.assert_not_called()
    output={'source':str(path.relative_to(PROJECT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'lines':len(text.splitlines()),
            'all_function_definitions_reviewed':len(entries),'ast_parse_seconds':parse_seconds,'allowed_imports':imports,
            'local_ast_security_preflight':'passed','working_tree_loader_dataclass_and_entrypoint_smoke':'passed',
            'official_backend_admission':'not run','functions':entries,'data_classes':DATACLASSES}
    (ROOT/'source-audit.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['TOSKY FUNCTION-BY-FUNCTION AUDIT','Source SHA-256: '+output['sha256'],'','Imports, model constants and entrypoint: locally parsed; public allowlist checked separately.','']
    for name,note in DATACLASSES.items():lines += [name, note, '']
    for entry in entries:lines += [f"{entry['name']} — lines {entry['line']}-{entry['end_line']}",entry['assessment'],'']
    (ROOT/'function-audit.txt').write_text('\n'.join(lines)+'\n')
    return {k:v for k,v in output.items() if k not in ('functions','data_classes')}


def benchmark_context():
    base=ROOT.parent/'tosky-race-022-2026-11-13'
    analysis=json.loads((base/'analysis.json').read_text());race=json.loads((base/'race-analysis.json').read_text())
    top=json.loads((ROOT/'public-top.json').read_text());data=top['data']
    records={}
    for name in ('pooh','chidori','tosky'):
        qualifier=analysis['agents'][name]['qualifying'];r=race['agents'][name]
        records[name]={'source': 'agents/ICDI_chidori_v4.py' if name=='tosky' else 'released '+name+' source',
                       'qualifying_correct':qualifier['correct'],'qualifying_executions':qualifier['tasks'],
                       'qualifying_correct_per_run':[run['correct'] for run in qualifier['runs']],
                       'race_number':22,'race_correct':r['correct'],'race_executions':r['executions'],
                       'mean_reported_run_usd':r['mean_usd'],'mean_sandbox_seconds':r['mean_sandbox_seconds'],
                       'inferences_per_episode':r['total_inferences']/r['executions'], 'failures':r['categories'],
                       'finished':r['end_reasons']['finished']}
    status=json.loads((ROOT/'pooh-status.json').read_text())['data']
    retained=[w for w in status['window'] if w['race_number'] in (21,22)]
    result={'checked_at':top['fetched_at'],'top':data,'benchmarks':records,
            'new_framework':{'source':'agents/TOSKY.py','qualifying_score':None,'race_score':None,'paid_evaluation_performed':False},
            'illustrative_first_race_raw_lead_over_incumbent':3*data['margin']+sum(w['delta'] for w in retained),
            'target_limitations':'Threshold and additive margin are time-varying. The first-race lead assumes the next three-race window retains races 21/22 for Pooh, missing new-agent history has zero delta, and unchanged rules; it is not a forecast. Public run costs are not independently verified invoices. Two included runs repeat 90 tasks, not 180 independent tasks.'}
    (ROOT/'benchmark-context.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    print(json.dumps({'source':source_audit(),'benchmark_context_saved':bool(benchmark_context())},indent=2))


if __name__=='__main__':main()
