"""Offline recovery-boundary checks. Actual HTTP is prohibited."""
import ast
import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from test_chidori_v3 import load, problem, response, Client, OfflineCase
from test_chidori_v4 import FULL as baseline, disputed_episode

ROOT = Path(__file__).resolve().parents[2]
agent = load(ROOT/'agents/Tosky_v3.py', 'tosky_v3_recovery')


def episode(module=agent):
    ep = disputed_episode(module)
    ep.room = lambda: 100
    ep.clock = lambda: 0
    ep.price_key = 'price';ep.stock_key = 'available';ep.budget = 100
    ep.rows['new'].update(available=True, category='equipment', name='Requested equipment')
    ep.full['new'].update(ep.rows['new'])
    ep.opened.add('new');ep.views['new'] = {'read'}
    return ep


def typed_answer(snapshot, *, state='met', same_scope=True):
    wire = snapshot['wire'];candidates = []
    for candidate in wire['candidates']:
        facts = [(eid, field) for eid, field in wire['facts'].items() if field['cid'] == candidate['cid']]
        kind = next(eid for eid, field in facts if field['scope'] == 'listing')
        attribute = next((eid for eid, field in facts if field['scope'] == 'attribute'), kind)
        candidates.append({'cid': candidate['cid'], 'kind': 'match', 'kind_evidence': [kind],
                           'needs': [{'rid': r['rid'], 'state': state, 'same_scope': same_scope,
                                      'evidence': [attribute] if state != 'unknown' else []} for r in wire['requirements']]})
    return {'revision': wire['revision'], 'candidates': candidates}


def recorded_episode(decision):
    ep = episode();packet = decision['packet']
    selected = (decision['answer'].get('assignments') or [{}])[0].get('id')
    if not selected:
        selected = packet['records'][0]['id']
    details = {r['id']: r for r in packet['records']}
    ep.rows = {ident: dict(r['listing']) for ident, r in details.items()}
    ep.full = {ident: dict(r['listing']) for ident, r in details.items() if r['attributes']}
    ep.public_attributes = lambda ident: details[ident]['attributes']
    ep.order = list(details);ep.opened = set(details)
    ep.items = [agent.Item(packet['items'][0]['name'], '', packet['items'][0]['quantity'],
                          [need for number, need in packet['live'] if number == 1])]
    # Projection for codec tests: only the selected record and its captured
    # requirements matter; semantic outcomes are not rescored here.
    if not ep.items[0].musts:ep.items[0].musts = ['Unresolved request: retained for format diagnosis']
    ep.brief.update(pick='any', in_stock_only=True, budget_scope=packet['budget_scope'])
    ep.budget = packet['budget'];ep.stock_key = 'in_stock'
    ep.cart_lines = {selected: packet['items'][0]['quantity']}
    ep.request = packet['request']
    return ep, selected


class SnapshotTests(OfflineCase):
    def test_all_previous_versions_are_byte_identical(self):
        path = ROOT/'logs/research/tosky-v3-recovery-2026-11-13/preserved-sources.json'
        for entry in json.loads(path.read_text()):
            self.assertEqual(hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest(), entry['sha256'])

    def test_global_limits_are_not_product_requirements(self):
        ep = episode();snap = ep.boundary_snapshot(['new'])
        self.assertEqual([r['text'] for r in snap['wire']['requirements']], ['Feature: wanted'])
        self.assertEqual(snap['wire']['limits']['budget'], 100)
        self.assertEqual(snap['wire']['limits']['max_steps'], ep.budget_steps)

    def test_runtime_fact_values_keep_json_types(self):
        ep = episode();snap = ep.boundary_snapshot(['new'])
        values = {(e['scope'], e['key']): e['value'] for e in snap['wire']['facts'].values()}
        self.assertIs(values['listing', 'available'], True)
        self.assertEqual(values['listing', 'price'], 2)
        self.assertIsInstance(values['listing', 'price'], int)

    def test_observed_empty_values_are_reported_and_cannot_be_positive_evidence(self):
        for value in (None, '', '   ', [], {}):
            ep = episode();ep.rows['new']['description'] = value
            snapshot = ep.boundary_snapshot(['new'])
            self.assertIn({'cid': 'c1', 'scope': 'listing', 'key': 'description'}, snapshot['wire']['empty_fields'])
            self.assertFalse(any(f['key'] == 'description' and f['scope'] == 'listing' for f in snapshot['wire']['facts'].values()))

    def test_any_objective_has_no_unrelated_alternatives(self):
        ep = episode();snap = ep.boundary_snapshot(['new'])
        self.assertEqual(snap['ids'], ['new'])

    def test_cheapest_objective_includes_lower_priced_known_candidate(self):
        ep = episode();ep.brief['pick'] = 'cheapest'
        with patch.object(ep, 'owner', return_value=1):snap = ep.boundary_snapshot(['new'])
        self.assertEqual(snap['ids'], ['new', 'old'])

    def test_complex_and_reference_decisions_keep_the_original_path(self):
        ep = episode();ep.refs = [(1, 'Feature: wanted')]
        self.assertIsNone(ep.boundary_snapshot(['new']))
        ep = episode();ep.items.append(agent.Item('another', '', 1, []))
        self.assertIsNone(ep.boundary_snapshot(['new']))

    def test_large_fields_are_explicitly_omitted_not_silently_clipped(self):
        ep = episode();ep.rows['new']['description'] = 'x'*900
        snap = ep.boundary_snapshot(['new'])
        self.assertIn({'cid': 'c1', 'scope': 'listing', 'key': 'description'}, snap['wire']['omitted_fields'])
        self.assertFalse(any(e['key'] == 'description' for e in snap['wire']['facts'].values()))

    def test_relevant_long_cached_text_uses_exact_marked_excerpts(self):
        ep = episode();ep.items[0].musts = ['surface: satin']
        value = 'General information. '*50+'The surface is satin. '+'Other details. '*70
        ep.rows['new']['description'] = value
        snapshot = ep.boundary_snapshot(['new'])
        eid, fact = next((eid, f) for eid, f in snapshot['wire']['facts'].items() if f['key'] == 'description')
        self.assertEqual(fact['value'], value);self.assertTrue(fact['preview']['partial'])
        self.assertEqual(fact['preview']['full_characters'], len(value))
        self.assertTrue(all(part in value for part in fact['preview']['parts']))
        self.assertTrue(any('The surface is satin.' in part for part in fact['preview']['parts']))
        packet = json.loads(ep.boundary_prompt(snapshot).split('SNAPSHOT:\n', 1)[1])
        self.assertEqual(packet['facts'][eid][3], fact['preview'])
        answer = typed_answer(snapshot);answer['candidates'][0]['needs'][0]['evidence'] = [eid]
        self.assertEqual(ep.boundary_decode(answer, snapshot)['status'], 'APPROVED')

    def test_excerpt_keeps_local_negation_and_does_not_certify_unknown_context(self):
        ep = episode();ep.items[0].musts = ['surface: satin']
        ep.rows['new']['description'] = 'Other details. '*80+'The surface is not satin. '+'Other details. '*70
        snapshot = ep.boundary_snapshot(['new'])
        fact = next(f for f in snapshot['wire']['facts'].values() if f['key'] == 'description')
        self.assertTrue(any('The surface is not satin.' in part for part in fact['preview']['parts']))
        self.assertIn('use unknown if its missing context matters', ep.boundary_prompt(snapshot))
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot, state='unknown'), snapshot)['status'], 'MISSING_EVIDENCE')

    def test_signature_tracks_real_evidence_and_not_a_step_counter(self):
        ep = episode();snap = ep.boundary_snapshot(['new']);ep.step += 1
        self.assertEqual(ep.boundary_signature(snap['ids']), snap['signature'])
        ep.rows['new']['price'] = 3
        self.assertNotEqual(ep.boundary_signature(snap['ids']), snap['signature'])

    def test_healthy_primary_review_request_matches_the_baseline(self):
        bodies = []
        for module in (baseline, agent):
            ep = episode(module);ep.verdicts = {};ep.checked = set();ep.review_contexts = {}
            client = Client([response({'ok': True, 'problems': [], 'better': []})])
            with patch.object(module, 'model_client', return_value=client), patch.object(ep, 'weak_lines', return_value=[]), patch.object(ep, 'code_fits', return_value=[]):
                self.assertEqual(ep.cart_check(), '')
            bodies.append(client.posts)
        self.assertEqual(bodies[0], bodies[1])

    def test_unavailable_primary_review_does_not_create_a_new_negative_verdict(self):
        ep = episode();ep.verdicts = {};ep.checked = set();ep.review_contexts = {}
        with patch.object(ep, 'json_ask', return_value=None), patch.object(ep, 'weak_lines', return_value=[]), patch.object(ep, 'code_fits', return_value=[]):
            self.assertEqual(ep.cart_check(), '')
        self.assertNotIn('new#0', ep.verdicts)
        with patch.object(ep, 'json_ask', return_value={'ok': True}), patch.object(ep, 'weak_lines', return_value=[]), patch.object(ep, 'code_fits', return_value=[]):
            self.assertEqual(ep.cart_check(), '')
        self.assertTrue(ep.verdicts['new#0'])


class CodecTests(OfflineCase):
    def test_valid_identifiers_approve_without_copying_field_values(self):
        ep = episode();snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'APPROVED')


class BoundaryIntegrationTests(OfflineCase):
    def run_boundary(self, ep, replies):
        client = Client(replies)
        with patch.object(agent, 'model_client', return_value=client), patch.object(agent.time, 'sleep') as sleep, patch.object(ep, 'env', side_effect=AssertionError('Review correction must not call the world')):
            approved = ep.adjudicate_review(ep.cart_ids())
        sleep.assert_not_called()
        return approved, client.posts

    def next_answer(self, ep):
        snapshot = ep.boundary_snapshot(ep.cart_ids())
        answer = typed_answer(snapshot)
        # Production builds the next local snapshot at entry to adjudication.
        answer['revision'] += 1
        return answer

    def test_grounded_negative_review_correction_is_cached_after_promotion(self):
        ep = episode();answer = self.next_answer(ep)
        ok, posts = self.run_boundary(ep, [response(json.dumps(answer))])
        self.assertTrue(ok);self.assertTrue(ep.verdicts['new#0'])
        self.assertNotIn('new#0', ep.review_messages)
        self.assertEqual(ep.review_contexts['new#0'], ep.cart_review_context(['new']))
        self.assertEqual(len(posts), 1);self.assertEqual(posts[0][1]['max_tokens'], 768)
        self.assertEqual(posts[0][1]['reasoning'], {'enabled': False})
        with patch.object(ep, 'json_ask', side_effect=AssertionError('No repeat review')):
            self.assertEqual(ep.cart_check(), '')

    def test_format_fault_repairs_representation_before_any_world_action(self):
        ep = episode();answer = self.next_answer(ep)
        bad = json.loads(json.dumps(answer));bad['candidates'][0]['needs'][0]['evidence'] = 'It is suitable.'
        ok, posts = self.run_boundary(ep, [response(json.dumps(bad)), response(json.dumps(answer))])
        self.assertTrue(ok);self.assertEqual(len(posts), 2)
        content = posts[1][1]['messages'][0]['content']
        self.assertIn('FORMAT-ONLY CORRECTION', content)
        self.assertIn('candidates[0].needs[0].evidence', content)
        self.assertEqual([p[1]['model'] for p in posts], [agent.SPARE, agent.LIGHT])
        self.assertEqual(ep.calls, 2);self.assertEqual(ep.inference_budget.pending, 0)

    def test_all_captured_prose_and_string_value_errors_get_format_only_handoff(self):
        decisions = json.loads((ROOT/'logs/research/tosky-v2-local10-2026-11-13/recorded-decisions.json').read_text())
        checked = {'prose': 0, 'value': 0}
        for decision in decisions:
            if decision['role'] != 'cart_proof':continue
            ep, ident = recorded_episode(decision);snapshot = ep.boundary_snapshot([ident])
            fault = ep.boundary_decode(decision['answer'], snapshot)
            category = 'prose' if fault['path'].endswith('.evidence') else 'value' if fault['path'].endswith('.value') else None
            if not category:continue
            key = ident+'#'+str(ep.version);ep.verdicts = {key: False};ep.review_messages = {key: 'Original negative review.'}
            corrected = typed_answer(snapshot);corrected['revision'] += 1
            ok, posts = self.run_boundary(ep, [response(json.dumps(decision['answer'])), response(json.dumps(corrected))])
            self.assertTrue(ok, decision['task_id']);self.assertEqual(len(posts), 2)
            self.assertIn(fault['path'], posts[1][1]['messages'][0]['content'])
            checked[category] += 1
        self.assertGreaterEqual(checked['prose'], 6);self.assertGreaterEqual(checked['value'], 3)

    def test_invalid_or_unavailable_correction_preserves_original_negative_reason(self):
        failure = SimpleNamespace(data=None, error={'status': 500, 'body': 'temporary'})
        for replies in ([response('[]'), response('{}')], [failure, failure]):
            ep = episode();original = dict(ep.review_messages)
            ok, posts = self.run_boundary(ep, replies)
            self.assertFalse(ok);self.assertEqual(len(posts), 2)
            self.assertFalse(ep.verdicts['new#0']);self.assertEqual(ep.review_messages, original)
            self.assertEqual(ep.inference_budget.pending, 0)
            count = ep.calls;ep.step += 1
            again, posts = self.run_boundary(ep, [])
            self.assertFalse(again);self.assertEqual(posts, []);self.assertEqual(ep.calls, count)

    def test_mismatch_unknown_and_better_option_return_specific_planner_feedback(self):
        for state, expected in [('mismatch', 'MISMATCH'), ('unknown', 'MISSING_EVIDENCE')]:
            ep = episode();snapshot = ep.boundary_snapshot(['new']);answer = typed_answer(snapshot, state=state);answer['revision'] += 1
            ok, posts = self.run_boundary(ep, [response(json.dumps(answer))])
            self.assertFalse(ok);self.assertEqual(len(posts), 1);self.assertFalse(ep.verdicts['new#0'])
            self.assertIn('Feature: wanted', ep.review_messages['new#0'])
            self.assertEqual(list(ep.boundary_results.values())[0]['status'], expected)
        ep = episode();ep.brief['pick'] = 'cheapest'
        with patch.object(ep, 'owner', return_value=1):
            answer = self.next_answer(ep)
            ok, posts = self.run_boundary(ep, [response(json.dumps(answer))])
        self.assertFalse(ok);self.assertIn('old', ep.review_messages['new#0'])
        self.assertEqual(list(ep.boundary_results.values())[0]['status'], 'BETTER_OPTION')

    def test_402_stops_without_retry_or_credit_reopening(self):
        ep = episode();original = dict(ep.review_messages)
        ok, posts = self.run_boundary(ep, [SimpleNamespace(data=None, error={'status': 402, 'body': 'cap'})])
        self.assertFalse(ok);self.assertTrue(ep.no_credit);self.assertEqual(len(posts), 1)
        self.assertEqual(ep.review_messages, original)
        self.assertEqual(self.run_boundary(ep, [])[1], [])

    def test_compatibility_retry_counts_toward_the_same_two_transmissions(self):
        bad = SimpleNamespace(data=None, error={'status': 422, 'body': 'unsupported reasoning'})
        ep = episode();answer = self.next_answer(ep)
        ok, posts = self.run_boundary(ep, [bad, response(json.dumps(answer))])
        self.assertTrue(ok);self.assertEqual(len(posts), 2)
        self.assertIn('reasoning', posts[0][1]);self.assertNotIn('reasoning', posts[1][1])
        self.assertGreater(ep.inference_budget.uncertain, 0)
        self.assertFalse(ep.reasoning_unsupported)
        self.assertEqual(ep.boundary_reasoning_unsupported, {agent.SPARE})
        ep = episode()
        ok, posts = self.run_boundary(ep, [bad, response('{}')])
        self.assertFalse(ok);self.assertEqual(len(posts), 2)
        self.assertFalse(ep.verdicts['new#0'])

    def test_transport_timeout_and_missing_cost_are_reserved_as_uncertain(self):
        ep = episode();answer = self.next_answer(ep)
        no_usage = response(json.dumps(answer));del no_usage.data['usage']
        ok, posts = self.run_boundary(ep, [no_usage])
        self.assertTrue(ok);self.assertGreater(ep.inference_budget.uncertain, 0)
        ep = episode();client = Client([])
        with patch.object(agent, 'model_client', return_value=client), patch.object(client, 'post_verbose', side_effect=TimeoutError('offline uncertainty')) as post:
            self.assertFalse(ep.adjudicate_review(['new']))
        self.assertEqual(post.call_count, 2);self.assertGreater(ep.inference_budget.uncertain, 0)
        self.assertEqual(ep.inference_budget.pending, 0)

    def test_evidence_or_requirement_change_during_correction_cannot_promote(self):
        for mutation in ('price', 'quantity', 'need', 'availability', 'notice'):
            ep = episode();answer = self.next_answer(ep)
            def changing_post(*_):
                if mutation == 'price':ep.rows['new']['price'] += 1
                elif mutation == 'quantity':ep.cart_lines['new'] += 1
                elif mutation == 'need':ep.items[0].musts.append('Additional live need')
                elif mutation == 'availability':ep.stock_key = None
                else:ep.pending_notices.append('A changed requirement is pending.')
                return response(json.dumps(answer))
            with patch.object(agent, 'model_client', return_value=SimpleNamespace(post_verbose=changing_post)):
                self.assertFalse(ep.adjudicate_review(['new']), mutation)
            self.assertFalse(ep.verdicts['new#0'])
            self.assertEqual(list(ep.boundary_results.values())[0]['path'], 'revision')

    def test_cheapest_corrected_review_tracks_existing_and_new_relevant_alternatives(self):
        ep = episode();ep.brief['pick'] = 'cheapest'
        with patch.object(ep, 'owner', return_value=1):
            answer = self.next_answer(ep);answer['candidates'][1].update(kind='mismatch', needs=[])
            self.assertTrue(self.run_boundary(ep, [response(json.dumps(answer))])[0])
            context = ep.cart_review_context(['new']);self.assertEqual(ep.review_contexts['new#0'], context)
            ep.rows['old']['attributes'] = {'feature': 'wanted'}
            self.assertNotEqual(context, ep.cart_review_context(['new']))
            context = ep.cart_review_context(['new'])
            ep.rows['another'] = {'id': 'another', 'price': .5, 'available': True};ep.order.append('another')
            self.assertNotEqual(context, ep.cart_review_context(['new']))

    def test_any_corrected_review_ignores_unrelated_record_updates(self):
        ep = episode();answer = self.next_answer(ep)
        self.assertTrue(self.run_boundary(ep, [response(json.dumps(answer))])[0])
        context = ep.cart_review_context(['new']);ep.rows['old']['price'] = 20
        self.assertEqual(context, ep.cart_review_context(['new']))

    def test_reference_scope_keeps_the_existing_grounded_dispute_implementation(self):
        from test_chidori_v4 import dispute_answer
        ep = episode();ep.refs = [(1, 'Feature: wanted')]
        with patch.object(ep, 'json_ask', return_value=dispute_answer()) as ask:
            self.assertTrue(ep.adjudicate_review(['new']))
        self.assertEqual(ask.call_count, 1);self.assertFalse(ep.boundary_results)

    def test_unbound_financial_fields_keep_the_original_review_path(self):
        from test_chidori_v4 import dispute_answer
        for field in ('price_key', 'stock_key'):
            ep = episode();ep.brief['in_stock_only'] = True;setattr(ep, field, None)
            with patch.object(ep, 'json_ask', return_value=dispute_answer()) as ask:
                self.assertTrue(ep.adjudicate_review(['new']))
            self.assertEqual(ask.call_count, 1);self.assertFalse(ep.boundary_results)

    def test_new_correction_is_not_invoked_for_a_positive_review(self):
        ep = episode();ep.verdicts['new#0'] = True
        self.assertFalse(self.run_boundary(ep, [])[0]);self.assertFalse(ep.boundary_results)

    def test_oversized_packet_does_not_spend_or_create_a_product_verdict(self):
        ep = episode();ep.items[0].musts = ['A long active runtime requirement. '*900]
        original = dict(ep.review_messages)
        ok, posts = self.run_boundary(ep, [])
        self.assertFalse(ok);self.assertEqual(posts, []);self.assertFalse(ep.no_credit)
        self.assertEqual(ep.review_messages, original);self.assertEqual(ep.calls, 0)

    def test_full_request_narrative_is_not_retransmitted_by_the_corrective_review(self):
        ep = episode();snapshot = ep.boundary_snapshot(['new']);before = ep.boundary_prompt(snapshot)
        ep.request = 'A long runtime instruction. '*900
        snapshot['wire']['request'] = ep.request
        self.assertEqual(before, ep.boundary_prompt(snapshot))

    def test_format_only_retry_cannot_promote_unknown_mismatch_or_component_scope(self):
        for mutation in ('unknown', 'mismatch', 'component', 'kind', 'duplicate'):
            ep = episode();good = self.next_answer(ep);bad = json.loads(json.dumps(good))
            if mutation in ('unknown', 'mismatch'):bad['candidates'][0]['needs'][0]['state'] = mutation
            elif mutation == 'component':bad['candidates'][0]['needs'][0]['same_scope'] = False
            elif mutation == 'kind':bad['candidates'][0]['kind'] = 'unknown'
            else:bad['candidates'].append(bad['candidates'][0])
            bad['candidates'][0]['needs'][0]['evidence'] = 'malformed evidence'
            ok, posts = self.run_boundary(ep, [response(json.dumps(bad)), response(json.dumps(good))])
            self.assertFalse(ok, mutation);self.assertFalse(ep.verdicts['new#0'])
            self.assertEqual(len(posts), 2)

    def test_corrective_review_cannot_spend_the_procedure_reserve(self):
        ep = episode();ep.budget_mode = 'procedure';ep.procedures_valid = False
        ep.inference_budget.actual = ep.inference_budget.limit-ep.inference_budget.protected
        ok, posts = self.run_boundary(ep, [])
        self.assertFalse(ok);self.assertEqual(posts, []);self.assertEqual(ep.budget_mode, 'procedure')
        self.assertEqual(ep.inference_budget.actual+ep.inference_budget.uncertain, ep.inference_budget.limit-ep.inference_budget.protected)
        self.assertFalse(ep.no_credit)


class CodecShapeTests(OfflineCase):
    def test_all_three_captured_detail_view_failures_retain_the_observed_facts(self):
        root = ROOT/'logs/research/tosky-v2-local10-2026-11-13'
        decisions = json.loads((root/'recorded-decisions.json').read_text())
        checked = 0
        for trace in json.loads((root/'recorded-replay.json').read_text()):
            for rejection in trace['rejections']:
                info = rejection.get('assignment_checks', {})
                if info.get('opened') is not False:continue
                decision = next(d for d in decisions if d['session_id'] == trace['problem_id'] and d['role'] == 'cart_proof' and d['answer'] == rejection['answer'])
                ep, ident = recorded_episode(decision)
                ep.roles['offer'] = info['offer_role'];ep.views[ident] = set(info['views'])
                ep.schemas[info['offer_role']] = {'type': 'object', 'properties': {'id': {'type': 'string'}}}
                self.assertFalse(ep.page_read(ident))
                snapshot = ep.boundary_snapshot([ident]);self.assertTrue(snapshot['wire']['candidates'][0]['details_observed'])
                self.assertTrue(any(f['scope'] == 'attribute' for f in snapshot['wire']['facts'].values()))
                self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'APPROVED')
                checked += 1
        self.assertEqual(checked, 3)

    def test_all_twenty_actual_incomplete_source_audits_are_not_a_universal_veto(self):
        decisions = json.loads((ROOT/'logs/research/tosky-v2-local10-2026-11-13/recorded-decisions.json').read_text())
        audits = [d for d in decisions if d['role'] == 'source_audit'];self.assertEqual(len(audits), 20)
        for decision in audits:
            ep = episode();ep.coverage_done = False;before = (ep.live(), ep.msg_limit, ep.budget_steps, list(ep.duties))
            with patch.object(ep, 'json_ask', return_value=decision['answer']):
                self.assertEqual(ep.ensure_source_coverage(), '')
            self.assertEqual(before, (ep.live(), ep.msg_limit, ep.budget_steps, list(ep.duties)))
            self.assertFalse(ep.coverage_repair)

    def test_semantic_unknown_and_scope_uncertainty_remain_unknown(self):
        for kwargs in ({'state': 'unknown'}, {'same_scope': False}):
            ep = episode();snapshot = ep.boundary_snapshot(['new']);before = dict(ep.verdicts)
            self.assertEqual(ep.boundary_decode(typed_answer(snapshot, **kwargs), snapshot)['status'], 'MISSING_EVIDENCE')
            self.assertEqual(ep.verdicts, before)

    def test_grounded_mismatch_is_not_approved(self):
        ep = episode();snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot, state='mismatch'), snapshot)['status'], 'MISMATCH')

    def test_wrong_revision_unknown_and_repeated_requirements_are_format_faults(self):
        ep = episode();snapshot = ep.boundary_snapshot(['new'])
        for change in ('revision', 'unknown', 'duplicate', 'prose'):
            answer = typed_answer(snapshot)
            if change == 'revision':answer['revision'] += 1
            elif change == 'unknown':answer['candidates'][0]['needs'][0]['rid'] = 'missing'
            elif change == 'duplicate':answer['candidates'][0]['needs'] *= 2
            else:answer['candidates'][0]['needs'][0]['evidence'] = 'This looks correct.'
            result = ep.boundary_decode(answer, snapshot)
            self.assertEqual(result['status'], 'FORMAT_ERROR', change);self.assertTrue(result['path'])

    def test_evidence_from_a_different_candidate_is_rejected(self):
        ep = episode();ep.brief['pick'] = 'cheapest';ep.rows['old']['attributes'] = {'feature': 'different'}
        with patch.object(ep, 'owner', return_value=1):snapshot = ep.boundary_snapshot(['new'])
        answer = typed_answer(snapshot)
        other = next(eid for eid, f in snapshot['wire']['facts'].items() if f['cid'] == 'c2')
        answer['candidates'][0]['needs'][0]['evidence'] = [other]
        self.assertEqual(ep.boundary_decode(answer, snapshot)['status'], 'FORMAT_ERROR')

    def test_quantity_stock_and_budget_constraints_are_checked_in_code(self):
        for change in ('quantity', 'stock', 'budget', 'unknown_stock', 'price'):
            ep = episode();ep.brief['in_stock_only'] = True
            if change == 'quantity':ep.cart_lines['new'] = 2
            elif change == 'stock':ep.rows['new']['available'] = False
            elif change == 'budget':ep.budget = 1
            elif change == 'unknown_stock':ep.stock_key = None
            else:ep.price_key = None
            snapshot = ep.boundary_snapshot(['new'])
            self.assertNotEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'APPROVED', change)

    def test_equal_or_higher_priced_candidates_do_not_create_a_cheapest_obligation(self):
        ep = episode();ep.brief['pick'] = 'cheapest';ep.rows['old']['price'] = 20
        ep.check_better['new#0'] = ['old']
        with patch.object(ep, 'owner', return_value=1):snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(snapshot['ids'], ['new'])

    def test_cheaper_candidate_must_match_the_requested_type_and_requirements(self):
        ep = episode();ep.brief['pick'] = 'cheapest'
        with patch.object(ep, 'owner', return_value=1):snapshot = ep.boundary_snapshot(['new'])
        answer = typed_answer(snapshot)
        self.assertEqual(ep.boundary_decode(answer, snapshot)['status'], 'BETTER_OPTION')
        answer['candidates'][1].update(kind='mismatch', needs=[])
        self.assertEqual(ep.boundary_decode(answer, snapshot)['status'], 'APPROVED')

    def test_unknown_availability_cannot_establish_a_better_eligible_alternative(self):
        ep = episode();ep.brief.update(pick='cheapest', in_stock_only=True)
        with patch.object(ep, 'owner', return_value=1):snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'MISSING_EVIDENCE')
        ep.rows['old']['available'] = True
        with patch.object(ep, 'owner', return_value=1):snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'BETTER_OPTION')

    def test_null_empty_or_nonfinite_stock_is_unknown(self):
        for value in (None, '', [], {}, float('nan'), float('inf')):
            ep = episode();ep.brief['in_stock_only'] = True;ep.rows['new']['available'] = value
            snapshot = ep.boundary_snapshot(['new'])
            self.assertFalse(snapshot['wire']['candidates'][0]['availability_known'])
            self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'MISSING_EVIDENCE')
            packet = json.loads(ep.boundary_prompt(snapshot).split('SNAPSHOT:\n', 1)[1])
            json.dumps(packet, allow_nan=False)

    def test_indifferent_and_extra_requirement_ids_do_not_become_new_obligations(self):
        ep = episode();ep.items[0].free = ['Another property'];snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(len(snapshot['wire']['requirements']), 1)
        answer = typed_answer(snapshot)
        answer['candidates'][0]['needs'].append({'rid': 'indifferent_topic', 'state': 'unknown', 'same_scope': False, 'evidence': []})
        self.assertEqual(ep.boundary_decode(answer, snapshot)['status'], 'FORMAT_ERROR')

    def test_no_literal_spelling_comparator_overrides_a_semantic_match(self):
        ep = episode();ep.items[0].musts = ['surface: gray'];ep.full['new']['attributes'] = {'surface': 'grey'}
        snapshot = ep.boundary_snapshot(['new'])
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'APPROVED')

    def test_runtime_property_name_binding_does_not_need_a_fixed_synonym_table(self):
        ep = episode();ep.items[0].musts = ['material: cotton'];ep.full['new']['attributes'] = {'fabric': 'cotton'}
        snapshot = ep.boundary_snapshot(['new']);answer = typed_answer(snapshot)
        eid = next(eid for eid, f in snapshot['wire']['facts'].items() if f['scope'] == 'attribute' and f['key'] == 'fabric')
        answer['candidates'][0]['needs'][0]['evidence'] = [eid]
        self.assertEqual(ep.boundary_decode(answer, snapshot)['status'], 'APPROVED')

    def test_all_42_actual_old_responses_are_protocol_faults_not_new_product_verdicts(self):
        path = ROOT/'logs/research/tosky-v2-local10-2026-11-13/recorded-decisions.json'
        rows = [d for d in json.loads(path.read_text()) if d['role'] == 'cart_proof']
        self.assertEqual(len(rows), 42)
        prose, typed_value_faults = 0, 0
        for decision in rows:
            ep, ident = recorded_episode(decision);snapshot = ep.boundary_snapshot([ident]);before = dict(ep.verdicts)
            result = ep.boundary_decode(decision['answer'], snapshot)
            self.assertEqual(result['status'], 'FORMAT_ERROR');self.assertEqual(ep.verdicts, before)
            prose += result['path'].endswith('.evidence')
            typed_value_faults += result['path'].endswith('.value')
        self.assertGreaterEqual(prose, 6);self.assertGreaterEqual(typed_value_faults, 3)

    def test_actual_datasheet_evidence_survives_offer_role_discovery(self):
        path = ROOT/'logs/research/tosky-v2-local10-2026-11-13/recorded-decisions.json'
        decision = next(d for d in json.loads(path.read_text()) if d['role'] == 'cart_proof' and d['task_id'] == 't-f3ca299b9f')
        ep, ident = recorded_episode(decision)
        ep.roles['offer'] = 'view_listing';ep.schemas['view_listing'] = {'type': 'object', 'properties': {'id': {'type': 'string'}}}
        ep.views[ident] = {'datasheet_for_item'}
        self.assertFalse(ep.page_read(ident))
        snapshot = ep.boundary_snapshot([ident]);self.assertTrue(snapshot['wire']['candidates'][0]['details_observed'])
        self.assertEqual(ep.boundary_decode(typed_answer(snapshot), snapshot)['status'], 'APPROVED')
