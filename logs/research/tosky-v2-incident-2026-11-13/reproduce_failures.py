"""Characterize completion-blocking defects offline; leave submitted code intact."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT/'agents/tests')]
from test_tosky_v2 import agent, proof_episode, cart_proof


def main():
    results = []
    with patch('requests.sessions.Session.request', side_effect=AssertionError('No network during diagnostic probes')) as network:
        # A source audit may add a genuine global availability requirement to
        # the per-property ledger, but the proof checker refuses a boolean
        # listing field as evidence for it.
        ep = proof_episode()
        ep.request += ' Only items currently in stock.'
        audit = {'complete': True, 'procedures_complete': True, 'constraints': [
                    {'item': 1, 'previous': 'surface: satin', 'need': 'surface: satin',
                     'source': 'with a satin surface', 'relation': 'literal'}],
                 'add': [{'item': 1, 'previous': None, 'need': 'availability: in stock',
                          'source': 'Only items currently in stock.', 'relation': 'literal'}], 'process': []}
        accepted = ep.apply_source_audit(audit)
        assert accepted
        ep.rows['one']['in_stock'] = True
        proof = cart_proof()
        proof['requirements'].append({'item': 1, 'need': 'availability: in stock', 'status': 'met',
            'evidence': [{'id': 'one', 'scope': 'listing', 'key': 'in_stock', 'value': True}]})
        assert not ep.validate_cart_proof(proof, ['one'])
        results.append({'case': 'availability_audit_and_proof_contract_disagree',
                        'audit_accepted_real_requirement': accepted,
                        'boolean_evidence_rejected': True,
                        'source_level_defect': 'The ledger can demand a true stock flag that the listing-evidence checker cannot represent.'})

        # The supplied runtime fields can use synonyms rather than token subsets.
        # No static synonym registry is needed to demonstrate the mismatch.
        ep = proof_episode();ep.items[0].musts = ['material: cotton']
        ep.rows['one']['attributes'] = {'fabric': 'cotton'}
        proof = cart_proof();proof['requirements'] = [{'item': 1, 'need': 'material: cotton', 'status': 'met',
            'binding': {'property': 'material', 'key': 'fabric', 'same_scope': True, 'reason': 'The observed fabric field supplies the requested material.'},
            'evidence': [{'id': 'one', 'scope': 'attribute', 'key': 'fabric', 'value': 'cotton'}]}]
        assert not ep.validate_cart_proof(proof, ['one'])
        results.append({'case': 'correct_runtime_semantic_binding_rejected',
                        'observed_value_matches': True, 'explicit_same_property_binding': True,
                        'accepted': False, 'source_level_defect': 'The key checker requires exact or nested word sets; semantic bindings cannot bridge synonyms.'})

        # A current proof can legitimately cite a value contained in a longer
        # attribute string. Conversely, this literal test rejects ordinary
        # synonymous values even if the model gives a correct positive judgment.
        ep = proof_episode();ep.items[0].musts = ['surface: gray']
        ep.rows['one']['attributes'] = {'surface': 'grey'}
        proof = cart_proof();proof['requirements'][0].update(need='surface: gray')
        proof['requirements'][0]['evidence'][0]['value'] = 'grey'
        assert not ep.validate_cart_proof(proof, ['one'])
        results.append({'case': 'ordinary_value_spelling_rejected', 'wanted': 'gray', 'observed': 'grey',
                        'accepted': False, 'source_level_defect': 'Substring matching turns a legitimate semantic equivalence into a definitive mismatch.'})

        # An unknown inference result and a definitive mismatch share the same
        # cached False state. A normal retry does not ask a now-available reviewer.
        ep = proof_episode()
        with patch.object(ep, 'evidence_decision', side_effect=[None, None, cart_proof()]) as model:
            first = ep.cart_check();second = ep.cart_check();third = ep.cart_check()
        assert first and second and third and model.call_count == 2
        assert ep.verdicts['one#0'] is False
        results.append({'case': 'unknown_review_cached_as_negative', 'ordinary_retries': 3,
                        'reviewer_calls': model.call_count, 'negative_verdict': True,
                        'source_level_defect': 'Unavailable and rejected are conflated; only a separate bounded adjudication path can reopen the result.'})

        # The reviewer sees clipped text, while the validator demands the entire
        # field verbatim rather than allowing a grounded fragment.
        ep = proof_episode();ep.rows['one']['description'] = 'Satin surface. '+('General product information. '*40)
        supplied = agent.clip(ep.rows['one'], 600)['description']
        assert supplied != ep.rows['one']['description']
        quoted = {'id': 'one', 'scope': 'listing', 'key': 'description', 'value': supplied}
        assert not ep.citations_valid([quoted], {'one'})
        results.append({'case': 'clipped_packet_cannot_supply_required_full_field_citation',
                        'full_characters': len(ep.rows['one']['description']), 'shown_characters': len(supplied),
                        'visible_field_rejected': True,
                        'source_level_defect': 'The decision packet and evidence validator disagree on which exact text was supplied.'})

        # Same category plus the current property list does not establish the
        # requested product subtype. The independent cheaper-item guard omits
        # the type_ok assessment required of the purchased candidate.
        ep = proof_episode();ep.brief['pick'] = 'cheapest';ep.items[0].name = 'walking footwear'
        ep.rows['one']['name'] = 'Walking footwear'
        ep.rows['lower'] = {'id': 'lower', 'category': 'equipment', 'name': 'A different product type',
                            'price': 8, 'attributes': {'surface': 'satin'}}
        ep.full['lower'] = ep.rows['lower'];ep.opened.add('lower');ep.order.append('lower')
        proof = cart_proof();proof['objective'].update(considered=['lower'], reason='The lower-priced record has the wrong requested product type.')
        assert not ep.validate_cart_proof(proof, ['one'])
        results.append({'case': 'cheaper_guard_omits_requested_product_type',
                        'model_checked_alternative': True, 'same_category': True, 'accepted': False,
                        'source_level_defect': 'The independent guard can veto a valid purchase using an ineligible cheaper subtype.'})

        # A fully enumerated, empty affordable candidate set is a different
        # situation from a failed keyword query. The new decline path has no
        # representation for it: no candidate means it refuses without review.
        ep = proof_episode();ep.cart_lines = {};ep.budget = 10
        ep.search_evidence = [{'step': 1, 'tool': 'search', 'args': {'query': ''}}]
        with patch.object(ep, 'evidence_decision') as reviewer:
            result = ep.decline_check()
        assert result and not reviewer.called and ep.decline_candidates() == []
        results.append({'case': 'no_affordable_candidate_has_no_decline_route',
                        'known_prices': [12], 'budget': 10, 'decline_held': True, 'reviewer_called': False,
                        'source_level_defect': 'The no-candidate branch cannot distinguish proven absence from insufficient search evidence.'})

        network.assert_not_called()
    (ROOT/'offline-counterexamples.json').write_text(json.dumps({'paid_calls': 0, 'candidate_modified': False,
        'scope': 'Deterministic contract counterexamples; these are not recovered scores or a claim that each triggered a particular network episode.',
        'cases': results}, indent=2)+'\n')
    print(json.dumps({'counterexamples_reproduced': len(results), 'network_calls': 0, 'candidate_modified': False}, indent=2))


if __name__ == '__main__':
    main()
