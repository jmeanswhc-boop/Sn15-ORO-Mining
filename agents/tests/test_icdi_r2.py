"""Offline regression tests; recorded responses are loaded, never sent to an API."""
import copy
import json
import os
from pathlib import Path
import unittest

from agents.tests.test_icdi_r1 import ROOT, load, scripted_json

SOURCE = Path(os.environ.get('ICDI_R2_TEST_SOURCE', ROOT / 'agents/ICDI_r2.py'))
RUN = ROOT / 'LV_logs/local-8751a6f6bc7948028534ac90e8c7d730'


def episode():
    ns = load(SOURCE)
    e = ns['Kyo']({'environment': {'binding': {'session_id': 'offline'},
         'policy_view': {'query': 'Choose a matte object.', 'max_steps': 40, 'tools': []}}})
    e.items = [ns['Item']('object', 'Objects', 1, ['finish: matte'])]
    e.brief = {'pick': 'any', 'buy_args': {}, 'refuse_args': {}}
    return ns, e


class ReplyContracts(unittest.TestCase):
    def test_resolution_can_refer_to_the_question_just_asked(self):
        _, e = episode()
        q, a = 'Which size do you prefer?', 'Any size works.'
        scripted_json(e, [{'kind': 'other', 'resolved': [q],
                          'answers': [{'question': 1, 'state': 'indifferent', 'quote': a}]}])
        got = e.hear([(1, q, a)])
        self.assertTrue(got)
        self.assertFalse(e.pending_answers)

    def test_unknown_resolution_is_still_rejected(self):
        _, e = episode()
        e.ask_json = lambda *a: {'kind': 'other', 'resolved': ['Never asked this'],
                               'answers': [{'question': 1, 'state': 'answered', 'quote': 'Small.'}]}
        self.assertFalse(e.hear([(1, 'Which size?', 'Small.')]))
        self.assertTrue(e.pending_answers)

    def test_answer_cannot_target_a_nonexistent_question(self):
        _, e = episode()
        e.ask_json = lambda *a: {'kind': 'other',
                               'answers': [{'question': 2, 'state': 'answered', 'quote': 'Small.'}]}
        self.assertFalse(e.hear([(1, 'Which size?', 'Small.')]))
        self.assertTrue(e.pending_answers)

    def test_unscoped_indifference_does_not_erase_a_firm_constraint(self):
        _, e = episode()
        a = 'Anything is fine with me.'
        scripted_json(e, [{'kind': 'change',
                          'drop': [{'item': 1, 'need': 'finish: matte', 'quote': a}],
                          'free': [{'item': 1, 'property': 'finish', 'quote': a}],
                          'answers': [{'question': 1, 'state': 'indifferent', 'quote': a}]}])
        e.take_heard(e.hear([(1, 'What surface texture do you want?', a)]), [])
        self.assertEqual(e.needs_now(), [(1, 'finish: matte')])
        self.assertNotIn('finish', e.items[0].free)
        self.assertFalse(e.pending_answers)

    def test_explicit_relaxation_remains_allowed(self):
        _, e = episode()
        a = 'Any finish is fine now.'
        scripted_json(e, [{'kind': 'change',
                          'drop': [{'item': 1, 'need': 'finish: matte', 'quote': a}],
                          'answers': [{'question': 1, 'state': 'indifferent', 'quote': a}]}])
        e.take_heard(e.hear([(1, 'What finish?', a)]), [])
        self.assertEqual(e.needs_now(), [])

    def test_indifference_to_the_exact_property_can_relax_it(self):
        _, e = episode()
        a = 'Anything is fine with me.'
        scripted_json(e, [{'kind': 'change',
                          'drop': [{'item': 1, 'need': 'finish: matte', 'quote': a}],
                          'answers': [{'question': 1, 'state': 'indifferent', 'quote': a}]}])
        e.take_heard(e.hear([(1, e.question(1, 'finish'), a)]), [])
        self.assertEqual(e.needs_now(), [])

    def test_scoped_clarification_can_relax_a_firm_requirement(self):
        _, e = episode()
        q, a = 'May the surface finish be anything, relaxing the earlier limit?', 'I leave that to you.'
        e.unclear = [{'item': 1, 'property': 'finish', 'question': q}]
        scripted_json(e, [{'kind': 'change', 'resolved': [q],
                          'drop': [{'item': 1, 'need': 'finish: matte', 'quote': a}],
                          'answers': [{'question': 1, 'state': 'indifferent', 'quote': a}]}])
        e.take_heard(e.hear([(1, q, a)]), [])
        self.assertEqual(e.needs_now(), [])
        self.assertFalse(e.unclear)

    @unittest.skipUnless(RUN.exists(), 'Recorded run is not present')
    def test_real_unfinished_runs_accept_their_recorded_replies(self):
        episodes = [json.loads(l)['episode'] for l in (RUN / 'episode_results.jsonl').read_text().splitlines()]
        failed = {e['session_id'] for e in episodes if e['verdict']['category'] == 'did not finish'}
        checked = 0
        for line in (RUN / 'sandbox/sandbox_output.jsonl').read_text().splitlines():
            row = json.loads(line)
            if row['problem_id'] not in failed:
                continue
            ns, e = episode()
            brief = next(t['brief'] for t in row['dialogue'] if t.get('kind') == 'brief')
            e.brief = brief
            e.items = [ns['Item'](i['name'], i['shelf'], i['quantity'], i['musts']) for i in brief['items']]
            e.unclear = copy.deepcopy(brief.get('unclear', []))
            questions = iter(t for t in row['dialogue'] if t.get('kind') == 'ask')
            calls = [json.loads(l) for l in (RUN / 'sandbox' / ('request_log_' + row['problem_id'] + '.jsonl')).read_text().splitlines()]
            replies = [c for c in calls if c.get('kind') == 'summary' and 'inference' in c['path']
                       and 'NEW SOURCE:' in str(c['json_data']['messages'][0].get('content', ''))]
            for q, c in zip(questions, replies):
                got = ns['_json_obj'](c['response']['choices'][0]['message']['content'])
                scripted_json(e, [got])
                result = e.hear([(1, q['q'], q['a'])])
                self.assertTrue(result, row['problem_id'])
                self.assertFalse(e.pending_answers)
                e.take_heard(result, [])
            checked += 1
        self.assertEqual(checked, 2)


class CurrentStateContracts(unittest.TestCase):
    def test_stock_only_notice_invalidates_decisions_and_read_cache_version(self):
        _, e = episode()
        e.verdicts['old-cart'] = True
        old_version = e.version
        e.take_heard({'sold_out': True}, [])
        self.assertGreater(e.version, old_version)
        self.assertFalse(e.verdicts)

    def test_retired_requirements_are_explicit_for_reviewers(self):
        _, e = episode()
        e.items[0].musts.append('surface texture: smooth')
        e.take_heard({'drop': [{'item': 1, 'need': 'finish: matte'}]}, [])
        self.assertIn('retired requirements (do not reapply): finish: matte', e.ledger())
        self.assertIn('requirements: surface texture: smooth', e.ledger())

    def test_reinstated_requirement_is_not_labelled_retired(self):
        _, e = episode()
        e.take_heard({'drop': [{'item': 1, 'need': 'finish: matte'}]}, [])
        e.take_heard({'needs': [{'item': 1, 'need': 'finish: matte'}]}, [])
        self.assertNotIn('retired requirements', e.ledger())

    def test_pending_reply_retry_gets_protocol_feedback(self):
        _, e = episode()
        batch = [(1, 'Which size?', 'Small.')]
        prompts = scripted_json(e, [{'kind': 'invalid'}, {'kind': 'other'}])
        self.assertFalse(e.hear(batch))
        e.hear(batch)
        self.assertIn('Previous interpretation failed validation', prompts[1])
        self.assertFalse(e.pending_answers)

    def test_blocked_empty_cart_finish_returns_a_recovery_reason(self):
        _, e = episode()
        e.roles = {'refuse': 'leave'}
        e.schemas = {'leave': {'type': 'object', 'properties': {}}}
        e.pending_answers = [(1, 'Which size?', 'Small.')]
        self.assertTrue(e.finish(again=True))
        self.assertFalse(e.finished)


class BoundedReviewContracts(unittest.TestCase):
    def test_identical_failed_procedure_does_not_repeat_inference(self):
        _, e = episode()
        e.process = [{'step': 'Get permission', 'when': 'order', 'messages': 1}]
        prompts = scripted_json(e, [{'steps': [{'step': 'Get permission', 'done': False, 'lines': []}]}])
        self.assertFalse(e.gate('order')[0])
        for _ in range(6):
            self.assertFalse(e.gate('order')[0])
        self.assertEqual(len(prompts), 1)

    def test_new_procedure_evidence_reopens_a_failed_gate(self):
        _, e = episode()
        e.process = [{'step': 'Get permission', 'when': 'order', 'messages': 1}]
        prompts = scripted_json(e, [{'steps': []},
            {'steps': [{'step': 'Get permission', 'done': True, 'lines': [1]}]}])
        self.assertFalse(e.gate('order')[0])
        e.done_log.append('Permission granted'); e.done_steps.append(2); e.done_versions.append(0)
        self.assertTrue(e.gate('order')[0])
        self.assertEqual(len(prompts), 2)

    def test_inconclusive_gate_can_retry_without_new_evidence(self):
        _, e = episode()
        e.process = [{'step': 'Get permission', 'when': 'order', 'messages': 1}]
        e.done_log = ['Permission granted']; e.done_steps = [2]; e.done_versions = [0]
        prompts = scripted_json(e, [None,
            {'steps': [{'step': 'Get permission', 'done': True, 'lines': [1]}]}])
        self.assertFalse(e.gate('order')[0])
        self.assertTrue(e.gate('order')[0])
        self.assertEqual(len(prompts), 2)

    def test_last_gate_action_can_receive_a_fresh_proof_review(self):
        _, e = episode()
        e.process = [{'step': 'Read the required records', 'when': 'choose', 'messages': 0}]
        e.schemas = {'read': {'type': 'object', 'properties': {}}}
        def act(name, args):
            e.step += 1
            return {'observation': {'ok': True}}
        e.act = act
        e.observe = lambda *a: ''
        pending = {'steps': [], 'call': {'tool': 'read', 'args': {}}}
        prompts = scripted_json(e, [pending, pending, pending,
            {'steps': [{'step': 'Read the required records', 'done': True, 'lines': [3]}]}])
        self.assertFalse(e.gate('choose')[0])
        self.assertTrue(e.gate('choose')[0])
        self.assertEqual(len(prompts), 4)

    def test_cart_dispute_has_one_recheck_and_never_defaults_to_approval(self):
        _, e = episode()
        e.cart_lines = {'sample': 1}
        prompts = scripted_json(e, [{'ok': False, 'problems': ['conflicting facts']}, None])
        self.assertTrue(e.check_cart())
        self.assertTrue(e.check_cart(reconsider=True))
        self.assertTrue(e.check_cart(reconsider=True))
        self.assertEqual(len(prompts), 2)
        self.assertFalse(e.verdicts[e.check_key(['sample'])])

    def test_review_can_correct_a_previous_rejection_with_evidence(self):
        _, e = episode()
        e.cart_lines = {'sample': 1}
        prompts = scripted_json(e, [{'ok': False, 'problems': ['outdated requirement']}, {'ok': True}])
        self.assertTrue(e.check_cart())
        self.assertEqual(e.check_cart(reconsider=True), '')
        self.assertIn('Previous internal review', prompts[1])
        self.assertEqual(e.check_cart(reconsider=True), '')
        self.assertEqual(len(prompts), 2)

    def test_unexecuted_planner_loop_is_bounded(self):
        _, e = episode()
        e.asked = True
        e.think = lambda *a: {'tool_calls': [{'function': {'name': 'unknown', 'arguments': '{}'}}]}
        e.work(42, 0.044)
        self.assertEqual(e.loop_calls, 3)
        self.assertEqual(e.step, 0)
        self.assertFalse(e.finished)

    def test_new_review_candidates_count_as_progress_without_a_store_call(self):
        _, e = episode()
        e.asked = True
        e.think = lambda *a: {'tool_calls': [{'function': {'name': 'unknown', 'arguments': '{}'}}]}
        def discover(calls):
            e.leads.append('candidate-' + str(e.loop_calls))
            return [('call', 'New candidate found in the records already read.')]
        e.run_calls = discover
        e.work(6, 0.044)
        self.assertEqual(e.loop_calls, 6)


if __name__ == '__main__':
    unittest.main()
