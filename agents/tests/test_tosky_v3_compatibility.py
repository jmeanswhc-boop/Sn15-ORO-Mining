"""Existing finance, transport and procedure checks retargeted to the recovery candidate."""
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import patch

import test_chidori_procedures as procedures
import test_chidori_v4_regression as previous
from test_tosky_v3 import agent, episode, OfflineCase


class PreservedProcedureTests(procedures.ProcedureStateTests):
    def setUp(self):
        item = patch.object(procedures, 'agent', agent)
        item.start();self.addCleanup(item.stop)
        super().setUp()


class PreservedFinancialTests(previous.V4FinancialTests):
    candidate = agent


class PreservedTransportTests(previous.V4TransportTests):
    candidate = agent
    filename = 'Tosky_v3.py'


class PreservedProcedureWireTests(previous.V4FullProcedureWireTests):
    def setUp(self):
        item = patch.object(previous, 'FULL', agent)
        item.start();self.addCleanup(item.stop)
        super().setUp()


class PreservedNinetyTaskTests(previous.V4NinetyTaskTests):
    def setUp(self):
        item = patch.object(previous, 'AGENTS', {'Tosky_v3.py': agent})
        item.start();self.addCleanup(item.stop)
        super().setUp()


class NinetyTaskReviewBudgetTests(OfflineCase):
    def test_new_and_existing_roles_share_ninety_bounded_isolated_ledgers(self):
        local = threading.local()
        class MeteredClient:
            def __init__(self, number):self.number, self.count = number, 0
            def post_verbose(self, path, body):
                self.count += 1
                if self.number%5 == 0:return SimpleNamespace(data=None, error={'status': 500, 'body': 'temporary'})
                if self.number%5 == 1:return SimpleNamespace(data=None, error={'status': 402, 'body': 'credit stop'})
                cost = None if self.number%5 == 2 else .0001
                return SimpleNamespace(data={'usage': {'cost': cost}, 'choices': [
                    {'message': {'content': '{}'}, 'finish_reason': 'stop'}]}, error=None)
        def run(number):
            ep = episode();local.client = MeteredClient(number)
            for index in range(24):
                if ep.no_credit:break
                ep.with_inference_role(('review_boundary', 'repair', 'coverage', 'procedure')[index%4], ep.think,
                    [{'role': 'user', 'content': 'Controlled mixed-role accounting workload.'}], [agent.SPARE, agent.LIGHT], 768)
            budget = ep.inference_budget
            self.assertEqual(budget.attempts, local.client.count);self.assertEqual(budget.pending, 0)
            self.assertLessEqual(budget.actual+budget.uncertain, budget.limit)
            self.assertLessEqual(budget.attempts, agent.CALL_CAP)
            if number%5 == 1:self.assertEqual(local.client.count, 1)
            return budget.actual+budget.uncertain
        with patch.object(agent, 'model_client', side_effect=lambda _: local.client), patch.object(agent.time, 'sleep'):
            with ThreadPoolExecutor(max_workers=6) as pool:total = sum(pool.map(run, range(90)))
        self.assertLessEqual(total, 3150000000)


class PreservedProcedureBoundaryTests(previous.V4IsolatedProcedureBoundaryTests):
    def setUp(self):
        item = patch.object(previous, 'FULL', agent)
        item.start();self.addCleanup(item.stop)
        super().setUp()
