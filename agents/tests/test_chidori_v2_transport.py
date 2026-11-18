"""Offline regressions through the real ProxyClient and a fake HTTP boundary."""
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_chidori_budget as budget_tests
import test_chidori_procedures as procedure_tests
import test_chidori_integration as integration_tests
from src.agent.proxy_client import ProxyClient


ROOT = Path(__file__).resolve().parents[2]
V1_PATH = ROOT / "agents/ICDI_chidori_v1.py"
V2_PATH = ROOT / "agents/ICDI_chidori_v2.py"
v2 = budget_tests.load(V2_PATH, "chidori_v2_transport_candidate")


class V2BudgetTests(budget_tests.BudgetTests):
    def setUp(self):
        item = patch.object(budget_tests, "agent", v2)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()

    def run_calls(self, responses, cap=64, prices=None, models=None, tools=None):
        with patch.object(v2, "BUDGET_PRICE_CONFIG", budget_tests.catalog() if prices is None else prices):
            return super().run_calls(responses, cap, prices, models, tools)

    def test_success_is_charged_once_and_catalog_is_cached(self):
        client = budget_tests.FakeClient([budget_tests.reply(), budget_tests.reply()])
        with patch.object(v2, "model_client", return_value=client):
            for _ in range(2):
                self.assertIsNotNone(self.ep.think([{"role": "user", "content": "x"}], [v2.LIGHT], 64))
        self.assertEqual(self.ep.inference_budget.actual, 200000)
        self.assertEqual(self.ep.calls, 2)
        self.assertEqual(client.gets, [])

    def test_original_methods_unchanged_outside_central_inference_hook(self):
        before = ast.parse(V1_PATH.read_text())
        after = ast.parse(V2_PATH.read_text())
        old_classes = {n.name: n for n in before.body if isinstance(n, ast.ClassDef)}
        new_classes = {n.name: n for n in after.body if isinstance(n, ast.ClassDef)}
        for name, old in old_classes.items():
            for method in [n for n in old.body if isinstance(n, ast.FunctionDef)]:
                if name == "BudgetControls" and method.name == "budget_post":
                    continue
                new = next(n for n in new_classes[name].body if isinstance(n, ast.FunctionDef) and n.name == method.name)
                self.assertEqual(ast.dump(method), ast.dump(new), (name, method.name))


class V2ProcedureTests(procedure_tests.ProcedureStateTests):
    def setUp(self):
        item = patch.object(procedure_tests, "agent", v2)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()


class V2BoundaryTests(integration_tests.BoundaryTests):
    def setUp(self):
        for module in (integration_tests, procedure_tests):
            item = patch.object(module, "agent", v2)
            item.start()
            self.addCleanup(item.stop)
        super().setUp()


class RealClientTransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        block = patch("requests.sessions.Session.request", side_effect=AssertionError("Real network forbidden"))
        blocked = block.start()
        self.addCleanup(block.stop)
        self.addCleanup(blocked.assert_not_called)
        self.client = ProxyClient(api_key="offline-test-token", max_retries=1, timeout=1,
                                  inference_stats_file=str(Path(self.directory.name) / "usage.jsonl"),
                                  inference_stats_problem_id="synthetic-transport")
        self.gets = []
        self.posts = []
        self.post_status = 200
        self.content = "{}"
        for name, handler in (("get", self.http_get), ("post", self.http_post)):
            item = patch("src.agent.proxy_client.requests." + name, side_effect=handler)
            item.start()
            self.addCleanup(item.stop)

    def response(self, status, data):
        return SimpleNamespace(status_code=status, text=json.dumps(data), json=lambda: data)

    def http_get(self, url, **kwargs):
        self.gets.append({"authorized": kwargs.get("headers", {}).get("Authorization") == "Bearer offline-test-token"})
        return self.response(401, {"error": "Inference key does not match active run"})

    def http_post(self, url, **kwargs):
        authorized = kwargs.get("headers", {}).get("Authorization") == "Bearer offline-test-token"
        self.posts.append({"authorized": authorized, "body": kwargs["json"]})
        if not authorized:
            return self.response(401, {"error": "Inference key does not match active run"})
        if self.post_status != 200:
            return self.response(self.post_status, {"error": {"code": self.post_status, "message": "synthetic failure"}})
        return self.response(200, {"usage": {"cost": 0.000001}, "choices": [{"message": {"content": self.content}, "finish_reason": "stop"}]})

    def test_v1_failure_is_reproduced_through_real_client(self):
        v1 = budget_tests.load(V1_PATH, "chidori_v1_failure_reproduction")
        episode = v1.Episode(budget_tests.problem())
        with patch.object(v1, "model_client", return_value=self.client):
            result = episode.think([{"role": "user", "content": "Synthetic request"}], [v1.LIGHT], 64)
        self.assertIsNone(result)
        self.assertEqual(self.gets, [{"authorized": False}])
        self.assertEqual(self.posts, [])
        self.assertTrue(episode.no_credit)

    def test_v2_uses_real_authenticated_post_without_catalog_get(self):
        episode = v2.Episode(budget_tests.problem())
        with patch.object(v2, "model_client", return_value=self.client):
            result = episode.think([{"role": "user", "content": "Synthetic request"}], [v2.LIGHT], 64)
        self.assertIsNotNone(result)
        self.assertEqual(self.gets, [])
        self.assertEqual(len(self.posts), 1)
        self.assertTrue(self.posts[0]["authorized"])
        self.assertEqual(episode.inference_budget.actual, 1000)

    def test_all_thirty_recorded_inputs_can_start_the_exact_briefing(self):
        path = ROOT / "LV_logs/local-d357453886534e12a9b57e574b2f8a73/problems.jsonl"
        if not path.exists():
            self.skipTest("Recorded verification inputs are unavailable")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 30)
        class CapturedBrief(Exception):
            pass
        for index, problem in enumerate(rows):
            with self.subTest(index=index):
                episode = v2.Episode(problem)
                def first_call(prompt, models, cap=1800):
                    result = episode.think([{"role": "user", "content": prompt}], models, cap)
                    self.assertIsNotNone(result, list(episode.trace))
                    raise CapturedBrief()
                with patch.object(v2, "model_client", return_value=self.client), patch.object(episode, "json_ask", side_effect=first_call):
                    with self.assertRaises(CapturedBrief):
                        episode.read_brief()
        self.assertEqual(self.gets, [])
        self.assertEqual(len(self.posts), 30)
        self.assertTrue(all(p["authorized"] for p in self.posts))
        self.assertTrue(all(512 <= p["body"]["max_tokens"] <= 4000 for p in self.posts))

    def test_real_client_402_stops_after_one_attempt(self):
        self.post_status = 402
        episode = v2.Episode(budget_tests.problem())
        with patch.object(v2, "model_client", return_value=self.client):
            result = episode.think([{"role": "user", "content": "x"}], [v2.LIGHT, v2.HEAVY], 64)
        self.assertIsNone(result)
        self.assertEqual(len(self.posts), 1)
        self.assertTrue(episode.no_credit)

    def test_invalid_financial_configuration_still_blocks_inference(self):
        episode = v2.Episode(budget_tests.problem())
        with patch.object(v2, "BUDGET_PRICE_CONFIG", {"data": []}), patch.object(v2, "model_client", return_value=self.client):
            self.assertIsNone(episode.think([{"role": "user", "content": "x"}], [v2.LIGHT], 64))
        self.assertEqual(self.gets, [])
        self.assertEqual(self.posts, [])

    def test_v1_source_is_preserved(self):
        self.assertEqual(hashlib.sha256(V1_PATH.read_bytes()).hexdigest(), "295f290c70a2fc71035052f23f78e40bd8c6e23782c6d457829454e7473baaf3")

    def test_maximum_published_context_tiers_are_budgeted(self):
        budget = v2.InferenceBudget()
        budget.load_prices(v2.BUDGET_PRICE_CONFIG)
        self.assertEqual(budget.prices[v2.LIGHT]["input"], 625)
        self.assertEqual(budget.prices[v2.LIGHT]["output"], 2500)
        self.assertEqual(budget.prices[v2.HEAVY]["input"], 2500)
        self.assertEqual(budget.prices[v2.SPARE]["output"], 750)

    def test_actual_catalog_shape_exposes_v1_parser_gap_and_v2_covers_tiers(self):
        snapshot = json.loads((ROOT / "logs/research/chidori-v2-failure-d3574538/pricing-snapshot.json").read_text())
        v1 = budget_tests.load(V1_PATH, "chidori_v1_catalog_reproduction")
        old_budget = v1.InferenceBudget()
        old_budget.load_prices(snapshot)
        self.assertEqual(old_budget.prices, {})
        normalized = {row["id"]: row for row in v2.BUDGET_PRICE_CONFIG["data"]}
        from decimal import Decimal
        for row in snapshot["data"]:
            rates = [row["pricing"]] + row["pricing"].get("overrides", [])
            for key in ("prompt", "completion", "input_cache_read", "input_cache_write"):
                expected = max(Decimal(rate.get(key, "0")) for rate in rates)
                self.assertEqual(Decimal(normalized[row["id"]]["pricing"][key]), expected)

    def test_v2_has_no_catalog_call_or_new_direct_network_import(self):
        tree = ast.parse(V2_PATH.read_text())
        imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertFalse(imports & {"requests", "http", "urllib", "socket"})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
                self.assertNotEqual(node.args[0].value, "/inference/models")


if __name__ == "__main__":
    unittest.main()
