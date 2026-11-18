"""Offline billing and retry checks. Every HTTP request is forbidden or mocked."""
import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "logs/research/champion-lineage-2026-11-12/source-06af2157-d650-7d40-8000-519949f07ca0.py"
CANDIDATE = ROOT / "agents/ICDI_chidori_v1_budget.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


agent = load(CANDIDATE, "chidori_budget_candidate")


def catalog():
    return {"data": [{"id": model, "context_length": 200000,
                      "top_provider": {"max_completion_tokens": 10000},
                      "pricing": {"prompt": "0.000001", "completion": "0.000005",
                                  "request": "0", "input_cache_read": "0.0000001",
                                  "input_cache_write": "0.00000125", "image": "0.01"}}
                     for model in (agent.LIGHT, agent.HEAVY, agent.SPARE)]}


def problem():
    return {"environment": {"binding": {"session_id": "synthetic-session"},
                            "policy_view": {"query": "Find one suitable item.", "tools": [],
                                            "max_steps": 20, "finished": False}}}


def reply(cost=0.0001, content="{}", choices=True):
    data = {"usage": {"cost": cost}}
    if choices:
        data["choices"] = [{"message": {"content": content}, "finish_reason": "stop"}]
    return SimpleNamespace(data=data, error=None)


class FakeClient:
    def __init__(self, replies, prices=None):
        self.replies = list(replies)
        self.prices = catalog() if prices is None else prices
        self.posts = []
        self.gets = []
        self.on_post = None

    def get(self, path):
        self.gets.append(path)
        return self.prices

    def post_verbose(self, path, body):
        self.posts.append((path, body))
        if self.on_post:
            self.on_post()
        if not self.replies:
            raise AssertionError("Unexpected inference attempt")
        result = self.replies.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.network = patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden"))
        network_mock = self.network.start()
        self.addCleanup(self.network.stop)
        self.addCleanup(network_mock.assert_not_called)
        self.ep = agent.Episode(problem())
        self.ep.room = lambda: 100
        self.ep.clock = lambda: 0
        sleep_patch = patch.object(agent.time, "sleep")
        self.sleeps = sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    def run_calls(self, responses, cap=64, prices=None, models=None, tools=None):
        client = FakeClient(responses, prices)
        with patch.object(agent, "model_client", return_value=client):
            value = self.ep.think([{"role": "user", "content": "Read this request."}],
                                  models or [agent.LIGHT], cap, tools)
        return value, client

    def test_original_methods_unchanged_outside_central_inference_hook(self):
        original = ast.parse(BASE.read_text())
        candidate = ast.parse(CANDIDATE.read_text())
        classes = {n.name: n for n in candidate.body if isinstance(n, ast.ClassDef)}
        for cls in original.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            updated = {n.name: n for n in classes[cls.name].body if isinstance(n, ast.FunctionDef)}
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and (cls.name, method.name) != ("E1", "one_call"):
                    self.assertEqual(ast.dump(method), ast.dump(updated[method.name]), (cls.name, method.name))

    def test_success_is_charged_once_and_catalog_is_cached(self):
        client = FakeClient([reply(), reply()])
        with patch.object(agent, "model_client", return_value=client):
            for _ in range(2):
                self.assertIsNotNone(self.ep.think([{"role": "user", "content": "x"}], [agent.LIGHT], 64))
        self.assertEqual(self.ep.inference_budget.actual, 200000)
        self.assertEqual(self.ep.calls, 2)
        self.assertEqual(client.gets, ["/inference/models"])
        self.assertEqual(self.ep.inference_budget.pending, 0)

    def test_reservation_exists_before_http(self):
        client = FakeClient([reply()])
        seen = []
        client.on_post = lambda: seen.append(self.ep.inference_budget.pending)
        with patch.object(agent, "model_client", return_value=client):
            self.ep.think([{"role": "user", "content": "x"}], [agent.LIGHT], 64)
        self.assertGreater(seen[0], 0)
        self.assertEqual(self.ep.inference_budget.pending, 0)

    def test_charged_response_without_choices_is_still_accounted(self):
        _, client = self.run_calls([reply(cost=0.0004, choices=False)])
        self.assertEqual(len(client.posts), 1)
        self.assertEqual(self.ep.inference_budget.actual, 400000)
        self.assertEqual(self.ep.spend, 0.0004)

    def test_missing_invalid_and_negative_costs_retain_reservation(self):
        for cost in (None, True, -1, float("nan"), float("inf"), "0.001"):
            with self.subTest(cost=cost):
                self.ep = agent.Episode(problem())
                self.ep.room = lambda: 100
                _, client = self.run_calls([reply(cost=cost)])
                self.assertEqual(len(client.posts), 1)
                self.assertGreater(self.ep.inference_budget.uncertain, 0)
                self.assertEqual(self.ep.inference_budget.actual, 0)

    def test_402_stops_before_sleep_retry_or_fallback(self):
        response = SimpleNamespace(data=None, error={"status": 402, "body": "cap reached"})
        value, client = self.run_calls([response], models=[agent.LIGHT, agent.HEAVY])
        self.assertIsNone(value)
        self.assertEqual(len(client.posts), 1)
        self.assertTrue(self.ep.no_credit)
        self.sleeps.assert_not_called()

    def test_timeout_is_reserved_and_each_retry_is_counted(self):
        _, client = self.run_calls([TimeoutError(), reply()])
        self.assertEqual(len(client.posts), 2)
        self.assertEqual(self.ep.calls, 2)
        self.assertGreater(self.ep.inference_budget.uncertain, 0)

    def test_rate_limit_counts_physical_attempts(self):
        error = SimpleNamespace(data=None, error={"status": 429, "body": "rate limited"})
        _, client = self.run_calls([error] * 10)
        self.assertEqual(len(client.posts), 4)
        self.assertEqual(self.ep.calls, len(client.posts))
        self.assertGreater(self.ep.inference_budget.uncertain, 0)

    def test_compatibility_retry_is_admitted_and_charged_separately(self):
        self.ep.clock = lambda: 100
        error = SimpleNamespace(data=None, error={"status": 422, "body": "reasoning unsupported"})
        _, client = self.run_calls([error, reply()], tools=[{"type": "function", "function": {"name": "inspect"}}])
        self.assertEqual(len(client.posts), 2)
        self.assertIn("reasoning", client.posts[0][1])
        self.assertNotIn("reasoning", client.posts[1][1])
        self.assertEqual(self.ep.calls, 2)
        self.assertGreater(self.ep.inference_budget.uncertain, 0)

    def test_exhaustion_never_sends_a_request(self):
        self.ep.inference_budget.actual = self.ep.inference_budget.limit
        _, client = self.run_calls([])
        self.assertEqual(client.posts, [])
        self.assertEqual(self.ep.calls, 0)
        self.assertTrue(self.ep.no_credit)

    def test_unknown_prices_do_not_send_inference(self):
        _, client = self.run_calls([], prices={"models": [agent.LIGHT]})
        self.assertEqual(client.posts, [])
        self.assertTrue(self.ep.no_credit)

    def test_bad_pricing_is_rejected(self):
        for key, value in (("prompt", "NaN"), ("completion", True), ("request", "-1"), ("extra_fee", "0.01")):
            prices = catalog()
            for row in prices["data"]:
                row["pricing"][key] = value
            with self.subTest(key=key):
                budget = agent.InferenceBudget()
                budget.load_prices(prices)
                self.assertEqual(budget.prices, {})

    def test_affordable_output_cap_and_procedure_allowance(self):
        _, client = self.run_calls([reply()], cap=8000)
        self.assertLess(client.posts[0][1]["max_tokens"], 8000)
        self.assertGreaterEqual(client.posts[0][1]["max_tokens"], agent.MIN_REPLY_TOKENS)
        self.assertGreaterEqual(self.ep.inference_budget.remaining(), self.ep.inference_budget.protected)

    def test_price_or_token_envelope_violation_stops_following_calls(self):
        value, client = self.run_calls([reply(cost=0.2)])
        self.assertIsNotNone(value)
        self.assertTrue(self.ep.inference_budget.breached)
        self.assertTrue(self.ep.no_credit)
        with patch.object(agent, "model_client", return_value=client):
            self.assertIsNone(self.ep.think([], [agent.LIGHT], 64))
        self.assertEqual(len(client.posts), 1)

    def test_procedure_reserve_survives_an_ordinary_work_stop(self):
        budget = self.ep.inference_budget
        budget.actual = budget.limit - budget.protected
        _, client = self.run_calls([reply()])
        self.assertEqual(client.posts, [])
        self.assertTrue(self.ep.no_credit)
        with patch.object(agent, "model_client", return_value=client):
            value = self.ep.with_procedure_budget(self.ep.think, [{"role": "user", "content": "Approve the basket."}], [agent.LIGHT], 64)
        self.assertIsNotNone(value)
        self.assertEqual(len(client.posts), 1)
        self.assertTrue(self.ep.no_credit)

    def test_reserved_work_cannot_restart_an_exhausted_remote_key(self):
        self.ep.no_credit = True
        self.ep.budget_locally_stopped = False
        client = FakeClient([])
        with patch.object(agent, "model_client", return_value=client):
            value = self.ep.with_procedure_budget(self.ep.think, [], [agent.LIGHT], 64)
        self.assertIsNone(value)
        self.assertEqual(client.posts, [])

    def test_optional_work_cannot_consume_reserved_physical_attempts(self):
        self.ep.calls = agent.CALL_CAP - 2
        self.ep.inference_budget.attempts = self.ep.calls
        _, client = self.run_calls([reply()])
        self.assertEqual(client.posts, [])
        with patch.object(agent, "model_client", return_value=client):
            result = self.ep.with_procedure_budget(self.ep.think, [{"role": "user", "content": "Required approval."}], [agent.LIGHT], 64)
        self.assertIsNotNone(result)
        self.assertEqual(len(client.posts), 1)

    def test_cache_write_rate_is_an_alternative_input_price(self):
        budget = agent.InferenceBudget()
        budget.load_prices(catalog())
        self.assertEqual(budget.prices[agent.LIGHT]["input"], 1250)

    def test_reported_prompt_token_underestimate_stops_inference(self):
        response = reply()
        response.data["usage"]["prompt_tokens"] = 100000
        self.run_calls([response])
        self.assertTrue(self.ep.inference_budget.breached)
        self.assertTrue(self.ep.no_credit)

    def test_ninety_isolated_allocations_do_not_share_a_counter(self):
        def spend_all(_):
            budget = agent.InferenceBudget()
            budget.load_prices(catalog())
            body = {"model": agent.LIGHT, "messages": [{"role": "user", "content": "x"}], "max_tokens": 64}
            while True:
                quote, _ = budget.quote(body)
                if quote is None:
                    break
                budget.pending += quote["units"]
                budget.settle(quote, {})
            self.assertLessEqual(budget.actual + budget.uncertain, budget.limit)
            return budget.actual + budget.uncertain
        with ThreadPoolExecutor(max_workers=7) as pool:
            total = sum(pool.map(spend_all, range(90)))
        self.assertLessEqual(total, 3150000000)


if __name__ == "__main__":
    unittest.main()
