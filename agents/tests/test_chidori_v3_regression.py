"""Retarget existing financial, procedure and real-client checks to both v3 files."""
import ast
import hashlib
import importlib.util
import json
import unittest
from unittest.mock import patch

import test_chidori_budget as budget_tests
import test_chidori_integration as integration
import test_chidori_procedures as procedures
import test_chidori_v2_transport as transport
from test_chidori_v3 import AGENTS, ROOT, cart_episode, notice


FULL = AGENTS["ICDI_chidori_v3.py"]
BUDGET = AGENTS["ICDI_chidori_v3_budget.py"]


class V3FinancialTests(transport.V2BudgetTests):
    candidate = FULL

    def setUp(self):
        item = patch.object(transport, "v2", self.candidate)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()

    def test_original_methods_unchanged_outside_central_inference_hook(self):
        full = ast.parse((ROOT / "agents/ICDI_chidori_v3.py").read_text())
        budget = ast.parse((ROOT / "agents/ICDI_chidori_v3_budget.py").read_text())
        for owner in ("BudgetControls", "InferenceBudget"):
            left = next(n for n in full.body if isinstance(n, ast.ClassDef) and n.name == owner)
            right = next(n for n in budget.body if isinstance(n, ast.ClassDef) and n.name == owner)
            self.assertEqual(ast.dump(left), ast.dump(right))

    def test_procedure_reserve_survives_an_ordinary_work_stop(self):
        budget = self.ep.inference_budget
        budget.actual = budget.limit - budget.protected
        _, client = self.run_calls([budget_tests.reply()])
        self.assertEqual(client.posts, [])
        self.assertFalse(self.ep.no_credit)
        self.assertTrue(self.ep.request_denied)
        with patch.object(self.candidate, "model_client", return_value=client):
            value = self.ep.with_procedure_budget(self.ep.think, [{"role": "user", "content": "Approve the basket."}], [self.candidate.LIGHT], 64)
        self.assertIsNotNone(value)
        self.assertEqual(len(client.posts), 1)
        self.assertFalse(self.ep.no_credit)


class V3BudgetFinancialTests(V3FinancialTests):
    candidate = BUDGET


class V3ProcedureTests(transport.V2ProcedureTests):
    def setUp(self):
        item = patch.object(transport, "v2", FULL)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()


class V3BoundaryTests(transport.V2BoundaryTests):
    def setUp(self):
        item = patch.object(transport, "v2", FULL)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()


class V3TransportTests(transport.RealClientTransportTests):
    candidate = FULL
    filename = "ICDI_chidori_v3.py"

    def setUp(self):
        for name, value in (("v2", self.candidate), ("V2_PATH", ROOT / "agents" / self.filename)):
            item = patch.object(transport, name, value)
            item.start()
            self.addCleanup(item.stop)
        super().setUp()

    def test_stockout_withdrawal_replacement_and_checkout_through_real_client(self):
        agent = self.candidate
        ep = cart_episode(agent)
        ep.roles["cart"] = "replace_cart"
        ep.schemas["replace_cart"] = {"type": "object", "properties": {"lines": {"type": "array", "items": {
            "type": "object", "properties": {"id": {"type": "string"}, "quantity": {"type": "integer"}}}}}}
        ep.items[0].musts = ["Color: red"]
        ep.rows["replacement-item"] = {"id": "replacement-item", "attributes": {"color": "blue"}}
        ep.order.append("replacement-item")
        environment = []
        inference = []
        def http(url, **kwargs):
            body = kwargs["json"]
            if url.endswith("/environment/call"):
                environment.append(body)
                final = body["action"]["name"] == "buy" and "replacement-item" in ep.cart_ids()
                observation = {"notice": "The selected listing is unavailable. Withdraw the earlier color requirement."} if len(environment) == 1 else {"ok": True}
                return self.response(200, {"step": body["step"], "finished": final, "observation": observation})
            self.assertEqual(kwargs["headers"]["Authorization"], "Bearer offline-test-token")
            inference.append(body)
            if body["messages"][0]["content"].startswith(agent.NOTE_PROMPT):
                answer = notice(gone=["sample-item"], sold_out=True, drop=[{"item": "one item", "need": "Color: red"}])
            else:
                answer = {"ok": True, "problems": [], "better": []}
            return self.response(200, {"usage": {"cost": 0.000001}, "choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}]})
        with patch("src.agent.proxy_client.requests.post", side_effect=http), patch.object(agent, "model_client", return_value=self.client), patch.object(agent, "ENV_CLIENT", self.client), patch.object(agent.time, "sleep"):
            rejected = ep.env("buy", {})
            self.assertFalse(rejected["finished"])
            ep.observe("buy", rejected["observation"])
            self.assertFalse(ep.pending_notices)
            self.assertNotIn("Color: red", ep.live_of(ep.items[0]))
            self.assertIsNone(ep.env("buy", {}))
            self.assertEqual(len(environment), 1)
            args = {"lines": [{"id": "replacement-item", "quantity": 1}]}
            self.assertIsNotNone(ep.env("replace_cart", args))
            ep.note_cart("replace_cart", args)
            self.assertEqual(ep.cart_ids(), ["replacement-item"])
            self.assertEqual(ep.cart_check(), "")
            self.assertTrue(ep.env("buy", {})["finished"])
        self.assertEqual([row["step"] for row in environment], [1, 2, 3])
        self.assertEqual(len(inference), 2)
        self.assertEqual(ep.inference_budget.actual, 2000)
        self.assertEqual(ep.calls, 2)

    def test_negative_review_and_denied_escalation_use_one_paid_attempt(self):
        agent = self.candidate
        ep = cart_episode(agent)
        ep.inference_budget.actual = agent.money_units("0.029")
        self.content = json.dumps({"ok": False, "problems": ["The required property is absent."], "better": []})
        with patch.object(agent, "model_client", return_value=self.client), patch.object(agent, "ENV_CLIENT", self.client), patch.object(ep, "weak_lines", return_value=[]), patch.object(ep, "code_fits", return_value=[]), patch.object(ep, "reread", return_value=""):
            self.assertTrue(ep.cart_check())
            self.assertFalse(ep.no_credit)
            self.assertTrue(ep.request_denied)
            self.assertIn("required property", ep.finish())
        self.assertEqual(len(self.posts), 1)
        self.assertTrue(self.posts[0]["authorized"])
        self.assertEqual(self.posts[0]["body"]["model"], agent.LIGHT)
        self.assertFalse(ep.verdicts["sample-item#0"])


class V3BudgetTransportTests(V3TransportTests):
    candidate = BUDGET
    filename = "ICDI_chidori_v3_budget.py"


class V3ScopeTests(unittest.TestCase):
    def test_previous_agent_hashes_are_unchanged(self):
        expected = {"ICDI_chidori_v2.py": "3d40d11553e4bf2ea6d67d90305030cee98ca85d9ec397e070e468c1bfa3cdc9",
                    "ICDI_chidori_v2_budget.py": "a4228519020b09f18582abbdd265f8000874263b942e1a3a3450e024b1fea5e6",
                    "ICDI_chidori_v1.py": "295f290c70a2fc71035052f23f78e40bd8c6e23782c6d457829454e7473baaf3",
                    "ICDI_chidori_v1_budget.py": "7a8e0d8da12870e9b5af9f59ccae03166cec723d70f9ae31f4e10795871e533c"}
        for filename, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / "agents" / filename).read_bytes()).hexdigest(), digest)

    def test_only_declared_methods_changed_and_prompts_are_preserved(self):
        changes = {("InferenceBudget", "load_prices"), ("InferenceBudget", "quote"),
                   ("BudgetControls", "__init__"), ("BudgetControls", "think"), ("E9", "turn_loop"),
                   ("E7", "read_note"), ("G5A", "cart_check"), ("G6B", "finish"), ("G6B", "dispatch")}
        additions = {("BudgetControls", name) for name in ("valid_notice", "cart_review_context", "terminal_recovery_hold", "env")}
        for filename in AGENTS:
            before = ast.parse((ROOT / "agents" / filename.replace("v3", "v2")).read_text())
            after = ast.parse((ROOT / "agents" / filename).read_text())
            def methods(tree):
                return {(cls.name, method.name): ast.dump(method) for cls in tree.body if isinstance(cls, ast.ClassDef)
                        for method in cls.body if isinstance(method, ast.FunctionDef)}
            old, new = methods(before), methods(after)
            self.assertEqual(set(new) - set(old), additions)
            self.assertEqual(set(old) - set(new), set())
            self.assertEqual({key for key in old if old[key] != new[key]}, changes)
            for old_node, new_node in zip(before.body, after.body):
                if isinstance(old_node, ast.Assign) and not any(isinstance(t, ast.Name) and t.id == "BUDGET_PRICE_CONFIG" for t in old_node.targets):
                    self.assertEqual(ast.dump(old_node), ast.dump(new_node))

    def test_local_submission_preflight_for_both_variants(self):
        allowed = {"json", "random", "re", "time", "typing", "decimal", "src.agent.proxy_client"}
        forbidden = {"eval", "exec", "__import__", "open", "write_text", "write_bytes", "unlink", "chmod", "getenv"}
        for filename in AGENTS:
            path = ROOT / "agents" / filename
            tree = ast.parse(path.read_text())
            self.assertLess(path.stat().st_size, 1000000)
            self.assertEqual(len([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "agent_main"]), 1)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    self.assertTrue(all(alias.name in allowed for alias in node.names))
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module, allowed)
                elif isinstance(node, ast.Call):
                    name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                    self.assertNotIn(name, forbidden)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    import re
                    self.assertIsNone(re.search(r"\bt-[0-9a-f]{10}\b", node.value))
            self.assertNotIn("/inference/models", path.read_text())


class V3RecordedTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "logs/research/chidori-v3-development/replay.py"
        spec = importlib.util.spec_from_file_location("chidori_recorded_trace_replay", path)
        cls.replay = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.replay)
        run = cls.replay.RUN
        if not run.exists():
            raise unittest.SkipTest("Recorded verification unavailable")
        problems = {row["problem_id"]: row for row in cls.replay.rows(run / "problems.jsonl")}
        traces = {row["problem_id"]: row for row in cls.replay.rows(run / "sandbox/sandbox_output.jsonl")}
        cls.fixtures = []
        for row in cls.replay.rows(run / "episode_results.jsonl"):
            episode = row["episode"]
            session = episode["session_id"]
            if episode["verdict"]["score"] != 1 or any(t.get("kind") == "local_inference_stop" for t in traces[session]["dialogue"]):
                continue
            records = [r for r in cls.replay.rows(run / "sandbox" / ("request_log_" + session + ".jsonl")) if r.get("kind") == "summary"]
            cls.fixtures.append((problems[session], records))
        cls.reference = cls.replay.load(ROOT / "agents/ICDI_chidori_v2.py")

    def test_replay_reproduces_all_eleven_uninterrupted_passing_controls(self):
        self.assertEqual(len(self.fixtures), 11)
        for problem, records in self.fixtures:
            result = self.replay.replay(self.reference, problem, records)
            self.assertTrue(result["completed_same_trace"], result)

    def test_recovery_changes_preserve_all_eleven_control_traces_at_fixed_prices(self):
        for filename, agent in AGENTS.items():
            adapted = self.reference if filename.endswith("_budget.py") else None
            with patch.object(agent, "BUDGET_PRICE_CONFIG", self.reference.BUDGET_PRICE_CONFIG):
                for problem, records in self.fixtures:
                    result = self.replay.replay(agent, problem, records, adapted)
                    self.assertTrue(result["completed_same_trace"], result)

    def test_full_pricing_replays_stop_at_new_primary_model_admission(self):
        for filename, agent in AGENTS.items():
            adapted = self.reference if filename.endswith("_budget.py") else None
            for problem, records in self.fixtures:
                result = self.replay.replay(agent, problem, records, adapted)
                self.assertFalse(result["completed_same_trace"])
                self.assertEqual(result.get("changed_fields"), ["messages", "model"], result)
                self.assertEqual(result.get("expected_model"), agent.SPARE, result)
                self.assertEqual(result.get("actual_model"), agent.LIGHT, result)


if __name__ == "__main__":
    unittest.main()
