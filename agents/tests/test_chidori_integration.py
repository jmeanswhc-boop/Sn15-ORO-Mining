"""Offline action-boundary integration and combined-budget checks."""
import ast
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import test_chidori_budget as budget_tests
from test_chidori_procedures import agent, episode, duty, parse, decision, event, tool


ROOT = Path(__file__).resolve().parents[2]


class CombinedBudgetTests(budget_tests.BudgetTests):
    def setUp(self):
        module_patch = patch.object(budget_tests, "agent", agent)
        module_patch.start()
        self.addCleanup(module_patch.stop)
        super().setUp()

    def test_original_methods_unchanged_outside_central_inference_hook(self):
        before = ast.parse((ROOT / "agents/ICDI_chidori_v1_budget.py").read_text())
        after = ast.parse((ROOT / "agents/ICDI_chidori_v1.py").read_text())
        for name in ("InferenceBudget", "BudgetControls"):
            old = next(n for n in before.body if isinstance(n, ast.ClassDef) and n.name == name)
            new = next(n for n in after.body if isinstance(n, ast.ClassDef) and n.name == name)
            self.assertEqual(ast.dump(old), ast.dump(new))


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        network_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden"))
        request_mock = network_patch.start()
        self.addCleanup(network_patch.stop)
        self.addCleanup(request_mock.assert_not_called)
        sleep_patch = patch.object(agent.time, "sleep")
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)
        self.ep = episode()
        self.calls = []
        self.responses = []
        env_patch = patch.object(agent.ENV_CLIENT, "post_verbose", side_effect=self.environment)
        env_patch.start()
        self.addCleanup(env_patch.stop)

    def environment(self, path, body):
        self.assertEqual(path, "/environment/call")
        self.calls.append(body)
        name = body["action"]["name"]
        observation = self.responses.pop(0) if self.responses else ("Approved." if name == "message" else {"ok": True})
        return SimpleNamespace(data={"step": body["step"], "finished": name in ("buy", "decline", "finish"),
                                     "observation": observation}, error=None)

    def test_normal_purchase_cannot_bypass_a_failed_gate(self):
        parse(self.ep, [duty()])
        with patch.object(self.ep, "json_ask", return_value=None):
            self.assertIsNone(self.ep.env("buy", {}))
        self.assertEqual(self.calls, [])

    def test_forced_purchase_honors_negative_gate(self):
        parse(self.ep, [duty()])
        self.ep.cart_lines["synthetic-item"] = 1
        with patch.object(self.ep, "fresh", return_value=True), patch.object(self.ep, "reread"), patch.object(self.ep, "gate", return_value=(False, "Approval missing")):
            text = self.ep.finish()
        self.assertEqual(text, "Approval missing")
        self.assertEqual(self.calls, [])

    def test_forced_decline_honors_negative_gate(self):
        parse(self.ep, [duty("end")])
        with patch.object(self.ep, "fresh", return_value=True), patch.object(self.ep, "proven_basket", return_value=False), patch.object(self.ep, "gate", return_value=(False, "Decline procedure missing")):
            text = self.ep.finish()
        self.assertEqual(text, "Decline procedure missing")
        self.assertEqual(self.calls, [])

    def test_complete_cart_recovery_reports_a_blocked_gate_truthfully(self):
        parse(self.ep, [duty()])
        self.ep.confirmed = "synthetic-item"
        self.ep.verdicts["synthetic-item#0"] = True
        patches = [patch.object(self.ep, "sold_out", return_value=False),
                   patch.object(self.ep, "planned_basket", return_value=({"synthetic-item": 1}, [])),
                   patch.object(self.ep, "cart_calls", return_value=[("cart", {})]),
                   patch.object(self.ep, "fill_facts"), patch.object(self.ep, "show_links"),
                   patch.object(self.ep, "cart_check"), patch.object(self.ep, "reread"),
                   patch.object(self.ep, "run_cart_calls", return_value=(["changed"], True)),
                   patch.object(self.ep, "gate", return_value=(False, "Approval is still missing."))]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        result = self.ep.complete_cart()
        self.assertEqual(result, "Approval is still missing.")
        self.assertEqual(self.calls, [])

    def test_one_step_direct_checkout_cannot_skip_approval(self):
        parse(self.ep, [duty()])
        self.ep.step = self.ep.budget_steps - 1
        with patch.object(self.ep, "line_faults", return_value=[]), patch.object(self.ep, "json_ask", return_value=None):
            self.ep.direct_checkout("offline-test")
        self.assertEqual(self.calls, [])

    def test_one_step_purchase_with_valid_cached_proof_can_finish(self):
        parse(self.ep, [duty()])
        s = event(self.ep)
        with patch.object(self.ep, "json_ask", return_value=decision(["d1"], [s])):
            self.assertTrue(self.ep.gate("order")[0])
        self.ep.step = self.ep.budget_steps - 1
        with patch.object(self.ep, "json_ask") as ask:
            self.assertTrue(self.ep.env("buy", {})["finished"])
            ask.assert_not_called()
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0]["step"], self.ep.budget_steps)

    def test_choose_message_precedes_the_first_cart_action(self):
        parse(self.ep, [duty("choose")])
        call = {"duty_id": "d1", "tool": "message", "args": {"text": "May I choose an item?"}}
        answers = [decision(["d1"], done=False, call=call), decision(["d1"], [1])]
        with patch.object(self.ep, "json_ask", side_effect=answers):
            result = self.ep.env("cart", {"item": "synthetic-item", "quantity": 1})
        self.assertIsNotNone(result)
        self.assertEqual([r["action"]["name"] for r in self.calls], ["message", "cart"])
        self.assertEqual(self.ep.procedure_first_selection, 2)
        self.assertEqual(self.ep.msg_used, 1)

    def test_choose_is_scheduled_before_the_scripted_opening(self):
        parse(self.ep, [duty("choose")])
        order = []
        with patch.object(self.ep, "gate", side_effect=lambda stage: (order.append(stage) or (True, ""))), patch.object(agent.G3B, "opening", side_effect=lambda: order.append("opening")):
            self.ep.opening()
        self.assertEqual(order, ["choose", "opening"])

    def test_ordinary_required_read_is_available_as_evidence(self):
        parse(self.ep, [duty(messages=0, cart_dependent=False)])
        self.ep.env("read", {"page": "synthetic-manual"})
        with patch.object(self.ep, "json_ask", return_value=decision(["d1"], [1])):
            self.assertTrue(self.ep.gate("order")[0])

    def test_failed_tool_observation_is_not_evidence(self):
        parse(self.ep, [duty(messages=0, cart_dependent=False)])
        self.responses = [{"error": "Read rejected"}]
        self.ep.env("read", {"page": "synthetic-manual"})
        self.assertEqual(self.ep.procedure_events, {})
        with patch.object(self.ep, "json_ask", return_value=decision(["d1"], [1])):
            self.assertFalse(self.ep.gate("order")[0])

    def test_ordinary_interview_cannot_consume_reserved_messages(self):
        parse(self.ep, [duty(), duty()])
        self.ep.msg_limit = 2
        self.assertEqual(self.ep.send([(1, "", "An optional property question")], 1), [])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ep.msg_used, 0)

    def test_required_messages_can_use_the_reserved_allowance(self):
        parse(self.ep, [duty(), duty()])
        self.ep.msg_limit = 2
        call1 = {"duty_id": "d1", "tool": "message", "args": {"text": "Approve the first procedure?"}}
        call2 = {"duty_id": "d2", "tool": "message", "args": {"text": "Approve the second procedure?"}}
        middle = {"steps": [{"id": "d1", "done": True, "evidence": [1]}, {"id": "d2", "done": False, "evidence": []}], "call": call2}
        answers = [decision(["d1", "d2"], done=False, call=call1), middle, decision(["d2"], [2])]
        with patch.object(self.ep, "json_ask", side_effect=answers):
            self.assertTrue(self.ep.gate("order")[0])
        self.assertEqual(self.ep.msg_used, 2)
        self.assertEqual(len(self.calls), 2)

    def test_zero_duty_tool_trace_matches_the_original(self):
        parse(self.ep, [])
        original = budget_tests.load(budget_tests.BASE, "chidori_original_trace")
        other = original.Episode({"environment": {"binding": self.ep.binding, "policy_view": self.ep.pv}})
        other.roles = dict(self.ep.roles)
        with patch.object(self.ep, "json_ask") as ask, patch.object(original.ENV_CLIENT, "post_verbose", side_effect=self.environment):
            for ep in (self.ep, other):
                ep.env("read", {"page": "synthetic-manual"})
                ep.env("buy", {})
            ask.assert_not_called()
        self.assertEqual(self.calls[:2], self.calls[2:])

    def test_finished_session_does_not_receive_another_call(self):
        parse(self.ep, [])
        self.ep.env("buy", {})
        self.assertTrue(self.ep.finished)
        self.assertIsNone(self.ep.env("read", {"page": "synthetic-manual"}))
        self.assertEqual(len(self.calls), 1)

    def test_shared_finish_tool_is_discriminated_by_arguments(self):
        parse(self.ep, [duty("end")])
        self.ep.roles.update(buy="finish", refuse="finish")
        self.ep.brief.update(buy_args={"mode": "purchase"}, refuse_args={"mode": "decline"})
        self.ep.schemas["finish"] = tool("finish", {"mode": {"type": "string", "enum": ["purchase", "decline"]}})["function"]["parameters"]
        with patch.object(self.ep, "json_ask", return_value=None):
            self.assertIsNone(self.ep.env("finish", {"mode": "decline"}))
        self.assertEqual(self.calls, [])
        self.assertTrue(self.ep.env("finish", {"mode": "purchase"})["finished"])

    def test_indistinguishable_finish_modes_are_blocked(self):
        parse(self.ep, [])
        self.ep.roles.update(buy="finish", refuse="finish")
        self.assertIsNone(self.ep.env("finish", {}))
        self.assertEqual(self.calls, [])

    def test_agent_main_completes_a_required_decline_procedure(self):
        data = {"environment": {"binding": self.ep.binding, "policy_view": dict(self.ep.pv)}}
        procedure = duty("end", cart_dependent=False)
        procedure["source"] = "Ask before declining."
        def completion(path, body):
            parts = []
            for message in body["messages"]:
                content = message.get("content")
                if isinstance(content, list):
                    parts.extend(p.get("text", "") for p in content)
                else:
                    parts.append(content or "")
            prompt = "\n".join(parts)
            if "tools" in body:
                message = {"tool_calls": [{"id": "synthetic-call", "type": "function", "function": {"name": "decline", "arguments": "{}"}}]}
            elif prompt.startswith(agent.BRIEF_PROMPT):
                brief = {"items": [{"name": "requested item", "quantity": 1, "musts": []}],
                         "roles": {"buy": "buy", "refuse": "decline", "ask": "message"},
                         "process": [procedure], "messages": 5, "steps": 20}
                message = {"content": json.dumps(brief)}
            elif prompt.startswith(agent.PROCEDURE_PROMPT):
                if any(c["action"]["name"] == "message" for c in self.calls):
                    result = decision(["d1"], [1])
                else:
                    result = decision(["d1"], done=False, call={"duty_id": "d1", "tool": "message", "args": {"text": "May I finish without buying?"}})
                message = {"content": json.dumps(result)}
            else:
                message = {"content": "{}"}
            return SimpleNamespace(data={"usage": {"cost": 0.0001}, "choices": [{"message": message, "finish_reason": "stop"}]}, error=None)
        client = SimpleNamespace(get=lambda _: budget_tests.catalog(), post_verbose=completion)
        with patch.object(agent, "model_client", return_value=client):
            trace = agent.agent_main(data)
        self.assertEqual([c["action"]["name"] for c in self.calls], ["message", "decline"], trace)
        self.assertFalse([r for r in trace if r.get("kind") in ("crash", "soft_crash")], trace)
        self.assertTrue(any(r.get("kind") == "end" and r.get("finished") for r in trace), trace)

    def test_ninety_task_mixed_billing_workload(self):
        thread_state = threading.local()
        class MixedClient:
            def __init__(self, variant):
                self.variant = variant
                self.count = 0
            def get(self, path):
                return budget_tests.catalog()
            def post_verbose(self, path, body):
                self.count += 1
                if self.variant % 5 == 0 and self.count % 3 == 0:
                    raise TimeoutError()
                if self.variant % 5 == 1 and self.count % 3 == 0:
                    return SimpleNamespace(data=None, error={"status": 429, "body": "limited"})
                if self.variant % 5 == 2:
                    return budget_tests.reply(cost=None)
                if self.variant % 5 == 3:
                    return budget_tests.reply(choices=False)
                return budget_tests.reply()
        def run(index):
            thread_state.client = MixedClient(index)
            ep = episode()
            ep.budget_mode = "procedure"
            for _ in range(agent.CALL_CAP):
                if ep.no_credit or ep.calls >= agent.CALL_CAP:
                    break
                ep.think([{"role": "user", "content": "Synthetic billing workload."}], [agent.LIGHT], 64)
            budget = ep.inference_budget
            self.assertEqual(ep.calls, thread_state.client.count)
            self.assertEqual(budget.pending, 0)
            self.assertLessEqual(budget.actual + budget.uncertain, budget.limit)
            return {"accounted": budget.actual + budget.uncertain, "attempts": ep.calls}
        with patch.object(agent, "model_client", side_effect=lambda _: thread_state.client):
            with ThreadPoolExecutor(max_workers=7) as pool:
                results = list(pool.map(run, range(90)))
        total = sum(r["accounted"] for r in results)
        self.__class__.billing_workload_totals = {"tasks": 90, "physical_attempts": sum(r["attempts"] for r in results),
                                                 "mock_accounted_agent_usd": total / 1000000000,
                                                 "assumed_simulator_and_uncertainty_reserve_usd": 0.85,
                                                 "is_paid_or_race_measurement": False}
        self.assertLessEqual(total, 3150000000)
        self.assertLessEqual(total + 850000000, 4000000000)

    def test_original_search_and_review_methods_are_preserved(self):
        original = ast.parse(budget_tests.BASE.read_text())
        candidate = ast.parse((ROOT / "agents/ICDI_chidori_v1.py").read_text())
        classes = {n.name: n for n in candidate.body if isinstance(n, ast.ClassDef)}
        changed = {("E1", "one_call"), ("G6B", "finish"), ("G6B", "dispatch"), ("G6A", "complete_cart")}
        for cls in original.body:
            if not isinstance(cls, ast.ClassDef):
                continue
            updated = {n.name: n for n in classes[cls.name].body if isinstance(n, ast.FunctionDef)}
            for method in cls.body:
                if isinstance(method, ast.FunctionDef) and (cls.name, method.name) not in changed:
                    self.assertEqual(ast.dump(method), ast.dump(updated[method.name]), (cls.name, method.name))


if __name__ == "__main__":
    unittest.main()
