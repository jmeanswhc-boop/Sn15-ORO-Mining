"""Offline procedure state checks; environment-boundary checks are added separately."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from test_chidori_budget import load, problem


ROOT = Path(__file__).resolve().parents[2]
agent = load(ROOT / "agents/ICDI_chidori_v1.py", "chidori_procedure_candidate")


def tool(name, props=None):
    return {"type": "function", "function": {"name": name, "parameters": {
        "type": "object", "properties": props or {}, "required": list(props or {}), "additionalProperties": False}}}


TOOLS = [tool("message", {"text": {"type": "string"}}), tool("read", {"page": {"type": "string"}}),
         tool("cart", {"item": {"type": "string"}, "quantity": {"type": "integer"}}), tool("buy"), tool("decline")]
REQUEST = "Ask before choosing. Approve the final basket before ordering. Read the manual before ordering. Ask before declining. One message may approve both."


def episode():
    data = problem()
    data["environment"]["policy_view"].update(query=REQUEST, tools=TOOLS)
    ep = agent.Episode(data)
    ep.roles = {"ask": "message", "open": "read", "cart": "cart", "buy": "buy", "refuse": "decline"}
    ep.msg_limit = 5
    ep.room = lambda: 100
    ep.clock = lambda: 0
    return ep


def duty(stage="order", messages=1, **extra):
    result = {"step": "Obtain the requested approval", "when": stage, "messages": messages,
              "actions": max(1, messages), "source": "Approve the final basket before ordering.",
              "cart_dependent": stage == "order"}
    result.update(extra)
    return result


def parse(ep, entries):
    ep.brief = {"process": entries}
    with patch.object(agent.BudgetControls, "read_brief", return_value=None):
        ep.read_brief()


def event(ep, name="message", observation="Approved.", step=None):
    ep.step = ep.step + 1 if step is None else step
    ep.procedure_events[ep.step] = {"step": ep.step, "tool": name, "args": {},
                                    "observation": observation, "signature": ep.procedure_cart_signature}
    return ep.step


def decision(ids, evidence=None, done=True, call=None, **extra):
    answer = {"steps": [{"id": i, "done": done, "evidence": evidence or []} for i in ids], "call": call}
    answer.update(extra)
    return answer


class ProcedureStateTests(unittest.TestCase):
    def setUp(self):
        request_patch = patch("requests.sessions.Session.request", side_effect=AssertionError("Network forbidden"))
        mock = request_patch.start()
        self.addCleanup(request_patch.stop)
        self.addCleanup(mock.assert_not_called)
        self.ep = episode()

    def check_gate(self, answer, stage="order"):
        with patch.object(self.ep, "json_ask", return_value=answer):
            return self.ep.gate(stage)[0]

    def test_explicit_zero_duties_needs_no_review(self):
        parse(self.ep, [])
        with patch.object(self.ep, "json_ask") as ask:
            self.assertTrue(self.ep.gate("order")[0])
            self.assertEqual(self.ep.reserve(), 1)
            self.assertEqual(self.ep.free_msgs(), 5)
            ask.assert_not_called()

    def test_absent_malformed_and_invented_duties_are_unknown(self):
        for entries in (None, {}, [duty(source="An instruction that is absent.")], [None]):
            with self.subTest(entries=entries):
                self.ep = episode()
                parse(self.ep, entries)
                self.assertFalse(self.ep.procedures_valid)
                self.assertFalse(self.check_gate({}))

    def test_missing_empty_and_explicitly_incomplete_responses_do_not_pass(self):
        for answer in (None, {}, decision(["d1"], done=False)):
            with self.subTest(answer=answer):
                self.ep = episode()
                parse(self.ep, [duty()])
                self.assertFalse(self.check_gate(answer))
                self.assertFalse(self.ep.duties[0]["completed"])

    def test_every_due_id_must_be_covered_exactly_once(self):
        for ids in (["d1"], ["d1", "d1"], ["d1", "unknown"], ["d1", {}]):
            with self.subTest(ids=ids):
                self.ep = episode()
                parse(self.ep, [duty(), duty()])
                s = event(self.ep)
                self.assertFalse(self.check_gate(decision(ids, [s])))

    def test_nonexistent_boolean_and_unexecuted_evidence_do_not_pass(self):
        parse(self.ep, [duty()])
        for proof in ([999], [True], [], ["1"]):
            self.assertFalse(self.check_gate(decision(["d1"], proof)))

    def test_correct_proof_is_cached_without_extra_inference(self):
        parse(self.ep, [duty()])
        s = event(self.ep)
        self.assertTrue(self.check_gate(decision(["d1"], [s])))
        with patch.object(self.ep, "json_ask") as ask:
            self.assertTrue(self.ep.gate("order")[0])
            ask.assert_not_called()

    def test_cart_change_invalidates_approval(self):
        parse(self.ep, [duty()])
        s = event(self.ep)
        self.assertTrue(self.check_gate(decision(["d1"], [s])))
        self.ep.cart_lines["synthetic-item"] = 1
        self.assertTrue(self.ep.gate_due("order"))
        self.assertFalse(self.check_gate(decision(["d1"], [s])))

    def test_one_time_read_survives_cart_changes(self):
        parse(self.ep, [duty(messages=0, cart_dependent=False)])
        s = event(self.ep, "read")
        self.assertTrue(self.check_gate(decision(["d1"], [s])))
        self.ep.cart_lines["synthetic-item"] = 1
        self.assertFalse(self.ep.gate_due("order"))

    def test_independent_duties_need_distinct_messages(self):
        parse(self.ep, [duty(), duty()])
        self.assertEqual(self.ep.msg_reserve, 2)
        self.assertEqual(self.ep.reserve(), 3)
        s = event(self.ep)
        self.assertFalse(self.check_gate(decision(["d1", "d2"], [s])))
        self.assertEqual(sum(d["completed"] for d in self.ep.duties), 1)

    def test_shared_message_requires_explicit_public_permission(self):
        shared = duty(group="shared", combine_source="One message may approve both.")
        parse(self.ep, [shared, shared])
        self.assertEqual(self.ep.msg_reserve, 1)
        s = event(self.ep)
        self.assertTrue(self.check_gate(decision(["d1", "d2"], [s])))

    def test_mutually_exclusive_branches_are_not_summed(self):
        parse(self.ep, [duty("choose"), duty(messages=2), duty("end", messages=3)])
        self.assertEqual(self.ep.msg_reserve, 4)
        self.assertEqual(self.ep.reserve(), 5)

    def test_minimum_message_count_cannot_be_satisfied_by_one_record(self):
        parse(self.ep, [duty(messages=2)])
        s = event(self.ep)
        self.assertFalse(self.check_gate(decision(["d1"], [s])))

    def test_before_selection_duty_cannot_be_repaired_after_selection(self):
        parse(self.ep, [duty("choose")])
        self.ep.procedure_first_selection = 1
        event(self.ep, step=2)
        self.assertFalse(self.check_gate(decision(["d1"], [2]), "choose"))

    def test_proposed_final_or_cart_action_cannot_execute_inside_gate(self):
        for name in ("buy", "decline", "cart"):
            self.ep = episode()
            parse(self.ep, [duty()])
            call = {"duty_id": "d1", "tool": name, "args": {}}
            with patch.object(self.ep, "env") as env:
                self.assertFalse(self.check_gate(decision(["d1"], done=False, call=call)))
                env.assert_not_called()

    def test_required_message_runs_then_needs_actual_proof(self):
        parse(self.ep, [duty()])
        call = {"duty_id": "d1", "tool": "message", "args": {"text": "May I order this basket?"}}
        def send(*_):
            event(self.ep)
            self.ep.msg_used += 1
            return [(1, "May I order?", "Approved.")]
        answers = [decision(["d1"], done=False, call=call), decision(["d1"], [1])]
        with patch.object(self.ep, "send", side_effect=send), patch.object(self.ep, "json_ask", side_effect=answers):
            self.assertTrue(self.ep.gate("order")[0])
        self.assertEqual(self.ep.gate_msgs, 1)

    def test_changed_shopper_need_returns_for_cart_repair(self):
        parse(self.ep, [duty()])
        event(self.ep)
        answer = decision(["d1"], [1], hold="The shopper requests a different quantity.")
        self.assertFalse(self.check_gate(answer))
        self.assertFalse(self.ep.duties[0]["completed"])
        self.assertEqual(self.ep.version, 1)

    def test_no_progress_rechecks_are_bounded(self):
        parse(self.ep, [duty()])
        with patch.object(self.ep, "json_ask", return_value=None) as ask:
            for _ in range(5):
                self.assertFalse(self.ep.gate("order")[0])
        self.assertEqual(ask.call_count, 2)


if __name__ == "__main__":
    unittest.main()
