"""V4 checks: every real network request is forbidden."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_chidori_procedures as procedures
from test_chidori_v3 import ROOT, load, episode, cart_episode, Client, response, OfflineCase, problem


AGENTS = {name: load(ROOT / "agents" / name, name[:-3]) for name in ("ICDI_chidori_v4.py", "ICDI_chidori_v4_budget.py")}
FULL = AGENTS["ICDI_chidori_v4.py"]


class ProcedureCompatibilityTests(procedures.ProcedureStateTests):
    def setUp(self):
        item = patch.object(procedures, "agent", FULL)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()


class RoleOutputTests(OfflineCase):
    def test_scoped_transient_failures_stop_after_two_accounted_attempts(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            failure = SimpleNamespace(data=None, error={"status": 500, "body": "temporary"})
            client = Client([failure, failure])
            with patch.object(agent,"model_client",return_value=client), patch.object(agent.time,"sleep") as sleep:
                self.assertIsNone(ep.with_inference_role("repair",ep.json_ask,"Repair decision",[agent.LIGHT,agent.SPARE],2048))
            self.assertEqual(ep.calls,2)
            self.assertEqual(len(client.posts),2)
            self.assertEqual(sleep.call_count,1)
            self.assertFalse(ep.no_credit)
    def test_procedure_role_reaches_transport_with_answer_room(self):
        ep = episode(FULL)
        client = Client([response()])
        with patch.object(FULL, "model_client", return_value=client):
            self.assertIsNotNone(ep.with_inference_role("procedure", ep.json_ask, "Carry out the public duty.", [FULL.LIGHT], 2048))
        self.assertEqual(client.posts[0][1]["max_tokens"], 2048)
        self.assertEqual(client.posts[0][1]["reasoning"], {"enabled": False})
        self.assertEqual(ep.inference_role, "")

    def test_normal_inference_keeps_existing_request_settings(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            client = Client([response()])
            with patch.object(agent, "model_client", return_value=client):
                ep.json_ask("Ordinary request", [agent.LIGHT], 700)
            self.assertEqual(client.posts[0][1]["max_tokens"], 700)
            self.assertNotIn("reasoning", client.posts[0][1])

    def test_unsupported_role_setting_retries_once_through_accounting(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            bad = SimpleNamespace(data=None, error={"status": 422, "body": "unsupported reasoning"})
            client = Client([bad, response()])
            with patch.object(agent, "model_client", return_value=client):
                self.assertIsNotNone(ep.with_inference_role("procedure", ep.json_ask, "Public duty", [agent.LIGHT], 2048))
            self.assertEqual(ep.calls, 2)
            self.assertGreater(ep.inference_budget.uncertain, 0)
            self.assertIn("reasoning", client.posts[0][1])
            self.assertNotIn("reasoning", client.posts[1][1])

    def test_remote_credit_stop_cannot_be_reopened_by_role_fallback(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            client = Client([SimpleNamespace(data=None, error={"status": 402, "body": "limit"})])
            with patch.object(agent, "model_client", return_value=client):
                self.assertIsNone(ep.with_inference_role("procedure", ep.json_ask, "Public duty", [agent.LIGHT], 2048))
                self.assertIsNone(ep.with_inference_role("repair", ep.json_ask, "Repair", [agent.LIGHT], 2048))
            self.assertEqual(len(client.posts), 1)
            self.assertTrue(ep.no_credit)

    def test_repeated_identical_public_read_does_not_restart_procedure_attempts(self):
        with patch.object(procedures, "agent", FULL):
            ep = procedures.episode()
            procedures.parse(ep, [procedures.duty(messages=0)])
            with patch.object(ep, "json_ask", return_value=None) as ask:
                ep.gate("order")
                ep.gate("order")
                ep.gate("order")
                self.assertEqual(ask.call_count, 2)
                procedures.event(ep, "read", "The same public record.")
                ep.gate("order")
                ep.gate("order")
                count = ask.call_count
                procedures.event(ep, "read", "The same public record.")
                ep.gate("order")
                self.assertEqual(ask.call_count, count)


def repair_episode(agent, array=False):
    data = problem()
    properties = {"lines": {"type": "array", "items": {"type": "object", "properties": {
        "item": {"type": "string"}, "quantity": {"type": "integer"}}, "required": ["item", "quantity"]}}} if array else {
            "mode": {"type": "string", "enum": ["include", "discard"]},
            "item": {"type": "string"}, "quantity": {"type": "integer"}}
    data["environment"]["policy_view"]["tools"].append({"type": "function", "function": {
        "name": "basket_change", "parameters": {"type": "object", "properties": properties, "required": list(properties)}}})
    ep = episode(agent, data)
    ep.coverage_done = True  # Isolate recovery; source coverage has its own tests.
    ep.items = [agent.Item("requested item", "", 1, [])]
    ep.roles = {"cart": "basket_change", "buy": "buy", "refuse": "decline", "open": "read"}
    ep.rows = {"old": {"id": "old", "price": 1}, "new": {"id": "new", "price": 2}}
    ep.order = ["old", "new"]
    return ep


def tool_response(name, arguments):
    result = response(None)
    result.data["choices"][0]["message"]["tool_calls"] = [{"id": "offline-action", "type": "function", "function": {
        "name": name, "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)}}]
    return result


class RecoveryFlowTests(OfflineCase):
    def test_ambiguous_false_acknowledgement_uses_public_cart_view_before_local_update(self):
        for agent in AGENTS.values():
            for changed in (True,False):
                ep = repair_episode(agent)
                ep.cart_lines = {"old":1}
                ep.roles["cart_view"] = "basket_view"
                ep.schemas["basket_view"] = {"type":"object","properties":{}}
                calls = []
                def post(path,body):
                    calls.append(body)
                    observation = {"has_items":False} if body["action"]["name"]=="basket_change" else {
                        "items":[] if changed else [{"id":"old","quantity":1}]}
                    return SimpleNamespace(data={"step":body["step"],"finished":False,"observation":observation},error=None)
                with patch.object(agent.ENV_CLIENT,"post_verbose",side_effect=post),patch.object(agent.time,"sleep"):
                    self.assertEqual(ep.repair_cart_calls([{"mode":"discard","item":"old","quantity":1}]),changed)
                self.assertEqual(ep.cart_lines,{} if changed else {"old":1})
                self.assertEqual(len(calls),2)

    def test_finished_session_never_runs_deterministic_builder_or_audits(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.finished = True
            ep.order_rejected = True
            with patch.object(ep,"proven_basket") as build,patch.object(ep,"recover") as recover,patch.object(ep,"json_ask") as ask:
                self.assertEqual(ep.finish(),"")
            build.assert_not_called()
            recover.assert_not_called()
            ask.assert_not_called()
    def test_required_procedure_actions_do_not_consume_optional_repair_action_quota(self):
        ep = repair_episode(FULL)
        ep.recovery_active = True
        ep.recovery_environment_limit = 0
        ep.procedure_busy = True
        calls,post = self.fake_environment(ep)
        with patch.object(FULL.ENV_CLIENT,"post_verbose",side_effect=post),patch.object(FULL.time,"sleep"):
            for _ in range(4):
                self.assertIsNotNone(ep.env("read",{"id":"public-page"}))
        self.assertEqual(ep.recovery_environment_calls,0)
        self.assertEqual(len(calls),4)

    def fake_environment(self, ep, success=True):
        calls = []
        def post(path, body):
            calls.append(body)
            terminal = body["action"]["name"] in ("buy", "decline")
            return SimpleNamespace(data={"step": body["step"], "finished": terminal,
                                         "observation": {"accepted": success}}, error=None)
        return calls, post

    def test_stockout_cleanup_preserves_surviving_quantities_for_both_schema_shapes(self):
        for agent in AGENTS.values():
            for array in (False, True):
                ep = repair_episode(agent, array)
                ep.cart_lines = {"old": 1, "new": 3}
                ep.items[0].quantity = 3
                ep.verdicts["new#0"] = True
                ep.review_contexts["new#0"] = ep.cart_review_context(["new"])
                ep.unavail.add("old")
                env_calls, post = self.fake_environment(ep)
                client = Client([tool_response("buy", {})])
                def dispatch(calls):
                    return [str(ep.env(c["function"]["name"], json.loads(c["function"]["arguments"]))) for c in calls]
                with patch.object(agent.ENV_CLIENT, "post_verbose", side_effect=post), patch.object(agent, "model_client", return_value=client), patch.object(ep, "dispatch", side_effect=dispatch), patch.object(agent.time, "sleep"):
                    self.assertTrue(ep.recover("The old line sold out."))
                self.assertTrue(ep.finished)
                self.assertEqual(ep.cart_lines, {"new": 3})
                self.assertEqual([c["action"]["name"] for c in env_calls], ["basket_change", "buy"])
                self.assertEqual(ep.inference_budget.attempts, 1)

    def test_failed_cart_observation_is_not_recorded_as_a_successful_removal(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.cart_lines = {"old": 1}
            ep.unavail.add("old")
            env_calls, post = self.fake_environment(ep, False)
            with patch.object(agent.ENV_CLIENT, "post_verbose", side_effect=post), patch.object(ep, "observe"), patch.object(agent.time, "sleep"):
                self.assertFalse(ep.recover("The old line sold out."))
            self.assertEqual(ep.cart_lines, {"old": 1})
            self.assertEqual(len(env_calls), 1)

    def test_truncated_invalid_repair_arguments_never_reach_environment(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            client = Client([tool_response("basket_change", '{"mode":"include",') for _ in range(2)])
            with patch.object(agent, "model_client", return_value=client), patch.object(agent.ENV_CLIENT, "post_verbose") as env:
                self.assertFalse(ep.recover("A previous action was incomplete."))
            env.assert_not_called()
            self.assertEqual(len(client.posts), 2)

    def test_unchanged_state_has_bounded_recovery_and_no_extra_attempt_on_repeat(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            client = Client([tool_response("buy", {}), tool_response("buy", {})])
            with patch.object(agent, "model_client", return_value=client), patch.object(ep, "dispatch", return_value=["Locally held; no step executed."]):
                self.assertFalse(ep.recover("A review is unresolved."))
                self.assertFalse(ep.recover("The same review is unresolved."))
            self.assertEqual(len(client.posts), 2)

    def test_finished_or_credit_stopped_session_cannot_enter_recovery(self):
        for agent in AGENTS.values():
            for field in ("finished", "no_credit"):
                ep = repair_episode(agent)
                setattr(ep, field, True)
                with patch.object(agent, "model_client") as model, patch.object(agent.ENV_CLIENT, "post_verbose") as env:
                    self.assertFalse(ep.recover("A repair is pending."))
                model.assert_not_called()
                env.assert_not_called()

    def test_healthy_fallback_does_not_enter_recovery(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.cart_lines = {"new": 1}
            ep.verdicts["new#0"] = True
            ep.review_contexts["new#0"] = ep.cart_review_context(["new"])
            env_calls, post = self.fake_environment(ep)
            with patch.object(ep, "reread", return_value=""), patch.object(ep, "recover", side_effect=AssertionError("Healthy path must not recover")), patch.object(agent.ENV_CLIENT, "post_verbose", side_effect=post), patch.object(agent.time, "sleep"):
                ep.finish()
            self.assertTrue(ep.finished)
            self.assertEqual(len(env_calls), 1)

    def test_repair_snapshot_preserves_withdrawals_obligations_and_incomplete_search(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.items[0].musts = ["First property: old", "Second property: retained"]
            ep.items[0].dropped = ["First property: old"]
            ep.items[0].told = ["First property: replacement"]
            ep.changes = ["The first property changed."]
            for number in range(20):
                ident = "candidate-"+str(number)
                ep.rows[ident] = {"id": ident}
                ep.order.append(ident)
            messages = ep.repair_messages("Repair current state.", candidates=4)
            text = messages[-1]["content"]
            self.assertIn("Second property: retained", text)
            self.assertIn("First property: replacement", text)
            self.assertIn("other_candidates_not_shown", text)
            self.assertIn("not the full search space", text)


def evidence(ident="new", key="feature", value="wanted"):
    return {"id": ident, "scope": "attribute", "key": key, "value": value}


def dispute_answer():
    source = evidence()
    return {"ok": True, "reason": "The cited public field settles this requirement.",
            "requirements": [{"item": 1, "need": "Feature: wanted", "ok": True, "evidence": [source]}],
            "comparisons": [], "evidence": [source]}


def disputed_episode(agent):
    ep = repair_episode(agent)
    ep.items[0].musts = ["Feature: wanted"]
    ep.cart_lines = {"new": 1}
    ep.full["new"] = {"id": "new", "attributes": {"feature": "wanted", "another feature": "unchanged"}}
    ep.verdicts["new#0"] = False
    ep.review_messages["new#0"] = "An earlier reviewer rejected the feature."
    ep.checked.add("new#0")
    ep.review_contexts["new#0"] = ep.cart_review_context(["new"])
    return ep


class DisputeTests(OfflineCase):
    def test_single_attribute_record_can_ground_evidence_without_guessing_ambiguous_blocks(self):
        for agent in AGENTS.values():
            ep = disputed_episode(agent)
            ep.full["new"] = {"id":"new","details":{"feature":"wanted"}}
            self.assertTrue(ep.grounded_evidence([evidence()],{"new"}))
            ep.full["new"]["other"] = {"feature":"different"}
            self.assertFalse(ep.grounded_evidence([evidence()],{"new"}))
    def test_unavailable_adjudication_does_not_suppress_normal_review_of_new_evidence(self):
        for agent in AGENTS.values():
            ep = disputed_episode(agent)
            ep.rows["new"]["price"] = 3
            with patch.object(ep,"json_ask",side_effect=[{"ok":False},{"ok":True}]) as ask, patch.object(ep,"weak_lines",return_value=[]), patch.object(ep,"code_fits",return_value=[]):
                self.assertEqual(ep.cart_check(),"")
            self.assertTrue(ep.verdicts["new#0"])
            self.assertEqual(ask.call_count,2)
    def test_approval_requires_complete_grounded_requirement_evidence(self):
        for agent in AGENTS.values():
            ep = disputed_episode(agent)
            with patch.object(ep, "json_ask", return_value=dispute_answer()):
                self.assertTrue(ep.adjudicate_review(["new"]))
            self.assertTrue(ep.verdicts["new#0"])
            self.assertNotIn("new#0", ep.review_messages)

    def test_bare_missing_wrong_or_invented_evidence_cannot_override_negative_review(self):
        answers = [{"ok": True}, dict(dispute_answer(), requirements=[]),
                   dict(dispute_answer(), evidence=[evidence(value="invented")]),
                   dict(dispute_answer(), evidence=[evidence(ident="unknown")]),
                   dict(dispute_answer(), evidence=[{"id": [], "key": []}])]
        for agent in AGENTS.values():
            for answer in answers:
                ep = disputed_episode(agent)
                with patch.object(ep, "json_ask", return_value=answer):
                    self.assertFalse(ep.adjudicate_review(["new"]))
                self.assertFalse(ep.verdicts["new#0"])

    def test_every_alternative_supporting_rejection_must_be_resolved(self):
        for agent in AGENTS.values():
            ep = disputed_episode(agent)
            ep.check_better["new#0"] = ["old"]
            with patch.object(ep, "json_ask", return_value=dispute_answer()):
                self.assertFalse(ep.adjudicate_review(["new"]))
            ep = disputed_episode(agent)
            ep.check_better["new#0"] = ["old"]
            ep.unavail.add("old")
            answer = dict(dispute_answer(), comparisons=[{"id": "old", "better": False,
                                                          "evidence": [{"id": "old", "unavailable": True}]}])
            with patch.object(ep, "json_ask", return_value=answer):
                self.assertTrue(ep.adjudicate_review(["new"]))

    def test_alternative_availability_and_verdict_changes_invalidate_only_dependent_context(self):
        for agent in AGENTS.values():
            ep = disputed_episode(agent)
            ep.check_better["new#0"] = ["old"]
            before = ep.cart_review_context(["new"])
            ep.verdicts["old#0"] = False
            after = ep.cart_review_context(["new"])
            self.assertNotEqual(before, after)
            ep.unavail.add("old")
            self.assertNotEqual(after, ep.cart_review_context(["new"]))
            ep.verdicts["new#0"] = True
            positive = ep.cart_review_context(["new"])
            ep.unavail.clear()
            self.assertEqual(positive, ep.cart_review_context(["new"]))

    def test_identical_unresolved_dispute_is_not_repeated(self):
        for agent in AGENTS.values():
            ep = disputed_episode(agent)
            with patch.object(ep, "json_ask", return_value={"ok": False}) as ask:
                self.assertFalse(ep.adjudicate_review(["new"]))
                self.assertFalse(ep.adjudicate_review(["new"]))
            self.assertEqual(ask.call_count, 1)

    def test_credit_stop_unavailable_cart_and_pending_notice_cannot_be_overridden(self):
        for agent in AGENTS.values():
            for field in ("no_credit", "unavailable", "pending"):
                ep = disputed_episode(agent)
                if field == "no_credit":
                    ep.no_credit = True
                elif field == "unavailable":
                    ep.unavail.add("new")
                else:
                    ep.pending_notices.append("A public instruction remains pending.")
                with patch.object(ep, "json_ask") as ask:
                    self.assertFalse(ep.adjudicate_review(["new"]))
                ask.assert_not_called()


class CoverageAndTerminalTests(OfflineCase):
    def test_source_quantity_correction_updates_requested_units_without_inventing_attribute_need(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.coverage_done = False
            ep.request = "Please purchase two units of the requested item."
            answer = {"complete":True,"missing":[{"item":1,"source":"two units","need":"Requested quantity: 2","quantity":2}]}
            with patch.object(ep,"json_ask",return_value=answer):
                ep.ensure_source_coverage()
            self.assertEqual(ep.items[0].quantity,2)
            self.assertEqual(ep.items[0].musts,[])
            ep = repair_episode(agent)
            ep.coverage_done = False
            ep.request = "Please purchase two units of the requested item."
            ep.items[0].quantity = 3
            ep.changes = ["The shopper changed the requested number of units."]
            with patch.object(ep,"json_ask",return_value=answer):
                ep.ensure_source_coverage()
            self.assertEqual(ep.items[0].quantity,3)

    def test_source_audit_adds_omission_without_losing_exclusions_alternatives_or_disclosures(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.coverage_done = False
            ep.request = "Choose a geometric print. Use red or blue. Do not use a metal clasp."
            ep.items[0].musts = ["Clasp: not metal", "Color: red or blue"]
            ep.items[0].told = ["Length: at least 8 cm"]
            answer = {"complete": True, "missing": [{"item": 1, "source": "geometric print", "need": "Print: geometric"}]}
            with patch.object(ep, "json_ask", return_value=answer) as ask:
                self.assertIn("Print: geometric", ep.ensure_source_coverage())
                self.assertEqual(ep.ensure_source_coverage(), "")
            self.assertEqual(ask.call_count, 1)
            self.assertEqual(set(ep.live_of(ep.items[0])), {"Clasp: not metal", "Color: red or blue", "Length: at least 8 cm", "Print: geometric"})
            self.assertEqual(ep.version, 1)

    def test_audit_cannot_restore_a_withdrawn_property_or_move_it_to_another_item(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.coverage_done = False
            ep.request = "The first item should be red and the second blue."
            ep.items[0].musts = ["Color: red"]
            ep.items[0].dropped = ["Color: red"]
            ep.items.append(agent.Item("second item", "", 2, []))
            answer = {"complete": True, "missing": [{"item": 1, "source": "first item should be red", "need": "Color: red"},
                                                    {"item": 2, "source": "second blue", "need": "Color: blue"}]}
            with patch.object(ep, "json_ask", return_value=answer):
                ep.ensure_source_coverage()
            self.assertNotIn("Color: red", ep.live_of(ep.items[0]))
            self.assertIn("Color: blue", ep.live_of(ep.items[1]))
            self.assertEqual(ep.items[1].quantity, 2)

    def test_invalid_or_unavailable_audit_is_advisory_and_does_not_claim_verified_coverage(self):
        for agent in AGENTS.values():
            for answer in (None, {}, {"complete": False, "missing": []},
                           {"complete": True, "missing": [{"item": 99, "source": "absent", "need": "Feature: invented"}]},
                           {"complete": True, "missing": [{"item": 1, "source": "unquoted claim", "need": "Feature: invented"}]}):
                ep = repair_episode(agent)
                ep.coverage_done = False
                before = ep.ledger()
                with patch.object(ep, "json_ask", return_value=answer) as ask:
                    self.assertEqual(ep.ensure_source_coverage(), "")
                    self.assertEqual(ep.ensure_source_coverage(), "")
                self.assertFalse(ep.coverage_done)
                self.assertEqual(ep.ledger(), before)
                self.assertEqual(ask.call_count, 1)

    def test_audit_batch_validation_is_atomic(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.coverage_done = False
            ep.request = "A geometric print is required."
            answer = {"complete": True, "missing": [{"item": 1, "source": "geometric print", "need": "Print: geometric"},
                                                    {"item": 1, "source": "fabricated", "need": "Feature: invented"}]}
            with patch.object(ep, "json_ask", return_value=answer):
                ep.ensure_source_coverage()
            self.assertEqual(ep.items[0].musts, [])
            self.assertEqual(ep.version, 0)

    def test_fallback_cannot_checkout_known_empty_cart_then_silently_decline(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.cart_touched = True
            ep.cart_empty_confirmed = True
            ep.cart_changed_at = 2
            ep.checkouts = 1
            ep.refused_at = 0
            with patch.object(ep, "recover", return_value=False), patch.object(ep, "proven_basket", return_value=False), patch.object(agent.ENV_CLIENT, "post_verbose") as env:
                self.assertIn("empty", ep.finish(True))
            env.assert_not_called()

    def test_fallback_replacement_requires_valid_review_and_unknown_is_not_approval(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.cart_lines = {"new": 1}
            ep.fallback_active = True
            with patch.object(ep, "cart_check", return_value="") as check:
                self.assertIn("No usable", ep.terminal_recovery_hold(True))
            self.assertTrue(ep.review_needed)
            self.assertEqual(check.call_count, 1)

    def test_valid_cached_review_survives_unchanged_fallback_without_another_model_call(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.cart_lines = {"new": 1}
            ep.fallback_active = True
            ep.verdicts["new#0"] = True
            ep.review_contexts["new#0"] = ep.cart_review_context(["new"])
            with patch.object(ep, "json_ask") as ask:
                self.assertEqual(ep.terminal_recovery_hold(True), "")
            ask.assert_not_called()

    def test_failed_cart_acknowledgement_cannot_empty_local_basket(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.cart_lines = {"old": 1}
            args = {"mode": "discard", "item": "old", "quantity": 1}
            ep.last_cart_receipt = ("basket_change", agent.dump(args), False)
            ep.note_cart("basket_change", args)
            self.assertEqual(ep.cart_lines, {"old": 1})
            self.assertFalse(ep.cart_empty_confirmed)


class InterviewPreservationTests(OfflineCase):
    def test_ambiguous_topic_is_skipped_only_after_grounded_same_property_decision(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.request = "The item must have a matte finish."
            ep.items[0].musts = ["Finish: matte"]
            ep.lanes = {1:["surface finish","size"]}
            ep.full["new"] = {"id":"new","attributes":{"surface finish":"matte","size":"small"}}
            answer = {"same_property":True,"item":1,"label":"surface finish","need":"Finish: matte", "source":"matte finish",
                      "evidence":[evidence(key="surface finish",value="matte")]}
            with patch.object(agent.E6,"plan"),patch.object(ep,"item_listings",return_value=["new"]),patch.object(ep,"json_ask",return_value=answer) as ask:
                ep.plan(3)
            self.assertEqual(ep.remaining(1),["size"])
            self.assertEqual(ask.call_count,1)
            ep.items[0].dropped = ["Finish: matte"]
            self.assertEqual(ep.remaining(1),["surface finish","size"])

    def test_uncertain_or_ungrounded_topic_mapping_preserves_existing_interview(self):
        for agent in AGENTS.values():
            for answer in ({"same_property":False}, {"same_property":True,"item":1,"label":"rim finish","need":"Finish: matte","source":"matte finish","evidence":[evidence(key="invented",value="matte")]}):
                ep = repair_episode(agent)
                ep.request = "The item must have a matte finish."
                ep.items[0].musts = ["Finish: matte"]
                ep.lanes = {1:["rim finish","size"]}
                ep.full["new"] = {"id":"new","attributes":{"rim finish":"matte","size":"small"}}
                with patch.object(agent.E6,"plan"),patch.object(ep,"item_listings",return_value=["new"]),patch.object(ep,"json_ask",return_value=answer):
                    ep.plan(3)
                self.assertEqual(ep.remaining(1),["rim finish","size"])

    def test_exclusions_and_allowed_ranges_do_not_remove_question_capacity(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.items[0].musts = ["Color: any value other than multicolored", "Size: at least 8 cm", "Finish: matte or glossy"]
            ep.lanes = {1: ["color", "size", "finish"]}
            self.assertEqual(ep.remaining(1), ["color", "size", "finish"])

    def test_skip_only_exact_binding_properties_and_keep_distinct_scopes(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.items[0].musts = ["Style: chosen", "Rim finish: matte"]
            ep.lanes = {1: ["style", "shaft style", "finish", "surface finish", "rim finish", "color"]}
            self.assertEqual(ep.remaining(1), ["shaft style", "finish", "surface finish", "color"])

    def test_withdrawn_disclosure_does_not_suppress_question_about_that_property(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.items[0].told = ["Color: red"]
            ep.items[0].dropped = ["Color: red"]
            ep.lanes = {1: ["color", "size"]}
            self.assertEqual(ep.remaining(1), ["color", "size"])

    def test_known_property_filter_is_item_scoped(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.items[0].musts = ["Size: small"]
            ep.items.append(agent.Item("second item", "", 1, ["Width: narrow"]))
            ep.lanes = {1: ["size", "width"], 2: ["size", "width"]}
            self.assertEqual(ep.remaining(1), ["width"])
            self.assertEqual(ep.remaining(2), ["size"])

    def test_ambiguous_multi_topic_answer_does_not_mark_all_topics_answered(self):
        for agent in AGENTS.values():
            ep = repair_episode(agent)
            ep.lanes = {1: ["first", "second", "third"]}
            ep.settle(1, ["first", "second", "third"], False, [], False)
            self.assertEqual(ep.lane_used[1], {"first"})
            self.assertIn("second", ep.remaining(1))
            self.assertIn("third", ep.remaining(1))


if __name__ == "__main__":
    unittest.main()
