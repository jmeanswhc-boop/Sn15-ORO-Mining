"""Trace-driven offline checks for both v3 variants. Real network is forbidden."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AGENTS = {name: load(ROOT / "agents" / name, name[:-3]) for name in ("ICDI_chidori_v3.py", "ICDI_chidori_v3_budget.py")}


def problem():
    tools = [{"type": "function", "function": {"name": name, "parameters": {"type": "object", "properties": props, "required": list(props)}}}
             for name, props in (("message", {"text": {"type": "string"}}), ("buy", {}), ("decline", {}), ("read", {"id": {"type": "string"}}))]
    return {"environment": {"binding": {"session_id": "offline-session"}, "policy_view": {
        "query": "Find one suitable item.", "tools": tools, "max_steps": 20, "finished": False}}}


def episode(agent, data=None):
    ep = agent.Episode(data or problem())
    ep.room = lambda: 100
    ep.clock = lambda: 0
    ep.brief_read = True
    if hasattr(ep, "procedures_valid"):
        ep.procedures_valid = True
    return ep


def response(content="{}", cost=0.000001):
    return SimpleNamespace(data={"usage": {"cost": cost}, "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}, error=None)


class Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.posts = []

    def get(self, *_):
        raise AssertionError("Catalog GET must not be used")

    def post_verbose(self, path, body):
        self.posts.append((path, body))
        if not self.replies:
            raise AssertionError("Unexpected inference attempt")
        return self.replies.pop(0)


class OfflineCase(unittest.TestCase):
    def setUp(self):
        item = patch("requests.sessions.Session.request", side_effect=AssertionError("Real network forbidden"))
        mocked = item.start()
        self.addCleanup(item.stop)
        self.addCleanup(mocked.assert_not_called)


class RequestAdmissionTests(OfflineCase):
    def test_expensive_denial_leaves_cheap_correction_available(self):
        for name, agent in AGENTS.items():
            with self.subTest(agent=name):
                ep = episode(agent)
                client = Client([response()])
                with patch.object(agent, "model_client", return_value=client), patch.object(agent.time, "sleep") as sleep:
                    self.assertIsNone(ep.think([{"role": "user", "content": "x" * 20000}], [agent.HEAVY], 3000))
                    self.assertTrue(ep.request_denied)
                    self.assertFalse(ep.no_credit)
                    self.assertEqual(client.posts, [])
                    self.assertIsNotNone(ep.think([{"role": "user", "content": "Read the new public notice."}], [agent.LIGHT], 700))
                    self.assertFalse(ep.request_denied)
                    self.assertEqual(len(client.posts), 1)
                    sleep.assert_not_called()

    def test_true_exhaustion_sends_no_request(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            ep.inference_budget.actual = ep.inference_budget.limit
            client = Client([])
            with patch.object(agent, "model_client", return_value=client):
                self.assertIsNone(ep.think([{"role": "user", "content": "x"}], [agent.LIGHT], 64))
            self.assertTrue(ep.no_credit)
            self.assertEqual(client.posts, [])

    def test_402_remains_global_and_has_no_retry(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            client = Client([SimpleNamespace(data=None, error={"status": 402, "body": "cap"})])
            with patch.object(agent, "model_client", return_value=client), patch.object(agent.time, "sleep") as sleep:
                self.assertIsNone(ep.think([{"role": "user", "content": "x"}], [agent.LIGHT, agent.HEAVY], 64))
                self.assertIsNone(ep.think([{"role": "user", "content": "retry"}], [agent.LIGHT], 64))
            self.assertTrue(ep.no_credit)
            self.assertEqual(len(client.posts), 1)
            sleep.assert_not_called()

    def test_invalid_pricing_still_halts_unpriced_work(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            client = Client([])
            with patch.object(agent, "BUDGET_PRICE_CONFIG", {"data": []}), patch.object(agent, "model_client", return_value=client):
                self.assertIsNone(ep.think([], [agent.LIGHT], 64))
            self.assertTrue(ep.no_credit)
            self.assertEqual(client.posts, [])

    def test_local_denial_does_not_enter_outage_sleep_loop(self):
        for agent in AGENTS.values():
            ep = episode(agent)
            ep.roles = {}
            ep.convo = [{"role": "user", "content": "Synthetic request"}]
            def denied(*_):
                ep.request_denied = True
                return None
            with patch.object(ep, "ready_to_ask", return_value=False), patch.object(ep, "trim"), patch.object(ep, "offered", return_value=[]), patch.object(ep, "think", side_effect=denied) as think, patch.object(agent.time, "sleep") as sleep:
                ep.turn_loop(20, 1.0)
            self.assertEqual(think.call_count, 1)
            sleep.assert_not_called()

    def test_all_nineteen_recorded_stops_allow_a_compact_request(self):
        run = ROOT / "LV_logs/local-46df07bb11a34426b51f83c581c88575"
        if not run.exists():
            self.skipTest("Recorded verification artifacts unavailable")
        problems = {p["problem_id"]: p for p in map(json.loads, (run / "problems.jsonl").read_text().splitlines())}
        rows = [r for r in map(json.loads, (run / "sandbox/sandbox_output.jsonl").read_text().splitlines()) if any(t.get("kind") == "local_inference_stop" for t in r["dialogue"])]
        self.assertEqual(len(rows), 19)
        for name, agent in AGENTS.items():
            for row in rows:
                with self.subTest(agent=name, session=row["problem_id"]):
                    ep = episode(agent, problems[row["problem_id"]])
                    brief = next(t["brief"] for t in row["dialogue"] if t.get("kind") == "brief")
                    with patch.object(ep, "json_ask", return_value=json.loads(json.dumps(brief))):
                        ep.read_brief()
                    ep.inference_budget.actual = sum(agent.money_units(t["reported_usd"]) for t in row["dialogue"] if t.get("kind") == "inference_budget")
                    ep.calls = ep.inference_budget.attempts = row["inference_total"]
                    client = Client([response()])
                    with patch.object(agent, "model_client", return_value=client):
                        self.assertIsNone(ep.think([{"role": "user", "content": "x" * 20000}], [agent.HEAVY], 3000))
                        self.assertFalse(ep.no_credit)
                        prompt = agent.NOTE_PROMPT + "\nITEMS:\n" + ep.ledger() + "\nNOTICE:\nThe current listing is unavailable."
                        self.assertIsNotNone(ep.think([{"role": "user", "content": prompt}], [agent.LIGHT], 2500))
                    self.assertEqual(len(client.posts), 1)
                    self.assertLessEqual(ep.inference_budget.actual + ep.inference_budget.uncertain, ep.inference_budget.limit)


def notice(**extra):
    result = {"kind": "change", "budget": None, "add": [], "drop": [], "gone": [],
              "sold_out": False, "price_moved": False, "summary": "The requirement changed."}
    result.update(extra)
    return result


def cart_episode(agent):
    ep = episode(agent)
    ep.items = [agent.Item("one item", "", 1, [])]
    ep.roles = {"buy": "buy", "refuse": "decline", "open": "read", "ask": "message"}
    ep.cart_lines = {"sample-item": 1}
    ep.rows = {"sample-item": {"id": "sample-item", "attributes": {"style": "commute"}}}
    ep.order = ["sample-item"]
    return ep


class RecoveryTests(OfflineCase):
    def test_failed_notice_stays_pending_then_applies_once(self):
        line = "The selected listing is no longer available today."
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            answer = notice(gone=["sample-item"], sold_out=True)
            with patch.object(ep, "json_ask", side_effect=[None, answer]) as ask:
                self.assertEqual(ep.read_note([line]), "")
                self.assertNotIn(line, ep.heard)
                self.assertEqual(ep.pending_notices, [line])
                self.assertIn(line, ep.loose({"notice": line}))
                self.assertTrue(ep.read_note([line]))
                version = ep.version
                self.assertEqual(ep.read_note([line]), "")
            self.assertEqual(ask.call_count, 2)
            self.assertIn(line, ep.heard)
            self.assertEqual(ep.pending_notices, [])
            self.assertTrue(ep.sold_out("sample-item"))
            self.assertEqual(ep.version, version)

    def test_malformed_notice_does_not_apply_partial_state(self):
        for agent in AGENTS.values():
            for answer in ({}, {"kind": "unknown"}, notice(add="bad"), notice(gone=True), notice(add=[{"item": 999, "need": "Color: red"}])):
                ep = cart_episode(agent)
                before = (ep.version, ep.ledger(), set(ep.unavail))
                with patch.object(ep, "json_ask", return_value=answer) as ask:
                    for _ in range(4):
                        ep.read_note(["A public notice needs to be interpreted first."])
                self.assertEqual((ep.version, ep.ledger(), set(ep.unavail)), before)
                self.assertTrue(ep.pending_notices)
                self.assertEqual(ask.call_count, 2)

    def test_withdrawn_requirement_is_applied_exactly_once(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            ep.items[0].musts = ["Color: red", "Capacity: at least ten"]
            answer = notice(drop=[{"item": 1, "need": "Color: red"}], add=[{"item": 1, "need": "Color: blue"}])
            with patch.object(ep, "json_ask", side_effect=[None, answer]):
                ep.read_note(["Replace the earlier color requirement with a blue one."])
                self.assertIn("Color: red", ep.live_of(ep.items[0]))
                ep.read_note([])
            self.assertNotIn("Color: red", ep.live_of(ep.items[0]))
            self.assertIn("Color: blue", ep.live_of(ep.items[0]))
            self.assertIn("Capacity: at least ten", ep.live_of(ep.items[0]))

    def test_item_names_and_numeric_strings_resolve_without_arbitrary_clamping(self):
        for agent in AGENTS.values():
            for target in ("one item", "1", 1):
                ep = cart_episode(agent)
                ep.items[0].musts = ["Color: red"]
                answer = notice(drop=[{"item": target, "need": "Color: red"}])
                with patch.object(ep, "json_ask", return_value=answer):
                    self.assertTrue(ep.read_note(["The earlier color requirement is withdrawn by the shopper."]))
                self.assertNotIn("Color: red", ep.live_of(ep.items[0]))
            for target in ("a different item", True, 99):
                ep = cart_episode(agent)
                self.assertFalse(ep.valid_notice(notice(drop=[{"item": target, "need": "Color: red"}])))

    def test_explicit_nonchange_notice_needs_no_terminal_hold(self):
        for agent in AGENTS.values():
            for kind in ("error", "other"):
                ep = episode(agent)
                with patch.object(ep, "json_ask", return_value={"kind": kind}) as ask:
                    ep.read_note(["This display contains a normal informational store message."])
                    self.assertEqual(ep.terminal_recovery_hold(False), "")
                    ep.read_note([])
                self.assertEqual(ask.call_count, 1)
                self.assertFalse(ep.pending_notices)

    def test_pending_notice_blocks_both_terminal_actions_without_environment_calls(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            ep.pending_notices = ["A new public change must still be interpreted."]
            with patch.object(ep, "json_ask", return_value=None), patch.object(agent, "ENV_CLIENT") as env, patch.object(ep, "gate", return_value=(True, "")):
                self.assertIsNone(ep.env("buy", {}))
                self.assertIn("uninterpreted", ep.recovery_blocked)
                self.assertIsNone(ep.env("decline", {}))
            env.post_verbose.assert_not_called()
            self.assertEqual(ep.step, 0)

    def test_recovered_change_requires_replanning_before_terminal_action(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            ep.pending_notices = ["The shopper changed the required color after this pick."]
            with patch.object(ep, "json_ask", return_value=notice(add=[{"item": 1, "need": "Color: blue"}])):
                self.assertIn("Apply it", ep.terminal_recovery_hold(True))
            self.assertFalse(ep.pending_notices)
            self.assertIn("Color: blue", ep.ledger())

    def test_negative_review_survives_denied_or_malformed_escalation(self):
        bad = {"ok": False, "problems": ["The style differs from the requirement."], "better": []}
        for agent in AGENTS.values():
            for escalation in (None, {}, {"ok": "yes"}, {"ok": None}):
                ep = cart_episode(agent)
                with patch.object(ep, "weak_lines", return_value=[]), patch.object(ep, "code_fits", return_value=[]), patch.object(ep, "json_ask", side_effect=[bad, escalation]) as ask:
                    text = ep.cart_check()
                    self.assertIn("style differs", text)
                    self.assertEqual(ep.cart_check(), text)
                self.assertEqual(ask.call_count, 2)
                self.assertFalse(ep.verdicts["sample-item#0"])
                self.assertTrue(ep.terminal_recovery_hold(True))

    def test_unknown_review_is_not_cached_as_pass_or_failure(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            with patch.object(ep, "weak_lines", return_value=[("sample-item", 1, ["Color"])]), patch.object(ep, "code_fits", return_value=[]), patch.object(ep, "json_ask", side_effect=[{}, {"ok": True}]) as ask:
                self.assertEqual(ep.cart_check(), "")
                self.assertNotIn("sample-item#0", ep.checked)
                self.assertNotIn("sample-item#0", ep.verdicts)
                self.assertEqual(ep.cart_check(), "")
            self.assertEqual(ask.call_count, 2)
            self.assertTrue(ep.verdicts["sample-item#0"])

    def test_fresh_evidence_can_overturn_a_negative_review(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            with patch.object(ep, "weak_lines", return_value=[]), patch.object(ep, "code_fits", return_value=[]), patch.object(ep, "json_ask", side_effect=[{"ok": False}, None, {"ok": True}]) as ask:
                self.assertTrue(ep.cart_check())
                ep.rows["sample-item"]["attributes"]["style"] = "requested style"
                self.assertEqual(ep.terminal_recovery_hold(True), "")
            self.assertEqual(ask.call_count, 3)
            self.assertTrue(ep.verdicts["sample-item#0"])

    def test_forced_finish_cannot_buy_a_known_bad_cart_or_blindly_decline(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            ep.verdicts["sample-item#0"] = False
            with patch.object(ep, "reread", return_value=""), patch.object(ep, "gate", return_value=(True, "")), patch.object(agent, "ENV_CLIENT") as env:
                self.assertIn("negative review", ep.finish(True))
                self.assertIn("negative review", ep.finish())
            env.post_verbose.assert_not_called()
            self.assertFalse(ep.finished)

    def test_stockout_blocks_repeat_buy_and_replacement_clears_hold(self):
        for agent in AGENTS.values():
            ep = cart_episode(agent)
            ep.unavail.add("sample-item")
            self.assertIn("unavailable", ep.terminal_recovery_hold(True))
            ep.cart_lines = {"replacement-item": 1}
            self.assertEqual(ep.terminal_recovery_hold(True), "")

    def test_budget_variant_honors_failed_procedure_gate_during_finish(self):
        agent = AGENTS["ICDI_chidori_v3_budget.py"]
        ep = cart_episode(agent)
        with patch.object(ep, "reread", return_value=""), patch.object(ep, "gate", return_value=(False, "Approval is unresolved.")), patch.object(agent, "ENV_CLIENT") as env:
            self.assertIn("Approval is unresolved", ep.finish())
        env.post_verbose.assert_not_called()


def recorded_inference():
    directory = ROOT / "LV_logs/local-46df07bb11a34426b51f83c581c88575/sandbox"
    if directory.exists():
        for path in sorted(directory.glob("request_log_*.jsonl")):
            for row in map(json.loads, path.read_text().splitlines()):
                if row.get("kind") == "summary" and row.get("path") == "/inference/chat/completions":
                    yield row


class PricingTests(OfflineCase):
    def budget(self, agent, limit="0.035"):
        budget = agent.InferenceBudget()
        budget.load_prices(agent.BUDGET_PRICE_CONFIG)
        budget.limit = agent.money_units(limit)
        return budget

    def body_at_estimate(self, model, tokens, cap=1000):
        body = {"model": model, "messages": [{"role": "user", "content": ""}], "max_tokens": cap}
        overhead = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 512 + 64
        body["messages"][0]["content"] = "x" * (tokens - overhead)
        return body

    def test_light_uses_base_rates_below_threshold_and_high_rates_at_boundary(self):
        for agent in AGENTS.values():
            budget = self.budget(agent, "10")
            for prompt, input_rate, output_rate in ((99999, 125, 500), (100000, 625, 2500), (100001, 625, 2500)):
                quote, reason = budget.quote(self.body_at_estimate(agent.LIGHT, prompt))
                self.assertEqual(reason, "")
                self.assertEqual(quote["prompt_estimate"], prompt)
                self.assertEqual(quote["units"], ((prompt * input_rate + 1000 * output_rate) * 5 + 3) // 4)

    def test_spare_tier_boundary_and_heavy_prices_are_preserved(self):
        for agent in AGENTS.values():
            budget = self.budget(agent, "10")
            for model, prompt, input_rate, output_rate in ((agent.SPARE, 271999, 125, 500), (agent.SPARE, 272000, 250, 750),
                                                         (agent.HEAVY, 99999, 2500, 10000), (agent.HEAVY, 100000, 2500, 10000)):
                quote, _ = budget.quote(self.body_at_estimate(model, prompt))
                self.assertEqual(quote["units"], ((prompt * input_rate + 1000 * output_rate) * 5 + 3) // 4)

    def test_short_prompts_keep_the_requested_reply_capacity(self):
        for agent in AGENTS.values():
            for model in (agent.LIGHT, agent.SPARE):
                budget = self.budget(agent)
                body = self.body_at_estimate(model, 32000, 4000)
                quote, reason = budget.quote(body)
                self.assertEqual(reason, "")
                self.assertEqual(quote["cap"], 4000)
                self.assertLess(quote["units"], agent.money_units("0.01"))

    def test_unknown_tier_layout_fails_closed_and_no_tier_keeps_conservative_fallback(self):
        for agent in AGENTS.values():
            for tiers in ([], {}, [None], [{"min_prompt_tokens": True, "pricing": {}}], [{"min_prompt_tokens": 100, "pricing": {"prompt": "0", "completion": "0"}}]):
                config = json.loads(json.dumps(agent.BUDGET_PRICE_CONFIG))
                config["data"][0]["context_tiers"] = tiers
                budget = agent.InferenceBudget()
                budget.load_prices(config)
                self.assertNotIn(agent.LIGHT, budget.prices)
            config = json.loads(json.dumps(agent.BUDGET_PRICE_CONFIG))
            del config["data"][0]["context_tiers"]
            budget = agent.InferenceBudget()
            budget.load_prices(config)
            quote, _ = budget.quote(self.body_at_estimate(agent.LIGHT, 1000))
            self.assertEqual(quote["units"], ((1000 * 625 + 1000 * 2500) * 5 + 3) // 4)

    def test_nonmonotone_tiers_use_maximum_of_all_possible_applicable_rates(self):
        for agent in AGENTS.values():
            config = json.loads(json.dumps(agent.BUDGET_PRICE_CONFIG))
            row = config["data"][0]
            row["context_tiers"][0]["pricing"] = row["pricing"]
            row["context_tiers"][1]["pricing"] = {"prompt": "0.0000001", "completion": "0.0000005"}
            budget = agent.InferenceBudget()
            budget.limit = agent.money_units("10")
            budget.load_prices(config)
            quote, _ = budget.quote(self.body_at_estimate(agent.LIGHT, 100000))
            self.assertEqual(quote["units"], ((100000 * 625 + 1000 * 2500) * 5 + 3) // 4)

    def test_every_recorded_bill_and_token_count_fits_the_new_quote(self):
        rows = list(recorded_inference())
        if not rows:
            self.skipTest("Recorded verification unavailable")
        self.assertEqual(len(rows), 405)
        for name, agent in AGENTS.items():
            for index, row in enumerate(rows):
                with self.subTest(agent=name, index=index):
                    budget = self.budget(agent)
                    quote, reason = budget.quote(row["json_data"])
                    self.assertIsNotNone(quote, reason)
                    usage = row["response"]["usage"]
                    self.assertGreaterEqual(quote["units"], agent.money_units(usage["cost"]))
                    self.assertGreaterEqual(quote["prompt_estimate"], usage["prompt_tokens"])
                    self.assertGreaterEqual(quote["cap"], usage["completion_tokens"])
                    budget.pending += quote["units"]
                    budget.settle(quote, row["response"])
                    self.assertFalse(budget.breached)


if __name__ == "__main__":
    unittest.main()
