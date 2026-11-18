"""Contract, provenance, originality, and rule-oriented tests for ICDI."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


AGENT_PATH = Path(__file__).parents[1] / "ICDI.py"
PROVENANCE_PATH = Path(__file__).parents[1] / "ICDI.provenance.json"
CATALOG_PATH = Path(__file__).parents[2] / "champions" / "catalog.csv"
SPEC = importlib.util.spec_from_file_location("icdi_agent", AGENT_PATH)
assert SPEC and SPEC.loader
icdi = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = icdi
SPEC.loader.exec_module(icdi)


def _tool(name: str, description: str = "", required: list[str] | None = None) -> dict:
    properties = {
        key: {"type": "string"}
        for key in required or []
    }
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _tool("inspect", "Inspect the store."),
    _tool("message", "Send one message.", ["text"]),
    _tool("buy", "Finish by ordering."),
    _tool("refuse", "Finish without ordering.", ["reason"]),
]


def _problem(max_steps: int = 8) -> dict:
    return {
        "environment": {
            "binding": {"session_id": "session-1"},
            "policy_view": {
                "query": "Find one suitable item.",
                "opening": None,
                "tools": TOOLS,
                "max_steps": max_steps,
                "finished": False,
            },
        }
    }


class ICDITests(unittest.TestCase):
    def test_source_identity_is_consistently_icdi(self) -> None:
        source = AGENT_PATH.read_text()
        self.assertTrue(source.startswith('"""ICDI:'))
        self.assertIn("class ICDI:", source)
        self.assertIn("You are ICDI", source)
        for inherited_identity in ("Cairn", "Kyo", "KYO", "YesYesYes", "ShoppingConductor"):
            self.assertNotIn(inherited_identity, source)

    def test_manifest_hash_matches_source(self) -> None:
        provenance = json.loads(PROVENANCE_PATH.read_text())
        digest = hashlib.sha256(AGENT_PATH.read_bytes()).hexdigest()
        self.assertEqual(digest, provenance["icdi_sha256"])

    def test_design_sources_are_downloaded_race_winners(self) -> None:
        provenance = json.loads(PROVENANCE_PATH.read_text())
        with CATALOG_PATH.open() as catalog_file:
            catalog = {row["agent_version_id"]: row for row in csv.DictReader(catalog_file)}
        sources = [provenance["design_foundation"], *provenance["post_kyo_champion_components"]]
        for source in sources:
            archived = catalog[source["agent_version_id"]]
            self.assertEqual(archived["download_status"], "downloaded")
            self.assertGreater(int(archived["race_win_count"]), 0)
            self.assertEqual(archived["archive_sha256"], source["archive_sha256"])

    def test_observation_shape_restores_supported_transports(self) -> None:
        restore = icdi.ObservationShape.restore
        self.assertEqual(restore('{"id":"p1","price":4}'), {"id": "p1", "price": 4})
        self.assertEqual(
            restore({"id": ["p1", "p2"], "price": [4, 7]}),
            [{"id": "p1", "price": 4}, {"id": "p2", "price": 7}],
        )
        self.assertEqual(
            restore({"prod1": {"price": 4}, "prod2": {"price": 7}}),
            [{"id": "prod1", "price": 4}, {"id": "prod2", "price": 7}],
        )
        self.assertEqual(
            restore("id=p1; price=4; stock=true"),
            {"id": "p1", "price": "4", "stock": "true"},
        )

    def test_schema_guard_rejects_missing_and_extra_arguments(self) -> None:
        schema = TOOLS[1]["function"]["parameters"]
        self.assertEqual(icdi.SchemaGuard.check({"text": "hello"}, schema), [])
        self.assertIn("args.text is required", icdi.SchemaGuard.check({}, schema))
        self.assertIn("args.extra is not accepted", icdi.SchemaGuard.check({"text": "hi", "extra": 1}, schema))

    def test_environment_port_sends_exactly_one_action(self) -> None:
        episode = icdi.Episode({"session_id": "session-1"}, "request", 4)
        port = icdi.EnvironmentPort(episode)
        sent: list[dict] = []

        def post_verbose(path, body):
            sent.append(body)
            return SimpleNamespace(
                data={"step": 1, "finished": False, "observation": '{"id":"p1","price":4}'},
                error=None,
            )

        with patch.object(port.client, "post_verbose", post_verbose):
            result = port.call("inspect", {})

        self.assertEqual(sent, [{"session_id": "session-1", "step": 1, "action": {"name": "inspect", "args": {}}}])
        self.assertNotIn("actions", sent[0])
        self.assertEqual(result["observation"], {"id": "p1", "price": 4})
        self.assertEqual(episode.step, 1)

    def test_final_two_steps_expose_only_terminal_tools(self) -> None:
        episode = icdi.Episode({}, "request", max_steps=4, step=2)
        contract = icdi.ToolContract(TOOLS)
        contract.bind_roles({"search": "inspect", "buy": "buy", "refuse": "refuse"})
        policy = icdi.StepPolicy(episode, contract)
        exposed = {tool["function"]["name"] for tool in policy.available_tools()}
        self.assertEqual(exposed, {"buy", "refuse"})
        self.assertEqual(policy.permits("buy"), (True, ""))
        self.assertFalse(policy.permits("inspect")[0])

    def test_task_revision_invalidates_terminal_audit(self) -> None:
        agent = icdi.ICDI(_problem())
        agent.contract.bind_roles({"message": "message", "buy": "buy", "refuse": "refuse"})
        agent.mission.replace({"requirements": ["old"]})
        agent.auditor._revision = agent.mission.revision
        agent.auditor._record_count = 2
        agent.auditor._decision = {"allow": True}
        change = {"changed": True, "plan_update": {"requirements": ["new"]}, "summary": "changed"}

        with patch.object(agent.reader, "revision", return_value=change):
            agent._refresh_after_message(("message", {"text": "question"}), {"observation": "new answer"})

        self.assertEqual(agent.mission.plan["requirements"], ["new"])
        self.assertIsNone(agent.auditor._decision)
        self.assertEqual(agent.auditor._revision, -1)

    def test_terminal_audit_cache_is_specific_to_the_proposed_action(self) -> None:
        gateway = icdi.ModelGateway()
        auditor = icdi.TerminalAuditor(gateway)
        mission = icdi.Mission({"requirements": ["request"]}, revision=1)
        trail = icdi.EvidenceTrail()
        decisions = [
            {"allow": False, "problems": ["cart incomplete"], "instruction": "reconsider"},
            {"allow": True, "problems": [], "instruction": ""},
        ]
        with patch.object(gateway, "json_object", side_effect=decisions) as review:
            buy = auditor.review("request", mission, trail, ("buy", {}))
            refuse = auditor.review("request", mission, trail, ("refuse", {"reason": "unavailable"}))
        self.assertFalse(buy["allow"])
        self.assertTrue(refuse["allow"])
        self.assertEqual(review.call_count, 2)

    def test_rate_limit_waits_before_sibling_model(self) -> None:
        gateway = icdi.ModelGateway()
        sent: list[str] = []

        def post_verbose(path, body):
            sent.append(body["model"])
            if len(sent) == 1:
                return SimpleNamespace(data=None, error={"status": 429})
            return SimpleNamespace(data={"choices": [{"message": {"content": "{}"}}]}, error=None)

        with patch.object(gateway.client, "post_verbose", post_verbose), patch.object(icdi.time, "sleep") as pause:
            result = gateway.complete([{"role": "user", "content": "read"}], icdi.MODELS.planner)

        self.assertEqual(result, {"content": "{}"})
        self.assertEqual(sent[:2], list(icdi.MODELS.planner[:2]))
        pause.assert_called_once_with(1.0)

    def test_malformed_planner_output_uses_a_sibling_model(self) -> None:
        gateway = icdi.ModelGateway()
        replies = iter(
            [
                SimpleNamespace(data={"choices": [{"message": {"content": "not json"}}]}, error=None),
                SimpleNamespace(data={"choices": [{"message": {"content": '{"ok":true}'}}]}, error=None),
            ]
        )
        with patch.object(gateway.client, "post_verbose", side_effect=lambda path, body: next(replies)):
            result = gateway.json_object("return json", {"request": "x"}, icdi.MODELS.planner)
        self.assertEqual(result, {"ok": True})

    def test_actor_output_without_tool_call_uses_a_sibling_model(self) -> None:
        gateway = icdi.ModelGateway()
        replies = iter(
            [
                SimpleNamespace(data={"choices": [{"message": {"content": "plain text"}}]}, error=None),
                SimpleNamespace(
                    data={
                        "choices": [
                            {
                                "message": {
                                    "content": "",
                                    "tool_calls": [
                                        {"id": "c1", "function": {"name": "inspect", "arguments": "{}"}}
                                    ],
                                }
                            }
                        ]
                    },
                    error=None,
                ),
            ]
        )
        with patch.object(gateway.client, "post_verbose", side_effect=lambda path, body: next(replies)):
            result = gateway.complete(
                [{"role": "user", "content": "act"}],
                icdi.MODELS.actor,
                tools=[TOOLS[0]],
            )
        self.assertEqual(result["tool_calls"][0]["function"]["name"], "inspect")

    def test_full_controller_uses_public_contract_until_finished(self) -> None:
        agent = icdi.ICDI(_problem(max_steps=3))
        plan = {
            "objective": "request",
            "requirements": ["request"],
            "selection_rule": "any",
            "questions": [],
            "roles": {"search": "inspect", "buy": "buy", "refuse": "refuse"},
            "completion_evidence": [],
        }
        choices = iter([("inspect", {}), ("buy", {})])

        def choose(*args, **kwargs):
            return next(choices)

        def world_call(name, arguments):
            agent.episode.step += 1
            agent.episode.finished = name == "buy"
            return {
                "step": agent.episode.step,
                "finished": agent.episode.finished,
                "observation": {"ok": True, "tool": name},
            }

        with (
            patch.object(agent.reader, "initial", return_value=plan),
            patch.object(agent.chooser, "choose", side_effect=choose),
            patch.object(agent.auditor, "review", return_value={"allow": True, "problems": [], "instruction": ""}),
            patch.object(agent.world, "call", side_effect=world_call),
        ):
            dialogue = agent.run()

        self.assertTrue(agent.episode.finished)
        self.assertEqual([item["tool_call"]["name"] for item in dialogue], ["inspect", "buy"])
        self.assertEqual(agent.episode.step, 2)

    def test_repeated_environment_failure_stops_deterministically(self) -> None:
        agent = icdi.ICDI(_problem(max_steps=3))
        plan = {
            "objective": "request",
            "requirements": ["request"],
            "selection_rule": "any",
            "questions": [],
            "roles": {"buy": "buy", "refuse": "refuse"},
            "completion_evidence": [],
        }
        with (
            patch.object(agent.reader, "initial", return_value=plan),
            patch.object(agent.chooser, "choose", return_value=("refuse", {"reason": "unavailable"})),
            patch.object(agent.auditor, "review", return_value={"allow": True, "problems": [], "instruction": ""}),
            patch.object(agent.world, "call", return_value=None) as environment,
        ):
            dialogue = agent.run()
        self.assertEqual(environment.call_count, 3)
        self.assertIn("repeated environment failures", dialogue[-1]["content"])

    def test_rule_oriented_static_properties(self) -> None:
        source = AGENT_PATH.read_text()
        tree = ast.parse(source)
        prohibited_imports = {
            "os", "posix", "nt", "subprocess", "commands", "socket", "http", "urllib",
            "requests", "ctypes", "cffi", "pickle", "shutil", "base64", "binascii", "codecs", "zlib",
        }
        imported: set[str] = set()
        calls: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)
        self.assertFalse(imported & prohibited_imports)
        self.assertFalse(calls & {"eval", "exec", "__import__", "open"})
        self.assertNotIn("time.monotonic", source)
        self.assertNotIn("time.time", source)
        self.assertNotIn('"actions"', source)
        self.assertLess(len(source.encode("utf-8")), 1_000_000)


if __name__ == "__main__":
    unittest.main()
