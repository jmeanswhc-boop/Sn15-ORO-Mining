"""Budget-only correction checks, including the actual ProxyClient transport."""
import ast
import hashlib
from pathlib import Path
from unittest.mock import patch
import unittest

import test_chidori_v2_transport as transport


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "agents/ICDI_chidori_v1_budget.py"
CANDIDATE = ROOT / "agents/ICDI_chidori_v2_budget.py"
FULL_V2 = transport.v2
budget_only = transport.budget_tests.load(CANDIDATE, "chidori_v2_budget_candidate")


class BudgetOnlyTests(transport.V2BudgetTests):
    def setUp(self):
        item = patch.object(transport, "v2", budget_only)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()

    def test_original_methods_unchanged_outside_central_inference_hook(self):
        before = ast.parse(BASE.read_text())
        after = ast.parse(CANDIDATE.read_text())
        old_classes = {n.name: n for n in before.body if isinstance(n, ast.ClassDef)}
        new_classes = {n.name: n for n in after.body if isinstance(n, ast.ClassDef)}
        self.assertEqual(set(old_classes), set(new_classes))
        for name, old in old_classes.items():
            for method in [n for n in old.body if isinstance(n, ast.FunctionDef)]:
                if name == "BudgetControls" and method.name == "budget_post":
                    continue
                new = next(n for n in new_classes[name].body if isinstance(n, ast.FunctionDef) and n.name == method.name)
                self.assertEqual(ast.dump(method), ast.dump(new), (name, method.name))

    def test_procedure_strategy_and_prompts_remain_budget_only(self):
        original = transport.budget_tests.load(BASE, "chidori_budget_scope_original")
        self.assertNotIn("ProcedureControls", vars(budget_only))
        self.assertEqual([c.__name__ for c in budget_only.Episode.__mro__], [c.__name__ for c in original.Episode.__mro__])
        self.assertEqual(budget_only.BRIEF_PROMPT, original.BRIEF_PROMPT)
        self.assertEqual(budget_only.GATE_PROMPT, original.GATE_PROMPT)
        self.assertEqual(budget_only.Episode.finish.__qualname__, "G6B.finish")

    def test_reviewed_pricing_and_existing_budget_settings_match_full_v2(self):
        self.assertEqual(budget_only.BUDGET_PRICE_CONFIG, FULL_V2.BUDGET_PRICE_CONFIG)
        for name in ("AGENT_USD_LIMIT", "PROCEDURE_USD_RESERVE", "MIN_REPLY_TOKENS"):
            self.assertEqual(getattr(budget_only, name), getattr(FULL_V2, name))

    def test_local_source_preflight(self):
        tree = ast.parse(CANDIDATE.read_text())
        self.assertLess(CANDIDATE.stat().st_size, 1000000)
        self.assertEqual(len([n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "agent_main"]), 1)
        allowed = {"json", "random", "re", "time", "typing", "decimal", "src.agent.proxy_client"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(alias.name in allowed for alias in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)
            elif isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                self.assertNotIn(name, {"eval", "exec", "__import__", "open", "write_text", "write_bytes", "unlink", "chmod"})


class BudgetOnlyTransportTests(transport.RealClientTransportTests):
    def setUp(self):
        for name, value in (("v2", budget_only), ("V1_PATH", BASE), ("V2_PATH", CANDIDATE)):
            item = patch.object(transport, name, value)
            item.start()
            self.addCleanup(item.stop)
        super().setUp()

    def test_v1_source_is_preserved(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(), "7a8e0d8da12870e9b5af9f59ccae03166cec723d70f9ae31f4e10795871e533c")


if __name__ == "__main__":
    unittest.main()
