"""Local source/contract checks; these do not certify private server-side policy."""
import ast
import hashlib
from pathlib import Path
import re
import unittest
from test_chidori_procedures import agent


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "agents/ICDI_chidori_v1.py"
BASE = ROOT / "logs/research/champion-lineage-2026-11-12/source-06af2157-d650-7d40-8000-519949f07ca0.py"


class PreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE.read_text()
        cls.tree = ast.parse(cls.source)

    def test_original_archive_hash_is_unchanged(self):
        self.assertEqual(hashlib.sha256(BASE.read_bytes()).hexdigest(),
                         "b32f88b0ef42100f10d9ad29e3019a54dd79dfaefe8ded2b72a88772f9247bf2")
        self.assertNotEqual(SOURCE.read_bytes(), BASE.read_bytes())

    def test_file_size_and_single_synchronous_entry_point(self):
        self.assertLess(SOURCE.stat().st_size, 1000000)
        entries = [n for n in self.tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "agent_main"]
        self.assertEqual(len(entries), 1)
        self.assertIsInstance(entries[0], ast.FunctionDef)

    def test_imports_use_only_supported_library_and_proxy_modules(self):
        allowed = {"json", "random", "re", "time", "typing", "decimal", "src.agent.proxy_client"}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                self.assertTrue(all(n.name in allowed for n in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertIn(node.module, allowed)

    def test_no_dynamic_execution_or_filesystem_calls(self):
        forbidden = {"eval", "exec", "__import__", "open", "write_text", "write_bytes", "unlink", "symlink_to", "hardlink_to", "chmod", "rmdir"}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                self.assertNotIn(name, forbidden)

    def test_only_supported_literal_proxy_routes(self):
        routes = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("get", "post", "post_verbose") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("/"):
                    routes.add(first.value)
        self.assertEqual(routes, {"/environment/call", "/inference/models", "/inference/chat/completions"})

    def test_no_concrete_benchmark_ids_or_phase_configuration(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertIsNone(re.search(r"\b(?:t-[0-9a-f]{10}|p-[0-9a-f]{12})\b", node.value))
        self.assertNotIn("getenv(", self.source)
        self.assertNotIn('binding["phase"]', self.source)
        self.assertNotIn('binding["race_id"]', self.source)

    def test_effective_methods_use_the_new_controls(self):
        expected = {"env": "ProcedureControls", "gate": "ProcedureControls", "read_brief": "ProcedureControls",
                    "opening": "ProcedureControls", "reserve": "ProcedureControls", "think": "BudgetControls",
                    "budget_post": "BudgetControls", "one_call": "E1", "finish": "G6B", "dispatch": "G6B"}
        for method, owner in expected.items():
            self.assertEqual(getattr(agent.Episode, method).__qualname__.split(".")[0], owner)


if __name__ == "__main__":
    unittest.main()
