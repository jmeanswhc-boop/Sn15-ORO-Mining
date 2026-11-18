"""Fictional offline contracts exposing failures in the unchanged agent.

These are expected to fail until the corresponding controller fixes exist.
They are diagnostics, not a claimed performance improvement.
"""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents/tests"))
from test_tosky_l17 import load, tool

SOURCE = ROOT / "agents/Tosky_l17.py"


def example():
    ns = load(SOURCE)
    agent = ns["Cairn"]({"environment": {"binding": {"session_id": "offline-contract"},
                         "policy_view": {"query": "Buy a fictional sample with a matte finish.",
                                          "max_steps": 40, "tools": [tool("Checkout", "Buy the cart")]}}})
    agent.roles = {"buy": "Checkout"}
    agent.items = [ns["Item"]("fictional sample", "Samples", 1, ["finish: matte"])]
    agent.brief = {"pick": "any", "buy_args": {}}
    agent.cart_lines = {"fictional-listing": 1}
    agent.cart_ids = lambda: list(agent.cart_lines)
    return ns, agent


class SafetyContracts(unittest.TestCase):
    def test_an_unchanged_rejected_cart_remains_blocked_on_repeat_review(self):
        ns, agent = example()
        agent.judges = lambda limit: [ns["LIGHT"]]
        agent.ask_json = lambda *args: {"ok": False, "problems": ["The finish is glossy."], "better": []}
        agent.record_of = lambda ident: "A fictional glossy sample."
        first = agent.check_cart(["fictional-listing"])
        second = agent.check_cart(["fictional-listing"])
        self.assertTrue(first)
        self.assertFalse(agent.verdicts[agent.check_key(["fictional-listing"])])
        self.assertTrue(second, "A cached rejection became permission to continue")

    def test_emergency_finish_cannot_buy_a_known_rejected_cart(self):
        ns, agent = example()
        agent.verdicts[agent.check_key(["fictional-listing"])]=False
        calls = []
        agent.gate = lambda stage: (True, "")
        agent.salvage = lambda reason: False
        agent.observe = lambda *args: ""

        def act(name, args):
            calls.append(name)
            agent.step += 1
            return {"observation": {}, "finished": False, "step": agent.step}

        agent.act = act
        agent.finish(again=True)
        self.assertEqual(calls, [], "The emergency path ignored an existing rejection")

    def test_stock_rejection_requires_cart_repair_before_another_checkout(self):
        ns, agent = example()
        calls = []
        agent.gate = lambda stage: (True, "")
        agent.salvage = lambda reason: False

        def observe(name, observation):
            agent.unavailable.add("fictional-listing")
            return "The cart's only listing is unavailable."

        def act(name, args):
            calls.append(name)
            agent.step += 1
            return {"observation": {"refusal": ["The selected listing has no stock."]},
                    "finished": False, "step": agent.step}

        agent.observe = observe
        agent.act = act
        agent.finish(again=True)
        self.assertTrue(agent.sold_out("fictional-listing"))
        agent.finish(again=True)
        self.assertEqual(calls, ["Checkout"], "The same sold-out cart was checked out twice")


def main():
    global SOURCE
    results = {}
    versions = {"before": ROOT / "logs/research/tosky-l17-run-9f2ac835/baseline-source.txt",
                "after": ROOT / "agents/Tosky_l17.py"}
    for label, path in versions.items():
        SOURCE = path
        with (DEST / ("diagnostic-contracts-" + label + ".txt")).open("w") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(SafetyContracts))
        results[label] = {"run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                          "failed_cases": [test.id() for test, _ in result.failures]}
    (DEST / "diagnostic-contracts.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
