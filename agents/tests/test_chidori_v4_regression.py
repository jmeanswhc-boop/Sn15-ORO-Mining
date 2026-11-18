"""Financial/transport regression checks for v4; no live inference."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import test_chidori_v2_transport as transport
import test_chidori_v3_regression as previous
import test_chidori_procedures as procedures
from test_chidori_v4 import AGENTS, FULL, ROOT, repair_episode, tool_response


class V4FinancialTests(previous.V3FinancialTests):
    candidate = FULL

    def test_original_methods_unchanged_outside_central_inference_hook(self):
        trees = [ast.parse((ROOT/"agents"/name).read_text()) for name in AGENTS]
        for owner in ("ScopedControls", "BudgetControls", "InferenceBudget"):
            nodes = [next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==owner) for tree in trees]
            self.assertEqual(ast.dump(nodes[0]),ast.dump(nodes[1]))


class V4BudgetFinancialTests(V4FinancialTests):
    candidate = AGENTS["ICDI_chidori_v4_budget.py"]


class V4TransportTests(transport.RealClientTransportTests):
    candidate = FULL
    filename = "ICDI_chidori_v4.py"

    def setUp(self):
        for key,value in (("v2",self.candidate),("V2_PATH",ROOT/"agents"/self.filename)):
            item = patch.object(transport,key,value)
            item.start()
            self.addCleanup(item.stop)
        super().setUp()

    def test_scoped_request_uses_real_authenticated_client_and_no_catalog(self):
        for role in ("procedure","repair","coverage","dispute","question_scope"):
            ep = self.candidate.Episode(transport.budget_tests.problem())
            with patch.object(self.candidate,"model_client",return_value=self.client):
                self.assertIsNotNone(ep.with_inference_role(role,ep.json_ask,"Public decision.",[self.candidate.LIGHT],2048))
            self.assertTrue(self.posts[-1]["authorized"])
            self.assertEqual(ep.calls,1)
            self.assertEqual(ep.inference_budget.actual,1000)
            self.assertIn("reasoning",self.posts[-1]["body"])
        self.assertEqual(self.gets,[])

    def run_repair_transport(self, empty=False, fallback=False):
        agent = self.candidate
        ep = repair_episode(agent)
        ep.coverage_done = fallback
        ep.items[0].musts = ["Feature: wanted"]
        ep.rows["new"]["attributes"] = {"feature":"wanted","another feature":"unchanged"}
        ep.full["new"] = dict(ep.rows["new"])
        ep.offer_seen["new"] = 0
        ep.cart_lines = {} if empty else {"new":1} if fallback else {"old":1,"new":1}
        ep.unavail.add("old")
        ep.order_rejected = empty
        events = []
        models = []
        def http(url,**kwargs):
            body = kwargs["json"]
            if url.endswith("/environment/call"):
                events.append(body)
                final = body["action"]["name"]=="buy"
                return self.response(200,{"step":body["step"],"finished":final,"observation":{"accepted":True}})
            self.assertEqual(kwargs["headers"]["Authorization"],"Bearer offline-test-token")
            models.append(body)
            if body.get("tools"):
                if not ep.cart_ids():
                    result = tool_response("basket_change",{"mode":"include","item":"new","quantity":1}).data
                else:
                    result = tool_response("buy",{}).data
            else:
                content = body["messages"][0]["content"]
                if content.startswith("Audit the request"):
                    result = {"usage":{"cost":0.000001},"choices":[{"message":{"content":'{"complete":true,"missing":[]}'},"finish_reason":"stop"}]}
                elif content.startswith(agent.CHECK_PROMPT):
                    result = {"usage":{"cost":0.000001},"choices":[{"message":{"content":'{"ok":true,"problems":[],"better":[]}'},"finish_reason":"stop"}]}
                else:
                    raise AssertionError("Unexpected model role")
            return self.response(200,result)
        with patch("src.agent.proxy_client.requests.post",side_effect=http),patch.object(agent,"model_client",return_value=self.client),patch.object(agent,"ENV_CLIENT",self.client),patch.object(agent.time,"sleep"):
            if fallback:
                ep.finish()
            else:
                ep.recover("Complete the known pending repair.")
        self.assertTrue(ep.finished,list(ep.trace))
        self.assertEqual(ep.cart_ids(),["new"])
        self.assertEqual(ep.calls,len(models))
        self.assertEqual([e["step"] for e in events],list(range(1,len(events)+1)))
        self.assertEqual(events[-1]["action"]["name"],"buy")
        self.assertLessEqual(ep.inference_budget.actual+ep.inference_budget.uncertain,ep.inference_budget.limit)
        return events,models

    def test_stockout_cleanup_source_audit_review_and_order_through_real_transport(self):
        events,models = self.run_repair_transport()
        self.assertEqual([event["action"]["name"] for event in events],["basket_change","buy"])
        self.assertTrue(any(body.get("tools") for body in models))

    def test_empty_rejected_basket_is_rebuilt_before_order_through_real_transport(self):
        events,_ = self.run_repair_transport(empty=True)
        self.assertEqual([event["action"]["name"] for event in events],["basket_change","buy"])

    def test_fallback_checks_new_cart_before_order_through_real_transport(self):
        events,models = self.run_repair_transport(fallback=True)
        self.assertEqual(len(events),1)
        self.assertTrue(any(isinstance(body["messages"][0]["content"],str) and body["messages"][0]["content"].startswith(self.candidate.CHECK_PROMPT) for body in models))


class V4BudgetTransportTests(V4TransportTests):
    candidate = AGENTS["ICDI_chidori_v4_budget.py"]
    filename = "ICDI_chidori_v4_budget.py"


class V4FullProcedureWireTests(unittest.TestCase):
    setUp = transport.RealClientTransportTests.setUp
    response = transport.RealClientTransportTests.response
    http_get = transport.RealClientTransportTests.http_get
    http_post = transport.RealClientTransportTests.http_post

    def test_procedure_executes_message_then_verifies_actual_evidence_before_order(self):
        with patch.object(procedures,"agent",FULL):
            ep = procedures.episode()
            procedures.parse(ep,[procedures.duty()])
        ep.coverage_done = True
        ep.cart_lines = {"public-item":1}
        ep.rows = {"public-item":{"id":"public-item"}}
        ep.order = ["public-item"]
        events,models = [],[]
        def http(url,**kwargs):
            body = kwargs["json"]
            if url.endswith("/environment/call"):
                events.append(body)
                message = body["action"]["name"]=="message"
                return self.response(200,{"step":body["step"],"finished":not message,"observation":"Approved." if message else {"ok":True}})
            self.assertEqual(kwargs["headers"]["Authorization"],"Bearer offline-test-token")
            self.assertEqual(body["reasoning"],{"enabled":False})
            self.assertGreaterEqual(body["max_tokens"],2048)
            models.append(body)
            answer = procedures.decision(["d1"],[1]) if events else procedures.decision(["d1"],done=False,call={
                "duty_id":"d1","tool":"message","args":{"text":"May I order this basket?"}})
            return self.response(200,{"usage":{"cost":0.000001},"choices":[{"message":{"content":json.dumps(answer)},"finish_reason":"stop"}]})
        with patch("src.agent.proxy_client.requests.post",side_effect=http),patch.object(FULL,"model_client",return_value=self.client),patch.object(FULL,"ENV_CLIENT",self.client),patch.object(FULL.time,"sleep"):
            self.assertTrue(ep.gate("order")[0])
            self.assertTrue(ep.env("buy",{})["finished"])
        self.assertEqual(len(models),2)
        self.assertEqual([e["action"]["name"] for e in events],["message","buy"])
        self.assertEqual(ep.duties[0]["evidence"],[1])
        self.assertEqual(ep.msg_used,1)
        self.assertEqual(ep.inference_budget.actual,2000)


class V4NinetyTaskTests(unittest.TestCase):
    def test_scoped_roles_keep_ninety_isolated_ledgers_inside_agent_allocation(self):
        for agent in AGENTS.values():
            thread = threading.local()
            class Mixed:
                def __init__(self,number):
                    self.number,self.count = number,0
                def post_verbose(self,path,body):
                    self.count += 1
                    if self.number%4==0:
                        return SimpleNamespace(data=None,error={"status":500,"body":"temporary"})
                    cost = None if self.number%4==1 else 0.0001
                    content = None if self.number%4==2 else "{}"
                    return SimpleNamespace(data={"usage":{"cost":cost},"choices":[{"message":{"content":content},"finish_reason":"stop"}]},error=None)
            def run(number):
                ep = agent.Episode(transport.budget_tests.problem())
                ep.room = lambda:100
                thread.client = Mixed(number)
                for index in range(20):
                    if ep.no_credit or ep.calls>=agent.CALL_CAP:
                        break
                    ep.with_inference_role(("coverage","repair","dispute","procedure")[index%4],ep.think,
                        [{"role":"user","content":"Synthetic scoped billing workload."}],[agent.LIGHT,agent.SPARE],2048)
                budget = ep.inference_budget
                self.assertEqual(ep.calls,thread.client.count)
                self.assertEqual(budget.pending,0)
                self.assertLessEqual(budget.actual+budget.uncertain,budget.limit)
                return budget.actual+budget.uncertain
            with patch("requests.sessions.Session.request",side_effect=AssertionError("Real network forbidden")) as blocked,patch.object(agent,"model_client",side_effect=lambda _:thread.client),patch.object(agent.time,"sleep"):
                with ThreadPoolExecutor(max_workers=7) as pool:
                    total = sum(pool.map(run,range(90)))
                blocked.assert_not_called()
            self.assertLessEqual(total,3150000000)


class V4IsolatedProcedureBoundaryTests(transport.V2BoundaryTests):
    """Preserve procedure boundaries independently of new product-review guards."""
    def setUp(self):
        item = patch.object(transport,"v2",FULL)
        item.start()
        self.addCleanup(item.stop)
        super().setUp()
        self.ep.coverage_done = True
        for name,value in (("terminal_recovery_hold",lambda order:""),("recover",lambda reason:False)):
            item = patch.object(self.ep,name,side_effect=value)
            item.start()
            self.addCleanup(item.stop)


class V4SourceTests(unittest.TestCase):
    def test_original_v3_hashes_and_base_methods_remain_unchanged(self):
        expected = {"ICDI_chidori_v3.py":"951f15e83b3230db2c00f01c73611b295fe116358dfd501f767ffbfeb417d52d",
                    "ICDI_chidori_v3_budget.py":"1fcdaa69e2f99a5fe2d547f02217b7324b36ba85c20fd9b0642934486b44813f"}
        for filename,digest in expected.items():
            source = ROOT/"agents"/filename
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),digest)
            original = ast.parse(source.read_text())
            updated = ast.parse((ROOT/"agents"/filename.replace("v3","v4")).read_text())
            def methods(tree):
                return {(c.name,f.name):ast.dump(f) for c in tree.body if isinstance(c,ast.ClassDef)
                        for f in c.body if isinstance(f,ast.FunctionDef)}
            before,after = methods(original),methods(updated)
            changes = {key for key in before if before[key]!=after[key]}
            self.assertEqual(changes,{("ProcedureControls","procedure_gate")} if "ProcedureControls" in vars(AGENTS[filename.replace("v3","v4")]) else set())

    def test_local_source_preflight_and_unchanged_financial_limits(self):
        allowed = {"json","random","re","time","typing","decimal","src.agent.proxy_client"}
        forbidden = {"eval","exec","__import__","open","write_text","write_bytes","unlink","chmod","getenv"}
        for filename,agent in AGENTS.items():
            path = ROOT/"agents"/filename
            tree = ast.parse(path.read_text())
            self.assertLess(path.stat().st_size,1000000)
            self.assertEqual(agent.AGENT_USD_LIMIT,"0.035")
            self.assertEqual(agent.PROCEDURE_USD_RESERVE,"0.010")
            self.assertEqual(len([n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="agent_main"]),1)
            for node in ast.walk(tree):
                if isinstance(node,ast.Import):
                    self.assertTrue(all(alias.name in allowed for alias in node.names))
                elif isinstance(node,ast.ImportFrom):
                    self.assertIn(node.module,allowed)
                elif isinstance(node,ast.Call):
                    name = node.func.id if isinstance(node.func,ast.Name) else node.func.attr if isinstance(node.func,ast.Attribute) else ""
                    self.assertNotIn(name,forbidden)
                elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                    import re
                    self.assertIsNone(re.search(r"\b(?:t-[0-9a-f]{10}|p-[0-9a-f]{12})\b",node.value))
            self.assertNotIn("/inference/models",path.read_text())


class V4RecordedPreservationTests(unittest.TestCase):
    def test_twenty_one_passing_paths_under_both_scripted_audit_outcomes(self):
        path = ROOT/"logs/research/chidori-v4-development/replay_controls.py"
        spec = importlib.util.spec_from_file_location("v4_preservation_runner",path)
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        if not runner.RUN.exists():
            self.skipTest("Recorded public verification unavailable")
        problems = {p["problem_id"]:p for p in runner.rows(runner.RUN/"problems.jsonl")}
        traces = {p["problem_id"]:p["dialogue"] for p in runner.rows(runner.RUN/"sandbox/sandbox_output.jsonl")}
        passed = [r["episode"] for r in runner.rows(runner.RUN/"episode_results.jsonl") if r["episode"]["verdict"]["score"]==1]
        self.assertEqual(len(passed),21)
        reference = runner.load(ROOT/"agents/ICDI_chidori_v3.py")
        for filename,agent in AGENTS.items():
            for mode in ("covered","unavailable"):
                for episode in passed:
                    session = episode["session_id"]
                    records = [r for r in runner.rows(runner.RUN/"sandbox"/("request_log_"+session+".jsonl")) if r.get("kind")=="summary"]
                    if filename.endswith("_budget.py"):
                        self.assertEqual(agent.obj_of(records[0]["response"]["choices"][0]["message"]["content"])["process"],[])
                    result = runner.replay(agent,reference,problems[session],records,traces[session],mode,filename.endswith("_budget.py"))
                    self.assertTrue(result["preserved_under_scripted_answers"],(filename,mode,episode["task_id"],result))
                    self.assertEqual(result["environment_actions_matched"],sum(r["path"]=="/environment/call" for r in records))
                    self.assertLessEqual(result["agent_usd"],float(agent.AGENT_USD_LIMIT))


if __name__=="__main__":
    unittest.main()
