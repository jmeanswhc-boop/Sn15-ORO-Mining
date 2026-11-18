"""Conditional preservation replay. Added model responses are explicitly scripted."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "LV_logs/local-57eb955c7dcc48dc8d62f192a5d8760e"
sys.path.insert(0, str(ROOT))
from src.agent.proxy_client import ProxyClient


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem+"_preservation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Divergence(BaseException):
    pass


def replay(agent, reference, problem, records, trace, audit_mode="covered", budget_variant=False):
    cursor = {"index":0,"t":0.0,"scripted":[],"action_count":0}
    first_end = float(records[0]["timestamp"])/1000
    first_elapsed = next(t["t"] for t in trace if t.get("kind")=="inference_budget")
    def http(url, **kwargs):
        index = cursor["index"]
        body = copy.deepcopy(kwargs["json"])
        if url.endswith("/inference/chat/completions"):
            if kwargs.get("headers",{}).get("Authorization") != "Bearer offline-control":
                raise Divergence("Missing inference authentication")
            content = body["messages"][0].get("content")
            if isinstance(content,str) and content.startswith("Audit the request against the current binding ledger"):
                answer = {"complete":True,"missing":[]} if audit_mode=="covered" else {}
                cursor["scripted"].append("source_audit_"+audit_mode)
                data = {"usage":{"cost":0.00015,"prompt_tokens":500,"completion_tokens":60},
                        "choices":[{"message":{"content":json.dumps(answer)},"finish_reason":"stop"}]}
                return SimpleNamespace(status_code=200,text=json.dumps(data),json=lambda:data)
            if isinstance(content,str) and content.startswith(agent.CHECK_PROMPT) and (index>=len(records) or records[index]["path"]!="/inference/chat/completions"):
                cursor["scripted"].append("new_fallback_review_true")
                data = {"usage":{"cost":0.00015,"prompt_tokens":500,"completion_tokens":60},
                        "choices":[{"message":{"content":'{"ok":true,"problems":[],"better":[]}'},"finish_reason":"stop"}]}
                return SimpleNamespace(status_code=200,text=json.dumps(data),json=lambda:data)
            if isinstance(content,str) and content.startswith("Resolve the specific disagreement"):
                cursor["scripted"].append("dispute_unresolved")
                data = {"usage":{"cost":0.00015,"prompt_tokens":500,"completion_tokens":60},
                        "choices":[{"message":{"content":'{"ok":false}'},"finish_reason":"stop"}]}
                return SimpleNamespace(status_code=200,text=json.dumps(data),json=lambda:data)
        if index>=len(records):
            raise Divergence("A new non-scripted request follows the recorded trace")
        expected = records[index]
        if not url.endswith(expected["path"]):
            raise Divergence("Route differs at request %d: expected %s, actual %s" % (index,expected["path"],body.get("action") or body.get("model")))
        original = copy.deepcopy(expected["json_data"])
        if index==0 and budget_variant:
            text = original["messages"][0]["content"]
            original["messages"][0]["content"] = agent.BRIEF_PROMPT+text[len(reference.BRIEF_PROMPT):]
        if expected["path"]=="/inference/chat/completions":
            if body.get("max_tokens",0)<expected["response"].get("usage",{}).get("completion_tokens",0):
                raise Divergence("Recorded completion exceeds new reply cap at request %d" % index)
            original.pop("max_tokens",None)
            body.pop("max_tokens",None)
        if body!=original:
            keys = sorted(k for k in set(body)|set(original) if body.get(k)!=original.get(k))
            detail = (" expected="+str(original.get("action"))+" actual="+str(body.get("action"))) if "action" in keys else ""
            raise Divergence("Payload differs at request %d in %s%s" % (index,keys,detail))
        cursor["index"] += 1
        cursor["t"] = first_elapsed+float(expected["timestamp"])/1000-first_end
        if expected["path"]=="/environment/call":
            cursor["action_count"] += 1
        data = copy.deepcopy(expected["response"])
        return SimpleNamespace(status_code=expected["status_code"],text=json.dumps(data),json=lambda:data)
    with TemporaryDirectory() as tmp, patch("requests.sessions.Session.request",side_effect=Divergence("Real network forbidden")) as blocked:
        client = ProxyClient(api_key="offline-control",max_retries=1,timeout=1,inference_stats_file=str(Path(tmp)/"usage.jsonl"))
        with patch("src.agent.proxy_client.requests.post",side_effect=http),patch("src.agent.proxy_client.requests.get",side_effect=Divergence("Unexpected GET")),patch.object(agent,"model_client",return_value=client),patch.object(agent,"ENV_CLIENT",client),patch.object(agent.time,"monotonic",side_effect=lambda:cursor["t"]),patch.object(agent.time,"sleep",side_effect=lambda seconds:None):
            ep = agent.Episode(problem)
            try:
                ep.run()
                divergence = None
            except Divergence as error:
                divergence = str(error)
        assert not blocked.called
    return {"preserved_under_scripted_answers":ep.finished and cursor["index"]==len(records),"divergence":divergence,
            "recorded_requests_consumed":cursor["index"],"recorded_requests":len(records),
            "scripted_new_responses":cursor["scripted"],"environment_actions_matched":cursor["action_count"],
            "agent_usd":ep.spend,"finished":ep.finished}


def main():
    problems = {p["problem_id"]:p for p in rows(RUN/"problems.jsonl")}
    traces = {p["problem_id"]:p["dialogue"] for p in rows(RUN/"sandbox/sandbox_output.jsonl")}
    episodes = [r["episode"] for r in rows(RUN/"episode_results.jsonl") if r["episode"]["verdict"]["score"]==1]
    reference = load(ROOT/"agents/ICDI_chidori_v3.py")
    results = []
    for filename in ("ICDI_chidori_v4.py","ICDI_chidori_v4_budget.py"):
        agent = load(ROOT/"agents"/filename)
        for mode in ("covered","unavailable"):
            for episode in episodes:
                session = episode["session_id"]
                records = [r for r in rows(RUN/"sandbox"/("request_log_"+session+".jsonl")) if r.get("kind")=="summary"]
                result = replay(agent,reference,problems[session],records,traces[session],mode,filename.endswith("_budget.py"))
                result.update(agent=filename,task=episode["task_id"],audit_mode=mode)
                results.append(result)
    (OUT/"preservation-replay.json").write_text(json.dumps({"no_paid_calls":True,"new_accuracy_measurement":False,
        "limitation":"New audit/review responses are scripted assumptions. This checks control flow, not their actual correctness.","results":results},indent=2)+"\n")
    for filename in ("ICDI_chidori_v4.py","ICDI_chidori_v4_budget.py"):
        for mode in ("covered","unavailable"):
            group = [r for r in results if r["agent"]==filename and r["audit_mode"]==mode]
            print(filename,mode,sum(r["preserved_under_scripted_answers"] for r in group),"/",len(group))
            for row in group:
                if not row["preserved_under_scripted_answers"]:
                    print(row["task"],row["divergence"])


if __name__=="__main__":
    main()
