"""Quote prospective changes and classify recorded responses without inference."""
import collections
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "LV_logs/local-57eb955c7dcc48dc8d62f192a5d8760e"
sys.path.insert(0, str(ROOT))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    spec = importlib.util.spec_from_file_location("chidori_feasibility", ROOT / "agents/ICDI_chidori_v3.py")
    agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent)
    by_session = {r["episode"]["session_id"]:r["episode"] for r in rows(RUN / "episode_results.jsonl")}
    purposes = collections.defaultdict(lambda:{"calls":0,"cost_usd":0.0,"seconds":0.0})
    heavy = []
    procedures = []
    tails = []
    all_bodies_have_prices = True
    with patch("requests.sessions.Session.request", side_effect=AssertionError("No real requests allowed")) as blocked:
        for session, ep in by_session.items():
            records = [r for r in rows(RUN/"sandbox"/("request_log_"+session+".jsonl")) if r.get("kind")=="summary"]
            budget = agent.InferenceBudget()
            budget.load_prices(agent.BUDGET_PRICE_CONFIG)
            last_env = max(i for i,r in enumerate(records) if r["path"] == "/environment/call")
            tail = [r for r in records[last_env+1:] if r["path"] == "/inference/chat/completions"]
            if ep["verdict"]["score"] != 1:
                tails.append({"task":ep["task_id"],"model_calls_after_last_environment_action":len(tail),
                              "cost_usd":sum(r["response"]["usage"]["cost"] for r in tail)})
            for r in records:
                if r["path"] != "/inference/chat/completions":
                    continue
                body, data = r["json_data"], r["response"]
                content = body["messages"][0].get("content")
                prompt = "\n".join(p.get("text","") for p in content) if isinstance(content,list) else content or ""
                purpose = next((name for name,value in vars(agent).items() if name.endswith("PROMPT") and
                                isinstance(value,str) and prompt.startswith(value[:100])),
                               "DIRECTOR" if prompt.startswith(agent.GUIDE) else "OTHER")
                bucket = purposes[body["model"]+" "+purpose]
                bucket["calls"] += 1
                bucket["cost_usd"] += data["usage"]["cost"]
                bucket["seconds"] += r["duration_ms"]/1000
                choice = data["choices"][0]
                message = choice["message"]
                if body["model"] == agent.HEAVY:
                    answer = agent.obj_of(message.get("content"))
                    heavy.append({"task":ep["task_id"],"cap":body["max_tokens"],"finish_reason":choice["finish_reason"],
                                  "parseable_object":isinstance(answer,dict),"empty":not message.get("content"),
                                  "cost_usd":data["usage"]["cost"],"purpose":purpose})
                if purpose == "PROCEDURE_PROMPT":
                    probes = []
                    for cap in (700,1600,2048):
                        request = dict(body, max_tokens=cap)
                        quote,reason = budget.quote(request,protected=0)
                        probes.append({"requested_cap":cap,"admitted":quote is not None,"quote":quote,"reason":reason})
                    procedures.append({"task":ep["task_id"],"remaining_before_usd":budget.remaining()/1e9,
                                       "cost_usd":data["usage"]["cost"],"probes":probes})
                quote, _ = budget.quote(body,protected=0)
                if quote is None:
                    all_bodies_have_prices = False
                budget.actual += agent.money_units(data["usage"]["cost"])
        assert not blocked.called
    result = {"no_paid_calls":True,"purposes":dict(purposes),"heavy_responses":heavy,"procedure_quotes":procedures,
              "post_environment_tails":tails,"unparseable_heavy_cost_usd":sum(r["cost_usd"] for r in heavy if not r["parseable_object"]),
              "unparseable_heavy_count":sum(not r["parseable_object"] for r in heavy),
              "all_recorded_bodies_admit_at_their_recorded_prefix_cost":all_bodies_have_prices}
    (OUT/"feasibility.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Unparseable heavy responses",result["unparseable_heavy_count"],"cost",result["unparseable_heavy_cost_usd"])
    print("Procedure quote examples",json.dumps(procedures[:1],indent=2))
    print("Post-environment tails",tails)
    print("Recorded prefix admission check",all_bodies_have_prices)


if __name__ == "__main__":
    main()
