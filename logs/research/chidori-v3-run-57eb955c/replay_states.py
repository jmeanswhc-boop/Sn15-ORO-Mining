"""Replay the supplied run through real ProxyClient code with HTTP mocked."""
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


def load():
    spec = importlib.util.spec_from_file_location("chidori_v3_recorded_state", ROOT / "agents/ICDI_chidori_v3.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Divergence(BaseException):
    pass


def replay(agent, problem, records, trace):
    timer = {"t": 0.0, "index": 0}
    first_end = float(records[0]["timestamp"]) / 1000
    first_elapsed = next(t["t"] for t in trace if t.get("kind") == "inference_budget")
    mismatches = []
    def http(url, **kwargs):
        index = timer["index"]
        if index >= len(records):
            raise Divergence("New request after the recorded trace ended")
        expected = records[index]
        if not url.endswith(expected["path"]):
            raise Divergence("Route mismatch at %d" % index)
        actual, original = copy.deepcopy(kwargs["json"]), copy.deepcopy(expected["json_data"])
        if expected["path"] == "/inference/chat/completions":
            if actual.get("max_tokens", 0) < expected["response"].get("usage", {}).get("completion_tokens", 0):
                raise Divergence("Recorded output exceeds new cap")
            original.pop("max_tokens", None)
            actual.pop("max_tokens", None)
            if kwargs.get("headers", {}).get("Authorization") != "Bearer offline-replay":
                raise Divergence("Inference authentication missing")
        if actual != original:
            changed = [k for k in set(actual) | set(original) if actual.get(k) != original.get(k)]
            # Timing is rounded in the source trace. Record a reasoning-boundary
            # mismatch explicitly rather than silently claiming exact replay.
            mismatches.append({"request_index": index, "fields": sorted(changed)})
            raise Divergence("Payload mismatch at %d in %s" % (index, changed))
        timer["t"] = first_elapsed + float(expected["timestamp"]) / 1000 - first_end
        timer["index"] += 1
        data = copy.deepcopy(expected["response"])
        return SimpleNamespace(status_code=expected["status_code"], text=json.dumps(data), json=lambda: data)

    def sleep(seconds):
        timer["t"] += max(0, seconds)

    with TemporaryDirectory() as tmp, patch("requests.sessions.Session.request", side_effect=Divergence("Real network forbidden")) as blocked:
        client = ProxyClient(api_key="offline-replay", max_retries=1, timeout=1,
                             inference_stats_file=str(Path(tmp) / "usage.jsonl"))
        with patch("src.agent.proxy_client.requests.post", side_effect=http), patch("src.agent.proxy_client.requests.get", side_effect=Divergence("Unexpected GET")), patch.object(agent, "ENV_CLIENT", client), patch.object(agent, "model_client", return_value=client), patch.object(agent.time, "monotonic", side_effect=lambda: timer["t"]), patch.object(agent.time, "sleep", side_effect=sleep):
            ep = agent.Episode(problem)
            before_finish = []
            original_finish = ep.finish
            def finish(*args, **kwargs):
                before = {"step": ep.step, "cart": dict(ep.cart_lines), "version": ep.version, "remaining_usd": ep.inference_budget.remaining()/1e9}
                result = original_finish(*args, **kwargs)
                before.update(detail=result, after_step=ep.step, after_cart=dict(ep.cart_lines), finished=ep.finished)
                before_finish.append(before)
                return result
            ep.finish = finish
            try:
                ep.run()
                result = {"exact_replay": timer["index"] == len(records), "divergence": None}
            except Divergence as error:
                result = {"exact_replay": False, "divergence": str(error)}
            if blocked.called:
                raise AssertionError("Network escaped mocking")
            result.update(requests_consumed=timer["index"], recorded_requests=len(records), mismatches=mismatches,
                          finished=ep.finished, steps=ep.step, steps_left=ep.left(), no_credit=ep.no_credit,
                          spend_usd=ep.spend, remaining_usd=ep.inference_budget.remaining()/1e9,
                          cart_lines=ep.cart_lines, cart_ids=ep.cart_ids(), fresh=ep.fresh(),
                          unavailable=list(ep.unavail), gone=ep.gone, pending_notices=ep.pending_notices,
                          duties=ep.duties, procedure_attempts=[{"key":list(key), "count":value} for key,value in ep.procedure_attempts.items()],
                          verdicts=ep.verdicts, better=ep.check_better, review_messages=ep.review_messages,
                          version=ep.version, finish_attempts=before_finish,
                          history_content_characters=sum(len(agent.dump(m.get("content"))) for m in ep.convo),
                          history_compaction_threshold=agent.HISTORY_CAP)
            result["final_reviews"] = {key:{"context_matches":ep.review_contexts.get(key) == ep.cart_review_context(key.rsplit("#",1)[0].split("|"))} for key in ep.verdicts}
            compact = "Repair the current public state and finish using supplied tools.\nREQUEST:\n" + ep.request
            compact += "\nREQUIREMENTS:\n" + ep.ledger() + "\nCART:\n" + ep.cart_text()
            compact += "\nUNAVAILABLE:\n" + agent.dump(list(ep.unavail))
            compact += "\nREVIEW ISSUES:\n" + agent.dump(ep.review_messages)
            compact += "\nRECENT REQUIREMENT CHANGES:\n" + agent.dump(ep.changes)
            compact += "\nSEEN CANDIDATES:\n" + "\n".join(ep.rec_text(i) for i in list(ep.rows)[-6:])
            protected = ep.inference_budget.protected if (ep.duties or not ep.procedures_valid) else 0
            probe_results = []
            for label,messages,cap,tools in (("full_planner",ep.convo,agent.DIRECTOR_CAP,ep.offered()),
                                             ("compact_repair",[{"role":"user","content":compact}],2000,ep.offered())):
                for model in (agent.LIGHT, agent.SPARE):
                    captured = []
                    def capture(body):
                        captured.append(body)
                        return agent.BudgetReply("offline_probe")
                    with patch.object(ep,"budget_post",side_effect=capture):
                        ep.one_call(model,messages,cap,tools)
                    quote,reason = ep.inference_budget.quote(captured[0],protected) if captured else (None,"not_constructed")
                    body = captured[0] if captured else {}
                    messages = body.get("messages") or []
                    estimate = len(json.dumps(body,ensure_ascii=False).encode("utf-8"))+512+64*len(messages)+128*len(body.get("tools") or [])
                    probe_results.append({"kind":label,"model":model,"admitted":quote is not None,"reason":reason,"quote":quote,
                                          "estimated_prompt_tokens":estimate})
            result["budget_probes"] = probe_results
    return result


def main():
    agent = load()
    eps = {r["episode"]["session_id"]:r["episode"] for r in rows(RUN/"episode_results.jsonl")}
    problems = {r["problem_id"]:r for r in rows(RUN/"problems.jsonl")}
    results = []
    for sandbox in rows(RUN/"sandbox/sandbox_output.jsonl"):
        session = sandbox["problem_id"]
        records = [r for r in rows(RUN/"sandbox"/("request_log_"+session+".jsonl")) if r.get("kind")=="summary"]
        result = replay(agent,problems[session],records,sandbox["dialogue"])
        result.update(task=eps[session]["task_id"],score=eps[session]["verdict"]["score"])
        results.append(result)
        if result["score"]!=1 or not result["exact_replay"]:
            print(result["task"],"exact",result["exact_replay"],result["divergence"],"finished",result["finished"],"cart",result["cart_lines"],"left",result["steps_left"],"finish",[r["detail"] for r in result["finish_attempts"]])
    (OUT/"replay-states.json").write_text(json.dumps({"no_paid_calls":True,"results":results},indent=2)+"\n")
    print("Exact replays",sum(r["exact_replay"] for r in results),"/",len(results))


if __name__ == "__main__":
    main()
