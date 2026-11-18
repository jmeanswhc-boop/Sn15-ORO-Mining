"""Offline deterministic replay. Recorded responses do not predict new model accuracy."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.agent.proxy_client import ProxyClient

RUN = ROOT / "LV_logs/local-46df07bb11a34426b51f83c581c88575"


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem + "_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReplayDivergence(BaseException):
    pass


def replay(agent, problem, records, reference=None):
    state = {"index": 0, "authorized_posts": 0, "outputs_within_caps": True}
    def http(url, **kwargs):
        index = state["index"]
        if index >= len(records):
            raise ReplayDivergence("A new request has no recorded response.")
        expected = records[index]
        if not url.endswith(expected["path"]):
            raise ReplayDivergence("Request route differs at index %d." % index)
        body = copy.deepcopy(kwargs["json"])
        original = copy.deepcopy(expected["json_data"])
        if index == 0 and reference is not None:
            content = original["messages"][0]["content"]
            if not content.startswith(reference.BRIEF_PROMPT):
                raise ReplayDivergence("The expected briefing prefix is absent.")
            original["messages"][0]["content"] = agent.BRIEF_PROMPT + content[len(reference.BRIEF_PROMPT):]
        if expected["path"] == "/inference/chat/completions":
            output = expected["response"].get("usage", {}).get("completion_tokens", 0)
            if body.get("max_tokens", 0) < output:
                raise ReplayDivergence("The recorded output does not fit the new reply cap.")
            body.pop("max_tokens", None)
            original.pop("max_tokens", None)
        if body != original:
            keys = sorted(k for k in set(body) | set(original) if body.get(k) != original.get(k))
            state["changed_fields"] = keys
            state["expected_model"] = original.get("model")
            state["actual_model"] = body.get("model")
            def preview(payload):
                messages = payload.get("messages") or [{}]
                return str(payload.get("model")) + ": " + str(messages[0].get("content"))[:90]
            raise ReplayDivergence("Request payload differs at index %d in %s. Expected %s; actual %s" % (index, ",".join(keys), preview(original), preview(body)))
        authorized = kwargs.get("headers", {}).get("Authorization") == "Bearer offline-test-token"
        if expected["path"] == "/inference/chat/completions" and not authorized:
            raise ReplayDivergence("Missing authentication.")
        state["authorized_posts"] += int(authorized)
        state["index"] += 1
        data = copy.deepcopy(expected["response"])
        return SimpleNamespace(status_code=expected["status_code"], text=json.dumps(data), json=lambda: data)

    with TemporaryDirectory() as tmp, patch("requests.sessions.Session.request", side_effect=ReplayDivergence("Real network forbidden")) as blocked:
        client = ProxyClient(api_key="offline-test-token", max_retries=1, timeout=1,
                             inference_stats_file=str(Path(tmp) / "usage.jsonl"))
        with patch("src.agent.proxy_client.requests.post", side_effect=http), patch("src.agent.proxy_client.requests.get", side_effect=ReplayDivergence("Unexpected catalog GET")), patch.object(agent, "model_client", return_value=client), patch.object(agent, "ENV_CLIENT", client), patch.object(agent.time, "sleep"):
            ep = agent.Episode(problem)
            ep.room = lambda: 280
            ep.clock = lambda: 0
            try:
                ep.run()
                result = {"completed_same_trace": ep.finished and state["index"] == len(records), "divergence": None}
            except ReplayDivergence as error:
                result = {"completed_same_trace": False, "divergence": str(error)}
        if blocked.called:
            raise AssertionError("Unexpected network escape")
    result.update(state)
    result.update(recorded_requests=len(records), finished=ep.finished, steps=ep.step,
                  pending_notices=len(ep.pending_notices) if hasattr(ep, "pending_notices") else None,
                  negative_reviews=sum(value is False for value in ep.verdicts.values()))
    return result


def main():
    problems = {r["problem_id"]: r for r in rows(RUN / "problems.jsonl")}
    traces = {r["problem_id"]: r for r in rows(RUN / "sandbox/sandbox_output.jsonl")}
    passed = [r["episode"] for r in rows(RUN / "episode_results.jsonl") if r["episode"]["verdict"]["score"] == 1]
    results = []
    reference = load(ROOT / "agents/ICDI_chidori_v2.py")
    configurations = [("ICDI_chidori_v2.py", "baseline"),
                      ("ICDI_chidori_v3.py", "recovery_isolated_with_v2_prices"),
                      ("ICDI_chidori_v3_budget.py", "recovery_isolated_with_v2_prices"),
                      ("ICDI_chidori_v3.py", "complete_v3"), ("ICDI_chidori_v3_budget.py", "complete_v3")]
    for filename, configuration in configurations:
        agent = load(ROOT / "agents" / filename)
        if configuration == "recovery_isolated_with_v2_prices":
            agent.BUDGET_PRICE_CONFIG = reference.BUDGET_PRICE_CONFIG
        for episode in passed:
            session = episode["session_id"]
            if any(t.get("kind") == "local_inference_stop" for t in traces[session]["dialogue"]):
                continue
            records = [r for r in rows(RUN / "sandbox" / ("request_log_" + session + ".jsonl")) if r.get("kind") == "summary"]
            adapted = filename.endswith("_budget.py")
            if adapted:
                assert records[0]["response"]["choices"][0]["message"]["content"]
                brief = agent.obj_of(records[0]["response"]["choices"][0]["message"]["content"])
                assert brief["process"] == [], "Cross-variant replay requires a zero-duty briefing."
            result = replay(agent, problems[session], records, reference if adapted else None)
            result.update(agent=filename, configuration=configuration, session=session, task=episode["task_id"])
            result["brief_schema_adapted"] = adapted
            result["stopped_at_newly_admitted_primary_model"] = (configuration == "complete_v3" and
                result.get("expected_model") == agent.SPARE and result.get("actual_model") == agent.LIGHT and
                result.get("changed_fields") == ["messages", "model"])
            results.append(result)
    report = {"is_paid_or_new_accuracy_measurement": False,
              "limitation": "Uses recorded model responses and unchanged public observations. A divergence requires separate analysis.",
              "results": results}
    (Path(__file__).parent / "replay-results.json").write_text(json.dumps(report, indent=2) + "\n")
    for filename, configuration in configurations:
        group = [r for r in results if r["agent"] == filename and r["configuration"] == configuration]
        print(filename, configuration, "complete", sum(r["completed_same_trace"] for r in group), "/", len(group))
        for row in group:
            if not row["completed_same_trace"]:
                print(" ", row["task"], row["index"], row["divergence"][:140])


if __name__ == "__main__":
    main()
