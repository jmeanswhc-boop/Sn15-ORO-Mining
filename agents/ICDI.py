"""ICDI: an evidence-led agent for dynamic ORO store worlds.

ICDI discovers each world's contract at runtime, executes exactly one supplied
tool per environment step, and keeps planning evidence separate from finish
authority. Its behavior depends only on the public task contract and the
observations returned during the current episode.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.proxy_client import ProxyClient


JsonObject = dict[str, Any]
ToolCall = tuple[str, JsonObject]


@dataclass(frozen=True)
class ICDIModels:
    planner: tuple[str, ...] = (
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-sonnet-5",
        "openai/gpt-6-luna",
    )
    actor: tuple[str, ...] = (
        "anthropic/claude-haiku-5.5",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-6-luna",
    )
    auditor: tuple[str, ...] = (
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-6-luna",
    )


MODELS = ICDIModels()


class JsonCodec:
    """Small, bounded JSON helpers used at model and environment boundaries."""

    @staticmethod
    def dump(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
        except (TypeError, ValueError):
            return json.dumps(str(value), ensure_ascii=False)

    @staticmethod
    def object_from_text(value: Any) -> JsonObject | None:
        if not isinstance(value, str):
            return None
        text = value.strip()
        fenced = re.fullmatch(r"```[A-Za-z0-9_-]*\s*(.*?)\s*```", text, re.S)
        if fenced:
            text = fenced.group(1)
        candidates = [text]
        left, right = text.find("{"), text.rfind("}")
        if 0 <= left < right:
            candidates.append(text[left : right + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @classmethod
    def bounded(cls, value: Any, text_limit: int = 12000, depth: int = 0) -> Any:
        if depth > 7:
            return "..."
        if isinstance(value, str):
            return value if len(value) <= text_limit else value[:text_limit] + "..."
        if isinstance(value, list):
            return [cls.bounded(item, text_limit, depth + 1) for item in value[:80]]
        if isinstance(value, dict):
            return {
                str(key): cls.bounded(item, text_limit, depth + 1)
                for key, item in list(value.items())[:100]
            }
        return value


class ObservationShape:
    """Restores transport-friendly result shapes before they reach the planner."""

    _identifier_key = re.compile(r"(^|[_. -])(id|sku|code)([_. -]|$)", re.I)
    _identifier_value = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{2,95}$")

    @classmethod
    def restore(cls, value: Any, depth: int = 0) -> Any:
        if depth > 7:
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("{", "[")):
                try:
                    return cls.restore(json.loads(stripped), depth + 1)
                except ValueError:
                    pass
            fields = cls._field_record(stripped, depth)
            return fields if fields is not None else value
        if isinstance(value, list):
            return [cls.restore(item, depth + 1) for item in value]
        if not isinstance(value, dict):
            return value
        restored = {str(key): cls.restore(item, depth + 1) for key, item in value.items()}
        columns = cls._columns(restored)
        if columns is not None:
            return columns
        keyed = cls._keyed_rows(restored)
        return keyed if keyed is not None else restored

    @classmethod
    def _field_record(cls, text: str, depth: int) -> JsonObject | None:
        pieces = [part.strip() for part in re.split(r"[;\n|]", text) if part.strip()]
        if len(pieces) < 3:
            return None
        record: JsonObject = {}
        for piece in pieces:
            match = re.fullmatch(r"([A-Za-z][A-Za-z0-9 _.-]{0,39})\s*[:=]\s*(.+)", piece)
            if not match:
                return None
            key, raw = match.group(1).strip(), match.group(2).strip()
            if key in record:
                return None
            record[key] = cls.restore(raw, depth + 1)
        if not any(cls._identifier_key.search(key) for key in record):
            return None
        return record

    @classmethod
    def _columns(cls, value: JsonObject) -> list[JsonObject] | None:
        if len(value) < 2 or not any(cls._identifier_key.search(key) for key in value):
            return None
        if not all(isinstance(column, list) for column in value.values()):
            return None
        lengths = {len(column) for column in value.values()}
        if len(lengths) != 1 or not lengths or next(iter(lengths)) == 0:
            return None
        return [
            {key: column[index] for key, column in value.items()}
            for index in range(next(iter(lengths)))
        ]

    @classmethod
    def _keyed_rows(cls, value: JsonObject) -> list[JsonObject] | None:
        if not value or not all(isinstance(item, dict) for item in value.values()):
            return None
        if not all(cls._identifier_value.fullmatch(key) for key in value):
            return None
        rows: list[JsonObject] = []
        for key, raw in value.items():
            row = dict(raw)
            has_identifier = any(cls._identifier_key.search(name) for name in row)
            rows.append(row if has_identifier else {"id": key, **row})
        return rows


class SchemaGuard:
    """Checks model-generated arguments against the supplied JSON schema."""

    @staticmethod
    def _kinds(schema: JsonObject) -> set[str]:
        declared = schema.get("type")
        if isinstance(declared, str):
            return {declared}
        if isinstance(declared, list):
            return {item for item in declared if isinstance(item, str)}
        return set()

    @classmethod
    def check(cls, value: Any, schema: JsonObject, path: str = "args") -> list[str]:
        if not isinstance(schema, dict):
            return []
        problems: list[str] = []
        if "enum" in schema and value not in schema["enum"]:
            problems.append(f"{path} is not an allowed value")
            return problems
        kinds = cls._kinds(schema)
        if value is None:
            return [] if not kinds or "null" in kinds else [f"{path} may not be null"]
        matches = (
            not kinds
            or ("object" in kinds and isinstance(value, dict))
            or ("array" in kinds and isinstance(value, list))
            or ("string" in kinds and isinstance(value, str))
            or ("boolean" in kinds and isinstance(value, bool))
            or ("integer" in kinds and isinstance(value, int) and not isinstance(value, bool))
            or ("number" in kinds and isinstance(value, (int, float)) and not isinstance(value, bool))
        )
        if not matches:
            return [f"{path} has the wrong type"]
        if isinstance(value, dict):
            properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            required = schema.get("required") if isinstance(schema.get("required"), list) else []
            problems.extend(f"{path}.{key} is required" for key in required if key not in value)
            if schema.get("additionalProperties") is False:
                problems.extend(f"{path}.{key} is not accepted" for key in value if key not in properties)
            for key, item in value.items():
                child = properties.get(key)
                if isinstance(child, dict):
                    problems.extend(cls.check(item, child, f"{path}.{key}"))
        if isinstance(value, list) and isinstance(schema.get("items"), dict):
            for index, item in enumerate(value):
                problems.extend(cls.check(item, schema["items"], f"{path}[{index}]"))
        return problems

    @classmethod
    def required_value(cls, schema: JsonObject, label: str) -> Any:
        options = schema.get("enum")
        if isinstance(options, list) and options:
            return options[0]
        kinds = cls._kinds(schema)
        if "object" in kinds or isinstance(schema.get("properties"), dict):
            properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
            required = schema.get("required") if isinstance(schema.get("required"), list) else []
            return {key: cls.required_value(properties.get(key, {}), key) for key in required}
        if "array" in kinds:
            return []
        if "boolean" in kinds:
            return False
        if "integer" in kinds:
            floor = schema.get("minimum")
            return int(floor) if isinstance(floor, (int, float)) else 1
        if "number" in kinds:
            floor = schema.get("minimum")
            return floor if isinstance(floor, (int, float)) else 1
        return label.replace("_", " ")


class ToolContract:
    """Indexes only the tools and schemas disclosed by the current world."""

    ROLE_KEYS = ("search", "detail", "message", "cart", "cart_view", "buy", "refuse")

    def __init__(self, tools: Any) -> None:
        self.tools: list[JsonObject] = [
            tool
            for tool in tools or []
            if isinstance(tool, dict)
            and isinstance(tool.get("function"), dict)
            and isinstance(tool["function"].get("name"), str)
        ]
        self.by_name = {tool["function"]["name"]: tool for tool in self.tools}
        self.roles: dict[str, str] = {}

    def describe(self) -> list[JsonObject]:
        return [
            {
                "name": tool["function"]["name"],
                "description": tool["function"].get("description", ""),
                "parameters": tool["function"].get("parameters", {}),
            }
            for tool in self.tools
        ]

    def bind_roles(self, proposed: Any) -> None:
        if not isinstance(proposed, dict):
            return
        self.roles = {
            role: name
            for role, name in proposed.items()
            if role in self.ROLE_KEYS and isinstance(name, str) and name in self.by_name
        }

    def schema(self, name: str) -> JsonObject:
        function = self.by_name.get(name, {}).get("function", {})
        schema = function.get("parameters", {})
        return schema if isinstance(schema, dict) else {}

    def selected(self, names: set[str] | None = None) -> list[JsonObject]:
        return self.tools if names is None else [tool for tool in self.tools if tool["function"]["name"] in names]

    def terminal_names(self) -> set[str]:
        return {self.roles[role] for role in ("buy", "refuse") if role in self.roles}

    def is_message(self, name: str) -> bool:
        return name == self.roles.get("message")

    def fallback_arguments(self, name: str) -> JsonObject:
        schema = self.schema(name)
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        required = schema.get("required") if isinstance(schema.get("required"), list) else []
        return {key: SchemaGuard.required_value(properties.get(key, {}), key) for key in required}


class ModelGateway:
    """Uses the supported inference proxy with bounded, staggered 429 retries."""

    ROUTE = "/inference/chat/completions"

    def __init__(self) -> None:
        self.client = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.failures: list[JsonObject] = []

    @staticmethod
    def _message(data: Any) -> JsonObject | None:
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            return None
        message = data["choices"][0].get("message")
        return message if isinstance(message, dict) else None

    def complete(
        self,
        messages: list[JsonObject],
        models: tuple[str, ...],
        tools: list[JsonObject] | None = None,
        max_tokens: int = 1200,
    ) -> JsonObject | None:
        for attempt, model in enumerate(models):
            body: JsonObject = {
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
            if tools:
                body.update(tools=tools, tool_choice="required")
            result = self.client.post_verbose(self.ROUTE, body)
            message = self._message(result.data)
            if message is not None and (not tools or message.get("tool_calls")):
                return message
            error = result.error if isinstance(result.error, dict) else {}
            self.failures.append(
                {
                    "model": model,
                    "status": error.get("status"),
                    "reason": "missing tool call" if message is not None and tools else "request failure",
                }
            )
            if error.get("status") == 429 and attempt + 1 < len(models):
                time.sleep(1.0 + 0.5 * attempt)
        return None

    def json_object(
        self,
        system: str,
        payload: JsonObject,
        models: tuple[str, ...],
        max_tokens: int = 1600,
    ) -> JsonObject | None:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": JsonCodec.dump(payload)},
        ]
        for attempt, model in enumerate(models):
            failures_before = len(self.failures)
            message = self.complete(messages, (model,), max_tokens=max_tokens)
            parsed = JsonCodec.object_from_text(message.get("content")) if message else None
            if parsed is not None:
                return parsed
            failed = self.failures[-1] if len(self.failures) > failures_before else {}
            if failed.get("status") == 429 and attempt + 1 < len(models):
                time.sleep(1.0 + 0.5 * attempt)
        return None


@dataclass
class Episode:
    binding: JsonObject
    query: str
    max_steps: int
    step: int = 0
    finished: bool = False

    def remaining(self) -> int:
        return max(0, self.max_steps - self.step)


class EnvironmentPort:
    """Executes one and only one disclosed world action per step."""

    ROUTE = "/environment/call"

    def __init__(self, episode: Episode) -> None:
        self.episode = episode
        self.client = ProxyClient(timeout=95, max_retries=3, retry_delay=1)

    def call(self, name: str, arguments: JsonObject) -> JsonObject | None:
        expected = self.episode.step + 1
        payload = {
            **self.episode.binding,
            "step": expected,
            "action": {"name": name, "args": arguments},
        }
        result = self.client.post_verbose(self.ROUTE, payload)
        if not isinstance(result.data, dict):
            return None
        response = dict(result.data)
        response["observation"] = ObservationShape.restore(response.get("observation"))
        returned_step = response.get("step")
        self.episode.step = returned_step if isinstance(returned_step, int) else expected
        self.episode.finished = bool(response.get("finished"))
        return response


@dataclass
class EvidenceEntry:
    step: int
    tool: str
    arguments: JsonObject
    observation: Any
    finished: bool


@dataclass
class EvidenceTrail:
    opening: Any = None
    entries: list[EvidenceEntry] = field(default_factory=list)

    def add(self, step: int, name: str, arguments: JsonObject, result: JsonObject) -> None:
        self.entries.append(
            EvidenceEntry(
                step=step,
                tool=name,
                arguments=arguments,
                observation=result.get("observation"),
                finished=bool(result.get("finished")),
            )
        )

    def repeated(self, name: str, arguments: JsonObject) -> bool:
        if not self.entries:
            return False
        latest = self.entries[-1]
        if latest.tool != name or latest.arguments != arguments:
            return False
        observation = latest.observation
        return not (isinstance(observation, dict) and observation.get("error"))

    def render(self, limit: int = 52000) -> JsonObject:
        records = [
            {
                "step": entry.step,
                "tool": entry.tool,
                "arguments": entry.arguments,
                "observation": JsonCodec.bounded(entry.observation),
            }
            for entry in self.entries
        ]
        payload: JsonObject = {"opening": JsonCodec.bounded(self.opening), "records": records}
        while len(JsonCodec.dump(payload)) > limit and len(records) > 8:
            records.pop(0)
        return payload


@dataclass
class Mission:
    plan: JsonObject = field(default_factory=dict)
    revision: int = 0

    def replace(self, value: JsonObject) -> None:
        self.plan = value
        self.revision += 1

    def apply_change(self, value: JsonObject) -> bool:
        if not value.get("changed"):
            return False
        update = value.get("plan_update")
        if isinstance(update, dict):
            self.plan = {**self.plan, **update}
        self.revision += 1
        return True


class MissionReader:
    """Turns the public request and schemas into a revisable working brief."""

    PLAN_SYSTEM = """You are the planning component of ICDI. Read a shopper request and the tools supplied by this world. Return exactly one JSON object. Do not invent product facts.

Required keys:
- objective: concise restatement of what must be accomplished
- requirements: every explicit product, quantity, constraint, budget, preference and process instruction
- selection_rule: how acceptable candidates must be compared, or "any"
- questions: unresolved information the request requires ICDI to ask the shopper
- roles: exact tool names for any applicable roles among search, detail, message, cart, cart_view, buy, refuse; omit unavailable roles
- completion_evidence: facts ICDI must observe before buying or refusing

Infer roles only from the supplied tool names, descriptions and schemas. Keep requirements in the shopper's words. The plan must generalize to the disclosed contract and must not assume a fixed catalog."""

    CHANGE_SYSTEM = """You are ICDI's revision reader. Determine whether a newly returned shopper message changes the task. Return exactly one JSON object with changed (boolean), plan_update (an object containing only revised objective, requirements, selection_rule, questions, or completion_evidence), and summary. Store errors and ordinary acknowledgements are not task changes."""

    def __init__(self, models: ModelGateway, contract: ToolContract) -> None:
        self.models = models
        self.contract = contract

    def initial(self, query: str, opening: Any) -> JsonObject:
        payload = {
            "request": query,
            "opening_observation": JsonCodec.bounded(opening),
            "tools": self.contract.describe(),
        }
        plan = self.models.json_object(self.PLAN_SYSTEM, payload, MODELS.planner, max_tokens=2200)
        if not isinstance(plan, dict):
            plan = {
                "objective": query,
                "requirements": [query],
                "selection_rule": "as requested",
                "questions": [],
                "roles": {},
                "completion_evidence": [],
            }
        self.contract.bind_roles(plan.get("roles"))
        return plan

    def revision(self, query: str, plan: JsonObject, observation: Any) -> JsonObject:
        payload = {
            "original_request": query,
            "current_plan": plan,
            "new_message_result": JsonCodec.bounded(observation),
        }
        return self.models.json_object(self.CHANGE_SYSTEM, payload, MODELS.planner) or {
            "changed": False,
            "plan_update": {},
            "summary": "unreadable revision",
        }


class StepPolicy:
    """Makes recovery capacity a deterministic function of disclosed step limits."""

    def __init__(self, episode: Episode, contract: ToolContract) -> None:
        self.episode = episode
        self.contract = contract

    def available_tools(self) -> list[JsonObject]:
        terminals = self.contract.terminal_names()
        if self.episode.remaining() <= 2 and terminals:
            return self.contract.selected(terminals)
        return self.contract.selected()

    def permits(self, name: str) -> tuple[bool, str]:
        terminals = self.contract.terminal_names()
        if self.episode.remaining() <= 2 and terminals and name not in terminals:
            return False, "The final two steps are reserved for a finish attempt and one recovery attempt."
        return True, ""


class TerminalAuditor:
    """Independently checks evidence before a proposed terminal world action."""

    SYSTEM = """You are ICDI's independent finish auditor. Review the public request, current plan, observations and proposed terminal action. Return exactly one JSON object with allow (boolean), problems (list of concrete missing or contradictory facts), and instruction (one concise correction). Allow a purchase only when the evidence supports every current requirement, quantity, budget, availability and requested comparison. Allow a refusal only when the evidence supports that the requested outcome cannot be completed. Do not invent facts and do not reproduce or predict any hidden grader."""

    def __init__(self, models: ModelGateway) -> None:
        self.models = models
        self._revision = -1
        self._record_count = -1
        self._action: ToolCall | None = None
        self._decision: JsonObject | None = None

    def reset(self) -> None:
        self._revision = -1
        self._record_count = -1
        self._action = None
        self._decision = None

    def review(self, query: str, mission: Mission, trail: EvidenceTrail, action: ToolCall) -> JsonObject:
        if (
            self._revision == mission.revision
            and self._record_count == len(trail.entries)
            and self._action == action
            and self._decision is not None
        ):
            return self._decision
        payload = {
            "request": query,
            "plan": mission.plan,
            "evidence": trail.render(),
            "proposed_action": {"name": action[0], "arguments": action[1]},
        }
        decision = self.models.json_object(self.SYSTEM, payload, MODELS.auditor, max_tokens=1200)
        if not isinstance(decision, dict) or not isinstance(decision.get("allow"), bool):
            decision = {
                "allow": False,
                "problems": ["The independent audit did not return a valid decision."],
                "instruction": "Reconsider the finish using the available evidence.",
            }
        self._revision = mission.revision
        self._record_count = len(trail.entries)
        self._action = action
        self._decision = decision
        return decision


class ActionChooser:
    """Chooses one reachable tool call from the current contract and evidence."""

    SYSTEM = """You are ICDI, a careful shopping agent operating through the tools supplied by the current world. Choose exactly one tool call and no plain-text answer.

Rules:
- Treat the current plan as a working interpretation, but the shopper's request and later messages remain authoritative.
- Use observations as evidence; never invent listing properties, prices, availability, cart contents or shopper answers.
- Read the supplied schemas and use exact argument names and allowed values.
- Search broadly enough to satisfy the requested comparison, then inspect promising candidates in enough detail to verify every requirement.
- Execute requested communication and process steps through the disclosed tools.
- Keep the cart aligned with the current requirements and verify it before buying.
- Refuse only when the observations justify that the requested outcome cannot be completed.
- When the final two steps are reserved, make the best supported terminal call now so one recovery step remains.
"""

    def __init__(self, models: ModelGateway, contract: ToolContract) -> None:
        self.models = models
        self.contract = contract

    @staticmethod
    def _decode(message: JsonObject | None) -> ToolCall | None:
        if not message:
            return None
        calls = message.get("tool_calls")
        if not isinstance(calls, list) or not calls:
            return None
        function = calls[0].get("function") if isinstance(calls[0], dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return None
        raw = function.get("arguments", {})
        arguments = JsonCodec.object_from_text(raw) if isinstance(raw, str) else raw if isinstance(raw, dict) else None
        return (function["name"], arguments) if isinstance(arguments, dict) else None

    def choose(
        self,
        episode: Episode,
        mission: Mission,
        trail: EvidenceTrail,
        tools: list[JsonObject],
        feedback: list[str],
    ) -> ToolCall | None:
        payload = {
            "request": episode.query,
            "steps": {"used": episode.step, "remaining": episode.remaining()},
            "plan": mission.plan,
            "evidence": trail.render(),
            "corrections": feedback[-4:],
        }
        message = self.models.complete(
            [
                {"role": "system", "content": self.SYSTEM},
                {"role": "user", "content": JsonCodec.dump(payload)},
            ],
            MODELS.actor,
            tools=tools,
            max_tokens=1100,
        )
        return self._decode(message)


class ICDI:
    """Coordinates one episode while keeping action and finish authority explicit."""

    def __init__(self, problem_data: JsonObject) -> None:
        environment = problem_data["environment"]
        policy = environment["policy_view"]
        self.episode = Episode(
            binding=dict(environment["binding"]),
            query=str(policy.get("query") or ""),
            max_steps=int(policy.get("max_steps") or 1),
            finished=bool(policy.get("finished")),
        )
        self.contract = ToolContract(policy.get("tools"))
        self.models = ModelGateway()
        self.world = EnvironmentPort(self.episode)
        self.reader = MissionReader(self.models, self.contract)
        self.mission = Mission()
        self.trail = EvidenceTrail(ObservationShape.restore(policy.get("opening")))
        self.policy = StepPolicy(self.episode, self.contract)
        self.auditor = TerminalAuditor(self.models)
        self.chooser = ActionChooser(self.models, self.contract)
        self.dialogue: list[JsonObject] = []

    def _record(self, action: ToolCall, result: JsonObject) -> None:
        name, arguments = action
        self.trail.add(self.episode.step, name, arguments, result)
        self.dialogue.append(
            {
                "role": "assistant",
                "content": "",
                "tool_call": {"name": name, "arguments": arguments},
                "environment_result": result,
            }
        )

    def _refresh_after_message(self, action: ToolCall, result: JsonObject) -> None:
        if not self.contract.is_message(action[0]):
            return
        change = self.reader.revision(self.episode.query, self.mission.plan, result.get("observation"))
        if self.mission.apply_change(change):
            self.auditor.reset()
            roles = self.mission.plan.get("roles")
            if isinstance(roles, dict):
                self.contract.bind_roles(roles)

    def _candidate(self, feedback: list[str]) -> ToolCall | None:
        available = self.policy.available_tools()
        for _ in range(2):
            action = self.chooser.choose(self.episode, self.mission, self.trail, available, feedback)
            if action is None:
                feedback.append("Return one valid call to one of the supplied tools.")
                continue
            name, arguments = action
            if name not in self.contract.by_name:
                feedback.append(f"The tool {name!r} is not in the current contract.")
                continue
            problems = SchemaGuard.check(arguments, self.contract.schema(name))
            if problems:
                feedback.append("Invalid arguments: " + "; ".join(problems[:6]))
                continue
            permitted, reason = self.policy.permits(name)
            if not permitted:
                feedback.append(reason)
                continue
            if self.trail.repeated(name, arguments):
                feedback.append("That exact successful call was just made; use its observation and choose a different next action.")
                continue
            return action
        return None

    def _fallback_terminal(self) -> ToolCall | None:
        terminals = self.contract.terminal_names()
        if not terminals:
            return None
        preferred = self.contract.roles.get("refuse") or self.contract.roles.get("buy")
        name = preferred if preferred in terminals else sorted(terminals)[0]
        return name, self.contract.fallback_arguments(name)

    def _approved(self, action: ToolCall, feedback: list[str]) -> bool:
        if action[0] not in self.contract.terminal_names() or self.episode.remaining() <= 1:
            return True
        verdict = self.auditor.review(self.episode.query, self.mission, self.trail, action)
        if verdict.get("allow"):
            return True
        problems = verdict.get("problems") if isinstance(verdict.get("problems"), list) else []
        instruction = str(verdict.get("instruction") or "Reconsider the terminal action.")
        feedback.append(instruction + (" Problems: " + "; ".join(map(str, problems[:5])) if problems else ""))
        return False

    def run(self) -> list[JsonObject]:
        if self.episode.finished:
            return self.dialogue
        self.mission.replace(self.reader.initial(self.episode.query, self.trail.opening))
        feedback: list[str] = []
        environment_failures = 0
        while not self.episode.finished and self.episode.remaining() > 0:
            action: ToolCall | None = None
            for _ in range(2):
                proposal = self._candidate(feedback)
                if proposal is None:
                    break
                if self._approved(proposal, feedback):
                    action = proposal
                    break
            if action is None:
                action = self._fallback_terminal()
            if action is None:
                self.dialogue.append({"role": "assistant", "content": "ICDI could not identify a valid reachable action."})
                break
            result = self.world.call(action[0], action[1])
            if result is None:
                environment_failures += 1
                feedback.append("The environment request failed without an observation; retry or choose a safe terminal action.")
                if environment_failures >= 3:
                    self.dialogue.append({"role": "assistant", "content": "ICDI stopped after repeated environment failures."})
                    break
                continue
            environment_failures = 0
            self._record(action, result)
            self._refresh_after_message(action, result)
            feedback.clear()
        return self.dialogue


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one ICDI episode through the public dynamic-world contract."""

    agent = ICDI(problem_data)
    try:
        return agent.run()
    except Exception as exc:
        agent.dialogue.append(
            {
                "role": "assistant",
                "content": f"ICDI stopped after {type(exc).__name__}.",
            }
        )
        return agent.dialogue


__all__ = ["agent_main"]
