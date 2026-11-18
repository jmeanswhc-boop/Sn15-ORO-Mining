PROCEDURE_PROMPT = """Carry out only the procedures explicitly required by the public instruction.
The local IDs in DUE must each occur exactly once in steps. An action proposal is not evidence.
Use {\"steps\":[{\"id\":\"local duty ID\",\"done\":false,\"evidence\":[executed step numbers]}],
\"call\":{\"duty_id\":\"one pending ID\",\"tool\":\"supplied nonterminal tool\",\"args\":{}},
\"hold\":\"required cart repair or empty string\",\"needs\":[]}.
Use call:null only if all duties are performed or no permitted action can perform the next one.
Mark done only when the successful recorded actions and replies actually perform that duty.
A product-property question does not obtain purchase approval. An approval must concern the current cart.
Use a separate message for independent duties unless their stated shared group permits one message.
Messages are concise and limited to the requested procedure. Read the shopper's reply for changed needs
or a refused approval. Report a changed requirement through needs or hold before allowing an order.
Do not select products, change the cart, order, or decline. No private grading information is available.
"""


class ProcedureControls:
    def __init__(self, problem_data):
        super().__init__(problem_data)
        self.procedures_valid = False
        self.procedure_busy = False
        self.procedure_events = {}
        self.procedure_cart_signature = self.cart_mark() + "#" + str(self.version)
        self.procedure_cart_changed_at = -1
        self.procedure_first_selection = None
        self.procedure_attempts = {}

    def read_brief(self):
        super().read_brief()
        raw = self.brief.get("process")
        duties = []
        valid = isinstance(raw, list)
        request = " ".join(self.request.split())
        for entry in raw if isinstance(raw, list) else []:
            if not isinstance(entry, dict):
                valid = False
                continue
            source = entry.get("source")
            source = " ".join(source.split()) if isinstance(source, str) else ""
            stage = entry.get("when")
            messages = count_of(entry.get("messages"))
            actions = count_of(entry.get("actions"))
            text = str(entry.get("step") or "").strip()
            if not source or source not in request or not text or stage not in ("choose", "order", "end") or messages is None or messages < 0:
                valid = False
                continue
            group = entry.get("group")
            combined = entry.get("combine_source")
            combined = " ".join(combined.split()) if isinstance(combined, str) else ""
            if not isinstance(group, str) or not group or not combined or combined not in request:
                group = ""
            duties.append({"id": "d%d" % (len(duties) + 1), "text": text, "source": source,
                           "stage": stage, "messages": messages, "actions": max(1, messages, actions or 1),
                           "cart_dependent": entry.get("cart_dependent") is not False if stage != "choose" else False,
                           "group": group, "evidence": [], "completed": False, "signature": None})
        self.duties = duties
        self.procedures_valid = valid
        self.refresh_procedure_reserve()
        self.trace.append({"kind": "procedures_parsed", "valid": valid, "count": len(duties)})

    def sync_procedures(self):
        signature = self.cart_mark() + "#" + str(self.version)
        if signature != self.procedure_cart_signature:
            self.procedure_cart_signature = signature
            self.procedure_cart_changed_at = self.step
        for duty in self.duties:
            if duty.get("cart_dependent") and duty.get("signature") != signature:
                duty["completed"] = False
                duty["evidence"] = []

    def pending_procedures(self, stage):
        self.sync_procedures()
        return [d for d in self.duties if d["stage"] == stage and not d.get("completed")]

    def procedure_resources(self, stage, field):
        groups = {}
        for duty in self.pending_procedures(stage):
            key = duty.get("group") or duty["id"]
            groups[key] = max(groups.get(key, 0), duty[field])
        return sum(groups.values())

    def refresh_procedure_reserve(self):
        self.msg_reserve = self.procedure_resources("choose", "messages") + max(
            self.procedure_resources("order", "messages"), self.procedure_resources("end", "messages"))

    def free_msgs(self):
        self.refresh_procedure_reserve()
        return max(0, self.msg_limit - self.msg_used - self.msg_reserve)

    def reserve(self):
        return 1 + self.procedure_resources("choose", "actions") + max(
            self.procedure_resources("order", "actions"), self.procedure_resources("end", "actions"))

    def gate_due(self, stage):
        return not self.procedures_valid or bool(self.pending_procedures(stage))

    def gate(self, stage):
        return self.with_procedure_budget(self.procedure_gate, stage)

    def procedure_gate(self, stage):
        if not self.procedures_valid:
            return False, "The required-procedure parse is unresolved."
        due = self.pending_procedures(stage)
        if not due:
            return True, ""
        if self.finished or self.procedure_busy:
            return False, "The procedure cannot run in the current state."
        if stage == "choose" and self.procedure_first_selection is not None:
            return False, "A required before-selection step missed its boundary."
        key = (stage, self.procedure_cart_signature, self.step, self.msg_used)
        if self.procedure_attempts.get(key, 0) >= 2:
            return False, "The procedure is unresolved with no new evidence."
        self.procedure_attempts[key] = self.procedure_attempts.get(key, 0) + 1
        self.procedure_busy = True
        performed = []
        seen_calls = set()
        try:
            rounds = min(CALL_CAP, 2 + sum(d["actions"] for d in due))
            for _ in range(rounds):
                due = self.pending_procedures(stage)
                if not due:
                    self.refresh_procedure_reserve()
                    return True, "\n".join(performed)
                records = []
                for event in self.procedure_events.values():
                    relevant = [d for d in due if not d["cart_dependent"] or
                                (event["signature"] == self.procedure_cart_signature and event["step"] > self.procedure_cart_changed_at)]
                    if relevant:
                        records.append(event)
                allowed = [t for t in self.tools if t["function"]["name"] not in
                           (self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("cart"))]
                prompt = PROCEDURE_PROMPT + "\nREQUEST:\n" + self.request + "\nDUE:\n" + dump(due)
                prompt += "\nCURRENT CART:\n" + self.cart_text() + "\nSUCCESSFUL RECORDS:\n" + dump(records)
                prompt += "\nTOOLS:\n" + dump(allowed) + "\nMESSAGES LEFT: %d; STEPS LEFT: %d" % (self.msg_limit - self.msg_used, self.left())
                answer = self.json_ask(prompt, [LIGHT], max(700, 200 + 180 * len(due)))
                entries = answer.get("steps") if isinstance(answer, dict) else None
                if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
                    return False, "The procedure decision is missing or malformed."
                ids = [e.get("id") for e in entries]
                if any(not isinstance(i, str) for i in ids) or len(ids) != len(due) or set(ids) != {d["id"] for d in due}:
                    return False, "The procedure decision does not cover every required ID."
                changed = []
                self.fold_reading({"needs": answer.get("needs")}, changed)
                hold = str(answer.get("hold") or "").strip()
                if changed or hold:
                    self.version += 1
                    self.sync_procedures()
                    return False, "The shopper requires a repair: " + "; ".join(changed + ([hold] if hold else []))
                claimed = {}
                for duty in self.duties:
                    if duty.get("completed"):
                        for step in duty.get("evidence", []):
                            claimed[step] = duty.get("group") or duty["id"]
                for duty in due:
                    entry = next(e for e in entries if e["id"] == duty["id"])
                    evidence = entry.get("evidence")
                    if entry.get("done") is not True or not isinstance(evidence, list) or not evidence:
                        continue
                    if any(not isinstance(s, int) or isinstance(s, bool) or s not in self.procedure_events for s in evidence):
                        continue
                    events = [self.procedure_events[s] for s in set(evidence)]
                    identity = duty.get("group") or duty["id"]
                    if any(s in claimed and claimed[s] != identity for s in evidence):
                        continue
                    if duty["cart_dependent"] and any(e["signature"] != self.procedure_cart_signature or e["step"] <= self.procedure_cart_changed_at for e in events):
                        continue
                    if duty["stage"] == "choose" and self.procedure_first_selection is not None and any(e["step"] >= self.procedure_first_selection for e in events):
                        continue
                    if sum(e["tool"] == self.roles.get("ask") for e in events) < duty["messages"] or len(events) < duty["actions"]:
                        continue
                    duty.update(completed=True, evidence=sorted(set(evidence)), signature=self.procedure_cart_signature)
                    for step in evidence:
                        claimed[step] = identity
                if not self.pending_procedures(stage):
                    self.refresh_procedure_reserve()
                    return True, "\n".join(performed)
                call = answer.get("call")
                if not isinstance(call, dict) or not isinstance(call.get("duty_id"), str) or call.get("duty_id") not in {d["id"] for d in self.pending_procedures(stage)}:
                    return False, "A required procedure remains unperformed."
                tool = call.get("tool")
                args = call.get("args")
                if not isinstance(tool, str) or tool not in self.schemas or tool in (self.roles.get("buy"), self.roles.get("refuse"), self.roles.get("cart")) or not isinstance(args, dict):
                    return False, "The proposed procedure action is not permitted."
                args, faults = shape_args(args, self.schemas[tool])
                mark = tool + dump(args)
                if faults or mark in seen_calls or self.left() < 2:
                    return False, "No valid new procedure action fits the remaining steps."
                if tool == self.roles.get("ask"):
                    if self.msg_used >= self.msg_limit:
                        return False, "The required message allowance is exhausted."
                    text = strings(args)
                    if not text:
                        return False, "The procedure message is empty."
                    sent = self.send([(1, "", max(text, key=len))], 1)
                    if not sent:
                        return False, "The procedure message was not sent."
                    self.gate_msgs += 1
                    performed.append("Performed required message at step %d." % self.step)
                else:
                    result = self.env(tool, args)
                    if result is None:
                        return False, "The procedure action did not return an observation."
                    self.absorb(result.get("observation"))
                    performed.append("Performed required tool %s at step %d." % (tool, self.step))
                seen_calls.add(mark)
            return False, "Required procedures remain unresolved after bounded attempts."
        finally:
            self.procedure_busy = False

    def opening(self):
        if self.procedures_valid and self.pending_procedures("choose"):
            self.gate("choose")
        return super().opening()

    def env(self, tool, args):
        if self.finished or self.left() <= 0:
            return None
        order = self.is_order(tool, args)
        decline = self.is_decline(tool, args)
        if (order and decline) or (tool in (self.roles.get("buy"), self.roles.get("refuse")) and not order and not decline):
            self.trace.append({"kind": "ambiguous_finish_blocked", "tool": tool})
            return None
        selecting = tool == self.roles.get("cart") or order
        if not self.procedure_busy:
            stages = (["choose"] if selecting else []) + (["order"] if order else ["end"] if decline else [])
            for stage in stages:
                go, text = self.gate(stage)
                if not go:
                    self.trace.append({"kind": "procedure_blocked", "stage": stage, "detail": text})
                    self.queued.append(text)
                    return None
            if not order and not decline and self.duties:
                reserve = self.reserve()
                if tool == self.roles.get("cart"):
                    # A changed basket can require a fresh approval even after a cached pass.
                    pending = [d for d in self.duties if d["stage"] == "order" and d["cart_dependent"]]
                    grouped = {}
                    for duty in pending:
                        key = duty.get("group") or duty["id"]
                        grouped[key] = max(grouped.get(key, 0), duty["actions"])
                    reserve = max(reserve, 1 + sum(grouped.values()))
                if self.left() <= reserve or (tool == self.roles.get("ask") and self.free_msgs() <= 0):
                    self.trace.append({"kind": "procedure_resources_reserved", "tool": tool})
                    return None
        result = super().env(tool, args)
        if result is None:
            return None
        observation = result.get("observation")
        successful = not failure_only(observation) and not (isinstance(observation, dict) and observation.get("error"))
        if successful:
            self.sync_procedures()
            if self.step not in self.procedure_events:
                shown = self.spoken(observation) if tool == self.roles.get("ask") else dump(observation)[:400]
                self.procedure_events[self.step] = {"step": self.step, "tool": tool, "args": dict(args),
                                                   "observation": shown, "signature": self.procedure_cart_signature}
            if selecting and self.procedure_first_selection is None:
                self.procedure_first_selection = self.step
        return result



