# Monetary amounts are integral nanodollars; prices are read through the proxy.
AGENT_USD_LIMIT = "0.035"
PROCEDURE_USD_RESERVE = "0.010"
MIN_REPLY_TOKENS = 512


def money_units(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0:
            return None
        return int((number * 1000000000).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, ValueError, OverflowError):
        return None


class BudgetReply:
    def __init__(self, reason):
        self.data = None
        self.error = {"status": 0, "body": reason}


class InferenceBudget:
    def __init__(self):
        self.limit = money_units(AGENT_USD_LIMIT)
        self.protected = money_units(PROCEDURE_USD_RESERVE)
        self.actual = 0
        self.uncertain = 0
        self.pending = 0
        self.attempts = 0
        self.breached = False
        self.prices = None

    def remaining(self, protected=0):
        return max(0, self.limit - self.actual - self.uncertain - self.pending - protected)

    def load_prices(self, catalog):
        self.prices = {}
        rows = catalog.get("data") if isinstance(catalog, dict) else None
        if not isinstance(rows, list):
            return
        for row in rows:
            if not isinstance(row, dict) or row.get("id") not in (LIGHT, HEAVY, SPARE):
                continue
            pricing = row.get("pricing")
            if not isinstance(pricing, dict):
                continue
            values = {k: money_units(v) for k, v in pricing.items()}
            if values.get("prompt") is None or values.get("completion") is None:
                continue
            used = {"prompt", "completion", "request", "input_cache_read", "input_cache_write", "internal_reasoning"}
            unused_text = {"image", "web_search", "audio", "input_audio_cache", "input_audio", "output_audio"}
            if any(v is None or (k not in used and k not in unused_text and v > 0) for k, v in values.items()):
                continue
            top = row.get("top_provider") or {}
            top = top if isinstance(top, dict) else {}
            context = count_of(top.get("context_length")) or count_of(row.get("context_length"))
            completion = count_of(top.get("max_completion_tokens"))
            if context is None or context <= 0:
                continue
            self.prices[row["id"]] = {
                "input": max(values["prompt"], values.get("input_cache_read", 0), values.get("input_cache_write", 0)),
                "plain_input": values["prompt"],
                "output": values["completion"] + values.get("internal_reasoning", 0),
                "request": values.get("request", 0),
                "context": context,
                "completion": completion,
            }

    def quote(self, body, protected=0):
        price = (self.prices or {}).get(body.get("model"))
        if price is None:
            return None, "pricing_unavailable"
        messages = body.get("messages")
        if not isinstance(messages, list):
            return None, "unsupported_message"
        for message in messages:
            if not isinstance(message, dict):
                return None, "unsupported_message"
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str) for p in content):
                    return None, "unsupported_message"
            elif content is not None and not isinstance(content, str):
                return None, "unsupported_message"
        # A conservative estimate, not a certified tokenizer or routed-price bound.
        prompt = len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 512 + 64 * len(messages) + 128 * len(body.get("tools") or [])
        requested = count_of(body.get("max_tokens"))
        if requested is None or requested <= 0 or prompt >= price["context"]:
            return None, "context_or_output_limit"
        cap = min(requested, price["context"] - prompt)
        if price["completion"] is not None:
            cap = min(cap, price["completion"])
        available = self.remaining(protected)
        input_rate = price["input"]
        cached = any(isinstance(m.get("content"), list) and any(p.get("cache_control") for p in m["content"]) for m in messages)
        if cached:
            input_rate = max(input_rate, (price["plain_input"] * 5 + 3) // 4)
        base = prompt * input_rate + price["request"]
        raw_room = available * 4 // 5 - base
        if raw_room < 0:
            return None, "allocation_exhausted"
        if price["output"]:
            cap = min(cap, raw_room // price["output"])
        if cap < min(MIN_REPLY_TOKENS, requested):
            return None, "allocation_exhausted"
        reserve = ((base + cap * price["output"]) * 5 + 3) // 4
        if reserve > available:
            return None, "allocation_exhausted"
        return {"units": reserve, "cap": cap, "prompt_estimate": prompt}, ""

    def settle(self, quote, data):
        self.pending -= quote["units"]
        usage = data.get("usage") if isinstance(data, dict) else None
        reported = usage.get("cost") if isinstance(usage, dict) else None
        cost = money_units(reported) if isinstance(reported, (int, float)) and not isinstance(reported, bool) else None
        if cost is None:
            self.uncertain += quote["units"]
        else:
            self.actual += cost
            if cost > quote["units"]:
                self.breached = True
        tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > quote["prompt_estimate"]:
            self.breached = True
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > quote["cap"]:
            self.breached = True
        return cost


class BudgetControls:
    def __init__(self, problem_data):
        super().__init__(problem_data)
        self.inference_budget = InferenceBudget()
        self.budget_mode = "work"
        self.brief_read = False
        self.last_budget_denial = ""
        self.budget_locally_stopped = False

    def budget_post(self, body):
        budget = self.inference_budget
        self.last_budget_denial = ""
        if self.no_credit or budget.breached or budget.attempts >= CALL_CAP or self.room() < CALL_FLOOR:
            self.last_budget_denial = "inference_stopped"
            return BudgetReply(self.last_budget_denial)
        if budget.prices is None:
            try:
                catalog = model_client(min(10, self.room())).get("/inference/models")
            except Exception:
                catalog = None
            budget.load_prices(catalog)
        protected = 0
        if self.budget_mode != "procedure" and (not self.brief_read or self.duties or not getattr(self, "procedures_valid", True)):
            protected = budget.protected
        if protected and self.budget_mode != "procedure":
            required = 2 + sum(max(1, d.get("actions", 1)) for d in self.duties)
            if budget.attempts >= CALL_CAP - min(CALL_CAP // 2, required):
                self.last_budget_denial = "procedure_attempts_reserved"
                return BudgetReply(self.last_budget_denial)
        quote, reason = budget.quote(body, protected)
        if quote is None:
            self.last_budget_denial = reason
            self.trace.append({"kind": "budget_denied", "model": body.get("model"), "reason": reason})
            return BudgetReply(reason)
        body = dict(body)
        body["max_tokens"] = quote["cap"]
        budget.pending += quote["units"]
        budget.attempts += 1
        self.calls = budget.attempts
        try:
            result = model_client(self.room()).post_verbose("/inference/chat/completions", body)
        except Exception:
            result = BudgetReply("uncertain_transport_failure")
            result.error["status"] = None
        cost = budget.settle(quote, result.data)
        self.spend = (budget.actual + budget.uncertain) / 1000000000
        self.trace.append({"kind": "inference_budget", "model": body.get("model"), "attempt": budget.attempts,
                           "reserved_usd": quote["units"] / 1000000000, "reported_usd": None if cost is None else cost / 1000000000,
                           "uncertain_usd": budget.uncertain / 1000000000, "cap": quote["cap"],
                           "envelope_breached": budget.breached})
        if budget.breached:
            self.no_credit = True
            self.budget_locally_stopped = False
        error = result.error if isinstance(result.error, dict) else {}
        data_error = result.data.get("error") if isinstance(result.data, dict) else None
        status = count_of(data_error.get("code")) if isinstance(data_error, dict) else error.get("status")
        if status == 402:
            self.no_credit = True
            self.budget_locally_stopped = False
        return result

    def think(self, messages, models, cap, tools=None):
        if self.no_credit:
            return None
        denied = []
        for model in models:
            if self.calls >= CALL_CAP or self.room() < CALL_FLOOR or self.no_credit:
                return None
            tries = 0
            while True:
                msg, status = self.one_call(model, messages, cap, tools)
                if msg is not None:
                    return msg
                if self.last_budget_denial:
                    denied.append(self.last_budget_denial)
                    break
                if status == 402 or self.no_credit:
                    return None
                retryable = status is None or (isinstance(status, int) and (status >= 500 or bool(_RETRY_STATUS.fullmatch(str(status)))))
                limit = 1 if status is None or status == 408 else len(WAITS)
                if not retryable or tries >= limit or self.calls >= CALL_CAP:
                    break
                wait = WAITS[tries]
                if self.waited + wait > WAIT_BUDGET or self.room() - 1.2 * wait < CALL_FLOOR:
                    break
                time.sleep(wait * random.uniform(0.8, 1.2))
                self.waited += wait
                tries += 1
        if denied:
            self.no_credit = True
            self.budget_locally_stopped = True
            self.trace.append({"kind": "local_inference_stop", "reasons": denied})
        return None

    def read_brief(self):
        super().read_brief()
        self.brief_read = True

    def gate(self, stage):
        return self.with_procedure_budget(super().gate, stage)

    def with_procedure_budget(self, callback, *args):
        previous = self.budget_mode
        was_locally_stopped = self.no_credit and self.budget_locally_stopped
        self.budget_mode = "procedure"
        if was_locally_stopped and not self.inference_budget.breached:
            self.no_credit = False
        try:
            return callback(*args)
        finally:
            self.budget_mode = previous
            if was_locally_stopped and not self.no_credit:
                self.no_credit = True
                self.budget_locally_stopped = True



