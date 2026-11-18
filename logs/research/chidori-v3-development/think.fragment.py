    def think(self, messages, models, cap, tools=None):
        self.request_denied = False
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
            self.request_denied = True
            budget = self.inference_budget
            self.no_credit = budget.breached or not budget.prices or budget.remaining() == 0
            self.budget_locally_stopped = self.no_credit
            self.trace.append({"kind": "request_budget_denied", "reasons": denied,
                               "remaining_usd": budget.remaining() / 1000000000})
            if self.no_credit:
                self.trace.append({"kind": "local_inference_stop", "reasons": denied})
        return None
