    def valid_notice(self, answer):
        if not isinstance(answer, dict) or answer.get("kind") not in ("change", "error", "other"):
            return False
        if answer["kind"] != "change":
            return True
        if not isinstance(answer.get("summary"), str):
            return False
        if not all(isinstance(answer.get(k), bool) for k in ("sold_out", "price_moved")):
            return False
        if "budget" not in answer or (answer["budget"] is not None and (not is_num(answer["budget"]) or answer["budget"] <= 0)):
            return False
        gone = answer.get("gone")
        if not isinstance(gone, list) or any(not isinstance(g, (str, int)) or isinstance(g, bool) for g in gone):
            return False
        for field in ("add", "drop"):
            entries = answer.get(field)
            if not isinstance(entries, list):
                return False
            for entry in entries:
                if not isinstance(entry, dict) or not isinstance(entry.get("need"), str) or not entry["need"].strip():
                    return False
                target = entry.get("item")
                number = count_of(target)
                if number is None and isinstance(target, str):
                    matches = [n for n, item in enumerate(self.items, 1) if words(item.name) == words(target)]
                    number = matches[0] if len(matches) == 1 else None
                if number is None or not 1 <= number <= len(self.items):
                    return False
                entry["item"] = number
        return True

    def cart_review_context(self, ids, planned=False):
        quantities = [it.quantity for it in self.items] if planned else [self.held(i) for i in ids]
        return dump([self.ledger(), quantities, [[i, self.rows.get(i), self.full.get(i), self.sold_out(i)] for i in ids]])

    def terminal_recovery_hold(self, order):
        if self.pending_notices:
            text = self.read_note([])
            if text:
                return "A pending notice changes the task: " + text + ". Apply it before ending."
            if self.pending_notices:
                return "A public store notice remains uninterpreted. Resolve it before ending."
        if not order:
            return ""
        ids = self.cart_ids()
        lost = [i for i in ids if self.sold_out(i)]
        if lost:
            return "Replace the unavailable cart lines before ordering: " + ", ".join(lost) + "."
        key = "|".join(sorted(ids)) + "#" + str(self.version)
        if self.verdicts.get(key) is False:
            previous = self.review_contexts.get(key)
            if previous is not None and previous != self.cart_review_context(ids):
                self.cart_check()
            if self.verdicts.get(key) is False:
                return self.review_messages.get(key) or "The current cart has a valid negative review. Repair it before ordering."
        return ""

    def env(self, tool, args):
        self.recovery_blocked = ""
        if self.finished or self.left() <= 0:
            return None
        order = self.is_order(tool, args)
        if order or self.is_decline(tool, args):
            text = self.terminal_recovery_hold(order)
            if text:
                self.recovery_blocked = text
                self.trace.append({"kind": "recovery_hold", "detail": text})
                return None
        return super().env(tool, args)
