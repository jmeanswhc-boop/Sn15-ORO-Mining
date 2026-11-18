from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

MAIN_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"
SETUP_MODEL = "anthropic/claude-sonnet-5.5"
JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
READ_MODEL = "anthropic/claude-haiku-5.5"

TASK_SECONDS = 265.0
WRAP_UP_SECONDS = 70.0
FORCE_SECONDS = 28.0
OBS_CHARS = 16000
HISTORY_CHARS = 35000
COMPACT_BLOCK = 20000
KEEP_RECENT = 6
MAX_TURNS = 80
RETRY_PAUSES = (1.5, 3.0, None)
RUNAWAY_USD = 0.10
SHELF_HOLDS = 2
OPEN_BEFORE_ASK = 8

SYSTEM_PROMPT = """Role: you act as a scrupulous purchasing agent for one customer of a simulated shop, and calling the tools you were given is the only thing you can do. Scoring is binary. The point is earned only by a perfect end state: the purchased product honours each condition the customer wrote, each hidden wish they disclose under questioning, and each change they make mid-task; units and spending limit are right; when they want the lowest price (or the top choice by some measure) that is what you picked; and each instruction in their text was obeyed. Should no product in the shop satisfy all of that, the correct ending is a refusal to buy, carried out exactly as their text prescribes.

Input: the customer's text, a draft breakdown of it, a status line, and whatever your tool calls return.

Rules:
1. Any tool call, failed or erroring ones included, burns one step out of a fixed allowance. Reasoning costs nothing; steps are scarce.
2. Independent calls (e.g. opening several offers' detail pages together) may share a reply and run in sequence. A checkout or a decline always goes out on its own.
3. Tools differ per shop in names, arguments and output layout. Read each description and JSON schema closely and use precisely the argument names and allowed values it lists. Let your first outputs teach you which tool exposes which facts (cost, availability, detailed attributes).
4. Find out soon which category labels the shop attaches to the wanted product type (offers display them). Among adjacent labels, the correct one is the label agreeing with the customer's phrasing taken whole; search and buy inside it.
5. Searches may be crippled: a cap on visible rows or pages, ordering by id rather than relevance or cost, all words mandatory, filters valid in only one mode. Hold the stated total up against the rows actually shown and keep narrowing (more precise words, category or price filters, further pages) until nothing relevant remains unseen.
6. The customer has unwritten wishes and discloses one only when a message asks about that attribute. Only the FIRST point of any message gets answered, so each message covers precisely ONE attribute: a brief, simple, warm question to the customer that names it (one or two sample values allowed). No commands, no talk of lists, labels, positions or rules, never two questions in one.
7. If they answer that it doesn't matter, they have no preference or anything goes, that attribute is open. If they name a preference or a limit, it binds from then on, verbatim (numeric limits too).
8. With a planner active, questions get chosen and sent for you once you have viewed details of two or three offers of the wanted type, so do that early. Replies then show up together with their meaning; treat each disclosed wish as binding and search again combining it with the written values.
9. When asking falls to you: view details of two or three offers first, next ask about attributes the story implies (the user of the item, the look it must have, the place or manner of use), then about attributes the offers carry that vary across candidates. Skip anything the text already decides, and stay inside the message allowance.
10. The text can prescribe working methods: how many offers to inspect or compare before deciding, what to verify, what to tell the customer, what a remark must contain, how to decline. They are scored like conditions; carry each one out literally and fully. Looking at or opening N offers means N fully suitable offers, each opened through the details tool.
11. A question from the customer to you (say, confirming a change) gets answered automatically while messages remain; the change binds you regardless.
12. Things merely hoped for ("ideally", "would be nice", "if possible") are not conditions: they only break ties among offers that already pass everything, and never outrank a lowest-price instruction.
13. Where the text leaves open something only the customer knows (for instance there is no price limit, or two wishes may clash and the customer must rank them), ask before deciding.
14. Mid-task changes happen: a fresh spending limit (up or down), a condition removed, swapped or added, a price increase, a product gone. A single customer line can bundle several. Such lines can hide in any tool output (an extra field of sentences) or ride along with a checkout attempt. Each binds from then on: test your choice and reserves against it, and if lowest price or top choice was requested, choose again from the new passing set.
15. Checkout can be REFUSED: the output then contains a "refusal" field, even if the inner output looks fine. Nothing ends and the cart stays as it was. Read the refusal, apply every change in it, take out of the cart whatever no longer passes (e.g. a sold-out product), put in the next verified candidate, recheck availability and cost, and try checkout again.
16. Products can also vanish or get pricier with no warning, hence availability and cost are rechecked just before each checkout; a product whose current cost times units exceeds the current limit has stopped qualifying.
17. Judging an offer decides your score. A condition holds only if the offer's own attribute for it shows the value. Key spelling varies (spaces, dots, underscores, prefixes) and one attribute may appear under several keys; any key naming it qualifies. Values match by sense: a single item of a comma-separated list suffices; singular equals plural; near synonyms qualify; letter case is irrelevant. Shared word beginnings are no match, nor is a wider or other class (girls is not women; plain unisex is not female).
18. Fall back on title or description only if the offer has no key at all for that condition. Should its own attribute say otherwise, it fails even when title or description shows the value. A condition that no offer in the shop states anywhere (not under any key, title or description) cannot be checked here, so on its own it never rules an offer out.
19. Values under some other, related attribute do not qualify (a length field cannot satisfy a style condition; lining material cannot satisfy material).
20. Compare numbers numerically after unit conversion; limits are inclusive; a range qualifies only if all of it respects the limit.
21. Obey exclusions ("leave out anything with X") and closed sets ("A or B and nothing else").
22. The product type is satisfied by the offer's category; prefer the category named like the text as a whole, and fall back to a narrower or adjacent category holding the same kind of product only when the right category holds no full fit; never require an attribute that echoes the category name, and never rule out an offer of the right category because its title or one of its own type attributes names another kind of item. Still, the product must be of the wanted type: if the shop labels offers by category, it belongs in the category whose label fits the customer's phrasing as a whole (asking for "A or B" points to a category named "A & B", not a narrower neighbour named "B"). It must be available when the text wants goods on hand, and cost times units may not exceed the limit (exactly equal is fine).
23. An attribute value listing several options, one of them the required one (a comma list), is a clean match and not conflicting evidence. Evidence conflicts only when the offer's own keys for that attribute give different, incompatible values; only then, and only with no lowest-price or top-choice rule to decide, favour the candidate with clean evidence.
24. Lowest price requested: the purchase must be the cheapest product among ALL products meeting every condition. Seeing one suitable product proves nothing. Use price filters when available, drop the price ceiling until all remaining rows show, sort visible candidates by cost yourself and verify upward from the cheapest; the first full fit, with every cheaper one examined and rejected, wins. Keep the following fits as reserves.
25. Top choice requested by some measure (most, largest, longest, highest and so on): weigh ALL suitable products on that measure likewise and take the extreme. An offer stating no value for the measure cannot win.
26. Maintain a ranked set of verified fits with a reserve, since the favourite may sell out during checkout.
27. Before any checkout: one cart line per wanted product (a single line when one product is wanted), each the chosen offer in its required units, nothing else (remove other lines); after your last cart change you reread its availability and cost, and it is available and within the limit.
28. Decline only when an exhaustive search leaves nothing that meets every condition, following the text's decline rule to the letter. If the text forbids offering substitutes, the decline remark must not name, describe or hint at any offer: no ids, no titles, no prices.
29. Pacing: send the first checkout while seven or more steps remain so a refusal can still be repaired. At 7 remaining steps only cart, stock-check and finishing tools stay usable. Commit once; flipping the cart back and forth wastes a step per change.

Output: a reply always includes one or more tool calls, with brief text: current conditions, verified fits with cost and availability, and your next action."""

SETUP_PROMPT = """Role: you analyse a customer's text and the shop's tool catalogue on behalf of a purchasing agent.

Input: the catalogue appears right after these instructions; the customer's text (and maybe an opening observation) follows it.

Rules:
1. Reply with ONE JSON object on a single line (no indentation), nothing else, using exactly these keys.
2. "product": the product type wanted, phrased as in the text.
3. "quantity": units of one offer to buy (an integer; 1 unless several are asked for). Things normally sold as a single unit, like a pair of shoes or a set, are quantity 1. "quantity_span": when quantity is above 1, the exact stretch of the text that asks for that many units, copied character for character; "" otherwise.
4. "products": when the text asks for more than one kind of product in one order, a list with one entry per kind: {"product": its type as phrased, "quantity": its units, "requirements": its own conditions in the format below}; a list with a single entry otherwise. "requirements": a list of strings, one per firm written condition on the product (with several products, start each with its product, as "<product> - property: constraint"), each formatted "property: constraint" and keeping the customer's exact value, limit, exclusion or closed list. Price, stock, quantity, budgets and mere wishes do not belong here, nor does the product type or category: words forming part of the category name (a season, an occasion or an audience inside it) are covered by the category and are not separate conditions.
5. "price_cap": the total spending ceiling as a number, or null if absent.
5b. "references": conditions whose value is not given in the text but has to be read off a different listing, which the text identifies by its seller and a few of its attributes. One entry each: {"product": the wanted product it constrains, "property": the attribute whose value must be read, "range": the category of that other listing, "seller": the seller named, "identify": the attributes that single it out, "tolerance": the allowed difference as written, or ""}. An empty list if none. Keep such a condition out of "requirements" until its value is known.
6. "currency": the money unit used in the text, or null.
7. "in_stock_only": true if the text accepts only goods actually available or on hand.
8. "priority": "cheapest" only for an explicit demand for the cheapest or lowest-priced suitable option; "best" only for an explicit demand for the best by some measure; "none" otherwise. Notes that cash is short or every penny matters merely justify the cap and do not demand the cheapest.
9. "priority_detail": the measure behind "best"; "" in all other cases.
10. "message_budget": an integer count of messages the customer permits, or null when unspecified.
11. "step_budget": an integer count of moves or steps permitted, or null when unspecified.
12. "rules": every procedural instruction for the agent (how many offers to inspect, what to verify or report, how to decline, what to avoid, what to include), quoted verbatim. Statements about what the customer will do personally are story, not rules.
13. "soft_wants": things wanted without strict need (worded ideally, preferably, would be nice, if possible); never put them under "requirements".
14. "ask_first": things only the customer can settle that should be asked before choosing, e.g. the budget when no ceiling is given, which of two wishes wins when the text says both may be impossible (then put both wishes in soft_wants too, not in requirements), or anything the text says to confirm with the customer. An empty list if none.
15. Every sentence must be accounted for: a requirement, the priority, a soft want, a rule, a budget, a hint, or mere story. When unsure whether something is a rule, list it under "rules".
16. "decline_no_names": true when the decline rule forbids offering, naming or suggesting substitutes or alternatives; false when it asks for something to be named; null when no decline rule exists.
17. "roles": an object mapping each role to an exact tool name from the catalogue, or null where nothing fits:
  "search", "details", "view", "message", "cart", "cart_view", "order", "decline".
  "message" = the tool for messaging the customer; "cart" = the tool adding to or removing from the cart; "order" = the tool that purchases or checks out; "decline" = the tool that closes the task without a purchase.
18. "finish_values": for the order tool and the decline tool, each argument that only accepts a fixed list of values, mapped to the allowed value that carries out that tool's own job (buying for "order", ending with no purchase for "decline"), read from the tool's and the argument's descriptions; written {"order": {"<argument>": "<value>"}, "decline": {"<argument>": "<value>"}}, with {} where no such argument exists.

Output: that single-line JSON object.
"""

INTERPRET_PROMPT = """Role: you decode a customer's latest words for a purchasing agent.

Input: below come the customer's original text, the conditions so far, earlier exchanges, the agent's question if one was asked, and the new customer words.

Rules:
1. "meaning": one brief sentence on what the customer intends.
2. "needs": newly disclosed firm conditions on the product, each formatted "property: constraint", keeping the exact value, limit, exclusion or closed list given. The property is the one the customer's own words name; when they answer a question about one attribute by setting another, the condition belongs to the attribute they set, never to the one the question named.
3. "no_preference": attributes the customer declared themselves indifferent about.
4. "dropped": former conditions the customer has given up.
5. "new_budget": the updated total spending ceiling as a number, or null.
6. "unavailable": true if the customer reports that a product can no longer be purchased (sold out, gone, out of stock, button disabled); false otherwise.
7. "requirements_now": the complete current list of firm product conditions after everything said so far, rebuilt from the original text, previous customer lines and this one, each formatted "property: constraint" (with several products wanted, "<product> - property: constraint", naming the product the customer or the question was about). Retain each earlier condition unless dropped or replaced; for a replaced value keep only the latest; append every newly disclosed need. Exclude price, stock, quantity and the product category.
8. "asked_us": true only when the customer's new words put a question to the agent or ask the agent to confirm something; false otherwise. "reply": when asked_us is true, a brief simple answer to return that addresses it and states the change in substance (e.g. confirming the new value); else "".
9. One line may contain several changes; list them all. A swapped condition counts as dropped (the old one) and as a need (the new one). If the customer says which of two wishes matters more, the winning wish is a need. Any budget the customer gives is new_budget. A mere hope ("ideally", "would be nice") is not a need.
10. Read literally yet generously. Folksy, figurative or roundabout wording naming a value means a firm wish for that value; treat a value as excluded only on a clear instruction to avoid it. A number with a comparison (at most, at least, under, over, no more than) is a limit and keeps its unit. A new budget supersedes the previous one, higher or lower, even if announced as good news.
11. When the customer merely says it doesn't matter, they have no preference or anything works, put the attribute that was asked about into no_preference.

Output: ONE JSON object holding exactly these keys, nothing else."""

VERIFY_PROMPT = """Role: you are the final gate before a purchasing agent checks out. The point is earned only if the purchased product satisfies every condition listed below.

Input: the customer's text, the current conditions, the product id, the records of it seen in the shop, and possibly the newest cart view plus labels recurring across offers.

Rules:
1. Base the judgement only on the records provided.
2. A condition holds only if the product's own attribute for it shows the value. Key spellings vary (spaces, dots, underscores, prefixes) and one attribute may sit under several keys; any key naming it qualifies. Values match by sense: one item of a comma list suffices, singular/plural and near synonyms qualify, letter case is irrelevant. Lookalike words and wider or other classes do not qualify.
3. Title or description only counts when the product has no key for that condition. If its own attribute says something else, it fails even when the title shows the value.
4. Values under some other, related attribute do not qualify.
5. Numbers: convert units first; limits are inclusive; a range qualifies only if all of it respects the limit.
6. Exclusions and closed sets must be obeyed.
7. It must be the wanted product type; where offers carry category labels, the right category is the one whose label fits the customer's phrasing as a whole. A narrower or adjacent category that holds the same kind of product counts only when the records show no fit at all inside the right category. The category label alone settles the type: a product listed in the right category passes the type check even when its title, description or a type attribute of its own names a different kind of item.
8. If the product's own attribute for a condition gives another value, that condition fails, whatever the title implies.
9. Cost times units may not exceed the budget (exactly equal is fine). If a record shows it out of stock, it fails.
10. A mere hope ("ideally", "would be nice") marked as not required is no condition.
11. With conflicting records (about price or stock, for example), the newest wins.
12. Use fits=false only for a condition the records plainly fail or that no record mentions at all; a missing record of stock or price is never grounds for failing.
13. When the order holds several products, judge the item only against the conditions of the product it is (by its category), and read the spending limit as covering the whole cart, not this item alone.
14. Sort every problem into one of two kinds. "contradicted": the product's own record states something for that condition and it is not the wanted value (another value, the wrong category, over budget, out of stock). "not_shown": nothing in the records speaks to that condition either way.

Output: ONE JSON object, nothing else: {"item": the product id, "fits": true or false, "problems": [{"condition": the condition exactly as listed, "kind": "contradicted" or "not_shown", "note": a few words}]}."""

PLAN_PROMPT = """Role: you decide which questions a purchasing agent puts to the customer. The customer carries hidden wishes; each one is the value of a single product attribute and surfaces only when a message asks about exactly that attribute. Only a message's first point is answered; asking about anything else yields a polite "don't mind".

Input: the customer's text, the conditions so far, the questions already put, items flagged ASK THE CUSTOMER FIRST, and the attributes found in the offers' details.

Rules:
1. Select up to QUESTION_COUNT distinct attributes, strongest candidates first.
2. First come items flagged below as ASK THE CUSTOMER FIRST that have not been asked; ask them plainly. For the budget, ask for their maximum spend. For a choice between two wishes, name both values with their attributes, e.g. "If no single listing offers both, do you prefer the <value> <property> or the <value> <property>?".
3. Then attributes the offers' details below truly contain, favouring (a) an attribute the story implies indirectly (who will own, wear or use it, how it should look, where or how it gets used) and (b) attributes whose values differ most across the offers, measures and sizes included (height, width, size, capacity, pack type), over logistics (warranty, condition, years) and over attributes that every offer shares.
4. Never choose an attribute the text already decides (its written conditions, price, stock, quantity), never repeat one already asked, never spend two questions on near-identical attributes (two kinds of style, two spellings of the same attribute), and skip attributes no offer has.
5. Use attribute names exactly as the offers spell them.
6. An attribute question names the attribute the way the offers name it and nothing more, in at most twelve words, for example "What <attribute> would you like?". Give no sample values and never offer a choice between two values. No instructions, no lists, no statements, no second question.
7. Mark each question "kind": "must_learn" for an item flagged ASK THE CUSTOMER FIRST, "attribute" otherwise.
8. When the order holds several products, give each question its "product" and close it with the product, as in "What <attribute> would you like for the <product>?"; spread the questions over the products whose details were seen.
9. If an earlier answer says the customer did not follow the question, do not ask that attribute the same way again; move on to the next attribute.

Output: ONE JSON object, nothing else: {"questions": [{"property": "...", "kind": "attribute" or "must_learn", "product": "... or empty", "question": "..."}, ...]}."""

LEAD_PROMPT = """Role: you pick the opening question a purchasing agent puts to a customer about one product. The customer holds a hidden wish about one attribute and reveals it only when a question names that attribute.

Input: the customer's text, the product, and the attributes the shop's listings carry for it, each shown with two sample values.

Rules:
1. Pick exactly one attribute from the list, spelled as listed.
2. First choice: an attribute that some sentence of the text is really about while never naming it, such as a component of the item, the way it attaches, holds or sizes up to something else, what sets its appearance, or the spot it has to share with other things.
3. When no sentence points at one, take the attribute that best tells this kind of product apart from its neighbours and that the text leaves open.
4. Skip attributes the text already settles, and skip price, stock, seller, shop, brand names and condition.

Output: ONE JSON object, nothing else: {"attribute": "..."}."""

AUDIT_PROMPT = """Role: a purchasing agent is ready to tell the customer the shop has nothing suitable; you double-check that.

Input: the customer's text, the current conditions and the offer records seen.

Rules:
1. Compare each record with the conditions.
2. A condition holds when the offer's own attribute for it shows the value (any key spelling, one item of a comma list, singular/plural and near synonyms all qualify; a wider or other class does not).
3. Title or description only counts when the offer has no key for that attribute.
4. Numbers are compared after unit conversion, with inclusive limits.
5. The cost must fit the budget and the product must not appear out of stock.
6. The category label settles the product type: an offer in the wanted category counts as that type whatever its title or its own type attribute calls it.

Output: ONE JSON object, nothing else: {"possible_fits": [ids of offers that seem to meet every condition, strongest first, 5 at most], "reason": "a single sentence"}."""

_llm = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=2, retry_delay=1)

_ID_HINT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-:.]{3,63}$")
_SKU_LIKE = re.compile(r"[A-Za-z0-9_\-]+")
_DECLINE_TEXT = "Sorry, nothing in the store meets all of your requirements, so I am not ordering anything."


def _trim(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, str):
        return value[:limit] + "..." if len(value) > limit else value
    if isinstance(value, dict):
        return {key: _trim(part, limit, depth + 1) for key, part in value.items()}
    if isinstance(value, list):
        return [_trim(part, limit, depth + 1) for part in value]
    return value


def _shrink(value: Any, depth: int = 0) -> Any:
    # older results keep ids, figures, short facts and shopper lines; long prose is cut
    if depth > 8:
        return None
    if isinstance(value, str):
        return value[:100] + "..." if len(value) > 120 else value
    if isinstance(value, dict):
        return {key: _shrink(part, depth + 1) for key, part in value.items()}
    if isinstance(value, list):
        if value and all(isinstance(part, str) for part in value):
            return value
        return [_shrink(part, depth + 1) for part in value]
    return value


def _unpaged(observation: dict[str, Any]) -> tuple[int, int] | None:
    # a result whose own count exceeds the rows it lists, with no non-empty continuation value beside them
    rows = next((part for part in observation.values() if isinstance(part, list) and part and all(isinstance(row, dict) for row in part)), None)
    if not rows:
        return None
    counts = [part for part in observation.values() if isinstance(part, int) and not isinstance(part, bool) and part > len(rows)]
    onward = [part for part in observation.values() if isinstance(part, str) and part.strip() and len(part) <= 80 and " " not in part.strip()]
    if not counts or onward:
        return None
    return max(counts), len(rows)


def _dump(value: Any) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + "...(cut)"


def _decode_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[A-Za-z]*", "", body).strip()
        if body.endswith("```"):
            body = body[:-3].strip()
    try:
        value = json.loads(body)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    left, right = body.find("{"), body.rfind("}")
    if 0 <= left < right:
        try:
            value = json.loads(body[left : right + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None


def _plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _short_sentences(value: Any, sink: list[str], depth: int = 0) -> None:
    if depth > 4:
        return
    if isinstance(value, str):
        if len(value.split()) >= 4:
            sink.append(value)
    elif isinstance(value, list):
        for part in value[:10]:
            _short_sentences(part, sink, depth + 1)


def _gather_ids(value: Any, sink: set[str], depth: int = 0) -> None:
    if depth > 6:
        return
    if isinstance(value, dict):
        for key, part in value.items():
            if isinstance(part, str) and "id" in key.lower() and _ID_HINT.match(part):
                sink.add(part)
            else:
                _gather_ids(part, sink, depth + 1)
    elif isinstance(value, list):
        for part in value:
            _gather_ids(part, sink, depth + 1)


def _nested_dicts(value: Any, depth: int = 0):
    if depth > 6:
        return
    if isinstance(value, list):
        for part in value:
            yield from _nested_dicts(part, depth + 1)
    elif isinstance(value, dict):
        yield value
        for part in value.values():
            if isinstance(part, (dict, list)):
                yield from _nested_dicts(part, depth + 1)


def _code_like(text: str) -> bool:
    return bool(re.search(r"\d", text)) and bool(_SKU_LIKE.fullmatch(text))


def _string_lines(observation: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for value in observation.values():
        if isinstance(value, list) and value and all(isinstance(part, str) for part in value):
            _short_sentences(value, found)
    return found


class Clock:
    def __init__(self) -> None:
        self.started = time.monotonic()
        self.cut = False

    def left(self) -> float:
        left = TASK_SECONDS - (time.monotonic() - self.started)
        # a runaway spend counts as a short clock, so the usual wrap-up path takes over
        return min(left, WRAP_UP_SECONDS - 1) if self.cut else left

    def elapsed(self) -> float:
        return time.monotonic() - self.started


class LLMDesk:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]]):
        self.clock = clock
        self.trace = trace
        self.cost = 0.0
        self.spend: dict[str, list[float]] = {}

    @staticmethod
    def _with_cache_marks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        marked = [dict(m) for m in messages]
        for spot in (0, len(marked) - 1):
            text = marked[spot].get("content")
            if isinstance(text, str) and text:
                marked[spot]["content"] = [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]
        return marked

    def ask(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int, role: str = "loop", effort: str | None = None) -> dict[str, Any] | None:
        for model in models:
            for pause in RETRY_PAUSES:
                remaining = self.clock.left()
                if remaining < FORCE_SECONDS - 10:
                    return None
                _llm.timeout = int(max(8, min(75, remaining - FORCE_SECONDS + 15)))
                cached = model.startswith("anthropic/") and len(messages) > 2
                request: dict[str, Any] = {"model": model, "messages": self._with_cache_marks(messages) if cached else messages, "temperature": 0, "max_tokens": max_tokens}
                if tools:
                    request.update(tools=tools, tool_choice="auto")
                if effort and model.startswith("anthropic/"):
                    request["reasoning"] = {"effort": effort}
                outcome = _llm.post_verbose("/inference/chat/completions", request)
                answer = self._take(outcome.data, role)
                if answer is not None:
                    return answer
                if self._ran_dry(outcome.data):
                    # the reply came back and spent its whole allowance without an answer; the same request repeats that, so try the next LLM
                    self.trace.append({"kind": "inference_failure", "model": model, "error": "ran out of tokens before answering"})
                    break
                self.trace.append({"kind": "inference_failure", "model": model, "error": str(outcome.error)[:300] if outcome.error else "empty"})
                status = (outcome.error or {}).get("status") if isinstance(outcome.error, dict) else None
                # a refused key or a bad request stays bad; a busy upstream tends to recover
                if status in (400, 401, 402, 403, 404, 422) or pause is None or self.clock.left() < FORCE_SECONDS + 30:
                    break
                time.sleep(pause)
        return None

    @staticmethod
    def _ran_dry(data: Any) -> bool:
        if not isinstance(data, dict) or not data.get("choices"):
            return False
        choice = data["choices"][0]
        message = choice.get("message") or {}
        return choice.get("finish_reason") == "length" and not (message.get("content") or message.get("tool_calls"))

    def _take(self, data: Any, role: str) -> dict[str, Any] | None:
        if not isinstance(data, dict) or not data.get("choices"):
            return None
        spent = (data.get("usage") or {}).get("cost")
        if isinstance(spent, (int, float)):
            self.cost += float(spent)
            tally = self.spend.setdefault(role, [0, 0.0])
            tally[0] += 1
            tally[1] += float(spent)
            if self.cost >= RUNAWAY_USD and not self.clock.cut:
                self.clock.cut = True
                self.trace.append({"kind": "runaway", "cost": round(self.cost, 5), "role": role})
        message = data["choices"][0].get("message") or {}
        if message.get("content") or message.get("tool_calls"):
            return message
        return None

    def ask_json(self, prompt: str, max_tokens: int = 700, models: list[str] | None = None, role: str = "judge", effort: str | None = None) -> dict[str, Any] | None:
        message = self.ask([{"role": "user", "content": prompt}], None, models or [JUDGE_MODEL, MAIN_MODEL], max_tokens, role, effort)
        return _decode_object((message or {}).get("content") or "")


class WorldDesk:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]], environment: dict[str, Any]):
        self.clock = clock
        self.trace = trace
        self.binding = dict(environment["binding"])
        self.policy = environment["policy_view"]
        self.max_steps = int(self.policy.get("max_steps") or 30)
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dead = False

    @property
    def query(self) -> str:
        return str(self.policy.get("query") or "")

    @property
    def opening(self) -> Any:
        return self.policy.get("opening")

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def over(self) -> bool:
        return self.finished or self.dead or self.steps_left() <= 0

    def call(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        payload.update({"action": actions[0]} if len(actions) == 1 else {"actions": actions})
        attempt = 0
        while attempt < 3:
            attempt += 1
            remaining = self.clock.left()
            if remaining < 2:
                return None
            _env.timeout = int(max(3, min(95, remaining - 1)))
            outcome = _env.post_verbose("/environment/call", payload)
            if isinstance(outcome.data, dict):
                return self._accept(outcome.data, len(actions))
            error = outcome.error or {}
            status = error.get("status")
            text = str(error.get("body") or "")
            self.trace.append({"kind": "env_failure", "status": status, "body": text[:300]})
            if status == 402:
                self.dead = True
                return None
            if status != 409:
                if status in (403, 422, 500, 503, 504):
                    return None
                continue
            if "finished" in text:
                self.finished = True
                return None
            expected = re.search(r"expected (\d+)", text)
            if not expected:
                return None
            self.step = int(expected.group(1)) - 1
            payload["step"] = self.step + 1
        return None

    def _accept(self, data: dict[str, Any], count: int) -> dict[str, Any]:
        self.step = data["step"] if isinstance(data.get("step"), int) else self.step + count
        if data.get("finished"):
            self.finished = True
        return data


class SchemaGuard:
    def __init__(self, policy: dict[str, Any]):
        self.tools = [tool for tool in policy.get("tools") or [] if isinstance(tool, dict) and isinstance(tool.get("function"), dict)]
        self.schemas = {tool["function"]["name"]: tool["function"].get("parameters") or {"type": "object", "properties": {}} for tool in self.tools}

    def knows(self, name: Any) -> bool:
        return name in self.schemas

    def props(self, name: str | None) -> dict[str, Any]:
        return (self.schemas.get(name or "") or {}).get("properties") or {}

    @staticmethod
    def matches(value: Any, kind: str) -> bool:
        if kind == "null":
            return value is None
        if kind == "string":
            return isinstance(value, str)
        if kind == "boolean":
            return isinstance(value, bool)
        if kind == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if kind == "number":
            return _plain_number(value)
        if kind == "array":
            return isinstance(value, list)
        if kind == "object":
            return isinstance(value, dict)
        return True

    @staticmethod
    def kinds_of(schema: dict[str, Any]) -> list[str] | None:
        kinds = schema.get("type")
        return [kinds] if isinstance(kinds, str) else kinds

    @staticmethod
    def takes_text(spec: dict[str, Any]) -> bool:
        return spec.get("type") == "string" or "string" in (spec.get("type") or [])

    def convert(self, value: Any, schema: dict[str, Any]) -> Any:
        kinds = self.kinds_of(schema)
        if not kinds:
            return value
        if any(self.matches(value, kind) for kind in kinds):
            whole_float = isinstance(value, float) and value.is_integer()
            return int(value) if "integer" in kinds and whole_float else value
        for kind in kinds:
            try:
                if kind == "integer" and isinstance(value, float) and value.is_integer():
                    return int(value)
                if kind == "integer" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.0+)?\s*", value):
                    return int(float(value))
                if kind == "number" and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                    figure = float(value)
                    return int(figure) if figure.is_integer() else figure
                if kind == "string" and _plain_number(value):
                    return str(value)
                if kind == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
                    return value.lower() == "true"
            except (TypeError, ValueError):
                continue
        return value

    def validate(self, value: Any, schema: Any, path: str) -> tuple[Any, list[str]]:
        if not isinstance(schema, dict):
            return value, []
        value = self.convert(value, schema)
        kinds = self.kinds_of(schema)
        if kinds and not any(self.matches(value, kind) for kind in kinds):
            return value, [f"{path}: expected {'/'.join(kinds)}"]
        issues: list[str] = []
        options = schema.get("enum")
        if "enum" in schema and value not in options:
            if isinstance(value, str):
                wanted = value.strip().lower()
                value = next((option for option in options if isinstance(option, str) and option.lower() == wanted), value)
            if value not in options:
                issues.append(f"{path}: must be one of {options}")
        if "const" in schema and value != schema["const"]:
            issues.append(f"{path}: must equal {schema['const']!r}")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                issues.append(f"{path}: shorter than {schema['minLength']} characters")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                value = value[: schema["maxLength"]]
        if _plain_number(value):
            low, high = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
            if "minimum" in schema and value < schema["minimum"]:
                issues.append(f"{path}: below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                issues.append(f"{path}: above maximum {schema['maximum']}")
            if "exclusiveMinimum" in schema and isinstance(low, (int, float)) and value <= low:
                issues.append(f"{path}: must be above {low}")
            if "exclusiveMaximum" in schema and isinstance(high, (int, float)) and value >= high:
                issues.append(f"{path}: must be below {high}")
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                issues.append(f"{path}: needs at least {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                issues.append(f"{path}: allows at most {schema['maxItems']} items")
            element = schema.get("items")
            if isinstance(element, dict):
                repaired_list = []
                for position, part in enumerate(value):
                    part, found = self.validate(part, element, f"{path}[{position}]")
                    repaired_list.append(part)
                    issues += found
                value = repaired_list
        if isinstance(value, dict):
            known = schema.get("properties") or {}
            for field in schema.get("required") or []:
                if field not in value:
                    issues.append(f"{path}.{field}: required")
            if schema.get("additionalProperties") is False:
                for field in list(value):
                    if field not in known:
                        issues.append(f"{path}.{field}: not an allowed argument")
            repaired = {}
            for field, part in value.items():
                if field in known:
                    part, found = self.validate(part, known[field], f"{path}.{field}")
                    issues += found
                repaired[field] = part
            value = repaired
        return value, issues

    def check_args(self, name: str, args: Any) -> tuple[Any, list[str]]:
        return self.validate(args, self.schemas[name], "args")

    def enum_signature(self, name: str, args: dict[str, Any]) -> str:
        known = self.props(name)
        return _dump({key: value for key, value in args.items() if isinstance(known.get(key), dict) and "enum" in known[key]})

    def message_args(self, name: str | None, text: str) -> dict[str, Any] | None:
        schema = self.schemas.get(name or "") or {}
        known = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        free_text = [key for key, spec in known.items() if isinstance(spec, dict) and self.takes_text(spec) and "enum" not in spec]
        field = next((key for key in required if key in free_text), None) or (free_text[0] if len(free_text) == 1 else None)
        if not field or any(key != field for key in required):
            return None
        return {field: text}

    def describe(self, name: str) -> str:
        return next((str(tool["function"].get("description") or "") for tool in self.tools if tool["function"].get("name") == name), "")

    def filler_args(self, name: str, text: str, chosen: dict[str, Any] | None = None) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        known = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = known.get(key) or {}
            if "enum" in spec and spec["enum"]:
                # a fixed choice is never free text: the value the setup reading named for this job, else the value whose
                # describing sentence shares most words with this action's own wording
                picked = (chosen or {}).get(key)
                if picked in spec["enum"]:
                    args[key] = picked
                    continue
                said = _words(text)
                notes = [line for line in re.split(r"(?<=[.;])\s+", f"{self.describe(name)} {spec.get('description') or ''}") if line.strip()]

                def fit(option: Any) -> int:
                    own = _words(option)
                    near = set().union(*[_words(line) for line in notes if own & _words(line)]) if notes else set()
                    return len((own | near) & said)

                args[key] = max(spec["enum"], key=fit)
            elif self.takes_text(spec):
                args[key] = text[: spec.get("maxLength", len(text))]
            elif spec.get("type") == "boolean":
                args[key] = True
            elif spec.get("type") in ("integer", "number"):
                args[key] = max(1, spec.get("minimum", 1))
        if len(known) == 1:
            for key, spec in known.items():
                if key not in args and isinstance(spec, dict) and spec.get("type") == "string" and "enum" not in spec:
                    args[key] = text[: spec.get("maxLength", len(text))]
        return args


class Ledger:
    def __init__(self) -> None:
        self.plan: dict[str, Any] = {}
        self.seen_ids: set[str] = set()
        self.stated: list[str] = []
        self.revealed: list[str] = []
        self.free: list[str] = []
        self.dropped: list[str] = []
        self.budget_now: Any = None
        self.current: list[str] | None = None
        self.version = 0
        self.heard: set[str] = set()
        self.said: list[str] = []
        self.records: list[Any] = []
        self.cart_ids: list[str] = []
        self.last_cart_view: Any = None
        self.add_signature: str | None = None
        self.shelf = Shelf()

    def remember(self, value: Any) -> None:
        self.records.append(value)

    def note_ids(self, value: Any) -> None:
        _gather_ids(value, self.seen_ids)

    def summary(self) -> str:
        if self.current is None:
            parts = [f"{r} (stated)" for r in self.stated if r not in self.dropped]
            parts.extend(f"{r} (revealed by the shopper)" for r in self.revealed if r not in self.dropped)
        else:
            parts = list(self.current)
        text = "; ".join(parts) or "see the request"
        if self.budget_now is not None:
            text += f"; total price at most {self.budget_now} {self.plan.get('currency') or ''}".rstrip()
        wanted = _basket(self.plan)
        if wanted:
            text += "; one order of: " + ", ".join(f"{entry['quantity']} x {entry['product']}" for entry in wanted)
        else:
            text += f"; quantity {self.plan.get('quantity') or 1}"
        if self.dropped:
            text += "; no longer required: " + "; ".join(self.dropped)
        if self.free:
            text += "; shopper does not care about: " + ", ".join(self.free)
        return text

    def records_of(self, item: str) -> list[str]:
        hits: list[str] = []

        def visit(value: Any, depth: int) -> None:
            if depth > 6 or len(hits) >= 8:
                return
            if isinstance(value, dict):
                if any(isinstance(part, str) and part == item for part in value.values()):
                    text = _dump(_trim(value, 400))[:2500]
                    if text not in hits:
                        hits.append(text)
                    return
                for part in value.values():
                    visit(part, depth + 1)
            elif isinstance(value, list):
                for part in value:
                    visit(part, depth + 1)

        for record in self.records:
            visit(record, 0)
        return hits[-4:]

    def property_menu(self, products: list[str] | None = None) -> tuple[str, int]:
        # in one order of several products each product gets its own menu, so no product's listings crowd out another's
        if products and self.shelf.range_key():
            parts, total = [], 0
            for product in products:
                menu, rich = self.single_menu(product)
                total += rich
                if menu:
                    parts.append(f"For the {product}:\n{menu}")
            return "\n".join(parts), total
        # one product: its own range when the shop labels ranges, so listings opened for a lookup or by mistake stay out
        own = self.single_menu(str(self.plan.get("product") or "")) if self.plan.get("product") else ("", 0)
        if own[0]:
            return own
        # before the product's own range is seen, listings opened for a lookup elsewhere must not stand in for it
        lookups = {self.shelf.main_range(str(entry.get("range") or ""), strict=True) for entry in self.plan.get("references") or [] if isinstance(entry, dict)} - {None}
        return self.tally_menu(True, None, lookups) if lookups else self.single_menu(None)

    def single_menu(self, product: str | None) -> tuple[str, int]:
        # attribute maps nested inside a listing come first; the listing's own row fields (seller, title, range) are not attributes
        menu = self.tally_menu(True, product)
        if menu[1] or not self.shelf.opened:
            return menu
        return self.tally_menu(False, product)

    def tally_menu(self, nested_only: bool, product: str | None = None, skip: set[str] | None = None) -> tuple[str, int]:
        counts: dict[str, int] = {}
        examples: dict[str, list[str]] = {}
        values: dict[str, set[str]] = {}
        rich = 0
        label_wanted = self.shelf.main_range(product, strict=True) if product else None
        if product and not label_wanted:
            return "", 0

        range_key = self.shelf.range_key() if product or skip else None

        def visit(value: Any, depth: int, owner: str | None) -> None:
            nonlocal rich
            if depth > 6:
                return
            if isinstance(value, dict):
                # a map that carries a listing's own id is that listing's row, under whatever key the shop names the id
                own = next((part for part in value.values() if isinstance(part, str) and part in self.shelf.rows), None) or _row_id(value)
                if own:
                    owner = self.shelf.range_of(own) or (str(value.get(range_key)) if range_key and value.get(range_key) else None)
                elif range_key and isinstance(value.get(range_key), str):
                    owner = value[range_key].strip().lower()
                facts = [(key, part) for key, part in value.items() if isinstance(part, str) and 0 < len(part) <= 70]
                mine = (not label_wanted or owner == label_wanted) and not (skip and owner in skip)
                if mine and len(facts) >= (3 if nested_only else 5) and not (nested_only and own):
                    rich += 1
                    for key, part in facts:
                        if _code_like(part) and ("-" in part or "_" in part):
                            continue
                        label = re.sub(r"[_.]+", " ", str(key)).strip().lower()
                        counts[label] = counts.get(label, 0) + 1
                        values.setdefault(label, set()).add(part.lower())
                        pool = examples.setdefault(label, [])
                        if part.lower() not in pool and len(pool) < 5:
                            pool.append(part.lower())
                for part in value.values():
                    if isinstance(part, (dict, list)):
                        visit(part, depth + 1, owner)
            elif isinstance(value, list):
                for part in value:
                    visit(part, depth + 1, owner)

        for record in self.records:
            visit(record, 0, None)
        # a field whose value differs on every listing names that listing, not a choice the shopper makes
        kept = [label for label in counts if not (counts[label] >= 4 and len(values[label]) == counts[label])]
        # a field many listings carry with several different values splits the candidates best; one shared value splits nothing
        # a field that is different on most listings it appears on is nearly a listing's name, so it ranks lower
        top = sorted(kept, key=lambda label: -(counts[label] * min(len(values[label]), 5) * (0.4 if counts[label] >= 3 and len(values[label]) >= 0.6 * counts[label] else 1)))[:12 if product else 28]
        return "\n".join(f"- {label} (in {counts[label]} listings, {len(values[label])} different values): " + " | ".join(examples.get(label, [])) for label in top), rich

    def recurring_labels(self) -> str:
        tally: dict[str, dict[str, int]] = {}
        for found in (d for record in self.records for d in _nested_dicts(record)):
            if len(found) < 4:
                continue
            for key, value in found.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not _code_like(value):
                    column = tally.setdefault(str(key), {})
                    column[value] = column.get(value, 0) + 1
        rows = []
        for key, column in tally.items():
            if sum(column.values()) < 4 or not 2 <= len(column) <= 8 or max(column.values()) < 2:
                continue
            ordered = sorted(column.items(), key=lambda pair: -pair[1])
            rows.append(f"- {key}: " + " | ".join(f"{value} (x{n})" for value, n in ordered))
        return "\n".join(rows[:12])


def _basket(plan: dict[str, Any]) -> list[dict[str, Any]]:
    # several kinds of product in one order; a single entry is an ordinary request
    entries = []
    for entry in plan.get("products") or []:
        if isinstance(entry, dict) and str(entry.get("product") or "").strip():
            units = entry.get("quantity")
            units = units if isinstance(units, int) and not isinstance(units, bool) and units > 0 else 1
            entries.append({"product": str(entry["product"]).strip(), "quantity": units, "requirements": [str(r) for r in entry.get("requirements") or [] if r]})
    return entries if len(entries) >= 2 else []


def _words(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(value).lower()))


def _split_need(need: str) -> tuple[set[str], set[str]]:
    head, _, tail = str(need).partition(":")
    return _words(head), _words(tail)


def _row_id(found: dict[str, Any]) -> str | None:
    loose = None
    for key, part in found.items():
        if isinstance(part, str) and _ID_HINT.match(part):
            name = str(key).lower()
            if name == "id":
                return part
            # "id" must be a word of the key (item_id, productId), not letters inside one, and a loose id carries a digit
            if "id" in _words(re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))) and any(ch.isdigit() for ch in part) and loose is None:
                loose = part
    return loose


class Shelf:
    # per listing id: its plain row fields, its facts, and whether its page was opened
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.facts: dict[str, dict[str, str]] = {}
        self.opened: set[str] = set()
        self.price_votes: dict[str, int] = {}
        self.range_memo: tuple[int, str | None] = (-1, None)

    def take(self, observation: Any, opened: bool) -> None:
        for found in _nested_dicts(observation):
            ident = _row_id(found)
            if not ident:
                continue
            flat = self.rows.setdefault(ident, {})
            flat.update({key: part for key, part in found.items() if not isinstance(part, (dict, list))})
            known = self.facts.setdefault(ident, {})
            for inner in _nested_dicts(found):
                if inner is not found and _row_id(inner) not in (None, ident):
                    continue
                known.update({str(key): part for key, part in inner.items() if isinstance(part, str) and 0 < len(part) <= 200})
            if opened:
                self.opened.add(ident)

    def learn_price(self, cart_view: Any) -> None:
        # the price field is the row field whose figure the cart repeats for the same listing
        for found in _nested_dicts(cart_view):
            ident = _row_id(found)
            row = self.rows.get(ident or "")
            if not row:
                continue
            figures = [part for part in found.values() if _plain_number(part) and part > 1]
            for key, part in row.items():
                if _plain_number(part) and part in figures:
                    self.price_votes[key] = self.price_votes.get(key, 0) + 1

    def sole_field(self, numeric: bool) -> str | None:
        # without a cart to learn from, a field only counts when it is the one field of its type most rows carry
        if len(self.rows) < 3:
            return None
        cover: dict[str, int] = {}
        for row in self.rows.values():
            for key, part in row.items():
                if (_plain_number(part) if numeric else isinstance(part, bool)):
                    cover[key] = cover.get(key, 0) + 1
        common = [key for key, n in cover.items() if n >= 0.8 * len(self.rows)]
        return common[0] if len(common) == 1 else None

    def price(self, ident: str) -> float | None:
        key = max(self.price_votes, key=lambda name: self.price_votes[name]) if self.price_votes else self.sole_field(True)
        value = (self.rows.get(ident) or {}).get(key) if key else None
        return float(value) if _plain_number(value) else None

    def sold_out(self, ident: str) -> bool:
        key = self.sole_field(False)
        return key is not None and (self.rows.get(ident) or {}).get(key) is False

    def range_key(self) -> str | None:
        if self.range_memo[0] != len(self.rows):
            self.range_memo = (len(self.rows), self.find_range_key())
        return self.range_memo[1]

    def find_range_key(self) -> str | None:
        # the label every row carries with the fewest different values is the shop's range (category) label
        if len(self.rows) < 4:
            return None
        tally: dict[str, set[str]] = {}
        for row in self.rows.values():
            for key, part in row.items():
                if isinstance(part, str) and 0 < len(part) <= 60 and not _code_like(part):
                    tally.setdefault(key, set()).add(part.strip().lower())
        cover = {key: sum(1 for row in self.rows.values() if isinstance(row.get(key), str) and 0 < len(row[key]) <= 60) for key in tally}
        usable = [key for key in tally if cover[key] >= 0.8 * len(self.rows) and len(tally[key]) < len(self.rows)]
        if not usable:
            return None
        return min(usable, key=lambda key: (len(tally[key]), -cover[key]))

    def range_of(self, ident: str) -> str | None:
        key = self.range_key()
        value = (self.rows.get(ident) or {}).get(key) if key else None
        return value.strip().lower() if isinstance(value, str) else None

    def main_range(self, product: str = "", strict: bool = False) -> str | None:
        # the range label with the widest word overlap with the wanted product (a word and its plural count as one), else the range of most opened listings
        labels = {self.range_of(ident) for ident in self.rows} - {None}
        wanted = {word for word in _words(product) if len(word) > 2}

        def same(word: str, other: str) -> bool:
            return word == other or other in (word + "s", word + "es") or word in (other + "s", other + "es")

        tail = re.findall(r"[a-z0-9]+", product.lower())[-1:]

        def overlap(label: str) -> tuple[int, int, int]:
            # whole words, a plural counting as its singular; then the head noun (last word) agreeing; then fewest words beyond the product
            have = {word for word in _words(label) if len(word) > 2}
            hits = sum(1 for word in wanted if any(same(word, other) for other in have))
            head = re.findall(r"[a-z0-9]+", label.lower())[-1:]
            agrees = int(bool(tail and head and same(tail[0], head[0])))
            extra = sum(1 for other in have if not any(same(word, other) for word in wanted))
            return hits, agrees, -extra

        scored = sorted(((overlap(label), label) for label in labels), reverse=True)
        if scored and scored[0][0][0] > 0:
            return scored[0][1]
        if strict:
            return None
        counts: dict[str, int] = {}
        for ident in self.opened:
            label = self.range_of(ident)
            if label:
                counts[label] = counts.get(label, 0) + 1
        return max(counts, key=lambda label: counts[label]) if counts else None

    def unopened(self, label: str | None, cap: float | None, quantity: int, below: float | None = None) -> list[tuple[float, str]]:
        found = []
        for ident in self.rows:
            if ident in self.opened or (label and self.range_of(ident) != label):
                continue
            cost = self.price(ident)
            if cost is None or self.sold_out(ident) or (cap is not None and _plain_number(cap) and cost * quantity > cap) or (below is not None and cost >= below):
                continue
            found.append((cost, ident))
        return sorted(found)

    def carried(self, need: str) -> bool:
        # does any opened listing state this property under some key of its own
        wanted, _ = _split_need(need)
        wanted = {word for word in wanted if len(word) > 2}
        if not wanted:
            return True
        return any(wanted & _words(key) for ident in self.opened for key in (self.facts.get(ident) or {}))

    def literal(self, ident: str, need: str) -> bool:
        # every word of the wanted value sits inside the value of a key that names the property
        wanted, value = _split_need(need)
        wanted = {word for word in wanted if len(word) > 2}
        if not wanted or not value:
            return False
        return any(wanted & _words(key) and value <= _words(part) for key, part in (self.facts.get(ident) or {}).items())


class SetupAdvisor:
    def __init__(self, llm: LLMDesk, world: WorldDesk, guard: SchemaGuard):
        self.llm = llm
        self.world = world
        self.guard = guard

    def read(self) -> dict[str, Any]:
        catalogue = "\n".join(
            f"- {tool['function']['name']}: {tool['function'].get('description', '')} | arguments: {_dump(tool['function'].get('parameters') or {})}"
            for tool in self.guard.tools
        )
        fixed_part = SETUP_PROMPT + "\n\nTOOLS:\n" + catalogue
        task_part = "REQUEST:\n" + self.world.query
        if self.world.opening is not None:
            task_part += "\n\nOPENING OBSERVATION:\n" + _dump(self.world.opening)
        parts = [{"type": "text", "text": fixed_part, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": task_part}]
        message = self.llm.ask([{"role": "user", "content": parts}], None, [SETUP_MODEL, BACKUP_MODEL], 1500, "setup")
        return _decode_object((message or {}).get("content") or "") or {}


class InterpretAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.query = query

    def read(self, lines: list[str], question: str | None) -> tuple[str, str]:
        book = self.ledger
        fresh = [line for line in lines if line not in book.heard]
        if not fresh:
            return "", ""
        book.heard.update(fresh)
        if self.clock.left() < WRAP_UP_SECONDS:
            return "", ""
        sections = [INTERPRET_PROMPT, "Input - the shopper's request (context only):\n" + self.query, "Input - requirements gathered so far:\n" + book.summary()]
        if book.said:
            sections.append("Input - earlier exchanges with the shopper, oldest first:\n" + "\n".join(book.said[-12:]))
        if question:
            sections.append("Input - the agent's question:\n" + question)
        sections.append("Input - what the shopper says now:\n" + "\n".join(fresh))
        book.said.append((f"asked: {question} -> " if question else "") + "shopper: " + " / ".join(fresh))
        verdict = self.llm.ask_json("\n\n".join(sections), 1500, [READ_MODEL, JUDGE_MODEL], "interpret")
        if not verdict:
            return "", ""
        reply = str(verdict.get("reply") or "").strip() if verdict.get("asked_us") is True else ""
        return self._apply(verdict), reply

    def _apply(self, verdict: dict[str, Any]) -> str:
        book = self.ledger
        changes = []
        latest = verdict.get("requirements_now")
        if isinstance(latest, list) and all(isinstance(entry, str) for entry in latest):
            book.current = [entry for entry in latest if entry.strip()]
        for need in verdict.get("needs") or []:
            if isinstance(need, str) and need and need not in book.revealed:
                book.revealed.append(need)
                changes.append(f"new requirement: {need}")
        for prop in verdict.get("no_preference") or []:
            if isinstance(prop, str) and prop and prop not in book.free:
                book.free.append(prop)
        for gone in verdict.get("dropped") or []:
            if isinstance(gone, str) and gone and gone not in book.dropped:
                book.dropped.append(gone)
                changes.append(f"dropped: {gone}")
        budget = verdict.get("new_budget")
        if _plain_number(budget) and budget > 0 and budget != book.budget_now:
            book.budget_now = budget
            changes.append(f"new budget: {budget}")
        if verdict.get("unavailable"):
            changes.append("an item became unavailable; remove it from the cart and do not order it")
        if changes:
            book.version += 1
        tail = " (" + "; ".join(changes) + ")" if changes else ""
        return "Meaning: " + str(verdict.get("meaning") or "") + tail


class PlanAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, query: str):
        self.llm = llm
        self.ledger = ledger
        self.query = query

    def questions(self, count: int, asked: list[str], menu: str) -> list[dict[str, Any]]:
        must_learn = self.ledger.plan.get("ask_first")
        sections = [PLAN_PROMPT.replace("QUESTION_COUNT", str(count)), "Input - the shopper's request:\n" + self.query, "Input - requirements gathered so far:\n" + self.ledger.summary()]
        if asked:
            sections.append("Input - questions already asked, with the shopper's answers:\n" + "\n".join(asked))
        if must_learn:
            sections.append("ASK THE CUSTOMER FIRST: " + "; ".join(str(entry) for entry in must_learn or []))
        sections.append("Input - properties in the listings' details seen so far (name, how many listings, example values):\n" + (menu or "(none seen yet)"))
        answer = self.llm.ask_json("\n\n".join(sections), 2000, [READ_MODEL, JUDGE_MODEL], "plan")
        return [entry for entry in (answer or {}).get("questions") or [] if isinstance(entry, dict) and str(entry.get("question") or "").strip()]

    def lead_key(self, product: str, choices: dict[str, str]) -> str | None:
        listing = "\n".join(f"- {label}: {samples}" for label, samples in choices.items())
        sections = [LEAD_PROMPT, "Input - the shopper's request:\n" + self.query, "Input - the product:\n" + (product or "the requested item"), "Input - attributes in the listings, with two sample values each:\n" + listing]
        answer = self.llm.ask_json("\n\n".join(sections), 1500, [JUDGE_MODEL, READ_MODEL], "lead", "low")
        key = re.sub(r"[_.]+", " ", str((answer or {}).get("attribute") or "")).strip().lower()
        # only a property the listings carry can be asked; anything else falls back to the plan's own order
        return key if key in choices else None


class VerifyAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, trace: list[dict[str, Any]], query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.query = query
        self.memo: dict[str, str] = {}
        self.passed: set[str] = set()

    def check(self, item: str, purpose: str) -> str:
        if self.clock.left() < WRAP_UP_SECONDS + 20:
            return ""
        book = self.ledger
        memo_key = f"{item}|{book.version}"
        if memo_key in self.memo:
            return self.memo[memo_key]
        records = book.records_of(item)
        if not records:
            return ""
        sections = [
            VERIFY_PROMPT,
            "Input - the shopper's request:\n" + self.query,
            "Input - current requirements (stated, revealed and changed):\n" + book.summary(),
            f"Input - the item to {purpose}: {item}\nInput - records of this item seen in the shop, oldest first:\n" + "\n".join(records),
        ]
        if book.last_cart_view is not None:
            sections.append("Input - latest cart view:\n" + _dump(book.last_cart_view)[:3000])
        labels = book.recurring_labels()
        if labels:
            sections.append("Input - labels that recur on this shop's listings, with counts (the item's category label has to be the one matching the request's wording as a whole, not a neighbouring one):\n" + labels)
        result = self.llm.ask_json("\n\n".join(sections), 1200, role="verify")
        problems = self.blocking(item, result) if result and result.get("fits") is False else []
        self.trace.append({"kind": "verify", "item": item, "result": result, "holds": problems})
        verdict = ""
        if problems:
            verdict = f"An independent check says {item} does NOT meet every requirement: " + "; ".join(problems[:5]) + ". Unless you are certain the check is wrong, choose another fully verified fit."
        elif result:
            self.passed.add(item)
        self.memo[memo_key] = verdict
        return verdict

    def blocking(self, item: str, result: dict[str, Any]) -> list[str]:
        # only evidence against the item stops it; a condition no opened listing states at all cannot be checked in this shop
        shelf = self.ledger.shelf
        kept = []
        for problem in result.get("problems") or []:
            if isinstance(problem, dict):
                need = str(problem.get("condition") or problem.get("requirement") or "")
                kind = str(problem.get("kind") or "contradicted").lower()
                note = str(problem.get("note") or "")
            else:
                need, kind, note = str(problem), "contradicted", ""
            if kind.startswith("not") and not shelf.carried(need):
                continue
            if need and shelf.literal(item, need):
                continue
            kept.append(f"{need} ({note})" if note else need)
        if not kept and not result.get("problems"):
            kept.append("the check found it unfit")
        return kept


class AuditAdvisor:
    def __init__(self, llm: LLMDesk, ledger: Ledger, clock: Clock, trace: list[dict[str, Any]], query: str):
        self.llm = llm
        self.ledger = ledger
        self.clock = clock
        self.trace = trace
        self.query = query
        self.done = False
        self.fits: list[str] = []

    def review(self) -> str:
        if self.done or self.clock.left() < WRAP_UP_SECONDS + 30:
            return ""
        self.done = True
        picked: list[str] = []
        used = 0
        for record in self.ledger.records[::-1]:
            text = _dump(_shrink(record))
            if used + len(text) <= 16000:
                picked.append(text)
                used += len(text)
        if not picked:
            return ""
        sections = [
            AUDIT_PROMPT,
            "Input - the shopper's request:\n" + self.query,
            "Input - current requirements:\n" + self.ledger.summary(),
            "Input - listing records seen, newest first:\n" + "\n".join(picked),
        ]
        result = self.llm.ask_json("\n\n".join(sections), 1200, role="audit")
        self.trace.append({"kind": "audit", "result": result})
        candidates = [str(x) for x in (result or {}).get("possible_fits") or [] if isinstance(x, (str, int))][:5]
        self.fits = candidates
        if not candidates:
            return ""
        why = f" ({result.get('reason')})" if result.get("reason") else ""
        return "An independent check thinks these listings may meet every requirement: " + ", ".join(candidates) + why + ". Verify them (details, stock, price) and buy the right one; decline only if each one truly fails."


class Advisors:
    def __init__(self, setup: SetupAdvisor, interpret: InterpretAdvisor, plan: PlanAdvisor, verify: VerifyAdvisor, audit: AuditAdvisor):
        self.setup = setup
        self.interpret = interpret
        self.plan = plan
        self.verify = verify
        self.audit = audit


ORDER_CHECKLIST = "Not run yet (no step used). Pre-order check: (1) the cart holds one line per wanted product (exactly one line when one product is wanted), each at its required quantity, and nothing else; (2) after your last cart change you re-read this item's stock and price with a tool that shows them, and it is in stock and price times quantity is within the current budget; (3) its own properties meet every stated requirement, every need the shopper revealed and every change announced; (4) if the cheapest or best was asked for, no better fitting item exists among everything you could find; (5) every procedural rule in the request has been followed. If anything is unchecked, check it now; otherwise call the order tool again."
DECLINE_CHECKLIST = "Not run yet (no step used). Before declining, be sure: you searched in several ways (different words, filters, pages), you checked every candidate's own properties, and each one fails a stated requirement, a revealed need, the budget or stock. Remember that close synonyms and any key spelling count. If something does fit, buy it instead; if nothing fits, call the decline tool again."
REFUSAL_NOTE = "THE ORDER WAS REFUSED and the episode continues; the cart is unchanged. Apply every change in the refusal, fix the cart (remove any line that is no longer acceptable, such as a sold-out item), re-check stock and price of the new pick, and order again."
NO_BUDGET_ASK = "the most the shopper wants to spend, since the request states no price limit (skip if the same thing is listed again here)"


class Driver:
    def __init__(self, clock: Clock, trace: list[dict[str, Any]], llm: LLMDesk, world: WorldDesk, guard: SchemaGuard, ledger: Ledger, advisors: Advisors):
        self.clock = clock
        self.trace = trace
        self.llm = llm
        self.world = world
        self.guard = guard
        self.ledger = ledger
        self.advisors = advisors
        self.plan: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.message_budget: int | None = None
        self.messages_used = 0
        self.order_attempts = 0
        self.refusals = 0
        self.order_checked = False
        self.decline_checked = False
        self.cart_used = False
        self.planner_on = True
        self.used_needles: set[str] = set()
        self.dry_search = False
        self.compacted = 0
        self.paging_noted = False
        self.asked: list[str] = []
        self.led: set[str] = set()
        self.qty_warned: set[str] = set()
        self.cart_checked: set[str] = set()
        self.recent: list[str] = []
        self.pending_reply = ""
        self.messages: list[dict[str, Any]] = []
        self.price_holds = 0
        self.decline_holds = 0
        self.basket: list[dict[str, Any]] = []

    @property
    def finished(self) -> bool:
        return self.world.finished

    @property
    def dead(self) -> bool:
        return self.world.dead

    def role(self, name: str) -> str | None:
        return self.roles.get(name)

    def message_limit(self) -> int:
        return 3 if self.message_budget is None else self.message_budget

    def finishers(self) -> set[str]:
        return {self.role("order"), self.role("decline")} - {None}

    # reading of the request

    def adopt_plan(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.ledger.plan = plan
        named = plan.get("roles") if isinstance(plan.get("roles"), dict) else {}
        self.roles = {role: tool for role, tool in named.items() if isinstance(tool, str) and self.guard.knows(tool)}
        allowance = plan.get("message_budget")
        if _plain_number(allowance) and allowance >= 0:
            self.message_budget = int(allowance)
        self.ledger.stated = [str(entry) for entry in plan.get("requirements") or [] if entry]
        self.basket = _basket(plan)
        if self.basket and not self.ledger.stated:
            self.ledger.stated = [f"{entry['product']} - {need}" for entry in self.basket for need in entry["requirements"]]
        span = str(plan.get("quantity_span") or "").strip().lower()
        if self.quantity() > 1 and (not span or span not in self.world.query.lower()):
            plan["quantity"] = 1
        self.ledger.budget_now = plan.get("price_cap")
        if plan and self.ledger.budget_now is None and self.role("message"):
            plan["ask_first"] = [NO_BUDGET_ASK] + [str(entry) for entry in plan.get("ask_first") or [] if entry]
        moves = plan.get("step_budget")
        if _plain_number(moves) and 0 < moves < self.world.max_steps:
            self.world.max_steps = int(moves)

    def first_reading(self) -> str:
        plan = self.plan
        if not plan:
            return ""
        lines = ["A first reading of the request (check it against the request itself):"]
        if plan.get("product"):
            lines.append(f"- product: {plan.get('product')}")
        if self.basket:
            lines.append("- ONE ORDER OF SEVERAL PRODUCTS: the cart must end with one line per product below, each at its own quantity, and the budget covers the whole cart")
            for entry in self.basket:
                lines.append(f"- {entry['quantity']} x {entry['product']}" + (": " + "; ".join(entry["requirements"]) if entry["requirements"] else ""))
        else:
            lines.append(f"- quantity: {plan.get('quantity') or 1}")
        lines.extend(f"- stated requirement: {entry}" for entry in plan.get("requirements") or [])
        if plan.get("price_cap") is not None:
            lines.append(f"- price cap (total): {plan.get('price_cap')} {plan.get('currency') or ''}".rstrip())
        if plan.get("in_stock_only"):
            lines.append("- only items in stock / on hand")
        goal = plan.get("priority")
        if goal not in ("cheapest", "best"):
            lines.append("- no cheapest/best rule was found: any item that meets every requirement within the budget is right, so do not spend steps hunting for the lowest price (unless the request itself says otherwise)")
        else:
            detail = f" ({plan.get('priority_detail')})" if plan.get("priority_detail") else ""
            lines.append(f"- choose the {goal}{detail} among all fitting items")
        for entry in self.references():
            lines.append(f"- LOOK UP FIRST: the {entry.get('property')} for the {entry.get('product')} is set by the {entry.get('range')} listing sold by {entry.get('seller')} with {entry.get('identify')}; search that range, open the listings of that seller, take the one that matches, read its {entry.get('property')}, then require it" + (f" (allowed difference: {entry.get('tolerance')})" if entry.get("tolerance") else ""))
        lines.extend(f"- nice to have, NOT required (only a tie-breaker): {entry}" for entry in plan.get("soft_wants") or [])
        lines.extend(f"- must be learned from the shopper first: {entry}" for entry in plan.get("ask_first") or [])
        lines.extend(f"- rule: {entry}" for entry in plan.get("rules") or [])
        if self.message_budget is not None:
            lines.append(f"- messages to the shopper allowed: {self.message_budget}")
        lines.append(f"- steps allowed: {self.world.max_steps}")
        hints = [entry for entry in plan.get("hint_properties") or [] if isinstance(entry, str)]
        if hints:
            lines.append("- properties the story may hint at (ask about these first, one per message): " + ", ".join(hints))
        return "\n".join(lines)

    def status(self) -> str:
        left = self.world.steps_left()
        pieces = [f"[status] steps used {self.world.step} of {self.world.max_steps}, {left} left"]
        if self.message_budget is None:
            pieces.append(f"messages sent {self.messages_used}")
        else:
            pieces.append(f"messages sent {self.messages_used} of {self.message_budget}")
        pieces.append(f"about {int(max(0, self.clock.left() - FORCE_SECONDS))} s left")
        text = "; ".join(pieces) + "."
        text += "\n[requirements now] " + self.ledger.summary()
        if self.planner_on and self.role("message") and left > 14 and self.messages_used < self.message_limit():
            gaps = self.open_gap()
            if gaps:
                text += "\n[before the questions] the shopper's questions go out once enough listings are opened; open the details of these first (a tool that takes several ids at once saves steps): " + "; ".join(f"{product} ({opened} opened): {', '.join(ids)}" for product, opened, ids in gaps)
        if left <= 2:
            return text + " LAST STEPS: order now if the cart holds a verified fit, otherwise decline."
        if left <= 5 or self.clock.left() < WRAP_UP_SECONDS:
            return text + " FINISH NOW: fix the cart to one verified line, re-check its stock, then order (or decline if nothing fits)."
        if self.order_attempts == 0 and left <= 14:
            return text + " Steps are running low: within the next few steps put your best verified fit in the cart and order; searching and reading details end when 7 steps remain, and a refused order needs about 4 steps to repair."
        return text

    def offered_tools(self) -> list[dict[str, Any]]:
        left = self.world.steps_left()
        order, decline, cart = self.role("order"), self.role("decline"), self.role("cart")
        narrow: set[str | None] | None = None
        if left <= 1:
            narrow = {order, decline}
        elif left <= 3:
            narrow = {order, decline, cart}
        elif left <= 7 and (self.order_attempts == 0 or self.refusals > 0):
            narrow = {order, decline, cart, self.role("cart_view"), self.role("view")}
        tools = self.guard.tools
        if narrow is not None and order and cart:
            kept = [tool for tool in self.guard.tools if tool["function"]["name"] in narrow]
            if kept:
                tools = kept
        messenger = self.role("message")
        if self.planner_on and messenger:
            tools = [tool for tool in tools if tool["function"]["name"] != messenger]
        return tools

    # conversation helpers

    def post_tool_replies(self, replies: list[tuple[str, str]], asked_labels: list[tuple[str, str]] | None = None) -> None:
        last = len(replies) - 1
        for position, (call_id, text) in enumerate(replies):
            if position == last:
                text += "\n" + self.status()
            self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})
            if asked_labels is not None and position < len(asked_labels):
                prop, question = asked_labels[position]
                heard = text.split("\n[status]")[0][:400]
                self.asked.append(f"- {prop}: asked \"{question}\" -> {heard}")

    def message_call(self, call_id: str, args: dict[str, Any]) -> dict[str, Any]:
        return {"id": call_id, "type": "function", "function": {"name": self.role("message"), "arguments": json.dumps(args)}}

    def ask_planned(self, force: bool = False) -> bool:
        messenger = self.role("message")
        limit = self.message_limit()
        if not messenger or self.messages_used >= limit:
            self.planner_on = False
            return False
        products = [str(entry["product"]) for entry in self.basket] if self.basket else None
        menu, rich = self.ledger.property_menu(products)
        urgent = bool(self.plan.get("ask_first")) and not self.asked
        # a hidden wish is a property whose values differ across the range, which shows only once enough of the range is opened
        late = self.world.step >= min(16, int(self.world.max_steps * 0.35))
        if not (force or urgent or late or (rich >= 4 and not self.open_gap())):
            return False
        urgent = urgent and not force
        if self.world.steps_left() <= 10 or self.clock.left() < WRAP_UP_SECONDS + 40:
            self.planner_on = False
            return False
        count = limit - self.messages_used
        if not menu and not urgent:
            # nothing opened yet: keep the questions for later rather than asking blind
            return False
        planned = self.advisors.plan.questions(count + len(products or []), self.asked, menu)
        planned = self.grounded(planned, menu) + self.uncovered(planned, products)
        one_at_a_time = urgent and rich < 4
        if not one_at_a_time:
            planned = self.lead_with_pick(planned, products, menu)
        planned = self.spread(planned, products)
        planned = (planned[:1] if one_at_a_time else planned)[:count]
        calls: list[dict[str, Any]] = []
        labels: list[tuple[str, str]] = []
        for position, entry in enumerate(planned):
            wording = self.plain_question(entry)
            args = self.guard.message_args(messenger, wording)
            if not args:
                continue
            args, issues = self.guard.check_args(messenger, args)
            if issues:
                continue
            calls.append(self.message_call(f"ask_{self.messages_used + position + 1}_{self.world.step}", args))
            labels.append((str(entry.get("property")), wording))
        if not calls:
            self.planner_on = False
            return False
        topics = ", then ".join(prop for prop, _ in labels)
        self.messages.append({"role": "assistant", "content": "I will ask the shopper about " + topics + ".", "tool_calls": calls})
        self.post_tool_replies(self.run_calls(calls, planned=True), labels)
        if self.messages_used >= limit or not one_at_a_time:
            self.planner_on = False
        return True

    def lead_with_pick(self, planned: list[dict[str, Any]], products: list[str] | None, menu: str) -> list[dict[str, Any]]:
        # a product's first attribute question goes to the property its story points at; the plan's order follows
        for product in products or [str(self.plan.get("product") or "")]:
            if product in self.led:
                continue
            own = self.ledger.single_menu(product)[0] if products else menu
            choices: dict[str, str] = {}
            for line in own.splitlines():
                if line.startswith("- "):
                    label, _, rest = line[2:].partition(" (in ")
                    choices[label] = " | ".join(rest.split("): ", 1)[-1].split(" | ")[:2])
            if len(choices) < 2:
                continue
            self.led.add(product)
            key = self.advisors.plan.lead_key(product, choices)
            self.trace.append({"kind": "lead", "product": product, "key": key})
            if not key or any(line.startswith(f"- {key}:") for line in self.asked):
                continue
            lane = _words(product) if products else set()

            def same(entry: dict[str, Any]) -> bool:
                name = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip().lower()
                return name == key and (not lane or bool(_words(entry.get("product")) & lane))

            planned = [entry for entry in planned if not same(entry)]
            spot = next((place for place, entry in enumerate(planned) if str(entry.get("kind") or "attribute") != "must_learn" and (not lane or _words(entry.get("product")) & lane)), len(planned))
            planned.insert(spot, {"property": key, "kind": "attribute", "product": product if products else "", "question": ""})
        return planned

    @staticmethod
    def grounded(planned: list[dict[str, Any]], menu: str) -> list[dict[str, Any]]:
        # an attribute question must name an attribute the menu carries; the shopper only answers about a listed property
        labels = [line[2:].split(" (in ")[0] for line in menu.splitlines() if line.startswith("- ")]
        kept = []
        for entry in planned:
            name = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip().lower()
            if str(entry.get("kind") or "attribute") == "must_learn" or not labels or name in labels:
                kept.append(entry)
        return kept or planned

    def uncovered(self, planned: list[dict[str, Any]], products: list[str] | None) -> list[dict[str, Any]]:
        # a product the plan left without any question gets the best-splitting attribute of its own menu
        if not products:
            return []
        named = [_words(entry.get("product")) for entry in planned]
        asked = " ".join(self.asked).lower()
        extra = []
        for product in products:
            if any(words & _words(product) for words in named):
                continue
            menu, _ = self.ledger.single_menu(product)
            for line in menu.splitlines():
                label = line[2:].split(" (in ")[0]
                values = line.split("listings, ")[-1].split(" different")[0]
                if values.isdigit() and int(values) >= 2 and label not in asked and not ({w for w in _words(label) if len(w) > 2} & _words(self.ledger.summary())):
                    extra.append({"property": label, "kind": "attribute", "product": product, "question": ""})
                    break
        return extra

    @staticmethod
    def spread(planned: list[dict[str, Any]], products: list[str] | None) -> list[dict[str, Any]]:
        # one order of several products: take the questions in turn per product, so each product's hidden wish gets a chance
        if not products:
            return planned
        lanes: dict[int, list[dict[str, Any]]] = {}
        for entry in planned:
            words = _words(entry.get("product"))
            lane = max(range(len(products)), key=lambda spot: len(words & _words(products[spot]))) if words else -1
            if str(entry.get("kind") or "attribute") == "must_learn":
                lane = -1
            lanes.setdefault(lane, []).append(entry)
        order = lanes.pop(-1, [])
        while any(lanes.values()):
            for lane in sorted(lanes):
                if lanes[lane]:
                    order.append(lanes[lane].pop(0))
        return order

    def open_gap(self) -> list[tuple[str, int, list[str]]]:
        # per wanted product: how many listings of its range are opened, and the unopened in-stock ones still to open
        shelf = self.ledger.shelf
        products = [str(entry["product"]) for entry in self.basket] if self.basket else [str(self.plan.get("product") or "")]
        target = OPEN_BEFORE_ASK if not self.basket else max(4, OPEN_BEFORE_ASK - 3)
        gaps = []
        for product in products:
            label = shelf.main_range(product, strict=bool(self.basket))
            if not label:
                continue
            opened = sum(1 for ident in shelf.opened if shelf.range_of(ident) == label)
            waiting = [ident for ident in shelf.rows if ident not in shelf.opened and shelf.range_of(ident) == label and not shelf.sold_out(ident)]
            if opened < min(target, opened + len(waiting)):
                gaps.append((product, opened, waiting[:target - opened]))
        return gaps

    def needle_hint(self) -> str:
        # a search that brought no new listing: every word must match and few rows show, so point at single words
        # from the wanted values of the lines not yet in the cart that no search has used so far
        shelf, book = self.ledger.shelf, self.ledger
        wanted = [str(entry["product"]) for entry in self.basket] or [str(self.plan.get("product") or "")]
        held = {shelf.range_of(ident) for ident in book.cart_ids}
        needs = book.current if book.current is not None else [need for need in book.stated + book.revealed if need not in book.dropped]
        picks: list[str] = []
        for product in wanted:
            label = shelf.main_range(product, strict=bool(self.basket))
            if self.basket and (not label or label in held):
                continue
            own = _words(product)
            for need in needs:
                head, _, rest = str(need).partition(" - ")
                if rest and ":" in rest and ":" not in head:
                    if self.basket and shelf.main_range(head, strict=True) != label:
                        continue
                    need = rest
                _, values = _split_need(need)
                fresh = [word for word in sorted(values, key=len, reverse=True) if len(word) >= 4 and not word.isdigit() and word not in own and word not in self.used_needles and word not in picks]
                if fresh:
                    picks.append(fresh[0])
        if not picks:
            return ""
        return "That search added no new listing. Here every search word must match and few rows show, so search one word at a time from a wanted value, with the range filter; words not searched yet: " + ", ".join(picks[:4]) + "."

    def plain_question(self, entry: dict[str, Any]) -> str:
        # an attribute question is built in code: one attribute, the shop's own name for it, and the product only in a several-product order
        wording = str(entry.get("question")).strip()
        if str(entry.get("kind") or "attribute") == "must_learn":
            return wording
        label = re.sub(r"[_.]+", " ", str(entry.get("property") or "")).strip()
        if not label:
            return wording
        product = str(entry.get("product") or "").strip() if self.basket else ""
        return f"What {label} would you like for the {product}?" if product else f"What {label} would you like?"

    def send_reply(self, text: str) -> bool:
        messenger = self.role("message")
        if not messenger or self.messages_used >= self.message_limit() or self.world.steps_left() <= 4 or self.clock.left() < WRAP_UP_SECONDS:
            return False
        args = self.guard.message_args(messenger, text)
        if not args:
            return False
        args, issues = self.guard.check_args(messenger, args)
        if issues:
            return False
        call = self.message_call(f"reply_{self.world.step}", args)
        self.messages.append({"role": "assistant", "content": "I will answer the shopper's question.", "tool_calls": [call]})
        self.post_tool_replies(self.run_calls([call], planned=True))
        return True

    # call screening

    def quantity_warning(self, name: str, args: dict[str, Any]) -> str:
        if self.basket:
            return ""
        wanted = self.plan.get("quantity")
        if isinstance(wanted, bool) or not isinstance(wanted, int) or wanted < 1:
            return ""
        known = self.guard.props(name)
        signature = self.guard.enum_signature(name, args)
        if self.ledger.add_signature is not None and signature != self.ledger.add_signature:
            return ""
        for key, spec in known.items():
            if isinstance(spec, dict) and spec.get("type") == "integer" and "enum" not in spec:
                asked_for = args.get(key, spec.get("default", 1))
                if isinstance(asked_for, int) and asked_for != wanted:
                    return f"the request needs a quantity of {wanted} of one listing, but this call asks for {asked_for}."
        return ""

    def cart_gate(self, name: str, args: dict[str, Any]) -> str:
        known = self.guard.props(name)
        signature = self.guard.enum_signature(name, args)
        if self.ledger.add_signature is None or signature == self.ledger.add_signature:
            target = next((v for k, v in args.items() if isinstance(v, str) and "enum" not in (known.get(k) or {}) and v.strip() and " " not in v.strip()), None)
            if target and f"{target}|{self.ledger.version}" not in self.cart_checked:
                self.cart_checked.add(f"{target}|{self.ledger.version}")
                return self.advisors.verify.check(target, "ADD TO THE CART")
        return ""

    def screen(self, call: dict[str, Any], allowed: set[str], planned: bool, count: int, queued_messages: int) -> tuple[str | None, Any]:
        function = call.get("function") or {}
        name = function.get("name")
        raw = function.get("arguments")
        messenger, cart = self.role("message"), self.role("cart")
        finishing = self.finishers()
        if not self.guard.knows(name):
            return f"Not run (no step used): '{name}' is not one of the tools. Use one of: {', '.join(sorted(allowed))}.", None
        if not planned and name not in allowed:
            if name == messenger and self.planner_on:
                reason = "questions to the shopper are planned and sent for you after you have opened a few listings' details"
            else:
                reason = "too few steps are left for this; only these tools can be used now: " + ", ".join(sorted(allowed))
            return f"Not run (no step used): {reason}.", None
        if isinstance(raw, dict):
            args = raw
        elif (raw or "").strip():
            args = _decode_object(raw or "{}")
            if args is None:
                return "Not run (no step used): the arguments were not a valid JSON object.", None
        else:
            args = {}
        args, issues = self.guard.check_args(name, args)
        if issues:
            return "Not run (no step used): invalid arguments: " + "; ".join(issues[:6]), None
        if name in finishing and count > 1:
            return "Not run (no step used): an order or a decline must be sent alone, after you have seen the results of your other calls.", None
        if name == messenger and self.message_budget is not None and self.messages_used + queued_messages >= self.message_budget:
            return "Not run (no step used): the message budget is used up. Do not message the shopper again.", None
        signature = name + _dump(args)
        if signature in self.recent and name not in finishing and name not in (messenger, cart):
            return "Not run (no step used): you just made this exact call and nothing has changed since; its result is above. Change the arguments or do something else.", None
        if name == cart:
            if self.world.steps_left() > 3:
                verdict = self.cart_gate(name, args)
                if verdict:
                    return "Not run (no step used). " + verdict + " If you are certain it fits, send the same call again.", None
            warning = self.quantity_warning(name, args)
            if warning and signature not in self.qty_warned:
                self.qty_warned.add(signature)
                return "Not run (no step used): " + warning + " If you are sure, send the same call again.", None
        return None, (name, args)

    def gate_finish(self, name: str, args: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        # one tool may both buy and refuse; then a filled cart means a purchase and an empty one a refusal
        merged = self.role("order") == self.role("decline")
        buying = name == self.role("order") and (not merged or bool(self.ledger.cart_ids))
        if buying:
            if not self.order_checked and self.world.steps_left() > 2 and self.clock.left() > WRAP_UP_SECONDS:
                self.order_checked = True
                lines = self.ledger.cart_ids[-len(self.basket):] if self.basket else self.ledger.cart_ids[-1:]
                verdict = " ".join(v for v in (self.advisors.verify.check(line, "ORDER") for line in lines[:4]) if v)
                return ("Not run yet (no step used). " + verdict) if verdict else ORDER_CHECKLIST, args
            hint = self.cheaper_left()
            return (("Not run yet (no step used). " + hint) if hint else None), args
        if name != self.role("decline"):
            return None, args
        if not self.decline_checked and self.world.steps_left() > 6 and self.clock.left() > WRAP_UP_SECONDS + 30:
            self.decline_checked = True
            audit = self.advisors.audit.review()
            return ("Not run yet (no step used). " + audit) if audit else DECLINE_CHECKLIST, args
        proof = self.decline_gap()
        if proof:
            return "Not run yet (no step used). " + proof, args
        return None, self.scrub_decline(args, name)

    def references(self) -> list[dict[str, Any]]:
        return [entry for entry in self.plan.get("references") or [] if isinstance(entry, dict) and str(entry.get("seller") or "").strip()]

    def seller_rows(self, observation: Any) -> str:
        # rows that carry a referenced seller's name in any of their own fields
        notes = []
        for entry in self.references():
            name = str(entry.get("seller")).strip().lower()
            hits = []
            for found in _nested_dicts(observation):
                ident = _row_id(found)
                if ident and any(isinstance(part, str) and part.strip().lower() == name for part in found.values()):
                    hits.append(ident)
            if hits:
                notes.append(f"listings sold by {entry.get('seller')} in this result: " + ", ".join(dict.fromkeys(hits)) + f" (open them to find the one with {entry.get('identify')})")
        return " ".join(notes)

    def quantity(self) -> int:
        wanted = self.plan.get("quantity")
        return wanted if isinstance(wanted, int) and not isinstance(wanted, bool) and wanted > 0 else 1

    def cheaper_left(self) -> str:
        # under a cheapest rule, no cheaper same-range listing within budget may stay unjudged
        # a hold needs room to act on it: opening, judging and swapping a line take several steps
        if self.plan.get("priority") != "cheapest" or not self.ledger.cart_ids or self.price_holds >= SHELF_HOLDS:
            return ""
        if self.world.steps_left() <= 8 or self.clock.left() < WRAP_UP_SECONDS + 20:
            return ""
        shelf = self.ledger.shelf
        verify = self.advisors.verify
        units = 1 if self.basket else self.quantity()
        cap = self.ledger.budget_now if _plain_number(self.ledger.budget_now) else None
        lines = self.ledger.cart_ids[-len(self.basket):] if self.basket else self.ledger.cart_ids[-1:]
        judged: list[tuple[float, str]] = []
        unread: list[tuple[float, str]] = []
        for pick in lines:
            cost = shelf.price(pick)
            if cost is None:
                continue
            label = shelf.range_of(pick)

            def cheaper(ident: str) -> bool:
                value = shelf.price(ident)
                return ident not in lines and value is not None and value < cost and not shelf.sold_out(ident) and (not label or shelf.range_of(ident) == label) and (cap is None or value * units <= cap)

            # a cheaper listing the loop opened but skipped goes to the checker before it is dropped
            for _, ident in sorted((shelf.price(i), i) for i in shelf.opened if cheaper(i) and i not in verify.passed)[:3]:
                verify.check(ident, "BUY INSTEAD")
            judged += sorted((shelf.price(i) or 0.0, i) for i in verify.passed if cheaper(i))
            if self.world.steps_left() > 10:
                unread += shelf.unopened(label, cap, units, below=cost)[:6]
        if not judged and not unread:
            return ""
        self.price_holds += 1
        parts = []
        if judged:
            parts.append("these listings passed the independent requirement check and are priced below the cart line of their range: " + ", ".join(f"{i} ({p:g})" for p, i in judged[:4]) + " (the shop's category label settles what kind of item a listing is, so a title or type field naming another kind does not rule it out)")
        if unread:
            parts.append("these listings of the same range within the budget are priced below the cart line and were never opened: " + ", ".join(f"{i} ({p:g})" for p, i in unread[:6]))
        return "The request asks for the cheapest fit, and " + "; ".join(parts) + ". Open and judge the cheaper ones, cart the cheapest that meets every requirement, then order. If each of them fails, send the order again."

    def decline_gap(self) -> str:
        # a refusal stands only when no audited fit survives the check and no in-budget listing of the range is left unopened
        if self.decline_holds >= SHELF_HOLDS or self.world.steps_left() <= 4 or self.clock.left() < WRAP_UP_SECONDS + 30:
            return ""
        shelf = self.ledger.shelf
        survivors = []
        for ident in self.advisors.audit.fits[:3]:
            if ident not in self.advisors.verify.passed and self.ledger.records_of(ident):
                self.advisors.verify.check(ident, "ORDER")
            if ident in self.advisors.verify.passed:
                survivors.append(ident)
        wanted = [entry["product"] for entry in self.basket] or [str(self.plan.get("product") or "")]
        unread = [i for name in wanted for _, i in shelf.unopened(shelf.main_range(name), self.ledger.budget_now, 1 if self.basket else self.quantity())[:3 if self.basket else 6]]
        if not survivors and not unread:
            return ""
        self.decline_holds += 1
        parts = []
        if survivors:
            parts.append("these listings passed the requirement check, so the shop does have a fit: " + ", ".join(survivors) + "; cart the best of them and order it")
        if unread:
            parts.append("these listings of the same range within the budget were never opened: " + ", ".join(unread) + "; open them before deciding")
        return "Before a refusal: " + "; ".join(parts) + ". Decline only if every one of them fails a requirement."

    def scrub_decline(self, args: dict[str, Any], name: str) -> dict[str, Any]:
        rule = self.plan.get("decline_no_names")
        if rule is False:
            return args
        leaks = any(
            token and token in self.ledger.seen_ids
            for value in args.values() if isinstance(value, str)
            for token in re.split(r"[\s.,;:!?()\[\]{}<>\"'`]+", value)
        )
        if not (rule or leaks):
            return args
        known = self.guard.props(name)
        scrubbed = {}
        for key, value in args.items():
            spec = known.get(key) or {}
            scrubbed[key] = _DECLINE_TEXT[: spec.get("maxLength", len(_DECLINE_TEXT))] if isinstance(value, str) and "enum" not in spec else value
        return scrubbed

    # execution

    def run_calls(self, calls: list[dict[str, Any]], planned: bool = False) -> list[tuple[str, str]]:
        replies: list[tuple[str, str]] = []
        allowed = {tool["function"]["name"] for tool in self.offered_tools()}
        finishing = self.finishers()
        messenger = self.role("message")
        ready: list[tuple[str, str, dict[str, Any]]] = []
        queued_messages = 0
        for call in calls:
            refusal, accepted = self.screen(call, allowed, planned, len(calls), queued_messages)
            if refusal is not None:
                replies.append((call["id"], refusal))
                continue
            name, args = accepted
            if name == messenger:
                queued_messages += 1
            ready.append((call["id"], name, args))
        if ready and ready[0][1] in finishing:
            call_id, name, args = ready[0]
            held, args = self.gate_finish(name, args)
            if held is not None:
                replies.append((call_id, held))
                return replies
            ready[0] = (call_id, name, args)
        elif ready and finishing:
            room = max(0, self.world.steps_left() - 1)
            if room < len(ready):
                replies.extend((call_id, "Not run (no step used): only the last step is left, and it must be the order (or the decline).") for call_id, _, _ in ready[room:])
                ready = ready[:room]
        for offset in range(0, len(ready), 16):
            self.execute(ready[offset : offset + 16], replies)
        position = {call["id"]: index for index, call in enumerate(calls)}
        replies.sort(key=lambda pair: position.get(pair[0], 0))
        return replies

    def execute(self, batch: list[tuple[str, str, dict[str, Any]]], replies: list[tuple[str, str]]) -> None:
        if self.world.over():
            replies.extend((call_id, "Not run: the episode is over or no steps are left.") for call_id, _, _ in batch)
            return
        batch = batch[: self.world.steps_left()]
        actions = [{"name": name, "args": args} for _, name, args in batch]
        first_step = self.world.step + 1
        answer = self.world.call(actions)
        self.trace.append({"kind": "env", "step": first_step, "actions": actions, "reply": answer})
        if answer is None:
            replies.extend((call_id, "The call failed (network or server). Try again.") for call_id, _, _ in batch)
            return
        observations = answer.get("observations") if isinstance(answer.get("observations"), list) else [answer.get("observation")]
        for index, (call_id, name, args) in enumerate(batch):
            if index < len(observations):
                text, note = self.absorb(name, args, observations[index])
                if index == len(observations) - 1 and answer.get("finished"):
                    text += " [the episode is finished]"
                replies.append((call_id, text + "\n[note] " + note if note else text))
            else:
                replies.append((call_id, "Not run: the episode finished before this call."))

    def book_cart(self, name: str, args: dict[str, Any]) -> None:
        known = self.guard.props(name)
        signature = self.guard.enum_signature(name, args)
        book = self.ledger
        if book.add_signature is None:
            book.add_signature = signature
        for key, value in args.items():
            spec = known.get(key) if isinstance(known.get(key), dict) else {}
            if isinstance(value, str) and "enum" not in spec and value.strip() and " " not in value.strip():
                if signature == book.add_signature:
                    book.cart_ids = [entry for entry in book.cart_ids if entry != value] + [value]
                else:
                    book.cart_ids = [entry for entry in book.cart_ids if entry != value]

    def absorb(self, name: str, args: dict[str, Any], observation: Any) -> tuple[str, str]:
        messenger, cart = self.role("message"), self.role("cart")
        finishing = self.finishers()
        if name == messenger:
            self.messages_used += 1
        if name == self.role("order"):
            self.order_attempts += 1
        if name == cart:
            self.cart_used = True
            self.order_checked = False
        self.recent = [] if name in (cart, messenger) or name in finishing else (self.recent + [name + _dump(args)])[-2:]
        self.ledger.remember(observation)
        before = len(self.ledger.shelf.rows)
        self.ledger.shelf.take(observation, name in {self.role("details"), self.role("view")} - {None})
        if name == self.role("search"):
            self.used_needles |= set().union(*[_words(value) for value in args.values() if isinstance(value, str)])
            self.dry_search = len(self.ledger.shelf.rows) == before
        if name == cart:
            self.book_cart(name, args)
        if name == self.role("cart_view"):
            self.ledger.last_cart_view = observation
            self.ledger.shelf.learn_price(observation)
        note = self.observe(name, observation)
        if name == self.role("search") and self.dry_search:
            hint = self.needle_hint()
            if hint:
                note = f"{note} {hint}".strip()
        lookup = self.seller_rows(observation) if name == self.role("search") else ""
        if lookup:
            note = f"{note} {lookup}".strip()
        heard: list[str] = []
        if name == messenger:
            if isinstance(observation, str):
                heard = [observation]
            elif isinstance(observation, dict) and "error" not in observation:
                _short_sentences([v for v in observation.values() if isinstance(v, (str, list))], heard)
            question = " ".join(str(v) for v in args.values() if isinstance(v, str))
        else:
            question = None
            if isinstance(observation, dict):
                for value in observation.values():
                    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                        _short_sentences(value, heard)
                    elif isinstance(value, str) and len(value.split()) >= 4 and ("refusal" in observation):
                        heard.append(value)
        if heard:
            meaning, reply = self.advisors.interpret.read(heard, question)
            if reply and self.role("message"):
                self.pending_reply = reply
            if meaning:
                note = f"{note} {meaning}" if note else meaning
        return _dump(observation), note

    def observe(self, name: str, observation: Any) -> str:
        notes = []
        if isinstance(observation, dict):
            self.ledger.note_ids(observation)
            if observation.get("error") == "budget_exceeded":
                if name == self.role("message") or self.message_budget is not None:
                    self.message_budget = self.messages_used if self.message_budget is None else min(self.message_budget, self.messages_used)
                notes.append("That budget is used up; do not call this tool again.")
            if "refusal" in observation:
                self.refusals += 1
                self.order_checked = False
                notes.append(REFUSAL_NOTE)
            elif len(observation) >= 2 and name != self.role("message"):
                lines = _string_lines(observation)
                if lines:
                    notes.append("This result carries a message from the shopper (a change). Apply it: " + " | ".join(lines[:4]))
            if name == self.role("search") and not self.paging_noted:
                gap = _unpaged(observation)
                if gap:
                    self.paging_noted = True
                    notes.append(f"This search reports {gap[0]} matches but shows {gap[1]} and offers no further page; to see the rest, narrow it (more precise words, a filter or price bounds) instead of guessing a page value.")
        elif isinstance(observation, list):
            self.ledger.note_ids(observation)
        return " ".join(notes)

    # finishing

    def finish_values(self, job: str) -> dict[str, Any]:
        found = self.plan.get("finish_values")
        values = found.get(job) if isinstance(found, dict) else None
        return values if isinstance(values, dict) else {}

    def force_finish(self) -> None:
        if self.world.over() or self.clock.left() < 2:
            return
        order, decline = self.role("order"), self.role("decline")
        if order and self.cart_used:
            action = {"name": order, "args": self.guard.filler_args(order, "Order the cart.", self.finish_values("order"))}
        elif decline:
            action = {"name": decline, "args": self.guard.filler_args(decline, _DECLINE_TEXT, self.finish_values("decline"))}
        else:
            return
        answer = self.world.call([action])
        self.trace.append({"kind": "forced_finish", "action": action, "reply": answer})
        refused = bool(answer) and not answer.get("finished") and isinstance(answer.get("observation"), dict) and "refusal" in answer["observation"]
        if refused and decline and self.world.steps_left() > 0:
            action = {"name": decline, "args": self.guard.filler_args(decline, _DECLINE_TEXT, self.finish_values("decline"))}
            answer = self.world.call([action])
            self.trace.append({"kind": "forced_finish", "action": action, "reply": answer})

    # main loop

    def compact_history(self) -> None:
        # older results are compacted a block at a time, so the cached prefix holds for many turns between passes
        if sum(len(str(m.get("content") or "")) for m in self.messages) < HISTORY_CHARS:
            return
        edge = len(self.messages) - KEEP_RECENT
        if edge <= self.compacted or sum(len(str(m.get("content") or "")) for m in self.messages[self.compacted:edge]) < COMPACT_BLOCK:
            return
        start, self.compacted = self.compacted, edge
        reader = json.JSONDecoder()
        for spot, message in list(enumerate(self.messages[:edge]))[start:]:
            text = message.get("content")
            if message.get("role") != "tool" or not isinstance(text, str) or len(text) < 1200:
                continue
            rest = text.split("\n[status]")[0]
            try:
                body, end = reader.raw_decode(rest)
            except ValueError:
                self.messages[spot] = dict(message, content=rest[:1200] + "...(cut)")
                continue
            self.messages[spot] = dict(message, content=_dump(_shrink(body)) + rest[end:])

    def opening_message(self) -> str:
        opening = self.world.opening
        text = "SHOPPER REQUEST:\n" + self.world.query
        if opening is not None:
            text += "\n\nOPENING OBSERVATION:\n" + _dump(opening)
        reading = self.first_reading()
        if reading:
            text += "\n\n" + reading
        text += "\n\n" + self.status()
        if opening is not None:
            self.ledger.remember(opening)
            self.ledger.note_ids(opening)
        return text

    def normalise_calls(self, message: dict[str, Any], counter: int) -> tuple[list[dict[str, Any]], int]:
        calls = []
        for raw in message.get("tool_calls") or []:
            if not isinstance(raw, dict):
                continue
            counter += 1
            function = raw.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments or {})
            calls.append({"id": raw.get("id") or f"call_{counter}", "type": "function", "function": {"name": function.get("name"), "arguments": arguments}})
        return calls, counter

    def turn_loop(self) -> None:
        idle = failures = counter = 0
        for _ in range(MAX_TURNS):
            if self.world.over():
                return
            if self.clock.left() < FORCE_SECONDS or self.clock.cut:
                # past the runaway spend line no further LLM turn is taken; code finishes with what the cart holds
                self.force_finish()
                return
            if self.pending_reply:
                text, self.pending_reply = self.pending_reply, ""
                if self.send_reply(text):
                    continue
            if self.planner_on and self.ask_planned():
                continue
            self.compact_history()
            message = self.llm.ask(self.messages, self.offered_tools(), [MAIN_MODEL, BACKUP_MODEL, MAIN_MODEL], 2500)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.force_finish()
                    return
                continue
            calls, counter = self.normalise_calls(message, counter)
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _dump(content)
            committing = {self.role("cart"), self.role("order"), self.role("decline")} - {None}
            if self.planner_on and any(call["function"]["name"] in committing for call in calls) and self.ask_planned(force=True):
                continue
            if calls:
                idle = 0
                self.trace.append({"kind": "model", "content": content[:2000], "calls": [call["function"] for call in calls]})
                self.messages.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                self.post_tool_replies(self.run_calls(calls))
                continue
            idle += 1
            self.messages.append({"role": "assistant", "content": content or "(no tool call)"})
            self.messages.append({"role": "user", "content": "Every reply must call at least one of the supplied tools. Continue.\n" + self.status()})
            if idle >= 3:
                self.force_finish()
                return

    def run(self) -> list[dict[str, Any]]:
        if self.world.finished:
            return self.trace
        self.adopt_plan(self.advisors.setup.read())
        self.trace.append({"kind": "plan", "plan": self.plan, "roles": self.roles})
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": self.opening_message()}]
        self.turn_loop()
        if not self.world.finished and not self.world.dead:
            self.force_finish()
        self.trace.append({"kind": "end", "finished": self.world.finished, "steps": self.world.step, "cost": round(self.llm.cost, 5), "spend": {k: [v[0], round(v[1], 5)] for k, v in self.llm.spend.items()}, "seconds": round(self.clock.elapsed(), 1)})
        return self.trace


def assemble(problem_data: dict[str, Any]) -> Driver:
    clock = Clock()
    trace: list[dict[str, Any]] = []
    world = WorldDesk(clock, trace, problem_data["environment"])
    guard = SchemaGuard(world.policy)
    llm = LLMDesk(clock, trace)
    ledger = Ledger()
    advisors = Advisors(
        SetupAdvisor(llm, world, guard),
        InterpretAdvisor(llm, ledger, clock, world.query),
        PlanAdvisor(llm, ledger, world.query),
        VerifyAdvisor(llm, ledger, clock, trace, world.query),
        AuditAdvisor(llm, ledger, clock, trace, world.query),
    )
    return Driver(clock, trace, llm, world, guard, ledger, advisors)


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    driver = assemble(problem_data)
    try:
        return driver.run()
    except Exception as exc:
        driver.trace.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:500]})
        try:
            if not driver.finished and not driver.dead:
                driver.force_finish()
        except Exception as again:
            driver.trace.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:500]})
        return driver.trace


__all__ = ["agent_main"]
