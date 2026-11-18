"""Winner: a shopping agent for ORO world sessions (validator v3, runtime contract 8).

Every world brings its own tools, wording and rules, so the agent reads them from the task. A helper model first turns
the request into a task sheet: the items with their quantities and requirements, the budget, how to choose, the
budgets for messages and steps, the procedures the request asks for along the way, and the world's tools by job.
A driver model then works the store one tool call per step with the sheet beside it. The code around the model keeps
the session inside its budgets (steps, messages, seconds), keeps every listing field readable, and holds the first
finishing call for an independent review of the whole record against the request, so a missing procedure, a wrong
cart line or an unproven cheapest pick is fixed before the order goes out.
"""

from __future__ import annotations

import json
import re
from os import getenv
from time import monotonic
from typing import Any

from src.agent.proxy_client import ProxyClient

# ----------------------------------------------------------------------------- settings
MODELS = {
    "openrouter": {
        "drive": ["z-ai/glm-5.3", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"],
        "recover": ["z-ai/glm-5.3", "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"],
        "think": ["anthropic/claude-sonnet-5.5", "z-ai/glm-5.3", "anthropic/claude-haiku-5.5"],
    },
    "chutes": {
        "drive": ["zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"],
        "recover": ["zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"],
        "think": ["zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE", "deepseek-ai/DeepSeek-V3.2-TEE"],
    },
}
# The network caps inference per run ($2 for 30 qualifying tasks, $5 for about 90 race tasks; past the cap every
# running task scores 0). Each job spends at most this much per task on the dearer models, then the cheap one:
# the task sheet. Past TASK_SPEND in all, every call goes to the cheap model.
SPEND = {"drive": 0.0, "think": 0.005, "recover": 0.0}
TASK_SPEND = 0.06
DEAR = ("anthropic/claude-sonnet", "anthropic/claude-opus")
CHEAP = "anthropic/claude-haiku-5.5"
PRICES = {"anthropic/claude-sonnet": (2.0, 10.0), "anthropic/claude-haiku": (0.1, 0.5), "z-ai/glm-5.3": (0.07, 1.4)}   # $ per million in, out
OTHER_PRICE = (1.5, 6.0)
TASK_SECONDS = 270.0         # the network allows 300 s per task; the rest is margin for the finishing call
WRAP_UP_SECONDS = 205.0       # from here on the model is told to finish
LATE_SECONDS = 20.0           # time given to the last finishing call
CALL_SECONDS = 45.0           # longest wait for one model reply
SHEET_SECONDS = 40.0          # longest wait for the task sheet
REVIEW_SECONDS = 35.0         # longest wait for one review
REVIEW_UNTIL = 228.0          # no review starts after this
REVIEW_STEPS = 4              # no review when this few steps are left (its fixes would not fit)
REVIEWS = 2                  # holds of one finishing attempt at most (a refused finish starts a new attempt)
DECLINE_HOLDS = 2            # declines held back before one is let through (declines almost never score)
FINISH_RESERVE = 6            # steps kept for cart, cart view, checkout, a refused checkout and its recovery
COMMIT_AFTER = 10             # records read without a cart line before the model is told to commit
CART_BY = 4                   # steps left at which only cart and finishing calls go out while the cart is empty
REPEAT_LIMIT = 6000           # characters of an earlier answer given back for a repeated information call
GUARD_LIMIT = 3              # calls those step guards hold back at most (then the model's call goes out)
BATCH_LIMIT = 6              # tool calls sent from one model reply
REPLY_LIMIT = 200             # characters kept of one shopper answer in the status line
NOTE_LIMIT = 220              # characters kept of one store note in the status line
CHEAP_SHOWN = 8               # cheapest in-stock listings shown to the review
MAX_TOKENS = 1800             # completion tokens per driver call
THINK_TOKENS = 2500           # completion tokens per sheet or review call
FIELD_LIMIT = 1600            # characters kept of one ordinary field of a listing
BLURB_LIMIT = 170             # characters kept of one long free-text field (description, blurb) in a list of rows
DETAIL_BLURB_LIMIT = 900      # the same, when the observation shows one listing
OBSERVATION_LIMIT = 14000     # characters kept of one observation
RECORD_LIMIT = 600            # characters of one observation kept in the review record
DETAIL_RECORD_LIMIT = 4500   # the same for an observation showing one to four listings (their full records)
RECORD_ROWS = 12              # result rows kept per search in the review record
TRIM_AT = 240000              # characters of conversation after which older observations are cut
KEEP_WHOLE = 10               # latest observations never cut
TRIMMED = 1500                # characters kept of an older observation
IDLE_LIMIT = 3               # replies in a row without a tool call before the model is replaced
HARD_SECONDS = 292.0          # nothing is sent after this (the network stops the task at 300 s)
STEP_SECONDS = 20.0           # longest wait for one environment reply
STEP_TRIES = 2                # sends of one step (a replayed step is answered from the record)

NUMBER_WORDS = {"no": 0, "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10}
ID_FIELD = re.compile(r"(?:^|[_\s.])(?:p?id|sku|ref|code|number)$|^id|identifier|^ref", re.I)
PRICE_FIELD = re.compile(r"price|cost", re.I)
TITLE_FIELD = re.compile(r"title|heading|headline|name", re.I)
GROUP_FIELD = re.compile(r"category|department|shelf|aisle|section", re.I)
STOCK_FIELD = re.compile(r"stock|available|on_?hand", re.I)
BLURB_FIELD = re.compile(r"^(?!.*(?:title|heading|headline))(?:.*(?:blurb|descr|about|overview|bullet|summary)"
                         r"|.*(?:details?|features?|text)$)", re.I)
LISTING_ID = re.compile(r"\b[a-z]{1,4}-[0-9a-f]{6,}\b")
DETAILS = re.compile(r"\b(?:facts|traits(?:_map)?|attributes|properties|specs|specifications)\"?\s*[:=]", re.I)
FAILED = re.compile(r"\"(?:error|call_trouble)\"|not found|no such|invalid|resolves to no|does not exist|unknown (?:id"
                    r"|listing|product)|\"worked\": false|\"ok\": false|\"success\": false", re.I)
NOT_UNDERSTOOD = re.compile(r"\b(?:put it (?:differently|simpler)|another way|simpler|fewer|one by one|one at a time"
                            r"|split (?:them|it|those|the)|separately|trickle|flood|one (?:subject|query|thing|topic)"
                            r"|first thing|repeat|once more|(?:send|try|say|ask|explain|give)\b[^.?!]{0,20}\bagain"
                            r"|lost (?:me|my place)|didn.t catch|couldn.t (?:quite )?grasp|not sinking|blur|gray wash"
                            r"|muddy|what were you getting at|come again|pardon|another go)\b|\beh\?", re.I)
DECIDED = re.compile(r"\b(?:must|need|needs|has to|have to|should|only accept|insist|holding to|settled|stick to"
                     r"|nothing else|any|anything|anyone|don.t (?:really )?mind|no (?:pick|view|preference|side|pull)"
                     r"|wide open|you choose|surprise me|whatever|flexible|not (?:bothered|fussy|attached|picky))\b",
                     re.I)

_proxy = ProxyClient(timeout=int(CALL_SECONDS), max_retries=1)
_world = ProxyClient(timeout=int(STEP_SECONDS), max_retries=1)

GUIDE = """You are a store assistant buying for one shopper. You act only through the supplied tools: every reply must contain one or more tool calls, never plain text alone. Each tool call uses one step of the task, including a call that fails.

How the task is graded: it scores 1 only when the finish is exactly right, else 0. A purchase is right when the cart holds every item the shopper asked for, each at the requested quantity, each listing meeting every requirement in force at checkout (stated ones, preferences the shopper revealed, changes from notes), in stock, the total within the budget, the cheapest or best one where the request asks for that, and every procedure the request asks for done at its time. When the store truly has nothing that meets everything, the right finish is the decline tool, used the way the request asks.

The task sheet: a helper read the request into a task sheet (items, requirements, budget, choice rule, budgets, procedures, tools by job). Use it as your notes; when it and the request disagree, the request wins.

The store:
- Tool names, arguments and result shapes belong to this store; read each tool's description and use the exact argument names and allowed values.
- Search results are short summaries and may come in no price order, a few rows per page; compare the reported total with the rows you have seen and page on when the choice depends on seeing them all. A filter on the category with a price limit narrows a large set. Keyword searches work best short: the wanted value plus the product noun. A filter that returns nothing usually means the value is spelled differently from the store's labels; copy labels exactly from result rows.
- Judge requirements from the listing's own record: its attributes (facts, properties, specifications), title and description. Read the full record of every listing you may buy. A tool that shows the full records of several listings at once (a compare or side-by-side tool) saves steps: prefer it to opening listings one by one.
- Some stores split a listing over two tools: one shows its attributes, another its page or offer (price, stock, description and the seller's remarks or pointers). The remarks can change a value that counts ("for the pattern, refer to a different product ..."), so a listing that seems to miss a requirement by its attributes may fit after all, and one that seems to fit may not. Before you buy a listing or rule out a near fit, see both views of it.
- Call tools with a listing's id exactly as the id field shows it, never a model or part code from its title.
- Stay focused: passing runs read about seven listing records; past about thirteen you are thrashing, so commit to the best fitting candidates.
- The kind of product is decided by the listing's category: only the category that is the requested item counts, never a neighbouring one.
- A requirement on a property is met when the listing's attribute for that property states the value (any key spelling; one element of a list is enough; plain synonyms, case and plural do not matter; a value containing the wanted words counts, "deep teal" is teal). A different value fails even if the title says otherwise. When no attribute names that property, the value stated elsewhere in the record counts, but a number of another kind does not (a weight is not a capacity).
- A requirement can point at another listing instead of giving a value ("the same colour as the kettle sold by X", "cheaper than the one on the page of ..."). Find that listing, open its full record, follow any page it refers you to, and take the value from there before choosing.
- Seller remarks on a listing (shop comments, notes) are rules, read every one:
  - A redirect ("for the number of cameras, look to the X smartphone with ..., not to this item") means this listing's value for that property is the value shown on that other listing: find it (by seller, kind and the properties named), open its full record, follow any further redirect, and use the final value. Never buy a listing whose redirect you have not resolved.
  - A pairing rule ("five goes together with dual, triple with single"; "1 tb goes with 128 gb") maps the listing's stated value to the value that counts: translate the value through the rule before matching a requirement.
  - A remark like "this is not the version with 6000 mah" tells near-identical listings apart; use it to find the exact listing a reference means.
- When one requested item must fit or work with another, choose the pair together from the listings' own words (what it fits, a shared model or series code).
- Never buy a listing shown as sold out or unavailable. The total (price times quantity over all lines) must not exceed the budget; equal is fine.

The shopper:
- The shopper has firm preferences the request does not state, each about one property of one requested item (for example material, type, style, pattern, colour, capacity, size, plug or interface, resolution, voltage, flavour, pack type). The shopper reveals one only when a message asks about that property (or one close to it). Open questions ("what will you use it for?") reveal nothing.
- Unasked preferences are the most common reason an order fails: an order placed without asking about the property the shopper cares about scores 0 even when the listing happens to fit. So use every message the request allows (keep one for each procedure that needs a message), and ask BEFORE you put anything in the cart.
- Ask once a first search and one or two records show which properties this kind of product has. Shoppers differ: some take a list of properties in one message and answer the one they care about, others get lost in a list and want one topic per message. So:
  - First message: one item and a short list of four to six of its open properties, likeliest first, by the attribute names its records use: "For the bed sheets, which material, sheet type, pattern, colour and thread count would you like?"
  - If the shopper answers it (a preference, or that they do not mind), keep asking short lists of different properties.
  - If the reply shows the list was not taken in (asks to split it up, go one by one, slow down, put it simpler, or did not understand), from then on ask exactly ONE property per message, in this plain form: "Which material would you like for the bed sheets?". Properties named only in a list that was not taken in are still unasked: ask the likeliest of them again, one by one.
  - Never offer values joined by "or".
- Ask the likeliest property first: one the request's story hints at (who uses it, the look, the setting), else the property that defines the product's kind or look and whose values differ across its records (material, type such as sheet type or card type, style, pattern, colour, capacity, plug or interface, resolution or megapixels, voltage). Skip properties the request already settles, identifiers, model names, warranty, and gender or age group unless hinted. With several items, spread the messages over the items, the item with most open properties first.
- Some shoppers are tired or distracted and sometimes do not understand even a simple one-property question. That message is spent and tells you nothing, not even "no preference": ask the next likeliest property in the same plain form (asking the same one again works no better).
- Never spend a message on asking for approval, confirming a pick or reporting progress unless the request asks for that step: every message is better used on one more property.
- An answer such as "no preference" or "don't mind" means that property is free. A stated preference is a requirement like any other, even when it names a different property than you asked about ("material has to be cotton"): search with it and check it in each record. An answer that points at another listing ("identical to the one on the page of X's listing with ...") is a reference: find that listing, open it, and take the value from there.

Procedures: the request can ask for steps along the way, such as getting the shopper's approval before ordering, telling the shopper something, checking or reading something first, or using a particular tool. The task sheet lists them. Do each at its time: a step due before the order comes after the cart is final (do it again if the cart changes afterwards), and an approval counts only when the shopper's reply grants it.

Choosing:
- Cheapest: the order must hold the lowest-priced listing among ALL fitting listings. Find the whole fitting set, check candidates from the cheapest up, and use a price limit just below your pick to prove nothing cheaper fits.
- Most or least of a measure: compare every fitting listing on that measure; a listing that does not state it cannot win.
- Otherwise any listing that meets everything is right; do not hunt further.
- Quantity is units of the listing: "a pair of" shoes, socks or earrings is 1 unit, since one listing sells the pair.
- Several items: one cart line per item, each at its quantity. Cart each item as soon as you have its fitting listing (a carted listing can sell out, and an early cart leaves steps to replace it).

Changes: notes from the shopper or the store can arrive in any tool result, often as a refused checkout: a new budget, a requirement added or dropped (a listing that failed only that requirement may now fit, or be the cheapest), a price change, a listing sold out. Apply every change, fix the cart and check out again.
- A replacement for a sold-out listing is chosen exactly as carefully as the first pick, never grabbed from search rows: open its full record and check every requirement again, including each preference the shopper revealed; where the request asks for the cheapest, take the next cheapest listing that FITS (open the cheaper candidates and rule them out). Re-read every sentence of the refusal: it may also change the budget or drop a requirement, and that can change which listing is right.

Finishing:
- Put exactly the chosen lines in the cart (remove anything else), look at the cart, do any procedure due before the order, then check out with a call on its own.
- Declining is almost never right: in the last races 0 of 126 and 3 of 81 declines scored. The store is almost always built so that a fitting listing exists; when none seems to, you have missed something: a value spelled differently, a requirement met by a list element or the description, a seller redirect or pairing rule, a reference to another listing, a later page of results, a change in a note (new budget, dropped requirement), or a budget per unit rather than in total. Search for it. Running short of steps proves nothing: then check out the best cart you have. Follow the request's rule for declining; a decline remark must name or describe no product and offer no substitute.
- Keep about six steps for the finish. Think briefly in text, then call the tools."""

SHEET_PROMPT = """Read the shopper's request and the store's tools below and write a task sheet for the agent that will do the shopping. Return ONE JSON object, compact, and nothing else, with these keys:
"items": one entry per distinct product to buy: {"item": the kind of product in the request's words, "category": the category the request names for it or "", "quantity": units of that one listing (1 when not stated; "a pair of" shoes, socks, gloves or earrings is 1 unit, since one listing sells the pair), "requirements": ["property: wanted value, limit, exclusion or closed list" for every firm requirement on this product, named with the request's own word for the property], "fits_item": the number (1-based) of another item it must fit or work with, or null}. A requirement whose value comes from another listing stays one requirement that keeps the whole description of that listing. Never put the price, the budget, stock or quantity into requirements. The story around the request (who it is for, what went wrong before) is context, not a requirement, unless it states a property's value outright. One product named in one phrase with one quantity is one item.
"budget": the most the shopper pays as a number in the request's unit, or null; "budget_unit": that unit in words or null; "budget_scope": "total", "each" or null.
"in_stock_only": true unless the request accepts unavailable items.
"choose": "cheapest", "most", "least" or "any"; "measure": what is maximised or minimised for most/least, else "".
"messages": how many messages to the shopper the request allows (integer), or null; "steps": how many steps the request allows (integer), or null.
"references": the requirements (copied exactly) whose value must first be read from another listing, else [].
"procedures": every step the request asks the agent to take besides finding and buying the right items, such as getting approval, telling or showing the shopper something, checking something with the shopper, reading or checking something first, using a named tool, each {"step": what to do in at most 16 words, "quote": the request's sentence that asks for it, copied exactly, "when": "before_choosing", "before_order", "before_decline" or "any", "message": true when it needs a message to the shopper}; [] when there are none. Questions about the products' properties are not procedures, even where the request invites questions; nor are requirements on items, budgets, limits, rules on how to choose, or what to do when nothing is found ("if nothing fits, just say so" is the decline rule, not a procedure).
"decline_rule": the request's words about what to do when nothing fits, or "".
"hints": [{"item": item number, "property": a product property the story hints the shopper cares about without giving its value, "words": the request's words it comes from}], else [].
"tools": {"search": name, "details": the tool showing one listing's attributes, "offer": a tool showing a listing's current price or stock when different from details, "compare": a tool showing several listings in full, "ask": the tool messaging the shopper, "cart": the tool changing the cart, "cart_view": the tool showing the cart, "order": the tool that checks out, "decline": the tool that ends without buying}, each the exact tool name or null; "order_args": fixed arguments that make the order tool check out ({} when none); "decline_args": fixed arguments that make the decline tool end without buying ({} when none)."""

REVIEW_PROMPT = """You review a shopping agent's work right before it {action}. The finish is graded 1 only when it is exactly right, so look for anything that would make it wrong. You get the shopper's request, a task sheet a helper made from it, the shopper's answers, and the numbered record of every tool call so far with a shortened result.

Check, in this order:
1. Procedures: every procedure the request asks for that is due by now has been done, at the right time, as shown in the record. A step due before the order must come after the last change to the cart; an approval counts only when the reply grants it. A procedure for declining must be done before declining.
2. For a purchase: the latest cart in the record holds exactly one line for each requested item at the requested quantity and nothing else; each line's listing is of the requested kind (category) and its own record (opened in full, not only a search row) shows every requirement met, including the shopper's answers and every change in notes; it is in stock; the total is within the budget; where the request asks for the cheapest (or most/least), the record shows that no cheaper (better) fitting listing exists: a cheaper in-stock listing of the requested category marked "row only" in the list below may fit, so the order is not ready until it is opened and ruled out (name its id); every listing a requirement points at was opened.
3. For ending without a purchase: be strict, since declines are rarely right. The record must show a real search proving that no listing meets everything (requirements read from full records, not guessed), and the decline must follow the request's rule (no product named or suggested when the request forbids it). If a listing seen in the record may fit, the decline is not ready.
   After a refused checkout (a listing sold out, a budget or requirement changed), check the new cart line as hard as the first: its own full record must be in the record after the refusal or before it, it must meet every requirement and every shopper answer still in force, and for the cheapest it must be the cheapest listing that still fits.
4. Seller remarks: a cart line whose record carries a redirect to another listing ("for X, look to the ... not to this item", "for the pattern, refer to a different product ...") or a pairing rule must have that redirect resolved (the other listing opened) or the rule applied. Where the store shows a listing's remarks on a separate page or offer view and the list below names a cart line (or, for a decline, a near fit) whose page was never shown, the check is missing.
5. Questions: if messages to the shopper are still unused and steps allow, and an item's main properties were never asked about, say so: an unasked preference makes the order fail.
Only report a problem the record shows or a check the record lacks; requirements the shopper said they do not mind are free. Do not invent requirements.
Mind the steps left (given at the end): each check costs one step, and the cart change and the checkout need about three more. Ask only for the checks that fit, the most decisive first; when few steps are left, report only a problem that makes the order certainly wrong. An order that runs out of steps scores 0.

Return ONE JSON object and nothing else: {{"ready": true or false, "problems": ["each problem in one sentence"], "next": "the single most useful next action, naming the tool and the listing ids, or empty when ready"}}"""


# ----------------------------------------------------------------------------- inference
class Reader:
    """Model calls with fallback models per job, a per-call timeout and a deadline."""

    def __init__(self, deadline: float) -> None:
        provider = getenv("INFERENCE_PROVIDER", "openrouter")
        jobs = MODELS.get(provider, MODELS["openrouter"])
        self.models = {job: list(names) for job, names in jobs.items()}
        self.deadline = deadline
        self.calls = 0
        self.errors: list[str] = []
        self.misses: dict[str, int] = {}
        self.spent = {job: 0.0 for job in self.models}

    def pick(self, job: str) -> str:
        """The job's model, or the cheap one once the job has spent its share on dearer models or the task its all."""
        model = self.models[job][0]
        if sum(self.spent.values()) >= TASK_SPEND:
            return CHEAP
        if model.startswith(DEAR) and self.spent.get(job, 0.0) >= SPEND.get(job, 0.0):
            return CHEAP
        return model

    def count(self, job: str, model: str, result: Any) -> None:
        """Adds the call's cost (as billed, else estimated from its tokens) to the job's spending."""
        usage = (result or {}).get("usage") if isinstance(result, dict) else None
        if not isinstance(usage, dict):
            return
        cost = usage.get("cost")
        if not isinstance(cost, (int, float)) or isinstance(cost, bool):
            price = next((p for prefix, p in PRICES.items() if model.startswith(prefix)), OTHER_PRICE)
            try:
                cost = (float(usage.get("prompt_tokens") or 0) * price[0]
                        + float(usage.get("completion_tokens") or 0) * price[1]) / 1e6
            except (TypeError, ValueError):
                return
        self.spent[job] = self.spent.get(job, 0.0) + max(0.0, float(cost))

    def drop_current(self, job: str = "drive") -> None:
        if len(self.models[job]) > 1:
            self.models[job].pop(0)

    def reply(self, messages: list[dict], tools: list[dict] | None = None, force: bool = False, job: str = "drive",
              seconds: float = CALL_SECONDS, tokens: int = MAX_TOKENS) -> dict | None:
        """The assistant message, or None when no model answered in time."""
        for _ in range(len(self.models[job]) * 2):
            left = self.deadline - monotonic()
            if left < 2:
                return None
            _proxy.timeout = int(min(seconds, left))   # (a reply never arrives after the deadline)
            model = self.pick(job)
            sent = cached(messages) if model.startswith("anthropic/") else messages
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "required" if force else "auto"
            self.calls += 1
            try:
                result = _proxy.post("/inference/chat/completions", json_data=body)
                self.count(job, model, result)
                message = (result or {}).get("choices", [{}])[0].get("message")
            except Exception as exc:  # a transport or decoding failure counts as no answer
                self.errors.append(f"{model}: {type(exc).__name__}")
                message = None
            if isinstance(message, dict):
                self.misses[model] = 0
                return message
            self.errors.append(f"{model}: no message")
            self.misses[model] = self.misses.get(model, 0) + 1
            if self.misses[model] >= 2:     # (one miss can be a passing hiccup; two in a row replace the model)
                self.drop_current(job)
        return None

    def json(self, system: str, user: str, seconds: float) -> dict | None:
        """One JSON object from the thinking model, or None; all its tries together stay within the seconds."""
        kept = self.deadline
        self.deadline = min(kept, monotonic() + seconds)
        try:
            message = self.reply([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                 job="think", seconds=seconds, tokens=THINK_TOKENS)
        finally:
            self.deadline = kept
        return json_object((message or {}).get("content"))


def cached(messages: list[dict]) -> list[dict]:
    """The conversation with cache marks on the guide and on the latest user line, so each call reads the
    conversation so far from the cache and pays in full only for what is new."""
    marked = [dict(message) for message in messages]
    marks = [0] + [index for index, message in enumerate(marked) if message.get("role") == "user"][-1:]
    for index in dict.fromkeys(marks):
        content = marked[index].get("content")
        if isinstance(content, str) and content:
            marked[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    return marked


def json_object(text: Any) -> dict | None:
    """The first JSON object in a model's text (fenced or bare), or None."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    start = text.find("{")
    while start >= 0:
        depth, quoted, escaped = 0, False, False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                escaped, quoted = (False, quoted) if escaped else (char == "\\", char != '"')
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start:index + 1])
                    except ValueError:
                        break
                    return value if isinstance(value, dict) else None
        start = text.find("{", start + 1)
    return None


# ----------------------------------------------------------------------------- observations
def fields(line: str) -> list[tuple[str, str]] | None:
    """The "key: value" fields of a listing written as one line ("id: p-1 | title: ... | price: 3"), or None."""
    parts = line.split(" | ")
    if len(parts) < 3:
        return None
    pairs = []
    for part in parts:
        key, sep, value = part.partition(": ")
        if not sep or len(key) > 40:
            if not pairs:
                return None
            key, value = pairs.pop()
            pairs.append((key, f"{value} | {part}"))
            continue
        pairs.append((key.strip(), value))
    return pairs


def shorten_line(line: str, blurb: int) -> str:
    pairs = fields(line)
    if pairs is None:
        return line if len(line) <= FIELD_LIMIT else line[:FIELD_LIMIT] + "…"
    kept = []
    for key, value in pairs:
        limit = blurb if BLURB_FIELD.search(key) else FIELD_LIMIT
        kept.append(f"{key}: {value if len(value) <= limit else value[:limit] + '…'}")
    return " | ".join(kept)


def shorten(value: Any, blurb: int, key: str = "") -> Any:
    """The observation with long free texts cut, while every attribute, price, stock and id field stays whole."""
    if isinstance(value, str):
        if BLURB_FIELD.search(key) and len(value) > blurb and not fields(value):
            return value[:blurb] + "…"
        return shorten_line(value, blurb)
    if isinstance(value, list):
        return [shorten(item, blurb, key) for item in value]
    if isinstance(value, dict):
        return {name: shorten(item, blurb, str(name)) for name, item in value.items()}
    return value


def as_text(observation: Any) -> str:
    blurb = DETAIL_BLURB_LIMIT if len(rows(observation)) <= 1 else BLURB_LIMIT
    text = json.dumps(shorten(observation, blurb), ensure_ascii=False)
    if len(text) > OBSERVATION_LIMIT:
        text = text[:OBSERVATION_LIMIT] + "… (cut)"
    return text


def tool_name(tool: Any) -> str:
    """The name of a tool in OpenAI function format, or "" for an entry that is not one."""
    function = tool.get("function") if isinstance(tool, dict) else None
    name = function.get("name") if isinstance(function, dict) else None
    return name if isinstance(name, str) else ""


def tool_text(tool: dict) -> str:
    function = tool["function"]
    return f"{function.get('name') or ''} {function.get('description') or ''}"


def talks(tool: dict) -> bool:
    """Whether a tool sends a message to the shopper, judged from its name and description."""
    text = tool_text(tool)
    return bool(re.search(r"shopper|customer|buyer|client|question|message|\bask", text, re.I)) and not re.search(
        r"search|find|list|cart|basket|\bbag|stash|select|\bhold|held|order|checkout|decline|give up|finish|close",
        text, re.I)


def ends(tool: dict) -> bool:
    """Whether a tool finishes the task (an order, a checkout or a decline), judged from its name and description."""
    return bool(re.search(r"\bfinal\b|check ?out|purchase|place (?:the |an |your )?order|settle|commit|decline|give ?up"
                          r"|abstain|no purchase|close ?(?:out|request)|end(?:s)? the (?:task|session|chat|request)"
                          r"|task ?end|finish", tool_text(tool), re.I))


def allowance(request: str) -> int | None:
    """How many messages to the shopper the request allows, when it says so ("Total of 3 messages")."""
    count = r"\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten"
    match = re.search(rf"\b({count}|no)\b(?:(?!\b(?:{count}|no)\b)[^.?!]){{0,30}}?\b(?:messages?|questions?)\b",
                      request, re.I)
    if not match:
        return None
    word = match.group(1).lower()
    return int(word) if word.isdigit() else NUMBER_WORDS[word]


def number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        return float(match.group()) if match else None
    return None


def row_of(pairs: list[tuple[str, Any]]) -> dict | None:
    """A listing summary (id, price, title, category, stock) from its fields, or None when it has no id or price."""
    ident = next((str(v) for k, v in pairs if isinstance(v, str) and ID_FIELD.search(k) and LISTING_ID.search(v)), None)
    if ident is None:
        ident = next((str(v) for k, v in pairs if isinstance(v, str) and ID_FIELD.search(k) and v.strip()), None)
    price = next((number(v) for k, v in pairs if PRICE_FIELD.search(k) and number(v) is not None), None)
    if not ident or price is None:
        return None
    title = next((str(v) for k, v in pairs if isinstance(v, str) and re.search(r"title|heading|headline", k, re.I)),
                 next((str(v) for k, v in pairs if isinstance(v, str) and TITLE_FIELD.search(k)), ""))
    group = next((str(v) for k, v in pairs if isinstance(v, str) and GROUP_FIELD.search(k)), "")
    stock = next((v for k, v in pairs if STOCK_FIELD.search(k) and (isinstance(v, bool) or str(v) in ("True", "False"))),
                 None)
    in_stock = stock if isinstance(stock, bool) else (None if stock is None else stock == "True")
    return {"id": ident.strip(), "price": price, "title": title[:70], "category": group[:40], "in_stock": in_stock}


def rows(value: Any, depth: int = 0) -> list[dict]:
    """The listings inside an observation, whether written as objects or as "key: value | ..." lines."""
    if depth > 6:
        return []
    if isinstance(value, str):
        pairs = fields(value)
        row = row_of(pairs) if pairs else None
        return [row] if row else []
    if isinstance(value, list):
        return [row for item in value for row in rows(item, depth + 1)]
    if not isinstance(value, dict):
        return []
    found = []
    row = row_of(list(value.items()))
    if row:
        found.append(row)
    for item in value.values():
        if isinstance(item, (list, dict)) or (isinstance(item, str) and " | " in item):
            found += rows(item, depth + 1)
    return found


def notes_in(observation: Any) -> list[str]:
    """The notes riding on an observation: plain sentences beside the payload, under whatever key the world uses
    (lists of text that are not listing lines, or text under a key that names a note or a refusal)."""
    if not isinstance(observation, dict):
        return []
    found = []
    for key, value in observation.items():
        texts = value if isinstance(value, list) else [value]
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            continue
        if any(fields(text) for text in texts):
            continue                       # (listing lines, not notes)
        named = re.search(r"note|notice|alert|warning|refus|reason|event|update|change", str(key), re.I)
        if named or (isinstance(value, list) and any(len(text.split()) >= 4 for text in texts)):
            found += texts
    return found


def cart_tool(tool: dict) -> bool:
    function = tool["function"]
    if not ((function.get("parameters") or {}).get("properties")):
        return False                       # (a tool without arguments only shows the cart)
    return bool(re.search(r"(?<![a-z])(?:cart|basket|bag|add.?to)(?![a-z])|stash", f"{function.get('name')} "
                          f"{function.get('description') or ''}", re.I)) and not ends(tool)


def arguments(call: dict) -> dict | None:
    raw = call["function"]["arguments"] or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def record_text(text: str, found: list[dict]) -> str:
    """One observation as the review reads it: a listing's full record whole (its facts, remarks and redirects are
    what the review checks), a list of results as compact rows, anything else shortened."""
    if len(found) > 4:
        lines = [f"{r['id']} {r['price']:g} {'in stock' if r.get('in_stock') is not False else 'SOLD OUT'} "
                 f"{r.get('title', '')[:60]!r}" for r in found[:RECORD_ROWS]]
        more = f" (+{len(found) - RECORD_ROWS} more rows)" if len(found) > RECORD_ROWS else ""
        return f"{len(found)} rows: " + "; ".join(lines) + more
    if found or DETAILS.search(text):
        return brief(text, DETAIL_RECORD_LIMIT)
    return brief(text, RECORD_LIMIT)


def opened_ids(text: str, found: list[dict]) -> list[str]:
    """The listings whose full records an observation shows (one to four of them, with their attributes); a
    datasheet without a price is no result row, so its ids are read from the text."""
    if not DETAILS.search(text) or len(found) > 4:
        return []
    ids = [row["id"] for row in found] or LISTING_ID.findall(text)[:1]   # (later ids may be redirect targets)
    return ids if 0 < len(ids) <= 4 else []


def call_key(name: str, args: dict) -> str:
    return f"{name} {json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def brief(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + "…"


# ----------------------------------------------------------------------------- task sheet
NOTHING_FOUND = re.compile(r"nothing (?:fits|found|suits|works)|come up empty|no (?:match|luck)|can't find|cannot find"
                           r"|don't offer|do not offer|substitute|just say so|tell me so", re.I)
GO_AHEAD = re.compile(r"approv|go.?ahead|confirm|ok(?:ay)? (?:it|the)|sign.?off|check with me|run .* by me|let me know"
                      r"|tell me|show me|before (?:you )?(?:order|buy|check)", re.I)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def real_procedures(procedures: Any, request: str) -> list[dict]:
    """The procedures whose quote is a real sentence of the request (most of its words appear in one sentence),
    leaving out decline rules read as procedures, so a made-up step never blocks the finish or eats a message."""
    sentences = [words(s) for s in re.split(r"(?<=[.!?])\s+", request) if s.strip()]
    kept = []
    for proc in procedures if isinstance(procedures, list) else []:
        if not isinstance(proc, dict):
            continue
        quote = str(proc.get("quote") or proc.get("step") or "")
        said = words(quote)
        if not said or not any(len(said & sentence) >= 0.7 * len(said) for sentence in sentences):
            continue
        if NOTHING_FOUND.search(quote) and not GO_AHEAD.search(quote):
            continue
        kept.append(proc)
    return kept


def tools_summary(tools: list[dict]) -> str:
    lines = []
    for tool in tools:
        function = tool["function"]
        params = (function.get("parameters") or {}).get("properties") or {}
        args = ", ".join(f"{name}" + (f"={sorted(map(str, spec['enum']))[:8]}" if isinstance(spec, dict)
                                      and isinstance(spec.get("enum"), list) and spec["enum"] else "")
                         for name, spec in params.items())
        lines.append(f"- {function.get('name')}({args}): {brief(function.get('description') or '', 300)}")
    return "\n".join(lines)


def sheet_text(sheet: dict) -> str:
    """The task sheet as short lines for the driver and the review."""
    lines = []
    for index, item in enumerate(sheet.get("items") or [], 1):
        if not isinstance(item, dict):
            continue
        musts = "; ".join(str(m) for m in (item.get("requirements") or [])) or "none stated"
        extra = f", must fit item {item['fits_item']}" if item.get("fits_item") else ""
        shelf = f" [category: {item['category']}]" if item.get("category") else ""
        lines.append(f"Item {index}: {item.get('item')}{shelf} x{item.get('quantity') or 1}{extra}; requirements: {musts}")
    budget = sheet.get("budget")
    if budget is not None:
        lines.append(f"Budget: {budget} {sheet.get('budget_unit') or ''} ({sheet.get('budget_scope') or 'total'})")
    choose = sheet.get("choose") or "any"
    lines.append(f"Choose: {choose}{' by ' + str(sheet.get('measure')) if sheet.get('measure') else ''}"
                 f"; in stock only: {sheet.get('in_stock_only', True)}")
    if sheet.get("references"):
        lines.append("Values to read from another listing first: " + "; ".join(map(str, sheet["references"])))
    for proc in sheet.get("procedures") or []:
        if isinstance(proc, dict):
            lines.append(f"Procedure ({proc.get('when')}): {proc.get('step')}")
    if sheet.get("decline_rule"):
        lines.append(f"If nothing fits: {sheet['decline_rule']}")
    for hint in sheet.get("hints") or []:
        if isinstance(hint, dict):
            lines.append(f"Hinted preference, item {hint.get('item')}: {hint.get('property')} ({hint.get('words')})")
    tools = {job: name for job, name in (sheet.get("tools") or {}).items() if name}
    if tools:
        lines.append("Tools: " + ", ".join(f"{job}={name}" for job, name in tools.items()))
    for key in ("order_args", "decline_args"):
        if sheet.get(key):
            lines.append(f"{key}: {json.dumps(sheet[key])}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- session
class Session:
    """One task: the environment binding, the step count, the task sheet and the conversation with the model."""

    def __init__(self, problem_data: dict[str, Any]) -> None:
        environment = problem_data["environment"]
        self.binding = dict(environment["binding"])
        policy = environment["policy_view"]
        self.tools = [tool for tool in (policy.get("tools") or []) if tool_name(tool)]
        self.tool_names = {tool_name(tool) for tool in self.tools}
        self.message_tools = {tool_name(tool) for tool in self.tools if talks(tool)}
        self.finish_tools = {tool_name(tool) for tool in self.tools if ends(tool)}
        self.cart_tools = {tool_name(tool) for tool in self.tools if cart_tool(tool)} - self.message_tools
        self.carted = False
        self.cart_held = False
        self.request = str(policy.get("query") or "")
        self.opening = policy.get("opening")
        self.allowed = allowance(self.request)
        try:
            self.max_steps = int(policy.get("max_steps") or 30)
        except (TypeError, ValueError):
            self.max_steps = 30
        self.finished = bool(policy.get("finished"))
        self.started = monotonic()
        self.reader = Reader(self.started + TASK_SECONDS)
        self.step = 0
        self.messages_sent = 0
        self.replies: list[str] = []
        self.notes: list[str] = []
        self.record: list[str] = []
        self.reviews = 0
        self.decline_holds = 0
        self.late = False
        self.guard_hits = 0
        self.answers: dict[str, tuple[int, str]] = {}   # (information calls already answered: step and result)
        self.refused_at = 0
        self.repeats = 0
        self.one_at_a_time = False
        self.seen: dict[str, dict] = {}
        self.opened: set[str] = set()
        self.paged: set[str] = set()          # listings whose page or offer view (with seller remarks) was shown
        self.call_ids: set[str] = set()
        self.bad_calls: set[str] = set()
        self.dialogue: list[dict] = []
        self.sheet: dict | None = None
        self.sheet_lines = ""
        self.messages: list[dict] = []

    # -- setup
    def prepare(self) -> None:
        """The task sheet from the helper model, then the opening of the driver's conversation."""
        try:
            self.read_sheet()
        except Exception as exc:  # a sheet the code cannot use: the driver works from the request alone
            self.reader.errors.append(f"sheet: {type(exc).__name__}")
            self.sheet, self.sheet_lines = None, ""
        intro = f"Shopper's request:\n{self.request}"
        if self.opening is not None:
            intro += f"\n\nOpening observation:\n{as_text(self.opening)}"
        if self.sheet_lines:
            intro += f"\n\nTask sheet:\n{self.sheet_lines}"
        intro += f"\n\nThis session allows at most {self.max_steps} steps (tool calls). Begin."
        self.messages = [{"role": "system", "content": GUIDE}, {"role": "user", "content": intro}]

    def read_sheet(self) -> None:
        """The task sheet in a safe shape, and the tool roles it names (they win over the guesses from the
        tools' names and descriptions)."""
        sheet = self.reader.json(SHEET_PROMPT, f"REQUEST:\n{self.request}\n\nTOOLS:\n{tools_summary(self.tools)}",
                                 SHEET_SECONDS)
        if not isinstance(sheet, dict):
            return
        for key in ("items", "references", "procedures", "hints"):
            if not isinstance(sheet.get(key), list):
                sheet[key] = []
        sheet["items"] = [item for item in sheet["items"] if isinstance(item, dict)]
        for item in sheet["items"]:
            if not isinstance(item.get("requirements"), list):
                item["requirements"] = [item["requirements"]] if item.get("requirements") else []
        named = sheet.get("tools") if isinstance(sheet.get("tools"), dict) else {}
        sheet["tools"] = {job: name for job, name in named.items() if isinstance(name, str) and name in self.tool_names}
        for key in ("order_args", "decline_args"):
            if not isinstance(sheet.get(key), dict):
                sheet[key] = {}
        sheet["procedures"] = real_procedures(sheet["procedures"], self.request)
        self.sheet = sheet
        self.sheet_lines = sheet_text(sheet)
        roles = sheet["tools"]
        if roles.get("ask"):
            self.message_tools = {roles["ask"]}
        if roles.get("order") and roles.get("decline"):
            self.finish_tools = {roles["order"], roles["decline"]}
        if roles.get("cart"):
            self.cart_tools = {roles["cart"]}
        self.finish_tools -= self.message_tools
        self.cart_tools -= self.message_tools | self.finish_tools
        messages = sheet.get("messages")
        if self.allowed is None and isinstance(messages, int) and not isinstance(messages, bool):
            self.allowed = messages

    # -- environment
    def send(self, name: str, args: dict) -> dict | None:
        """The environment's answer to the next step, or None when it did not answer (the step number is then
        kept, so the next call resends that step and the environment replays it if it did run)."""
        envelope = {**self.binding, "step": self.step + 1, "action": {"name": name, "args": args}}
        for _ in range(STEP_TRIES):
            left = self.started + HARD_SECONDS - monotonic()
            if left < 1:
                return None
            _world.timeout = int(min(STEP_SECONDS, max(1.0, left)))
            try:
                result = _world.post("/environment/call", json_data=envelope)
            except Exception as exc:
                result = None
                self.reader.errors.append(f"environment: {type(exc).__name__}")
            if isinstance(result, dict) and "observation" in result:
                answered = result.get("step")     # (the environment's own count wins when it reports one)
                self.step = answered if isinstance(answered, int) and answered > self.step else self.step + 1
                self.finished = bool(result.get("finished"))
                return result
        return None

    # -- bookkeeping shown to the model
    def elapsed(self) -> float:
        return monotonic() - self.started

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def must_finish(self) -> bool:
        return self.steps_left() <= FINISH_RESERVE or self.elapsed() >= WRAP_UP_SECONDS

    def status(self) -> str:
        line = (f"[steps used {self.step} of {self.max_steps}; {self.steps_left()} left; "
                f"{int(self.elapsed())} s of about {int(TASK_SECONDS)} s used")
        if self.message_tools:
            line += f"; messages sent to the shopper {self.messages_sent}"
            if self.allowed is not None:
                line += f" of {self.allowed} allowed"
        line += "]"
        if self.replies:
            line += " The shopper's answers so far (each preference is a requirement): " + " | ".join(self.replies)
            if self.one_at_a_time:
                line += " (This shopper needs one property per message.)"
        if self.notes:
            line += " Notes received (apply each): " + " | ".join(self.notes[-4:])
        if self.cart_tools and not self.carted and len(self.opened) >= COMMIT_AFTER and not self.must_finish():
            line += (f" You have read {len(self.opened)} records and carted nothing: commit now to the best fitting "
                     "listing you have seen (a title or description can state a value no attribute names), cart it, "
                     "and keep the remaining steps for checking out and for changes.")
        if self.must_finish():
            line += (" Finish now: make the cart exactly right (the best fitting listing for each item), do any "
                     "procedure due before the order, and check out.")
        return line

    # -- the loop
    def run(self) -> list[dict]:
        self.prepare()
        idle = 0
        while not self.finished and self.step < self.max_steps:
            self.trim()
            job = "recover" if self.refused_at else "drive"     # (a refused checkout: the replacement matters most)
            message = self.reader.reply(self.messages, self.tools, force=idle > 0, job=job)
            if message is None:
                break
            calls = [call for call in (message.get("tool_calls") or []) if isinstance(call, dict)]
            if not calls:
                idle += 1
                self.messages += [{"role": "assistant", "content": message.get("content") or "(no tool call)"},
                                  {"role": "user", "content": "Reply with a tool call. " + self.status()}]
                if idle >= IDLE_LIMIT:
                    self.reader.drop_current(job)
                    idle = 0
                continue
            idle = 0
            self.act_safely(message, calls)
        if not self.finished and self.step < self.max_steps:
            self.finish_late()
        return self.dialogue

    def act_safely(self, message: dict, calls: list[dict]) -> None:
        """act(), with the conversation rolled back to a well-formed state if anything in it fails."""
        length = len(self.messages)
        try:
            self.act(message, calls)
        except Exception as exc:
            self.reader.errors.append(f"act: {type(exc).__name__}")
            del self.messages[length:]
            self.messages.append({"role": "user", "content": "An internal error interrupted that reply; nothing more "
                                  "was sent. " + self.status()})

    def trim(self) -> None:
        """Older observations cut short once the conversation grows long, so calls stay fast and within context
        (done in one go, so the cached prefix is rebuilt rarely)."""
        if sum(len(str(message.get("content") or "")) for message in self.messages) < TRIM_AT:
            return
        tools = [message for message in self.messages if message.get("role") == "tool"]
        for message in tools[:-KEEP_WHOLE]:
            content = message.get("content")
            if isinstance(content, str) and len(content) > TRIMMED:
                message["content"] = content[:TRIMMED] + "… (older result cut; read it again if needed)"

    def finish_late(self) -> None:
        """One last model call, after the time or the model ran out, so the task still ends with a finish."""
        self.reader.deadline = self.started + TASK_SECONDS + LATE_SECONDS
        self.reviews = REVIEWS                 # (this call is sent as it is: there is no time for a review)
        self.late = True                       # (nor for holding a decline back)
        late = ("Time is up. Make the single finishing call now: check out the cart if it holds anything; decline "
                "only if the cart is empty.")
        last = self.messages[-1] if self.messages else {}
        if last.get("role") == "user" and isinstance(last.get("content"), str):
            last["content"] += " " + late
        else:
            self.messages.append({"role": "user", "content": late})
        message = self.reader.reply(self.messages, self.tools, force=True, job="recover" if self.refused_at else "drive")
        calls = [call for call in ((message or {}).get("tool_calls") or []) if isinstance(call, dict)]
        if calls:
            self.act_safely(message, calls[:1])
        if not self.finished and self.step < self.max_steps:
            self.blind_checkout()

    def blind_checkout(self) -> None:
        """When no model answers at the end, the cart as it stands is checked out by the code itself (a cart that
        holds the chosen listings can still score; no finish never does)."""
        named = (self.sheet or {}).get("tools") or {}
        order = named.get("order") if isinstance(named, dict) else None
        if not isinstance(order, str) or order not in self.tool_names:
            guessed = [tool for tool in sorted(self.finish_tools)
                       if not re.search(r"declin|give|abstain|cancel|skip|reject|refus|no.?purchase", tool, re.I)]
            order = guessed[0] if len(guessed) == 1 else None
        if not self.carted or order is None:
            return
        args = dict((self.sheet or {}).get("order_args") or {})
        try:
            result = self.send(order, args)
            if result is not None:
                self.take(order, args, result)
        except Exception as exc:  # the dialogue so far still stands
            self.reader.errors.append(f"blind checkout: {type(exc).__name__}")

    def act(self, message: dict, calls: list[dict]) -> None:
        # (a batch never eats the steps kept for finishing; once finishing, the calls go one by one)
        room = 1 if self.must_finish() else min(BATCH_LIMIT, self.steps_left() - FINISH_RESERVE)
        kept = [self.clean_call(call, index) for index, call in enumerate(calls[: max(1, room)])]
        content = message.get("content")
        self.messages.append({"role": "assistant", "content": content if isinstance(content, str) else "",
                              "tool_calls": kept})
        held = False
        for call in kept:
            name = call["function"]["name"]
            args = None if call["id"] in self.bad_calls else arguments(call)
            if self.finished:
                text = json.dumps({"error": "the session has already finished"})
            elif held:
                text = json.dumps({"error": "not sent: an earlier call in this reply was held back"})
            elif args is None:
                text = json.dumps({"error": "the arguments were not a JSON object; nothing was sent"})
            elif name not in self.tool_names:
                text = json.dumps({"error": f"there is no tool named {name!r}; nothing was sent"})
            elif (self.guard_hits < GUARD_LIMIT and self.steps_left() <= CART_BY and not self.carted
                  and self.cart_tools and self.finish_tools and name not in self.cart_tools | self.finish_tools):
                self.guard_hits += 1
                text = json.dumps({"error": f"Not sent: only {self.steps_left()} steps are left and the cart is empty. "
                                            "Put the best fitting listing you have seen in the cart now, then check "
                                            "out."})
            elif (self.guard_hits < GUARD_LIMIT and self.steps_left() <= 1 and self.finish_tools
                  and name not in self.finish_tools):
                # (the last step is kept for the finish, so the task never ends unfinished)
                self.guard_hits += 1
                text = json.dumps({"error": "Not sent: only one step is left and it must be the finishing call. "
                                            "Check out the cart as it is now (decline only if it is empty)."})
            elif name in self.finish_tools and (hold := self.decline_hold(name, args) or self.review(name, args)):
                held = True
                text = json.dumps({"error": hold})
            elif name in self.cart_tools and (hold := self.ask_first()):
                held = True
                text = json.dumps({"error": hold})
            elif (earlier := self.repeat_of(name, args)) is not None:
                text = earlier
            elif (result := self.send(name, args)) is None:
                text = json.dumps({"error": "the environment did not answer; nothing changed, try again"})
            else:
                text = self.take_safely(name, args, result)
            self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})
        self.messages.append({"role": "user", "content": self.status()})

    def repeat_of(self, name: str, args: dict) -> str | None:
        """The earlier answer to the very same information call (same tool, same arguments) made since the last
        refused finish, returned without spending a step; None for anything that should go out."""
        if name in self.cart_tools | self.finish_tools | self.message_tools:
            return None
        earlier = self.answers.get(call_key(name, args))
        if earlier is None or earlier[0] <= self.refused_at:
            return None
        self.repeats += 1
        nudge = (" You keep repeating calls: decide now with what you have and move on to the cart and checkout."
                 if self.repeats >= 3 else "")
        return (f"Not sent again (no step used): you made this exact call at step {earlier[0]} and nothing has changed "
                f"since. Its result was: {earlier[1]}{nudge}")

    def take_safely(self, name: str, args: dict, result: dict) -> str:
        """take(), but a failure in the bookkeeping never hides an answered step from the model (the step is spent;
        hiding its answer would make the model send it again and again)."""
        try:
            return self.take(name, args, result)
        except Exception as exc:
            self.reader.errors.append(f"take: {type(exc).__name__}")
            try:
                return as_text(result.get("observation"))
            except Exception:
                return json.dumps({"observation": brief(result.get("observation"), OBSERVATION_LIMIT)})

    def take(self, name: str, args: dict, result: dict) -> str:
        """Book one answered step: the dialogue, the record, the listings seen, answers and notes."""
        observation = result.get("observation")
        self.dialogue.append({"step": self.step, "action": {"name": name, "args": args},
                              "observation": observation, "finished": self.finished})
        text = as_text(observation)
        found = rows(observation)
        self.record.append(f"{self.step}. {name}({brief(args, 300)}) -> {record_text(text, found)}")
        for row in found:
            self.seen[row["id"]] = {**self.seen.get(row["id"], {}), **{k: v for k, v in row.items() if v not in ("", None)}}
        self.opened.update(opened_ids(text, found))           # (full records, not search rows)
        failed = bool(FAILED.search(text[:300]))      # (an error leads its observation; a description may not)
        if name == self.page_tool() and not failed:
            asked = [v.strip() for v in args.values() if isinstance(v, str) and v.strip() in self.seen.keys() | self.opened]
            self.paged.update(asked or [row["id"] for row in found[:4]])
        if name in self.message_tools:
            self.messages_sent += 1
            lost = bool(NOT_UNDERSTOOD.search(text)) and not DECIDED.search(text)
            self.one_at_a_time = self.one_at_a_time or lost
            self.replies.append(f"to {brief(args, 120)}: {brief(text, REPLY_LIMIT)}"
                                + (" (not understood: this message is spent and says nothing about the properties it "
                                   "named; from now on ask exactly one property per message, plain form)" if lost
                                   else ""))
        outside = {k: v for k, v in result.items() if k not in ("observation", "step", "finished", "replayed")}
        for note in notes_in(observation) + notes_in(outside):
            note = brief(note, NOTE_LIMIT)
            if note not in self.notes:
                self.notes.append(note)
        if outside:
            text += " " + as_text(outside)
        if name in self.cart_tools and not failed:
            self.carted = True
        self.answers[call_key(name, args)] = (self.step, brief(text, REPEAT_LIMIT))
        if name in self.finish_tools and not self.finished:
            self.reviews = 0        # (a refused finish: the next attempt is reviewed again)
            self.refused_at = self.step   # (stock and prices may have changed: calls may be repeated)
            self.notes.append(f"finish refused: {brief(text, NOTE_LIMIT)} (choose any replacement as carefully as the "
                              "first pick: full record, every requirement and answer, the choice rule)")
        return text

    # -- the hold on the first cart call while messages are left
    def unused_messages(self) -> int:
        """Messages still free for questions: the allowance less those sent and one per procedure needing one."""
        if not self.message_tools or self.allowed is None:
            return 0
        kept = sum(1 for proc in (self.sheet or {}).get("procedures") or []
                   if isinstance(proc, dict) and proc.get("message"))
        return max(0, self.allowed - self.messages_sent - kept)

    def ask_first(self) -> str | None:
        """The first cart call is held once while questions are left unasked and the steps and time allow them."""
        unused = self.unused_messages()
        if self.cart_held or self.carted or unused <= 0 or self.elapsed() > 150:
            return None
        if self.steps_left() < FINISH_RESERVE + unused + 3:
            return None
        self.cart_held = True
        return (f"Not sent yet (no step used). You still have {unused} message(s) to the shopper unused. Orders placed "
                "without asking about the property the shopper cares about score 0, so ask first, "
                + ("one property of one item per message, in the plain form \"Which <property> would you like for "
                   "the <item>?\", the likeliest open property first."
                   if self.one_at_a_time else
                   "about one item per message with a short list of four to six of its open properties, likeliest "
                   "first, different from those already answered (one property per message if the shopper struggles "
                   "with lists).")
                + " Then re-check your choice against the answers and put it in the cart.")

    # -- the hold on declines
    def decline_hold(self, name: str, args: dict) -> str | None:
        """A decline is held back a few times with the usual reasons a fitting listing was missed; nearly every
        decline in past races scored 0."""
        if self.late or self.decline_holds >= DECLINE_HOLDS or not self.is_decline(name, args):
            return None
        if self.steps_left() <= 3 or self.elapsed() >= HARD_SECONDS - 25:     # (room left for cart and checkout)
            return None
        self.decline_holds += 1
        return ("Not sent yet (no step used). Declines almost never score: in the last races 0 of 126 and 3 of 81 "
                "were right, because a fitting listing nearly always exists. Before declining, check what was likely "
                "missed: a value spelled differently or met by one element of a list or by the description; a seller "
                "redirect or pairing rule, often shown only on a listing's page or offer view (open that view of the "
                "near fits); a listing a requirement or an answer points at; a later page of results or "
                "a wider search without a filter; a note that changed the budget or dropped a requirement; a budget "
                "per unit rather than in total. Find the best fitting listing, put it in the cart and check out. If "
                "steps run short, check out the closest fit you have rather than declining.")

    # -- the review before finishing
    def review(self, name: str, args: dict) -> str | None:
        """None to send the finishing call; otherwise the review's findings, returned to the model instead."""
        if (self.reviews >= REVIEWS or self.elapsed() >= REVIEW_UNTIL or not self.record
                or self.steps_left() <= REVIEW_STEPS):
            return None
        self.reviews += 1
        try:
            return self.findings(name, args)
        except Exception as exc:  # a review that fails lets the finish go out
            self.reader.errors.append(f"review: {type(exc).__name__}")
            return None

    def page_tool(self) -> str | None:
        """The tool showing a listing's page or offer, when the store keeps it apart from the attributes tool."""
        named = (self.sheet or {}).get("tools") or {}
        offer = named.get("offer") if isinstance(named, dict) else None
        details = named.get("details") if isinstance(named, dict) else None
        return offer if isinstance(offer, str) and offer and offer != details else None

    def unpaged_lines(self) -> str:
        """Opened listings whose page or offer view was never shown (their seller remarks are unseen)."""
        tool = self.page_tool()
        if not tool:
            return ""
        missing = sorted(self.opened - self.paged)[:12]
        return (f"\n\nLISTINGS WHOSE ATTRIBUTES WERE OPENED BUT WHOSE {tool} VIEW (seller remarks, pointers, stock) WAS "
                f"NEVER SHOWN:\n{', '.join(missing)}") if missing else ""

    def findings(self, name: str, args: dict) -> str | None:
        declining = self.is_decline(name, args)
        action = "ends the task without buying" if declining else "checks out (places the order)"
        cheap = sorted((r for r in self.seen.values() if r.get("in_stock") is not False), key=lambda r: r["price"])
        cheap_lines = "; ".join(f"{r['id']} {r['price']:g} {r.get('title', '')!r} [{r.get('category', '')}]"
                                f" ({'opened' if r['id'] in self.opened else 'row only, record never opened'})"
                                for r in cheap[:CHEAP_SHOWN])
        user = (f"REQUEST:\n{self.request}\n\nTASK SHEET:\n{self.sheet_lines or '(none)'}\n\n"
                f"SHOPPER'S ANSWERS:\n" + ("\n".join(self.replies) or "(none)") +
                "\n\nNOTES:\n" + ("\n".join(self.notes) or "(none)") +
                f"\n\nCHEAPEST IN-STOCK LISTINGS SEEN IN RESULTS (they may not fit):\n{cheap_lines or '(none)'}" +
                self.unpaged_lines() +
                "\n\nRECORD:\n" + "\n".join(self.record) +
                f"\n\nTHE AGENT NOW CALLS: {name}({json.dumps(args, ensure_ascii=False)})\n"
                f"Steps left: {self.steps_left()}; messages sent {self.messages_sent}"
                + (f" of {self.allowed} allowed" if self.allowed is not None else ""))
        verdict = self.reader.json(REVIEW_PROMPT.format(action=action), user, REVIEW_SECONDS)
        if not isinstance(verdict, dict) or verdict.get("ready") is not False:
            return None
        problems = verdict.get("problems")
        problems = [str(p) for p in problems if p][:5] if isinstance(problems, list) else []
        if not problems:
            return None
        advice = str(verdict.get("next") or "").strip()
        return ("Not sent yet (no step used). A review of the record before finishing found: " + " ".join(problems)
                + (f" Suggested next: {advice}" if advice else "")
                + " Fix what is right to fix, then make the finishing call again; if a finding is wrong, "
                  "make the same finishing call again.")

    def is_decline(self, name: str, args: dict) -> bool:
        sheet = self.sheet or {}
        named = sheet.get("tools") if isinstance(sheet.get("tools"), dict) else {}
        if named.get("decline") == name and named.get("order") != name:
            return True
        if named.get("order") == name and named.get("decline") != name:
            return False
        for key, declines in (("decline_args", True), ("order_args", False)):
            fixed = sheet.get(key)
            if isinstance(fixed, dict) and fixed and all(args.get(k) == v for k, v in fixed.items()):
                return declines
        # (only the tool name and choice values count, never a free-text remark that may say "nothing else")
        tool = next((t for t in self.tools if tool_name(t) == name), {})
        params = ((tool.get("function") or {}).get("parameters") or {}).get("properties") or {}
        choices = [k for k, spec in params.items() if isinstance(spec, dict) and isinstance(spec.get("enum"), list)]
        values = [args.get(k) for k in choices] if choices else [
            v for v in args.values() if isinstance(v, (str, bool)) and len(str(v).split()) == 1]
        short = [name] + [str(v) for v in values if isinstance(v, (str, bool)) and len(str(v)) <= 24]
        return any(re.search(r"declin|give ?up|abstain|no.?purchase|refus|cancel|skip|reject|nothing", text, re.I)
                   for text in short)

    def clean_call(self, call: dict, index: int) -> dict:
        """The model's tool call in a well-formed shape, with an id that is unique in the conversation."""
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = function.get("name") if isinstance(function.get("name"), str) else ""
        raw = function.get("arguments")
        if not isinstance(raw, str):
            try:
                raw = json.dumps(raw if raw is not None else {})
            except (TypeError, ValueError):
                raw = "{}"
        ident = call.get("id") if isinstance(call.get("id"), str) and call.get("id") else ""
        if not ident or ident in self.call_ids:
            ident = f"call_{len(self.call_ids)}_{index}"
        self.call_ids.add(ident)
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except ValueError:
            parsed = None
        if not isinstance(parsed, dict):
            self.bad_calls.add(ident)       # (answered with an error; the history keeps well-formed arguments)
            parsed = {}
        return {"id": ident, "type": "function", "function": {"name": name or "unnamed_call",
                                                             "arguments": json.dumps(parsed, ensure_ascii=False)}}


# ----------------------------------------------------------------------------- entry point
def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        session = Session(problem_data)
    except Exception as exc:  # a task the agent cannot read: nothing was sent
        return [{"error": f"{type(exc).__name__}: {exc}"[:300]}]
    try:
        return session.run()
    except Exception as exc:  # the dialogue so far is still returned; the environment keeps the steps taken
        session.dialogue.append({"error": f"{type(exc).__name__}: {exc}"[:300]})
        return session.dialogue


__all__ = ["agent_main"]
