from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

__all__ = ["agent_main"]


_MAIN_MODEL = "anthropic/claude-haiku-5.5"
_STRONG_MODEL = "anthropic/claude-sonnet-5.5"
_proxy = ProxyClient(timeout=120, max_retries=2)
_SHORT_CLIENTS: dict[int, Any] = {}


_TIME_LIMIT = 300.0
_T_RESEARCH = 130.0
_T_QUESTIONS = 170
_T_FIRST_QUESTION = 225
_T_REPLY_LOOKUP = 185
_T_REPLY_LOOKUP_DEADLINE = 195.0
_T_LATE_BROWSE = 200
_T_LATE_DEADLINE = 215.0
_T_DECIDE = 200
_T_LADDER = 220
_T_CHECK = 230
_T_DECLINE_HOLD = 235
_T_DECLINE_REVIEW = 245
_T_ORDER_MARGIN = 25


_SPEND_SOFT = 0.06
_SPEND_HARD = 0.15
_SPENT = [0.0]
_TASK_START = [time.monotonic()]


_ID_KEY = re.compile(r"^(id|ids|ref|.*_id|.*id_code|.*idcode|.*_ref|product_id.*|item_id.*|.*id_number)$", re.I)
_NEXT_KEY = re.compile(r"(next|follow|continue|cursor|page_token)", re.I)
_PRICE_KEY = re.compile(r"(price|cost|amount)", re.I)
_STOCK_KEY = re.compile(r"(stock|on_hand|avail|ready|orderable)", re.I)
_CAT_FIELD = re.compile(r"^(category|department|shelf|group|group_path|aisle|section)$", re.I)
_REC_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ]{0,40}:\s")
_REC_SPLIT = re.compile(r"\s\|\s(?=[A-Za-z_][A-Za-z0-9_ ]{0,40}:\s)")
_PAIR_SPLIT = re.compile(r";\s*(?=[^=;]{1,60}=)")


_QUESTION_FORM = re.compile(r"^Do you have a preference on [^?.!\n]{2,90}\?$")
_FINAL_VALUE_RE = re.compile(r"FINAL VALUE:\s*\**\s*([^\n]+)", re.I)
_SOLD_RE = re.compile(r"sold out|out of stock|no longer (?:available|in stock|sold|on sale)|unavailable|not available", re.I)
_MESSAGE_LIMIT_RE = re.compile(r"(\d+)\s+(?:more\s+)?(?:messages?|questions?)\b", re.I)


_MODULES: list[dict[str, Any]] = [

    {"name": "spread_items", "flags": {"multi_item"}, "stage": "planner",
     "text": "Several items: every item needs its own questions. A question without an item reaches only the items "
             "whose listings have that property, so before a second property of any item, ask each OTHER item about "
             "a property its own listings have (name that item); the item with the fewest stated requirements comes "
             "first."},
    {"name": "facts_view_keys", "flags": {"page_and_facts"}, "stage": "browse",
     "text": "This store has separate page and facts views: read the FACTS view of candidates for their properties and "
             "KEYS; brand, category, fine print, notes and remarks are not datasheet keys."},
    {"name": "value_only_search", "flags": {"search_modes"}, "stage": "buy",
     "text": "Search also with the requirement values alone (e.g. two value words without the item's kind word) in "
             "text mode, and read the next page of those results too."},
    {"name": "value_only_first_query", "flags": {"search_modes"}, "stage": "summary",
     "text": "Make the FIRST query just the two most distinctive requirement values (no item kind word), the "
             "shopper's revealed value first, e.g. 'X Y'."},
    {"name": "category_not_title", "flags": set(), "stage": "buy",
     "text": "A row's kind is its category field, never its title words: read every cheaper row of the right category "
             "even when its title names something else."},
    {"name": "same_unit_measures", "flags": {"measure"}, "stage": "buy",
     "text": "Measures: a value in another unit counts after converting it to the requirement's unit; among listings "
             "that fit, prefer one that states the value in the requirement's own unit; a value that lists several "
             "options meets the requirement when one of those options does."},
    {"name": "lookup_start_page", "flags": {"lookup"}, "stage": "browse",
     "text": "Lookups: the referenced listing is the seller's row of that kind whose details (facts or page notes) show "
             "the identifying property values named for it; the same value under a related key counts (e.g. 'blade "
             "count: 3' for 'number of blades 3'). Read the seller's rows of that kind until one shows them all; if none does, "
             "take the one that shows the most. Follow only that page's pointer."},
    {"name": "lookup_other_kind", "flags": {"lookup_other"}, "stage": "planner",
     "text": "A looked-up listing of another kind than the item to buy only supplies a value: the properties that "
             "identify it are not properties of the item to buy, so never ask them for that item."},
    {"name": "lookup_identifiers", "flags": {"lookup_same"}, "stage": "planner",
     "text": "Lookups: never ask the property a lookup reads (its value comes from the pages). The properties that "
             "identify a looked-up listing in the checklist (its seller aside) are not requirements of the item to "
             "buy, which makes them likely hidden requirements of it: ask them early, one per message, at most two of "
             "them per item and none once that item has revealed a requirement; then go on to the other items."},
]


_PROMPT_NOTE_DROP = ("A shopper sent this note while their order was being prepared: \"{note}\". Does the note let go of (drop) "
              "one of their requirements, so that it no longer applies? Reply with JSON only: {{\"drops\": true or false, "
              "\"property\": \"<the property dropped, or null>\"}}.")

_NOTE_MULTI_ID_FAIL = ("That call with {n} ids returned no listing: this store's tool may take fewer ids at once. Call it again "
               "with at most 2 ids per call (or read them one by one); do not repeat the {n}-id call.")

_NOTE_ID_CAP = "Not run (no step used): this tool takes at most 2 ids per call here; call it with 2 ids at a time."

_PROMPT_DECLINE_REVIEW = (
    "A shopping assistant wants to decline the request below because it believes no listing fits. Check the listings it "
    "has seen. Judge every stated requirement on the listing's own details (fact tables first; units convert; a value "
    "that lists several options counts when one of them matches; a key with a related name holding the same value "
    "counts; a missing key is not a contradiction when the title or page states the value). For each item, find the "
    "cheapest in-stock listing of the item's kind that meets every stated, looked-up and revealed requirement within the "
    "budget. Reply with JSON only: {{\"buy\": [{{\"id\": \"<listing id>\", \"why\": \"<short reason>\"}}, ...]}} "
    "with one listing per item when every item has one, or {{\"decline\": true, \"why\": \"<the item that has "
    "none>\"}}.\n\nREQUEST AND FINDINGS:\n{context}\n\nLISTINGS SEEN (id: details):\n{listings}")

_PROMPT_CHAIN_CLEANING = (
    "Below are a shopping request, the reports of looking up listings it (or a shopper reply) refers to, and a draft "
    "list of the requirements of the items to buy. Correct the draft: (1) remove every value that only identifies a "
    "listing along a reference chain -- the seller or brand, and the properties named to pick out a listing ('<seller>'s "
    "<kind> with <property> <value>'), for the first listing and for every listing the chain passes through; (2) for a "
    "requirement that comes from a chain, keep only the chain's final value, and when the last page says which value "
    "goes together with its own, use that other value; (3) keep every requirement the request states for the item to "
    "buy and every value the shopper's replies require. Reply with JSON only: {{\"requirements\": \"item: property "
    "value; item: property value\"}}.\n\n{context}\n\nDRAFT REQUIREMENTS: {draft}")

_PROMPT_TOOL_PROTOCOL = (
    "Below are the tools of an online store: name, description and parameters (type, description, allowed values). "
    "Read how they are used and reply with JSON only:\n"
    "{\"roles\": {\"<tool name>\": \"<role>\"},\n"
    " \"ending\": {\"<tool whose role is ending>\": {\"param\": \"<the parameter that chooses>\", \"order\": "
    "\"<its value that places the order>\", \"decline\": \"<its value that turns the request down>\"}},\n"
    " \"cart\": {\"<tool whose role is cart>\": {\"param\": \"<the parameter that says whether units go in or out, "
    "or null>\", \"remove\": [\"<its values that take units out>\"]}},\n"
    " \"price_filter\": {\"tool\": \"<a tool that lists the listings of one category under a price cap, or null>\", "
    "\"cap\": \"<its price-cap parameter>\", \"shelf\": \"<its category parameter>\", \"mode_param\": \"<the parameter "
    "that switches it to listing a category, or null>\", \"mode_value\": \"<the value for that, or null>\"},\n"
    " \"search_modes\": <true when a search tool has a parameter that switches between a word search and listing a "
    "category, else false>,\n"
    " \"details\": [\"<every tool that shows the details of one listing>\"]}\n"
    "Roles: 'message' (sends a message to the shopper), 'cart' (puts listings in the basket, or changes or removes "
    "basket lines), 'order' (places the order for the basket and ends the session), 'decline' (turns the request "
    "down without buying and ends the session), 'ending' (one tool that ends the session either way, the choice "
    "made by an argument), 'other' (searching, reading listings, showing the basket, balances and anything else). "
    "Use the tool and parameter names and the allowed values exactly as written.\n\nTOOLS:\n")

_PROMPT_CHECKLIST = (
    "Turn the shopper's request below into a checklist for a shopping assistant, as JSON only:\n"
    "{\"items\": [{\"name\": \"<what to buy, in the request's words>\", \"quantity\": <n>,\n"
    "  \"must\": [\"<property>: <value>\", ...],\n"
    "  \"lookups\": [{\"listing\": \"<the other listing: seller or brand, kind of item, identifying properties>\",\n"
    "               \"read\": \"<property to read there>\", \"relation\": \"<how the item to buy must relate to it>\"}],\n"
    "  \"hints\": [\"<property of this item the story may hint at>\"]}],\n"
    " \"budget\": \"<amount, and whether per item or for everything>\", \"exclude\": \"<e.g. sold out>\",\n"
    " \"messages\": <message limit or null>, \"steps\": <step limit or null>, \"other\": \"<any other rule>\",\n"
    " \"confirm\": <true only when the request explicitly asks to be told, listed or confirmed what will arrive, the basket "
    "or the total, or asks a direct question about them (e.g. 'what will the order contain?'); wishes about planning, "
    "storage, timing or use are hints, not such a request: then false>}\n"
    "Rules: item names in a request are usually the store's category names and can contain '&' or commas (e.g. "
    "'pots & pans', 'cups, mugs & saucers'): keep such a name whole as ONE item, and a quantity before "
    "it ('2 pairs of X') belongs to that whole item. 'must' holds only requirements the listing to buy must itself meet, with the request's exact values; a rule "
    "that the item must fit or be compatible with something the request names directly (e.g. 'it must work with the "
    "device I own, model X') is a 'must' entry ('compatible with: X'), not a lookup, and a conditional rule ('X unless Y') is a "
    "'must' entry with its condition. A "
    "lookup is a value that has to be read from another listing; the properties that identify that other listing "
    "describe it only and are NOT requirements of the item to buy. That listing may say the value is on a further "
    "listing; do not assume that 'compatible' or 'goes with' means 'equal': the pages say which value goes with which. "
    "Hints: details of the story (use, place, people, feel, timing) that may point to a hidden requirement.")

_PROMPT_REPLY_LOOKUPS = (
    "Some shopper replies below may say that a requirement's value is to be read from another listing (the reply "
    "names that listing, by its seller, kind or properties, instead of giving the value). List each such reference as "
    "JSON only: "
    "{\"lookups\": [{\"item\": \"<item the requirement is for>\", \"listing\": \"<the other listing: seller or brand, "
    "kind, identifying properties>\", \"read\": \"<property to read>\", \"relation\": \"<how the item must relate>\"}]} "
    "(an empty list when there is none).")

_PROMPT_LOOKUP_IDENTIFIERS = ("Here is how a shopper describes another listing: \"{listing}\". List the property names it uses to "
             "identify that listing (e.g. the property name in '<property> <value>'), leaving out the seller, the brand, the kind "
             "of item and pointers to other pages. Reply with JSON only: {{\"properties\": [\"<property>\", ...]}}.")

_PROMPT_STORY_HINTS = (
    "Hidden requirements are listing properties the shopper did not state; the story often hints at one indirectly "
    "(a remark about where the item goes, who uses it, how it is used or what it must go with can point to one key of "
    "the item's listings). For each item, name at most ONE property "
    "that a phrase of the story clearly hints at: it must be a key of that item's listings in the research KEYS above, "
    "not a property the request states, and you must quote the exact words of the request that hint at it. Give none "
    "for an item when no phrase clearly points to one property. Reply with JSON only: {\"hints\": [{\"item\": "
    "\"<item name as in the checklist>\", \"property\": \"<key>\", \"phrase\": \"<exact words>\"}]}")

_PROMPT_ASKS_QUESTION = ("Does the shopper's request below ask a direct question that needs an answer about the purchase, the "
              "basket or the total (for example whether the plan works, or what will arrive)? Polite requests phrased as questions "
              "('Could you ...?', 'Can you ...?') do not count. Reply with JSON only: {\"answer\": \"yes\" or \"no\"}.")

_PROMPT_STATED_QUANTITY = ("How many units of '{name}' does the request below ask for? Use only a number the request states for this "
             "item ('2 x', 'two pairs', 'a pair'); who will use it or how many people there are does not count; if no "
             "number is stated, the answer is 1. Reply with JSON only: {{\"quantity\": <n>}}.")

_PROMPT_BROWSE_SYSTEM = (
    "You research an online store for a shopping assistant, using only the supplied tools. Search with short queries "
    "of 2-3 words (the store reads only the first words), e.g. a seller or brand name or the item's kind word plus "
    "one distinctive value. Search results are rows without properties: read the details of the promising rows, "
    "cheapest of the right kind first (rows show prices; a cheaper row is not skipped for its title), "
    "several side by side in one call when the store has a tool for that. Never read a listing twice. A result page "
    "may give a cursor for the next page. When the store has separate tools for a listing's page and for its facts, "
    "notes and pointers to other listings are on the page view, and requirements are judged on the facts view: a value "
    "missing from a page, title or description is no evidence, so read the facts view before excluding a listing. Judge properties on the details (facts), not on titles. A fit or "
    "compatibility list that names a model family or series (e.g. 'X 35xx') covers the models in it (X 3520). A requirement is met "
    "only when the details state that value itself; a related, stronger or weaker value is a different value (e.g. "
    "'X-free' does not meet 'low X'), and a value that inserts words inside the phrase ('X Y Z' for 'X Z') does not "
    "state it. A listing's notes "
    "(annotations) are store information too: when a page says its value goes together with another value, the item "
    "to buy needs that other value. When your task is "
    "done, or when you are told to stop, reply in plain text without a tool call: that text is your report.")

_PROMPT_QUESTION_PLANNER = (
    "You choose the next question to a shopper, to uncover hidden requirements on listing properties. Each question "
    "asks about ONE property. A question that names no item is matched against every item, so in a request with "
    "several items leave the item out (null) unless that property was already asked without an item and you want to "
    "check it for another item; in a request with one item, name it. A reply that gives a value is a requirement; a reply with no preference, or "
    "one saying the shopper did not understand, means that property is not a requirement: never ask it again. "
    "A reply asking you to ask one thing at a time, to a question about a single property, means that property IS a "
    "requirement for one or more items: ask it next naming each item whose listings have it, one per message, and "
    "keep going through ALL those items even after one of them revealed a requirement. Choose "
    "in this order: (1) properties the story hints at; (1b) a property the request states for one item, asked for "
    "each OTHER item whose listings have it; (2) for each item still without a requirement, the property a buyer of "
    "that kind of item usually decides on first, even when the candidates seen so far all share one value; (3) a key "
    "whose values differ among the candidates in the research reports, the key with the most different values first "
    "(without an item when several items' listings have it); (4) other properties of the items still without a "
    "requirement. Avoid condition, warranty, options, and properties the request or a lookup already "
    "fixes. Ask only properties whose keys the research reports (KEYS) or the request show; when a story hint names "
    "several possible properties (e.g. 'X or Y'), ask the one the store's listings have; never ask a bare "
    "'type'. Use each property key exactly as the store spells it (underscores may become spaces), never a paraphrase "
    "(never a shortened or reworded name), as "
    "the store shows them. Reply with JSON only: {\"property\": \"<property key>\", \"item\": <item name as in "
    "the checklist, or null>} or {\"done\": true} when nothing useful is left.")

_PROMPT_REPLY_KIND = (
    "A shopping assistant asked a shopper the question below. Classify the shopper's reply as one of: 'requirement' "
    "(it states a value the item must have, or that it must be the same as another listing), 'no_preference', "
    "'not_understood' (it asks to repeat, simplify or explain), 'one_at_a_time' (it asks for the questions to come one "
    "at a time). Also name the item the reply is about, using exactly one of these names, "
    "or null: {names}. For a 'requirement', also give the property as the reply names it (not the question's word), the relation ('must be', 'at most', 'at least', "
    "'not', or 'same as' when the value is to be read from another listing) and the value, read carefully: a reply that "
    "accepts only one value V means the property must be V; a reply that accepts every value except V means it must "
    "NOT be V (relation 'not'). "
    "Reply with JSON only: {{\"kind\": \"...\", \"item\": ..., \"property\": ..., \"relation\": ..., \"value\": ...}}")

_PROMPT_REQUIREMENTS_SUMMARY = ("Write one store search query per item (up to 4) that finds listings meeting its requirements. Each query has "
               "exactly 3 words: the word that names the item's kind (e.g. 'kettles', 'lamps', 'jackets') as ONE word, "
               "plus the most distinctive single word of TWO requirement values together (always include the newest "
               "revealed requirement; a brand or dealer name the requirement names counts as a value). Keep the word that names a "
               "value itself; leave out words that only qualify it (degree, negation, comparison), property names, and "
               "conditions on price, budget, stock or delivery.")

_PROMPT_BUY_SYSTEM = (
    "You are a shopping assistant in an online store. Buy what the shopper asks for: every stated requirement, every "
    "lookup result and every requirement the shopper's replies revealed must hold, and among the listings that fit "
    "choose the best one (the cheapest in-stock listing within the budget unless the shopper says otherwise). Decline "
    "only when no listing can fit.\n"
    "The order is all or nothing: never order only part of the requested items. If some item has no listing that "
    "meets its STATED requirements (the request's own rules and the budget) after a real search, empty the basket and "
    "decline the whole request. If the stated requirements are met and only a lookup-derived or revealed value cannot "
    "be confirmed, buy the best fitting listing instead of declining.\n"
    "A requirement is met only when the listing's details state that value itself; a related, stronger or weaker value "
    "is a different value. Prefer a listing whose value is exactly the required phrase; a value that inserts words "
    "inside the phrase ('X Y Z' for 'X Z') does not state it (a value holding the phrase whole, e.g. in a list, does). Search results show titles only, and a title that does not mention a required value says "
    "nothing about it: after each search, read the details of its in-stock, in-budget rows of the right kind, the "
    "cheapest first (the best fit is the cheapest one that fits; never skip a cheaper row because its title does not "
    "mention a requirement), side by side when the store has a tool for that, before you judge them or search again. A listing's notes (annotations) are store information: when a page says its value goes "
    "together with another value, the item to buy needs that other value. Buy every item: if no listing meets every "
    "requirement of an item after reading every right-kind, in-budget row your searches showed and at least three "
    "searches aimed at the missing requirement (for a 'not X' rule, "
    "search the item's kind word with other common values of that property), take the one that meets the most rather "
    "than leaving it out; this applies only to values derived from lookups: never buy a listing that lacks a value "
    "the shopper's replies revealed. Before "
    "adding a listing, check its details against every FINAL REQUIREMENT yourself (the research reports' verdicts may "
    "be wrong). The store may also check a requirement on OTHER basket lines that have the same property key (e.g. if "
    "one item must have value X under key K, another item's listing with key K of another value can break it; a "
    "different key such as 'item-K' does not). So when you can choose, take for the other items a listing "
    "that lacks that key or has the same value there; but never decline or leave an item out for this reason. Among "
    "listings that fit equally, prefer the one with fewer of those keys, then the one with fewer properties. A listing of another category never counts, even when it states the "
    "values; if no right-kind listing fits after a real search, decline. The best fit is the cheapest one, so for every item "
    "list the item's shelf with the store's filter, naming the category exactly as listings show it (e.g. "
    "'Desks', not 'desk'), with a LOW price cap first, about a quarter of the budget, so that only a few rows remain "
    "(the filter shows a few rows per page and does not sort by price); if none fits, double the cap; read the "
    "candidates before choosing (text search alone misses cheap fits). Where the store has separate page and facts "
    "views, judge requirements on the facts view: a value missing from a page, title or description is no evidence, so "
    "never exclude a listing (or decline) before reading its facts view. A listing's "
    "category must be the requested kind of item. When a listing you "
    "choose has a fine-print note (annotation), search its title words once to see same-title twins from other sellers, "
    "and prefer an equal-price twin with no note that shows each required value under every spelling of that key the "
    "twins use (e.g. both 'x y' and 'x_y'); this spelling preference is only between same-title equal-price twins: "
    "between different listings one spelling of the key is enough, and the cheaper fit wins. When the store has separate page and facts views, open BOTH views of "
    "every listing you order, in this episode, before finishing (the store checks that). Before finishing, if steps allow, read the carted listings once more "
    "to check they are still in stock. Search with short queries (2-3 words; try the suggested searches first); rows do not show "
    "properties, so read the promising rows' details, several side by side when the store has a tool for that; never "
    "read a listing twice; when a page shows the right kind but no fit, read its next page. Then put the chosen "
    "listings in the basket with the requested quantities, each item's whole quantity on ONE listing line (do not "
    "split it over several listings), and finish with the order. If no listing within the budget states a required "
    "value in its details, a listing whose title or page states it counts (prefer one whose other required values are "
    "in its details). If the store refuses, read the "
    "notice, fix the basket and try again: after a refused finish or a notice, re-open every carted listing's details "
    "(stock and price), replace any line that is sold out with the next cheapest fit (look at the cheapest rows of your result pages "
    "again rather than choosing from memory), then finish. Any tool result can carry notices from the shopper or the store (a new "
    "budget, an item sold out or repriced, a requirement dropped): read them every time and update the requirements "
    "(a new budget replaces the old one) before you finish.\n"
    "Every tool call has a 'notes' argument: write there 'REQUIREMENTS: <item>: <value>; ...' with every requirement "
    "value, then 'NOTES: ...' with the listings read, whether each fits, and what you will do next.\n"
    "Every reply must be exactly one call to one of the supplied tools; each tool call is one environment step. Never "
    "answer in plain text. Continue until a result reports finished=true.")

_PROMPT_TWIN_QUERIES = (
    "Below are a shopping assistant's notes and the latest store results. For every listing that is in the basket "
    "(or about to be ordered) and whose details have a fine-print note (an annotation or remark), write one search "
    "query of 3-4 distinctive product words taken from its title (skip promotional words), to find same-title twins "
    "from other sellers. Reply with JSON only: "
    "{\"queries\": [\"...\"]} (an empty list when no basket listing has a note).")


class Episode:


    def __init__(self, problem_data: dict[str, Any]):
        environment = problem_data["environment"]
        self.binding = environment["binding"]
        self.policy = environment["policy_view"]
        self.max_steps = int(self.policy["max_steps"])
        self.step = 0
        self.finished = bool(self.policy.get("finished"))
        self.dialogue: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.id_cap: dict[str, int] = {}
        self.seen: dict[str, dict[str, Any]] = {}
        self.sold: set[str] = set()
        self.named: set[str] = set()
        self.read_only: set[str] = set()
        self.changes = 0
        self.memo: dict[str, tuple[Any, int, int, int]] = {}
        self.repeats = 0

    def left(self) -> int:
        return self.max_steps - self.step

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def call(self, name: str, args: dict[str, Any], tool_call: dict[str, Any] | None = None, content: str = "") -> Any:

        try:
            self.named |= _ids_in_args(args)
        except Exception:
            pass
        key = json.dumps([name, args], sort_keys=True, default=str) if name in self.read_only and args else ""
        if key in self.memo:
            result, step, changes, answered = self.memo[key]
            if changes == self.changes and answered < 2:
                self.memo[key] = (result, step, changes, answered + 1)
                self.repeats += 1
                return {"note": f"The same call as step {step}, and nothing has changed since: its result again "
                                "(no step used).", "result": result}
        self.step += 1
        payload = {**self.binding, "step": self.step, "action": {"name": name, "args": args}}
        result = _proxy.post("/environment/call", json_data=payload)
        if result is None:
            raise RuntimeError("environment call failed")
        self.finished = bool(result["finished"])
        call = tool_call or {"id": f"code-{self.step}", "type": "function",
                             "function": {"name": name, "arguments": json.dumps(args)}}
        self.dialogue.append({"role": "assistant", "content": content, "tool_call": call, "environment_result": result})
        try:
            observation = _records_from_text(result["observation"])
        except Exception:
            observation = result["observation"]
        try:
            self._note_listings(observation)
        except Exception:
            pass
        noticed = bool(_notice_texts(observation))
        if name not in self.read_only or noticed:
            self.changes += 1
        if key and not noticed:
            self.memo[key] = (observation, self.step, self.changes, 0)
        return observation

    def _note_listings(self, observation: Any) -> None:
        fresh: dict[str, dict[str, Any]] = {}
        _remember_listings(observation, fresh)
        for ident, row in fresh.items():
            before = self.seen.get(ident) or {}
            turned_off = any(isinstance(value, bool) and not value and before.get(key) is True
                             and _STOCK_KEY.search(str(key)) for key, value in row.items())
            if turned_off:
                self.sold.add(ident)
        _remember_listings(observation, self.seen)


def _typed_value(text: str) -> Any:

    value = text.strip()
    words = {"None": None, "null": None, "True": True, "true": True, "False": False, "false": False}
    if value in words:
        return words[value]
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _fact_pairs(text: str) -> dict[str, Any] | None:

    parts = _PAIR_SPLIT.split(text.strip())
    if len(parts) < 2 or any("=" not in part for part in parts):
        return None
    table: dict[str, Any] = {}
    for part in parts:
        key, _, value = part.partition("=")
        if not key.strip() or len(key) > 60:
            return None
        table[key.strip()] = _typed_value(value)
    return table


def _record_from_line(text: Any) -> dict[str, Any] | None:

    if not isinstance(text, str) or " | " not in text or not _REC_KEY.match(text):
        return None
    fields = _REC_SPLIT.split(text)
    if len(fields) < 3:
        return None
    record: dict[str, Any] = {}
    for field in fields:
        key, colon, value = field.partition(":")
        if not colon or not key.strip():
            return None
        value = value.strip()
        table = _fact_pairs(value) if "=" in value else None
        record[key.strip()] = table if table is not None else _typed_value(value)
    return record


def _records_from_text(node: Any, depth: int = 0) -> Any:

    if depth > 6:
        return node
    if isinstance(node, str):
        record = _record_from_line(node)
        return node if record is None else record
    if isinstance(node, list):
        return [_records_from_text(item, depth + 1) for item in node]
    if isinstance(node, dict):
        return {key: _records_from_text(value, depth + 1) for key, value in node.items()}
    return node


def _remember_listings(node: Any, memory: dict[str, dict[str, Any]]) -> None:

    if isinstance(node, dict):
        ident = next((value for key, value in node.items() if isinstance(value, str) and _ID_KEY.match(key)), None)
        if ident:
            record = memory.setdefault(ident, {})
            for key, value in node.items():
                if value is None or isinstance(value, (str, int, float, bool)):
                    if not (isinstance(value, str) and len(value) > 300):
                        record[key] = value
                elif isinstance(value, dict) and value and all(
                        inner is None or isinstance(inner, (str, int, float, bool)) for inner in value.values()):
                    record[key] = dict(list(value.items())[:40])
        for value in node.values():
            if isinstance(value, (dict, list)):
                _remember_listings(value, memory)
    elif isinstance(node, list):
        for value in node:
            _remember_listings(value, memory)


def _row_ids(node: Any) -> list[str]:

    ids: list[str] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            if x and all(isinstance(row, dict) for row in x):
                for row in x:
                    for key, value in row.items():
                        if isinstance(value, str) and _ID_KEY.match(key) and value not in ids:
                            ids.append(value)
                            break
            for value in x:
                walk(value)

    walk(node)
    return ids


def _small_page(observation: Any) -> tuple[str, Any, int] | None:

    if not isinstance(observation, dict):
        return None
    for level in [observation] + [value for value in observation.values() if isinstance(value, dict)]:
        lists = [value for value in level.values() if isinstance(value, list) and value
                 and all(isinstance(row, dict) for row in value)]
        nexts = [(key, value) for key, value in level.items()
                 if _NEXT_KEY.search(key) and value not in (None, False, "", 0)]
        if lists and nexts and len(lists[0]) <= 12:
            return nexts[0][0], nexts[0][1], len(lists[0])
    return None


def _ids_in_args(args: dict[str, Any]) -> set[str]:

    found: set[str] = set()
    for value in args.values():
        for item in (value if isinstance(value, list) else [value]):
            if isinstance(item, str):
                found.add(item)
    return found


def _clip(value: Any, limit: int = 600) -> Any:

    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, list):
        return [_clip(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: _clip(item, limit) for key, item in value.items()}
    return value


def _tool_text(observation: Any) -> str:


    def trim(x: Any, in_rows: bool) -> Any:
        if isinstance(x, str):
            return x if not in_rows or len(x) <= 200 else x[:200] + "..."
        if isinstance(x, dict):
            return {k: trim(v, in_rows) for k, v in x.items() if v not in (None, "", [], {})}
        if isinstance(x, list):
            rows = len(x) >= 2 and all(isinstance(v, dict) for v in x)
            return [trim(v, in_rows or rows) for v in x]
        return x
    return json.dumps(trim(_clip(observation), False), ensure_ascii=False, separators=(",", ":"))


def _compact(text: str, limit: int = 1500) -> str:

    try:
        parsed = json.loads(text)
    except Exception:
        return text[:300] + "..."

    def shorten(x: Any) -> Any:
        if isinstance(x, str):
            return x if len(x) <= 60 else "..."
        if isinstance(x, list):
            return [shorten(item) for item in x]
        if isinstance(x, dict):
            return {key: shorten(item) for key, item in x.items()}
        return x

    short = json.dumps(shorten(parsed), ensure_ascii=False)
    return short if len(short) <= limit else short[:limit] + "..."


def _json_from_text(text: str) -> Any:

    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except Exception:
        return None


def _call_arguments(tool_call: dict[str, Any]) -> dict[str, Any]:

    raw = tool_call["function"].get("arguments", "{}")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(parsed, dict):
        raise ValueError("tool arguments must be a JSON object")
    return parsed


def _flat_facts(record: dict[str, Any]) -> dict[str, Any]:

    flat: dict[str, Any] = {}
    for key, value in (record or {}).items():
        if isinstance(value, dict):
            flat.update({k: v for k, v in value.items() if isinstance(v, (str, int, float, bool))})
        elif isinstance(value, (str, int, float, bool)):
            flat[key] = value
    return flat


def _row_price(row: dict[str, Any]) -> float:

    for key, value in row.items():
        if _PRICE_KEY.search(key) and isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return float("inf")


def _in_stock(record: dict[str, Any]) -> bool:

    flags = [value for key, value in record.items() if _STOCK_KEY.search(str(key)) and isinstance(value, bool)]
    return all(flags) if flags else True


class StoreTools:


    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = tools
        self.params = {(t.get("function") or {}).get("name", ""): ((t.get("function") or {}).get("parameters") or {})
                       for t in tools}
        protocol = _read_tool_protocol(tools, self.params)
        self.roles: dict[str, str] = protocol["roles"]
        self.ask, self.ask_arg = "", ""
        for tool in tools:
            name = (tool.get("function") or {}).get("name", "")
            props = self.params[name].get("properties") or {}
            if (self.roles.get(name) == "message" and len(props) == 1
                    and (next(iter(props.values())) or {}).get("type", "string") == "string"):
                self.ask, self.ask_arg = name, next(iter(props))
                break
        self.close = {n for n, r in self.roles.items() if r in ("order", "decline", "ending") and n != self.ask}
        self.cart = {n for n, r in self.roles.items() if r == "cart" and n not in self.close and n != self.ask}
        self.ending: dict[str, dict[str, Any]] = protocol["ending"]
        self.cart_actions: dict[str, dict[str, Any]] = protocol["cart"]
        self.sweep: dict[str, Any] | None = protocol["price_filter"]
        self.search_modes: bool = protocol["search_modes"]
        self.details: list[str] = protocol["details"]
        taken = self.close | self.cart | {self.ask}
        self.browse = [tool for tool in tools if (tool.get("function") or {}).get("name") not in taken]
        self.browse_names = {(tool.get("function") or {}).get("name") for tool in self.browse}

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:


        role = self.roles.get(name)
        if role == "decline":
            return True
        spec = self.ending.get(name) if role == "ending" else None
        if not spec:
            return False
        value = _plain_value(args.get(spec["param"]))
        if spec.get("decline") is not None:
            return value == _plain_value(spec["decline"])
        return bool(value) and value != _plain_value(spec.get("order"))

    def order_call(self) -> tuple[str, dict[str, Any]]:


        for name, role in self.roles.items():
            if role == "order" and not self.params.get(name, {}).get("required"):
                return name, {}
        for name, spec in self.ending.items():
            required = set(self.params.get(name, {}).get("required") or [])
            if spec.get("order") is not None and required <= {spec["param"]}:
                return name, {spec["param"]: spec["order"]}
        return "", {}

    def cart_effect(self, name: str, args: dict[str, Any], seen: dict[str, Any]) -> tuple[str, str, int] | None:


        args = {k: v for k, v in args.items() if k != "notes"}
        ids = [v for v in args.values() if isinstance(v, str) and v in seen]
        if len(ids) != 1:
            return None
        listing = ids[0]
        numbers = [int(v) for v in args.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        units = numbers[0] if numbers else 1
        spec = self.cart_actions.get(name) or {}
        param = spec.get("param")
        if param and param in args:
            return listing, ("remove" if _plain_value(args[param]) in spec["remove"] else "add"), units
        if numbers:
            return listing, ("remove" if units <= 0 else "set"), units
        return listing, "add", units


def _plain_value(value: Any) -> str:
    return str(value if value is not None else "").strip().lower()


def _tool_rows(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:


    rows = []
    for tool in tools:
        fn = tool.get("function") or {}
        params = {}
        for key, spec in ((fn.get("parameters") or {}).get("properties") or {}).items():
            spec = spec or {}
            entry = {"type": spec.get("type", "string")}
            if spec.get("description"):
                entry["description"] = spec["description"]
            if spec.get("enum"):
                entry["allowed"] = list(spec["enum"])[:12] + (["..."] if len(spec["enum"]) > 12 else [])
            params[key] = entry
        rows.append({"name": fn.get("name"), "description": fn.get("description"), "parameters": params})
    return rows


def _read_tool_protocol(tools: list[dict[str, Any]], params: dict[str, dict[str, Any]]) -> dict[str, Any]:


    protocol: dict[str, Any] = _checked_protocol(None, params)
    messages = [{"role": "system", "content": "You read tool descriptions. Reply with one JSON object."},
                {"role": "user", "content": _PROMPT_TOOL_PROTOCOL + json.dumps(_tool_rows(tools), ensure_ascii=False)}]
    for model in (_MAIN_MODEL, _STRONG_MODEL):
        try:
            data, _ = _chat_json(model, messages)
        except Exception:
            continue
        protocol = _checked_protocol(data, params)
        roles = set(protocol["roles"].values())
        if "cart" in roles and roles & {"order", "ending"}:
            break
    return protocol


def _checked_protocol(data: Any, params: dict[str, dict[str, Any]]) -> dict[str, Any]:


    data = data if isinstance(data, dict) else {}

    def has_param(tool: str, param: Any) -> bool:
        return isinstance(param, str) and param in (params.get(tool, {}).get("properties") or {})

    def allowed(tool: str, param: str, value: Any) -> Any:

        if value is None:
            return None
        choices = ((params.get(tool, {}).get("properties") or {}).get(param) or {}).get("enum")
        if not choices:
            return str(value)
        return next((c for c in choices if _plain_value(c) == _plain_value(value)), None)

    known = ("message", "cart", "order", "decline", "ending", "other")
    roles = {str(k): str(v).strip().lower() for k, v in (data.get("roles") or {}).items()
             if str(k) in params and str(v).strip().lower() in known} if isinstance(data.get("roles"), dict) else {}
    ending: dict[str, dict[str, Any]] = {}
    for tool, spec in (data.get("ending") or {}).items() if isinstance(data.get("ending"), dict) else []:
        if roles.get(tool) == "ending" and isinstance(spec, dict) and has_param(tool, spec.get("param")):
            order, decline = allowed(tool, spec["param"], spec.get("order")), allowed(tool, spec["param"], spec.get("decline"))
            if order is not None or decline is not None:
                ending[tool] = {"param": spec["param"], "order": order, "decline": decline}
    for tool in [t for t, r in roles.items() if r == "ending" and t not in ending]:
        roles[tool] = "other"
    cart: dict[str, dict[str, Any]] = {}
    for tool, spec in (data.get("cart") or {}).items() if isinstance(data.get("cart"), dict) else []:
        if roles.get(tool) == "cart" and isinstance(spec, dict) and has_param(tool, spec.get("param")):
            removes = [allowed(tool, spec["param"], v) for v in (spec.get("remove") or []) if isinstance(v, str)]
            cart[tool] = {"param": spec["param"], "remove": {_plain_value(v) for v in removes if v is not None}}
    sweep = None
    pf = data.get("price_filter") if isinstance(data.get("price_filter"), dict) else {}
    tool = pf.get("tool")
    if isinstance(tool, str) and roles.get(tool, "other") == "other" and has_param(tool, pf.get("cap"))\
            and has_param(tool, pf.get("shelf")):
        props = params[tool].get("properties") or {}
        cap_type = (props[pf["cap"]] or {}).get("type")
        if cap_type in ("integer", "number") and (props[pf["shelf"]] or {}).get("type", "string") == "string":
            mode = None
            if has_param(tool, pf.get("mode_param")) and allowed(tool, pf["mode_param"], pf.get("mode_value")) is not None:
                mode = (pf["mode_param"], allowed(tool, pf["mode_param"], pf.get("mode_value")))
            sweep = {"name": tool, "cap": pf["cap"], "int": cap_type == "integer", "shelf": pf["shelf"], "mode": mode}
    details = [t for t in (data.get("details") or []) if isinstance(t, str) and t in params
               and roles.get(t, "other") == "other" and len(params[t].get("properties") or {}) == 1]\
        if isinstance(data.get("details"), list) else []
    return {"roles": roles, "ending": ending, "cart": cart, "price_filter": sweep,
            "search_modes": data.get("search_modes") is True, "details": details}


def _priced_rows(observation: Any) -> list[tuple[str, float, dict[str, Any]]]:

    rows: list[tuple[str, float, dict[str, Any]]] = []

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            for value in x.values():
                walk(value)
        elif isinstance(x, list):
            if x and all(isinstance(row, dict) for row in x):
                for row in x:
                    ident = next((v for k, v in row.items() if isinstance(v, str) and _ID_KEY.match(k)), None)
                    if ident and not any(_STOCK_KEY.search(k) and v is False for k, v in row.items()):
                        rows.append((ident, _row_price(row), row))
            else:
                for value in x:
                    walk(value)

    walk(observation)
    return rows


def _price_ladder(episode: Episode, sweep: dict[str, Any], listing: str) -> list[tuple[float, str, dict[str, Any]]]:


    record = episode.seen.get(listing) or {}
    price = _row_price(record)
    shelf = next((v for k, v in record.items() if isinstance(v, str) and _CAT_FIELD.match(k)), None)
    if price == float("inf") or not shelf or price <= 0:
        return []
    step_down = 1 if sweep["int"] else 0.01
    found: dict[str, tuple[float, dict[str, Any]]] = {}
    cap = price - step_down
    for _ in range(3):
        if episode.left() < 8 or cap <= 0:
            break
        args: dict[str, Any] = {sweep["shelf"]: shelf, sweep["cap"]: int(cap) if sweep["int"] else round(cap, 2)}
        if sweep["mode"]:
            args[sweep["mode"][0]] = sweep["mode"][1]
        try:
            observation = episode.call(sweep["name"], args)
        except Exception:
            break
        page = [(ident, p, row) for ident, p, row in _priced_rows(observation) if p < price and ident != listing]
        if not page:
            break
        for ident, p, row in page:
            found[ident] = (p, row)
        cap = min(p for _, p, _ in page) - step_down
    cheaper = ((p, ident, row) for ident, (p, row) in found.items() if ident not in episode.named)
    return sorted(cheaper, key=lambda x: x[0])[:6]


def _many_ids_failed(args: dict[str, Any], observation: Any) -> int:

    n = max((len(v) for v in args.values() if isinstance(v, list) and all(isinstance(x, str) for x in v)), default=0)
    if n < 3 or _row_ids(observation) or len(json.dumps(observation, default=str)) > 400:
        return 0
    return n


def _over_id_cap(episode: Episode, name: str, args: dict[str, Any]) -> bool:

    cap = episode.id_cap.get(name)
    if not cap:
        return False
    return any(isinstance(v, list) and len(v) > cap for v in args.values())


def _strong_model() -> str:

    return _STRONG_MODEL if _SPENT[0] < _SPEND_SOFT else _MAIN_MODEL


def _inference_client() -> Any:

    left = _TIME_LIMIT - (time.monotonic() - _TASK_START[0]) - 15
    if left >= 120:
        return _proxy
    limit = max(10, int(left) // 5 * 5)
    if limit not in _SHORT_CLIENTS:
        _SHORT_CLIENTS[limit] = ProxyClient(timeout=limit, max_retries=1)
    return _SHORT_CLIENTS[limit]


def _add_spend(reply: Any) -> None:

    try:
        usage = (reply or {}).get("usage") or {}
        cost = float(usage.get("cost") or 0)
        if usage.get("is_byok"):
            cost += float((usage.get("cost_details") or {}).get("upstream_inference_cost") or 0)
        _SPENT[0] += cost
    except Exception:
        pass


def _cache_marks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:


    sent = [dict(message) for message in messages]
    first_user = next((i for i, m in enumerate(sent) if m.get("role") == "user"), None)
    marks = sorted({0, len(sent) - 1} | ({first_user} if first_user is not None else set()))
    for i in marks[:4]:
        content = sent[i].get("content")
        if isinstance(content, str) and content:
            sent[i]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
    return sent


def _chat(model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
          tool_choice: str = "auto", max_tokens: int | None = None) -> dict[str, Any]:


    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens or 8000}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
        if model.startswith("anthropic/") and len(messages) > 1:
            body["messages"] = _cache_marks(messages)
    reply = _inference_client().post("/inference/chat/completions", json_data=body)
    if reply is None:
        raise RuntimeError("inference request failed")
    _add_spend(reply)
    choice = reply["choices"][0]
    message = choice.get("message") or {}
    if choice.get("finish_reason") == "content_filter" and not (message.get("content") or message.get("tool_calls")):
        body["temperature"] = 0.2
        again = _inference_client().post("/inference/chat/completions", json_data=body)
        _add_spend(again)
        if again is not None:
            return again["choices"][0].get("message") or {}
    return message


def _chat_json(model: str, messages: list[dict[str, Any]], max_tokens: int = 4000) -> tuple[Any, str]:

    text = ""
    for _ in range(2):
        text = _chat(model, messages, max_tokens=max_tokens).get("content") or ""
        data = _json_from_text(text)
        if data is not None:
            return data, text
        messages = messages + [{"role": "assistant", "content": text.strip() or "(no reply)"},
                               {"role": "user", "content": "That reply was not a JSON object. Reply again with the "
                                                           "JSON object only."}]
    return None, text


def _plain_lower(text: Any) -> str:

    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", str(text).lower())


def _value_norm(text: Any) -> str:

    t = re.sub(r"[^a-z0-9.~+ ]+", " ", _plain_lower(text))
    t = re.sub(r"(?<=\d)(?=[a-z])|(?<=[a-z])(?=\d)", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _value_tokens(text: Any) -> list[str]:

    tokens = []
    for token in _value_norm(text).split():
        token = token.strip(".")
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if token:
            tokens.append(token)
    return tokens


def _item_index(text: str, names: list[str]) -> int | None:

    if len(names) == 1:
        return 0
    words = _value_tokens(text)
    for i, name in enumerate(names):
        if words and _value_tokens(name) == words:
            return i
    ranked = sorted(((len(set(_value_tokens(name)) & set(words)), i) for i, name in enumerate(names)), reverse=True)
    if ranked and ranked[0][0] > 0 and (len(ranked) < 2 or ranked[0][0] > ranked[1][0]):
        return ranked[0][1]
    return None


def _listing_item(facts: dict[str, Any], names: list[str]) -> int | None:

    category = " ".join(str(v) for k, v in facts.items() if re.search(r"(category|department|shelf|group)", k, re.I))
    return _item_index(category, names)


def _norm_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _norm_prop(prop: str, names: list[str]) -> str:

    nouns = {w.rstrip("s") for name in names for w in re.findall(r"[a-z]+", name.lower()) if len(w) > 2}
    words = re.findall(r"[a-z0-9]+", (prop or "").lower())
    kept = [w for w in words if w.rstrip("s") not in nouns] or words
    return "".join(kept)


def _match_name(text: str, names: list[str]) -> str:

    t = _norm_text(text)
    return next((n for n in names if _norm_text(n) == t or _norm_text(n) in t or t in _norm_text(n)), "")


def _kind_words(text: str) -> set[str]:
    return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower()) if len(w) > 2 and w not in ("and", "the")}


def _item_text(item: dict[str, Any]) -> str:
    must = "; ".join(str(m) for m in item.get("must") or []) or "none stated"
    return f"{item.get('name')} x {item.get('quantity') or 1} (requirements: {must})"


def _final_value(report: str) -> str | None:

    found = _FINAL_VALUE_RE.findall(report or "")
    if not found:
        return None
    value = found[-1].strip().strip("*").strip()
    if not value or "not found" in value.lower():
        return None
    value = value.split(":")[-1].strip().strip("*'\". ")
    return value if 0 < len(value) <= 40 else None


def _notice_texts(observation: Any) -> list[str]:

    if not isinstance(observation, dict) or len(observation) != 2:
        return []
    for value in observation.values():
        if isinstance(value, list) and value and all(isinstance(x, str) for x in value):
            return [x for x in value if x.strip()]
    return []


def _situation_flags(plan: dict[str, Any] | None, store: StoreTools) -> set[str]:

    flags: set[str] = set()
    items = [i for i in (plan or {}).get("items") or [] if isinstance(i, dict)]
    if len(items) > 1:
        flags.add("multi_item")
    if any(i.get("lookups") for i in items):
        flags.add("lookup")
    for item in items:
        kind = _kind_words(str(item.get("name") or ""))
        for lookup in item.get("lookups") or []:
            if isinstance(lookup, dict):
                same = bool(kind) and kind <= _kind_words(str(lookup.get("listing") or ""))
                flags.add("lookup_same" if same else "lookup_other")
    if (plan or {}).get("confirm"):
        flags.add("confirm")
    if any(re.search(r"\d\s*[a-z]", str(m), re.I) for item in items for m in item.get("must") or []):
        flags.add("measure")
    if store.sweep:
        flags.add("price_filter")
    if any((p or {}).get("type") == "array" for tool in store.browse
           for p in (((tool.get("function") or {}).get("parameters") or {}).get("properties") or {}).values()):
        flags.add("compare_tool")
    if store.search_modes:
        flags.add("search_modes")
    if len(store.details) >= 2:
        flags.add("page_and_facts")
    if not store.ask:
        flags.add("no_messages")
    return flags


def _module_text(flags: set[str], stage: str) -> str:

    return "\n".join(m["text"] for m in _MODULES if m["stage"] == stage and set(m["flags"]) <= flags)


def _checklist(request: str) -> dict[str, Any] | None:

    try:
        message = _chat(_STRONG_MODEL, [
            {"role": "system", "content": "You prepare shopping checklists. Reply with one JSON object."},
            {"role": "user", "content": f"SHOPPER REQUEST:\n{request}\n\n{_PROMPT_CHECKLIST}"}], max_tokens=1500)
        data = _json_from_text(message.get("content") or "")
        if isinstance(data, dict) and isinstance(data.get("items"), list) and data["items"]:
            return data
    except Exception:
        pass
    return None


def _lookup_task(n: int, name: Any, lookup: dict[str, Any]) -> str:

    return (f"For item {n + 1} ({name}), find this listing: {lookup.get('listing')}, and read its "
            f"'{lookup.get('read')}'. Work in hops. Each hop: (a) search the seller or brand name plus ONE "
            "identifying value word (2-3 words, e.g. '<seller> <value>'); (b) read ALL of that seller's rows of "
            "the right kind in ONE side-by-side call (rows do not show properties); (c) open the PAGE view of the "
            "listing that has the identifying properties (notes and pointers are on the page; when the store has "
            "separate page and facts views, open the page even if you read its facts) and read what it says. If it "
            "says the value is to be taken from another "
            "listing, do the next hop for that listing. Stop when a page states the value; if it says which "
            "value goes together with its own, the item to buy needs that other value. If a filter returns "
            "nothing, do not retry it. If searches of the seller plus a value do not show the seller's row with those "
            "values, search the seller name alone and read the next pages of that search (a seller can have many "
            "rows). Report: the chain (id, seller, what each page says) and a last line "
            f"'FINAL VALUE: <property>: <value the item to buy must have>' ({lookup.get('relation')}), or "
            "'FINAL VALUE: not found' with the next listing to look for.")


def _lookup_identifiers(listing: str) -> list[str]:

    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
            {"role": "user", "content": _PROMPT_LOOKUP_IDENTIFIERS.format(listing=listing)}])
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    return [str(x).strip() for x in data.get("properties") or [] if isinstance(x, str) and str(x).strip()][:4]


def _reply_lookups(talk: list[tuple[str, str]]) -> list[dict[str, Any]]:

    if not talk:
        return []
    conversation = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk)
    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You read shopper replies. Reply with one JSON object."},
            {"role": "user", "content": f"{conversation}\n\n{_PROMPT_REPLY_LOOKUPS}"}])
        data = data or {}
        return [x for x in data.get("lookups", []) if isinstance(x, dict) and x.get("listing")][:3]
    except Exception:
        return []


def _story_hints(context: str, request: str, names: list[str]) -> list[tuple[str, str]]:

    try:
        data, _ = _chat_json(_strong_model(), [
            {"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
            {"role": "user", "content": f"{context}\n\n{_PROMPT_STORY_HINTS}"}])
    except Exception:
        return []
    hints: list[tuple[str, str]] = []
    request_text = re.sub(r"\s+", " ", request.lower())
    for hint in (data or {}).get("hints") or [] if isinstance(data, dict) else []:
        if not isinstance(hint, dict):
            continue
        item, prop = str(hint.get("item") or ""), str(hint.get("property") or "").strip()
        phrase = re.sub(r"\s+", " ", str(hint.get("phrase") or "").lower()).strip(" .,'\"")
        if not prop or len(phrase) < 6 or phrase not in request_text:
            continue
        name = _match_name(item, names)
        if name or len(names) == 1:
            hints.append((name or names[0], prop))
    return hints


def _asks_question(request: str) -> bool:

    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
            {"role": "user", "content": f"REQUEST:\n{request}\n\n{_PROMPT_ASKS_QUESTION}"}])
    except Exception:
        return False
    return isinstance(data, dict) and str(data.get("answer", "")).strip().lower().startswith("y")


def _stated_quantity(request: str, name: str) -> int | None:

    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You read shopping requests. Reply with one JSON object."},
            {"role": "user", "content": f"REQUEST:\n{request}\n\n" + _PROMPT_STATED_QUANTITY.format(name=name)}])
        return int(data.get("quantity")) if isinstance(data, dict) and data.get("quantity") is not None else None
    except Exception:
        return None


def _reply_kind(question: str, reply: str, names: list[str]) -> tuple[str, str, str]:

    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You classify shopper replies. Reply with one JSON object."},
            {"role": "user", "content": f"QUESTION: {question}\nREPLY: {reply}\n\n"
                                        + _PROMPT_REPLY_KIND.format(names=json.dumps(names))}])
    except Exception:
        return "", "", ""
    if not isinstance(data, dict):
        return "", "", ""
    kind = str(data.get("kind") or "").strip().lower()
    item = str(data.get("item") or "").strip()
    reading = ""
    if kind == "requirement" and data.get("property") and data.get("value"):
        reading = f"{item or 'the item'}: {data.get('property')} {data.get('relation') or 'must be'} {data.get('value')}"
    return kind, item, reading


def _next_question(context: str, talk: list[tuple[str, str]], left: int, avoid: str = "",
                   extra: str = "") -> tuple[str, str, list[str]] | None:

    conversation = "\n".join(f"Q{i + 1}: {q}\nA{i + 1}: {a}" for i, (q, a) in enumerate(talk)) or "(no questions yet)"
    messages = [{"role": "system", "content": "You plan questions for a shopping assistant. Reply with one JSON object."},
                {"role": "user", "content": f"{context}\n\nQUESTIONS AND REPLIES SO FAR:\n{conversation}\n\n"
                                            f"Messages left: {left}.\n\n{_PROMPT_QUESTION_PLANNER}"
                                            + (f"\n{extra}" if extra else "") + (f"\n\n{avoid}" if avoid else "")}]
    data: Any = None
    for _ in range(2):
        try:
            data, text = _chat_json(_MAIN_MODEL, messages)
        except Exception:
            return None
        if isinstance(data, dict) and data.get("done") and left > 0:
            messages += [{"role": "assistant", "content": text or ""},
                         {"role": "user", "content": "Messages are still left and an unused message is a wasted chance: "
                                                     "do not return done; pick the most promising remaining question."}]
            continue
        break
    if not isinstance(data, dict) or data.get("done"):
        return None
    item, prop = str(data.get("item") or "").strip(), str(data.get("property") or "").strip()
    if item.lower() in ("null", "none"):
        item = ""
    follow = [str(x).strip() for x in (data.get("follow_up") or []) if isinstance(x, str) and str(x).strip()]
    if not prop and not follow:
        return None
    return item, prop, follow


def _requirements_and_queries(context: str, talk: list[tuple[str, str]], extra: str = "",
                              model: str | None = None) -> tuple[str, list[str]]:

    conversation = "\n".join(f"Q: {q}\nA: {a}" for q, a in talk) or "(none)"
    task = ("First write every requirement value of each item as 'item: property value; item: property value': the "
            "stated ones, the FINAL value of every lookup (follow what the pages say; when the last page says its value "
            "goes together with another value, the item to buy needs that other value; the properties that identify a "
            "looked-up listing are not requirements of the item to buy), and every value the shopper's replies require, "
            "taken literally (keep bounds and comparisons exactly: 'at most N' stays 'at most N', never 'N') even when "
            "it looks like a placeholder (a reply wanting a placeholder-like word as the value under a key requires that "
            "exact word; ignore replies that give no preference or did not understand). "
            + (f"{extra} " if extra else "") + "Then: " + _PROMPT_REQUIREMENTS_SUMMARY
            + " Return JSON only: {\"requirements\": \"...\", \"queries\": [\"...\"]}.")
    try:
        data, _ = _chat_json(model or _MAIN_MODEL, [
            {"role": "system", "content": "You write search queries for an online store. Reply with one JSON object and nothing else."},
            {"role": "user", "content": f"{context}\n\nQUESTIONS AND SHOPPER REPLIES:\n{conversation}\n\n{task}"}])
        data = data or {}
        queries = [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:4]
        return str(data.get("requirements") or "").strip(), queries
    except Exception:
        return "", []


def _clean_requirements(context: str, draft: str) -> str:

    try:
        data, _ = _chat_json(_strong_model(), [
            {"role": "system", "content": "You check shopping requirement lists. Reply with one JSON object."},
            {"role": "user", "content": _PROMPT_CHAIN_CLEANING.format(context=context[-12000:], draft=draft)}])
    except Exception:
        return draft
    if isinstance(data, dict) and isinstance(data.get("requirements"), str) and data["requirements"].strip():
        return data["requirements"].strip()
    return draft


def _dropped_requirement(note: str) -> str | None:

    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You read shopper notes. Reply with one JSON object."},
            {"role": "user", "content": _PROMPT_NOTE_DROP.format(note=note)}])
    except Exception:
        return None
    if isinstance(data, dict) and data.get("drops") is True:
        return str(data.get("property") or "a requirement").strip() or "a requirement"
    return None


def _review_row(ident: str, record: dict[str, Any]) -> str:

    record = {k: v for k, v in record.items() if v not in (None, "", [], {})}
    tables = {k: v for k, v in record.items() if isinstance(v, dict)}
    scalars = {k: v for k, v in record.items() if not isinstance(v, dict)}
    short = {k: v for k, v in scalars.items() if not (isinstance(v, str) and len(v) > 160)}
    long = {k: v[:300] + "..." for k, v in scalars.items() if isinstance(v, str) and len(v) > 160}
    return f"{ident}: " + json.dumps({**short, **tables, **long}, ensure_ascii=False)[:3000]


def _review_rows(listings: dict[str, dict[str, Any]], names: list[str], budget: int = 60000) -> list[str]:


    groups: dict[Any, list[tuple[str, dict[str, Any]]]] = {}
    for ident, record in listings.items():
        if len(record) <= 2 or not _in_stock(record):
            continue
        index = _listing_item(_flat_facts(record), names) if names else None
        groups.setdefault(index, []).append((ident, record))
    queues = []
    for group in groups.values():
        group.sort(key=lambda x: (not any(isinstance(v, dict) for v in x[1].values()), _row_price(x[1])))
        queues.append(group)
    rows: list[str] = []
    used = 0
    while used < budget and any(queues):
        for queue in queues:
            if queue and used < budget:
                ident, record = queue.pop(0)
                row = _review_row(ident, record)
                rows.append(row)
                used += len(row)
    return rows


def _decline_review(context: str, listings: dict[str, dict[str, Any]],
                    names: list[str] | None = None) -> list[tuple[str, str]] | None:

    rows = _review_rows(listings, names or [])
    if not rows:
        return None
    try:
        data, _ = _chat_json(_strong_model(), [
            {"role": "system", "content": "You check shopping decisions. Reply with one JSON object."},
            {"role": "user", "content": _PROMPT_DECLINE_REVIEW.format(context=context[-9000:], listings="\n".join(rows))}])
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("decline") or not data.get("buy"):
        return None
    fits = [(str(x.get("id")), str(x.get("why") or "")) for x in data.get("buy") or []
            if isinstance(x, dict) and str(x.get("id")) in listings]
    return fits or None


def _swap_candidates(seen: dict[str, dict[str, Any]], names: list[str], k: int = 3) -> str:

    groups: dict[Any, list[tuple[float, str]]] = {}
    for ident, record in seen.items():
        if len(record) <= 2 or not _in_stock(record):
            continue
        index = _listing_item(_flat_facts(record), names) if names else None
        groups.setdefault(index, []).append((_row_price(record), ident))
    parts = []
    for index, group in groups.items():
        group.sort()
        label = names[index] if (isinstance(index, int) and index < len(names)) else "any item"
        parts.append(f"{label}: " + ", ".join(i if p == float("inf") else f"{i} ({p:g})" for p, i in group[:k]))
    return "; ".join(parts)


def _twin_queries(messages: list[dict[str, Any]]) -> list[str]:

    recent = [m for m in messages if m.get("role") in ("tool", "assistant")][-14:]
    lines = []
    for message in recent:
        if message["role"] == "tool":
            lines.append("RESULT: " + str(message.get("content"))[:2500])
        else:
            for tc in message.get("tool_calls") or []:
                lines.append("CALL: " + str(tc["function"].get("name")) + " " + str(tc["function"].get("arguments"))[:800])
    try:
        data, _ = _chat_json(_MAIN_MODEL, [
            {"role": "system", "content": "You review a shopping basket. Reply with one JSON object."},
            {"role": "user", "content": "\n".join(lines) + "\n\n" + _PROMPT_TWIN_QUERIES}])
        data = data or {}
        return [q for q in data.get("queries", []) if isinstance(q, str) and q.strip()][:3]
    except Exception:
        return []


def _browse(episode: Episode, store: StoreTools, context: str, task: str, allowance: int, deadline: float,
            extra: str = "", model: str | None = None) -> str:


    model = model or _MAIN_MODEL
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _PROMPT_BROWSE_SYSTEM + (f"\n{extra}" if extra else "")},
        {"role": "user", "content": f"{context}\n\nYOUR TASK: {task}\nYou may make at most {allowance} tool calls."}]
    used = 0
    for _ in range(allowance + 2):
        if episode.finished:
            return ""
        if used >= allowance or episode.elapsed() > deadline:
            break
        try:
            reply = _chat(model, messages, store.browse)
        except Exception:
            return ""
        calls = reply.get("tool_calls") or []
        if not calls:
            return (reply.get("content") or "").strip()
        content = reply.get("content") or ""
        messages.append({"role": "assistant", "content": content, "tool_calls": calls})
        for call in calls:
            name = call["function"]["name"]
            try:
                args = _call_arguments(call)
            except ValueError as exc:
                args, observation = None, {"error": str(exc)}
            if args is None:
                pass
            elif used >= allowance:
                observation = {"error": "tool-call allowance used: write your report now"}
            elif _over_id_cap(episode, name, args):
                observation = {"error": _NOTE_ID_CAP}
            else:
                observation = _browse_call(episode, call, args, content)
                used += 1
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": _tool_text(observation)})
    messages.append({"role": "user", "content": "Stop now and write your report in plain text."})
    try:
        return (_chat(model, messages, store.browse, tool_choice="none").get("content") or "").strip()
    except Exception:
        return ""


def _browse_call(episode: Episode, call: dict[str, Any], args: dict[str, Any], content: str) -> Any:

    name = call["function"]["name"]
    try:
        observation = episode.call(name, args, call, content)
    except ValueError as exc:
        observation = {"error": str(exc)}
    try:
        failed = _many_ids_failed(args, observation)
    except Exception:
        failed = 0
    if failed:
        episode.id_cap[name] = 2
        observation = {"result": observation, "note": _NOTE_MULTI_ID_FAIL.format(n=failed)}
    return observation


def _has_final_value(report: str) -> bool:
    upper = report.upper()
    return "FINAL VALUE" in upper and "NOT FOUND" not in upper


def _named_item(text: str, names: list[str]) -> str:


    key = _norm_text(text)
    exact = next((n for n in names if _norm_text(n) == key), None)
    if exact:
        return exact
    return next((n for n in names if _norm_text(n) and (_norm_text(n) in key or key in _norm_text(n))), text)


def _related_names(text: str, names: list[str]) -> set[str]:

    key = _norm_text(text)
    return {_norm_text(n) for n in names if _norm_text(n) == key or _norm_text(n) in key or key in _norm_text(n)}


class Shopping:


    def __init__(self, episode: Episode, store: StoreTools, request: str, plan: dict[str, Any], stated_limit: int):
        self.episode, self.store, self.request, self.plan = episode, store, request, plan
        self.items, self.category_items = self._prepare_items()
        self.names = [str(item.get("name") or "") for item in self.items]
        self.flags = _situation_flags(plan, store)
        self.msg_limit = self._message_limit(stated_limit)
        self.confirm = bool(plan.get("confirm")) and self.msg_limit > 0
        self.q_limit = self.msg_limit - (1 if self.confirm else 0)
        self.base = self._base_context()

        self.reserve_buy = max(10, 3 * len(self.items) + 4)
        self.browse_budget = max(0, episode.max_steps - self.reserve_buy - min(self.msg_limit, 6))

        self.look = ""
        self.reports = ""
        self.not_requirements: list[str] = []
        self.derived: dict[int, list[str]] = {}
        self.talk: list[tuple[str, str]] = []
        self.revealed: list[str] = []
        self.dropped: list[str] = []
        self.notes_seen: set[str] = set()
        self.requirements = ""
        self.queries: list[str] = []
        self.picks: list[str] = []


    def _prepare_items(self) -> tuple[list[dict[str, Any]], list[str]]:


        items = self._merged_items([item for item in self.plan["items"] if isinstance(item, dict)])
        items, category_items = self._grouped_items(items)
        self.plan["items"] = items
        for item in items:
            try:
                quantity = int(item.get("quantity") or 1)
            except (TypeError, ValueError):
                quantity = 1
            if quantity > 1:
                stated = _stated_quantity(self.request, str(item.get("name") or ""))
                if stated and stated != quantity:
                    item["quantity"] = stated
        for item in items:
            identifiers = " ".join(re.sub(r"[^a-z0-9]+", " ", str(lookup.get("listing") or "").lower())
                                   for lookup in item.get("lookups") or [] if isinstance(lookup, dict))
            if identifiers:
                item["must"] = [m for m in item.get("must") or []
                                if re.sub(r"[^a-z0-9]+", " ", str(m).lower()).strip() not in f" {identifiers} "]
        return items, category_items

    @staticmethod
    def _merged_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:

        merged: list[dict[str, Any]] = []
        for item in items:
            same = next((m for m in merged if _norm_text(str(m.get("name"))) == _norm_text(str(item.get("name")))), None)
            if same is None:
                merged.append(item)
                continue
            same["quantity"] = item.get("quantity") or same.get("quantity")
            same["must"] = list(dict.fromkeys(list(same.get("must") or []) + list(item.get("must") or [])))
            same["lookups"] = list(same.get("lookups") or []) + list(item.get("lookups") or [])
            same["hints"] = list(same.get("hints") or []) + list(item.get("hints") or [])
        return merged

    @staticmethod
    def _grouped_items(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:


        groups: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            key = "|".join(sorted(_norm_text(str(m)) for m in item.get("must") or []))
            if key:
                groups.setdefault(key, []).append(item)
        category_items: list[str] = []
        for group in groups.values():
            if len(group) < 3 or any(member.get("lookups") for member in group):
                continue
            first = group[0]
            words = [str(member.get("name") or "").split() for member in group]
            shared_lead = (all(w and w[0].lower() == words[0][0].lower() for w in words)
                           and all(len(w) > 1 for w in words))
            lead = words[0][0] if shared_lead else ""
            parts = [" ".join(w[1:]) if lead else " ".join(w) for w in words]
            first["name"] = ((lead + " ") if lead else "") + ", ".join(parts[:-1]) + " & " + parts[-1]
            category_items.append(str(first["name"]))
            first["quantity"] = max(int(member.get("quantity") or 1) for member in group)
            items = [item for item in items if item is first or item not in group]
        return items, category_items

    def _message_limit(self, stated_limit: int) -> int:


        try:
            limit = int(self.plan.get("messages")) if self.plan.get("messages") is not None else stated_limit or 3
        except (TypeError, ValueError):
            limit = 3
        if stated_limit:
            limit = min(limit, stated_limit)
        if not self.store.ask:
            limit = 0
        return limit

    def _base_context(self) -> str:
        plan = self.plan
        lines = "\n".join(f"ITEM {n + 1}: {_item_text(item)}" for n, item in enumerate(self.items))
        one_item = "".join(f"\nONE ITEM: '{c}' is a single store category: buy exactly ONE listing of it (any one of "
                           "these kinds), never one listing per kind." for c in self.category_items)
        return (f"SHOPPER REQUEST:\n{self.request}\n\nCHECKLIST:\n{lines}\nBUDGET: {plan.get('budget')}"
                f"\nEXCLUDE: {plan.get('exclude')}\nOTHER: {plan.get('other')}" + one_item)


    def _browse_counted(self, context: str, task: str, allowance: int, deadline: float) -> str:

        before = self.episode.step
        report = _browse(self.episode, self.store, context, task, allowance, deadline, _module_text(self.flags, "browse"))
        self.browse_budget -= self.episode.step - before
        return report

    def lookups(self) -> None:

        reports: list[str] = []
        for n, item in enumerate(self.items):
            for lookup in item.get("lookups") or []:
                if not isinstance(lookup, dict) or self.episode.finished:
                    continue
                allowance = min(8, self.browse_budget)
                if allowance < 2:
                    break
                report = self._lookup(_lookup_task(n, item.get("name"), lookup), allowance)
                reports.append(f"LOOKUP for item {n + 1} ({lookup.get('listing')} -> {lookup.get('read')}):\n{report}")
                value = _final_value(report)
                if value:
                    self.derived.setdefault(n, []).append(value)
                self.not_requirements.append(f"item {n + 1}: {lookup.get('listing')} (only identifies the listing to "
                                             f"read '{lookup.get('read')}' from)")
        self.look = "\n\n".join(reports)

    def _lookup(self, task: str, allowance: int) -> str:
        report = self._browse_counted(self.base, task, allowance, _T_RESEARCH)
        if _has_final_value(report) or self.browse_budget < 4 or self.episode.finished:
            return report
        more = self._browse_counted(self.base + f"\n\nREPORT SO FAR:\n{report}", "Continue this lookup from the report "
                                    "so far (do not repeat its reads). " + task, min(5, self.browse_budget), _T_RESEARCH)
        report = report + "\n\nCONTINUED:\n" + more
        if (_has_final_value(more) or "next listing" not in more.lower() or self.browse_budget < 6
                or self.episode.finished):
            return report
        more = self._browse_counted(self.base + f"\n\nREPORT SO FAR:\n{report}", "Continue this lookup from the "
                                    "report so far (do not repeat its reads). " + task, min(5, self.browse_budget - 2),
                                    _T_RESEARCH)
        return report + "\n\nCONTINUED:\n" + more

    def research(self) -> None:

        research: list[str] = []
        per_item = max(2, min(4, self.browse_budget // max(1, len(self.items))))
        context = self.base + (f"\n\n{self.look}" if self.look else "")
        for n, item in enumerate(self.items):
            if self.episode.finished or self.browse_budget < 2 or self.episode.elapsed() > _T_RESEARCH:
                break
            task = (f"Item {n + 1}: {_item_text(item)}. Find in-stock listings of this kind that meet its requirements "
                    "(and the lookup results above, if any): at most 2 searches of 2-3 words (the item's kind word plus "
                    "one distinctive requirement value), then READ the details of the promising rows (side by side when "
                    "the store has a tool for that); do not end without reading details. Report: up to 6 CANDIDATES (id, "
                    "price, in stock, the facts that matter, which requirements each meets), and KEYS: the keys of the "
                    "listings' property/facts table (their datasheet) with the values seen; never record fields such as "
                    "title, name, description, seller, shop, maker, price or stock.")
            if self.derived.get(n):
                task += self._derived_search(item, self.derived[n][0])
            report = self._browse_counted(context, task, min(per_item, self.browse_budget), _T_RESEARCH)
            research.append(f"RESEARCH for item {n + 1} ({item.get('name')}):\n{report}")
        self.reports = "\n\n".join(r for r in [self.look] + research if r)

    @staticmethod
    def _derived_search(item: dict[str, Any], value: str) -> str:

        if item.get("must"):
            return (f" The lookup gave this item the value '{value}': make your FIRST search one stated requirement "
                    f"value of the item plus that value plus the item's kind word (e.g. '<stated value> {value} "
                    f"{item.get('name')}') and read its rows before any other search.")
        return (f" The lookup gave this item the value '{value}' and the item has no other stated requirement: "
                f"make your FIRST search that value alone ('{value}'), read its next pages too (cheaper rows come "
                "later), and never add words that only identified the looked-up listings.")


    def interview(self) -> None:

        Interview(self).run()


    def _conversation(self) -> str:
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in self.talk)

    def reply_lookups(self) -> None:

        found: list[str] = []
        context = self.base + f"\n\nQUESTIONS AND SHOPPER REPLIES:\n{self._conversation()}"
        episode = self.episode
        extra = _module_text(self.flags, "browse")
        for lookup in _reply_lookups(self.talk):
            if episode.finished or episode.left() < self.reserve_buy + 4 or episode.elapsed() > _T_REPLY_LOOKUP:
                break
            task = _lookup_task(0, lookup.get("item"), lookup).replace("For item 1 (", "For the item (")
            report = _browse(episode, self.store, context, task, min(8, episode.left() - self.reserve_buy),
                             _T_REPLY_LOOKUP_DEADLINE, extra)
            if (not _has_final_value(report) and episode.left() >= self.reserve_buy + 4 and not episode.finished
                    and episode.elapsed() < _T_LATE_BROWSE):
                more = _browse(episode, self.store, self.base + f"\n\nREPORT SO FAR:\n{report}", "Continue this lookup "
                               "from the report so far (do not repeat its reads). " + task,
                               min(6, episode.left() - self.reserve_buy), _T_LATE_DEADLINE, extra)
                report = report + "\n\nCONTINUED:\n" + more
            found.append(f"LOOKUP from a shopper reply ({lookup.get('item')}: {lookup.get('listing')} -> "
                         f"{lookup.get('read')}):\n{report}")
            self.not_requirements.append(f"{lookup.get('item')}: {lookup.get('listing')} (only identifies the listing "
                                         f"to read '{lookup.get('read')}' from)")
        if found:
            self.look = "\n\n".join(([self.look] if self.look else []) + found)
            self.reports = "\n\n".join(([self.reports] if self.reports else []) + found)

    def _not_requirements_text(self) -> str:

        text = ""
        if self.not_requirements:
            text += ("\n\nNOT REQUIREMENTS (seller, brand and properties that only identify a looked-up listing; never "
                     "list them as requirements of the item to buy):\n" + "\n".join(self.not_requirements))
        return text

    def _revealed_text(self) -> str:

        text = ""
        if self.revealed:
            text += "\n\nREVEALED REQUIREMENTS (exact readings of the shopper's replies):\n" + "\n".join(self.revealed)
        if self.dropped:
            text += ("\n\nDROPPED REQUIREMENTS (the shopper's notes let go of these; they no longer apply and any value "
                     "fits):\n" + "\n".join(self.dropped))
        return text

    def final_requirements(self) -> None:


        not_reqs, revealed = self._not_requirements_text(), self._revealed_text()
        context = self.base + (f"\n\n{self.look}" if self.look else "") + not_reqs + revealed
        self.requirements, self.queries = _requirements_and_queries(
            context, self.talk, _module_text(self.flags, "summary"), model=_strong_model() if self.look else None)
        if self.look and self.requirements:
            self.requirements = _clean_requirements(
                self.base + f"\n\n{self.look}\n\nQUESTIONS AND SHOPPER REPLIES:\n{self._conversation()}" + not_reqs
                + revealed, self.requirements)

    def item_picks(self) -> None:

        if len(self.items) <= 1 or not self.requirements or self.episode.finished:
            return
        spare = self.episode.left() - (2 * len(self.items) + 6)
        per_item = min(5, spare // len(self.items)) if spare > 0 else 0
        context = (self.base + f"\n\n{self.reports}\n\nQUESTIONS AND SHOPPER REPLIES:\n{self._talk_text()}"
                   + self._revealed_text() + self._not_requirements_text())
        for n, item in enumerate(self.items):
            if per_item < 2 or self.episode.elapsed() > _T_LATE_BROWSE or self.episode.finished:
                break
            task = (f"Item {n + 1} only: {item.get('name')} x {item.get('quantity') or 1}. Its requirements are the "
                    f"'{item.get('name')}' entries of these FINAL REQUIREMENTS: {self.requirements}. Find the cheapest "
                    "in-stock listing of this kind within the budget that meets every one of them: search with its "
                    "requirement values, read the details of the cheapest rows of the right kind side by side, and read "
                    "the next page of the best search. Report one line 'CHOSEN: <id> (<price>)' with the details that "
                    "meet each requirement, or 'CHOSEN: none' with the closest listing and what it lacks.")
            report = _browse(self.episode, self.store, context, task, per_item, _T_LATE_DEADLINE,
                             _module_text(self.flags, "browse"))
            self.picks.append(f"ITEM {n + 1} ({item.get('name')}): {report}")

    def _talk_text(self) -> str:
        return self._conversation() or "(no questions were asked)"

    def buy_context(self) -> str:

        context = (self.base + f"\n\n{self.reports}\n\nQUESTIONS AND SHOPPER REPLIES:\n{self._talk_text()}"
                   + self._revealed_text() + self._not_requirements_text())
        if self.picks:
            context += ("\n\nPER-ITEM SELECTION (one focused search per item; use these choices unless a check shows "
                        "they fail):\n" + "\n\n".join(self.picks))
        if self.requirements:
            context += f"\n\nFINAL REQUIREMENTS (from the request, the lookups and the replies): {self.requirements}"
        if self.queries:
            context += (f"\nSUGGESTED SEARCHES: {' | '.join(self.queries)}\nYour FIRST call is the search "
                        f"'{self.queries[0]}', exactly as written; then read its cheapest rows of the right kind.")
        return context + (f"\n\nSteps left: {self.episode.left()}. Now buy: confirm the best fitting listing of every "
                          "item (search more if a revealed requirement is not met by the reported candidates), put "
                          "them in the basket and finish with the order.")

    def buy(self) -> None:
        BuyLoop(self.episode, self.store, self.buy_context(), allow_ask=False, summary=self.confirm,
                n_lines=len(self.items), extra=_module_text(self.flags, "buy"),
                seen_notes=self.notes_seen, cheaper_sweep="price_filter" in self.flags, names=self.names).run()


class Interview:


    def __init__(self, shopping: Shopping):
        self.s = shopping
        names = shopping.names
        self.queue: list[tuple[str, str]] = []
        self.asked: set[tuple[str, str]] = set()
        self.asked_list: list[tuple[str, str]] = []
        self.planned: set[str] = set()
        self.covered: list[str] = []
        self.reasked: set[str] = set()
        self.answer_checked = False
        self.last_prop, self.last_itemless = "", False
        for name, item in zip(names, shopping.items):
            for must in item.get("must") or []:
                key = str(must).split(":")[0].strip()
                if key:
                    self.asked.add((_norm_text(name), _norm_prop(key, names)))
                    if len(shopping.items) == 1:
                        self.asked.add((_norm_text(""), _norm_prop(key, names)))
        hints = "; ".join(f"item {n + 1} ({item.get('name')}): {', '.join(str(h) for h in item.get('hints') or [])}"
                          for n, item in enumerate(shopping.items) if item.get("hints"))
        self.context = shopping.base + f"\nSTORY HINTS: {hints or 'none'}\n\n{shopping.reports}"
        for name, prop in _story_hints(self.context, shopping.request, names):
            if self._new(name, prop) and all(self._prop(prop) != self._prop(q) for _, q in self.queue):
                self.queue.append((name, prop))
        self._queue_lookup_identifiers()

    def _prop(self, prop: str) -> str:
        return _norm_prop(prop, self.s.names)

    def _new(self, item: str, prop: str) -> bool:
        return (_norm_text(item), self._prop(prop)) not in self.asked

    def _queue_lookup_identifiers(self) -> None:

        names = self.s.names
        for n, item in enumerate(self.s.items):
            kind = _kind_words(str(item.get("name") or ""))
            for lookup in item.get("lookups") or []:
                if not isinstance(lookup, dict) or not kind or not (kind <= _kind_words(str(lookup.get("listing") or ""))):
                    continue
                identifiers = [p for p in _lookup_identifiers(str(lookup.get("listing") or ""))
                               if self._prop(p) != self._prop(str(lookup.get("read") or ""))]
                for prop in identifiers[:2]:
                    if self._new(names[n], prop) and (names[n], prop) not in self.queue:
                        self.queue.append((names[n], prop))

    def _may_ask(self) -> bool:
        s, episode = self.s, self.s.episode
        return (not episode.finished and bool(s.store.ask) and len(s.talk) < s.q_limit
                and episode.left() > s.reserve_buy - 2
                and (episode.elapsed() < _T_QUESTIONS or (not s.talk and episode.elapsed() < _T_FIRST_QUESTION)))

    def run(self) -> None:
        s = self.s
        while self._may_ask():
            if (s.revealed and not s.confirm and len(s.talk) == s.q_limit - 1 and s.q_limit >= 2
                    and not self.answer_checked):
                self.answer_checked = True
                if _asks_question(s.request):
                    s.confirm = True
                    break
            self.queue = [(i, p) for i, p in self.queue if self._new(i, p)]
            bare = False
            if self.queue:
                item_name, prop = self.queue.pop(0)
                bare = not item_name and len(s.items) == 1
                self.planned.add(_norm_text(item_name))
            else:
                pick = self._planned_question()
                if pick is None:
                    break
                item_name, prop = pick
            if len(s.items) == 1 and not bare:
                item_name = str(s.items[0].get("name") or "")
            self._ask(item_name, prop)

    def _avoid_text(self, attempt: int) -> str:


        s = self.s
        avoid = ("ALREADY ASKED, do not ask again: " + "; ".join(f"{p} ({i or 'no item'})" for i, p in self.asked_list)
                 ) if attempt else ""
        if self.covered:
            avoid = ("ANSWERED FOR EVERY ITEM (a question without an item reaches every item, and the shopper "
                     "has answered it): " + ", ".join(self.covered) + ". Do not ask these again for any item; choose "
                     "another property.") + ("\n" + avoid if avoid else "")
        uncovered = sorted((item for name, item in zip(s.names, s.items) if _norm_text(name) not in self.planned),
                           key=lambda item: len(item.get("must") or []))
        if len(s.items) > 1 and s.talk and uncovered:
            cover = (f"NEXT QUESTION: it must be about the item '{uncovered[0].get('name')}' (give it as the "
                     "item): a property its OWN listings have (see its research report and KEYS).")
            avoid = cover + ("\n" + avoid if avoid else "")
        return avoid

    def _planned_question(self) -> tuple[str, str] | None:

        s = self.s
        names = s.names
        for attempt in range(2):
            pick = _next_question(self.context, s.talk, s.q_limit - len(s.talk), self._avoid_text(attempt),
                                  _module_text(s.flags, "planner"))
            if pick is None:
                return None
            item_name, prop, follow = pick
            if len(s.items) == 1:
                item_name = names[0]
            elif item_name:
                item_name = _named_item(item_name, names)
            if item_name:
                self.planned.update(_related_names(item_name, names))
            if follow and s.talk and self.last_prop and self.last_itemless:
                follow = [self._numbered_item(f) for f in follow]
                self.queue = [(f, self.last_prop) for f in follow if self._new(f, self.last_prop)]
                if self.queue:
                    item_name, prop = self.queue.pop(0)
            if not prop:
                return None
            if len(s.items) > 1 and not any(self._prop(p) == self._prop(prop) and not i for i, p in self.asked_list):
                item_name = ""
            if self._new(item_name, prop):
                return item_name, prop
        return None

    def _numbered_item(self, text: str) -> str:

        names = self.s.names
        m = re.fullmatch(r"(?i)item\s*(\d+)\b.*", text)
        return names[int(m.group(1)) - 1] if m and 0 < int(m.group(1)) <= len(names) else text

    def _ask(self, item_name: str, prop: str) -> None:
        s = self.s
        names = s.names
        question = (f"Do you have a preference on {prop} for the {item_name}?" if item_name
                    else f"Do you have a preference on {prop}?")
        observation = s.episode.call(s.store.ask, {s.store.ask_arg: question})
        if isinstance(observation, dict) and isinstance(observation.get("value"), str) and len(observation) == 1:
            reply = observation["value"]
        else:
            reply = json.dumps(observation, ensure_ascii=False)
        s.talk.append((question, reply))
        self.asked.add((_norm_text(item_name), self._prop(prop)))
        self.asked_list.append((item_name, prop))
        self.last_prop, self.last_itemless = prop, not item_name
        for note in _notice_texts(observation):
            if note not in s.notes_seen:
                s.notes_seen.add(note)
                dropped = _dropped_requirement(note)
                if dropped:
                    s.dropped.append(f"{dropped} (the shopper's note: '{note}')")
        kind, about, reading = _reply_kind(question, reply, names)
        if reading:
            s.revealed.append(reading)
        if kind == "one_at_a_time" and len(s.items) == 1 and item_name and self._prop(prop) not in self.reasked:
            self.reasked.add(self._prop(prop))
            self.queue.insert(0, ("", prop))
        if item_name or len(s.items) <= 1:
            return
        if kind == "requirement":
            for name in names:
                self.asked.add((_norm_text(name), self._prop(prop)))
            self.queue = [(i, p) for i, p in self.queue if self._prop(p) != self._prop(prop)]
            if self._prop(prop) not in {self._prop(c) for c in self.covered}:
                self.covered.append(prop)
            if about:
                self.planned.update(_norm_text(n) for n in names if _norm_text(n) == _norm_text(about))
        if kind == "one_at_a_time":
            self.queue = [(name, prop) for name, item in zip(names, s.items)
                          if not any(_norm_text(prop) in _norm_text(str(m).split(":")[0]) for m in item.get("must") or [])
                          and self._new(name, prop)]


_ID_CAP_REFUSED = object()


def _row_title(row: dict[str, Any]) -> str:

    return str(row.get(next((k for k in row if re.search("title|name", k, re.I)), ""), ""))


def _short_notes(call: dict[str, Any]) -> dict[str, Any]:

    try:
        args = _call_arguments(call)
    except ValueError:
        return call
    notes = args.get("notes")
    if not isinstance(notes, str) or len(notes) <= 200:
        return call
    short = json.dumps({**args, "notes": notes[:200] + "..."}, ensure_ascii=False)
    return {**call, "function": {**call["function"], "arguments": short}}


class BuyLoop:


    def __init__(self, episode: Episode, store: StoreTools, context: str, allow_ask: bool, msg_cap: int = 0,
                 summary: bool = False, n_lines: int = 0, extra: str = "", seen_notes: set[str] | None = None,
                 cheaper_sweep: bool = False, names: list[str] | None = None):
        self.episode, self.store, self.context = episode, store, context
        self.allow_ask, self.msg_cap, self.summary, self.n_lines = allow_ask, msg_cap, summary, n_lines
        self.cheaper_sweep = cheaper_sweep
        self.names = list(names or [])
        self.tools = self._loop_tools()
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": _PROMPT_BUY_SYSTEM + (f"\n{extra}" if extra else "")},
            {"role": "user", "content": context}]
        self.start_step = episode.step
        self.order_name, self.order_args = store.order_call()
        self.sweep = store.sweep
        self.pending: list[str] = []
        self.shortened: dict[int, str] = {}
        self.short_calls: dict[int, list[dict[str, Any]]] = {}

        self.decide_warned = self.steps_warned = self.last_steps_warned = False

        self.decline_held = self.decline_reviewed = self.checked = False
        self.summary_sent = self.refused_noted = False

        self.idle = self.self_refused = self.sent_msgs = self.swap_notes = self.failures = 0
        self.read_ids: set[str] = set()
        self.notes_seen: set[str] = set(seen_notes or ())
        self.paged_queries: set[str] = set()
        self.sold_noted: set[str] = set()
        self.swept: set[str] = set()
        self.added: set[str] = set()

    def _loop_tools(self) -> list[dict[str, Any]]:


        store = self.store
        keep = [t for t in store.tools
                if self.allow_ask or self.summary or (t.get("function") or {}).get("name") != store.ask]
        tools = json.loads(json.dumps(keep))
        for tool in tools:
            params = tool.setdefault("function", {}).setdefault("parameters", {"type": "object", "properties": {}})
            params.setdefault("properties", {})["notes"] = {
                "type": "string", "description": "Your working notes: REQUIREMENTS line and NOTES line."}
            params["required"] = list(params.get("required") or []) + ["notes"]
        return tools


    def run(self) -> None:
        episode = self.episode
        while not episode.finished and episode.step < episode.max_steps:
            if self._must_order_now():
                episode.call(self.order_name, dict(self.order_args))
                break
            self._warn()
            try:
                reply = _chat(_MAIN_MODEL, self._messages_to_send(), self.tools)
            except RuntimeError:
                self.failures += 1
                if self.failures >= 2 or episode.elapsed() > _TIME_LIMIT - 2 * _T_ORDER_MARGIN:
                    self._order_what_is_carted()
                    break
                continue
            content = reply.get("content") or ""
            calls = reply.get("tool_calls") or []
            if not calls:
                self.idle += 1
                if self.idle > 2:
                    self._order_what_is_carted()
                    break
                self.messages += [{"role": "assistant", "content": content},
                                  {"role": "user", "content": "Choose one of the supplied tools to continue."}]
                continue
            self.idle = 0
            self.messages.append({"role": "assistant", "content": content, "tool_calls": calls})
            for call in calls:
                self._answer(call, content)
            if self.pending:
                self.messages.append({"role": "user", "content": " ".join(self.pending)})
                self.pending.clear()

    def _order_what_is_carted(self) -> None:

        _order_carted(self.episode, self.store)

    def _must_order_now(self) -> bool:

        episode = self.episode
        return bool(self.order_name) and (
            episode.elapsed() > _TIME_LIMIT - _T_ORDER_MARGIN
            or (_SPENT[0] > _SPEND_HARD and episode.step - self.start_step >= 6))

    def _warn(self) -> None:
        episode = self.episode
        if not self.decide_warned and (episode.elapsed() > _T_DECIDE or episode.step - self.start_step >= 25
                                       or _SPENT[0] > _SPEND_SOFT):
            self.decide_warned = True
            self._tell("Decide now: put the best fitting listings you have read in the basket and finish with the order.")
        if not self.steps_warned and self.n_lines and episode.left() <= self.n_lines + 7 and not self.decide_warned:
            self.steps_warned = True
            self._tell(f"{episode.left()} steps are left: add your chosen listings to the basket now (searching more "
                       "leaves no steps for the basket check), then finish.")
        if not self.last_steps_warned and episode.left() <= 4:
            self.last_steps_warned = True
            self._tell("Only 4 steps are left: put the best fitting listings in the basket now (if they are not there "
                       "yet) and finish with the order.")

    def _tell(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def _messages_to_send(self) -> list[dict[str, Any]]:


        whole = [i for i, m in enumerate(self.messages) if m.get("role") == "tool" and i not in self.shortened]
        if len(whole) > 16:
            for i in whole[:-8]:
                text = str(self.messages[i].get("content"))
                self.shortened[i] = _compact(text) if len(text) > 300 else text
            cut = whole[-8]
            for i, m in enumerate(self.messages[:cut]):
                if m.get("role") == "assistant" and m.get("tool_calls") and i not in self.short_calls:
                    self.short_calls[i] = [_short_notes(call) for call in m["tool_calls"]]
        sent = []
        for i, m in enumerate(self.messages):
            if i in self.shortened:
                m = {**m, "content": self.shortened[i]}
            elif i in self.short_calls:
                m = {**m, "tool_calls": self.short_calls[i]}
            sent.append(m)
        return sent


    def _answer(self, call: dict[str, Any], content: str) -> None:
        name = call["function"].get("name")
        before = self.episode.step
        repeats = self.episode.repeats
        try:
            args = {k: v for k, v in _call_arguments(call).items() if k != "notes"}
        except ValueError as exc:
            self._tool_result(call, json.dumps({"error": str(exc)}))
            return
        observation = self._respond(name, args, call, content)
        if observation is _ID_CAP_REFUSED:
            self._tool_result(call, json.dumps({"error": _NOTE_ID_CAP}))
            return
        refused = self.episode.step == before and self.episode.repeats == repeats and not self.episode.finished
        self.self_refused = self.self_refused + 1 if refused else 0
        self._tool_result(call, _tool_text(observation))

    def _tool_result(self, call: dict[str, Any], text: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call["id"], "content": text})

    def _respond(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:
        episode = self.episode
        if episode.finished:
            return {"error": "the episode has finished"}
        if self.self_refused >= 4 and name != self.store.ask:
            return episode.call(name, args, call, content)
        for hold in (self._ladder_hold, self._browse_refusal, self._decline_hold, self._review_decline,
                     self._summary_hold, self._summary_message, self._pre_finish_check, self._message_rules):
            observation = hold(name, args, call, content)
            if observation is not None:
                return observation
        return self._run_call(name, args, call, content)


    def _ladder_hold(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        episode = self.episode
        if name not in self.store.cart or not self.sweep:
            return None
        effect = self.store.cart_effect(name, args, episode.seen)
        if not effect or effect[1] not in ("add", "set"):
            return None
        listing = effect[0]
        if listing in self.swept or listing in self.added or len(self.swept) >= self.n_lines + 2:
            return None
        if episode.left() < 10 + self.n_lines or episode.elapsed() >= _T_LADDER:
            return None
        self.swept.add(listing)
        cheaper = _price_ladder(episode, self.sweep, listing)
        if not cheaper:
            return None
        rows = "; ".join(f"{ident} {price:g} {_row_title(row)[:60]}" for price, ident, row in cheaper)
        return {"error": "Not added yet: the store's price filter shows cheaper in-stock rows of the same shelf "
                         "that you have not opened (cheapest first): " + rows
                         + f". The best fit is the cheapest listing that meets every requirement: read these rows' "
                         f"details, cheapest first, before adding {listing}; add {listing} only if none of them fits."}

    def _browse_refusal(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        summary_step = 1 if self.summary and not self.summary_sent else 0
        if name in self.store.browse_names and self.n_lines and self.episode.left() <= self.n_lines + 2 + summary_step + 2:
            return {"error": "Not run (no step used): only the steps to add your picks, finish and recover from a "
                             "refused finish are left. Add the best fitting listings you have read now and finish."}
        return None

    def _is_decline(self, name: str, args: dict[str, Any]) -> bool:
        return name in self.store.close and self.store.is_decline(name, args)

    def _decline_hold(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        episode = self.episode
        if (not self._is_decline(name, args) or self.decline_held or episode.left() < 5
                or episode.elapsed() >= _T_DECLINE_HOLD):
            return None
        self.decline_held = True
        return {"error": "Not declined yet (no step used). A decline fails unless some item has no listing of "
                         "its kind at all. First search the bare required value words with the item's kind "
                         "word (e.g. '<kind> <value>') and read the next pages; a value of several words (e.g. "
                         "'a/b blend') gets one search per distinctive word ('<kind> a', '<kind> b'); read the FACTS view of every "
                         "candidate you excluded only from its page, title or description; a listing whose title or page "
                         "states a required value counts when none states it in its details within the budget. "
                         "Only requirements that the request states, the lookups give or the shopper's replies "
                         "reveal count: a remark about where or how the item will be used is not a requirement, "
                         "and a property the shopper had no preference on accepts any value. "
                         "If every item has a listing, buy the best fitting basket within the budget instead."}

    def _review_decline(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        episode = self.episode
        if not self._is_decline(name, args) or self.decline_reviewed or episode.elapsed() >= _T_DECLINE_REVIEW:
            return None
        left = episode.left()
        if not (left >= 4 if self.decline_held else max(1, self.n_lines) + 1 <= left < 5):
            return None
        self.decline_reviewed = True
        fits = _decline_review(self.context, episode.seen, self.names)
        if not fits:
            return episode.call(name, args, call, content)
        listed = "; ".join(f"{ident} ({why})" for ident, why in fits)
        if self.decline_held:
            advice = (". Open their details if you have not, put them in the basket (the quantities the request asks "
                      "for) and finish with the order.")
        else:
            advice = (". Few steps are left: put them in the basket now (the quantities the request asks for) and "
                      "finish with the order.")
        return {"error": "Not declined (no step used). A check of the listings you have seen found fits: " + listed
                         + advice}

    def _summary_hold(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        if (self.summary and not self.summary_sent and name in self.store.close
                and not self.store.is_decline(name, args) and self.episode.left() >= 2):
            return {"error": "Not finished yet (no step used): the shopper asked to be told what will arrive. Send "
                             "ONE message now with every basket line (item, quantity, price) and the total against "
                             "the budget, then finish with the order."}
        return None

    def _summary_message(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        if name != self.store.ask or not self.summary or self.allow_ask:
            return None
        if self.summary_sent:
            return {"error": "Message not sent (no step used): the summary was already sent; finish with the order."}
        self.summary_sent = True
        return self.episode.call(name, args, call, content)

    def _pre_finish_check(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        episode = self.episode
        if name not in self.store.close or self.checked or episode.left() < 4 or episode.elapsed() >= _T_CHECK:
            return None
        self.checked = True
        twin_queries = _twin_queries(self.messages)
        twins = (" (2) Run these searches now, one after the other: " + " | ".join(twin_queries) + "; read the "
                 "equal-price same-title twins of the basket listings side by side, and keep or swap to the "
                 "twin that has no fine-print note AND shows each required value under every spelling of that "
                 "key the twins use (e.g. both 'x y' and 'x_y').") if twin_queries else ""
        cheaper = ("(5) with the store's price filter, list each item's category capped just below "
                   "its line's price and read the cheaper rows of the right kind side by side; a "
                   "cheaper row that meets every FINAL REQUIREMENT replaces the line. "
                   if self.cheaper_sweep and episode.left() >= self.n_lines + 8 else "")
        return {"error": "This is a check, not a refusal (no step used). Before finishing: (0) view the basket "
                         "and make sure it holds exactly the requested items and quantities, no extra or "
                         "sold-out lines; (1) every line's "
                         "details state each FINAL REQUIREMENT of its item;" + twins + " (3) each line's "
                         "category is the requested kind and it is in stock; (3b) re-read the notices in the "
                         "latest results: the basket total must fit the newest budget; (4) when the store has separate "
                         "page and facts views, every ordered listing has been opened in BOTH views in this "
                         "episode: open the missing ones now. " + cheaper + "Swap a line only for a listing you "
                         "have read that fits better; then finish with the order. Do not decline because of "
                         "this check."}

    def _message_rules(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:

        if name != self.store.ask:
            return None
        if self.sent_msgs >= self.msg_cap:
            return {"error": "Message not sent (no step used): the message limit is used up."}
        if not _QUESTION_FORM.match(str(args.get(self.store.ask_arg, "")).strip()):
            return {"error": "Message not sent (no step used). Send exactly one short question in this form and "
                             "nothing else: 'Do you have a preference on <property> for the <item>?'"}
        return None


    def _run_call(self, name: str, args: dict[str, Any], call: dict[str, Any], content: str) -> Any:
        store, episode = self.store, self.episode
        browsing = name in store.browse_names
        if name == store.ask:
            self.sent_msgs += 1
        if browsing and _over_id_cap(episode, name, args):
            return _ID_CAP_REFUSED
        if browsing:
            self.read_ids |= _ids_in_args(args)
        observation = episode.call(name, args, call, content)
        if name in store.cart:
            effect = store.cart_effect(name, args, episode.seen)
            if effect:
                (self.added.add if effect[1] in ("add", "set") else self.added.discard)(effect[0])
        if browsing:
            self._note_failed_ids(name, args, observation)
            self._note_unread_rows(observation)
        self._note_notices(observation)
        if browsing:
            self._note_next_page(args, observation)
        if name in store.close and not episode.finished and not self.refused_noted:
            self.refused_noted = True
            self.pending.append("The finish was refused. A notice in that reply overrides the request (a new budget "
                                "replaces the old one; a dropped requirement no longer applies). Re-open every carted "
                                "listing's details; if a line is sold out, replace it with the cheapest remaining fit "
                                "you have already read (a fit at the same price counts), then finish again.")
        return observation


    def _note_failed_ids(self, name: str, args: dict[str, Any], observation: Any) -> None:
        failed = _many_ids_failed(args, observation)
        if failed:
            self.episode.id_cap[name] = 2
            self.pending.append(_NOTE_MULTI_ID_FAIL.format(n=failed))

    def _note_unread_rows(self, observation: Any) -> None:
        rows = _row_ids(observation)
        if len(rows) < 2:
            return
        unread = [r for r in rows if r not in self.read_ids]
        if unread:
            self.pending.append("Rows of this result not yet read: " + ", ".join(unread[:8]) + ". Read the "
                                "cheapest ones of the right category before choosing or searching again.")

    def _note_notices(self, observation: Any) -> None:

        fresh: list[str] = []
        for note in _notice_texts(observation):
            if note in self.notes_seen:
                continue
            self.notes_seen.add(note)
            fresh.append(note)
            dropped = _dropped_requirement(note)
            if dropped:
                self.pending.append(
                    f"The shopper's note '{note}' is the shopper's own message (this store "
                    "delivers the shopper's messages beside tool results). It drops the requirement "
                    f"on {dropped}: that requirement no longer applies. Only the stated, looked-up and "
                    "revealed requirements that remain count (a remark about where or how the item "
                    "is used is not one), so the item's best listing is now the cheapest in-stock "
                    "listing of its kind within the budget that meets the remaining requirements: "
                    "the cheapest one may not be among the rows "
                    "you have seen: search the item's category for its cheapest in-stock rows (with "
                    "the store's price filter and a low price cap when it has one), read them, "
                    "re-check the stock of the carted line, and swap the basket line.")
        new_sold = self.episode.sold - self.sold_noted
        self.sold_noted |= new_sold
        if self.swap_notes >= 2 or not (new_sold or any(_SOLD_RE.search(x) for x in fresh)):
            return
        candidates = _swap_candidates(self.episode.seen, self.names)
        if not candidates:
            return
        self.swap_notes += 1
        opening = ("Listing " + ", ".join(sorted(new_sold)) + " now shows out of stock. " if new_sold
                   else "A notice says a listing sold out. ")
        self.pending.append(opening + "Alternatives you have already seen that were in stock when last read, cheapest "
                            "first: " + candidates + ". Re-open the carted lines; if one is sold out, swap it for the "
                            "cheapest of these that meets every requirement of its item (open its details first if "
                            "you have not read them).")

    def _note_next_page(self, args: dict[str, Any], observation: Any) -> None:

        page = _small_page(observation)
        if not page:
            return
        query = json.dumps({k: v for k, v in sorted(args.items()) if not _NEXT_KEY.search(k)
                            and not re.search(r"page|token|notes", k, re.I)}, sort_keys=True)
        if query in self.paged_queries or len(self.paged_queries) >= 3 or self.episode.left() <= self.n_lines + 10:
            return
        self.paged_queries.add(query)
        self.pending.append(f"This result shows only {page[2]} rows and has a next page ({page[0]}: "
                            f"{json.dumps(page[1])}). A cheaper fit is often on a later page: read the next "
                            "pages of the search that names the requirement value (pass that token), up to "
                            "3 pages, checking the right-category rows of each page, before choosing.")


def _request_text(policy: dict[str, Any]) -> str:

    request = policy["query"]
    if policy.get("opening") is not None:
        request += "\n\n" + json.dumps(policy["opening"])
    return request


def _stated_message_limit(request: str) -> int:

    found = _MESSAGE_LIMIT_RE.search(request)
    return int(found.group(1)) if found else 0


def _order_carted(episode: Episode, store: StoreTools) -> None:


    order, args = store.order_call()
    carted = any(((step.get("tool_call") or {}).get("function") or {}).get("name") in store.cart
                 for step in episode.dialogue)
    if order and carted and not episode.finished and episode.step < episode.max_steps:
        try:
            episode.call(order, dict(args))
        except Exception:
            pass


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:

    _SPENT[0] = 0.0
    _TASK_START[0] = time.monotonic()
    episode = Episode(problem_data)
    store = StoreTools(episode.policy["tools"])
    episode.read_only = set(store.browse_names)
    try:
        _shop(episode, store)
    except Exception:
        _order_carted(episode, store)
    return episode.dialogue


def _shop(episode: Episode, store: StoreTools) -> None:

    request = _request_text(episode.policy)
    plan = _checklist(request)
    stated_limit = _stated_message_limit(request)
    if plan is None:
        BuyLoop(episode, store, f"SHOPPER REQUEST:\n{request}", allow_ask=bool(store.ask),
                msg_cap=stated_limit or 3).run()
        return
    shopping = Shopping(episode, store, request, plan, stated_limit)
    shopping.lookups()
    shopping.research()
    shopping.interview()
    if episode.finished:
        return
    shopping.reply_lookups()
    shopping.final_requirements()
    shopping.item_picks()
    shopping.buy()
