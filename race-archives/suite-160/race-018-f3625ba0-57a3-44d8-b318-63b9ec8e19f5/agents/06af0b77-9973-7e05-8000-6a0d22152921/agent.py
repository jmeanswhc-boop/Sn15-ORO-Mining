"""world_agent_will_v4 — upcoming-race agent (WILL v4).

Spine: v5 (KT fit + dialect/basket/reask + clock/spend).
R16–R17 read (PARTIAL prior guess): process + request dominate; Cairn wins on
process gate + multi-prop list asks + ~300s clock; YesYesYes R16 won on
finish/DNF recovery without late_ask. Forward levers for packed flips:
  - late_question before checkout (basket-reserve intact)
  - multi-prop list ask ramp (LIST_SIZE), crowded → solo_ask (dialect safe)
  - process/basket harden + message reserve
  - ~300s deadline + pre-decline recovery search
Kept: plain reask, scoop dialect, process_boost, page_budget, finish nudge.
"""
from __future__ import annotations
import json
import random
import re
import time
from typing import Any
from src.agent.proxy_client import ProxyClient

# --- shopper dialect scoops (enrich LLM parse only; never stop asking) ---
WAIVE_HINTS = (
    "don't mind", "do not mind", "not fussed", "not picky", "just crack on", "never bothered me",
    "skip past it", "you pick", "you choose", "you decide", "your call", "up to you", "leave it open",
    "leave that open", "any of them", "pick any", "pick whatever", "pick whichever", "whatever works",
    "anything works", "anything is fine", "no preference", "not particular", "nowt springs to mind",
    "never given that a thought", "haven't given that a thought", "haven't given it a thought",
    "couldn't tell you which", "could not tell you which", "couldn't tell you owt", "i've no say",
    "no say there", "shrug from me", "happy with whatever", "happy with owt", "happy with anything",
    "not tied to owt", "i'm not tied to owt", "not a favourite", "no favourite", "no favorite",
    "i trust you", "your judgment", "your judgement", "wide open", "no opinion", "not bothered",
    "all the same", "same to me", "any would do", "choose for me", "just choose", "surprise",
    # Race13 shopper dialect
    "no dog in that fight", "go with what looks sensible", "wet cardboard", "anything goes",
    "can't say i've got a view", "blank slate", "unmarked plank", "any option will do",
    "i'd shrug", "open slather", "use your own judgement", "use your own judgment",
    "pick whichever seems sensible", "what do most people pick", "which would you choose",
    "any option is acceptable", "both are alright", "i'm open on this", "what would you suggest",
    "sail with the tide", "neither here nor there", "not a favourite there", "choose as you see fit",
    "i've no fixed view", "glad of a sit-down", "too tired to be choosy",
    # R17 race evidence (Cairn-pass / WILL-process gaps): waive phrases we missed → same-prop reask
    "leaves me cold", "choose freely", "zero feelings", "no feelings on",
    "toss a coin", "shuffle the deck", "read tea leaves", "agreeable to all",
    "no horse in that race", "nothing in my life hinges", "go ahead and decide",
    "decide for me", "equally happy with any", "any option there would suit",
    "would suit me", "content with anything", "leave that one open",
    "pick any card", "any card from the deck", "never lost sleep over",
)
MUDDLE_HINTS = (
    "lost me", "once more", "give it me again", "give it to me again", "till float", "didn't land",
    "did not land", "all at sea", "souff", "saturday lad", "have another go", "come again",
    "simpler", "another way", "one ear", "struggling with that last bit", "try it again", "try again",
    "another go", "not sunk", "none the wiser", "say it again", "repeat that", "repeat it",
    "not following", "didn't follow", "did not follow", "can't follow", "cannot follow",
    "what are you asking", "not sinking in", "give us it once more", "you've lost me",
    # Race13 shopper dialect
    "everyday words", "too much jargon", "what were you getting at", "first-year",
    "instant coffee", "last bit again", "straight through me", "hit me with it again",
    "gentler", "dropped chisel", "what did you actually mean", "wet dog", "what was the point",
    "came up empty", "explain it again", "pencil behind", "stuck on your last", "loosen it up",
    "green timber", "plain talk", "second clause", "quite muddled", "return to your previous",
    "tea went cold", "half the steps missing", "what exactly do you mean", "not a word of it stuck",
    "bleep of the monitors", "am i right in thinking you asked", "could not catch it",
    "put that in simpler words", "head is still on the ward",
    # R17 muddle dialect (card/deck metaphors that are NOT waives)
    "could not tell what you were asking", "need it explained again", "completely adrift",
    "send that again", "everyday language", "hand of unfamiliar cards", "shuffled deck right now",
    "lost you somewhere", "take another pass", "did not help sort it",
)
BATCH_HINTS = (
    "one by one", "one at a time", "just one thing", "just one item", "single ask", "one ask",
    "lone question", "split those up", "space them out", "ask me less", "bundle of questions",
    "crowd of asks", "one after another", "hold the rest back", "smaller bites", "dribs and drabs",
    "too many pieces on the bench", "pop just the one query", "only one query", "first question",
    # Race13 shopper dialect
    "back pocket", "just give me the first", "stacked three questions", "pick the first",
    "let me answer it before the rest", "each query by itself", "first query and then the second",
    "kindly ask me about one matter", "one matter, then another", "ward phone keeps ringing",
)


def scoop_budget(blob: str) -> float | None:
    found = re.search(r"(\d+(?:\.\d+)?)\s*cents?\s+of\s+a\s+dollar", (blob or "").lower())
    if not found:
        return None
    try:
        return float(found.group(1))
    except Exception:
        return None


def scoop_waive(blob: str) -> bool:
    low = (blob or "").lower()
    if not low.strip():
        return False
    if re.search(r"\d", low) and scoop_budget(blob) is None:
        if not any(p in low for p in WAIVE_HINTS):
            return False
    return any(p in low for p in WAIVE_HINTS)


def scoop_muddle(blob: str) -> bool:
    low = (blob or "").lower()
    if re.search(r"\b(?:at least|at most|must|require|same as|i want|i need)\b", low) and re.search(r"\d|[a-z]{3,}", low):
        if any(p in low for p in ("at least", "at most", "same as", "i want", "i need")) and not any(p in low for p in MUDDLE_HINTS):
            return False
    return any(p in low for p in MUDDLE_HINTS)



PROCESS_PHRASES = (
    'tell me the total', 'tell me what is in', 'before you order', 'before you finish', 'before i finish',
    'your total', 'cart total', 'quote the total', 'confirm the total', 'what is the total',
    'how much is the', 'how much will', 'what do i owe', 'your basket', 'the basket',
    'show me the basket', 'confirm the basket', 'tell me the basket', 'read the total back',
    'say the total', 'repeat the total', 'list what is in the cart', 'list the cart',
    'confirm before', 'get my approval', 'ask me to confirm', 'check with me before',
    'run it by me', 'show me what you will buy', 'tell me what you will order',
    'account for the basket', 'account of the basket', 'final basket', 'before placing',
    # R17 process gap vs Cairn: broader account/approval wording in requests
    'before checking out', 'before checkout', 'before you check out', 'before purchasing',
    'tell me the cart', 'read back the', 'what you are about to', 'what you are going to buy',
    'double-check with me', 'double check with me', 'ok it with me', 'okay it with me',
    'get the go-ahead', 'go ahead from me', 'my say-so', 'my say so', 'sign off',
    'recap the order', 'recap before', 'summary of the basket', 'summary before',
)

def plain_reask(product: str, label: str, examples: list[str] | None=None) -> str:
    """One short, everyday re-ask of a single property after muddle/crowded."""
    base = f'In plain words: for the {product}, which {label} do you want'
    if examples and len(examples) >= 2:
        return base + f' — {examples[0]} or {examples[1]}?'
    if examples and examples[0]:
        return base + f' (for example {examples[0]})?'
    return base + '?'

def scoop_batch(blob: str) -> bool:
    low = (blob or "").lower()
    return bool(low.strip()) and any(p in low for p in BATCH_HINTS)


def scoop_trim(val: str) -> str:
    val = val.split(",")[0]
    val = re.split(r"\s+(?:like a|no two ways|plain and|because|and that(?:'s| is)|and the rest|or i(?:'ll| will)|full stop)\b", val)[0]
    return val.strip(" .")


def scoop_asked_prop(question: str) -> str:
    low = (question or "").lower()
    found = re.search(r"\bwhat\s+(.+?)\s+would you like\b", low)
    if not found:
        found = re.search(r"\bwhich\s+(.+?)\s+would you like\b", low)
    if not found:
        found = re.search(r"\bwhat\s+(.+?)\s+do you need\b", low)
    if not found:
        return ""
    prop = re.sub(r"\s+", " ", found.group(1)).strip(" ?.")
    prop = re.sub(r"^(?:a|an|the|your)\s+", "", prop)
    return prop.split("(")[0].strip()[:48]


def scoop_hard(blob: str) -> list[str]:
    low = (blob or "").lower().replace("?", " ")
    if not low.strip():
        return []
    if any(w in low for w in ("the page", "listing", "sold by", "shown on", "whatever's on the page", "whatever the page")):
        return scoop_pointer(blob)
    found: list[str] = []

    def keep(prop: str, val: str) -> None:
        prop = re.sub(r"\s+", " ", (prop or "").strip().lower())
        if " of the " in prop:
            prop = prop.split(" of the ")[0].strip()
        val = scoop_trim(val)
        if not prop or not val or len(val) > 40 or len(found) >= 2:
            return
        if any(old.startswith(prop + ":") for old in found):
            return
        found.append(f"{prop}: {val}")

    patterns = [
        (r"\bregarding\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s*,\s*i(?:'m| am) sticking to\s+([^,]{1,48})", 1, 2),
        # Race13: "Regarding style of the X — I want modern and nothing sanded down..."
        (r"\bregarding\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s*[—\-–,]\s*i want\s+(?:a\s+)?([^,]{1,48}?)\s+and nothing sanded", 1, 2),
        (r"\bregarding\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s*[—\-–,]\s*i want\s+(?:a\s+)?([^,]{1,48})", 1, 2),
        # Race13: "Square and true — that's how I want occasion — meaning sport and no other."
        (r"\bsquare and true\s*[—\-–,].*?how i want\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,40}?)?\s*[—\-–,]\s*meaning\s+([^,]{1,48}?)\s+and no other", 1, 2),
        (r"\bsquare and true\s*[—\-–,].*?how i want\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})\s*[—\-–,]\s*meaning\s+([^,]{1,48})", 1, 2),
        # Race13: "display size has got to match at least 2 inch — the rest..."
        (r"\b([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+has got to match\s+([^,]{1,48}?)\s*(?:—|,|$)", 1, 2),
        # Race13: "Pencil it in permanently — protein source of the X equals whey."
        (r"\bpencil it in permanently\s*[—\-–,]\s*([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+equals\s+([^,]{1,48})", 1, 2),
        (r"\bi just need the\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+to come in as\s+([^,]{1,48})", 1, 2),
        (r"\bi(?:'m| am) holding out for\s+([^,]{1,40}?)\s+on the\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})", 2, 1),
        (r"\banything with\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+apart from\s+([^,]{1,48}?)\s+gets binned\b", 1, 2),
        (r"\bi(?:'m| am) just not shifting on the\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s*,\s*so i(?:'ll| will) be sticking with\s+([^,]{1,48})", 1, 2),
        (r"\bi(?:'ve| have) got to have my\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+sorted out as\s+([^,]{1,48})", 1, 2),
        (r"\bmy\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+wants to be\s+([^,]{1,48})", 1, 2),
        (r"\bi(?:'ll| will) stand firm on\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,40}?)?\s*,\s*and that means\s+([^,]{1,48})", 1, 2),
        (r"\bthe\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+has got to land on\s+([^,]{1,48})", 1, 2),
        (r"\babout\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,4})\s*,\s*my mind is set on\s+([^,]{1,40})", 1, 2),
        (r"\bonly looking at options with\s+((?:at least|at most|no more than|no less than)\s+[^,]{1,40}?)\s+showing under\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})", 2, 1),
        (r"\b([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+wants to come in as\s+([^,]{1,40})", 1, 2),
        (r"\b([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,50}?)?\s+has got to come as\s+([^,]{1,40})", 1, 2),
        (r"\bsticking to\s+([^,]{1,32}?)\s+when it comes to\s+([a-z]+(?:\s+[a-z]+){0,4})", 2, 1),
        (r"\bi(?:'ll| will) only take\s+([^,]{1,40}?)\s+(?:for|on|as)\s+(?:the\s+)?([a-z0-9]+(?:\s+[a-z0-9]+){0,5})", 2, 1),
        (r"\bwhen it comes to\s+([a-z0-9]+(?:\s+[a-z0-9]+){0,5})\s*,\s*i(?:'ll| will) only take\s+([^,]{1,40})", 1, 2),
        # "Please don't think me rude, but color of the X set to gold is all I'll accept."
        (r"\b([a-z0-9]+(?:\s+[a-z0-9]+){0,5})(?:\s+of the\s+.{0,60}?)?\s+set to\s+([^,]{1,40}?)\s+is all i(?:'ll| will) accept", 1, 2),
    ]
    for pat, pi, vi in patterns:
        for match in re.finditer(pat, low):
            prop, val = match.group(pi), match.group(vi)
            if "page" in val or "listing" in val or "whatever" in val:
                continue
            keep(prop, val)
            if found:
                return found
    return found


def scoop_pointer(blob: str) -> list[str]:
    low = (blob or "").lower().replace("?", " ")
    if not any(w in low for w in ("the page", "listing", "sold by", "shown on", "whatever's on the page", "whatever the page", "works alongside", "mates up with")):
        return []
    point = (
        r"(?:just the same as |the same as |the type that works alongside |a type that works alongside |"
        r"a type that mates up with |mates up with |set apart from |anything other than |definitely not |"
        r"within\s+[^,]{0,40}?\s+either way of |no higher than )?"
        r"(?:whatever's on the page|whatever the page|what's given on the listing|the value shown on the page|"
        r"the one on the page|the details on the listing).{0,200}"
    )
    found: list[str] = []
    for pat in (
        rf"\bi just need the\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,60}}?)?\s+to come in as\s+({point})",
        rf"\bi(?:'ve| have) got to have my\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,60}}?)?\s+sorted out as\s+({point})",
        rf"\bthe\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,60}}?)?\s+has got to land on\s+({point})",
        rf"\banything with\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,60}}?)?\s+apart from\s+({point})",
        rf"\b([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,60}}?)?\s+(?:come in as|coming in as|wants to come in as)\s+({point})",
        # Race13: "Regarding style — I want a type that mates up with whatever's on the page for ..."
        rf"\bregarding\s+([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,40}}?)?\s*[—\-–,]\s*i want\s+({point})",
        # Race13: "gender has got to match definitely not whatever's on the page for ..."
        rf"\b([a-z0-9]+(?:\s+[a-z0-9]+){{0,5}})(?:\s+of the\s+.{{0,60}}?)?\s+has got to match\s+({point})",
    ):
        for match in re.finditer(pat, low):
            prop = match.group(1).split(" of the ")[0].strip()
            val = match.group(2).strip(" .,")[:180]
            if prop and val:
                found.append(f"{prop}: {val}")
                return found
    return found


MODEL_FAST = 'anthropic/claude-haiku-5.5'
MODEL_DEEP = 'anthropic/claude-sonnet-5.5'
MODEL_FALLBACK = 'openai/gpt-6-luna'
TURN_BUDGET = 42
CALL_CEILING = 60
BUDGET_SOFT = 0.055
BUDGET_HARD = 0.078
SPEND_STOP = 0.085
SPEND_LIGHT = 0.035
DEEP_ROOM = 0.02
DEADLINE = 295.0            # champs use ~300s; 250 starved late process+ask under packed flips
CLOSING = 70.0
LAST_CALL = 28.0
QUESTION_TIME = 45.0
LATE_ASK_STEPS = 8          # need this many steps left to risk one late question before buy
LATE_ASK_SECS = 40.0        # wall room above CLOSING (was 55; clock now nearer 300)
LIST_SIZE = 8               # Cairn uses 12; ramp carefully so dialect crowded still flips to solo
LIST_MIXED = 4
LIST_STEPS = (1, 3)         # first asks stay narrow; then open the list
LIST_STEPS_MIXED = (2, 3)
RECOVERY_STEPS = 8          # YesYesYes-style pre-decline hunt budget
RETRY_GAPS = (4, 10, 20, 35)
RETRY_BUDGET_S = 45
VIEW_CHARS = 450
OBS_KEEP = 15000
CHAT_KEEP = 150000
FINISH_TRIES = 4
PAGE_SWEEP = 8
CHEAP_PROBES = 4
_llm = ProxyClient(timeout=50, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=95, max_retries=3, retry_delay=1)
PLAYBOOK = 'You are World — a precise purchasing agent for one shopper in this store. You can act only by calling the store\'s tools. The episode is graded 1 or 0: it earns 1 only when the finish is exactly right. A purchase must contain every requested item at the requested quantity, each meeting every requirement in force when you check out, inside the budget; or, when the store truly has nothing that meets everything, the episode must end with the no-purchase tool in the way the request asks.\n\nWORKING RULES\n- Every tool call is one step and steps are limited; thinking costs nothing. Tool names, argument names and result shapes are particular to this store: read each tool\'s description and schema and use the exact argument names and allowed values.\n- Search tools can be narrow: a category filter, a keyword mode that may only read the first words of the query, a page size of a few rows, a token or offset for the next page. Compare the reported total with the rows you have seen and page on (with the exact token or offset the store gives) when the answer depends on seeing all of them. Keyword queries work best short and value-first (the wanted value plus the product noun). A filtered search that returns no rows usually means the filter value or the filter parameter is wrong: retry with the label exactly as result rows show it, or with another filter parameter.\n- Search rows are summaries. Stores split a listing\'s record differently: one view can show its attributes (properties), another only its offer and description, and a side-by-side view may carry either. The ledger names the tool that shows attributes once World knows it. Judge every requirement on a property from those attributes, so read them for each listing you may buy; one call that shows several listings saves steps when it shows what you need. Never open a listing again when nothing has changed since you read it.\n- Questions to the shopper are handled for you by World\'s interviewer, so you never message the shopper yourself. Its answers arrive in the conversation and are added to the ledger.\n- The request can also ask for steps along the way: an approval, a message, something to check or read first. The ledger lists them. World carries out the ones due just before the order or the ending itself when you call the checkout or the ending tool, so never wait for those and never end without a purchase because of them; do any others with the store\'s tools (product questions go through World\'s interviewer).\n\nFIT RULES\n- The kind of product is decided by the listing\'s category label: only the label that matches the requested item as a whole counts, never a neighbouring one.\n- A requirement on a property is met when the listing\'s attribute for that property states the value. Key spelling does not matter (spaces, underscores, camel case, prefixes), and one element of a comma or slash list is enough. Values match by their words: case, singular or plural and plain synonyms are fine, and a value that contains the required value\'s words counts ("deep teal" meets teal, "extra roomy fit" meets roomy). A value with different words does not count (an abbreviation or code is not the word it stands for, and "teal green" is not "sea green"). Units are converted before comparing numbers, limits include the edge value, and a range must lie wholly inside a limit.\n- An attribute whose key names the requirement\'s property (under that name or a plain synonym) decides: if it says something else, the listing fails whatever other keys or the title say. A key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing about the requirement. World adds [evidence] lines after opened listings with the keys whose names match a requirement.\n- When the listing has no attribute for that property, the wanted value stated under another attribute (for example a list of styles or features that includes it) or in the title or description counts. A number of another kind does not (a weight is not a capacity), nor a bare figure whose text does not say what it measures.\n- When the category label itself states a requirement\'s value (a season or audience in the label), that requirement is met by the category.\n- A requirement can point at another listing instead of giving a value ("the same X as the listing sold by <shop> with <property>", "within N of the X on the page of ..."). Resolve it first: find that listing (search its category, brand, shop or words), open its page in full with the single-item or side-by-side tool (a search row is not enough), and when that page refers you to a further listing for the property, find and open that one the same way; the last page gives the value. The order counts only when every page of that chain was opened. When the wanted value is anything but that other listing\'s value, the rule excludes only listings sharing that exact value, and never end without a purchase because of it alone.\n- When a listing you are about to buy sends you to another page for more about it, look at that page before you buy.\n- When one requested item must fit another (an accessory for a device), choose the two together from the listings\' own words (a list of what it fits, a model or series code that both pages carry). When no accessory that meets its own requirements fits the device you picked, switch to another device that meets its requirements and that such an accessory fits; never end without a purchase while that switch is untried.\n- Respect exclusions and closed lists. Wishes ("ideally", "would be nice") are tie-breakers only.\n- Never buy an item that the store shows as sold out; the total (price times quantity, summed over every line) may not exceed the budget; equal is fine.\n\nSELECTION\n- When the request asks for the cheapest (lowest total, smallest price), the order must hold the lowest-priced listing among ALL listings that fit. Find the whole fitting set (page through it: result rows are often in no price order, and World reads the next pages of a short result itself), check candidates in ascending price, and take the first that fully fits. When a search takes a maximum price, use it: list the category below the price of your current best pick, so only cheaper candidates are left to check.\n- When the request asks for the most or least of some measure, compare every fitting listing on that measure and take the extreme; a listing that does not state the measure cannot win.\n- Otherwise any listing that meets everything is right: do not waste steps hunting for something better.\n- Cart each item as soon as you have opened a listing that meets every requirement for it (with a cheapest, most or least pick: once you have compared the fitting set), then move on. Stores often sell out a carted line soon after it is added, and an early cart leaves steps to replace it.\n- A request for several different items needs one cart line per item, each at its requested quantity.\n\nMID-RUN SHIFTS\n- The shopper or the store can change things: a new budget, a requirement added or dropped, a price change, a listing sold out. They appear inside tool results, often when you try to check out (a refusal); a note counts whichever tool\'s result shows it, and the ledger lists the requirements in force. The episode goes on after a refusal: apply every change, fix the cart and check out again.\n- Re-read a chosen listing\'s record shortly before checking out, because price and stock can change silently.\n\nCLOSING\n- Put exactly the chosen lines in the cart (remove anything else), look at the cart once, then check out with a call on its own.\n- End without a purchase only after a real search finds no listing that meets everything. Running short of steps proves nothing of the kind: then check out the best cart you have. Follow the request\'s wording for that ending, and when a remark is needed, keep it short and name no listing.\n- Every reply must call at least one tool. Keep your text short: what you know, then the next move.'
BRIEFING = 'You prepare a shopping task for an agent. Read the shopper\'s request and the store\'s tools and return ONE JSON object (compact, no indentation, nothing else) with these keys:\n\n"items": a list with one entry per distinct product the shopper wants bought, each {"name": the kind of product in the request\'s words, "shelf": the store\'s category label for it when a tool lists labels (copy it exactly), else "", "quantity": units of that one listing (integer, 1 when not stated; a pair or a set that is one listing counts as 1), "musts": a list of firm requirements on that product, each "property: constraint" with the exact value, limit, exclusion or closed list. Name the property with the request\'s own word for it ("meant for a lakeside occasion" gives "occasion: lakeside", "a 12-inch blade" gives "blade length: 12 inch"); do not swap it for a neighbouring property. A must whose value is read from another listing is ONE must holding the whole description of that listing ("rim colour: unlike the rim colour of the kettle Ovarno sells with a glass lid"): words that describe that other listing (its shop, its kind, "with <property> <value> and <property> <value>" after it) stay inside that one must and are never musts of this product}. Never put price, budget, stock or quantity into musts. The story around the request (the person meant to use it, where it will stand, what went wrong before, the occasion behind it) is context, not a must, unless it states a property\'s value or limit outright; and never turn words that are part of what the product is called (a season or an audience named with the product) into musts: those belong to "name". A use or occasion the request makes a property of the product itself ("blazers for business wear", "a dress made for parties", "the occasion has to be party") IS a must on that property ("occasion: business"), unless the store\'s category label already names it; an event, person or setting the product is bought for ("to wear at his wedding", "for a colleague"), and how it should sit, feel or look in use, stay context. A product the request names in one phrase joined by "and" (one "I want ..." with one quantity, often the name of one store department) is ONE item; make separate items only when the request gives each product its own quantity or its own requirements.\n"budget": the most the shopper will pay in total as a number in the request\'s own unit, or null.\n"budget_unit": that unit in words (for example "cents"), or null.\n"budget_scope": "total" when the limit covers the whole order, "each" when it is per item, null when there is no limit.\n"in_stock_only": true unless the request clearly accepts items that are not available.\n"pick": "cheapest" when the request asks for the lowest price or smallest total, "most" or "least" when it asks for the extreme of some measure, else "any". Remarks that money is tight are not a pick rule.\n"pick_measure": what is maximised or minimised for "most"/"least", else "".\n"messages": how many messages the shopper allows the agent to send (integer), or null when not stated.\n"steps": how many steps, moves or tool calls the request allows (integer), or null.\n"lookups": a list of the musts (copied exactly) whose value must first be read from another listing the request points at, else [].\n"rules": a list of every instruction the request gives about how to work or how to end, each in at most 12 words.\n"wishes": a list of things the shopper would merely like.\n"ask_first": a list of things the request explicitly says to check with the shopper before choosing (empty when none).\n"process": a list of the steps the request requires the agent to take besides finding and buying the right items (getting the shopper\'s approval, telling or showing the shopper something, checking something with them, reading something first, using a particular tool), each {"step": what must be done, in at most 15 words, "when": "choose" (before the products are chosen), "order" (right before checking out) or "end" (just before ending without a purchase), "messages": how many messages to the shopper it takes (0 when none)}; [] when the request asks for no such step. Requirements on the items, the budget, stock, the limits on messages and steps, and how to end when nothing fits are not steps.\n"quiet_decline": true when the request says not to suggest or name alternatives when nothing fits, false when it asks for them, null when it says nothing.\n"fits": a list of {"item": item number, "with": item number} for each requested item that the request says must work with, fit or suit ANOTHER requested item (the one the agent picks); [] when none.\n"roles": an object naming the exact tool for each job, or null when no tool does it: "search", "open" (shows one listing\'s attributes or properties; when two single-listing tools exist, the one whose description promises attributes, properties or specifications), "offer" (shows one listing\'s current offer, such as its price and stock, when that is a different tool from "open"), "compare" (several listings in full), "ask" (sends a message to the shopper), "cart" (changes the cart), "cart_view" (shows the cart), "buy" (checks out), "refuse" (ends without a purchase).\n"buy_args": the fixed arguments that make the "buy" tool check out (for a tool with a mode argument, that mode), else {}.\n"refuse_args": the fixed arguments that make the "refuse" tool end without a purchase, else {}.\n'
INTAKE_PLAN = 'You choose the questions a shopping agent sends to a shopper. Shoppers hold a few firm preferences they did not write down, each about one property of one requested product. A question reveals a preference only when it names that exact property; a property the shopper has no view on gets "don\'t mind", and one the shopper does not think about at all gets a puzzled reply, so every property named spends part of a message.\n\nYou get the request, the items with what the request already settles for each, and a numbered list of attribute keys seen on opened listings of each item\'s category, with how many of those listings carry the key and example values. An unwritten preference may be on a property that no listed key names.\n\nChoose the properties in this order:\n0. What the request itself asks you to check with the shopper (listed under ASK FIRST when there is any), each named by the property it is about.\n1. Hinted properties. The request\'s story often points at an unwritten preference without giving its value: a sentence about something the shopper cannot judge or leaves to you, someone whose taste decides, the product\'s looks or the setting it belongs in, the person meant to use or wear it, or what is inside it. For each such sentence, name the one property it points at and the item it is about, in the order the story gives them.\nName a hinted property by a listed key of that item when one fits the hint (a hint about who will wear it: the key for gender or audience; about its looks: the listed key for colour, pattern or finish that most of its listings carry); a property no listing of the item carries comes after the item\'s three most carried keys.\n2. Then the listed keys, ranked: favour the properties a buyer of that kind of product chooses between first (often its type or style, its form or fit, its size, capacity or dimensions, its material, colour or pattern, whichever that kind of product is bought by) and keys carried by most of the item\'s listings; once five or more listings were read, avoid keys seen on only one or two of them.\nSkip a property the request already settles for that item (a requirement names that property; a key whose values only repeat a required value, such as a pattern key showing the required decoration, is NOT settled), a second name for a property already chosen, a key whose values each list several purchase options at once, identifiers, model names, warranty and other logistics. Pack sizes, counts and quantities per pack are properties like any other, and a key marked [not the stated ...] is another property than that requirement, whatever words its name shares with it. Rank every item on its own: its hinted properties first, then its listed keys.\n"label" is the property\'s plain name: for a listed key, all of the key\'s words in plain spelling ("lid colour" for lidColour, "rim finish" for rim_finish; never shortened to "colour" or "finish"); only the property\'s name, no example values, no brackets, no second property.\nChoose up to COUNT for each item, best first. Return ONE JSON object and nothing else: {"ask": [{"item": item number, "label": "...", "hint": "at most 8 of the story\'s words it comes from, or an empty string"}]}, each item\'s picks in rank order'
REPLY_READ = 'A shopping agent asked a shopper some questions. Read each answer and return ONE JSON object and nothing else:\n{"needs": [{"item": item number, "need": "property: constraint", "points": true when the value is to be read from another listing, else false}], "drop": [{"item": item number, "property": "a requirement the shopper now gives up"}], "free": [{"item": item number, "property": "..."}], "crowded": [numbers of the questions whose answer complains that the message asked about too many things or wants a single question; not answers that only say they did not follow it], "budget": new total limit as a number or null}\n- "needs": every firm preference an answer states, with the exact value or limit and its unit ("at least 30 cm", "oval", "not wicker"). Name the need by the property the answer itself names, which may differ from the one asked about; several answers about one property make one need. Copy the value word for word as the answer says it, keeping its nouns and units ("at least 3 shelves", not "at least 3").\n- An answer can give the value by pointing at another listing (a value like, unlike or near the one on some shop\'s product, that product picked out by a few of its properties): that is ONE need on the property the answer names, with "points": true. Its value is the answer\'s own words copied exactly, from the comparison words to the end of that description, including what it says about a listing that page may name; never shorten or reword it. The properties describing the other product are not needs of this item.\n- "free": the property of each question the shopper said they do not mind about.\n- Folksy or roundabout wording that names a value still states that value.'
NOTE_READ = 'While a shopping agent works, the store showed this note (it may come from the shopper or the store). Return ONE JSON object and nothing else:\n{"kind": "change" when the note changes the task (budget, requirements, a price, availability, a refusal of the order), "error" when it only reports that a call was malformed or not accepted, "other" for anything else, "budget": the new total limit as a number in the request\'s unit, or null, "add": [{"item": item number, "need": "property: constraint"}], "drop": [{"item": item number, "need": "the requirement no longer wanted"}], "gone": [listing ids said to be unavailable], "sold_out": true when it says a chosen listing is gone, sold out or no longer available (even without naming it), "price_moved": true when it says a price changed, "summary": "one short sentence"}\nA replaced requirement is a drop of the old one plus an add of the new one. Report every change in the note.'
CART_JUDGE = 'You are the last check before a shopping agent checks out. Decide from the records shown whether this cart is exactly right.\nReading rules: a requirement is met when the listing\'s attribute for that property states the value (any key spelling, one element of a list is enough, case, singular/plural and plain synonyms count, a value containing the required value\'s words counts, such as "deep teal" for teal, units converted, limits include the edge, a range must lie wholly inside); an attribute whose key names the same property as the requirement decides, and if it says something else the listing fails; a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when the listing has no attribute for that property, the wanted value stated under another attribute or in the title or description counts (a number of another kind does not: a weight is not a capacity; nor does a bare figure whose text does not say what it measures); a requirement the category label itself states is met by that category; the category label must match the requested item as a whole; a rule of being unlike another listing\'s value counts as met unless the records show that same value. Each requested item must appear as a separate cart line at the requested quantity, nothing else may be in the cart, nothing shown out of stock, and the total of price times quantity over all lines must be within the budget. The newest record of a listing wins.\nWhen the pick rule is cheapest, most or least, also look at the other listings shown: if one of them clearly fits every requirement and beats the cart\'s choice on that rule, the cart is wrong.\nReturn ONE JSON object and nothing else: {"ok": true or false, "problems": ["short, concrete problem", ...], "better": [listing ids that should replace a cart line, best first]}. Use ok=false only for a problem the records clearly show or a requirement no record addresses.'
READ_RULES = 'the listing\'s attribute for a property decides when its key names that same property (any key spelling, one element of a list is enough, synonyms and unit conversions count, a value containing the required value\'s words counts, such as "deep teal" for teal, limits include the edge); a key for a different property, even a related one (a handle colour is not the body\'s colour), decides nothing; when no attribute names that property, the wanted value stated under another attribute or in the title or description counts (a bare figure that does not say what it measures counts for nothing); a requirement the category label itself states is met by that category; the category label must match the item; a rule of being unlike another listing\'s value counts as met unless the records show that same value; out-of-stock listings and listings that break the budget do not fit.'
AUDIT_PROMPT = 'A shopping agent wants to end without buying because it thinks nothing fits. Check the listing records shown against the requirements, with these reading rules: ' + READ_RULES + '\nRecords marked ROW were seen only as a search result row: their page (the attributes) has not been read, so they can be neither passed nor failed on a property the row does not show.\nReturn ONE JSON object and nothing else: {"fits": {"item number": [listing ids that appear to meet every requirement, cheapest first, at most 4]}, "open": [ids of ROW listings of a requested kind, inside the budget, whose page could show that they fit, most promising first, at most 6], "reason": "one sentence"}'
UNMIX_PROMPT = 'A shopper\'s request was read into firm requirements on a product. Some requirements take their value from another listing the request points at, and the words that pick out that other listing (its kind, its shop, the properties named after it with "with ...") are sometimes read by mistake as requirements on the product being bought. Using the request\'s wording, say which of the OTHER REQUIREMENTS below only describe that other listing. Return ONE JSON object and nothing else: {"describe_other": [those requirements copied exactly; empty when every one is a real requirement on the product]}'
LINK_TRAIL = 'A shopping agent is about to check out. Some requirements take their value from another listing instead of stating it, and a listing\'s page can send the reader on to a further listing for that property. The order counts only when the agent has opened in full every listing along each such chain. Follow each requirement\'s chain from the requirement\'s own description, link by link, through the pages\' own words, using the records below (OPENED: the agent has opened that page; ROW: seen only as a search result row, its page not read).\nReturn ONE JSON object and nothing else: {"chains": [{"need": the requirement copied exactly, "links": [{"id": the listing id when a record below is that link, else "", "about": how the requirement or the previous page describes that listing, "find": "for a link not found: search words (its shop and the values the description gives)"}], "done": true when the last link is an OPENED page that states the property\'s value itself (also when the requirement asks for a value other than or unlike that one: the page\'s value is what it is measured against), else false, "value": the property\'s value as stated on that last page, or ""}]}\nA record is a link only when it matches the whole description of that link: the shop or brand, the kind of product AND every property value the description gives (for example "a 10 pieces multipack" rules out the same shop\'s listing with 10 capsules and no multipack); when no record matches all of it, that link is not found yet, even if the same shop sells other listings of that kind. Give "find" for such a link: the shop\'s name and the description\'s values, as a few search words.\nList each chain\'s links in order and stop at the first link whose page is not OPENED or not found, since its words are still unknown.'
FINISH_STEPS = 'A shopping agent is about to STAGE. The shopper\'s request asks for the steps below along the way. Decide from the numbered RECORD of what the agent has done whether each step due before this STAGE is done for the cart as it is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or the order, that line comes after the cart last changed and its reply grants it; questions on a product\'s properties do not do a step.\nWhen a step is still missing, give the one store call that does it next: a tool from TOOLS with its arguments. A message to the shopper goes through the messaging tool: one short sentence or question about that step alone, in plain words, naming no product properties or values the step does not need. Give no call when every due step is done or when the messages or steps left cannot do the missing one.\nReturn ONE JSON object and nothing else: {"steps": [{"step": the step, "done": true or false, "lines": [the RECORD line numbers that show it done]}], "call": {"tool": "...", "args": {...}} or null, "hold": what a reply asks to change before going on, or "", "needs": [{"item": item number, "need": "property: constraint"} for each firm requirement such a reply states]}'
SWAP_HINT = "A listing in a shopping agent's cart has just sold out. Find replacements for that cart line among the other listing records shown, with these reading rules: " + READ_RULES + '\nReturn ONE JSON object and nothing else: {"fits": [{"id": listing id, "price": its price as a number as the record shows it} for every listing that appears to meet every requirement of the sold-out line\'s item, at most 6], "reason": "one sentence"}'
SHAPE_PROMPT = 'A store tool returned the result below. Name its parts by the exact keys the result and the tool\'s parameters use. Return ONE JSON object and nothing else: {"rows": the key of the list of listings, "count": the key holding how many listings matched in all (not how many are shown), or null, "next": the key holding the value that asks for the following page, or null, "next_arg": the tool parameter that takes that value, or null, "price": the key, inside a row, of the listing\'s price, or null, "stock": the key, inside a row, that says whether the listing can be bought now, or null}'
KEY_GUARD = 3
SHARE_GUARD = 0.0
GAP_GUARD = 0.34
SKIP_PROP_RE = '\\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\\w*|video|url|links?|images?|photos?|condition)\\b'
CATCHALL_RE = '^(?!.*\\b(?:types?|styles?|kinds?|shapes?)\\b).*\\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\\w*|accessor\\w*|extras?|misc\\w*)\\b'
SEED_EACH = 6
FIT_CODE_RE = '[A-Z][A-Za-z]{0,5} ?\\d{3,6}'
PART_CODE_RE = '[A-Z]{1,4}\\d{3,6}'

def scan_codes(text: Any) -> list[str]:
    return [re.sub('\\s+', ' ', c).strip().lower() for c in re.findall('\\b' + FIT_CODE_RE + '\\b', str(text or ''))]

def scan_fit_marks(text: Any) -> dict[str, Any]:
    text = str(text or '')
    own = re.search('[\\[(]\\s*(?:part\\s+)?(' + PART_CODE_RE + ')\\s*[\\])]', text, re.I)
    codes: list[str] = []
    for hit in re.finditer('\\bmodel\\s*(?:no\\.?|number|#)\\s*[:.]?\\s*(' + FIT_CODE_RE + ')\\b', text, re.I):
        codes += scan_codes(hit.group(1))
    for hit in re.finditer('\\b(?:also sold as|also known as|sold also as)\\s+([^.;]+)', text, re.I):
        codes += scan_codes(hit.group(1))
    works: list[str] = []
    for hit in re.finditer('\\b(?<!not )(?<!never )(?:works with|compatible with|fits)\\s*:?\\s+([^.;]+)', text, re.I):
        if not re.search('\\bpart\\s+' + PART_CODE_RE + '\\s+also\\s+$', text[max(0, hit.start() - 40):hit.start()], re.I):
            works += scan_codes(hit.group(1))
    also = [(hit.group(1).upper(), scan_codes(hit.group(2))) for hit in re.finditer('\\bpart\\s+(' + PART_CODE_RE + ')\\s+also fits\\s*:?\\s*([^.;]+)', text, re.I)]
    more = re.search('\\b(?:more|details|see|read)\\s+(?:on|at|under|about)?\\s*(?:the\\s+)?part\\s+(' + PART_CODE_RE + ')', text, re.I)
    return {'own': own.group(1).upper() if own else '', 'codes': codes, 'works': works, 'also': also, 'more': more.group(1).upper() if more else ''}

def jdump(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    except (TypeError, ValueError):
        return str(value)

def trim_tree(value: Any, limit: int, depth: int=0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: trim_tree(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [trim_tree(v, limit, depth + 1) for v in value]
    return value

def as_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    text = text.strip()
    fence = re.match('^```[A-Za-z]*\\s*(.*?)\\s*```$', text, re.S)
    if fence:
        text = fence.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    left, right = (text.find('{'), text.rfind('}'))
    if 0 <= left < right:
        try:
            got = json.loads(text[left:right + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None

def canon_key(key: str) -> str:
    key = re.sub('([a-z0-9])([A-Z])', '\\1 \\2', str(key))
    return re.sub('[\\s_.\\-]+', ' ', key).strip().lower()

def word_bag(text: str) -> frozenset[str]:
    return frozenset((w[:-1] if len(w) > 3 and w.endswith('s') else w for w in canon_key(text).split()))

def root_bag(text: str) -> set[str]:
    out = set()
    for w in canon_key(text).split():
        w = w[:-2] if re.search('(?:ch|sh|ss|x|z)es$', w) else w[:-1] if len(w) > 3 and w.endswith('s') and (not w.endswith('ss')) else w
        if len(w) >= 3 and w not in ('and', 'for', 'the', 'with'):
            out.add(w)
    return out

def keys_overlap(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)

def schema_kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]

def value_matches(value: Any, kind: str) -> bool:
    if kind == 'string':
        return isinstance(value, str)
    if kind == 'integer':
        return isinstance(value, int) and (not isinstance(value, bool))
    if kind == 'number':
        return isinstance(value, (int, float)) and (not isinstance(value, bool))
    if kind == 'boolean':
        return isinstance(value, bool)
    if kind == 'array':
        return isinstance(value, list)
    if kind == 'object':
        return isinstance(value, dict)
    if kind == 'null':
        return value is None
    return True

def coerce_args(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    if not isinstance(schema, dict):
        return value
    kinds = schema_kinds(schema)
    if kinds and (not any((value_matches(value, k) for k in kinds))):
        for kind in kinds:
            if kind == 'integer' and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch('\\s*-?\\d+(\\.\\d+)?\\s*', value):
                number = float(value)
                value = int(number) if kind == 'integer' or number.is_integer() else number
                break
            if kind == 'string' and isinstance(value, (int, float)) and (not isinstance(value, bool)):
                value = str(value)
                break
            if kind == 'boolean' and isinstance(value, str) and (value.strip().lower() in ('true', 'false')):
                value = value.strip().lower() == 'true'
                break
            if kind == 'array' and (not isinstance(value, list)) and (value is not None):
                value = [value]
                break
        if not any((value_matches(value, k) for k in kinds)):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            match = [o for o in options if isinstance(o, str) and o.lower() == low]
            if match:
                value = match[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str) and isinstance(schema.get('maxLength'), int) and (len(value) > schema['maxLength']):
        value = value[:schema['maxLength']]
    if isinstance(value, str) and isinstance(schema.get('minLength'), int) and (len(value) < schema['minLength']):
        faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and (not isinstance(value, bool)):
        if isinstance(schema.get('minimum'), (int, float)) and value < schema['minimum']:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get('maximum'), (int, float)) and value > schema['maximum']:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [coerce_args(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: coerce_args(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value

def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []

def as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.0+)?\\s*', value):
        return int(float(value))
    return None

def looks_like_id(value: Any, named: bool=False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or ' ' in value.strip():
        return False
    return named or bool(re.search('\\d', value))

def detect_id_field(rows: list[dict[str, Any]]) -> str | None:
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all((looks_like_id(v, 'id' in str(key).lower()) for v in values)):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ('id' in str(key).lower(), sum((len(v) for v in values)) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None

def tree_has_id(node: Any, row_id: str, depth: int=0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any((isinstance(v, dict) and row_id in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node)) or any((tree_has_id(v, row_id, depth + 1) for v in node))
    if isinstance(node, dict):
        return any((tree_has_id(v, row_id, depth + 1) for v in node.values()))
    return False

def attr_payload(node: Any) -> dict[str, Any] | None:
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and (not isinstance(v, bool))}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best

class Want:

    def __init__(self, name: str, shelf: str, qty: int, musts: list[str]):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.musts = musts
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []

class WorldAgent:

    def __init__(self, problem: dict[str, Any]):
        environment = problem['environment']
        self.binding = dict(environment['binding'])
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.tools = [t for t in self.view.get('tools') or [] if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas = {t['function']['name']: t['function'].get('parameters') if isinstance(t['function'].get('parameters'), dict) else {'type': 'object', 'properties': {}} for t in self.tools}
        steps = as_int(self.view.get('max_steps'))
        if steps is None and isinstance(self.view.get('max_steps'), float):
            steps = int(self.view['max_steps'])
        self.max_steps = steps if steps and steps > 0 else 30
        self.t0 = time.monotonic()
        self.out_of_time = False
        self.solo_ask = False  # after muddle/crowded: one property, plain words, with examples
        self.listing = True    # multi-prop list asks until crowded/muddle turns them off
        self.list_asks = 0
        self.process_boost = False
        self.finish_nudge = False
        self.late_asked = False
        self.recovered = False
        self.intake_lanes: dict[int, list[str]] = {}
        self.intake_used: dict[int, set[str]] = {}
        self.intake_found: dict[int, int] = {}
        self.intake_spent: dict[int, int] = {}
        self.loop_cap = max(TURN_BUDGET, min(110, self.max_steps))
        self.call_cap = max(CALL_CEILING, self.loop_cap + 25)
        self.step = 0
        self.finished = bool(self.view.get('finished'))
        self.broke = False
        self.no_credit = False
        self.waited = 0
        self.misses = 0
        self.cost = 0.0
        self.calls = 0
        self.loop_calls = 0
        self.trace: list[dict[str, Any]] = []
        self.brief: dict[str, Any] = {}
        self.roles: dict[str, str] = {}
        self.items: list[Want] = []
        self.budget: Any = None
        self.msg_limit = 0
        self.msg_used = 0
        self.process: list[dict[str, Any]] = []
        self.reserve = 0
        self.gate_msgs = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.asked = False
        self.qa: list[tuple[str, str]] = []
        self.rows: dict[str, dict[str, Any]] = {}
        self.full: dict[str, dict[str, Any]] = {}
        self.seen_order: list[str] = []
        self.heard: set[str] = set()
        self.notes: list[str] = []
        self.version = 0
        self.cart_touched = False
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_view: Any = None
        self.note_named = False
        self.cart_lines: dict[str, int] = {}
        self.checked: set[str] = set()
        self.audited = False
        self.offer_seen: dict[str, int] = {}
        self.gone: dict[str, int] = {}
        self.unavailable: set[str] = set()
        self.sold_lines: set[str] = set()
        self.note_sold = False
        self.held = 0
        self.id_args: set[tuple[str, str]] = set()
        self.cart_said = ''
        self.refused_at = -1
        self.leads: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.late_held = False
        self.lead_items: dict[str, list[str]] = {}
        self.verdicts: dict[str, bool] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.id_key: str | None = None
        self.facts_from: dict[str, bool] = {}
        self.queued: list[str] = []
        self.buys = 0
        self.refusals = 0
        self.recent: list[str] = []
        self.chat: list[dict[str, Any]] = []
        self.opened: set[str] = set()
        self.refs: list[tuple[int, str]] = []
        self.ref_values: dict[str, str] = {}
        self.trails = 0
        self.trail_blocks = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.tool_keys: dict[str, set[str]] = {}
        self.view_keys: dict[str, tuple[int, set[str]]] = {}
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.pages_read = 0
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.cheaper_marks: set[str] = set()
        self.wider_held = False
        self.paged_obs: list[Any] = []
        self.plan_order: list[int] = []
        self.line_holds = 0
        self.bound: list[tuple[int, int]] = []
        self.fit_holds = 0
        self.searched: list[dict[str, Any]] = []
        self.link_searches = 0
        self.link_seen: dict[str, list[str]] = {}

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def time_left(self) -> float:
        return DEADLINE - (time.monotonic() - self.t0)

    def page_budget(self) -> int:
        """Upcoming packed races punish over-paging into the clock."""
        secs = self.time_left()
        if secs <= CLOSING + 20 or self.cost >= SPEND_LIGHT:
            return 2
        if secs <= CLOSING + 80:
            return 4
        return PAGE_SWEEP

    def late_models(self) -> list[str]:
        return [MODEL_DEEP, MODEL_FAST] if self.cost < SPEND_LIGHT else [MODEL_FAST, MODEL_DEEP]

    def asks_left(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def cart_fingerprint(self) -> str:
        return jdump(sorted(self.cart_lines.items()))

    def paging_tools(self) -> set[str]:
        page = self.roles.get('offer') or self.roles.get('open')
        return {t for t in (page, self.roles.get('compare')) if t}

    def attr_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.solo_string_arg(t)), None)
        if known:
            return known
        tool = self.roles.get('open')
        return tool if self.facts_from.get(tool or '') is not False and self.solo_string_arg(tool) else None

    def batch_attr_tool(self) -> bool | None:
        return self.facts_from.get(self.roles.get('compare') or '')

    def batch_attr_missing(self) -> bool:
        seen = self.batch_attr_tool()
        return seen is False or (seen is None and (self.facts_from.get(self.attr_tool() or '') is True or bool(self.roles.get('offer'))))

    def finish_due(self, stage: str) -> bool:
        return any((p['when'] == stage for p in self.process)) and (stage, self.cart_fingerprint() + f'#{self.version}') not in self.gate_ok

    def reserve_steps(self) -> int:
        talk = any((p['when'] == 'order' and p['messages'] > 0 for p in self.process))
        basket_owed = self.needs_basket_note('order') and (not self.basket_note_after_cart())
        due = self.finish_due('order') or basket_owed
        return 2 if (talk or basket_owed) and self.msg_used < self.msg_limit and due else 1

    def list_cap(self) -> int:
        return LIST_SIZE if len(self.items) == 1 else LIST_MIXED

    def list_now(self) -> int:
        """Labels named in the next list ask: ramp narrow → fuller after the shopper has taken a list."""
        if self.solo_ask or not self.listing:
            return 1
        steps = LIST_STEPS if len(self.items) == 1 else LIST_STEPS_MIXED
        return min(steps[self.list_asks], self.list_cap()) if self.list_asks < len(steps) else self.list_cap()

    def answered_label(self, labels: list[str], new: list[str]) -> int:
        """Index of the first asked label a disclosed need matches; -1 when none."""
        for i, label in enumerate(labels):
            bag = word_bag(label)
            for need in new:
                prop = need.split(':', 1)[0].strip()
                if keys_overlap(bag, word_bag(prop)) or canon_key(label) in canon_key(prop) or canon_key(prop) in canon_key(label):
                    return i
        return -1

    def judge_pool(self, limit: float=BUDGET_SOFT) -> list[str]:
        if self.cost >= SPEND_STOP or self.time_left() < LAST_CALL:
            return []
        if self.cost <= limit and self.cost + DEEP_ROOM <= BUDGET_HARD and self.cost < SPEND_LIGHT:
            return [MODEL_DEEP, MODEL_FAST]
        return [MODEL_FAST] if self.cost <= BUDGET_HARD else []

    def model_turn(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None=None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= self.call_cap or self.cost >= SPEND_STOP:
                return None
            if self.time_left() < LAST_CALL:
                self.out_of_time = True
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.raw_model_call(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                passing = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or tries >= (1 if status in (None, 408) else len(RETRY_GAPS)) or self.waited + RETRY_GAPS[tries] > RETRY_BUDGET_S:
                    break
                if self.time_left() - RETRY_GAPS[tries] < LAST_CALL:
                    self.out_of_time = True
                    break
                pause = RETRY_GAPS[tries] * (0.8 + 0.4 * random.random())
                self.waited += RETRY_GAPS[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.no_credit = True
                return None
        return None

    def raw_model_call(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        result = _llm.post_verbose('/inference/chat/completions', body)
        data = result.data
        if isinstance(data, dict) and data.get('choices'):
            usage = data.get('usage') or {}
            if isinstance(usage.get('cost'), (int, float)):
                self.cost += float(usage['cost'])
            message = data['choices'][0].get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return (message, 200)
            self.trace.append({'kind': 'llm_fail', 'model': model, 'status': 0})
            return (None, 0)
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = as_int(data['error'].get('code'))
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:200]})
        return (None, status)

    def json_model_call(self, prompt: str, models: list[str], max_tokens: int=1800) -> dict[str, Any] | None:
        for model in models:
            message = self.model_turn([{'role': 'user', 'content': prompt}], [model], max_tokens)
            got = as_object((message or {}).get('content'))
            if got is not None:
                return got
            if self.broke or self.no_credit or self.calls >= self.call_cap or self.out_of_time or self.cost >= SPEND_STOP:
                return None
        return None

    def invoke_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                seen = data.get('observation')
                if name in self.paging_tools() and (not (isinstance(seen, dict) and seen.get('error'))):
                    for value in args.values():
                        self.opened.update((v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str)))
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low:
                    self.finished = True
                    return None
                found = re.search('expected (\\d+)', body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broke = True
                return None
        return None

    def parse_briefing(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {jdump(fn.get('parameters') or {})}")
        prompt = BRIEFING + '\nTOOLS:\n' + '\n'.join(lines) + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nOPENING NOTE FROM THE STORE:\n' + jdump(trim_tree(self.view.get('opening'), 1500))
        brief = self.json_model_call(prompt, [MODEL_FAST], 4000) or {}
        if not self.no_credit and (not as_list(brief.get('items')) or not isinstance(brief.get('roles'), dict) or (not brief['roles'].get('buy'))):
            brief = self.json_model_call(prompt, [MODEL_DEEP, MODEL_FAST], 2500) or brief
        elif not self.no_credit and self.same_shelf(brief):
            again = self.json_model_call(prompt, [MODEL_DEEP], 2500) or {}
            if as_list(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.schemas}
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: ' + jdump(self.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).'
            fixed = self.json_model_call(ask, [MODEL_FAST, MODEL_DEEP], 1500) or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and (fixed['buy_args'] != fixed['refuse_args']):
                brief['buy_args'], brief['refuse_args'] = (fixed['buy_args'], fixed['refuse_args'])
        for entry in as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = as_int(entry.get('quantity'))
            count = quantity if quantity and quantity > 0 else 1
            musts = [str(m) for m in as_list(entry.get('musts')) if m]
            self.items.append(Want(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), count, musts))
        merged: list[Want] = []
        became: dict[int, int] = {}
        for old, item in enumerate(self.items, 1):
            twin = next((m for m in merged if item.shelf.strip() and canon_key(m.shelf) == canon_key(item.shelf) and (not any((canon_key(a.split(':')[0]) == canon_key(b.split(':')[0]) and canon_key(a.partition(':')[2]) != canon_key(b.partition(':')[2]) for a in m.musts for b in item.musts)))), None)
            if twin is None:
                merged.append(item)
                became[old] = len(merged)
            else:
                became[old] = merged.index(twin) + 1
                twin.musts += [m for m in item.musts if m not in twin.musts]
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged
        if not self.items:
            self.items = [Want('the requested product', '', 1, [])]
        for pair in as_list(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = (became.get(as_int(pair.get('item')) or 0), became.get(as_int(pair.get('with')) or 0))
                if part and main and (part != main) and (1 <= part <= len(self.items)) and (1 <= main <= len(self.items)) and ((part, main) not in self.bound):
                    self.bound.append((part, main))
        for lookup in as_list(brief.get('lookups')):
            text = str(lookup).strip()
            if not any((text in item.musts for item in self.items)):
                key = canon_key(text.split(':')[0])
                text = next((m for item in self.items for m in item.musts if canon_key(m.split(':')[0]) == key), text)
            number = next((n for n, item in enumerate(self.items, 1) if text in item.musts), 1)
            if text and (number, text) not in self.refs:
                self.refs.append((number, text))
        self.split_mixed_need()
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0):
            self.budget = budget
        limit = as_int(brief.get('messages'))
        if limit is not None and limit >= 0:
            self.msg_limit = limit
        elif self.roles.get('ask'):
            self.msg_limit = 3
        steps = as_int(brief.get('steps'))
        if steps is not None and 10 <= steps < self.max_steps:
            self.max_steps = steps
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, as_int(entry.get('messages')) or 0)})
        if self.roles.get('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    @staticmethod
    def same_shelf(brief: dict[str, Any]) -> bool:
        shelves = [canon_key(str(e.get('shelf') or '')) for e in as_list(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        return len(shelves) != len(set(shelves))

    def split_mixed_need(self) -> None:
        for number, item in enumerate(self.items, 1):
            pointers = [need for n, need in self.refs if n == number and need in item.musts]
            rest = [m for m in item.musts if m not in pointers]
            if not pointers or not rest:
                continue
            prompt = UNMIX_PROMPT + '\n\nREQUEST:\n' + self.request + f'\n\nPRODUCT: {item.name}\nREQUIREMENTS POINTING AT ANOTHER LISTING:\n' + '\n'.join(pointers) + '\nOTHER REQUIREMENTS:\n' + '\n'.join(rest)
            got = self.json_model_call(prompt, [MODEL_DEEP, MODEL_FAST], 1500) or {}
            moved = [m for m in rest if m in [str(x).strip() for x in as_list(got.get('describe_other'))]]
            if not moved:
                continue
            item.musts = [m for m in item.musts if m not in moved]
            joined = pointers[0] + ' (that listing: ' + '; '.join(moved) + ')'
            item.musts = [joined if m == pointers[0] else m for m in item.musts]
            self.refs = [(n, joined if (n, need) == (number, pointers[0]) else need) for n, need in self.refs]
            self.trace.append({'kind': 'unmix', 'item': number, 'moved': moved})

    def is_described(self, number: int, need: str) -> bool:
        if need in [m for _, m in self.refs] or need in [str(x) for x in as_list(self.brief.get('lookups'))]:
            return False
        prop, _, value = need.partition(':')
        if not prop.strip() or len(re.findall('[a-z0-9]+', value.lower())) < 4:
            return False
        if re.search('\\b(?:listing|page|shown|other than|unlike|same as)\\b', value, re.I):
            return False
        item = self.items[number - 1]
        if any(({w for w in word_bag(other.name) if len(w) >= 4} & word_bag(value) for other in self.items if other is not item)):
            return False
        cat_key = self.shelf_field()
        keys: set[str] = set()
        read = 0
        for record in self.full.values():
            if cat_key and (not self.item_owns(item, str(record.get(cat_key) or ''))):
                continue
            block = attr_payload(record) or {}
            if block:
                read += 1
                keys |= {canon_key(k) for k in block}
        return read >= 3 and (not any((word_bag(k) & word_bag(prop) for k in keys)))

    def ledger_text(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            live = [m for m in item.musts + item.told if m not in item.dropped]
            told = [m for m in live if self.is_described(number, m)]
            needs = [m + (f' (the page it points at gives: {self.ref_values[m]})' if m in self.ref_values else '') for m in live if m not in told]
            head = f'item {number}: {item.name}' if many else f'item: {item.name}'
            if item.shelf:
                head += f' (store label: {item.shelf})'
            head += f', quantity {item.qty}'
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if told:
                text += "; described in the shopper's words, which no listing attribute states (decides nothing): " + '; '.join(told)
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
            parts.append(text)
        brief = self.brief
        if self.budget is not None:
            scope = 'for each item' if brief.get('budget_scope') == 'each' else 'in total'
            parts.append(f"budget: at most {self.budget} {brief.get('budget_unit') or ''} {scope}".replace('  ', ' '))
        pick = brief.get('pick')
        if pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif pick in ('most', 'least'):
            parts.append(f"pick: the {pick} {brief.get('pick_measure') or ''} among fitting listings".replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if brief.get('in_stock_only', True) is not False:
            parts.append('in stock only')
        for wish in as_list(brief.get('wishes')):
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in as_list(brief.get('rules')):
            parts.append(f'rule from the request: {rule}')
        if brief.get('quiet_decline') is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (World does it when you check out)', 'end': 'at an ending without a purchase (World does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        facts = self.attr_tool()
        if facts and (self.facts_from.get(facts) is True or self.roles.get('offer')):
            bare = sorted((t for t in self.paging_tools() | {self.roles.get('offer') or ''} if t and self.facts_from.get(t) is False))
            pages = sorted((t for t in self.paging_tools() if t != facts))
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show none" if bare else '') + (f"; a listing's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ''))
        return '\n'.join(('- ' + p for p in parts))

    def progress_banner(self) -> str:
        left = self.steps_left()
        text = f'[World] step {self.step} of {self.max_steps} used, {left} left.'
        if left <= self.reserve_steps() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; end without a purchase only when your search showed that an item has no fitting listing (running out of steps does not show that).'
        elif left <= 6 or (left <= len(self.items) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to the chosen lines and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and (not self.cart_touched) and (self.step >= self.max_steps // 2):
            text += " Half the steps are used and the cart is empty: cart each item's fitting listing now (the store may still sell one out)."
        secs = self.time_left()
        if secs < CLOSING + 50:
            text += f" Time is short: about {max(0, int(secs))} seconds are left for this task. Cart the fitting listings you have and check out now."
        elif secs < CLOSING + 110 and self.buys == 0:
            text += " Move to carting now: open one fitting listing per item if needed, put each in the cart, then check out."
        if self.leads and self.buys == 0 and self.steps_left() <= max(8, self.max_steps // 3):
            text += " Fitting leads are already known: cart the best ones and check out; do not keep searching."
        live = {need for _, need in self.open_requirements()}
        lookups = [need for _, need in self.refs if need in live and need not in self.ref_values]
        if lookups and self.buys == 0:
            text += '\n[resolve first: open the listing each of these points at, and every listing its page refers to; the last page gives the value] ' + ' | '.join(lookups[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.compat_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        text += '\n[ledger]\n' + self.ledger_text()
        return text

    def row_id(self, node: Any, in_list: bool=False, depth: int=0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_like_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any((isinstance(v, dict) for v in node.values()))
        if not (in_list or depth == 0 or nested):
            return None
        found = detect_id_field([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def ingest_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all((looks_like_id(r.get(self.id_key), True) for r in rows)):
            return True
        found = detect_id_field(rows)
        if found:
            self.id_key = found
            return True
        return False

    def ingest_payload(self, observation: Any) -> None:

        def walk(node: Any, depth: int, in_list: bool=False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                row_id = self.row_id(node, in_list, depth)
                if row_id:
                    if row_id not in self.rows:
                        self.seen_order.append(row_id)
                        self.rows[row_id] = {}
                    old = self.rows[row_id]
                    if any((old.get(k) is True and v is False for k, v in node.items())):
                        self.unavailable.add(row_id)
                    elif any((old.get(k) is False and v is True for k, v in node.items())):
                        self.unavailable.discard(row_id)
                    self.rows[row_id].update(node)
                    if attr_payload(node):
                        self.full[row_id] = dict(self.rows[row_id])
                    elif row_id in self.full:
                        self.full[row_id].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.ingest_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def known_shelves(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get('enum') or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.schemas.values():
            walk(schema, 0)
        return found

    def shelf_field(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and (not re.search('\\d{3}', value)):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.known_shelves()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum((1 for v in values if v.strip().lower() in known)) / len(values)
            mean = sum((len(v) for v in values)) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def attr_key_census(self) -> list[tuple[str, str, int, list[str]]]:
        cat_key = self.shelf_field()
        groups: dict[str, dict[str, Any]] = {}
        for row_id in self.seen_order:
            record = self.full.get(row_id)
            if not record:
                continue
            block = attr_payload(record) or {}
            label = str(record.get(cat_key) or 'listings') if cat_key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = canon_key(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group['keys'].setdefault(name, {'n': 0, 'values': [], 'per': {}})
                slot['n'] += 1
                text = str(value).strip().lower()
                slot['per'][row_id] = text
                if text not in slot['values']:
                    slot['values'].append(text)
        wanted = [(canon_key(item.shelf or ''), canon_key(item.name), {w for w in word_bag(item.shelf or '') | word_bag(item.name) if len(w) >= 4}) for item in self.items]
        matched = {label for label in groups if any((canon_key(label) in (shelf, name) or {w for w in word_bag(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_item(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = list(slot['per'].values())
                if not values:
                    continue
                mean = sum((len(v) for v in values)) / len(values)
                if mean > 40 or sum((1 for v in values if ' / ' in v)) * 2 >= len(values):
                    continue
                recur = sum((len(set(re.findall('[a-z]{3,}', v)) & set(re.findall('[a-z]{3,}', ' '.join((o for o in values if o is not v))))) > 0 for v in values))
                if len(values) >= 4 and len(set(values)) == len(values) and (mean > 12) and (recur * 2 < len(values)):
                    continue
                twin = next((spec for other, spec in kept if self.twin_rows(other, spec, name, slot)), None)
                if twin is not None:
                    for row_id, text in slot['per'].items():
                        twin['per'].setdefault(row_id, text)
                        if text not in twin['values']:
                            twin['values'].append(text)
                    twin['n'] = len(twin['per'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def unset_props(self, label: str, name: str, values: list[str]) -> str:
        number = self.owner_item(label)
        if not number:
            return ''
        for must in self.items[number - 1].musts:
            prop, _, value = must.partition(':')
            if re.search('\\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            words = set(re.findall('[a-z0-9]{3,}', value.lower()))
            shown = set(re.findall('[a-z0-9]{3,}', ' '.join(values).lower()))
            if prop.strip() and canon_key(prop) != name and word_bag(prop) & word_bag(name) and words and (not words <= shown):
                return f' [not the stated {prop.strip()}: another property]'
        return ''

    @staticmethod
    def twin_rows(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        a, b = (set(name.split()), set(other.split()))
        if not a or not b or (not a & b):
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = (re.findall('[a-z0-9]+', x.lower()), re.findall('[a-z0-9]+', y.lower()))
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and (len(long) <= 2 * len(short) + 2))
        both = [i for i in spec['per'] if i in slot['per'] and spec['per'][i].strip() and slot['per'][i].strip()]
        same = sum((1 for i in both if agree(spec['per'][i], slot['per'][i])))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec['values']) & set(slot['values'])) - {''})

    def category_buckets(self) -> list[list[str]]:
        cat_key = self.shelf_field()
        groups: dict[str, list[str]] = {}
        for row_id in self.seen_order:
            label = str(self.rows[row_id].get(cat_key) or '') if cat_key else ''
            groups.setdefault(label, []).append(row_id)
        wanted = [(canon_key(item.shelf or ''), canon_key(item.name), {w for w in word_bag(item.shelf or '') | word_bag(item.name) if len(w) >= 4}) for item in self.items]
        matched = [ids for label, ids in groups.items() if any((canon_key(label) in (shelf, name) or {w for w in word_bag(label) if len(w) >= 4} & long for shelf, name, long in wanted)) or self.owner_item(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.items))]

    def solo_string_arg(self, tool: str | None) -> str | None:
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def listing_price(self, row_id: str) -> float | None:
        value = (self.rows.get(row_id) or {}).get(self.price_key) if self.price_key else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and (not isinstance(v, bool))), None)
        if isinstance(value, str) and re.fullmatch('\\s*\\d+(\\.\\d+)?\\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and (not isinstance(value, bool)) else None

    def is_buyable(self, row_id: str) -> bool:
        if self.is_sold_out(row_id):
            return False
        value = (self.rows.get(row_id) or {}).get(self.stock_key) if self.stock_key else None
        return not (value is False or (isinstance(value, (int, float)) and (not isinstance(value, bool)) and (value <= 0)))

    def extract_records(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            row_id = self.row_id(observation)
            if row_id:
                return [(row_id, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.row_id(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.row_id(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.row_id(r, True)] if i]
        return []

    def track_view_keys(self, name: str, observation: Any) -> None:
        for row_id, record in self.extract_records(observation):
            keys = set(record)
            self.tool_keys.setdefault(name, set()).update(keys)
            version, have = self.view_keys.get(row_id, (-1, set()))
            self.view_keys[row_id] = (self.version, (have if version == self.version else set()) | keys)

    def already_viewed(self, name: str, row_id: str) -> bool:
        version, have = self.view_keys.get(row_id, (-1, set()))
        keys = self.tool_keys.get(name)
        return bool(keys) and version == self.version and (keys <= have) and (row_id not in self.gone)

    def infer_result_shape(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.shapes:
            return self.shapes[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all((isinstance(r, dict) for r in v))]
        if not lists:
            return None
        self.shapes[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE_PROMPT + '\n\nTOOL: ' + name + ' | parameters: ' + jdump(self.schemas.get(name) or {}) + '\n\nRESULT (first rows only):\n' + jdump(trim_tree(sample, 120))[:4000]
        got = self.json_model_call(prompt, [MODEL_FAST, MODEL_DEEP], 2500) or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props), 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.shapes[name] = shape
        self.price_key = self.price_key or shape['price']
        self.stock_key = self.stock_key or shape['stock']
        self.trace.append({'kind': 'shape', 'tool': name, 'shape': shape})
        return shape

    def next_page_token(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.infer_result_shape(name, observation)
        if not shape or not shape['count'] or (not shape['next']) or (not shape['next_arg']) or (not isinstance(observation, dict)):
            return None
        rows, count, value = (observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next']))
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or (not isinstance(count, int)):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return (count, len(rows), shape['next_arg'], value)

    def rows_price_ordered(self, observation: Any) -> bool:
        prices = [self.listing_price(i) for i, _ in self.extract_records(observation)]
        return len(prices) >= 3 and None not in prices and all((a <= b for a, b in zip(prices, prices[1:])))

    def auto_page_search(self, name: str, args: dict[str, Any], observation: Any) -> str:
        self.paged_obs = []
        pick = self.brief.get('pick')
        plan = self.next_page_token(name, observation) if pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and (not isinstance(given, bool)) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if pick == 'cheapest' and self.rows_price_ordered(observation):
            return ''
        pages = -(-(count - shown) // shown)
        sweep = self.page_budget()
        if pages > 5 or self.pages_read + min(pages, sweep) > sweep or self.steps_left() - self.reserve_steps() - min(pages, sweep) < 10:
            return f"\n[World] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: search with narrower words (the requirement's value) or page on with {param}={jdump(value)}."
        pages = min(pages, sweep - self.pages_read)
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = coerce_args({**args, param: value}, self.schemas.get(name), 'args', faults)
            if faults:
                break
            reply = self.invoke_tool(name, call)
            if reply is None or self.finished:
                break
            self.pages_read += 1
            seen = reply.get('observation')
            self.on_observation(name, seen)
            self.paged_obs.append(seen)
            shown_text = self.render_obs(seen)
            self.read_cache[name + jdump(call)] = (self.version, self.step, shown_text)
            texts.append(f'{param}={jdump(value)}: {shown_text}')
            after = self.next_page_token(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': name, 'read': len(texts)})
        rest = f"\n[World] {count} listings matched and World has read {shown * (len(texts) + 1)} rows of them; narrow the search with the requirement's value or page on to see the rest." if value is not None and texts else ''
        return '\n[World read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def ignore_cap(self, name: str, args: dict[str, Any], observation: Any) -> str:
        props = (self.schemas.get(name) or {}).get('properties') or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search('price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and (not isinstance(v, bool)) and re.search('max|most|under|below|limit|cap|upto|up_to', k, re.I) and priced(k)]
        if not caps or not self.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.extract_records(observation) if (self.listing_price(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        other_listings = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join((f'{k}={jdump(o[0])}' for k, o in other_listings.items() if o)) + f' and the same {caps[0]} to see only cheaper listings' if any(other_listings.values()) else ''
        return f'\n[World] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def rank_cheap_rows(self, observations: list[Any]) -> str:
        if self.brief.get('pick') != 'cheapest' or not self.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.extract_records(seen)}
        rows = [(self.listing_price(i), i) for i in found if self.is_buyable(i) and self.listing_price(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[World] buyable rows here, cheapest first: ' + ', '.join((f'{i} ({p:g})' for p, i in sorted(rows)[:6]))

    def expand_search_hint(self) -> str:
        if self.wider_held or self.brief.get('pick') != 'cheapest' or self.steps_left() < 14 + 2 * len(self.items):
            return ''
        self.wider_held = True
        lines = [f'{i} at {self.listing_price(i):g}' for i in self.cart_product_ids() if self.listing_price(i) is not None]
        return f"Not run yet (no step used). {self.steps_left()} steps are left and the pick is the cheapest. Searches match a listing's title, so a requirement's value (often only in the attributes) finds few of the fitting listings, and rows come in no price order. Before checking out, run a few more searches for each item under other names for the product (its plain one-word name, synonyms, words from fitting titles you have seen) and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') + '; read the attributes of the cheapest such rows, swap in any that fits, then check out.'

    def pull_attributes(self, ids: list[str], keep: int) -> list[str]:
        tool = self.attr_tool()
        target = self.solo_string_arg(tool)
        if not tool or not target:
            return []
        texts = []
        for row_id in [i for i in ids if i in self.rows and i not in self.full and (not self.is_sold_out(i))]:
            if self.steps_left() <= keep or self.finished or self.facts_from.get(tool) is False:
                break
            faults: list[str] = []
            args = coerce_args({target: row_id}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            reply = self.invoke_tool(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.on_observation(tool, seen)
            self.track_view_keys(tool, seen)
            if isinstance(seen, dict) and self.row_id(seen):
                self.facts_from[tool] = self.facts_from.get(tool, False) or bool(attr_payload(seen))
            texts.append(f'{row_id}: ' + self.render_obs(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def probe_cheaper(self) -> list[str]:
        if self.brief.get('pick') != 'cheapest':
            return []
        mark = self.cart_fingerprint() + f'#{self.version}'
        cat_key = self.shelf_field()
        if mark in self.cheaper_marks or not cat_key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_product_ids()
        picks: list[str] = []
        for line in cart:
            price = self.listing_price(line)
            kind = str((self.rows.get(line) or {}).get(cat_key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.seen_order if i not in cart and i not in self.full and self.is_buyable(i) and (str((self.rows.get(i) or {}).get(cat_key) or '').strip().lower() == kind) and ((self.listing_price(i) if self.listing_price(i) is not None else float('inf')) < price)]
            pool.sort(key=lambda i: self.listing_price(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.pull_attributes(picks[:CHEAP_PROBES], self.reserve_steps() + 4)

    def on_observation(self, name: str, observation: Any) -> str:
        self.ingest_payload(observation)
        lines = self.note_candidates(observation)
        if not lines:
            return ''
        meaning = self.apply_store_note(lines)
        if meaning:
            self.queued.append(f'(seen on {name}) {meaning}')
        return meaning

    def seed_attr_reads(self) -> None:
        tool = self.roles.get('compare')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in schema_kinds(v)), None)
        if not tool or not target or any((k != target for k in schema.get('required') or [])) or self.batch_attr_missing():
            self.seed_attr_reads_solo()
            return
        if self.steps_left() <= 12:
            return
        cap = props[target].get('maxItems') if isinstance(props[target].get('maxItems'), int) else 4
        ranked = self.category_buckets()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.full][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = coerce_args({target: picks}, schema, 'args', faults)
        if faults:
            return
        before = len(self.full)
        reply = self.invoke_tool(tool, args)
        if reply is not None:
            seen = reply.get('observation')
            self.on_observation(tool, seen)
            self.track_view_keys(tool, seen)
            rows = self.extract_records(seen)
            if rows:
                self.facts_from[tool] = self.facts_from.get(tool, False) or any((attr_payload(r) for _, r in rows))
            self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.full) - before})
            if rows and self.facts_from.get(tool) is False:
                self.seed_attr_reads_solo()
                return
            if len(self.full) == before and len(picks) > 2 and (self.steps_left() > 12):
                faults = []
                args = coerce_args({target: picks[:2]}, schema, 'args', faults)
                if not faults:
                    reply = self.invoke_tool(tool, args)
                    if reply is not None:
                        self.on_observation(tool, reply.get('observation'))

    def seed_attr_reads_solo(self) -> None:
        tool = self.attr_tool()
        target = self.solo_string_arg(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.open_requirements() for w in re.findall('[a-z0-9]{3,}', need.split(':', 1)[-1].lower())}
        each = SEED_EACH if len(self.items) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.category_buckets():
            opened = sum((1 for i in ids if i in self.full))
            pool = [i for i in ids if i not in self.full and self.is_buyable(i)]
            if self.brief.get('pick') == 'cheapest':
                pool.sort(key=lambda i: self.listing_price(i) if self.listing_price(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall('[a-z0-9]{3,}', jdump(self.rows.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.pull_attributes(first[:cap][:8], 22)
        self.pull_attributes((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    def intake_ready(self) -> bool:
        if self.asked or not self.roles.get('ask') or self.asks_left() <= 0:
            return False
        if self.time_left() <= CLOSING + QUESTION_TIME:
            return False
        if self.steps_left() <= 8:
            return False
        opened = len(self.full)
        if opened >= 3 * len(self.items) + 1:
            return True
        return self.step >= max(8, self.max_steps // 4)

    def compose_ask_args(self, text: str) -> dict[str, Any] | None:
        name = self.roles.get('ask')
        schema = self.schemas.get(name or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        target = next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get('required') or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        faults: list[str] = []
        args = coerce_args(args, schema, 'args', faults)
        return None if faults else args

    def run_intake(self) -> str:
        self.asked = True
        if self.asks_left() <= 0:
            return ''
        self.seed_attr_reads()
        lanes = self.build_lanes(self.asks_left())
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        used: dict[int, set[str]] = {n: set() for n in lanes}
        self.intake_lanes = {n: list(v) for n, v in lanes.items()}
        self.intake_used = {n: set(used[n]) for n in used}
        self.intake_found = dict(found)
        self.intake_spent = dict(spent)
        follow: list[tuple[int, list[str], bool]] = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        muddle_tries: dict[tuple[int, str], int] = {}
        while self.asks_left() > 0 and (not self.finished) and (self.steps_left() > 6) and (self.time_left() > CLOSING + QUESTION_TIME):
            if follow:
                number, known_shelves, shown = follow.pop(0)
            else:
                number, known_shelves = self.queued_ask(lanes, found, spent, used)
                shown = False
                if not known_shelves:
                    break
            self.intake_used = {n: set(used[n]) for n in used}
            self.intake_found = dict(found)
            self.intake_spent = dict(spent)
            # solo_ask / plain / shown: one property. Else list_now() multi-prop when listing is on.
            ask_plain = self.solo_ask or shown
            shelves = known_shelves[:1] if (self.solo_ask or ask_plain or not self.listing) else known_shelves
            batch = self.ask_shopper([(number, shelves[0], self.text_model_call(number, shelves, shown=True if self.solo_ask else shown, plain=ask_plain))])
            if not batch:
                break
            sent += batch
            spent[number] += 1
            self.list_asks += 1
            had = [len(item.told) for item in self.items]
            heard = self.parse_replies(batch, (MODEL_FAST, MODEL_DEEP))
            # Multi-prop replies: free/don't-mind on a list is noisy; keep firm needs only.
            if len(shelves) > 1:
                heard.pop('free', None)
            self.apply_reply_parse(heard, told)
            fresh_needs = [need for item, size in zip(self.items, had) for need in item.told[size:]]
            fresh = [n for n, item in enumerate(self.items, 1) if len(item.told) > had[n - 1]]
            for n in fresh:
                found[n] = found.get(n, 0) + 1
            answer = batch[0][2]
            crowded_here = 1 in {as_int(x) for x in as_list(heard.get('crowded'))} or scoop_batch(answer)
            # Settle which asked labels were covered (Cairn pattern, adapted).
            if len(shelves) == 1:
                used[number].add(canon_key(shelves[0]))
            elif crowded_here and not fresh:
                self.listing = False
                self.solo_ask = True
                self.trace.append({'kind': 'lists_off', 'item': number, 'why': 'crowded'})
                used[number].add(canon_key(shelves[0]))
                follow.insert(0, (number, shelves[:1], True))
                muddle_tries[(number, canon_key(shelves[0]))] = muddle_tries.get((number, canon_key(shelves[0])), 0) + 1
                continue
            else:
                hit = self.answered_label(shelves, fresh_needs) if fresh_needs else -1
                used[number].update(canon_key(label) for label in shelves[: max(hit, 0) + 1])
                if hit < 0:
                    # No disclosed need: only the first label is spent; rest re-queue soon.
                    ahead = list(reversed(shelves[1:]))
                    rest = [label for label in lanes.get(number) or [] if canon_key(label) not in used[number] and label not in shelves]
                    lanes[number] = ahead + rest
                else:
                    ahead = shelves[hit + 1 :]
                    rest = [label for label in lanes.get(number) or [] if canon_key(label) not in used[number] and label not in shelves]
                    lanes[number] = ahead + rest
                self.trace.append({'kind': 'list_settled', 'item': number, 'asked': shelves, 'hit': hit, 'next': (lanes.get(number) or [])[:4]})
            self.intake_used = {n: set(used[n]) for n in used}
            # R17 evidence: undetected waive → same-prop plain reask → missed other asks → process fail.
            # Waive means the property is spent; move on. Never reask a waive with examples.
            waived = (not fresh) and scoop_waive(answer) and (not scoop_muddle(answer)) and (not crowded_here)
            if waived:
                label0 = shelves[0]
                item = self.items[number - 1]
                if all(canon_key(label0) != canon_key(f) for f in item.free):
                    item.free.append(label0)
                used[number].add(canon_key(label0))
                self.intake_used = {n: set(used[n]) for n in used}
                self.trace.append({'kind': 'waive_skip', 'item': number, 'label': label0, 'a': answer[:160]})
                continue
            # True muddle/crowded: same property again in PLAIN words (never reed-stop).
            if (scoop_muddle(answer) or crowded_here) and not fresh:
                self.solo_ask = True
                self.listing = False
                key = (number, canon_key(shelves[0]))
                muddle_tries[key] = muddle_tries.get(key, 0) + 1
                if muddle_tries[key] <= 2:
                    follow.insert(0, (number, shelves[:1], True))
                    continue
            if fresh:
                continue
            # After a non-answer that is not muddle: do not burn a message re-asking with examples
            # when the label is already marked used (spent). Advance to the next planned property.
            if self.solo_ask or not self.listing:
                if not shown and canon_key(shelves[0]) not in used[number] and len(self.example_values(number, shelves[0])) >= 2:
                    follow.insert(0, (number, shelves[:1], True))
                continue
            head = shelves[0].split()[-1]
            if len(shelves) == 1 and len(shelves[0].split()) >= 2 and canon_key(head) not in used[number]:
                follow.insert(0, (number, [head], False))
            elif not shown and len(shelves) == 1 and canon_key(shelves[0]) not in used[number] and len(self.example_values(number, shelves[0])) >= 2:
                follow.insert(0, (number, shelves, True))
        found_items = [n for n in found if found[n] > 0]
        searched = self.search_by_value(found_items) if found_items else ''
        if told:
            self.version += 1
        lines = [f'Q: {q}\nA: {a}' for _, q, a in sent]
        summary = "World's interviewer asked the shopper (each question used one step):\n" + '\n'.join(lines)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none')
        summary += '. Apply them to every pick (search again with the new values where needed).'
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Find that listing, open its page in full, and open every listing its page refers to; the last page gives the value, and the order counts only after all of them are opened.'
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def search_skel(self) -> tuple[str, dict[str, Any], str] | None:
        tool = self.roles.get('search')
        schema = self.schemas.get(tool or '') or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        named = [k for k in texts if re.search('(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword', k, re.I)]
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I)]
            if not named:
                known_shelves = {str(r.get(self.shelf_field()) or '').strip().lower() for r in self.rows.values()} - {''} if self.shelf_field() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.searched if a.get(k)}) >= 2 and any((len(str(a.get(k)).split()) >= 2 for a in self.searched)) and (not any((str(a.get(k) or '').strip().lower() in known_shelves for a in self.searched)))]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in (schema.get('required') or []))}
                cat_key = self.shelf_field()
                shelves = {canon_key(item.shelf or '') for item in self.items} | {canon_key(item.name) for item in self.items}
                shelves |= {canon_key(str(row.get(cat_key) or '')) for row in self.rows.values()} if cat_key else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get('required') or []) or not isinstance(v, str) or (canon_key(v) not in shelves and (not any(({w for w in word_bag(v) if len(w) >= 4} & {w for w in word_bag(s) if len(w) >= 4} for s in shelves if s))))}
                worded = [a for a in self.searched if isinstance(a.get(field), str) and re.search('[a-z]', a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get('enum'), list):
                        used = [jdump(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next((a[k] for a in worded if k in a and jdump(a[k]) == best))
                return (tool, keep, field)
        return None

    def words_of_value(self, number: int) -> list[str]:
        out = []
        for n, need in self.open_requirements():
            if n != number or need in [m for _, m in self.refs] or ':' not in need:
                continue
            value = need.split(':', 1)[1].strip()
            if not value or re.search('\\b(?:not|no|other than|unlike|alongside|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\\b', value, re.I):
                continue
            out.append(value)
        return out

    def search_by_value(self, numbers: list[int]) -> str:
        plan = self.search_skel()
        if not plan or self.steps_left() < 22:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = 8
        for number in numbers:
            values = self.words_of_value(number)
            item = self.items[number - 1]
            fresh = [v for v in values if any((':' in t and t.split(':', 1)[1].strip() == v for t in item.told))]
            if not fresh or budget <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = coerce_args({**base, field: ' '.join(words)}, self.schemas.get(tool), 'args', faults)
                if faults or self.finished:
                    break
                reply = self.invoke_tool(tool, args)
                budget -= 1
                if reply is None:
                    break
                observation = reply.get('observation')
                self.on_observation(tool, observation)
                cat_key = self.shelf_field()
                found = [i for i, _ in self.extract_records(observation)]
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                prices = [self.listing_price(i) for i in found]
                if cap is not None and prices and all((p is not None and p > 10 * cap for p in prices)):
                    cap = None
                rows = [i for i in found if self.is_buyable(i) and (not cat_key or self.item_owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))) and (cap is None or (self.listing_price(i) or 0) <= cap)]
                seen.append(f'{jdump(args.get(field))}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
                if rows:
                    opened = self.pull_attributes([i for i in rows if i not in self.full][:min(3, budget)], self.reserve_steps() + 16)
                    budget -= len(opened)
                    if opened:
                        seen.append('opened: ' + ' | '.join(opened)[:2400])
                    break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[World searched each such item's requirement values together and read the first fitting-kind rows] " + '\n'.join(seen)

    def queued_ask(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.plan_order)}
        take = self.list_now()
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if canon_key(label) not in used[number]]
            if left:
                return (number, left[:take])
        return (0, [])

    def text_model_call(self, number: int, known_shelves: list[str], shown: bool=False, plain: bool=False) -> str:
        product = self.items[number - 1].name
        if self.solo_ask or plain or len(known_shelves) <= 1:
            label = known_shelves[0]
            examples = self.example_values(number, label)
            if len(examples) < 2 and shown:
                examples = examples or self.example_values(number, label.split()[-1] if label.split() else label)
            if self.solo_ask or plain:
                return plain_reask(product, label, examples if examples else None)
            return self.format_question(number, label, examples if shown else None)
        # Multi-prop list ask (Halo/Cairn strength); crowded replies flip listing off upstream.
        labels = known_shelves[: self.list_now()]
        if len(labels) == 2:
            return f'What {labels[0]} or {labels[1]} would you like for the {product}?'
        head = ', '.join(labels[:-1]) + f' or {labels[-1]}'
        return f'What {head} would you like for the {product}?'

    def lane_attr_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.items[number - 1]
        settled = [word_bag(m.split(':', 1)[0]) for m in item.musts]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_item(label) == number]
        measured = {key: sum((1 for v in values if re.match('^\\s*[\\d.,/x -]+\\s*[a-z]{0,6}\\s*$', v))) * 2 > len(values) for _, key, _, values in table}
        if len(self.items) == 1 and (not rows):
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any((keys_overlap(prop, word_bag(key)) for prop in settled)) and (not re.search(SKIP_PROP_RE, key))]
        return [key for key in keys if not re.search(CATCHALL_RE, key)] + [key for key in keys if re.search(CATCHALL_RE, key)]

    def owner_item(self, label: str) -> int:
        if not label:
            return 0
        words = canon_key(label)
        long = root_bag(label)
        best, top = (0, 0.0)
        for number, item in enumerate(self.items, 1):
            if words == canon_key(item.shelf or ''):
                score = 3.0
            elif words == canon_key(item.name):
                score = 2.0
            else:
                mine = root_bag(item.shelf or '') | root_bag(item.name)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = (number, score)
        return best

    def item_owns(self, item: Want, label: str) -> bool:
        return bool(label) and item in self.items and (self.owner_item(label) == self.items.index(item) + 1)

    def records_for_item(self, number: int) -> int:
        cat_key = self.shelf_field()
        item = self.items[number - 1]
        return sum((1 for record in self.full.values() if cat_key and self.item_owns(item, str(record.get(cat_key) or '')) and attr_payload(record)))

    def build_lanes(self, count: int) -> dict[int, list[str]]:
        table = self.attr_key_census()
        settled = []
        for number, item in enumerate(self.items, 1):
            settled.append(f'item {number}: {item.name} - settled by the request: ' + ('; '.join(item.musts) or 'nothing'))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r}:')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + self.unset_props(label, name, values) + (' [several features in one: ask last]' if re.search(CATCHALL_RE, name) else ''))
        prompt = INTAKE_PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        first = [str(x) for x in as_list(self.brief.get('ask_first')) if str(x).strip()]
        if first:
            prompt += '\n\nASK FIRST (the request asks to be asked about these):\n' + '\n'.join(first)
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.json_model_call(prompt, [MODEL_DEEP, MODEL_FAST], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in as_list(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, as_int(pick.get('item') or 1) or 1), len(self.items))
            label = re.split('\\s+(?:and|or)\\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            if all((canon_key(label) != canon_key(other) for other in lanes[number])):
                lanes[number].append(label)
            if str(pick.get('hint') or '').strip():
                hinted[number].add(canon_key(label))
                if number not in self.plan_order:
                    self.plan_order.append(number)
        for number in lanes:
            keys = self.lane_attr_keys(number, table)
            read = max(1, self.records_for_item(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner_item(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub('our\\b', 'or', re.sub('(?<=[a-z]{3})s\\b', '', canon_key(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if canon_key(label) in own:
                    return own[canon_key(label)]
                near = [s for k, s in own.items() if keys_overlap(word_bag(label), word_bag(k)) or keys_overlap(word_bag(k), word_bag(label))]
                return max(near) if near else 0.0
            hinted[number] = {canon_key(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all((canon_key(spelled(pick)) != canon_key(other) for other in lane)):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= SHARE_GUARD and (not re.search(CATCHALL_RE, k))][:KEY_GUARD]:
                if any((canon_key(key) == canon_key(other) for other in lane[:3])):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if canon_key(other) not in hinted[number] and share_of(other) + GAP_GUARD <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if canon_key(other) != canon_key(key)]
                if weak is not None and len(self.items) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any((canon_key(x) in hinted[number] and all((canon_key(x) != canon_key(y) for y in placed[:3])) for x in lane[:3])):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all((canon_key(key) != canon_key(other) for other in lanes[number])):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(CATCHALL_RE, canon_key(x))] + [x for x in lanes[number] if re.search(CATCHALL_RE, canon_key(x))]
        self.trace.append({'kind': 'plan', 'ask': trim_tree(plan.get('ask'), 120), 'lanes': {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def format_question(self, number: int, label: str, example_values: list[str] | None=None) -> str:
        text = f'What {label} would you like for the {self.items[number - 1].name}'
        return text + (f', for example {example_values[0]} or {example_values[1]}?' if example_values else '?')

    def example_values(self, number: int, label: str) -> list[str]:
        want = canon_key(label)
        settled = ' '.join(self.items[number - 1].musts).lower()
        for _, name, _, values in self.attr_key_census():
            if name == want:
                return [v for v in values if v and v not in settled and (len(v) <= 30)][:2]
        return []

    def ask_shopper(self, questions: list[tuple[int, str, str]], floor: int=6) -> list[tuple[int, str, str]]:
        tool = self.roles['ask']
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.finished or self.steps_left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.compose_ask_args(text)
            if not args:
                break
            reply = self.invoke_tool(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            answer = reply.get('observation')
            words = self.extract_speech(answer)
            sent.append((number, text, words))
            self.done_log.append(f'step {self.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_fingerprint()})')
            self.done_steps.append(self.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300]})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def apply_reply_parse(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.items) > 1
        for entry in as_list(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, as_int(entry.get('item')) or 1), len(self.items))
            need = str(entry['need']).strip()
            if need not in self.items[number - 1].told:
                self.items[number - 1].told.append(need)
                told.append(f'item {number}: {need}' if many else need)
            if entry.get('points') is True and (number, need) not in self.refs:
                self.refs.append((number, need))
        for entry in as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                item = self.items[min(max(1, as_int(entry.get('item')) or 1), len(self.items)) - 1]
                gone = canon_key(str(entry['property']).split(':')[0])
                hit = [m for m in item.musts + item.told if m not in item.dropped and canon_key(m.split(':')[0]) == gone]
                item.dropped.extend(hit)
                told.extend((f'dropped {m}' for m in hit))
        for entry in as_list(heard.get('free')):
            if isinstance(entry, dict) and entry.get('property'):
                number = min(max(1, as_int(entry.get('item')) or 1), len(self.items))
                self.items[number - 1].free.append(str(entry['property']))
        if isinstance(heard.get('budget'), (int, float)) and (not isinstance(heard.get('budget'), bool)) and (heard['budget'] > 0):
            self.budget = heard['budget']

    def parse_replies(self, sent: list[tuple[int, str, str]], models: tuple[str, ...]=(MODEL_DEEP, MODEL_FAST)) -> dict[str, Any]:
        prompt = REPLY_READ + '\n\nITEMS:\n' + '\n'.join((f'item {n}: {it.name}' for n, it in enumerate(self.items, 1)))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join((f'{i}. (about item {n}) Q: {q}\nA: {a}' for i, (n, q, a) in enumerate(sent, 1)))
        heard = self.json_model_call(prompt, list(models), 2500) or {}
        return self.blend_reply_parse(sent, heard)

    def blend_reply_parse(self, sent: list[tuple[int, str, str]], heard: dict[str, Any]) -> dict[str, Any]:
        """Merge dialect scoops into LLM hear. Confusion does NOT stop later asks."""
        heard = dict(heard or {})
        needs = [n for n in as_list(heard.get('needs')) if isinstance(n, dict)]
        free = [n for n in as_list(heard.get('free')) if isinstance(n, dict)]
        crowded = [x for x in as_list(heard.get('crowded'))]
        have = {(int(n.get('item') or 1), str(n.get('need') or '').strip().lower()) for n in needs}
        for index, (number, question, answer) in enumerate(sent, 1):
            cash = scoop_budget(answer)
            if cash is not None and cash > 0 and not isinstance(heard.get('budget'), (int, float)):
                heard['budget'] = cash
            if scoop_batch(answer) and index not in {as_int(x) for x in crowded}:
                crowded.append(index)
            if scoop_waive(answer) and not scoop_muddle(answer) and not scoop_hard(answer) and not scoop_pointer(answer):
                prop = scoop_asked_prop(question)
                if prop and not any(canon_key(str(f.get('property') or '')) == canon_key(prop) and int(f.get('item') or 1) == number for f in free):
                    free.append({'item': number, 'property': prop})
            for need in scoop_hard(answer):
                key = (number, need.lower())
                if key in have or not need:
                    continue
                points = any(w in need.lower() for w in ('page', 'listing', 'sold by', 'shown'))
                needs.append({'item': number, 'need': need, 'points': points})
                have.add(key)
            for need in scoop_pointer(answer):
                key = (number, need.lower())
                if key in have or not need:
                    continue
                needs.append({'item': number, 'need': need, 'points': True})
                have.add(key)
        heard['needs'] = needs
        heard['free'] = free
        heard['crowded'] = crowded
        return heard

    def late_question(self) -> str:
        """Cairn/Halo lever, adapted: one more planned ask before checkout.

        Uses a spare shopper message when the cart is ready. Keeps one message
        in reserve when a basket/total process note is still due, so late-ask
        does not steal our process account strength. When listing is still on,
        names up to list_now() properties in that one message.
        """
        if self.late_asked or not self.roles.get('ask') or self.msg_used >= self.msg_limit:
            return ''
        basket_due = self.needs_basket_note('order') and (not self.basket_note_after_cart())
        # Need this ask, plus room for basket note if still owed.
        if self.msg_limit - self.msg_used < (2 if basket_due else 1):
            return ''
        if self.steps_left() < LATE_ASK_STEPS or self.time_left() < CLOSING + LATE_ASK_SECS:
            return ''
        if not self.judge_pool():
            return ''
        self.late_asked = True
        if not self.intake_lanes:
            self.intake_lanes = self.build_lanes(max(1, self.list_cap()))
            self.intake_used = {n: set() for n in self.intake_lanes}
            self.intake_found = {n: 0 for n in self.intake_lanes}
            self.intake_spent = {n: 0 for n in self.intake_lanes}
        number, shelves = self.queued_ask(self.intake_lanes, self.intake_found, self.intake_spent, self.intake_used)
        if not shelves:
            self.trace.append({'kind': 'late_ask', 'skipped': 'no property left'})
            return ''
        # Late phase: prefer a short list if still listing; else one plain property.
        ask_labels = shelves[: self.list_now()] if self.listing and not self.solo_ask else shelves[:1]
        for label in ask_labels:
            self.intake_used.setdefault(number, set()).add(canon_key(label))
        plain = self.solo_ask or len(ask_labels) == 1
        text = self.text_model_call(number, ask_labels, shown=plain, plain=plain)
        told: list[str] = []
        budget = self.budget
        batch = self.ask_shopper([(number, ask_labels[0], text)], floor=max(2, self.reserve_steps() + 1))
        if not batch:
            self.trace.append({'kind': 'late_ask', 'skipped': 'ask failed'})
            return ''
        self.list_asks += 1
        heard = self.parse_replies(batch, (MODEL_FAST, MODEL_DEEP))
        if len(ask_labels) > 1:
            heard.pop('free', None)
        self.apply_reply_parse(heard, told)
        if self.budget != budget and self.budget is not None:
            told.append(f'budget now {self.budget}')
        question, answer = batch[0][1], batch[0][2]
        if scoop_batch(answer) or scoop_muddle(answer):
            self.listing = False
            self.solo_ask = True
        self.trace.append({'kind': 'late_ask', 'q': question[:200], 'a': answer[:200], 'told': told[:8], 'labels': ask_labels})
        if not told:
            return ''
        self.version += 1
        summary = (
            "World asked the shopper one more question before checking out (one step):\n"
            f"Q: {question}\nA: {answer}\n"
            "New firm requirements from this answer: " + '; '.join(told)
            + '. The cart was not checked out: apply them, fix the cart, then check out.'
        )
        pointed = [need for _, need in self.refs if any((need in t for t in told))]
        if pointed:
            summary += ' These take their value from another listing: ' + ' | '.join(pointed) + '. Open that chain in full before buying.'
        searched = self.search_by_value([number])
        return summary + (('\n' + searched) if searched else '')

    def run_finish_guard(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        need_account = stage == 'order' and self.needs_basket_note(stage) and (not self.basket_note_after_cart())
        if (not steps and not need_account) or self.finished or (steps and (not self.finish_due(stage)) and (not need_account)):
            return (True, '')
        if (not steps) and need_account and (stage, self.cart_fingerprint() + f'#{self.version}') in self.gate_ok:
            return (True, '')
        ending = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        tools = [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {jdump(t['function'].get('parameters') or {})}" for t in self.tools if t['function']['name'] not in ending]
        said: list[str] = []
        nudged = False
        if need_account and self.roles.get('ask') and self.msg_used < self.msg_limit and self.steps_left() >= 2 and self.cart_lines:
            text = self.compose_basket_note()
            batch = self.ask_shopper([(1, '', text)], 1)
            if batch:
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                self.trace.append({'kind': 'gate_account', 'text': text[:240]})
        if not steps:
            self.gate_ok.add((stage, self.cart_fingerprint() + f'#{self.version}'))
            return (True, 'Before going on, World did the steps the request asks for:\n' + '\n'.join(said) if said else '')
        for _ in range(FINISH_TRIES):
            prompt = FINISH_STEPS.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request + '\n\nSTEPS DUE NOW:\n' + '\n'.join((f"- {p['step']}" for p in steps)) + '\n\nLEDGER:\n' + self.ledger_text() + '\n\nCART NOW: ' + self.describe_cart() + '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join((f'{i}. {line}' for i, line in enumerate(self.done_log, 1))) or '(nothing yet)') + ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') + f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.steps_left()} (the {('order' if stage == 'order' else 'ending')} itself takes one)" + '\n\nTOOLS:\n' + '\n'.join(tools)
            got = self.json_model_call(prompt, [MODEL_DEEP, MODEL_FAST] if nudged else [MODEL_FAST, MODEL_DEEP], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.apply_reply_parse({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': trim_tree(call, 200), 'steps': trim_tree(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.version += 1
                return (False, ('Before going on, World did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + ("The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'))
            name = str((call or {}).get('tool') or '')
            shown = [e for e in as_list(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and (any((p['messages'] > 0 for p in steps)) or need_account)
            lines = range(1, len(self.done_log) + 1)
            proof = {i for i in lines if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any(((as_int(n) or 0) in proof for n in as_list(e.get('lines'))))]
            if not call and unproven and (not said) and (not nudged) and self.roles.get('ask') and (self.msg_used < self.msg_limit) and (self.steps_left() >= 2):
                nudged = True
                self.trace.append({'kind': 'gate_unproven', 'steps': trim_tree(unproven, 120)})
                self.done_log.append(f'(World: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join((str(e.get('step') or '') for e in unproven))[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.steps_left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.schemas or name in ending:
                if not call:
                    self.gate_ok.add((stage, self.cart_fingerprint() + f'#{self.version}'))
                break
            if self.steps_left() < 2 or (name == self.roles.get('ask') and self.msg_used >= self.msg_limit):
                break
            faults: list[str] = []
            args = coerce_args(call.get('args') if isinstance(call.get('args'), dict) else {}, self.schemas[name], 'args', faults)
            if faults:
                break
            if name == self.roles.get('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                if need_account and (not self.basket_note_after_cart()) and self.cart_lines:
                    text = self.compose_basket_note()
                batch = self.ask_shopper([(1, '', text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.invoke_tool(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.ingest_payload(seen)
            self.done_log.append(f'step {self.step}: {name} {jdump(args)} -> {self.render_obs(seen)[:300]} (cart then: {self.cart_fingerprint()})')
            self.done_steps.append(self.step)
            said.append(f'{name} {jdump(args)} -> {self.render_obs(seen)[:400]}')
        return (True, 'Before going on, World did the steps the request asks for:\n' + '\n'.join(said) if said else '')

    def extract_speech(self, observation: Any) -> str:
        if isinstance(observation, str):
            return observation
        found: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 4:
                return
            if isinstance(node, str) and len(node.split()) >= 2:
                found.append(node)
            elif isinstance(node, dict):
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        return ' '.join(found)[:800] if found else jdump(observation)[:400]

    def note_candidates(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and (not self.row_id(observation)):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all((isinstance(v, str) for v in value)):
                    lines.extend((v for v in value if len(v.split()) >= 5))
                elif isinstance(value, dict) and (not self.row_id(value, False, 1)):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard]

    def apply_store_note(self, lines: list[str]) -> str:
        self.heard.update(lines)
        prompt = NOTE_READ + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger_text() + '\n\nTHE NOTE:\n' + '\n'.join(lines)
        got = self.json_model_call(prompt, [MODEL_FAST, MODEL_FALLBACK], 2500) or {}
        blob = '\n'.join(lines)
        cash = scoop_budget(blob)
        if cash is not None and cash > 0 and not isinstance(got.get('budget'), (int, float)):
            got['budget'] = cash
            if got.get('kind') not in (None, 'change'):
                got['kind'] = 'change'
        adds = [e for e in as_list(got.get('add')) if isinstance(e, dict)]
        have = {str(e.get('need') or '').strip().lower() for e in adds}
        for need in scoop_hard(blob) + scoop_pointer(blob):
            if need.lower() in have:
                continue
            adds.append({'item': 1, 'need': need})
            have.add(need.lower())
            if got.get('kind') not in (None, 'change'):
                got['kind'] = 'change'
        got['add'] = adds
        gone = [str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))]
        self.unavailable.update((g for g in gone if g in self.rows))
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any((g in self.rows for g in gone))
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and (not isinstance(budget, bool)) and (budget > 0) and (budget != self.budget):
            self.budget = budget
            changed.append(f'budget now {budget}')
        added: set[str] = set()
        drops = [e for e in as_list(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        adds = [e for e in as_list(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]
        for entry in drops:
            item = self.items[min(max(1, as_int(entry.get('item')) or 1), len(self.items)) - 1]
            target = str(entry['need']).strip()
            live = [m for m in item.musts + item.told if m not in item.dropped]
            hit = [m for m in live if m == target]
            if not hit:
                hit = [m for m in live if canon_key(m.split(':')[0]) == canon_key(target.split(':')[0])][:1]
            item.dropped.extend(hit or [target])
            changed.append(f'dropped {target}')
        for entry in adds:
            item = self.items[min(max(1, as_int(entry.get('item')) or 1), len(self.items)) - 1]
            need = str(entry['need']).strip()
            if need in item.dropped:
                item.dropped.remove(need)
            if need not in item.musts + item.told:
                item.told.append(need)
            added.add(need)
            changed.append(f'added {need}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.brief.get('pick') == 'cheapest' and (drops or gone or self.note_sold or any((c.startswith('budget now') for c in changed))):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.version += 1
            self.recent = []
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    def open_requirements(self) -> list[tuple[int, str]]:
        out = []
        for number, item in enumerate(self.items, 1):
            for need in item.musts + item.told:
                if need not in item.dropped:
                    out.append((number, need))
        return out

    def evidence_lines(self, observation: Any) -> str:
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                row_id = self.row_id(node, False, depth)
                if row_id and attr_payload(node) and (row_id not in ids):
                    ids.append(row_id)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.open_requirements()
        if not ids or not needs:
            return ''
        many = len(self.items) > 1
        lines = []
        for row_id in ids[:12]:
            block = attr_payload(self.full.get(row_id) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(':', 1)[0].strip()
                hits = [f'{k}={str(v)[:50]}' for k, v in block.items() if keys_overlap(word_bag(prop), word_bag(k))]
                tag = f'(item {number}) ' if many else ''
                if hits:
                    parts.append(f'{tag}{prop} -> ' + ', '.join(hits[:3]))
            if parts:
                lines.append(f'{row_id}: ' + '; '.join(parts))
        if not lines:
            return ''
        return "[evidence: keys whose names contain a requirement's property; one naming a different property (another part or aspect) decides nothing, and a key with another name may still hold the same property] " + ' | '.join(lines)


    def boost_process_from_request(self) -> None:
        """If briefing under-counted procedure steps, add/repair an order-stage account step."""
        if self.process_boost:
            return
        blob = (' '.join((str(x) for x in as_list(self.brief.get('rules')))) + ' ' + (self.request or '')).lower()
        if not any((p in blob for p in PROCESS_PHRASES)):
            return
        # Brief often lists the step with messages=0; force a shopper message so the gate can prove it.
        keys = ('basket', 'total', 'confirm', 'approval', 'tell', 'show', 'quote', 'account', 'cart', 'recap', 'sign off', 'go-ahead', 'go ahead')
        repaired = False
        for step in self.process:
            if step.get('when') != 'order':
                continue
            text = str(step.get('step') or '').lower()
            if step.get('messages', 0) > 0 or any((k in text for k in keys)):
                if step.get('messages', 0) <= 0:
                    step['messages'] = 1
                    repaired = True
                self.reserve = max(self.reserve, 1)
                self.process_boost = True
                self.trace.append({'kind': 'process_boost', 'repaired': repaired, 'step': step})
                return
        self.process.append({'step': 'tell the shopper the basket and total before ordering', 'when': 'order', 'messages': 1})
        self.reserve = max(self.reserve, 1)
        self.process_boost = True
        self.trace.append({'kind': 'process_boost', 'step': self.process[-1]})

    def needs_basket_note(self, stage: str='order') -> bool:
        keys = ('basket', 'total', 'confirm', 'approval', 'tell', 'show', 'quote', 'account', 'cart', 'recap', 'sign off', 'go-ahead', 'go ahead')
        for step in self.process:
            if step['when'] != stage:
                continue
            if step['messages'] > 0:
                return True
            blob = str(step.get('step') or '').lower()
            if any((k in blob for k in keys)):
                return True
        blob = (' '.join((str(x) for x in as_list(self.brief.get('rules')))) + ' ' + (self.request or '')).lower()
        return any((p in blob for p in PROCESS_PHRASES))

    def basket_note_after_cart(self) -> bool:
        markers = (
            'here is the basket', 'your basket', 'basket before', 'before i finish', 'total is',
            'order total', 'total:', 'total before', 'i will place the order', 'the basket:',
            'cart total', 'before checking out', 'recap', 'here is what', 'about to order',
        )
        for i, line in enumerate(self.done_log):
            if i >= len(self.done_steps):
                break
            if self.done_steps[i] <= self.cart_at:
                continue
            low = line.lower()
            if any((m in low for m in markers)):
                return True
        return False

    def compose_basket_note(self) -> str:
        parts: list[str] = []
        total = 0.0
        priced = True
        for ident, qty in self.cart_lines.items():
            row = self.full.get(ident) or self.rows.get(ident) or {}
            title = next((str(v).strip() for k, v in row.items() if isinstance(v, str) and k != self.id_key and len(v.strip()) >= 4 and any((x in k.lower() for x in ('name', 'title', 'text')))), '')
            cost = self.listing_price(ident)
            label = f'{qty} x {ident}' + (f' ({title[:60]})' if title else '')
            if cost is None:
                priced = False
                parts.append(label)
            else:
                total += float(cost) * float(qty)
                parts.append(f'{label} at {cost:g} each')
        if not parts:
            return 'Before I finish: the basket is empty, so nothing will be ordered.'
        text = 'Before I finish, here is the basket: ' + '; '.join(parts) + '.'
        if priced:
            text += f' The total is {total:g}.'
        return text + ' I will place the order now.'


    def describe_cart(self) -> str:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            return jdump(trim_tree(self.cart_view, 300))[:3000]
        return "lines by World's record of the cart calls (listing: units): " + jdump(self.cart_lines)

    def units_held(self, row_id: str) -> int:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if row_id in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [as_int(v) for k, v in node.items() if not isinstance(v, str) and as_int(v) is not None and re.search('\\b(?:qty|quantity|count|units?)\\b', canon_key(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(row_id, 1)

    def cart_product_ids(self) -> list[str]:
        if self.cart_view is not None and self.cart_seen_at >= self.cart_at:
            text = jdump(self.cart_view)
            return [i for i in self.rows if i in text]
        return list(self.cart_lines)

    def listing_blob(self, row_id: str) -> str:
        record = self.full.get(row_id) or self.rows.get(row_id)
        return jdump(trim_tree(record, 400))[:3000] if record else '(no record seen)'

    def other_listings(self, skip: list[str], limit: int=30) -> str:
        lines = []
        order = list(reversed(self.seen_order))
        if self.brief.get('pick') == 'cheapest':
            order.sort(key=lambda i: (i not in self.full, self.listing_price(i) if self.listing_price(i) is not None else float('inf')))
        for row_id in order:
            if row_id in skip or self.is_sold_out(row_id):
                continue
            record = self.full.get(row_id) or self.rows.get(row_id)
            lines.append(jdump(trim_tree(record, 160))[:900])
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def compat_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for row_id in self.seen_order:
            record = self.full.get(row_id) or self.rows.get(row_id) or {}
            text = ' '.join((str(v) for v in record.values() if isinstance(v, str)))
            text += ' ' + '. '.join((f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)))
            marks[row_id] = scan_fit_marks(text)
            for part, codes in marks[row_id]['also']:
                notes.setdefault(part, set()).update(codes)
        return {'marks': marks, 'notes': notes}

    def compat_set(self, row_id: str, index: dict[str, Any]) -> set[str]:
        mark = index['marks'].get(row_id) or {}
        found = set(mark.get('works') or [])
        if mark.get('own'):
            found |= index['notes'].get(mark['own'], set())
        return found

    def ids_for_item(self, number: int) -> list[str]:
        cat_key = self.shelf_field()
        item = self.items[number - 1]
        return [i for i in self.seen_order if cat_key and self.item_owns(item, str((self.rows.get(i) or {}).get(cat_key) or ''))]

    def compat_faults(self) -> list[str]:
        if not self.bound:
            return []
        index = self.compat_index()
        cart = self.cart_product_ids()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.ids_for_item(main)]
            parts = [i for i in cart if i in self.ids_for_item(part)]
            if len(devices) != 1 or not parts:
                continue
            codes = set(index['marks'].get(devices[0], {}).get('codes') or [])
            if not codes:
                continue
            for line in parts:
                fits = self.compat_set(line, index)
                if fits and (not fits & codes):
                    good = [i for i in self.ids_for_item(part) if self.compat_set(i, index) & codes and (not self.is_sold_out(i)) and (i not in cart)][:4]
                    faults.append(f"{line} ({self.items[part - 1].name}) works with {', '.join(sorted(fits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})" + (f"; seen parts of that kind that work with it: {', '.join(good)}" if good else '; no seen part of that kind works with it, so look for one or switch the device: ' + self.compat_plan(short=True)))
        return faults

    def compat_plan(self, short: bool=False) -> str:
        if not self.bound:
            return ''
        index = self.compat_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.ids_for_item(p):
                    if self.is_sold_out(i):
                        continue
                    for code in self.compat_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            devices = {code: [i for i in self.ids_for_item(main) if code in (index['marks'].get(i, {}).get('codes') or [])][:2] for code, _ in ranked}
            text = '; '.join((f'{code}: ' + ', '.join((f'{n} {self.items[p - 1].name}' for p, n in sorted(per.items()))) + (f" (device seen: {', '.join(devices[code])})" if devices[code] else ' (device not seen yet)') for code, per in ranked))
            if text:
                lines.append(f"codes the {self.items[main - 1].name} candidates' parts name -> {text}")
        if not lines:
            return ''
        return ' | '.join(lines) if short else "[fit plan: a device fits a part when the part's text names the device's model code or another name it is sold as; codes are not searchable, so list the device's category and read the codes on its rows] " + ' | '.join(lines)

    def cart_line_faults(self) -> list[str]:
        cat_key = self.shelf_field()
        ids = [i for i in self.cart_product_ids() if (self.units_held(i) or 0) > 0]
        if not ids or not cat_key:
            return []
        label = {i: str((self.rows.get(i) or {}).get(cat_key) or '') for i in ids}
        known = {n for n, item in enumerate(self.items, 1) if any((self.item_owns(item, str(r.get(cat_key) or '')) for r in self.rows.values()))}
        owner = {i: [n for n, item in enumerate(self.items, 1) if self.item_owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(self.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.units_held(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
        if len(known) == len(self.items):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.is_sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.listing_price(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all((p is not None for p in prices.values())):
            if self.brief.get('budget_scope') == 'each':
                over = [i for i in ids if self.budget < prices[i] <= 10 * self.budget]
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in over]
            else:
                total = sum((prices[i] * self.units_held(i) for i in ids))
                if self.budget + 1e-09 < total <= 10 * self.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
        return faults

    def qty_fixes(self) -> list[tuple[str, int, int]]:
        cat_key = self.shelf_field()
        if not cat_key:
            return []
        out = []
        for row_id in self.cart_product_ids():
            have = self.units_held(row_id) or 0
            owners = [n for n, item in enumerate(self.items, 1) if self.item_owns(item, str((self.rows.get(row_id) or {}).get(cat_key) or ''))]
            mine = [i for i in self.cart_product_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.item_owns(self.items[owners[0] - 1], str((self.rows.get(i) or {}).get(cat_key) or ''))]
            if have > 0 and len(owners) == 1 and (len(mine) == 1) and (have != self.items[owners[0] - 1].qty) and (not self.is_sold_out(row_id)):
                out.append((row_id, have, self.items[owners[0] - 1].qty))
        return out

    def fix_qtys(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and 'string' in schema_kinds(v) and ('enum' not in v)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in schema_kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and (len(v['enum']) >= 2)), None)
        if not name or not target or (not count):
            return []
        calls: list[dict[str, Any]] = []
        for row_id, have, want in wrong:
            if self.cart_is_qty_set(name):
                calls.append({target: row_id, count: want})
            elif mode:
                options = [str(o) for o in props[mode]['enum']]
                off = next((o for o in options if re.search('remov|delet|drop|clear|take', o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: row_id, mode: off, count: have}, {target: row_id, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.steps_left():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = coerce_args(args, schema, 'args', faults)
            if faults:
                break
            reply = self.invoke_tool(name, args)
            if reply is None or self.finished:
                break
            self.track_cart_edit(name, args)
            self.cart_said = jdump(reply.get('observation'))
            self.on_observation(name, reply.get('observation'))
            done.append(jdump(args) + ' -> ' + self.render_obs(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    def verdict_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.version}'

    def judge_cart(self, ids: list[str] | None=None, limit: float=BUDGET_SOFT) -> str:
        lead = ids is not None
        ids = ids if lead else self.cart_product_ids()
        key = self.verdict_key(ids)
        if key in self.checked or not self.judge_pool(limit) or (not ids):
            return ''
        self.checked.add(key)
        prompt = CART_JUDGE + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger_text()
        if lead:
            quantity = '; '.join((f'item {n}: quantity {it.qty}' for n, it in enumerate(self.items, 1)))
            prompt += '\n\nCART (planned, judge these listings as if they were carted at the requested quantities: ' + quantity + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.describe_cart()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join((f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.is_sold_out(i) else '') + self.listing_blob(i) for i in ids))
        if self.brief.get('pick') in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.other_listings(ids, 15)
        got = self.json_model_call(prompt, self.judge_pool(limit), 3000)
        self.trace.append({'kind': 'check', 'cart': ids, 'got': got})
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in as_list(got.get('problems'))][:5]
        better = [str(b) for b in as_list(got.get('better')) if str(b) in self.rows and (not self.is_sold_out(str(b)))][:4]
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def follow_links(self) -> tuple[str, bool]:
        live = {need for _, need in self.open_requirements()}
        pending = [need for _, need in self.refs if need in live and need not in self.ref_values]
        tool = self.roles.get('offer') or self.roles.get('open')
        target = self.solo_string_arg(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.trails < (6 if self.steps_left() > 15 else 4) and (self.cost < BUDGET_HARD) and (self.time_left() > CLOSING):
            self.trails += 1
            records = []
            for row_id in reversed(self.seen_order):
                record = self.full.get(row_id) or self.rows.get(row_id) if row_id in self.opened else None
                if record and len(records) < 45:
                    records.append('OPENED ' + jdump(trim_tree(record, 300))[:1400])
            for row_id in reversed(self.seen_order):
                if len(records) >= 60:
                    break
                if row_id not in self.opened or not (self.full.get(row_id) or self.rows.get(row_id)):
                    records.append('ROW ' + jdump(trim_tree(self.rows.get(row_id), 80))[:300])
            prompt = LINK_TRAIL + '\n\nREQUIREMENTS WITH A VALUE ON ANOTHER LISTING:\n' + '\n'.join(pending)
            prompt += '\n\nRECORDS (newest first):\n' + '\n'.join(records)
            got = self.json_model_call(prompt, [MODEL_FAST] + ([MODEL_DEEP] if self.cost + DEEP_ROOM <= BUDGET_HARD else []), 4500) or {}
            self.trace.append({'kind': 'trail', 'pending': pending, 'got': trim_tree(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in as_list(got.get('chains')):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get('need') or '').strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in as_list(chain.get('links')) if isinstance(l, dict)]
                ids = [str(l.get('id') or '').strip() for l in links]
                if chain.get('done') is True and links and all((i in self.opened for i in ids)) and str(chain.get('value') or '').strip():
                    self.ref_values[need] = str(chain['value']).strip()
                    self.version += 1
                    continue
                if links and all((i in self.opened for i in ids)) and (not str(links[-1].get('find') or '').strip()):
                    self.ref_values[need] = str(chain.get('value') or '').strip() or f'as stated on {ids[-1]}'
                    self.version += 1
                    continue
                closed = [i for i in ids if i in self.rows and i not in self.opened]
                if closed:
                    reach += [i for i in closed if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.rows), {})
                    about = str(gap.get('about') or '') or 'the listing it points at'
                    words = str(gap.get('find') or '').strip()
                    lost.append(f'{need} -> find {about}' + (f' (search: {words})' if words else ''))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.ref_values]
            if not reach and gaps and tool and target and (self.steps_left() > 12) and (self.link_searches < 3):
                reach = self.locate_link(*gaps[0])
                if self.trace and self.trace[-1].get('kind') == 'link_known':
                    seen.append(f"{self.trace[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or (not target) or (self.steps_left() < len(reach) + 4) or self.finished:
                break
            for row_id in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = coerce_args({target: row_id}, self.schemas.get(tool), 'args', faults)
                if faults:
                    break
                reply = self.invoke_tool(tool, args)
                if reply is None or self.finished:
                    break
                self.on_observation(tool, reply.get('observation'))
                seen.append(f'{row_id}: ' + self.render_obs(reply.get('observation'))[:1500])
        text = ''
        if seen:
            text += '[World opened the pages a requirement points at]\n' + '\n'.join(seen) + '\n'
        if self.ref_values:
            text += '[values read from the pages requirements point at] ' + '; '.join((f'{k} -> {v}' for k, v in self.ref_values.items())) + '\n'
        hold = bool(lost) and self.trail_blocks < (3 if self.steps_left() > 15 else 2) and (self.steps_left() > 5)
        if hold:
            self.trail_blocks += 1
            text += 'The order counts only after every page a requirement points at has been opened in full (a search row is not enough), following each page that refers to a further listing. Still to find and open: ' + ' | '.join(lost[:3]) + ". Search for it by its shop's name together with the values its description gives (the shop's other listings of that kind are not it), open the row of that shop whose page shows those values, apply the value its page gives, then check out."
        return (text, hold)

    def rows_by_shop(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for row_id in ids:
            row = self.rows.get(row_id) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if row_id not in self.opened and any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(row_id)
        return out

    def shop_opened(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for row_id in self.opened:
            row = self.rows.get(row_id) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I)]
            if any((re.search('(?<![a-z0-9])' + re.escape(n.lower()) + '(?![a-z0-9])', low) for n in names)):
                out.append(row_id)
        return out

    def locate_link(self, about: str, words: str) -> list[str]:
        known: list[str] = []
        named = {w for row_id in self.rows_by_shop(about, list(self.rows)) + self.shop_opened(about) for k, v in (self.rows.get(row_id) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search('seller|shop|store|vendor|brand|maker|merchant', str(k), re.I) for w in re.findall('[a-z0-9]{3,}', v.lower())}
        values = {w for w in re.findall('[a-z0-9]+', (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for', 'of'}
        for row_id in self.shop_opened(about):
            stack, leaves = ([self.full.get(row_id) or self.rows.get(row_id) or {}], [])
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = ' '.join(leaves).lower()
            if values and all((re.search('(?<![a-z0-9])' + re.escape(v) + '(?![a-z0-9])', text) for v in values)):
                self.trace.append({'kind': 'link_known', 'id': row_id})
                return []
        plan = self.search_skel()
        memo = ' '.join(words.lower().split())
        if memo and memo in self.link_seen:
            self.link_searches += 1
            known = self.rows_by_shop(about, self.link_seen[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = coerce_args({**base, field: words}, self.schemas.get(tool), 'args', faults)
            if not faults:
                self.link_searches += 1
                reply = self.invoke_tool(tool, args)
                if reply is not None:
                    observation = reply.get('observation')
                    self.on_observation(tool, observation)
                    self.link_seen[memo] = [i for i, _ in self.extract_records(observation)]
                    known = self.rows_by_shop(about, self.link_seen[memo])
                    self.trace.append({'kind': 'link_search', 'words': words[:80], 'rows': known[:4]})
        kind = {w for w in re.findall('[a-z0-9]{3,}', (words or about).lower())} - named - {'the', 'and', 'with', 'from', 'that', 'comes', 'listing', 'page', 'its', 'for'}

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall('[a-z0-9]{3,}', jdump(self.rows.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.rows_by_shop(about, list(reversed(self.seen_order)))))[:2]

    def price_tool(self) -> str | None:
        return self.roles.get('offer') or self.roles.get('open')

    def refresh_offers(self) -> str:
        tool = self.price_tool()
        target = self.solo_string_arg(tool)
        if not tool or not target:
            return ''
        since = max(self.step - 2, self.refused_at)
        stale = [i for i in self.cart_product_ids() if self.offer_seen.get(i, -99) < since and i not in self.gone]
        if not stale or self.steps_left() < len(stale) + 3:
            return ''
        lines = []
        for row_id in stale:
            faults: list[str] = []
            args = coerce_args({target: row_id}, self.schemas.get(tool), 'args', faults)
            if faults:
                break
            before = dict(self.rows.get(row_id) or {})
            reply = self.invoke_tool(tool, args)
            if reply is None or self.finished:
                break
            observation = reply.get('observation')
            self.offer_seen[row_id] = self.step
            self.on_observation(tool, observation)
            after = self.rows.get(row_id) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and (not isinstance(v, (dict, list)))]
            if moved:
                lines.append(f"{row_id} changed ({', '.join(moved[:4])}): {jdump(trim_tree(observation, 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.version += 1
        return '\n'.join(lines)

    def is_sold_out(self, row_id: str) -> bool:
        return row_id in self.unavailable or row_id in self.gone

    def suggest_swap(self) -> str:
        lost = [i for i in self.cart_product_ids() if i not in self.gone and self.is_sold_out(i)]
        if not lost:
            return ''
        for row_id in lost:
            self.gone[row_id] = self.step
        self.leads = [i for i in self.leads if i not in lost]
        self.version += 1
        picks = self.swap_candidates(lost)
        text = f"[World] Sold out now: {', '.join(lost)} (the store shows it cannot be bought). It will not come back, so do not open it again."
        if picks and self.steps_left() <= 6:
            return text + f" Few steps are left: in this turn add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other listings already read that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.brief.get('pick') == 'cheapest':
            return text + f" Listings already read that appear to fit that item instead, cheapest first: {', '.join(picks)}. The pick rule is the cheapest, so unless you have already seen the whole fitting set, run one search under {picks[0]}'s price (the search's maximum price, if it has one) before you swap the sold-out line for the cheapest fitting listing."
        if picks:
            return text + f" Listings already read that appear to fit that item instead: {', '.join(picks)}. In this turn take the sold-out line out of the cart and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line from the cart and add the best other listing that fits; search for one if none you have read fits.'

    def locate_sold(self) -> str:
        ids = [i for i in self.cart_product_ids() if i not in self.gone]
        if any((self.is_sold_out(i) for i in ids)):
            return ''
        if len(ids) == 1:
            self.unavailable.add(ids[0])
            return ''
        return self.refresh_offers() if ids else ''

    def swap_candidates(self, lost: list[str]) -> list[str]:
        if not self.judge_pool():
            return []
        records = []
        for row_id in reversed(self.seen_order):
            if row_id in lost or row_id in self.gone or self.is_sold_out(row_id):
                continue
            records.append(jdump(trim_tree(self.full.get(row_id) or self.rows.get(row_id), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP_HINT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger_text()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join((f'{i}: {self.listing_blob(i)}' for i in lost)) + '\n\nCART NOW:\n' + self.describe_cart()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.json_model_call(prompt, self.judge_pool(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in as_list((got or {}).get('fits')):
            row_id = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if row_id in self.rows and row_id not in lost and (row_id not in self.gone) and (row_id not in self.cart_product_ids()) and (row_id not in [i for _, i in fits]):
                fits.append((float(price) if isinstance(price, (int, float)) and (not isinstance(price, bool)) else float('inf'), row_id))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.brief.get('pick') == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    def recovery_search(self, missing: list[int] | None=None) -> str:
        """YesYesYes R16 lever, adapted: before a decline stands, hunt fresh rows for unfitted items.

        Keeps cart/checkout room. Does not replace dialect or process gates.
        """
        if self.recovered or self.finished:
            return ''
        if missing:
            need = list(missing)
        else:
            need = []
            for n in range(1, len(self.items) + 1):
                ids = self.lead_items.get(str(n)) or self.lead_items.get(n) or []
                if not any((i for i in ids if not self.is_sold_out(i))):
                    need.append(n)
            if not need:
                need = list(range(1, len(self.items) + 1))
        keep = self.reserve_steps() + len(self.items) + 2
        room = min(RECOVERY_STEPS, self.steps_left() - keep)
        if room < 3 or self.time_left() < CLOSING + 25:
            return ''
        self.recovered = True
        before = len(self.full)
        searched = self.search_by_value(need[:3])
        # Open a few unread promising rows of each missing kind if search found any.
        opened = 0
        cat_key = self.shelf_field()
        for number in need[:3]:
            if self.steps_left() <= keep + 1:
                break
            item = self.items[number - 1]
            cand = [
                i for i in reversed(self.seen_order)
                if i not in self.full and not self.is_sold_out(i)
                and (not cat_key or self.item_owns(item, str((self.rows.get(i) or {}).get(cat_key) or '')))
            ][:2]
            got = self.pull_attributes(cand, keep)
            opened += len(got)
        self.trace.append({'kind': 'recovery', 'missing': need[:3], 'opened': opened, 'full_delta': len(self.full) - before})
        if len(self.full) <= before and not searched:
            return ''
        return (
            'World ran a recovery search before ending without a purchase: '
            + (searched or f'opened {opened} unread listing(s) of the requested kind')
            + '. Re-judge the new rows; end without a purchase only if each still fails.'
        )

    def run_audit(self) -> str:
        if self.audited or not self.judge_pool():
            return ''
        self.audited = True
        got = self.run_audit_turn()
        named = self.audit_matches(got)
        if not named:
            rows = [str(i) for i in as_list((got or {}).get('open')) if str(i) in self.rows and str(i) not in self.opened and (not self.is_sold_out(str(i)))]
            if rows and self.open_listings(rows):
                got = self.run_audit_turn()
                named = self.audit_matches(got)
        if not named:
            # R16 DNF edge: hunt before giving up when audit finds nothing.
            note = self.recovery_search()
            if note:
                got = self.run_audit_turn()
                named = self.audit_matches(got)
                if named:
                    self.lead_items = named
                    self.leads = [i for ids in named.values() for i in ids]
                    return note + ' A careful review now thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + '. Open them and buy what truly fits.'
                return note
            return ''
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        return 'A careful review thinks these listings may fit: ' + '; '.join((f"item {k}: {', '.join(v)}" for k, v in named.items())) + (f" ({got.get('reason')})" if got.get('reason') else '') + '. Open them and buy what truly fits; end without a purchase only if each one fails.'

    def run_audit_turn(self) -> dict[str, Any] | None:
        records = []
        for row_id in reversed(self.seen_order):
            if self.is_sold_out(row_id):
                continue
            record = self.full.get(row_id) if row_id in self.full else None
            records.append('OPENED ' + jdump(trim_tree(record, 200))[:1200] if record else 'ROW ' + jdump(trim_tree(self.rows.get(row_id), 120))[:500])
            if len(records) >= 30:
                break
        if not records:
            return None
        prompt = AUDIT_PROMPT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger_text()
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.json_model_call(prompt, self.judge_pool(), 4000)
        self.trace.append({'kind': 'audit', 'got': got})
        return got

    def audit_matches(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get('fits')
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in as_list(v) if str(i) in self.rows and (not self.is_sold_out(str(i)))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_listings(self, rows: list[str]) -> bool:
        many, joiner = (self.batch_arg_key(), self.roles.get('compare'))
        tool = self.attr_tool()
        target = self.solo_string_arg(tool)
        if many and joiner and (not self.batch_attr_missing()):
            plan = [(joiner, {many: rows[:self.batch_capacity()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.steps_left() < len(plan) + 5:
            return False
        before = len(self.full)
        for name, args in plan:
            faults: list[str] = []
            args = coerce_args(args, self.schemas.get(name), 'args', faults)
            if faults:
                return False
            reply = self.invoke_tool(name, args)
            if reply is None or self.finished:
                break
            self.on_observation(name, reply.get('observation'))
        self.trace.append({'kind': 'audit_open', 'rows': rows, 'opened': len(self.full) - before})
        return len(self.full) > before

    def build_cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.roles.get('cart') or ''
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        for key, spec in props.items():
            shape = spec.get('items') if isinstance(spec, dict) and 'array' in schema_kinds(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get('properties'), dict):
                inner = shape['properties']
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in schema_kinds(s) and ('enum' not in s)), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in schema_kinds(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and 'string' in schema_kinds(s) and ('enum' not in s)]
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in schema.get('required') or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and 'integer' in schema_kinds(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2)), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for row_id in drop:
            have = self.cart_lines.get(row_id, 1)
            if mode:
                calls.append({target: row_id, mode: props[mode]['enum'][1], **({count: have} if count else {})})
            elif count and self.cart_is_qty_set(name):
                calls.append({target: row_id, count: 0})
            elif count and isinstance(props[count].get('minimum'), int) and (props[count]['minimum'] <= -have):
                calls.append({target: row_id, count: -have})
            else:
                return None
        for row_id, want in lines.items():
            have = self.cart_lines.get(row_id, 0) if row_id in self.cart_product_ids() else 0
            if have == want:
                continue
            if have and (not self.cart_is_qty_set(name)):
                return None
            calls.append({target: row_id, **({mode: props[mode]['enum'][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def execute_checkout(self) -> str:
        cart, buy = (self.roles.get('cart'), self.roles.get('buy'))
        if self.completed or not self.confirmed or self.is_sold_out(self.confirmed) or (not cart) or (not buy):
            return ''
        late = self.late_question()
        if late:
            # Hand control back to the model with the new requirement; do not mark completed.
            self.queued.append(late)
            return 'Not run (World asked first): ' + late
        self.completed = True
        cat_key = self.shelf_field()

        def kind(row_id: str) -> str:
            return str((self.rows.get(row_id) or {}).get(cat_key) or '').strip().lower() if cat_key else row_id
        plan: dict[str, int] = {}
        for number, ids in self.lead_items.items():
            index = as_int(number)
            pick = next((i for i in ids if not self.is_sold_out(i)), None)
            if pick and index and (1 <= index <= len(self.items)) and (kind(pick) not in {kind(p) for p in plan}):
                plan[pick] = self.items[index - 1].qty
        if self.confirmed not in plan:
            return ''
        now = self.cart_product_ids()
        keep = {i: self.cart_lines.get(i, 1) for i in now if i not in plan and (not self.is_sold_out(i)) and (kind(i) not in {kind(p) for p in plan})}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.items):
            return ''
        lines = {**keep, **plan}
        calls = self.build_cart_calls(lines, drop)
        if not calls or len(calls) + 1 > self.steps_left():
            return ''
        self.pull_attributes(list(lines), len(calls) + 3)
        self.judge_cart(list(lines), BUDGET_HARD)
        if not self.verdicts.get(self.verdict_key(list(lines))):
            return ''
        done = []
        for args in calls:
            faults: list[str] = []
            args = coerce_args(args, self.schemas.get(cart), 'args', faults)
            if faults:
                return ''
            reply = self.invoke_tool(cart, args)
            if reply is None or self.finished:
                return ''
            self.track_cart_edit(cart, args)
            self.on_observation(cart, reply.get('observation'))
            done.append(jdump(args))
        faults = []
        args = coerce_args(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
        if faults:
            return ''
        replies = []
        for attempt in range(2):
            if attempt:
                self.refresh_offers()
                self.judge_cart(list(lines), BUDGET_HARD)
                if not self.verdicts.get(self.verdict_key(list(lines))) or self.steps_left() < 1 or self.finished or any((self.is_sold_out(i) for i in lines)):
                    break
            go, said = self.run_finish_guard('order')
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.invoke_tool(buy, args)
            self.trace.append({'kind': 'completed', 'lines': lines, 'drop': drop, 'reply': trim_tree(reply, 200)})
            if reply is None:
                break
            self.buys += 1
            if not reply.get('finished'):
                self.refused_at = self.step
            self.on_observation(buy, reply.get('observation'))
            replies.append(self.render_obs(reply.get('observation')))
            if self.finished or reply.get('finished'):
                break
        if not replies:
            return ''
        return 'Not run (World acted instead): the independent check confirmed ' + self.confirmed + ', so World set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ' | then: '.join(replies)

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('buy'):
            return False
        fixed = self.brief.get('buy_args') if isinstance(self.brief.get('buy_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def is_decline(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.roles.get('refuse'):
            return False
        fixed = self.brief.get('refuse_args') if isinstance(self.brief.get('refuse_args'), dict) else {}
        return all((args.get(k) == v for k, v in fixed.items()))

    def available_tools(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        ask = self.roles.get('ask')
        keep: set[str] | None = None
        ends = {self.roles.get('buy'), self.roles.get('refuse')} - {None}
        secs = self.time_left()
        if left <= self.reserve_steps() or secs <= CLOSING:
            keep = set(ends)
        elif left <= self.reserve_steps() + 3 or secs <= CLOSING + 25:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view')}
        elif left <= len(self.items) + 5 and self.buys == 0:
            keep = set(ends) | {self.roles.get('cart'), self.roles.get('cart_view'), self.roles.get('offer')}
        tools = [t for t in self.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def mute_decline_text(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get('properties') or {}
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in schema_kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get('required') or []):
                continue
            text = str(value or '')
            named = any((row_id in text for row_id in self.rows)) or self.brief.get('quiet_decline') is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def normalize_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.roles.get(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.id_key:
            return (args, '')
        props = (self.schemas.get(name) or {}).get('properties') or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in schema_kinds(v) or 'array' in schema_kinds(v)) and ('enum' not in v)]
        if len(free) != 1 or (name != self.roles.get('cart') and (name, free[0]) not in self.id_args):
            return (args, '')

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.rows:
                return (value.strip(), '')
            if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
                return (value, '')
            low = value.strip().lower()
            hits = [i for i, row in self.rows.items() if any((k != self.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items()))]
            if len(hits) == 1:
                return (hits[0], '')
            if hits:
                return (value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id")
            return (value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.id_key} field)')
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return (args, problem)
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return (out, '')

    def cart_is_qty_set(self, name: str) -> bool:
        props = self.schemas.get(name, {}).get('properties') or {}
        if any((isinstance(s, dict) and isinstance(s.get('enum'), list) and (len(s['enum']) >= 2) for s in props.values())):
            return False
        return any((isinstance(s, dict) and 'integer' in schema_kinds(s) and (s.get('minimum') == 0) for s in props.values()))

    def track_cart_edit(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.step
        spec = self.schemas.get(name, {}).get('properties') or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and (value in options[1:]):
                removal = True
        loose_count = next((as_int(v) for v in args.values() if as_int(v) is not None and (not isinstance(v, str))), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        row_id = next((v for v in line.values() if isinstance(v, str) and v in self.rows), None)
                        count = next((as_int(v) for v in line.values() if as_int(v) is not None and (not isinstance(v, str))), 1)
                        if row_id:
                            lines.append((row_id, count))
                    elif isinstance(line, str) and line in self.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.rows:
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.cart_is_qty_set(name)
        for row_id, count in lines:
            have = self.cart_lines.get(row_id, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[row_id] = have
            else:
                self.cart_lines.pop(row_id, None)

    def render_obs(self, observation: Any) -> str:
        text = jdump(trim_tree(observation, VIEW_CHARS))
        if len(text) > OBS_KEEP:
            text = text[:OBS_KEEP] + '...(cut)'
        return text

    def batch_arg_key(self) -> str | None:
        schema = self.schemas.get(self.roles.get('compare') or '') or {}
        props = schema.get('properties') or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in schema_kinds(v)), None)
        if not target or any((k != target for k in schema.get('required') or [])):
            return None
        return target

    def batch_capacity(self) -> int:
        spec = ((self.schemas.get(self.roles.get('compare') or '') or {}).get('properties') or {}).get(self.batch_arg_key() or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def merge_open_calls(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        opener, joiner, target = (self.roles.get('open'), self.roles.get('compare'), self.solo_string_arg(self.roles.get('open')))
        many = self.batch_arg_key()
        if not opener or not joiner or (not target) or (not many) or (opener == joiner):
            return calls
        if self.batch_attr_missing():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = as_object(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            row_id = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and (set(args) == {target}) and (row_id in self.rows) and (row_id not in self.gone):
                picks.append((index, row_id))
        picks = picks[:self.batch_capacity()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': jdump({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def dispatch_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.merge_open_calls(calls)
        for index, call in enumerate(calls):
            allowed = {t['function']['name'] for t in self.available_tools()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: World opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished:
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.roles.get('ask'):
                out.append((cid, "Not run (no step used): World's interviewer handles every message to the shopper."))
                continue
            if name not in self.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = as_object(raw) if (raw or '').strip() else {}
                if args is None:
                    out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                    continue
            faults: list[str] = []
            args = coerce_args(args, self.schemas[name], 'args', faults)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.normalize_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            many = self.batch_arg_key() if name == self.roles.get('compare') else None
            if many and isinstance(args.get(many), list) and (len(args[many]) > self.batch_capacity()) and self.compare_bad:
                room = self.batch_capacity()
                trimmed = f"\n[World] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.solo_string_arg(name)
            if target and name in (self.roles.get('open'), self.roles.get('offer')) and (str(args.get(target)) in self.gone):
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.roles.get('cart') and any(((as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str))):
                view = jdump(self.cart_view) if self.cart_seen_at >= self.cart_at else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.rows and (v not in self.cart_lines) and (v not in view) and (v not in self.cart_said)]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.describe_cart()))
                    continue
            ending = self.is_checkout(name, args) or self.is_decline(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.steps_left() <= self.reserve_steps() and (self.roles.get('buy') or self.roles.get('refuse')):
                out.append((cid, "Not run (no step used): the last steps are kept for checking out (with the shopper's approval of the basket when it is due) or ending without a purchase."))
                continue
            mark = name + jdump(args)
            if not ending and name != self.roles.get('cart') and (mark in self.recent):
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = {self.roles.get(r) for r in ('search', 'open', 'offer', 'compare')} - {None}
            cached = self.read_cache.get(mark)
            if name in readers and cached and (cached[0] == self.version):
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and (name != self.roles.get('compare')) and self.already_viewed(name, str(args.get(target))):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.listing_blob(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.already_viewed(name, str(i))]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join((f'{i}: {self.listing_blob(str(i))}' for i in args[many]))[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[World] Left out {', '.join((str(i) for i in args[many] if i not in keep))}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + jdump(args)
            if ending and self.steps_left() > 2 and self.time_left() > CLOSING - 25:
                if self.is_checkout(name, args):
                    wider = self.expand_search_hint()
                    if wider:
                        out.append((cid, wider + '\nCart now: ' + self.describe_cart()))
                        continue
                    reread = self.refresh_offers()
                    if self.note_sold:
                        self.locate_sold()
                    if self.cart_touched and self.cart_seen_at < self.cart_at and self.roles.get('cart_view') and (self.steps_left() > 3):
                        faults = []
                        view_args = coerce_args({}, self.schemas.get(self.roles['cart_view']), 'args', faults)
                        view = self.invoke_tool(self.roles['cart_view'], view_args) if not faults else None
                        if view is not None:
                            self.cart_view = view.get('observation')
                            self.cart_seen_at = self.step
                            change = self.on_observation(self.roles['cart_view'], self.cart_view)
                            if change:
                                self.queued = []
                                out.append((cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.describe_cart() + '\nApply the change, fix the cart if needed, then check out.'))
                                continue
                    swap = self.suggest_swap()
                    held = [i for i in self.cart_product_ids() if i in self.gone] if self.held < 2 else []
                    if swap or held:
                        self.held += 0 if swap else 1
                        out.append((cid, 'Not run yet (no step used). ' + (swap or f"{', '.join(held)} in the cart is sold out: replace that line first.") + '\nCart now: ' + self.describe_cart()))
                        continue
                    chain, hold = self.follow_links()
                    if hold:
                        out.append((cid, 'Not run yet (no step used). ' + chain + '\nCart now: ' + self.describe_cart()))
                        continue
                    misfit = self.compat_faults() if self.fit_holds < 2 and self.steps_left() > self.reserve_steps() + 3 else []
                    if misfit:
                        self.fit_holds += 1
                        self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
                        out.append((cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.\nCart now: ' + self.describe_cart()))
                        continue
                    faults = self.cart_line_faults() if self.line_holds < 2 and self.steps_left() > self.reserve_steps() + 2 else []
                    fixed = []
                    if faults and all((' is held at ' in f for f in faults)):
                        fixed = self.fix_qtys(self.qty_fixes())
                        faults = self.cart_line_faults() if fixed else faults
                        if fixed and (not faults):
                            self.line_holds += 1
                            out.append((cid, "Not run yet (no step used). World set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.\nCart now: ' + self.describe_cart()))
                            continue
                    if faults:
                        self.line_holds += 1
                        self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
                        out.append((cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out.\nCart now: " + self.describe_cart()))
                        continue
                    read = self.pull_attributes(self.cart_product_ids(), self.reserve_steps() + 3) + self.probe_cheaper()
                    verdict = self.judge_cart()
                    if verdict:
                        seen = "World read these listings' attributes for the check (one step each):\n" + '\n'.join(read) + '\n' if read else ''
                        out.append((cid, 'Not run yet. ' + chain + seen + verdict + '\nCart now: ' + self.describe_cart()))
                        continue
                    if reread and self.cost > BUDGET_SOFT:
                        out.append((cid, "Not run yet (no step used). World re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'))
                        continue
                elif self.time_left() > CLOSING + 15 and self.steps_left() > 5:
                    review = self.run_audit()
                    if review:
                        out.append((cid, 'Not run yet (no step used). ' + review))
                        continue
                    if self.leads and self.steps_left() > 4 and (self.refuse_blocks == 0):
                        self.refuse_blocks = 1
                        self.trace.append({'kind': 'refuse_blocked', 'leads': list(self.leads)})
                        out.append((cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, put the best fitting one in the cart (respecting the pick rule) and check out. End without a purchase only if each of them clearly fails a requirement; then say which one in your text and send the ending call again.'))
                        continue
                    if self.leads and self.steps_left() > 4 and (self.refuse_blocks == 1):
                        self.refuse_blocks = 2
                        first = self.leads[0]
                        self.judge_cart([first], BUDGET_HARD)
                        if self.verdicts.get(self.verdict_key([first])) and first in self.leads:
                            self.confirmed = first
                            self.trace.append({'kind': 'refuse_blocked', 'confirmed': first})
                            out.append((cid, f"Not run yet (no step used). An independent check of {first}'s records says it meets every requirement. Add it to the cart at the requested quantity and check out; end without a purchase only if its record clearly breaks a requirement."))
                            continue
            if self.is_decline(name, args) and self.steps_left() <= 5 and (not self.late_held) and self.time_left() > CLOSING + 15:
                lines = [i for i in self.built if not self.is_sold_out(i)]
                fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
                missing = [i for i in lines if i not in self.cart_product_ids() or (i in self.dropped and (not fresh))]
                if len(lines) >= len(self.items) and len(missing) + 1 <= self.steps_left():
                    self.late_held = True
                    self.trace.append({'kind': 'late_decline_held', 'lines': lines, 'missing': missing})
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo + ' End without a purchase only if your search showed that some requested item has no listing that meets its requirements; then send the ending call again.'))
                    continue
            if self.is_decline(name, args) and (not self.leads) and (not self.pushed) and (self.steps_left() > self.max_steps // 3) and self.time_left() > CLOSING + 15:
                self.pushed = True
                note = self.recovery_search()
                if note:
                    out.append((cid, 'Not run yet. ' + note))
                    continue
                out.append((cid, f"Not run yet (no step used). {self.steps_left()} steps are still left, so search in ways you have not tried before ending: browse the category of each requested item and page through it with the token or offset the store returns, try single-word value queries and the property's other names, and open the listings that might fit. End without a purchase only after that."))
                continue
            if self.is_decline(name, args):
                done = self.execute_checkout()
                if done:
                    out.append((cid, done + (' [episode finished]' if self.finished else '')))
                    continue
                args = self.mute_decline_text(name, args)
            if self.is_checkout(name, args):
                late = self.late_question()
                if late:
                    out.append((cid, 'Not run yet. ' + late + '\nCart now: ' + self.describe_cart()))
                    continue
            if ending and self.steps_left() >= 2 and self.time_left() > LAST_CALL + 12:
                go, said = self.run_finish_guard('order' if self.is_checkout(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.describe_cart()))
                    continue
                if said:
                    trimmed = '\n[World] ' + said + trimmed
            reply = self.invoke_tool(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and (len(args[many]) > 1):
                shown = jdump(observation)
                if not any((str(i) in shown for i in args[many])):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[World] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.batch_capacity()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.id_args.update(((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all((isinstance(x, str) and x in self.rows for x in (v if isinstance(v, list) else [v])))))
            self.ingest_payload(observation)
            if name == self.roles.get('compare'):
                rows = self.extract_records(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any((attr_payload(r) for _, r in rows))
            if name in readers - {self.roles.get('search')}:
                self.track_view_keys(name, observation)
            plain = {tool for job, tool in self.roles.items() if job != 'search'}
            paged = self.auto_page_search(name, args, observation) if name not in plain and (not ending) else ''
            if self.solo_string_arg(name) and name not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask')):
                listing = self.row_id(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(attr_payload(observation))
                    if name == self.roles.get('open') and (not self.facts_from[name]):
                        other = next((t for t in self.schemas if t != name and self.solo_string_arg(t) and (self.facts_from.get(t) is not False) and (t not in (self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse'), self.roles.get('ask'), self.roles.get('search')))), None)
                        if other:
                            self.roles['open'] = other
                    if not self.facts_from[name] and name != self.roles.get('open') and (not self.roles.get('offer')):
                        self.roles['offer'] = name
                    if name == self.price_tool():
                        self.offer_seen[listing] = self.step
            if name == self.roles.get('cart'):
                self.track_cart_edit(name, args)
                self.cart_said = jdump(observation)
                self.recent = []
                counts = [as_int(v) for v in args.values() if not isinstance(v, str) and as_int(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and (value in self.rows):
                        if counts[0] <= 0 and (not tree_has_id(observation, value)):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_product_ids() if not self.is_sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.roles.get('search') and self.extract_records(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
            if name == self.roles.get('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.step
            if self.is_checkout(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.step
            text = self.render_obs(observation) + paged + trimmed
            if name not in plain and (not ending):
                text += self.rank_cheap_rows([observation] + self.paged_obs) + self.ignore_cap(name, args, observation)
            base = text
            proof = self.evidence_lines(observation)
            if proof:
                text += '\n' + proof
            if name == self.roles.get('cart') and any((as_int(v) == 0 for v in args.values() if not isinstance(v, str))):
                if any((isinstance(v, str) and v in self.rows and tree_has_id(observation, v) for v in args.values())):
                    text += '\n[World] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.note_candidates(observation)
            if lines:
                meaning = self.apply_store_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.is_checkout(name, args) and (not reply.get('finished'))
            again = refused and any((line in self.extract_speech(observation) for line in self.sold_lines))
            loose = self.note_sold and (refused or (not self.note_named and (not self.is_checkout(name, args))))
            if (loose or again) and self.cart_product_ids() and (not self.finished):
                seen = self.locate_sold()
                if seen:
                    text += "\n[World re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.suggest_swap()
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on World's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.version, self.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def force_end(self, again: bool=False) -> str:
        if self.finished or self.broke or self.steps_left() <= 0:
            return ''
        buy, refuse = (self.roles.get('buy'), self.roles.get('refuse'))
        fresh = self.cart_view is not None and self.cart_seen_at >= self.cart_at
        holding = bool(self.cart_product_ids()) if fresh else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and (not fresh) and (self.steps_left() >= 2))):
            faults: list[str] = []
            args = coerce_args(dict(self.brief.get('buy_args') or {}), self.schemas.get(buy), 'args', faults)
            if not faults:
                if self.steps_left() >= 2:
                    self.run_finish_guard('order')
                if self.finished or self.steps_left() <= 0:
                    return ''
                reply = self.invoke_tool(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': trim_tree(reply, 200)})
                if self.finished or self.steps_left() <= 0:
                    return ''
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.step
                    self.on_observation(buy, reply.get('observation'))
                    return 'the checkout was refused: ' + self.render_obs(reply.get('observation'))
        if refuse:
            faults = []
            args = coerce_args(self.mute_decline_text(refuse, dict(self.brief.get('refuse_args') or {})), self.schemas.get(refuse), 'args', faults)
            if not faults:
                if self.steps_left() >= 2:
                    self.run_finish_guard('end')
                if self.finished or self.steps_left() <= 0:
                    return ''
                reply = self.invoke_tool(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': trim_tree(reply, 200)})
                if reply is not None and (not self.finished):
                    self.on_observation(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.render_obs(reply.get('observation'))
        return ''

    def trim_chat(self) -> None:
        size = sum((len(jdump(m.get('content'))) for m in self.chat))
        if size <= CHAT_KEEP:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if len(content) > 600:
                ids = re.findall('"([A-Za-z0-9][A-Za-z0-9_\\-:.]{5,40})"', content)
                ids = [i for i in dict.fromkeys(ids) if i in self.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def execute(self) -> list[dict[str, Any]]:
        if self.finished:
            return self.trace
        self.parse_briefing()
        self.boost_process_from_request()
        self.trace.append({'kind': 'brief', 'brief': self.brief})
        opening = self.view.get('opening')
        if opening is not None:
            self.ingest_payload(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.render_obs(opening)
        intro += '\n\nStart by searching for each item in its category and opening several listings in full, so the interviewer can see which properties these listings carry.\n\n' + self.progress_banner()
        self.chat = [{'role': 'system', 'content': PLAYBOOK}, {'role': 'user', 'content': intro}]
        self.drive_loop(self.loop_cap, BUDGET_HARD)
        if not self.finished:
            turned = self.force_end(again=True)
            if turned and (not self.finished) and (not self.broke) and (not self.no_credit) and (self.steps_left() > 1) and (self.time_left() > CLOSING):
                self.chat.append({'role': 'user', 'content': '[World] World tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.progress_banner()})
                self.drive_loop(self.loop_calls + 10, min(SPEND_STOP, BUDGET_HARD + 0.01))
            if not self.finished:
                self.force_end()
        self.trace.append({'kind': 'end', 'finished': self.finished, 'steps': self.step, 'cost': round(self.cost, 5), 'calls': self.calls, 'time_left': round(self.time_left(), 1), 'out_of_time': self.out_of_time})
        return self.trace

    def drive_loop(self, loop_cap: int, spend_cap: float) -> None:
        idle = 0
        while (not self.finished and (not self.broke) and (self.steps_left() > 0) and (self.loop_calls < loop_cap)
               and (self.cost < spend_cap) and (self.time_left() > CLOSING)):
            if self.intake_ready():
                summary = self.run_intake()
                if summary:
                    self.chat.append({'role': 'user', 'content': summary + '\n' + self.progress_banner()})
                continue
            # Packed-race finish bias: once, when cart is full and time is mid-late.
            if (not self.finish_nudge and self.time_left() < CLOSING + 90 and self.buys == 0
                    and len(self.items) > 0 and len(self.cart_product_ids()) >= len(self.items)
                    and self.steps_left() <= self.reserve_steps() + 6):
                self.finish_nudge = True
                self.chat.append({'role': 'user', 'content': 'The cart already covers each requested item. Look at the cart once if needed, then check out now.\n' + self.progress_banner()})
            self.trim_chat()
            self.loop_calls += 1
            message = self.model_turn(self.chat, [MODEL_FAST, MODEL_FALLBACK], 3000, self.available_tools())
            if message is None:
                self.misses += 1
                if self.no_credit or self.calls >= self.call_cap or self.misses > 2 or self.out_of_time or self.cost >= SPEND_STOP:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else as_object(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'k{self.loop_calls}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': jdump(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = jdump(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.progress_banner()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.roles.get('cart'), self.roles.get('buy'), self.roles.get('refuse')} - {None}
            if not self.asked and self.roles.get('ask') and (self.asks_left() > 0) and any((c['function']['name'] in committing for c in calls)) and (self.steps_left() > 8):
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.run_intake()
                self.chat.append({'role': 'user', 'content': (summary or 'The interviewer had nothing to ask.') + '\n' + self.progress_banner()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.dispatch_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.progress_banner()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = WorldAgent(problem_data)
    except Exception as exc:
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return episode.execute()
    except Exception as exc:
        episode.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not episode.finished:
                episode.force_end()
        except Exception as again:
            episode.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return episode.trace
__all__ = ['agent_main']
