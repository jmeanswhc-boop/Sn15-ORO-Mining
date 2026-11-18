"""Shopping agent for ORO Bench store worlds.

Design: a tool-calling driver model works the store with the store's own tool list, and a layer of free guards decides
whether each call may spend a step. The guards rest on two pieces of our own: a fact reader that compares every opened
listing's property list with the shopper's needs (structured as property / test / values), and a shelf that keeps every
listing seen, so that cheaper unopened rows, conflicting facts and overlooked fits are caught before a step is spent.
Questions to the shopper are planned from the property table of the listings read and sent by the code itself.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

DRIVER = "anthropic/claude-haiku-5.5"
DRIVER_SPARE = "openai/gpt-6-luna"
REASONER = "anthropic/claude-sonnet-5.5"
HELPER = "anthropic/claude-haiku-5.5"
PLANNER = "google/gemini-3.8-flash"

EPISODE_SECONDS = 262.0
CLOSING_SECONDS = 70.0
LAST_CALL_SECONDS = 28.0
RESULT_CLIP = 15000
TEXT_CLIP = 420
TURN_CAP = 80
SPEND_SOFT = 0.035   # USD this task: past it the expensive seat gives way to the cheap one
SPEND_HARD = 0.044   # past it: FINISH NOW, only finishing tools offered
TRIM_AT = 80000      # driver prompt tokens: past it old tool results are clipped (100k+ prompts bill 5x)
SPEND_STOP = 0.050   # past it: the episode is finished at once (the run key is capped for all tasks together)
SETUP_SHARE = 0.015  # SB: the reading spend counted against the work budget; an overlong reading beyond it is not
SPEND_PER_LINE = 0.12   # SL: each list line past the first raises the spend lines by this share of them ...
SPEND_SCALE_MAX = 1.5   # ... up to this many times (a five-line fit basket needs reads a one-line request never makes)
DEFAULT_QUESTIONS = 3
_AP_S = r"[^.?!;]"
_AP_YES = r"(?:yes|approv\w*|ok(?:ay)?|okay\w*|ok'?d|o\.k\.?|go[- ]ahead|green[- ]light|confirm\w*|sign[- ]?off|consent|permission|agree\w*|blessing)"
_AP_BASKET = r"(?:basket|cart|order|picks?|selection|items|list|purchase|choices?)"
_AP_ORDER = r"(?:order\w*|buy\w*|bought|purchas\w*|check(?:ing)?[- ]?out|checkout|plac\w*|finali[sz]\w*|pay\w*|commit\w*|complet\w*|goes through)"
_AP_TOLD = r"(?:show|tell|give|send|quote|message)\s+(?:me|us)\s+(?:the |a |my )?(?:final |full |grand |whole |order )?(?:total|price|cost|sum|amount)\b"
# the request plainly asks to be shown the basket and to approve it before the order ("show me the final basket and ask for my
# yes"): one message is held back for it. An invitation to ask about needs before buying ("check with me before you buy, and
# I'll answer you straight") is not approval and must not match.
APPROVAL_RX = re.compile(
    r"\b(?:show|tell|read|list|send|give|run|walk|present|share|message)\b" + _AP_S + r"{0,60}\b" + _AP_BASKET + r"\b" + _AP_S + r"{0,80}\b(?:" + _AP_YES + r"|say[- ]so)\b"
    r"|\b(?:ask|wait|get|need|want|have|require|until)\b" + _AP_S + r"{0,25}\b(?:my|the shopper'?s?|your shopper'?s?|explicit|a|an|the final)\s+(?:final\s+|explicit\s+)?(?:yes|approval|ok(?:ay)?|go[- ]ahead|consent|confirmation|sign[- ]?off|agreement|permission|green[- ]light|blessing)\b"
    r"|\b(?:let me|I(?: want| need| would like|'d like| must| have| get) to|I'll|I will|I)\s+(?:first\s+)?(?:see|review|approve|ok(?:ay)?|look (?:at|over)|go over|check|sign off on|confirm)\b" + _AP_S + r"{0,30}\b" + _AP_BASKET + r"\b"
    r"|\b(?:don'?t|do not|never|no|nothing)\b" + _AP_S + r"{0,40}\b" + _AP_ORDER + r"\b" + _AP_S + r"{0,40}\b(?:until|unless|before|without)\b" + _AP_S + r"{0,30}\b(?:I|me|my)\b" + _AP_S + r"{0,30}\b(?:say|said|says|give|gave|approv\w*|ok(?:ay)?|okay\w*|agree\w*|confirm\w*|yes|go[- ]ahead|consent|sign\w*|nod|word|asking|checking)\b"
    r"|\b(?:confirm|check|clear|run|go over)\b\s+(?:(?:the |my |your |this |that )?(?:final |whole |full )?" + _AP_BASKET + r"|it|this|that|everything|them)\s+(?:with|by|past)\s+me\b" + _AP_S + r"{0,40}\b(?:before|first|ahead)\b"
    r"|\b(?:get|ask|wait for)\b" + _AP_S + r"{0,20}\b(?:me|my)\b" + _AP_S + r"{0,10}\b(?:to )?(?:confirm|approve|ok(?:ay)?|sign off)\b"
    r"|(?=" + _AP_S + r"*\b" + _AP_BASKET + r"\b)(?=" + _AP_S + r"*\bme\b)" + _AP_S + r"*\bfor (?:my |your shopper'?s )?(?:approval|confirmation|sign[- ]?off|ok(?:ay)?|go[- ]ahead|consent)\b"
    # AT: to be told the basket's total before the order is to be shown the basket ("Before placing the order, give me the total")
    r"|\bbefore\b" + _AP_S + r"{0,40}\b" + _AP_ORDER + r"\b" + _AP_S + r"{0,30}\b" + _AP_TOLD
    + r"|\b" + _AP_TOLD + _AP_S + r"{0,30}\bbefore\b" + _AP_S + r"{0,30}\b" + _AP_ORDER + r"\b", re.I)
# a sentence that could ask for the shopper's say before the order in other words (the shopper, the order, and a before or an
# approval word): the model reads it (none of the team's replay packs but pack-4 livetext has one)
APPROVAL_CUE_RX = re.compile(r"(?=" + _AP_S + r"*\b(?:me|my|I|I'm|I've|I'll|I'd|mine)\b)"
                             r"(?=" + _AP_S + r"*\b(?:basket|cart|order\w*|purchas\w*|buy\w*|bought|check(?:ing)?[- ]?out|checkout|pay\w*|plac\w*|finali[sz]\w*|picks?|selection)\b)"
                             r"(?=" + _AP_S + r"*\b(?:before|until|till|first|once|beforehand|prior|hold off|wait\w*|approv\w*|ok(?:ay)?|yes|confirm\w*|sign\w*|agree\w*|go[- ]ahead|consent|thumbs|nod|happy with)\b)" + _AP_S + r"+", re.I)
APPROVAL_PROMPT = """A shopper wrote a request to a shopping assistant. It contains the sentences below. Does any of them require the assistant to show the shopper the basket (or the order) it is about to place, and to get the shopper's approval, yes or go-ahead, before it places the order? Only an approval of the chosen basket or order counts. An invitation to ask the shopper questions about needs or details before buying (for example "ask me before you buy, and I'll answer you straight") is not an approval rule, and neither is an instruction about what to check in the store. Answer with ONE JSON object on a single line: {"approval": true} or {"approval": false}.

SENTENCES:
"""

OBLIGATION_PROMPT = """A shopper wrote the request below to a shopping assistant. Find its PROCESS RULES: sentences that tell the assistant how it must work (something it must do, or must not do, or an order of steps), not what to buy. A product need (a property, a count, a price limit, a shop, a stock state) is not a process rule. For each rule return one record:
- "quote": the rule's own words;
- "when": "before_choosing", "before_ordering" or "any_time";
- "act": what the assistant must do, in a few words;
- "witness": how one could see that it was done, one of:
  "order_message": a message to the shopper that shows the basket or order the assistant is about to place (its items, counts or total), sent before the order is placed. Any wording counts and anyone may be named (me, the customer, the shopper, the buyer): the shopper wants to see, approve, accept, agree to, sign off on, or be told the final selection or its total before it is bought;
  "question_message": a message that asks the shopper something (needs, wishes, choices). An invitation to ask before buying ("check with me about anything unclear before you buy") is this kind, not order_message;
  "view": opening a page (a listing, a seller, a referenced listing) before choosing or ordering;
  "other": anything else.
A rule the request rules out or calls unnecessary ("no need to run it past me", "just buy it, don't check back") is not a rule: leave it out. Leave out anything the request does not state.
Answer with ONE JSON object on a single line: {"rules": [records]}, or {"rules": []} when there is none.

REQUEST:
"""

# a request sentence that only states the step or message limit (the same wording the limit reading uses)
LIMIT_SENTENCE = re.compile(r"\b\d{1,4} (?:messages|questions|steps|moves|actions|tool calls|calls|turns)\b", re.I)

RULE_CHECK_PROMPT = """A shopping assistant is about to end its work with the ENDING below. The shopper's request set the WORKING RULES below (how the assistant must work), and the assistant's ACTIONS so far are listed in order. Check each rule against the actions. A rule about what to buy (a property, a count, a price, a shop, a stock state) is not checked here. These are handled by the assistant's own code, so count them as followed: the step limit, the limit on messages, buying nothing when nothing fits, and showing the basket for approval just before the order when a rule asks for that. List only the rules that are clearly NOT followed yet and that further actions could still put right before the ending, each with the next action(s) that would. A rule whose condition does not hold, a rule already followed, and a rule you cannot judge from the actions count as followed. Answer with ONE JSON object on a single line: {"missing": [{"quote": "the rule's words", "do": "the next action(s), in a few words"}]}, or {"missing": []}."""

REPLY_PROMPT = """A shopping assistant showed a shopper the basket it is about to order, and the shopper replied. Does the reply object to the basket: does it ask to change, add or remove something in it (an item, a count, a property, the price), or say not to place the order? An acknowledgement, thanks, a request to repeat or to send it again, a reply about something else, or an unclear reply is NOT an objection. Answer with ONE JSON object on a single line: {"change": true or false, "what": "the change asked for, in a few words, or empty"}."""

LOOP_PROMPT = """You are a careful shopping assistant working for one shopper in a simulated store. You act only through the tools you are given. The errand is scored all-or-nothing: it counts only if your final action is right in every way. The bought listing must meet every need the shopper stated, every need they reveal later and every change they announce; the quantity and the total price must be right; when the shopper asked for the cheapest (or the best), no other fitting listing may beat it; and every working rule in the request must be obeyed. When nothing in the store meets everything, the right ending is a decline, done the way the request says. When the request is a shopping list of several different products, the cart holds one line per product, each at its own quantity, all bought in ONE order; the budget is for the whole list, and each line must meet that product's own needs.

STEPS AND TOOLS
- Each tool call costs one step, failed calls included, and steps are limited; thinking is free. Calls that do not depend on each other can share one reply (for example opening several listings); they run in order. A checkout or a decline is always sent alone.
- Every store names its tools, arguments and result fields its own way. Use the argument names and allowed values of the schemas exactly. The first results show you where price, stock, the store section and the property list appear.
- You never write to the shopper yourself. Questions to the shopper are planned and sent by a separate planner, and the answers are shown to you with their meaning.
- Results that begin with "Not run (no step used)" come from my guards. They cost nothing; read them and act on them.

WHAT COUNTS AS EVIDENCE
- A need is met when the listing's own property list (its facts or attributes, shown when you open the listing) states it. Only when no key of the listing mentions that property at all (not even inside a longer key name) may its title or its options/variant field show it; the description never proves a need. A listing whose own property says something else fails even if its title agrees.
- Property names vary (spaces, underscores, dots, prefixes, several keys for one property). Any key that names the property counts. Where a listing's keys disagree, follow the shortlist and the independent check, and do not keep swapping your pick over it.
- Values: one item of a list is enough; letter case, singular or plural and hyphens do not matter; a value that holds the required words plus a describing word still counts (a shade such as "pale yellow" is yellow), and a colour can also match by meaning (crimson is a red). A material named inside a mix still names that material, unless the shopper asks for a pure or 100% one. Everything else is strict: a value that lacks a required word, or names a different form, type, format or kind, does not count ("cup" does not state "paper cup", "bundle" is not "single unit"); a size letter does not state a size word ("M" does not state "medium"); a value under a different property does not count.
- Numbers: compare as numbers in the same unit; limits include the edge; a range in a listing meets a limit only when the whole range does.
- An exclusion ("nothing with X") is met when the listing's own property states a value other than X; a listing that does not state that property at all does not show it.
- The kind of product is decided by the store's section or category label on the listing, not by its title. A listing in the section that matches the request's wording as a whole IS the requested kind even when its title names another garment or object, so never skip it because of its title. A listing in a different, even neighbouring, section is not the requested kind. When the request names two kinds joined by "or" or "and", the right section is the one whose label names both, not a narrower neighbour that names one of them.
- Stock is the listing's own stock field; delivery or made-to-order wording in a description does not change it. Price times quantity must be within the current budget (equal is fine).

SEARCHING
- Search can be capped, paged, sorted by price or by id, and may require every word. Compare each result's reported total with the rows you can see, and use further pages, section or price filters and narrower words until the relevant set is covered.
- Open a listing's details before you rely on it; its facts appear only there.

CHEAPEST OR BEST
- For "cheapest", the answer is the lowest-priced listing among ALL listings that meet every need. Gather the candidates of the right section, open them in ascending price and take the first that fits; every cheaper one must have been opened and rejected. Keep the next fits as backups.
- For "best by a measure", compare all fitting listings on that measure; a listing that does not state the measure cannot win.
- Without such a rule any listing that meets everything within budget is right; do not spend steps hunting for lower prices.

CHANGES AND REFUSALS
- The shopper may change things while you work: the budget up or down, needs added, dropped or replaced, an item selling out or changing price. Changes can arrive inside any tool result or as a refusal of your checkout. Each change binds from that moment: re-check your pick and your backups, and choose again when the cheapest or best was asked for.
- A refused checkout leaves the episode running and the cart unchanged. Apply the changes, remove a line that no longer qualifies, add the next verified fit and check out again.

FINISHING
- At checkout the cart holds exactly one line: the chosen listing at the required quantity.
- Make the first checkout attempt while at least seven steps remain, so that a refusal can still be repaired. Do not swap cart lines back and forth.
- Decline only after a real search finds nothing that meets every need, and follow the request's rule for declining. A decline remark never names or points to a listing.
- Every reply must contain at least one tool call. Keep the reply text short: the needs as they stand, verified fits with price, and your next move."""

BRIEF_PROMPT = """You prepare a work order for a shopping assistant. Below are a shopper's request and the store's tools. Answer with ONE JSON object on a single line, no prose, with exactly these keys:
"item": the kind of product wanted, in the shopper's words.
"count": how many units of one listing to buy in this order (an integer; 1 unless the shopper asks to buy several; people in the story who would each like one, such as customers, regulars or friends asking after it, do not set the count; a pair or a set sold as one listing is 1).
"needs": every firm condition on the product itself, one object each: {"property": the property (never the value itself; for an excluded value, the property it is a value of), named with the shopper's own word for it whenever the request uses one, otherwise in a store's plain words (for example material, colour, capacity), "test": "is" when it must be one of the values, "not" when it must be none of them, "min" or "max" only when the listing's own number must stay at or below, or at or above, the shopper's number, "range" for such an interval, "other" for anything else (for example a range or capability the item must reach or cover), "values": the value the shopper names for the property, as strings (numbers too; two numbers for "range"); when the shopper names a value and then explains or illustrates it, keep the named value and leave the explanation in "said", "unit": the unit written with the number or "", "said": the shopper's own words for it}. Leave out price, stock, quantity, wishes, the product kind itself and any word of the product's name (an audience, a season, a form or an occasion that is part of the kind or of a store section's name is not a separate need), and conditions no listing can state, such as delivery, arrival or timing.
"ceiling": the most the shopper will pay in total, as a number, or null.
"parts": [] unless the shopper wants two or more different products bought together in one order (a shopping list); then one object per product, in the request's order: {"item": that product in the shopper's words, "count": units of one listing for it, "needs": its own firm conditions in the same form as "needs"}; "item", "count" and "needs" above then describe the first product only.
"money": the money unit the request uses, or null.
"on_hand_only": true when only items in stock may be bought.
"pick": "cheapest" only when the shopper asks for the lowest-priced fitting item; "best" only when they ask for the best by some measure; otherwise "any". Remarks about a tight budget are not a request for the cheapest.
"best_by": for "best", the measure and whether higher or lower wins; otherwise "".
"messages": how many messages the shopper allows, as an integer, or null.
"steps": how many moves or steps the shopper allows, as an integer, or null.
"rules": each instruction about HOW to work (how many listings to compare, what to check or say, how to decline, what to avoid), copied word for word.
"wishes": things the shopper would merely like (ideally, preferably, if possible).
"learn_first": things that can only be learned from the shopper before choosing, for example the budget when none is given, or which of two wishes matters more when both may not be possible (then put both wishes under "wishes"). Each item names one specific thing; a remark that some unwritten details matter is not an item. [] when none.
"quiet_decline": true when a decline must not offer, name or suggest alternatives; false when it must name something; null when no rule is given.
"tools": the exact tool name for each job, or null: "search", "details" (shows one listing's full property list), "view" (shows one listing's price or stock without the property list, when that is a separate tool), "message" (writes to the shopper), "cart" (adds or removes cart lines), "cart_view", "order" (checks out), "decline" (ends without buying).
"cart_add": the arguments of the cart tool, other than the listing id and the quantity, that make it ADD a line, as an object (for example {"<argument>": "<allowed value>"}); {} when none are needed.
"cart_remove": the same for removing a line; {} when the tool cannot remove.
Every sentence of the request belongs somewhere: a need, the pick rule, a wish, a rule, a limit, or story. When unsure whether a sentence is a rule, put it under "rules"."""

HEAR_PROMPT = """A shopper just said something to a shopping assistant. Work out what it changes. Return ONE JSON object and nothing else, with these keys:
"gist": one short sentence with the shopper's meaning.
"needs_now": the complete list of firm needs on the product after everything said so far, each as {"property", "test", "values", "unit", "said"} in the same form as the current list below. Keep every earlier need unless the shopper dropped or replaced it (for a replaced value keep only the newest), add every newly revealed need, and leave out price, stock, quantity and the product kind.
"budget_now": a new total price limit as a number when the shopper gives one (higher or lower, even when they call it good news), else null.
"dropped": properties the shopper no longer requires.
"indifferent": properties the shopper says they do not mind about.
"gone": true when the shopper says an item can no longer be bought (sold out, unavailable, gone, button disabled), else false.
Read the shopper literally and kindly. A colourful or roundabout phrase that names a value means they firmly want that value; a value is excluded only when they clearly say to avoid it. A number with at most, at least, under, over or no more than is a limit, kept with its unit. When the shopper only says they do not mind, put the property that was asked about under "indifferent" and change nothing else. A mere wish is not a need. When the shopper says which of two wishes matters more, the chosen one is a need. Name each property the way the shopper names it; when the shopper answers about another property than the one asked, use the shopper's property. One line can carry several changes; report all of them."""

REFERENCE_RULE = (" Also return the key \"reference\": when the shopper defines a property's wanted value by pointing at ANOTHER listing (for example"
                  " \"set apart from the value on the page for <shop>'s <product>, with <facts>\", \"the same as on <shop>'s page\", \"anything but what the"
                  " page for the <product> that <shop> has listed says\"), an object {\"property\": that property, \"relation\": \"differ\" when the"
                  " value must be anything EXCEPT that listing's (anything but / bar / except / set apart from / not), else \"same\", \"listing\": the"
                  " shop, the product and the facts that identify"
                  " that other listing, in the shopper's words}; else null. The facts that identify the other listing describe THAT listing, never this"
                  " product: do not add them to needs_now, and add no need for that property from such a line.")

CHECK_PROMPT = """You are the last check before a shopping assistant puts a listing in the cart or checks out. Judge the listing only from the records shown, with these rules:
- A need is met when the listing's own property list states it. Any key naming the property counts; one list item is enough; case, singular or plural and hyphens do not matter; a value holding the required words plus a describing word counts (a shade counts for a colour, and a colour may also match by meaning). A material named inside a mix still counts unless a pure or 100% material is required. A different form, type, format or kind fails; a size letter does not state a size word; a value under a different property does not count. When two of the listing's own keys for the property state different values, the need is not met, unless the two keys have the same name written another way (spaces, underscores, dots or case: one property listed twice, so either value counts) or the property holds several labels at once (certifications, features) and one key simply lists more of them. Only when no key mentions the property at all, not even inside a longer key name, may the title or an options/variant field show it; the description never proves a need. A key about a different property (a count, a size or a type) never stands in for the needed one.
- Numbers compare in the same unit; edges count; a range meets a limit only when all of it does.
- An exclusion is met when the listing's own property states a value other than the excluded one; when no key states that property, it is not met.
- The kind of product is decided by the listing's section or category label: it must be the section that matches the request's wording as a whole (for two kinds joined by "or" or "and", the section naming both, not a narrower neighbour). A title that names another kind of item does not matter.
- Price times quantity must be within the current budget. When the request wants items on hand, the listing's stock field must not say it is out of stock. When records disagree, the newest one counts.
- Wishes are not needs.
An automatic reading of the property list is included below; trust what it says is stated, but the decision is yours.
Return ONE JSON object and nothing else: {"id": the listing id, "ok": true or false, "why": [short reasons, each naming an unmet need]}. Say false only for a need that the records clearly fail or do not state at all; a missing stock or price record alone is not a failure."""

RESCUE_PROMPT = """An automatic reader compared a listing's own properties with a shopper's needs. For each need below it found a property that names the need, but whose value does not contain the required words. The reader's summary of a need can be wrong: the request and the shopper's own words decide. Decide for each one whether the stated value gives the shopper what they asked for in their own words. Count it as meant when it is exactly the value the shopper named, or the same thing in other words: a shade or tone of a required colour, another spelling or an abbreviation of the same word, or the same value written another way. Do NOT count a different form, type, format or variant (a pack format, a container type, a neckline, a style or a fit that differs), a broader or narrower class, a mix when a pure or 100% material is required, or a size letter for a size word.
Return ONE JSON object and nothing else: {"meant": [the numbers of the needs whose value is meant], "why": "one sentence"}."""

ASK_PROMPT = """You choose the questions a shopping assistant may send to a shopper. The shopper holds some needs back and reveals one only when a message asks about that property. Each message is answered only about the first thing it asks. Choose up to QUESTION_COUNT different things to ask, best first:
1. Everything under MUST LEARN that has not been asked yet, as a short plain question (kind "open"). For a budget, ask the most they want to spend; for two competing wishes, ask which one matters more and name both values. A question must name what it asks about: never ask what else or which other details matter, because such a question gets no answer; unwritten details are found by asking about properties.
2. Then properties from the TABLE that describe the product itself: its type, form, size, fit or cut, its look (colour, pattern, finish, style), who it is for, its material, build or condition, its specifications (processor, memory, storage, screen, capacity, power and the like, even when their values look like model codes), or its function. Leave out warranty, delivery, seller or brand facts, and never ask for a model name, product code or option string as such. Rank first the ones the story hints at (who will use it, how it must look, where or how it is used, what it must work with); then the ones that the most listings carry (the count in the table); among equally common ones prefer those whose values differ across the candidates that still meet the known needs.
Never pick a property the known needs already settle, one already asked, or two names of the same property. Name each property in full, as the listings name it but without prefixes, codes or filler words ("sneaker height", not "height"; "graphics memory", not "graphics memory new"), the way a shopper would recognise it.
Return ONE JSON object and nothing else: {"picks": [{"kind": "property", "name": "..."} or {"kind": "open", "name": "...", "text": "the question"}, ...]}."""

SWEEP_PROMPT = """A shopping assistant is about to decline, saying nothing in the store fits. Check the listing records below against the needs with these rules: a need is met when the listing's own property list states it, or its title or options/variant field when no key mentions the property at all (any key naming the property, one list item is enough, a value holding the required words counts, a colour may match by meaning, a material named inside a mix counts unless a pure one is required; a different form or type and a size letter for a size word do not); numbers compare in the same unit with the edges included; an exclusion is met when the property is stated with another value; the listing's section must be the requested kind; the price must be within the budget and the listing must not be shown out of stock.
Return ONE JSON object and nothing else: {"maybe": [ids of listings that seem to meet every need, cheapest first, at most 5], "why": "one sentence"}."""

FINISH_PROMPT = """One tool of a shop, named below, is used to finish the task: with some fixed argument values it checks out (places the order for what is in the cart), with others it ends the task without buying. From its description and arguments, find those values. Return ONE JSON object and nothing else: {"order": {"<argument>": "<allowed value>"}, "decline": {"<argument>": "<allowed value>"}}, using only the tool's own argument names and allowed values; give {} for "decline" if the tool cannot end without buying."""

NEUTRAL_DECLINE = "I'm sorry, nothing in the store meets every one of your requirements, so I am not buying anything."
GUARD = "Not run (no step used): "
SEARCH_FILLER = {"a", "an", "and", "of", "the", "for", "with", "in", "to", "or", "by", "on"}   # words a store search may read as operators or drop


# ---------------------------------------------------------------- small helpers

def _dump(value: Any, limit: int = RESULT_CLIP) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(value)
    return text if len(text) <= limit else text[:limit] + "...(cut)"


def _trim(value: Any, limit: int = TEXT_CLIP, depth: int = 0) -> Any:
    """Copy of an observation with very long strings shortened (descriptions are never evidence)."""
    if depth > 8:
        return value
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, list):
        return [_trim(v, limit, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _trim(v, limit, depth + 1) for k, v in value.items()}
    return value


def _shorten(text: str, limit: int = 4000) -> str:
    """A long tool result cut down: a leading JSON value is dumped again with every string clipped to 60 characters, and
    whatever follows it (notes, an old status) keeps its first 300 characters."""
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except ValueError:
        return text[:limit] + "...(cut)"
    rest = text[end:].strip()
    return _dump(_trim(value, 60), limit) + ((" " + rest[:300]) if rest else "")


def _json_object(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str):
        return None
    body = text.strip()
    fence = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", body, re.S)
    if fence:
        body = fence.group(1)
    for attempt in (body, body[body.find("{"): body.rfind("}") + 1] if "{" in body else ""):
        if not attempt:
            continue
        try:
            value = json.loads(attempt)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    # several objects with words between them ("{...} Wait - ... {...}"): the last complete one is the model's final answer
    decoder, found, at = json.JSONDecoder(), None, body.find("{")
    while at >= 0:
        try:
            value, end = decoder.raw_decode(body, at)
        except ValueError:
            break   # a cut or broken object: never one of its inner pieces
        if isinstance(value, dict):
            found = value
        at = body.find("{", end)
    return found


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_int(value: Any) -> int | None:
    if _is_number(value) and float(value).is_integer() and value >= 0:
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+\s*", value):
        return int(value)
    return None


def _as_written(value: Any, text: str, taken: list[Any]) -> tuple[Any, str]:
    """A number a model took from the shopper's words, put back as the shopper wrote it. When the value itself is not
    written but 100 or 1000 times it (or a 100th or 1000th of it) is, and no other field uses that number, the written
    number and the word after it (its unit) are returned; otherwise the value and ''."""
    if not _is_number(value) or value <= 0:
        return value, ""
    used = [float(t) for t in taken if _is_number(t)]
    found = [(raw, float(raw.replace(",", ""))) for raw in re.findall(r"\d[\d,]*(?:\.\d+)?", text or "")]
    found = [(raw, number) for raw, number in found if not any(abs(number - u) < 1e-9 for u in used)]
    if not found or any(abs(number - value) < 1e-9 for _, number in found):
        return value, ""
    for factor in (100, 0.01, 1000, 0.001):
        for raw, number in found:
            if abs(number - value * factor) <= 1e-6 * max(1.0, number):
                unit = re.search(r"(?<![\d.,])" + re.escape(raw) + r"\s+([^\W\d_]+)", text)
                return (int(number) if number.is_integer() else number), (unit.group(1) if unit else "")
    return value, ""


def _id_like(value: Any) -> bool:
    return isinstance(value, str) and 3 <= len(value) <= 80 and not re.search(r"\s", value) and bool(re.search(r"\d", value))


# ---------------------------------------------------------------- argument schemas

def _kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind if isinstance(k, str)] if isinstance(kind, list) else []


def _fits_kind(value: Any, kind: str) -> bool:
    tests = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": _is_number,
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }
    test = tests.get(kind)
    return True if test is None else test(value)


def _convert(value: Any, kinds: list[str]) -> Any:
    """Bring a value to one of the allowed JSON types when an obvious conversion exists."""
    if not kinds or any(_fits_kind(value, k) for k in kinds):
        if "integer" in kinds and isinstance(value, float) and value.is_integer():
            return int(value)
        return value
    for kind in kinds:
        if kind in ("integer", "number") and isinstance(value, str):
            text = value.strip()
            if re.fullmatch(r"-?\d+", text):
                return int(text)
            if kind == "number" and re.fullmatch(r"-?\d+\.\d+", text):
                return float(text)
            if kind == "integer" and re.fullmatch(r"-?\d+\.0+", text):
                return int(float(text))
        if kind == "integer" and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind == "string" and _is_number(value):
            return str(value)
        if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
    return value


def conform(value: Any, schema: Any, where: str = "args") -> tuple[Any, list[str]]:
    """Check (and gently repair) a tool argument against its JSON schema. Returns the value and a list of problems."""
    if not isinstance(schema, dict):
        return value, []
    kinds = _kinds(schema)
    value = _convert(value, kinds)
    if kinds and not any(_fits_kind(value, k) for k in kinds):
        return value, [where + " must be " + " or ".join(kinds)]
    issues: list[str] = []
    options = schema.get("enum")
    if isinstance(options, list) and value not in options:
        folded = {str(o).strip().lower(): o for o in options if isinstance(o, str)}
        if isinstance(value, str) and value.strip().lower() in folded:
            value = folded[value.strip().lower()]
        else:
            issues.append(where + " must be one of " + _dump(options, 300))
    if "const" in schema and value != schema["const"]:
        issues.append(where + " must be " + _dump(schema["const"], 100))
    if isinstance(value, str):
        if isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
            value = value[: schema["maxLength"]]
        if isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
            issues.append(where + " is too short")
    if _is_number(value):
        for bound, broken in (("minimum", lambda v, b: v < b), ("maximum", lambda v, b: v > b),
                              ("exclusiveMinimum", lambda v, b: v <= b), ("exclusiveMaximum", lambda v, b: v >= b)):
            limit = schema.get(bound)
            if _is_number(limit) and broken(value, limit):
                issues.append(where + " breaks " + bound + " " + str(limit))
    if isinstance(value, list):
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            issues.append(where + " has too few items")
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            issues.append(where + " has too many items")
        inner = schema.get("items")
        if isinstance(inner, dict):
            repaired = []
            for index, element in enumerate(value):
                element, sub = conform(element, inner, where + "[" + str(index) + "]")
                repaired.append(element)
                issues.extend(sub)
            value = repaired
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for name in schema.get("required") or []:
            if name not in value:
                issues.append(where + "." + str(name) + " is required")
        repaired_obj = {}
        for name, element in value.items():
            if name not in props:
                if schema.get("additionalProperties") is False:
                    issues.append(where + "." + str(name) + " is not an allowed argument")
                repaired_obj[name] = element
                continue
            element, sub = conform(element, props[name], where + "." + str(name))
            repaired_obj[name] = element
            issues.extend(sub)
        value = repaired_obj
    return value, issues


# ---------------------------------------------------------------- the fact reader

def _stem(word: str) -> str:
    if word.isdigit() or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("sses", "xes", "ches", "shes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: Any) -> list[str]:
    """Lower-case word and number tokens; numbers are canonical (6.00 -> 6), a percent sign stays on its number."""
    raw = str(text).lower()
    raw = re.sub(r"(\d)([a-z])", r"\1 \2", raw)
    raw = re.sub(r"([a-z])(\d)", r"\1 \2", raw)
    out: list[str] = []
    for token in re.findall(r"\d+(?:\.\d+)?%?|[a-z]+", raw):
        if token[0].isdigit():
            percent = token.endswith("%")
            number = float(token.rstrip("%"))
            token = (str(int(number)) if number.is_integer() else str(number)) + ("%" if percent else "")
        else:
            token = _stem(token)
        if token not in out:
            out.append(token)
    return out


def _roots(word: str) -> set[str]:
    """The word plus what is left after one English inflection ending (embroidered / embroidery -> embroider).
    FV: an f / fe plural ends in -ves, which _stem leaves as -ve (shelves -> shelve, knives -> knive): both sides root to -f."""
    out = {word}
    for ending in ("ing", "ery", "ed", "ry", "y", "d"):
        if word.endswith(ending) and len(word) - len(ending) >= 4:
            out.add(word[: -len(ending)])
    if len(word) >= 5 and word.endswith("ve"):
        out.add(word[:-2] + "f")
    elif len(word) >= 5 and word.endswith("fe"):
        out.add(word[:-1])
    return out


def _same(a: str, b: str) -> bool:
    """Equal tokens, two forms of one word, or two spellings that differ by one inserted letter (five letters or more)."""
    if a == b:
        return True
    if a[0].isdigit() or b[0].isdigit():
        return False
    if _roots(a) & _roots(b):
        return True
    longer, shorter = (a, b) if len(a) > len(b) else (b, a)
    if len(shorter) < 5 or len(longer) != len(shorter) + 1:
        return False
    return any(longer[:i] + longer[i + 1:] == shorter for i in range(len(longer)))


def _covers(have: list[str], want: list[str]) -> bool:
    return bool(want) and all(any(_same(w, h) for h in have) for w in want)


def _key_grade(key: Any, prop: Any) -> int:
    """2 when a fact key names exactly the property, 1 when it names it with extra words, 0 otherwise."""
    have, want = _words(key), _words(prop)
    if not _covers(have, want):
        return 0
    return 2 if len(have) == len(want) else 1


def _pieces(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for element in value:
            parts.extend(_pieces(element))
        return parts
    text = str(value).strip()
    return [p.strip() for p in re.split(r"[,;|/\n]+", text) if p.strip()]


def _value_says(piece: str, wanted: str) -> bool:
    """Does one list item of a fact value state the wanted value? Every word (and number) of the wanted value must be there;
    a piece joining values with '+' (a combination) states only a combination."""
    if "+" in piece and "+" not in wanted and len([p for p in piece.split("+") if p.strip()]) > 1:
        return False
    return _covers(_words(piece), _words(wanted))


def _says_in(value: Any, wanted: str) -> bool:
    """UV: does a fact value state the wanted value in one of its list items, or whole when the wanted value itself holds a list
    separator ("read speed: mb/s" is stated by '205 mb/s', which _pieces splits at the slash)?"""
    return any(_value_says(p, wanted) for p in _pieces(value)) or (bool(re.search(r"[,;|/]", wanted)) and _value_says(str(value), wanted))


def _says_amount(value: Any, wanted: str) -> bool:
    """UN: does a fact value state an amount the request gives apart from its unit ("model: 3 inches" read as 3 + inches)? As
    before, any list item holding the number (or range) states it; but not a number glued into a code that runs on into another
    number ('3in1', 'v2x3'), which names no amount."""
    if not _says_in(value, wanted):
        return False
    numbers = [w for w in _words(wanted) if w[0].isdigit()]
    text = str(value).lower()
    free = {_words(m.group(1))[0] for m in re.finditer(r"(?<![0-9.])(\d+(?:\.\d+)?)(?![0-9]|\.[0-9])", text)
            if not re.match(r"[a-z]+[0-9]", text[m.end():])}
    return all(n in free for n in numbers)


def _free_count(text: Any, wanted: str) -> bool:
    """UB: a bare number the request gives with no unit ("pack size: 6") is not stated by a title or listing text where it is
    the letter-led head of a serial code ("[ZE6-53853]": letters, the number, a dash and digits) or runs on into another
    number ('3in1'); "6pk", "6 pack" and a model name such as "G1" keep it."""
    low = str(text).lower()
    free = {_words(m.group(1))[0] for m in re.finditer(r"(?<![0-9.])(\d+(?:\.\d+)?)(?![0-9]|\.[0-9])", low)
            if not re.match(r"[a-z]+[0-9]", low[m.end():]) and not (re.search(r"[a-z]$", low[: m.start()]) and re.match(r"-\d", low[m.end():]))}
    return all(n in free for n in _words(wanted) if n[0].isdigit())


def _bare_number(need: dict[str, Any], wanted: Any) -> bool:
    return not str(need.get("unit") or "").strip() and bool(re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(wanted)))


def _says_need(value: Any, wanted: str, need: dict[str, Any]) -> bool:
    """A fact value states one wanted value of a need: UN when the need gives the value's unit apart, else UV (a fact under a
    key naming the property keeps a glued number: 'win11 pro' states operating system 11; UB is for titles and text)."""
    return _says_amount(value, wanted) if str(need.get("unit") or "").strip() else _says_in(value, wanted)


def _quantities(piece: str) -> list[tuple[float, float, str]]:
    """Numbers in a fact item as (low, high, unit); 'a-b' or 'a to b' becomes one range."""
    text = str(piece).lower().replace(",", "")
    found: list[tuple[float, float, str]] = []
    pattern = r"(-?\d+(?:\.\d+)?)\s*(?:(?:-|~|to)\s*(-?\d+(?:\.\d+)?))?\s*([a-z%\"]+)?"
    for low, high, unit in re.findall(pattern, text):
        a = float(low)
        b = float(high) if high else a
        if high and a < 0 and b > 0:
            a = abs(a)
        if a > b:
            a, b = b, a
        found.append((a, b, _stem(unit) if unit else ""))
    return found


def _unit_ok(have: str, want: str) -> bool | None:
    """True when the units agree, None when a comparison would need a conversion we do not have."""
    if not want or not have:
        return True
    if _same(have, want):
        return True
    return None


def _presence_need(need: dict[str, Any]) -> bool:
    """YN: a need that only asks for the property to be there ("with a date feature" read as date feature: yes)."""
    values = [str(v).strip().lower() for v in need.get("values") or [] if str(v).strip()]
    return need.get("test") == "is" and bool(values) and all(v in ("yes", "true") for v in values)


def read_need(need: dict[str, Any], facts: dict[str, Any] | None) -> tuple[str, str]:
    """State of one need on one listing: met / clash (an exact key says otherwise and no key holds the value) / mixed (exact
    keys disagree) / weak (only a looser key says otherwise, or the value sits under another key) / absent (no key names the
    property) / unsure (the reader cannot decide)."""
    if facts is None:
        return "unsure", "facts not read"
    prop = need.get("property") or ""
    test = need.get("test") or "is"
    values = [str(v) for v in need.get("values") or [] if str(v).strip()]
    hits = []
    for key, value in facts.items():
        if value is None or isinstance(value, (dict, bool)) or not str(value).strip():
            continue
        grade = _key_grade(key, prop)
        if grade:
            hits.append((grade, str(key), value))
    if values and all(_names_other_listing(v) for v in values):
        return "unsure", "its value is set by another listing"
    if test == "other" or (test in ("is", "not") and not values):
        return "unsure", "needs judgement"
    if test == "not":
        for grade, key, value in hits:
            for piece in _pieces(value):
                for banned in values:
                    if _value_says(piece, banned):
                        return "clash", key + " says '" + str(value) + "'"
        if hits:
            return "met", hits[0][1] + " says '" + str(hits[0][2]) + "'"
        if all(_covers(_words(v), _words(prop)) for v in values):
            # the property was written as the excluded value itself: no key can be known, so any value that states it clashes
            for key, value in facts.items():
                if isinstance(value, (str, int, float)) and not isinstance(value, bool) and any(_value_says(p, b) for p in _pieces(value) for b in values):
                    return "clash", str(key) + " says '" + str(value) + "'"
            return "met", "no key states " + " / ".join(values)
        return "absent", prop + " not stated (only a stated other value shows an exclusion is met)"
    if not hits and test == "is" and _presence_need(need):
        # YN: "with a date feature" (date feature: yes) is stated by a key the property ends in holding the rest ("feature: date")
        named = _words(prop)
        for key, value in facts.items():
            have = _words(key)
            rest = [w for w in named if w not in have]
            if have and rest and _covers(named, have) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
                if any(_covers(_words(p), rest) for p in _pieces(value)):
                    return "met", str(key) + " says '" + str(value) + "'"
    if not hits and test == "is":
        head = _words(prop)[:1]
        for key, value in facts.items():
            if head and _covers(_words(key), head) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
                if any(_says_need(value, v, need) for v in values):
                    return "met", str(key) + " says '" + str(value) + "' (a related key; none names the property itself)"
    if not hits:
        return "absent", prop + " not stated"
    if test == "is":
        says = [(grade, key, value) for grade, key, value in hits if any(_says_need(value, v, need) for v in values)]
        exact = [h for h in hits if h[0] == 2]
        if says:
            # two keys with the same name are one property listed twice: either value counts (live race 4: 8 accepted, 2 rejected)
            twin = {" ".join(_words(h[1])) for h in says}
            against = [h for h in exact if h not in says and " ".join(_words(h[1])) not in twin]
            exact_says = any(h[0] == 2 for h in says)
            listed = any(len(_pieces(h[2])) > 1 for h in says)
            if against and (exact_says or not listed):
                return "mixed", says[0][1] + " says '" + str(says[0][2]) + "' but " + against[0][1] + " says '" + str(against[0][2]) + "'"
            return "met", says[0][1] + " says '" + str(says[0][2]) + "'"
        shown = exact[0] if exact else hits[0]
        note = shown[1] + " says '" + str(shown[2]) + "'"
        elsewhere = [str(k) for k, v in facts.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
                     and any(_says_need(v, w, need) for w in values)]
        if exact and not elsewhere:
            return "clash", note
        return "weak", note + ("; the value appears under " + elsewhere[0] if elsewhere else "")
    numbers = []
    for v in values:
        parsed = _quantities(v)
        if parsed:
            numbers.append(parsed[0][0])
    if not numbers:
        return "unsure", "limit not numeric"
    want_unit = _stem(str(need.get("unit") or "").lower().strip())
    low_limit = min(numbers)
    high_limit = max(numbers)
    decided_fail = []
    undecided = False
    for grade, key, value in hits:
        for piece in _pieces(value):
            for a, b, unit in _quantities(piece):
                agree = _unit_ok(unit, want_unit)
                if agree is None:
                    undecided = True
                    continue
                if test == "min":
                    good = a >= low_limit
                elif test == "max":
                    good = b <= high_limit
                else:
                    good = a >= low_limit and b <= high_limit
                if good:
                    return "met", key + " says '" + str(value) + "'"
                if grade == 2:
                    decided_fail.append(key + " says '" + str(value) + "'")
    if decided_fail and not undecided:
        return "clash", decided_fail[0]
    return "unsure", "number not comparable"


def _section_says(need: dict[str, Any], item: dict[str, Any] | None) -> bool:
    """A need that only restates the store section's own label is met by the section (the section decides the kind)."""
    section = (item or {}).get("section")
    if not section:
        return False
    have = _words(section)
    return any(_covers(have, _words(v)) for v in need.get("values") or [])


def _shared_word(values: Any) -> bool:
    """Some word of three or more letters appears in two or more of these values."""
    seen: dict[str, int] = {}
    for value in values:
        for word in set(w for w in _words(value) if w.isalpha() and len(w) >= 3):
            seen[word] = seen.get(word, 0) + 1
    return any(n >= 2 for n in seen.values())


_FACT_VALUES: dict[str, set[str]] = {}   # property key -> short values opened listings state (reset with each Shelf)


def _note_values(facts: Any) -> None:
    for key, value in facts.items() if isinstance(facts, dict) else []:
        if isinstance(value, str) and "," not in value and "/" not in value and 0 < len(_words(value)) <= 3:
            _FACT_VALUES.setdefault(" ".join(_words(key)), set()).add(" ".join(_words(value)))


def _title_names_other(need: dict[str, Any], item: dict[str, Any] | None, others: Any = None) -> str:
    """When no key names the property: a value that opened listings state for it, named by this title, that is no form of
    a wanted value, while the title names none of the wanted ones ("satin nickel" for a wanted "polished brass"). A title
    word that is another need's wanted value ("privacy" for the function) says nothing about this property."""
    title = (item or {}).get("title")
    wanted = [_words(v) for v in need.get("values") or [] if _words(v)]
    if not title or need.get("test") != "is" or not wanted or _presence_need(need):
        return ""   # YN: a title naming another feature says nothing against a wanted one being there
    have = _words(title)
    if any(_covers(have, w) for w in wanted):
        return ""
    taken =[_words(v) for n in others or [] if isinstance(n, dict) and n is not need and n.get("test") != "not"
             for v in n.get("values") or [] if _words(v)]
    wanted = wanted + [t for t in taken if _covers(have, t)]   # skipped below like a form of a wanted value
    for key in sorted(_FACT_VALUES):
        if not (_key_grade(key, need.get("property")) or _key_grade(need.get("property"), key)):
            continue
        for value in sorted(_FACT_VALUES[key]):
            words = value.split()
            if any(_covers(words, w) or _covers(w, words) for w in wanted):
                continue
            if _covers(have, words):
                return value
    return ""


def _title_says(need: dict[str, Any], item: dict[str, Any] | None) -> bool:
    """The title may show a need only when no key names the property (callers check that); the description never does."""
    title = (item or {}).get("title")
    if not title:
        return False
    have = _words(title)
    # TU: a figure counts only as a free number ('3in1' states no 3), with or without a unit in the need ("model: 3 inches")
    return any(_covers(have, _words(v)) and (not re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(v)) or _free_count(title, v))
               for v in need.get("values") or [])


def _option_says(need: dict[str, Any], facts: dict[str, Any] | None) -> bool:
    """A listing's options/variant field may show a need only when no key names the property (callers check that)."""
    values = [str(v) for v in need.get("values") or [] if str(v).strip()]
    for key, value in (facts or {}).items():
        if re.search(r"option|variant", str(key), re.I) and isinstance(value, (str, int, float)) and not isinstance(value, bool):
            if any(_says_need(value, wanted, need) for wanted in values):
                return True
    return False


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# FT: "Make sure the <part> fits the <host> you end up buying": the part must suit the host listing bought with it
_FIT_RULE = re.compile(r"\b(?:make sure|ensure|check) (?:that )?(?:the |each |every |all (?:the )?|your )?(?P<part>[a-z][a-z0-9 &'-]{1,80}?) "
                       r"(?:fits?|works? with|suits?|matches|goes with|(?:is|are) compatible with|(?:is|are) made for) (?:the |each |your |my )?"
                       r"(?P<host>[a-z][a-z0-9 &'-]{1,60}?)(?: (?:you|i|we) (?:end up |will )?(?:buy|choose|pick|order|get|take)\w*)?\s*[.!]?\s*$", re.I)
# FD: any sentence that says one list product must fit / be compatible with / work with another (named before and after the word)
_FIT_WORD = re.compile(r"\b(?:fits?|fitting|compatib\w*|works? with|suits?|goes with|made for|for use with|pairs? with|matches|matching)\b", re.I)
# FT: a page that names the part series (models, codes) fitting it: "Compatible part series: ZE9"
_FIT_SAYS = re.compile(r"(?i:\b(?:compatib\w*|fits?|for use with|works? with|pairs? with|matching)\b)[^.;\n]{0,40}?"
                       r"(?i:\b(?:series|models?|lines?|codes?|range|famil(?:y|ies)|types?)\b)\s*(?i:no\.?|number|is|are)?\s*[:#-]?\s*"
                       r"(?P<codes>[A-Z][A-Z0-9]{0,5}\d[A-Z0-9]{0,4}(?:\s*(?:,|/|&|\bor\b|\band\b)\s*[A-Z][A-Z0-9]{0,5}\d[A-Z0-9]{0,4})*)(?![A-Za-z0-9])")


def _fit_codes(item: dict[str, Any]) -> list[str]:
    """FT: the part series a listing's own page or row says fit it ("Compatible part series: ZE9" -> ["ZE9"])."""
    texts = [v for v in ((item or {}).get("record") or {}).values() if isinstance(v, str) and v != (item or {}).get("title")]
    texts += [str(v) for v in ((item or {}).get("facts") or {}).values()]
    found = [code for text in texts for m in _FIT_SAYS.finditer(text) for code in re.findall(r"[A-Z][A-Z0-9]{0,5}\d[A-Z0-9]{0,4}", m.group("codes"))]
    return list(dict.fromkeys(found))


def _carries(item: dict[str, Any], code: str) -> bool:
    """FT: a listing names the series code as a token of its title or its own text ("[ZE9-56311]" names ZE9; "ZE91" does not)."""
    texts = [str((item or {}).get("title") or "")] + [v for v in ((item or {}).get("record") or {}).values() if isinstance(v, str)]
    texts += [str(v) for v in ((item or {}).get("facts") or {}).values()]
    return any(re.search(r"(?<![A-Za-z0-9])" + re.escape(code) + r"(?![A-Za-z0-9])", t) for t in texts)


_ONWARD = re.compile(r"\bnot\b|\binstead\b|\brather than\b|\btaken from\b|\bgo by\b|\brefer to\b|\blook (?:to|at)\b|\btake it from\b|^\s*for [^,]{2,50},"
                     r"|\belsewhere\b|\bversion\b[^.]{0,80}\bexists\b|\bdifferent version\b", re.I)
# NS: more ways a page sends a property to another shop's listing ("Take its plug type from another product, ...", "isn't usb the way X's is")
_HOP = re.compile(r"\btake\b[^.;]{0,60}\bfrom\b|\banother (?:product|listing|item|page|shop|seller)\b|\bsee\b|n't\b|\bthe way\b|\bsame as\b|\bunlike\b"
                  r"|\blike\b|\bmatch\w*\b", re.I)
INFERRED_SAID = "inferred from another listing"
_FILLER = {"the", "a", "an", "for", "listing", "sold", "by", "with", "and", "in", "of", "from", "that", "which", "it", "its", "this", "item",
           "not", "product", "names", "direct", "or", "through", "is", "are", "to",
           "listed", "has", "page", "written", "what", "s", "said", "outright", "naming", "some", "other", "very", "same", "as", "on",
           "anything", "bar", "but", "except"}


def _record_sentences(item: dict[str, Any]) -> list[str]:
    title = (item or {}).get("title")
    return [s for value in ((item or {}).get("record") or {}).values()
            if isinstance(value, str) and value != title and len(value.split()) >= 4 for s in _sentences(value)]


def _listing_words(item: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Words the listing states (fact values, its section and title, its own plain sentences) and words that only appear in
    its negated or pointing sentences."""
    said = set(_words(" ".join(str(v) for v in ((item or {}).get("facts") or {}).values())))
    said |= set(_words(str(item.get("section") or "") + " " + str(item.get("title") or "")))
    denied: set[str] = set()
    for sentence in _record_sentences(item):
        (denied if _ONWARD.search(sentence) else said).update(_words(sentence))
    return said, denied - said


_POINTER_VERBS = set(_words("listed at weighing weighs priced rated measuring measures comes come as is are has have having offered featuring"
                            " check see refer look follow one ones"))   # RP: "the Belnovex one with ..." names the listing, no detail
# NH: words a pointer uses to tell its listing apart that no listing states ("a different binder", "any binding cover")
_HUNT_SKIP = {"different", "another", "any", "else", "elsewhere", "version", "here", "made"}
_CHAIN_BOUND = {"with", "a", "an", "the", "of", "and", "from", "listing", "has", "have", "its", "for", "that", "which", "is", "are"}


def _chain_details(sentence: str, name: str) -> str:
    """The details a pointer gives to tell the named shop's page from its other pages: the "has listed, with ..." clause of a
    shopper's pointer, or on a page the words from just before the shop's name to the end of that clause ("the Luribel desktop
    computer with windows 11 home", "from Dormivexko (10 ft cable length, nylon material)", "from Vonorako, the men's slip on
    sneakers")."""
    at = re.search(r"\b" + re.escape(name) + r"\b", sentence, re.I)
    if not at:
        return ""
    listed = re.match(r"\s*(?:has|have|had) listed,?\s*(?:with (.+?))?(?:, said outright|, and at any|,\s*direct(?:ly)? or through|$)", sentence[at.end():], re.I)
    if listed:
        return listed.group(1) or ""
    head = sentence[:at.start()]
    head = head[max(head.rfind(" the "), head.rfind(","), head.rfind(":") + 1, 0):]   # RL: "see a different product: canned meat from X"
    # RL: a reason after the details ("..., since that detail comes from that product") is no detail either
    return re.split(r",? (?:not|rather than|instead of|since|because|when it comes to)\b|;|, the value| - |,\s*direct(?:ly)? or through", head + sentence[at.start():], flags=re.I)[0]


def _value_fit(have: list[str], want: list[str]) -> bool | None:
    """Does a page value state the wanted one? None when both carry a number but name different units (1000 gb / 1 tb): words
    cannot compare them, so that detail is not counted either way."""
    if _covers(have + sorted(_FILLER), want):
        return True
    def units(words: list[str]) -> set[str]:
        return {words[i + 1] for i, w in enumerate(words[:-1]) if w[0].isdigit() and not words[i + 1][0].isdigit()}
    mine, theirs = units(want), units(have)
    nums = [w for w in want if w[0].isdigit()]
    if mine and not theirs and nums and set(nums) <= set(have) and len(want) == 2 * len(nums):
        return True   # a bare page number ("6") states "6 count": the page names no unit that could disagree
    if mine and theirs and not mine & theirs:
        return None
    return False


def _root(word: str) -> str:
    """RL: a verb form's root, so a pointer's "running windows 10 pro" meets a page's "run windows 10 pro"."""
    for end, size in (("ing", 6), ("ed", 5)):
        if len(word) >= size and word.endswith(end) and word.isalpha():
            base = word[:-len(end)]
            return base[:-1] if len(base) > 2 and base[-1] == base[-2] else base
    return word


def _states(said: list[str], core: list[str]) -> bool:
    """RL: one sentence or fact states a detail: it has every word of it, and each figure with the word that follows it ("6 kg")."""
    said, core = [_root(w) for w in said], [_root(w) for w in core]
    if not set(core) <= set(said):
        return False
    for i, w in enumerate(core[:-1]):
        if any(c.isdigit() for c in w) and not any(said[k] == w and said[k + 1] == core[i + 1] for k in range(len(said) - 1)):
            return False
    return True


def _detail_fit(item: dict[str, Any], details: str, skip: set[str], shop_keys: frozenset[str] = frozenset(),
                unsure: list[list[str]] | None = None, vocab: frozenset[str] | None = None) -> float | None:
    """Share of a pointer's details that this page itself states (None: nothing in them can be checked). A detail that names a
    property ("color black", "512gb system memory", "nylon material" for a "cable material" key) is checked against the page's
    own value of that property; any other detail against the words the page states, not those of its negated or pointing
    sentences. skip: words that tell no page of the shop from another (the shop's and the section's names). unsure: gets the
    pointer's words of each detail the page states only in other units. vocab (DP): the words of every opened listing's property
    names, values, titles and sections; a chunk none of whose words is among them ("or from the listing it defers to", "the rest I can bend") is
    the pointer's own clause, not a detail, and is left out (a chunk with a figure always counts)."""
    facts = item.get("facts") or {}
    said, _ = _listing_words(item)
    keys = [(_words(k), v) for k, v in facts.items() if _words(k)]
    named = {kw[0] for kw, _ in keys if len(kw) == 1}   # a one-word property name ("color") tells no page apart; a word inside a longer key may
    oks: list[bool | float | None] = []
    # RL: each of the page's own plain sentences and each fact ("key value") is checked whole as well: a detail every content
    # word of which one of them states ("8gb & up graphics memory" said in the notes, "cans as the packaging type", "listed at
    # 6 kg" on a page "weighing 6 kg") is stated, whatever key the table files it under and whatever verb the pointer uses
    own = [[w for piece in t.split() for w in _words(piece) if w not in _FILLER and w not in skip] for t in _record_sentences(item)
           if not _ONWARD.search(t) and not _DIFFER_WORD.search(t)]   # word by word: repeats kept ("a weight (in g) of 340 g")
    own += [[w for piece in (str(k) + " " + str(v)).split() for w in _words(piece) if w not in _FILLER] for k, v in facts.items()
            if isinstance(v, (str, int, float))]
    for chunk in re.split(r",\s*(?:and\s+)?|\s+and\s+|\(|\)", details or ""):
        cw = [w for piece in chunk.split() for w in _words(piece)]   # in order, repeats kept ("cable with a cable length")
        if not [w for w in cw if w not in _FILLER and w not in skip]:
            continue
        if vocab is not None and not any(w in vocab or any(c.isdigit() for c in w) for w in cw if w not in _FILLER and w not in skip):
            continue   # DP: no opened listing names or states any of its words, and it gives no figure
        core = [w for w in cw if w not in _FILLER and w not in skip and w not in _POINTER_VERBS]
        if len(core) >= 2 and any(_states(sw, core) for sw in own):
            oks.append(True)   # RL: one of the page's own sentences or facts states it
            continue
        spans: dict[tuple[int, int], list[Any]] = {}
        for kw, value in keys:
            for cut in range(len(kw)):   # the key or its tail ("cable material" -> "material")
                part = kw[cut:]
                if len(part) == 1 and part[0] in _FILLER:
                    continue
                for j in range(len(cw) - len(part) + 1):
                    if cw[j:j + len(part)] == part and not (cut and j and cw[j - 1][0].isdigit()) and not set(part) <= skip:
                        # "1 gb" is a unit, not a "hard disk gb" key; "cable" in "the USB cable" is the kind, not a "cable" key
                        spans.setdefault((j, len(part)), []).append(value)
        chosen: list[tuple[int, int, list[Any]]] = []
        taken: set[int] = set()
        for (j, n), values in sorted(spans.items(), key=lambda s: (-s[0][1], s[0][0])):
            if not taken & set(range(j, j + n)):
                taken |= set(range(j, j + n))
                chosen.append((j, n, values))
        chosen.sort(key=lambda c: c[0])
        pairs: list[bool | None] = []
        used: set[int] = set()
        for k, (j, n, values) in enumerate(chosen):
            # the value sits after the property ("weight 2 lb") or before it ("a macbook laptop type"), never past another property
            low = chosen[k - 1][0] + chosen[k - 1][1] if k else 0
            high = chosen[k + 1][0] if k + 1 < len(chosen) else len(cw)
            after: list[int] = []
            for i in range(j + n, high):
                if cw[i] in _CHAIN_BOUND and after:
                    break
                if cw[i] not in _CHAIN_BOUND:
                    after.append(i)
            before: list[int] = []
            for i in reversed(range(low, j)):
                if cw[i] in _CHAIN_BOUND:
                    break
                before.insert(0, i)
            used |= set(range(j, j + n))
            sides = [(side, [_value_fit(_words(v), [cw[i] for i in side]) for v in values]) for side in (after, before) if side]
            fits = [f for _, side_fits in sides for f in side_fits]
            if fits:
                pairs.append(True if any(fits) else False if False in fits else None)
                for side, side_fits in sides:
                    if any(side_fits) or not any(fits):
                        used |= set(side)   # the side that states the value; the other side's words are details of their own
                if pairs[-1] is None and unsure is not None:
                    unsure += [[cw[i] for i in side if cw[i] not in _FILLER] for side, side_fits in sides if None in side_fits]
        oks += pairs
        rest = [w for i, w in enumerate(cw) if not (pairs and i in used) and w not in _FILLER and w not in named and w not in skip
                and w not in _POINTER_VERBS]   # RL: "a turret style camera at 4k": "at" is the pointer's word
        if rest:
            seen = sorted(said) + [w for v in facts.values() for piece in str(v).split() for w in _words(piece)]
            # no key of this page names the detail: its title or values must hold the phrase. "color black" on a page whose
            # options field says "Black" counts a little less than a page with its own color key, and only when the first word
            # is a whole key on another page of the shop ("men slip on" keeps "men": it is a value, not a property)
            # DP: a property of two or three words ("sleeve length long sleeve") that is a key on another page of the shop but not
            # on this one is checked the same way, by its value words alone
            lead = next((n for n in (3, 2, 1) if len(rest) > n and " ".join(rest[:n]) in shop_keys), 0)
            # RL: a figure counts only with the word after it ("6 kg" is not stated by a "2 kg" page that says "wifi 6" elsewhere)
            texts = own + [[w for piece in str(item.get("title") or "").split() for w in _words(piece)]]
            figures = all(any(_states(t, [w, rest[i + 1]]) for t in texts) for i, w in enumerate(rest[:-1]) if any(c.isdigit() for c in w))
            # RL: the property may come last ("wireless connectivity" on a page that states "wireless" but has no connectivity key)
            tail = next((n for n in (3, 2, 1) if len(rest) > n and " ".join(rest[-n:]) in shop_keys), 0)
            oks.append(True if _covers(seen, rest) and figures else 0.9 if lead and (lead > 1 or len(rest) == 2) and _covers(seen, rest[lead:])
                       else 0.9 if tail and figures and _covers(seen, rest[:-tail]) else False)
    # a detail that words cannot check ranks just below one the page states, well above one it contradicts
    return sum(ok if isinstance(ok, float) else 1.0 if ok else 0.98 if ok is None else 0.0 for ok in oks) / len(oks) if oks else None


_LISTING_WORD = re.compile(r"\b(?:page|listing|listed|sold|seller|posted)\b", re.I)
_DIFFER_WORD = re.compile(r"\b(?:anything but|anything bar|anything except|other than|set apart|bar|except|not|unlike|differ\w*)\b", re.I)


def _names_other_listing(text: Any) -> bool:
    """The text points at another listing: a word for a listing or a seller together with a proper name inside a sentence (a
    capitalised word right after a lower-case word or a comma, so never a sentence's first word) or a possessive name."""
    text = str(text or "")
    return bool(_LISTING_WORD.search(text)) and bool(re.search(r"(?<=[a-z,] )[A-Z][a-z][\w-]+|\b[A-Z][a-z]+'s\b", text))


def _listing_names(sentence: str, name: str) -> bool:
    """LW: a listing word sends a page's reader to an unseen shop only when it takes that name ("Check the Dressers listing from
    Lutami (...)"); a page's blurb that merely holds such a word ("SOLD PER CANLibby's offers ready-to-eat ...", "Delivered by
    Seller") names no other listing."""
    return bool(re.search(r"\b(?:page|listing|listed|sold|seller|posted)\b\W+(?:[\w-]+\W+){0,3}?(?:from|by|of)\W+" + re.escape(name) + r"\b",
                          sentence, re.I))


def _pointer(item: dict[str, Any], prop: str) -> str | None:
    """A sentence of the listing's own text that sends this property to another listing ("For color, go by the X from Y,
    not this item" / "Color is to be taken from Y's shorts ..."): the text naming that other listing, or None."""
    def names(said: str) -> bool:
        return bool(_key_grade(said, prop) or _key_grade(prop, said) or _covers(_words(said), _words(prop)))
    for sentence in (c for s in _record_sentences(item) for c in [s] + re.split(r";\s+", s)[1:] if c.strip()):
        target = None
        lead = re.match(r"\s*for (?:the )?([^,]{2,50}),\s*(.+)", sentence, re.I)
        if lead and names(lead.group(1)):
            target = lead.group(2)
        taken = re.match(r"(.+?)\b(?:is|are|should be|must be)\s+(?:to be\s+)?(?:taken|read) from\s+(?:a different product:\s*)?(.+)", sentence, re.I)
        if not target and taken and names(taken.group(1)):
            target = taken.group(2)
        # PTR: "Take the length from the X, not from this pair" / "Its material isn't taken from this jacket; see the X for that"
        take = re.match(r"\s*(?:take|read|get) (?:the |its )?(.+?) from (.+)", sentence, re.I)
        if not target and take and names(take.group(1)):
            target = take.group(2)
        if not target and re.search(r"\b(?:isn't|is not|aren't|are not) (?:taken|read) from this\b", sentence, re.I):
            own = re.match(r"\s*(?:its |the )?(.+?) (?:isn't|is not|aren't|are not) (?:taken|read) from this\b", sentence, re.I)
            later = re.search(r"\b(?:see|look to|refer to|go by)\s+(.+)", sentence[own.end():] if own else "", re.I)
            if own and later and names(own.group(1)):
                target = later.group(1)
        if not target:
            continue
        target = re.sub(r"^\s*(?:go by|refer to|look to|look at|take it from|use|check|see|follow)\s+", "", target, flags=re.I)
        name = re.search(r"\b(?:the\s+)?[A-Z][\w-]+", target)
        if not re.match(r"\s*the\s", target):   # PTR: "the canned meat by X" keeps its kind and its "by X"
            target = target[name.start():] if name else target
        target = re.split(r",?\s+(?:(?:and |but )?not|instead|rather than|since|because|as (?:it|that|this)|for that)\b", target, maxsplit=1)[0].strip(" .")
        if re.search(r"\b[A-Z][\w-]+", target):
            return target
    return None


def _text_value(item: dict[str, Any], prop: str) -> str | None:
    """No key names the property: its value as one of the listing's own plain sentences states it ("with mixed fruits as
    the fruit type", "formal in occasion", "a solid pattern", "power consumption of 1-500")."""
    p = r"\s+".join(re.escape(w) for w in _words(prop))
    if not p:
        return None
    patterns = (r"\bwith (?:a |an )?([\w.-]+(?: [\w.-]+){0,2}) as (?:the |its )" + p + r"\b",
                r"\b" + p + r" (?:is|of) (?:a |an )?([\w.-]+(?: [\w.-]+)?)",
                r"\b([\w.-]+) in " + p + r"\b",
                r"\b(?:a|an) ([\w.-]+(?: [\w.-]+)?) " + p + r"\b")
    for sentence in _record_sentences(item):
        if _ONWARD.search(sentence):
            continue
        low = " ".join(_words(sentence))
        for pattern in patterns:
            found = re.search(pattern, low)
            if found and found.group(1).split()[-1] not in _FILLER:
                return found.group(1)
    return None


def _relation(words: Any) -> str:
    """RL: how a pointer's wanted value relates to the value of the page it points at, from the words before "the one pointed to"
    ("as" / "the very same as" -> is, "not" -> not, "the counterpart of" -> pair, "no lower than" -> min, "no higher than" -> max)."""
    text = " ".join(str(words or "").lower().split())
    if re.search(r"\bnot\b|\banything (?:bar|but|except)\b|\bother than\b|\bdiffer", text):
        return "not"
    if re.search(r"\bcounterpart\b|\bpartner\b|\bgoes (?:together )?with\b|\bpairs? with\b", text):
        return "pair"
    if re.search(r"\bno (?:lower|less|smaller|fewer|shorter|lighter|slower|weaker) than\b|\bat least\b", text):
        return "min"
    if re.search(r"\bno (?:higher|more|larger|greater|bigger|longer|heavier|faster|stronger) than\b|\bat most\b", text):
        return "max"
    return "is"


def _relation_end(text: Any) -> str:
    """RP: the relation a reply gives right before "the one pointed to": only the closing words count ("... falls away from not
    the one" -> not, "nothing except the counterpart of the one" -> pair, "no lower than the one" -> min), since the reply's
    framing words around them ("Do not show me anything where ...") say only how firm the shopper is."""
    tail = " ".join(str(text or "").lower().split()[-4:])
    if re.search(r"(?:^|\s)not$", tail):
        return "not"
    if re.search(r"\b(?:counterpart|partner) of$", tail):
        return "pair"
    if re.search(r"\bno (?:lower|less|smaller|fewer|shorter|lighter|slower|weaker) than$|\bat least$", tail):
        return "min"
    if re.search(r"\bno (?:higher|more|larger|greater|bigger|longer|heavier|faster|stronger) than$|\bat most$", tail):
        return "max"
    return "is"


def _speech_cut(details: str, vocab: frozenset[str]) -> str:
    """RP: a reply wraps its pointer in the shopper's own words after the details ("..., full stop", "... suits our household, and
    that is final"): from the end, a chunk with no word an opened listing names or states, or a chunk of two or more words left
    with fewer than two such words, is the shopper's clause and is cut, and the last kept chunk loses its trailing words no opened
    listing states. Only the end is cut, and a figure always counts, so the details themselves stay whole."""
    details = details or ""
    seps = list(re.finditer(r",\s*(?:and\s+)?|;\s*|\s+and\s+", details))
    starts = [0] + [m.end() for m in seps]
    chunks = [details[s:e] for s, e in zip(starts, [m.start() for m in seps] + [len(details)])]

    def known(word: str) -> bool:
        return any(w in vocab or any(c.isdigit() for c in w) for w in _words(word))
    while chunks:
        words = chunks[-1].split()
        core = [w for w in words if [x for x in _words(w) if x not in _FILLER and x not in _POINTER_VERBS]]
        kept = [w for w in core if known(w)]
        if core and (not kept or (len(core) >= 2 and len(kept) < 2)) and not any(c.isdigit() for c in chunks[-1]):
            chunks.pop()
            continue
        text = chunks[-1]
        while words and not known(words[-1]) and words[-1] in core and not (len(words) > 1 and any(c.isdigit() for c in words[-2])):
            words.pop()   # NC: a word right after a figure is its unit ("52 g"), never the shopper's clause
            text = re.sub(r"\s*\S+\s*$", "", text)
        # NC: the cut ends the shopper's own text where it ends; nothing cut leaves the details word for word
        if len(chunks) == len(starts) and text == chunks[-1]:
            return details
        return (details[:starts[len(chunks) - 1]] + text).strip(" ,;")
    return ""


_TAIL_START = re.compile(r"\s*(?:;|,\s*(?:and\s+)?(?=(?:the|i|we|my|our|you|it|that|this|like|as|since|because|so|but|which|full stop)\b)"
                         r"|\s(?=(?:suits?|when it comes to|because|since)\b))", re.I)


def _reply_tail_cut(text: str) -> str:
    """RP: a reply's page reference ends where the details of that page end; what follows is the shopper's own talk (", as I am
    not changing that", "; I have no patience left", " suits our household, and that is final", " when it comes to length,
    because ..."). The details themselves are property-value pairs joined by commas and "and", never a pronoun or an article."""
    at = re.search(r"\bwith\b", text)
    if not at:
        return text
    return text[:at.end()] + _TAIL_START.split(text[at.end():], maxsplit=1)[0]


_SELF_WORDS = {"its", "that", "this", "value", "values", "of", "our", "own", "is", "listing", "item", "product"}


def _pair_rule(items: list[dict[str, Any]], prop: str, value: str, near: int = 0) -> str | None:
    """The store's own pairing rule stated in listing text ("Our color pairing rule: black goes with pink, red with blue";
    "polyester goes together with taslan"; RL: "Its pack type, carton, goes together with single", "card type sd, which goes
    together with microsdxc", "That pink pairs with red"): the value it puts together with this one, or None. The first `near`
    items are the pages a reference means: there the rule need not name the property."""
    head = [w for w in _words(prop) if w not in _FILLER]
    strip = set(head) | {"that", "the", "this", "and", "one"}

    def core(text: str) -> list[str]:
        return [w for w in _words(text) if w not in strip]
    unit = strip - {w for w in _words(value) if w in head}   # RP: "4 channel goes with 8 channel": a word the value itself holds stays

    def shown(text: str) -> str:
        return " ".join(w for w in _words(text) if w not in unit)
    want = core(value)
    if not want:
        return None
    for rank, item in enumerate(items):
        for sentence in _record_sentences(item):
            if not re.search(r"\b(?:goes (?:together )?with|pairs with|is paired with)\b", sentence, re.I) \
                    or not (rank < near or any(w in _words(sentence) for w in head)):
                continue
            body = re.split(r":|\bis that\b", sentence, maxsplit=1)[-1]
            body = re.sub(r",\s*(?:which\s+)?(?=(?:goes|pairs|is paired)\b)", " ", body, flags=re.I)   # RL: "<prop>, carton, goes with"
            for clause in re.split(r",|\band\b", body):
                pair = re.match(r"\s*(.+?)\s+(?:goes\s+(?:together\s+)?|pairs\s+|is\s+paired\s+)?with\s+(.+)", clause.strip(" ."))
                if not pair:
                    continue
                a, b = core(pair.group(1)), core(pair.group(2))
                if a and b and len(a) == len(want) and all(_same(x, y) for x, y in zip(a, want)):
                    return shown(pair.group(2))
                # PV: "Its network connections value of 3g goes together with 4g" / "Its sugar free dietary needs value goes together
                # with low sugar": the page names its own value inside words that only point at itself and its own property keys
                own = _SELF_WORDS | {w for key in ((item.get("facts") or {}) if isinstance(item.get("facts"), dict) else {}) for w in _words(key)}
                if a and b and len(a) > len(want) and re.match(r"\s*(?:its|that|this)\b", pair.group(1), re.I):
                    at = next((i for i in range(len(a) - len(want) + 1) if all(_same(x, y) for x, y in zip(a[i:], want))), None)
                    if at is not None and all(w in own for w in a[:at] + a[at + len(want):]):
                        return shown(pair.group(2))
            for clause in re.split(r",|\band\b", body):
                pair = re.match(r"\s*(.+?)\s+(?:goes\s+(?:together\s+)?|pairs\s+|is\s+paired\s+)?with\s+(.+)", clause.strip(" ."))
                if not pair:
                    continue
                a, b = core(pair.group(1)), core(pair.group(2))
                if a and b and len(b) == len(want) and all(_same(x, y) for x, y in zip(b, want)):
                    return shown(pair.group(1))
    return None


def _same_value(a: Any, b: Any) -> bool:
    """RL: two values written alike, word by word ("Carton" / "carton", "8 channel" / "8 channels")."""
    wa, wb = _words(a), _words(b)
    return bool(wa) and len(wa) == len(wb) and all(_same(x, y) for x, y in zip(wa, wb))


def _stated_pair(item: dict[str, Any], prop: str) -> tuple[str, str] | None:
    """RL: a page that states its own value of the property inside its pairing rule ("Its pack type, carton, goes together with
    single"): (its value, the value paired with it), or None."""
    p = r"\s+".join(re.escape(w) for w in _words(prop))
    if not p:
        return None
    for sentence in _record_sentences(item):
        low = " ".join(_words(sentence))
        found = re.search(r"\b" + p + r" ([\w.-]+(?: [\w.-]+){0,2}?) (?:goes (?:together )?with|pairs with|is paired with) (?:that |the |a |an )?([\w.-]+(?: [\w.-]+){0,2})", low)
        if found and not re.search(r"\bnot\b|\binstead\b|\brather than\b", sentence, re.I):
            own = found.group(1).strip()
            partner = re.sub(r"\s+one$", "", found.group(2).strip())
            if own and partner and own.split()[-1] not in _FILLER:
                return own, partner
    return None


def _text_says(need: dict[str, Any], item: dict[str, Any] | None) -> bool:
    """No key names the property: a free-text sentence of the listing's own record (not its title) may state the value,
    unless that sentence negates or sends the reader to another listing."""
    for sentence in _record_sentences(item or {}):
        if _ONWARD.search(sentence):
            continue
        if any(_covers(_words(sentence), _words(v)) and (not _bare_number(need, v) or _free_count(sentence, v)) for v in need.get("values") or []):
            return True
    return False


def _related_against(need: dict[str, Any], facts: Any, opened: list[Any]) -> str:
    """RA: an 'is' need whose property shares a word with a fact key that names it only partly, where that key here states none
    of the wanted values while other opened listings use the same key for a wanted value's words: the listing says otherwise."""
    if need.get("test") != "is" or not isinstance(facts, dict):
        return ""
    values = [str(v) for v in need.get("values") or [] if str(v).strip()]
    words = [w for w in _words(need.get("property") or "") if w.isalpha() and len(w) >= 4]
    wanted = {w for v in values for w in _words(v) if w.isalpha() and len(w) >= 3}
    if not values or not words or not wanted:
        return ""
    against = ""
    for key, value in facts.items():
        if not isinstance(value, (str, int, float)) or isinstance(value, bool) or _key_grade(key, need.get("property")):
            continue
        if not any(_same(w, k) for w in words for k in _words(key)):
            continue
        if any(_says_in(value, v) for v in values):
            return ""
        used = any(isinstance(f.get(key), (str, int, float)) and any(_same(w, h) for w in wanted for h in _words(f.get(key)))
                   for f in opened if isinstance(f, dict) and f is not facts)
        if used and not any(_same(w, h) for w in wanted for h in _words(value)):
            against = str(key) + " says '" + str(value) + "'"
    return against


def read_listing(needs: list[dict[str, Any]], facts: dict[str, Any] | None, item: dict[str, Any] | None = None) -> dict[str, Any]:
    states = []
    for need in needs:
        state, note = read_need(need, facts)
        if facts is not None and need.get("test") == "is" and state != "met":
            if _section_says(need, item):
                state, note = "met", "its store section names it"
            elif state == "absent" and _title_says(need, item):
                state, note = "met", "no key names it and its title states it"
            elif state == "absent" and _option_says(need, facts):
                state, note = "met", "no key names it and its options field states it"
            elif state == "absent" and _text_says(need, item):
                state, note = "met", "no key names it and its own listing text states it"
            elif state == "absent" and _title_names_other(need, item, needs):
                state, note = "clash", "no key names it and its title names another " + str(need.get("property")) + ": " + _title_names_other(need, item, needs)
        if state == "clash" and need.get("said") == INFERRED_SAID:
            state, note = "weak", note + "; the wanted value was inferred from another listing"
        states.append((need, (state, note)))
    clashes = [(n, note) for n, (state, note) in states if state == "clash"]
    loose = [(n, state, note) for n, (state, note) in states if state in ("weak", "absent", "unsure", "mixed")]
    if facts is None:
        verdict = "unread"
    elif clashes:
        verdict = "fails"
    elif loose:
        verdict = "open"
    else:
        verdict = "fits"
    return {"verdict": verdict, "states": states, "clashes": clashes, "loose": loose}


def _longer_key(prop: Any, lines: list[str], keys: Any) -> str:
    """The store key naming a heard property together with the 1-2 words the shopper put before it ("style" heard in "with
    sneakers style" -> "sneakers style"), when an opened listing carries exactly that key; else ""."""
    want = _words(prop)
    if not want or len(want) > 2:
        return ""
    names: dict[str, str] = {}
    for key in keys:
        label = re.sub(r"\s+", " ", re.sub(r"[_.]+", " ", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key)))).strip().lower()
        names.setdefault(" ".join(_words(label)), label)
    for line in lines:
        said = [_stem(t) for t in re.findall(r"[a-z]+", str(line).lower())]
        for i in range(1, len(said) - len(want) + 1):
            if said[i:i + len(want)] != want:
                continue
            for extra in (2, 1):
                name = " ".join(said[i - extra:i + len(want)]) if i >= extra else ""
                if name in names and len(_words(name)) > len(want):
                    return names[name]
    return ""


def _need_sig(need: dict[str, Any]) -> str:
    """A need's property, test and values, for telling an unchanged need from a replaced one."""
    return _dump([" ".join(_words(need.get("property"))), need.get("test"), sorted(str(v).lower() for v in need.get("values") or [])])


def need_text(need: dict[str, Any]) -> str:
    prop = str(need.get("property") or "?")
    values = [str(v) for v in need.get("values") or []]
    unit = str(need.get("unit") or "")
    test = need.get("test")
    if test == "is":
        return prop + " is " + " or ".join(values) + (" (read from the listing the shopper pointed at)" if need.get("said") == INFERRED_SAID else "")
    if test == "not":
        return prop + " is not " + " / ".join(values)
    if test == "min":
        return prop + " at least " + " ".join(values[:1]) + (" " + unit if unit else "")
    if test == "max":
        return prop + " at most " + " ".join(values[:1]) + (" " + unit if unit else "")
    if test == "range":
        return prop + " between " + " and ".join(values[:2]) + (" " + unit if unit else "")
    return prop + ": " + str(need.get("said") or " ".join(values))


def _test_word(test: str) -> str:
    """A need's test written in other words or signs (the seats' own wording, never product words) read as the test it names."""
    t = re.sub(r"[\s_-]+", " ", test.strip().lower())
    if re.fullmatch(r"at least|>=|min(imum)?|gte|no less than", t):
        return "min"
    if re.fullmatch(r"at most|<=|max(imum)?|lte|no more than|up to", t):
        return "max"
    if re.fullmatch(r"(is )?not( equals?| in| one of)?|!=|none of|excludes?", t):
        return "not"
    if re.fullmatch(r"(is )?(equals?|eq|==?|exactly|one of|in|any of)|includes|contains", t):
        return "is"
    if t == "between":
        return "range"
    return t


def clean_needs(raw: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if isinstance(entry, str) and entry.strip():
            prop, _, rest = entry.partition(":")
            entry = {"property": prop.strip(), "test": "other", "values": [rest.strip()] if rest.strip() else [], "said": entry}
        if not isinstance(entry, dict):
            continue
        prop = str(entry.get("property") or "").strip()
        if not prop:
            continue
        test = _test_word(str(entry.get("test") or "is"))
        if test not in ("is", "not", "min", "max", "range", "other"):
            test = "other"
        values = entry.get("values")
        if not isinstance(values, list):
            values = [values] if values not in (None, "") else []
        out.append({"property": prop, "test": test, "values": [str(v) for v in values if str(v).strip()],
                    "unit": str(entry.get("unit") or "").strip(), "said": str(entry.get("said") or "").strip()})
    return out


# ---------------------------------------------------------------- the shelf: every listing seen

UNSORTED_PAGES = 6      # pages of an unsorted search still worth opening for a cheapest pick


class Shelf:
    def __init__(self, request: str = "") -> None:
        _FACT_VALUES.clear()
        self.request_words = set(_words(request))
        self.items: dict[str, dict[str, Any]] = {}
        self.stats: dict[str, dict[str, Any]] = {}
        self.id_key: str | None = None
        self.known_ids: set[str] = set()

    def _note_rows(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            for key, value in row.items():
                entry = self.stats.setdefault(key, {"rows": 0, "values": {}, "kinds": {}})
                entry["rows"] += 1
                kind = "bool" if isinstance(value, bool) else "num" if _is_number(value) else "str" if isinstance(value, str) else "none" if value is None else "other"
                entry["kinds"][kind] = entry["kinds"].get(kind, 0) + 1
                marker = value if isinstance(value, str) and len(value) <= 80 else ("<null-" + str(entry["rows"]) + ">" if value is None else ("<long-" + str(entry["rows"]) + ">" if isinstance(value, str) and len(value) <= 240 else "<long>"))
                entry["chars"] = entry.get("chars", 0) + (len(value) if isinstance(value, str) else 0)
                entry["values"][marker] = entry["values"].get(marker, 0) + 1

    def _pick_key(self, kind: str) -> str | None:
        best, best_rows = None, 0
        named, named_rows = None, 0   # SH: a qualifying field whose name says its role
        role = r"price|cost|amount" if kind == "num" else r"stock|avail" if kind == "bool" else None
        for key, entry in self.stats.items():
            share = entry["kinds"].get(kind, 0)
            if share and share >= entry["rows"] * 0.6 and entry["rows"] > best_rows:
                best, best_rows = key, entry["rows"]
            if role and share and share >= entry["rows"] * 0.6 and entry["rows"] > named_rows and re.search(role, str(key), re.I):
                named, named_rows = key, entry["rows"]
        return named if named is not None and named_rows >= best_rows * 0.6 else best

    def price_key(self) -> str | None:
        return self._pick_key("num")

    def stock_key(self) -> str | None:
        return self._pick_key("bool")

    def title_key(self) -> str | None:
        """The row field that behaves like a title: present on most rows, different on almost every row, short."""
        best, best_len = None, 1e9
        section = self.section_key()
        for key, entry in self.stats.items():
            if key in (self.id_key, section) or entry["rows"] < 2 or entry["kinds"].get("str", 0) < entry["rows"] * 0.8:
                continue
            filled = entry["rows"] - entry["values"].get("", 0)
            ratio = len(entry["values"]) / entry["rows"]
            mean = entry.get("chars", 0) / max(1, entry["rows"])
            if filled >= entry["rows"] * 0.8 and ratio >= 0.8 and 12 <= mean < best_len:
                best, best_len = key, mean
        return best

    def section_key(self) -> str | None:
        """The row field that behaves like a store section: short labels shared by many rows; a label that echoes the
        request's own words wins a tie (a brand rarely appears in the request)."""
        best, best_score, best_ratio = None, 9.0, 2.0
        for key, entry in self.stats.items():
            if key == self.id_key or entry["rows"] < 2:
                continue
            strings = entry["kinds"].get("str", 0)
            if strings < entry["rows"] * 0.8 or "<long>" in entry["values"]:
                continue
            letters = [v for v in entry["values"] if isinstance(v, str) and re.search(r"[A-Za-z]", v) and not v.startswith("<")]
            if not letters:
                continue
            ratio = len(entry["values"]) / entry["rows"]
            echoed = sum(n for v, n in entry["values"].items() if v in letters and _covers(list(self.request_words), _words(v)[:1]))
            score = ratio - 0.5 * (echoed / entry["rows"])
            if score < best_score:
                best, best_score, best_ratio = key, score, ratio
        return best if best_ratio < 0.85 else None

    def _row_id(self, row: dict[str, Any]) -> str | None:
        if self.id_key and _id_like(row.get(self.id_key)):
            return row[self.id_key]
        for key, value in row.items():
            if _id_like(value) and not isinstance(value, bool):
                return value
        return None

    def _choose_id_key(self, rows: list[dict[str, Any]]) -> None:
        if self.id_key:
            return
        candidates = []
        for key in rows[0]:
            values = [r.get(key) for r in rows]
            if all(_id_like(v) for v in values) and len(set(values)) == len(values):
                candidates.append((len({str(v)[:2] for v in values}) != 1, key))
        if candidates:
            self.id_key = sorted(candidates, key=lambda c: c[0])[0][1]

    @staticmethod
    def _facts_of(record: dict[str, Any]) -> dict[str, Any] | None:
        best = None
        for value in record.values():
            if isinstance(value, dict) and all(not isinstance(v, (dict, list)) for v in value.values()):
                if best is None or len(value) > len(best):
                    best = value
        listed = None   # SH: a property dict with list-of-scalar values, larger than any flat one
        for value in record.values():
            if isinstance(value, dict) and any(isinstance(v, list) for v in value.values()) and all(
                    not isinstance(v, dict) and not (isinstance(v, list) and any(isinstance(x, (dict, list)) for x in v)) for v in value.values()):
                if listed is None or len(value) > len(listed):
                    listed = value
        if listed is not None and (best is None or len(listed) > len(best)):
            return {k: (", ".join(str(x) for x in v) if isinstance(v, list) else v) for k, v in listed.items()}
        if best is None:   # SH: a list of two-field {name, value} records
            for value in record.values():
                pairs = value if isinstance(value, list) and len(value) >= 2 and all(isinstance(p, dict) and len(p) == 2 for p in value) else None
                keys = list(pairs[0].keys()) if pairs else []
                if not pairs or any(list(p.keys()) != keys for p in pairs):
                    continue
                for name_key in keys:
                    names = [p[name_key] for p in pairs]
                    other = [k for k in keys if k != name_key][0]
                    if all(isinstance(n, str) and n.strip() for n in names) and len(set(names)) == len(names) \
                            and all(not isinstance(p[other], (dict, list)) for p in pairs):
                        flat = {p[name_key]: p[other] for p in pairs}
                        if best is None or len(flat) > len(best):
                            best = flat
                        break
        return best

    def _merge(self, item_id: str, record: dict[str, Any], step: int) -> dict[str, Any]:
        item = self.items.get(item_id)
        if item is None:
            item = {"id": item_id, "price": None, "stock": None, "section": None, "title": None, "facts": None, "gone": False, "first": step, "record": {}}
            self.items[item_id] = item
        price_key, stock_key, section_key = self.price_key(), self.stock_key(), self.section_key()
        if price_key and _is_number(record.get(price_key)):
            item["price"] = record[price_key]
        if stock_key and isinstance(record.get(stock_key), bool):
            item["stock"] = record[stock_key]
        if section_key and isinstance(record.get(section_key), str):
            item["section"] = record[section_key]
        title_key = self.title_key()
        if title_key and isinstance(record.get(title_key), str) and record[title_key].strip():
            item["title"] = record[title_key]
        fresh = {k: v for k, v in record.items() if not isinstance(v, (dict, list))}
        merged = {k: v for k, v in (item.get("record") or {}).items() if k not in fresh and not (isinstance(v, str) and len(v) > 300)}
        merged.update(fresh)
        item["record"] = merged
        return item

    def take(self, observation: Any, role: str, step: int) -> list[str]:
        """File every listing in a result. Returns the ids touched."""
        touched: list[str] = []
        self._collect_ids(observation, 0)
        if role not in ("search", "details", "view"):
            return touched
        for rows in self._row_lists(observation, 0):
            self._choose_id_key(rows)
            self._note_rows([r for r in rows if self._row_id(r) not in self.items])
            for row in rows:
                item_id = self._row_id(row)
                if item_id:
                    self._merge(item_id, row, step)
                    touched.append(item_id)
        if role in ("details", "view"):
            for record in self._records(observation, 0):
                item_id = self._row_id(record)
                if not item_id:
                    continue
                item = self._merge(item_id, record, step)
                facts = self._facts_of(record)
                if role == "view":
                    item["viewed"] = True
                if role == "details":
                    item["facts"] = dict(facts) if isinstance(facts, dict) else {}
                    _note_values(item["facts"])
                touched.append(item_id)
        return touched

    def take_detail_rows(self, observation: Any, step: int) -> list[str]:
        """Listing records in a list (a tool that shows several listings' details at once): every record with an id and its
        own flat property list is filed as opened, as a details result is."""
        touched: list[str] = []
        for rows in self._row_lists(observation, 0):
            self._choose_id_key(rows)
            self._note_rows([r for r in rows if self._row_id(r) not in self.items])
            for row in rows:
                item_id = self._row_id(row)
                facts = self._facts_of(row)
                if item_id and isinstance(facts, dict) and len(facts) >= 3:
                    item = self._merge(item_id, row, step)
                    item["facts"] = dict(facts)
                    _note_values(item["facts"])
                    touched.append(item_id)
        return touched

    def _row_lists(self, value: Any, depth: int) -> list[list[dict[str, Any]]]:
        found: list[list[dict[str, Any]]] = []
        if depth > 4:
            return found
        if isinstance(value, list):
            dicts = [v for v in value if isinstance(v, dict)]
            if dicts and len(dicts) == len(value) and any(len(d) >= 3 for d in dicts):
                found.append(dicts)
                return found
            for element in value:
                found.extend(self._row_lists(element, depth + 1))
        elif isinstance(value, dict):
            if len(value) >= 2 and all(_id_like(k) and isinstance(v, dict) and len(v) >= 3 for k, v in value.items()):
                # SH: rows keyed by id; the key is filed as "id" when the record carries no id-like value of its own
                found.append([v if any(_id_like(x) for x in v.values()) or "id" in v else dict(v, id=k) for k, v in value.items()])
                return found
            for element in value.values():
                if isinstance(element, (list, dict)):
                    found.extend(self._row_lists(element, depth + 1))
        return found

    def _records(self, value: Any, depth: int) -> list[dict[str, Any]]:
        if depth > 3 or not isinstance(value, dict):
            return []
        if any(_id_like(v) for v in value.values()) and len(value) >= 2:
            return [value]
        out: list[dict[str, Any]] = []
        for element in value.values():
            if isinstance(element, dict):
                out.extend(self._records(element, depth + 1))
        return out

    def _collect_ids(self, value: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(value, dict):
            for element in value.values():
                if _id_like(element):
                    self.known_ids.add(element)
                else:
                    self._collect_ids(element, depth + 1)
        elif isinstance(value, list):
            for element in value:
                self._collect_ids(element, depth + 1)

    def affordable(self, item: dict[str, Any], cap: Any, count: int) -> bool:
        if item.get("gone") or item.get("stock") is False:
            return False
        if _is_number(cap) and _is_number(item.get("price")) and item["price"] * max(1, count) > cap + 1e-9:
            return False
        return True

    def ranked(self, needs: list[dict[str, Any]], cap: Any, count: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        out = []
        for item in self.items.values():
            if item["facts"] is None or not self.affordable(item, cap, count):
                continue
            out.append((item, read_listing(needs, item["facts"], item)))
        order = {"fits": 0, "open": 1, "fails": 2}
        out.sort(key=lambda pair: (order.get(pair[1]["verdict"], 3), pair[0]["price"] if _is_number(pair[0]["price"]) else 1e18))
        return out

    def cheaper_rows(self, item: dict[str, Any], cap: Any, count: int) -> list[dict[str, Any]]:
        price = item.get("price")
        if not _is_number(price) or not item.get("section"):
            return []
        rows = []
        for other in self.items.values():
            if other["id"] == item["id"] or not _is_number(other.get("price")) or other["price"] >= price:
                continue
            if other.get("section") != item.get("section"):
                continue
            if self.affordable(other, cap, count):
                rows.append(other)
        rows.sort(key=lambda r: r["price"])
        return rows


def _label(item: dict[str, Any]) -> str:
    text = str(item["id"])
    if _is_number(item.get("price")):
        text += " at " + str(item["price"])
    if item.get("section"):
        text += " (" + str(item["section"]) + ")"
    return text


# ---------------------------------------------------------------- the network side

class Wire:
    def __init__(self, binding: dict[str, Any], started: float, trace: list[dict[str, Any]]):
        self.binding = binding
        self.started = started
        self.trace = trace
        self.models = ProxyClient(timeout=70, max_retries=1, retry_delay=1)
        self.store = ProxyClient(timeout=95, max_retries=2, retry_delay=1)
        self.spent = 0.0
        self.setup_extra = 0.0   # SB: reading spend beyond SETUP_SHARE, left out of the work budget
        self.scale = 1.0         # SL: the spend lines' multiple for this request (set once its list is read)
        self.step = 0
        self.finished = False
        self.broken = False
        self.driver_prompt = 0   # prompt tokens of the last call that carried tools (the driver's context size)
        self.key_dead = False    # KD: the run's key is spent; a message now would stop the whole run
        self.inflight_waited = 0.0   # IF: seconds spent waiting out in-flight reservation refusals this episode
        self.busy_waited = 0.0       # TM: seconds spent waiting out rate-limit (429) refusals this episode
        self.message_tool = None

    def left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

    def work_spent(self) -> float:
        return self.spent - self.setup_extra

    def ask_model(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, models: list[str], max_tokens: int) -> dict[str, Any] | None:
        for model in models:
            if self.key_dead:
                return None
            if model == REASONER and self.work_spent() >= SPEND_SOFT * getattr(self, "scale", 1.0):
                model = HELPER
            room = self.left()
            if room < LAST_CALL_SECONDS - 10:
                return None
            self.models.timeout = int(max(8, min(75, room - LAST_CALL_SECONDS + 15)))
            sent = messages
            if model.startswith("anthropic/") and len(messages) > 2:
                sent = [dict(m) for m in messages]
                for index in (0, len(sent) - 1):
                    content = sent[index].get("content")
                    if isinstance(content, str) and content:
                        sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
            body: dict[str, Any] = {"model": model, "messages": sent, "temperature": 0, "max_tokens": max_tokens}
            if tools:
                body["tools"] = tools
                body["tool_choice"] = "auto"
            reply = self.models.post_verbose("/inference/chat/completions", body)
            for attempt in range(5):   # IF: the key's in-flight reservation is full; it clears as other calls finish
                error = reply.error if isinstance(reply.error, dict) else {}
                said = str(error.get("body") or "").lower()
                # TM: a 429 is the proxy's or the provider's rate limit, which clears in seconds. Episodes that start together
                # call at the same moment, and an episode that took six 429s in a row at its first turn ended without a step
                # (the client tries once per call); the same call is sent again after a short, spread-out wait
                busy = error.get("status") == 429
                if busy:
                    pause = min((2.0 + random.uniform(0.0, 2.0)) * (attempt + 1), self.left() - LAST_CALL_SECONDS - 12, 60.0 - self.busy_waited)
                else:
                    pause = min(3.0 * (attempt + 1), self.left() - LAST_CALL_SECONDS - 12, 45.0 - self.inflight_waited)
                if not busy and (attempt >= 4 or error.get("status") != 402 or not ("in_flight" in said or "in-flight" in said)) or pause < 1:
                    break
                self.trace.append({"kind": "fire", "set": "TM" if busy else "IF", "model": model, "wait": round(pause, 1)})
                time.sleep(pause)
                if busy:
                    self.busy_waited += pause
                else:
                    self.inflight_waited += pause
                self.models.timeout = int(max(8, min(75, self.left() - LAST_CALL_SECONDS + 15)))
                reply = self.models.post_verbose("/inference/chat/completions", body)
            data = reply.data
            if isinstance(data, dict) and data.get("choices"):
                usage = data.get("usage") or {}
                if _is_number(usage.get("cost")):
                    self.spent += float(usage["cost"])
                if tools and _is_number(usage.get("prompt_tokens")):
                    self.driver_prompt = int(usage["prompt_tokens"])
                message = data["choices"][0].get("message") or {}
                if message.get("content") or message.get("tool_calls"):
                    return message
            self.trace.append({"kind": "model_failure", "model": model, "error": str(reply.error)[:300] if reply.error else "empty reply"})
            error = reply.error if isinstance(reply.error, dict) else {}
            said = str(error.get("body") or "").lower()
            if (error.get("status") == 403 and "key limit exceeded" in said) or \
                    (error.get("status") == 402 and ("insufficient credits" in said or "requires more credits" in said)):
                self.key_dead = True   # KD: the operator's key_exhausted rule
                self.trace.append({"kind": "fire", "set": "KD", "status": error.get("status")})
                return None
        return None

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 900) -> dict[str, Any] | None:
        message = self.ask_model([{"role": "user", "content": prompt}], None, models, max_tokens)
        value = _json_object((message or {}).get("content"))
        if value is None and message is not None and self.left() > 60:
            message = self.ask_model([{"role": "user", "content": prompt}], None, models, min(2 * max_tokens, 3200))
            value = _json_object((message or {}).get("content"))
        return value

    def act(self, actions: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.key_dead and self.message_tool and any(isinstance(a, dict) and a.get("name") == self.message_tool for a in actions):
            self.trace.append({"kind": "fire", "set": "KD", "blocked": self.message_tool})
            return None   # KD: the shopper's reader would call the spent key and stop the run
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        if len(actions) == 1:
            payload["action"] = actions[0]
        else:
            payload["actions"] = actions
        for _ in range(3):
            room = self.left()
            if room < 2:
                return None
            self.store.timeout = int(max(3, min(95, room - 1)))
            reply = self.store.post_verbose("/environment/call", payload)
            if isinstance(reply.data, dict):
                data = reply.data
                self.step = data["step"] if isinstance(data.get("step"), int) else self.step + len(actions)
                if data.get("finished"):
                    self.finished = True
                return data
            error = reply.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.trace.append({"kind": "store_failure", "status": status, "body": body[:300]})
            if status == 402:
                self.broken = True
                return None
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None
                expected = re.search(r"expected (\d+)", body)
                if not expected:
                    return None
                self.step = int(expected.group(1)) - 1
                payload["step"] = self.step + 1
                continue
            if status in (403, 422, 500, 503, 504):
                return None
        return None


# ---------------------------------------------------------------- the shopper's needs

class Ledger:
    def __init__(self) -> None:
        self.needs: list[dict[str, Any]] = []
        self.version = 0
        self.ceiling: Any = None
        self.money = ""
        self.count = 1
        self.dropped: list[str] = []
        self.indifferent: list[str] = []
        self.heard: set[str] = set()
        self.exchanges: list[str] = []

    def text(self, words: bool = False) -> str:
        parts = [need_text(n) + (" (shopper's words: \"" + n["said"] + "\")" if words and n.get("said") else "") for n in self.needs] or ["see the request"]
        if _is_number(self.ceiling):
            parts.append("total price at most " + str(self.ceiling) + (" " + self.money if self.money else ""))
        parts.append("quantity " + str(self.count))
        if self.dropped:
            parts.append("no longer needed: " + ", ".join(self.dropped))
        if self.indifferent:
            parts.append("shopper does not mind: " + ", ".join(self.indifferent))
        return "; ".join(parts)

    def as_json(self) -> str:
        return _dump(self.needs, 4000)


# ---------------------------------------------------------------- one episode

class Session:
    def __init__(self, problem_data: dict[str, Any]):
        self.started = time.monotonic()
        environment = problem_data["environment"]
        self.policy = environment["policy_view"]
        self.request = str(self.policy.get("query") or "")
        self.trace: list[dict[str, Any]] = []
        self.wire = Wire(dict(environment["binding"]), self.started, self.trace)
        self.tools = [t for t in self.policy.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict)]
        self.schemas = {t["function"]["name"]: t["function"].get("parameters") or {"type": "object", "properties": {}} for t in self.tools}
        self.step_limit = int(self.policy.get("max_steps") or 30)
        self.wire.finished = bool(self.policy.get("finished"))
        self.ledger = Ledger()
        self.shelf = Shelf(self.request)
        self.brief: dict[str, Any] = {}
        self.role: dict[str, str] = {}
        self.question_limit: int | None = None
        self.questions_sent = 0
        self.asked: list[str] = []
        self.asking = True
        self.ask_floor = 0
        self.searches: dict[str, dict[str, Any]] = {}
        self.section_searches: dict[str, dict[str, Any]] = {}   # section label -> the agent's own search args that named it
        self.page_values: set[Any] = set()
        self.text_totals: dict[frozenset[str], int] = {}   # SM: a first-page text search's content words -> the total it reported
        self.mode_votes = {"all": 0, "any": 0}
        self.lone_votes = {"all": 0, "any": 0}   # SV: single searches of 2+ words (under 50 rows: every word; 300 or more: any word)
        self.mode_proof = ""
        self.search_mode: str | None = None   # "all": every word is required (a word more, fewer rows); "any": any word matches
        self.token_sources: dict[str, set[str]] = {}   # page token -> the searches (paging left out, lower case) that returned it
        self.paging_notes: dict[str, str] = {}
        self.removed: dict[str, int] = {}
        self.finish_args: dict[str, dict[str, Any]] = {}
        self.cart: list[str] = []
        self.cart_touched = False
        self.bag: dict[str, int] = {}       # units per listing, as the newest cart view showed them
        self.bag_fresh = False              # a cart view was read after the last cart change
        self.adds: dict[str, int] = {}      # add calls per listing since it was last taken out
        self.add_mode: str | None = None
        self.orders_tried = 0
        self.refused = 0
        self.approval = False        # FB: the request asks to approve the basket before the order
        self.confirmed_cart = ""     # the basket (ids and units) last sent for approval, with the message count right after it
        self.cart_step = -1          # the step of the last cart change
        self.order_bounced = False
        self.decline_bounced = False
        self.forced_buy = False   # BS fired once
        self.list_pick: list[tuple[str, int]] = []   # the basket AS named for a list about to be declined
        self.swept = False
        self.read_after_cart: set[str] = set()
        self.stock_tools: set[str] = set()   # tools whose results have carried a listing's stock field
        self.reread_tried: set[str] = set()  # cart lines BF already re-opened since the last cart change
        self.multi_bad = False   # MV: the multi-listing reader returned rows with no property list (it is not a details read)
        self.multi_view_ok: bool | None = None   # MV2: whether the multi-listing reader's rows show what a single view shows
        self.read_by: dict[str, set[str]] = {}   # AV: role (details / view) -> listings read alone with that role's tool
        self.av_tried: set[tuple[str, str]] = set()
        self.verdicts: dict[str, str] = {}
        self.rescued: dict[str, bool] = {}
        self.guard_marks: dict[str, int] = {}
        self.recent: list[str] = []
        self.recent_text: dict[str, str] = {}   # DS: the filed result of each call in self.recent
        # RM: the filed result of each page opened since the last cart write, message, finish or notice; the same page opened
        # again is answered from here, with no step used (a page changes only through those: a price or stock change hits a cart line)
        self.read_memory: dict[str, tuple[int, str]] = {}
        self.read_answered: set[str] = set()
        self.chat: list[dict[str, Any]] = []
        self.call_no = 0

    # ---------------- clocks and counts
    def spend_line(self, base: float) -> float:
        """SL: a spend line for this request (scaled by its list lines)."""
        return base * getattr(self.wire, "scale", 1.0)

    def steps_left(self) -> int:
        return self.step_limit - self.wire.step

    def message_room(self) -> int:
        limit = self.question_limit if self.question_limit is not None else DEFAULT_QUESTIONS
        return max(0, limit - self.questions_sent)

    def question_room(self) -> int:
        """Messages the planner may spend on questions: one stays back for the basket's approval when the request asks for it."""
        return max(0, self.message_room() - (1 if self.approval else 0))

    def mark(self, key: str) -> int:
        """Count how often a guard fired for one key; the first time returns 1."""
        self.guard_marks[key] = self.guard_marks.get(key, 0) + 1
        return self.guard_marks[key]

    # ---------------- setup
    def real_parts(self, parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """VP: a shopping-list part is a product of its own only when the request names it as one. A part whose item is just the
        value of a property line ("meat products: chicken breast; meat products: luncheon meat" -> one listing stating both) goes
        back to the product it was said for as a need; a part whose item is only the listing a pointer names ("as stated on the
        page of <Shop>'s <kind> with ...") is that page, not something to buy, and is dropped."""
        low = " ".join(self.request.lower().split())

        def tokens(text: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+", text.lower()))   # every word in order (_words keeps one of each)
        values: dict[str, str] = {}
        for prop, value in re.findall(r"(?:^|[;.:]\s|\.\s)\s*([a-z][a-z &/-]{1,40}?):\s*([^;.:\n]{1,60}?)\s*(?=[;.]|$)", low):
            values.setdefault(tokens(value), prop.strip())
        flat = " " + tokens(low) + " "

        def pointed(item: str) -> bool:
            # every mention of the item stands right after a pointer's words for another listing's page
            said = " " + tokens(item) + " "
            hits = flat.count(said)
            cues = len(re.findall(r" pages? (?:of|for|from|by)(?: the)?(?: [a-z0-9]+ s)?" + re.escape(said), flat))   # never "of the same listing from <kind>"
            return hits > 0 and hits == cues
        # the words that describe a page the shopper points to ("Look at the page of Tabelno's binders & binder accessories with
        # binder type binding cover, and at any listing it points to")
        regions = [" " + tokens(m) + " " for m in re.findall(r"\bpages? (?:of|for) ([^.;]+)", low)]

        def only_there(text: str) -> bool:
            said = " " + tokens(text) + " "
            return said.strip() != "" and flat.count(said) > 0 and flat.count(said) == sum(r.count(said) for r in regions)

        def page_part(part: dict[str, Any]) -> bool:
            # a "product" made only of a pointed page's words: its kind stands in the page's description and every value it
            # needs is said nowhere else
            needs = [n for n in part.get("needs") or [] if isinstance(n, dict)]
            values = [" ".join(str(v) for v in (n.get("values") if isinstance(n.get("values"), list) else [n.get("values")])) for n in needs]
            return bool(needs) and any((" " + tokens(str(part.get("item") or "")) + " ") in r for r in regions) and all(only_there(v) for v in values)
        kept: list[dict[str, Any]] = []
        folded: list[tuple[int, dict[str, Any]]] = []
        dropped: list[str] = []
        for part in parts:
            item = str(part.get("item") or "").strip()
            key = tokens(item)
            if key in values and not part.get("needs"):
                folded.append((len(kept), {"property": values[key], "test": "is", "values": [item], "said": values[key] + ": " + item}))
            elif pointed(item) or page_part(part):
                dropped.append(item)
            else:
                kept.append(part)
        if not folded and not dropped:
            return parts
        if len(kept) >= 2:
            for at, need in folded:
                owner = kept[max(at - 1, 0)]
                owner["needs"] = list(owner.get("needs") or []) + [need]
        else:
            if kept:
                self.ledger.needs = clean_needs(kept[0].get("needs"))
                self.ledger.count = _as_int(kept[0].get("count")) or self.ledger.count
            self.ledger.needs += clean_needs([need for _, need in folded])
        self.trace.append({"kind": "fire", "set": "VP", "kept": [p.get("item") for p in kept], "folded": [n["said"] for _, n in folded], "dropped": dropped})
        return kept

    def read_request(self) -> None:
        lines = []
        for tool in self.tools:
            fn = tool["function"]
            lines.append("- " + fn["name"] + ": " + str(fn.get("description") or "") + " | arguments: " + _dump(fn.get("parameters") or {}, 2000))
        fixed = BRIEF_PROMPT + "\n\nTOOLS:\n" + "\n".join(lines)
        varying = "REQUEST:\n" + self.request
        if self.policy.get("opening") is not None:
            varying += "\n\nOPENING OBSERVATION:\n" + _dump(self.policy.get("opening"), 4000)
        content = [{"type": "text", "text": fixed, "cache_control": {"type": "ephemeral"}}, {"type": "text", "text": varying}]
        message = self.wire.ask_model([{"role": "user", "content": content}], None, [REASONER, DRIVER_SPARE], 4000)   # BX: room for a long reading
        brief = _json_object((message or {}).get("content")) or {}
        if not isinstance(brief.get("tools"), dict) and self.wire.left() > CLOSING_SECONDS + 120:
            # an unreadable reading (cut by the token cap, or not one object): ask once more with room to finish
            self.trace.append({"kind": "fire", "set": "JX", "first": len(str((message or {}).get("content") or ""))})
            message = self.wire.ask_model([{"role": "user", "content": content}], None, [REASONER, DRIVER_SPARE], 4000)
            brief = _json_object((message or {}).get("content")) or brief
        self.brief = brief
        roles = brief.get("tools") if isinstance(brief.get("tools"), dict) else {}
        self.role = {job: name for job, name in roles.items() if isinstance(name, str) and name in self.schemas}
        self.resolve_finish()
        limit = _as_int(brief.get("messages"))
        if limit is None:   # the reading missed it: the request's own words ("You may send 5 messages")
            found = re.search(r"\b(\d{1,3}) (?:messages|questions)\b", self.request, re.I)
            limit = int(found.group(1)) if found else None
        if limit is not None:
            self.question_limit = limit
        steps = _as_int(brief.get("steps"))
        if steps is None:
            found = re.search(r"\b(?:within|in|at most|up to|no more than) (\d{1,4}) (?:steps|moves|actions|tool calls|calls|turns)\b", self.request, re.I) \
                or re.search(r"\b(\d{1,4}) (?:steps|moves|actions|tool calls|turns)\b", self.request, re.I)
            steps = int(found.group(1)) if found else None
        if steps and steps < self.step_limit:
            self.step_limit = steps
        self.ledger.needs = clean_needs(brief.get("needs"))
        ceiling, unit = _as_written(brief.get("ceiling"), self.request, [brief.get("steps"), brief.get("messages"), brief.get("count")])
        if unit:
            self.trace.append({"kind": "fire", "set": "AF", "brief": [brief.get("ceiling"), brief.get("money")], "written": [ceiling, unit]})
            brief["ceiling"], brief["money"] = ceiling, unit
        self.ledger.ceiling = ceiling if _is_number(ceiling) else None
        self.ledger.money = str(brief.get("money") or "")
        self.ledger.count = _as_int(brief.get("count")) or 1
        parts = [p for p in brief.get("parts") or [] if isinstance(p, dict) and p.get("item")]
        if len(parts) == 1 and _words(brief.get("item") or "") and _words(brief.get("item")) != _words(parts[0].get("item")):
            # PL: the reading put the list's first product in "item" (with its count and needs) and only the others in "parts"
            # ("3 of ... from 1 Coffee; ... 1 of ... from Binders"); one part alone would make it a one-product request and the
            # other line would never be bought
            parts = [{"item": brief.get("item"), "count": brief.get("count"), "needs": brief.get("needs") or []}] + parts
            self.trace.append({"kind": "fire", "set": "PL", "parts": [p.get("item") for p in parts]})
        if len(parts) >= 2:
            parts = self.real_parts(parts)
        self.parts = parts if len(parts) >= 2 else []
        if self.parts:
            self.ledger.needs = []
            self.trace.append({"kind": "fire", "set": "AM", "parts": [[p.get("item"), p.get("count")] for p in self.parts]})
        # FT: each "make sure the <part> fits the <host>" sentence binds two list products (the part's line must suit the host's)
        self.fit_rules: list[tuple[int, int, str]] = []
        self.fit_loose: set[tuple[int, int]] = set()   # FD: pairs the request names only as "everything fits the X": kept by the data
        self.fit_gone: set[tuple[int, str]] = set()   # HS: (part, series) searched out with no fit
        self.fit_asked: dict[int, str] = {}   # SW: part -> the key of its series-free search for its stated values
        for sentence in _sentences(self.request) if self.parts else []:
            found = _FIT_RULE.search(sentence)
            part, host = (self.part_named(found.group("part")), self.part_named(found.group("host"))) if found else (None, None)
            word = None if found else _FIT_WORD.search(sentence)
            if word and not re.match(r"\s*:", sentence[word.end():]):   # FD: a fit word, not a property name ("compatibility: rugged")
                part, host = self.part_named(sentence[: word.start()]), self.part_named(sentence[word.end():])
                if part is None and host is not None and re.search(r"\b(?:everything|all|each|every|the rest|others?)\b", sentence[: word.start()], re.I):
                    for other in range(len(self.parts)):
                        if other != host:
                            self.fit_rules.append((other, host, sentence.strip()))
                            self.fit_loose.add((other, host))
                    continue
            if part is not None and host is not None and part != host:
                self.fit_rules.append((part, host, sentence.strip()))
        if self.fit_rules:
            self.trace.append({"kind": "fire", "set": "FT", "rules": [[self.parts[a].get("item"), self.parts[b].get("item")] for a, b, _ in self.fit_rules]})
        clauses = []
        for sentence in _sentences(self.request):
            pointer = re.search(r"\b(?:has|have|had) listed\b|\blisted by [A-Z][\w-]+|\bsold by [A-Z][\w-]+"
                                # LO: "Look at the page of <Shop>'s <kind> with <details>, and at any listing it points to"
                                r"|\bpages? (?:of|for) (?:the )?[A-Z][\w-]+'s\b[^,;]*?(?=,?\s*with\b)", sentence) if _names_other_listing(sentence) else None
            if pointer:
                tail = re.match(r",?\s*with (.+?)(?:, said outright\b|, either\b|,? and at any listing\b|$)", sentence[pointer.end():], re.I)
                if tail:
                    clauses.append(" ".join(_words(tail.group(1))))
        # PS: "For <prop> (<other name>), go by the page of <kind> from <Shop> with <details>, or by the listing it defers to" and
        # "in the style of the <Shop> <details> listing": the details only say which page is meant, and the property is read there
        self.pointer_props: dict[str, dict[str, str]] = {}
        self.pointer_clauses: list[str] = []
        for sentence in _sentences(self.request):
            lead = re.match(r"\s*for (?:the )?([^,()]{2,60}?)\s*(?:\(([^)]{2,40})\))?,\s*(?:go by|take it from|look to|refer to)\s+(.+)", sentence, re.I)
            target = lead.group(3) if lead else sentence
            shop = re.search(r"\b(?:from|by) ([A-Z][\w-]{3,})\b", target) if lead else re.search(r"\bin the style of (?:the )?([A-Z][\w-]{3,})\b", sentence)
            if not shop or (lead and not _names_other_listing(target)):
                # RL: "[For the <product>,] <prop>: <relation> the one pointed to (or stated) on the page of <Shop>'s <kind> with
                # <details>": the words before "pointed to" say how the wanted value relates to the page's own one
                for clause in re.split(r";\s+", sentence):
                    named = re.match(r"\s*(?:for (?:the )?([^,:;]{2,60}),\s*)?([a-z][^,:;()]{1,50}?)\s*:\s*([^:;]{0,80}?)\b(?:pointed to|stated)\b(.+)", clause, re.I)
                    rest = named.group(4) if named else ""
                    owner = re.search(r"\b([A-Z][\w-]{3,})'s\b", rest) or re.search(r"\b(?:from|by) ([A-Z][\w-]{3,})\b", rest)
                    if not named or not owner or not _names_other_listing(rest):
                        continue
                    tail = re.split(r",?\s+or (?:by|from) the listing\b|\band at any listing\b", rest[owner.end():], maxsplit=1)[0]
                    if _words(tail):
                        self.pointer_clauses.append(tail)
                    prop = named.group(2).strip().lower()
                    known = self.pointer_props.get(prop)
                    if known and known["shop"] != owner.group(1).lower():
                        self.pointer_props[prop] = dict(known, shop="")   # two pages for one property: neither settles it
                        continue
                    self.pointer_props[prop] = {"shop": owner.group(1).lower(), "target": rest, "alias": "", "rel": _relation(named.group(3)),
                                                "product": str(named.group(1) or "").strip().lower()}
                continue
            tail = re.split(r",?\s+or (?:by|from) the listing\b|,\s+said outright\b|\blisting\b", target[shop.end():], maxsplit=1)[0]
            if _words(tail):
                self.pointer_clauses.append(tail)
            if lead:
                self.pointer_props[lead.group(1).strip().lower()] = {"shop": shop.group(1).lower(), "target": target,
                                                                     "alias": str(lead.group(2) or "").strip().lower()}
        if clauses:
            def borrowed(need: dict[str, Any]) -> bool:
                said = " ".join(_words(need.get("said") or " ".join(str(v) for v in need.get("values") or [])))
                return bool(said) and any(said in c for c in clauses) and " ".join(_words(self.request)).count(said) == sum(c.count(said) for c in clauses)
            dropped = [need_text(n) for n in self.ledger.needs if borrowed(n)] + [need_text(n) for p in self.parts for n in clean_needs(p.get("needs")) if borrowed(n)]
            if dropped:
                self.trace.append({"kind": "fire", "set": "WF", "dropped": dropped})
                self.ledger.needs = [n for n in self.ledger.needs if not borrowed(n)]
                for part in self.parts:
                    part["needs"] = [n for n in clean_needs(part.get("needs")) if not borrowed(n)]
                brief["needs"] = self.ledger.needs if not self.parts else brief.get("needs")
        if self.pointer_props or self.pointer_clauses:
            self.point_needs()
        learn = [str(x) for x in brief.get("learn_first") or [] if str(x).strip()]
        if brief and self.ledger.ceiling is None and self.role.get("message"):
            learn = ["the most the shopper wants to spend in total (no price limit was given; skip if listed again)"] + learn
        brief["learn_first"] = learn
        self.trace.append({"kind": "brief", "brief": brief, "roles": self.role})

    def resolve_finish(self) -> None:
        """When one tool both checks out and declines (told apart by a fixed argument), learn which values select each."""
        order, decline = self.role.get("order"), self.role.get("decline")
        tool = order or decline
        if not tool or (order and decline and order != decline):
            return
        schema = self.schemas.get(tool) or {}
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        choices = [k for k in schema.get("required") or [] if isinstance(props.get(k), dict) and len(props[k].get("enum") or []) >= 2]
        if not choices:
            return
        fn = next((t["function"] for t in self.tools if t["function"]["name"] == tool), {})
        prompt = FINISH_PROMPT + "\n\nTOOL " + tool + ": " + str(fn.get("description") or "") + " | arguments: " + _dump(schema, 2000)
        result = self.wire.ask_json(prompt, [REASONER, DRIVER], 300) or {}
        picks: dict[str, dict[str, Any]] = {}
        for kind in ("order", "decline"):
            fixed = result.get(kind)
            if isinstance(fixed, dict) and fixed and all(isinstance(props.get(k), dict) and (not props[k].get("enum") or v in props[k]["enum"]) for k, v in fixed.items()):
                picks[kind] = dict(fixed)
        if order and order == decline and len(picks) == 1 and len(choices) == 1 and len(props[choices[0]]["enum"]) == 2:
            known = next(iter(picks.values())).get(choices[0])
            other = [v for v in props[choices[0]]["enum"] if v != known]
            if known in props[choices[0]]["enum"] and other:
                picks["decline" if "order" in picks else "order"] = {choices[0]: other[0]}
        if len(picks) == 2 and picks["order"] != picks["decline"]:
            self.finish_args = picks
            self.role["order"] = self.role["decline"] = tool
        self.trace.append({"kind": "finish_args", "tool": tool, "args": self.finish_args, "result": result})

    def final_kind(self, name: str, args: dict[str, Any]) -> str:
        """'order' or 'decline' for a finishing call, '' otherwise; a shared finishing tool is told apart by its fixed arguments."""
        order, decline = self.role.get("order"), self.role.get("decline")
        if not name or name not in (order, decline):
            return ""
        if order != decline:
            return "order" if name == order else "decline"
        for kind in ("decline", "order"):
            fixed = self.finish_args.get(kind)
            if fixed and all(args.get(k) == v for k, v in fixed.items()):
                return kind
        return "order"

    def final_action(self, kind: str) -> dict[str, Any] | None:
        name = self.role.get(kind)
        if not name:
            return None
        args = self.neutral_args(name)
        if self.role.get("order") == self.role.get("decline"):
            args.update(self.finish_args.get(kind) or {})
        if kind == "decline":
            args = self.decline_args(name, args)
        return {"name": name, "args": args}

    def opening_text(self) -> str:
        b = self.brief
        out = ["SHOPPER REQUEST:\n" + self.request]
        if self.policy.get("opening") is not None:
            out.append("OPENING OBSERVATION:\n" + _dump(self.policy.get("opening"), 6000))
        if b:
            notes = ["My reading of the request (check it against the request itself):"]
            if b.get("item"):
                notes.append("- product: " + str(b.get("item")))
            if getattr(self, "parts", []):
                notes.append("- a SHOPPING LIST of " + str(len(self.parts)) + " different products bought together in ONE order: one cart line per product, each at its own quantity; the budget is for the whole list")
                for part in self.parts:
                    notes.append("  - " + str(_as_int(part.get("count")) or 1) + " x " + str(part.get("item"))
                                 + "".join("; " + need_text(n) for n in clean_needs(part.get("needs"))))
            else:
                notes.append("- quantity: " + str(self.ledger.count))
            for need in self.ledger.needs:
                notes.append("- need: " + need_text(need))
            if self.ledger.ceiling is not None:
                notes.append("- total price at most " + str(self.ledger.ceiling) + " " + self.ledger.money)
            if b.get("on_hand_only"):
                notes.append("- only listings in stock")
            pick = b.get("pick")
            if pick == "cheapest":
                notes.append("- choose the CHEAPEST listing among all that fit; open candidates in ascending price")
            elif pick == "best":
                notes.append("- choose the best listing by: " + str(b.get("best_by") or "the stated measure"))
            else:
                notes.append("- no cheapest or best rule: any listing that meets everything within budget is right")
            for wish in b.get("wishes") or []:
                notes.append("- wish only, never a requirement (tie-breaker at most): " + str(wish))
            for rule in b.get("rules") or []:
                notes.append("- working rule: " + str(rule))
            for item in b.get("learn_first") or []:
                notes.append("- the planner will ask the shopper: " + str(item))
            notes.append("- steps allowed: " + str(self.step_limit))
            if self.role.get("message") and self.message_args("?") is None:
                notes.append("- questions: the planner cannot use this store's message tool, so YOU send them with " + str(self.role.get("message"))
                             + ", one property of the product per message (the shopper answers only the first thing a message asks), before you buy")
            if self.finish_args:
                notes.append("- " + str(self.role.get("order")) + " both checks out and declines: " + _dump(self.finish_args.get("order"))
                             + " places the order, " + _dump(self.finish_args.get("decline")) + " ends without buying")
            out.append("\n".join(notes))
        return "\n\n".join(out)

    # ---------------- status line
    def shortlist(self) -> str:
        ranked = self.shelf.ranked(self.ledger.needs, self.ledger.ceiling, self.ledger.count)
        if not ranked:
            return ""
        fits = [i for i, r in ranked if r["verdict"] == "fits"]
        open_ = [(i, r) for i, r in ranked if r["verdict"] == "open"]
        fails = [(i, r) for i, r in ranked if r["verdict"] == "fails"]
        parts = []
        if fits:
            parts.append("facts state every need: " + "; ".join(_label(i) for i in fits[:5]))
        if open_:
            parts.append("not proven: " + "; ".join(_label(i) + " [" + ", ".join(n.get("property", "?") for n, _, _ in r["loose"][:2]) + "]" for i, r in open_[:3]))
        if fails:
            parts.append("facts seem to say otherwise (a same-meaning value can still count): " + "; ".join(_label(i) + " [" + r["clashes"][0][1][:60] + "]" for i, r in fails[:3]))
        return " | ".join(parts)

    def pointed(self, prop: Any) -> str | None:
        """PS: the request's pointer property (or its other name) that this property names, or None."""
        for name, ptr in getattr(self, "pointer_props", {}).items():
            for p in (name, ptr.get("alias")):
                if p and (_key_grade(prop, p) == 2 or _key_grade(p, prop) == 2):
                    return name
        return None

    def reply_pointer(self, lines: list[str], prop: str, hint: str) -> None:
        """RP: a reply that ties a property to another listing ("<relation> the one pointed to (or stated) on the page of <Shop>'s
        <kind> with <details>", "the same as the figure from the page of <kind> from <Shop> with <details>") points like the
        request's own pointer: it joins pointer_props, so the chain walk settles the need from the page it means. The property is
        the one the reply reader names (on a list, "<property> of the <product>" names its product); the relation is the words
        right before "the one", else the reader's (hint); a second page for one property settles neither."""
        prop = " ".join(str(prop or "").lower().split())
        if not prop:
            return
        for line in lines:
            at = re.search(r"\bthe one (?:pointed to|stated)\b", line, re.I) or re.search(r"\b(?:on|from) the (?:page|listing)\b", line, re.I)
            rest = (line[at.end():] if re.match(r"the one", at.group(0), re.I) else line[at.start():]) if at else ""
            owner = re.search(r"\b([A-Z][\w-]{3,})'s\b", rest) or re.search(r"\b(?:from|by) ([A-Z][\w-]{3,})\b", rest)
            if not at or not owner or not _names_other_listing(rest):
                continue
            rest = _reply_tail_cut(rest)
            name, product = prop, ""
            for item in [str(p.get("item") or "").strip().lower() for p in self.parts]:
                if item and re.search(r"\bof the " + re.escape(item) + r"\b", line[:at.start()], re.I):
                    product = item
                    name = re.sub(r"\s+of the " + re.escape(item) + r"$", "", name).strip() or name
            known = self.pointer_props.get(name)
            if known and known["shop"] != owner.group(1).lower():
                self.pointer_props[name] = dict(known, shop="")
                continue
            rel = _relation_end(line[:at.start()]) if re.match(r"the one", at.group(0), re.I) else hint
            self.pointer_props[name] = {"shop": owner.group(1).lower(), "target": rest, "alias": "", "rel": rel, "product": product}
            self.trace.append({"kind": "fire", "set": "RP", "property": name, "shop": owner.group(1).lower(), "relation": self.pointer_props[name]["rel"],
                               "product": product})

    def point_needs(self) -> None:
        """PS: a need on a pointer property holds no value of its own: a value made only of the pointer's words ("bra type: with
        wire" from "go by the page ... with wire") is the page's description, not the shopper's. On a list, the pointer belongs to the
        product whose own words name its shop; the same property copied onto the list's other products is dropped."""
        if self.parts:
            for part in self.parts:
                part["needs"] = clean_needs(part.get("needs"))
        def flat(text: Any) -> str:
            return " " + " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", str(text).lower())) + " "
        clauses = [flat(c) for c in getattr(self, "pointer_clauses", [])]
        whole = flat(self.request)

        def borrowed(need: dict[str, Any]) -> bool:
            # its words stand only inside a pointer's description of the page ("material cotton" in "... from Belshianan with
            # wire, material cotton, size class regular"), never in the shopper's own words for the product
            said = flat(need.get("said") or "")
            return said.strip() != "" and any(said in c for c in clauses) and whole.count(said) == sum(c.count(said) for c in clauses)
        cleared, dropped = [], []
        if self.parts:
            for part in self.parts:
                dropped += [str(part.get("item")) + ": " + need_text(n) for n in part["needs"] if borrowed(n)]
                part["needs"] = [n for n in part["needs"] if not borrowed(n)]
        else:
            dropped += [need_text(n) for n in self.ledger.needs if borrowed(n)]
            self.ledger.needs = [n for n in self.ledger.needs if not borrowed(n)]
        groups = [self.ledger.needs] + [part["needs"] for part in self.parts]
        for name, ptr in self.pointer_props.items():
            here = set(_words(ptr["target"]))
            for needs in groups:
                for need in needs:
                    if self.pointed(need.get("property")) == name and need.get("values") \
                            and all(_covers(sorted(here), _words(v)) for v in need["values"]):
                        cleared.append(need_text(need))
                        need["values"] = []
                        need["test"] = "is"
            if len(self.parts) >= 2:
                def names_shop(part: dict[str, Any]) -> bool:
                    return any(ptr["shop"] in _words(str(n.get("said") or "") + " " + " ".join(n.get("values") or []))
                               for n in part["needs"] if not re.match(r"\s*for\b", str(n.get("said") or ""), re.I)) \
                        or ptr["shop"] in _words(part.get("item") or "")
                product = ptr.get("product") or ""
                # RL: "For the IP Security Cameras, input voltage: ..." names the product the pointer belongs to
                owners = [part for part in self.parts if product and (_covers(_words(part.get("item") or ""), _words(product))
                                                                      or _covers(_words(product), _words(part.get("item") or "")))] \
                    or [part for part in self.parts if names_shop(part)]
                for part in self.parts:
                    if owners and part not in owners:
                        gone = [n for n in part["needs"] if self.pointed(n.get("property")) == name and not n.get("values")]
                        dropped += [str(part.get("item")) + ": " + need_text(n) for n in gone]
                        part["needs"] = [n for n in part["needs"] if n not in gone]
        self.trace.append({"kind": "fire", "set": "PS", "pointers": {k: v["shop"] for k, v in self.pointer_props.items()},
                           "cleared": cleared, "dropped": dropped})

    def onward_shop(self, text: str) -> str | None:
        """RL: the shop a page's pointer hands a property on to: "from/by X" or "X's" first, else "the X <kind>" ("the Mini PC
        from Voanko" -> voanko, "the Kokora Mini PC" -> kokora), else the name the pointed text starts with ("Anramovo in a
        multipack pack type"); a section's, a list line's or a maker's word is no shop."""
        kinds = {w for i in self.shelf.items.values() for w in _words(str(i.get("section") or ""))}
        kinds |= {w for p in getattr(self, "parts", []) for w in _words(str(p.get("item") or ""))}
        makers = self.makers()
        for pattern in (r"\b(?:from|by) ([A-Z][\w-]{3,})", r"\b([A-Z][\w-]{3,})'s\b", r"\bthe ([A-Z][\w-]{3,})\b", r"^\s*([A-Z][\w-]{3,})\b"):
            for m in re.finditer(pattern, text):
                if not set(_words(m.group(1))) & kinds and m.group(1).lower() not in makers:
                    return m.group(1).lower()
        return None

    def makers(self) -> set[str]:
        """The makers the opened rows name (a brand field): "the Ralenmo camera" names a maker, not a shop."""
        return {str(v).strip().lower() for i in self.shelf.items.values() for k, v in (i.get("record") or {}).items()
                if isinstance(v, str) and re.search(r"brand|producer|maker|manufacturer", str(k), re.I)}

    def settle_pointer(self, shop: str, follow: list[dict[str, Any]], chain: list[dict[str, Any]] | None = None, sure: bool = True) -> str:
        """PS: the page a request pointer means is found (it states every detail the pointer gives): its own value for the
        pointed property becomes the need, read from the listing the shopper pointed at. A page that sends the property on to
        another listing settles nothing; that listing's shop takes the pointer over. RL: the pointer's relation decides the need:
        the same value, not that value, at least / at most it, or the value the store's pairing rule puts with it (the rule read
        on the meant page first, then on the chain's earlier pages); no pairing rule seen -> nothing is settled. Each set of
        meant pages is read once per view of them (PK: more of a page shown reads it again). Not sure (the closest pages show only part of the pointer's details): the pointer is only
        handed on, when every such page sends the property to the same shop; nothing is settled from them."""
        note = ""
        tried = getattr(self, "ps_tried", set())
        for name, ptr in list(getattr(self, "pointer_props", {}).items()):
            if ptr["shop"] != shop or name in getattr(self, "inferred", {}):
                continue
            # PK: a meant page read again with more of it shown (its page notes after its details) is read once more: the notes
            # may send the property on ("go by the Belnovex one, not by this item") where the details alone settled nothing; so is
            # a pairing pointer once the chain holds more pages (the pairing rule may sit on a page opened later)
            def shown(items: list[dict[str, Any]]) -> tuple[Any, ...]:
                return tuple(sorted((str(i["id"]), tuple(sorted(i.get("record") or {})), i.get("facts") is not None) for i in items))
            seen = (name, shop, shown(follow), sure, shown(chain or []) if (ptr.get("rel") or "is") == "pair" else ())
            if seen in tried:
                continue
            tried.add(seen)
            self.ps_tried = tried
            rel = ptr.get("rel") or "is"
            # RL: the meant pages that send the property on; the pointer moves only when they all name one shop
            onward = [_pointer(item, name) or (_pointer(item, ptr["alias"]) if ptr["alias"] else None) for item in follow]
            if any(onward):
                nxt = {self.onward_shop(o) for o in onward if o}
                if len(nxt) == 1 and None not in nxt and (all(onward) or sure):
                    self.pointer_props[name] = dict(ptr, shop=next(iter(nxt)))
                continue
            if not sure:
                continue
            view = self.role.get("view")
            if view and view != self.role.get("details") and not all(i.get("viewed") for i in follow):
                continue   # VW: a meant page whose own page is unread settles nothing yet: its notes may send the property on
                # ("..., not by this item"), and its details alone would settle the need on the wrong page's value
            values: dict[str, str] = {}
            for item in follow:
                for prop in [name] + ([ptr["alias"]] if ptr["alias"] else []):
                    hits = sorted([(-_key_grade(k, prop), str(k), str(v).strip()) for k, v in (item.get("facts") or {}).items()
                                   if _key_grade(k, prop) and isinstance(v, (str, int, float)) and str(v).strip()])
                    value = hits[0][2] if hits else _text_value(item, prop)
                    if not value and rel == "pair":
                        stated = _stated_pair(item, prop)
                        value = stated[0] if stated else None
                    if value:
                        values[str(item["id"])] = value
                        break
            if len({v.lower() for v in values.values()}) != 1 or len(values) != len(follow):
                continue
            value = next(iter(values.values()))
            wanted, test = value, rel if rel in ("not", "min", "max") else "is"
            if rel == "pair":
                pages = list(follow) + [i for i in chain or [] if i not in follow]
                stated = [st for i in follow for p in [name] + ([ptr["alias"]] if ptr["alias"] else []) for st in [_stated_pair(i, p)] if st]
                partner = next((b for a, b in stated if _same_value(a, value)), None) or _pair_rule(pages, name, value, near=len(follow))
                self.trace.append({"kind": "fire", "set": "RL", "shop": shop, "property": name, "value": value, "partner": partner})
                if not partner:
                    note += ("\n[reference chain] " + shop.title() + "'s page " + ", ".join(values) + " gives " + name + " '" + value
                             + "'; the shopper wants its counterpart, and no pairing rule naming it has been seen on the chain yet: read the"
                             " meant page's own notes (and the pages it points to) for what goes with '" + value + "'")
                    continue
                wanted = re.sub(r"^(?:a|an|the)\s+", "", partner.strip(), flags=re.I)   # "goes with a zipper" -> zipper
            groups = [self.ledger.needs] if not self.parts else [part["needs"] for part in self.parts]
            hit = False
            for needs in groups:
                for need in needs:
                    if self.pointed(need.get("property")) == name and not need.get("values"):
                        need.update(values=[wanted], test=test, said=INFERRED_SAID)
                        hit = True
            if not hit:
                product = ptr.get("product") or ""
                owner = [part for part in self.parts if product and _covers(_words(part.get("item") or ""), _words(product))] if self.parts else []
                target = self.ledger.needs if not self.parts else owner[0]["needs"] if len(owner) == 1 else None
                if target is not None:
                    target.append({"property": name, "test": test, "values": [wanted], "unit": "", "said": INFERRED_SAID})
                    hit = True
            self.trace.append({"kind": "fire", "set": "PS", "shop": shop, "page": list(values), "property": name, "value": value,
                               "relation": rel, "wanted": wanted, "set_need": hit})
            if hit:
                self.inferred = dict(getattr(self, "inferred", {}), **{name: (wanted, test)})
                self.ledger.version += 1
                says = {"is": "the product to buy must have the same " + name,
                        "not": "the product to buy must NOT have that " + name,
                        "min": "the product's " + name + " must be at least that",
                        "max": "the product's " + name + " must be at most that",
                        "pair": "the store's pairing rule puts '" + wanted + "' with it, so the product to buy must have " + name + " '" + wanted + "'"}[rel if rel in ("not", "min", "max", "pair") else "is"]
                note += ("\n[reference chain] " + shop.title() + "'s page " + ", ".join(values) + " gives " + name + " '" + value
                         + "': " + says + " (now a known need)")
        return note

    def chain_note(self) -> str:
        """Reference chain pages still to open (see NV): shop by shop, from the request's or shopper's pointer onwards. A pointer
        that gives details ("(10 ft cable length, nylon material)", "with style modern", "the men's slip on sneakers") means the
        one page of that shop which states them: the shop stays open until such a page is opened, and only that page's own
        pointers carry the chain on; the shop's other pages are look-alikes whose pointers lead off the chain (DD)."""
        sellers: dict[str, list[dict[str, Any]]] = {}
        for item in self.shelf.items.values():
            for key, value in (item.get("record") or {}).items():
                if isinstance(value, str) and re.search(r"merchant|seller|shop|store|vendor|sold", str(key), re.I) \
                        and not re.search(r"note|text|desc", str(key), re.I) and 0 < len(value.split()) <= 3:
                    sellers.setdefault(value.strip().lower(), []).append(item)
        strict = (r"\bthat ([A-Z][\w-]{3,}) (?:has|have|had) ", r"\b(?:from|by) ([A-Z][\w-]{3,})", r"\b([A-Z][\w-]{3,})'s\b")
        makers = self.makers()
        sections = sorted({str(i.get("section")) for i in self.shelf.items.values() if i.get("section")}, key=len, reverse=True)
        # PW: the section a list line names ends at its colon ("from Beads: set of beads in the same color as the Ranovovo ..."): the
        # words after it are the product's own and may name the shop a pointer means
        product_words = {w for s in sections + re.findall(r"of the same listing from ([^;.:]+)", self.request) + [str(p.get("item") or "") for p in getattr(self, "parts", [])]
                         for w in _words(s)}

        # NS: words the shopper's side writes in lower case somewhere ("that", "what") are no shop names ("That's as muddy ...")
        plain = set(re.findall(r"(?<![\w'-])[a-z][\w-]{3,}", "\n".join([self.request] + sorted(self.ledger.heard))))

        def pointers(text: str, loose: bool, own: str = "") -> list[tuple[str, str, str, bool]]:
            found: list[tuple[str, str, str, bool]] = []
            for sentence in _sentences(text):
                names = [m.group(1) for p in strict for m in re.finditer(p, sentence)]
                if own and not (_ONWARD.search(sentence) or _HOP.search(sentence)):
                    # SH: a page naming its own shop without sending anything elsewhere describes itself ("Anraanan's pants come
                    # in a plain pattern for summer"); its pointer to another page of that shop is the sentence that does
                    names = [n for n in names if n.lower() != own]
                # NS: a page names an unseen shop only in a pointer sentence; "ships from Manila", "powered by Intel Core" are no hop
                cue = bool(_ONWARD.search(sentence) or _HOP.search(sentence))
                hop = not loose or cue or bool(_LISTING_WORD.search(sentence))
                # NS: the request or a reply never opens a sentence with an unseen shop's name ("Declan's home office", "That's ...")
                opener = None if loose else re.match(r"\W*([A-Z][\w-]*)", sentence)
                names = [n for n in names if n.lower() in sellers or (hop and n.lower() not in plain and not (opener and opener.group(1) == n)
                                                                      and (not loose or cue or _listing_names(sentence, n)))]   # LW
                if loose:
                    names += [w for w in re.findall(r"\b[A-Z][\w-]{3,}\b", sentence) if w.lower() in sellers]
                    if _ONWARD.search(sentence):
                        # RL: a page's pointer may name the shop bare ("Take plug type from the Kokora Mini PC (...)")
                        names += [m.group(1) for m in re.finditer(r"\bthe ([A-Z][\w-]{3,})\b", sentence) if m.group(1).lower() not in makers]
                shelf = next((s for s in sections if _covers(_words(sentence), _words(s))), "")
                spots = [(s, sentence.lower().find(s.lower())) for s in sections if s.lower() in sentence.lower()]
                for n in names:
                    if len(spots) >= 2:
                        # RL: "For the IP Security Cameras, plug type: ... on the page of Belvovex's canned meat": the shop's own
                        # section is the one named nearest its name
                        at = sentence.find(n)
                        shelf = min(spots, key=lambda sp: (abs(sp[1] - at), -len(sp[0])))[0]
                    # SH: a page's own shop is a pointer only when named after the cue ("go by the other Anraanan Pants listing");
                    # named before it, the page describes itself ("Shisano's camera here is ...; for input voltage, look to ...")
                    cues = [m.start() for m in (_ONWARD.search(sentence), _HOP.search(sentence)) if m]
                    if own and n.lower() == own and not (cues and re.search(r"\b" + re.escape(n) + r"\b", sentence[min(cues):], re.I)):
                        continue
                    if not set(_words(n)) & product_words and n.lower() not in [f[0] for f in found]:
                        every = bool(re.search(r"\bevery\b[^.;]{0,80}\b" + re.escape(n) + r"\b", sentence))
                        found.append((n.lower(), shelf, _chain_details(sentence, n), every))   # a product or section word ("from Mass Gainer") is no shop
            return found
        vocab = frozenset(w for i in self.shelf.items.values() if isinstance(i.get("facts"), dict)
                          for text in [str(k) + " " + str(v) for k, v in i["facts"].items()] + [str(i.get("title") or ""), str(i.get("section") or "")]
                          for w in _words(text))   # DP: words opened listings state
        queue: list[tuple[str, str, str, bool]] = []
        for text in [self.request] + sorted(self.ledger.heard):
            if _names_other_listing(text) or re.search(r"\b[A-Z][a-z]+'s\b", text):
                found = pointers(text, False)
                if text != self.request:   # RP: a reply's own closing words are no details
                    found = [(n, shelf, _speech_cut(_reply_tail_cut(details), vocab), every) for n, shelf, details, every in found]
                queue += [p for p in found if p[0] not in [q[0] for q in queue]]

        def onward(p: tuple[str, str, str, bool]) -> bool:
            # SH: a meant page may send the chain on to ANOTHER page of a shop already walked ("For length, go by the other
            # Anraanan Pants listing (loose fit, cotton material, casual style), not this item"): a shop comes back when the
            # details differ from every earlier pointer to it
            said = " ".join(_words(p[2]))
            return all(q[0] != p[0] or (said and said != " ".join(_words(q[2])) and q[2].strip()) for q in queue)
        lines: list[str] = []
        aside: list[tuple[str, list[str], str]] = []
        done = 0
        settled_note = ""
        self.chain_unviewed: list[str] = []   # VS: meant pages read only through their details (the code opens their own pages)
        self.chain_hunt: list[tuple[str, str, str]] = []   # NH: pointers with details whose meant page no opened page surely is
        chained: list[dict[str, Any]] = []   # RL: the meant pages of the shops already walked
        while done < len(queue) and done < 8:   # RL: race-7 chains run up to six shops
            shop, shelf, details, every = queue[done]
            done += 1
            rows = [i for i in sellers.get(shop, []) if not shelf or i.get("section") == shelf] or sellers.get(shop, [])
            opened = [i for i in rows if i.get("facts") is not None]
            label = shop.title() + (" (" + shelf + ")" if shelf else "")
            skip = set(_words(shop)) | {w for i in rows for w in _words(i.get("section") or "")}
            shop_keys = frozenset(" ".join(_words(k)) for i in rows for k in (i.get("facts") or {}))
            fit = {str(i["id"]): (_detail_fit(i, details, skip, shop_keys, None, vocab) if details.strip() else None) for i in rows}
            unread = sorted([i for i in rows if i.get("facts") is None], key=lambda i: -(fit[str(i["id"])] or 0))
            follow = opened
            if details.strip() and not every and not any((fit[str(i["id"])] or 0) >= 0.9 for i in opened):
                self.chain_hunt.append((shop, shelf, details))
            if not rows:
                if any('"' + shop + '"' in str(e.get("label") or "").lower() for e in self.searches.values()):
                    continue   # NS: searched by its name alone and no listing of it came back: no shop of this store ("Declan's home office")
                lines.append(label + ": no listing of this shop seen yet, search the shop's name alone")
            elif every:
                follow = [i for i in opened if fit[str(i["id"])] is None or fit[str(i["id"])] >= 0.5]
                if unread:
                    lines.append(label + ": the pointer means EVERY such page of this shop, still to open: " + ", ".join(str(i["id"]) for i in unread[:12]))
            elif not opened:
                lines.append(label + ": seen, not opened: " + ", ".join(str(i["id"]) for i in unread[:6]))
            elif any(fit[str(i["id"])] is not None for i in opened):
                best = max(fit[str(i["id"])] or 0 for i in opened)
                follow = [i for i in opened if best >= 0.5 and (fit[str(i["id"])] or 0) == best]
                if follow:
                    settled_note += self.settle_pointer(shop, follow, chained, sure=best >= 0.9)   # PS
                    view = self.role.get("view")
                    unseen = [str(i["id"]) for i in follow if not i.get("viewed")] if view and view != self.role.get("details") else []
                    if unseen and best >= 0.5:
                        # VW: the chain goes on only from the meant page's own page (its notes), which its details do not show
                        lines.append(label + ": open the page itself (" + view + ") of " + ", ".join(unseen[:3]) + ", not only its details: its"
                                     " notes may send the property on to another listing")
                        self.chain_unviewed += [i for i in unseen[:3] if i not in self.chain_unviewed]
                wanted = " ".join(details.split())
                unsure: list[list[str]] = []
                for item in follow:
                    _detail_fit(item, details, skip, shop_keys, unsure, vocab)
                unsure = [u for k, u in enumerate(unsure) if u and u not in unsure[:k]]
                # a detail the closest page states only in other units: an unread page stating it as the pointer does may be the one
                literal = [i for i in unread if any(u and _covers(sorted(_listing_words(i)[0]), u) for u in unsure)]
                if not follow:
                    # VS: a page's notes are on its own page only, so the details a pointer gives ("listed at 6 kg") may sit in the notes of an
                    # opened page whose own page was never shown
                    view = self.role.get("view")
                    blind = [str(i["id"]) for i in opened if not i.get("viewed")] if view and view != self.role.get("details") else []
                    self.chain_unviewed += [i for i in blind[:4] if i not in self.chain_unviewed]
                    lines.append(label + ": none of its opened pages (" + ", ".join(str(i["id"]) for i in opened[:6]) + ") shows '" + wanted + "'"
                                 + (" in what was read; only the details of " + ", ".join(blind[:4]) + " were read, and their own pages (" + str(view) + ") may state it in their"
                                    " notes; else the page meant is another one: " if blind else ", so the page meant is another one: ")
                                 + ("open " + ", ".join(str(i["id"]) for i in unread[:6]) if unread else "search the shop's name with those details once"))
                elif best < 0.9 and unread:
                    lines.append(label + ": the closest opened page (" + str(follow[0]["id"]) + ") shows only part of '" + wanted + "'; also open " + ", ".join(str(i["id"]) for i in unread[:6]))
                elif literal:
                    lines.append(label + ": the closest opened page (" + str(follow[0]["id"]) + ") states " + " and ".join("'" + " ".join(u) + "'" for u in unsure[:2])
                                 + " only in other units; also open " + ", ".join(str(i["id"]) for i in literal[:6]) + ", which state it as the pointer does")
                off = [str(i["id"]) for i in opened if i not in follow]
                if off:
                    aside.append((label, off, " do not show '" + wanted + "'" + ((" (closest so far: " if literal else " (the page meant is ")
                                 + ", ".join(str(i["id"]) for i in follow[:3]) + ")" if follow else "")))
                    fired = getattr(self, "dd_fired", set())
                    if (shop, details) not in fired:
                        self.dd_fired = fired | {(shop, details)}
                        self.trace.append({"kind": "fire", "set": "DD", "shop": shop, "meant": [str(i["id"]) for i in follow[:3]], "off": off[:6], "best": round(best, 2)})
            chained += [i for i in follow if i not in chained]
            for item in follow:
                for value in (item.get("record") or {}).values():
                    if isinstance(value, str) and len(value) > 40:
                        queue += [p for p in pointers(value, True, shop) if onward(p)]
        text = ""
        if lines:
            text += ("\n[reference chain] pages still to open (the store checks that every page a pointer names was opened; a shop's own"
                     " listing is the row whose seller field is that shop, whatever its title): " + " | ".join(lines))
        # SH: a page one pointer passes over can be the page another pointer to the same shop means: it is on the chain
        meant = {str(i["id"]) for i in chained}
        aside_text = [label + ": " + ", ".join(rest[:6]) + said for label, off, said in aside for rest in [[i for i in off if i not in meant]] if rest]
        if aside_text:
            text += ("\n[reference chain] look-alike pages (same shop, but not the details the pointer gives): what they say and where they point"
                     " is NOT part of the chain, ignore it: " + " | ".join(aside_text))
        return text + settled_note


    def status(self) -> str:
        left = self.steps_left()
        bits = ["[status] steps used " + str(self.wire.step) + " of " + str(self.step_limit) + ", " + str(left) + " left"]
        limit = self.question_limit if self.question_limit is not None else DEFAULT_QUESTIONS
        bits.append("questions sent " + str(self.questions_sent) + " of " + str(limit))
        spare = int(max(0, self.wire.left() - LAST_CALL_SECONDS))
        if spare < 120:
            bits.append("about " + str(spare // 30 * 30) + " s left")
        text = "; ".join(bits) + ".\n[needs now] " + self.ledger.text()
        board = self.shortlist()
        if board:
            text += "\n[shortlist from opened listings] " + board
        if self.cart:
            shown = [i + (" (no longer available: remove it)" if self.shelf.items.get(i, {}).get("gone") else "") for i in self.cart]
            text += "\n[cart as I track it] " + ", ".join(shown)
            if self.approval and self.role.get("message") and self.message_room() > 0:
                text += ("\n[approval] the approval the request asks for is taken for you: your checkout call first shows this cart to the shopper"
                         " as the final basket and places the order only when the reply does not object. Do not wait for an approval; check out"
                         " once the cart holds the final basket.")   # AW: the driver held fitting carts back waiting for an approval no one asked
        if left > 6:
            text += self.chain_note() + self.fit_note()
        if left <= 2:
            text += "\nLAST STEPS: check out now if the cart holds a verified fit, otherwise decline."
        elif left <= 5 or self.wire.left() < CLOSING_SECONDS or self.wire.work_spent() >= self.spend_line(SPEND_HARD):
            text += "\nFINISH NOW: one verified line in the cart, then check out (or decline if nothing fits)."
        elif self.orders_tried == 0 and left <= 17:
            text += "\nSteps are running low: put your best verified fit in the cart and send the first checkout while at least 10 steps remain; searching ends when 7 steps remain, first checkouts are often refused because something changed, and a refused checkout needs about 4 steps to repair."
        usable = [t["function"]["name"] for t in self.offered_tools()]
        if len(usable) < len(self.driver_tools()):
            text += "\nUsable now: " + ", ".join(usable) + " (other tools are refused)."
        return text

    # ---------------- shopper lines
    def hear(self, lines: list[str], asked: str | None = None, pairs: list[tuple[str, str]] | None = None) -> str:
        fresh = [line for line in lines if line not in self.ledger.heard]
        if not fresh:
            return ""
        if self.wire.left() < LAST_CALL_SECONDS:
            return ""   # LN: too late to act on; left unheard
        self.ledger.heard.update(fresh)
        if not self.approval and self.role.get("message") and any(APPROVAL_RX.search(line) for line in fresh):
            self.approval = True   # FC: a reply or a notice asks to approve the basket
            self.trace.append({"kind": "fire", "set": "FC", "heard": True})
        prompt = HEAR_PROMPT
        if any(_names_other_listing(line) or re.search(r"\b[A-Z][a-z]+'s\b", line) for line in fresh):
            # only for a line naming someone's listing (a proper name with 's: every live race #3 reference line, ~9% of the
            # other lines), so every other line keeps the unchanged prompt for the team's free replay
            prompt += REFERENCE_RULE
        if getattr(self, "parts", []):
            prompt += (" The request is a shopping list of several products: name each need's property together with its product as the shopper"
                       " says it (for example \"length of the curtains and blinds\"), so that no need is applied to another product.")
        prompt += "\n\nTHE REQUEST (for context):\n" + self.request
        prompt += "\n\nCURRENT NEEDS (JSON):\n" + self.ledger.as_json()
        prompt += "\nCURRENT LIMITS: " + self.ledger.text()
        if self.ledger.exchanges:
            prompt += "\n\nEARLIER EXCHANGES (oldest first):\n" + "\n".join(self.ledger.exchanges[-10:])
        if pairs and len(pairs) > 1:
            prompt += "\n\nTHE ASSISTANT ASKED (several questions in one turn):\n" + "\n".join("Q" + str(i) + ": " + q for i, (q, _) in enumerate(pairs, 1))
            prompt += ("\n\nTHE SHOPPER NOW SAID (each answer replies only to the question with the same number; a \"don't mind\" answer is about"
                       " that question's property alone and never undoes a value given in another answer):\n"
                       + "\n".join("A" + str(i) + ": " + a for i, (_, a) in enumerate(pairs, 1)))
        else:
            if asked:
                prompt += "\n\nTHE ASSISTANT ASKED:\n" + asked
            prompt += "\n\nTHE SHOPPER NOW SAID:\n" + "\n".join(fresh)
        self.ledger.exchanges.append(("asked: " + asked + " -> " if asked else "") + "shopper: " + " / ".join(fresh))
        result = self.wire.ask_json(prompt, [HELPER, REASONER], 1600)
        if not result:
            return ""
        if isinstance(result.get("dropped"), list):
            said = _words(" ".join(fresh))
            named = [p for p in result["dropped"] if isinstance(p, str) and any(any(_same(w, h) for h in said) for w in _words(p) if len(w) > 2)]
            if len(named) != len(result["dropped"]):
                self.trace.append({"kind": "fire", "set": "CX", "unnamed": [p for p in result["dropped"] if p not in named]})
                result["dropped"] = named
        changes = []
        before = _dump(self.ledger.needs)
        prior = [_need_sig(n) for n in self.ledger.needs]
        if isinstance(result.get("needs_now"), list):
            fresh_needs = clean_needs(result.get("needs_now"))
            for need in fresh_needs:
                for prop_known, (value_known, test_known) in getattr(self, "inferred", {}).items():
                    if (_key_grade(need["property"], prop_known) or _key_grade(prop_known, need["property"])) and (
                            not need["values"] or need["values"] == [value_known] or any(_names_other_listing(v) for v in need["values"])):
                        need.update(values=[value_known], test=test_known, said=INFERRED_SAID)
            pointers = " ".join(str(r.get("listing") or "") for r in getattr(self, "ref_queue", []))
            plain = [line for line in fresh if not (_names_other_listing(line) or re.search(r"\b[A-Z][a-z]+'s\b", line))]

            def identifies(need: dict[str, Any]) -> bool:
                values = [str(v) for v in need.get("values") or [] if str(v).strip()]
                return bool(pointers and values) and _need_sig(need) not in prior and all(_covers(_words(pointers), _words(v)) for v in values) \
                    and not any(_covers(_words(line), _words(v)) for line in plain for v in values)
            borrowed = [n for n in fresh_needs if identifies(n)]
            if borrowed:
                self.trace.append({"kind": "fire", "set": "RF", "dropped": [need_text(n) for n in borrowed]})
                fresh_needs = [n for n in fresh_needs if n not in borrowed]
            keys = {k for it in self.shelf.items.values() if it.get("facts") for k in it["facts"]}
            for need in fresh_needs:
                longer = _longer_key(need.get("property"), fresh, keys) if _need_sig(need) not in prior else ""
                if longer:
                    self.trace.append({"kind": "fire", "set": "FP", "heard": need.get("property"), "now": longer})
                    need["property"] = longer
            if fresh_needs or not self.ledger.needs:
                self.ledger.needs = fresh_needs
        if _dump(self.ledger.needs) != before:
            changes.append("needs now: " + "; ".join(need_text(n) for n in self.ledger.needs))
        still: list[dict[str, Any]] = []
        for prop in result.get("indifferent") or []:
            if not isinstance(prop, str) or not prop:
                continue
            held = [n for n in self.ledger.needs if n.get("test") != "not" and (_key_grade(prop, n.get("property")) or _key_grade(n.get("property"), prop))]
            if held:
                still.extend(n for n in held if n not in still)    # "don't mind" never cancels what the request requires
            elif prop not in self.ledger.indifferent:
                self.ledger.indifferent.append(prop)
        for prop in result.get("dropped") or []:
            if isinstance(prop, str) and prop and prop not in self.ledger.dropped:
                self.ledger.dropped.append(prop)
                changes.append("dropped: " + prop)
        gone_props = [p for p in (result.get("dropped") or []) if isinstance(p, str) and p.strip()]
        if gone_props:
            kept = [n for n in self.ledger.needs if not (_need_sig(n) in prior and any(
                _key_grade(p, n.get("property")) == 2 or _key_grade(n.get("property"), p) == 2 for p in gone_props))]
            if len(kept) != len(self.ledger.needs):
                self.trace.append({"kind": "fire", "set": "CD", "dropped": gone_props,
                                   "removed": [need_text(n) for n in self.ledger.needs if n not in kept]})
                self.ledger.needs = kept
                changes.append("needs now: " + ("; ".join(need_text(n) for n in kept) or "none beyond the item, budget and quantity"))
        budget, unit = _as_written(result.get("budget_now"), "\n".join(fresh), [])
        if _is_number(budget) and budget > 0 and budget != self.ledger.ceiling:
            if unit:
                self.trace.append({"kind": "fire", "set": "AF", "heard": [result.get("budget_now"), budget, unit]})
                self.ledger.money = unit
            self.ledger.ceiling = budget
            changes.append("new budget: " + str(budget))
        if result.get("gone") is True:
            said = _words(" ".join(fresh))
            named = [i for i in self.cart if i in " ".join(fresh) or (len(_words((self.shelf.items.get(i) or {}).get("title") or "")) > 2
                     and sum(1 for w in set(_words((self.shelf.items.get(i) or {}).get("title") or "")) if len(w) > 3 and any(_same(w, h) for h in said)) >= 2)]
            hit = named or (self.cart[:1] if len(self.cart) == 1 else [])
            for item_id in hit:
                if item_id in self.shelf.items:
                    self.shelf.items[item_id]["gone"] = True
            if hit:
                changes.append("an item is no longer available (" + ", ".join(hit) + "): take it out of the cart and do not buy it")
            else:
                self.read_after_cart = set()
                self.reread_tried = set()
                changes.append("one cart line is no longer available: re-check each cart line's stock and replace only the one that is gone")
        if changes:
            self.ledger.version += 1
        meaning = "Meaning: " + str(result.get("gist") or "") + (" (" + "; ".join(changes) + ")" if changes else "")
        ref = result.get("reference")
        if isinstance(ref, dict) and str(ref.get("property") or "").strip() and str(ref.get("listing") or "").strip():
            prop, other = str(ref["property"]).strip(), str(ref["listing"]).strip()[:200]
            said = " ".join(fresh).lower()
            pair = "right for the details" in said
            same = pair or "a match for the details" in said or str(ref.get("relation") or "").lower().startswith("same")
            self.trace.append({"kind": "fire", "set": "BQ", "property": prop, "listing": other, "same": same, "pair": pair})
            self.ref_queue = getattr(self, "ref_queue", []) + [{"prop": prop, "listing": other, "same": same, "pair": pair}]
            meaning += (" The shopper ties the " + prop + " to ANOTHER listing (" + other + "); the facts named there only identify that listing and are not needs on this product. "
                        "Find that listing (search its shop and product words), open it, read its own " + prop + ", then choose a candidate whose " + prop
                        + (" is the one the store's own pairing rule (stated in listing text) puts together with it." if pair else " is the same." if same else " is different from it.")
                        + " Its own text may send this property on to yet another listing: then find and open that one too, and so on, until a page"
                        " states the value (the store checks that every page on the way was opened, and every listing you buy). A listing whose own"
                        " text says it is not that version, or that the wanted one is elsewhere, is not it: look for the one that matches."
                        + " Keep looking until it is found: search the shop's name alone, then with the product words (the store checks that you opened it).")
            if self.role.get("view") and self.role.get("view") != self.role.get("details"):
                # PV: with a separate page tool the pointer sits in the page notes, which the property list does not show
                meaning += (" Open that listing's page with " + str(self.role["view"]) + " as well as its property list: a note there that sends the "
                            + prop + " to another listing wins over the listing's own " + prop + " field.")
        if isinstance(ref, dict) and str(ref.get("property") or "").strip() and str(ref.get("listing") or "").strip():
            try:
                self.reply_pointer(fresh, str(ref["property"]), "pair" if pair else "is" if same else "not")
            except Exception as exc:   # a helper's failure must not end the episode
                self.trace.append({"kind": "fire", "set": "CS", "where": "RP", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
        if still:
            meaning += " A 'don't mind' answer does not cancel what the request requires: " + "; ".join(need_text(n) for n in still) + " still holds."
        return meaning

    @staticmethod
    def voice(observation: Any, from_message_tool: bool) -> list[str]:
        """Sentences from the shopper inside a result (a message reply, a notice list, a refusal)."""
        lines: list[str] = []
        if from_message_tool:
            if isinstance(observation, str):
                return [observation] if observation.strip() else []
            if isinstance(observation, dict) and "error" not in observation:
                for value in observation.values():
                    if isinstance(value, str) and len(value.split()) >= 3:
                        lines.append(value)
                    elif isinstance(value, list):
                        lines.extend(v for v in value if isinstance(v, str) and len(v.split()) >= 3)
            return lines
        if not isinstance(observation, dict):
            return lines
        refused = "refusal" in observation
        for value in observation.values():
            if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                lines.extend(v for v in value[:10] if len(v.split()) >= 4)
            elif isinstance(value, str) and refused and len(value.split()) >= 4:
                lines.append(value)
        return lines

    # ---------------- checks
    def facts_note(self, item: dict[str, Any]) -> str:
        reading = read_listing(self.ledger.needs, item.get("facts"), item)
        rows = []
        for need, (state, note) in reading["states"]:
            rows.append("- " + need_text(need) + ": " + state + " (" + note + ")")
        return "\n".join(rows) if rows else "(no structured needs)"

    def item_records(self, item_id: str) -> list[str]:
        item = self.shelf.items.get(item_id)
        if not item:
            return []
        record = dict(item.get("record") or {})
        record["property list"] = item.get("facts")
        return [_dump(_trim(record, 600), 3000)]

    def llm_check(self, item_id: str, purpose: str) -> str:
        if self.wire.left() < CLOSING_SECONDS + 20:
            return ""
        key = item_id + "|" + str(self.ledger.version)
        if key in self.verdicts:
            return self.verdicts[key]
        item = self.shelf.items.get(item_id)
        if not item or item.get("facts") is None:
            return ""
        prompt = CHECK_PROMPT + "\n\nTHE REQUEST:\n" + self.request
        prompt += "\n\nCURRENT NEEDS AND LIMITS (the shopper's own words decide when they differ from the summary):\n" + self.ledger.text(True)
        prompt += "\n\nLISTING TO " + purpose + ": " + item_id + "\n" + "\n".join(self.item_records(item_id))
        prompt += "\n\nAUTOMATIC READING OF ITS PROPERTY LIST:\n" + self.facts_note(item)
        if self.ledger.money:
            prompt += "\n(Prices in the listing records are in the same money unit as the budget: " + self.ledger.money + ".)"
        sections = self.section_census()
        if sections:
            prompt += "\n\nSECTION LABELS SEEN ON THIS STORE'S LISTINGS (with counts):\n" + sections
        result = self.wire.ask_json(prompt, [REASONER, DRIVER], 1100)
        self.trace.append({"kind": "check", "item": item_id, "result": result})
        verdict = ""
        if result and result.get("ok") is False:
            reasons = "; ".join(str(r) for r in (result.get("why") or [])[:4])
            verdict = "An independent check finds that " + item_id + " does not meet every need: " + reasons + "."
        self.verdicts[key] = verdict
        return verdict

    def rescue(self, item_id: str, clashes: list[tuple[dict[str, Any], str]]) -> bool:
        """Ask the reasoner whether conflicting values are the required values in other words (a shade of a colour)."""
        key = item_id + "|" + str(self.ledger.version)
        if key in self.rescued:
            return self.rescued[key]
        if self.wire.left() < CLOSING_SECONDS + 15:
            return False
        lines = []
        for number, (need, note) in enumerate(clashes, 1):
            words = (" (shopper's words: \"" + str(need.get("said")) + "\")") if need.get("said") else ""
            lines.append(str(number) + ". need: " + need_text(need) + words + " | the listing's own property: " + note)
        prompt = RESCUE_PROMPT + "\n\nTHE REQUEST:\n" + self.request + "\n\nLISTING " + item_id + "\n" + "\n".join(lines)
        result = self.wire.ask_json(prompt, [REASONER, DRIVER], 400)
        meant = result.get("meant") if isinstance(result, dict) else None
        ok = isinstance(meant, list) and len(meant) >= len(clashes)
        self.trace.append({"kind": "rescue", "item": item_id, "result": result, "ok": ok})
        self.rescued[key] = ok
        return ok

    def section_census(self) -> str:
        counts: dict[str, int] = {}
        for item in self.shelf.items.values():
            if item.get("section"):
                counts[item["section"]] = counts.get(item["section"], 0) + 1
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        return "\n".join("- " + k + " (x" + str(v) + ")" for k, v in ranked)

    def line_needs(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """LC: on a shopping list each line answers to its own product's needs: the needs the request states for the list product
        whose section the listing is in, plus the needs the shopper revealed for that product ("length of the pants"). A listing
        of no single list product is read against the shared needs, as before."""
        parts = getattr(self, "parts", [])
        mine = [p for p in parts if self.in_part(item, p)]
        if len(mine) != 1:
            return self.ledger.needs
        product = _words(mine[0].get("item") or "")
        told = []
        for need in self.ledger.needs:
            named = re.match(r"(.+?)\s+(?:of|for) (?:the |a |an )?(.+)$", str(need.get("property") or ""), re.I)
            if named and product and _covers(_words(named.group(2)), product):
                told.append(dict(need, property=named.group(1).strip()))
        return clean_needs(mine[0].get("needs")) + told

    def line_clash(self, item_id: str) -> str:
        """LC: a list line whose own facts contradict its product's needs (and the reasoner does not read them as meant): a guard
        naming the clash and that product's listings whose facts state every need."""
        item = self.shelf.items.get(item_id)
        if not item or not item.get("facts") or not getattr(self, "parts", []):
            return ""
        needs = self.line_needs(item)
        reading = read_listing(needs, item["facts"], item)
        if reading["verdict"] != "fails" or self.rescue(item_id, reading["clashes"]):
            return ""
        if self.mark("line-clash|" + item_id + "|" + str(self.ledger.version)) > 2 or self.steps_left() <= 3:
            return ""
        others = [o for o in self.better_fits(item, needs) if o.get("section") == item.get("section")]
        self.trace.append({"kind": "fire", "set": "LC", "item": item_id, "clashes": [need_text(n) for n, _ in reading["clashes"][:3]], "others": [o["id"] for o in others[:4]]})
        text = (item_id + " is the line for one product of the list, and its own facts contradict that product's needs: "
                + "; ".join(need_text(n) + " but " + note for n, note in reading["clashes"][:3]) + ".")
        if others:
            return text + " Listings of that product whose facts state every need: " + "; ".join(_label(o) for o in others[:4]) + ". Put one of them in its place."
        return text + " Find a listing of that product whose facts state every need (search its required value with the product's word), or decline if none exists."

    def better_fits(self, item: dict[str, Any] | None, needs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        ranked = self.shelf.ranked(self.ledger.needs if needs is None else needs, self.ledger.ceiling, self.ledger.count)
        out = [i for i, r in ranked if r["verdict"] == "fits" and (item is None or i["id"] != item["id"])]
        if item is not None and item.get("section"):
            same = [i for i in out if i.get("section") == item["section"]]
            out = same or out
        return out

    def judge_pick(self, item_id: str, purpose: str) -> str:
        """Every free check on a listing about to enter the cart or be bought. Returns a guard message or ''."""
        item = self.shelf.items.get(item_id)
        if item is None:
            return ""
        version = str(self.ledger.version)
        priced = version + "|" + str(item.get("price"))
        if item.get("gone") and self.mark("gone|" + item_id) <= 2:
            return item_id + " is no longer available (the shopper or the store said so). Take it out of the cart and choose another verified fit."
        if item.get("facts") is None:
            if self.role.get("details") and self.steps_left() > 4 and self.mark("unread|" + item_id) == 1:
                return "you have not opened " + item_id + "'s details, so its property list is unknown. Open it first (its facts decide every need), then add the one that fits."
            return ""
        needs = self.line_needs(item)
        reading = read_listing(needs, item["facts"], item)
        positive = [n for n in needs if n.get("test") != "not"]
        store_has_facts = any(i.get("facts") for i in self.shelf.items.values())
        titled = all(_title_says(n, item) for n in positive)
        if not item["facts"] and positive and store_has_facts and not titled and self.steps_left() > 4 and self.mark("empty|" + item_id) == 1:
            return item_id + " has an empty property list, and its title does not state the needs either (the description never counts). Choose a listing whose facts state the needs; sending the same call again goes ahead."
        if reading["verdict"] == "fails":
            if not self.rescue(item_id, reading["clashes"]):
                times = self.mark("clash|" + item_id + "|" + version)
                if times <= 2 and self.steps_left() > 3:
                    others = self.better_fits(item, needs)
                    text = item_id + "'s own facts contradict the needs: " + "; ".join(need_text(n) + " but " + note for n, note in reading["clashes"][:3]) + "."
                    if others:
                        text += " Listings whose facts state every need: " + "; ".join(_label(o) for o in others[:4]) + "."
                    else:
                        text += " Find a listing whose facts state every need, or decline if none exists."
                    return text
        wanted = [w for w in _words(self.brief.get("item") or "") if len(w) > 2]
        if wanted and item.get("section") and self.steps_left() > 4:
            def covered(label: str) -> set[str]:
                have = _words(label)
                return {w for w in wanted if any(_same(w, h) for h in have)}
            mine = covered(item["section"])
            wider = sorted({i["section"] for i in self.shelf.items.values() if i.get("section") and covered(i["section"]) > mine})
            if wider and self.mark("section|" + str(item["section"])) == 1:
                return (item_id + " is in the section '" + str(item["section"]) + "', but the request asks for '" + str(self.brief.get("item")) + "', and the section '"
                        + wider[0] + "' matches that wording more fully. Choose from '" + wider[0] + "' unless the request clearly means the narrower section; sending the same call again goes ahead.")
        if reading["verdict"] == "open" and self.brief.get("pick") not in ("cheapest", "best") and self.steps_left() > 4:
            proven = [o for o in self.better_fits(item) if not item.get("section") or o.get("section") == item.get("section")]
            if proven and self.mark("open|" + item_id + "|" + version) == 1:
                missing = "; ".join(need_text(n) + " (" + note + ")" for n, _, note in reading["loose"][:3])
                return item_id + "'s own facts do not clearly state every need: " + missing + ". These listings' facts state every need: " + "; ".join(_label(o) for o in proven[:4]) + ". Prefer one of them unless you are sure; sending the same call again goes ahead."
        if self.brief.get("pick") == "cheapest" and self.steps_left() > 14:
            cheaper = self.shelf.cheaper_rows(item, self.ledger.ceiling, self.ledger.count)
            verdicts = {c["id"]: read_listing(self.ledger.needs, c["facts"], c)["verdict"] for c in cheaper if c.get("facts") is not None}
            fitting = [c for c in cheaper if verdicts.get(c["id"]) == "fits"]
            unclear = [c for c in cheaper if verdicts.get(c["id"]) == "open" and any(
                state in ("mixed", "weak") for _, state, _ in read_listing(self.ledger.needs, c["facts"], c)["loose"])]
            unread = [c for c in cheaper if c.get("facts") is None][:6]
            if fitting and self.mark("cheaper-fit|" + item_id + "|" + priced) <= 2:
                first = fitting[0]
                proof = "; ".join(need_text(n) + ": " + note for n, (state, note) in read_listing(self.ledger.needs, first["facts"], first)["states"])
                return ("the cheapest fitting listing is wanted, and cheaper listings in the same section have facts that state every need: " + "; ".join(_label(c) for c in fitting[:4])
                        + ". For " + first["id"] + ": " + (proof or "every need is stated") + ". A key about a different property does not cancel these. "
                        + "Choose the cheapest of them unless it fails the budget or the stock; send the same call again only if you are sure.")
            room = self.steps_left() - 8
            if (unread or unclear) and len(unread) <= max(0, room) and self.mark("cheaper-unread|" + item_id + "|" + priced) == 1:
                text = "the cheapest fitting listing is wanted, and cheaper listings in the same section are still undecided."
                if unread:
                    text += " Seen but never opened (cheapest first): " + "; ".join(_label(c) for c in unread) + " (open them in one reply)."
                if unclear:
                    text += " Opened, facts not clear-cut: " + "; ".join(_label(c) for c in unclear[:4]) + " (judge them by the rules; adding one triggers an independent check)."
                return text + " Take the cheapest that fits; send the same call again only if none of them can fit."
        if self.brief.get("pick") == "cheapest" and self.steps_left() > 14 and self.search_mode == "all":
            part = next((p for p in getattr(self, "parts", []) if self.in_part(item, p)), None)
            product, needs = (str(part.get("item") or ""), clean_needs(part.get("needs"))) if part else self.search_product(set())
            todo = [q for q in self.value_query(product, needs, set(_words(item.get("title") or "")), str(item.get("section") or ""))
                    if frozenset(w for w in _words(q) if w not in SEARCH_FILLER) not in self.text_totals]
            if todo and self.mark("value-search|" + item_id + "|" + priced) == 1:
                self.trace.append({"kind": "fire", "set": "SM", "value_searches": todo[:3]})
                return ("the cheapest fitting listing is wanted, and this store's search needs every word, so a required value with the product's own word "
                        "lists exactly the listings that carry it. Not run yet: " + "; ".join('"' + q + '"' for q in todo[:3]) + ". Run them, read every page "
                        "whose total fits, and open the cheaper rows first; send the same call again only if none of them can fit.")
        if self.brief.get("pick") == "cheapest" and self.steps_left() > 14 and self.mark("probe|" + item_id + "|" + priced) == 1:
            found = self.price_probe(item)
            if found:
                return ("the cheapest fitting listing is wanted, and a search of its section capped just under " + item_id + "'s price shows cheaper listings: "
                        + "; ".join(_label(c) for c in found[:6]) + ". Open them (cheapest first, in one reply) and take the cheapest that fits; send the same call again only if none of them can fit.")
        if self.brief.get("pick") == "cheapest" and self.steps_left() > 14:
            more = self.unpaged(item)
            if more and self.mark("pages|" + item_id + "|" + priced + "|" + more) == 1 and self.mark("pages-all|" + item_id) <= 3:
                return "the cheapest fitting listing is wanted, and " + more + ". Open the next page(s) and check the cheaper rows first; send the same call again only if you are sure."
        verdict = self.llm_check(item_id, purpose)
        if verdict and self.mark("llm|" + item_id + "|" + version) == 1:
            others = self.better_fits(item)
            if others:
                verdict += " Listings whose facts state every need: " + "; ".join(_label(o) for o in others[:4]) + "."
            return verdict + " Unless you are certain the check is wrong, choose another verified fit (sending the same call again overrides)."
        return ""

    # ---------------- questions
    def property_table(self, part: dict[str, Any] | None = None) -> tuple[str, int, list[str]]:
        """With a list product: only that product's listings, keys under their full names (none folded into a shorter key)."""
        groups: dict[str, dict[str, Any]] = {}
        read = [i for i in self.shelf.items.values() if i.get("facts") and (part is None or self.in_part(i, part))]
        counted = len(read)
        if part is None and not getattr(self, "parts", []) and self.brief.get("item"):
            own = [i for i in read if self.in_part(i, {"item": self.brief.get("item")})]
            if own:
                read = own   # pages of other sections (where a reference sent us) say nothing about this product's properties
            elif getattr(self, "pointer_props", None) or getattr(self, "chain_hunt", None):
                # QO: the request points to other listings' pages and none of the product's own listings is open yet, so what is
                # open is those pages (an IP camera's questions were asked on canned meat properties: meat type, pack type, ...)
                read = []
                counted = 0
        live = {i["id"] for i in read if read_listing(self.ledger.needs, i["facts"], i)["verdict"] != "fails"}
        for item in read:
            seen_here: set[str] = set()
            for key, value in item["facts"].items():
                if not isinstance(value, (str, int, float)) or isinstance(value, bool) or not str(value).strip():
                    continue
                text = str(value).strip().lower()
                if len(text) > 70:
                    continue
                if part is not None:
                    key = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(key))
                name = " ".join(_words(re.sub(r"[_.]+", " ", str(key)))) or str(key).lower()
                group = groups.setdefault(name, {"label": re.sub(r"[_.]+", " ", str(key)).strip().lower(), "count": 0, "values": [], "live": set(), "all": set(), "combos": 0, "codes": 0})
                if name not in seen_here:
                    group["count"] += 1
                    seen_here.add(name)
                    group["all"].add(text)
                    if " / " in text or re.search(r"#|[a-z]\s*:\s*[a-z]", text):
                        group["combos"] += 1
                    if len(text) > 40 or re.search(r"[a-z]\d", text):
                        group["codes"] += 1
                if text not in group["values"] and len(group["values"]) < 6:
                    group["values"].append(text)
                if item["id"] in live:
                    group["live"].add(text)
        for name in (sorted(groups, key=lambda n: -len(n.split())) if part is None else []):
            words = name.split()
            if len(words) < 2 or name not in groups:
                continue
            short = next((other for other in groups if other != name and len(other.split()) == 1 and _same(other, words[-1])), None)
            if short:   # the same property under a longer key: count it once, under the shorter name
                into, group = groups[short], groups.pop(name)
                into["count"] = min(len(read), into["count"] + group["count"])
                into["all"] |= group["all"]
                into["live"] |= group["live"]
                into["combos"] += group["combos"]
                into["codes"] += group["codes"]
                into["values"] = (into["values"] + [v for v in group["values"] if v not in into["values"]])[:6]
        rows = []
        mine = " for the " + str((part or {}).get("item") or "").strip().lower() + "?"
        asked_here = [a[2:].split(": asked", 1)[0] for a in self.asked if part is not None and mine in a.lower()]
        for name, group in groups.items():
            if any(_key_grade(name, a) or _key_grade(a, name) for a in asked_here):
                continue   # already asked about this list product
            if (group["count"] >= 4 and len(group["all"]) >= group["count"] and group["codes"] * 2 > group["count"]
                    and not _shared_word(group["all"])) or group["combos"] * 2 > group["count"]:
                continue  # identifier-like: a model name or code on every listing, or option strings; specs with model numbers stay
            settled = any(_key_grade(name, n.get("property")) for n in self.ledger.needs) and not self.family_open(name)
            rows.append((len(group["live"]) > 1, group["count"], name, group, settled))
        rows.sort(key=lambda r: (not r[0], -r[1]) if part is None else r[2])   # a list product's table: name order, no ranking
        lines = []
        for split, count, name, group, settled in rows[:26]:
            tag = " [settled by a known need]" if settled else (" [a known need names only its family: the exact one is still open]" if self.family_open(name)
                                                              else " [values differ among candidates]" if split else " [same value among candidates]")
            lines.append("- " + group["label"] + " (in " + str(count) + " listings)" + tag + ": " + " | ".join(group["values"]))
        spare = [name for split, count, name, group, settled in rows if split and not settled and count >= 2]
        return "\n".join(lines), counted, spare

    def family_open(self, prop: Any) -> bool:
        """A known need naming only a family in one word ("core") leaves the exact variant open when opened listings show two or
        more narrower values for that property that hold the word and a model number ("core i5-10500", "core i7-12700")."""
        for need in self.ledger.needs:
            if need.get("test") != "is" or not (_key_grade(prop, need.get("property")) or _key_grade(need.get("property"), prop)):
                continue
            words = [w for v in need.get("values") or [] for w in _words(v)]
            if len(words) != 1 or not words[0].isalpha():
                continue
            narrower = set()
            for item in self.shelf.items.values():
                for key, value in (item.get("facts") or {}).items():
                    if isinstance(value, str) and (_key_grade(key, prop) or _key_grade(prop, key)):
                        low = value.strip().lower()
                        if words[0] in _words(low) and re.search(r"\d", low) and len(_words(low)) > 1:
                            narrower.add(low)
            if len(narrower) >= 2:
                return True
        return False

    @staticmethod
    def store_label(name: str, table: str) -> str:
        """The property as the store's table names it: a name that is a shortened form of table labels (all its words inside
        a longer label) becomes the highest-ranked such label."""
        labels = re.findall(r"^- (.+?) \(in \d+ listings\)", table or "", re.M)
        low = name.strip().lower()
        if not low or low in labels:
            return name
        have = set(_words(low))
        wider = [label for label in labels if have and have < set(_words(label))]
        return wider[0] if wider else name

    def more_picks(self, table: str, picks: list[dict[str, Any]], settled: list[Any], count: int) -> list[dict[str, Any]]:
        """Table rows for the questions the planner left unused (a held-back need is revealed only when its property is
        asked): rows in the table's order that no known need settles, not asked or picked yet, not another name of a picked
        property, with more than one value seen."""
        taken = [str(p.get("name") or "") for p in picks if p.get("kind") != "open" and str(p.get("name") or "").strip()]
        taken += [m.group(1) for m in (re.match(r"- (.+?): asked", line) for line in self.asked) if m]
        extra: list[dict[str, Any]] = []
        rows = []
        for name, listings, tag, values in re.findall(r"^- (.+?) \(in (\d+) listings\)( \[[^\]]*\])?: (.*)$", table or "", re.M):
            shown = [v for v in values.split(" | ") if v.strip()]
            if "settled" in tag or not shown:
                continue
            rows.append((len(shown) < 2, name))
        for _, name in sorted(rows, key=lambda row: row[0]):
            if any(_key_grade(name, other) or _key_grade(other, name) for other in list(settled) + taken):
                continue
            extra.append({"kind": "property", "name": name})
            taken.append(name)
            if len(extra) >= count:
                break
        return extra

    def part_labels(self, part: dict[str, Any]) -> set[str]:
        """The store section labels of one list product: the labels holding every longer word of its name; when no label does,
        the labels whose every longer word is in the name ("women's mini dress" -> "Women's Dresses")."""
        labels = {str(i.get("section")) for i in self.shelf.items.values() if i.get("section")}
        key = (str(part.get("item")), len(labels))
        cache = getattr(self, "label_cache", {})
        if key not in cache:
            name = [w for w in _words(str(part.get("item") or "")) if len(w) > 2]
            tier = {lab for lab in labels if name and all(any(_same(w, h) for h in _words(lab)) for w in name)}
            if not tier:
                tier = {lab for lab in labels if [w for w in _words(lab) if len(w) > 2]
                        and all(any(_same(w, h) for h in name) for w in _words(lab) if len(w) > 2)}
            cache[key] = tier
            self.label_cache = cache
        return cache[key]

    def in_part(self, listing: dict[str, Any], part: dict[str, Any]) -> bool:
        """A listing of this list product's section (see part_labels)."""
        return str(listing.get("section")) in self.part_labels(part)

    def open_part(self, part: dict[str, Any], count: int = 1) -> bool:
        """Open one seen listing of a list product that has nothing to ask about yet: within budget and not shown sold out, the
        row whose title names most of the product's stated values first, then the cheapest (at most twice per product)."""
        if self.mark("lag-open|" + str(part.get("item") or "").lower()) > 2:
            return False
        values = {w for n in clean_needs(part.get("needs")) if n.get("test") != "not" for v in n.get("values") or [] for w in _words(v) if len(w) > 2}
        rows = [i for i in self.shelf.items.values() if i.get("facts") is None and self.in_part(i, part)
                and self.shelf.affordable(i, self.ledger.ceiling, _as_int(part.get("count")) or 1)]
        shown = self.value_hits(clean_needs(part.get("needs")), part.get("item"))   # SQ
        rows.sort(key=lambda i: (-len(shown.get(str(i["id"]), ())), -len(values & set(_words(i.get("title") or ""))),
                                 i["price"] if _is_number(i.get("price")) else 1e18, str(i["id"])))
        many = self.multi_reader() if count > 1 and len(rows) > 1 and not self.multi_bad else None
        opened = []
        if many:
            tool, key, cap = many
            args, issues = conform({key: [str(r["id"]) for r in rows[: min(count, cap)]]}, self.schemas[tool])
            if not issues:
                done = self.execute([(tool, args)])
                if done and self.bare_rows(done[0][2], {str(r["id"]) for r in rows[: min(count, cap)]}):
                    self.multi_bad = True
                    self.trace.append({"kind": "fire", "set": "MV", "tool": tool, "where": "LA"})
                opened = [str(r["id"]) for r in rows[: min(count, cap)] if (self.shelf.items.get(str(r["id"])) or {}).get("facts") is not None]
        if not opened:
            opened = self.open_rows(rows, 1)
        if opened:
            self.trace.append({"kind": "fire", "set": "LA", "part": part.get("item"), "opened": opened})
            self.chat.append({"role": "user", "content": "[before asking about the " + str(part.get("item")) + " I opened " + _label(self.shelf.items[opened[0]])
                              + " to see its properties]\n" + self.status()})
        return bool(opened)

    def value_hits(self, needs: list[dict[str, Any]], product: Any) -> dict[str, set[int]]:
        """SQ: a store whose search needs every word returns only listings that carry each search word (in the title, the shop or
        a property value), so a row a search returned states each stated value the search's words cover, even when its title does
        not ("2 pin IP Security Cameras" found a camera whose title never says 2 pin). Per row id: the stated values (their
        indexes) its own title or a search that returned it covers. Values made of the product's own words prove nothing; empty
        unless the store is known to need every word."""
        if self.search_mode != "all":
            return {}
        own = _words(product or "")
        wanted = [w for w in (_words(v) for n in needs if n.get("test") == "is" for v in n.get("values") or []) if w and not _covers(own, w)]
        out: dict[str, set[int]] = {}
        if not wanted:
            return out
        tool = self.role.get("search") or ""
        for key, entry in self.searches.items():
            try:
                query = json.loads(key)
            except ValueError:
                continue
            words = list(self.search_words(tool, query)) if isinstance(query, dict) else []
            said = {k for k, w in enumerate(wanted) if words and _covers(words, w)}
            for row_id in entry["ids"] if said else []:
                out.setdefault(str(row_id), set()).update(said)
        for row_id, item in self.shelf.items.items():
            title = _words(item.get("title") or "")
            said = {k for k, w in enumerate(wanted) if _covers(title, w)}
            if said:
                out.setdefault(str(row_id), set()).update(said)
        return out

    def pre_read(self) -> bool:
        """PR: a single product's questions are planned from the property lists of at least three of its own listings. With one
        list every property looks equally common and the planner picks rare ones (size, type) over the ones most listings carry.
        Open up to three of its seen, unopened rows in one reply (stated values in the title first, then the cheapest), once."""
        item = self.brief.get("item")
        if not item or self.steps_left() <= 16:
            return False
        part = {"item": item}
        own = sum(1 for i in self.shelf.items.values() if i.get("facts") and self.in_part(i, part))
        if own >= 3:
            return False
        values = {w for n in self.ledger.needs if n.get("test") != "not" for v in n.get("values") or [] for w in _words(v) if len(w) > 2}
        rows = [i for i in self.shelf.items.values() if i.get("facts") is None and self.in_part(i, part) and not i.get("gone")
                and self.shelf.affordable(i, self.ledger.ceiling, self.ledger.count)]
        if not rows or self.mark("pre-read") > 1:
            return False
        shown = self.value_hits(self.ledger.needs, item)   # SQ
        rows.sort(key=lambda i: (-len(shown.get(str(i["id"]), ())), -len(values & set(_words(i.get("title") or ""))),
                                 i["price"] if _is_number(i.get("price")) else 1e18, str(i["id"])))
        opened = self.open_rows(rows, 3 - own)
        if opened:
            self.trace.append({"kind": "fire", "set": "PR", "own": own, "opened": opened})
            self.chat.append({"role": "user", "content": "[before asking the shopper anything I opened the property lists of "
                              + "; ".join(_label(self.shelf.items[i]) for i in opened if i in self.shelf.items) + "]\n" + self.status()})
        return bool(opened)

    def part_props(self, part: dict[str, Any], settled: list[Any]) -> list[str]:
        """The property keys of one list product's opened listings (its section), most common first, without the settled ones
        and the ones already asked."""
        counts: dict[str, int] = {}
        for item in self.shelf.items.values():
            if not item.get("facts") or not self.in_part(item, part):
                continue
            for key, value in item["facts"].items():
                if str(value).strip():
                    counts[key] = counts.get(key, 0) + 1
        asked = [a[2:].split(": asked", 1)[0] for a in self.asked]
        skip = [s for s in list(settled) + [n.get("property") for n in clean_needs(part.get("needs"))] + asked if s]
        keys = [k for k in counts if not any(_key_grade(k, s) or _key_grade(s, k) for s in skip)]
        return sorted(keys, key=lambda k: (-counts[k], k))

    def cover_parts(self, picks: list[dict[str, Any]], settled: list[Any], table: str) -> list[dict[str, Any]]:
        """Give every list product at least one question: an uncovered product takes its most common unsettled property in place
        of the last extra pick of the most-asked product."""
        def owner(pick: dict[str, Any]) -> str:
            if pick.get("item"):
                return str(pick["item"]).strip().lower()
            if pick.get("kind") == "open":
                return ""
            return self.item_for(self.store_label(re.sub(r"\s+", " ", str(pick.get("name") or "").strip()), table))[9:]
        picks = list(picks)
        for part in self.parts:
            mine = re.sub(r"\s+", " ", str(part.get("item") or "").strip()).lower()
            if any(owner(p) == mine for p in picks):
                continue
            props = self.part_props(part, settled)
            if not props:
                continue
            counts: dict[str, int] = {}
            for p in picks:
                counts[owner(p)] = counts.get(owner(p), 0) + 1
            spare = [i for i in range(len(picks) - 1, -1, -1) if owner(picks[i]) and counts.get(owner(picks[i]), 0) > 1]
            if not spare:
                continue
            self.trace.append({"kind": "fire", "set": "BY", "part": mine, "property": props[0], "replaced": picks[spare[0]].get("name")})
            picks[spare[0]] = {"kind": "property", "name": props[0], "item": part.get("item")}
        return picks

    def item_for(self, name: str) -> str:
        """On a shopping list: " for the <item>" for the product whose opened listings (its section) carry this property most."""
        best, most = None, 0
        for part in getattr(self, "parts", []):
            count = 0
            for item in self.shelf.items.values():
                if item.get("facts") and self.in_part(item, part) \
                        and any(_key_grade(key, name) or _key_grade(name, key) for key in item["facts"]):
                    count += 1
            if count > most:
                best, most = part, count
        return " for the " + re.sub(r"\s+", " ", str(best.get("item")).strip()).lower() if best else ""

    def message_args(self, text: str) -> dict[str, Any] | None:
        """The message tool's arguments for one question. The text goes into its free-text slot: the one required string, else
        the one named like a message among several, else the only string. Every other required argument takes its schema
        default, an enum value that reads as neutral (else the first), a number's minimum (else 1), false, or a short subject
        line; None when a slot cannot be filled."""
        tool = self.role.get("message")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = [k for k in schema.get("required") or [] if isinstance(k, str)]
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]

        def named(keys: list[str]) -> str | None:
            hits = [k for k in keys if re.search(r"text|message|msg|body|content|chat|question|say|note|utter", k, re.I)]
            return hits[0] if len(hits) == 1 else None
        needed = [k for k in required if k in texts]
        target = (needed[0] if len(needed) == 1 else named(needed)) if needed else (texts[0] if len(texts) == 1 else named(texts))
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in required:
            spec = props.get(key)
            if key == target:
                continue
            if not isinstance(spec, dict):
                return None
            kinds = _kinds(spec)
            if "default" in spec:
                args[key] = spec["default"]
            elif isinstance(spec.get("enum"), list) and spec["enum"]:
                args[key] = next((o for o in spec["enum"] if isinstance(o, str)
                                  and re.search(r"neutral|polite|friendly|normal|default|question|ask|general|info", o, re.I)), spec["enum"][0])
            elif "integer" in kinds or "number" in kinds:
                args[key] = spec["minimum"] if _is_number(spec.get("minimum")) else 1
            elif "boolean" in kinds:
                args[key] = False
            elif "string" in kinds:
                args[key] = "Question"
            else:
                return None
        return args

    def plan_questions(self, force: bool = False) -> bool:
        tool = self.role.get("message")
        room = self.question_room()
        if not tool or room <= 0 or not self.asking:
            self.asking = False
            return False
        if self.steps_left() <= 10 or self.wire.left() < CLOSING_SECONDS + 40:
            self.asking = False
            return False
        table, read, spare = self.property_table()
        learn = [str(x) for x in self.brief.get("learn_first") or []]
        urgent = bool(learn) and not self.asked
        if read < 3 and (self.wire.step < 8 or read < 2) and not urgent and not force:
            return False
        if read < self.ask_floor and not urgent:
            return False
        if len(getattr(self, "parts", [])) >= 2 and not force and not urgent and self.wire.step < 12 \
                and all(not self.part_props(part, []) for part in self.parts):
            return False   # a list waits for one opened listing per product before its questions are planned
        if not urgent and not getattr(self, "parts", []) and self.pre_read():
            return True   # PR: the questions wait for three of the product's own property lists
        only_one = urgent and read < 3 and not force
        focus = None
        sequential = len(getattr(self, "parts", [])) >= 2 and not urgent
        if sequential:
            room = min(room, 1)   # a list: one question per turn, planned with the last answer known
            told = getattr(self, "part_told", set())
            asked_n = getattr(self, "part_asked", {})
            ready = [(i, p) for i, p in enumerate(self.parts) if self.part_props(p, [])]
            def due(ip: tuple[int, dict[str, Any]]) -> tuple[bool, int]:
                label = str(ip[1].get("item") or "").lower()
                return label in told, asked_n.get(label, 0)   # a product whose held-back need is told goes last
            first = min(due(ip) for ip in enumerate(self.parts))
            if not force and self.steps_left() > 14 and not ready:
                return False   # no product of the list has an opened listing yet: the driver opens some before they are asked about
            lag = [p for i, p in enumerate(self.parts) if due((i, p)) == first and not self.part_props(p, [])]
            if lag and ready and not force and self.steps_left() > 14 and min(due(ip) for ip in ready) > first:
                if self.open_part(lag[0]):
                    return True   # the product due next had nothing to ask about: one of its listings is open now
                if self.steps_left() > 20:
                    return False
            if ready:
                i, focus = min(ready, key=lambda ip: (due(ip), ip[0]))
                # LR: only listings that may still be bought count: reference-chain pages of the product's own section (a
                # universal-plug camera for "plug type: 2 pin") are read for the chain, and its candidates stayed unopened
                wanted = clean_needs(focus.get("needs"))
                opened_here = sum(1 for it in self.shelf.items.values() if it.get("facts") and self.in_part(it, focus)
                                  and read_listing(wanted, it["facts"], it)["verdict"] != "fails")
                if opened_here < 2 and not due((i, focus))[1] and not force and self.steps_left() > 16 and self.open_part(focus, 3 - opened_here):
                    return True   # the product's first question waits for 2-3 of its listings to be opened
                table, _, spare = self.property_table(focus)
        prompt = ASK_PROMPT.replace("QUESTION_COUNT", str(1 if only_one else room))
        prompt += "\n\nTHE REQUEST:\n" + self.request + "\n\nKNOWN NEEDS AND LIMITS:\n" + self.ledger.text()
        if self.asked:
            prompt += "\n\nALREADY ASKED (with the answers):\n" + "\n".join(self.asked)
        if learn:
            prompt += "\n\nMUST LEARN: " + "; ".join(learn)
        if focus is not None:
            prompt += ("\n\nTHE PRODUCT TO ASK ABOUT NOW: the " + str(focus.get("item")) + " (one product of the shopping list). Pick a property of this "
                       "product only; the table below holds the rows of its own listings, in name order. Hidden needs sit on everyday "
                       "properties (colour or finish, material, who it is for, size, style) as often as on technical ones: prefer what most of "
                       "its listings carry (the count).")
        prompt += "\n\nTABLE OF PROPERTIES IN THE LISTINGS OPENED SO FAR (name, how many listings carry it, whether its values differ among candidates that still meet the known needs, example values):\n" + (table or "(nothing opened yet)")
        result = self.wire.ask_json(prompt, [HELPER, PLANNER], 1800)
        picks = [p for p in (result or {}).get("picks") or [] if isinstance(p, dict) and str(p.get("name") or p.get("text") or "").strip()]
        # a property a known need already fixes is never asked: its answer can only be a filler, which reads as "don't mind"
        settled = [n.get("property") for n in self.ledger.needs if n.get("property") and not self.family_open(n.get("property"))]
        # PS: nor is a property the request reads from another listing, or one the list's product in focus already states
        settled += [p for name, ptr in getattr(self, "pointer_props", {}).items() for p in (name, ptr.get("alias")) if p]
        if focus is not None:
            settled += [n.get("property") for n in clean_needs(focus.get("needs")) if n.get("property") and not self.family_open(n.get("property"))]
        chosen = [p for p in picks if p.get("kind") == "open" or not any(_key_grade(p.get("name"), s) or _key_grade(s, p.get("name")) for s in settled)]
        if not chosen and (picks or result is None) and not only_one:
            chosen = [{"kind": "property", "name": name} for name in spare]
        picks = chosen
        if not only_one and len(picks) < room:
            picks = picks + self.more_picks(table, picks, settled, room - len(picks))
        picks = picks[: (1 if only_one else room)]
        if focus is not None:
            for pick in picks:
                if pick.get("kind") != "open":
                    pick["item"] = focus.get("item")
        if len(getattr(self, "parts", [])) >= 2 and not only_one and room >= len(self.parts):
            picks = self.cover_parts(picks, settled, table)
        calls, labels = [], []
        for pick in picks:
            name = re.sub(r"\s+", " ", str(pick.get("name") or "").strip())
            if pick.get("kind") == "open" and str(pick.get("text") or "").strip():
                text = re.sub(r"\s+", " ", str(pick.get("text")).strip())[:240]
            elif name:
                name = self.store_label(name, table)
                owner = (" for the " + re.sub(r"\s+", " ", str(pick["item"]).strip()).lower()) if pick.get("item") else self.item_for(name)
                text = "What " + name.lower() + " do you need" + owner + "?"
            else:
                continue
            args = self.message_args(text)
            if not args:
                continue
            args, issues = conform(args, self.schemas[tool])
            if issues:
                continue
            self.call_no += 1
            calls.append((tool, args))
            labels.append((name or "question", text))
        if not calls:
            if picks or read >= 3 or self.asked:
                self.asking = False
            else:
                self.ask_floor = read + 2
            return False
        self.planner_sending = True
        try:
            replies = self.execute(calls)
        finally:
            self.planner_sending = False
        lines = []
        pairs = []
        for (name, text), (call_id, result_text, observation) in zip(labels, replies):
            answer = " ".join(self.voice(observation, True)) or result_text[:300]
            pairs.append((text, answer))
            self.asked.append("- " + name + ": asked \"" + text + "\" -> " + answer[:300])
            lines.append("Q: " + text + "\nA: " + answer)
        meaning = ""
        heard = [a for _, a in pairs if a]
        version = self.ledger.version
        needs_before = _dump(self.ledger.needs)
        if heard:
            meaning = self.hear(heard, " / ".join(q for q, _ in pairs), [(q, a) for q, a in pairs if a])
        if focus is not None:
            label = str(focus.get("item") or "").lower()
            self.part_asked = dict(getattr(self, "part_asked", {}), **{label: getattr(self, "part_asked", {}).get(label, 0) + 1})
            if self.ledger.version != version or _dump(self.ledger.needs) != needs_before:
                self.part_told = set(getattr(self, "part_told", set())) | {label}
        summary = "[questions sent by the planner and the shopper's answers]\n" + "\n".join(lines)
        if meaning:
            summary += "\n" + meaning
        self.chat.append({"role": "user", "content": summary + "\n" + self.status()})
        self.asking = self.question_room() > 0
        return True

    # ---------------- cart bookkeeping
    def item_in_args(self, name: str, args: dict[str, Any]) -> str | None:
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for key, value in args.items():
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if isinstance(value, str) and "enum" not in spec and value in self.shelf.items:
                return value
        for key, value in args.items():
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if isinstance(value, str) and "enum" not in spec and _id_like(value):
                return value
        return None

    def cart_direction(self, name: str, args: dict[str, Any]) -> str:
        fields = (self.schemas.get(name) or {}).get("properties") or {}
        if any(isinstance(v, int) and not isinstance(v, bool) and v == 0 and isinstance(fields.get(k), dict) and "enum" not in fields[k]
               for k, v in args.items()):
            return "remove"  # a count of 0 takes the line out, whatever the brief says
        add = self.brief.get("cart_add") if isinstance(self.brief.get("cart_add"), dict) else {}
        remove = self.brief.get("cart_remove") if isinstance(self.brief.get("cart_remove"), dict) else {}
        if remove and all(args.get(k) == v for k, v in remove.items()):
            return "remove"
        if add:
            return "add" if all(args.get(k) == v for k, v in add.items()) else "remove"
        props = (self.schemas.get(name) or {}).get("properties") or {}
        signature = _dump({k: v for k, v in args.items() if isinstance(props.get(k), dict) and "enum" in props[k]})
        if self.add_mode is None:
            self.add_mode = signature
        return "add" if signature == self.add_mode else "remove"

    def quantity_issue(self, name: str, args: dict[str, Any]) -> str:
        if getattr(self, "parts", []):
            return ""
        wanted = self.ledger.count
        props = (self.schemas.get(name) or {}).get("properties") or {}
        for key, spec in props.items():
            if isinstance(spec, dict) and "integer" in _kinds(spec) and "enum" not in spec:
                value = args.get(key, spec.get("default", 1))
                if isinstance(value, int) and value != wanted:
                    return "the request needs " + str(wanted) + " unit(s) of one listing, but this call sets " + str(value) + "."
        return ""

    # ---------------- guards before any step
    def screen(self, calls: list[tuple[str, str, Any]]) -> tuple[list[tuple[str, str, dict[str, Any]]], list[tuple[str, str]]]:
        offered = {t["function"]["name"] for t in self.offered_tools()}
        finals = {self.role.get("order"), self.role.get("decline")} - {None}
        cart_tool = self.role.get("cart")
        ready: list[tuple[str, str, dict[str, Any]]] = []
        refused: list[tuple[str, str]] = []
        for call_id, name, raw in calls:
            if name not in self.schemas:
                refused.append((call_id, GUARD + "'" + str(name) + "' is not a tool here. Use one of: " + ", ".join(sorted(offered)) + "."))
                continue
            if name not in offered:
                why = "questions to the shopper are planned and sent for you" if name == self.role.get("message") else "too few steps are left for it; usable now: " + ", ".join(sorted(offered))
                refused.append((call_id, GUARD + why + "."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_object(raw or "{}") if str(raw or "").strip() else {}
                if args is None:
                    refused.append((call_id, GUARD + "the arguments were not one JSON object."))
                    continue
            if isinstance(args, dict):
                spec = self.schemas[name] if isinstance(self.schemas[name], dict) else {}
                needed = set(spec.get("required") or [])
                props = spec.get("properties") or {}
                args = {k: v for k, v in args.items() if k in needed or not (v is None or (isinstance(v, str) and not v.strip()
                        and "" not in ((props.get(k) if isinstance(props.get(k), dict) else {}).get("enum") or [])))}
            args, issues = conform(args, self.schemas[name])
            if issues:
                refused.append((call_id, GUARD + "invalid arguments: " + "; ".join(issues[:6])))
                continue
            slots = self.text_slots(name)
            if name == self.role.get("search") and slots and not any(isinstance(args.get(s), str) and args[s].strip() for s in slots):
                refused.append((call_id, GUARD + "this search carries no words and no section name, so it can only come back empty. Put the search words or a section name in "
                                + " or ".join(slots) + "."))
                continue
            if name == self.role.get("search"):
                args, moved = self.foreign_token(name, args)
                if moved:
                    self.paging_notes[call_id] = moved
            scale_note = self.price_scale_note(name, args) if name == self.role.get("search") else ""
            if scale_note and self.mark("scale|" + _dump(args)) == 1:
                refused.append((call_id, GUARD + scale_note))
                continue
            if name in finals and len(calls) > 1:
                refused.append((call_id, GUARD + "a checkout or a decline must be sent alone, after you have seen your other results."))
                continue
            signature = name + _dump(args)
            if name not in finals and name != cart_tool and signature in self.recent:
                # DS: the code itself may have sent the call (a guard opening listings), so its result may never have been shown
                shown = self.recent_text.get(signature)
                refused.append((call_id, GUARD + ("you just made this exact call and nothing has changed (no step used); its result was: " + shown[:6000]
                                                  if shown else "you just made this exact call and nothing has changed; its result is above.")))
                continue
            if signature in self.read_memory and signature not in self.read_answered and not any(v in self.cart for v in args.values() if isinstance(v, str)):
                # RM: a page opened before and unchanged since (no cart write, message, finish or notice in between); asked again
                # after this answer, it goes to the store
                self.read_answered.add(signature)
                at, shown = self.read_memory[signature]
                self.trace.append({"kind": "fire", "set": "RM", "call": signature[:120], "first": at})
                refused.append((call_id, GUARD + "this page was opened at step " + str(at) + " and nothing has changed since, so its result still holds: "
                                + shown[:6000]))
                continue
            if name == cart_tool:
                note = self.screen_cart(name, args, signature)
                if note:
                    refused.append((call_id, GUARD + note))
                    continue
            ready.append((call_id, name, args))
        if ready and ready[0][1] in finals:
            call_id, name, args = ready[0]
            refusal = getattr(self, "last_refusal", None)
            if self.final_kind(name, args) == "decline" and refusal and refusal.get("job") == "decline" and self.steps_left() > 2 \
                    and getattr(self, "work_step", -1) <= refusal["step"] and self.mark("xd|" + str(refusal["step"])) == 1:
                self.trace.append({"kind": "fire", "set": "XD", "refused_at": refusal["step"]})
                return [], refused + [(call_id, "Not run yet (no step used). The store refused the last decline, and nothing has been searched, opened or put in "
                                       "the cart since. A decline sent again right after a refused one is almost never accepted: apply every change the "
                                       "refusal states, then search again, open the candidates or check the cart for a listing that fits now; decline "
                                       "again only after that.")]
            note = self.screen_final(name, args) or self.rule_gate(name, args)
            if note:
                return [], refused + [(call_id, note)]
            if self.final_kind(name, args) == "decline":
                ready[0] = (call_id, name, self.decline_args(name, args))
        if ready and finals and ready[0][1] not in finals:
            keep = self.steps_left() - 1
            if keep < len(ready):
                for call_id, _, _ in ready[max(0, keep):]:
                    refused.append((call_id, GUARD + "only the last step is left, and it must be the checkout or the decline."))
                ready = ready[: max(0, keep)]
        return ready, refused

    def reachable_rows(self, entry: dict[str, Any]) -> int:
        """Rows a search can show at most: when its paging argument is an offset (the first next-page value equals the rows a
        page holds) with a schema maximum, the rows up to that offset plus one page; otherwise every row it reports."""
        props = (self.schemas.get(self.role.get("search") or "") or {}).get("properties") or {}
        caps = [spec["maximum"] for spec in props.values() if isinstance(spec, dict) and "integer" in _kinds(spec) and _is_number(spec.get("maximum"))]
        if caps and entry["size"] in self.page_values:
            return int(max(caps)) + entry["size"]
        return entry["total"]

    def text_slots(self, name: str) -> list[str]:
        """The tool's search-text arguments: every string argument except a required one with a fixed list of values (the
        search mode); an optional list of values (section names) is search text."""
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        required = set(schema.get("required") or [])
        return [key for key, spec in props.items() if isinstance(spec, dict) and "string" in _kinds(spec) and not ("enum" in spec and key in required)]

    def joined_units(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """QJ: a one-digit figure written apart from a one-letter unit ("4 g", "5 v") is two words of under two characters, which a
        keyword search may skip, so the value drops out of the search (every smartphone comes back); joined ("4g") it stays a word
        of the search. Figures of more digits keep their spacing: their number is a word of its own."""
        if name != self.role.get("search") or not isinstance(args, dict):
            return args
        props = (self.schemas.get(name) or {}).get("properties") or {}
        out = dict(args)
        for key in self.text_slots(name):
            value = out.get(key)
            if not isinstance(value, str) or (isinstance(props.get(key), dict) and props[key].get("enum")):
                continue
            joined = re.sub(r"(?<![\w.,-])(\d)\s+([b-z])(?![\w.-])(?!\s*\d)", r"\1\2", value, flags=re.I)
            if joined != value:
                out[key] = joined
                self.trace.append({"kind": "fire", "set": "QJ", "from": value[:120], "to": joined[:120]})
        return out

    def note_search(self, args: dict[str, Any], observation: Any) -> None:
        """Keep, per query (paging arguments left out), the rows in arrival order and the largest count the store reports."""
        query = {k: v for k, v in sorted(args.items()) if not (isinstance(v, (str, int)) and not isinstance(v, bool) and v in self.page_values)}
        entry = self.searches.setdefault(_dump(query), {"label": _dump(query, 200), "ids": [], "prices": [], "sections": set(), "total": 0, "last": 0, "size": 0})
        holders = [observation] if isinstance(observation, dict) else []
        holders += [v for v in (observation.values() if isinstance(observation, dict) else []) if isinstance(v, dict)]
        for holder in holders:
            for value in holder.values():
                if isinstance(value, bool) or value is None:
                    continue
                if isinstance(value, int):
                    entry["total"] = max(entry["total"], value)
                    self.page_values.add(value)
                elif isinstance(value, str) and 0 < len(value) <= 64 and not re.search(r"\s", value):
                    self.page_values.add(value)
                    if value not in [str(v) for v in args.values()]:
                        self.token_sources.setdefault(value, set()).add(_dump(query).lower())
        # CT: a store paging by token names the next page's token in the reply; keep the newest one per query, and learn which
        # argument takes it from a call that passed a token a reply named
        onward = [v for holder in holders for k, v in holder.items() if isinstance(v, str) and re.search(r"next|onward|cursor|continu|token", str(k), re.I)
                  and 0 < len(v) <= 64 and not re.search(r"\s", v) and v not in [str(a) for a in args.values()]]
        entry["onward"] = onward[0] if onward else None
        for k, v in args.items():
            if isinstance(v, str) and v in self.token_sources:
                self.token_arg = k
        lists = self.shelf._row_lists(observation, 0)
        rows = lists[0] if lists else []
        entry["last"] = len(rows)
        entry["size"] = max(entry["size"], len(rows))
        try:
            if getattr(self, "page_mode", None) is None and len(rows) >= 2:
                props = (self.schemas.get(self.role.get("search") or "") or {}).get("properties") or {}
                pkeys = [k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v) and _is_number(v.get("maximum")) and v["maximum"] <= 50]
                nexts = [v for holder in holders for k, v in holder.items() if re.search(r"next", str(k), re.I) and isinstance(v, int) and not isinstance(v, bool)]
                if len(pkeys) == 1 and nexts:
                    spec = props[pkeys[0]]
                    asked = args.get(pkeys[0]) if isinstance(args.get(pkeys[0]), int) else (int(spec["minimum"]) if _is_number(spec.get("minimum")) else 1)
                    if nexts[0] == asked + 1:
                        self.page_mode = "number"
                    elif nexts[0] == asked + len(rows):
                        self.page_mode = "offset"
        except Exception as exc:   # a helper's failure must not end the episode
            self.trace.append({"kind": "fire", "set": "CS", "where": "CP", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
        for row in rows:
            row_id = self.shelf._row_id(row)
            item = self.shelf.items.get(row_id or "")
            if not item or row_id in entry["ids"]:
                continue
            entry["ids"].append(row_id)
            if _is_number(item.get("price")):
                entry["prices"].append(item["price"])
            if item.get("section"):
                entry["sections"].add(item["section"])

    def search_words(self, name: str, args: dict[str, Any]) -> frozenset[str]:
        """SM: the content words of a text search (paging values, section names and filler left out)."""
        labels = {str(i.get("section")).strip().lower() for i in self.shelf.items.values() if i.get("section")}
        out: set[str] = set()
        for key in self.text_slots(name):
            value = args.get(key)
            if not isinstance(value, str) or not value.strip() or value in self.page_values or value.strip().lower() in labels \
                    or re.search(r"shelf|section|categor|department|aisle", key, re.I):
                continue
            out |= {w for w in _words(value) if w not in SEARCH_FILLER}
        return frozenset(out)

    def search_product(self, words: frozenset[str] | set[str]) -> tuple[str, list[dict[str, Any]]]:
        """SM: the product a search is for (the list product whose name shares most words with it, else the request's item) and its needs."""
        products = [(str(p.get("item") or ""), clean_needs(p.get("needs"))) for p in getattr(self, "parts", [])] \
            or [(str(self.brief.get("item") or ""), list(self.ledger.needs))]
        return max(products, key=lambda p: len(set(_words(p[0])) & set(words)))

    def value_query(self, product: str, needs: list[dict[str, Any]], words: frozenset[str] | set[str], section: str = "") -> list[str]:
        """SM: short searches a store that needs every word answers with the listings carrying one required value: the value,
        then the product's own word (a word of its name that the search or its section's titles use)."""
        label = [w for w in re.findall(r"[a-z]+", product.lower()) if len(w) > 2 and w not in SEARCH_FILLER]
        owners = set(re.findall(r"([a-z]+)'s\b", product.lower()))
        label = [w for w in label if w not in owners] or label
        titles = [set(_words(i.get("title") or "")) for i in self.shelf.items.values() if section and i.get("section") == section]
        kind = max(label, key=lambda w: (_stem(w) in words, sum(1 for t in titles if _stem(w) in t)), default="")
        out = []
        for need in needs:
            if need.get("test") not in ("is", "min", "max", "other"):
                continue
            value = next((str(v).strip() for v in need.get("values") or [] if str(v).strip()), "")
            if not value or len(value.split()) > 3 or not kind:
                continue
            text = value + " " + kind if _stem(kind) not in _words(value) else value
            if text.lower() not in [o.lower() for o in out]:
                out.append(text)
        return out

    def learn_search_mode(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """SM: learn from first-page totals whether this store's text search needs every word (a search with a word more
        reports fewer rows) or takes any word (it reports more), and say so when a long search comes back nearly empty."""
        if name != self.role.get("search") or any(isinstance(v, (str, int)) and not isinstance(v, bool) and v in self.page_values for v in args.values()):
            return ""
        words = self.search_words(name, args)
        holders = [observation] if isinstance(observation, dict) else []
        holders += [v for v in (observation.values() if isinstance(observation, dict) else []) if isinstance(v, dict)]
        totals = [v for h in holders for k, v in h.items() if isinstance(v, int) and not isinstance(v, bool) and re.search(r"total|found|hits|count|num|match", str(k), re.I)]
        if not words or not totals or words in self.text_totals:
            return ""
        total = max(totals)
        before = self.search_mode
        for other, seen in self.text_totals.items():
            if seen == total or not (other < words or words < other):
                continue
            small, big = (other, words) if other < words else (words, other)
            small_total, big_total = (seen, total) if other < words else (total, seen)
            vote = "all" if big_total < small_total else "any"
            self.mode_votes[vote] += 1
            if vote == "all":
                self.mode_proof = "'" + " ".join(sorted(small)) + "' found " + str(small_total) + ", '" + " ".join(sorted(big)) + "' found " + str(big_total)
        self.text_totals[words] = total
        if len(words) >= 2 and (total < 50 or total >= 300):   # SV: a lone search of two or more words counts half a vote
            self.lone_votes["all" if total < 50 else "any"] += 1
            if total < 50 and not self.mode_proof:
                self.mode_proof = "'" + " ".join(sorted(words)) + "' found only " + str(total)
        votes = {k: self.mode_votes[k] + 0.5 * self.lone_votes[k] for k in self.mode_votes}
        self.search_mode = "all" if votes["all"] >= 1 and votes["all"] > 2 * votes["any"] else "any" if votes["any"] >= 1 and votes["any"] > 2 * votes["all"] else None
        if self.search_mode != before:
            self.trace.append({"kind": "fire", "set": "SM", "mode": self.search_mode, "votes": dict(self.mode_votes), "lone": dict(self.lone_votes)})
        if self.steps_left() <= 6:
            return ""
        product, needs = self.search_product(words)
        example = next(iter(self.value_query(product, needs, words)), "")
        how = ("Search a required value with the product's own word, the value first" + (' (for example "' + example + '")' if example else "")
               + ", and read every page when the total fits; leave property names (size, type, material) and filler words out of the search text.")
        if len(words) >= 3 and total < 8 and self.search_mode != "any" and (self.search_mode == "all" or total <= 2) and self.mark("sm-narrow") <= 3:
            self.trace.append({"kind": "fire", "set": "SM", "narrow": total, "words": len(words)})
            why = ("this store's search needs every word (" + self.mode_proof + ")") if self.search_mode == "all" else "the store may need every word"
            return "This search reports only " + str(total) + " row(s) for " + str(len(words)) + " words: " + why + ", so each extra word drops listings, fits among them. " + how
        if self.search_mode == "all" and before != "all" and self.mark("sm-learned") == 1:
            return "This store's search needs every word (" + self.mode_proof + "): each added word narrows the rows. " + how
        return ""

    def joined_note(self, name: str, args: dict[str, Any]) -> str:
        """JN: a stated value written as a number and a short unit apart ("128 gb", "4 g") is usually one word in listings
        ("128GB", "4G"), and a search with the two apart misses those listings in a store whose search needs every word.
        Once per value, name the same search with the joined form."""
        text = " ".join(str(args[k]) for k in self.text_slots(name) if isinstance(args.get(k), str) and args[k] not in self.page_values)
        query = " " + " ".join(text.lower().split()) + " "
        if not query.strip():
            return ""
        needs = list(self.ledger.needs) + [n for p in getattr(self, "parts", []) for n in clean_needs(p.get("needs"))]
        said = []
        for need in needs:
            unit = str(need.get("unit") or "").strip().lower()
            for value in need.get("values") or []:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    value = "%g" % value
                if unit and re.fullmatch(r"\d+(?:\.\d+)?", str(value).strip()):
                    value = str(value).strip() + " " + unit
                m = re.fullmatch(r"(\d+(?:\.\d+)?)\s+([a-z]{1,4})", str(value).strip().lower())
                if not m:
                    continue
                spaced, joined = m.group(1) + " " + m.group(2), m.group(1) + m.group(2)
                if " " + spaced + " " in query and self.mark("joined|" + joined) == 1:
                    query = query.replace(" " + spaced + " ", " " + joined + " ")
                    said.append((spaced, joined))
        if not said:
            return ""
        self.trace.append({"kind": "fire", "set": "JN", "values": said})
        return ("The request writes " + ", ".join('"' + a + '"' for a, _ in said) + ", but listings usually write a number and its unit as one word ("
                + ", ".join('"' + b + '"' for _, b in said) + "), and a search that needs every word misses those listings. Also search \""
                + query.strip() + "\" and read its pages.")

    def search_drift(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """A search that names a known section but whose rows are mostly from other sections: the store let the text alone
        decide the rows. Names the section word its listings' titles use most, to put first in the search text."""
        sections = {str(i["section"]).strip().lower(): i["section"] for i in self.shelf.items.values() if i.get("section")}
        named = [v.strip().lower() for v in args.values() if isinstance(v, str) and v.strip().lower() in sections]
        if not named:
            return ""
        want = named[0]
        lists = self.shelf._row_lists(observation, 0)
        rows = [self.shelf.items.get(self.shelf._row_id(row) or "") for row in (lists[0] if lists else [])]
        rows = [i for i in rows if i and i.get("section")]
        off = sum(1 for i in rows if str(i["section"]).strip().lower() != want)
        if len(rows) < 3 or off * 2 <= len(rows):
            return ""
        query = {k: v for k, v in sorted(args.items()) if not (isinstance(v, (str, int)) and not isinstance(v, bool) and v in self.page_values)}
        if self.mark("drift|" + _dump(query)) > 1:
            return ""
        inside = [set(_words(i.get("title") or "")) for i in self.shelf.items.values() if str(i.get("section") or "").strip().lower() == want]
        outside = [set(_words(i.get("title") or "")) for i in self.shelf.items.values() if i.get("section") and str(i["section"]).strip().lower() != want]
        brands = {w for i in self.shelf.items.values() for key, value in (i.get("record") or {}).items()
                  if re.search(r"brand|maker|shop|seller|store", str(key), re.I) and isinstance(value, str) for w in _words(value)}

        def typical(stem: str) -> float:
            return sum(1 for t in inside if stem in t) / max(1, len(inside)) - sum(1 for t in outside if stem in t) / max(1, len(outside))
        label = [w for w in re.findall(r"[a-z]+", want) if len(w) > 2 and w != "and"]
        owners = set(re.findall(r"([a-z]+)'s\b", want))  # "women's", "men's": who it is for, not what it is
        word = max([w for w in label if w not in owners] or label, key=lambda w: typical(_stem(w)), default="")
        text = [w for k in self.text_slots(name) if isinstance(args.get(k), str) and args[k].strip().lower() != want
                and args[k] not in self.page_values for w in args[k].split()]  # a paging cursor is no search text
        text = [w for w in text if not any(_same(_stem(w.lower()), _stem(s)) for s in label) and w.lower() not in ("of", "the", "for", "and", "a")]
        said = {_stem(w.lower()) for w in text} | {_stem(w) for w in label}
        pool = {w for t in inside for w in t if len(w) > 3 and not w[0].isdigit() and w not in brands and w not in said}
        extra = max(sorted(pool), key=typical, default="")
        if extra and (sum(1 for t in inside if extra in t) * 3 < len(inside) or typical(extra) < 0.25):
            extra = ""
        example = " ".join(([word] if word else []) + text[:2])
        if extra:
            example += '" or "' + " ".join([extra] + text[:2])
        self.trace.append({"kind": "fire", "set": "BE", "section": sections[want], "off": off, "rows": len(rows)})
        return ("Only " + str(len(rows) - off) + " of these " + str(len(rows)) + " rows are from " + str(sections[want])
                + ": the store did not limit this text search to that section, so the search text alone decides the rows. Put the product's"
                + " own word first in the search text, then one or two need words" + (' (for example "' + example + '")' if example else "")
                + ", or list the section itself.")

    def unpaged(self, item: dict[str, Any]) -> str:
        """A price-ordered search of the pick's section that still has rows left, the last one seen cheaper than the pick."""
        for entry in self.searches.values():
            prices = entry["prices"]
            if len(prices) < 4 or any(b < a for a, b in zip(prices, prices[1:])):
                continue
            if item.get("section") not in entry["sections"] or not entry["last"]:
                continue
            seen = len(entry["ids"])
            if entry["total"] <= seen or not _is_number(item.get("price")) or prices[-1] >= item["price"]:
                continue
            return ("the search " + entry["label"] + " lists rows cheapest first and has more left (" + str(seen) + " of " + str(entry["total"])
                    + " seen; the last one seen costs " + str(prices[-1]) + ", below this pick's " + str(item["price"]) + ")")
        # rows in no price order: a cheaper fit can be on any later page of a search that returned the pick; the search
        # with the fewest pages left is named, when it can be covered with steps to spare
        choice = None
        for entry in self.searches.values():
            prices = entry["prices"]
            if not entry["size"] or not any(b < a for a, b in zip(prices, prices[1:])):
                continue
            if item.get("id") not in entry["ids"] and item.get("section") not in entry["sections"]:
                continue
            seen = len(entry["ids"])
            pages = -(-(min(entry["total"], self.reachable_rows(entry)) - seen) // entry["size"])
            if pages <= 0 or self.steps_left() < min(pages, 2) + 10:
                continue
            rank = (item.get("id") not in entry["ids"], pages)
            if choice is None or rank < choice[0]:
                choice = (rank, pages, seen, entry)
        if choice:
            _, pages, seen, entry = choice
            return ("the search " + entry["label"] + " returns rows in no price order and has more left (" + str(seen) + " of " + str(entry["total"])
                    + " seen, about " + str(pages) + " more page(s)); a cheaper fit can be on the next one, so open at least the next page or two")
        return ""

    def best_fit(self) -> dict[str, Any] | None:
        fits = self.better_fits(None)
        if not fits:
            return None
        if self.brief.get("pick") == "cheapest":
            return fits[0]
        for item_id in self.cart:
            if any(f["id"] == item_id for f in fits):
                return self.shelf.items.get(item_id)
        return None

    def screen_cart(self, name: str, args: dict[str, Any], signature: str) -> str:
        if getattr(self, "parts", []):
            return ""
        direction = self.cart_direction(name, args)
        item_id = self.item_in_args(name, args)
        swaps = sum(self.removed.values())
        best = self.best_fit() if swaps >= 1 else None
        if best is not None and item_id and self.mark("swap|" + direction + "|" + item_id) == 1:
            cheapest = " at the lowest price" if self.brief.get("pick") == "cheapest" else ""
            if direction != "add" and item_id == best["id"]:
                return ("you have already taken lines out of the cart " + str(swaps) + " times, and every swap costs steps. " + _label(best)
                        + " is the listing whose own facts state every need" + cheapest + ": keep it and check out (sending the same call again goes ahead).")
            if direction == "add" and item_id != best["id"] and best["id"] in self.cart:
                return ("you have already taken lines out of the cart " + str(swaps) + " times. " + _label(best) + " is in the cart and its facts state every need"
                        + cheapest + ": keep it and check out instead of swapping again (sending the same call again goes ahead).")
        if direction != "add":
            return ""
        opened = sum(1 for i in self.shelf.items.values() if i.get("facts") is not None)
        if item_id and self.asking and self.role.get("message") and self.question_room() > 0 and opened < 3 and self.steps_left() > 12 \
                and self.mark("ask-first") == 1:
            return ("you can still ask the shopper " + str(self.question_room()) + " question(s), and only " + str(opened) + " listing(s) are opened, too few to tell "
                    "which property the shopper may care about. Open a few more listings of the section first (their details), then choose; sending the same call again goes ahead.")
        held = self.bag.get(item_id or "") if self.bag_fresh else None
        if item_id and item_id in self.cart and (held is None or held >= self.ledger.count) and self.mark("again|" + item_id) == 1:
            shown = (" The cart view shows " + str(held) + " unit(s) of it.") if held is not None else " Look at the cart to see how many units it holds."
            return (item_id + " is already in the cart; depending on the store, adding it again either sets its units or adds units on top of those already there." + shown + " The request needs "
                    + str(self.ledger.count) + "; send the same call again only if more units are really needed.")
        if item_id and self.steps_left() > 3:
            note = self.judge_pick(item_id, "ADD TO THE CART")
            if note:
                return note
        issue = self.quantity_issue(name, args)
        if issue and self.mark("qty|" + signature) == 1:
            return issue + " If you are sure, send the same call again."
        return ""

    def screen_final(self, name: str, args: dict[str, Any]) -> str:
        if self.final_kind(name, args) == "order":
            seen = self.read_every_view()
            if seen:
                return seen
            if self.steps_left() <= 2 or self.wire.left() < CLOSING_SECONDS:
                return self.confirm_cart()
            if self.cart_touched and not self.cart and self.mark("empty-cart|" + str(self.wire.step)) == 1:
                return GUARD + "as far as I can tell the cart is empty. Add the chosen listing first (or view the cart if you think otherwise)."
            fresh = self.reread_cart()
            if fresh:
                return fresh
            if getattr(self, "parts", []):
                if self.brief.get("pick") == "cheapest" and self.steps_left() > 12:
                    for line_id in self.cart:
                        line_item = self.shelf.items.get(line_id)
                        if line_item and self.steps_left() > 12 and self.mark("probe|" + line_id + "|" + str(line_item.get("price"))) == 1:
                            found = [c for c in self.price_probe(line_item) if c.get("facts") is None]
                            if found:
                                return ("Not run yet (no step used). The cheapest basket is wanted, and a search of the section of " + _label(line_item)
                                        + " capped just under its price shows cheaper listings: " + "; ".join(_label(c) for c in found[:6])
                                        + ". Open them (cheapest first, in one reply) and put the cheapest that fits that product's needs in its place; send the checkout again only if none fits.")
                        more = self.unpaged(line_item) if line_item else ""
                        if more and self.mark("list-pages|" + line_id + "|" + more) == 1 and self.mark("list-pages-all") <= 3:
                            return ("Not run yet (no step used). The cheapest basket is wanted, and for the line " + _label(line_item) + " " + more
                                    + ". Open the next page(s) and check that product's cheaper rows first; send the checkout again only if you are sure.")
                for line_id in self.cart:
                    clash = self.line_clash(line_id)
                    if clash:
                        return "Not run yet (no step used). " + clash
                fault = self.fit_fault() if self.steps_left() > 6 else ""
                if fault and self.mark("ft-order|" + ",".join(sorted(self.cart))) == 1:
                    self.trace.append({"kind": "fire", "set": "FT", "order": list(self.cart)})
                    return "Not run yet (no step used). " + fault
                if not self.order_bounced:
                    self.order_bounced = True
                    return ("Not run yet (no step used). Before checkout: the cart holds exactly one line per product on the shopping list ("
                            + str(len(self.parts)) + "), each at its own quantity; each line's own facts meet that product's needs; the total is within "
                            "the budget for the whole list. If all of that holds, call the checkout tool again.")
                return self.confirm_cart()
            if len(self.cart) > 1 and self.mark("lines|" + ",".join(self.cart)) <= 2:
                return GUARD + "the cart seems to hold several lines (" + ", ".join(self.cart) + "). Keep only the chosen listing, then check out."
            units = self.bag.get(self.cart[-1]) if self.cart and self.bag_fresh else None
            if units is not None and units != self.ledger.count and self.mark("units|" + self.cart[-1] + "|" + str(units)) <= 2:
                return (GUARD + "the cart view shows " + str(units) + " unit(s) of " + self.cart[-1] + ", but the request needs " + str(self.ledger.count)
                        + ". Fix the quantity in the cart first (adding puts units on top of those already there; take the extra ones out), then check out.")
            if self.cart and not self.bag_fresh and self.adds.get(self.cart[-1], 0) > 1 and self.role.get("cart_view") and self.mark("units-unknown|" + self.cart[-1]) == 1:
                return (GUARD + self.cart[-1] + " was added " + str(self.adds[self.cart[-1]]) + " times since it was last taken out, and adding can put units on top of those already there. "
                        + "Look at the cart and make sure it holds " + str(self.ledger.count) + " unit(s) of it, then check out.")
            if self.cart:
                note = self.judge_pick(self.cart[-1], "BUY")
                if note:
                    return "Not run yet (no step used). " + note
            if not self.order_bounced:
                self.order_bounced = True
                unread = self.cart and self.cart[-1] not in self.read_after_cart and self.steps_left() > 4
                checklist = "Not run yet (no step used). Before checkout: (1) the cart holds exactly one line, the chosen listing at the required quantity; (2) its facts meet every need as they stand now, including revealed and changed ones; (3) if the cheapest or best was asked for, nothing you found beats it; (4) every working rule of the request is followed"
                tool = self.stock_tool()
                checklist += ("; (5) after your last cart change, re-read its stock and price with a tool whose result shows them"
                              + (" (" + tool + ")" if tool else "") + "." if unread else ".")
                return checklist + " If all of that holds, call the checkout tool again."
            return self.confirm_cart()
        refusal = getattr(self, "last_refusal", None)
        if refusal and self.role.get("order") and self.steps_left() >= 2 and self.wire.left() > LAST_CALL_SECONDS \
                and self.mark("refused-final|" + str(refusal["step"])) == 1:
            if refusal["job"] == "order" and self.cart and {_need_sig(n) for n in self.ledger.needs} <= refusal["sigs"]:
                lines = [(i, self.line_units(i)) for i in self.cart]
                if not self.budget_cut({"observation": refusal["obs"]}, lines) and not any((self.shelf.items.get(i) or {}).get("gone") for i in self.cart):
                    self.trace.append({"kind": "fire", "set": "FR", "kept": list(self.cart)})
                    return ("Not run yet (no step used). The checkout was refused only to tell you of a change, and the change adds no need: a dropped "
                            "or relaxed need, or a raised budget, never makes a fitting cart fail, so a decline now scores 0. If the cheapest or best was "
                            "asked for, compare the cart with the listings you have seen under the new terms and swap only to one that now beats it; "
                            "otherwise check out the same cart again.")
            elif refusal["job"] == "decline" and self.steps_left() >= 3:
                backups = self.backups()
                self.trace.append({"kind": "fire", "set": "FR", "redecline": True})
                return ("Not run yet (no step used). Your decline was refused because something changed, and after such a change a listing usually "
                        "fits: a dropped need widens the choice and a new budget changes what is affordable. Re-check the listings you have seen "
                        "against the new terms and buy the best fit" + (" (verified fits: " + "; ".join(_label(i) for i in backups[:3]) + ")" if backups else "")
                        + ". Decline again only if none fits.")
        nb = getattr(self, "nb_pick", None)
        held = self.shelf.items.get(nb) if nb else None
        if nb and nb in self.cart and held and not held.get("gone") and self.role.get("order") and self.steps_left() >= 1 \
                and not getattr(self, "parts", []) and self.mark("nb-order") == 1:
            self.trace.append({"kind": "fire", "set": "NO", "order": nb})
            self.finish_now()
            if self.wire.finished:
                return GUARD + ("the order was placed instead: " + _label(held) + " contradicts no need, and a decline scores 0 whenever a "
                                "listing could meet the needs.")
        veto = self.commit_veto()
        if veto:
            return veto
        if self.refused and self.steps_left() >= 3 and self.wire.left() > LAST_CALL_SECONDS + 10 and self.mark("repair-not-cancel") == 1:
            backups = self.backups()
            if backups:
                return ("Not run yet (no step used). A refused checkout is repaired, not cancelled: " + _label(backups[0])
                        + " is a verified fit that is not shown out of stock. Take the line that no longer qualifies out of the cart, put this one in, and check out; send the decline again only if it truly fails.")
        if self.steps_left() <= 4 or self.wire.left() < CLOSING_SECONDS + 20:
            return ""
        overlooked = self.better_fits(None) if not getattr(self, "parts", []) else []
        if not overlooked and not getattr(self, "parts", []) and self.steps_left() > 6 and self.mark("decline-open") == 1:
            opened = self.open_seen(min(6, self.steps_left() - 4))
            if opened:
                overlooked = self.better_fits(None)
                self.trace.append({"kind": "fire", "set": "AG", "opened": opened, "fits": [i["id"] for i in overlooked[:4]]})
        if overlooked and self.mark("decline-fits") == 1:
            return "Not run yet (no step used). These listings' own facts state every need, they are within budget and not shown out of stock: " + "; ".join(_label(i) + self.proof(i, self.ledger.needs) for i in overlooked[:4]) + ". Unless their section is not the requested kind, buy the right one (the cheapest if that was asked) instead of declining."
        if not self.swept and not getattr(self, "parts", []):
            self.swept = True
            note = self.sweep()
            if note:
                return "Not run yet (no step used). " + note
        if getattr(self, "parts", []) and self.steps_left() > 6 and self.mark("list-decline") == 1:
            note = self.list_fits()
            if note:
                return "Not run yet (no step used). " + note
        if not getattr(self, "parts", []) and self.steps_left() > 4 and self.mark("decline-unclear") == 1:
            maybe = self.unclear_fits()
            if maybe:
                self.trace.append({"kind": "fire", "set": "BB", "maybe": [i["id"] for i in maybe[:4]]})
                return ("Not run yet (no step used). Nothing opened is proven to fit, but these opened listings are in the requested section, within budget and in stock, "
                        "and none of their own facts contradicts a need (they leave some need unstated): " + "; ".join(_label(i) for i in maybe[:4])
                        + ". A decline is almost never the right ending while such listings exist: buy the one whose facts come closest to every need"
                        + (" (the cheapest of the closest, as the cheapest was asked)" if self.brief.get("pick") == "cheapest" else "")
                        + ", unless its record clearly fails a need; send the decline again only if each of them clearly fails.")
        if not self.decline_bounced:
            self.decline_bounced = True
            return "Not run yet (no step used). Before declining be sure you searched in several ways (other words, section or price filters, further pages), opened the candidates' details, and that each fails a need, the budget or stock by its own facts. If one fits, buy it; otherwise send the decline again."
        if not getattr(self, "parts", []) and not self.forced_buy and self.steps_left() >= 2 and self.wire.left() > LAST_CALL_SECONDS + 10:
            self.forced_buy = True
            held = [self.shelf.items[i] for i in self.cart if i in self.shelf.items and not self.shelf.items[i].get("gone")]
            if held:
                reading = read_listing(self.ledger.needs, held[-1].get("facts"), held[-1]) if held[-1].get("facts") is not None else {"verdict": "open"}
                if reading["verdict"] != "fails":
                    self.trace.append({"kind": "fire", "set": "BS", "cart": held[-1]["id"]})
                    return ("Not run yet (no step used). A decline scores 0 whenever a listing could meet the needs, and the cart holds " + _label(held[-1])
                            + ", whose own facts contradict no need. Check it out; send the decline again only if its facts clearly fail a need.")
            elif not self.cart:
                positive = [n for n in self.ledger.needs if n.get("test") != "not"]
                keyed = [i for i in self.better_fits(None) if all(read_need(n, i.get("facts"))[0] == "met" for n in positive)]
                pick = next(iter(keyed), None)   # proven by its own property keys only: a decline over unproven needs can be right
                for item in (self.unclear_fits() if pick is None else []):
                    reading = read_listing(self.ledger.needs, item.get("facts"), item)
                    met = sum(1 for _, (state, _) in reading["states"] if state == "met")
                    if met >= 1 and met >= len(reading.get("loose") or []):
                        pick = item
                        self.trace.append({"kind": "fire", "set": "NB", "pick": item["id"], "met": met, "loose": len(reading.get("loose") or [])})
                        self.nb_pick = item["id"]
                        break
                add = self.add_action(pick["id"]) if pick else None
                if add:
                    self.trace.append({"kind": "fire", "set": "BS", "add": pick["id"]})
                    self.execute([(add["name"], add["args"])])
                    return ("Not run yet (no step used). A decline scores 0 whenever a listing could meet the needs, so instead I put " + _label(pick)
                            + " in the cart: its facts come closest to every need and contradict none. Check it out now; send the decline again only if its own facts clearly fail a need.")
        return ""

    def open_rows(self, rows: list[dict[str, Any]], count: int, role: str = "details") -> list[str]:
        """Open up to count of these rows with the details tool (or the tool of another role), in one reply. Returns the ids opened."""
        tool = self.role.get(role)
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        if not tool or not target or any(k != target for k in required) or count <= 0:
            return []
        if role == "view" and tool != self.role.get("details"):
            shown = self.view_many(rows[:count])
            if shown is not None:
                return shown
        calls = self.row_calls(rows[:count], role)
        if calls:
            self.execute(calls)
        return [str(args[target]) for _, args in calls]

    def row_calls(self, rows: list[dict[str, Any]], role: str = "details") -> list[tuple[str, dict[str, Any]]]:
        """The calls that open these rows with the tool of one role (its one id argument), not yet sent."""
        tool = self.role.get(role)
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        if not tool or not target or any(k != target for k in required):
            return []
        calls = []
        for item in rows:
            args, issues = conform({target: item["id"]}, schema)
            if not issues:
                calls.append((tool, args))
        return calls

    def page_keys(self) -> set[str]:
        """MV2: the fields that every page shown alone by the view tool carries and no search row does (a seller's notes)."""
        alone = [set(i.get("record") or {}) for i in self.shelf.items.values() if i.get("viewed") and not i.get("viewed_many")]
        plain = [set(i.get("record") or {}) for i in self.shelf.items.values() if not i.get("viewed") and i.get("facts") is None]
        if not alone or not plain:
            return set()
        return set.intersection(*alone) - set().union(*plain)

    def view_many(self, rows: list[dict[str, Any]]) -> list[str] | None:
        """MV2: show several listings' own pages with one call of the multi-listing reader (every call is a step, and a reference
        hop opens up to six pages), once its rows have shown what a single view shows (page_keys). Each page shown is
        remembered as a single view of it (RM). None: not usable here, or its first rows lacked those fields (then never again)."""
        many = self.multi_reader() if len(rows) >= 2 and self.multi_view_ok is not False else None
        keys = self.page_keys() if many else set()
        view = self.role.get("view")
        schema = self.schemas.get(view or "") or {}
        required = list(schema.get("required") or [])
        if not many or not keys or not view or len(required) != 1:
            return None
        tool, key, cap = many
        ids = list(dict.fromkeys(str(r["id"]) for r in rows))
        done: list[str] = []
        for start in range(0, len(ids), cap):
            chunk = ids[start:start + cap]
            args, issues = conform({key: chunk}, self.schemas[tool])
            if issues or self.steps_left() <= 8:
                break
            _, _, observation = self.execute([(tool, args)])[0]
            got = {}
            for found in self.shelf._row_lists(observation, 0):
                for row in found:
                    row_id = self.shelf._row_id(row)
                    if row_id in chunk and keys <= set(row):
                        got[row_id] = row
            if not got:
                self.multi_view_ok = False
                self.trace.append({"kind": "fire", "set": "MV2", "tool": tool, "bare": chunk})
                break
            self.multi_view_ok = True
            for row_id, row in got.items():
                item = self.shelf._merge(row_id, row, self.wire.step)
                item["viewed"] = item["viewed_many"] = True
                self.read_memory[view + _dump({required[0]: row_id})] = (self.wire.step, _dump(_trim(row, 600), 3000))
                done.append(row_id)
        if done:
            self.trace.append({"kind": "fire", "set": "MV2", "tool": tool, "viewed": done})
        return done or None

    def resolve_reference(self) -> bool:
        """Find the listing a need points at (shop name + product words), open the shop's rows, read the property from the
        one whose facts best match the named facts, and write that value into the need. When that listing has no such key
        but its own text sends the property on to another listing, follow that pointer (at most three hops). A "right for"
        reference wants the value the store's own pairing rule puts with the found one."""
        queue = getattr(self, "ref_queue", [])
        if not queue or self.steps_left() < 10 or self.wire.left() < CLOSING_SECONDS + 30:
            return False
        ref = queue.pop(0)
        text = str(ref.get("listing") or "")
        hop = int(ref.get("hop") or 0)
        prop = re.split(r"\s+of\s+the\s+", str(ref["prop"]), maxsplit=1)[0]
        shop = None
        for pattern in (r"\bsold by ([A-Z][\w-]+)", r"\b(?:from|by) ([A-Z][\w-]+)", r"\b([A-Z][\w-]+)'s\b", r"\bthat ([A-Z][\w-]+) (?:has|have|had) ",
                        r"\bthe ([A-Z][\w-]+)", r"(?<=[a-z,] )([A-Z][a-z][\w-]+)"):
            found = re.search(pattern, text)
            if found:
                shop = found.group(1)
                break
        tool = self.role.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        if not shop or not tool or len(slots) != 1:
            return False
        origin = ref.get("origin") or shop
        raw_head = re.split(r",|\(|\s+with\s|\s+that\s|\s+which\s|\s+in\s", text, maxsplit=1)[0]
        rest = text[len(raw_head):]
        head = re.sub(r"\b(?:the listing for|sold by|from|by|the|a|an|product|listing|listed|page|for|what'?s|written|on|very|same|as|anything|bar|but|except|has)\b|"
                      + re.escape(shop) + r"(?:'s)?", " ", raw_head, flags=re.I)
        args, issues = conform({slots[0]: (shop + " " + " ".join(head.split())).strip()}, schema)
        if issues:
            return False
        self.execute([(tool, args)])

        def sold_by(item: dict[str, Any]) -> bool:
            return any(isinstance(v, str) and v.strip().lower() == shop.lower() for v in (item.get("record") or {}).values())
        rows = [i for i in self.shelf.items.values() if sold_by(i)]
        if not rows and self.steps_left() > 10:
            # the keyword search wants every word in a row: the shop name alone still finds its rows
            alone, issues = conform({slots[0]: shop}, schema)
            if not issues:
                self.execute([(tool, alone)])
                rows = [i for i in self.shelf.items.values() if sold_by(i)]
        named = _words(rest)
        wanted = set(_words(re.sub(re.escape(shop) + r"(?:'s)?", " ", text))) - set(_words(prop)) - _FILLER

        def row_hint(item: dict[str, Any]) -> int:
            return len(wanted & set(_words(" ".join(str(v) for v in (item.get("record") or {}).values() if isinstance(v, (str, int, float))))))
        unread = sorted([i for i in rows if i.get("facts") is None], key=lambda i: (-row_hint(i), str(i["id"])))

        def match(item: dict[str, Any]) -> tuple[int, int]:
            facts = item.get("facts") or {}
            pairs = sum(1 for k, v in facts.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)
                        and any(_covers(named, _words(k) + _words(p)) for p in _pieces(v)))
            said, denied = _listing_words(item)
            return pairs, len(wanted & said) - len(wanted & denied)
        # clauses naming the listing itself or how it may be named ("that <shop> has listed", "said outright or by naming some
        # other listing", "direct or through a listing it names") are not facts of it
        clauses = len([c for c in re.split(r",|\band\b|\(|\)", rest) if _words(c) and not re.search(r"\b(?:listing|listed|names|naming|outright|direct|through)\b", c, re.I)])

        def settled() -> bool:
            # an opened row of the shop shows every named fact and says something about the property: that is the named listing
            return any(i.get("facts") and (match(i)[0] >= clauses if clauses else match(i)[1] >= 1)
                       and (any(_key_grade(k, prop) for k in i["facts"]) or _pointer(i, prop) or _text_value(i, prop)) for i in rows)
        room = min(5, self.steps_left() - 9)
        for item in unread:
            if room <= 0 or settled():
                break
            self.open_rows([item], 1)
            room -= 1
        opened = sorted([i for i in rows if i.get("facts")], key=lambda i: tuple(-x for x in match(i)) + (str(i["id"]),))
        lead = next((i for i in opened if match(i)[0] == match(opened[0])[0] and match(i)[1] >= 0 and _pointer(i, prop)), None) if opened else None
        if lead and lead is not opened[0]:
            # the world writes a pointer for this property only on the listing a reference names: on a tie, that row is the one
            opened.remove(lead)
            opened.insert(0, lead)
        view = self.role.get("view")
        if opened and view and view != self.role.get("details") and not opened[0].get("viewed") and self.steps_left() > 9:
            # PTR: the listing's page carries its notes ("For <prop>, go by the X one, not by this item"), which its property
            # list does not show; a page of a reference chain is also one to have seen
            self.open_rows([opened[0]], 1, "view")
        value = None
        pointer = _pointer(opened[0], prop) if opened and hop < 3 else None
        if opened and not pointer:
            hits = sorted([(-_key_grade(k, prop), k, v) for k, v in opened[0]["facts"].items() if _key_grade(k, prop) and str(v).strip()], key=lambda h: h[0])
            value = str(hits[0][2]).strip() if hits else _text_value(opened[0], prop)
        onward = None
        if not value and hop < 3:
            # PTR: a pointer on the listing itself wins over its own field for this property ("..., not by this item")
            onward = (opened[0], pointer) if pointer else next(((i, p) for i in opened for p in [_pointer(i, prop)] if p), None)
        paired = None
        if value and ref.get("pair"):
            paired = _pair_rule([i for i in self.shelf.items.values() if i.get("facts") is not None], prop, value)
        self.trace.append({"kind": "fire", "set": "DP", "shop": shop, "hop": hop, "rows": [i["id"] for i in rows][:6], "pick": opened[0]["id"] if opened else None,
                           "value": value, "onward": onward[1][:120] if onward else None, "paired": paired})
        if onward:
            self.ref_queue = [{"prop": ref["prop"], "listing": onward[1], "same": ref.get("same", True), "pair": ref.get("pair"), "hop": hop + 1, "origin": origin}] + queue
            self.chat.append({"role": "user", "content": "The listing the shopper pointed at (" + _label(onward[0]) + ") sends the " + prop + " on to another listing: "
                              + onward[1] + ". I am looking that one up next.\n" + self.status()})
            return True
        if not value or (ref.get("pair") and not paired):
            self.chat.append({"role": "user", "content": "I looked for the listing the shopper pointed at (" + text + ") but could not settle its " + prop
                              + (" (found '" + value + "', but no store rule says what goes together with it)" if value else "")
                              + ". That property then rules no candidate out.\n" + self.status()})
            return True
        wanted_value = paired or value
        changed = False
        for need in self.ledger.needs:
            if (_key_grade(need.get("property"), ref["prop"]) or _key_grade(ref["prop"], need.get("property"))) and (
                    not need.get("values") or any(re.search(r"\blisting\b|\bsold by\b|" + re.escape(origin) + "|" + re.escape(shop), str(v), re.I) for v in need.get("values") or [])):
                need["values"] = [wanted_value]
                need["test"] = "is" if ref.get("same", True) else "not"
                need["said"] = INFERRED_SAID
                changed = True
        if not changed:
            self.ledger.needs.append({"property": ref["prop"], "test": "is" if ref.get("same", True) else "not", "values": [wanted_value], "said": INFERRED_SAID})
        self.inferred = dict(getattr(self, "inferred", {}), **{str(ref["prop"]): (wanted_value, "is" if ref.get("same", True) else "not")})
        self.ledger.version += 1
        product = (re.split(r"\s+of\s+the\s+", str(ref["prop"]), maxsplit=1) + [""])[1] or str(self.brief.get("item") or "")
        seen = ""
        if ref.get("same", True) and self.steps_left() > 8:
            # the wanted value is new to the search so far: look for it at once, so the next choice sees listings that state it
            again, issues = conform({slots[0]: " ".join((wanted_value + " " + product).split()[:6])}, schema)
            if not issues:
                seen = "\nI searched for it: " + str(self.execute([(tool, again)])[0][1])[:1500]
        self.chat.append({"role": "user", "content": "I found the listing the shopper pointed at: " + _label(opened[0]) + " sold by " + shop + ". Its own "
                          + prop + " is '" + value + "'" + (", and the store's own rule puts '" + paired + "' together with it" if paired else "")
                          + ", so the wanted " + prop + (" is '" + wanted_value + "'." if ref.get("same", True) else " must differ from it.")
                          + " The facts named with it only identified that listing." + seen + "\n" + self.status()})
        return True

    def commit_basket(self) -> list[tuple[str, int]] | None:
        """The basket to buy instead of declining: per product the best-checked opened listing in stock and within the budget
        whose own record contradicts none of that product's needs (proven fits first, then the fewest needs left open, then
        the cheapest). None when some product has no such listing, or no basket of them fits the budget."""
        def candidates(section: list[dict[str, Any]], needs: list[dict[str, Any]], count: int, room: Any, stated: set[str]) -> list[dict[str, Any]]:
            out = []
            for item in section:
                if item.get("facts") is None or item.get("gone") or item.get("stock") is False or not _is_number(item.get("price")):
                    continue
                if _is_number(room) and item["price"] * count > room + 1e-9:
                    continue
                reading = read_listing(needs, item["facts"], item)
                if reading["verdict"] != "fails" and self.states_stated(reading, stated):
                    off = sum(1 for n, state, _ in reading.get("loose") or [] if state == "weak" and n.get("said") == INFERRED_SAID)
                    out.append((reading["verdict"] != "fits", off, len(reading.get("loose") or []), item["price"], item["id"], item))
            out.sort(key=lambda r: r[:5])
            return [r[5] for r in out]
        if not getattr(self, "parts", []):
            wanted = [w for w in _words(self.brief.get("item") or "") if len(w) > 2]

            def covered(label: Any) -> int:
                have = _words(label or "")
                return sum(1 for w in wanted if any(_same(w, h) for h in have))
            top = max([covered(i.get("section")) for i in self.shelf.items.values() if i.get("section")] or [0])
            # KS: with no section naming the product, only a listing whose title names it is one (not a reference page of another kind)
            section = [i for i in self.shelf.items.values() if (covered(i.get("section")) == top if top else covered(i.get("title")) > 0)]
            picks = candidates(section, self.ledger.needs, self.ledger.count, self.ledger.ceiling, getattr(self, "stated_sigs", set()))
            return [(picks[0]["id"], self.ledger.count)] if picks else None
        if self.fit_groups():
            fitted = self.fit_basket()   # FT: every part line of the host's series
            if fitted is None:
                return None
            lines = [(_as_int(part.get("count")) or 1, fitted[k]) for k, (part, _, _) in enumerate(self.list_plans())]
            if _is_number(self.ledger.ceiling) and sum(count * item["price"] for count, item in lines) > self.ledger.ceiling + 1e-9:
                return None
            return [(item["id"], count) for count, item in lines]
        chosen = []
        for part, section, needs in self.list_plans():
            count = _as_int(part.get("count")) or 1
            picks = candidates(section, needs, count, self.ledger.ceiling, {_need_sig(n) for n in clean_needs(part.get("needs"))})
            if not picks:
                return None
            chosen.append((count, picks))
        basket = [(count, picks[0]) for count, picks in chosen]
        if _is_number(self.ledger.ceiling) and sum(count * item["price"] for count, item in basket) > self.ledger.ceiling + 1e-9:
            basket = [(count, min(picks, key=lambda i: (i["price"], i["id"]))) for count, picks in chosen]
            if sum(count * item["price"] for count, item in basket) > self.ledger.ceiling + 1e-9:
                return None
        return [(item["id"], count) for count, item in basket]

    @staticmethod
    def states_stated(reading: dict[str, Any], stated: set[str]) -> bool:
        """Every need the request itself stated reads met on this listing (or cannot be decided by code)."""
        return all(state in ("met", "unsure") for need, (state, _) in reading["states"] if _need_sig(need) in stated)

    def cart_buyable(self) -> bool:
        """The cart holds, for every product, a line in stock whose own record contradicts none of that product's needs, and
        the whole is within the budget."""
        lines = [self.shelf.items.get(i) for i in self.cart]
        if not lines or any(line is None or line.get("gone") or line.get("stock") is False for line in lines):
            return False
        units = {i: (self.bag.get(i) if self.bag_fresh and self.bag.get(i) else None) for i in self.cart}
        if getattr(self, "parts", []):
            for part, section, needs in self.list_plans():
                mine = [line for line in lines if any(i["id"] == line["id"] for i in section)]
                stated = {_need_sig(n) for n in clean_needs(part.get("needs"))}
                if not mine or any(line.get("facts") is None or read_listing(needs, line["facts"], line)["verdict"] == "fails"
                                   or not self.states_stated(read_listing(needs, line["facts"], line), stated) for line in mine):
                    return False
                for line in mine:
                    units[line["id"]] = units[line["id"]] or _as_int(part.get("count")) or 1
        elif any(line.get("facts") is None or read_listing(self.ledger.needs, line["facts"], line)["verdict"] == "fails"
                 or not self.states_stated(read_listing(self.ledger.needs, line["facts"], line), getattr(self, "stated_sigs", set())) for line in lines):
            return False
        total = sum((line["price"] if _is_number(line.get("price")) else 0) * (units.get(line["id"]) or self.ledger.count) for line in lines)
        return not _is_number(self.ledger.ceiling) or total <= self.ledger.ceiling + 1e-9

    def put_basket(self, basket: list[tuple[str, int]]) -> bool:
        """Make the cart hold exactly this basket (lines not in it taken out, missing ones added). False when a call cannot be
        built or the steps do not suffice to also check out."""
        keep = {item_id for item_id, _ in basket}
        if self.lines_slot():
            # LS: a line-list cart takes every line in one call (calls built one by one would each start from the old cart)
            sets = self.lines_set()
            if not sets and any(i not in keep for i in self.cart):
                return False
            lines = list(basket) if sets else [(i, n) for i, n in basket if i not in self.cart]
            call = self.lines_call(lines) if lines else None
            if (lines and not call) or self.steps_left() < 2:
                return False
            if call:
                self.execute([(call["name"], call["args"])])
                self.cart = [i for i, _ in basket]
                self.bag = {i: n for i, n in basket}
            return True
        outs = [self.remove_action(i) for i in self.cart if i not in keep]
        adds = [self.add_action(i, n) for i, n in basket if i not in self.cart]
        if not all(outs) or not all(adds) or self.steps_left() < len(outs) + len(adds) + 1:
            return False
        if outs or adds:
            self.execute([(a["name"], a["args"]) for a in outs + adds])
        return True

    def commit_veto(self) -> str:
        """A decline while something can be bought: refused with no expiry; see commit_basket. The first refusal (with steps to
        spare) only advises; after that the code puts the best-checked basket in the cart and checks it out itself, because a
        driver that empties the cart and declines again only burns the steps the purchase needs."""
        if not self.role.get("order") or self.steps_left() < 1 or self.wire.left() < LAST_CALL_SECONDS + 5:
            return ""
        holding = bool(self.cart) and self.cart_buyable()
        basket = None if holding else self.commit_basket()
        if not holding and not basket:
            return ""
        times = self.mark("commit-any")
        if times == 1 and self.steps_left() > 10:
            if holding:
                self.trace.append({"kind": "fire", "set": "DK", "cart": list(self.cart), "times": times})
                return ("Not run yet (no step used). The cart already holds " + "; ".join(_label(self.shelf.items[i]) for i in self.cart if i in self.shelf.items)
                        + ", and their own records contradict none of the needs. A decline scores 0 whenever something can be bought, and the store almost never wants one "
                        "while such a basket exists: check out now." + (" The checkout itself shows the cart to the shopper for the approval the request asks for." if self.approval else ""))
            named = "; ".join(str(n) + " x " + _label(self.shelf.items.get(i) or {"id": i}) for i, n in basket)
            self.trace.append({"kind": "fire", "set": "DK", "advice": basket})
            return ("Not run yet (no step used). A decline scores 0 whenever something can be bought, and the store almost never wants one while opened listings "
                    "contradict none of the needs. The best-checked so far: " + named + ". Search on for a listing whose record states the needs still open; "
                    "then buy the best you have. The decline will not be sent while such listings exist.")
        if not holding and not self.put_basket(basket):
            return ""
        order = self.final_action("order")
        self.trace.append({"kind": "fire", "set": "DK", "checkout": list(self.cart), "times": times})
        if not order:
            return "Not run yet (no step used). The best-checked basket is in the cart; check out now."
        held = self.confirm_cart()
        if held:
            return held
        self.execute([(order["name"], order["args"])])
        return "The decline was not sent; the best-checked basket in the cart was checked out instead."

    def commit_deadline(self) -> bool:
        """When the steps left only just suffice to buy and the cart holds no buyable basket, put the best-checked basket in
        the cart once and tell the driver to check out (searching on would end in a forced decline)."""
        if self.orders_tried or not self.role.get("order") or self.wire.left() < CLOSING_SECONDS or self.cart_buyable():
            return False
        basket = self.commit_basket()
        if not basket:
            return False
        keep = {item_id for item_id, _ in basket}
        need = sum(1 for i in self.cart if i not in keep) + sum(1 for i, _ in basket if i not in self.cart) + 1
        if self.steps_left() > need + 3 or self.mark("commit-deadline") > 1 or not self.put_basket(basket):
            return False
        self.trace.append({"kind": "fire", "set": "DK", "deadline": basket})
        self.chat.append({"role": "user", "content": "The steps left only just suffice to buy, so I put the best-checked listings in the cart: "
                          + "; ".join(str(n) + " x " + _label(self.shelf.items.get(i) or {"id": i}) for i, n in basket)
                          + ". Their own records contradict none of the needs. Check out now; decline only if one of them clearly fails a need by its own record.\n" + self.status()})
        return True

    def line_units(self, item_id: str) -> int:
        if self.bag_fresh and self.bag.get(item_id):
            return self.bag[item_id]
        item = self.shelf.items.get(item_id)
        for part in getattr(self, "parts", []):
            if item and self.in_part(item, part):
                return _as_int(part.get("count")) or 1
        return _as_int(self.ledger.count) or 1

    def basket_text(self, basket: list[tuple[str, int]], titles: bool = True) -> str:
        """The approval message: each line's units, title and price, the total and the budget, and one yes-or-no question. It
        asks about no property and raises no doubt, so it reads as the basket shown for approval."""
        def amount(value: float) -> str:
            return str(int(value)) if float(value).is_integer() else "%.2f" % value
        lines, total = [], 0.0
        for item_id, units in basket:
            item = self.shelf.items.get(item_id) or {}
            title = re.sub(r"\s+", " ", str(item.get("title") or "")).strip() if titles else ""
            if len(title) > 70:
                title = title[:70].rsplit(" ", 1)[0] + "..."
            line = str(units) + " x " + (title + " (" + item_id + ")" if title else item_id)
            if _is_number(item.get("price")):
                total += item["price"] * units
                line += " at " + amount(item["price"]) + " each"
            lines.append(line)
        money = (" " + self.ledger.money) if self.ledger.money else ""
        text = "Final basket before I place the order: " + "; ".join(lines) + "."
        if total:
            text += " Total " + amount(total) + money
            if _is_number(self.ledger.ceiling):
                text += ", within your budget of " + amount(self.ledger.ceiling) + money
            text += "."
        return text + " Shall I place this order? Please reply yes to approve."

    def confirm_cart(self, basket: list[tuple[str, int]] | None = None) -> str:
        """FB: when the request asks to approve the basket before the order, send the basket about to be ordered (again when it
        changed since and a message is left). A cart not viewed since its last change is viewed first: a change that filling the
        cart sets off shows on the next call, and would make the approval stale. "" lets the checkout go; otherwise a note for
        the driver (the view or the reply brought a change). basket: the closing path's lines, which bypass the cart records."""
        tool = self.role.get("message")
        if not tool or not self.approval or self.message_room() <= 0 or self.wire.finished or self.wire.broken or self.wire.key_dead \
                or self.wire.left() < 8:
            return ""
        lines = basket if basket is not None else [(i, self.line_units(i)) for i in self.cart]
        if not lines or self.steps_left() < 2 or _dump(sorted(lines)) + "#" + str(self.questions_sent) == self.confirmed_cart:
            return ""
        view = self.role.get("cart_view")
        if basket is None and view and not (self.schemas.get(view) or {}).get("required") and self.steps_left() >= 3 \
                and self.wire.left() > LAST_CALL_SECONDS and (not self.bag_fresh or self.wire.step <= self.cart_step) \
                and self.mark("approval-view|" + _dump(sorted(lines))) == 1:
            done = self.execute([(view, {})])
            changes = self.voice(done[0][2], False) if done else []
            if changes and not self.wire.finished:
                self.trace.append({"kind": "fire", "set": "FB", "view_change": changes[:3]})
                return ("Not run yet (no step used for the checkout). Before asking the shopper to approve the basket I viewed the cart, and the result "
                        "brought a change, which binds from now on: " + " | ".join(changes[:3]) + ". Fix the cart if the change asks for it, then check out again.")
            lines = [(i, self.line_units(i)) for i in self.cart]
            if not lines or self.steps_left() < 2 or _dump(sorted(lines)) + "#" + str(self.questions_sent) == self.confirmed_cart:
                return ""
        args = None
        for titles in (True, False):
            args = self.message_args(self.basket_text(lines, titles))
            if args:
                args, issues = conform(args, self.schemas[tool])
                args = None if issues else args
            if args:
                break
        if not args:
            return ""
        self.planner_sending = True   # the reply is read here, not by the driver's message path
        try:
            done = self.execute([(tool, args)])
        finally:
            self.planner_sending = False
        self.confirmed_cart = _dump(sorted(lines)) + "#" + str(self.questions_sent)   # a later message makes it stale
        observation = done[0][2] if done else None
        self.trace.append({"kind": "fire", "set": "FB", "sent": lines, "asked": self.approval, "steps_left": self.steps_left()})
        changes = self.voice(observation, False)   # a notice riding on the reply (the reader's own answer is a plain string)
        if changes and basket is None and not self.wire.finished:
            meaning = self.hear(changes)
            return ("Not run yet (one step went to the shopper's approval). The reply brought a change, which binds from now on: " + " | ".join(changes[:3])
                    + (" " + meaning if meaning else "") + " Fix the cart if the change asks for it, then check out again.")
        objection = self.basket_objection(observation, self.basket_text(lines, False)) if basket is None and not self.wire.finished else ""
        if objection:
            self.confirmed_cart = ""   # the shown basket is not the one to order: it is shown again after the repair, if a message is left
            return ("Not run yet (one step went to the shopper's approval). The shopper's reply to the basket asks for a change: " + objection
                    + " Make that change in the cart, then check out again.")
        return ""

    def basket_objection(self, observation: Any, shown: str) -> str:
        """RY: the shopper's reply to the shown basket, read for an objection only. "" when there is none (an acknowledgement,
        a request to repeat, an unclear reply) or when too little time or too few steps are left to repair anything."""
        said: list[str] = []
        stack = [observation]
        while stack and len(said) < 6:
            node = stack.pop(0)
            if isinstance(node, str):
                if len(node.split()) >= 3:
                    said.append(node.strip()[:400])
            elif isinstance(node, dict):
                stack.extend(node.values())
            elif isinstance(node, list):
                stack.extend(node)
        if not said or self.wire.left() < LAST_CALL_SECONDS + 20 or self.steps_left() < 4:
            return ""
        answer = self.wire.ask_json(REPLY_PROMPT + "\n\nTHE ASSISTANT SENT:\n" + shown + "\n\nTHE SHOPPER REPLIED:\n" + "\n".join(said), [HELPER], 300)
        if not isinstance(answer, dict) or answer.get("change") is not True:
            return ""
        what = str(answer.get("what") or "").strip()[:240]
        self.trace.append({"kind": "fire", "set": "RY", "reply": said[:2], "what": what})
        self.hear(said, shown)   # the change can name a need or a limit: the ledger takes it in
        return what or " | ".join(said)[:240]

    def read_every_view(self) -> str:
        """AV: a store with two single-listing read tools (a page and a property list) may count either one as looking at a
        listing, so every cart line is read with both before the checkout. Returns a note for the driver, or "" when none."""
        view, details = self.role.get("view"), self.role.get("details")
        if not view or not details or view == details or self.wire.left() < CLOSING_SECONDS:
            return ""
        calls: list[tuple[str, dict[str, Any]]] = []
        due: list[tuple[str, str]] = []
        for job, tool in (("details", details), ("view", view)):
            schema = self.schemas.get(tool) or {}
            props = schema.get("properties") or {}
            required = list(schema.get("required") or [])
            slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
            target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
            if not target or any(k != target for k in required):
                continue
            for item_id in self.cart:
                if item_id in self.read_by.get(job, set()) or (job, item_id) in self.av_tried or item_id not in self.shelf.items:
                    continue
                args, issues = conform({target: item_id}, schema)
                if not issues:
                    calls.append((tool, args))
                    due.append((job, item_id))
        if not calls or self.steps_left() < len(calls) + 1:
            return ""
        self.av_tried.update(due)
        done = self.execute(calls)
        notes = [text.split("[note]", 1)[1].strip() for _, text, _ in done if "[note]" in text]
        self.trace.append({"kind": "fire", "set": "AV", "reads": [job + ":" + item_id for job, item_id in due], "notes": len(notes)})
        if not notes:
            return ""
        return ("The checkout was not sent: before it I read the cart's lines with " + details + " and " + view + " as well. " + " ".join(notes)
                + " Fix the cart first if a line no longer fits, then check out.")

    def reread_cart(self) -> str:
        """Open again the cart lines not opened since the last cart change (stock and price can change when a listing is put in
        the cart, and the cart view need not show it). Returns what changed, or "" when nothing did."""
        lines = [i for i in self.cart if i not in self.read_after_cart and i not in self.reread_tried and i in self.shelf.items]
        if not lines or self.steps_left() <= len(lines) + 4:
            return ""
        tool = self.stock_tool()
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        if not tool or not target or any(k != target for k in required):
            return ""
        before = {i: (self.shelf.items[i].get("price"), self.shelf.items[i].get("stock")) for i in lines}
        calls = []
        for item_id in lines:
            args, issues = conform({target: item_id}, schema)
            if not issues:
                calls.append((tool, args))
        if not calls:
            return ""
        done = self.execute(calls)
        self.reread_tried.update(lines)
        changed = []
        for item_id in lines:
            item = self.shelf.items.get(item_id) or {}
            price, stock = before[item_id]
            if item.get("stock") is False or item.get("gone"):
                changed.append(_label(item) + " now shows no stock")
            elif _is_number(item.get("price")) and _is_number(price) and item["price"] != price:
                changed.append(_label(item) + " now costs " + str(item["price"]) + " (it was " + str(price) + ")")
        notes = [text.split("[note]", 1)[1].strip() for _, text, _ in done if "[note]" in text]
        self.trace.append({"kind": "fire", "set": "BF", "lines": lines, "changed": changed, "notes": len(notes)})
        if not changed and not notes:
            return ""
        return ("The checkout was not sent: I opened the cart's lines again after your last cart change. " + " ".join(
            ([("; ".join(changed) + ".")] if changed else []) + notes)
            + " Fix the cart first (take out a line that is no longer in stock or no longer fits the budget and put in the next fitting listing), then check out.")

    def unclear_fits(self) -> list[dict[str, Any]]:
        """Opened listings of the requested section, within budget and in stock, whose own facts contradict no need (some need left
        unstated): the best-checked candidates, fewest needs left open first, then the cheapest."""
        wanted = [w for w in _words(self.brief.get("item") or "") if len(w) > 2]

        def covered(label: Any) -> int:
            have = _words(label or "")
            return sum(1 for w in wanted if any(_same(w, h) for h in have))
        best = max([covered(i.get("section")) for i in self.shelf.items.values() if i.get("section")] or [0])
        out = []
        opened = [i.get("facts") for i in self.shelf.items.values() if isinstance(i.get("facts"), dict)]
        for item, reading in self.shelf.ranked(self.ledger.needs, self.ledger.ceiling, self.ledger.count):
            if reading["verdict"] != "open" or (best and covered(item.get("section")) != best):
                continue
            against = [_related_against(n, item.get("facts"), opened) for n, _, _ in reading.get("loose") or []]
            if any(against):
                if self.mark("ra|" + str(item["id"])) == 1:
                    self.trace.append({"kind": "fire", "set": "RA", "drop": item["id"], "why": next(a for a in against if a)[:120]})
                continue
            out.append((len(reading.get("loose") or []), item["price"] if _is_number(item.get("price")) else 1e18, item))
        out.sort(key=lambda p: (p[0], p[1]))
        return [item for _, _, item in out]

    def list_plans(self) -> list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]:
        """Per list product: (the product, its section's listings, its needs: the stated ones plus each heard need that names
        it, or that names no product while its section's opened listings carry the property)."""
        def covers(words: list[str], label: Any) -> bool:
            have = _words(str(label or ""))
            return bool(words) and all(any(_same(w, h) for h in have) for w in words)
        plans = []
        for part in self.parts:
            section = [i for i in self.shelf.items.values() if self.in_part(i, part)]
            keys = {k for i in section if i.get("facts") for k in i["facts"]}
            needs = clean_needs(part.get("needs"))
            for need in self.ledger.needs:
                # the shopper's own line that states the need ("material of the jumpsuits & rompers ... cotton") names its product
                said = [str(need.get("property")) + " " + str(need.get("said") or "")]
                said += [line for line in self.ledger.heard if all(any(_same(w, h) for h in _words(line)) for v in need.get("values") or [] for w in _words(v))]
                named = [p for p in self.parts if any(covers([w for w in _words(str(p.get("item") or "")) if len(w) > 2], text) for text in said)]
                if (part in named) or (not named and any(_key_grade(k, need.get("property")) for k in keys)):
                    needs.append(dict(need, property=re.split(r"\s+of\s+the\s+", str(need.get("property") or ""))[0]))
            plans.append((part, section, needs))
        return plans

    # ---------------- FT: a part that must fit the host listing bought with it
    def part_named(self, text: str) -> int | None:
        """FT: the list product a phrase names: every longer word of its name is in the phrase (the longest such name wins)."""
        said = _words(text)
        best, size = None, 0
        for k, part in enumerate(self.parts):
            name = [w for w in _words(str(part.get("item") or "")) if len(w) > 2]
            if name and all(any(_same(w, h) for h in said) for w in name) and len(name) > size:
                best, size = k, len(name)
        return best

    def fit_groups(self) -> list[tuple[int, list[int]]]:
        """FT: per host product, the list products that must fit it (FD: else the pairs the listings show)."""
        groups: dict[int, list[int]] = {}
        for part, host in [(a, b) for a, b, _ in getattr(self, "fit_rules", [])] or self.fit_seen():
            if part not in groups.get(host, []):
                groups.setdefault(host, []).append(part)
        return list(groups.items())

    def fit_seen(self) -> list[tuple[int, int]]:
        """FD: with no fit sentence in the request (and the cheapest not asked), the pairs the data shows: a list product whose
        opened listing names part series, and another whose opened listing carries one of those series. Soft (fit_soft)."""
        if getattr(self, "fit_rules", []) or len(getattr(self, "parts", [])) < 2 or self.brief.get("pick") == "cheapest":
            return []
        plans = self.list_plans()
        named = [{c for i in section if i.get("facts") is not None for c in _fit_codes(i)} for _, section, _ in plans]
        pairs = [(part, host) for host, codes in enumerate(named) if codes for part, (_, section, _) in enumerate(plans)
                 if part != host and not named[part] and any(_carries(i, c) for i in section if i.get("facts") is not None for c in codes)]
        if pairs:
            self.mark("fd|" + _dump(sorted(pairs)))
        return pairs

    def fit_dead(self, host: dict[str, Any], part_ks: list[int]) -> bool:
        """HS: every series this host listing names left some part that must fit it with no fit found after the searches."""
        codes = _fit_codes(host)
        gone = getattr(self, "fit_gone", set())
        return bool(codes) and all(any((a, c) in gone for a in part_ks) for c in codes)

    def fit_where(self, a: int, plans: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]) -> tuple[set[str], bool]:
        """SW: the series (named by some seen listing) that the part's rows returned by its series-free search for its stated
        values carry, and whether that search listed every row it found on a page the store did not fill (then a series it
        does not show has no listing stating those values)."""
        entry = self.searches.get(getattr(self, "fit_asked", {}).get(a, ""))
        if not entry:
            return set(), False
        rows = [self.shelf.items[i] for i in entry["ids"] if i in self.shelf.items and self.in_part(self.shelf.items[i], plans[a][0])]
        codes = {c for i in self.shelf.items.values() for c in _fit_codes(i)}
        carried = {c for c in codes if any(_carries(i, c) for i in rows)}
        widest = max([e["size"] for e in self.searches.values()] or [0])
        worded = all(re.search(r"[a-z]{2}", str(v), re.I) for n in clean_needs(plans[a][0].get("needs")) if n.get("test") == "is" for v in n.get("values") or [])
        return carried, worded and bool(carried) and len(entry["ids"]) >= entry["total"] and entry["total"] < widest   # a "6 pack" search is a hint only

    @staticmethod
    def fit_value(need: dict[str, Any], value: Any) -> str:
        """SW: a stated value as search words: a worded value of up to 3 words as it is; a bare number with the first word of
        its property ("pack size: 6" -> "6 pack") or with its unit when the need gives one ("512 gb"); else ""."""
        text = str(value).strip()
        if re.search(r"[a-z]{2}", text, re.I):
            return text if len(text.split()) <= 3 else ""
        unit = str(need.get("unit") or "").strip().lower()
        if re.fullmatch(r"\d+(?:\.\d+)?", text) and re.fullmatch(r"[a-z]{1,6}", unit):
            return text + " " + unit   # HV: "storage: 512 gb" -> "512 gb", as a property value the search reads ("512 storage" finds none)
        head = next(iter(w for w in str(need.get("property") or "").split() if re.fullmatch(r"[A-Za-z]{3,}", w)), "")
        return text + " " + head if head and re.fullmatch(r"\d+(?:\.\d+)?", text) else ""

    def fit_soft(self, part: int, host: int) -> bool:
        """FD: a fitting part is only preferred (the request names the pair by no fit sentence of its own): a part line of
        another series stays buyable when none of the host's series is known."""
        return not any(a == part and b == host for a, b, _ in getattr(self, "fit_rules", [])) or (part, host) in getattr(self, "fit_loose", set())

    def fit_sentence(self, part: int, host: int) -> str:
        return next((s for a, b, s in getattr(self, "fit_rules", []) if a == part and b == host), "")

    def fit_fits(self, k: int, plans: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]) -> list[dict[str, Any]]:
        """FT: a list product's opened listings in stock, priced and within the budget whose own facts state its stated needs and
        contradict none of its needs: proven fits first, then the cheapest."""
        part, section, needs = plans[k]
        count = _as_int(part.get("count")) or 1
        stated = {_need_sig(n) for n in clean_needs(part.get("needs"))}
        out = []
        for item in section:
            if item.get("facts") is None or item.get("gone") or item.get("stock") is False or not _is_number(item.get("price")):
                continue
            if _is_number(self.ledger.ceiling) and item["price"] * count > self.ledger.ceiling + 1e-9:
                continue
            reading = read_listing(needs, item["facts"], item)
            if reading["verdict"] != "fails" and self.states_stated(reading, stated):
                out.append((reading["verdict"] != "fits", item["price"], str(item["id"]), item))
        out.sort(key=lambda r: r[:3])
        return [r[3] for r in out]

    def fit_host(self, host_k: int, plans: list[tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]]) -> tuple[dict[str, Any] | None, bool]:
        """FT: the host listing the parts must fit: the cart's line of the host product, else its best opened fit (and whether it is
        the cart's)."""
        held = [self.shelf.items[i] for i in self.cart if i in self.shelf.items and self.in_part(self.shelf.items[i], plans[host_k][0])]
        if held:
            return held[0], True
        return next(iter(self.fit_fits(host_k, plans)), None), False

    def fit_basket(self) -> dict[int, dict[str, Any]] | None:
        """FT: per list product, its listing in the best basket known that keeps every fit rule. Each host listing (an opened fit
        of its product) takes, for every part that must fit it, the part's best opened fit naming a series the host's listing names
        (a host naming none restricts nothing); the other products take their best opened fit. Under a cheapest rule the basket
        with the fewest unproven lines and then the lowest total wins; otherwise the cart's host first, then the best host. None
        when some product has no such listing."""
        plans = self.list_plans()
        fits = [self.fit_fits(k, plans) for k in range(len(plans))]
        counts = [_as_int(p.get("count")) or 1 for p, _, _ in plans]
        cheap = self.brief.get("pick") == "cheapest"
        pick: dict[int, dict[str, Any]] = {}
        for host_k, part_ks in self.fit_groups():
            best = None
            for host in sorted(fits[host_k][:12], key=lambda h: h["id"] not in self.cart):
                codes = _fit_codes(host)
                chosen: dict[int, dict[str, Any]] = {host_k: host}
                misses = 0
                for a in part_ks:
                    match = [i for i in fits[a] if not codes or any(_carries(i, c) for c in codes)]
                    if not match and self.fit_soft(a, host_k) and fits[a]:
                        match, misses = fits[a][:1], misses + 1   # FD: preferred only
                    if not match:
                        break
                    chosen[a] = match[0]
                if len(chosen) <= len(part_ks):
                    continue
                key = (misses, sum(1 for k, i in chosen.items() if read_listing(plans[k][2], i["facts"], i)["verdict"] != "fits"),
                       sum(i["price"] * counts[k] for k, i in chosen.items()))
                if best is None or key < best[0]:
                    best = (key, chosen)
                if not cheap and not misses:
                    break
            if best is None:
                return None
            pick.update(best[1])
        for k in range(len(plans)):
            if k not in pick:
                if not fits[k]:
                    return None
                pick[k] = fits[k][0]
        return pick

    def fit_note(self) -> str:
        """FT: status text: per host product, the host listing the parts must fit with the series its listing names, each part's
        opened fits of that series and the cart lines that name none of it; under a cheapest rule the cheapest basket known that
        keeps every fit."""
        groups = self.fit_groups()
        if not groups:
            return ""
        plans = self.list_plans()
        out = []
        for host_k, part_ks in groups:
            kind = str(plans[host_k][0].get("item"))
            names = ", ".join(str(plans[a][0].get("item")) for a in part_ks)
            host, held = self.fit_host(host_k, plans)
            codes = _fit_codes(host) if host else []
            if not codes and all(self.fit_soft(a, host_k) for a in part_ks) and (part_ks[0], host_k) not in getattr(self, "fit_loose", set()):
                continue
            if not codes:
                out.append("the " + names + " must fit the " + kind + " you buy: a " + kind + " listing names the part series that fit it"
                           " (e.g. 'Compatible part series: X'), and a fitting part names that series (e.g. in its title)"
                           + ("; " + _label(host) + (" in the cart" if held else "") + " names none yet" + ("; open its page" if not host.get("viewed") else "") if host else
                              "; choose the " + kind + " first"))
                continue
            bits = []
            soft = all(self.fit_soft(a, host_k) for a in part_ks)
            for a in part_ks:
                part = plans[a][0]
                known = [i for i in self.fit_fits(a, plans) if any(_carries(i, c) for c in codes)]
                wrong = [i for i in self.cart if i in self.shelf.items and self.in_part(self.shelf.items[i], part)
                         and not any(_carries(self.shelf.items[i], c) for c in codes)]
                bit = str(part.get("item")) + ": " + ("opened fits of that series " + "; ".join(_label(i) for i in known[:3]) if known else
                                                        "no opened fit of that series yet" + ("" if self.fit_soft(a, host_k) else
                                                                                             " (search " + codes[0] + " with its words, open the results)"))
                if wrong and (known or not self.fit_soft(a, host_k)):
                    bit += "; cart line " + ", ".join(wrong) + " names no " + "/".join(codes) + ": replace it"
                bits.append(bit)
            if soft:   # FD: the listings show the pairing; the request sets no fit rule
                out.append("the " + kind + " " + _label(host) + (" in the cart" if held else " (its best opened fit)") + " names part series "
                           + "/".join(codes) + " on its listing, and " + names + " listings carry such series: a part naming " + "/".join(codes)
                           + " is made for it, so take one when it meets the needs (not required). " + " | ".join(bits))
                continue
            out.append("the " + names + " must fit the " + kind + " " + _label(host) + (" in the cart" if held else " (its best opened fit)")
                       + ", whose listing names part series " + "/".join(codes) + ": only a part listing naming " + "/".join(codes) + " fits it. " + " | ".join(bits))
        basket = self.fit_basket() if self.brief.get("pick") == "cheapest" else None
        if basket:
            counts = [_as_int(p.get("count")) or 1 for p, _, _ in plans]
            out.append("cheapest basket known that keeps every fit: " + "; ".join(str(counts[k]) + " x " + _label(basket[k]) for k in sorted(basket))
                       + " (together " + str(sum(basket[k]["price"] * counts[k] for k in basket)) + ").")
        return "\n[fits] " + " ".join(out)

    def fit_fault(self) -> str:
        """FT: why a checkout breaks a fit rule: a part line naming none of the series the cart's host listing names (with that
        series' opened fits), or under a cheapest rule a cheaper basket known that keeps every fit; "" when none."""
        plans = self.list_plans()
        for host_k, part_ks in self.fit_groups():
            host, held = self.fit_host(host_k, plans)
            codes = _fit_codes(host) if host and held else []
            basket = self.fit_basket() if codes and self.fit_dead(host, part_ks) else None
            if basket and basket.get(host_k) and basket[host_k]["id"] != host["id"] and self.mark("hs-fault|" + str(host["id"])) == 1:
                counts = [_as_int(p.get("count")) or 1 for p, _, _ in plans]
                return ("The " + str(plans[host_k][0].get("item")) + " in the cart, " + _label(host) + ", names part series " + "/".join(codes)
                        + ", but no listing of that series meeting the needs was found for every part that must fit it. This basket keeps every fit"
                        " rule: " + "; ".join(str(counts[k]) + " x " + _label(basket[k]) for k in sorted(basket)) + ". Put it in the cart instead, then check out.")
            for a in part_ks if codes else []:
                part = plans[a][0]
                wrong = [i for i in self.cart if i in self.shelf.items and self.in_part(self.shelf.items[i], part)
                         and not any(_carries(self.shelf.items[i], c) for c in codes)]
                if not wrong:
                    continue
                known = [i for i in self.fit_fits(a, plans) if any(_carries(i, c) for c in codes)]
                if self.fit_soft(a, host_k):
                    if not known or self.mark("fd-fault|" + str(a) + "|" + "/".join(codes)) > 1:
                        continue   # FD: preferred only, and said once
                    return ("The " + str(plans[host_k][0].get("item")) + " in the cart, " + _label(host) + ", names part series " + "/".join(codes)
                            + " on its listing, and the " + str(part.get("item")) + " line " + ", ".join(wrong) + " names none of it, while "
                            + "; ".join(_label(i) for i in known[:3]) + " (opened, that series) meet" + ("s" if len(known) == 1 else "")
                            + " its needs and is made for it. Take that one unless the cart line is better on a stated need, then check out.")
                return ("The request says: '" + self.fit_sentence(a, host_k) + "' The " + str(plans[host_k][0].get("item")) + " in the cart, " + _label(host)
                        + ", names part series " + "/".join(codes) + " on its listing, but the " + str(part.get("item")) + " line " + ", ".join(wrong)
                        + " names none of it, so it does not fit. " + ("Replace it with " + "; ".join(_label(i) for i in known[:3]) + " (opened, that series)"
                        if known else "Search " + codes[0] + " with the " + str(part.get("item")) + "'s words, open the results and take one that names "
                        + codes[0] + " and meets its needs") + ", then check out.")
        if self.brief.get("pick") != "cheapest":
            return ""
        basket = self.fit_basket()
        if not basket or {i["id"] for i in basket.values()} == set(self.cart):
            return ""
        counts = [_as_int(p.get("count")) or 1 for p, _, _ in plans]
        total = sum(basket[k]["price"] * counts[k] for k in basket)
        held = [self.shelf.items.get(i) for i in self.cart]
        if any(i is None or not _is_number(i.get("price")) for i in held):
            return ""
        now = sum(i["price"] * self.line_units(i["id"]) for i in held)
        if total >= now - 1e-9:
            return ""
        return ("A cheaper basket that keeps every fit rule is known: " + "; ".join(str(counts[k]) + " x " + _label(basket[k]) for k in sorted(basket))
                + " (together " + str(total) + "; the cart comes to " + str(now) + "). Put it in the cart instead of the dearer lines, then check out.")

    def query_call(self, text: str) -> tuple[str, dict[str, Any]] | None:
        """FT: one plain search for a text: the search tool's search-text argument alone (no paging or section argument)."""
        tool = self.role.get("search")
        schema = self.schemas.get(tool or "") or {}
        required = list(schema.get("required") or [])
        slots = [k for k in self.text_slots(tool or "") if not re.search(r"next|page|cursor|token|onward|continu|shelf|section|categor|department|aisle", k, re.I)]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        if not tool or not target or any(k != target for k in required):
            return None
        args, issues = conform({target: text}, schema)
        return None if issues else (tool, args)

    def fit_work(self) -> bool:
        """FT: the code reads what the fit rules need, once per host listing and series. The host's page when its listing names
        no series yet (the cart's host line, else its best opened fits: the two best when the cheapest is asked); then per part
        that must fit it, a search for the series with the part's first stated worded value (else its name), and up to 3 of the
        cheapest unopened rows of the part's section that name the series opened (a second round only while no opened fit of
        that series is known). True when it did something: what it read goes to the driver."""
        groups = self.fit_groups()
        if not groups or self.steps_left() < 16 or self.wire.left() < CLOSING_SECONDS + 60 or self.wire.work_spent() >= self.spend_line(SPEND_HARD):
            return False
        plans = self.list_plans()
        cheap = self.brief.get("pick") == "cheapest"
        done: list[str] = []
        for host_k, part_ks in groups:
            if all(self.fit_soft(a, host_k) and (a, host_k) not in getattr(self, "fit_loose", set()) for a in part_ks):
                continue   # FD: pairs only the data shows get no searches of their own
            # SW: before the parts are searched by a host's series, one series-free search per part for its stated worded values
            # (in one reply): the series its rows carry show which hosts can take every part. HV: the host gets one for its own
            # stated values too, so the host rows opened for a series are ones that state them ("512 gb Rugged Tablets")
            hard = [a for a in part_ks if not self.fit_soft(a, host_k) or (a, host_k) in getattr(self, "fit_loose", set())]
            asks = []
            for a in [host_k] + hard:
                value = " ".join(dict.fromkeys(self.fit_value(n, v) for n in clean_needs(plans[a][0].get("needs")) if n.get("test") == "is" for v in n.get("values") or []
                                               if self.fit_value(n, v)))
                if value and a not in self.fit_asked and self.steps_left() >= 18:
                    call = self.query_call(" ".join((value + " " + str(plans[a][0].get("item") or "")).split()[:10]))
                    if call:
                        self.fit_asked[a] = _dump(dict(sorted(call[1].items())))
                        asks.append(call)
            if asks:
                self.execute(asks)
                done.append("searched " + ", ".join("'" + str(next(iter(c[1].values()), "")) + "'" for c in asks) + " with no series, to see which series have them")
                plans = self.list_plans()
            where = {a: self.fit_where(a, plans) for a in hard}
            named = {c for i in plans[host_k][1] for c in _fit_codes(i)}
            for a, (carried, whole) in where.items():
                if whole:
                    self.fit_gone.update((a, c) for c in named - carried)   # SW: the store listed every row stating the values
            backed = [a for a in hard if where[a][1]]   # only a search that listed every row it found rules a series out

            def takes(h: dict[str, Any]) -> int:
                codes = _fit_codes(h)
                return sum(1 for a in part_ks if any(_carries(i, c) for i in self.fit_fits(a, plans) for c in codes) or any(c in where.get(a, (set(), False))[0] for c in codes))
            host, held = self.fit_host(host_k, plans)
            pool = ([host] if held and host else []) + [h for h in self.fit_fits(host_k, plans) if not (held and host and h["id"] == host["id"])]
            hosts = [h for h in pool if not self.fit_dead(h, part_ks)]   # HS: a series with a part none of whose fits was found is left
            hosts.sort(key=lambda h: -takes(h))
            if backed and not any(all(any(c in where[a][0] for c in _fit_codes(h)) for a in backed) for h in hosts) and self.mark("sw-open|" + str(host_k)) <= 3:
                # SW: no opened host listing names a series every searched part has: open the best seen one that does
                count = _as_int(plans[host_k][0].get("count")) or 1
                rows = [i for i in self.seen_rows(plans[host_k][1], plans[host_k][2]) if _fit_codes(i) and not self.fit_dead(i, part_ks)
                        and all(any(c in where[a][0] for c in _fit_codes(i)) for a in backed) and self.shelf.affordable(i, self.ledger.ceiling, count)]
                opened = self.open_rows(rows, min(2 if cheap else 1, self.steps_left() - 15))
                if opened:
                    done.append("opened the " + str(plans[host_k][0].get("item")) + " rows " + ", ".join(opened) + " (series "
                                + "/".join(sorted({c for i in rows[:2] for c in _fit_codes(i)})) + ", which every searched part has)")
                    continue
            for host in hosts[: 2 if cheap else 1]:
                codes = _fit_codes(host)
                if not codes and not host.get("viewed") and self.role.get("view") and self.mark("ft-view|" + str(host["id"])) == 1:
                    if self.open_rows([host], 1, "view"):
                        done.append("read the page of " + str(host["id"]))
                    codes = _fit_codes(host)
                for code in codes[:2]:
                    for a in part_ks:
                        if self.steps_left() < 16:
                            break
                        if self.fit_soft(a, host_k) and ((a, host_k) not in getattr(self, "fit_loose", set())
                                                         or not any(_carries(i, code) for i in plans[a][1])):
                            continue   # FD: a loose pair is worked only for a product whose rows carry the series
                        part = plans[a][0]
                        count = _as_int(part.get("count")) or 1
                        known = [i for i in self.fit_fits(a, self.list_plans()) if _carries(i, code)]
                        if known and not cheap:
                            continue
                        if self.mark("ft-search|" + str(a) + "|" + code) == 1:
                            value = next((str(v).strip() for n in clean_needs(part.get("needs")) if n.get("test") == "is" for v in n.get("values") or []
                                          if re.search(r"[a-z]", str(v), re.I) and len(str(v).split()) <= 3), "")
                            text = code + " " + (value or str(part.get("item") or ""))
                            call = self.query_call(" ".join(text.split()[:6]))
                            if call:
                                self.execute([call])
                                done.append("searched '" + str(next(iter(call[1].values()), "")) + "'")
                        if known and self.mark("ft-round|" + str(a) + "|" + code) > 1:
                            continue
                        if not known and self.mark("ft-round|" + str(a) + "|" + code) > 2:
                            self.fit_gone.add((a, code))   # HS: searched and opened twice, no fit of this series
                            continue
                        rows = [i for i in self.shelf.items.values() if self.in_part(i, part) and _carries(i, code) and i.get("facts") is None
                                and i.get("stock") is not False and not i.get("gone") and _is_number(i.get("price"))
                                and (not _is_number(self.ledger.ceiling) or i["price"] * count <= self.ledger.ceiling + 1e-9)]
                        said = set((self.searches.get(self.fit_asked.get(a, "")) or {}).get("ids") or [])
                        stated = [n for n in clean_needs(part.get("needs")) if n.get("test") == "is"]
                        # TO: a row whose title already states more of the part's needs ("6pk" for pack size 6) is opened first;
                        # SW: then the rows its need search returned
                        rows.sort(key=lambda i: (-sum(1 for n in stated if _title_says(n, i)), i["id"] not in said, i["price"], str(i["id"])))
                        opened = self.open_rows(rows, min(3, self.steps_left() - 15))
                        if opened:
                            done.append("opened the " + str(part.get("item")) + " rows of series " + code + " " + ", ".join(opened))
        if not done:
            return False
        self.trace.append({"kind": "fire", "set": "FT", "work": done[:8]})
        self.chat.append({"role": "user", "content": "For the fit rule" + ("s" if len(getattr(self, "fit_rules", [])) > 1 else "") + " of the request I "
                          + "; ".join(done) + " (those steps are used).\n" + self.status()})
        return True

    def view_meant(self) -> bool:
        """VS: the code opens the own pages (the view tool) of the reference-chain pages read only through their details, as
        chain_note lists them: the chain goes on from a page's notes, a pointer's details may sit only there, and a pointed-to
        page counts as looked at only when its own page was shown. At most 4 pages a turn, each once."""
        view = self.role.get("view")
        if not view or view == self.role.get("details") or self.steps_left() < 8 or self.wire.left() < CLOSING_SECONDS + 30:
            return False
        rows = [self.shelf.items[i] for i in getattr(self, "chain_unviewed", []) if i in self.shelf.items and not self.shelf.items[i].get("viewed")]
        rows = [i for i in rows if self.mark("vs|" + str(i["id"])) == 1][: min(4, self.steps_left() - 7)]
        opened = self.open_rows(rows, len(rows), "view") if rows else []
        if not opened:
            return False
        self.trace.append({"kind": "fire", "set": "VS", "viewed": opened})
        self.chat.append({"role": "user", "content": "The reference chain's pages " + ", ".join(opened) + " had been read only through their details, so I opened"
                          " their own pages (those steps are used): their notes are below in the status.\n" + self.status()})
        return True

    def line_queries(self) -> bool:
        """LQ: before the code hunts a reference chain, every line of a shopping list with stated worded values that no search has
        asked for by its own words gets one: those values with its name, all in one reply, once (a name alone is the driver's search). A pointer's shop may sell the line's own kind, and
        the hunt's rows of that shop then stood in for the line's candidates: on a 43-step task the hunt took the steps, the line's
        own need ("plug type: 2 pin") was never searched and a pointer page whose plug is unknown was bought."""
        if not getattr(self, "parts", None) or self.mark("lq") > 1:
            return False
        said = []
        for key in self.searches:
            try:
                args = json.loads(key)
            except (TypeError, ValueError):
                continue
            if isinstance(args, dict):
                said.append(_words(" ".join(str(v) for v in args.values() if isinstance(v, str))))
        asks = []
        for part, _, _ in self.list_plans():
            value = " ".join(dict.fromkeys(self.fit_value(n, v) for n in clean_needs(part.get("needs")) if n.get("test") == "is" for v in n.get("values") or []
                                           if self.fit_value(n, v)))
            text = " ".join((value + " " + str(part.get("item") or "")).split()[:10])
            words = _words(text)
            if not value or not words or any(all(any(_same(w, h) for h in earlier) for w in words) for earlier in said):
                continue
            call = self.query_call(text)
            if call:
                asks.append(call)
        if not asks or self.steps_left() < 12 + len(asks):
            return False
        self.execute(asks)
        texts = [str(next(iter(c[1].values()), "")) for c in asks]
        self.trace.append({"kind": "fire", "set": "LQ", "q": texts})
        self.chat.append({"role": "user", "content": "Before following the reference chain I searched each list line by its own stated values: '" + "', '".join(texts)
                          + "' (those steps are used). A line is bought from its own rows, not from a pointer's shop unless that row meets the line's needs.\n" + self.status()})
        return True

    def hop_work(self) -> bool:
        """NH: a reference-chain pointer that gives details ("the Anraanan pants (plain, summer)") means one page of that shop,
        and a search for the shop's name alone shows a page of rows where look-alikes come first. The store's search needs every
        word and reads titles, shops, sections and property values, not property names, so the shop's name with the pointer's
        value words finds that page. Once per pointer that chain_note still lists as open: that search, and when it finds no row of
        the shop (a word the index does not hold: a property name, a value only the notes give) the shop with the pointer's
        one most telling word (no figure, not a word of the section's name), then the shop with the section's name. The shop's
        rows it returned are ranked by the pointer's words on what is known of them; up to 5 own pages (their notes say which
        one the pointer meant and where the chain goes on) and up to 3 property lists are opened in one reply."""
        hunt = getattr(self, "chain_hunt", [])
        if not hunt or self.steps_left() < 12 or self.wire.left() < CLOSING_SECONDS + 60 or self.wire.work_spent() >= self.spend_line(SPEND_SOFT):
            return False
        if self.line_queries():   # LQ
            return True
        names = {w for i in self.shelf.items.values() for k in (i.get("facts") or {}) for w in re.findall(r"[a-z0-9]+", str(k).lower())}
        # NV: the request's own property names ("plug type: 2 pin") are names too before any listing's facts are read; "qianan ip
        # security cameras plug type" found no row, while the pointer's values ("universal", "aa") find its page
        names |= {w for n in list(self.ledger.needs) + [n for p in (getattr(self, "parts", None) or []) for n in clean_needs(p.get("needs"))]
                  for w in re.findall(r"[a-z0-9]+", str(n.get("property") or "").lower())}
        # ST: a name read as "graphic_card_series" is the same word as the pointer's "graphics memory"
        stems = {w.rstrip("s") for w in names if len(w) > 3}
        for shop, shelf, details in hunt:
            tag = "nh|" + shop + "|" + " ".join(details.lower().split())
            tries = self.guard_marks.get(tag, 0)
            said = [w for w in dict.fromkeys(re.findall(r"[a-z0-9]+", details.lower())) if len(w) > 1 and w not in _FILLER
                    and w not in _POINTER_VERBS and w not in names and w.rstrip("s") not in stems and w not in shop.split()]
            kind = re.findall(r"[a-z0-9]+", str(shelf or "").lower())
            said = [w for w in said if w not in _HUNT_SKIP]
            odd = [w for w in said if not w[0].isdigit() and w.rstrip("s") not in {k.rstrip("s") for k in kind}]   # a value comes first
            ladder = []
            for words in (said[:5], odd[:1], kind[:4]):
                if words and words not in ladder:
                    ladder.append(words)
            if self.lean_hop(shop, shelf, details, tag):
                return True
            if tries >= len(ladder):
                continue
            call = self.query_call(shop + " " + " ".join(ladder[tries]))
            if not call:
                self.guard_marks[tag] = len(ladder)
                continue
            self.mark(tag)
            self.execute([call])
            ids = (self.searches.get(_dump(dict(sorted(call[1].items())))) or {}).get("ids") or []
            mine = [self.shelf.items[i] for i in ids if i in self.shelf.items and self.seller_name(self.shelf.items[i]) == shop]
            mine = [i for i in mine if not shelf or i.get("section") == shelf] or mine
            if mine:
                # NH2: rows of the shop came back; when none of them proves to be the meant page (the pointer stays hunted: "the
                # Qinovex Mini PC listed at 6 kg" found the 2 kg and 5 kg ones), the next turn searches the shop with the section's
                # name, which lists every page of that kind
                last = len(ladder) - 1 if kind[:4] and ladder[-1] == kind[:4] and tries < len(ladder) - 1 else len(ladder)
                self.guard_marks[tag] = last
            known = {str(i["id"]): set(re.findall(r"[a-z0-9]+", (_dump(i.get("record") or {}) + " " + " ".join(str(v) for v in (i.get("facts") or {}).values())).lower()))
                     for i in mine}
            mine.sort(key=lambda i: -len(set(said) & known[str(i["id"])]))
            todo = [i for i in mine if i.get("facts") is None or not i.get("viewed")][:5]
            split = len(mine) >= 2 and self.role.get("view") and self.role.get("view") != self.role.get("details")
            shown = None
            if split:
                # LH: several pages of the shop came back and only one is meant (2 of 3 such opens were look-alikes, and every call
                # is a step): their own pages (notes) are opened now, and only the property list of the page that the pointer's
                # words fit best is read, on the next turn (lean_hop)
                todo = [i for i in mine if not i.get("viewed")][:6]
                shown = self.view_many(todo) if self.steps_left() > 8 else None
                calls = self.row_calls(todo, "view") if self.steps_left() > 8 and shown is None else []
            else:
                calls = self.row_calls([i for i in todo if i.get("facts") is None][:3]) if self.steps_left() > 8 else []
            if not split and self.role.get("view") and self.role.get("view") != self.role.get("details") and self.steps_left() > 8:
                calls += self.row_calls([i for i in todo if not i.get("viewed")], "view")
            if calls:
                self.execute(calls)
            q = str(next(iter(call[1].values()), ""))
            self.trace.append({"kind": "fire", "set": "NH", "shop": shop, "q": q, "rows": len(ids), "mine": [str(i["id"]) for i in mine[:6]],
                               "opened": [str(i["id"]) for i in todo]})
            found = ("it returned " + str(len(mine)) + " listing(s) of " + shop.title()
                     + ((", and I opened " + ", ".join(str(i["id"]) for i in todo) + (" (own pages only: the property list of the best fit comes next)" if split
                                                                                       else " (own pages, and the property lists of the first ones)")) if todo and (calls or shown) else "")
                     if mine else "it returned no listing of " + shop.title())
            self.chat.append({"role": "user", "content": "For the reference chain I searched '" + q + "': the shop's name with the pointer's value words"
                              " (the store's search needs every word and reads property values, not property names); " + found
                              + " (those steps are used).\n" + self.status()})
            return True
        return False

    def lean_hop(self, shop: str, shelf: str, details: str, tag: str) -> bool:
        """LH: a hunted pointer whose shop has pages opened as pages only (notes read, property list not): read the property list of
        the one whose own statements (title, notes; not its negated or pointing sentences) fit the pointer's details best, one per
        turn and at most five per pointer; chain_note then settles the pointer or the next one is read. The fit is chain_note's
        own (the same property names and words of opened listings); a page it cannot check yet (the details name a property
        and its property list is unread) is ranked by how many of the pointer's value words its title and notes hold ("binding
        cover" in "PVC Binding Cover 200 microns"), then by price."""
        if self.guard_marks.get("lh|" + tag, 0) >= 5 or self.steps_left() <= 8 or not self.role.get("details"):
            return False
        rows = [i for i in self.shelf.items.values() if self.seller_name(i) == shop and (not shelf or i.get("section") == shelf)
                and not i.get("gone")]
        pend = [i for i in rows if i.get("viewed") and i.get("facts") is None]
        if not pend:
            return False
        skip = set(_words(shop)) | {w for i in rows for w in _words(i.get("section") or "")}
        shop_keys = frozenset(" ".join(_words(k)) for i in rows for k in (i.get("facts") or {}))
        vocab = frozenset(w for i in self.shelf.items.values() if isinstance(i.get("facts"), dict)
                          for text in [str(k) + " " + str(v) for k, v in i["facts"].items()] + [str(i.get("title") or ""), str(i.get("section") or "")]
                          for w in _words(text))
        names = {w for i in self.shelf.items.values() for k in (i.get("facts") or {}) for w in _words(k)}
        names |= {w for n in list(self.ledger.needs) + [n for p in (getattr(self, "parts", None) or []) for n in clean_needs(p.get("needs"))]
                  for w in _words(str(n.get("property") or ""))}
        said = {w for w in _words(details) if w not in _FILLER and w not in _POINTER_VERBS and w not in _HUNT_SKIP and w not in skip
                and w not in names}
        fit = {str(i["id"]): _detail_fit(i, details, skip, shop_keys, None, vocab) or 0.0 for i in pend}
        # a page's own clause and its pointer often share one sentence ("Shisano's IP security camera here is a turret style with
        # us plug type; for input voltage, look to ..."): each clause is read on its own
        own = {str(i["id"]): len(said & {w for t in _record_sentences(i) for c in t.split(";") if not _ONWARD.search(c) and not _DIFFER_WORD.search(c)
                                         for w in _words(c)} | said & set(_words(str(i.get("title") or "")))) for i in pend}
        pend.sort(key=lambda i: (-fit[str(i["id"])], -own[str(i["id"])], i["price"] if _is_number(i.get("price")) else 1e18, str(i["id"])))
        self.mark("lh|" + tag)
        opened = self.open_rows(pend, 1)
        if not opened:
            return False
        self.trace.append({"kind": "fire", "set": "LH", "shop": shop, "opened": opened, "fit": round(fit[opened[0]], 2), "own": own[opened[0]], "said": len(said), "waiting": len(pend) - 1})
        self.chat.append({"role": "user", "content": "For the reference chain I read the property list of " + opened[0] + ", the page of "
                          + shop.title() + " that the pointer's words fit best (its own page was opened already); "
                          + str(len(pend) - 1) + " more of its pages are open as pages only (that step is used).\n" + self.status()})
        return True

    @staticmethod
    def seller_name(item: dict[str, Any]) -> str:
        """The shop a row is listed by, lower case, from its seller-like field (as chain_note reads it)."""
        for key, value in (item.get("record") or {}).items():
            if isinstance(value, str) and re.search(r"merchant|seller|shop|store|vendor|sold", str(key), re.I) \
                    and not re.search(r"note|text|desc", str(key), re.I) and 0 < len(value.split()) <= 3:
                return value.strip().lower()
        return ""

    def seen_rows(self, section: list[dict[str, Any]], needs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """A section's rows seen but never opened and not shown out of stock: those whose title, or a narrowing search that
        returned them, names a need's value first, then the cheapest."""
        widest = max([entry["total"] for entry in self.searches.values()] or [0])
        values = {w for n in needs if n.get("test") != "not" for v in n.get("values") or [] for w in _words(v) if len(w) > 2}
        evidence: dict[str, int] = {}
        for entry in self.searches.values():
            if widest and entry["total"] * 2 >= widest:
                continue  # the store did not narrow this search: its text is no evidence about the rows
            said = len(values & set(_words(entry["label"])))
            for row_id in entry["ids"]:
                evidence[row_id] = max(evidence.get(row_id, 0), said)
        rows = [i for i in section if i.get("facts") is None and i.get("stock") is not False]
        rows.sort(key=lambda i: (-max(len(values & set(_words(i.get("title") or ""))), evidence.get(i["id"], 0)),
                                 i["price"] if _is_number(i.get("price")) else 1e18))
        return rows

    def proof(self, item: dict[str, Any], needs: list[dict[str, Any]]) -> str:
        """FE: the opened listing's own facts that state its needs, quoted, for a note that names it as a fit: the driver does not
        trust a bare id it never saw opened (a guard opens listings itself)."""
        facts = item.get("facts") if isinstance(item.get("facts"), dict) else {}
        said = []
        for need, (state, note) in read_listing(needs, facts, item)["states"] if facts else []:
            if state != "met":
                continue
            words = _words(need.get("property") or "")
            key = next((k for k in facts if words and all(any(_same(w, h) for h in _words(k)) for w in words)), None)
            said.append(str(key) + " '" + str(facts[key])[:60] + "'" if key else str(need.get("property") or "") + " (" + str(note)[:60] + ")")
        return (" [its own facts: " + ", ".join(said[:4]) + "]") if said else ""

    def list_fits(self) -> str:
        """A shopping list about to be declined: per product, its opened fits (own facts meet its stated needs and the needs
        heard for it, in stock); a product without one gets up to 3 of its section's unopened rows opened first (need words in
        the title or in a narrowing search that returned them, then the cheapest). Names one fit per product when every product
        has one and the cheapest fits together are within the budget; else ""."""
        def covers(words: list[str], label: Any) -> bool:
            have = _words(str(label or ""))
            return bool(words) and all(any(_same(w, h) for h in have) for w in words)
        plans = self.list_plans()

        def fits_of(section: list[dict[str, Any]], needs: list[dict[str, Any]]) -> list[dict[str, Any]]:
            # proven fits first; else the best-checked: own facts contradict no need (fewest needs left unstated, then cheapest)
            usable = [i for i in section if i.get("facts") is not None and i.get("stock") is not False and not i.get("gone") and _is_number(i.get("price"))]
            readings = [(i, read_listing(needs, i["facts"], i)) for i in usable]
            proven = sorted([i for i, r in readings if r["verdict"] == "fits"], key=lambda i: i["price"])
            if proven:
                return proven
            return [i for _, i in sorted([(len(r.get("loose") or []), i) for i, r in readings if r["verdict"] == "open"], key=lambda p: (p[0], p[1]["price"]))]
        tool = self.role.get("details")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        calls = []
        if tool and target and not any(k != target for k in required):
            room = max(0, min(6, self.steps_left() - 4))
            for part, section, needs in plans:
                usable = [i for i in section if i.get("facts") is not None and i.get("stock") is not False and not i.get("gone") and _is_number(i.get("price"))]
                if any(read_listing(needs, i["facts"], i)["verdict"] == "fits" for i in usable) or len(calls) >= room:
                    continue   # a proven fit is known; a product with only unproven candidates still gets rows opened
                for item in self.seen_rows(section, needs)[: min(3, room - len(calls))]:
                    args, issues = conform({target: item["id"]}, schema)
                    if not issues:
                        calls.append((tool, args))
        if calls:
            self.execute(calls)
        chosen = []
        for part, section, needs in plans:
            fits = fits_of(section, needs)
            if not fits:
                self.trace.append({"kind": "fire", "set": "AS", "opened": len(calls), "missing": str(part.get("item"))})
                return ""
            chosen.append((part, fits[0]))
        if self.fit_groups():
            basket = self.fit_basket()   # FT: every part line of the host's series
            if basket is None:
                self.trace.append({"kind": "fire", "set": "AS", "opened": len(calls), "missing": "fit"})
                return ""
            chosen = [(part, basket[k]) for k, (part, _, _) in enumerate(plans)]
        total = sum(item["price"] * (_as_int(part.get("count")) or 1) for part, item in chosen)
        if self.ledger.ceiling is not None and _is_number(self.ledger.ceiling) and total > self.ledger.ceiling and tool and target \
                and self.steps_left() > 8 and self.mark("list-cheaper|" + str(self.ledger.version)) == 1:
            # every line whose own price exceeds what the budget leaves for it, dearest first (DA once took the dearest line only)
            for k in sorted(range(len(chosen)), key=lambda j: -chosen[j][1]["price"] * (_as_int(chosen[j][0].get("count")) or 1)):
                if total <= self.ledger.ceiling or self.steps_left() <= 8:
                    break
                part, item = chosen[k]
                count = _as_int(part.get("count")) or 1
                room_price = (self.ledger.ceiling - (total - item["price"] * count)) / count
                if not 0 < room_price < item["price"]:
                    continue
                # the line's section searched under what the budget leaves for it, cheapest unopened rows opened
                rows = [c for c in self.price_probe(dict(item, price=int(room_price) + 1)) if c.get("facts") is None][:3]
                if not rows:
                    rows = self.shelf_sweep(item, room_price)
                opening = []
                for row in rows:
                    args, issues = conform({target: row["id"]}, schema)
                    if not issues:
                        opening.append((tool, args))
                if opening:
                    self.execute(opening)
                self.trace.append({"kind": "fire", "set": "DA", "part": str(part.get("item")), "room": room_price, "opened": [r["id"] for r in rows]})
                section = [i for i in self.shelf.items.values() if self.in_part(i, part)]
                fits = fits_of(section, plans[k][2])
                if fits:
                    chosen[k] = (part, fits[0])
                    total = sum(it["price"] * (_as_int(p.get("count")) or 1) for p, it in chosen)
        if self.ledger.ceiling is not None and _is_number(self.ledger.ceiling) and total > self.ledger.ceiling:
            return ""
        self.trace.append({"kind": "fire", "set": "AS", "opened": len(calls), "fits": [item["id"] for _, item in chosen]})
        self.list_pick = [(item["id"], _as_int(part.get("count")) or 1) for part, item in chosen]
        return ("for every product of the list a listing in stock is either proven by its own facts or contradicts none of that product's needs (the best checked): "
                + "; ".join(str(_as_int(part.get("count")) or 1) + " x " + _label(item) + self.proof(item, plans[k][2]) + " for " + str(part.get("item"))
                            for k, (part, item) in enumerate(chosen))
                + " (together " + str(total) + ", within the budget). Unless one is not the requested kind, put them in the cart together at their quantities and check out instead of declining (a decline is almost never right while such listings exist).")

    def shelf_sweep(self, item: dict[str, Any], room: float) -> list[dict[str, Any]]:
        """With no price filter on the search: list the line's whole store section (a browse tool that takes a section name and a
        row offset, else the search tool with the section name), two pages in one reply, and return the section's unopened rows
        in stock priced within what the budget leaves for the line, cheapest first (at most six, steps allowing)."""
        label = str(item.get("section") or "").strip()
        if not label or self.steps_left() <= 12:
            return []
        known = {self.role.get(r) for r in ("details", "view", "message", "cart", "cart_view", "order", "decline")}
        choice = None
        for name, schema in self.schemas.items():
            props = (schema or {}).get("properties") or {}
            texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
            offsets = [k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v) and not _is_number(v.get("maximum"))]
            if name in known or len(props) != 2 or len(texts) != 1 or len(offsets) != 1:
                continue
            rank = 1 if name == self.role.get("search") else 0
            if choice is None or rank < choice[0]:
                choice = (rank, name, texts[0], offsets[0])
        if not choice:
            return []
        _, tool, text_key, page_key = choice
        size = max([e["size"] for e in self.searches.values() if _is_number(e.get("size"))] or [8])
        calls = []
        for page in range(2):
            args, issues = conform({text_key: label, page_key: int(page * size)}, self.schemas[tool])
            if not issues:
                calls.append((tool, args))
        if not calls:
            return []
        self.execute(calls)
        rows = [i for i in self.shelf.items.values() if i.get("section") == item.get("section") and i.get("facts") is None and i["id"] != item.get("id")
                and i.get("stock") is not False and not i.get("gone") and _is_number(i.get("price")) and i["price"] <= room + 1e-9]
        rows.sort(key=lambda i: (i["price"], str(i["id"])))
        self.trace.append({"kind": "fire", "set": "BS", "tool": tool, "section": label, "rows": len(rows)})
        return rows[:max(0, min(6, self.steps_left() - 8))]

    def open_seen(self, count: int) -> list[str]:
        """Open listings seen in searches but never opened: within budget, in stock, in the section that matches the request
        best; those whose title names a need's value first, then the cheapest. Returns the ids opened."""
        tool = self.role.get("details")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        if not tool or not target or any(k != target for k in required) or count <= 0:
            return []
        wanted = [w for w in _words(self.brief.get("item") or "") if len(w) > 2]

        def covered(label: Any) -> int:
            have = _words(label or "")
            return sum(1 for w in wanted if any(_same(w, h) for h in have))
        best = max([covered(i.get("section")) for i in self.shelf.items.values() if i.get("section")] or [0])
        sections = {i.get("section") for i in self.shelf.items.values() if i.get("section") and best and covered(i.get("section")) == best}
        values = {w for n in self.ledger.needs if n.get("test") != "not" for v in n.get("values") or [] for w in _words(v) if len(w) > 2}
        rows = [i for i in self.shelf.items.values() if i.get("facts") is None and self.shelf.affordable(i, self.ledger.ceiling, self.ledger.count)
                and (not sections or i.get("section") in sections)]
        widest = max([entry["total"] for entry in self.searches.values()] or [0])
        evidence: dict[str, int] = {}
        for entry in self.searches.values():
            if widest and entry["total"] * 2 >= widest:
                continue  # the store did not narrow this search: its text is no evidence about the rows
            said = len(values & set(_words(entry["label"])))
            for row_id in entry["ids"]:
                evidence[row_id] = max(evidence.get(row_id, 0), said)
        rows.sort(key=lambda i: (-max(len(values & set(_words(i.get("title") or ""))), evidence.get(i["id"], 0)),
                                 i["price"] if _is_number(i.get("price")) else 1e18))
        calls = []
        for item in rows[:count]:
            args, issues = conform({target: item["id"]}, schema)
            if not issues:
                calls.append((tool, args))
        if calls:
            self.execute(calls)
        return [str(args[target]) for _, args in calls]

    def sweep(self) -> str:
        if self.wire.left() < CLOSING_SECONDS + 30:
            return ""
        blocks, total = [], 0
        for item in sorted(self.shelf.items.values(), key=lambda i: i["price"] if _is_number(i.get("price")) else 1e18):
            if item.get("facts") is None or not self.shelf.affordable(item, self.ledger.ceiling, self.ledger.count):
                continue
            text = _dump(_trim(dict(item["record"], **{"property list": item["facts"]}), 160), 2500)
            if total + len(text) > 22000:
                break
            blocks.append(text)
            total += len(text)
        if not blocks:
            return ""
        prompt = SWEEP_PROMPT + "\n\nTHE REQUEST:\n" + self.request + "\n\nCURRENT NEEDS AND LIMITS (the shopper's own words decide):\n" + self.ledger.text(True)
        prompt += "\n\nOPENED LISTINGS (cheapest first):\n" + "\n".join(blocks)
        result = self.wire.ask_json(prompt, [REASONER, DRIVER], 900)
        self.trace.append({"kind": "sweep", "result": result})
        maybe = [str(x) for x in (result or {}).get("maybe") or [] if isinstance(x, (str, int))][:5]
        if not maybe:
            return ""
        return "An independent check thinks these listings may meet every need: " + ", ".join(maybe) + (" (" + str(result.get("why")) + ")" if result.get("why") else "") + ". Verify them and buy the right one; decline only if each truly fails."

    def decline_args(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        props = (self.schemas.get(name) or {}).get("properties") or {}
        cleaned = {}
        for key, value in args.items():
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            if isinstance(value, str) and "enum" not in spec:
                limit = spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else len(NEUTRAL_DECLINE)
                cleaned[key] = NEUTRAL_DECLINE[:limit]
            else:
                cleaned[key] = value
        return cleaned

    # ---------------- running calls
    def trim_chat(self) -> None:
        """Clip the long strings of old tool results (all but the last 6 messages) so the driver's prompt stays under the
        long-prompt price step; runs again only after the chat has grown by ~30k characters (~15k tokens) since the last trim."""
        size = sum(len(str(m.get("content") or "")) for m in self.chat)
        if size < getattr(self, "trimmed_size", 0) + 30000:
            return
        for m in self.chat[2:-6]:
            content = m.get("content")
            if m.get("role") == "tool" and isinstance(content, str) and len(content) > 900 and not content.startswith("[trimmed] "):
                m["content"] = "[trimmed] " + _shorten(content)
        self.trimmed_size = sum(len(str(m.get("content") or "")) for m in self.chat)
        self.trace.append({"kind": "fire", "set": "CT", "tokens": self.wire.driver_prompt, "chars": [size, self.trimmed_size]})

    def driver_tools(self) -> list[dict[str, Any]]:
        """The driver's tool list, fixed at its first use: a changed list voids the provider's prompt cache. The narrowing of
        offered_tools() reaches the driver through status() and the screen() guard instead."""
        if getattr(self, "fixed_tools", None) is None:
            message_tool = self.role.get("message") if self.message_args("?") is not None else None
            self.fixed_tools = [t for t in self.tools if t["function"]["name"] != message_tool]
        return self.fixed_tools

    def offered_tools(self) -> list[dict[str, Any]]:
        left = self.steps_left()
        role = self.role
        allowed: set[str] | None = None
        if left <= 1:
            allowed = {role.get("order"), role.get("decline")}
        elif left <= 3:
            allowed = {role.get("order"), role.get("decline"), role.get("cart"), role.get("view") or self.stock_tool()}
        elif (left <= 7 or self.wire.work_spent() >= self.spend_line(SPEND_HARD)) and (self.orders_tried == 0 or self.refused > 0):
            allowed = {role.get("order"), role.get("decline"), role.get("cart"), role.get("cart_view"), role.get("view"), role.get("details")}
        tools = self.tools
        if allowed is not None and left > 3 and not self.shelf.items and role.get("search"):
            allowed.add(role.get("search"))   # SN: nothing found yet, so searching stays open
        if allowed is not None and role.get("order") and role.get("cart"):
            narrowed = [t for t in self.tools if t["function"]["name"] in allowed]
            tools = narrowed or tools
        message_tool = role.get("message") if self.message_args("?") is not None else None   # a tool the planner cannot fill stays with the driver
        return [t for t in tools if t["function"]["name"] != message_tool]

    def execute(self, actions: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, str, Any]]:
        """Send actions to the store; file results; return (call id, text, observation) in order."""
        out: list[tuple[str, str, Any]] = []
        role_of: dict[str, str] = {}
        for job, name in self.role.items():
            if name and (name not in role_of or job in ("details", "cart", "search")):
                role_of[name] = job   # one tool in two roles keeps the richer one (details over view, cart over cart_view)
        cap = self.policy.get("max_calls_per_turn")
        chunk_size = cap if isinstance(cap, int) and cap > 0 else 16
        pending = list(actions)
        while pending:
            chunk, pending = pending[:chunk_size], pending[chunk_size:]
            if self.wire.finished or self.wire.broken or self.steps_left() <= 0:
                out.extend(("x", "Not run: the episode is over or no steps are left.", None) for _ in chunk)
                continue
            chunk = [(n, self.joined_units(n, a)) for n, a in chunk[: self.steps_left()]]
            sent = [{"name": n, "args": a} for n, a in chunk]
            before = self.wire.step
            reply = self.wire.act(sent)
            if reply is not None and any(role_of.get(n) in ("search", "details", "view", "cart", "cart_view") for n, _ in chunk):
                self.work_step = self.wire.step   # XD: the last step that searched, opened or changed the cart
            self.trace.append({"kind": "store", "step": before + 1, "actions": sent, "reply": reply})
            if reply is None:
                out.extend(("x", "The call failed (network or server). Try again.", None) for _ in chunk)
                continue
            observations = reply.get("observations") if isinstance(reply.get("observations"), list) else [reply.get("observation")]
            for index, (name, args) in enumerate(chunk):
                if index >= len(observations):
                    out.append(("x", "Not run: the episode finished before this call.", None))
                    continue
                observation = observations[index]
                job = self.final_kind(name, args) or role_of.get(name, "")
                filed = self.file_result(name, job, args, observation, index == len(observations) - 1 and bool(reply.get("finished")))
                self.recent_text = {k: v for k, v in self.recent_text.items() if k in self.recent}
                if name + _dump(args) in self.recent:
                    self.recent_text[name + _dump(args)] = filed   # DS
                if job in ("cart", "message", "order", "decline") or self.voice(observation, False):
                    self.read_memory, self.read_answered = {}, set()   # RM: something may have changed
                elif job in ("details", "view") and getattr(self, "last_touched", None):
                    self.read_memory[name + _dump(args)] = (self.wire.step, filed)
                out.append(("x", filed, observation))
        return out

    def file_result(self, name: str, job: str, args: dict[str, Any], observation: Any, ended: bool) -> str:
        notes: list[str] = []
        if job == "message":
            self.questions_sent += 1
        if job == "order":
            self.orders_tried += 1
        signature = name + _dump(args)
        self.recent = [] if job in ("cart", "message", "order", "decline") else (self.recent + [signature])[-2:]
        touched = self.shelf.take(observation, job if job in ("search", "details", "view", "cart_view") else "search" if not job else "other", self.wire.step)
        self.last_touched = touched   # RM: only a page the shelf took is remembered
        if job in ("details", "view") and touched:
            self.read_by.setdefault(job, set()).update(touched)
        if observation in ({}, [], None, "") and job not in ("message", "order", "decline"):
            notes.append("This call returned nothing: it failed (a limit or a bad argument, such as too many ids at once, a stale page token or an unknown id); it does not mean the store has no such listings.")
        try:
            if touched and not getattr(self, "parts", []):
                seen_sections = {str(i.get("section")).strip().lower() for i in self.shelf.items.values() if i.get("section")}
                kind = [n for n in self.ledger.needs if n.get("test") == "is" and n.get("values")
                        and all(str(v).strip().lower() in seen_sections for v in n["values"])]
                if kind and self.mark("kind-need") == 1:
                    self.ledger.needs = [n for n in self.ledger.needs if n not in kind]
                    self.ledger.version += 1
                    if not str(self.brief.get("item") or "").strip():
                        self.brief["item"] = str(kind[0]["values"][0])
                    self.trace.append({"kind": "fire", "set": "CO", "needs": [need_text(n) for n in kind]})
                    notes.append("'" + str(kind[0]["values"][0]) + "' is a section of this store, the kind of product the request asks for, not a "
                                 + str(kind[0].get("property") or "property") + ": that need is dropped, and no listing has to show it.")
        except Exception as exc:   # a helper's failure must not end the episode
            self.trace.append({"kind": "fire", "set": "CS", "where": "CO", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
        stocked = self.stock_ids(observation) if job not in ("cart", "message", "order", "decline") else set()
        if stocked:
            self.stock_tools.add(name)
            self.read_after_cart.update(stocked)
        if job not in ("search", "details", "view", "cart_view", "cart", "message", "order", "decline"):
            opened = [i for i in self.shelf.take_detail_rows(observation, self.wire.step) if i not in touched]
            if opened:
                self.trace.append({"kind": "fire", "set": "AQ", "tool": name, "opened": len(opened)})
                touched = touched + opened
        details_tool = self.role.get("details")
        if job == "view" and details_tool and details_tool != name and any((self.shelf.items.get(i) or {}).get("facts") is None for i in touched) \
                and self.mark("view-note") == 1:
            notes.append(name + " shows a listing's price, stock and notes, not its property list. Open candidates with " + details_tool
                         + " (their facts decide every need); keep " + name + " for re-checking a carted line's stock and price just before checkout.")
        if job not in ("message", "order", "decline", "cart") and isinstance(args, dict) and self._ids_in(observation, 0):
            labels = {str(i.get("section")).strip().lower() for i in self.shelf.items.values() if i.get("section")}
            for key, value in args.items():
                if isinstance(value, str) and value.strip().lower() in labels:
                    self.section_slots = dict(getattr(self, "section_slots", {}), **{name: key})   # learned by value, any tool
        if job == "search":
            shelf_key = next((k for k, v in args.items() if isinstance(v, str) and v.strip() and re.search(r"shelf|section|categor|department|aisle", k, re.I)), None)
            if shelf_key and self._ids_in(observation, 0):
                self.section_searches[args[shelf_key].strip().lower()] = dict(args)
            try:
                mode_note = self.learn_search_mode(name, args, observation)
            except Exception as exc:   # a helper's failure must not end the episode
                mode_note = ""
                self.trace.append({"kind": "fire", "set": "CS", "where": "SM", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
            self.note_search(args, observation)
            try:
                named = {str(p.get("item") or "").strip().lower() for p in getattr(self, "parts", [])} | {str(self.brief.get("item") or "").strip().lower()}
                odd = [k for k, v in args.items() if isinstance(v, str) and v.strip().lower() in named - {""} and k != shelf_key]
                if odd and not self._ids_in(observation, 0) and self.mark("section-as|" + odd[0]) == 1:
                    notes.append("'" + args[odd[0]].strip() + "' is the store section the request names (\"... from " + args[odd[0]].strip() + "\"), not a value of "
                                 + odd[0] + ": the request sets no brand, maker or shop, so no listing needs one.")
            except Exception as exc:   # a helper's failure must not end the episode
                self.trace.append({"kind": "fire", "set": "CS", "where": "CN", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
            drift = self.search_drift(name, args, observation)
            if drift:
                notes.append(drift)
            try:
                joined = self.joined_note(name, args)
            except Exception as exc:   # a helper's failure must not end the episode
                joined = ""
                self.trace.append({"kind": "fire", "set": "CS", "where": "JN", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
            if joined:
                notes.append(joined)
            if mode_note:
                notes.append(mode_note)
        if job == "cart":
            self.cart_touched = True
            self.cart_step = self.wire.step
            self.read_after_cart = set()
            self.reread_tried = set()
            self.bag_fresh = False
            item_id = self.item_in_args(name, args)
            if item_id:
                if self.cart_direction(name, args) == "add":
                    self.cart = [i for i in self.cart if i != item_id] + [item_id]
                    self.adds[item_id] = self.adds.get(item_id, 0) + 1
                else:
                    self.cart = [i for i in self.cart if i != item_id]
                    self.adds[item_id] = 0
                    self.removed[item_id] = self.removed.get(item_id, 0) + 1
            listed = list(dict.fromkeys(self._ids_in(observation, 0)))
            counts = self.bag_counts(observation, set(listed)) if listed else {}
            if listed and set(counts) == set(listed) and set(self.cart) <= set(listed) and (not item_id or (item_id in listed) == (item_id in self.cart)):
                # the result lists the whole cart, a count on every line: take it as the store shows it
                self.cart = [i for i in self.cart if i in listed] + [i for i in listed if i not in self.cart]
                self.bag = counts
                self.bag_fresh = True
            self.order_bounced = False
        if job == "cart_view":
            shown = list(dict.fromkeys(self._ids_in(observation, 0)))   # the store's own line order
            seen = set(shown)
            self.cart = [i for i in self.cart if i in seen] + [i for i in shown if i not in self.cart]
            self.bag = self.bag_counts(observation, seen)
            self.bag_fresh = True
        if isinstance(observation, dict):
            if observation.get("error") == "budget_exceeded":
                if job == "message":
                    self.question_limit = self.questions_sent
                    self.asking = False
                notes.append("That budget is used up; do not call this tool again.")
        if (isinstance(observation, dict) and "refusal" in observation) or (job in ("order", "decline") and not ended and self.voice(observation, False)):
            self.refused += 1
            self.last_refusal = {"job": job, "step": self.wire.step, "obs": observation, "sigs": {_need_sig(n) for n in self.ledger.needs}}
            self.order_bounced = False   # the next checkout is checked again, stock re-read included
            self.read_after_cart = set()
            self.reread_tried = set()
            if job == "decline":
                notes.append("THE DECLINE WAS NOT ACCEPTED and the episode goes on. Apply every change the shopper or the store states in this result; if a listing fits now, buy it, otherwise decline again.")
            else:
                notes.append("THE CHECKOUT WAS REFUSED and the episode goes on; the cart is unchanged. Apply every change in the refusal, fix the cart (remove a line that no longer qualifies, such as a sold-out item), re-check the new pick's price and stock, and check out again.")
                backups = self.backups()
                if backups:
                    notes.append("Verified backups (their facts state every need, within budget, not shown out of stock when last seen): "
                                 + "; ".join(_label(i) for i in backups[:3]) + ". If the cart line no longer qualifies, swap in the first that still does.")
        if job == "message" and not getattr(self, "planner_sending", False):
            heard = self.voice(observation, True)   # a question the driver sent itself (the planner cannot fill this message tool)
            meaning = self.hear(heard) if heard else ""
            if meaning:
                notes.append(meaning)
        if job != "message":
            lines = self.voice(observation, False)
            if lines:
                notes.append("This result carries a message from the shopper or the store, a real change that binds from now on (the store relays"
                             " the shopper's changes inside its results): " + " | ".join(lines[:4]))
                meaning = self.hear(lines)
                if meaning:
                    notes.append(meaning)
        text = _dump(_trim(observation))
        if ended:
            text += " [the episode is finished]"
        if notes:
            text += "\n[note] " + " ".join(notes)
        return text

    def foreign_token(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """A search sent with a page token that only other searches returned, on words and filters never fetched without one:
        the token would skip the first rows of these words, so it is left out and the first page runs (with a note why)."""
        required = set((self.schemas.get(name) or {}).get("required") or [])
        labels = {str(i.get("section")).strip().lower() for i in self.shelf.items.values() if i.get("section")}
        moved = [k for k, v in args.items() if isinstance(v, str) and v in self.token_sources and k not in required and v.strip().lower() not in labels]
        if not moved:
            return args, ""
        rest = {k: v for k, v in args.items() if k not in moved}
        query = {k: v for k, v in sorted(rest.items()) if not (isinstance(v, (str, int)) and not isinstance(v, bool) and v in self.page_values)}
        key = _dump(query).lower()
        if not query or key in {k.lower() for k in self.searches} or any(key in self.token_sources[args[k]] for k in moved):
            return args, ""
        self.trace.append({"kind": "fire", "set": "PT", "dropped": {k: args[k] for k in moved}, "query": _dump(query, 200)})
        return rest, ("this search ran without " + ", ".join(k + " '" + str(args[k]) + "'" for k in moved) + ": that page token came from a "
                      "different search (other words or filters), and a token only continues the search that returned it; with these words "
                      "it would have skipped their first rows. These are the first rows of these words; page on with the token this search returns.")

    def price_scale_note(self, name: str, args: dict[str, Any]) -> str:
        """A price ceiling far below every listing price seen and below the budget by the same factor: likely converted into a
        larger money unit than the store's."""
        props = (self.schemas.get(name) or {}).get("properties") or {}
        paging = {k for k, v in args.items() if v in self.page_values}
        paging |= {k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v) and _is_number(v.get("maximum")) and v["maximum"] <= 50}
        numeric = [k for k, v in props.items() if isinstance(v, dict) and {"number", "integer"} & set(_kinds(v)) and "enum" not in v and k not in paging]
        if len(numeric) != 1 or not _is_number(args.get(numeric[0])) or args[numeric[0]] <= 0:
            return ""
        cap = float(args[numeric[0]])
        prices = sorted(i["price"] for i in self.shelf.items.values() if _is_number(i.get("price")) and i["price"] > 0)
        budget = self.ledger.ceiling if _is_number(self.ledger.ceiling) else None
        if len(prices) < 5 or not budget or cap * 50 > prices[len(prices) // 10] or cap * 50 > budget:
            return ""
        return ("this price filter (" + numeric[0] + " " + str(args[numeric[0]]) + ") is far below every listing price seen (the cheapest about "
                + str(prices[0]) + ") and below the budget of " + str(budget) + " by the same factor: the listings' prices are in the store's own unit"
                + (" (" + self.ledger.money + ")" if self.ledger.money else "") + ", so give the filter in that unit (for example " + str(int(round(cap * 100)))
                + "). Sending the same call again goes ahead.")

    def price_probe(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """Send the agent's own search of the item's section again with the price ceiling just under the item's unit price.
        Returns the section's cheaper, affordable rows on the shelf afterwards, cheapest first (empty when it cannot be done)."""
        tool = self.role.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        price = item.get("price")
        base = self.section_searches.get(str(item.get("section") or "").strip().lower())
        paging = {k for k, v in base.items() if v in self.page_values} if base else set()
        paging |= {k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v) and _is_number(v.get("maximum")) and v["maximum"] <= 50}
        numeric = [k for k, v in props.items() if isinstance(v, dict) and {"number", "integer"} & set(_kinds(v)) and "enum" not in v and k not in paging]
        numeric = [k for k in numeric if self.price_cap_arg(tool or "", k)]
        cap_key = numeric[0] if len(numeric) == 1 else None   # one free number on the search that its text ties to a price: the ceiling
        if not _is_number(price) or price <= 0:
            return []
        if not tool or not cap_key or base is None:
            return self.section_price_probe(item)
        args = {k: v for k, v in base.items() if k not in paging}
        spec = props.get(cap_key) or {}
        below = price - 1 if float(price).is_integer() and price >= 2 else price * 0.99
        args[cap_key] = int(below) if "integer" in _kinds(spec) else below
        args, issues = conform(args, schema)
        if issues:
            return []
        self.trace.append({"kind": "fire", "set": "BR", "item": item.get("id"), "args": args})
        self.execute([(tool, args)])
        return self.shelf.cheaper_rows(item, self.ledger.ceiling, self.ledger.count)

    def price_cap_arg(self, tool: str, key: str) -> bool:
        """A number argument that caps the price: its name, its own text or the tool's text about it speaks of a price or an upper
        bound, and nothing there speaks of paging (a row offset or page number sent as a price ceiling only shows an empty page)."""
        fn = next((t["function"] for t in self.tools if t["function"]["name"] == tool), {})
        spec = ((fn.get("parameters") or {}).get("properties") or {}).get(key)
        name = re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " ")
        about = " ".join([name, str((spec or {}).get("description") or "") if isinstance(spec, dict) else ""]
                         + re.findall(r"[^.;]*\b" + re.escape(key) + r"\b[^.;]*", str(fn.get("description") or "")))
        if re.search(r"page|start|begin|from|offset|skip|after|cursor|next|token|row", about, re.I):
            return False
        return bool(re.search(r"price|cost|ceiling|budget|max|cheap|under|below|limit|\bcap\b|upper", about, re.I))

    def section_price_probe(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        """price_probe through a tool whose section argument was learned by value (see PP): that section, the price ceiling just
        under the item's unit price. Returns the section's cheaper, affordable rows on the shelf afterwards, cheapest first."""
        price, section = item.get("price"), str(item.get("section") or "").strip()
        if not _is_number(price) or price <= 0 or not section:
            return []
        for name, slot in sorted((getattr(self, "section_slots", {}) or {}).items()):
            schema = self.schemas.get(name) or {}
            props = schema.get("properties") or {}
            numeric = [k for k, v in props.items() if isinstance(v, dict) and {"number", "integer"} & set(_kinds(v)) and "enum" not in v
                       and not (_is_number(v.get("maximum")) and v["maximum"] <= 50) and self.price_cap_arg(name, k)]
            if len(numeric) != 1 or any(k not in (slot, numeric[0]) for k in schema.get("required") or []):
                continue
            spec = props.get(numeric[0]) or {}
            below = price - 1 if float(price).is_integer() and price >= 2 else price * 0.99
            args, issues = conform({slot: section, numeric[0]: int(below) if "integer" in _kinds(spec) else below}, schema)
            if issues:
                continue
            self.trace.append({"kind": "fire", "set": "PP", "item": item.get("id"), "tool": name, "args": args})
            self.execute([(name, args)])
            return self.shelf.cheaper_rows(item, self.ledger.ceiling, self.ledger.count)
        return []

    def stock_ids(self, observation: Any) -> set[str]:
        """Listings whose record in this result carries the store's stock field."""
        key = self.shelf.stock_key()
        if not key:
            return set()
        found: set[str] = set()
        records = list(self.shelf._records(observation, 0))
        for rows in self.shelf._row_lists(observation, 0):
            records.extend(r for r in rows if isinstance(r, dict))
        for record in records:
            if isinstance(record.get(key), bool):
                item_id = self.shelf._row_id(record)
                if item_id and item_id in self.shelf.items:
                    found.add(item_id)
        return found

    def stock_tool(self) -> str | None:
        """The tool to re-check a listing's stock with: one whose results have shown stock (the view role first), else the
        view role, else the details role."""
        for tool in (self.role.get("view"), self.role.get("details")):
            if tool and tool in self.stock_tools:
                return tool
        return self.role.get("view") or self.role.get("details")

    def bag_counts(self, observation: Any, ids: set[str]) -> dict[str, int]:
        """Units per listing in a cart view: in each cart row, the one whole-number field that is not the price."""
        price = self.shelf.price_key()
        counts: dict[str, int] = {}
        for rows in self.shelf._row_lists(observation, 0):
            for row in rows:
                row_id = next((v for v in row.values() if isinstance(v, str) and v in ids), None)
                numbers = [v for k, v in row.items() if k != price and isinstance(v, int) and not isinstance(v, bool) and v >= 0]
                if row_id and len(numbers) == 1:
                    counts[row_id] = numbers[0]
        return counts

    def _ids_in(self, value: Any, depth: int) -> list[str]:
        found: list[str] = []
        if depth > 5:
            return found
        if isinstance(value, dict):
            for element in value.values():
                if isinstance(element, str) and element in self.shelf.items:
                    found.append(element)
                elif isinstance(element, (dict, list)):
                    found.extend(self._ids_in(element, depth + 1))
        elif isinstance(value, list):
            for element in value:
                found.extend(self._ids_in(element, depth + 1))
        return found

    def multi_reader(self) -> tuple[str, str, int] | None:
        """A tool with no role that reads several listings in one call: its one required argument is a list of strings and
        its description speaks of listings or ids. Returns (tool, argument, most ids per call)."""
        taken = set(self.role.values())
        for tool in self.tools:
            fn = tool["function"]
            name = fn.get("name")
            schema = self.schemas.get(name) or {}
            props = schema.get("properties") or {}
            required = [k for k in schema.get("required") or [] if k in props]
            if name in taken or len(required) != 1 or any(k != required[0] for k in props if k in (schema.get("required") or [])):
                continue
            spec = props[required[0]] if isinstance(props[required[0]], dict) else {}
            items = spec.get("items") if isinstance(spec.get("items"), dict) else {}
            text = str(fn.get("description") or "")
            if "array" in _kinds(spec) and "string" in _kinds(items) and re.search(r"listing|item|product|\bids?\b", text, re.I):
                most = re.search(r"up to (\d+)", text)
                cap = spec.get("maxItems") if isinstance(spec.get("maxItems"), int) else int(most.group(1)) if most else 6
                return name, required[0], max(2, min(int(cap), 10))
        return None

    def bare_rows(self, observation: Any, ids: set[str]) -> bool:
        """MV: True when a multi-listing read returned rows for these ids and none of them carries a property list."""
        seen = [row for found in self.shelf._row_lists(observation, 0) for row in found if self.shelf._row_id(row) in ids]
        return bool(seen) and all(self.shelf._facts_of(row) is None for row in seen)

    def merged_reads(self, ready: list[tuple[str, str, dict[str, Any]]], results: dict[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
        """Send the turn's details calls (two or more) as multi-listing reads; fill their results; return the other calls."""
        details = self.role.get("details")
        group = [r for r in ready if r[1] == details]
        many = self.multi_reader() if details and len(group) >= 2 and not self.multi_bad else None
        schema = self.schemas.get(details or "") or {}
        required = list(schema.get("required") or [])
        slot = required[0] if len(required) == 1 else None
        if not many or not slot or any(set(a) - {slot} or not isinstance(a.get(slot), str) for _, _, a in group):
            return ready
        tool, key, cap = many
        ids = list(dict.fromkeys(str(a[slot]) for _, _, a in group))
        rows: dict[str, Any] = {}
        notes: list[str] = []
        for start in range(0, len(ids), cap):
            chunk = ids[start:start + cap]
            args, issues = conform({key: chunk}, self.schemas[tool])
            if issues:
                return ready
            _, text, observation = self.execute([(tool, args)])[0]
            if "\n[note] " in text:
                notes.append(text.split("\n[note] ", 1)[1])   # a store notice or guard note of the merged call reaches the driver
            if self.bare_rows(observation, set(chunk)):
                # MV: its rows carry no property list, so they do not stand in for details reads: the calls go out alone
                self.multi_bad = True
                self.trace.append({"kind": "fire", "set": "MV", "tool": tool, "where": "MR", "ids": chunk})
                break
            for found in self.shelf._row_lists(observation, 0):
                for row in found:
                    row_id = self.shelf._row_id(row)
                    if row_id in chunk and self.shelf._facts_of(row) is not None:
                        rows[row_id] = row
                        self.shelf.take(row, "details", self.wire.step)
        self.trace.append({"kind": "fire", "set": "MR", "tool": tool, "ids": ids, "read": len(rows)})
        alone = []
        for call in group:
            call_id, _, args = call
            row = rows.get(str(args[slot]))
            if row is not None:
                results[call_id] = _dump(_trim(row)) + " [read together with the turn's other listings in one call of " + tool + "]"
            elif self.multi_bad:
                alone.append(call)
            else:
                results[call_id] = "Not returned when read together with the turn's other listings (" + tool + "); open it alone if you still need it."
        first = next((call_id for call_id, _, _ in group if call_id in results), None)
        if notes and first:
            results[first] += "\n[note] " + " ".join(notes)
        return [r for r in ready if r[1] != details] + alone

    def merged_views(self, ready: list[tuple[str, str, dict[str, Any]]], results: dict[str, str]) -> list[tuple[str, str, dict[str, Any]]]:
        """MV2: the turn's view calls (two or more) go out as one multi-listing read when its rows show the same pages; their
        results are filled; returns the other calls (and any view the read did not show)."""
        view = self.role.get("view")
        schema = self.schemas.get(view or "") or {}
        required = list(schema.get("required") or [])
        group = [r for r in ready if r[1] == view and view != self.role.get("details") and len(required) == 1 and set(r[2]) == {required[0]}
                 and isinstance(r[2].get(required[0]), str)]
        if len(group) < 2:
            return ready
        rows = [self.shelf.items.get(str(a[required[0]])) or {"id": str(a[required[0]])} for _, _, a in group]
        shown = set(self.view_many(rows) or [])
        if not shown:
            return ready
        left = []
        for call in group:
            call_id, name, args = call
            memo = self.read_memory.get(name + _dump(args))
            if str(args[required[0]]) in shown and memo:
                results[call_id] = memo[1] + " [shown together with the turn's other pages in one call]"
            else:
                left.append(call)
        return [r for r in ready if r not in group] + left

    def handle(self, calls: list[tuple[str, str, Any]]) -> list[tuple[str, str]]:
        ready, refused = self.screen(calls)
        results = dict(refused)
        if ready:
            ready = self.merged_reads(ready, results)
            ready = self.merged_views(ready, results)
            done = self.execute([(n, a) for _, n, a in ready]) if ready else []
            for (call_id, _, _), (_, text, _) in zip(ready, done):
                results[call_id] = text + ("\n[note] " + self.paging_notes[call_id] if call_id in self.paging_notes else "")
            self.paging_notes.clear()
            try:
                note = self.band_probe() or self.sweep_pages() or self.early_probe()
            except Exception as exc:   # a helper's failure must not end the episode
                self.trace.append({"kind": "fire", "set": "CS", "where": "probes", "error": (type(exc).__name__ + ": " + str(exc))[:300]})
                note = ""
            if note and ready:
                results[ready[-1][0]] = results.get(ready[-1][0], "") + "\n[note] " + note
        return [(call_id, results.get(call_id, GUARD + "skipped.")) for call_id, _, _ in calls]

    def band_probe(self) -> str:
        """Cheapest single-product requests: lower the section search's price ceiling until all listings at or under it fit on
        one page (at most 3 searches), then name that band (see CF)."""
        if self.brief.get("pick") != "cheapest" or getattr(self, "parts", []) or self.steps_left() <= 20 or self.wire.left() < CLOSING_SECONDS + 60:
            return ""
        section = str(self.brief.get("item") or "").strip().lower()
        base = self.section_searches.get(section)
        if base is None and len(self.section_searches) == 1:
            section, base = next(iter(self.section_searches.items()))
        tool = self.role.get("search")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        if not tool:
            return ""
        if base is None:
            shelf_arg = next((k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and re.search(r"shelf|section|categor|department|aisle", k, re.I)), None)
            if not shelf_arg or not self.searches or self.mark("band-ask|" + str(self.ledger.version)) != 1:
                return ""
            return ("the cheapest fitting listing is wanted, and this store lists search rows in no price order, so a text search cannot show which "
                    "listings are cheapest: send one search of the " + str(self.brief.get("item") or "requested") + " section itself (its " + shelf_arg
                    + " argument) with the budget as its price ceiling; its cheapest listings will then be listed for you.")
        paging = {k for k, v in base.items() if v in self.page_values}
        paging |= {k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v) and _is_number(v.get("maximum")) and v["maximum"] <= 50}
        numeric = [k for k, v in props.items() if isinstance(v, dict) and {"number", "integer"} & set(_kinds(v)) and "enum" not in v and k not in paging]
        if len(numeric) != 1:
            return ""
        cap_key = numeric[0]
        paging.discard(cap_key)
        integer = "integer" in _kinds(props.get(cap_key) or {})

        def entry_of(args: dict[str, Any]) -> dict[str, Any] | None:
            query = {k: v for k, v in sorted(args.items()) if not (isinstance(v, (str, int)) and not isinstance(v, bool) and v in self.page_values)}
            return self.searches.get(_dump(query))

        def section_prices(top: float) -> list[float]:
            return sorted({i["price"] for i in self.shelf.items.values() if str(i.get("section") or "").strip().lower() == section
                           and _is_number(i.get("price")) and 0 < i["price"] <= top})

        query = {k: v for k, v in base.items() if k not in paging}
        entry = entry_of(base) or entry_of(query)   # kept under the arguments it was sent with (a first page number included)
        size = entry["size"] if entry else 0
        n_hi = entry["total"] if entry else 0
        if size <= 0 or n_hi <= size:
            return ""   # the section search already shows all its rows
        shown = entry["prices"][:size]
        if len(shown) >= 3 and all(a <= b for a, b in zip(shown, shown[1:])):
            return ""   # this store lists rows cheapest first: the first page is the cheapest band already
        hi = base.get(cap_key)
        if not _is_number(hi) and _is_number(self.ledger.ceiling):
            hi = self.ledger.ceiling / max(1, self.ledger.count or 1)
        if not _is_number(hi) or hi <= 2:
            return ""
        if self.mark("band|" + section + "|" + str(self.ledger.version)) != 1:
            return ""
        whole = all(float(i["price"]).is_integer() for i in self.shelf.items.values() if _is_number(i.get("price")))
        lo, n_lo, band_cap, band_n = 0.0, 0, None, None
        target = size - 1
        for _ in range(3):
            # the count of rows under a ceiling grows with the ceiling; rows come in no price order, so the prices seen are a
            # sample of the section: estimate the count at each seen price and take the highest one expected to fit a page
            pool = [p for p in section_prices(hi) if lo < p < hi]
            c = None
            for rank, p in enumerate(pool, 1):
                if n_lo + (n_hi - n_lo) * rank / (len(pool) + 1) <= target:
                    c = p
            if c is None and pool:
                c = pool[0]   # a listing costs this much, so this ceiling shows at least one row
            if c is None:
                c = lo + (hi - lo) * (target - n_lo) / max(1, n_hi - n_lo)
            c = int(c) if (integer or whole) else round(c, 2)
            while c in self.page_values and c - 1 > lo:
                c -= 1
            if c <= lo or c >= hi:
                break
            args = dict(query)
            args[cap_key] = c
            args, issues = conform(args, schema)
            if issues:
                break
            self.trace.append({"kind": "fire", "set": "CF", "args": args})
            self.execute([(tool, args)])
            got = entry_of(args)
            n = got["total"] if got else 0
            if n <= max(size, got["size"] if got else 0):
                lo, n_lo = c, n
                if n > 0:
                    band_cap, band_n = c, n
                if n >= size - 3:
                    break
            else:
                hi, n_hi = c, n
        top = band_cap if band_cap is not None else hi
        rows = sorted([i for i in self.shelf.items.values() if str(i.get("section") or "").strip().lower() == section and _is_number(i.get("price"))
                       and i["price"] <= top and i.get("stock") is not False], key=lambda i: i["price"])
        if not rows:
            return ""
        said = ("all " + str(band_n) + " listings of the section at or under " + str(band_cap) + " (the cheapest band) are") if band_cap is not None else \
            ("the section has " + str(n_hi) + " listings at or under " + str(hi) + "; those shown so far are")
        return ("the cheapest fitting listing is wanted, and this store lists search rows in no price order, so the section search was sent "
                "again with a lower price ceiling: " + said + ": " + "; ".join(_label(i) + " '" + str(i.get("title") or "")[:60] + "'" for i in rows[:11])
                + ". The cheapest of them whose own facts state every need is the cheapest fit" + (" of the whole section" if band_cap is not None else " among them")
                + ": open their property lists" + (" with " + self.role["details"] if self.role.get("details") else "")
                + " cheapest first (in one reply), skipping only a listing whose title clearly contradicts a need.")

    def sweep_pages(self) -> str:
        """Cheapest single-product requests: read a search for a stated value to its end and name its cheapest rows (see CH)."""
        if self.brief.get("pick") != "cheapest" or self.steps_left() <= 16 or self.wire.left() < CLOSING_SECONDS + 60:
            return ""
        used = self.guard_marks.get("sweep-pages", 0)
        allowed = 8 if getattr(self, "parts", []) else 6
        tool = self.role.get("search")
        props = (self.schemas.get(tool or "") or {}).get("properties") or {}
        pages = [k for k, v in props.items() if isinstance(v, dict) and "integer" in _kinds(v) and _is_number(v.get("maximum")) and v["maximum"] <= 50]
        token_arg = getattr(self, "token_arg", None) if len(pages) != 1 else None   # CT: a store paging by token
        if len(pages) != 1 and not token_arg:
            required = set((self.schemas.get(tool or "") or {}).get("required") or [])
            named = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and k not in required
                     and re.search(r"next|page|cursor|token|onward|continu", k, re.I)]
            token_arg = named[0] if len(named) == 1 else None
        if not tool or (len(pages) != 1 and not token_arg) or used >= allowed:
            return ""
        page_key = pages[0] if len(pages) == 1 else token_arg
        first = int(props[page_key]["minimum"]) if not token_arg and _is_number(props[page_key].get("minimum")) else 1
        last = int(props[page_key]["maximum"]) if not token_arg else 10 ** 6
        groups = [(str(p.get("item") or "").strip().lower(), _as_int(p.get("count")) or 1, list(p.get("needs") or [])) for p in self.parts] \
            if getattr(self, "parts", []) else [(str(self.brief.get("item") or "").strip().lower(), self.ledger.count, self.ledger.needs)]
        groups = [(sec, cnt, [w for w in (_words(str(v)) for n in needs if isinstance(n, dict) and n.get("test") == "is" for v in n.get("values") or []) if w])
                  for sec, cnt, needs in groups]
        sections = {sec for sec, _, _ in groups}
        for key in reversed(list(self.searches)):
            entry = self.searches[key]
            size, total = entry["size"], entry["total"]
            if size <= 0 or total <= 2 or total > size * (last - first + 1):
                continue
            try:
                query = json.loads(key)
            except ValueError:
                continue
            if not isinstance(query, dict):
                continue
            text = [w for k in self.text_slots(tool) if isinstance(query.get(k), str) and query[k].strip().lower() not in sections for w in _words(query[k])]
            match = [(sec, cnt) for sec, cnt, wanted in groups if text and any(all(any(_same(w, t) for t in text) for w in words) for words in wanted)]
            if len(match) != 1:
                continue
            section, count = match[0]
            shown = entry["prices"][:size]
            if len(shown) >= 3 and all(a <= b for a, b in zip(shown, shown[1:])):
                continue   # rows come cheapest first: page 1 holds the cheapest already
            if len(entry["ids"]) < total and getattr(self, "page_mode", None) != "number" and not (token_arg and entry.get("onward")):
                continue   # the store pages by row offset or by token (or it is not known yet): no pages are guessed
            if self.mark("sweep|" + key) != 1:
                continue
            plain = {k: v for k, v in query.items() if k != page_key}
            start = first + -(-len(entry["ids"]) // size)
            todo = list(range(start, first + -(-total // size)))[: max(0, min(self.steps_left() - 14, allowed - used))]
            if todo and token_arg:
                # CT: each page's token comes from the page before it, so the pages go one call at a time, each by the token
                # the store itself named
                read = []
                for _ in todo:
                    onward = entry.get("onward")
                    if not onward or self.steps_left() <= 14:
                        break
                    read.append(onward)
                    self.guard_marks["sweep-pages"] = used + len(read)
                    self.execute([(tool, dict(plain, **{page_key: onward}))])
                    if entry.get("onward") == onward:
                        break
                self.trace.append({"kind": "fire", "set": "CH", "query": plain, "pages": read, "by": "token"})
                todo = read
            elif todo:
                self.guard_marks["sweep-pages"] = used + len(todo)
                self.trace.append({"kind": "fire", "set": "CH", "query": plain, "pages": todo})
                self.execute([(tool, dict(plain, **{page_key: p})) for p in todo])
            ids = list(entry["ids"])
            for other in (self.searches.get(_dump(dict(sorted(plain.items())))), self.searches.get(key)):
                if other:
                    ids += [i for i in other["ids"] if i not in ids]
            rows = [self.shelf.items[i] for i in ids if i in self.shelf.items]
            inside = [i for i in rows if str(i.get("section") or "").strip().lower() == section]
            rows = [i for i in (inside or rows) if _is_number(i.get("price")) and self.shelf.affordable(i, self.ledger.ceiling, count)]
            rows.sort(key=lambda i: i["price"])
            if not rows:
                return ""
            return ("the cheapest fitting listing" + (" of " + section if getattr(self, "parts", []) else "") + " is wanted, and the search " + entry["label"] + " lists its " + str(total) + " matches in no price order: "
                    + ("its remaining pages were read, and " if todo else "") + "its cheapest listings within the budget are: "
                    + "; ".join(_label(i) + (" [opened]" if i.get("facts") is not None else "") + " '" + str(i.get("title") or "")[:60] + "'" for i in rows[:6])
                    + ". Open the unopened ones' property lists" + (" with " + self.role["details"] if self.role.get("details") else "")
                    + " cheapest first (in one reply): the first whose own facts state every need is the cheapest fit among the listings that name that value.")
        return ""

    def early_probe(self) -> str:
        """Cheapest single-product requests: once a listing proven by its own facts is known and more than 12 steps remain, send
        the section search capped just under its price (price_probe) and name the cheaper rows never opened."""
        if self.brief.get("pick") != "cheapest" or getattr(self, "parts", []) or self.steps_left() <= 12 or self.wire.left() < CLOSING_SECONDS + 30:
            return ""
        best = next(iter(sorted(self.better_fits(None), key=lambda i: i["price"] if _is_number(i.get("price")) else 1e18)), None)
        if not best or not _is_number(best.get("price")):
            return ""
        if self.mark("probe|" + best["id"] + "|" + str(self.ledger.version) + "|" + str(best.get("price"))) != 1:
            return ""
        unread = [c for c in self.price_probe(best) if c.get("facts") is None][:6]
        if not unread:
            return ""
        return ("the cheapest fitting listing is wanted, and a search of its section capped just under " + _label(best)
                + " (the cheapest listing proven so far) shows cheaper listings never opened: " + "; ".join(_label(c) for c in unread)
                + ". Open them (cheapest first, in one reply) before choosing.")

    # ---------------- the end
    def backups(self) -> list[dict[str, Any]]:
        """Single-product requests: verified fits outside the cart, not gone, not shown out of stock (cheapest first when the
        cheapest was asked)."""
        if getattr(self, "parts", []):
            return []
        return [i for i in self.better_fits(None) if i["id"] not in self.cart and not i.get("gone") and i.get("stock") is not False]

    def remove_action(self, item_id: str) -> dict[str, Any] | None:
        """A cart call that takes the line out: the brief's remove arguments, or a count of 0 when the tool sets counts."""
        tool = self.role.get("cart")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        args: dict[str, Any] = {}
        remove = self.brief.get("cart_remove") if isinstance(self.brief.get("cart_remove"), dict) else {}
        for key, value in remove.items():
            if key in props:
                args[key] = value
        id_slot = None
        counted = False
        for key, spec in props.items():
            if key in args or not isinstance(spec, dict):
                continue
            if "string" in _kinds(spec) and "enum" not in spec and id_slot is None:
                id_slot = key
            elif "integer" in _kinds(spec) and "enum" not in spec:
                args[key] = max(1, self.line_units(item_id)) if remove else 0
                counted = True
            elif "enum" in spec and key in (schema.get("required") or []) and spec["enum"]:
                return None
        if not id_slot and self.lines_slot() and self.lines_set():
            return self.lines_call([(i, self.line_units(i)) for i in self.cart if i != item_id])   # LS: the cart set without the line
        if not id_slot or not (remove or counted):
            return None
        args[id_slot] = item_id
        args, issues = conform(args, schema)
        if issues or not tool:
            return None
        return {"name": tool, "args": args}

    def lines_slot(self) -> tuple[str, str, str | None] | None:
        """LS: a cart tool whose argument is a list of line objects, each an id string and a count ({lines: [{id, qty}]}):
        (list key, id key, count key or None); else None."""
        schema = self.schemas.get(self.role.get("cart") or "") or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            items = spec.get("items") if isinstance(spec, dict) and "array" in _kinds(spec) else None
            inner = items.get("properties") if isinstance(items, dict) else None
            if not isinstance(inner, dict) or any(k != key for k in schema.get("required") or []):
                continue
            ids = [k for k, v in inner.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
            counts = [k for k, v in inner.items() if isinstance(v, dict) and "integer" in _kinds(v) and "enum" not in v]
            if len(ids) == 1 and len(counts) <= 1:
                return key, ids[0], (counts[0] if counts else None)
        return None

    def lines_set(self) -> bool:
        """LS: the line-list cart tool says it sets the whole cart (lines left out are taken out), not that it adds lines."""
        tool = self.role.get("cart")
        fn = next((t["function"] for t in self.tools if t["function"]["name"] == tool), {})
        return bool(re.search(r"\b(?:sets?|exactly|replac\w*|overwrit\w*)\b", str(fn.get("description") or ""), re.I))

    def lines_call(self, lines: list[tuple[str, int]]) -> dict[str, Any] | None:
        """LS: one call of the line-list cart tool carrying these lines."""
        slot = self.lines_slot()
        tool = self.role.get("cart")
        if not slot or not tool:
            return None
        key, id_key, count_key = slot
        rows = [dict({id_key: item_id}, **({count_key: max(1, int(count or 1))} if count_key else {})) for item_id, count in lines]
        args, issues = conform({key: rows}, self.schemas.get(tool) or {})
        return None if issues else {"name": tool, "args": args}

    def add_action(self, item_id: str, count: int | None = None) -> dict[str, Any] | None:
        tool = self.role.get("cart")
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        args: dict[str, Any] = {}
        add = self.brief.get("cart_add") if isinstance(self.brief.get("cart_add"), dict) else {}
        for key, value in add.items():
            if key in props:
                args[key] = value
        id_slot = None
        for key, spec in props.items():
            if key in args or not isinstance(spec, dict):
                continue
            if "string" in _kinds(spec) and "enum" not in spec and id_slot is None:
                id_slot = key
            elif "integer" in _kinds(spec) and "enum" not in spec:
                args[key] = count if count is not None else self.ledger.count
            elif "enum" in spec and key in (schema.get("required") or []) and spec["enum"]:
                return None
        if not id_slot:
            # LS: a tool that sets the whole cart gets the lines already held too
            held = [(i, self.line_units(i)) for i in self.cart if i != item_id] if self.lines_set() else []
            return self.lines_call(held + [(item_id, count if count is not None else self.ledger.count)]) if self.lines_slot() else None
        args[id_slot] = item_id
        args, issues = conform(args, schema)
        if issues or not tool:
            return None
        return {"name": tool, "args": args}

    def finish_now(self) -> None:
        if self.wire.finished or self.wire.broken or self.steps_left() <= 0 or self.wire.left() < 2:
            return
        order, decline = self.role.get("order"), self.role.get("decline")
        action = None
        shown: list[tuple[str, int]] | None = None   # the closing adds' basket, which bypasses the cart records
        if order and self.cart:
            action = self.final_action("order")
        elif order and self.steps_left() >= 2:
            basket = self.commit_basket() or []
            adds = [self.add_action(item_id, count) for item_id, count in basket] if self.steps_left() >= len(basket) + 1 else []
            if basket and self.lines_slot():
                adds = [self.lines_call(basket)]   # LS: one call holds every line
            if adds and all(adds):
                reply = None
                for add in adds:
                    reply = self.wire.act([add])
                    self.trace.append({"kind": "closing_add", "action": add, "reply": reply})
                    if not reply or self.wire.finished:
                        break
                if reply and not self.wire.finished:
                    action = self.final_action("order")
                    shown = list(basket)
                    self.cart = [item_id for item_id, _ in basket]   # the closing adds, which bypass the cart records
        if action is None and decline:
            action = self.final_action("decline")
        if action is None:
            return
        if self.final_kind(action["name"], action["args"]) == "order" and self.steps_left() >= 2:
            self.confirm_cart(shown)
            if self.wire.finished or self.steps_left() <= 0:
                return
        reply = self.wire.act([action])
        self.trace.append({"kind": "closing", "action": action, "reply": reply})
        refused = bool(reply) and not reply.get("finished")
        held = shown if shown is not None else [(i, self.line_units(i)) for i in self.cart]
        if refused and order and held and self.steps_left() > 0 and not self.wire.finished and self.wire.left() > 2 \
                and not self.budget_cut(reply, held):
            if self.steps_left() >= 2 and self.wire.left() > 6:
                self.stock_check([i for i, _ in held])   # a line sold out since keeps the old ending
            lines = [self.shelf.items.get(i) or {} for i, _ in held]
            again = self.final_action("order") if not any(line.get("gone") or line.get("stock") is False for line in lines) else None
            if again and self.steps_left() > 0 and not self.wire.finished:
                reply = self.wire.act([again])
                self.trace.append({"kind": "closing", "set": "RO", "action": again, "reply": reply})
                refused = bool(reply) and not reply.get("finished")
        if refused and order and not getattr(self, "parts", []) and not self.wire.finished and self.wire.left() > 6:
            backup = next((b for b in self.backups() if b["id"] not in self.cart), None)   # a backup already in the cart is no swap
            out = [self.remove_action(i) for i in self.cart if not backup or i != backup["id"]]
            add = self.add_action(backup["id"]) if backup else None
            if backup and add and all(out) and self.steps_left() >= len(out) + 2:
                for step in out + [add]:
                    self.trace.append({"kind": "closing_swap", "action": step, "reply": self.wire.act([step])})
                    if self.wire.finished:
                        return
                again = self.final_action("order")
                if again:
                    reply = self.wire.act([again])
                    self.trace.append({"kind": "closing", "action": again, "reply": reply})
                    refused = bool(reply) and not reply.get("finished")
        if refused and decline and self.steps_left() > 0 and not self.wire.finished:
            action = self.final_action("decline")
            self.trace.append({"kind": "closing", "action": action, "reply": self.wire.act([action])})

    def stock_check(self, ids: list[str]) -> None:
        """Open these listings again with the tool whose results show stock (one step each), so a line sold out since shows."""
        tool = self.stock_tool()
        schema = self.schemas.get(tool or "") or {}
        props = schema.get("properties") or {}
        required = list(schema.get("required") or [])
        slots = [k for k, v in props.items() if isinstance(v, dict) and "string" in _kinds(v) and "enum" not in v]
        target = next((k for k in required if k in slots), None) or (slots[0] if len(slots) == 1 else None)
        if not tool or not target or any(k != target for k in required):
            return
        calls = []
        for item_id in ids[: max(0, self.steps_left() - 1)]:
            args, issues = conform({target: item_id}, schema)
            if not issues:
                calls.append((tool, args))
        if calls:
            self.execute(calls)

    def budget_cut(self, reply: Any, lines: list[tuple[str, int]]) -> bool:
        """A refusal naming an amount below the basket's total: the budget may now be under it."""
        said = " ".join(self.voice(reply.get("observation"), False)) if isinstance(reply, dict) else ""
        total = sum((self.shelf.items.get(i) or {}).get("price") * n for i, n in lines if _is_number((self.shelf.items.get(i) or {}).get("price")))
        amounts = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", said)]
        return bool(total) and any(0 < a < total for a in amounts)

    def neutral_args(self, name: str) -> dict[str, Any]:
        schema = self.schemas.get(name) or {}
        props = schema.get("properties") or {}
        args: dict[str, Any] = {}
        for key in schema.get("required") or []:
            spec = props.get(key) or {}
            if "string" in _kinds(spec) and "enum" not in spec:
                args[key] = NEUTRAL_DECLINE[: spec.get("maxLength", len(NEUTRAL_DECLINE))]
            elif spec.get("enum"):
                args[key] = spec["enum"][0]
            elif "boolean" in _kinds(spec):
                args[key] = True
            elif "integer" in _kinds(spec) or "number" in _kinds(spec):
                args[key] = max(1, spec.get("minimum", 1))
        if len(props) == 1:
            key, spec = next(iter(props.items()))
            if key not in args and isinstance(spec, dict) and "string" in _kinds(spec) and "enum" not in spec:
                args[key] = NEUTRAL_DECLINE[: spec.get("maxLength", len(NEUTRAL_DECLINE))]
        return args

    def working_rules(self) -> list[dict[str, Any]]:
        """RG: the request's process rules the code does not enforce itself: every OB record but the basket approval
        (confirm_cart) and the bare step / message limits (a rule whose only numbers are those limits)."""
        records = getattr(self, "obligations", None) or []
        limits = {str(n) for n in (self.step_limit, self.question_limit) if isinstance(n, int)}
        out = []
        for rule in records:
            numbers = set(re.findall(r"\d+", rule.get("quote") or ""))
            quote = rule.get("quote") or ""
            bare_limit = bool(numbers) and numbers <= limits and LIMIT_SENTENCE.search(quote) is not None
            # the basket approval, a question to the shopper (the driver has no message tool) and the bare limits stay out
            if rule.get("witness") in ("order_message", "question_message") or not quote or bare_limit:
                continue
            out.append(rule)
        return out

    def rule_gate(self, name: str, args: dict[str, Any]) -> str:
        """RG: one check of the working rules before the episode ends (checkout or decline): the model reads the rules against
        the actions taken so far and names what is still undone; the finish is refused once with that list. "" lets it go."""
        working = self.working_rules()
        if not working or self.steps_left() < 4 or self.wire.left() < LAST_CALL_SECONDS + 40 or self.mark("rule-gate") != 1:
            return ""
        log = []
        for entry in self.trace:
            if entry.get("kind") in ("store", "closing_add") and isinstance(entry.get("actions") or entry.get("action"), (list, dict)):
                acts = entry.get("actions") if isinstance(entry.get("actions"), list) else [entry.get("action")]
                for act in acts:
                    if isinstance(act, dict):
                        log.append("step " + str(entry.get("step", "?")) + ": " + str(act.get("name")) + " " + _dump(act.get("args"), 300))
        if self.final_kind(name, args) == "order":
            ending = "place the order for the cart: " + self.basket_text([(i, self.line_units(i)) for i in self.cart], False)
        else:
            ending = "decline: buy nothing and end"
        prompt = (RULE_CHECK_PROMPT + "\n\nENDING: " + ending + "\n\nWORKING RULES:\n"
                  + "\n".join("- " + r["quote"] + " (" + r.get("act", "") + ")" for r in working[:8])
                  + "\n\nACTIONS (oldest first):\n" + "\n".join(log[-60:] or ["(none)"]))
        answer = self.wire.ask_json(prompt, [HELPER], 1200)
        listed = answer.get("missing") if isinstance(answer, dict) else None
        missing = [m for m in listed if isinstance(m, dict) and str(m.get("do") or "").strip()] if isinstance(listed, list) else []
        self.trace.append({"kind": "fire", "set": "RG", "rules": len(working), "missing": missing[:4], "read": isinstance(listed, list)})
        if not missing:
            return ""
        return ("Not run yet (no step used). Before this ends, the request's working rules are not all followed yet: "
                + " ".join("(" + str(k + 1) + ") \"" + str(m.get("quote") or "")[:160] + "\" -> " + str(m.get("do"))[:160] for k, m in enumerate(missing[:4]))
                + ". Do that now, then send the " + ("checkout" if self.final_kind(name, args) == "order" else "decline")
                + " again. If a rule can no longer be followed, send it again as it is.")

    def read_obligations(self, texts: list[str]) -> list[dict[str, Any]] | None:
        """OB: the request's process rules as records {quote, when, act, witness}, read by the model. None when the reading
        failed, [] when the request states none. The code checks the order_message witness itself (confirm_cart: a message
        sent while the cart holds exactly the basket being ordered); the other kinds stay with the driver as working rules."""
        copied = list(dict.fromkeys(t.strip() for t in texts[1:] if isinstance(t, str) and t.strip()))[:12]
        prompt = OBLIGATION_PROMPT + self.request.strip()
        if copied:
            prompt += "\n\nRULES A FIRST READING COPIED FROM IT:\n" + "\n".join("- " + t for t in copied)
        if self.policy.get("opening") is not None:   # OO: a rule the shop's first observation carries binds like one in the request
            prompt += "\n\nTHE SHOP'S FIRST OBSERVATION (rules the shopper states in it bind the same way as rules in the request):\n" \
                + _dump(self.policy.get("opening"), 3000)
        answer = self.wire.ask_json(prompt, [HELPER, REASONER], 900)
        if not isinstance(answer, dict) or not isinstance(answer.get("rules"), list):
            return None
        kinds = ("order_message", "question_message", "view", "other")
        records = []
        for rule in answer["rules"]:
            if isinstance(rule, dict):
                witness = str(rule.get("witness") or "").strip().lower()
                records.append({"quote": str(rule.get("quote") or "")[:200], "when": str(rule.get("when") or "")[:20],
                                "act": str(rule.get("act") or "")[:120], "witness": witness if witness in kinds else "other"})
        return records

    # ---------------- main loop
    def run(self) -> list[dict[str, Any]]:
        if self.wire.finished:
            return self.trace
        self.read_request()
        self.wire.message_tool = self.role.get("message")
        rules = self.brief.get("rules")
        texts = [self.request] + ([str(r) for r in rules] if isinstance(rules, list) else [str(rules)] if rules else [])
        self.approval = False
        self.obligations = self.read_obligations(texts) if self.role.get("message") else []
        if self.obligations is not None:   # OB: the model's reading decides; the patterns below only stand in when it failed
            self.approval = bool(self.role.get("message")) and any(o["witness"] == "order_message" for o in self.obligations)
            self.trace.append({"kind": "fire", "set": "OB", "approval": self.approval, "rules": self.obligations[:6]})
            if not self.approval and self.role.get("message"):   # OX: a second, narrow reading of the approval cue sentences
                said = texts + ([_dump(self.policy.get("opening"), 3000)] if self.policy.get("opening") is not None else [])
                cues = list(dict.fromkeys(m.group(0).strip() for t in said for m in APPROVAL_CUE_RX.finditer(t)))[:6]
                if cues:
                    answer = self.wire.ask_json(APPROVAL_PROMPT + "\n".join("- " + cue for cue in cues), [HELPER], 200)
                    self.approval = isinstance(answer, dict) and answer.get("approval") is True
                    self.trace.append({"kind": "fire", "set": "OX", "cues": cues[:3], "approval": self.approval})
        elif any(APPROVAL_RX.search(t) for t in texts + ([_dump(self.policy.get("opening"), 3000)] if self.policy.get("opening") is not None else [])):
            self.approval = True
            self.trace.append({"kind": "fire", "set": "FB", "approval": True})
        elif self.role.get("message"):
            cues = list(dict.fromkeys(m.group(0).strip() for t in texts for m in APPROVAL_CUE_RX.finditer(t)))[:6]
            if cues:
                answer = self.wire.ask_json(APPROVAL_PROMPT + "\n".join("- " + cue for cue in cues), [HELPER], 200)
                self.approval = isinstance(answer, dict) and answer.get("approval") is True
                self.trace.append({"kind": "fire", "set": "FC", "cues": cues[:3], "approval": self.approval})
        self.stated_sigs = {_need_sig(n) for n in self.ledger.needs}
        for need in self.ledger.needs:
            for value in need.get("values") or []:
                pointer = re.search(r"\b(right for|a match for) the details on (?:the )?(?:listing for )?(?:the )?(.+)", str(value), re.I)
                if pointer or _names_other_listing(value):
                    self.ref_queue = getattr(self, "ref_queue", []) + [{"prop": need["property"], "listing": pointer.group(2) if pointer else str(value),
                                                                        "same": bool(pointer) or not _DIFFER_WORD.search(str(value)),
                                                                        "pair": bool(pointer) and pointer.group(1).lower() == "right for"}]
                    self.trace.append({"kind": "fire", "set": "DR", "stated": need["property"]})
        opening = self.policy.get("opening")
        if opening is not None:
            self.shelf.take(opening, "search", 0)
        self.chat = [{"role": "system", "content": LOOP_PROMPT}, {"role": "user", "content": self.opening_text() + "\n\n" + self.status()}]
        self.wire.setup_extra = max(0.0, self.wire.spent - SETUP_SHARE)
        # SL: a request of several list lines gets proportionally higher spend lines (the run's cap is on the mean, not one task)
        self.wire.scale = min(SPEND_SCALE_MAX, 1.0 + SPEND_PER_LINE * max(0, len(getattr(self, "parts", None) or []) - 1))
        if self.wire.scale > 1.0:
            self.trace.append({"kind": "fire", "set": "SL", "lines": len(self.parts), "scale": round(self.wire.scale, 3)})
        if self.wire.setup_extra:
            self.trace.append({"kind": "fire", "set": "SB", "setup": round(self.wire.spent, 5)})
        silent = failures = blocked = 0
        for _ in range(TURN_CAP):
            if self.wire.finished or self.wire.broken or self.steps_left() <= 0:
                break
            if self.wire.left() < LAST_CALL_SECONDS or self.wire.work_spent() >= self.spend_line(SPEND_STOP) or self.wire.key_dead:
                self.finish_now()
                break
            if self.steps_left() == 2 and self.approval and self.role.get("order") and self.role.get("message") and self.message_room() > 0 \
                    and self.cart_buyable() and _dump(sorted((i, self.line_units(i)) for i in self.cart)) + "#" + str(self.questions_sent) != self.confirmed_cart:
                self.trace.append({"kind": "fire", "set": "RK", "cart": list(self.cart)})   # RK: the last two steps = approval message + order
                self.finish_now()
                break
            if self.asking and self.plan_questions():
                continue
            if self.commit_deadline():   # references are followed by the driver (FW notes); the code resolver settled wrong rows when it ran
                continue
            if self.fit_work():   # FT
                continue
            if self.view_meant():   # VS
                continue
            if self.hop_work():   # NH
                continue
            if self.wire.driver_prompt > TRIM_AT:
                self.trim_chat()
            message = self.wire.ask_model(self.chat, self.driver_tools(), [DRIVER, DRIVER_SPARE, DRIVER], 2500)
            if message is None:
                failures += 1
                if failures >= 2:
                    self.finish_now()
                    break
                continue
            calls: list[tuple[str, str, Any]] = []
            echoed = []
            for call in message.get("tool_calls") or []:
                if not isinstance(call, dict):
                    continue
                self.call_no += 1
                fn = call.get("function") or {}
                call_id = call.get("id") or ("c" + str(self.call_no))
                raw = fn.get("arguments") if isinstance(fn.get("arguments"), str) else json.dumps(fn.get("arguments") or {})
                calls.append((call_id, fn.get("name"), raw))
                echoed.append({"id": call_id, "type": "function", "function": {"name": fn.get("name"), "arguments": raw}})
            content = message.get("content") or ""
            if not isinstance(content, str):
                content = _dump(content)
            committing = {self.role.get("cart"), self.role.get("order"), self.role.get("decline")} - {None}
            if self.asking and any(name in committing for _, name, _ in calls) and self.plan_questions(force=True):
                continue
            if not calls:
                silent += 1
                self.chat.append({"role": "assistant", "content": content or "(no tool call)"})
                self.chat.append({"role": "user", "content": "Every reply must call one of the supplied tools. Continue.\n" + self.status()})
                if silent >= 3:
                    self.finish_now()
                    break
                continue
            silent = 0
            usable = {t["function"]["name"] for t in self.offered_tools()}
            if len(usable) < len(self.driver_tools()) and not any(name in usable for _, name, _ in calls):
                blocked += 1
                if blocked >= 2:   # the driver keeps calling tools that are no longer usable: finish by code
                    self.trace.append({"kind": "fire", "set": "TS", "calls": [name for _, name, _ in calls]})
                    self.finish_now()
                    break
            else:
                blocked = 0
            self.trace.append({"kind": "driver", "content": content[:2000], "calls": [{"name": n, "arguments": r} for _, n, r in calls]})
            self.chat.append({"role": "assistant", "content": content or None, "tool_calls": echoed})
            answers = self.handle(calls)
            for index, (call_id, text) in enumerate(answers):
                if index == len(answers) - 1:
                    text += "\n" + self.status()
                self.chat.append({"role": "tool", "tool_call_id": call_id, "content": text})
        if not self.wire.finished and not self.wire.broken:
            self.finish_now()
        self.trace.append({"kind": "end", "finished": self.wire.finished, "steps": self.wire.step, "cost": round(self.wire.spent, 5), "seconds": round(time.monotonic() - self.started, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    session = Session(problem_data)
    try:
        return session.run()
    except Exception as exc:  # keep the episode alive: report, then finish cleanly
        session.trace.append({"kind": "crash", "error": (type(exc).__name__ + ": " + str(exc))[:500]})
        try:
            if not session.wire.finished and not session.wire.broken:
                session.finish_now()
        except Exception as again:
            session.trace.append({"kind": "crash", "error": (type(again).__name__ + ": " + str(again))[:500]})
        return session.trace


__all__ = ["agent_main"]
