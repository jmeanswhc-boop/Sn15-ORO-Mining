"""Ghost: a shopping agent for ORO store worlds.

One episode is one shopper's request inside a generated store. The store's tools differ from world to world, so
Ghost reads every tool from its schema and learns result shapes from what comes back. A small model proposes the
store calls; the code around it owns the procedure:

- it reads the request once into a ledger (items, firm requirements, budget, pick rule, message and step limits,
  procedures the request asks for) and keeps that ledger current as the shopper and the store say more;
- it runs the interview itself: one plain question about one property of one item per message, chosen from the
  story's hints and from the attributes the listings carry, because distracted shoppers answer nothing else;
- it resolves requirements whose value lives on another listing's page by opening every page of that chain, and
  reads store rules (values that "go together") from the pages it has seen;
- it opens both views of a listing it is about to buy where the store splits attributes from the offer page;
- it checks the cart before every checkout and, for a cheapest pick, looks below the cart's price first;
- it declines only when a requirement the shopper stated outright is met by nothing the store can sell, after the
  category was searched; running out of steps or an unverified chain never ends an episode without a purchase;
- it keeps every episode inside its step, message, time and spending budgets and always ends it itself.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# ----------------------------------------------------------------------------------------------------------------
# models and budgets
# ----------------------------------------------------------------------------------------------------------------
QUICK = 'anthropic/claude-haiku-5.5'          # drives the store loop and reads short texts
QUICK_SPARE = 'openai/gpt-6-luna'              # stands in when the quick model is unavailable
DEEP = 'anthropic/claude-sonnet-5.5'           # reads the request, checks the cart, reviews a decline
DEEP_KIN = (DEEP, 'anthropic/claude-sonnet-5', 'openai/gpt-6-sol')

EPISODE_SECONDS = 300.0        # the validator's time limit for one task
TIME_FLOOR = 14.0              # never start a call with less than this left
EXPLORE_FLOOR = 55.0           # stop exploring, start finishing, at this much time left
CALL_CAP = 64                  # model requests per episode
LOOP_CAP = 40                  # loop turns per episode
SPEND_SOFT = 0.048             # USD: deep model stops being offered above this
SPEND_HARD = 0.072             # USD: all model work stops above this
DEEP_ROOM = 0.018              # USD a deep call may still cost near the soft cap
BACKOFFS = (3, 8, 16, 30)
BACKOFF_TOTAL = 42
BUSY_SECONDS = 20.0
ROW_CHARS = 420
OBS_CHARS = 14000
HISTORY_CHARS = 140000
DEFAULT_STEPS = 30
DEFAULT_MESSAGES = 3
PAGE_LIMIT = 6                 # extra result pages Ghost reads itself for a cheapest/most/least pick
CHEAPER_OPENS = 3
CHAIN_DEPTH = 6
CHAIN_STEPS = 14
RECOVERY_STEPS = 8
SWEEP_STEPS = 6
WIDEN_PER_ITEM = 5
LIST_SIZE = 10                 # properties one message may name for a single item
LIST_MIXED = 4                 # properties per item when one message covers several items
MIX_ITEMS = 3                  # most items one message covers

# ----------------------------------------------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------------------------------------------
GUIDE = (
    "You are Ghost, a buyer acting for one shopper inside an online store. You act only through the store's tools, "
    "one tool call per step, and the episode scores 1 only when the ending is exactly right: a checkout whose cart "
    "holds every requested item at its quantity, each meeting every requirement in force at that moment, within the "
    "budget; or, when the store truly sells nothing that meets a requirement the shopper stated, the no-purchase "
    "ending done the way the request asks. Anything else scores 0, so there is no credit for effort.\n\n"
    "THE STORE\n"
    "- Tool names, argument names and result shapes belong to this store: read each schema and use the exact "
    "argument names and allowed values. Every call is a step and steps are scarce; thinking is free.\n"
    "- Searches are narrow: a category filter, a keyword mode that reads only a few words, short pages with a token "
    "or page number for the next page. Compare the total reported with the rows shown and page on only when the "
    "answer depends on seeing all rows. Keyword queries work best short: the wanted value plus the product noun. A "
    "filter that returns nothing usually has the wrong label: use the label exactly as rows show it.\n"
    "- Rows are summaries. A listing's attributes live in its full record; stores sometimes split that record into an "
    "attribute view and an offer page (price, stock, free-text notes). Read the attribute view of every listing you "
    "might buy, and never open a view again when nothing has changed since you read it.\n"
    "- Ghost sends every message to the shopper itself; you never call the messaging tool. Answers arrive in the "
    "conversation and land in the ledger.\n"
    "- Free-text notes on a listing's page can point to ANOTHER listing for one property (\"for X, see the ... "
    "from <shop>\"), state store rules (\"<value> goes together with <value>\"), or mention versions sold elsewhere. "
    "Ghost follows such chains itself when a requirement needs them; you just open pages when asked to.\n\n"
    "WHAT FITS\n"
    "- The product kind is the listing's category label; neighbouring categories do not count. When a label matches "
    "the request's words exactly (\"Pants & Jeans\" for \"pants or jeans\"), rows under any other label (\"Jeans\") are "
    "not the item.\n"
    "- A property requirement is met when the listing's attribute keys naming that property state the value (any "
    "key spelling or prefix; one element of a list is enough; synonyms, case and plural are fine; a value containing "
    "the wanted words counts; units convert; limits include the edge). Keys that name the property decide: when "
    "they all state the value the listing fits even if the title is silent; when they exist and none states it the "
    "listing FAILS even if the title says the value; when they disagree the listing is doubtful and loses to one "
    "whose keys agree. A key for a different property decides nothing, and a broader or neighbouring value is not "
    "the value (print is not logo, skinny is not slim). Only when no key names the property does the title, "
    "description or another attribute count. Ghost grades every opened listing on each requirement under "
    "[evidence]; trust those grades.\n"
    "- Exclusions and closed lists are strict. Wishes are tie-breakers only. Store rules pairing values (\"a goes "
    "with b\") make the paired value acceptable where the request or the shopper asked for the other.\n"
    "- When one item must fit another (an accessory for a device), judge the pair by the listings' own words: a "
    "model or series code both pages carry, or a list of what it fits.\n"
    "- Never buy anything shown sold out; the total of price times quantity over all lines stays within the budget.\n\n"
    "CHOOSING\n"
    "- Cheapest pick: the order must hold the lowest-priced fitting listing among all buyable ones. Rows come in no "
    "price order, so see the whole set (filtered category listing under a price cap when the store has one) and "
    "check candidates from the cheapest up. Most/least of a measure: compare every fitting listing; one that does "
    "not state the measure cannot win. Otherwise any fitting listing is right: do not hunt for a better one. Between "
    "candidates that both seem to fit, prefer the one whose attribute KEY names the required property outright and "
    "states the value; a value found only in the title, the description or under a neighbouring key is weaker.\n"
    "- Under a cheapest pick a cheaper listing Ghost grades DOUBTFUL or title-only is the right buy: never replace it "
    "with a dearer one whose keys agree; replace a line only when it FAILS, sells out or a cheaper fitting listing "
    "appears. When Ghost reports that a listing fits by its record, do not re-open it or search for the same value "
    "again: cart it (if it is not carted) and check out.\n"
    "- Work item by item. Cart each item as soon as one listing meets everything you know, then move on; a carted "
    "line can be replaced later, an empty cart cannot be rescued when steps run out. A request for several items "
    "needs one line per item at its quantity. Decide before you cart: every swap of a line costs two steps, so "
    "compare candidates from their records first and change the cart only when a line is sold out, a change "
    "demands it, or a clearly better listing turns up under a cheapest/most/least rule.\n\n"
    "CHANGES\n"
    "- The shopper or the store can change the task: a new budget, a requirement added or dropped, a price change, a "
    "sold-out line, a refused checkout with a reason. Such notes appear inside tool results, most often as the reply "
    "to a checkout. Apply them at once: a line reported sold out must leave the cart (never check out with it), a "
    "dropped requirement stops binding, a new one binds; then fix the cart and check out again. Ghost keeps the "
    "last few steps free for exactly this.\n\n"
    "ENDING\n"
    "- Set the cart to exactly the chosen lines, then call the checkout tool on its own. Ghost runs its own checks "
    "first and may hand the call back with what to fix.\n"
    "- The no-purchase ending is right only when a requirement the shopper stated outright is met by no buyable "
    "listing after the category was searched; a requirement you could not verify, a missing page, or a shortage of "
    "steps is never a reason. When in doubt, buy the best fitting candidates.\n"
    "- Every reply calls at least one tool. Keep text to a line: what you know and the next move."
)

BRIEF = (
    "You prepare a shopping errand for an agent. From the shopper's request and the store's tools, return ONE JSON "
    "object (compact, nothing else) with these keys.\n"
    "\"items\": one entry per distinct product the shopper wants bought: {\"name\": the kind of product in the "
    "request's own words, \"shelf\": the store's category label for it when a tool schema lists labels (copied "
    "exactly) else \"\", \"quantity\": units of that listing (integer; 1 when unstated; a pair or set sold as one "
    "listing is 1), \"needs\": firm requirements on that product as \"property: constraint\" strings with the exact "
    "value, limit, exclusion or closed list, naming the property with the request's word for it (\"dial glass: "
    "sapphire\", never \"sapphire glass: required\": the constraint is the value itself). A requirement whose "
    "value is read from another listing is ONE need carrying the whole description of that listing. Never put price, "
    "budget, stock or quantity into needs. The story (who uses it, where, what went wrong before, the occasion) is "
    "context, not a need, unless it states a property's value or limit outright; a word that is part of the product's "
    "name (a season, an audience) belongs to \"name\". A use the request makes a property of the product itself "
    "(\"a dress made for parties\", \"mules for a wedding\") is a need unless the category label already states it, and "
    "so is anything the shopper says they will check or verify on the listing. A comparative or vague "
    "word without a value (longer, bigger, decent, nicer, something warm) is a hint, never a need. One \"I want X and Y\" "
    "phrase with one quantity is ONE item; split only when each product gets its own quantity or requirements.}\n"
    "\"budget\": the most the shopper pays in total, as a number in the request's unit, or null. \"budget_unit\": that "
    "unit in words or null. \"budget_scope\": \"total\" or \"each\" or null.\n"
    "\"in_stock_only\": true unless the request clearly accepts unavailable items.\n"
    "\"pick\": \"cheapest\" when the request asks for the lowest price or total, \"most\"/\"least\" for the extreme "
    "of a measure (\"pick_measure\" names it, else \"\"), else \"any\". Tight money is not a pick rule.\n"
    "\"messages\": how many messages the agent may send the shopper (integer) or null. \"steps\": the step limit the "
    "request states (integer) or null.\n"
    "\"lookups\": the needs (copied exactly) whose value must be read from another listing, else [].\n"
    "\"rules\": every instruction about how to work or how to end, each in at most 12 words.\n"
    "\"wishes\": things the shopper would merely like.\n"
    "\"hints\": sentences of the story that point at an unwritten preference without naming its value (something the "
    "shopper cannot judge or leaves to you, a person whose taste or body decides, looks, setting, what is inside, how "
    "it will be used or carried): [{\"item\": item number, \"text\": the sentence, at most 14 words, \"property\": "
    "the one listing property it most likely concerns, in plain words}].\n"
    "\"ask_first\": things the request explicitly says to ask the shopper about before choosing, else [].\n"
    "\"process\": steps the request requires beyond finding and buying (an approval, telling or showing the shopper "
    "something, checking something with them, reading or opening something first, using a tool a certain way): "
    "[{\"step\": at most 15 words, \"when\": \"choose\" | \"order\" | \"end\", \"messages\": messages it costs}]; "
    "requirements, budget, stock, message and step limits and the way to end when nothing fits are NOT steps.\n"
    "\"quiet_decline\": true when the request says not to suggest or name alternatives when nothing fits, false when "
    "it asks for them, null otherwise.\n"
    "\"fits\": [{\"item\": n, \"with\": m}] for each item that must work with or fit another requested item, else [].\n"
    "\"browse\": how one tool lists a whole store category: {\"tool\": name, \"category\": the argument taking the "
    "category label, \"max_price\": the argument taking a price cap or null, \"fixed\": other arguments that select "
    "that mode} or null.\n"
    "\"roles\": the exact tool for each job or null: \"search\", \"open\" (one listing's attributes; with two "
    "single-listing tools, the one promising attributes, properties or specifications), \"offer\" (one listing's "
    "offer page: price, stock, notes, when that is a different tool), \"compare\" (several listings in full), "
    "\"ask\" (message the shopper), \"cart\" (change the cart), \"cart_view\", \"buy\" (check out), \"refuse\" "
    "(end without buying).\n"
    "\"buy_args\": the fixed arguments that make the buy tool check out (a mode value), else {}. \"refuse_args\": "
    "the fixed arguments that end without a purchase, else {}."
)

PLAN = (
    "You choose which properties a shopping agent asks a shopper about. The shopper holds a few unwritten firm "
    "preferences, each on ONE property of ONE requested item, and reveals one only when a message asks about exactly "
    "that property of that item; any other question is wasted. Messages are scarce, so rank carefully.\n"
    "You get the request, the items with what the request already settles, the story hints (sentences that point at "
    "an unwritten preference), and the attribute keys seen on opened listings of each item's category with coverage "
    "counts and sample values.\n"
    "Rank each item's candidate properties:\n"
    "1. anything the request explicitly says to ask the shopper about;\n"
    "2. properties the story hints at: a sentence about who wears or uses it, its looks, where it stands, what it "
    "holds, how it moves or is stored, what the shopper cannot judge; name the listed key that fits the hint (the key "
    "for colour first and pattern second when the hint is about looks, photos, matching an outfit or a room; gender "
    "or age group for the wearer; size, capacity or pack size for what it holds; height, length or dimensions for "
    "where it stands), always a key the listings carry when one fits, never an invented quality such as comfort or "
    "durability;\n"
    "3. then the keys a buyer of that kind of product decides on first, favouring keys most listings carry.\n"
    "Skip a property the request settles (a need names it), a second spelling of a property already chosen, keys "
    "whose values list several purchase options at once, identifiers, model names, warranty and logistics.\n"
    "\"label\" is the property's plain name: for a listed key, all its words in plain spelling (\"lid colour\" for "
    "lidColour), never shortened, no values, no second property.\n"
    "Return ONE JSON object and nothing else: {\"ask\": [{\"item\": item number, \"label\": \"...\", \"hinted\": "
    "true when a story hint or an explicit instruction points at it}]} with up to COUNT entries per item, best first."
)

HEAR = (
    "A shopping agent asked a shopper about one or several properties of one item per message (the properties asked "
    "about are listed with each question). Read each answer and return ONE JSON object and nothing else: {\"needs\": [{\"item\": item number, \"need\": \"property: constraint\", \"points\": true when the "
    "value must be read from another listing, else false}], \"drop\": [{\"item\": n, \"property\": \"a requirement the "
    "shopper now gives up\"}], \"free\": [{\"item\": n, \"property\": \"a property the shopper says they do not mind\"}] "
    "(an answer that states no preference at all makes every property that question asked about free), "
    "\"lost\": [question numbers whose answer only says the shopper did not follow, did not understand, wants it "
    "repeated or put differently, and states nothing], \"crowded\": [question numbers whose answer complains that the "
    "message asked about too many things or wants one question or one thing at a time], \"budget\": a new total limit as "
    "a number, or null}\n"
    "- A message may name several properties and even several items; the answer then concerns ONE of them, named in the "
    "answer itself: read the item and property from the answer's words.\n"
    "- A need is a firm preference the answer states, named by the property the ANSWER names (it may differ from the "
    "one asked), with the exact value or limit and unit copied from the answer (\"at least 3 shelves\", \"not wicker\"). "
    "Words that only join the property to its value (set to, landing at, coming in as, has to match, matching, lines "
    "up with, comes out to, in the form of, equal to, equals, comes to, set at, at) are not part of the value: \"size landing at large\" states large and "
    "\"pattern matching animal print\" states animal print. Name the property by its plain name without store "
    "prefixes (pattern, not fa pattern; closure, not shoes closuretype).\n"
    "- An answer may give the value by pointing at another listing (identical to, different from, the partner of, "
    "within some amount of the value on the page of some shop's listing described by a few properties): that is ONE "
    "need on the property the answer names, \"points\": true, whose value is the answer's own words from the "
    "comparison phrase to the end, copied exactly. The properties describing that other listing are not needs.\n"
    "- Folksy wording that names a value still states it. A refusal to bargain over a value states it as wanted."
)

NOTE = (
    "While a shopping agent worked, a store tool returned the text below (from the store or the shopper). Return "
    "ONE JSON object and nothing else: {\"kind\": \"change\" when it changes the task (budget, a requirement added, "
    "dropped or replaced, a price, availability, a refused checkout with a reason), \"error\" when it only reports a "
    "malformed or refused call, \"other\" otherwise; \"budget\": the new total limit as a number in the request's unit "
    "or null; \"add\": [{\"item\": n, \"need\": \"property: constraint\"}]; \"drop\": [{\"item\": n, \"need\": \"the "
    "requirement no longer wanted\"}]; \"gone\": [listing ids said to be unavailable]; \"sold_out\": true when it says "
    "a chosen listing is gone or sold out, even unnamed; \"price_moved\": true when a price changed; \"summary\": one "
    "short sentence}. A replaced requirement is one drop plus one add. Report every change the text carries."
)

SHAPE = (
    "A store tool returned the result below. Name its parts with the exact keys the result and the tool's "
    "parameters use. Return ONE JSON object and nothing else: {\"rows\": the key of the list of listings, \"count\": "
    "the key holding how many listings matched in all (not how many are shown) or null, \"next\": the key whose "
    "value asks for the following page or null, \"next_arg\": the tool parameter that takes that value or null, "
    "\"price\": the key inside a row holding the listing's price or null, \"stock\": the key inside a row saying "
    "whether it can be bought now or null, \"seller\": the key inside a row naming the shop or seller or null}"
)

READING = (
    "a listing meets a property requirement when the attribute keys naming that property state the value (any key "
    "spelling or store prefix, one element of a list is enough, synonyms, case and plural count, a value containing "
    "the wanted words counts, units convert, limits include the edge, a range must lie wholly inside a limit); a key "
    "names the property when its words are the property's words plus at most a qualifier such as the product's name "
    "or type/style/filter (sleeve length and sleeve style both name sleeves; heel height does not name heel type; "
    "collar type does not name type); when every key naming the property states the value the listing fits, whether "
    "or not the title repeats it; when such keys exist and none states the value the listing FAILS even if the title "
    "or description says the value; when such keys disagree (one states it, another states something else) the "
    "listing is DOUBTFUL and loses to one whose keys agree; without any such key, a value stated in the title, "
    "description or another attribute counts, weaker than agreeing keys; a broader, narrower or neighbouring value is "
    "not the value (print is not logo, skinny is not slim) and a value under a key for a different property decides "
    "nothing; a bare figure of unknown meaning counts for nothing; a value the category label itself states is met "
    "by the category; a store rule pairing two values makes either acceptable for the other; \"different from X\" is "
    "met by any value other than X; \"within N of X\" is a numeric band around X; out-of-stock listings and listings "
    "that break the budget do not fit"
)

CHECK = (
    "You are the last check before a shopping agent checks out. From the records shown, decide whether the cart is "
    "exactly right. Reading rules: " + READING + ". Each requested item needs its own cart line at the requested "
    "quantity, nothing else may be in the cart, nothing shown out of stock, and price times quantity summed over "
    "the lines must stay within the budget. The newest record of a listing wins. When the pick rule is cheapest, "
    "most or least, also look at the other listings shown: if one clearly fits every requirement and beats the "
    "cart's line on that rule, the cart is wrong. Under \"better\", also name a listing that fits with stronger "
    "evidence (an attribute key naming the property outright) when the cart's line fits only through its title, "
    "description or a neighbouring key and the pick rule does not forbid the swap. Under a cheapest pick a cheaper "
    "line graded DOUBTFUL or title-only stays: a dearer listing with agreeing keys is never \"better\" then; only "
    "a FAILS grade, a sold-out line or a cheaper fitting listing changes a cheapest cart.\n"
    "Return ONE JSON object and nothing else: {\"ok\": true or false, \"problems\": [\"short concrete problem\"], "
    "\"better\": [listing ids that should replace a line, best first]}. Use ok=false only for a problem the records "
    "clearly show or a requirement no record addresses."
)

AUDIT = (
    "A shopping agent is about to end without buying because it believes nothing fits. Judge that from the records, "
    "with these reading rules: " + READING + ". Records marked ROW were seen only in search results; their "
    "attributes were not read, so they can neither pass nor fail on a property the row does not show.\n"
    "First compare every requirement with the REQUEST and the ANSWERS: a requirement that only repeats the story "
    "(who, when, where, why, how it should feel) and names no property value goes under \"not_firm\"; a requirement "
    "is FIRM, never not_firm, when the shopper names a property with a value, says they will check, verify or "
    "double-check it themselves, insists on it, or an answer states it. A value that picked up joining words goes "
    "under \"corrected\" with the plain value. Judge the listings without the not_firm requirements and with the "
    "corrected values. The [grades] under a record are code-computed from its keys and are authoritative: a FAILS "
    "grade rules the listing out, a strong grade means the record meets that requirement.\n"
    "Return ONE JSON object and nothing else: {\"not_firm\": [{\"item\": n, \"need\": copied exactly}], "
    "\"corrected\": [{\"item\": n, \"need\": copied exactly, \"value\": \"property: plain value\"}], \"fits\": "
    "{\"item number\": [ids of listings that appear to meet every requirement, cheapest first, at most 4]}, "
    "\"maybe\": {\"item number\": [ids that meet every requirement a record addresses and contradict none, cheapest "
    "first, at most 4]}, \"open\": [ids of ROW listings of a requested kind within the budget whose attributes could "
    "show a fit, most promising first, at most 6], \"search\": [at most 4 of {\"item\": n, \"query\": \"two or three "
    "keywords not tried yet that could surface a fitting listing\"}], \"blocking\": [{\"item\": n, \"need\": the "
    "requirement copied exactly that no listing seems able to meet}], \"reason\": \"one sentence\"}"
)

SWAP = (
    "A listing in a shopping agent's cart has sold out. Find replacements for that line among the other records "
    "shown, with these reading rules: " + READING + ".\nReturn ONE JSON object and nothing else: {\"fits\": "
    "[{\"id\": listing id, \"price\": its price as the record shows it} for every listing that appears to meet "
    "every requirement of the sold-out line's item, at most 6], \"reason\": \"one sentence\"}"
)

GATE = (
    "A shopping agent is about to STAGE. The request asks for the steps below along the way. From the numbered "
    "RECORD of what the agent has done, decide whether each step due before this STAGE is done for the cart as it "
    "is now. A step counts as done only when a RECORD line shows it being done and, for a step about the cart or "
    "the order, that line comes after the cart last changed and the reply grants it; product questions do not do a "
    "step. When a step is missing, give the ONE store call that does it next (a tool from TOOLS with arguments). A "
    "message to the shopper goes through the messaging tool: one short plain sentence about that step alone, "
    "naming no product properties the step does not need. Give no call when every due step is done or when the "
    "messages or steps left cannot do it.\n"
    "Return ONE JSON object and nothing else: {\"steps\": [{\"step\": ..., \"done\": true or false, \"lines\": "
    "[RECORD line numbers]}], \"call\": {\"tool\": \"...\", \"args\": {...}} or null, \"hold\": what a reply asks to "
    "change before going on, or \"\", \"needs\": [{\"item\": n, \"need\": \"property: constraint\"} for each firm "
    "requirement such a reply states]}"
)

REF_PARSE = (
    "A shopping requirement takes its value from another listing instead of stating it. Read it and return ONE JSON "
    "object and nothing else: {\"relation\": \"identical\" | \"different\" | \"partner\" | \"within\" | \"other\", "
    "\"amount\": for within: the allowed distance as a number, else null, \"unit\": its unit or \"\", \"property\": the "
    "property whose value is wanted, in plain words, \"anchor\": {\"category\": the kind or category of the listing "
    "it points at, \"shop\": the shop, seller or brand named, or \"\", \"traits\": [\"property: value\" for each "
    "property that picks the listing out]}, \"follow\": true when the requirement says to follow what that page "
    "names or points to onward, else false}"
)

PAGE_READ = (
    "A shopping agent follows a chain of listing pages to learn the value of one property. Read the page text and "
    "return ONE JSON object and nothing else: {\"redirect\": when the page sends the reader to ANOTHER listing for "
    "that property: {\"category\": its kind, \"shop\": its shop or brand or \"\", \"traits\": [\"property: value\" "
    "picking it out]}, else null; \"value\": the property's value as this page states it when it does not redirect "
    "for that property (from its attributes or text), else \"\"; \"rules\": [[\"value a\", \"value b\"] for every "
    "pairing rule the text states, such as \"a goes together with b\" or \"pairs a with b\"]; \"negations\": "
    "[\"property: value\" for each value the text says this listing is NOT]}"
)

LINK_PICK = (
    "Several listing records are candidates for the ONE listing a description points at. Return ONE JSON object and "
    "nothing else: {\"id\": the listing id whose shop or brand, kind and every described property value all match "
    "the description, or \"\" when none matches all of it, \"close\": [ids that match the shop and kind but could "
    "not be judged on every described property because that attribute is missing]}"
)

PICKER = (
    "A shopping agent must choose what to buy now; steps are nearly gone and no listing is verified to meet every "
    "requirement. Choose, for each item, the listing most likely to satisfy the request from the records shown, with "
    "these reading rules: " + READING + ". Prefer a listing that meets every requirement a record addresses and "
    "contradicts none, and among those the one whose attribute keys name the required properties outright; then "
    "apply the pick rule (cheapest when asked, else any). A listing whose record contradicts a plain requirement the "
    "shopper stated (in the request or an answer) must not be chosen: give \"\" for that item instead. The [grades] "
    "under a record are code-computed from its keys and authoritative: never choose a listing graded FAILS on any "
    "requirement while a listing without a FAILS grade exists; prefer strong grades, then title-only, then doubtful. "
    "A requirement whose value lives on another listing's page and was not read yet rules nothing out: judge the "
    "other requirements and still choose. Return ONE JSON object and "
    "nothing else: {\"choice\": {\"item number\": listing id or \"\"}, \"reason\": \"one sentence\"}"
)

FIT_WORDS = (
    r"(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|"
    r"designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|"
    r"supports?|takes? (?:part )?series)"
)
FIT_EXCEPT = r"\b(?:except(?:\s+for)?|excluding|other than|but not|apart from|not including|not for)\b"
CODE_RX = r"[A-Z][A-Za-z]{0,7}(?:[- ]?\d{2,6}|[- ]?\d{1,5}[xX]{1,4})"
SERIES_RX = r"\b([A-Z]{1,5}\d{1,3})(?=-\d{2,6}\b)"
NOT_PROPERTY = r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition|shipping|delivery|origin)\b"
CATCH_ALL = r"^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b"


# ----------------------------------------------------------------------------------------------------------------
# small helpers: text, json, schemas, records
# ----------------------------------------------------------------------------------------------------------------
def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def clip(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return '...'
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + '...'
    if isinstance(value, dict):
        return {k: clip(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [clip(v, limit, depth + 1) for v in value]
    return value


def parse_json_object(text: Any) -> dict[str, Any] | None:
    """The first JSON object in a model reply, with or without a code fence."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    fence = re.match(r'^```[A-Za-z]*\s*(.*?)\s*```$', body, re.S)
    if fence:
        body = fence.group(1)
    try:
        got = json.loads(body)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    start, end = body.find('{'), body.rfind('}')
    if 0 <= start < end:
        try:
            got = json.loads(body[start:end + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None


def plain(key: Any) -> str:
    """Words of an attribute key: camelCase, underscores, dots and dashes become spaces."""
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', str(key))
    return re.sub(r'[\s_.\-]+', ' ', text).strip().lower()


def stem_word(word: str) -> str:
    if re.search(r'(?:ch|sh|ss|x|z)es$', word):
        return word[:-2]
    if len(word) > 3 and word.endswith('s') and not word.endswith('ss'):
        return word[:-1]
    return word


def stems(text: str) -> frozenset[str]:
    return frozenset(stem_word(word) for word in plain(text).split())


def content_words(text: str) -> set[str]:
    return {w for w in stems(text) if len(w) >= 3 and w not in ('and', 'for', 'the', 'with', 'from', 'that')}


NOISE_WORDS = frozenset({'fa', 'm', 'w', 'mus', 'kid', 'is', 's', 'the', 'of', 'and', 'a', 'an'})
PROP_LINKS = frozenset({'type', 'style', 'kind', 'filter', 'general', 'main', 'primary', 'item', 'product', 'clothing', 'class', 'option', 'detail'})
VALUE_STOP = frozenset({'a', 'an', 'the', 'of', 'and', 'or', 'with', 'in', 'on', 'for', 'to', 'is', 'be', 'as', 'at', 'by', 'its', 'it',
                        'that', 'this', 'any', 'some', 'only', 'just', 'one', 'made', 'from'})
LIMIT_RX = r"\b(?:at most|at least|under|over|below|above|less|more|between|within|up to|max|min|maximum|minimum|no more than|no less than|or less|or more|exactly|around|about)\b|\d|[<>=\u2264\u2265]"
EXCL_RX = r"^(?:not|no|never|other than|anything but|except|excluding|without|none of)\b\s*"
GRADE_RANK = {'strong': 0, 'skip': 0, 'weak': 1, 'conflict': 1, 'unknown': 3, 'unread': 3, 'fails': 9}
SOLD_RX = r"sold out|out of stock|no longer|unavailable|zero left|none left|\bgone\b|ran dry|dried up|beat me to it|can'?t be ordered|cannot be ordered|grayed out|all gone|nobody has any|run out|no stock|last of those"


def prop_stems(text: str) -> frozenset[str]:
    """The words of a property name that carry meaning: store prefixes and link words dropped."""
    return frozenset(w for w in stems(text) if w not in NOISE_WORDS)


def same_property(a: str, b: str) -> bool:
    """Whether two property names mean the same property: equal words, one contained in the other, or the same
    letters run together ("closuretype" and "closure type")."""
    sa, sb = prop_stems(a), prop_stems(b)
    if not sa or not sb:
        return False
    if sa == sb or sa <= sb or (len(sb) >= 2 and sb <= sa):
        return True
    return joined_stems(a) == joined_stems(b)


def joined_stems(text: str) -> str:
    """A property name's meaningful stems run together in order: "shoes closure type" and "shoes closuretype" agree."""
    return ''.join(w for w in (stem_word(x) for x in plain(text).split()) if w not in NOISE_WORDS)


def key_names(prop: str, key: str, extra_ok: frozenset[str] = frozenset()) -> bool:
    """Whether an attribute key names a property: it carries the property's words and adds nothing but link words
    ("type", "style", "filter") or words of the product's own name. "sleeve length" and "sleeve style" both name
    "sleeves"; "heel height" does not name "heel type"; "collar type" does not name "type"."""
    ps, ks = prop_stems(prop), prop_stems(key)
    if not ps or not ks:
        return False
    if ps == ks or joined_stems(prop) == joined_stems(key):
        return True
    if len(ps) >= 2:
        return ps <= ks or (ks <= ps and len(ks) >= 2)
    return ps <= ks and all(w in PROP_LINKS or w in extra_ok for w in ks - ps)


SAME_WORDS = (frozenset({'female', 'women', 'woman', 'ladies', 'lady', 'womens'}), frozenset({'male', 'men', 'man', 'mens', 'gents', 'gentlemen'}),
              frozenset({'kids', 'kid', 'children', 'child', 'childrens'}), frozenset({'baby', 'babies', 'infant', 'infants', 'toddler', 'toddlers'}))
EMPTY_VALUES = frozenset({'required', 'yes', 'true', 'needed', 'must', 'present', 'any', 'none', 'ok', 'fine'})


def word_matches(word: str, token: str) -> bool:
    if token == word or (len(word) >= 3 and token.startswith(word) and len(token) - len(word) <= 3) or (len(token) >= 3 and word.startswith(token) and len(word) - len(token) <= 3):
        return True
    return any(word in group and token in group for group in SAME_WORDS)


def value_states(text: Any, value: str) -> bool:
    """Whether a text states a wanted value: every content word of the value appears, allowing a plural, a short
    suffix (knit/knitted, zip/zipper, square/squares) or a universal synonym (female/women), but not a different
    word (slim is not slimming)."""
    tokens = [stem_word(t) for t in re.findall(r'[a-z0-9]+', str(text).lower())]
    words = [stem_word(w) for w in re.findall(r'[a-z0-9]+', value.lower()) if w not in VALUE_STOP]
    if not words or not tokens:
        return False
    for word in words:
        if not any(word_matches(word, t) for t in tokens):
            return False
    return True


def grade_record(block: dict[str, Any] | None, text: str, need: 'Need', extra_ok: frozenset[str] = frozenset()) -> str:
    """How a read listing stands on one plain requirement, by its attribute keys: 'strong' when every key naming
    the property states the value; 'conflict' when such keys disagree; 'fails' when they exist and none states it;
    'weak' when only the title, text or a key for another property states it; 'unknown' when nothing read states
    it; 'skip' for limits, numbers and unresolved chains, which code does not judge."""
    value = (need.resolved if need.points and need.resolved else need.value) or ''
    low = value.lower().strip()
    if not low or (need.points and not need.resolved):
        return 'skip'
    excluded = re.match(EXCL_RX, low) is not None
    body = re.sub(EXCL_RX, '', low, count=1).strip() if excluded else low
    if not body or re.search(LIMIT_RX, body) or body in EMPTY_VALUES:
        return 'skip'
    own = prop_stems(need.prop)
    alternatives = []
    for part in re.split(r'\s+or\s+|,|/|\s+nor\s+', body):
        words = [w for w in re.findall(r'[a-z0-9]+', part) if stem_word(w) not in own]
        words = [w for w in words if stem_word(w) not in extra_ok] or words
        if words:
            alternatives.append(' '.join(words))
    if not alternatives:
        return 'skip'

    def states(where: Any) -> bool:
        return any(value_states(where, a) for a in alternatives)
    naming = [(k, v) for k, v in (block or {}).items() if key_names(need.prop, k, extra_ok)]
    if excluded:
        if any(states(v) for _, v in naming):
            return 'fails'
        return 'strong' if naming else ('fails' if states(text) else 'unknown')
    if naming:
        yes = [k for k, v in naming if states(v)]
        if len(yes) == len(naming):
            return 'strong'
        return 'conflict' if yes else 'fails'
    if any(states(v) for v in (block or {}).values()) or states(text):
        return 'weak'
    return 'unknown'


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
    if isinstance(value, str) and re.fullmatch(r'\s*\d+(\.0+)?\s*', value):
        return int(float(value))
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        found = re.search(r'-?\d+(?:[.,]\d+)?', value.replace(',', ''))
        if found and re.fullmatch(r'\s*[^\d]{0,4}-?\d+(?:\.\d+)?\s*[a-zA-Z%]{0,6}\s*', value.replace(',', '')):
            return float(found.group(0))
    return None


def schema_kinds(schema: dict[str, Any]) -> list[str]:
    kind = schema.get('type')
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def value_fits(value: Any, kind: str) -> bool:
    checks = {
        'string': lambda v: isinstance(v, str),
        'integer': lambda v: isinstance(v, int) and not isinstance(v, bool),
        'number': lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        'boolean': lambda v: isinstance(v, bool),
        'array': lambda v: isinstance(v, list),
        'object': lambda v: isinstance(v, dict),
        'null': lambda v: v is None,
    }
    test = checks.get(kind)
    return test(value) if test else True


def coerce(value: Any, kinds: list[str]) -> Any:
    for kind in kinds:
        if kind == 'integer' and isinstance(value, float) and value.is_integer():
            return int(value)
        if kind in ('integer', 'number') and isinstance(value, str) and re.fullmatch(r'\s*-?\d+(\.\d+)?\s*', value):
            number = float(value)
            return int(number) if kind == 'integer' or number.is_integer() else number
        if kind == 'string' and isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
            return value.strip().lower() == 'true'
        if kind == 'array' and value is not None and not isinstance(value, list):
            return [value]
    return value


def conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value towards a JSON schema (types, enums, lengths, bounds, required keys); report what cannot bend."""
    if not isinstance(schema, dict):
        return value
    kinds = schema_kinds(schema)
    if kinds and not any(value_fits(value, k) for k in kinds):
        value = coerce(value, kinds)
        if not any(value_fits(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    options = schema.get('enum')
    if isinstance(options, list) and value not in options:
        if isinstance(value, str):
            low = value.strip().lower()
            hit = [o for o in options if isinstance(o, str) and o.lower() == low]
            if not hit:
                hit = [o for o in options if isinstance(o, str) and (low in o.lower() or o.lower() in low) and len(low) >= 3]
            if len(hit) == 1 or (hit and all(h == hit[0] for h in hit)):
                value = hit[0]
        if value not in options:
            faults.append(f'{where}: allowed values are {options}')
    if isinstance(value, str):
        if isinstance(schema.get('maxLength'), int) and len(value) > schema['maxLength']:
            value = value[:schema['maxLength']]
        if isinstance(schema.get('minLength'), int) and len(value) < schema['minLength']:
            faults.append(f'{where} is too short')
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get('minimum'), schema.get('maximum')
        if isinstance(low, (int, float)) and value < low:
            faults.append(f'{where} is below {low}')
        if isinstance(high, (int, float)) and value > high:
            faults.append(f'{where} is above {high}')
    if isinstance(value, list):
        if isinstance(schema.get('maxItems'), int) and len(value) > schema['maxItems']:
            faults.append(f"{where} allows at most {schema['maxItems']} entries")
        if isinstance(schema.get('minItems'), int) and len(value) < schema['minItems']:
            faults.append(f"{where} needs at least {schema['minItems']} entries")
        if isinstance(schema.get('items'), dict):
            value = [conform(v, schema['items'], f'{where}[{i}]', faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get('properties') if isinstance(schema.get('properties'), dict) else {}
        if schema.get('additionalProperties') is False:
            for name in [n for n in value if n not in props]:
                faults.append(f'{where}.{name} is not an allowed argument')
        for name in schema.get('required') or []:
            if name not in value:
                faults.append(f'{where}.{name} is required')
        value = {k: conform(v, props.get(k), f'{where}.{k}', faults) if k in props else v for k, v in value.items()}
    return value


def looks_like_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not 3 <= len(text) <= 64 or ' ' in text:
        return False
    return named or bool(re.search(r'\d', text))


def guess_id_key(rows: list[dict[str, Any]]) -> str | None:
    """The key whose values look like identifiers in every row and are distinct across rows."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(looks_like_id(v, 'id' in str(key).lower()) for v in values):
            continue
        texts = [v.strip() for v in values]
        if len(rows) > 1 and len(set(texts)) < len(texts):
            continue
        score = ('id' in str(key).lower(), sum(len(v) for v in texts) / len(texts))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def attribute_block(record: Any) -> dict[str, Any] | None:
    """The nested object holding a listing's attributes: the largest dict of scalars inside the record."""
    if not isinstance(record, dict):
        return None
    best = None
    for value in record.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def free_text(record: Any) -> str:
    """Every string of three or more words inside a record, joined: the material notes and descriptions live in."""
    found: list[str] = []

    def walk(node: Any, depth: int) -> None:
        if depth > 6:
            return
        if isinstance(node, str):
            if len(node.split()) >= 3:
                found.append(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value, depth + 1)
        elif isinstance(node, list):
            for value in node:
                walk(value, depth + 1)
    walk(record, 0)
    return ' '.join(found)


def mentions_id(node: Any, ident: str, depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(mentions_id(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(mentions_id(v, ident, depth + 1) for v in node.values())
    return False


def codes_in(text: Any) -> list[str]:
    return list(dict.fromkeys(m.strip() for m in re.findall(CODE_RX, str(text or ''))))


def code_family(code: str) -> str:
    return re.sub(r'[- ]', '', code).upper()


def code_names(stated: str, code: str) -> bool:
    a, b = code_family(stated), code_family(code)
    return a == b or (len(a) >= 4 and (a.startswith(b) or b.startswith(a)))


def fit_marks(text: str) -> dict[str, Any]:
    """Compatibility statements in a listing's text: the codes it works with, its own code, exceptions, its series."""
    works: list[str] = []
    excluded: list[str] = []
    for found in re.finditer(FIT_WORDS + r'[:\s]+([^.;\n]{0,160})', text, re.I):
        chunk = found.group(1)
        if re.search(FIT_EXCEPT, chunk, re.I):
            parts = re.split(FIT_EXCEPT, chunk, maxsplit=1, flags=re.I)
            works += codes_in(parts[0])
            excluded += codes_in(parts[1] if len(parts) > 1 else '')
        else:
            works += codes_in(chunk)
    own = next((c for c in codes_in(re.sub(FIT_WORDS + r'[:\s]+[^.;\n]{0,160}', ' ', text, flags=re.I))), None)
    series = list(dict.fromkeys(re.findall(SERIES_RX, text)))
    return {'works': list(dict.fromkeys(works)), 'excluded': list(dict.fromkeys(excluded)), 'own': own, 'series': series}


def fit_verdict(part: dict[str, Any], device: dict[str, Any]) -> bool | None:
    """Whether a part's statements fit a device's codes: True, False, or None when neither side says."""
    device_codes = [c for c in [device.get('own')] + list(device.get('works') or []) if c]
    device_series = set(device.get('series') or [])
    if not device_codes and not device_series:
        return None
    if any(code_names(c, d) for c in part.get('excluded') or [] for d in device_codes):
        return False
    if any(code_names(c, d) for c in part.get('works') or [] for d in device_codes):
        return True
    if device_series and set(part.get('series') or []) & device_series:
        return True
    if part.get('works') or part.get('series'):
        return False
    return None


# ----------------------------------------------------------------------------------------------------------------
# the ledger: what the shopper wants
# ----------------------------------------------------------------------------------------------------------------
class Need:
    """One firm requirement on one item.

    `text` is "property: constraint". `source` says who stated it: 'request', 'shopper', 'store' (a note), or
    'chain' (resolved from another listing's page). A pointer need (`points`) takes its value from another listing;
    `resolved` holds the plain constraint once the chain was followed."""

    def __init__(self, text: str, source: str, points: bool = False):
        self.text = text.strip()
        self.source = source
        self.points = points
        self.resolved: str | None = None
        self.dropped = False

    @property
    def prop(self) -> str:
        return self.text.split(':', 1)[0].strip()

    @property
    def value(self) -> str:
        return self.text.split(':', 1)[1].strip() if ':' in self.text else ''

    @property
    def shown(self) -> str:
        if self.points and self.resolved:
            return f'{self.text} (resolved: {self.resolved})'
        return self.text


class Want:
    """One requested product."""

    def __init__(self, name: str, shelf: str, qty: int):
        self.name = name
        self.shelf = shelf
        self.qty = qty
        self.needs: list[Need] = []
        self.free: list[str] = []          # properties the shopper does not mind
        self.asked: list[str] = []         # properties already asked about
        self.lost: list[str] = []          # properties whose question the shopper did not follow

    def live(self) -> list[Need]:
        return [n for n in self.needs if not n.dropped]

    def has_need(self, text: str) -> bool:
        return any(n.text == text.strip() for n in self.needs if not n.dropped)

    def add_need(self, text: str, source: str, points: bool = False) -> Need | None:
        text = str(text or '').strip()
        if not text or self.has_need(text):
            return None
        for old in self.needs:
            if old.dropped and old.text == text:
                old.dropped = False
                old.source = source
                return old
        need = Need(text, source, points)
        self.needs.append(need)
        return need

    def drop_property(self, prop: str) -> list[str]:
        gone = []
        for need in self.live():
            if same_property(need.prop, prop) or need.text == prop:
                need.dropped = True
                gone.append(need.text)
        return gone


# ----------------------------------------------------------------------------------------------------------------
# the shop: this world's tools
# ----------------------------------------------------------------------------------------------------------------
class Shop:
    """The world's tools as the schema describes them, and which job each one does."""

    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = [t for t in tools if isinstance(t, dict) and isinstance(t.get('function'), dict) and isinstance(t['function'].get('name'), str)]
        self.schemas: dict[str, dict[str, Any]] = {}
        self.descriptions: dict[str, str] = {}
        for tool in self.tools:
            fn = tool['function']
            params = fn.get('parameters') if isinstance(fn.get('parameters'), dict) else {'type': 'object', 'properties': {}}
            self.schemas[fn['name']] = params
            self.descriptions[fn['name']] = str(fn.get('description') or '')
        self.roles: dict[str, str] = {}
        self.buy_args: dict[str, Any] = {}
        self.refuse_args: dict[str, Any] = {}
        self.browse: dict[str, Any] | None = None
        self.id_args: set[tuple[str, str]] = set()   # (tool, argument) pairs that took listing ids

    def names(self) -> list[str]:
        return list(self.schemas)

    def role(self, job: str) -> str | None:
        return self.roles.get(job)

    def role_of(self, name: str) -> str | None:
        return next((job for job, tool in self.roles.items() if tool == name), None)

    def props(self, name: str | None) -> dict[str, Any]:
        schema = self.schemas.get(name or '') or {}
        return schema.get('properties') if isinstance(schema.get('properties'), dict) else {}

    def required(self, name: str | None) -> list[str]:
        schema = self.schemas.get(name or '') or {}
        return [k for k in schema.get('required') or [] if isinstance(k, str)]

    def text_args(self, name: str | None) -> list[str]:
        return [k for k, v in self.props(name).items() if isinstance(v, dict) and 'string' in schema_kinds(v) and 'enum' not in v]

    def single_target(self, name: str | None) -> str | None:
        """The one text argument a single-listing tool takes, when the tool takes only that."""
        texts = self.text_args(name)
        required = self.required(name)
        target = next((k for k in required if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not name or not target or any(k != target for k in required):
            return None
        return target

    def list_target(self, name: str | None) -> str | None:
        props = self.props(name)
        target = next((k for k, v in props.items() if isinstance(v, dict) and 'array' in schema_kinds(v)), None)
        if not target or any(k != target for k in self.required(name)):
            return None
        return target

    def list_room(self, name: str | None) -> int:
        spec = self.props(name).get(self.list_target(name) or '') or {}
        return spec['maxItems'] if isinstance(spec.get('maxItems'), int) else 4

    def shape_args(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        faults: list[str] = []
        shaped = conform(dict(args), self.schemas.get(name), 'args', faults)
        return (shaped if isinstance(shaped, dict) else dict(args)), faults

    def tool_lines(self, skip: set[str] | None = None) -> str:
        lines = []
        for tool in self.tools:
            fn = tool['function']
            if skip and fn['name'] in skip:
                continue
            lines.append(f"- {fn['name']}: {fn.get('description', '')} | parameters: {dumps(fn.get('parameters') or {})}")
        return '\n'.join(lines)

    def enum_labels(self) -> set[str]:
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

    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.role('ask')
        if not name:
            return None
        props = self.props(name)
        texts = self.text_args(name)
        target = next((k for k in self.required(name) if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in self.required(name):
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get('enum'), list) and spec['enum']:
                    args[key] = spec['enum'][0]
                else:
                    return None
        shaped, faults = self.shape_args(name, args)
        return None if faults else shaped

    def set_browse(self, spec: Any) -> None:
        if not isinstance(spec, dict) or spec.get('tool') not in self.schemas:
            return
        props = self.props(spec['tool'])
        category, price = spec.get('category'), spec.get('max_price')
        if category not in props:
            return
        fixed = spec.get('fixed') if isinstance(spec.get('fixed'), dict) else {}
        self.browse = {
            'tool': spec['tool'],
            'category': category,
            'price': price if price in props else None,
            'fixed': {k: v for k, v in fixed.items() if k in props and k not in (category, price)},
        }

    def cart_sets_count(self, name: str) -> bool:
        """Whether the cart tool sets a line's count outright (a count argument allowing 0, no add/remove mode)."""
        props = self.props(name)
        if any(isinstance(s, dict) and isinstance(s.get('enum'), list) and len(s['enum']) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and 'integer' in schema_kinds(s) and s.get('minimum') == 0 for s in props.values())

    def cart_parts(self, name: str) -> dict[str, Any]:
        props = self.props(name)
        texts = self.text_args(name)
        target = next((k for t, k in self.id_args if t == name and k in texts), None) or next((k for k in self.required(name) if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and 'integer' in schema_kinds(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get('enum'), list) and len(v['enum']) >= 2), None)
        lines = None
        for key, spec in props.items():
            inner = spec.get('items') if isinstance(spec, dict) and 'array' in schema_kinds(spec) else None
            if isinstance(inner, dict) and isinstance(inner.get('properties'), dict):
                lines = (key, inner['properties'])
        return {'target': target, 'count': count, 'mode': mode, 'lines': lines, 'props': props}

    def remove_option(self, name: str) -> str | None:
        parts = self.cart_parts(name)
        if not parts['mode']:
            return None
        options = [str(o) for o in parts['props'][parts['mode']]['enum']]
        return next((o for o in options if re.search(r'remov|delet|drop|clear|take|discard|minus|subtract|out', o, re.I)), options[1])

    def add_option(self, name: str) -> str | None:
        parts = self.cart_parts(name)
        if not parts['mode']:
            return None
        off = self.remove_option(name)
        options = [str(o) for o in parts['props'][parts['mode']]['enum']]
        return next((o for o in options if o != off), options[0])


# ----------------------------------------------------------------------------------------------------------------
# memory: every listing seen, and what the store said about each
# ----------------------------------------------------------------------------------------------------------------
class Memory:
    """Listing records built from search rows and opened views, with the keys that matter in this store."""

    def __init__(self, shop: Shop):
        self.shop = shop
        self.rows: dict[str, dict[str, Any]] = {}        # merged record per listing id
        self.full: dict[str, dict[str, Any]] = {}        # ids whose attribute block has been read
        self.pages: dict[str, str] = {}                  # free text of an opened offer page / full record
        self.order: list[str] = []
        self.id_key: str | None = None
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.seller_key: str | None = None
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.views: dict[str, dict[str, int]] = {}       # ident -> tool -> version seen
        self.tool_keys: dict[str, set[str]] = {}
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.version = 0
        self.rules: list[tuple[str, str]] = []           # store rules pairing two values
        self.negations: dict[str, list[str]] = {}

    # -- identity --------------------------------------------------------------------------------------------
    def ident(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key and looks_like_id(node.get(self.id_key), True):
            return node[self.id_key].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = guess_id_key([node])
        if found and ('id' in str(found).lower() or node[found].strip() in self.rows):
            return node[found].strip()
        return None

    def learn_rows(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_key and all(looks_like_id(r.get(self.id_key), True) for r in rows):
            return True
        found = guess_id_key(rows)
        if found:
            self.id_key = found
            return True
        return False

    def absorb(self, observation: Any) -> list[str]:
        """Merge every listing record in an observation; return the ids seen."""
        seen: list[str] = []

        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.ident(node, in_list, depth)
                if ident:
                    if ident not in self.rows:
                        self.order.append(ident)
                        self.rows[ident] = {}
                    old = self.rows[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unavailable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unavailable.discard(ident)
                    old.update(node)
                    if attribute_block(node):
                        self.full[ident] = dict(old)
                    elif ident in self.full:
                        self.full[ident].update(node)
                    seen.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_rows(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)
        return seen

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.ident(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.ident(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.ident(r, True)] if i]
        return []

    def note_view(self, tool: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            self.tool_keys.setdefault(tool, set()).update(record.keys())
            self.views.setdefault(ident, {})[tool] = self.version
            text = free_text(record)
            if text:
                self.pages[ident] = (self.pages.get(ident, '') + ' ' + text).strip()[-4000:]

    def viewed_with(self, ident: str, tool: str | None) -> bool:
        return bool(tool) and self.views.get(ident, {}).get(tool) == self.version

    def opened_with(self, ident: str, tool: str | None) -> bool:
        """Whether this view of the listing was ever opened in the episode (what counts as reading its page)."""
        return bool(tool) and tool in self.views.get(ident, {})

    def fresh_view(self, ident: str, tool: str | None) -> bool:
        """Whether this view has shown everything it gives for this listing since the last change."""
        keys = self.tool_keys.get(tool or '')
        return bool(keys) and self.viewed_with(ident, tool) and ident not in self.gone

    # -- fields ----------------------------------------------------------------------------------------------
    def price_of(self, ident: str) -> float | None:
        record = self.rows.get(ident) or {}
        value = record.get(self.price_key) if self.price_key else None
        if value is None:
            for key, v in record.items():
                if re.search(r'price|cost|amount', plain(key)) and isinstance(v, (int, float)) and not isinstance(v, bool):
                    value = v
                    break
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r'\s*\d+(\.\d+)?\s*', value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        record = self.rows.get(ident) or {}
        value = record.get(self.stock_key) if self.stock_key else None
        if value is None:
            for key, v in record.items():
                if re.search(r'stock|avail|on hand|hand|buyable|sold', plain(key)) and isinstance(v, bool):
                    value = v
                    break
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def seller_of(self, ident: str) -> str:
        record = self.rows.get(ident) or {}
        if self.seller_key and isinstance(record.get(self.seller_key), str):
            return record[self.seller_key].strip()
        for key, value in record.items():
            if isinstance(value, str) and 1 < len(value.strip()) <= 40 and re.search(r'seller|shop|store|vendor|brand|maker|merchant|company|sold', plain(key)):
                return value.strip()
        return ''

    def category_key(self) -> str | None:
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r'\d{3}', value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.shop.enum_labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40 or key == self.seller_key:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def label_of(self, ident: str) -> str:
        key = self.category_key()
        return str((self.rows.get(ident) or {}).get(key) or '') if key else ''

    def record_text(self, ident: str, limit: int = 3000) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        if not record:
            return '(no record seen)'
        text = dumps(clip(record, 400))[:limit]
        page = self.pages.get(ident, '')
        if page and page[:200] not in text:
            text += ' | page text: ' + page[:900]
        return text

    def row_text(self, ident: str, limit: int = 900) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        return dumps(clip(record, 160))[:limit]

    def rules_text(self) -> str:
        parts = [f'{a} goes together with {b}' for a, b in self.rules[:12]]
        return '; '.join(parts)


# ----------------------------------------------------------------------------------------------------------------
# the oracle: model calls and store calls, within time and money
# ----------------------------------------------------------------------------------------------------------------
_llm = ProxyClient(timeout=60, max_retries=1)
_env = ProxyClient(timeout=90, max_retries=3)


class Oracle:
    """Talks to the models and to the store, keeping the episode inside its clock and its spending caps."""

    def __init__(self, binding: dict[str, Any], started: float, trace: list[dict[str, Any]]):
        self.binding = dict(binding)
        self.started = started
        self.trace = trace
        self.step = 0
        self.finished = False
        self.cost = 0.0
        self.calls = 0
        self.waited = 0.0
        self.no_credit = False
        self.broken = False
        self.busy: dict[str, float] = {}
        self.last_finish = ''

    def time_left(self) -> float:
        return EPISODE_SECONDS - (time.monotonic() - self.started)

    def deep_ok(self, limit: float = SPEND_SOFT) -> bool:
        return self.cost <= limit and self.cost + DEEP_ROOM <= SPEND_HARD and self.time_left() > EXPLORE_FLOOR

    def judges(self, limit: float = SPEND_SOFT) -> list[str]:
        if self.deep_ok(limit):
            return [DEEP, QUICK]
        return [QUICK] if self.cost <= SPEND_HARD else []

    # -- models ------------------------------------------------------------------------------------------------
    def think(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, tag: str = 'loop') -> dict[str, Any] | None:
        """One model reply. A rate-limited deep model hands over to its kin at once; other failures back off a little."""
        if self.time_left() < EXPLORE_FLOOR + 10:
            models = list(dict.fromkeys(QUICK if m == DEEP else m for m in models))
        for model in models:
            for choice in (DEEP_KIN if model == DEEP else (model,)):
                if self.calls >= CALL_CAP or self.time_left() < TIME_FLOOR or self.no_credit:
                    return None
                if time.monotonic() < self.busy.get(choice, 0.0):
                    continue
                self.calls += 1
                tries = 0
                status: Any = None
                grown = False
                while True:
                    _llm.timeout = max(8, min(55, int(self.time_left() - TIME_FLOOR)))
                    message, status = self.call_model(choice, messages, max_tokens, tools, tag)
                    if message is not None:
                        return message
                    if status == 0 and not grown and self.last_finish == 'length' and self.time_left() > EXPLORE_FLOOR:
                        grown = True
                        max_tokens = min(8000, max_tokens * 2)
                        continue
                    if status == 429 and model == DEEP:
                        self.busy[choice] = time.monotonic() + BUSY_SECONDS
                        self.calls -= 1
                        break
                    transient = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                    if not transient or tries >= (1 if status in (None, 408) else len(BACKOFFS)) or self.waited + BACKOFFS[tries] > BACKOFF_TOTAL:
                        break
                    if self.time_left() < BACKOFFS[tries] + TIME_FLOOR + 10:
                        break
                    pause = BACKOFFS[tries] * (0.8 + 0.4 * random.random())
                    self.waited += BACKOFFS[tries]
                    tries += 1
                    time.sleep(pause)
                if status == 402:
                    self.no_credit = True
                    return None
                if status != 429:
                    break
        return None

    def call_model(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, tag: str = '') -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            for index in {0, len(sent) - 1}:
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        if model.startswith('anthropic/') and max_tokens < 3000:
            max_tokens = 3000    # these models spend hidden reasoning tokens out of the same budget
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
            details = usage.get('prompt_tokens_details') if isinstance(usage.get('prompt_tokens_details'), dict) else {}
            self.trace.append({'kind': 'llm', 'tag': tag, 'model': model, 'in': usage.get('prompt_tokens'), 'cached': details.get('cached_tokens'), 'out': usage.get('completion_tokens'), 'cost': round(float(usage['cost']), 5) if isinstance(usage.get('cost'), (int, float)) else None})
            choice = data['choices'][0] or {}
            message = choice.get('message') or {}
            if message.get('content') or message.get('tool_calls'):
                return message, 200
            self.trace.append({'kind': 'llm_empty', 'model': model, 'finish': choice.get('finish_reason'), 'keys': sorted(message.keys()), 'reasoning': str(message.get('reasoning') or '')[:120], 'max_tokens': max_tokens})
            self.last_finish = str(choice.get('finish_reason') or '')
            return None, 0
        error = result.error or {}
        status = error.get('status')
        if isinstance(data, dict) and isinstance(data.get('error'), dict):
            status = as_int(data['error'].get('code')) or status
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:160]})
        return None, status

    def ask_json(self, prompt: str, models: list[str], max_tokens: int = 1800, tag: str = 'json') -> dict[str, Any] | None:
        for model in models:
            message = self.think([{'role': 'user', 'content': prompt}], [model], max_tokens, tag=tag)
            got = parse_json_object((message or {}).get('content'))
            if got is not None:
                return got
            if self.broken or self.no_credit or self.calls >= CALL_CAP:
                return None
        return None

    # -- the store -------------------------------------------------------------------------------------------
    def act(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One environment step. Recovers from step-number disagreements; None when the call could not be made."""
        if self.finished:
            return None
        payload = dict(self.binding)
        payload['step'] = self.step + 1
        payload['action'] = {'name': name, 'args': args}
        _env.timeout = max(10, min(90, int(self.time_left() - 4)))
        _env.max_retries = 3 if self.time_left() > EXPLORE_FLOOR else 1
        for _ in range(4):
            result = _env.post_verbose('/environment/call', payload)
            if isinstance(result.data, dict):
                data = result.data
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.finished = True
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if 'finish' in low or 'quarantin' in low or 'finaliz' in low or 'ended' in low:
                    self.finished = True
                    return None
                found = re.search(r'expected (\d+)', body)
                if found:
                    self.step = int(found.group(1)) - 1
                    payload['step'] = self.step + 1
                    continue
                if 'reuse' in low or 'different action' in low or 'already' in low:
                    self.step += 1
                    payload['step'] = self.step + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.broken = True
                return None
        return None


# ----------------------------------------------------------------------------------------------------------------
# the pilot: one episode
# ----------------------------------------------------------------------------------------------------------------
class Pilot:
    """Runs one shopping episode: reads the request, drives the store loop, interviews the shopper, resolves
    chains, checks the cart, and always ends the episode itself."""

    def __init__(self, problem: dict[str, Any]):
        started = time.monotonic()
        environment = problem['environment']
        self.view = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.trace: list[dict[str, Any]] = []
        self.oracle = Oracle(environment['binding'], started, self.trace)
        self.shop = Shop(self.view.get('tools') or [])
        self.memory = Memory(self.shop)
        steps = as_int(self.view.get('max_steps'))
        self.max_steps = steps if steps and steps > 0 else DEFAULT_STEPS
        self.oracle.finished = bool(self.view.get('finished'))
        # the ledger
        self.wants: list[Want] = []
        self.budget: float | None = None
        self.budget_unit = ''
        self.budget_scope = 'total'
        self.pick = 'any'
        self.pick_measure = ''
        self.in_stock_only = True
        self.msg_limit = 0
        self.msg_used = 0
        self.reserve = 0
        self.process: list[dict[str, Any]] = []
        self.rules: list[str] = []
        self.wishes: list[str] = []
        self.hints: list[dict[str, Any]] = []
        self.ask_first: list[str] = []
        self.quiet_decline: bool | None = None
        self.bound: list[tuple[int, int]] = []
        self.brief: dict[str, Any] = {}
        # conversation and bookkeeping
        self.chat: list[dict[str, Any]] = []
        self.loop_turns = 0
        self.done_log: list[str] = []
        self.done_steps: list[int] = []
        self.notes: list[str] = []
        self.queued: list[str] = []
        self.heard_lines: set[str] = set()
        self.qa: list[tuple[int, str, str]] = []
        self.asked_any = False
        self.interviewed = False
        self.searched: list[dict[str, Any]] = []
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.recent: list[str] = []
        self.pages_read = 0
        self.paged_obs: list[Any] = []
        # the cart
        self.cart_lines: dict[str, int] = {}
        self.cart_view: Any = None
        self.cart_at = -1
        self.cart_seen_at = -1
        self.cart_touched = False
        self.cart_said = ''
        self.built: list[str] = []
        self.dropped: set[str] = set()
        self.buys = 0
        self.refused_at = -1
        self.offer_seen: dict[str, int] = {}
        self.note_sold = False
        self.note_named = False
        self.sold_lines: set[str] = set()
        # judgement state
        self.checked: set[str] = set()
        self.verdicts: dict[str, bool] = {}
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.confirmed: str | None = None
        self.completed = False
        self.audited = 0
        self.refuse_blocks = 0
        self.pushed = False
        self.late_held = False
        self.line_holds = 0
        self.fit_holds = 0
        self.view_holds = 0
        self.empty_holds = 0
        self.gate_ok: set[tuple[str, str]] = set()
        self.gate_msgs = 0
        self.wider_held = False
        self.swept: set[str] = set()
        self.cheaper_marks: set[str] = set()
        self.chain_steps = 0
        self.chain_blocks = 0
        self.chain_misses: dict[str, int] = {}
        self.facts_from: dict[str, bool] = {}
        self.compare_ok = 0
        self.compare_bad = 0
        self.browse_misses = 0
        self.blocking: list[tuple[int, str]] = []
        self.plan_hinted: dict[int, set[str]] = {}
        self.chain_pass = 0
        self.shape_tries: dict[str, int] = {}
        self.shopper_mode = 'unknown'     # 'normal' answers lists of properties; 'tired' answers one property at a time
        self.listing = True               # one message may name several properties until the shopper objects
        self.better: list[str] = []
        self.swaps = 0
        self.blanks: dict[int, int] = {}      # per item: replies that disclosed nothing to a list question
        self.search_rows: dict[str, list[str]] = {}
        self.value_done: set[tuple[int, str]] = set()
        self.value_hits: dict[str, set[str]] = {}
        self.sold_fits: list[str] = []
        self.tried_words: set[str] = set()
        self.value_pass = False
        self.laddered = False
        self.last_total: int | None = None
        self.exact_cache: tuple[int, dict[int, set[str]]] = (-1, {})
        self.kw_dead: set[tuple[str, str]] = set()   # (argument, value) of a search mode that ignores the keyword
        self.code_removed: set[str] = set()
        self.code_added: set[str] = set()
        self.last_search_obs: dict[str, Any] = {}

    # -- budgets ------------------------------------------------------------------------------------------------
    def left(self) -> int:
        return self.max_steps - self.oracle.step

    def time_left(self) -> float:
        return self.oracle.time_left()

    def finished(self) -> bool:
        return self.oracle.finished

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def tail(self) -> int:
        """Steps kept back for the ending: the checkout, one retry after a refusal, and a message a process step
        at the order needs."""
        talk = any(p['when'] == 'order' and p['messages'] > 0 for p in self.process)
        spare = 4 if self.buys == 0 else 0
        return spare + (2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1)

    def gate_due(self, stage: str) -> bool:
        return any(p['when'] == stage for p in self.process) and (stage, self.cart_mark() + f'#{self.memory.version}') not in self.gate_ok

    def cart_mark(self) -> str:
        return dumps(sorted(self.cart_lines.items()))

    def room(self) -> int:
        """Steps still free for exploring: what is left minus the ending and one cart call per item."""
        return self.left() - self.tail() - len(self.wants) - 1

    # -- the brief ----------------------------------------------------------------------------------------------
    def read_request(self) -> None:
        prompt = BRIEF + '\nTOOLS:\n' + self.shop.tool_lines() + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nWHAT THE STORE SHOWS FIRST:\n' + dumps(clip(self.view.get('opening'), 1500))
        brief = self.oracle.ask_json(prompt, [QUICK], 4000, tag='brief') or {}
        weak = not as_list(brief.get('items')) or not isinstance(brief.get('roles'), dict) or not brief['roles'].get('buy')
        shelves = [plain(str(e.get('shelf') or '')) for e in as_list(brief.get('items')) if isinstance(e, dict)]
        shelves = [s for s in shelves if s]
        if not self.oracle.no_credit and (weak or len(shelves) != len(set(shelves))):
            again = self.oracle.ask_json(prompt, [DEEP, QUICK], 2500, tag='brief') or {}
            if as_list(again.get('items')) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.shop.roles = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.shop.schemas}
        self.guess_roles()
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.shop.role('buy'), self.shop.role('refuse')
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = (f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: '
                   + dumps(self.shop.schemas.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed '
                   'arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).')
            fixed = self.oracle.ask_json(ask, [QUICK, DEEP], 1200, tag='brief') or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and fixed['buy_args'] != fixed['refuse_args']:
                brief['buy_args'], brief['refuse_args'] = fixed['buy_args'], fixed['refuse_args']
        self.shop.buy_args = dict(brief['buy_args'])
        self.shop.refuse_args = dict(brief['refuse_args'])
        self.shop.set_browse(brief.get('browse'))
        # items and needs
        lookups = [str(x).strip() for x in as_list(brief.get('lookups')) if str(x).strip()]
        for entry in as_list(brief.get('items')):
            if not isinstance(entry, dict):
                continue
            quantity = as_int(entry.get('quantity'))
            want = Want(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1)
            for text in as_list(entry.get('needs')):
                text = str(text or '').strip()
                if text:
                    want.add_need(text, 'request', points=text in lookups or self.points_elsewhere(text))
            self.wants.append(want)
        for want in self.wants:
            for need in list(want.needs):
                if re.match(r'^(?:in[ _-]?stock|stock|availability|available|on hand|price|budget|cost|quantity|qty|units?|delivery|shipping|message|step)s?\b', plain(need.prop)) or re.search(r'\b(?:in stock|on hand|in the store|available)\b', need.value.lower()) and not re.search(r'[a-z]{4,}:', need.text):
                    if re.search(r'stock|hand|avail', need.text.lower()):
                        self.in_stock_only = True
                    want.needs.remove(need)
                    self.trace.append({'kind': 'need_dropped', 'need': need.text})
        self.merge_twins()
        if not self.wants:
            self.wants = [Want('the requested product', '', 1)]
        for pair in as_list(brief.get('fits')):
            if isinstance(pair, dict):
                part, main = as_int(pair.get('item')) or 0, as_int(pair.get('with')) or 0
                if part != main and 1 <= part <= len(self.wants) and 1 <= main <= len(self.wants) and (part, main) not in self.bound:
                    self.bound.append((part, main))
        budget = brief.get('budget')
        if isinstance(budget, str) and re.fullmatch(r'\s*\d+(\.\d+)?\s*', budget):
            budget = float(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = float(budget)
        self.budget_unit = str(brief.get('budget_unit') or '')
        self.budget_scope = 'each' if brief.get('budget_scope') == 'each' else 'total'
        self.pick = brief.get('pick') if brief.get('pick') in ('cheapest', 'most', 'least') else 'any'
        self.pick_measure = str(brief.get('pick_measure') or '')
        self.in_stock_only = brief.get('in_stock_only') is not False
        limit = as_int(brief.get('messages'))
        self.msg_limit = limit if limit is not None and limit >= 0 else (DEFAULT_MESSAGES if self.shop.role('ask') else 0)
        steps = as_int(brief.get('steps'))
        if steps is not None and 8 <= steps < self.max_steps:
            self.max_steps = steps
        self.rules = [str(r) for r in as_list(brief.get('rules')) if str(r).strip()]
        self.wishes = [str(w) for w in as_list(brief.get('wishes')) if str(w).strip()]
        self.ask_first = [str(a) for a in as_list(brief.get('ask_first')) if str(a).strip()]
        self.hints = [h for h in as_list(brief.get('hints')) if isinstance(h, dict) and str(h.get('text') or '').strip()]
        self.quiet_decline = brief.get('quiet_decline') if isinstance(brief.get('quiet_decline'), bool) else None
        for entry in as_list(brief.get('process')):
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                self.process.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, as_int(entry.get('messages')) or 0)})
        if self.shop.role('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    @staticmethod
    def points_elsewhere(text: str) -> bool:
        return bool(re.search(r"\b(?:same as|identical to|different from|unlike|partner|counterpart|within .{1,30} of|the one (?:named|pointed|on the page)|on the page of|listing (?:sold|from|by)|points? (?:at|to))\b", text, re.I))

    def guess_roles(self) -> None:
        """Fill the jobs the brief left out from the schemas: endings first, then the messaging and cart tools."""
        roles = self.shop.roles
        texts = {name: (name + ' ' + self.shop.descriptions.get(name, '')).lower() for name in self.shop.schemas}
        if 'refuse' not in roles:
            hit = next((n for n, t in texts.items() if re.search(r'give up|decline|no purchase|without (?:a )?purchase|cancel|abandon|skip|close out|pass', t) and n not in roles.values()), None)
            if hit:
                roles['refuse'] = hit
        if 'buy' not in roles:
            hit = next((n for n, t in texts.items() if re.search(r'check ?out|purchase|place (?:the )?order|buy|settle|commit|wrap', t) and n not in roles.values() and not self.shop.text_args(n)), None)
            if hit:
                roles['buy'] = hit
        if 'ask' not in roles:
            hit = next((n for n, t in texts.items() if re.search(r'shopper|customer|buyer|message to|ask the', t) and self.shop.text_args(n) and n not in roles.values()), None)
            if hit:
                roles['ask'] = hit
        if 'cart_view' not in roles:
            hit = next((n for n, t in texts.items() if re.search(r'cart|basket|bag', t) and not self.shop.props(n) and n not in roles.values()), None)
            if hit:
                roles['cart_view'] = hit
        if 'cart' not in roles:
            hit = next((n for n, t in texts.items() if re.search(r'cart|basket|bag', t) and self.shop.text_args(n) and n not in roles.values()), None)
            if hit:
                roles['cart'] = hit
        if 'search' not in roles:
            hit = next((n for n, t in texts.items() if re.search(r'search|find|look ?up|locate|seek|browse|list', t) and self.shop.text_args(n) and not self.shop.single_target(n) and n not in roles.values()), None)
            if hit:
                roles['search'] = hit
        if 'open' not in roles:
            singles = [n for n in self.shop.schemas if self.shop.single_target(n) and n not in roles.values()]
            if singles:
                roles['open'] = singles[0]
                if len(singles) > 1 and 'offer' not in roles:
                    roles['offer'] = singles[1]

    def merge_twins(self) -> None:
        merged: list[Want] = []
        for want in self.wants:
            twin = next((m for m in merged if want.shelf.strip() and m.qty == want.qty and plain(m.shelf) == plain(want.shelf)
                         and not any(same_property(a.prop, b.prop) and plain(a.value) != plain(b.value) for a in m.live() for b in want.live())), None)
            if twin is None:
                merged.append(want)
            else:
                for need in want.needs:
                    twin.add_need(need.text, need.source, need.points)
                self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': want.name})
        self.wants = merged

    # -- the ledger as text -------------------------------------------------------------------------------------
    def ledger(self) -> str:
        parts = []
        many = len(self.wants) > 1
        for number, want in enumerate(self.wants, 1):
            head = (f'item {number}: {want.name}' if many else f'item: {want.name}') + (f' (store label: {want.shelf})' if want.shelf else '') + f', quantity {want.qty}'
            needs = [n.shown for n in want.live()]
            text = head + '; requirements: ' + ('; '.join(needs) if needs else 'none beyond the kind of product')
            if want.free:
                text += '; shopper does not mind: ' + ', '.join(want.free)
            parts.append(text)
        exact = self.exact_labels()
        key = self.memory.category_key()
        for number, want in enumerate(self.wants, 1):
            if exact.get(number) and key:
                others = sorted({str(r.get(key)) for r in self.memory.rows.values() if isinstance(r.get(key), str) and plain(str(r.get(key))) not in exact[number] and content_words(str(r.get(key))) & content_words(want.name)})
                if others:
                    parts.append(f"item {number}'s kind is the store label {want.shelf!r}; rows labelled {', '.join(repr(o) for o in others[:3])} are not this item")
        if self.budget is not None:
            parts.append(f"budget: at most {self.budget:g} {self.budget_unit} {'for each item' if self.budget_scope == 'each' else 'in total'}".replace('  ', ' '))
        if self.pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif self.pick in ('most', 'least'):
            parts.append(f'pick: the {self.pick} {self.pick_measure} among fitting listings'.replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if self.in_stock_only:
            parts.append('in stock only')
        for wish in self.wishes:
            parts.append(f'wish (tie-breaker only): {wish}')
        for rule in self.rules:
            parts.append(f'rule from the request: {rule}')
        if self.quiet_decline is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        for entry in self.process:
            when = {'choose': 'before choosing', 'order': 'at the order (Ghost does it at checkout)', 'end': 'at an ending without a purchase (Ghost does it then)'}[entry['when']]
            parts.append(f"step the request asks for, {when}: {entry['step']}")
        if self.memory.rules:
            parts.append('store rules seen (paired values are interchangeable): ' + self.memory.rules_text())
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.shop.role('offer')):
            bare = sorted(t for t in self.page_tools() | {self.shop.role('offer') or ''} if t and self.facts_from.get(t) is False)
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show the offer page (price, stock, notes) only" if bare else ''))
        return '\n'.join('- ' + p for p in parts)

    def needs_now(self) -> list[tuple[int, Need]]:
        return [(n, need) for n, want in enumerate(self.wants, 1) for need in want.live()]

    def status(self) -> str:
        left = self.left()
        text = f'[Ghost] step {self.oracle.step} of {self.max_steps} used, {left} left; {self.msgs_free()} message(s) free.'
        if left <= self.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; the no-purchase ending is for a stated requirement nothing meets, never for a shortage of steps.'
        elif left <= 6 or (left <= len(self.wants) + 5 and self.buys == 0):
            text += ' Few steps left: set the cart to your picks and check out now.'
        elif self.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif self.buys == 0 and not self.cart_touched and self.oracle.step >= self.max_steps // 2:
            text += ' Half the steps are used and the cart is empty: cart each item\'s best fitting listing now.'
        missing = self.items_without_lines()
        if missing and self.cart_touched and left > 6:
            text += ' Items with no cart line yet: ' + ', '.join(self.wants[n - 1].name for n in missing) + '.'
        pending = [need.text for _, need in self.needs_now() if need.points and not need.resolved]
        if pending and self.buys == 0:
            text += '\n[chains] Ghost resolves these from the pages they point at: ' + ' | '.join(pending[:3])
        if self.leads:
            text += '\n[review candidates] ' + ', '.join(self.leads)
        plan = self.fit_plan() if self.bound and self.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if self.notes:
            text += '\n[changes so far] ' + ' | '.join(self.notes[-4:])
        if self.time_left() < EXPLORE_FLOOR + 45:
            text += f'\n[time] about {max(0, int(self.time_left()))} s are left: settle and check out within the next few calls.'
        text += '\n[ledger]\n' + self.ledger()
        return text

    def items_without_lines(self) -> list[int]:
        key = self.memory.category_key()
        if not key:
            return []
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0 and not self.memory.sold_out(i)]
        out = []
        for n, want in enumerate(self.wants, 1):
            known = any(self.owns(want, str(r.get(key) or '')) for r in self.memory.rows.values())
            if known and not any(self.owns(want, self.memory.label_of(i)) for i in ids):
                out.append(n)
        return out

    # -- tools by job -------------------------------------------------------------------------------------------
    def page_tools(self) -> set[str]:
        page = self.shop.role('offer') or self.shop.role('open')
        return {t for t in (page, self.shop.role('compare')) if t}

    def facts_tool(self) -> str | None:
        known = next((t for t, ok in self.facts_from.items() if ok and self.shop.single_target(t)), None)
        if known:
            return known
        tool = self.shop.role('open')
        return tool if self.facts_from.get(tool or '') is not False and self.shop.single_target(tool) else None

    def offer_tool(self) -> str | None:
        return self.shop.role('offer') or self.shop.role('open')

    def view_tools(self) -> list[str]:
        """Every single-listing view tool in this store, attribute view first."""
        out = []
        for name in (self.facts_tool(), self.shop.role('open'), self.shop.role('offer')):
            if name and name not in out and self.shop.single_target(name) and name not in (self.shop.role('cart'), self.shop.role('buy'), self.shop.role('refuse'), self.shop.role('ask'), self.shop.role('search')):
                out.append(name)
        return out

    def readers(self) -> set[str]:
        return {self.shop.role(r) for r in ('search', 'open', 'offer', 'compare')} - {None}

    def endings(self) -> set[str]:
        return {self.shop.role('buy'), self.shop.role('refuse')} - {None}

    # -- observations ---------------------------------------------------------------------------------------------
    def show(self, observation: Any) -> str:
        text = dumps(clip(observation, ROW_CHARS))
        return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + '...(cut)'

    def observe(self, tool: str, observation: Any) -> str:
        """Absorb an observation: records, views, free-text notes. Returns the meaning of any note read."""
        self.memory.absorb(observation)
        if tool in self.readers() - {self.shop.role('search')} or self.shop.single_target(tool):
            self.memory.note_view(tool, observation)
            if self.shop.single_target(tool):
                listing = self.memory.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.facts_from[tool] = self.facts_from.get(tool, False) or bool(attribute_block(observation))
                    if tool == self.offer_tool():
                        self.offer_seen[listing] = self.oracle.step
                    self.harvest_rules(listing, observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            self.queued.append(f'(seen on {tool}) {meaning}')
        return meaning

    def harvest_rules(self, ident: str, observation: Any) -> None:
        """Pull pairing rules out of a page's text by pattern: "a goes together with b", "pairs a with b, c with d",
        "a is the partner of b". Sentence by sentence, so a rule never spans two statements."""
        text = free_text(observation)
        value = r"[a-z0-9][a-z0-9 /+'.-]{0,28}?"
        for sentence in re.split(r'(?<=[.;!?])\s+', text):
            low = sentence.strip().lower()
            if not low:
                continue
            for a, b in re.findall(r"(?:^|\b(?:its|their|the|that|this|rule for [a-z ]+:)\s+)(" + value + r")\s+(?:goes|go) (?:together )?with\s+(" + value + r")(?=$|[,.;]|\s+(?:and|as|for)\b)", low):
                self.add_rule(a, b)
            for clause in re.findall(r"\bpairs?\s+(.{3,160}?)(?=[.;]|$)", low):
                for a, b in re.findall(r"(" + value + r")\s+with\s+(" + value + r")(?=$|,|\s+and\b)", clause):
                    self.add_rule(a, b)
            for a, b in re.findall(r"(?:^|\b(?:its|the|that|this)\s+)(" + value + r")\s+is the (?:partner|counterpart) (?:of|to)\s+(" + value + r")(?=$|[,.;])", low):
                self.add_rule(a, b)

    def add_rule(self, a: str, b: str) -> None:
        def clean(v: str) -> str:
            v = v.strip().lower().strip('.,;:')
            v = re.sub(r"^(?:its|the|their|that|this|a|an|our|and|also|store rule for [a-z ]+:)\s+", '', v).strip()
            words = v.split()
            if len(words) > 1 and re.fullmatch(r'(?:pattern|style|material|colou?r|type|value|feature|features|closure|upholstery|finish|size|card|class)', words[-1]):
                v = ' '.join(words[:-1])
            return v
        a, b = clean(a), clean(b)
        if not a or not b or a == b or len(a) > 40 or len(b) > 40 or a in ('it', 'they', 'which', 'one') or b in ('it', 'they', 'which', 'one'):
            return
        if (a, b) in self.memory.rules or (b, a) in self.memory.rules:
            return
        self.memory.rules.append((a, b))
        self.trace.append({'kind': 'rule', 'pair': [a, b]})

    def loose_lines(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.memory.ident(observation):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not self.memory.ident(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.heard_lines]

    def take_note(self, lines: list[str]) -> str:
        self.heard_lines.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + self.ledger() + '\n\nTHE TEXT:\n' + '\n'.join(lines)
        got = self.oracle.ask_json(prompt, [QUICK, QUICK_SPARE], 2000, tag='note') or {}
        gone = [str(g) for g in as_list(got.get('gone')) if isinstance(g, (str, int))]
        self.memory.unavailable.update(g for g in gone if g in self.memory.rows)
        self.note_sold = bool(gone) or got.get('sold_out') is True
        self.note_named = any(g in self.memory.rows for g in gone)
        if self.note_sold and len(lines) == 1:
            self.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = float(budget)
            changed.append(f'budget now {budget:g}')
        for entry in [e for e in as_list(got.get('drop')) if isinstance(e, dict) and str(e.get('need') or '').strip()]:
            want = self.wants[min(max(1, as_int(entry.get('item')) or 1), len(self.wants)) - 1]
            target = str(entry['need']).strip()
            hit = [n for n in want.live() if n.text == target] or [n for n in want.live() if same_property(n.prop, target.split(':')[0])][:1]
            for need in hit:
                need.dropped = True
            changed.append(f'dropped {target}')
        for entry in [e for e in as_list(got.get('add')) if isinstance(e, dict) and str(e.get('need') or '').strip()]:
            want = self.wants[min(max(1, as_int(entry.get('item')) or 1), len(self.wants)) - 1]
            text = str(entry['need']).strip()
            if want.add_need(text, 'store', points=self.points_elsewhere(text)):
                changed.append(f'added {text}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if self.pick == 'cheapest' and (changed or self.note_sold):
            self.wider_held = False
            self.cheaper_marks = set()
        if changed or summary:
            self.memory.version += 1
            self.recent = []
            self.checked = set()
            self.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    # -- result shapes and paging ---------------------------------------------------------------------------------
    def read_shape(self, tool: str, observation: Any) -> dict[str, Any] | None:
        if self.memory.shapes.get(tool) is not None or self.shape_tries.get(tool, 0) >= 2:
            return self.memory.shapes.get(tool)
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.shape_tries[tool] = self.shape_tries.get(tool, 0) + 1
        self.memory.shapes[tool] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + tool + ' | parameters: ' + dumps(self.shop.schemas.get(tool) or {}) + '\n\nRESULT (first rows only):\n' + dumps(clip(sample, 120))[:4000]
        got = self.oracle.ask_json(prompt, [QUICK, DEEP], 1500, tag='shape') or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = self.shop.props(tool)

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        shape = {'rows': rows, 'count': named(got.get('count'), observation), 'next': named(got.get('next'), observation), 'next_arg': named(got.get('next_arg'), props),
                 'price': named(got.get('price'), first), 'stock': named(got.get('stock'), first), 'seller': named(got.get('seller'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        self.memory.shapes[tool] = shape
        self.memory.price_key = self.memory.price_key or shape['price']
        self.memory.stock_key = self.memory.stock_key or shape['stock']
        self.memory.seller_key = self.memory.seller_key or shape['seller']
        self.trace.append({'kind': 'shape', 'tool': tool, 'shape': shape})
        return shape

    def paging(self, tool: str, observation: Any) -> tuple[int, int, str, Any] | None:
        shape = self.read_shape(tool, observation)
        if not shape or not shape['count'] or not shape['next'] or not shape['next_arg'] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next'])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), shape['next_arg'], value

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.memory.price_of(i) for i, _ in self.memory.records_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def more_pages(self, tool: str, args: dict[str, Any], observation: Any) -> str:
        """For a cheapest/most/least pick, read the following pages of a short result itself."""
        self.paged_obs = []
        plan = self.paging(tool, observation) if self.pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or '').strip())
        if value is None or later or count <= shown:
            return ''
        if self.pick == 'cheapest' and self.price_sorted(observation):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.pages_read + 2 <= PAGE_LIMIT and self.room() >= 10:
            pages = 2
        if pages > 2 or self.pages_read + pages > PAGE_LIMIT or self.room() - pages < 8:
            return f"\n[Ghost] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: narrow the search with the requirement's value or page on with {param}={dumps(value)}."
        texts = []
        for _ in range(pages):
            call, faults = self.shop.shape_args(tool, {**args, param: value})
            if faults:
                break
            reply = self.oracle.act(tool, call)
            if reply is None or self.finished():
                break
            self.pages_read += 1
            seen = reply.get('observation')
            self.observe(tool, seen)
            self.paged_obs.append(seen)
            shown_text = self.show(seen)
            self.read_cache[tool + dumps(call)] = (self.memory.version, self.oracle.step, shown_text)
            texts.append(f'{param}={dumps(value)}: {shown_text}')
            after = self.paging(tool, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.trace.append({'kind': 'pages', 'tool': tool, 'read': len(texts)})
        rest = f"\n[Ghost] {count} listings matched and {shown * (len(texts) + 1)} rows were read; narrow the search or page on to see the rest." if value is not None and texts else ''
        return '\n[Ghost read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest if texts else ''

    def cap_ignored(self, tool: str, args: dict[str, Any], observation: Any) -> str:
        props = self.shop.props(tool)

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search(r'price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r'max|most|under|below|limit|cap|upto|up_to|budget|spend', k, re.I) and priced(k)]
        if not caps or not self.memory.price_key:
            return ''
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.memory.records_in(observation) if (self.memory.price_of(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join(f'{k}={dumps(o[0])}' for k, o in others.items() if o) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Ghost] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        if self.pick != 'cheapest' or not self.memory.price_key:
            return ''
        found = {i for seen in observations for i, _ in self.memory.records_in(seen)}
        rows = [(self.memory.price_of(i), i) for i in found if self.memory.buyable(i) and self.memory.price_of(i) is not None]
        if len(rows) < 2:
            return ''
        return '\n[Ghost] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in sorted(rows)[:6])

    def evidence(self, observation: Any) -> str:
        """Which attribute keys of the opened listings name a requirement's property, with their values."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.memory.ident(node, False, depth)
                if ident and attribute_block(node) and ident not in ids:
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.needs_now()
        if not ids or not needs:
            return ''
        lines = []
        for ident in ids[:12]:
            text = self.grade_text(ident)
            if text:
                lines.append(f'{ident}: ' + text)
        if not lines:
            return ''
        return ("[evidence, graded by code from the keys naming each property: ok = every such key states the value; FAILS = such "
                "keys exist and say another value (do not buy); DOUBTFUL = such keys disagree; title-only = no such key, the title or "
                "text states it] " + ' | '.join(lines))

    # -- items and categories -----------------------------------------------------------------------------------
    def exact_labels(self) -> dict[int, set[str]]:
        """Per item, the category labels seen whose words are exactly the item's words ("Pants & Jeans" for "pants or
        jeans"). Once one is seen, rows under a partial label ("Jeans") are not that item."""
        count = len(self.memory.rows)
        if self.exact_cache[0] == count:
            return self.exact_cache[1]
        key = self.memory.category_key()
        found: dict[int, set[str]] = {n: set() for n in range(1, len(self.wants) + 1)}
        if key:
            for record in self.memory.rows.values():
                label = record.get(key)
                if not isinstance(label, str) or not label.strip():
                    continue
                words = content_words(label)
                if not words:
                    continue
                for number, want in enumerate(self.wants, 1):
                    if words == content_words(want.name) or (want.shelf and (words == content_words(want.shelf) or plain(label) == plain(want.shelf))):
                        found[number].add(plain(label))
                        if not want.shelf:
                            want.shelf = label.strip()
        if key and self.memory.rows:
            request_words = content_words(self.request)
            seen_labels = {str(r.get(key)).strip() for r in self.memory.rows.values() if isinstance(r.get(key), str) and r.get(key).strip()}
            for number, want in enumerate(self.wants, 1):
                if found[number]:
                    continue
                mine = content_words(want.name)
                candidates = {label: content_words(label) for label in seen_labels if content_words(label) and content_words(label) <= request_words and content_words(label) & mine}
                for label, words in candidates.items():
                    if any(words > other for other in candidates.values()):
                        found[number].add(plain(label))
                        if not want.shelf:
                            want.shelf = label
                        self.trace.append({'kind': 'exact_label', 'item': number, 'label': label})
            for number, want in enumerate(self.wants, 1):
                if found[number] or want.shelf:
                    continue
                mine = content_words(want.name)
                if any(content_words(str(r.get(key) or '')) & mine for r in self.memory.rows.values()):
                    continue
                rows = list(self.memory.rows) if len(self.wants) == 1 else self.rows_searched_for(want)
                labels = [str(self.memory.rows[i].get(key) or '').strip() for i in rows if isinstance(self.memory.rows.get(i, {}).get(key), str)]
                labels = [x for x in labels if x]
                if len(labels) >= 3:
                    top = max(set(labels), key=labels.count)
                    if labels.count(top) * 2 >= len(labels):
                        want.shelf = top
                        found[number].add(plain(top))
                        self.trace.append({'kind': 'adopt_label', 'item': number, 'label': top})
        self.exact_cache = (count, found)
        return found

    def rows_searched_for(self, want: Want) -> list[str]:
        """Rows returned by searches whose keywords named this item."""
        mine = content_words(want.name)
        out: list[str] = []
        for mark, ids in self.search_rows.items():
            tool = self.shop.role('search') or ''
            args = parse_json_object(mark[len(tool):]) if mark.startswith(tool) else None
            if isinstance(args, dict) and any(isinstance(v, str) and content_words(v) & mine for v in args.values()):
                out += [i for i in ids if i not in out]
        return out

    def owner_of(self, label: str) -> int:
        if not label:
            return 0
        words = plain(label)
        long = content_words(label)
        exact = self.exact_labels()
        best, top = 0, 0.0
        for number, want in enumerate(self.wants, 1):
            if words == plain(want.shelf or ''):
                score = 3.0
            elif words == plain(want.name):
                score = 2.0
            else:
                mine = content_words(want.shelf or '') | content_words(want.name)
                score = len(long & mine) / max(1, len(long | mine))
                if exact.get(number) and words not in exact[number]:
                    score = 0.0
            if score > top:
                best, top = number, score
        return best

    def firm_marked(self, need: Need) -> bool:
        """Whether the request marks a requirement as insisted on or to be checked by the shopper: such a requirement
        is never set aside as story context."""
        words = [w for w in re.findall(r'[a-z0-9]{3,}', need.value.lower()) if w not in VALUE_STOP][:2]
        if not words:
            return False
        for sentence in re.split(r'(?<=[.!?])\s+', self.request):
            low = sentence.lower()
            if all(w in low for w in words) and re.search(r"\b(?:check|verify|double-check|make sure|confirm|insist|has to|have to|got to|gotta|must|non-negotiable|sticking to|holding firm|only taking|won't take|no substitute)\b", low):
                return True
        return False

    # -- evidence grades ----------------------------------------------------------------------------------------
    def kind_ok_words(self) -> frozenset[str]:
        """Words that may qualify an attribute key without changing the property it names: the items' names, the
        category labels seen, and the words most row titles of the store share (the product nouns)."""
        words: set[str] = set()
        for want in self.wants:
            words |= prop_stems(want.name) | prop_stems(want.shelf or '')
        key = self.memory.category_key()
        counts: dict[str, int] = {}
        rows = list(self.memory.rows)[:300]
        for ident in rows:
            record = self.memory.rows[ident]
            label = record.get(key) if key else None
            if isinstance(label, str):
                words |= prop_stems(label)
            for w in set(prop_stems(self.title_of(ident))):
                if len(w) >= 3:
                    counts[w] = counts.get(w, 0) + 1
        if len(rows) >= 4:
            words |= {w for w, n in counts.items() if n * 10 >= len(rows) * 3}
        return frozenset(words)

    def grade(self, ident: str, need: Need) -> str:
        record = self.memory.full.get(ident)
        if record is None:
            return 'unread'
        label = self.memory.label_of(ident)
        if label and need.value and not need.points and not re.match(EXCL_RX, need.value.lower()) and value_states(label, need.value):
            return 'strong'
        block = attribute_block(record) or {}
        title = ' '.join(str(v) for k, v in record.items() if isinstance(v, str) and k not in block and len(str(v)) <= 200
                         and not re.search(r'desc|detail|note|about|summary|text|body|info|remark|spec', plain(k)))
        return grade_record(block, title, need, self.kind_ok_words())

    def grades_of(self, ident: str, number: int) -> list[tuple[Need, str]]:
        return [(need, self.grade(ident, need)) for need in self.wants[number - 1].live()]

    def listing_score(self, ident: str, number: int) -> int:
        """Lower is better: the sum of grade ranks over the item's requirements (0 when every graded one is strong)."""
        return sum(GRADE_RANK.get(g, 3) for _, g in self.grades_of(ident, number))

    def clean_fit(self, ident: str, number: int) -> bool:
        """Every requirement graded strong by code (none left to a limit or chain that code does not judge)."""
        grades = [g for _, g in self.grades_of(ident, number)]
        return all(g == 'strong' for g in grades)

    def likely_fit(self, ident: str, number: int) -> bool:
        """Every requirement graded by code as met in some way (agreeing keys, a title, or keys that disagree): under a
        cheapest pick a cheaper listing graded so beats a dearer one with stronger evidence."""
        grades = [g for _, g in self.grades_of(ident, number)]
        return all(g in ('strong', 'weak', 'conflict', 'skip') for g in grades)

    def grade_text(self, ident: str, number: int | None = None) -> str:
        """'pattern: logo -> FAILS (fa.pattern=print, pattern=print); collar: not crew neck -> ok (neckline=hooded)'."""
        if ident not in self.memory.full:
            return ''
        number = number or self.owner_of(self.memory.label_of(ident)) or 1
        block = attribute_block(self.memory.full[ident]) or {}
        ok_words = self.kind_ok_words()
        names = {'strong': 'ok', 'weak': 'title-only', 'conflict': 'DOUBTFUL (keys disagree)', 'fails': 'FAILS', 'unknown': 'nothing states it', 'skip': 'not graded by code'}
        parts = []
        for need, grade in self.grades_of(ident, number):
            if grade == 'skip':
                continue
            keys = [f'{k}={str(v)[:40]}' for k, v in block.items() if key_names(need.prop, k, ok_words)][:3]
            parts.append(f"{need.text} -> {names.get(grade, grade)}" + (f" ({', '.join(keys)})" if keys else ''))
        return '; '.join(parts)

    def owns(self, want: Want, label: str) -> bool:
        return bool(label) and want in self.wants and self.owner_of(label) == self.wants.index(want) + 1

    def item_ids(self, number: int) -> list[str]:
        key = self.memory.category_key()
        want = self.wants[number - 1]
        return [i for i in self.memory.order if key and self.owns(want, str((self.memory.rows.get(i) or {}).get(key) or ''))]

    def item_groups(self) -> list[list[str]]:
        key = self.memory.category_key()
        groups: dict[str, list[str]] = {}
        for ident in self.memory.order:
            label = str(self.memory.rows[ident].get(key) or '') if key else ''
            groups.setdefault(label, []).append(ident)
        matched = [ids for label, ids in groups.items() if self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.wants))]

    def records_of_item(self, number: int) -> int:
        return sum(1 for i in self.item_ids(number) if i in self.memory.full and attribute_block(self.memory.full[i]))

    def shelf_of(self, number: int) -> str:
        key = self.memory.category_key()
        want = self.wants[number - 1]
        labels = [str(r.get(key) or '') for r in self.memory.rows.values()] if key else []
        labels = [label for label in labels if label and self.owns(want, label)]
        return max(set(labels), key=labels.count) if labels else want.shelf

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Attribute keys seen on opened listings, grouped by category label: (label, key, carried-by, samples)."""
        key = self.memory.category_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.memory.order:
            record = self.memory.full.get(ident)
            if not record:
                continue
            block = attribute_block(record) or {}
            label = str(record.get(key) or 'listings') if key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            done: set[str] = set()
            for name, value in block.items():
                words = plain(name)
                if not words or words in done:
                    continue
                done.add(words)
                slot = group['keys'].setdefault(words, {'n': 0, 'values': []})
                slot['n'] += 1
                text = str(value).strip().lower()
                if text and text not in slot['values']:
                    slot['values'].append(text)
        matched = {label for label in groups if self.owner_of(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]['n']):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group['keys'].items(), key=lambda kv: (-kv[1]['n'], len(kv[0]))):
                values = slot['values']
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if ' / ' in v) * 2 >= len(values):
                    continue
                twin = next((spec for other, spec in kept if same_property(other, name) and (set(spec['values']) & set(values) or len(name.split()) > 1)), None)
                if twin is not None:
                    twin['n'] = max(twin['n'], slot['n'])
                    twin['values'] += [v for v in values if v not in twin['values']]
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            for name, slot in kept[:14]:
                table.append((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]))
        return table

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        want = self.wants[number - 1]
        settled = [n.prop for n in want.live()] + want.free + want.asked
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        if len(self.wants) == 1 and not rows:
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        measured = {key: sum(1 for v in values if re.match(r'^\s*[\d.,/x -]+\s*[a-z]{0,6}\s*$', v)) * 2 > len(values) for _, key, _, values in table}
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any(same_property(s, key) for s in settled) and not re.search(NOT_PROPERTY, key)]
        return [k for k in keys if not re.search(CATCH_ALL, k)] + [k for k in keys if re.search(CATCH_ALL, k)]

    # -- the interview ------------------------------------------------------------------------------------------
    def ready_to_ask(self) -> bool:
        if self.interviewed or not self.shop.role('ask') or self.msgs_free() <= 0:
            return False
        if self.left() <= self.tail() + len(self.wants) + 2 or self.time_left() < EXPLORE_FLOOR + 35:
            return False
        if all(self.records_of_item(n) >= 1 for n in range(1, len(self.wants) + 1)) and len(self.memory.full) >= 2:
            return True
        return self.oracle.step >= max(6, self.max_steps // 4)

    def plan_questions(self, count: int) -> dict[int, list[str]]:
        table = self.key_table()
        settled = [f'item {n}: {w.name} - settled by the request: ' + ('; '.join(x.text for x in w.live()) or 'nothing') + (f"; shopper does not mind: {', '.join(w.free)}" if w.free else '') + (f"; already asked: {', '.join(w.asked)}" if w.asked else '') for n, w in enumerate(self.wants, 1)]
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r} (item {self.owner_of(label) or "?"}):')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + (' [several features in one: ask last]' if re.search(CATCH_ALL, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + self.request + '\n\nITEMS:\n' + '\n'.join(settled)
        if self.ask_first:
            prompt += '\n\nTHE REQUEST SAYS TO ASK ABOUT:\n' + '\n'.join(self.ask_first)
        if self.hints:
            prompt += '\n\nSTORY HINTS:\n' + '\n'.join(f"- item {h.get('item')}: \"{h.get('text')}\" -> {h.get('property')}" for h in self.hints[:8])
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        plan = self.oracle.ask_json(prompt, [DEEP, QUICK], 2500, tag='plan') or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.wants) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in as_list(plan.get('ask')):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, as_int(pick.get('item') or 1) or 1), len(self.wants))
            label = re.split(r'\s+(?:and|or)\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            want = self.wants[number - 1]
            if any(same_property(label, s) for s in [n.prop for n in want.live()] + want.free + want.asked):
                continue
            if all(plain(label) != plain(other) for other in lanes[number]):
                lanes[number].append(label)
            if pick.get('hinted') is True:
                hinted[number].add(plain(label))
        for number in lanes:
            keys = self.lane_keys(number, table)
            known = [key for label, key, _, _ in table if self.owner_of(label) == number or len(self.wants) == 1]
            if len(known) >= 3:
                listed = [x for x in lanes[number] if any(same_property(x, k) for k in known)]
                invented_hinted = [x for x in lanes[number] if x not in listed and plain(x) in hinted[number]]
                invented_other = [x for x in lanes[number] if x not in listed and plain(x) not in hinted[number]]
                lanes[number] = listed[:1] + invented_hinted + listed[1:] + invented_other
            for key in keys:
                if all(plain(key) != plain(other) for other in lanes[number]):
                    lanes[number].append(key)
        self.plan_hinted = hinted
        self.trace.append({'kind': 'plan', 'ask': clip(plan.get('ask'), 120), 'lanes': {n: v[:6] for n, v in lanes.items()}})
        return lanes

    def ask_order(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int]) -> list[int]:
        """Items in the order the next message serves them: items not yet asked about first, then items with no need
        found yet, items with story hints before the rest, then the least asked."""
        return sorted(lanes, key=lambda n: (spent.get(n, 0) > 0, found.get(n, 0) > 0, not any(plain(x) in self.plan_hinted.get(n, set()) for x in lanes[n]), spent.get(n, 0), n))

    def open_labels(self, number: int, lanes: dict[int, list[str]]) -> list[str]:
        want = self.wants[number - 1]
        return [x for x in lanes[number] if not any(same_property(x, s) for s in want.asked + want.free + [n.prop for n in want.live()])]

    def compose_question(self, parts: list[tuple[int, list[str]]]) -> str:
        def joined(labels: list[str]) -> str:
            return labels[0] if len(labels) == 1 else f"{', '.join(labels[:-1])} or {labels[-1]}"
        if len(parts) == 1:
            number, labels = parts[0]
            return f'What {joined(labels)} would you like for the {self.wants[number - 1].name}?'
        pieces = []
        for index, (number, labels) in enumerate(parts):
            name = self.wants[number - 1].name
            pieces.append(f'what {joined(labels)} would you like for the {name}' if index == 0 else f'what {joined(labels)} for the {name}')
        text = ', and '.join(pieces)
        return 'W' + text[1:] + '?'

    def fit_question(self, parts: list[tuple[int, list[str]]]) -> tuple[list[tuple[int, list[str]]], str]:
        """Shorten the lists until the message fits the messaging tool's length limit."""
        props = self.shop.props(self.shop.role('ask'))
        limit = min([v['maxLength'] for v in props.values() if isinstance(v, dict) and isinstance(v.get('maxLength'), int)] or [800])
        parts = [(n, list(l)) for n, l in parts if l]
        while parts:
            text = self.compose_question(parts)
            if len(text) <= limit:
                return parts, text
            longest = max(range(len(parts)), key=lambda i: len(parts[i][1]))
            if len(parts[longest][1]) > 1:
                parts[longest] = (parts[longest][0], parts[longest][1][:-1])
            else:
                parts = parts[:-1]
        return [], ''

    def interview(self) -> str:
        """Ask the shopper. The shopper answers about the first property the message names that they have a view on, so
        a message names many properties (of several items when messages are scarce); a reply that discloses nothing
        settles only the first label, and the rest are rotated to the front for the next message. A shopper who asks
        for one thing at a time gets one property per message from then on."""
        self.interviewed = True
        if self.msgs_free() <= 0 or not self.shop.role('ask'):
            return ''
        if self.room() >= 6 and self.time_left() > EXPLORE_FLOOR + 40:
            self.widen()
        lanes = self.plan_questions(LIST_SIZE)
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        told: list[str] = []
        asked: list[tuple[int, str, str]] = []
        while self.msgs_free() > 0 and not self.finished() and self.left() > self.tail() + len(self.wants) + 1 and self.time_left() > EXPLORE_FLOOR + 20:
            avail = [(n, self.open_labels(n, lanes)) for n in self.ask_order(lanes, found, spent)]
            avail = [(n, l) for n, l in avail if l]
            if not avail:
                break
            if not self.listing:
                parts = [(avail[0][0], avail[0][1][:1])]
            else:
                uncovered = [n for n, _ in avail if spent.get(n, 0) == 0]
                size = min(MIX_ITEMS, len(avail), max(1, -(-len(uncovered) // max(1, self.msgs_free())))) if uncovered else 1
                chosen = avail[:size]
                per = LIST_SIZE if len(chosen) == 1 else (LIST_MIXED if len(chosen) == 2 else 3)
                parts = [(n, l[:per]) for n, l in chosen]
            parts, text = self.fit_question(parts)
            if not parts:
                break
            listed = [l for _, ls in parts for l in ls]
            sent = self.send([(parts[0][0], text)], labels=listed)
            if not sent:
                break
            asked += sent
            for n, _ in parts:
                spent[n] = spent.get(n, 0) + 1
            had = [len(w.live()) for w in self.wants]
            heard = self.hear(sent, parts)
            self.take_heard(heard, told, sent, parts)
            gained = {n: [x for x in w.live()[had[n - 1]:]] for n, w in enumerate(self.wants, 1) if len(w.live()) > had[n - 1]}
            for n in gained:
                found[n] = found.get(n, 0) + 1
            crowded = any(as_int(k) == 1 for k in as_list(heard.get('crowded')))
            lost = any(as_int(k) == 1 for k in as_list(heard.get('lost')))
            self.settle(parts, gained, crowded, lost, lanes)
            self.trace.append({'kind': 'reply_mode', 'listing': self.listing, 'gained': {n: [x.text for x in v] for n, v in gained.items()}, 'crowded': crowded, 'lost': lost, 'parts': parts})
        searched = self.value_search(list(range(1, len(self.wants) + 1)), everything=True)
        self.value_pass = True
        if told:
            self.memory.version += 1
            self.checked = set()
        if not asked:
            return ''
        summary = "Ghost asked the shopper (each message was one step):\n" + '\n'.join(f'Q: {q}\nA: {a}' for _, q, a in asked)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none') + '. Apply them to every pick.'
        pointed = [need.text for _, need in self.needs_now() if need.points and not need.resolved]
        if pointed:
            chained = self.resolve_pointers()
            if chained:
                summary += '\n' + chained
        if searched:
            summary += '\n' + searched
        if self.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(self.queued)
            self.queued = []
        return summary

    def settle(self, parts: list[tuple[int, list[str]]], gained: dict[int, list[Need]], crowded: bool, lost: bool, lanes: dict[int, list[str]]) -> None:
        """Apply what a reply tells about the properties the message named. The shopper answers about the first named
        property they track, so labels named before an answered one are not tracked; after a reply that discloses
        nothing, only the first label is settled and the rest move to the front in reverse order."""
        def mark_used(number: int, labels: list[str]) -> None:
            want = self.wants[number - 1]
            want.asked.extend(x for x in labels if x not in want.asked)

        def rotate(number: int, labels: list[str]) -> None:
            lane = lanes.get(number) or []
            front = [x for x in labels if x in lane]
            lanes[number] = front + [x for x in lane if x not in front]
        if gained:
            answered = False
            for number, labels in parts:
                needs = gained.get(number) or []
                if needs and not answered:
                    answered = True
                    hit = next((i for i, label in enumerate(labels) if any(same_property(label, n.prop) for n in needs)), None)
                    if hit is None:
                        rotate(number, labels)
                    else:
                        mark_used(number, labels[:hit])
                        rotate(number, labels[hit + 1:])
                elif not answered:
                    mark_used(number, labels)
                else:
                    rotate(number, labels)
            return
        if crowded and len([l for _, ls in parts for l in ls]) > 1:
            self.listing = False
            self.shopper_mode = 'tired'
            self.trace.append({'kind': 'lists_off'})
            return
        if lost and self.listing and len([l for _, ls in parts for l in ls]) > 1:
            for number, labels in parts:
                mark_used(number, labels)
            return
        if lost:
            for number, labels in parts:
                mark_used(number, labels)
                self.wants[number - 1].lost.extend(x for x in labels if x not in self.wants[number - 1].lost)
            return
        first_number, first_labels = parts[0]
        mark_used(first_number, first_labels[:1])
        if len(first_labels) == 1 and len(parts) == 1:
            if first_labels[0] not in self.wants[first_number - 1].free:
                self.wants[first_number - 1].free.append(first_labels[0])
            return
        # the shopper answered about the first listed property they track, which they do not mind: the next message
        # must put a different tracked property first, so the rest is reversed once, then ordered middle-out
        self.blanks[first_number] = self.blanks.get(first_number, 0) + 1
        rest = first_labels[1:]
        if self.blanks[first_number] % 2 == 1:
            order = list(reversed(rest))
        else:
            centre = (len(rest) - 1) / 2
            order = [rest[i] for i in sorted(range(len(rest)), key=lambda i: (abs(i - centre), i))]
        lane = lanes.get(first_number) or []
        lanes[first_number] = [x for x in order if x in lane] + [x for x in lane if x not in order]
        for number, labels in parts[1:]:
            rotate(number, labels)

    def send(self, questions: list[tuple[int, str]], floor: int | None = None, labels: list[str] | None = None) -> list[tuple[int, str, str]]:
        tool = self.shop.role('ask')
        floor = self.tail() + 1 if floor is None else floor
        sent: list[tuple[int, str, str]] = []
        for number, text in questions:
            if not tool or self.finished() or self.left() <= floor or self.msg_used >= self.msg_limit:
                break
            args = self.shop.message_args(text)
            if not args:
                break
            reply = self.oracle.act(tool, args)
            if reply is None:
                break
            self.msg_used += 1
            self.asked_any = True
            answer = reply.get('observation')
            words = self.spoken(answer)
            sent.append((number, text, words))
            self.qa.append((number, text, words))
            self.done_log.append(f'step {self.oracle.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.oracle.step)
            self.trace.append({'kind': 'ask', 'q': text, 'a': words[:300], 'about': labels or []})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def spoken(self, observation: Any) -> str:
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
        return ' '.join(found)[:800] if found else dumps(observation)[:400]

    def hear(self, sent: list[tuple[int, str, str]], parts: list[tuple[int, list[str]]] | None = None) -> dict[str, Any]:
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join(f'item {n}: {w.name}' for n, w in enumerate(self.wants, 1))
        about = '; '.join(f"item {n} ({self.wants[n - 1].name}): {', '.join(labels)}" for n, labels in (parts or [])) or 'see the question'
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(f"{i}. (properties asked about: {about}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = self.oracle.ask_json(prompt, [QUICK, DEEP], 2000, tag='hear') or {}
        if any(isinstance(e, dict) and e.get('points') is True for e in as_list(got.get('needs'))) and self.oracle.deep_ok():
            again = self.oracle.ask_json(prompt, [DEEP], 2000, tag='hear')
            if again:
                got = again
        return got

    def take_heard(self, heard: dict[str, Any], told: list[str], sent: list[tuple[int, str, str]] | None = None, parts: list[tuple[int, list[str]]] | None = None) -> None:
        many = len(self.wants) > 1
        listed = [l for _, ls in (parts or []) for l in ls]
        for entry in as_list(heard.get('needs')):
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, as_int(entry.get('item')) or 1), len(self.wants))
            text = str(entry['need']).strip()
            points = entry.get('points') is True or self.points_elsewhere(text)
            need = self.wants[number - 1].add_need(text, 'shopper', points)
            if need:
                told.append(f'item {number}: {text}' if many else text)
        for entry in as_list(heard.get('drop')):
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                want = self.wants[min(max(1, as_int(entry.get('item')) or 1), len(self.wants)) - 1]
                gone = want.drop_property(str(entry['property']).split(':')[0])
                told.extend(f'dropped {g}' for g in gone)
        if len(listed) <= 1:
            for entry in as_list(heard.get('free')):
                if isinstance(entry, dict) and entry.get('property'):
                    number = min(max(1, as_int(entry.get('item')) or 1), len(self.wants))
                    if str(entry['property']) not in self.wants[number - 1].free:
                        self.wants[number - 1].free.append(str(entry['property']))
        budget = heard.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = float(budget)
            told.append(f'budget now {budget:g}')

    # -- searching with what the shopper said -----------------------------------------------------------------
    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool, its fixed arguments and its keyword field, learnt from the searches made so far."""
        tool = self.shop.role('search')
        if not tool:
            return None
        props = self.shop.props(tool)
        texts = self.shop.text_args(tool)
        named = [k for k in texts if re.search(r'(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword|words|text', k, re.I)]
        key = self.memory.category_key()
        labels = {str(r.get(key) or '').strip().lower() for r in self.memory.rows.values()} - {''} if key else set()
        for args in reversed(self.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I) and args[k].strip().lower() not in labels]
            if fields:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get('enum'), list) or k in self.shop.required(tool))}
                keep = {k: v for k, v in keep.items() if not isinstance(v, str) or k in self.shop.required(tool) or v.strip().lower() not in labels}
                return tool, self.keyword_mode(tool, keep), field
        if named:
            field = named[0]
            keep: dict[str, Any] = {}
            for k in self.shop.required(tool):
                spec = props.get(k) or {}
                if k != field and isinstance(spec.get('enum'), list) and spec['enum']:
                    keep[k] = next((o for o in spec['enum'] if re.search(r'word|phrase|text|typed|keyword|search|match', str(o), re.I)), spec['enum'][0])
            return tool, self.keyword_mode(tool, keep), field
        return None

    def keyword_mode(self, tool: str, keep: dict[str, Any]) -> dict[str, Any]:
        """Fixed search arguments with every mode value set to one in which keywords are read: a mode seen to ignore
        the keyword is replaced, and a mode whose name promises words, text or typing is preferred."""
        props = self.shop.props(tool)
        out = dict(keep)
        for key, value in keep.items():
            options = (props.get(key) or {}).get('enum')
            if not isinstance(options, list) or not options or not isinstance(value, str):
                continue
            wordy = [o for o in options if re.search(r'word|phrase|text|typed|keyword|search|match|query|free', str(o), re.I)]
            if (key, value) in self.kw_dead:
                out[key] = next((o for o in wordy if (key, o) not in self.kw_dead), next((o for o in options if o != value and (key, o) not in self.kw_dead), value))
            elif wordy and value not in wordy:
                out[key] = wordy[0]
        return out

    @staticmethod
    def positive_part(value: str) -> str:
        """The searchable part of a requirement's value: "baggy" from "baggy, not slim or straight"; '' for limits,
        exclusions, pointers and contentless words."""
        if not value:
            return ''
        parts = [p.strip() for p in re.split(r',|;|\s+but\s+|\(|\)', value) if p.strip()]
        parts = [p for p in parts if not re.match(EXCL_RX, p.lower()) and not re.search(r'\b(?:not|no|other than|unlike|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\b', p, re.I)]
        if parts and re.search('[a-z]', parts[0], re.I) and parts[0].strip().lower() not in EMPTY_VALUES:
            return parts[0]
        return ''

    def value_words(self, number: int) -> list[str]:
        """Plain requirement values worth searching for: the positive part of each value, never limits or pointers."""
        out = []
        for n, need in self.needs_now():
            if n != number or need.points:
                continue
            part = self.positive_part(need.value)
            if part and part not in out:
                out.append(part)
        return out

    def search_words(self, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
        args, faults = self.shop.shape_args(tool, {**base, field: words})
        if faults or self.finished():
            return []
        reply = self.oracle.act(tool, args)
        if reply is None:
            return []
        self.observe(tool, reply.get('observation'))
        self.read_shape(tool, reply.get('observation'))
        self.searched = (self.searched + [dict(args)])[-8:]
        self.tried_words.update(w for v in args.values() if isinstance(v, str) for w in re.findall(r'[a-z0-9]{3,}', v.lower()))
        self.last_search_obs[tool + dumps(args)] = reply.get('observation')
        self.last_search_obs['__last__'] = (tool, args, reply.get('observation'))
        return [i for i, _ in self.memory.records_in(reply.get('observation'))]

    def kind_rows(self, number: int, found: list[str], below: float | None = None) -> list[str]:
        key = self.memory.category_key()
        want = self.wants[number - 1]
        wanted = set(re.findall(r'[a-z0-9]{3,}', ' '.join(self.value_words(number)).lower()))
        rows = [i for i in dict.fromkeys(found) if self.memory.buyable(i) and (not key or self.owns(want, str((self.memory.rows.get(i) or {}).get(key) or '')))]
        if below is not None:
            rows = [i for i in rows if (self.memory.price_of(i) if self.memory.price_of(i) is not None else below) < below]
        return sorted(rows, key=lambda i: (i in self.memory.full, -len(wanted & set(re.findall(r'[a-z0-9]{3,}', dumps(self.memory.rows.get(i) or {}).lower()))), self.memory.price_of(i) or 0))

    def page_more(self, tool: str, args: dict[str, Any], pages: int) -> tuple[int, list[str]] | None:
        """Read up to `pages` further pages of the search just made with `args`; returns (steps used, ids seen)."""
        if pages <= 0:
            return None
        mark = tool + dumps(args)
        seen_obs = self.last_search_obs.get(mark)
        last = self.last_search_obs.get('__last__')
        if seen_obs is None and isinstance(last, tuple) and last[0] == tool:
            args, seen_obs = last[1], last[2]
        plan = self.paging(tool, seen_obs) if seen_obs is not None else None
        if not plan or plan[3] is None or plan[0] <= plan[1]:
            return None
        used, ids = 0, []
        value = plan[3]
        for _ in range(pages):
            if value is None or self.finished() or self.room() < 4:
                break
            call, faults = self.shop.shape_args(tool, {**args, plan[2]: value})
            if faults:
                break
            reply = self.oracle.act(tool, call)
            if reply is None:
                break
            used += 1
            seen = reply.get('observation')
            self.observe(tool, seen)
            ids += [i for i, _ in self.memory.records_in(seen)]
            after = self.paging(tool, seen)
            value = after[3] if after else None
        return used, ids

    def covered(self, number: int, value: str) -> bool:
        """Whether two read buyable listings of the item's kind already state this value with agreeing keys."""
        needs = [n for n in self.wants[number - 1].live() if self.positive_part(n.value) == value]
        if not needs:
            return True
        hits = 0
        for ident in self.item_ids(number):
            if ident in self.memory.full and self.memory.buyable(ident) and all(self.grade(ident, n) in ('strong', 'skip') for n in needs):
                hits += 1
                if hits >= 2:
                    return True
        return False

    def value_search(self, numbers: list[int], everything: bool = False) -> str:
        """Search each plain requirement's value as keywords (the bare value, then with the category's words, then with
        the product's words), read the first rows of the item's kind, and page on while no read row states the value
        with agreeing keys; shopper answers always, request needs when `everything`."""
        plan = self.search_template()
        if not plan or self.room() < 6 or self.time_left() < EXPLORE_FLOOR + 25:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = min(8, max(2, self.room() // 2))
        for number in numbers:
            want = self.wants[number - 1]
            values = self.value_words(number)
            fresh = [v for v in values if (everything or any(n.source == 'shopper' and n.value == v for n in want.live()))
                     and (number, v) not in self.value_done and not self.covered(number, v)]
            for value in fresh[:2]:
                if budget <= 1:
                    break
                self.value_done.add((number, value))
                cap = self.budget if isinstance(self.budget, (int, float)) else None
                label = self.shelf_of(number)
                if label and value_states(label, value):
                    continue
                needs_here = [n for n in want.live() if self.positive_part(n.value) == value]

                def strong_known() -> bool:
                    return bool(needs_here) and any(i in self.memory.full and self.memory.buyable(i) and all(self.grade(i, n) == 'strong' for n in needs_here) for i in self.item_ids(number))

                def within(ids: list[str]) -> list[str]:
                    return [i for i in self.kind_rows(number, ids) if cap is None or (self.memory.price_of(i) or 0) <= cap or self.budget_scope == 'each']
                noun = ' '.join(sorted(content_words(want.name))[:3]) or ' '.join(stem_word(w) for w in plain(want.name).split()[-2:])
                forms = [value]
                if label and plain(label) != plain(want.name):
                    forms.append(' '.join([value] + [w for w in plain(label).split() if w not in ('&', 'and', '/')][:3]))
                forms.append(' '.join([value, noun]))
                for words in dict.fromkeys(forms):
                    if budget <= 1 or self.finished():
                        break
                    rows = within(self.search_words(tool, base, field, words))
                    budget -= 1
                    self.value_hits.setdefault(value, set()).update(rows)
                    opened = self.fill_facts([i for i in rows if i not in self.memory.full][:min(2, budget)], self.tail() + len(self.wants) + 2)
                    budget -= len(opened)
                    if needs_here and not strong_known() and budget >= 3:
                        more = self.page_more(tool, {**base, field: words}, 2)
                        if more:
                            budget -= more[0]
                            new_rows = [i for i in within(more[1]) if i not in rows]
                            rows += new_rows
                            self.value_hits[value].update(new_rows)
                            extra = self.fill_facts([i for i in new_rows if i not in self.memory.full][:min(2, max(0, budget - 1))], self.tail() + len(self.wants) + 2)
                            budget -= len(extra)
                            opened += extra
                    seen.append(f'{words!r}: ' + (', '.join(rows[:6]) or 'no row of that kind') + (' | opened: ' + ' | '.join(opened)[:2000] if opened else ''))
                    if (needs_here and strong_known()) or (not needs_here and len(rows) >= 3):
                        break
        if not seen:
            return ''
        self.trace.append({'kind': 'value_search', 'seen': [x[:200] for x in seen]})
        return "[Ghost searched the requirements' values as keywords and read the first rows of that kind] " + '\n'.join(seen)

    # -- opening listings ---------------------------------------------------------------------------------------
    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        tool = self.facts_tool()
        target = self.shop.single_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.memory.rows and i not in self.memory.full and not self.memory.sold_out(i)]:
            if self.left() <= keep or self.finished() or self.facts_from.get(tool) is False or self.time_left() < TIME_FLOOR + 10:
                break
            args, faults = self.shop.shape_args(tool, {target: ident})
            if faults:
                break
            reply = self.oracle.act(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(tool, seen)
            texts.append(f'{ident}: ' + self.show(seen)[:1500])
        if texts:
            self.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def open_view(self, tool: str, ident: str) -> dict[str, Any] | None:
        target = self.shop.single_target(tool)
        if not target or self.finished():
            return None
        args, faults = self.shop.shape_args(tool, {target: ident})
        if faults:
            return None
        reply = self.oracle.act(tool, args)
        if reply is None:
            return None
        self.observe(tool, reply.get('observation'))
        return reply

    def complete_views(self, ids: list[str], keep: int) -> list[str]:
        """Open every single-listing view this store has for each id that lacks one: the attribute view and the
        offer page both count as reading the listing's page before buying it."""
        done = []
        for ident in ids:
            for tool in self.view_tools():
                if self.left() <= keep or self.finished() or self.time_left() < TIME_FLOOR + 10:
                    return done
                if self.memory.opened_with(ident, tool) or ident in self.memory.gone:
                    continue
                reply = self.open_view(tool, ident)
                if reply is None:
                    return done
                done.append(f'{tool} {ident}: ' + self.show(reply.get('observation'))[:600])
        if done:
            self.trace.append({'kind': 'views', 'n': len(done)})
        return done

    def widen(self) -> None:
        """Open a couple of listings per item, so the interview can see which attributes they carry."""
        tool = self.shop.role('compare')
        target = self.shop.list_target(tool)
        if tool and target and self.facts_from.get(tool) is not False and self.room() >= 4:
            cap = self.shop.list_room(tool)
            ranked = self.item_groups()
            share = max(1, cap // max(1, len(ranked)))
            picks: list[str] = []
            for ids in ranked:
                picks += [i for i in ids if i not in self.memory.full and self.memory.buyable(i)][:share]
            picks = picks[:cap]
            if len(picks) >= 2:
                args, faults = self.shop.shape_args(tool, {target: picks})
                if not faults:
                    before = len(self.memory.full)
                    reply = self.oracle.act(tool, args)
                    if reply is not None:
                        seen = reply.get('observation')
                        self.observe(tool, seen)
                        rows = self.memory.records_in(seen)
                        if rows:
                            self.facts_from[tool] = self.facts_from.get(tool, False) or any(attribute_block(r) for _, r in rows)
                        self.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(self.memory.full) - before})
                        if len(self.memory.full) > before:
                            return
        tool = self.facts_tool()
        if not tool or not self.shop.single_target(tool):
            return
        wanted = {w for _, need in self.needs_now() for w in re.findall(r'[a-z0-9]{3,}', need.value.lower())}
        each = 2 if len(self.wants) > 1 else 3
        picks = []
        for number, ids in enumerate(self.item_groups(), 1):
            opened = sum(1 for i in ids if i in self.memory.full)
            pool = [i for i in ids if i not in self.memory.full and self.memory.buyable(i)]
            if self.pick == 'cheapest':
                pool.sort(key=lambda i: self.memory.price_of(i) if self.memory.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall(r'[a-z0-9]{3,}', dumps(self.memory.rows.get(i) or {}).lower()))))
            picks += pool[:max(0, each - opened)]
        self.fill_facts(picks[:WIDEN_PER_ITEM], self.tail() + len(self.wants) + 4)

    # -- cheapest pick: look below the cart's price -----------------------------------------------------------
    def list_category(self, number: int, cap: float | None, more: int = 0) -> list[str] | None:
        spec = self.shop.browse
        label = self.shelf_of(number)
        if not spec or not label or self.browse_misses >= 2 or (cap is not None and not spec['price']):
            return None
        args = {**spec['fixed'], spec['category']: label}
        if cap is not None:
            args[spec['price']] = cap
        args, faults = self.shop.shape_args(spec['tool'], args)
        if faults or self.finished():
            self.browse_misses += 1
            return None
        reply = self.oracle.act(spec['tool'], args)
        if reply is None:
            self.browse_misses += 1
            return None
        seen = reply.get('observation')
        self.observe(spec['tool'], seen)
        self.read_shape(spec['tool'], seen)
        found = [i for i, _ in self.memory.records_in(seen)]
        page = self.paging(spec['tool'], seen)
        self.last_total = page[0] if page else None
        if isinstance(seen, dict) and seen.get('error'):
            self.browse_misses += 1
            return None
        for _ in range(more):
            page = self.paging(spec['tool'], seen)
            if not page or page[3] is None or self.finished():
                break
            call, faults = self.shop.shape_args(spec['tool'], {**args, page[2]: page[3]})
            reply = self.oracle.act(spec['tool'], call) if not faults else None
            if reply is None:
                break
            seen = reply.get('observation')
            self.observe(spec['tool'], seen)
            found += [i for i, _ in self.memory.records_in(seen)]
        return found

    def other_values(self, number: int) -> list[str]:
        table = self.key_table()
        keys = self.lane_keys(number, table)[:3]
        said = ' '.join(self.value_words(number)).lower() + ' ' + self.wants[number - 1].name.lower()
        lanes = []
        for key in keys:
            values = next((v for label, name, _, v in table if name == key and (len(self.wants) == 1 or self.owner_of(label) == number)), [])
            lanes.append([v for v in values if v and len(v) <= 20 and v not in said and re.search('[a-z0-9]', v)][:2])
        return [lane[k] for k in range(2) for lane in lanes if k < len(lane)]

    def cheaper_sweep(self) -> bool:
        key = self.memory.category_key()
        plan = self.search_template()
        mark = self.cart_mark() + f'#{self.memory.version}'
        if mark in self.swept or self.room() < 2 or self.time_left() < EXPLORE_FLOOR + 25 or not key:
            return False
        self.swept.add(mark)
        limit = self.oracle.step + min(SWEEP_STEPS, self.room())
        ran = False
        for line in self.cart_ids():
            price = self.memory.price_of(line)
            label = str((self.memory.rows.get(line) or {}).get(key) or '')
            number = next((n for n, want in enumerate(self.wants, 1) if label and self.owns(want, label)), 0)
            if price is None or not number or self.oracle.step >= limit:
                continue
            found = self.list_category(number, int(price) - 1 if price > 1 else price * 0.9, 1 if limit - self.oracle.step >= 2 else 0)
            if found is not None:
                ran = True
            elif plan:
                base = self.value_words(number)[:2]
                for other in self.other_values(number)[:2]:
                    if self.oracle.step >= limit:
                        break
                    self.search_words(plan[0], plan[1], plan[2], ' '.join(base + [other, self.wants[number - 1].name]))
                    ran = True
        self.trace.append({'kind': 'sweep', 'ran': ran})
        return ran

    def cheaper_rows(self) -> list[str]:
        if self.pick != 'cheapest':
            return []
        mark = self.cart_mark() + f'#{self.memory.version}'
        key = self.memory.category_key()
        if mark in self.cheaper_marks or not key:
            return []
        self.cheaper_marks.add(mark)
        cart = self.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = self.memory.price_of(line)
            kind = str((self.memory.rows.get(line) or {}).get(key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.memory.order if i not in cart and i not in self.memory.full and self.memory.buyable(i)
                    and str((self.memory.rows.get(i) or {}).get(key) or '').strip().lower() == kind
                    and (self.memory.price_of(i) if self.memory.price_of(i) is not None else float('inf')) < price]
            pool.sort(key=lambda i: self.memory.price_of(i) or 0.0)
            picks += [i for i in pool[:CHEAPER_OPENS if len(cart) == 1 else 2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], self.tail() + 2)

    def keyword_ignored(self, tool: str, args: dict[str, Any], ids: list[str]) -> str:
        """A keyword that changed nothing against an earlier search differing only in that keyword was ignored by
        this search mode: say so, with the mode values that could take it."""
        plan = self.search_template()
        field = plan[2] if plan else None
        if not field or not isinstance(args.get(field), str) or not ids:
            return ''
        bare = {k: v for k, v in args.items() if k != field}
        for mark, rows in self.search_rows.items():
            if not mark.startswith(tool) or rows != ids:
                continue
            other = parse_json_object(mark[len(tool):])
            if not isinstance(other, dict) or other.get(field) == args.get(field) or {k: v for k, v in other.items() if k != field} != bare:
                continue
            props = self.shop.props(tool)
            modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list) and k != field}
            self.kw_dead.update(modes.items())
            hint = ''.join(f"; the other values of {k} are {', '.join(dumps(o) for o in props[k]['enum'] if o != v)}" for k, v in modes.items())
            return f"\n[Ghost] {field}={args[field]!r} changed nothing: these rows are exactly those of the same search with {field}={other.get(field)!r}, so this mode ignores the keyword{hint}."
        return ''

    def code_swap(self) -> str:
        """Cheapest pick, one item: when a read, buyable, cheaper listing of the kind states every plain requirement
        with agreeing keys, put it in the cart instead of the current line."""
        if self.pick != 'cheapest' or len(self.wants) != 1 or self.left() < 4 or self.time_left() < TIME_FLOOR + 25:
            return ''
        cart = self.cart_ids()
        if len(cart) != 1:
            return ''
        line = cart[0]
        price = self.memory.price_of(line)
        if price is None:
            return ''
        cheaper = [i for i in self.item_ids(1) if i != line and i in self.memory.full and self.memory.buyable(i) and i not in self.code_removed
                   and self.memory.price_of(i) is not None and self.memory.price_of(i) < price and self.likely_fit(i, 1)]
        if not cheaper or line in self.code_added:
            return ''
        best = min(cheaper, key=lambda i: self.memory.price_of(i) or 0.0)
        done = self.set_cart({best: self.wants[0].qty}, [line])
        if not done:
            return ''
        self.code_removed.add(line)
        self.code_added.add(best)
        self.trace.append({'kind': 'code_swap', 'in': best, 'out': line})
        return f"Ghost swapped the cart to {best} ({self.memory.price_of(best):g}), a cheaper listing that meets every requirement by its record: " + '; '.join(done)

    def price_ladder(self) -> str:
        """Cheapest pick, one item: list the category under falling price caps until the cheapest rows are in view,
        read the cheap rows that mention the value and, once a fitting listing is known, every cheaper row below it;
        cart the cheapest read listing that code grades as fitting."""
        if self.pick != 'cheapest' or self.laddered or len(self.wants) != 1 or self.finished():
            return ''
        self.laddered = True
        spec = self.shop.browse
        if not spec or not spec['price'] or not self.shelf_of(1) or self.room() < 8 or self.time_left() < EXPLORE_FLOOR + 40:
            return ''
        budget = float(self.budget) if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
        cap = budget
        lines: list[str] = []
        for _ in range(3):
            if self.room() < 6 or self.finished():
                break
            found = self.list_category(1, cap)
            if found is None:
                break
            total = self.last_total
            kind = self.kind_rows(1, found)
            prices = sorted(p for p in (self.memory.price_of(i) for i in kind) if p is not None)
            lines.append((f"category listing under {cap:g}" if cap is not None else 'category listing') + f": {len(found)} rows shown of {total if total is not None else 'an unknown total'}")
            if total is None or total <= len(found) or not prices:
                break
            lower = prices[0] - 1 if prices[0] > 1 else prices[0] * 0.9
            if cap is not None and lower >= cap:
                break
            cap = lower
        qty = self.wants[0].qty

        def price(ident: str) -> float:
            return self.memory.price_of(ident) if self.memory.price_of(ident) is not None else float('inf')
        pool = sorted([i for i in self.item_ids(1) if self.memory.buyable(i) and self.memory.price_of(i) is not None and (budget is None or price(i) * qty <= budget)], key=price)
        best = next((i for i in pool if i in self.memory.full and self.likely_fit(i, 1)), None)
        wanted = {stem_word(w) for v in self.value_words(1) for w in re.findall(r'[a-z0-9]{3,}', v.lower()) if w not in VALUE_STOP}
        hits = {i for v in self.value_words(1) for i in self.value_hits.get(v, set())}
        mention = [i for i in pool if i not in self.memory.full and (i in hits or (wanted and wanted & {stem_word(w) for w in re.findall(r'[a-z0-9]{3,}', dumps(self.memory.rows.get(i) or {}).lower())}))]
        blind = [i for i in pool if i not in self.memory.full and i not in mention]
        opened = 0
        limit = min(5, max(0, self.room() - 3))
        for ident in mention + blind:
            if opened >= limit or self.room() < 4 or self.finished():
                break
            if best is not None and price(ident) >= price(best):
                continue
            if ident in blind and best is None:
                continue
            if not self.fill_facts([ident], self.tail() + 2):
                break
            opened += 1
            if self.likely_fit(ident, 1) and (best is None or price(ident) < price(best)):
                best = ident
        unjudged = [need.text for need, g in (self.grades_of(best, 1) if best else []) if g == 'skip']
        self.trace.append({'kind': 'ladder', 'rounds': len(lines), 'opened': opened, 'pick': best, 'unjudged': unjudged, 'cheapest_seen': [(self.memory.price_of(i), i) for i in pool[:5]]})
        self.wider_held = True

        def verdict(ident: str) -> str:
            if ident not in self.memory.full:
                return 'unread'
            bad = [need.prop for need, g in self.grades_of(ident, 1) if g == 'fails']
            if bad:
                return 'FAILS ' + ', '.join(bad[:2])
            return 'fits by its record' if self.likely_fit(ident, 1) else 'unclear'
        shown = [i for i in pool if best is None or price(i) <= price(best)][:8]
        text = ('[Ghost looked for the cheapest fitting listing itself] ' + '; '.join(lines) +
                f". Buyable rows of the kind at or below the pick, cheapest first: {', '.join(f'{i} ({self.memory.price_of(i):g}: {verdict(i)})' for i in shown) or 'none'}.")
        if best is not None and all(i in self.memory.full for i in pool if price(i) < price(best)):
            text += ' Every cheaper row of the kind has been read and fails, so nothing cheaper fits.'
        if best is not None:
            self.swept.add(self.cart_mark() + f'#{self.memory.version}')
        if best and unjudged:
            text += f" {best} ({self.memory.price_of(best):g}) is the cheapest read listing that meets every requirement code can judge; judge {'; '.join(unjudged)} yourself from its record, then cart it (or the next cheapest that fits) and check out."
        elif best:
            if not self.cart_ids() and self.shop.role('cart') and self.left() >= 3:
                done = self.set_cart({best: qty}, [])
                if done:
                    text += f" {best} ({self.memory.price_of(best):g}) is the cheapest read listing that meets every requirement, so Ghost carted it: " + '; '.join(done) + '. Check out unless a change demands otherwise.'
                    return text
            text += f" {best} ({self.memory.price_of(best):g}) is the cheapest read listing that meets every requirement: cart it and check out."
        else:
            text += ' No read row meets every requirement yet; find one (search the value as a keyword, open rows that mention it), then Ghost checks the cheaper rows below it before checkout.'
        return text

    def wider_hold(self) -> str:
        if self.wider_held or self.pick != 'cheapest':
            return ''
        self.wider_held = True
        if self.cheaper_sweep() or self.room() < 10:
            return ''
        lines = [f'{i} at {self.memory.price_of(i):g}' for i in self.cart_ids() if self.memory.price_of(i) is not None]
        return ("Not run yet (no step used). The pick is the cheapest and steps remain. Searches match titles, so a requirement's value finds "
                "few of the fitting listings and rows come in no price order: run a few more searches under other names for the product "
                "and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') +
                "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")

    # -- chains: requirements whose value lives on another listing's page --------------------------------------
    def resolve_pointers(self) -> str:
        """Follow every pointer requirement to the page that states its value, opening each page on the way."""
        pending = [(n, need) for n, need in self.needs_now() if need.points and not need.resolved]
        if not pending or self.buys > 0:
            return ''
        notes: list[str] = []
        for number, need in pending:
            if self.chain_steps >= CHAIN_STEPS or self.room() < 3 or self.time_left() < EXPLORE_FLOOR + 20 or self.finished():
                break
            if self.chain_misses.get(need.text, 0) >= 2:
                continue
            parsed = self.oracle.ask_json(REF_PARSE + '\n\nREQUIREMENT:\n' + need.text + f'\n\nITEM IT BELONGS TO: {self.wants[number - 1].name}', [QUICK, DEEP], 900, tag='ref') or {}
            anchor = parsed.get('anchor') if isinstance(parsed.get('anchor'), dict) else None
            prop = str(parsed.get('property') or need.prop).strip() or need.prop
            if not anchor or not str(anchor.get('category') or anchor.get('shop') or '').strip():
                self.chain_misses[need.text] = self.chain_misses.get(need.text, 0) + 1
                continue
            before = self.oracle.step
            value, path = self.follow_chain(anchor, prop, 0, [])
            self.chain_steps += self.oracle.step - before
            if value is None:
                self.chain_misses[need.text] = self.chain_misses.get(need.text, 0) + 1
                notes.append(f'{need.text}: the listing it points at was not found yet' + (f' (pages opened: {", ".join(path)})' if path else ''))
                self.trace.append({'kind': 'chain_miss', 'need': need.text, 'path': path})
                continue
            resolved = self.compose(str(parsed.get('relation') or 'identical'), prop, value, parsed.get('amount'), str(parsed.get('unit') or ''))
            need.resolved = resolved
            self.memory.version += 1
            self.checked = set()
            notes.append(f'{need.text} -> {resolved} (pages opened in order: {", ".join(path)})')
            self.trace.append({'kind': 'chain', 'need': need.text, 'resolved': resolved, 'path': path})
        if not notes:
            return ''
        return '[Ghost followed the pages that requirements point at]\n' + '\n'.join(notes) + '\nApply the resolved values when choosing; the order counts only with every page of each chain opened, which Ghost has done for the chains marked resolved.'

    def follow_chain(self, anchor: dict[str, Any], prop: str, depth: int, trail: list[str]) -> tuple[str | None, list[str]]:
        if depth > CHAIN_DEPTH or self.room() < 2 or self.finished():
            return None, trail
        ident = self.find_anchor(anchor, prop)
        if not ident or ident in trail:
            return None, trail
        trail = trail + [ident]
        self.complete_views([ident], self.tail() + 1)
        record = self.memory.record_text(ident, 3500)
        prompt = PAGE_READ + f'\n\nPROPERTY WANTED: {prop}\n\nTHE PAGE (listing {ident}):\n' + record
        read = self.oracle.ask_json(prompt, [QUICK, DEEP], 1200, tag='page') or {}
        for pair in as_list(read.get('rules')):
            if isinstance(pair, list) and len(pair) == 2:
                self.add_rule(str(pair[0]), str(pair[1]))
        for neg in as_list(read.get('negations')):
            self.memory.negations.setdefault(ident, []).append(str(neg))
        redirect = read.get('redirect') if isinstance(read.get('redirect'), dict) else None
        if redirect and str(redirect.get('category') or redirect.get('shop') or '').strip():
            return self.follow_chain(redirect, prop, depth + 1, trail)
        value = str(read.get('value') or '').strip()
        if not value:
            block = attribute_block(self.memory.full.get(ident) or {}) or {}
            for key, shown in block.items():
                if same_property(prop, key) and str(shown).strip():
                    value = str(shown).strip()
                    break
        return (value or None), trail

    def find_anchor(self, anchor: dict[str, Any], prop: str) -> str | None:
        """The one listing a description points at: by shop, kind and the traits named, searching when unseen."""
        shop = str(anchor.get('shop') or '').strip()
        category = str(anchor.get('category') or '').strip()
        traits = [str(t) for t in as_list(anchor.get('traits')) if str(t).strip()]
        trait_words = {w for t in traits for w in re.findall(r'[a-z0-9.]{2,}', t.lower().split(':', 1)[-1])}
        shop_words = {w for w in re.findall(r'[a-z0-9]{3,}', shop.lower())}
        cat_words = content_words(category)

        def score(ident: str) -> tuple[int, int, int]:
            row = self.memory.rows.get(ident) or {}
            text = dumps(row).lower() + ' ' + self.memory.pages.get(ident, '').lower()
            seller = self.memory.seller_of(ident).lower()
            shop_hit = 2 if shop_words and all(w in seller for w in shop_words) else (1 if shop_words and any(w in text for w in shop_words) else 0)
            label = self.memory.label_of(ident).lower()
            cat_hit = 1 if (cat_words and (cat_words & content_words(label) or all(w in text for w in list(cat_words)[:2]))) else 0
            return shop_hit, cat_hit, len(trait_words & set(re.findall(r'[a-z0-9.]{2,}', text)))

        def candidates() -> list[str]:
            pool = [i for i in self.memory.order if not self.memory.sold_out(i) or True]
            ranked = sorted(pool, key=lambda i: score(i), reverse=True)
            return [i for i in ranked if score(i)[0] >= (2 if shop_words else 0) and (score(i)[1] or not cat_words)][:4]
        found = candidates()
        plan = self.search_template()
        tries = 0
        while len(found) < 1 and plan and tries < 2 and self.room() >= 3:
            words = ' '.join(x for x in [shop, category] if x) if tries == 0 else ' '.join([category] + [t.split(':', 1)[-1].strip() for t in traits[:2]])
            if not words.strip():
                break
            self.search_words(plan[0], plan[1], plan[2], words[:60])
            tries += 1
            found = candidates()
        if not found:
            return None
        unread = [i for i in found[:3] if i not in self.memory.full]
        if unread:
            self.fill_facts(unread, self.tail() + 1)
        if len(found) == 1 and not traits:
            return found[0]
        records = '\n'.join(f'{i} (seller: {self.memory.seller_of(i) or "?"}): ' + self.memory.record_text(i, 1200) for i in found[:4])
        prompt = LINK_PICK + f'\n\nDESCRIPTION: shop or brand {shop or "(not named)"}; kind {category or "(not named)"}; traits: ' + ('; '.join(traits) or '(none)') + '\n\nCANDIDATE RECORDS:\n' + records
        got = self.oracle.ask_json(prompt, [QUICK, DEEP], 600, tag='link') or {}
        choice = str(got.get('id') or '').strip()
        if choice in found:
            return choice
        close = [str(c) for c in as_list(got.get('close')) if str(c) in found]
        if close:
            return close[0]
        return found[0] if score(found[0])[0] == 2 and (score(found[0])[2] >= max(1, len(trait_words) // 2)) else None

    def compose(self, relation: str, prop: str, value: str, amount: Any, unit: str) -> str:
        value = value.strip().rstrip('.')
        if relation == 'different':
            return f'{prop}: not {value} (any value other than {value})'
        if relation == 'partner':
            partner = self.partner_of(value)
            if partner:
                return f'{prop}: {partner} (the store pairs {value} with {partner})'
            return f"{prop}: the value the store's pairing rule puts with {value} (rule not read yet; look for it on listing pages)"
        if relation == 'within':
            number = as_number(amount)
            base = as_number(value)
            if number is not None and base is not None:
                return f'{prop}: between {base - number:g} and {base + number:g} {unit} (within {number:g} {unit} of {value})'.replace('  ', ' ')
            return f'{prop}: within {amount} {unit} of {value}'.replace('  ', ' ')
        return f'{prop}: {value}'

    def partner_of(self, value: str) -> str | None:
        words = content_words(value)
        for a, b in self.memory.rules:
            if content_words(a) == words or (words and words <= content_words(a)):
                return b
            if content_words(b) == words or (words and words <= content_words(b)):
                return a
        return None

    # -- the cart ----------------------------------------------------------------------------------------------
    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        self.cart_touched = True
        self.cart_at = self.oracle.step
        parts = self.shop.cart_parts(name)
        props = parts['props']
        removal = False
        for key, value in args.items():
            options = (props.get(key) or {}).get('enum')
            if isinstance(options, list) and len(options) >= 2 and value == self.shop.remove_option(name):
                removal = True
        loose_count = next((as_int(v) for v in args.values() if as_int(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (props.get(key) or {}).get('items') or {}
                if not value and isinstance(shape, dict) and (shape.get('type') == 'object' or 'properties' in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.memory.rows), None)
                        count = next((as_int(v) for v in line.values() if as_int(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.memory.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.memory.rows:
                lines.append((value, loose_count))
        if whole:
            self.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = self.shop.cart_sets_count(name)
        for ident, count in lines:
            have = self.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.cart_lines[ident] = have
            else:
                self.cart_lines.pop(ident, None)

    def cart_fresh(self) -> bool:
        return self.cart_view is not None and self.cart_seen_at >= self.cart_at

    def cart_ids(self) -> list[str]:
        if self.cart_fresh():
            text = dumps(self.cart_view)
            return [i for i in self.memory.rows if i in text]
        return list(self.cart_lines)

    def held_units(self, ident: str) -> int:
        if self.cart_fresh():
            stack: list[Any] = [self.cart_view]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [as_int(v) for k, v in node.items() if not isinstance(v, str) and as_int(v) is not None and re.search(r'\b(?:qty|quantity|count|units?|number|amount|how many|pieces)\b', plain(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.cart_lines.get(ident, 1)

    def cart_text(self) -> str:
        if self.cart_fresh():
            return dumps(clip(self.cart_view, 300))[:3000]
        return "lines by Ghost's record of the cart calls (listing: units): " + dumps(self.cart_lines)

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """The cart tool calls that leave exactly `lines` in the cart, removing `drop`; None when the tool's shape
        does not allow it."""
        name = self.shop.role('cart') or ''
        parts = self.shop.cart_parts(name)
        if parts['lines']:
            key, inner = parts['lines']
            id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in schema_kinds(s) and 'enum' not in s), None)
            count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in schema_kinds(s)), None)
            if not id_key:
                return None
            return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        target, count, mode = parts['target'], parts['count'], parts['mode']
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        sets = self.shop.cart_sets_count(name)
        for ident in drop:
            have = self.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: self.shop.remove_option(name), **({count: have} if count else {})})
            elif count and sets:
                calls.append({target: ident, count: 0})
            elif count and isinstance(parts['props'][count].get('minimum'), int) and parts['props'][count]['minimum'] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.cart_lines.get(ident, 0) if ident in self.cart_ids() else 0
            if have == want:
                continue
            if have and not sets:
                if mode:
                    calls.append({target: ident, mode: self.shop.remove_option(name), **({count: have} if count else {})})
                else:
                    return None
            calls.append({target: ident, **({mode: self.shop.add_option(name)} if mode else {}), **({count: want} if count else {})})
        return calls

    def set_cart(self, lines: dict[str, int], drop: list[str]) -> list[str] | None:
        name = self.shop.role('cart')
        calls = self.cart_calls(lines, drop)
        if not name or calls is None or len(calls) + 1 > self.left():
            return None
        done = []
        for args in calls:
            args, faults = self.shop.shape_args(name, args)
            if faults:
                return None
            reply = self.oracle.act(name, args)
            if reply is None or self.finished():
                return None
            self.note_cart(name, args)
            self.cart_said = dumps(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(dumps(args) + ' -> ' + self.show(reply.get('observation'))[:300])
        return done

    def line_faults(self) -> list[str]:
        key = self.memory.category_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not key:
            return []
        label = {i: str((self.memory.rows.get(i) or {}).get(key) or '') for i in ids}
        known = {n for n, want in enumerate(self.wants, 1) if any(self.owns(want, str(r.get(key) or '')) for r in self.memory.rows.values())}
        owner = {i: [n for n, want in enumerate(self.wants, 1) if self.owns(want, label[i])] for i in ids}
        faults: list[str] = []
        for n, want in enumerate(self.wants, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({want.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({want.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != want.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({want.name}) wants {want.qty}')
        if len(known) == len(self.wants):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for: take it out")
        for i in ids:
            if self.memory.sold_out(i):
                faults.append(f'{i} is sold out')
        prices = {i: self.memory.price_of(i) for i in ids}
        if isinstance(self.budget, (int, float)) and all(p is not None for p in prices.values()):
            if self.budget_scope == 'each':
                faults += [f'{i} costs {prices[i]:g}, over the {self.budget:g} limit for each item' for i in ids if self.budget < prices[i] <= 10 * self.budget]
            else:
                total = sum(prices[i] * self.held_units(i) for i in ids)
                if self.budget + 1e-9 < total <= 10 * self.budget:
                    faults.append(f'the total is {total:g}, over the budget of {self.budget:g}')
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        key = self.memory.category_key()
        if not key:
            return []
        out = []
        for ident in self.cart_ids():
            have = self.held_units(ident) or 0
            owners = [n for n, want in enumerate(self.wants, 1) if self.owns(want, str((self.memory.rows.get(ident) or {}).get(key) or ''))]
            mine = [i for i in self.cart_ids() if (self.cart_lines.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owns(self.wants[owners[0] - 1], str((self.memory.rows.get(i) or {}).get(key) or ''))]
            if have > 0 and len(owners) == 1 and len(mine) == 1 and have != self.wants[owners[0] - 1].qty and not self.memory.sold_out(ident):
                out.append((ident, have, self.wants[owners[0] - 1].qty))
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.shop.role('cart') or ''
        parts = self.shop.cart_parts(name)
        target, count, mode = parts['target'], parts['count'], parts['mode']
        if not name or not target or not count:
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.shop.cart_sets_count(name):
                calls.append({target: ident, count: want})
            elif mode:
                calls += [{target: ident, mode: self.shop.remove_option(name), count: have}, {target: ident, mode: self.shop.add_option(name), count: want}]
            else:
                return []
        if len(calls) + 3 > self.left():
            return []
        done = []
        for args in calls:
            args, faults = self.shop.shape_args(name, args)
            if faults:
                break
            reply = self.oracle.act(name, args)
            if reply is None or self.finished():
                break
            self.note_cart(name, args)
            self.cart_said = dumps(reply.get('observation'))
            self.observe(name, reply.get('observation'))
            done.append(dumps(args) + ' -> ' + self.show(reply.get('observation'))[:300])
        self.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    # -- compatibility between requested items ---------------------------------------------------------------------
    def fit_index(self) -> dict[str, dict[str, Any]]:
        marks: dict[str, dict[str, Any]] = {}
        for ident in self.memory.order:
            record = self.memory.full.get(ident) or self.memory.rows.get(ident) or {}
            text = free_text(record) + ' ' + '. '.join(f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str))
            text += ' ' + ' '.join(str(v) for v in record.values() if isinstance(v, str))
            marks[ident] = fit_marks(text + ' ' + self.memory.pages.get(ident, ''))
        return marks

    def fit_faults(self) -> list[str]:
        if not self.bound:
            return []
        marks = self.fit_index()
        cart = self.cart_ids()
        faults = []
        for part, main in self.bound:
            devices = [i for i in cart if i in self.item_ids(main)]
            parts = [i for i in cart if i in self.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            device = marks.get(devices[0]) or {}
            for line in parts:
                if fit_verdict(marks.get(line) or {}, device) is False:
                    good = [i for i in self.item_ids(part) if fit_verdict(marks.get(i) or {}, device) and not self.memory.sold_out(i) and i not in cart][:4]
                    named = (marks.get(line) or {}).get('works') or (marks.get(line) or {}).get('series') or ['other models']
                    faults.append(f"{line} ({self.wants[part - 1].name}) names {', '.join(sorted(named)[:4])}, not {devices[0]} ({', '.join(sorted(set([device.get('own') or ''] + list(device.get('series') or [])) - {''}))})" + (f"; seen parts that work with it: {', '.join(good)}" if good else ''))
        return faults

    def fit_plan(self) -> str:
        if not self.bound:
            return ''
        marks = self.fit_index()
        lines = []
        for main in sorted({m for _, m in self.bound}):
            parts = [p for p, m in self.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.item_ids(p):
                    if self.memory.sold_out(i):
                        continue
                    mark = marks.get(i) or {}
                    for code in set(mark.get('works') or []) | set(mark.get('series') or []):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]
            texts = []
            for code, per in ranked:
                devices = [i for i in self.item_ids(main) if any(code_names(code, c) for c in [(marks.get(i) or {}).get('own') or ''] if c) or code in ((marks.get(i) or {}).get('series') or [])][:2]
                texts.append(f'{code}: ' + ', '.join(f'{n} {self.wants[p - 1].name}' for p, n in sorted(per.items())) + (f" (device seen: {', '.join(devices)})" if devices else ' (device not seen yet)'))
            if texts:
                lines.append(f"codes the {self.wants[main - 1].name} candidates' parts name -> " + '; '.join(texts))
        if not lines:
            return ''
        return "[fit plan: a part fits a device when the part's text names the device's model code or series, or both carry the same series code] " + ' | '.join(lines)

    # -- checks before an order ---------------------------------------------------------------------------------
    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.memory.version}'

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        planned = ids is not None
        ids = ids if planned else self.cart_ids()
        key = self.check_key(ids)
        judges = self.oracle.judges(limit)
        if key in self.checked or not judges or not ids:
            return ''
        self.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        answers = [line for line in self.done_log if 'message to the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers[-6:])
        if planned:
            prompt += '\n\nCART (planned; judge these as carted at the requested quantities: ' + '; '.join(f'item {n}: quantity {w.qty}' for n, w in enumerate(self.wants, 1)) + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + self.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS (with code-computed grades from the keys naming each property: ok fits, FAILS is wrong, DOUBTFUL loses to an ok listing):\n" + '\n'.join(f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if self.memory.sold_out(i) else '') + self.memory.record_text(i) + (' [grades: ' + self.grade_text(i) + ']' if self.grade_text(i) else '') for i in ids)
        if self.pick in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 15)
        got = self.oracle.ask_json(prompt, judges, 2500, tag='check')
        self.trace.append({'kind': 'check', 'cart': ids, 'got': clip(got, 200)})
        if got and got.get('ok') is False and self.pick == 'cheapest' and self.cheapest_stands(ids, got):
            got = {**got, 'ok': True}
        self.verdicts[key] = bool(got) and got.get('ok') is not False
        if not got or got.get('ok') is not False:
            return ''
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in as_list(got.get('problems'))][:5]
        better = [str(b) for b in as_list(got.get('better')) if str(b) in self.memory.rows and not self.memory.sold_out(str(b))][:4]
        self.better = better
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def title_of(self, ident: str) -> str:
        record = self.memory.rows.get(ident) or {}
        for key, value in record.items():
            if isinstance(value, str) and len(value) > 3 and re.search(r'title|name|headline|label', plain(key)) and not re.search(r'category|brand|shop|seller|company', plain(key)):
                return plain(value)
        return ''

    def twins_of(self, ident: str) -> list[str]:
        """Other buyable listings with the same title and category label: the same product under another id."""
        title = self.title_of(ident)
        if not title:
            return []
        label = self.memory.label_of(ident)
        return [other for other in self.memory.order if other != ident and self.memory.buyable(other) and self.title_of(other) == title and self.memory.label_of(other) == label]

    def unread_candidates(self, number: int) -> list[str]:
        """Unread rows of the item's kind worth buying when nothing read fits: twins of a fitting line that sold out,
        then rows a value search returned; cheapest first under a cheapest pick."""
        want = self.wants[number - 1]
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
        ok = lambda i: i not in self.memory.full and self.memory.buyable(i) and i not in self.cart_ids() and (cap is None or (self.memory.price_of(i) or 0) * want.qty <= cap) and self.owns(want, self.memory.label_of(i))  # noqa: E731
        out: list[str] = []
        for lost in self.sold_fits:
            out += [t for t in self.twins_of(lost) if ok(t) and t not in out]
        hits = [i for v in self.value_words(number) for i in sorted(self.value_hits.get(v, set()))]
        out += [i for i in hits if ok(i) and i not in out]
        if self.pick == 'cheapest':
            out.sort(key=lambda i: self.memory.price_of(i) if self.memory.price_of(i) is not None else float('inf'))
        return out

    def swap_lost(self) -> str:
        """Replace every sold-out cart line with the best read replacement of its kind when the steps allow it: the
        cheapest listing code grades as fitting under a cheapest pick, else the best-graded one; without one by
        code, the review's replacement."""
        cart = self.shop.role('cart')
        lost = [i for i in self.cart_ids() if self.memory.sold_out(i)]
        if not cart or not lost or self.left() < 2 * len(lost) + 1:
            return ''
        picks: dict[str, int] = {}
        for line in lost:
            number = self.owner_of(self.memory.label_of(line)) or (1 if len(self.wants) == 1 else 0)
            if not number:
                return ''
            want = self.wants[number - 1]
            cap = self.budget if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
            pool = [i for i in self.item_ids(number) if i in self.memory.full and self.memory.buyable(i) and i not in self.cart_ids() and i not in picks
                    and (cap is None or (self.memory.price_of(i) or 0) * want.qty <= cap)]
            fitting = [i for i in pool if self.likely_fit(i, number)] or [i for i in pool if not any(g == 'fails' for _, g in self.grades_of(i, number)) and self.leads and i in self.leads]
            if line in self.sold_fits or self.likely_fit(line, number):
                twins = [t for t in self.twins_of(line) if t not in picks and t not in self.cart_ids() and (cap is None or (self.memory.price_of(t) or 0) * want.qty <= cap)]
                fitting = [t for t in twins if t not in self.memory.full or self.likely_fit(t, number)] + [i for i in fitting if i not in twins]
            if not any(i in self.memory.full and self.clean_fit(i, number) or i not in self.memory.full for i in fitting) and self.left() >= 2 * len(lost) + 4 and self.time_left() > EXPLORE_FLOOR:
                extra = [i for i in self.unread_candidates(number) if i not in fitting][:2]
                opened = self.fill_facts(extra, self.tail() + 2 * len(lost) + 1) if extra else []
                self.trace.append({'kind': 'swap_extra', 'line': line, 'extra': extra, 'opened': len(opened)})
                if opened:
                    fitting = [i for i in extra if i in self.memory.full and self.likely_fit(i, number)] + [i for i in fitting if i not in extra]
            if not fitting:
                return ''
            if self.pick == 'cheapest':
                fitting.sort(key=lambda i: self.memory.price_of(i) if self.memory.price_of(i) is not None else float('inf'))
            else:
                fitting.sort(key=lambda i: (self.listing_score(i, number), self.memory.order.index(i)))
            picks[fitting[0]] = want.qty
        keep = {i: self.held_units(i) or 1 for i in self.cart_ids() if i not in lost}
        done = self.set_cart({**keep, **picks}, lost)
        if not done:
            return ''
        self.trace.append({'kind': 'swap_lost', 'lost': lost, 'in': list(picks)})
        return 'Ghost replaced the sold-out line(s) with read listings that meet the requirements: ' + '; '.join(done)

    def cheapest_stands(self, ids: list[str], got: dict[str, Any]) -> bool:
        """Under a cheapest pick, a verdict that only prefers dearer listings over a buyable line code grades as
        fitting (even doubtfully) does not count."""
        if len(self.wants) != 1 or len(ids) != 1:
            return False
        line = ids[0]
        price = self.memory.price_of(line)
        if price is None or not self.memory.buyable(line) or not self.likely_fit(line, 1):
            return False
        better = [str(b) for b in as_list(got.get('better')) if str(b) in self.memory.rows]
        if any((self.memory.price_of(b) if self.memory.price_of(b) is not None else 0) < price for b in better):
            return False
        problems = ' '.join(str(p) for p in as_list(got.get('problems'))).lower()
        if re.search(r'sold out|out of stock|budget|quantity|fails', problems):
            return False
        return True

    def drop_sold_lines(self) -> str:
        """Take known sold-out lines out of the cart when a checkout is about to go out with few steps left."""
        cart = self.shop.role('cart')
        lost = [i for i in self.cart_ids() if self.memory.sold_out(i)]
        if not cart or not lost or self.left() < len(lost) + 1:
            return ''
        keep = {i: self.held_units(i) or 1 for i in self.cart_ids() if i not in lost}
        done = self.set_cart(keep, lost)
        if not done:
            return ''
        self.trace.append({'kind': 'drop_sold', 'lost': lost})
        return 'Ghost took the sold-out line(s) out of the cart first: ' + '; '.join(done)

    def swap_to_better(self, better: list[str]) -> str:
        """Replace a cart line with a candidate the review found better, when that candidate's attributes were read
        and it is buyable (and cheaper, under a cheapest pick)."""
        key = self.memory.category_key()
        cart = self.cart_ids()
        for cand in better:
            if cand in cart or not self.memory.buyable(cand) or cand not in self.memory.full:
                continue
            owner = self.owner_of(self.memory.label_of(cand)) or (1 if len(self.wants) == 1 else 0)
            if owner and any(g == 'fails' for _, g in self.grades_of(cand, owner)):
                continue
            kind = self.memory.label_of(cand).lower()
            same = [i for i in cart if key and self.memory.label_of(i).lower() == kind] or (cart if len(self.wants) == 1 else [])
            if self.pick == 'cheapest' and same and all((self.memory.price_of(cand) or 0) >= (self.memory.price_of(i) or 0) for i in same):
                continue
            number = self.owner_of(self.memory.label_of(cand)) or (1 if len(self.wants) == 1 else 0)
            if not number:
                continue
            lines = {i: self.held_units(i) or 1 for i in cart if i not in same}
            lines[cand] = self.wants[number - 1].qty
            done = self.set_cart(lines, same)
            if done:
                self.trace.append({'kind': 'swap_better', 'in': cand, 'out': same})
                return 'Ghost swapped the cart to the better listing the review named: ' + '; '.join(done)
        return ''

    def others(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.memory.order))
        if self.pick == 'cheapest':
            order.sort(key=lambda i: (i not in self.memory.full, self.memory.price_of(i) if self.memory.price_of(i) is not None else float('inf')))
        for ident in order:
            if ident in skip or self.memory.sold_out(ident):
                continue
            lines.append(self.memory.row_text(ident) + (' [grades: ' + self.grade_text(ident) + ']' if ident in self.memory.full and self.grade_text(ident) else ''))
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def reread_offers(self) -> str:
        tool = self.offer_tool()
        target = self.shop.single_target(tool)
        if not tool or not target:
            return ''
        since = max(self.oracle.step - 2, self.refused_at)
        stale = [i for i in self.cart_ids() if self.offer_seen.get(i, -99) < since and i not in self.memory.gone]
        if not stale or self.left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            before = dict(self.memory.rows.get(ident) or {})
            reply = self.open_view(tool, ident)
            if reply is None or self.finished():
                break
            self.offer_seen[ident] = self.oracle.step
            after = self.memory.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {dumps(clip(reply.get('observation'), 300))[:900]}")
        self.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            self.memory.version += 1
            self.checked = set()
        return '\n'.join(lines)

    def swap_hint(self) -> str:
        lost = [i for i in self.cart_ids() if i not in self.memory.gone and self.memory.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            self.memory.gone[ident] = self.oracle.step
            number = self.owner_of(self.memory.label_of(ident)) or (1 if len(self.wants) == 1 else 0)
            if number and ident not in self.sold_fits and (ident not in self.memory.full or self.likely_fit(ident, number) or self.confirmed == ident):
                self.sold_fits.append(ident)
        self.leads = [i for i in self.leads if i not in lost]
        self.memory.version += 1
        self.checked = set()
        picks = self.replacements(lost)
        text = f"[Ghost] Sold out now: {', '.join(lost)}. It will not come back, so do not open it again."
        if picks and self.left() <= 6:
            return text + f" Few steps are left: add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other read listings that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and self.pick == 'cheapest':
            return text + f" Read listings that appear to fit instead, cheapest first: {', '.join(picks)}. The pick is the cheapest: unless you have seen the whole fitting set, run one search under {picks[0]}'s price before swapping in the cheapest fitting listing."
        if picks:
            return text + f" Read listings that appear to fit instead: {', '.join(picks)}. Take the sold-out line out and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        ids = [i for i in self.cart_ids() if i not in self.memory.gone]
        if any(self.memory.sold_out(i) for i in ids):
            return ''
        if len(ids) == 1:
            self.memory.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        judges = [QUICK, DEEP] if self.oracle.deep_ok() else ([QUICK] if self.oracle.cost <= SPEND_HARD else [])
        if not judges:
            return []
        records = []
        for ident in reversed(self.memory.order):
            if ident in lost or self.memory.sold_out(ident):
                continue
            records.append(self.memory.row_text(ident, 1200) + (' [grades: ' + self.grade_text(ident) + ']' if ident in self.memory.full and self.grade_text(ident) else ''))
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join(f'{i}: {self.memory.record_text(i)}' for i in lost) + '\n\nCART NOW:\n' + self.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = self.oracle.ask_json(prompt, judges, 2500, tag='swap')
        fits: list[tuple[float, str]] = []
        for entry in as_list((got or {}).get('fits')):
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in self.memory.rows and ident not in lost and not self.memory.sold_out(ident) and ident not in self.cart_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.pick == 'cheapest' else [i for _, i in fits]
        self.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        return ranked[:4]

    # -- a decline, and what to do instead -------------------------------------------------------------------------
    def audit_call(self) -> dict[str, Any] | None:
        cap = self.budget if isinstance(self.budget, (int, float)) else None
        key = self.memory.category_key()
        opened: list[str] = []
        rows: list[str] = []
        for strict in (True, False):
            for ident in reversed(self.memory.order):
                label = str((self.memory.rows.get(ident) or {}).get(key) or '') if key else ''
                if self.memory.sold_out(ident) or (strict and ((cap is not None and self.budget_scope == 'total' and (self.memory.price_of(ident) or 0) > cap) or (label and not self.owner_of(label)))):
                    continue
                if ident in self.memory.full and len(opened) < 12:
                    opened.append('OPENED ' + self.memory.record_text(ident, 700) + (' [grades: ' + self.grade_text(ident) + ']' if self.grade_text(ident) else ''))
                elif ident not in self.memory.full and len(rows) < 10:
                    rows.append('ROW ' + self.memory.row_text(ident, 300))
            if opened or rows:
                break
        if not opened and not rows:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger()
        answers = [line for line in self.done_log if 'message to the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers)
        tried = [str(v) for args in self.searched for v in args.values() if isinstance(v, str)]
        if tried:
            prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(dict.fromkeys(tried))
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(opened + rows)
        got = self.oracle.ask_json(prompt, self.oracle.judges() or [QUICK], 3500, tag='audit')
        self.trace.append({'kind': 'audit', 'got': clip(got, 200)})
        return got

    def take_review(self, got: dict[str, Any] | None) -> None:
        changed = []
        aside: set[int] = set()
        for entry in as_list((got or {}).get('not_firm')):
            if isinstance(entry, dict):
                number, text = as_int(entry.get('item')) or 0, str(entry.get('need') or '').strip()
                if 1 <= number <= len(self.wants) and number not in aside:
                    need = next((n for n in self.wants[number - 1].live() if n.text == text and n.source == 'request' and not n.points), None)
                    if need and self.firm_marked(need):
                        self.trace.append({'kind': 'kept_firm', 'need': text})
                    elif need:
                        aside.add(number)
                        need.dropped = True
                        changed.append(f'set aside {text}')
        for entry in as_list((got or {}).get('corrected')):
            if isinstance(entry, dict):
                number, text, value = as_int(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip()
                if 1 <= number <= len(self.wants) and ':' in value:
                    need = next((n for n in self.wants[number - 1].live() if n.text == text and not n.points), None)
                    if need:
                        need.dropped = True
                        self.wants[number - 1].add_need(value, need.source)
                        changed.append(f'{text} read as {value}')
        if changed:
            self.memory.version += 1
            self.checked = set()
            self.trace.append({'kind': 'review', 'changed': changed})
            self.notes.append('a review of the request: ' + '; '.join(changed))

    @staticmethod
    def ids_by_item(got: dict[str, Any] | None, field: str, memory: Memory) -> dict[str, list[str]]:
        table = (got or {}).get(field)
        if not isinstance(table, dict):
            return {}
        named = {str(k): [str(i) for i in as_list(v) if str(i) in memory.rows and not memory.sold_out(str(i))][:4] for k, v in table.items()}
        return {k: v for k, v in named.items() if v}

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        keep = self.tail() + len(self.wants) + 2
        room = min(RECOVERY_STEPS, self.left() - keep)
        if room < 3 or self.time_left() < EXPLORE_FLOOR + 25:
            return False
        before = len(self.memory.full)
        last = self.oracle.step + room
        searching = self.oracle.step + max(1, room // 2)
        plan = self.search_template()
        queries = [(min(max(1, as_int(e.get('item')) or 1), len(self.wants)), str(e['query']).strip()) for e in as_list((got or {}).get('search')) if isinstance(e, dict) and str(e.get('query') or '').strip()]
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(self.value_words(n)[:2] + [self.wants[n - 1].name])) for n in missing]
        found: dict[int, list[str]] = {n: [] for n in missing}
        for number in missing[:3]:
            if self.oracle.step < searching:
                cap = self.budget if isinstance(self.budget, (int, float)) and self.shop.browse and self.shop.browse['price'] and self.budget_scope == 'total' else None
                found[number] += self.list_category(number, cap) or []
        for number, words in queries[:4]:
            if plan and self.oracle.step < searching:
                found.setdefault(number, [])
                found[number] += self.search_words(plan[0], plan[1], plan[2], words)
        for number, rows in found.items():
            picks = [i for i in self.kind_rows(number, rows) if i not in self.memory.full][:2]
            self.fill_facts(picks[:max(0, last - self.oracle.step)], keep)
        self.trace.append({'kind': 'recovery', 'missing': missing, 'queries': queries[:4], 'opened': len(self.memory.full) - before})
        return len(self.memory.full) > before

    def open_rows(self, rows: list[str]) -> bool:
        tool = self.facts_tool()
        if not tool or not self.shop.single_target(tool) or self.left() < 5:
            return False
        before = len(self.memory.full)
        self.fill_facts(rows[:3], self.tail() + 1)
        return len(self.memory.full) > before

    def audit(self) -> str:
        """Before a decline stands: review the evidence, search once more for what is missing, and name candidates.
        Returns advice for the model when a decline is not justified, else ''."""
        if self.audited >= 2 or not (self.oracle.judges() or [QUICK]):
            return ''
        self.audited += 1
        got = self.audit_call()
        if got is None:
            return ''
        self.take_review(got)
        fits = self.ids_by_item(got, 'fits', self.memory)
        maybe = self.ids_by_item(got, 'maybe', self.memory)
        missing = [n for n in range(1, len(self.wants) + 1) if not fits.get(str(n)) and not maybe.get(str(n))]
        if missing and self.audited == 1:
            rows = [str(i) for i in as_list((got or {}).get('open')) if str(i) in self.memory.rows and str(i) not in self.memory.full and not self.memory.sold_out(str(i))]
            opened = bool(rows) and self.open_rows(rows)
            if self.recovery_search(got, missing) or opened:
                again = self.audit_call()
                if again is not None:
                    got = again
                    self.take_review(got)
                    fits = self.ids_by_item(got, 'fits', self.memory) or fits
                    maybe = self.ids_by_item(got, 'maybe', self.memory) or maybe
                    missing = [n for n in range(1, len(self.wants) + 1) if not fits.get(str(n)) and not maybe.get(str(n))]
        if got is not None:
            self.blocking = [(as_int(e.get('item')) or 0, str(e.get('need') or '')) for e in as_list(got.get('blocking')) if isinstance(e, dict)]
        named = {k: v for k, v in {**maybe, **fits}.items()}
        self.lead_items = named
        self.leads = [i for ids in named.values() for i in ids]
        if not named:
            return ''
        if missing:
            return ('A careful review found candidates for some items: ' + '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items()) +
                    f". Items with no candidate yet: {', '.join(self.wants[n - 1].name for n in missing)}. Search those once more (their category listing, another name for the product) before any ending; a decline is only right when a requirement the shopper stated outright has no buyable listing.")
        return 'A careful review thinks these listings may fit: ' + '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them if needed, cart the best fitting one per item and check out.'

    def decline_allowed(self) -> bool:
        """A decline stands only when what blocks every candidate was stated outright by the shopper (in the request
        or an answer), is not a chain or a store note, and the store was actually searched for the item."""
        pointers = [need for _, need in self.needs_now() if need.points and not need.resolved]
        if pointers:
            return False
        tried = ' '.join(sorted(self.tried_words)) + ' ' + ' '.join(str(v) for args in self.searched for v in args.values() if isinstance(v, str)).lower()
        if self.sold_fits:
            self.trace.append({'kind': 'decline_gate', 'reason': 'a fitting line sold out: the store sells the kind'})
            return False
        if not self.blocking:
            self.trace.append({'kind': 'decline_gate', 'audited': self.audited, 'code_blockers': (self.code_blockers(tried) if self.audited >= 1 else [])[:3]})
            return False
        for number, text in self.blocking:
            if not 1 <= number <= len(self.wants):
                return False
            want = self.wants[number - 1]
            need = next((n for n in want.live() if n.text == text or same_property(n.prop, text.split(':')[0])), None)
            if need is None or need.points or need.source not in ('request', 'shopper'):
                return False
            if not self.searched and not self.item_ids(number):
                return False
            cap = self.budget if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
            read = [i for i in self.item_ids(number) if i in self.memory.full and self.memory.buyable(i) and (cap is None or (self.memory.price_of(i) or 0) * want.qty <= cap)]
            grades = [self.grade(i, need) for i in read]
            if any(g in ('strong', 'weak', 'conflict') for g in grades):
                return False
            if all(g == 'unread' for g in grades) and self.item_ids(number):
                return False
            ok_words = self.kind_ok_words()
            carried = sum(1 for i in read if any(key_names(need.prop, k, ok_words) for k in (attribute_block(self.memory.full.get(i) or {}) or {})))
            if grades and grades[0] != 'skip' and (len(read) < 3 or carried * 10 < len(read) * 3):
                self.trace.append({'kind': 'decline_gate', 'reason': 'property rarely carried', 'need': need.text, 'read': len(read), 'carried': carried})
                return False
            words = [w for w in re.findall(r'[a-z0-9]{3,}', need.value.lower()) if w not in VALUE_STOP and stem_word(w) not in prop_stems(need.prop)]
            if words and (not grades or grades[0] != 'skip') and not any(w[:4] in tried for w in words):
                return False
        return True

    def code_blockers(self, tried: str) -> list[tuple[int, str]]:
        """Plain requirements (stated by the shopper or in the request) that no read, buyable, affordable listing of
        the item's kind meets in any way, after at least three such listings were read and the value itself was
        searched as a keyword: evidence by code that the store sells nothing fitting."""
        out: list[tuple[int, str]] = []
        for number, want in enumerate(self.wants, 1):
            cap = self.budget if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
            read = [i for i in self.item_ids(number) if i in self.memory.full and self.memory.buyable(i) and (cap is None or (self.memory.price_of(i) or 0) * want.qty <= cap)]
            if len(read) < 3:
                continue
            for need in want.live():
                if need.points or need.source not in ('request', 'shopper'):
                    continue
                grades = [self.grade(i, need) for i in read]
                if any(g in ('strong', 'weak', 'conflict', 'skip', 'unread') for g in grades):
                    continue
                words = [w for w in re.findall(r'[a-z0-9]{3,}', need.value.lower()) if w not in VALUE_STOP and stem_word(w) not in prop_stems(need.prop)]
                if words and any(w[:4] in tried for w in words):
                    out.append((number, need.text))
        return out

    def forced_pick(self) -> dict[int, str]:
        """When no verified cart exists and the episode must end: the most likely listing per item, by a judge."""
        judges = self.oracle.judges(SPEND_HARD) or [QUICK]
        key = self.memory.category_key()
        records = []
        for ident in reversed(self.memory.order):
            if self.memory.sold_out(ident) or (key and not self.owner_of(self.memory.label_of(ident))):
                continue
            if not self.memory.buyable(ident):
                continue
            records.append(('OPENED ' if ident in self.memory.full else 'ROW ') + self.memory.row_text(ident, 700) + (' [grades: ' + self.grade_text(ident) + ']' if ident in self.memory.full and self.grade_text(ident) else ''))
            if len(records) >= 24:
                break
        if not records:
            return {}
        prompt = PICKER + '\n\nREQUEST:\n' + self.request + '\n\nREQUIREMENTS IN FORCE:\n' + self.ledger() + '\n\nRECORDS (newest first):\n' + '\n'.join(records)
        got = self.oracle.ask_json(prompt, judges, 800, tag='pick') or {}
        choice = got.get('choice') if isinstance(got.get('choice'), dict) else {}
        out: dict[int, str] = {}
        for k, v in choice.items():
            number = as_int(k)
            ident = str(v or '').strip()
            if number and 1 <= number <= len(self.wants) and ident in self.memory.rows and not self.memory.sold_out(ident):
                out[number] = ident
        for number, ids in self.lead_items.items():
            n = as_int(number)
            if n and n not in out:
                pick = next((i for i in ids if not self.memory.sold_out(i)), None)
                if pick:
                    out[n] = pick
        for n in range(1, len(self.wants) + 1):
            if n not in out:
                pick = self.code_pick(n)
                if pick:
                    out[n] = pick
        self.trace.append({'kind': 'forced_pick', 'choice': out, 'reason': str(got.get('reason') or '')[:160]})
        return out

    def code_pick(self, number: int) -> str | None:
        """The most plausible read listing of an item's kind, by code: buyable, within budget, whose record states every
        plain requirement's words where any listing does; cheapest first under a cheapest pick, else the earliest seen."""
        want = self.wants[number - 1]
        ids = [i for i in self.item_ids(number) if i in self.memory.full and self.memory.buyable(i)]
        if not ids and len(self.wants) == 1:
            ids = [i for i in self.memory.order if i in self.memory.full and self.memory.buyable(i)]
        if not ids:
            unread = self.unread_candidates(number)
            if unread:
                self.trace.append({'kind': 'code_pick', 'item': number, 'pick': unread[0], 'unread': True})
                return unread[0]
            return None
        cap = self.budget if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
        if cap is not None:
            inside = [i for i in ids if (self.memory.price_of(i) or 0) * want.qty <= cap]
            ids = inside or ids
        scored = []
        for ident in ids:
            grades = [g for _, g in self.grades_of(ident, number)]
            failing = sum(1 for g in grades if g == 'fails')
            order = self.memory.price_of(ident) if self.pick == 'cheapest' and self.memory.price_of(ident) is not None else float(self.memory.order.index(ident))
            scored.append((failing, self.listing_score(ident, number), order, ident))
        scored.sort()
        best = scored[0]
        if best[0] > 0:
            unread = self.unread_candidates(number)
            if not unread:
                cap_here = self.budget if isinstance(self.budget, (int, float)) and self.budget_scope == 'total' else None
                unread = [i for i in self.item_ids(number) if i not in self.memory.full and self.memory.buyable(i) and i not in self.cart_ids()
                          and (cap_here is None or (self.memory.price_of(i) or 0) * want.qty <= cap_here)]
                if self.pick == 'cheapest':
                    unread.sort(key=lambda i: self.memory.price_of(i) if self.memory.price_of(i) is not None else float('inf'))
            if unread:
                self.trace.append({'kind': 'code_pick', 'item': number, 'pick': unread[0], 'unread': True})
                return unread[0]
        self.trace.append({'kind': 'code_pick', 'item': number, 'pick': best[3], 'fails': best[0], 'score': best[1]})
        return best[3]

    def buy_best(self, picks: dict[int, str]) -> str:
        """Set the cart to the picks (keeping fitting lines of other items) and check out. Returns the store's reply
        text, or '' when nothing could be done."""
        cart, buy = self.shop.role('cart'), self.shop.role('buy')
        if not cart or not buy or not picks or self.finished():
            return ''
        key = self.memory.category_key()
        now = self.cart_ids()
        lines: dict[str, int] = {}
        for number, ident in picks.items():
            lines[ident] = self.wants[number - 1].qty
        covered = {self.memory.label_of(i).lower() for i in lines} if key else set()
        keep = {i: self.held_units(i) or 1 for i in now if i not in lines and not self.memory.sold_out(i) and (not key or self.memory.label_of(i).lower() not in covered) and self.owner_of(self.memory.label_of(i))}
        drop = [i for i in now if i not in lines and i not in keep]
        done = self.set_cart({**keep, **lines}, drop)
        if done is None:
            return ''
        self.complete_views(list(lines), self.tail())
        args, faults = self.shop.shape_args(buy, dict(self.shop.buy_args))
        if faults or self.left() < 1:
            return ''
        self.gate('order')
        if self.finished() or self.left() < 1:
            return ''
        reply = self.oracle.act(buy, args)
        self.trace.append({'kind': 'buy_best', 'lines': lines, 'reply': clip(reply, 200)})
        if reply is None:
            return ''
        self.buys += 1
        if not reply.get('finished'):
            self.refused_at = self.oracle.step
            self.observe(buy, reply.get('observation'))
            self.after_refusal(reply.get('observation'))
        return 'Ghost set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + self.show(reply.get('observation'))

    def after_refusal(self, observation: Any) -> str:
        """A refused checkout: a message about stock marks a single cart line sold out (several lines are re-read),
        fitting lines that sold out are remembered, and a replacement is carted when the steps allow."""
        text = self.spoken(observation).lower()
        cart = self.cart_ids()
        if re.search(SOLD_RX, text) and cart:
            if len(cart) == 1:
                self.memory.unavailable.add(cart[0])
            elif not any(self.memory.sold_out(i) for i in cart):
                self.find_sold()
        hint = self.swap_hint()
        done = self.swap_lost() if hint else ''
        return done

    # -- steps the request asks for ---------------------------------------------------------------------------
    def gate(self, stage: str) -> tuple[bool, str]:
        steps = [p for p in self.process if p['when'] == stage]
        if not steps or self.finished() or not self.gate_due(stage):
            return True, ''
        tools = self.shop.tool_lines(self.endings())
        said: list[str] = []
        nudged = False
        for _ in range(3):
            prompt = (GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + self.request +
                      '\n\nSTEPS DUE NOW:\n' + '\n'.join(f"- {p['step']}" for p in steps) + '\n\nLEDGER:\n' + self.ledger() + '\n\nCART NOW: ' + self.cart_text() +
                      '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(f'{i}. {line}' for i, line in enumerate(self.done_log, 1)) or '(nothing yet)') +
                      ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(self.notes[-4:]) if self.notes else '') +
                      f"\n\nMESSAGES LEFT: {max(0, self.msg_limit - self.msg_used)}   STEPS LEFT: {self.left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)" + '\n\nTOOLS:\n' + tools)
            got = self.oracle.ask_json(prompt, [DEEP, QUICK] if nudged else [QUICK, DEEP], 2500, tag='gate')
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                self.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            self.trace.append({'kind': 'gate', 'stage': stage, 'call': clip(call, 200), 'steps': clip(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                self.memory.version += 1
                self.checked = set()
                return False, ('Before going on, Ghost did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + "The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'
            name = str((call or {}).get('tool') or '')
            shown = [e for e in as_list(got.get('steps')) if isinstance(e, dict)]
            talk = stage == 'order' and any(p['messages'] > 0 for p in steps)
            proof = {i for i in range(1, len(self.done_log) + 1) if not talk or (i <= len(self.done_steps) and self.done_steps[i - 1] > self.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any((as_int(n) or 0) in proof for n in as_list(e.get('lines')))]
            if not call and unproven and not said and not nudged and self.shop.role('ask') and self.msg_used < self.msg_limit and self.left() >= 2:
                nudged = True
                self.done_log.append(f'(Ghost: no line made since the cart last changed at step {self.cart_at} shows ' + '; '.join(str(e.get('step') or '') for e in unproven)[:200] + f' done, and {self.msg_limit - self.msg_used} message(s) and {self.left()} steps are left: give the call that does it)')
                self.done_steps.append(-1)
                continue
            if not call or name not in self.shop.schemas or name in self.endings():
                if not call:
                    self.gate_ok.add((stage, self.cart_mark() + f'#{self.memory.version}'))
                break
            if self.left() < 2 or (name == self.shop.role('ask') and self.msg_used >= self.msg_limit):
                break
            args, faults = self.shop.shape_args(name, call.get('args') if isinstance(call.get('args'), dict) else {})
            if faults:
                break
            if name == self.shop.role('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = self.send([(1, text)], 1)
                if not batch:
                    break
                self.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = self.oracle.act(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            self.memory.absorb(seen)
            self.done_log.append(f'step {self.oracle.step}: {name} {dumps(args)} -> {self.show(seen)[:300]} (cart then: {self.cart_mark()})')
            self.done_steps.append(self.oracle.step)
            said.append(f'{name} {dumps(args)} -> {self.show(seen)[:400]}')
        return True, 'Before going on, Ghost did the steps the request asks for:\n' + '\n'.join(said) if said else ''

    # -- the loop -----------------------------------------------------------------------------------------------
    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        return name == self.shop.role('buy') and all(args.get(k) == v for k, v in self.shop.buy_args.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        return name == self.shop.role('refuse') and all(args.get(k) == v for k, v in self.shop.refuse_args.items())

    def offered(self) -> list[dict[str, Any]]:
        """The tools the model may call now: everything but the messaging tool, narrowed at the very end."""
        left = self.left()
        ask = self.shop.role('ask')
        ends = self.endings()
        keep: set[str] | None = None
        if left <= self.tail():
            keep = set(ends)
        elif left <= self.tail() + 3:
            keep = set(ends) | {self.shop.role('cart'), self.shop.role('cart_view')}
        elif left <= len(self.wants) + 5 and self.buys == 0:
            keep = set(ends) | {self.shop.role('cart'), self.shop.role('cart_view'), self.shop.role('offer')}
        tools = [t for t in self.shop.tools if t['function']['name'] != ask]
        if keep is not None and ends:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            tools = narrowed or tools
        return tools

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A decline's remark: short, naming no listing, unless the request wants alternatives suggested."""
        props = self.shop.props(name)
        out = dict(args)
        for key, spec in props.items():
            if not isinstance(spec, dict) or 'string' not in schema_kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in self.shop.required(name):
                continue
            text = str(value or '')
            named = any(ident in text for ident in self.memory.rows) or self.quiet_decline is not False
            if named or not text:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if name not in {self.shop.role(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not self.memory.id_key:
            return args, ''
        props = self.shop.props(name)
        free = [k for k, v in props.items() if isinstance(v, dict) and ('string' in schema_kinds(v) or 'array' in schema_kinds(v)) and 'enum' not in v]
        if len(free) != 1 or (name != self.shop.role('cart') and (name, free[0]) not in self.shop.id_args):
            return args, ''

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.memory.rows:
                return value.strip(), ''
            if not isinstance(value, str) or looks_like_id(value, True) or len(value.strip()) < 8:
                return value, ''
            low = value.strip().lower()
            hits = [i for i, row in self.memory.rows.items() if any(k != self.memory.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ''
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f'{value!r} is not a listing id; pass the id from the result rows (their {self.memory.id_key} field)'
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
            problem = next((p for _, p in fixed if p), '')
            if problem:
                return args, problem
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ''

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call where the store has one."""
        opener, joiner = self.shop.role('open'), self.shop.role('compare')
        target, many = self.shop.single_target(opener), self.shop.list_target(joiner)
        if not opener or not joiner or not target or not many or opener == joiner or self.facts_from.get(joiner) is False:
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = parse_json_object(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and set(args) == {target} and ident in self.memory.rows and ident not in self.memory.gone:
                picks.append((index, ident))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': dumps({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        self.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    def compare_room(self) -> int:
        tool = self.shop.role('compare')
        spec = self.shop.props(tool).get(self.shop.list_target(tool) or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if self.compare_bad:
            return max(2, min(self.compare_bad - 1, max(self.compare_ok, 2)))
        return max(4, self.compare_ok)

    def before_buy(self, cid: str) -> tuple[str, str] | None:
        """Everything Ghost does or asks before a checkout goes out; a (cid, text) reply holds the call back."""
        cart_text = '\nCart now: ' + self.cart_text()
        lost = [i for i in self.cart_ids() if self.memory.sold_out(i)]
        if lost and self.left() >= 2:
            done = self.swap_lost()
            if not done and self.left() >= len(lost) + 2:
                done = self.drop_sold_lines()
            if done:
                self.notes.append(done[:200])
                cart_text = '\nCart now: ' + self.cart_text()
            if any(self.memory.sold_out(i) for i in self.cart_ids()) and self.left() >= 3:
                return cid, f"Not run yet (no step used). {', '.join(lost)} is sold out: take it out of the cart and add another fitting listing first." + cart_text
        missing = self.items_without_lines()
        empty = bool(missing) and len(missing) == len(self.wants) and not self.cart_ids()
        if missing and ((empty and self.left() >= 3 and self.empty_holds < 3 and self.time_left() > TIME_FLOOR + 30) or (self.line_holds < 1 and self.room() >= 2 * len(missing) + 1 and self.time_left() > EXPLORE_FLOOR)):
            self.line_holds += 1
            self.empty_holds += 1 if empty else 0
            names = ', '.join(self.wants[n - 1].name for n in missing)
            return cid, f'Not run yet (no step used). The cart has no line for: {names}. Cart the best fitting listing for each of them first (search its category if none was read), then check out.' + cart_text
        wider = self.wider_hold()
        if wider:
            return cid, wider + cart_text
        reread = self.reread_offers()
        if self.note_sold:
            self.find_sold()
        if self.cart_touched and not self.cart_fresh() and self.shop.role('cart_view') and self.left() > 3:
            view_args, faults = self.shop.shape_args(self.shop.role('cart_view'), {})
            view = self.oracle.act(self.shop.role('cart_view'), view_args) if not faults else None
            if view is not None:
                self.cart_view = view.get('observation')
                self.cart_seen_at = self.oracle.step
                change = self.observe(self.shop.role('cart_view'), self.cart_view)
                if change:
                    self.queued = []
                    return cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + self.cart_text() + '\nApply the change, fix the cart if needed, then check out.'
        swap = self.swap_hint()
        if swap and self.left() <= 8:
            done = self.swap_lost()
            if done:
                swap = ''
                cart_text = '\nCart now: ' + self.cart_text()
                self.notes.append(done[:200])
        if swap:
            return cid, 'Not run yet (no step used). ' + swap + cart_text
        chain = self.resolve_pointers()
        pending = [need.text for _, need in self.needs_now() if need.points and not need.resolved]
        if pending and self.chain_blocks < 1 and self.room() >= 4 and self.time_left() > EXPLORE_FLOOR:
            self.chain_blocks += 1
            return cid, ('Not run yet (no step used). ' + (chain + '\n' if chain else '') + 'These requirements take their value from another listing and that listing was not found yet: ' + ' | '.join(pending[:3]) +
                         ". Search for it by its shop or brand name together with its kind, open its page, and then check out; Ghost will read the value from the pages." + cart_text)
        misfit = self.fit_faults() if self.fit_holds < 2 and self.left() > self.tail() + 3 else []
        if misfit:
            self.fit_holds += 1
            self.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
            return cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.' + cart_text
        faults = self.line_faults() if self.line_holds < 3 and self.left() > self.tail() + 2 else []
        if faults and all(' is held at ' in f for f in faults):
            fixed = self.fix_counts(self.count_fixes())
            faults = self.line_faults() if fixed else faults
            if fixed and not faults:
                self.line_holds += 1
                return cid, "Not run yet (no step used). Ghost set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.' + cart_text
        if faults:
            self.line_holds += 1
            self.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
            return cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out." + cart_text
        read = self.fill_facts(self.cart_ids(), self.tail() + 3) + self.cheaper_rows()
        swapped_code = self.code_swap()
        if swapped_code:
            read.append(swapped_code)
        viewed = self.complete_views(self.cart_ids(), self.tail() + 1)
        verdict = self.check_cart()
        if verdict and self.better and self.swaps < 2 and self.room() >= 3:
            swapped = self.swap_to_better(self.better)
            if swapped:
                self.swaps += 1
                self.complete_views(self.cart_ids(), self.tail() + 1)
                again = self.check_cart()
                if not again:
                    return None
                return cid, 'Not run yet. ' + swapped + '\n' + again + cart_text
        if verdict and self.view_holds < 2:
            self.view_holds += 1
            seen = "Ghost read these listings' views for the check (one step each):\n" + '\n'.join(read + viewed) + '\n' if read or viewed else ''
            return cid, 'Not run yet. ' + (chain + '\n' if chain else '') + seen + verdict + cart_text
        if reread and self.oracle.cost > SPEND_SOFT:
            return cid, "Not run yet (no step used). Ghost re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return None

    def before_refuse(self, cid: str) -> tuple[str, str] | None:
        """A decline is handed back unless the evidence truly supports it; otherwise Ghost buys the best candidates."""
        if self.left() <= 2 or self.time_left() < TIME_FLOOR + 20:
            return None
        if not self.pushed and not self.leads and self.left() > self.max_steps // 3 and self.time_left() > EXPLORE_FLOOR + 30:
            self.pushed = True
            return cid, f"Not run yet (no step used). {self.left()} steps are still left: search in ways not tried yet (each item's category listing paged on, one-word value queries, the product's other names) and open the listings that might fit. End without a purchase only after that."
        review = self.audit() if self.left() > 5 else ''
        if review:
            return cid, 'Not run yet (no step used). ' + review
        if self.leads and self.left() > 4 and self.refuse_blocks == 0:
            self.refuse_blocks = 1
            return cid, 'Not run yet (no step used). The review found listings that appear to fit: ' + ', '.join(self.leads) + '. Open them if needed, cart the best fitting one per item (respecting the pick rule) and check out. End without a purchase only if each clearly fails a requirement the shopper stated; then say which, and send the ending call again.'
        if self.decline_allowed():
            self.trace.append({'kind': 'decline_allowed', 'blocking': self.blocking})
            return None
        picks = self.forced_pick()
        if picks and self.left() >= len(picks) + 2:
            self.confirmed = next(iter(picks.values()))
            done = self.buy_best(picks)
            if done:
                return cid, 'Not run (Ghost acted instead): no requirement the shopper stated outright rules out every listing, so ' + done + (' [episode finished]' if self.finished() else '')
        if self.refuse_blocks < 2:
            self.refuse_blocks += 1
            return cid, 'Not run yet (no step used). Ending without a purchase earns nothing unless a requirement the shopper stated has no buyable listing; an unverified chain, a missing page or a shortage of steps is not that. Cart the most likely fitting listing for each item and check out.'
        return None

    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for call in calls:
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Ghost opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if self.finished():
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == self.shop.role('ask'):
                out.append((cid, 'Not run (no step used): Ghost sends every message to the shopper itself.'))
                continue
            if name not in self.shop.schemas:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            args = raw if isinstance(raw, dict) else (parse_json_object(raw) if (raw or '').strip() else {})
            if args is None:
                out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                continue
            args, faults = self.shop.shape_args(name, args)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            many = self.shop.list_target(name) if name == self.shop.role('compare') else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and self.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Ghost] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = self.shop.single_target(name)
            if target and name in (self.shop.role('open'), self.shop.role('offer')) and str(args.get(target)) in self.memory.gone:
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {self.memory.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == self.shop.role('cart') and any(isinstance(v, str) and self.memory.sold_out(v) for v in args.values()) and not any((as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str)) and self.shop.remove_option(name) not in args.values():
                gone_ids = [v for v in args.values() if isinstance(v, str) and self.memory.sold_out(v)]
                out.append((cid, f"Not run (no step used): {', '.join(gone_ids)} is sold out and cannot be bought. Add another fitting listing instead."))
                continue
            if name == self.shop.role('cart') and any((as_int(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = dumps(self.cart_view) if self.cart_fresh() else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in self.memory.rows and v not in self.cart_lines and v not in view and v not in self.cart_said]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + self.cart_text()))
                    continue
            ending = self.is_buy(name, args) or self.is_refuse(name, args)
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            if not ending and self.left() <= self.tail() and self.endings():
                out.append((cid, 'Not run (no step used): the last steps are kept for checking out (or for the no-purchase ending).'))
                continue
            mark = name + dumps(args)
            if not ending and name != self.shop.role('cart') and mark in self.recent:
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = self.readers()
            cached = self.read_cache.get(mark)
            if name in readers and cached and cached[0] == self.memory.version:
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and name != self.shop.role('compare') and self.memory.fresh_view(str(args.get(target)), name):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + self.memory.record_text(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.memory.fresh_view(str(i), name)]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(f'{i}: {self.memory.record_text(str(i))}' for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Ghost] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + dumps(args)
            if ending and self.left() > 2 and self.time_left() > TIME_FLOOR + 25:
                held = self.before_buy(cid) if self.is_buy(name, args) else self.before_refuse(cid)
                if held:
                    out.append(held)
                    continue
            elif self.is_buy(name, args) and self.left() >= 2:
                cleaned = self.swap_lost() or self.drop_sold_lines()
                if cleaned:
                    trimmed = '\n[Ghost] ' + cleaned + trimmed
            if self.is_refuse(name, args) and self.left() <= 5 and not self.late_held:
                lines = [i for i in self.built if not self.memory.sold_out(i)]
                missing = [i for i in lines if i not in self.cart_ids() or (i in self.dropped and not self.cart_fresh())]
                if len(lines) >= len(self.wants) and len(missing) + 1 <= self.left():
                    self.late_held = True
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo))
                    continue
            if self.is_refuse(name, args):
                args = self.quiet(name, args)
            if ending and self.left() >= 2 and self.time_left() > TIME_FLOOR + 15:
                go, said = self.gate('order' if self.is_buy(name, args) else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + self.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Ghost] ' + said + trimmed
            reply = self.oracle.act(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.' if not self.finished() else 'Not run: the episode is over.'))
                continue
            observation = reply.get('observation')
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = dumps(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.compare_bad = min(self.compare_bad or 999, len(args[many]))
                    trimmed += f'\n[Ghost] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
                else:
                    self.compare_ok = max(self.compare_ok, len(args[many]))
            self.recent = (self.recent + [mark])[-3:]
            self.shop.id_args.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.memory.rows for x in (v if isinstance(v, list) else [v])))
            self.memory.absorb(observation)
            if name == self.shop.role('compare'):
                rows = self.memory.records_in(observation)
                if rows:
                    self.facts_from[name] = self.facts_from.get(name, False) or any(attribute_block(r) for _, r in rows)
                self.memory.note_view(name, observation)
            if self.shop.single_target(name) and name not in (self.shop.role('cart'), self.shop.role('buy'), self.shop.role('refuse'), self.shop.role('ask')):
                listing = self.memory.ident(observation) if isinstance(observation, dict) else None
                if listing:
                    self.memory.note_view(name, observation)
                    self.facts_from[name] = self.facts_from.get(name, False) or bool(attribute_block(observation))
                    if name == self.shop.role('open') and not self.facts_from[name]:
                        other = next((t for t in self.shop.schemas if t != name and self.shop.single_target(t) and self.facts_from.get(t) is not False and t not in (self.shop.role('cart'), self.shop.role('buy'), self.shop.role('refuse'), self.shop.role('ask'), self.shop.role('search'))), None)
                        if other:
                            self.shop.roles['open'] = other
                            self.shop.roles['offer'] = name
                    if not self.facts_from[name] and name != self.shop.role('open') and not self.shop.role('offer'):
                        self.shop.roles['offer'] = name
                    if name == self.offer_tool():
                        self.offer_seen[listing] = self.oracle.step
                    self.harvest_rules(listing, observation)
            plain_tools = {tool for job, tool in self.shop.roles.items() if job != 'search'}
            paged = self.more_pages(name, args, observation) if name not in plain_tools and not ending else ''
            if name == self.shop.role('cart'):
                self.note_cart(name, args)
                self.cart_said = dumps(observation)
                self.recent = []
                counts = [as_int(v) for v in args.values() if not isinstance(v, str) and as_int(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and value in self.memory.rows:
                        if counts[0] <= 0 and not mentions_id(observation, value):
                            self.dropped.add(value)
                        elif counts[0] > 0:
                            self.dropped.discard(value)
                lines = [i for i in self.cart_ids() if not self.memory.sold_out(i) and i not in self.dropped]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.shop.role('search') and self.memory.records_in(observation):
                self.searched = (self.searched + [dict(args)])[-8:]
                self.tried_words.update(w for v in args.values() if isinstance(v, str) for w in re.findall(r'[a-z0-9]{3,}', v.lower()))
                self.read_shape(name, observation)
                ids_here = [i for i, _ in self.memory.records_in(observation)]
                ignored = self.keyword_ignored(name, args, ids_here)
                self.search_rows[mark] = ids_here
                if ignored:
                    trimmed += ignored
            if name == self.shop.role('cart_view'):
                self.cart_view = observation
                self.cart_seen_at = self.oracle.step
            if self.is_buy(name, args):
                self.buys += 1
                if not reply.get('finished'):
                    self.refused_at = self.oracle.step
            text = self.show(observation) + paged + trimmed
            if name not in plain_tools and not ending:
                text += self.cheapest_here([observation] + self.paged_obs) + self.cap_ignored(name, args, observation)
            base = text
            proof = self.evidence(observation)
            if proof:
                text += '\n' + proof
            if name == self.shop.role('cart') and any(as_int(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.memory.rows and mentions_id(observation, v) for v in args.values()):
                    text += '\n[Ghost] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
            self.note_sold = False
            lines = self.loose_lines(observation)
            if lines:
                meaning = self.take_note(lines)
                if meaning:
                    text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
            refused = self.is_buy(name, args) and not reply.get('finished')
            again = refused and any(line in self.spoken(observation) for line in self.sold_lines)
            loose = self.note_sold and (refused or (not self.note_named and not self.is_buy(name, args)))
            if (loose or again) and self.cart_ids() and not self.finished():
                seen = self.find_sold()
                if seen:
                    text += "\n[Ghost re-read the cart's lines after the refusal]\n" + seen
            self.note_sold = False
            swap = self.swap_hint()
            if swap and refused and not self.finished():
                done = self.swap_lost()
                if done:
                    swap = ''
                    text += '\n[Ghost] ' + done
                    if self.left() <= 6 and self.left() >= 1 and self.time_left() > TIME_FLOOR + 10:
                        again_args, again_faults = self.shop.shape_args(name, dict(args))
                        if not again_faults:
                            retry = self.oracle.act(name, again_args)
                            self.trace.append({'kind': 'swap_buy', 'reply': clip(retry, 200)})
                            if retry is not None:
                                self.buys += 1
                                if not retry.get('finished'):
                                    self.refused_at = self.oracle.step
                                    self.observe(name, retry.get('observation'))
                                text += '\n[Ghost checked out again with the replaced line] ' + self.show(retry.get('observation')) + (' [episode finished]' if retry.get('finished') else '')
            if swap:
                text += '\n' + swap
            if self.queued:
                text += "\n[notes seen on Ghost's own calls] " + ' | '.join(self.queued)
                self.queued = []
            if name in readers:
                self.read_cache[mark] = (self.memory.version, self.oracle.step, base)
            if reply.get('finished'):
                text += ' [episode finished]'
            out.append((cid, text))
        return out

    def finish(self, again: bool = False) -> str:
        """End the episode by code: check out a cart that has lines; otherwise buy the most likely candidates, and
        decline only when the evidence allows it or nothing at all can be bought."""
        if self.finished() or self.oracle.broken or self.left() <= 0:
            return ''
        buy, refuse = self.shop.role('buy'), self.shop.role('refuse')
        holding = bool(self.cart_ids()) if self.cart_fresh() else bool(self.cart_lines)
        if buy and not holding and self.left() >= 3 and self.time_left() > TIME_FLOOR + 20:
            if not self.blocking and self.left() > 6 and self.time_left() > EXPLORE_FLOOR:
                self.audit()
            if not self.decline_allowed():
                picks = self.forced_pick()
                if picks and self.left() >= len(picks) + 2:
                    done = self.buy_best(picks)
                    if self.finished() or done:
                        return ''
            holding = bool(self.cart_ids()) if self.cart_fresh() else bool(self.cart_lines)
        if buy and (holding or (self.cart_touched and not self.cart_fresh() and self.left() >= 2)):
            args, faults = self.shop.shape_args(buy, dict(self.shop.buy_args))
            if not faults:
                lost = [i for i in self.cart_ids() if self.memory.sold_out(i)]
                if lost and self.time_left() > TIME_FLOOR + 20 and not self.swap_lost():
                    if self.left() >= len(lost) + 3:
                        self.drop_sold_lines()
                        if not self.decline_allowed():
                            picks = self.forced_pick()
                            if picks and self.left() >= len(picks) + 2 and self.buy_best(picks):
                                return ''
                    if self.finished() or self.left() <= 0:
                        return ''
                if self.left() >= 3 and self.time_left() > TIME_FLOOR + 20:
                    self.complete_views(self.cart_ids(), 2)
                if self.left() >= 2 and self.time_left() > TIME_FLOOR + 15:
                    self.gate('order')
                if self.finished() or self.left() <= 0:
                    return ''
                reply = self.oracle.act(buy, args)
                self.trace.append({'kind': 'forced_buy', 'reply': clip(reply, 200)})
                if self.finished() or self.left() <= 0:
                    return ''
                if reply is not None and again:
                    self.buys += 1
                    self.refused_at = self.oracle.step
                    self.observe(buy, reply.get('observation'))
                    self.after_refusal(reply.get('observation'))
                    return 'the checkout was refused: ' + self.show(reply.get('observation'))
        if refuse:
            args, faults = self.shop.shape_args(refuse, self.quiet(refuse, dict(self.shop.refuse_args)))
            if not faults:
                if self.left() >= 2 and self.time_left() > TIME_FLOOR + 15:
                    self.gate('end')
                if self.finished() or self.left() <= 0:
                    return ''
                reply = self.oracle.act(refuse, args)
                self.trace.append({'kind': 'forced_refuse', 'reply': clip(reply, 200)})
                if reply is not None and not self.finished():
                    self.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + self.show(reply.get('observation'))
        return ''

    def trim(self) -> None:
        size = sum(len(dumps(m.get('content'))) for m in self.chat)
        if size <= HISTORY_CHARS:
            return
        tools = [i for i, m in enumerate(self.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.chat[index].get('content') or ''
            if isinstance(content, str) and len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in self.memory.rows][:20]
                self.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def work(self, turn_cap: int, spend_cap: float) -> None:
        idle = 0
        while not self.finished() and not self.oracle.broken and self.left() > 0 and self.loop_turns < turn_cap and self.oracle.cost < spend_cap and self.time_left() > EXPLORE_FLOOR:
            if self.ready_to_ask():
                early = ''
                if not self.value_pass and self.room() >= 10:
                    # search the request's values first: the listings found carry the attribute keys the questions draw on
                    self.value_pass = True
                    early = self.value_search(list(range(1, len(self.wants) + 1)), everything=True)
                summary = self.interview()
                ladder = self.price_ladder()
                if early or summary or ladder:
                    self.chat.append({'role': 'user', 'content': '\n'.join(x for x in (early, summary, ladder) if x) + '\n' + self.status()})
                continue
            if not self.value_pass and (self.interviewed or not self.shop.role('ask') or self.msgs_free() <= 0) and len(self.memory.full) >= 1 and self.room() >= 8:
                self.value_pass = True
                searched = self.value_search(list(range(1, len(self.wants) + 1)), everything=True)
                ladder = self.price_ladder()
                if searched or ladder:
                    self.chat.append({'role': 'user', 'content': '\n'.join(x for x in (searched, ladder) if x) + '\n' + self.status()})
                    continue
            pending = [need for _, need in self.needs_now() if need.points and not need.resolved]
            if pending and self.chain_pass < 1 and (self.interviewed or not self.shop.role('ask') or self.msgs_free() <= 0) and self.room() >= 6 and len(self.memory.rows) >= 1:
                self.chain_pass += 1
                chained = self.resolve_pointers()
                if chained:
                    self.chat.append({'role': 'user', 'content': chained + '\n' + self.status()})
                    continue
            self.trim()
            self.loop_turns += 1
            message = self.oracle.think(self.chat, [QUICK, QUICK_SPARE], 3000, self.offered())
            if message is None:
                idle += 1
                if self.oracle.no_credit or self.oracle.calls >= CALL_CAP or idle > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else parse_json_object(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({'id': str(call.get('id') or f'g{self.loop_turns}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': dumps(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = dumps(content) if content else ''
            if not calls:
                idle += 1
                self.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                self.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.shop.role('cart'), self.shop.role('buy'), self.shop.role('refuse')} - {None}
            if not self.interviewed and self.shop.role('ask') and self.msgs_free() > 0 and any(c['function']['name'] in committing for c in calls) and self.left() > self.tail() + len(self.wants) + 2:
                self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                for call in calls:
                    self.chat.append({'role': 'tool', 'tool_call_id': call['id'], 'content': 'Not run (no step used): the shopper is asked first.'})
                summary = self.interview()
                self.chat.append({'role': 'user', 'content': (summary or 'Ghost had nothing to ask.') + '\n' + self.status()})
                continue
            self.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            self.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                self.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def run(self) -> list[dict[str, Any]]:
        if self.finished():
            return self.trace
        self.read_request()
        self.trace.append({'kind': 'brief', 'brief': clip(self.brief, 300), 'roles': self.shop.roles, 'max_steps': self.max_steps, 'messages': self.msg_limit})
        opening = self.view.get('opening')
        if opening is not None:
            self.memory.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + self.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + self.show(opening)
        intro += ('\n\nStart with one search per requested item (its category when a filter exists), then open the attributes of the '
                  'most promising row or two per item, so Ghost can see which properties these listings carry and ask the shopper. '
                  'Cart a fitting listing for each item as soon as you have one.\n\n' + self.status())
        self.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(LOOP_CAP, SPEND_HARD)
        if not self.finished():
            turned = self.finish(again=True)
            if turned and not self.finished() and not self.oracle.broken and not self.oracle.no_credit and self.left() > 1 and self.time_left() > EXPLORE_FLOOR:
                self.chat.append({'role': 'user', 'content': '[Ghost] Ghost tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(self.loop_turns + 10, SPEND_HARD + 0.01)
            if not self.finished():
                self.finish()
        self.trace.append({'kind': 'end', 'finished': self.finished(), 'steps': self.oracle.step, 'cost': round(self.oracle.cost, 5), 'calls': self.oracle.calls, 'seconds': round(time.monotonic() - self.oracle.started, 1)})
        return self.trace


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        pilot = Pilot(problem_data)
    except Exception as exc:  # noqa: BLE001 - the harness must always get a result
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    try:
        return pilot.run()
    except Exception as exc:  # noqa: BLE001
        pilot.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not pilot.finished():
                pilot.finish()
        except Exception as again:  # noqa: BLE001
            pilot.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return pilot.trace


__all__ = ['agent_main']
