"""Best: a shopping agent for ORO store worlds.

An episode is one shopper's request inside a generated store whose tools, result shapes and wording change from
world to world. The code owns the procedure and a small model proposes the store calls:

- the request is read into a ledger (items, firm requirements, budget, pick rule, limits, procedures) and a second,
  stronger read corrects it (requirements hidden in a use or an occasion, story details that are not requirements);
- the interview is run by code: one plain question about one property of one item per message for shoppers who
  answer nothing else, short lists only once the shopper has shown that lists get answered; the properties come
  from the attributes the store's own listings carry and from the hints in the request;
- a requirement whose value lives on another listing's page is resolved by opening every page of that chain, both
  views of each link where the store splits attributes from the offer page, and pairing rules found on pages are
  kept;
- the kind of product is the store's own category label, matched exactly when an exact label exists; a listing
  counts as meeting a requirement only when its evidence is consistent: a key naming the property states the value
  and no other key or the title says otherwise;
- the cart is checked before every checkout; for a cheapest pick the store is searched below the cart's price;
- an ending without a purchase stands only when a requirement the shopper stated outright is met by nothing the
  store sells after its category was searched; everything else ends with the best fitting purchase;
- every episode stays inside its step, message, time and spending budgets and is always ended by the code itself.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

# ------------------------------------------------------------------------------------------------------------
# settings
# ------------------------------------------------------------------------------------------------------------
FAST = 'anthropic/claude-haiku-5.5'           # drives the store loop and the small readings
FAST_ALT = 'openai/gpt-6-luna'                # takes over when the fast model is unavailable
SLOW = 'anthropic/claude-sonnet-5.5'          # reviews the brief, plans questions, checks the cart, judges a decline
SLOW_ALTS = ('anthropic/claude-sonnet-5', 'openai/gpt-6-sol')

CLOCK = 300.0            # seconds the validator allows one task
CLOCK_FLOOR = 14.0       # never start a model or store call with less than this left
WRAP_UP_AT = 55.0        # with this much time left, stop exploring and finish
MODEL_CALLS = 84         # model requests per episode
LOOP_TURNS = 40
SPEND_SOFT = 0.034       # USD: above this no slow-model call is started
SPEND_HARD = 0.060       # USD: above this the loop stops; only the ending's few fast calls remain
ENDING_ROOM = 0.008      # USD the fast model may still spend on ending the episode above the hard cap
SLOW_ROOM = 0.022        # USD a slow call may still cost near the soft cap
RETRY_WAITS = (3, 8, 16, 30)
RETRY_BUDGET = 42.0
COOL_OFF = 20.0
ROW_CHARS = 420
OBS_CHARS = 14000
CHAT_CHARS = 140000
STEPS_IF_UNKNOWN = 30
MESSAGES_IF_UNKNOWN = 3
OWN_PAGES = 6            # result pages read by code for a cheapest/most/least pick
CHEAPER_OPENS = 4
CHAIN_HOPS = 6
CHAIN_STEPS = 14
RESCUE_STEPS = 8
SWEEP_STEPS = 6
OPENS_PER_ITEM = 3
LIST_SIZE = 1            # properties per message: one, always (a list hides every need but the first it names)

# ------------------------------------------------------------------------------------------------------------
# prompts
# ------------------------------------------------------------------------------------------------------------
GUIDE = (
    "You are Best, a buyer acting for one shopper inside an online store. You act only through the store's tools, "
    "one tool call per step. The episode scores 1 only when the ending is exactly right: a checkout whose cart holds "
    "every requested item at its quantity, each meeting every requirement in force at that moment, inside the "
    "budget; or, when the store truly sells nothing meeting a requirement the shopper stated, the no-purchase ending "
    "done the way the request asks. Anything else scores 0; effort earns nothing.\n\n"
    "THE STORE\n"
    "- Tool names, argument names and result shapes belong to this store: read each schema and use the exact "
    "argument names and allowed values. Every call is a step and steps are scarce; thinking is free.\n"
    "- Searches are narrow: a category filter, a keyword mode that reads only a few words (every word must match, so "
    "keep queries to two or three words: the wanted value plus the product noun, and try the value's other word "
    "forms when one finds nothing: knit, knitted, knits), short pages with a token or page number for the next page. Compare the total reported with the rows shown; page on only when the answer depends "
    "on seeing all rows. A filter that returns nothing usually has the wrong label: use the label exactly as rows "
    "show it.\n"
    "- Rows are summaries. A listing's attributes live in its full record; stores sometimes split that record into an "
    "attribute view and an offer page (price, stock, free-text notes). Read the attribute view of every listing you "
    "might buy. Never open a view again when nothing has changed since you read it.\n"
    "- Best sends every message to the shopper itself; you never call the messaging tool. Answers arrive in the "
    "conversation and land in the ledger.\n"
    "- Free-text notes on a page can point to ANOTHER listing for one property (\"for X, see the ... from <shop>\"), "
    "state store rules (\"<value> goes together with <value>\"), or mention versions sold elsewhere. Best follows "
    "such chains itself when a requirement needs them; you open pages when asked.\n\n"
    "WHAT FITS\n"
    "- The kind of product is the listing's category label. When the store has a label that matches the request's "
    "words exactly, only listings under that label count; a neighbouring label (a broader, narrower or related "
    "category) does not, even when its titles sound right.\n"
    "- A property requirement is met when the listing's attribute for that property states the value (any key "
    "spelling; one element of a list is enough; synonyms, case and plural are fine; a value containing the wanted "
    "words counts; units convert; limits include the edge). Evidence has grades: a listing whose keys for the "
    "property all state the value is solid (spellings of one key are one attribute, a list containing the value "
    "counts, and Best grades this for you: trust its strong grade over your own reading); one whose keys disagree "
    "with each other is doubtful (the store's hidden truth often sides against it); one with no key for the "
    "property fits through its title or description. Always prefer a solid candidate over a doubtful one, but a "
    "doubtful candidate is still a candidate: buy it rather than end without a purchase. A key for a different "
    "property decides nothing; a bare number of unknown meaning counts for nothing.\n"
    "- A use or occasion the shopper insists on for the product (it has to be for that; they will check it) is a "
    "requirement on the listing's occasion, purpose or audience; a use named only as the reason for shopping is a "
    "wish. Exclusions and closed "
    "lists are strict. Wishes are tie-breakers only. Store rules pairing values (\"a goes with b\") make the paired "
    "value acceptable where the other was asked for.\n"
    "- When one item must fit another (an accessory for a device), judge the pair by the listings' own words: a "
    "model or series code both pages carry, or a list of what it fits.\n"
    "- Never buy anything shown sold out; price times quantity summed over all lines stays inside the budget.\n\n"
    "CHOOSING\n"
    "- Cheapest pick: the order must hold the lowest-priced fitting listing among all buyable ones. Rows come in no "
    "price order, so see the whole set (a category listing under a price cap when the store has one; lower the cap "
    "until every match is visible) and check candidates from the cheapest up. Most/least of a measure: compare every "
    "fitting listing; one that does not state the measure cannot win. Otherwise any consistently fitting listing is "
    "right: do not hunt for a better one.\n"
    "- Work item by item. Cart each item as soon as one listing meets everything you know, then move on; a carted "
    "line can be replaced, an empty cart cannot be rescued when steps run out. Several items need one line per item "
    "at its quantity. Decide before you cart: a swap costs two steps, so compare candidates from their records first "
    "and change the cart only when a line is sold out, a change demands it, or a clearly better listing turns up "
    "under a cheapest/most/least rule.\n\n"
    "CHANGES\n"
    "- The shopper or the store can change the task: a new budget, a requirement added or dropped, a price change, a "
    "sold-out line, a refused checkout with a reason. Such notes arrive inside tool results, most often attached to "
    "a result or as the reply to a checkout. Apply them at once: a line reported sold out leaves the cart (never "
    "check out with it), a dropped requirement stops binding, a new one binds; then fix the cart and check out "
    "again. Best keeps the last steps free for exactly this.\n\n"
    "ENDING\n"
    "- Set the cart to exactly the chosen lines, then call the checkout tool on its own. Best runs its own checks "
    "first and may hand the call back with what to fix.\n"
    "- The no-purchase ending is right only when a requirement the shopper stated outright is met by no buyable "
    "listing after the category was searched; a requirement you could not verify, a missing page or a shortage of "
    "steps is never a reason. In doubt, buy the best fitting candidates.\n"
    "- Every reply calls at least one tool. Keep text to a line: what you know and the next move."
)

BRIEF = (
    "You prepare a shopping errand for an agent. From the shopper's request and the store's tools, return ONE JSON "
    "object (compact, nothing else) with these keys.\n"
    "\"items\": one entry per distinct product the shopper wants bought: {\"name\": the kind of product in the "
    "request's own words, \"shelf\": the store's category label for it when a tool schema lists labels (copied "
    "exactly) else \"\", \"quantity\": units of that listing (integer; 1 when unstated; a pair or set sold as one "
    "listing is 1), \"needs\": firm requirements on that product as \"property: constraint\" strings with the exact "
    "value, limit, exclusion or closed list, naming the property with the request's word for it. A requirement "
    "whose value is read from another listing is ONE need carrying the whole description of that listing. Never put "
    "price, budget, stock or quantity into needs. A use or occasion the shopper names for the product is a need on its "
    "occasion, purpose or audience only when the shopper states it as a condition on the product (it has to / must / "
    "is got to be for that; they will check it on the listing); named merely as the reason for shopping or as "
    "background (\"another skirt for class\", \"before my shift\") it is a wish. Anything the shopper says they "
    "will check on the listing is a need. The rest of "
    "the story (what broke before, who they are, the weather, when they get paid) is context, not a need, unless it "
    "states a property's value or limit outright; a word that is part of the product's name (a season, an audience) "
    "belongs to \"name\". A comparative or vague word without a value (longer, bigger, decent, warm) is a hint, never "
    "a need. One \"X and Y\" phrase with one quantity is ONE item; split only when each product gets its own "
    "quantity or requirements.}\n"
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

REVIEW = (
    "A fast reader turned a shopper's request into the errand below. Check it against the request sentence by "
    "sentence and return ONE JSON object and nothing else with the corrections:\n"
    "{\"add\": [{\"item\": n, \"need\": \"property: constraint\"} for each firm requirement on a product the "
    "reader missed: a use or occasion the shopper insists on for the product (has to be for that, will check it), a "
    "property the shopper says they will check, an exclusion, a closed list, a limit with its value],\n"
    " \"drop\": [{\"item\": n, \"need\": the need copied exactly} for each listed need that only repeats the story "
    "(what broke before, who the shopper is, the weather, when they pay, the activity they shop for without insisting "
    "the product be made for it) and that the request never demands of the product, or that duplicates another "
    "need],\n"
    " \"items\": null, or the corrected list of items [{\"name\", \"quantity\"}] when the reader split one product "
    "in two, merged two, or got a quantity wrong,\n"
    " \"pick\": null or the corrected pick rule (\"cheapest\", \"most\", \"least\", \"any\"),\n"
    " \"budget\": null or the corrected total limit as a number,\n"
    " \"process\": null or the corrected list of steps the request requires beyond finding and buying "
    "[{\"step\", \"when\": \"choose\"|\"order\"|\"end\", \"messages\"}],\n"
    " \"hints\": [{\"item\": n, \"text\": a sentence of the story pointing at an unwritten preference, at most 14 "
    "words, \"property\": the one listing property it most likely concerns}] the reader missed, else []}.\n"
    "Be strict about \"add\": a need must be a property the listing itself can show; stock, availability, price, the "
    "budget, quantity, message and step limits and the way to end are never needs. Be strict about \"drop\": keep "
    "every requirement the shopper states outright, even folksy ones, but drop needs the reader invented from the "
    "story (an event, a place, a person, something that broke) and needs about stock or price."
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
    "about are listed with each question). Read each answer and return ONE JSON object and nothing else: "
    "{\"needs\": [{\"item\": item number, \"need\": \"property: constraint\", \"points\": true when the value must "
    "be read from another listing, else false}], \"drop\": [{\"item\": n, \"property\": \"a requirement the shopper "
    "now gives up\"}], \"free\": [{\"item\": n, \"property\": \"a property the shopper says they do not mind\"}] "
    "(an answer that states no preference at all makes every property that question asked about free), "
    "\"lost\": [question numbers whose answer only says the shopper did not follow, wants it simpler, repeated, or one "
    "thing at a time, and states nothing], \"budget\": a new total limit as a number, or null}\n"
    "- A need is a firm preference the answer states, named by the property the ANSWER names (it may differ from the "
    "one asked), with the exact value or limit and unit copied from the answer (\"at least 3 shelves\", \"not wicker\"). "
    "Words that only join the property to its value (set to, landing at, coming in as, has to match, reading) are not "
    "part of the value: \"size landing at large\" states large.\n"
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
    "a listing meets a property requirement when its attribute for that property states the value (any key "
    "spelling, one element of a list is enough, synonyms, case and plural count, a value containing the wanted "
    "words counts, units convert, limits include the edge, a range must lie wholly inside a limit; a neighbouring or "
    "broader value is not the stated value); when two keys "
    "name the same property and disagree, or the title names a different value for that property, the listing is "
    "doubtful and a candidate whose keys agree is preferred, but a doubtful listing still fits when no solid one "
    "exists; a key for a different "
    "property decides nothing; without any key naming the property, the value stated in the title, description or "
    "another attribute counts, and a title stating a different value fails it; a bare figure of unknown meaning "
    "counts for nothing; a value the category label itself states is met by the category; a store rule pairing two "
    "values makes either acceptable for the other; \"different from X\" is met by any value other than X; \"within "
    "N of X\" is a numeric band around X; out-of-stock listings and listings that break the budget do not fit; a "
    "listing under a category label other than the item's own label is not the item's kind"
)

CHECK = (
    "You are the last check before a shopping agent checks out. From the records shown, decide whether the cart is "
    "exactly right. Reading rules: " + READING + ". Each requested item needs its own cart line at the requested "
    "quantity, nothing else may be in the cart, nothing shown out of stock, and price times quantity summed over "
    "the lines must stay within the budget. The newest record of a listing wins. When the pick rule is cheapest, "
    "most or least, also look at the other listings shown: if one clearly fits every requirement and beats the "
    "cart's line on that rule, the cart is wrong. Under \"better\", also name a listing that fits with stronger "
    "evidence (keys naming the property outright, all agreeing) when the cart's line fits only through its title, "
    "description or a neighbouring key, or carries a disagreeing key, and the pick rule does not forbid the swap. "
    "The EVIDENCE BY CODE lines, when given, are authoritative on key agreement: they read every spelling of a key "
    "as one attribute, a list containing the value counts, and a sibling key naming one element of that list is not "
    "a disagreement. Never call a line graded strong doubtful, and never name a pricier listing as better than a "
    "strong line under a cheapest pick.\n"
    "Return ONE JSON object and nothing else: {\"ok\": true or false, \"problems\": [\"short concrete problem\"], "
    "\"better\": [listing ids that should replace a line, best first]}. Use ok=false only for a problem the records "
    "clearly show or a requirement no record addresses."
)

AUDIT = (
    "A shopping agent is about to end without buying because it believes nothing fits. Judge that from the records, "
    "with these reading rules: " + READING + ". Records marked ROW were seen only in search results; their "
    "attributes were not read, so they can neither pass nor fail on a property the row does not show.\n"
    "First compare every requirement with the REQUEST and the ANSWERS: a value that picked up joining words goes "
    "under \"corrected\" with the plain value. Judge the listings with the corrected values. Keep the answer "
    "compact: ids only, no commentary outside the JSON.\n"
    "Return ONE JSON object and nothing else: {"
    "\"corrected\": [{\"item\": n, \"need\": copied exactly, \"value\": \"property: plain value\"}], \"fits\": "
    "{\"item number\": [ids of listings that appear to meet every requirement, cheapest first, at most 4]}, "
    "\"maybe\": {\"item number\": [ids of ROW records whose row contradicts nothing, or of opened records that state every "
    "requirement except one that only an unopened offer page could show, cheapest first, at most 4]}, \"open\": [ids of ROW listings of a requested kind within the budget whose attributes could "
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
    "contradicts none, and among those the one whose attribute keys name the required properties outright and "
    "agree; then apply the pick rule (cheapest when asked, else any). Imperfect evidence (a disagreeing key, a value "
    "found only in the title) still makes a candidate: choose it. Give \"\" for an item only when every record of "
    "its kind plainly states a different value for a requirement the shopper stated outright. Return ONE JSON object "
    "and nothing else: {\"choice\": {\"item number\": listing id or \"\"}, "
    "\"reason\": \"one sentence\"}"
)

FIT_WORDS = (
    r"(?:works? (?:with|on)|compatible with|compatibility|fitment|fits?|to fit|for use with|usable with|"
    r"designed for|made for|built for|intended for|suits?|goes with|pairs? with|replacement for|replaces|"
    r"supports?|takes? (?:part )?series)"
)
FIT_EXCEPT = r"\b(?:except(?:\s+for)?|excluding|other than|but not|apart from|not including|not for)\b"
CODE_RX = r"[A-Z][A-Za-z]{0,7}(?:[- ]?\d{2,6}|[- ]?\d{1,5}[xX]{1,4})"
SERIES_RX = r"\b([A-Z]{1,5}\d{1,3})(?=-\d{2,6}\b)"
SKIP_KEYS = r"\b(?:options?|variants?|models?|sku|ids?|codes?|asin|upc|ean|barcodes?|warrant\w*|video|url|links?|images?|photos?|condition|shipping|delivery|origin)\b"
MIXED_KEYS = r"^(?!.*\b(?:types?|styles?|kinds?|shapes?)\b).*\b(?:features?|functions?|highlights?|specs?|specifications?|descriptions?|includ\w*|accessor\w*|extras?|misc\w*)\b"
POINTER_RX = r"\b(?:same as|identical to|different from|unlike|partner|counterpart|within .{1,30} of|the one (?:named|pointed|on the page)|on the page of|listing (?:sold|from|by)|points? (?:at|to))\b"


# ------------------------------------------------------------------------------------------------------------
# small helpers: text, numbers, json, schemas
# ------------------------------------------------------------------------------------------------------------
def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), default=str)


def shorten(value: Any, limit: int, depth: int = 0) -> Any:
    """A copy of a JSON-like value with every string cut to `limit` characters."""
    if depth > 7:
        return '...'
    kind = type(value)
    if kind is str:
        return value if len(value) <= limit else value[:limit] + '...'
    if kind is dict:
        return {key: shorten(inner, limit, depth + 1) for key, inner in value.items()}
    if kind is list:
        return [shorten(inner, limit, depth + 1) for inner in value]
    return value


def first_json(text: Any) -> dict[str, Any] | None:
    """The first JSON object inside a model reply, fenced or not."""
    if not isinstance(text, str):
        return None
    body = text.strip()
    if body.startswith('```'):
        body = re.sub(r'^```[A-Za-z]*\s*|\s*```$', '', body).strip()
    decoder = json.JSONDecoder()
    start = body.find('{')
    while 0 <= start < len(body):
        try:
            found, _ = decoder.raw_decode(body, start)
        except ValueError:
            start = body.find('{', start + 1)
            continue
        return found if isinstance(found, dict) else None
    return None


class Words:
    """Plain spellings of attribute keys and property names, and how alike two of them are."""

    GLUE = frozenset(('and', 'for', 'the', 'with', 'from', 'that'))

    @staticmethod
    def plain(key: Any) -> str:
        pieces = re.findall(r'[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+|[^\W_]+', str(key))
        return ' '.join(piece.lower() for piece in pieces)

    @classmethod
    def stems(cls, text: Any) -> frozenset[str]:
        out = []
        for word in cls.plain(text).split():
            if re.search(r'(?:ch|sh|ss|x|z)es$', word):
                word = word[:-2]
            elif word.endswith('s') and len(word) > 3 and not word.endswith('ss'):
                word = word[:-1]
            out.append(word)
        return frozenset(out)

    @classmethod
    def content(cls, text: Any) -> set[str]:
        return {w for w in cls.stems(text) if len(w) > 2 and w not in cls.GLUE}

    @classmethod
    def alike(cls, a: Any, b: Any) -> bool:
        """Whether two property names mean the same property: equal word sets, or one inside the other."""
        left, right = cls.stems(a), cls.stems(b)
        if not left or not right:
            return False
        if left == right or left < right:
            return True
        return right < left and len(right) > 1

    @staticmethod
    def spoken(observation: Any) -> str:
        """What a shopper (or the store) said inside an observation: every string of two or more words."""
        if isinstance(observation, str):
            return observation
        said: list[str] = []
        for node, _, _ in nodes(observation, 4):
            if isinstance(node, str) and len(node.split()) > 1:
                said.append(node)
        return ' '.join(said)[:800] if said else compact(observation)[:400]


def whole(value: Any) -> int | None:
    """An integer when the value is one (an int, a whole float, or a string of digits)."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value) if isinstance(value, (int, float, str)) else None
    except ValueError:
        return None
    if number is None or number != number or number != int(number):
        return None
    if isinstance(value, str) and not re.fullmatch(r'\s*-?\d+(?:\.0+)?\s*', value):
        return None
    return int(number)


def amount(value: Any) -> float | None:
    """A number read from a number or a short priced string such as '$12.50' or '64 gb'."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    text = value.replace(',', '').strip()
    hit = re.fullmatch(r'[^\d-]{0,4}(-?\d+(?:\.\d+)?)\s*[a-zA-Z%]{0,6}', text)
    return float(hit.group(1)) if hit else None


class Schema:
    """One JSON-schema node: bends a value towards it and reports what cannot bend."""

    CHECKS: dict[str, tuple[type, ...]] = {'string': (str,), 'integer': (int,), 'number': (int, float), 'boolean': (bool,), 'array': (list,), 'object': (dict,)}

    def __init__(self, spec: Any):
        self.spec: dict[str, Any] = spec if isinstance(spec, dict) else {}

    @staticmethod
    def kinds(spec: Any) -> list[str]:
        kind = spec.get('type') if isinstance(spec, dict) else None
        if isinstance(kind, str):
            return [kind]
        return [k for k in kind or [] if isinstance(k, str)]

    @classmethod
    def accepts(cls, value: Any, kind: str) -> bool:
        if kind == 'null':
            return value is None
        allowed = cls.CHECKS.get(kind)
        if allowed is None:
            return True
        if isinstance(value, bool) and bool not in allowed:
            return False
        return isinstance(value, allowed)

    @staticmethod
    def bend(value: Any, kinds: list[str]) -> Any:
        for kind in kinds:
            if kind in ('integer', 'number'):
                number = amount(value) if isinstance(value, str) and re.fullmatch(r'\s*-?\d+(\.\d+)?\s*', value) else (value if isinstance(value, float) else None)
                if number is not None:
                    return int(number) if kind == 'integer' and number == int(number) else number
            elif kind == 'string' and isinstance(value, (int, float)) and not isinstance(value, bool):
                return str(value)
            elif kind == 'boolean' and isinstance(value, str) and value.strip().lower() in ('true', 'false'):
                return value.strip().lower() == 'true'
            elif kind == 'array' and value is not None and not isinstance(value, list):
                return [value]
        return value

    def fit(self, value: Any, where: str, faults: list[str]) -> Any:
        spec = self.spec
        if not spec:
            return value
        kinds = self.kinds(spec)
        if kinds and not any(self.accepts(value, k) for k in kinds):
            value = self.bend(value, kinds)
            if not any(self.accepts(value, k) for k in kinds):
                faults.append(f"{where} should be {'/'.join(kinds)}")
                return value
        value = self._listed(value, where, faults)
        if isinstance(value, str):
            value = self._text(value, where, faults)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            self._range(value, where, faults)
        elif isinstance(value, list):
            value = self._array(value, where, faults)
        elif isinstance(value, dict):
            value = self._object(value, where, faults)
        return value

    def _listed(self, value: Any, where: str, faults: list[str]) -> Any:
        options = self.spec.get('enum')
        if not isinstance(options, list) or value in options:
            return value
        if isinstance(value, str) and len(value.strip()) >= 3:
            low = value.strip().lower()
            exact = [o for o in options if isinstance(o, str) and o.lower() == low]
            loose = exact or [o for o in options if isinstance(o, str) and (low in o.lower() or o.lower() in low)]
            if loose and len(set(loose)) == 1:
                return loose[0]
        faults.append(f'{where}: allowed values are {options}')
        return value

    def _text(self, value: str, where: str, faults: list[str]) -> str:
        longest, shortest = self.spec.get('maxLength'), self.spec.get('minLength')
        if isinstance(longest, int) and len(value) > longest:
            value = value[:longest]
        if isinstance(shortest, int) and len(value) < shortest:
            faults.append(f'{where} is too short')
        return value

    def _range(self, value: float, where: str, faults: list[str]) -> None:
        low, high = self.spec.get('minimum'), self.spec.get('maximum')
        if isinstance(low, (int, float)) and value < low:
            faults.append(f'{where} is below {low}')
        if isinstance(high, (int, float)) and value > high:
            faults.append(f'{where} is above {high}')

    def _array(self, value: list[Any], where: str, faults: list[str]) -> list[Any]:
        most, least = self.spec.get('maxItems'), self.spec.get('minItems')
        if isinstance(most, int) and len(value) > most:
            faults.append(f'{where} allows at most {most} entries')
        if isinstance(least, int) and len(value) < least:
            faults.append(f'{where} needs at least {least} entries')
        inner = self.spec.get('items')
        if isinstance(inner, dict):
            child = Schema(inner)
            value = [child.fit(v, f'{where}[{i}]', faults) for i, v in enumerate(value)]
        return value

    def _object(self, value: dict[str, Any], where: str, faults: list[str]) -> dict[str, Any]:
        props = self.spec.get('properties') if isinstance(self.spec.get('properties'), dict) else {}
        if self.spec.get('additionalProperties') is False:
            faults.extend(f'{where}.{name} is not an allowed argument' for name in value if name not in props)
        faults.extend(f'{where}.{name} is required' for name in self.spec.get('required') or [] if name not in value)
        out = {}
        for name, inner in value.items():
            out[name] = Schema(props[name]).fit(inner, f'{where}.{name}', faults) if name in props else inner
        return out


# ------------------------------------------------------------------------------------------------------------
# records: walking observations, finding identifiers, attributes and prose
# ------------------------------------------------------------------------------------------------------------
def nodes(value: Any, deepest: int):
    """Every node of a JSON-like value, depth first: (node, depth, inside_a_list)."""
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    while stack:
        node, depth, listed = stack.pop()
        yield node, depth, listed
        if depth >= deepest:
            continue
        if isinstance(node, dict):
            stack.extend((inner, depth + 1, False) for inner in reversed(list(node.values())))
        elif isinstance(node, list):
            stack.extend((inner, depth + 1, True) for inner in reversed(node))


class IdSense:
    """Recognises listing identifiers in rows and records."""

    @staticmethod
    def is_id(value: Any, named: bool = False) -> bool:
        if not isinstance(value, str):
            return False
        text = value.strip()
        if len(text) < 3 or len(text) > 64 or ' ' in text:
            return False
        return named or any(ch.isdigit() for ch in text)

    @classmethod
    def key_of(cls, rows: list[dict[str, Any]]) -> str | None:
        """The key whose values look like identifiers in every row and differ between rows."""
        ranked: list[tuple[tuple[bool, float], str]] = []
        for key in rows[0]:
            named = 'id' in str(key).lower()
            values = [row.get(key) for row in rows]
            if not all(cls.is_id(v, named) for v in values):
                continue
            texts = [str(v).strip() for v in values]
            if len(set(texts)) < len(texts):
                continue
            ranked.append(((named, sum(map(len, texts)) / len(texts)), str(key)))
        ranked.sort(reverse=True)
        return ranked[0][1] if ranked else None


def facts_of(record: Any) -> dict[str, Any] | None:
    """The attribute block of a listing record: its largest nested object of scalar values."""
    if not isinstance(record, dict):
        return None
    blocks = []
    for value in record.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2:
                blocks.append(scalars)
    blocks.sort(key=len, reverse=True)
    return blocks[0] if blocks else None


def prose_of(record: Any) -> str:
    """Every string of three or more words inside a record: where notes and descriptions live."""
    return ' '.join(node for node, _, _ in nodes(record, 6) if isinstance(node, str) and len(node.split()) >= 3)


def names_id(node: Any, ident: str) -> bool:
    """Whether some object inside `node` carries `ident` as one of its string values."""
    for inner, _, _ in nodes(node, 5):
        if isinstance(inner, dict) and any(isinstance(v, str) and v.strip() == ident for v in inner.values()):
            return True
    return False


def codes_in(text: Any) -> list[str]:
    return list(dict.fromkeys(m.strip() for m in re.findall(CODE_RX, str(text or ''))))


def same_code(a: str, b: str) -> bool:
    left, right = re.sub(r'[- ]', '', a).upper(), re.sub(r'[- ]', '', b).upper()
    return left == right or (len(left) >= 4 and (left.startswith(right) or right.startswith(left)))


def fit_marks(text: str) -> dict[str, Any]:
    """Compatibility statements in a listing's text: codes it works with, its own code, exceptions, its series."""
    works: list[str] = []
    excluded: list[str] = []
    for found in re.finditer(FIT_WORDS + r'[:\s]+([^.;\n]{0,160})', text, re.I):
        chunk = found.group(1)
        halves = re.split(FIT_EXCEPT, chunk, maxsplit=1, flags=re.I)
        works += codes_in(halves[0])
        if len(halves) > 1:
            excluded += codes_in(halves[1])
    rest = re.sub(FIT_WORDS + r'[:\s]+[^.;\n]{0,160}', ' ', text, flags=re.I)
    own = codes_in(rest)
    return {'works': list(dict.fromkeys(works)), 'excluded': list(dict.fromkeys(excluded)), 'own': own[0] if own else None, 'series': list(dict.fromkeys(re.findall(SERIES_RX, text)))}


def fit_verdict(part: dict[str, Any], device: dict[str, Any]) -> bool | None:
    """Whether a part's statements fit a device's codes: True, False, or None when neither side says."""
    targets = [c for c in [device.get('own'), *(device.get('works') or [])] if c]
    lines = set(device.get('series') or [])
    if not targets and not lines:
        return None
    for code in part.get('excluded') or []:
        if any(same_code(code, t) for t in targets):
            return False
    for code in part.get('works') or []:
        if any(same_code(code, t) for t in targets):
            return True
    if lines & set(part.get('series') or []):
        return True
    return False if (part.get('works') or part.get('series')) else None


# ------------------------------------------------------------------------------------------------------------
# the ledger: requested items and their requirements
# ------------------------------------------------------------------------------------------------------------
class Rule:
    """One firm requirement on one item: "property: constraint".

    `source` is who stated it: 'request', 'shopper', 'store' (a note), 'chain' (read from another page). A pointer
    rule (`points`) takes its value from another listing; `resolved` holds the plain constraint afterwards."""

    __slots__ = ('text', 'source', 'points', 'resolved', 'dropped')

    def __init__(self, text: str, source: str, points: bool = False):
        self.text = text.strip()
        self.source = source
        self.points = points
        self.resolved: str | None = None
        self.dropped = False

    @property
    def prop(self) -> str:
        return self.text.partition(':')[0].strip()

    @property
    def value(self) -> str:
        return self.text.partition(':')[2].strip()

    def shown(self) -> str:
        return f'{self.text} (resolved: {self.resolved})' if self.points and self.resolved else self.text


class Item:
    """One requested product."""

    def __init__(self, name: str, shelf: str, qty: int):
        self.name = name
        self.shelf = shelf                 # the store's category label, once known
        self.label_sure = False            # whether `shelf` was confirmed by the store's own rows
        self.qty = qty
        self.rules: list[Rule] = []
        self.free: list[str] = []          # properties the shopper does not mind
        self.asked: list[str] = []         # properties asked about already
        self.lost: list[str] = []          # properties whose question the shopper did not follow

    def active(self) -> list[Rule]:
        return [r for r in self.rules if not r.dropped]

    def has(self, text: str) -> bool:
        wanted = text.strip()
        return any(r.text == wanted for r in self.active())

    def require(self, text: Any, source: str, points: bool = False) -> Rule | None:
        text = str(text or '').strip()
        if not text or self.has(text):
            return None
        for old in self.rules:
            if old.text == text:
                old.dropped, old.source = False, source
                return old
        rule = Rule(text, source, points)
        self.rules.append(rule)
        return rule

    def release(self, prop: str) -> list[str]:
        gone = []
        for rule in self.active():
            if rule.text == prop or Words.alike(rule.prop, prop):
                rule.dropped = True
                gone.append(rule.text)
        return gone

    def settled(self) -> list[str]:
        return [r.prop for r in self.active()] + self.free + self.asked


# ------------------------------------------------------------------------------------------------------------
# the tool box: this world's tools and the job each one does
# ------------------------------------------------------------------------------------------------------------
class Toolbox:
    """The world's tools as their schemas describe them, plus which job each tool does."""

    JOBS = ('search', 'open', 'offer', 'compare', 'ask', 'cart', 'cart_view', 'buy', 'refuse')

    def __init__(self, tools: Any):
        self.raw: list[dict[str, Any]] = []
        self.specs: dict[str, dict[str, Any]] = {}
        self.blurbs: dict[str, str] = {}
        for tool in tools if isinstance(tools, list) else []:
            fn = tool.get('function') if isinstance(tool, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get('name'), str):
                continue
            params = fn.get('parameters')
            self.raw.append(tool)
            self.specs[fn['name']] = params if isinstance(params, dict) else {'type': 'object', 'properties': {}}
            self.blurbs[fn['name']] = str(fn.get('description') or '')
        self.jobs: dict[str, str] = {}
        self.buy_args: dict[str, Any] = {}
        self.refuse_args: dict[str, Any] = {}
        self.browse: dict[str, Any] | None = None
        self.id_slots: set[tuple[str, str]] = set()      # (tool, argument) pairs that took listing ids

    # -- jobs --------------------------------------------------------------------------------------------
    def names(self) -> list[str]:
        return list(self.specs)

    def tool(self, job: str) -> str | None:
        return self.jobs.get(job)

    def job_of(self, name: str) -> str | None:
        for job, tool in self.jobs.items():
            if tool == name:
                return job
        return None

    def endings(self) -> set[str]:
        return {t for t in (self.tool('buy'), self.tool('refuse')) if t}

    def readers(self) -> set[str]:
        return {t for t in (self.tool(j) for j in ('search', 'open', 'offer', 'compare')) if t}

    def committers(self) -> set[str]:
        return {t for t in (self.tool(j) for j in ('cart', 'buy', 'refuse')) if t}

    # -- schemas -----------------------------------------------------------------------------------------
    def props(self, name: str | None) -> dict[str, Any]:
        spec = self.specs.get(name or '') or {}
        found = spec.get('properties')
        return found if isinstance(found, dict) else {}

    def required(self, name: str | None) -> list[str]:
        spec = self.specs.get(name or '') or {}
        return [k for k in spec.get('required') or [] if isinstance(k, str)]

    def text_args(self, name: str | None) -> list[str]:
        out = []
        for key, spec in self.props(name).items():
            if isinstance(spec, dict) and 'string' in Schema.kinds(spec) and 'enum' not in spec:
                out.append(key)
        return out

    def one_target(self, name: str | None) -> str | None:
        """The single text argument of a one-listing tool, when the tool takes only that."""
        if not name:
            return None
        texts = self.text_args(name)
        musts = self.required(name)
        target = None
        for key in musts:
            if key in texts:
                target = key
                break
        if target is None and len(texts) == 1:
            target = texts[0]
        if target is None or any(k != target for k in musts):
            return None
        return target

    def many_target(self, name: str | None) -> str | None:
        target = None
        for key, spec in self.props(name).items():
            if isinstance(spec, dict) and 'array' in Schema.kinds(spec):
                target = key
                break
        if target is None or any(k != target for k in self.required(name)):
            return None
        return target

    def many_room(self, name: str | None) -> int:
        spec = self.props(name).get(self.many_target(name) or '') or {}
        return spec['maxItems'] if isinstance(spec.get('maxItems'), int) else 4

    def shape(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        faults: list[str] = []
        shaped = Schema(self.specs.get(name)).fit(dict(args), 'args', faults)
        return (shaped if isinstance(shaped, dict) else dict(args)), faults

    def lines(self, skip: set[str] | None = None) -> str:
        out = []
        for name, spec in self.specs.items():
            if skip and name in skip:
                continue
            out.append(f'- {name}: {self.blurbs.get(name, "")} | parameters: {compact(spec)}')
        return '\n'.join(out)

    def offered(self, keep: set[str] | None) -> list[dict[str, Any]]:
        """The tool list handed to the model: never the messaging tool; only `keep` when given."""
        ask = self.tool('ask')
        tools = [t for t in self.raw if t['function']['name'] != ask]
        if keep:
            narrowed = [t for t in tools if t['function']['name'] in keep]
            if narrowed:
                return narrowed
        return tools

    def labels(self) -> set[str]:
        """Every enum option in any schema, lowercased: the labels this store names itself."""
        found: set[str] = set()
        for spec in self.specs.values():
            for node, _, _ in nodes(spec, 6):
                if isinstance(node, dict):
                    found.update(str(o).strip().lower() for o in node.get('enum') or [] if isinstance(o, str))
        return found

    # -- messaging ---------------------------------------------------------------------------------------
    def message_args(self, text: str) -> dict[str, Any] | None:
        name = self.tool('ask')
        if not name:
            return None
        texts = self.text_args(name)
        target = next((k for k in self.required(name) if k in texts), texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in self.required(name):
            if key in args:
                continue
            options = (self.props(name).get(key) or {}).get('enum')
            if not isinstance(options, list) or not options:
                return None
            args[key] = options[0]
        shaped, faults = self.shape(name, args)
        return None if faults else shaped

    # -- browsing a category -----------------------------------------------------------------------------
    def set_browse(self, spec: Any) -> None:
        if not isinstance(spec, dict) or spec.get('tool') not in self.specs:
            return
        props = self.props(spec['tool'])
        category = spec.get('category')
        if category not in props:
            return
        price = spec.get('max_price') if spec.get('max_price') in props else None
        fixed = spec.get('fixed') if isinstance(spec.get('fixed'), dict) else {}
        self.browse = {'tool': spec['tool'], 'category': category, 'price': price,
                       'fixed': {k: v for k, v in fixed.items() if k in props and k not in (category, price)}}

    # -- the cart tool's shape ---------------------------------------------------------------------------
    def cart_parts(self, name: str) -> dict[str, Any]:
        props = self.props(name)
        texts = self.text_args(name)
        target = None
        for tool, key in self.id_slots:
            if tool == name and key in texts:
                target = key
        if target is None:
            target = next((k for k in self.required(name) if k in texts), texts[0] if len(texts) == 1 else None)
        count = mode = lines = None
        for key, spec in props.items():
            if not isinstance(spec, dict):
                continue
            kinds = Schema.kinds(spec)
            if count is None and 'integer' in kinds:
                count = key
            if mode is None and isinstance(spec.get('enum'), list) and len(spec['enum']) >= 2:
                mode = key
            inner = spec.get('items') if 'array' in kinds else None
            if isinstance(inner, dict) and isinstance(inner.get('properties'), dict):
                lines = (key, inner['properties'])
        return {'target': target, 'count': count, 'mode': mode, 'lines': lines, 'props': props}

    def cart_sets_count(self, name: str) -> bool:
        """Whether the cart tool sets a line's count outright: a count that may be 0 and no add/remove mode."""
        parts = self.cart_parts(name)
        if parts['mode'] or not parts['count']:
            return False
        spec = parts['props'].get(parts['count']) or {}
        return spec.get('minimum') == 0

    def remove_option(self, name: str) -> str | None:
        parts = self.cart_parts(name)
        if not parts['mode']:
            return None
        options = [str(o) for o in parts['props'][parts['mode']]['enum']]
        for option in options:
            if re.search(r'remov|delet|drop|clear|take|discard|minus|subtract|out', option, re.I):
                return option
        return options[1]

    def add_option(self, name: str) -> str | None:
        parts = self.cart_parts(name)
        if not parts['mode']:
            return None
        off = self.remove_option(name)
        options = [str(o) for o in parts['props'][parts['mode']]['enum']]
        return next((o for o in options if o != off), options[0])


# ------------------------------------------------------------------------------------------------------------
# the catalog: every listing seen, what the store said about each, and how well each fits
# ------------------------------------------------------------------------------------------------------------
class Catalog:
    """Listing records merged from search rows and opened views, with the keys that matter in this store."""

    def __init__(self, box: Toolbox):
        self.box = box
        self.rows: dict[str, dict[str, Any]] = {}       # merged record per listing id
        self.full: dict[str, dict[str, Any]] = {}       # ids whose attribute block was read
        self.pages: dict[str, str] = {}                 # prose of opened views
        self.order: list[str] = []
        self.id_key: str | None = None
        self.price_key: str | None = None
        self.stock_key: str | None = None
        self.seller_key: str | None = None
        self.shapes: dict[str, dict[str, Any] | None] = {}
        self.views: dict[str, dict[str, int]] = {}      # ident -> tool -> version when last viewed
        self.tool_keys: dict[str, set[str]] = {}
        self.unavailable: set[str] = set()
        self.gone: dict[str, int] = {}
        self.version = 0
        self.rules: list[tuple[str, str]] = []          # store rules pairing two values
        self.negations: dict[str, list[str]] = {}
        self._kind_cache: tuple[Any, str | None] = (None, None)
        self._facets: dict[str, tuple[Any, set[str]]] = {}

    # -- identity ----------------------------------------------------------------------------------------
    def ident(self, node: Any, listed: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_key:
            value = node.get(self.id_key)
            if IdSense.is_id(value, True):
                return value.strip()
        if not listed and depth > 0 and not any(isinstance(v, dict) for v in node.values()):
            return None
        key = IdSense.key_of([node])
        if key is None:
            return None
        value = str(node[key]).strip()
        return value if 'id' in key.lower() or value in self.rows else None

    def learn_rows(self, items: list[Any]) -> bool:
        rows = [n for n in items if isinstance(n, dict) and len(n) >= 3]
        if not rows or 2 * len(rows) < len(items):
            return False
        if self.id_key and all(IdSense.is_id(r.get(self.id_key), True) for r in rows):
            return True
        key = IdSense.key_of(rows)
        if key is None:
            return False
        self.id_key = key
        return True

    def absorb(self, observation: Any) -> list[str]:
        """Merge every listing record inside an observation; the ids seen, in order."""
        seen: list[str] = []
        stack: list[tuple[Any, int, bool]] = [(observation, 0, False)]
        while stack:
            node, depth, listed = stack.pop()
            if depth > 6:
                continue
            if isinstance(node, list):
                rows = self.learn_rows(node)
                stack.extend((inner, depth + 1, rows) for inner in reversed(node))
                continue
            if not isinstance(node, dict):
                continue
            ident = self.ident(node, listed, depth)
            if not ident:
                stack.extend((inner, depth + 1, False) for inner in reversed(list(node.values())))
                continue
            self._merge(ident, node)
            seen.append(ident)
        return seen

    def _merge(self, ident: str, node: dict[str, Any]) -> None:
        if ident not in self.rows:
            self.order.append(ident)
            self.rows[ident] = {}
        old = self.rows[ident]
        for key, value in node.items():
            if old.get(key) is True and value is False:
                self.unavailable.add(ident)
            elif old.get(key) is False and value is True:
                self.unavailable.discard(ident)
        old.update(node)
        if facts_of(node):
            self.full[ident] = dict(old)
        elif ident in self.full:
            self.full[ident].update(node)

    def records_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        """Listing records at the top of an observation: the observation itself, or the rows of its lists."""
        if isinstance(observation, dict):
            ident = self.ident(observation)
            if ident:
                return [(ident, observation)]
            found = []
            for value in observation.values():
                if isinstance(value, list):
                    found += self._rows(value)
                elif isinstance(value, dict):
                    for inner in value.values():
                        if isinstance(inner, list):
                            found += self._rows(inner)
            return found
        return self._rows(observation) if isinstance(observation, list) else []

    def _rows(self, items: list[Any]) -> list[tuple[str, dict[str, Any]]]:
        out = []
        for row in items:
            ident = self.ident(row, True) if isinstance(row, dict) else None
            if ident:
                out.append((ident, row))
        return out

    # -- views -------------------------------------------------------------------------------------------
    def note_view(self, tool: str, observation: Any) -> None:
        for ident, record in self.records_in(observation):
            self.tool_keys.setdefault(tool, set()).update(record.keys())
            self.views.setdefault(ident, {})[tool] = self.version
            text = prose_of(record)
            if text:
                self.pages[ident] = (self.pages.get(ident, '') + ' ' + text).strip()[-4000:]

    def viewed_now(self, ident: str, tool: str | None) -> bool:
        return bool(tool) and self.views.get(ident, {}).get(tool) == self.version

    def opened_with(self, ident: str, tool: str | None) -> bool:
        """Whether this view of the listing was opened at any point: what counts as reading its page."""
        return bool(tool) and tool in self.views.get(ident, {})

    def fresh_view(self, ident: str, tool: str | None) -> bool:
        """Whether this view has shown everything it gives for the listing since the last change."""
        return bool(self.tool_keys.get(tool or '')) and self.viewed_now(ident, tool) and ident not in self.gone

    # -- fields ------------------------------------------------------------------------------------------
    def price_of(self, ident: str) -> float | None:
        record = self.rows.get(ident) or {}
        value = record.get(self.price_key) if self.price_key else None
        if value is None:
            for key, inner in record.items():
                if isinstance(inner, (int, float)) and not isinstance(inner, bool) and re.search(r'price|cost|amount', Words.plain(key)):
                    value = inner
                    break
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str):
            value = amount(value) if re.fullmatch(r'\s*\d+(\.\d+)?\s*', value) else None
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def sold_out(self, ident: str) -> bool:
        return ident in self.unavailable or ident in self.gone

    def buyable(self, ident: str) -> bool:
        if self.sold_out(ident):
            return False
        record = self.rows.get(ident) or {}
        flag = record.get(self.stock_key) if self.stock_key else None
        if flag is None:
            for key, inner in record.items():
                if isinstance(inner, bool) and re.search(r'stock|avail|on hand|hand|buyable|sold', Words.plain(key)):
                    flag = inner
                    break
        if flag is False:
            return False
        return not (isinstance(flag, (int, float)) and not isinstance(flag, bool) and flag <= 0)

    def seller_of(self, ident: str) -> str:
        record = self.rows.get(ident) or {}
        named = record.get(self.seller_key) if self.seller_key else None
        if isinstance(named, str):
            return named.strip()
        for key, inner in record.items():
            if isinstance(inner, str) and 1 < len(inner.strip()) <= 40 and re.search(r'seller|shop|store|vendor|brand|maker|merchant|company|sold', Words.plain(key)):
                return inner.strip()
        return ''

    def kind_key(self) -> str | None:
        """The row key holding the category label: short strings, present on most rows, few distinct values,
        preferably ones the tool schemas list."""
        stamp = (len(self.rows), self.seller_key)
        if self._kind_cache[0] == stamp:
            return self._kind_cache[1]
        values: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.rows.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r'\d{3}', value):
                    values.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.rows))
        known = self.box.labels()
        best: tuple[tuple[float, int, int, float], str] | None = None
        for key, seen in values.items():
            if present[key] < 0.8 * total or len(seen) > 40 or key == self.seller_key:
                continue
            named = bool(re.search(r'categor|department|group|kind|shelf|section|aisle|family|class|type', Words.plain(key)))
            listed = sum(1 for v in seen if v.strip().lower() in known) / len(seen)
            mean = sum(len(v) for v in seen) / len(seen)
            score = (not named, -listed, len(seen), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        answer = best[1] if best else None
        self._kind_cache = (stamp, answer)
        return answer

    def label_of(self, ident: str) -> str:
        key = self.kind_key()
        return str((self.rows.get(ident) or {}).get(key) or '') if key else ''

    def title_of(self, ident: str) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        for key, value in record.items():
            if isinstance(value, str) and re.search(r'title|headline|name', Words.plain(key)) and len(value) > 3:
                return value
        return ''

    def record_text(self, ident: str, limit: int = 3000) -> str:
        record = self.full.get(ident) or self.rows.get(ident)
        if not record:
            return '(no record seen)'
        text = compact(shorten(record, 400))[:limit]
        page = self.pages.get(ident, '')
        if page and page[:200] not in text:
            text += ' | page text: ' + page[:900]
        return text

    def row_text(self, ident: str, limit: int = 900) -> str:
        record = self.full.get(ident) or self.rows.get(ident) or {}
        return compact(shorten(record, 160))[:limit]

    def rules_text(self) -> str:
        return '; '.join(f'{a} goes together with {b}' for a, b in self.rules[:12])

    # -- evidence ----------------------------------------------------------------------------------------
    @staticmethod
    def simple_value(value: str) -> bool:
        """Whether a constraint is a plain value that word matching can judge (no number, limit or exclusion)."""
        return bool(value) and not re.search(r'\d|\b(?:not|no|other than|unlike|at most|at least|under|over|below|above|less|more|between|within|up to|or|any|different|same|identical|partner)\b', value, re.I)

    @staticmethod
    def alternatives(value: str) -> list[frozenset[str]]:
        """The word sets a plain value may be stated by: "none (sleeveless)" or "rectangle or square" give several."""
        out: list[frozenset[str]] = []
        for part in re.split(r'[()/]|\bor\b', value.lower()):
            words = Words.content(part) or Words.stems(part)
            if words and words not in out:
                out.append(frozenset(words))
        return out

    @classmethod
    def states(cls, shown: Any, wanted: Any) -> bool:
        """Whether an attribute value states the wanted value: one of its elements carries every word of one of the
        wanted value's alternatives."""
        alts = wanted if isinstance(wanted, list) else [frozenset(wanted)]
        for element in re.split(r'[,;/|]| and ', str(shown).lower()):
            words = Words.stems(element)
            if any(alt and alt <= words for alt in alts):
                return True
        return False

    def vocabulary(self, prop: str) -> set[str]:
        """Values seen under keys naming this property across the records read."""
        found: set[str] = set()
        for record in self.full.values():
            for key, value in (facts_of(record) or {}).items():
                if Words.alike(prop, key):
                    for element in re.split(r'[,;/|]| and ', str(value).lower()):
                        element = element.strip()
                        if 2 < len(element) <= 30:
                            found.add(element)
        return found

    def facet_values(self, key: str) -> set[str]:
        """Every value element seen under keys spelt like `key` across the records read, stemmed."""
        name = Words.plain(key)
        stamp = (len(self.full), name)
        cached = self._facets.get(name)
        if cached and cached[0] == stamp:
            return cached[1]
        found: set[str] = set()
        for record in self.full.values():
            for k, v in (facts_of(record) or {}).items():
                if Words.plain(k) == name:
                    for element in re.split(r'[,;/|]| and ', str(v).lower()):
                        words = ' '.join(sorted(Words.stems(element)))
                        if words:
                            found.add(words)
        self._facets[name] = (stamp, found)
        return found

    def other_facet(self, a: str, b: str) -> bool:
        """Whether two differently spelt keys hold value sets that never meet: then they describe different facets."""
        if Words.plain(a) == Words.plain(b):
            return False
        va, vb = self.facet_values(a), self.facet_values(b)
        return len(va) >= 3 and len(vb) >= 3 and not (va & vb)

    def evidence(self, ident: str, prop: str, value: str) -> dict[str, Any]:
        """How a read listing stands on one plain-valued requirement.

        grade: 'strong' (keys naming the property state the value and agree; a title naming another value is
        only noted), 'weak' (no key; title or description states it), 'unknown'
        (nothing says), 'conflict' (keys naming the property disagree), 'fails' (every such key states another
        value, or the title states another value and no key states the wanted one)."""
        record = self.full.get(ident) or self.rows.get(ident) or {}
        block = facts_of(record) or {}
        wanted = self.alternatives(value)
        keys = {k: v for k, v in block.items() if Words.alike(prop, k)}
        # spellings of one key (clock shape / clock_shape) are one attribute: its values are read together
        groups: dict[str, list[str]] = {}
        for k in keys:
            groups.setdefault(Words.plain(k), []).append(k)
        agree = [k for name, ks in groups.items() if any(self.states(keys[k], wanted) for k in ks) for k in ks]
        listed: set[str] = set()
        for k in agree:
            for element in re.split(r'[,;/|]| and ', str(keys[k]).lower()):
                listed.add(' '.join(sorted(Words.stems(element))))
        # a sibling key naming one element of an agreeing key's list is the same list cut short, not a conflict
        differ = [k for k in keys if k not in agree and str(keys[k]).strip() and ' '.join(sorted(Words.stems(keys[k]))) not in listed]
        # a key whose values never overlap an agreeing key's values describes another facet (style: casual beside
        # skirt style: wrap), not a disagreement
        differ = [k for k in differ if not any(self.other_facet(k, a) for a in agree)]
        title = self.title_of(ident).lower()
        title_words = Words.stems(title)
        in_title = any(alt <= title_words for alt in wanted)
        others = [v for v in self.vocabulary(prop) if not self.states(v, wanted) and len(Words.content(v)) >= 1]
        clash = next((v for v in others if Words.stems(v) <= title_words and not any(Words.stems(v) <= alt for alt in wanted)), None)
        if agree and not differ:
            grade = 'strong'
        elif agree:
            grade = 'conflict'
        elif differ:
            grade = 'fails'
        elif in_title or any(alt <= Words.stems(self.pages.get(ident, '') + ' ' + prose_of(record)) for alt in wanted):
            grade = 'weak'
        else:
            grade = 'fails' if clash else 'unknown'
        return {'grade': grade, 'agree': {k: str(keys[k])[:40] for k in agree}, 'differ': {k: str(keys[k])[:40] for k in differ}, 'title_clash': clash or ''}


# ------------------------------------------------------------------------------------------------------------
# the clock, the models and the store, inside time and money
# ------------------------------------------------------------------------------------------------------------
class Clock:
    def __init__(self) -> None:
        self.started = time.monotonic()

    def left(self) -> float:
        return CLOCK - (time.monotonic() - self.started)

    def used(self) -> float:
        return time.monotonic() - self.started


class Brain:
    """Model calls, with fall-backs between models and a spending cap per episode."""

    def __init__(self, clock: Clock, trace: list[dict[str, Any]]):
        self.clock = clock
        self.trace = trace
        self.http = ProxyClient(timeout=60, max_retries=1)
        self.cost = 0.0
        self.calls = 0
        self.waited = 0.0
        self.broke = False                      # the inference key has no credit left
        self.cooling: dict[str, float] = {}     # model -> monotonic time until which it is rate limited

    def slow_ok(self, limit: float = SPEND_SOFT) -> bool:
        return self.cost <= limit and self.cost + SLOW_ROOM <= SPEND_HARD and self.clock.left() > WRAP_UP_AT

    def judges(self, limit: float = SPEND_SOFT) -> list[str]:
        if self.slow_ok(limit):
            return [SLOW, FAST]
        return [FAST] if self.cost <= SPEND_HARD else []

    def lineup(self, models: list[str]) -> list[str]:
        """The models to try in order: a slow model brings its kin; late in the episode only fast models run."""
        late = self.clock.left() < WRAP_UP_AT + 10
        out: list[str] = []
        for model in models:
            picks = [FAST] if (late and model == SLOW) else ([model, *SLOW_ALTS] if model == SLOW else [model])
            out.extend(p for p in picks if p not in out)
        return out

    def complete(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None, tag: str = 'loop') -> dict[str, Any] | None:
        """One model reply, or None. Rate-limited slow models hand over to their kin; other faults back off."""
        for model in self.lineup(models):
            if self.calls >= MODEL_CALLS or self.clock.left() < CLOCK_FLOOR or self.broke:
                return None
            if time.monotonic() < self.cooling.get(model, 0.0):
                continue
            self.calls += 1
            status: Any = None
            for attempt, wait in enumerate((0, *RETRY_WAITS)):
                if wait:
                    if self.waited + wait > RETRY_BUDGET or self.clock.left() < wait + CLOCK_FLOOR + 10:
                        break
                    self.waited += wait
                    time.sleep(wait * (0.8 + 0.4 * random.random()))
                self.http.timeout = max(8, min(55, int(self.clock.left() - CLOCK_FLOOR)))
                message, status = self._post(model, messages, max_tokens, tools, tag)
                if message is not None:
                    return message
                if status == 402:
                    self.broke = True
                    return None
                if status == 0:
                    # an empty or cut-off reply: another model would spend the same tokens on the same prompt
                    return None
                if status == 429 and model != FAST:
                    self.cooling[model] = time.monotonic() + COOL_OFF
                    self.calls -= 1
                    break
                passing = status is None or status in (408, 425, 429) or (isinstance(status, int) and status >= 500)
                if not passing or (status in (None, 408) and attempt >= 1):
                    break
        return None

    def _post(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None, tag: str) -> tuple[dict[str, Any] | None, Any]:
        sent = messages
        if model.startswith('anthropic/') and len(messages) > 1:
            sent = [dict(m) for m in messages]
            for index in sorted({0, len(sent) - 1}):
                content = sent[index].get('content')
                if isinstance(content, str) and content:
                    sent[index]['content'] = [{'type': 'text', 'text': content, 'cache_control': {'type': 'ephemeral'}}]
        body: dict[str, Any] = {'model': model, 'messages': sent, 'max_tokens': max_tokens, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        result = self.http.post_verbose('/inference/chat/completions', body)
        data = result.data if isinstance(result.data, dict) else None
        choices = data.get('choices') if data else None
        if choices:
            usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
            spent = usage.get('cost')
            if isinstance(spent, (int, float)) and not isinstance(spent, bool):
                self.cost += float(spent)
            details = usage.get('prompt_tokens_details') if isinstance(usage.get('prompt_tokens_details'), dict) else {}
            self.trace.append({'kind': 'llm', 'tag': tag, 'model': model, 'in': usage.get('prompt_tokens'), 'cached': details.get('cached_tokens'), 'out': usage.get('completion_tokens'), 'cost': round(float(spent), 5) if isinstance(spent, (int, float)) else None})
            message = (choices[0] or {}).get('message') if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and (message.get('content') or message.get('tool_calls')):
                return message, 200
            self.trace.append({'kind': 'llm_empty', 'model': model, 'finish': (choices[0] or {}).get('finish_reason'), 'max_tokens': max_tokens})
            return None, 0
        error = result.error or {}
        status = error.get('status')
        if data and isinstance(data.get('error'), dict):
            status = whole(data['error'].get('code')) or status
            error = {'body': data['error'].get('message')}
        self.trace.append({'kind': 'llm_fail', 'model': model, 'status': status, 'body': str(error.get('body') or '')[:160]})
        return None, status

    def json(self, prompt: str, models: list[str], max_tokens: int = 1800, tag: str = 'json') -> dict[str, Any] | None:
        for model in models:
            # fast models think before they answer and their output is cheap: give them room, keep slow models tight
            room = max_tokens if model in (SLOW, *SLOW_ALTS) else max(max_tokens, 3500)
            message = self.complete([{'role': 'user', 'content': prompt}], [model], room, tag=tag)
            found = first_json((message or {}).get('content'))
            if found is not None:
                return found
            if self.broke or self.calls >= MODEL_CALLS:
                break
        return None


class Store:
    """Environment steps, with recovery from step-number disagreements."""

    def __init__(self, binding: dict[str, Any], clock: Clock, trace: list[dict[str, Any]]):
        self.binding = dict(binding)
        self.clock = clock
        self.trace = trace
        self.http = ProxyClient(timeout=90, max_retries=3)
        self.step = 0
        self.over = False
        self.broken = False

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """One environment step; None when the call could not be made (the episode may be over)."""
        if self.over:
            return None
        self.http.timeout = max(10, min(90, int(self.clock.left() - 4)))
        self.http.max_retries = 3 if self.clock.left() > WRAP_UP_AT else 1
        for _ in range(4):
            payload = {**self.binding, 'step': self.step + 1, 'action': {'name': name, 'args': args}}
            result = self.http.post_verbose('/environment/call', payload)
            data = result.data
            if isinstance(data, dict):
                self.step = data['step'] if isinstance(data.get('step'), int) else self.step + 1
                if data.get('finished'):
                    self.over = True
                return data
            error = result.error or {}
            status = error.get('status')
            body = str(error.get('body') or '')
            self.trace.append({'kind': 'env_fail', 'status': status, 'body': body[:200]})
            if status == 409:
                low = body.lower()
                if any(word in low for word in ('finish', 'quarantin', 'finaliz', 'ended')):
                    self.over = True
                    return None
                expected = re.search(r'expected (\d+)', body)
                if expected:
                    self.step = int(expected.group(1)) - 1
                    continue
                if any(word in low for word in ('reuse', 'different action', 'already')):
                    self.step += 1
                    continue
                return None
            if status == 402:
                self.broken = True
            if status in (400, 401, 402, 403, 404, 422):
                return None
        return None


# ------------------------------------------------------------------------------------------------------------
# the episode: shared state, budgets and the ledger
# ------------------------------------------------------------------------------------------------------------
class Episode:
    """Everything one episode knows: the request as a ledger, the store, the models, the conversation."""

    def __init__(self, problem: dict[str, Any]):
        self.clock = Clock()
        environment = problem['environment']
        self.view: dict[str, Any] = environment['policy_view']
        self.request = str(self.view.get('query') or '')
        self.trace: list[dict[str, Any]] = []
        self.brain = Brain(self.clock, self.trace)
        self.store = Store(environment['binding'], self.clock, self.trace)
        self.store.over = bool(self.view.get('finished'))
        self.box = Toolbox(self.view.get('tools') or [])
        self.catalog = Catalog(self.box)
        steps = whole(self.view.get('max_steps'))
        self.max_steps = steps if steps and steps > 0 else STEPS_IF_UNKNOWN
        # the ledger
        self.items: list[Item] = []
        self.budget: float | None = None
        self.budget_unit = ''
        self.budget_each = False
        self.pick = 'any'
        self.measure = ''
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
        self.turns = 0
        self.record: list[str] = []          # what was done, for process-step gates
        self.record_steps: list[int] = []
        self.notes: list[str] = []
        self.queued: list[str] = []
        self.heard_lines: set[str] = set()
        self.qa: list[tuple[int, str, str]] = []
        self.interviewed = False
        self.searched: list[dict[str, Any]] = []
        self.read_cache: dict[str, tuple[int, int, str]] = {}
        self.recent: list[str] = []
        self.pages_read = 0
        self.paged_obs: list[Any] = []
        self.facts_from: dict[str, bool] = {}
        self.shape_tries: dict[str, int] = {}
        self.shopper_mode = 'unknown'        # 'normal' answers lists; 'tired' answers one property at a time
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
        # judgement
        self.checked: set[str] = set()
        self.leads: list[str] = []
        self.lead_items: dict[str, list[str]] = {}
        self.audited = 0
        self.audit_ok = 0
        self.swap_fits: list[str] = []
        self.refuse_blocks = 0
        self.pushed = False
        self.late_held = False
        self.line_holds = 0
        self.fit_holds = 0
        self.view_holds = 0
        self.proof_holds = 0
        self.doubt_opens = 0
        self.ladder_done: dict[int, bool] = {}
        self.ladder_note = ''
        self.last_match: tuple[int, int] | None = None
        self.gate_ok: set[tuple[str, str]] = set()
        self.gate_msgs = 0
        self.wider_held = False
        self.swept: set[str] = set()
        self.cheaper_marks: set[str] = set()
        self.chain_steps = 0
        self.chain_blocks = 0
        self.chain_misses: dict[str, int] = {}
        self.chain_pass = 0
        self.compare_ok = 0
        self.compare_bad = 0
        self.browse_misses = 0
        self.blocking: list[tuple[int, str]] = []
        self.plan_hinted: dict[int, set[str]] = {}
        self.better: list[str] = []
        self.swaps = 0

    # -- budgets -----------------------------------------------------------------------------------------
    def steps_left(self) -> int:
        return self.max_steps - self.store.step

    def time_left(self) -> float:
        return self.clock.left()

    def done(self) -> bool:
        return self.store.over

    def msgs_free(self) -> int:
        return self.msg_limit - self.msg_used - max(0, self.reserve - self.gate_msgs)

    def tail(self) -> int:
        """Steps kept for the ending: the checkout, a retry after a refusal, a message a process step needs."""
        talk = any(p['when'] == 'order' and p['messages'] > 0 for p in self.process)
        spare = 4 if self.buys == 0 else 0      # a refused first checkout needs remove + add + check out
        return spare + (2 if talk and self.msg_used < self.msg_limit and self.gate_due('order') else 1)

    def gate_due(self, stage: str) -> bool:
        return any(p['when'] == stage for p in self.process) and (stage, self.cart_mark()) not in self.gate_ok

    def cart_mark(self) -> str:
        return compact(sorted(self.cart_lines.items())) + f'#{self.catalog.version}'

    def room(self) -> int:
        """Steps free for exploring: what is left minus the ending and one cart call per item."""
        return self.steps_left() - self.tail() - len(self.items) - 1

    def needs(self) -> list[tuple[int, Rule]]:
        return [(n, rule) for n, item in enumerate(self.items, 1) for rule in item.active()]

    def item_at(self, number: Any) -> Item:
        index = whole(number) or 1
        return self.items[min(max(1, index), len(self.items)) - 1]

    def bump(self) -> None:
        """The task changed: records must be re-read and the cart re-checked."""
        self.catalog.version += 1
        self.recent = []
        self.checked = set()

    # -- reading the request -----------------------------------------------------------------------------
    def read_request(self) -> None:
        prompt = BRIEF + '\nTOOLS:\n' + self.box.lines() + '\n\nREQUEST:\n' + self.request
        if self.view.get('opening') is not None:
            prompt += '\n\nWHAT THE STORE SHOWS FIRST:\n' + compact(shorten(self.view.get('opening'), 1500))
        brief = self.brain.json(prompt, [FAST], 4000, tag='brief') or {}
        weak = not self._items_of(brief) or not isinstance(brief.get('roles'), dict) or not brief['roles'].get('buy')
        if weak and not self.brain.broke:
            again = self.brain.json(prompt, [SLOW, FAST], 2500, tag='brief') or {}
            if self._items_of(again) and isinstance(again.get('roles'), dict) and again['roles'].get('buy'):
                brief = again
        self.brief = brief
        self._take_roles(brief)
        self._take_items(brief)
        self._take_terms(brief)
        if not weak:
            self.review_brief()
        if self.box.tool('ask'):
            talk = [p['messages'] for p in self.process if p['when'] in ('order', 'end') and p['messages']]
            self.reserve = min(max(talk), self.msg_limit) if talk else 0

    @staticmethod
    def _items_of(brief: dict[str, Any]) -> list[dict[str, Any]]:
        found = brief.get('items')
        return [e for e in found if isinstance(e, dict)] if isinstance(found, list) else []

    def _take_roles(self, brief: dict[str, Any]) -> None:
        roles = brief.get('roles') if isinstance(brief.get('roles'), dict) else {}
        self.box.jobs = {k: v for k, v in roles.items() if isinstance(v, str) and v in self.box.specs}
        self.guess_roles()
        for key in ('buy_args', 'refuse_args'):
            if not isinstance(brief.get(key), dict):
                brief[key] = {}
        buy, refuse = self.box.tool('buy'), self.box.tool('refuse')
        if buy and buy == refuse and (not brief['buy_args'] or not brief['refuse_args'] or brief['buy_args'] == brief['refuse_args']):
            ask = (f'The store tool {buy} both checks out and ends without a purchase, depending on its arguments. Its schema: '
                   + compact(self.box.specs.get(buy)) + '\nReturn ONE JSON object and nothing else: {"buy_args": the fixed '
                   'arguments that check out, "refuse_args": the fixed arguments that end without a purchase} (only the arguments that choose the mode).')
            fixed = self.brain.json(ask, [FAST, SLOW], 1200, tag='brief') or {}
            if isinstance(fixed.get('buy_args'), dict) and isinstance(fixed.get('refuse_args'), dict) and fixed['buy_args'] != fixed['refuse_args']:
                brief['buy_args'], brief['refuse_args'] = fixed['buy_args'], fixed['refuse_args']
        self.box.buy_args = dict(brief['buy_args'])
        self.box.refuse_args = dict(brief['refuse_args'])
        self.box.set_browse(brief.get('browse'))

    def _take_items(self, brief: dict[str, Any]) -> None:
        lookups = {str(x).strip() for x in brief.get('lookups') or [] if str(x).strip()} if isinstance(brief.get('lookups'), list) else set()
        for entry in self._items_of(brief):
            quantity = whole(entry.get('quantity'))
            item = Item(str(entry.get('name') or 'the product'), str(entry.get('shelf') or ''), quantity if quantity and quantity > 0 else 1)
            for text in entry.get('needs') if isinstance(entry.get('needs'), list) else []:
                text = str(text or '').strip()
                if text:
                    item.require(text, 'request', points=text in lookups or self.points_elsewhere(text))
            self.items.append(item)
        self.merge_twins()
        if not self.items:
            self.items = [Item('the requested product', '', 1)]
        for pair in brief.get('fits') if isinstance(brief.get('fits'), list) else []:
            if isinstance(pair, dict):
                part, main = whole(pair.get('item')) or 0, whole(pair.get('with')) or 0
                if part != main and 1 <= part <= len(self.items) and 1 <= main <= len(self.items) and (part, main) not in self.bound:
                    self.bound.append((part, main))

    def _take_terms(self, brief: dict[str, Any]) -> None:
        budget = brief.get('budget')
        if isinstance(budget, str):
            budget = amount(budget)
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            self.budget = float(budget)
        self.budget_unit = str(brief.get('budget_unit') or '')
        self.budget_each = brief.get('budget_scope') == 'each'
        self.pick = brief.get('pick') if brief.get('pick') in ('cheapest', 'most', 'least') else 'any'
        self.measure = str(brief.get('pick_measure') or '')
        self.in_stock_only = brief.get('in_stock_only') is not False
        limit = whole(brief.get('messages'))
        self.msg_limit = limit if limit is not None and limit >= 0 else (MESSAGES_IF_UNKNOWN if self.box.tool('ask') else 0)
        steps = whole(brief.get('steps'))
        if steps is not None and 8 <= steps < self.max_steps:
            self.max_steps = steps
        strings = lambda key: [str(x) for x in brief.get(key) or [] if str(x).strip()] if isinstance(brief.get(key), list) else []  # noqa: E731
        self.rules = strings('rules')
        self.wishes = strings('wishes')
        self.ask_first = strings('ask_first')
        self.hints = [h for h in brief.get('hints') or [] if isinstance(h, dict) and str(h.get('text') or '').strip()] if isinstance(brief.get('hints'), list) else []
        self.quiet_decline = brief.get('quiet_decline') if isinstance(brief.get('quiet_decline'), bool) else None
        self.process = self._steps_of(brief.get('process'))

    @staticmethod
    def _steps_of(entries: Any) -> list[dict[str, Any]]:
        out = []
        for entry in entries if isinstance(entries, list) else []:
            if isinstance(entry, dict) and str(entry.get('step') or '').strip():
                when = str(entry.get('when') or '').strip().lower()
                out.append({'step': str(entry['step']).strip(), 'when': when if when in ('choose', 'order', 'end') else 'order', 'messages': max(0, whole(entry.get('messages')) or 0)})
        return out

    def review_brief(self) -> None:
        """A stronger reader corrects the brief: missed requirements (a use, an occasion), story read as needs,
        wrong items or quantities, the pick rule, the budget, the process steps."""
        if not self.brain.slow_ok():
            return
        lines = []
        for n, item in enumerate(self.items, 1):
            lines.append(f'item {n}: {item.name}, quantity {item.qty}; needs: ' + ('; '.join(r.text for r in item.active()) or 'none'))
        shown = {'items': lines, 'budget': self.budget, 'budget_scope': 'each' if self.budget_each else 'total', 'pick': self.pick, 'process': self.process, 'hints': self.hints}
        prompt = REVIEW + '\n\nREQUEST:\n' + self.request + '\n\nTHE READER\'S ERRAND:\n' + compact(shown)
        got = self.brain.json(prompt, [SLOW], 1500, tag='review') or {}
        changed: list[str] = []
        fixed = got.get('items')
        if isinstance(fixed, list) and fixed and all(isinstance(e, dict) and str(e.get('name') or '').strip() for e in fixed) and len(fixed) != len(self.items):
            rebuilt = []
            for entry in fixed[:6]:
                quantity = whole(entry.get('quantity'))
                item = Item(str(entry['name']).strip(), '', quantity if quantity and quantity > 0 else 1)
                twin = next((old for old in self.items if Words.content(old.name) & Words.content(item.name)), None)
                if twin:
                    item.shelf = twin.shelf
                    for rule in twin.active():
                        item.require(rule.text, rule.source, rule.points)
                rebuilt.append(item)
            self.items = rebuilt
            self.bound = []
            changed.append('items: ' + ', '.join(f'{i.name} x{i.qty}' for i in rebuilt))
        elif isinstance(fixed, list) and len(fixed) == len(self.items):
            for item, entry in zip(self.items, fixed, strict=False):
                quantity = whole(entry.get('quantity')) if isinstance(entry, dict) else None
                if quantity and quantity > 0 and quantity != item.qty:
                    item.qty = quantity
                    changed.append(f'{item.name}: quantity {quantity}')
        for entry in got.get('drop') if isinstance(got.get('drop'), list) else []:
            if isinstance(entry, dict) and str(entry.get('need') or '').strip():
                item = self.item_at(entry.get('item'))
                target = str(entry['need']).strip()
                for rule in item.active():
                    if rule.text == target and rule.source == 'request' and not rule.points:
                        rule.dropped = True
                        changed.append(f'dropped {target}')
        for entry in got.get('add') if isinstance(got.get('add'), list) else []:
            if isinstance(entry, dict) and ':' in str(entry.get('need') or ''):
                item = self.item_at(entry.get('item'))
                text = str(entry['need']).strip()
                if not any(Words.alike(rule.prop, text.partition(':')[0]) for rule in item.active()) and item.require(text, 'request', self.points_elsewhere(text)):
                    changed.append(f'added {text}')
        if got.get('pick') in ('cheapest', 'most', 'least', 'any') and got['pick'] != self.pick:
            self.pick = got['pick']
            changed.append(f'pick {self.pick}')
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != self.budget:
            self.budget = float(budget)
            changed.append(f'budget {budget:g}')
        steps = self._steps_of(got.get('process')) if isinstance(got.get('process'), list) else None
        if steps is not None and steps != self.process:
            self.process = steps
            changed.append('process steps: ' + '; '.join(p['step'] for p in steps) if steps else 'no process steps')
        for hint in got.get('hints') if isinstance(got.get('hints'), list) else []:
            if isinstance(hint, dict) and str(hint.get('text') or '').strip() and len(self.hints) < 8:
                self.hints.append(hint)
        if changed:
            self.trace.append({'kind': 'review', 'changed': changed})

    @staticmethod
    def points_elsewhere(text: str) -> bool:
        return bool(re.search(POINTER_RX, text, re.I))

    def guess_roles(self) -> None:
        """Fill the jobs the brief left out from the schemas: endings first, then messaging and cart tools."""
        jobs = self.box.jobs
        texts = {name: (name + ' ' + self.box.blurbs.get(name, '')).lower() for name in self.box.specs}

        def free(name: str) -> bool:
            return name not in jobs.values()

        def first(pattern: str, test) -> str | None:  # noqa: ANN001
            for name, text in texts.items():
                if free(name) and re.search(pattern, text) and test(name):
                    return name
            return None
        if 'refuse' not in jobs:
            hit = first(r'give up|decline|no purchase|without (?:a )?purchase|cancel|abandon|skip|close out|pass', lambda n: True)
            if hit:
                jobs['refuse'] = hit
        if 'buy' not in jobs:
            hit = first(r'check ?out|purchase|place (?:the )?order|buy|settle|commit|wrap', lambda n: not self.box.text_args(n))
            if hit:
                jobs['buy'] = hit
        if 'ask' not in jobs:
            hit = first(r'shopper|customer|buyer|message to|ask the', lambda n: bool(self.box.text_args(n)))
            if hit:
                jobs['ask'] = hit
        if 'cart_view' not in jobs:
            hit = first(r'cart|basket|bag', lambda n: not self.box.props(n))
            if hit:
                jobs['cart_view'] = hit
        if 'cart' not in jobs:
            hit = first(r'cart|basket|bag', lambda n: bool(self.box.text_args(n)))
            if hit:
                jobs['cart'] = hit
        if 'search' not in jobs:
            hit = first(r'search|find|look ?up|locate|seek|browse|list', lambda n: bool(self.box.text_args(n)) and not self.box.one_target(n))
            if hit:
                jobs['search'] = hit
        if 'open' not in jobs:
            singles = [n for n in self.box.specs if self.box.one_target(n) and free(n)]
            if singles:
                jobs['open'] = singles[0]
                if len(singles) > 1 and 'offer' not in jobs:
                    jobs['offer'] = singles[1]

    def merge_twins(self) -> None:
        merged: list[Item] = []
        for item in self.items:
            twin = None
            for other in merged:
                if not item.shelf.strip() or other.qty != item.qty or Words.plain(other.shelf) != Words.plain(item.shelf):
                    continue
                if any(Words.alike(a.prop, b.prop) and Words.plain(a.value) != Words.plain(b.value) for a in other.active() for b in item.active()):
                    continue
                twin = other
                break
            if twin is None:
                merged.append(item)
                continue
            for rule in item.rules:
                twin.require(rule.text, rule.source, rule.points)
            self.trace.append({'kind': 'items_merged', 'into': twin.name, 'dropped': item.name})
        self.items = merged

    # -- the ledger as text ------------------------------------------------------------------------------
    def ledger(self) -> str:
        parts = []
        many = len(self.items) > 1
        for number, item in enumerate(self.items, 1):
            head = (f'item {number}: {item.name}' if many else f'item: {item.name}') + (f' (store label: {item.shelf})' if item.shelf else '') + f', quantity {item.qty}'
            shown = [r.shown() for r in item.active()]
            text = head + '; requirements: ' + ('; '.join(shown) if shown else 'none beyond the kind of product')
            if item.free:
                text += '; shopper does not mind: ' + ', '.join(item.free)
            parts.append(text)
        if self.budget is not None:
            parts.append(f"budget: at most {self.budget:g} {self.budget_unit} {'for each item' if self.budget_each else 'in total'}".replace('  ', ' '))
        if self.pick == 'cheapest':
            parts.append('pick: the cheapest listing that fits (lowest price)')
        elif self.pick in ('most', 'least'):
            parts.append(f'pick: the {self.pick} {self.measure} among fitting listings'.replace('  ', ' '))
        else:
            parts.append('pick: any listing that fits everything')
        if self.in_stock_only:
            parts.append('in stock only')
        parts.extend(f'wish (tie-breaker only): {w}' for w in self.wishes)
        parts.extend(f'rule from the request: {r}' for r in self.rules)
        if self.quiet_decline is True:
            parts.append('if nothing fits: end without naming or suggesting any listing')
        stages = {'choose': 'before choosing', 'order': 'at the order (Best does it at checkout)', 'end': 'at an ending without a purchase (Best does it then)'}
        parts.extend(f"step the request asks for, {stages[p['when']]}: {p['step']}" for p in self.process)
        if self.catalog.rules:
            parts.append('store rules seen (paired values are interchangeable): ' + self.catalog.rules_text())
        facts = self.facts_tool()
        if facts and (self.facts_from.get(facts) is True or self.box.tool('offer')):
            bare = sorted(t for t in self.page_tools() | {self.box.tool('offer') or ''} if t and self.facts_from.get(t) is False)
            parts.append(f'listing attributes come from {facts}' + (f"; {', '.join(bare)} show the offer page (price, stock, notes) only" if bare else ''))
        return '\n'.join('- ' + p for p in parts)

    # -- tools by job ------------------------------------------------------------------------------------
    def page_tools(self) -> set[str]:
        page = self.box.tool('offer') or self.box.tool('open')
        return {t for t in (page, self.box.tool('compare')) if t}

    def facts_tool(self) -> str | None:
        for tool, ok in self.facts_from.items():
            if ok and self.box.one_target(tool):
                return tool
        tool = self.box.tool('open')
        return tool if self.facts_from.get(tool or '') is not False and self.box.one_target(tool) else None

    def offer_tool(self) -> str | None:
        return self.box.tool('offer') or self.box.tool('open')

    def view_tools(self) -> list[str]:
        """Every single-listing view tool in this store, attribute view first."""
        out = []
        taken = self.box.committers() | {self.box.tool('ask'), self.box.tool('search')}
        for name in (self.facts_tool(), self.box.tool('open'), self.box.tool('offer')):
            if name and name not in out and name not in taken and self.box.one_target(name):
                out.append(name)
        return out


# ------------------------------------------------------------------------------------------------------------
# observing: records, notes, result shapes, paging, evidence
# ------------------------------------------------------------------------------------------------------------
class Observer:
    """Reads what tool results carry: listing records, store notes, result shapes, page tokens."""

    def __init__(self, ep: Episode):
        self.ep = ep

    def show(self, observation: Any) -> str:
        text = compact(shorten(observation, ROW_CHARS))
        return text if len(text) <= OBS_CHARS else text[:OBS_CHARS] + '...(cut)'

    def observe(self, tool: str, observation: Any) -> str:
        """Absorb an observation: records, views, notes. Returns the meaning of any note read."""
        ep = self.ep
        ep.catalog.absorb(observation)
        single = ep.box.one_target(tool)
        if single or tool in ep.box.readers() - {ep.box.tool('search')}:
            ep.catalog.note_view(tool, observation)
            listing = ep.catalog.ident(observation) if single and isinstance(observation, dict) else None
            if listing:
                ep.facts_from[tool] = ep.facts_from.get(tool, False) or bool(facts_of(observation))
                if tool == ep.offer_tool():
                    ep.offer_seen[listing] = ep.store.step
                self.harvest_rules(observation)
        lines = self.loose_lines(observation)
        if not lines:
            return ''
        meaning = self.take_note(lines)
        if meaning:
            ep.queued.append(f'(seen on {tool}) {meaning}')
        return meaning

    # -- store rules on pages ----------------------------------------------------------------------------
    def harvest_rules(self, observation: Any) -> None:
        """Pairing rules in a page's prose: "a goes together with b", "pairs a with b, c with d", "a is the
        partner of b". Sentence by sentence, so a rule never spans two statements."""
        value = r"[a-z0-9][a-z0-9 /+'.-]{0,28}?"
        for sentence in re.split(r'(?<=[.;!?])\s+', prose_of(observation)):
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
        pair = [self.clean_value(a), self.clean_value(b)]
        if not all(pair) or pair[0] == pair[1] or any(len(v) > 40 or v in ('it', 'they', 'which', 'one') for v in pair):
            return
        rules = self.ep.catalog.rules
        if (pair[0], pair[1]) in rules or (pair[1], pair[0]) in rules:
            return
        rules.append((pair[0], pair[1]))
        self.ep.trace.append({'kind': 'rule', 'pair': pair})

    @staticmethod
    def clean_value(text: str) -> str:
        value = text.strip().lower().strip('.,;:')
        value = re.sub(r"^(?:its|the|their|that|this|a|an|our|and|also|store rule for [a-z ]+:)\s+", '', value).strip()
        words = value.split()
        if len(words) > 1 and re.fullmatch(r'(?:pattern|style|material|colou?r|type|value|feature|features|closure|upholstery|finish|size|card|class)', words[-1]):
            value = ' '.join(words[:-1])
        return value

    # -- notes from the store or the shopper -------------------------------------------------------------
    def loose_lines(self, observation: Any) -> list[str]:
        """Strings of five or more words that sit outside listing records: notes, refusals, messages."""
        catalog = self.ep.catalog
        lines: list[str] = []
        if isinstance(observation, dict) and not catalog.ident(observation):
            for value in observation.values():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not catalog.ident(value, False, 1):
                    lines.extend(v for v in value.values() if isinstance(v, str) and len(v.split()) >= 6)
        return [line for line in lines if line not in self.ep.heard_lines]

    def take_note(self, lines: list[str]) -> str:
        ep = self.ep
        ep.heard_lines.update(lines)
        prompt = NOTE + '\n\nITEMS AND REQUIREMENTS NOW:\n' + ep.ledger() + '\n\nTHE TEXT:\n' + '\n'.join(lines)
        got = ep.brain.json(prompt, [FAST, FAST_ALT], 2000, tag='note') or {}
        gone = [str(g) for g in got.get('gone') or [] if isinstance(g, (str, int))] if isinstance(got.get('gone'), list) else []
        ep.catalog.unavailable.update(g for g in gone if g in ep.catalog.rows)
        ep.note_sold = bool(gone) or got.get('sold_out') is True
        ep.note_named = any(g in ep.catalog.rows for g in gone)
        if ep.note_sold and len(lines) == 1:
            ep.sold_lines.update(lines)
        if got.get('kind') not in (None, 'change'):
            return ''
        changed = []
        budget = got.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0 and budget != ep.budget:
            ep.budget = float(budget)
            changed.append(f'budget now {budget:g}')
        for entry in got.get('drop') if isinstance(got.get('drop'), list) else []:
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            item = ep.item_at(entry.get('item'))
            target = str(entry['need']).strip()
            hits = [r for r in item.active() if r.text == target] or [r for r in item.active() if Words.alike(r.prop, target.partition(':')[0])][:1]
            for rule in hits:
                rule.dropped = True
            changed.append(f'dropped {target}')
        for entry in got.get('add') if isinstance(got.get('add'), list) else []:
            if isinstance(entry, dict) and str(entry.get('need') or '').strip():
                item = ep.item_at(entry.get('item'))
                text = str(entry['need']).strip()
                if item.require(text, 'store', points=ep.points_elsewhere(text)):
                    changed.append(f'added {text}')
        if gone:
            changed.append('unavailable: ' + ', '.join(gone))
        if got.get('price_moved') is True:
            changed.append('a price changed')
        summary = str(got.get('summary') or '').strip()
        if ep.pick == 'cheapest' and (changed or ep.note_sold):
            ep.wider_held = False
            ep.cheaper_marks = set()
        if changed or summary:
            ep.bump()
            ep.notes.append(summary or '; '.join(changed))
        return (summary + (' (' + '; '.join(changed) + ')' if changed else '')).strip()

    # -- result shapes and paging ------------------------------------------------------------------------
    def read_shape(self, tool: str, observation: Any) -> dict[str, Any] | None:
        ep = self.ep
        catalog = ep.catalog
        if catalog.shapes.get(tool) is not None or ep.shape_tries.get(tool, 0) >= 2 or not isinstance(observation, dict):
            return catalog.shapes.get(tool)
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        ep.shape_tries[tool] = ep.shape_tries.get(tool, 0) + 1
        catalog.shapes[tool] = None
        sample = {k: (v[:3] if k in lists else v) for k, v in observation.items()}
        prompt = SHAPE + '\n\nTOOL: ' + tool + ' | parameters: ' + compact(ep.box.specs.get(tool) or {}) + '\n\nRESULT (first rows only):\n' + compact(shorten(sample, 120))[:4000]
        got = ep.brain.json(prompt, [FAST, SLOW], 1500, tag='shape') or {}
        rows = got.get('rows') if got.get('rows') in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = ep.box.props(tool)
        pick = lambda key, where: key if isinstance(key, str) and key in where else None  # noqa: E731
        shape = {'rows': rows, 'count': pick(got.get('count'), observation), 'next': pick(got.get('next'), observation), 'next_arg': pick(got.get('next_arg'), props),
                 'price': pick(got.get('price'), first), 'stock': pick(got.get('stock'), first), 'seller': pick(got.get('seller'), first)}
        if shape['count'] and (isinstance(observation[shape['count']], bool) or not isinstance(observation[shape['count']], int)):
            shape['count'] = None
        catalog.shapes[tool] = shape
        catalog.price_key = catalog.price_key or shape['price']
        catalog.stock_key = catalog.stock_key or shape['stock']
        catalog.seller_key = catalog.seller_key or shape['seller']
        ep.trace.append({'kind': 'shape', 'tool': tool, 'shape': shape})
        return shape

    def paging(self, tool: str, observation: Any) -> tuple[int, int, str, Any] | None:
        """(matched, shown, next-page argument, next-page value) when a result has more pages."""
        shape = self.read_shape(tool, observation)
        if not shape or not isinstance(observation, dict) or not (shape['count'] and shape['next'] and shape['next_arg']):
            return None
        rows, count, token = observation.get(shape['rows']), observation.get(shape['count']), observation.get(shape['next'])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if token is not None and (isinstance(token, bool) or not isinstance(token, (str, int))):
            return None
        return count, len(rows), shape['next_arg'], token

    def price_sorted(self, observation: Any) -> bool:
        prices = [self.ep.catalog.price_of(i) for i, _ in self.ep.catalog.records_in(observation)]
        return len(prices) >= 3 and None not in prices and prices == sorted(prices)

    def more_pages(self, tool: str, args: dict[str, Any], observation: Any) -> str:
        """For a cheapest/most/least pick, read the following pages of a short result by code."""
        ep = self.ep
        ep.paged_obs = []
        plan = self.paging(tool, observation) if ep.pick in ('cheapest', 'most', 'least') else None
        if not plan:
            return ''
        count, shown, param, token = plan
        given = args.get(param)
        later = (given not in (None, 0, 1)) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or '').strip())
        if token is None or later or count <= shown or (ep.pick == 'cheapest' and self.price_sorted(observation)):
            return ''
        pages = -(-(count - shown) // shown)
        if pages > 2 and ep.pages_read + 2 <= OWN_PAGES and ep.room() >= 10:
            pages = 2
        if pages > 2 or ep.pages_read + pages > OWN_PAGES or ep.room() - pages < 8:
            return f"\n[Best] This search matched {count} listings and showed {shown}, in no price order. The pick counts every fitting listing: narrow the search with the requirement's value (or a lower price cap) or page on with {param}={compact(token)}."
        texts = []
        for _ in range(pages):
            call, faults = ep.box.shape(tool, {**args, param: token})
            if faults or ep.done():
                break
            reply = ep.store.call(tool, call)
            if reply is None or ep.done():
                break
            ep.pages_read += 1
            seen = reply.get('observation')
            self.observe(tool, seen)
            ep.paged_obs.append(seen)
            shown_text = self.show(seen)
            ep.read_cache[tool + compact(call)] = (ep.catalog.version, ep.store.step, shown_text)
            texts.append(f'{param}={compact(token)}: {shown_text}')
            after = self.paging(tool, seen)
            token = after[3] if after else None
            if token is None:
                break
        ep.trace.append({'kind': 'pages', 'tool': tool, 'read': len(texts)})
        if not texts:
            return ''
        rest = f"\n[Best] {count} listings matched and {shown * (len(texts) + 1)} rows were read; narrow the search or page on to see the rest." if token is not None else ''
        return '\n[Best read the next pages of this search itself, one step a page]\n' + '\n'.join(texts) + rest

    def cap_ignored(self, tool: str, args: dict[str, Any], observation: Any) -> str:
        ep = self.ep
        props = ep.box.props(tool)

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search(r'price|cost|amount|budget|spend|money', key + ' ' + str(spec.get('description') or ''), re.I))
        caps = sorted((k for k, v in args.items() if k in props and isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r'max|most|under|below|limit|cap|upto|up_to|budget|spend', k, re.I) and priced(k)), key=lambda k: args[k])
        if not caps or not ep.catalog.price_key:
            return ''
        cap = args[caps[0]]
        over = [i for i, _ in ep.catalog.records_in(observation) if (ep.catalog.price_of(i) or 0) > cap]
        if not over:
            return ''
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get('enum'), list)}
        others = {k: [o for o in props[k]['enum'] if o != v] for k, v in modes.items()}
        hint = '; list the category with ' + ', '.join(f'{k}={compact(o[0])}' for k, o in others.items() if o) + f' and the same {caps[0]} to see only cheaper listings' if any(others.values()) else ''
        return f'\n[Best] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}.'

    def cheapest_here(self, observations: list[Any]) -> str:
        ep = self.ep
        if ep.pick != 'cheapest' or not ep.catalog.price_key:
            return ''
        found = {i for seen in observations for i, _ in ep.catalog.records_in(seen)}
        rows = sorted((ep.catalog.price_of(i), i) for i in found if ep.catalog.buyable(i) and ep.catalog.price_of(i) is not None)
        if len(rows) < 2:
            return ''
        return '\n[Best] buyable rows here, cheapest first: ' + ', '.join(f'{i} ({p:g})' for p, i in rows[:6])

    # -- evidence on opened listings ---------------------------------------------------------------------
    def opened_in(self, observation: Any) -> list[str]:
        ids: list[str] = []
        for node, depth, _ in nodes(observation, 6):
            if isinstance(node, dict):
                ident = self.ep.catalog.ident(node, False, depth)
                if ident and facts_of(node) and ident not in ids:
                    ids.append(ident)
        return ids

    def evidence_lines(self, ids: list[str]) -> str:
        """How each opened listing stands on every plain-valued requirement of its kind, graded by code."""
        ep = self.ep
        if not ids:
            return ''
        many = len(ep.items) > 1
        lines = []
        for ident in ids[:12]:
            owner = ep.kinds.owner_of(ep.catalog.label_of(ident)) or (1 if len(ep.items) == 1 else 0)
            parts = []
            for number, rule in ep.needs()[:10]:
                if owner and number != owner:
                    continue
                value = rule.resolved or rule.value
                if not Catalog.simple_value(value):
                    continue
                proof = ep.catalog.evidence(ident, rule.prop, value)
                detail = ', '.join(f'{k}={v}' for k, v in {**proof['agree'], **proof['differ']}.items()) or ('title' if proof['grade'] == 'weak' else '')
                if proof['title_clash'] and proof['grade'] != 'strong':
                    detail += f"; title says {proof['title_clash']}"
                parts.append(('(item %d) ' % number if many else '') + f"{rule.prop} -> {proof['grade']}" + (f' ({detail})' if detail else ''))
            if parts:
                lines.append(f'{ident}: ' + '; '.join(parts))
        if not lines:
            return ''
        return '[evidence by code, authoritative on key agreement: strong = the keys naming the property state the value (all spellings read as one attribute; a list holding the value counts); weak = title or text only; conflict/fails = doubtful, prefer a solid candidate when one exists] ' + ' | '.join(lines)


# ------------------------------------------------------------------------------------------------------------
# kinds: which listings belong to which requested item
# ------------------------------------------------------------------------------------------------------------
class Kinds:
    """Matches store category labels to the requested items, exactly when the store has an exact label."""

    def __init__(self, ep: Episode):
        self.ep = ep
        self._cache: dict[str, int] = {}
        self._cache_size = -1

    def labels_seen(self) -> set[str]:
        key = self.ep.catalog.kind_key()
        return {str(r.get(key)).strip() for r in self.ep.catalog.rows.values() if key and isinstance(r.get(key), str) and r.get(key).strip()}

    def score(self, item: Item, label: str) -> float:
        words = Words.plain(label)
        if item.shelf and words == Words.plain(item.shelf):
            return 3.0
        if words == Words.plain(item.name):
            return 2.5
        theirs = Words.content(label)
        mine = Words.content(item.name) | Words.content(item.shelf)
        if theirs and theirs == Words.content(item.name):
            return 2.0
        return len(theirs & mine) / max(1, len(theirs | mine))

    def owner_of(self, label: str) -> int:
        """The item a category label belongs to (1-based), or 0. A label that only partly matches an item does
        not belong to it once the store has shown a label matching that item exactly."""
        if not label:
            return 0
        ep = self.ep
        if self._cache_size != len(ep.catalog.rows):
            self._cache, self._cache_size = {}, len(ep.catalog.rows)
            self.confirm_labels()
        if label in self._cache:
            return self._cache[label]
        best, top = 0, 0.0
        for number, item in enumerate(ep.items, 1):
            value = self.score(item, label)
            if value > top:
                best, top = number, value
        if best and top < 2.0 and ep.items[best - 1].label_sure:
            best = 0
        self._cache[label] = best
        return best

    def owns(self, item: Item, label: str) -> bool:
        return bool(label) and item in self.ep.items and self.owner_of(label) == self.ep.items.index(item) + 1

    def confirm_labels(self) -> None:
        """Pin each item to the store's own label when a seen label matches it exactly."""
        for item in self.ep.items:
            exact = [label for label in self.labels_seen() if self.score(item, label) >= 2.0]
            if exact and not item.label_sure:
                item.shelf = sorted(exact, key=lambda l: -self.score(item, l))[0]
                item.label_sure = True
                self.ep.trace.append({'kind': 'label', 'item': item.name, 'label': item.shelf})

    def item_ids(self, number: int) -> list[str]:
        catalog = self.ep.catalog
        key = catalog.kind_key()
        if not key:
            return []
        item = self.ep.items[number - 1]
        return [i for i in catalog.order if self.owns(item, str((catalog.rows.get(i) or {}).get(key) or ''))]

    def item_groups(self) -> list[list[str]]:
        catalog = self.ep.catalog
        key = catalog.kind_key()
        groups: dict[str, list[str]] = {}
        for ident in catalog.order:
            groups.setdefault(str(catalog.rows[ident].get(key) or '') if key else '', []).append(ident)
        matched = [ids for label, ids in groups.items() if self.owner_of(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=len, reverse=True)[:max(1, len(self.ep.items))]

    def records_of_item(self, number: int) -> int:
        catalog = self.ep.catalog
        return sum(1 for i in self.item_ids(number) if i in catalog.full and facts_of(catalog.full[i]))

    def shelf_of(self, number: int) -> str:
        catalog = self.ep.catalog
        item = self.ep.items[number - 1]
        if item.label_sure:
            return item.shelf
        key = catalog.kind_key()
        labels = [str(r.get(key) or '') for r in catalog.rows.values()] if key else []
        labels = [label for label in labels if label and self.owns(item, label)]
        return max(set(labels), key=labels.count) if labels else item.shelf

    def key_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Attribute keys seen on opened listings, grouped by category label: (label, key, carried-by, samples)."""
        catalog = self.ep.catalog
        key = catalog.kind_key()
        groups: dict[str, dict[str, Any]] = {}
        for ident in catalog.order:
            record = catalog.full.get(ident)
            if not record:
                continue
            label = str(record.get(key) or 'listings') if key else 'listings'
            group = groups.setdefault(label, {'n': 0, 'keys': {}})
            group['n'] += 1
            seen: set[str] = set()
            for name, value in (facts_of(record) or {}).items():
                words = Words.plain(name)
                if not words or words in seen:
                    continue
                seen.add(words)
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
                if sum(len(v) for v in values) / len(values) > 40 or 2 * sum(1 for v in values if ' / ' in v) >= len(values):
                    continue
                twin = next((spec for other, spec in kept if Words.alike(other, name) and (set(spec['values']) & set(values) or len(name.split()) > 1)), None)
                if twin is not None:
                    twin['n'] = max(twin['n'], slot['n'])
                    twin['values'].extend(v for v in values if v not in twin['values'])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]['n'])
            table.extend((label, name, slot['n'], [v[:40] for v in slot['values'][:4]]) for name, slot in kept[:16])
        return table

    def lane_keys(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        item = self.ep.items[number - 1]
        settled = item.settled()
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner_of(label) == number]
        if len(self.ep.items) == 1 and not rows:
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        measured = {key: 2 * sum(1 for v in values if re.match(r'^\s*[\d.,/x -]+\s*[a-z]{0,6}\s*$', v)) > len(values) for _, key, _, values in table}
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any(Words.alike(s, key) for s in settled) and not re.search(SKIP_KEYS, key)]
        return [k for k in keys if not re.search(MIXED_KEYS, k)] + [k for k in keys if re.search(MIXED_KEYS, k)]


# ------------------------------------------------------------------------------------------------------------
# the interview: one plain question about one property of one item per message
# ------------------------------------------------------------------------------------------------------------
class Interviewer:
    def __init__(self, ep: Episode):
        self.ep = ep

    def ready(self) -> bool:
        ep = self.ep
        if ep.interviewed or not ep.box.tool('ask') or ep.msgs_free() <= 0:
            return False
        if ep.steps_left() <= ep.tail() + len(ep.items) + 2 or ep.time_left() < WRAP_UP_AT + 35:
            return False
        if all(ep.kinds.records_of_item(n) >= 1 for n in range(1, len(ep.items) + 1)) and len(ep.catalog.full) >= 2:
            return True
        return ep.store.step >= max(6, ep.max_steps // 4)

    def plan(self, count: int) -> dict[int, list[str]]:
        ep = self.ep
        table = ep.kinds.key_table()
        settled = []
        for n, item in enumerate(ep.items, 1):
            text = f'item {n}: {item.name} - settled by the request: ' + ('; '.join(r.text for r in item.active()) or 'nothing')
            if item.free:
                text += f"; shopper does not mind: {', '.join(item.free)}"
            if item.asked:
                text += f"; already asked: {', '.join(item.asked)}"
            settled.append(text)
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f'category {label!r} (item {ep.kinds.owner_of(label) or "?"}):')
                last = label
            listing.append(f'  {index}. {name} (on {carried}): ' + ' | '.join(values) + (' [several features in one: ask last]' if re.search(MIXED_KEYS, name) else ''))
        prompt = PLAN.replace('COUNT', str(max(1, count))) + '\n\nREQUEST:\n' + ep.request + '\n\nITEMS:\n' + '\n'.join(settled)
        if ep.ask_first:
            prompt += '\n\nTHE REQUEST SAYS TO ASK ABOUT:\n' + '\n'.join(ep.ask_first)
        if ep.hints:
            prompt += '\n\nSTORY HINTS:\n' + '\n'.join(f"- item {h.get('item')}: \"{h.get('text')}\" -> {h.get('property')}" for h in ep.hints[:8])
        prompt += '\n\nATTRIBUTE KEYS SEEN:\n' + ('\n'.join(listing) or '(no full listing records yet)')
        got = ep.brain.json(prompt, [SLOW, FAST], 2500, tag='plan') or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(ep.items) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in got.get('ask') if isinstance(got.get('ask'), list) else []:
            if not isinstance(pick, dict):
                continue
            number = min(max(1, whole(pick.get('item')) or 1), len(ep.items))
            label = re.split(r'\s+(?:and|or)\s+|[/,;(]', str(pick.get('label') or ''))[0].strip().rstrip('?.')
            if not label or len(label.split()) > 5:
                continue
            item = ep.items[number - 1]
            if any(Words.alike(label, s) for s in item.settled()):
                continue
            if all(Words.plain(label) != Words.plain(other) for other in lanes[number]):
                lanes[number].append(label)
            if pick.get('hinted') is True:
                hinted[number].add(Words.plain(label))
        for number in lanes:
            keys = ep.kinds.lane_keys(number, table)
            known = [key for label, key, _, _ in table if ep.kinds.owner_of(label) == number or len(ep.items) == 1]
            if len(known) >= 3:
                listed = [x for x in lanes[number] if any(Words.alike(x, k) for k in known)]
                made_up = [x for x in lanes[number] if x not in listed]
                lanes[number] = listed[:1] + [x for x in made_up if Words.plain(x) in hinted[number]] + listed[1:] + [x for x in made_up if Words.plain(x) not in hinted[number]]
            # the judge's picks and the keys most listings carry take turns: a hint can mislead, coverage rarely does
            picks = list(lanes[number])
            fresh = [k for k in keys if all(not Words.alike(k, other) for other in picks)]
            merged: list[str] = picks[:1]
            for pick, key in zip(picks[1:] + [None] * len(fresh), fresh + [None] * len(picks), strict=False):
                for label in (key, pick):
                    if label and all(not Words.alike(label, other) for other in merged):
                        merged.append(label)
            lanes[number] = merged
        ep.plan_hinted = hinted
        ep.trace.append({'kind': 'plan', 'ask': shorten(got.get('ask'), 120), 'lanes': {n: v[:6] for n, v in lanes.items()}})
        return lanes

    def next_question(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int]) -> tuple[int, list[str]]:
        """Hinted properties first, then items without a found need, then the items asked least; one property per
        message until the shopper has shown that lists get answered."""
        ep = self.ep

        def rank(n: int) -> tuple[bool, bool, int, int]:
            hinted = any(Words.plain(x) in ep.plan_hinted.get(n, set()) for x in lanes[n])
            return (not hinted, found.get(n, 0) > 0, spent.get(n, 0), n)
        for number in sorted(lanes, key=rank):
            item = ep.items[number - 1]
            labels = [x for x in lanes[number] if not any(Words.alike(x, s) for s in item.settled())]
            if labels:
                return number, labels[:LIST_SIZE if ep.shopper_mode == 'normal' else 1]
        return 0, []

    def run(self) -> str:
        ep = self.ep
        ep.interviewed = True
        if ep.msgs_free() <= 0 or not ep.box.tool('ask'):
            return ''
        if ep.room() >= 6 and ep.time_left() > WRAP_UP_AT + 40:
            ep.opener.widen()
        lanes = self.plan(ep.msgs_free() + 1)
        found = {n: 0 for n in lanes}
        spent = {n: 0 for n in lanes}
        told: list[str] = []
        asked: list[tuple[int, str, str]] = []
        while ep.msgs_free() > 0 and not ep.done() and ep.steps_left() > ep.tail() + len(ep.items) + 1 and ep.time_left() > WRAP_UP_AT + 20:
            number, labels = self.next_question(lanes, found, spent)
            if not labels:
                break
            item = ep.items[number - 1]
            item.asked.extend(labels)
            spent[number] += 1
            text = (f'What {labels[0]} would you like for the {item.name}?' if len(labels) == 1
                    else f"What {', '.join(labels[:-1])} or {labels[-1]} would you like for the {item.name}?")
            sent = self.send([(number, text)], labels=labels)
            if not sent:
                break
            asked += sent
            had = [len(w.active()) for w in ep.items]
            free_before = len(item.free)
            heard = self.hear(sent, {1: labels})
            self.take_heard(heard, told, sent, {1: labels})
            gained = [n for n, w in enumerate(ep.items, 1) if len(w.active()) > had[n - 1]]
            for n in gained:
                found[n] = found.get(n, 0) + 1
            if gained and len(labels) > 1:
                # a list question is answered for one property only: the others were not asked yet
                named = [r.prop for r in item.active()[had[number - 1]:]] if number in gained else []
                others = [x for x in labels if not any(Words.alike(x, p) for p in named) and not any(Words.alike(x, f) for f in item.free)]
                item.asked = [x for x in item.asked if x not in others]
            lost_now = any(whole(k) == 1 for k in heard.get('lost') or []) if isinstance(heard.get('lost'), list) else False
            if lost_now:
                ep.shopper_mode = 'tired'
                item.lost.extend(x for x in labels if x not in item.lost)
            elif ep.shopper_mode == 'unknown' and not gained and len(item.free) > free_before:
                ep.shopper_mode = 'normal'
            ep.trace.append({'kind': 'reply_mode', 'mode': ep.shopper_mode, 'lost': lost_now, 'gained': gained})
        searched = self.value_search([n for n in found if found[n] > 0]) if told else ''
        if told:
            ep.bump()
        if not asked:
            return ''
        summary = 'Best asked the shopper (each message was one step):\n' + '\n'.join(f'Q: {q}\nA: {a}' for _, q, a in asked)
        summary += '\nNew firm requirements from these answers: ' + ('; '.join(told) if told else 'none') + '. Apply them to every pick.'
        if any(rule.points and not rule.resolved for _, rule in ep.needs()):
            chained = ep.chainer.resolve()
            if chained:
                summary += '\n' + chained
        if searched:
            summary += '\n' + searched
        if ep.queued:
            summary += '\n[notes seen meanwhile] ' + ' | '.join(ep.queued)
            ep.queued = []
        return summary

    def send(self, questions: list[tuple[int, str]], floor: int | None = None, labels: list[str] | None = None) -> list[tuple[int, str, str]]:
        ep = self.ep
        tool = ep.box.tool('ask')
        floor = ep.tail() + 1 if floor is None else floor
        sent: list[tuple[int, str, str]] = []
        for number, text in questions:
            if not tool or ep.done() or ep.steps_left() <= floor or ep.msg_used >= ep.msg_limit:
                break
            args = ep.box.message_args(text)
            if not args:
                break
            reply = ep.store.call(tool, args)
            if reply is None:
                break
            ep.msg_used += 1
            answer = reply.get('observation')
            words = Words.spoken(answer)
            sent.append((number, text, words))
            ep.qa.append((number, text, words))
            ep.record.append(f'step {ep.store.step}: message to the shopper {text!r} -> reply {words[:300]!r} (cart then: {ep.cart_mark()})')
            ep.record_steps.append(ep.store.step)
            ep.trace.append({'kind': 'ask', 'q': text, 'a': words[:300], 'about': labels or []})
            if isinstance(answer, dict) and answer.get('error'):
                break
        return sent

    def hear(self, sent: list[tuple[int, str, str]], about: dict[int, list[str]] | None = None) -> dict[str, Any]:
        ep = self.ep
        about = about or {}
        prompt = HEAR + '\n\nITEMS:\n' + '\n'.join(f'item {n}: {w.name}' for n, w in enumerate(ep.items, 1))
        prompt += '\n\nQUESTIONS AND ANSWERS:\n' + '\n'.join(f"{i}. (about item {n}; properties asked about: {', '.join(about[i]) if about.get(i) else 'see the question'}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        got = ep.brain.json(prompt, [FAST, SLOW], 2000, tag='hear') or {}
        pointed = any(isinstance(e, dict) and e.get('points') is True for e in got.get('needs') or []) if isinstance(got.get('needs'), list) else False
        if pointed and ep.brain.slow_ok():
            again = ep.brain.json(prompt, [SLOW], 2000, tag='hear')
            if again:
                got = again
        return got

    def take_heard(self, heard: dict[str, Any], told: list[str], sent: list[tuple[int, str, str]] | None = None, about: dict[int, list[str]] | None = None) -> None:
        ep = self.ep
        many = len(ep.items) > 1
        for entry in heard.get('needs') if isinstance(heard.get('needs'), list) else []:
            if not isinstance(entry, dict) or not str(entry.get('need') or '').strip():
                continue
            number = min(max(1, whole(entry.get('item')) or 1), len(ep.items))
            text = str(entry['need']).strip()
            rule = ep.items[number - 1].require(text, 'shopper', entry.get('points') is True or ep.points_elsewhere(text))
            if rule:
                told.append(f'item {number}: {text}' if many else text)
        for entry in heard.get('drop') if isinstance(heard.get('drop'), list) else []:
            if isinstance(entry, dict) and str(entry.get('property') or '').strip():
                gone = ep.item_at(entry.get('item')).release(str(entry['property']).partition(':')[0])
                told.extend(f'dropped {g}' for g in gone)
        for entry in heard.get('free') if isinstance(heard.get('free'), list) else []:
            if isinstance(entry, dict) and entry.get('property'):
                ep.item_at(entry.get('item')).free.append(str(entry['property']))
        for index in heard.get('lost') if isinstance(heard.get('lost'), list) else []:
            k = whole(index)
            if sent and k and 1 <= k <= len(sent):
                number, question, _ = sent[k - 1]
                names = (about or {}).get(k) or [re.sub(r'^what\s+(.*?)\s+would you like.*$', r'\1', question, flags=re.I)]
                item = ep.items[number - 1]
                item.lost.extend(n for n in names if n not in item.lost)
        budget = heard.get('budget')
        if isinstance(budget, (int, float)) and not isinstance(budget, bool) and budget > 0:
            ep.budget = float(budget)
            told.append(f'budget now {budget:g}')

    # -- searching with what the shopper said ------------------------------------------------------------
    def search_template(self) -> tuple[str, dict[str, Any], str] | None:
        """The search tool, its fixed arguments and its keyword field, learnt from the searches made so far."""
        ep = self.ep
        tool = ep.box.tool('search')
        if not tool:
            return None
        props = ep.box.props(tool)
        texts = ep.box.text_args(tool)
        named = [k for k in texts if re.search(r'(?:^|_)(?:q|query|search|text|phrase|keywords?|terms?|words?)(?:_|$)|query|phrase|keyword|words|text', k, re.I)]
        key = ep.catalog.kind_key()
        labels = {str(r.get(key) or '').strip().lower() for r in ep.catalog.rows.values()} - {''} if key else set()
        musts = set(ep.box.required(tool))
        for args in reversed(ep.searched):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search('[a-z]', args[k], re.I) and args[k].strip().lower() not in labels]
            if not fields:
                continue
            field = fields[0]
            keep = {}
            for k, v in args.items():
                if k == field or not (isinstance((props.get(k) or {}).get('enum'), list) or k in musts):
                    continue
                if isinstance(v, str) and k not in musts and v.strip().lower() in labels:
                    continue
                keep[k] = v
            return tool, keep, field
        if not named:
            return None
        keep = {}
        for k in musts:
            spec = props.get(k) or {}
            if k != named[0] and isinstance(spec.get('enum'), list) and spec['enum']:
                keep[k] = next((o for o in spec['enum'] if re.search(r'word|phrase|text|typed|keyword|search|match', str(o), re.I)), spec['enum'][0])
        return tool, keep, named[0]

    def value_words(self, number: int) -> list[str]:
        out = []
        for n, rule in self.ep.needs():
            if n != number or rule.points:
                continue
            value = rule.value
            if value and not re.search(r'\b(?:not|no|other than|unlike|listing|page|shown|at most|at least|under|over|below|above|less|more|between|within|up to|or)\b', value, re.I):
                out.append(value)
        return out

    def search_words(self, tool: str, base: dict[str, Any], field: str, words: str) -> list[str]:
        ep = self.ep
        args, faults = ep.box.shape(tool, {**base, field: words})
        if faults or ep.done():
            return []
        reply = ep.store.call(tool, args)
        if reply is None:
            return []
        ep.observer.observe(tool, reply.get('observation'))
        ep.observer.read_shape(tool, reply.get('observation'))
        ep.searched = (ep.searched + [dict(args)])[-8:]
        return [i for i, _ in ep.catalog.records_in(reply.get('observation'))]

    def kind_rows(self, number: int, found: list[str], below: float | None = None) -> list[str]:
        ep = self.ep
        key = ep.catalog.kind_key()
        item = ep.items[number - 1]
        wanted = set(re.findall(r'[a-z0-9]{3,}', ' '.join(self.value_words(number)).lower()))
        rows = [i for i in dict.fromkeys(found) if ep.catalog.buyable(i) and (not key or ep.kinds.owns(item, str((ep.catalog.rows.get(i) or {}).get(key) or '')))]
        if below is not None:
            rows = [i for i in rows if (ep.catalog.price_of(i) if ep.catalog.price_of(i) is not None else below) < below]
        return sorted(rows, key=lambda i: (i in ep.catalog.full, -len(wanted & set(re.findall(r'[a-z0-9]{3,}', compact(ep.catalog.rows.get(i) or {}).lower()))), ep.catalog.price_of(i) or 0))

    def value_search(self, numbers: list[int]) -> str:
        ep = self.ep
        plan = self.search_template()
        if not plan or ep.room() < 6 or ep.time_left() < WRAP_UP_AT + 25:
            return ''
        tool, base, field = plan
        seen: list[str] = []
        budget = min(8, max(2, ep.room() // 2))
        for number in numbers:
            item = ep.items[number - 1]
            fresh = [v for v in self.value_words(number) if any(r.source == 'shopper' and r.value == v for r in item.active())]
            if not fresh or budget <= 1:
                continue
            cap = ep.budget if isinstance(ep.budget, (int, float)) else None
            words = fresh[0] if len(fresh[0].split()) <= 2 else ' '.join(fresh[:1] + [item.name])
            found = self.search_words(tool, base, field, words)
            budget -= 1
            rows = [i for i in self.kind_rows(number, found) if cap is None or (ep.catalog.price_of(i) or 0) <= cap or ep.budget_each]
            if not rows and budget > 1 and words != ' '.join(fresh[:1] + [item.name]):
                words = ' '.join(fresh[:1] + [item.name])
                found = self.search_words(tool, base, field, words)
                budget -= 1
                rows = [i for i in self.kind_rows(number, found) if cap is None or (ep.catalog.price_of(i) or 0) <= cap or ep.budget_each]
            seen.append(f'{words!r}: ' + (', '.join(rows[:6]) or 'no row of that kind'))
            if rows:
                opened = ep.opener.fill_facts([i for i in rows if i not in ep.catalog.full][:min(2, budget)], ep.tail() + len(ep.items) + 2)
                budget -= len(opened)
                if opened:
                    seen.append('opened: ' + ' | '.join(opened)[:2400])
        if not seen:
            return ''
        ep.trace.append({'kind': 'value_search', 'seen': [s[:200] for s in seen]})
        return "[Best searched the shopper's values together with the product name and read the first rows of that kind] " + '\n'.join(seen)


# ------------------------------------------------------------------------------------------------------------
# opening listings: attribute views, offer pages, wider looks, cheaper looks
# ------------------------------------------------------------------------------------------------------------
class Opener:
    def __init__(self, ep: Episode):
        self.ep = ep

    def fill_facts(self, ids: list[str], keep: int) -> list[str]:
        ep = self.ep
        tool = ep.facts_tool()
        target = ep.box.one_target(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in ids:
            if ident not in ep.catalog.rows or ident in ep.catalog.full or ep.catalog.sold_out(ident):
                continue
            if ep.steps_left() <= keep or ep.done() or ep.facts_from.get(tool) is False or ep.time_left() < CLOCK_FLOOR + 10:
                break
            args, faults = ep.box.shape(tool, {target: ident})
            if faults:
                break
            reply = ep.store.call(tool, args)
            if reply is None:
                break
            seen = reply.get('observation')
            ep.observer.observe(tool, seen)
            texts.append(f'{ident}: ' + ep.observer.show(seen)[:1500])
        if texts:
            ep.trace.append({'kind': 'fill_facts', 'n': len(texts)})
        return texts

    def open_view(self, tool: str, ident: str) -> dict[str, Any] | None:
        ep = self.ep
        target = ep.box.one_target(tool)
        if not target or ep.done():
            return None
        args, faults = ep.box.shape(tool, {target: ident})
        if faults:
            return None
        reply = ep.store.call(tool, args)
        if reply is None:
            return None
        ep.observer.observe(tool, reply.get('observation'))
        return reply

    def complete_views(self, ids: list[str], keep: int) -> list[str]:
        """Open every single-listing view this store has for each id that lacks one: the attribute view and the
        offer page both count as reading the listing's page before buying it."""
        ep = self.ep
        done = []
        for ident in ids:
            for tool in ep.view_tools():
                if ep.steps_left() <= keep or ep.done() or ep.time_left() < CLOCK_FLOOR + 10:
                    return done
                if ep.catalog.opened_with(ident, tool) or ident in ep.catalog.gone:
                    continue
                reply = self.open_view(tool, ident)
                if reply is None:
                    return done
                done.append(f'{tool} {ident}: ' + ep.observer.show(reply.get('observation'))[:600])
        if done:
            ep.trace.append({'kind': 'views', 'n': len(done)})
        return done

    def widen(self) -> None:
        """Open a few listings per item so the interview can see which attributes they carry."""
        ep = self.ep
        catalog = ep.catalog
        tool = ep.box.tool('compare')
        target = ep.box.many_target(tool)
        if tool and target and ep.facts_from.get(tool) is not False and ep.room() >= 4:
            cap = ep.box.many_room(tool)
            ranked = ep.kinds.item_groups()
            share = max(1, cap // max(1, len(ranked)))
            picks: list[str] = []
            for ids in ranked:
                picks += [i for i in ids if i not in catalog.full and catalog.buyable(i)][:share]
            picks = picks[:cap]
            if len(picks) >= 2:
                args, faults = ep.box.shape(tool, {target: picks})
                if not faults:
                    before = len(catalog.full)
                    reply = ep.store.call(tool, args)
                    if reply is not None:
                        seen = reply.get('observation')
                        ep.observer.observe(tool, seen)
                        rows = catalog.records_in(seen)
                        if rows:
                            ep.facts_from[tool] = ep.facts_from.get(tool, False) or any(facts_of(r) for _, r in rows)
                        ep.trace.append({'kind': 'widen', 'ids': picks, 'opened': len(catalog.full) - before})
                        if len(catalog.full) > before:
                            return
        tool = ep.facts_tool()
        if not tool or not ep.box.one_target(tool):
            return
        wanted = {w for _, rule in ep.needs() for w in re.findall(r'[a-z0-9]{3,}', rule.value.lower())}
        each = 2 if len(ep.items) > 1 else OPENS_PER_ITEM
        if ep.max_steps >= 60:
            each += 1
        picks = []
        for ids in ep.kinds.item_groups():
            opened = sum(1 for i in ids if i in catalog.full)
            pool = [i for i in ids if i not in catalog.full and catalog.buyable(i)]
            if ep.pick == 'cheapest':
                pool.sort(key=lambda i: catalog.price_of(i) if catalog.price_of(i) is not None else float('inf'))
            pool.sort(key=lambda i: -len(wanted & set(re.findall(r'[a-z0-9]{3,}', compact(catalog.rows.get(i) or {}).lower()))))
            picks += pool[:max(0, each - opened)]
        self.fill_facts(picks[:OPENS_PER_ITEM * max(1, len(ep.items))], ep.tail() + len(ep.items) + 4)

    # -- cheapest pick: look below the cart's price ------------------------------------------------------
    def list_category(self, number: int, cap: float | None, more: int = 0) -> list[str] | None:
        ep = self.ep
        spec = ep.box.browse
        label = ep.kinds.shelf_of(number)
        if not spec or not label or ep.browse_misses >= 2 or (cap is not None and not spec['price']):
            return None
        args = {**spec['fixed'], spec['category']: label}
        if cap is not None:
            args[spec['price']] = cap
        args, faults = ep.box.shape(spec['tool'], args)
        if faults or ep.done():
            ep.browse_misses += 1
            return None
        reply = ep.store.call(spec['tool'], args)
        if reply is None:
            ep.browse_misses += 1
            return None
        seen = reply.get('observation')
        ep.observer.observe(spec['tool'], seen)
        ep.observer.read_shape(spec['tool'], seen)
        found = [i for i, _ in ep.catalog.records_in(seen)]
        if isinstance(seen, dict) and seen.get('error'):
            ep.browse_misses += 1
            return None
        page = ep.observer.paging(spec['tool'], seen)
        ep.last_match = (page[0], page[1]) if page else (len(found), len(found))
        for _ in range(more):
            page = ep.observer.paging(spec['tool'], seen)
            if not page or page[3] is None or ep.done():
                break
            call, faults = ep.box.shape(spec['tool'], {**args, page[2]: page[3]})
            reply = ep.store.call(spec['tool'], call) if not faults else None
            if reply is None:
                break
            seen = reply.get('observation')
            ep.observer.observe(spec['tool'], seen)
            found += [i for i, _ in ep.catalog.records_in(seen)]
        return found

    def other_values(self, number: int) -> list[str]:
        ep = self.ep
        table = ep.kinds.key_table()
        keys = ep.kinds.lane_keys(number, table)[:3]
        said = ' '.join(ep.interviewer.value_words(number)).lower() + ' ' + ep.items[number - 1].name.lower()
        lanes = []
        for key in keys:
            values = next((v for label, name, _, v in table if name == key and (len(ep.items) == 1 or ep.kinds.owner_of(label) == number)), [])
            lanes.append([v for v in values if v and len(v) <= 20 and v not in said and re.search('[a-z0-9]', v)][:2])
        return [lane[k] for k in range(2) for lane in lanes if k < len(lane)]

    def cheaper_sweep(self) -> bool:
        ep = self.ep
        key = ep.catalog.kind_key()
        plan = ep.interviewer.search_template()
        mark = ep.cart_mark()
        if mark in ep.swept or ep.room() < 2 or ep.time_left() < WRAP_UP_AT + 25 or not key:
            return False
        ep.swept.add(mark)
        limit = ep.store.step + min(SWEEP_STEPS, ep.room())
        ran = False
        for line in ep.basket.cart_ids():
            price = ep.catalog.price_of(line)
            label = str((ep.catalog.rows.get(line) or {}).get(key) or '')
            number = ep.kinds.owner_of(label) if label else 0
            if price is None or not number or ep.store.step >= limit:
                continue
            found = self.list_category(number, int(price) - 1 if price > 1 else price * 0.9, 1 if limit - ep.store.step >= 2 else 0)
            if found is not None:
                ran = True
                matched, shown = ep.last_match or (0, 0)
                prices = sorted(p for p in (ep.catalog.price_of(i) for i in found) if p is not None)
                if matched > shown and len(prices) >= 4 and ep.store.step < limit and ep.room() >= 4:
                    # more rows match than the store shows: look once more under a lower cap, where fewer hide
                    lower = prices[min(len(prices) - 1, max(2, len(prices) // 2))]
                    if lower < price:
                        self.list_category(number, lower, 0)
            elif plan:
                base = ep.interviewer.value_words(number)[:2]
                for other in self.other_values(number)[:2]:
                    if ep.store.step >= limit:
                        break
                    ep.interviewer.search_words(plan[0], plan[1], plan[2], ' '.join(base + [other, ep.items[number - 1].name]))
                    ran = True
        ep.trace.append({'kind': 'sweep', 'ran': ran})
        return ran

    def cheaper_rows(self) -> list[str]:
        ep = self.ep
        if ep.pick != 'cheapest':
            return []
        mark = ep.cart_mark()
        key = ep.catalog.kind_key()
        if mark in ep.cheaper_marks or not key:
            return []
        ep.cheaper_marks.add(mark)
        cart = ep.basket.cart_ids()
        picks: list[str] = []
        for line in cart:
            price = ep.catalog.price_of(line)
            kind = str((ep.catalog.rows.get(line) or {}).get(key) or '').strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in ep.catalog.order if i not in cart and i not in ep.catalog.full and ep.catalog.buyable(i)
                    and str((ep.catalog.rows.get(i) or {}).get(key) or '').strip().lower() == kind
                    and (ep.catalog.price_of(i) if ep.catalog.price_of(i) is not None else float('inf')) < price]
            pool.sort(key=lambda i: ep.catalog.price_of(i) or 0.0)
            picks += [i for i in pool[:CHEAPER_OPENS if len(cart) == 1 else 2] if i not in picks]
        return self.fill_facts(picks[:CHEAPER_OPENS], ep.tail() + 2)

    def price_ladder(self, number: int) -> str:
        """For a cheapest pick: list the item's category under falling price caps until every cheaper row is
        visible, then read rows cheapest first until one states every requirement. Returns a note for the model."""
        ep = self.ep
        catalog = ep.catalog
        spec = ep.box.browse
        if ep.pick != 'cheapest' or not spec or not spec['price'] or ep.ladder_done.get(number) or ep.room() < 8:
            return ''
        item = ep.items[number - 1]
        if not ep.kinds.shelf_of(number) or any(rule.points and not rule.resolved for _, rule in ep.needs() if _ == number):
            return ''
        light = ep.ladder_done.get(number) is False      # re-armed after a sell-out: no new listings, few opens
        ep.ladder_done[number] = True
        cap = (ep.budget if ep.budget_each else ep.budget / max(1, item.qty)) if isinstance(ep.budget, (int, float)) else None
        if cap is None:
            return ''
        before = ep.store.step
        listings = 3 if light else 0
        bound = cap if not light else (ep.last_match and cap) or cap
        while listings < 3 and ep.room() >= 7 and ep.time_left() > WRAP_UP_AT + 25:
            found = self.list_category(number, bound, 0)
            listings += 1
            if found is None:
                break
            matched, shown = ep.last_match or (0, 0)
            prices = sorted(p for p in (catalog.price_of(i) for i in found if catalog.buyable(i)) if p is not None)
            if matched <= shown or len(prices) < 4:
                break
            lower = prices[max(2, len(prices) // 2)]
            if lower >= bound:
                break
            bound = lower
        pool = sorted((i for i in ep.kinds.item_ids(number) if catalog.buyable(i) and catalog.price_of(i) is not None and catalog.price_of(i) <= cap), key=lambda i: catalog.price_of(i) or 0)
        verified = None
        rejected: list[str] = []
        opens = 0
        limit = 2 if light else min(6, max(2, ep.room() // 2))
        for ident in pool:
            if ep.room() < 5 or ep.time_left() < WRAP_UP_AT + 20:
                break
            if ident not in catalog.full:
                if opens >= limit or not self.fill_facts([ident], ep.tail() + 3):
                    break
                opens += 1
            grades = ep.basket.grades_of(ident, number)
            if grades and all(g in ('strong', 'weak') for g in grades.values()):
                verified = ident
                break
            rejected.append(f"{ident} ({catalog.price_of(ident):g}: " + ', '.join(f'{p} {g}' for p, g in grades.items()) + ')')
        ep.trace.append({'kind': 'ladder', 'item': number, 'steps': ep.store.step - before, 'bound': bound, 'pool': len(pool), 'verified': verified, 'rejected': len(rejected)})
        if not pool:
            return ''
        complete = (ep.last_match or (0, 0))[0] <= (ep.last_match or (0, 0))[1]
        text = f"[Best priced the {item.name} category itself, {ep.store.step - before} steps] cheapest buyable rows under the budget, in price order: " + ', '.join(f'{i} ({catalog.price_of(i):g})' for i in pool[:10])
        if verified:
            text += f". Read cheapest-first, {verified} at {catalog.price_of(verified):g} is the first whose attributes state every plain requirement" + (f"; cheaper rows read and failing: {'; '.join(rejected[:6])}" if rejected else '') + '.'
            settled = complete and not any(i not in catalog.full for i in pool if (catalog.price_of(i) or 0) < (catalog.price_of(verified) or 0))
            text += ' Every row below it is visible and was read, so it is the cheapest fit for the requirements known now.' if settled else ' Rows cheaper than it that were not read may still fit: read them if steps allow.'
            ep.ladder_note = f"{verified} at {catalog.price_of(verified):g} is the cheapest read listing of the kind whose keys state every plain requirement" + ('; every cheaper row was read and fails' if settled else '; cheaper unread rows may exist') + '. Do not call a pricier listing better on price grounds.'
        else:
            text += '. None of the rows read states every requirement' + (f" ({'; '.join(rejected[:6])})" if rejected else '') + '; search by the requirement\'s value for a listing that does, and read the next cheapest rows.'
        return text

    def wider_hold(self) -> str:
        ep = self.ep
        if ep.wider_held or ep.pick != 'cheapest':
            return ''
        ep.wider_held = True
        if self.cheaper_sweep() or ep.room() < 10:
            return ''
        lines = [f'{i} at {ep.catalog.price_of(i):g}' for i in ep.basket.cart_ids() if ep.catalog.price_of(i) is not None]
        return ("Not run yet (no step used). The pick is the cheapest and steps remain. Searches match titles, so a requirement's value finds "
                "few of the fitting listings and rows come in no price order: run a few more searches under other names for the product "
                "and look only at buyable rows priced below the cart's line" + (f" ({'; '.join(lines)})" if lines else '') +
                "; read the attributes of the cheapest such rows, swap in any that fits, then check out.")


# ------------------------------------------------------------------------------------------------------------
# chains: requirements whose value lives on another listing's page
# ------------------------------------------------------------------------------------------------------------
class Chainer:
    def __init__(self, ep: Episode):
        self.ep = ep

    def resolve(self) -> str:
        """Follow every pointer requirement to the page that states its value, opening each page on the way."""
        ep = self.ep
        pending = [(n, rule) for n, rule in ep.needs() if rule.points and not rule.resolved]
        if not pending or ep.buys > 0:
            return ''
        notes: list[str] = []
        for number, rule in pending:
            if ep.chain_steps >= CHAIN_STEPS or ep.room() < 3 or ep.time_left() < WRAP_UP_AT + 20 or ep.done():
                break
            if ep.chain_misses.get(rule.text, 0) >= 2:
                continue
            parsed = ep.brain.json(REF_PARSE + '\n\nREQUIREMENT:\n' + rule.text + f'\n\nITEM IT BELONGS TO: {ep.items[number - 1].name}', [FAST, SLOW], 900, tag='ref') or {}
            anchor = parsed.get('anchor') if isinstance(parsed.get('anchor'), dict) else None
            prop = str(parsed.get('property') or rule.prop).strip() or rule.prop
            if not anchor or not str(anchor.get('category') or anchor.get('shop') or '').strip():
                ep.chain_misses[rule.text] = ep.chain_misses.get(rule.text, 0) + 1
                continue
            before = ep.store.step
            value, path = self.follow(anchor, prop, 0, [])
            ep.chain_steps += ep.store.step - before
            if value is None:
                ep.chain_misses[rule.text] = ep.chain_misses.get(rule.text, 0) + 1
                notes.append(f'{rule.text}: the listing it points at was not found yet' + (f' (pages opened: {", ".join(path)})' if path else ''))
                ep.trace.append({'kind': 'chain_miss', 'need': rule.text, 'path': path})
                continue
            rule.resolved = self.compose(str(parsed.get('relation') or 'identical'), prop, value, parsed.get('amount'), str(parsed.get('unit') or ''))
            ep.bump()
            notes.append(f'{rule.text} -> {rule.resolved} (pages opened in order: {", ".join(path)})')
            ep.trace.append({'kind': 'chain', 'need': rule.text, 'resolved': rule.resolved, 'path': path})
        if not notes:
            return ''
        return '[Best followed the pages that requirements point at]\n' + '\n'.join(notes) + '\nApply the resolved values when choosing; the order counts only with every page of each chain opened, which Best has done for the chains marked resolved.'

    def follow(self, anchor: dict[str, Any], prop: str, depth: int, trail: list[str]) -> tuple[str | None, list[str]]:
        ep = self.ep
        if depth > CHAIN_HOPS or ep.room() < 2 or ep.done():
            return None, trail
        ident = self.find_anchor(anchor, prop)
        if not ident or ident in trail:
            return None, trail
        trail = trail + [ident]
        ep.opener.complete_views([ident], ep.tail() + 1)
        prompt = PAGE_READ + f'\n\nPROPERTY WANTED: {prop}\n\nTHE PAGE (listing {ident}):\n' + ep.catalog.record_text(ident, 3500)
        read = ep.brain.json(prompt, [FAST, SLOW], 1200, tag='page') or {}
        for pair in read.get('rules') if isinstance(read.get('rules'), list) else []:
            if isinstance(pair, list) and len(pair) == 2:
                ep.observer.add_rule(str(pair[0]), str(pair[1]))
        for negation in read.get('negations') if isinstance(read.get('negations'), list) else []:
            ep.catalog.negations.setdefault(ident, []).append(str(negation))
        redirect = read.get('redirect') if isinstance(read.get('redirect'), dict) else None
        if redirect and str(redirect.get('category') or redirect.get('shop') or '').strip():
            return self.follow(redirect, prop, depth + 1, trail)
        value = str(read.get('value') or '').strip()
        if not value:
            for key, shown in (facts_of(ep.catalog.full.get(ident) or {}) or {}).items():
                if Words.alike(prop, key) and str(shown).strip():
                    value = str(shown).strip()
                    break
        return (value or None), trail

    def find_anchor(self, anchor: dict[str, Any], prop: str) -> str | None:
        """The one listing a description points at: by shop, kind and the traits named, searching when unseen."""
        ep = self.ep
        catalog = ep.catalog
        shop = str(anchor.get('shop') or '').strip()
        category = str(anchor.get('category') or '').strip()
        traits = [str(t) for t in anchor.get('traits') or [] if str(t).strip()] if isinstance(anchor.get('traits'), list) else []
        trait_words = {w for t in traits for w in re.findall(r'[a-z0-9.]{2,}', t.lower().partition(':')[2] or t.lower())}
        shop_words = set(re.findall(r'[a-z0-9]{3,}', shop.lower()))
        cat_words = Words.content(category)

        def score(ident: str) -> tuple[int, int, int]:
            text = compact(catalog.rows.get(ident) or {}).lower() + ' ' + catalog.pages.get(ident, '').lower()
            seller = catalog.seller_of(ident).lower()
            shop_hit = 2 if shop_words and all(w in seller for w in shop_words) else (1 if shop_words and any(w in text for w in shop_words) else 0)
            label = catalog.label_of(ident).lower()
            cat_hit = 1 if (cat_words and (cat_words & Words.content(label) or all(w in text for w in list(cat_words)[:2]))) else 0
            return shop_hit, cat_hit, len(trait_words & set(re.findall(r'[a-z0-9.]{2,}', text)))

        def candidates() -> list[str]:
            ranked = sorted(catalog.order, key=score, reverse=True)
            return [i for i in ranked if score(i)[0] >= (2 if shop_words else 0) and (score(i)[1] or not cat_words)][:4]
        found = candidates()
        plan = ep.interviewer.search_template()
        tries = 0
        while not found and plan and tries < 2 and ep.room() >= 3:
            words = ' '.join(x for x in [shop, category] if x) if tries == 0 else ' '.join([category] + [t.partition(':')[2].strip() or t for t in traits[:2]])
            if not words.strip():
                break
            ep.interviewer.search_words(plan[0], plan[1], plan[2], words[:60])
            tries += 1
            found = candidates()
        if not found:
            return None
        unread = [i for i in found[:3] if i not in catalog.full]
        if unread:
            ep.opener.fill_facts(unread, ep.tail() + 1)
        if len(found) == 1 and not traits:
            return found[0]
        records = '\n'.join(f'{i} (seller: {catalog.seller_of(i) or "?"}): ' + catalog.record_text(i, 1200) for i in found[:4])
        prompt = LINK_PICK + f'\n\nDESCRIPTION: shop or brand {shop or "(not named)"}; kind {category or "(not named)"}; traits: ' + ('; '.join(traits) or '(none)') + '\n\nCANDIDATE RECORDS:\n' + records
        got = ep.brain.json(prompt, [FAST, SLOW], 600, tag='link') or {}
        choice = str(got.get('id') or '').strip()
        if choice in found:
            return choice
        close = [str(c) for c in got.get('close') or [] if str(c) in found] if isinstance(got.get('close'), list) else []
        if close:
            return close[0]
        top = score(found[0])
        return found[0] if top[0] == 2 and top[2] >= max(1, len(trait_words) // 2) else None

    def compose(self, relation: str, prop: str, value: str, distance: Any, unit: str) -> str:
        value = value.strip().rstrip('.')
        if relation == 'different':
            return f'{prop}: not {value} (any value other than {value})'
        if relation == 'partner':
            partner = self.partner_of(value)
            if partner:
                return f'{prop}: {partner} (the store pairs {value} with {partner})'
            return f"{prop}: the value the store's pairing rule puts with {value} (rule not read yet; look for it on listing pages)"
        if relation == 'within':
            number, base = amount(distance), amount(value)
            if number is not None and base is not None:
                return f'{prop}: between {base - number:g} and {base + number:g} {unit} (within {number:g} {unit} of {value})'.replace('  ', ' ')
            return f'{prop}: within {distance} {unit} of {value}'.replace('  ', ' ')
        return f'{prop}: {value}'

    def partner_of(self, value: str) -> str | None:
        words = Words.content(value)
        for a, b in self.ep.catalog.rules:
            for first, second in ((a, b), (b, a)):
                theirs = Words.content(first)
                if theirs == words or (words and words <= theirs):
                    return second
        return None


# ------------------------------------------------------------------------------------------------------------
# the basket: what the cart holds, and whether it matches the request
# ------------------------------------------------------------------------------------------------------------
class Basket:
    def __init__(self, ep: Episode):
        self.ep = ep

    def note_cart(self, name: str, args: dict[str, Any]) -> None:
        """Update the record of the cart from one cart call."""
        ep = self.ep
        ep.cart_touched = True
        ep.cart_at = ep.store.step
        parts = ep.box.cart_parts(name)
        props = parts['props']
        removal = any(isinstance((props.get(k) or {}).get('enum'), list) and len(props[k]['enum']) >= 2 and v == ep.box.remove_option(name) for k, v in args.items())
        loose_count = next((whole(v) for v in args.values() if not isinstance(v, str) and whole(v) is not None), None)
        lines: list[tuple[str, int | None]] = []
        whole_cart = False
        for key, value in args.items():
            if isinstance(value, list):
                inner = (props.get(key) or {}).get('items') or {}
                if not value and isinstance(inner, dict) and (inner.get('type') == 'object' or 'properties' in inner):
                    whole_cart = True
                for line in value:
                    if isinstance(line, dict):
                        whole_cart = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in ep.catalog.rows), None)
                        count = next((whole(v) for v in line.values() if not isinstance(v, str) and whole(v) is not None), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in ep.catalog.rows:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in ep.catalog.rows:
                lines.append((value, loose_count))
        if whole_cart:
            ep.cart_lines = {i: c for i, c in lines if c and c > 0}
            return
        sets = ep.box.cart_sets_count(name)
        for ident, count in lines:
            have = ep.cart_lines.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif sets:
                have = count if count is not None else 1
            else:
                have += count if count is not None else 1
            if have > 0:
                ep.cart_lines[ident] = have
            else:
                ep.cart_lines.pop(ident, None)

    def cart_fresh(self) -> bool:
        return self.ep.cart_view is not None and self.ep.cart_seen_at >= self.ep.cart_at

    def cart_ids(self) -> list[str]:
        ep = self.ep
        if self.cart_fresh():
            text = compact(ep.cart_view)
            return [i for i in ep.catalog.rows if i in text]
        return list(ep.cart_lines)

    def held_units(self, ident: str) -> int:
        ep = self.ep
        if self.cart_fresh():
            for node, _, _ in nodes(ep.cart_view, 8):
                if not isinstance(node, dict) or ident not in [v.strip() for v in node.values() if isinstance(v, str)]:
                    continue
                counts = [whole(v) for k, v in node.items() if not isinstance(v, str) and whole(v) is not None and re.search(r'\b(?:qty|quantity|count|units?|number|amount|how many|pieces)\b', Words.plain(k))]
                if len(counts) == 1:
                    return counts[0]
        return ep.cart_lines.get(ident, 1)

    def cart_text(self) -> str:
        ep = self.ep
        if self.cart_fresh():
            return compact(shorten(ep.cart_view, 300))[:3000]
        return "lines by Best's record of the cart calls (listing: units): " + compact(ep.cart_lines)

    def cart_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        """The cart tool calls that leave exactly `lines` in the cart, removing `drop`; None when the tool's
        shape does not allow it."""
        ep = self.ep
        name = ep.box.tool('cart') or ''
        parts = ep.box.cart_parts(name)
        if parts['lines']:
            key, inner = parts['lines']
            id_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'string' in Schema.kinds(s) and 'enum' not in s), None)
            count_key = next((k for k, s in inner.items() if isinstance(s, dict) and 'integer' in Schema.kinds(s)), None)
            if not id_key:
                return None
            return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        target, count, mode = parts['target'], parts['count'], parts['mode']
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        sets = ep.box.cart_sets_count(name)
        for ident in drop:
            have = ep.cart_lines.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: ep.box.remove_option(name), **({count: have} if count else {})})
            elif count and sets:
                calls.append({target: ident, count: 0})
            elif count and isinstance(parts['props'][count].get('minimum'), int) and parts['props'][count]['minimum'] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        present = self.cart_ids()
        for ident, want in lines.items():
            have = ep.cart_lines.get(ident, 0) if ident in present else 0
            if have == want:
                continue
            if have and not sets:
                if not mode:
                    return None
                calls.append({target: ident, mode: ep.box.remove_option(name), **({count: have} if count else {})})
            calls.append({target: ident, **({mode: ep.box.add_option(name)} if mode else {}), **({count: want} if count else {})})
        return calls

    def set_cart(self, lines: dict[str, int], drop: list[str]) -> list[str] | None:
        ep = self.ep
        name = ep.box.tool('cart')
        calls = self.cart_calls(lines, drop)
        if not name or calls is None or len(calls) + 1 > ep.steps_left():
            return None
        done = []
        for args in calls:
            args, faults = ep.box.shape(name, args)
            if faults:
                return None
            reply = ep.store.call(name, args)
            if reply is None or ep.done():
                return None
            self.note_cart(name, args)
            ep.cart_said = compact(reply.get('observation'))
            ep.observer.observe(name, reply.get('observation'))
            done.append(compact(args) + ' -> ' + ep.observer.show(reply.get('observation'))[:300])
        return done

    def line_faults(self) -> list[str]:
        """What is wrong with the cart as a set of lines: missing items, extra kinds, wrong counts, sold-out
        lines, a total over the budget."""
        ep = self.ep
        catalog = ep.catalog
        key = catalog.kind_key()
        ids = [i for i in self.cart_ids() if (self.held_units(i) or 0) > 0]
        if not ids or not key:
            return []
        label = {i: str((catalog.rows.get(i) or {}).get(key) or '') for i in ids}
        known = {n for n, item in enumerate(ep.items, 1) if any(ep.kinds.owns(item, str(r.get(key) or '')) for r in catalog.rows.values())}
        owner = {i: [n for n, item in enumerate(ep.items, 1) if ep.kinds.owns(item, label[i])] for i in ids}
        faults: list[str] = []
        for n, item in enumerate(ep.items, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f'item {n} ({item.name}) has no line in the cart')
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"item {n} ({item.name}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.held_units(i)
                if owner[i] == [n] and have != item.qty:
                    faults.append(f'{i} is held at {have} unit(s); item {n} ({item.name}) wants {item.qty}')
        if len(known) == len(ep.items):
            for i in ids:
                if not owner[i]:
                    sure = ep.items[0].shelf if len(ep.items) == 1 and ep.items[0].label_sure else ''
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the shopper asked for" + (f" (the store's label for it is {sure!r})" if sure else '') + ': take it out')
        faults.extend(f'{i} is sold out' for i in ids if catalog.sold_out(i))
        prices = {i: catalog.price_of(i) for i in ids}
        if isinstance(ep.budget, (int, float)) and all(p is not None for p in prices.values()):
            if ep.budget_each:
                faults.extend(f'{i} costs {prices[i]:g}, over the {ep.budget:g} limit for each item' for i in ids if ep.budget < prices[i] <= 10 * ep.budget)
            else:
                total = sum(prices[i] * self.held_units(i) for i in ids)
                if ep.budget + 1e-9 < total <= 10 * ep.budget:
                    faults.append(f'the total is {total:g}, over the budget of {ep.budget:g}')
        return faults

    def count_fixes(self) -> list[tuple[str, int, int]]:
        ep = self.ep
        key = ep.catalog.kind_key()
        if not key:
            return []
        out = []
        cart = self.cart_ids()
        for ident in cart:
            have = self.held_units(ident) or 0
            owners = [n for n, item in enumerate(ep.items, 1) if ep.kinds.owns(item, str((ep.catalog.rows.get(ident) or {}).get(key) or ''))]
            if have <= 0 or len(owners) != 1 or ep.catalog.sold_out(ident):
                continue
            item = ep.items[owners[0] - 1]
            mine = [i for i in cart if (ep.cart_lines.get(i, 1) or 0) > 0 and ep.kinds.owns(item, str((ep.catalog.rows.get(i) or {}).get(key) or ''))]
            if len(mine) == 1 and have != item.qty:
                out.append((ident, have, item.qty))
        return out

    def fix_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        ep = self.ep
        name = ep.box.tool('cart') or ''
        parts = ep.box.cart_parts(name)
        target, count, mode = parts['target'], parts['count'], parts['mode']
        if not name or not target or not count:
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if ep.box.cart_sets_count(name):
                calls.append({target: ident, count: want})
            elif mode:
                calls += [{target: ident, mode: ep.box.remove_option(name), count: have}, {target: ident, mode: ep.box.add_option(name), count: want}]
            else:
                return []
        if len(calls) + 3 > ep.steps_left():
            return []
        done = []
        for args in calls:
            args, faults = ep.box.shape(name, args)
            if faults:
                break
            reply = ep.store.call(name, args)
            if reply is None or ep.done():
                break
            self.note_cart(name, args)
            ep.cart_said = compact(reply.get('observation'))
            ep.observer.observe(name, reply.get('observation'))
            done.append(compact(args) + ' -> ' + ep.observer.show(reply.get('observation'))[:300])
        ep.trace.append({'kind': 'count_fix', 'calls': done})
        return done

    # -- compatibility between requested items ------------------------------------------------------------
    def fit_index(self) -> dict[str, dict[str, Any]]:
        catalog = self.ep.catalog
        marks: dict[str, dict[str, Any]] = {}
        for ident in catalog.order:
            record = catalog.full.get(ident) or catalog.rows.get(ident) or {}
            pieces = [prose_of(record), catalog.pages.get(ident, '')]
            pieces += [f'{k} {v}' for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str)]
            pieces += [v for v in record.values() if isinstance(v, str)]
            marks[ident] = fit_marks('. '.join(pieces))
        return marks

    def fit_faults(self) -> list[str]:
        ep = self.ep
        if not ep.bound:
            return []
        marks = self.fit_index()
        cart = self.cart_ids()
        faults = []
        for part, main in ep.bound:
            devices = [i for i in cart if i in ep.kinds.item_ids(main)]
            parts = [i for i in cart if i in ep.kinds.item_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            device = marks.get(devices[0]) or {}
            for line in parts:
                if fit_verdict(marks.get(line) or {}, device) is not False:
                    continue
                good = [i for i in ep.kinds.item_ids(part) if fit_verdict(marks.get(i) or {}, device) and not ep.catalog.sold_out(i) and i not in cart][:4]
                named = (marks.get(line) or {}).get('works') or (marks.get(line) or {}).get('series') or ['other models']
                own = sorted(set([device.get('own') or ''] + list(device.get('series') or [])) - {''})
                faults.append(f"{line} ({ep.items[part - 1].name}) names {', '.join(sorted(named)[:4])}, not {devices[0]} ({', '.join(own)})" + (f"; seen parts that work with it: {', '.join(good)}" if good else ''))
        return faults

    def fit_plan(self) -> str:
        ep = self.ep
        if not ep.bound:
            return ''
        marks = self.fit_index()
        lines = []
        for main in sorted({m for _, m in ep.bound}):
            parts = [p for p, m in ep.bound if m == main]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in ep.kinds.item_ids(p):
                    if ep.catalog.sold_out(i):
                        continue
                    mark = marks.get(i) or {}
                    for code in set(mark.get('works') or []) | set(mark.get('series') or []):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            texts = []
            for code, per in sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:4]:
                devices = [i for i in ep.kinds.item_ids(main) if any(same_code(code, c) for c in [(marks.get(i) or {}).get('own') or ''] if c) or code in ((marks.get(i) or {}).get('series') or [])][:2]
                texts.append(f'{code}: ' + ', '.join(f'{n} {ep.items[p - 1].name}' for p, n in sorted(per.items())) + (f" (device seen: {', '.join(devices)})" if devices else ' (device not seen yet)'))
            if texts:
                lines.append(f"codes the {ep.items[main - 1].name} candidates' parts name -> " + '; '.join(texts))
        if not lines:
            return ''
        return "[fit plan: a part fits a device when the part's text names the device's model code or series, or both carry the same series code] " + ' | '.join(lines)

    # -- evidence on the cart's lines ---------------------------------------------------------------------
    def grades_of(self, ident: str, number: int) -> dict[str, str]:
        """Grade of each plain-valued requirement of item `number` on a read listing."""
        ep = self.ep
        out = {}
        for n, rule in ep.needs():
            value = rule.resolved or rule.value
            if n == number and Catalog.simple_value(value):
                out[rule.prop] = ep.catalog.evidence(ident, rule.prop, value)['grade']
        return out

    def doubtful_lines(self) -> list[tuple[str, int, str]]:
        """Cart lines whose evidence on a plain-valued requirement is shaky: (ident, item number, property)."""
        ep = self.ep
        catalog = ep.catalog
        out = []
        for ident in self.cart_ids():
            if ident not in catalog.full:
                continue
            number = ep.kinds.owner_of(catalog.label_of(ident)) if catalog.kind_key() else (1 if len(ep.items) == 1 else 0)
            if not number:
                continue
            for prop, grade in self.grades_of(ident, number).items():
                if grade in ('conflict', 'fails'):
                    out.append((ident, number, prop))
        return out

    def proof_faults(self) -> tuple[list[str], list[str]]:
        """Cart lines whose evidence on a stated requirement is shaky, with read alternatives that stand better.

        Returns (faults, better ids). A line fails when a requirement grades conflict or fails; it is doubtful when
        a requirement grades mixed or weak while another read, buyable listing of the same kind grades strong on
        every requirement (and is not pricier under a cheapest pick)."""
        ep = self.ep
        catalog = ep.catalog
        key = catalog.kind_key()
        faults: list[str] = []
        better: list[str] = []
        order = {'strong': 0, 'weak': 1, 'mixed': 2, 'unknown': 3, 'fails': 4, 'conflict': 4}
        for ident in self.cart_ids():
            if ident not in catalog.full:
                continue
            number = ep.kinds.owner_of(catalog.label_of(ident)) if key else (1 if len(ep.items) == 1 else 0)
            if not number:
                continue
            grades = self.grades_of(ident, number)
            if not grades:
                continue
            worst = max(grades.values(), key=lambda g: order[g])
            if worst in ('strong', 'unknown'):
                continue
            price = catalog.price_of(ident)
            rivals = []
            for other in ep.kinds.item_ids(number):
                if other == ident or other not in catalog.full or not catalog.buyable(other) or other in better:
                    continue
                theirs = self.grades_of(other, number)
                if not theirs or any(g != 'strong' for g in theirs.values()):
                    continue
                if ep.pick == 'cheapest' and price is not None and (catalog.price_of(other) or float('inf')) > price:
                    continue
                if isinstance(ep.budget, (int, float)) and not ep.budget_each and (catalog.price_of(other) or 0) * ep.items[number - 1].qty > ep.budget:
                    continue
                rivals.append(other)
            shaky = ', '.join(f'{p} {g}' for p, g in grades.items() if g != 'strong')
            if rivals:
                faults.append(f"{ident}: {shaky}, while {', '.join(rivals[:3])} state every requirement outright with agreeing keys")
                better.extend(rivals[:2])
            elif worst == 'fails':
                faults.append(f"{ident}: {shaky} (a key naming the property states another value and nothing states the wanted one; the store will not count it)")
        return faults, better


# ------------------------------------------------------------------------------------------------------------
# the judge: checks before an order, what to do instead of a decline, the steps a request asks for
# ------------------------------------------------------------------------------------------------------------
class Judge:
    def __init__(self, ep: Episode):
        self.ep = ep

    # -- the cart check -----------------------------------------------------------------------------------
    def check_key(self, ids: list[str]) -> str:
        return '|'.join(sorted(ids)) + f'#{self.ep.catalog.version}'

    def check_cart(self, ids: list[str] | None = None, limit: float = SPEND_SOFT) -> str:
        ep = self.ep
        planned = ids is not None
        ids = ids if planned else ep.basket.cart_ids()
        key = self.check_key(ids)
        judges = ep.brain.judges(limit)
        if key in ep.checked or not judges or not ids:
            return ''
        ep.checked.add(key)
        prompt = CHECK + '\n\nREQUEST:\n' + ep.request + '\n\nREQUIREMENTS IN FORCE:\n' + ep.ledger()
        answers = [line for line in ep.record if 'message to the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers[-6:])
        if planned:
            prompt += '\n\nCART (planned; judge these as carted at the requested quantities: ' + '; '.join(f'item {n}: quantity {w.qty}' for n, w in enumerate(ep.items, 1)) + '):\n' + ', '.join(ids)
        else:
            prompt += '\n\nCART:\n' + ep.basket.cart_text()
        prompt += "\n\nRECORDS OF THE CART'S LISTINGS:\n" + '\n'.join(f'{i}: ' + ('SOLD OUT (the store says it can no longer be bought). ' if ep.catalog.sold_out(i) else '') + ep.catalog.record_text(i, 2000) for i in ids)
        proof = ep.observer.evidence_lines(ids)
        if proof:
            prompt += '\n\n' + proof
        if ep.pick == 'cheapest' and ep.ladder_note:
            prompt += '\n\nCODE VERDICT ON PRICE: ' + ep.ladder_note
        if ep.pick in ('cheapest', 'most', 'least'):
            prompt += '\n\nOTHER LISTINGS SEEN (newest first):\n' + self.others(ids, 10)
        got = ep.brain.json(prompt, judges, 1200, tag='check')
        ep.trace.append({'kind': 'check', 'cart': ids, 'got': shorten(got, 200)})
        if not got or got.get('ok') is not False:
            return ''
        ep.leads = [i for i in ep.leads if i not in ids]
        problems = [str(p) for p in got.get('problems') or []][:5] if isinstance(got.get('problems'), list) else []
        better = [str(b) for b in got.get('better') or [] if str(b) in ep.catalog.rows and not ep.catalog.sold_out(str(b))][:4] if isinstance(got.get('better'), list) else []
        ep.better = better
        text = 'A careful review says this cart is not right yet: ' + '; '.join(problems or ['see the requirements'])
        if better:
            text += '. Listings that look better: ' + ', '.join(better)
        return text + '. Fix the cart (or show the review is wrong by re-reading the records), then check out.'

    def drop_sold_lines(self) -> str:
        """Take known sold-out lines out of the cart when a checkout is about to go out with few steps left."""
        ep = self.ep
        lost = [i for i in ep.basket.cart_ids() if ep.catalog.sold_out(i)]
        if not ep.box.tool('cart') or not lost or ep.steps_left() < len(lost) + 1:
            return ''
        keep = {i: ep.basket.held_units(i) or 1 for i in ep.basket.cart_ids() if i not in lost}
        done = ep.basket.set_cart(keep, lost)
        if not done:
            return ''
        ep.trace.append({'kind': 'drop_sold', 'lost': lost})
        return 'Best took the sold-out line(s) out of the cart first: ' + '; '.join(done)

    def swap_to_better(self, better: list[str]) -> str:
        """Replace a cart line with a candidate found better, when that candidate's attributes were read and it is
        buyable (and cheaper, under a cheapest pick)."""
        ep = self.ep
        catalog = ep.catalog
        key = catalog.kind_key()
        cart = ep.basket.cart_ids()
        for cand in better:
            if cand in cart or not catalog.buyable(cand) or cand not in catalog.full:
                continue
            kind = catalog.label_of(cand).lower()
            same = [i for i in cart if key and catalog.label_of(i).lower() == kind] or (cart if len(ep.items) == 1 else [])
            if ep.pick == 'cheapest' and same and all((catalog.price_of(cand) or 0) >= (catalog.price_of(i) or 0) for i in same):
                continue
            number = ep.kinds.owner_of(catalog.label_of(cand)) or (1 if len(ep.items) == 1 else 0)
            if not number:
                continue
            lines = {i: ep.basket.held_units(i) or 1 for i in cart if i not in same}
            lines[cand] = ep.items[number - 1].qty
            done = ep.basket.set_cart(lines, same)
            if done:
                ep.trace.append({'kind': 'swap_better', 'in': cand, 'out': same})
                return 'Best swapped the cart to the better listing: ' + '; '.join(done)
        return ''

    def others(self, skip: list[str], limit: int = 30) -> str:
        ep = self.ep
        order = list(reversed(ep.catalog.order))
        if ep.pick == 'cheapest':
            order.sort(key=lambda i: (i not in ep.catalog.full, ep.catalog.price_of(i) if ep.catalog.price_of(i) is not None else float('inf')))
        lines = []
        for ident in order:
            if ident in skip or ep.catalog.sold_out(ident):
                continue
            lines.append(ep.catalog.row_text(ident, 600))
            if len(lines) >= limit:
                break
        return '\n'.join(lines)

    def reread_offers(self) -> str:
        ep = self.ep
        tool = ep.offer_tool()
        if not tool or not ep.box.one_target(tool):
            return ''
        since = max(ep.store.step - 2, ep.refused_at)
        stale = [i for i in ep.basket.cart_ids() if ep.offer_seen.get(i, -99) < since and i not in ep.catalog.gone]
        if not stale or ep.steps_left() < len(stale) + 3:
            return ''
        lines = []
        for ident in stale:
            before = dict(ep.catalog.rows.get(ident) or {})
            reply = ep.opener.open_view(tool, ident)
            if reply is None or ep.done():
                break
            ep.offer_seen[ident] = ep.store.step
            after = ep.catalog.rows.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {compact(shorten(reply.get('observation'), 300))[:900]}")
        ep.trace.append({'kind': 'reread', 'ids': stale, 'changed': len(lines)})
        if lines:
            ep.bump()
        return '\n'.join(lines)

    def swap_hint(self) -> str:
        ep = self.ep
        lost = [i for i in ep.basket.cart_ids() if i not in ep.catalog.gone and ep.catalog.sold_out(i)]
        if not lost:
            return ''
        for ident in lost:
            ep.catalog.gone[ident] = ep.store.step
        ep.leads = [i for i in ep.leads if i not in lost]
        ep.bump()
        if ep.pick == 'cheapest' and ep.room() >= 8:
            ep.ladder_done = {n: False for n in ep.ladder_done}
        picks = self.replacements(lost)
        text = f"[Best] Sold out now: {', '.join(lost)}. It will not come back, so do not open it again."
        if picks and ep.steps_left() <= 6:
            return text + f" Few steps are left: add {picks[0]} at the requested quantity and remove the sold-out line (two calls), then check out. Other read listings that appear to fit: {', '.join(picks[1:]) or 'none'}."
        if picks and ep.pick == 'cheapest':
            return text + f" Read listings that appear to fit instead, cheapest first: {', '.join(picks)}. The pick is the cheapest: unless you have seen the whole fitting set, run one search under {picks[0]}'s price before swapping in the cheapest fitting listing."
        if picks:
            return text + f" Read listings that appear to fit instead: {', '.join(picks)}. Take the sold-out line out and add {picks[0]} at the requested quantity, then check out."
        return text + ' Remove that line and add the best other listing that fits; search for one if none you have read fits.'

    def find_sold(self) -> str:
        ep = self.ep
        ids = [i for i in ep.basket.cart_ids() if i not in ep.catalog.gone]
        if any(ep.catalog.sold_out(i) for i in ids):
            return ''
        if len(ids) == 1:
            ep.catalog.unavailable.add(ids[0])
            return ''
        return self.reread_offers() if ids else ''

    def replacements(self, lost: list[str]) -> list[str]:
        ep = self.ep
        judges = [FAST, SLOW] if ep.brain.slow_ok() else ([FAST] if ep.brain.cost <= SPEND_HARD + ENDING_ROOM else [])
        if not judges:
            return []
        records = []
        for ident in reversed(ep.catalog.order):
            if ident in lost or ep.catalog.sold_out(ident):
                continue
            records.append(ep.catalog.row_text(ident, 600))
            if len(records) >= 14:
                break
        if not records:
            return []
        prompt = SWAP + '\n\nREQUEST:\n' + ep.request + '\n\nREQUIREMENTS IN FORCE:\n' + ep.ledger()
        prompt += '\n\nSOLD-OUT CART LINE:\n' + '\n'.join(f'{i}: {ep.catalog.record_text(i)}' for i in lost) + '\n\nCART NOW:\n' + ep.basket.cart_text()
        prompt += '\n\nOTHER LISTING RECORDS SEEN (newest first):\n' + '\n'.join(records)
        got = ep.brain.json(prompt, judges, 1200, tag='swap')
        fits: list[tuple[float, str]] = []
        cart = ep.basket.cart_ids()
        for entry in (got or {}).get('fits') or [] if isinstance((got or {}).get('fits'), list) else []:
            ident = str(entry.get('id') if isinstance(entry, dict) else entry)
            price = entry.get('price') if isinstance(entry, dict) else None
            if ident in ep.catalog.rows and ident not in lost and not ep.catalog.sold_out(ident) and ident not in cart and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float('inf'), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if ep.pick == 'cheapest' else [i for _, i in fits]
        ep.trace.append({'kind': 'swap', 'lost': lost, 'fits': fits})
        ep.swap_fits = ranked[:4]
        return ranked[:4]

    # -- a decline, and what to do instead ---------------------------------------------------------------
    def audit_call(self) -> dict[str, Any] | None:
        ep = self.ep
        catalog = ep.catalog
        cap = ep.budget if isinstance(ep.budget, (int, float)) else None
        key = catalog.kind_key()
        opened: list[str] = []
        rows: list[str] = []
        for strict in (True, False):
            for ident in reversed(catalog.order):
                label = str((catalog.rows.get(ident) or {}).get(key) or '') if key else ''
                pricey = cap is not None and not ep.budget_each and (catalog.price_of(ident) or 0) > cap
                if catalog.sold_out(ident) or (strict and (pricey or (label and not ep.kinds.owner_of(label)))):
                    continue
                if ident in catalog.full and len(opened) < 10:
                    opened.append('OPENED ' + catalog.record_text(ident, 700))
                elif ident not in catalog.full and len(rows) < 6:
                    rows.append('ROW ' + catalog.row_text(ident, 300))
            if opened or rows:
                break
        if not opened and not rows:
            return None
        prompt = AUDIT + '\n\nREQUEST:\n' + ep.request + '\n\nREQUIREMENTS IN FORCE:\n' + ep.ledger()
        answers = [line for line in ep.record if 'message to the shopper' in line]
        if answers:
            prompt += '\n\nANSWERS FROM THE SHOPPER:\n' + '\n'.join(answers[-6:])
        tried = list(dict.fromkeys(str(v) for args in ep.searched for v in args.values() if isinstance(v, str)))
        if tried:
            prompt += '\n\nSEARCHES ALREADY TRIED:\n' + ' | '.join(tried)
        prompt += '\n\nLISTING RECORDS SEEN (newest first):\n' + '\n'.join(opened + rows)
        judges = ep.brain.judges() or ([FAST] if ep.brain.cost <= SPEND_HARD + ENDING_ROOM else [])
        got = ep.brain.json(prompt, judges, 1800, tag='audit') if judges else None
        ep.trace.append({'kind': 'audit', 'got': shorten(got, 200)})
        return got

    def take_review(self, got: dict[str, Any] | None) -> None:
        ep = self.ep
        changed = []
        for entry in (got or {}).get('corrected') or [] if isinstance((got or {}).get('corrected'), list) else []:
            if not isinstance(entry, dict):
                continue
            number, text, value = whole(entry.get('item')) or 0, str(entry.get('need') or '').strip(), str(entry.get('value') or '').strip()
            if 1 <= number <= len(ep.items) and ':' in value:
                rule = next((r for r in ep.items[number - 1].active() if r.text == text and not r.points), None)
                if rule:
                    rule.dropped = True
                    ep.items[number - 1].require(value, rule.source)
                    changed.append(f'{text} read as {value}')
        if changed:
            ep.bump()
            ep.trace.append({'kind': 'review', 'changed': changed})
            ep.notes.append('a review of the request: ' + '; '.join(changed))

    def ids_by_item(self, got: dict[str, Any] | None, field: str) -> dict[str, list[str]]:
        catalog = self.ep.catalog
        table = (got or {}).get(field)
        if not isinstance(table, dict):
            return {}
        out = {}
        for k, v in table.items():
            ids = [str(i) for i in (v if isinstance(v, list) else [v]) if str(i) in catalog.rows and not catalog.sold_out(str(i))][:4]
            if ids:
                out[str(k)] = ids
        return out

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        ep = self.ep
        keep = ep.tail() + len(ep.items) + 2
        room = min(RESCUE_STEPS, ep.steps_left() - keep)
        if room < 3 or ep.time_left() < WRAP_UP_AT + 25:
            return False
        before = len(ep.catalog.full)
        last = ep.store.step + room
        searching = ep.store.step + max(1, room // 2)
        plan = ep.interviewer.search_template()
        queries = [(min(max(1, whole(e.get('item')) or 1), len(ep.items)), str(e['query']).strip()) for e in (got or {}).get('search') or [] if isinstance(e, dict) and str(e.get('query') or '').strip()] if isinstance((got or {}).get('search'), list) else []
        queries = [q for q in queries if q[0] in missing] or [(n, ' '.join(ep.interviewer.value_words(n)[:2] + [ep.items[n - 1].name])) for n in missing]
        found: dict[int, list[str]] = {n: [] for n in missing}
        for number in missing[:3]:
            if ep.store.step < searching:
                cap = ep.budget if isinstance(ep.budget, (int, float)) and ep.box.browse and ep.box.browse['price'] and not ep.budget_each else None
                found[number] += ep.opener.list_category(number, cap) or []
        for number, words in queries[:4]:
            if plan and ep.store.step < searching:
                found.setdefault(number, [])
                found[number] += ep.interviewer.search_words(plan[0], plan[1], plan[2], words)
        for number, rows in found.items():
            picks = [i for i in ep.interviewer.kind_rows(number, rows) if i not in ep.catalog.full][:2]
            ep.opener.fill_facts(picks[:max(0, last - ep.store.step)], keep)
        ep.trace.append({'kind': 'recovery', 'missing': missing, 'queries': queries[:4], 'opened': len(ep.catalog.full) - before})
        return len(ep.catalog.full) > before

    def open_rows(self, rows: list[str]) -> bool:
        ep = self.ep
        tool = ep.facts_tool()
        if not tool or not ep.box.one_target(tool) or ep.steps_left() < 5:
            return False
        before = len(ep.catalog.full)
        ep.opener.fill_facts(rows[:3], ep.tail() + 1)
        return len(ep.catalog.full) > before

    def audit(self) -> str:
        """Before a decline stands: review the evidence, search once more for what is missing, name candidates.
        Returns advice for the model when a decline is not justified, else ''."""
        ep = self.ep
        if ep.audited >= 2 or not (ep.brain.judges() or [FAST]):
            return ''
        ep.audited += 1
        got = self.audit_call()
        if got is None:
            return ''
        ep.audit_ok += 1
        self.take_review(got)
        fits = self.ids_by_item(got, 'fits')
        maybe = self.ids_by_item(got, 'maybe')
        missing = [n for n in range(1, len(ep.items) + 1) if not fits.get(str(n)) and not maybe.get(str(n))]
        if missing and ep.audited == 1:
            rows = [str(i) for i in (got or {}).get('open') or [] if str(i) in ep.catalog.rows and str(i) not in ep.catalog.full and not ep.catalog.sold_out(str(i))] if isinstance((got or {}).get('open'), list) else []
            opened = bool(rows) and self.open_rows(rows)
            if self.recovery_search(got, missing) or opened:
                again = self.audit_call()
                if again is not None:
                    got = again
                    self.take_review(got)
                    fits = self.ids_by_item(got, 'fits') or fits
                    maybe = self.ids_by_item(got, 'maybe') or maybe
                    missing = [n for n in range(1, len(ep.items) + 1) if not fits.get(str(n)) and not maybe.get(str(n))]
        ep.blocking = [(whole(e.get('item')) or 0, str(e.get('need') or '')) for e in got.get('blocking') or [] if isinstance(e, dict)] if isinstance(got.get('blocking'), list) else []
        named = {**maybe, **fits}
        if named or not ep.lead_items:
            ep.lead_items = named
            ep.leads = [i for ids in named.values() for i in ids]
        if not named:
            return ''
        if missing:
            return ('A careful review found candidates for some items: ' + '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items()) +
                    f". Items with no candidate yet: {', '.join(ep.items[n - 1].name for n in missing)}. Search those once more (their category listing, another name for the product) before any ending; a decline is only right when a requirement the shopper stated outright has no buyable listing.")
        return 'A careful review thinks these listings may fit: ' + '; '.join(f"item {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got and got.get('reason') else '') + '. Open them if needed, cart the best fitting one per item and check out.'

    def decline_allowed(self) -> bool:
        """A decline stands only when what blocks every candidate was stated outright by the shopper (in the
        request or an answer), is not a chain or a store note, and the store was actually searched for the item."""
        ep = self.ep
        if any(rule.points and not rule.resolved for _, rule in ep.needs()):
            return False
        if ep.blocking:
            for number, text in ep.blocking:
                if not 1 <= number <= len(ep.items):
                    return False
                item = ep.items[number - 1]
                rule = next((r for r in item.active() if r.text == text or Words.alike(r.prop, text.partition(':')[0])), None)
                if rule is None or rule.points or rule.source not in ('request', 'shopper'):
                    return False
                if not ep.searched and not ep.kinds.item_ids(number):
                    return False
            return True
        return ep.audit_ok >= 2 and not ep.lead_items and bool(ep.searched) and all(r.source in ('request', 'shopper') and not r.points for _, r in ep.needs())

    def code_pick(self, numbers: list[int] | None = None) -> dict[int, str]:
        """Without a judge: per item, the read, buyable, in-budget listing of its kind whose code-graded evidence is
        best (no conflict or failure on a stated requirement; most requirements stated outright), cheapest first."""
        ep = self.ep
        catalog = ep.catalog
        rank = {'strong': 0, 'weak': 1, 'mixed': 3, 'unknown': 9, 'conflict': 9, 'fails': 9}
        out: dict[int, str] = {}
        for number in numbers or range(1, len(ep.items) + 1):
            item = ep.items[number - 1]
            best: tuple[tuple[int, int, float], str] | None = None
            for ident in ep.kinds.item_ids(number):
                if ident not in catalog.full or not catalog.buyable(ident):
                    continue
                price = catalog.price_of(ident)
                if isinstance(ep.budget, (int, float)) and price is not None and (price if ep.budget_each else price * item.qty) > ep.budget:
                    continue
                grades = ep.basket.grades_of(ident, number)
                if any(rank[g] >= 9 for g in grades.values()):
                    continue
                score = (sum(rank[g] for g in grades.values()), -len(grades), price if price is not None else float('inf'))
                if best is None or score < best[0]:
                    best = (score, ident)
            if best:
                out[number] = best[1]
        if out:
            ep.trace.append({'kind': 'code_pick', 'choice': out})
        return out

    def evidenced(self, ident: str, number: int) -> bool:
        """Whether a read listing states every plain-valued requirement of the item somewhere (an unread row passes)."""
        ep = self.ep
        if ident not in ep.catalog.full:
            return True
        return all(g in ('strong', 'weak', 'mixed') for g in ep.basket.grades_of(ident, number).values())

    def forced_pick(self) -> dict[int, str]:
        """When no verified cart exists and the episode must end: the most likely listing per item, by a judge; a
        read listing that states nothing for a stated requirement is never taken."""
        ep = self.ep
        catalog = ep.catalog
        judges = ep.brain.judges(SPEND_HARD) or ([FAST] if ep.brain.cost <= SPEND_HARD + ENDING_ROOM else [])
        key = catalog.kind_key()
        records = []
        for ident in reversed(catalog.order):
            if catalog.sold_out(ident) or (key and not ep.kinds.owner_of(catalog.label_of(ident))) or not catalog.buyable(ident):
                continue
            records.append(('OPENED ' if ident in catalog.full else 'ROW ') + catalog.row_text(ident, 700))
            if len(records) >= 24:
                break
        out: dict[int, str] = {}
        if not records or not judges:
            for number, ids in ep.lead_items.items():
                n = whole(number)
                pick = next((i for i in ids if not catalog.sold_out(i) and catalog.buyable(i) and self.evidenced(i, n)), None) if n else None
                if n and pick:
                    out[n] = pick
            if not out and len(ep.items) == 1:
                pick = next((i for i in ep.swap_fits if not catalog.sold_out(i) and catalog.buyable(i) and self.evidenced(i, 1)), None)
                if pick:
                    out[1] = pick
            ep.trace.append({'kind': 'forced_pick', 'choice': out, 'reason': 'leads only (no judge available)'})
            return out or self.code_pick()
        prompt = PICKER + '\n\nREQUEST:\n' + ep.request + '\n\nREQUIREMENTS IN FORCE:\n' + ep.ledger() + '\n\nRECORDS (newest first):\n' + '\n'.join(records)
        proof = ep.observer.evidence_lines([i for i in reversed(catalog.order) if i in catalog.full and catalog.buyable(i)][:12])
        if proof:
            prompt += '\n\n' + proof
        got = ep.brain.json(prompt, judges, 800, tag='pick') or {}
        choice = got.get('choice') if isinstance(got.get('choice'), dict) else {}
        rejected = []
        for k, v in choice.items():
            number = whole(k)
            ident = str(v or '').strip()
            if number and 1 <= number <= len(ep.items) and ident in catalog.rows and not catalog.sold_out(ident):
                if self.evidenced(ident, number):
                    out[number] = ident
                else:
                    rejected.append(ident)
        for number, ids in ep.lead_items.items():
            n = whole(number)
            if n and n not in out:
                pick = next((i for i in ids if not catalog.sold_out(i) and catalog.buyable(i) and self.evidenced(i, n)), None)
                if pick:
                    out[n] = pick
        if not out and len(ep.items) == 1:
            pick = next((i for i in ep.swap_fits if not catalog.sold_out(i) and catalog.buyable(i) and self.evidenced(i, 1)), None)
            if pick:
                out[1] = pick
        ep.trace.append({'kind': 'forced_pick', 'choice': out, 'rejected': rejected, 'reason': str(got.get('reason') or '')[:160]})
        return out or ({} if ep.blocking else self.code_pick())

    def buy_best(self, picks: dict[int, str]) -> str:
        """Set the cart to the picks (keeping fitting lines of other items) and check out. Returns the store's
        reply text, or '' when nothing could be done."""
        ep = self.ep
        catalog = ep.catalog
        cart, buy = ep.box.tool('cart'), ep.box.tool('buy')
        if not cart or not buy or not picks or ep.done():
            return ''
        key = catalog.kind_key()
        now = ep.basket.cart_ids()
        lines = {ident: ep.items[number - 1].qty for number, ident in picks.items()}
        covered = {catalog.label_of(i).lower() for i in lines} if key else set()
        keep = {i: ep.basket.held_units(i) or 1 for i in now if i not in lines and not catalog.sold_out(i) and (not key or catalog.label_of(i).lower() not in covered) and ep.kinds.owner_of(catalog.label_of(i))}
        drop = [i for i in now if i not in lines and i not in keep]
        done = ep.basket.set_cart({**keep, **lines}, drop)
        if done is None:
            return ''
        ep.opener.complete_views(list(lines), ep.tail())
        args, faults = ep.box.shape(buy, dict(ep.box.buy_args))
        if faults or ep.steps_left() < 1:
            return ''
        self.gate('order')
        if ep.done() or ep.steps_left() < 1:
            return ''
        reply = ep.store.call(buy, args)
        ep.trace.append({'kind': 'buy_best', 'lines': lines, 'reply': shorten(reply, 200)})
        if reply is None:
            return ''
        ep.buys += 1
        if not reply.get('finished'):
            ep.refused_at = ep.store.step
            ep.observer.observe(buy, reply.get('observation'))
            if ep.note_sold:
                self.find_sold()
            ep.note_sold = False
        return 'Best set the cart (' + '; '.join(done) + ') and checked out. Store reply: ' + ep.observer.show(reply.get('observation'))

    # -- steps the request asks for ----------------------------------------------------------------------
    def gate(self, stage: str) -> tuple[bool, str]:
        ep = self.ep
        steps = [p for p in ep.process if p['when'] == stage]
        if not steps or ep.done() or not ep.gate_due(stage):
            return True, ''
        tools = ep.box.lines(ep.box.endings())
        said: list[str] = []
        nudged = False
        for _ in range(3):
            prompt = (GATE.replace('STAGE', 'place the order' if stage == 'order' else 'end without a purchase') + '\n\nREQUEST:\n' + ep.request +
                      '\n\nSTEPS DUE NOW:\n' + '\n'.join(f"- {p['step']}" for p in steps) + '\n\nLEDGER:\n' + ep.ledger() + '\n\nCART NOW: ' + ep.basket.cart_text() +
                      '\n\nRECORD (messages and step calls so far):\n' + ('\n'.join(f'{i}. {line}' for i, line in enumerate(ep.record, 1)) or '(nothing yet)') +
                      ('\n\nSTORE NOTES SEEN:\n' + '\n'.join(ep.notes[-4:]) if ep.notes else '') +
                      f"\n\nMESSAGES LEFT: {max(0, ep.msg_limit - ep.msg_used)}   STEPS LEFT: {ep.steps_left()} (the {'order' if stage == 'order' else 'ending'} itself takes one)" + '\n\nTOOLS:\n' + tools)
            got = ep.brain.json(prompt, [SLOW, FAST] if nudged else [FAST, SLOW], 2500, tag='gate')
            if got is None:
                break
            told: list[str] = []
            hold = ''
            if said:
                ep.interviewer.take_heard({'needs': got.get('needs')}, told)
                hold = str(got.get('hold') or '').strip()
            call = got.get('call') if isinstance(got.get('call'), dict) else None
            ep.trace.append({'kind': 'gate', 'stage': stage, 'call': shorten(call, 200), 'steps': shorten(got.get('steps'), 120), 'hold': hold[:200], 'told': told})
            if told or hold:
                ep.bump()
                return False, ('Before going on, Best did the steps the request asks for:\n' + '\n'.join(said) + '\n' if said else '') + "The shopper's reply changes the task: " + '; '.join(told + ([hold] if hold else [])) + '. Fix the cart for it, then check out again.'
            name = str((call or {}).get('tool') or '')
            shown = [e for e in got.get('steps') or [] if isinstance(e, dict)] if isinstance(got.get('steps'), list) else []
            talk = stage == 'order' and any(p['messages'] > 0 for p in steps)
            proof = {i for i in range(1, len(ep.record) + 1) if not talk or (i <= len(ep.record_steps) and ep.record_steps[i - 1] > ep.cart_at)}
            unproven = [e for e in shown if e.get('done') is not True or not any((whole(n) or 0) in proof for n in (e.get('lines') if isinstance(e.get('lines'), list) else []))]
            if not call and unproven and not said and not nudged and ep.box.tool('ask') and ep.msg_used < ep.msg_limit and ep.steps_left() >= 2:
                nudged = True
                ep.record.append(f'(Best: no line made since the cart last changed at step {ep.cart_at} shows ' + '; '.join(str(e.get('step') or '') for e in unproven)[:200] + f' done, and {ep.msg_limit - ep.msg_used} message(s) and {ep.steps_left()} steps are left: give the call that does it)')
                ep.record_steps.append(-1)
                continue
            if not call or name not in ep.box.specs or name in ep.box.endings():
                if not call:
                    ep.gate_ok.add((stage, ep.cart_mark()))
                break
            if ep.steps_left() < 2 or (name == ep.box.tool('ask') and ep.msg_used >= ep.msg_limit):
                break
            args, faults = ep.box.shape(name, call.get('args') if isinstance(call.get('args'), dict) else {})
            if faults:
                break
            if name == ep.box.tool('ask'):
                text = next((v for v in args.values() if isinstance(v, str)), '')
                batch = ep.interviewer.send([(1, text)], 1)
                if not batch:
                    break
                ep.gate_msgs += 1
                said.append(f'Q: {text}\nA: {batch[0][2]}')
                continue
            reply = ep.store.call(name, args)
            if reply is None:
                break
            seen = reply.get('observation')
            ep.catalog.absorb(seen)
            ep.record.append(f'step {ep.store.step}: {name} {compact(args)} -> {ep.observer.show(seen)[:300]} (cart then: {ep.cart_mark()})')
            ep.record_steps.append(ep.store.step)
            said.append(f'{name} {compact(args)} -> {ep.observer.show(seen)[:400]}')
        return True, 'Before going on, Best did the steps the request asks for:\n' + '\n'.join(said) if said else ''


# ------------------------------------------------------------------------------------------------------------
# the driver: the model proposes store calls, the code runs the procedure around them
# ------------------------------------------------------------------------------------------------------------
class Driver:
    def __init__(self, ep: Episode):
        self.ep = ep

    def is_buy(self, name: str, args: dict[str, Any]) -> bool:
        box = self.ep.box
        return name == box.tool('buy') and all(args.get(k) == v for k, v in box.buy_args.items())

    def is_refuse(self, name: str, args: dict[str, Any]) -> bool:
        box = self.ep.box
        return name == box.tool('refuse') and all(args.get(k) == v for k, v in box.refuse_args.items())

    def offered(self) -> list[dict[str, Any]]:
        """The tools the model may call now: never the messaging tool, and only the ending tools at the very end."""
        ep = self.ep
        left = ep.steps_left()
        ends = ep.box.endings()
        keep: set[str] | None = None
        if ends:
            if left <= ep.tail():
                keep = set(ends) | ({ep.box.tool('cart')} if not ep.basket.cart_ids() or self.items_without_lines() else set())
            elif left <= ep.tail() + 3:
                keep = ends | {ep.box.tool('cart'), ep.box.tool('cart_view')}
            elif left <= len(ep.items) + 5 and ep.buys == 0:
                keep = ends | {ep.box.tool('cart'), ep.box.tool('cart_view'), ep.box.tool('offer')}
        return ep.box.offered({k for k in keep if k} if keep else None)

    def quiet(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A decline's remark: short and naming no listing, unless the request wants alternatives suggested."""
        ep = self.ep
        out = dict(args)
        for key, spec in ep.box.props(name).items():
            if not isinstance(spec, dict) or 'string' not in Schema.kinds(spec) or 'enum' in spec:
                continue
            value = out.get(key)
            if value is None and key not in ep.box.required(name):
                continue
            text = str(value or '')
            names = any(ident in text for ident in ep.catalog.rows)
            if names or not text or ep.quiet_decline is not False:
                out[key] = 'Sorry, nothing in the store meets all of your requirements, so I am not buying anything.'[:spec.get('maxLength', 200)]
        return out

    def resolve_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Titles passed where a listing id belongs become the id, when exactly one listing carries that title."""
        ep = self.ep
        box, catalog = ep.box, ep.catalog
        if name not in {box.tool(r) for r in ('cart', 'open', 'offer', 'compare')} - {None} or not catalog.id_key:
            return args, ''
        free = [k for k, v in box.props(name).items() if isinstance(v, dict) and ('string' in Schema.kinds(v) or 'array' in Schema.kinds(v)) and 'enum' not in v]
        if len(free) != 1 or (name != box.tool('cart') and (name, free[0]) not in box.id_slots):
            return args, ''

        def fix(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in catalog.rows:
                return value.strip(), ''
            if not isinstance(value, str) or IdSense.is_id(value, True) or len(value.strip()) < 8:
                return value, ''
            low = value.strip().lower()
            hits = [i for i, row in catalog.rows.items() if any(k != catalog.id_key and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ''
            if hits:
                return value, f"several listings carry the text {value!r} ({', '.join(hits[:6])}); pass one listing id"
            return value, f'{value!r} is not a listing id; pass the id from the result rows (their {catalog.id_key} field)'
        out = dict(args)
        key = free[0]
        if key not in args:
            return args, ''
        fixed = [fix(v) for v in args[key]] if isinstance(args[key], list) else [fix(args[key])]
        problem = next((p for _, p in fixed if p), '')
        if problem:
            return args, problem
        out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ''

    def compare_room(self) -> int:
        ep = self.ep
        tool = ep.box.tool('compare')
        spec = ep.box.props(tool).get(ep.box.many_target(tool) or '') or {}
        if isinstance(spec.get('maxItems'), int):
            return spec['maxItems']
        if ep.compare_bad:
            return max(2, min(ep.compare_bad - 1, max(ep.compare_ok, 2)))
        return max(4, ep.compare_ok)

    def join_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-listing opens in one turn become one side-by-side call where the store has one."""
        ep = self.ep
        opener, joiner = ep.box.tool('open'), ep.box.tool('compare')
        target, many = ep.box.one_target(opener), ep.box.many_target(joiner)
        if not opener or not joiner or not target or not many or opener == joiner or ep.facts_from.get(joiner) is False:
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get('function') or {}
            args = first_json(fn.get('arguments')) if isinstance(fn.get('arguments'), str) else None
            ident = str((args or {}).get(target) or '').strip()
            if fn.get('name') == opener and args and set(args) == {target} and ident in ep.catalog.rows and ident not in ep.catalog.gone:
                picks.append((index, ident))
        picks = picks[:self.compare_room()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        joined = {i for i, _ in picks}
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({'id': call['id'], 'type': 'function', 'function': {'name': joiner, 'arguments': compact({many: [i for _, i in picks]})}})
            elif index in joined:
                out.append({'id': call['id'], 'joined': True})
            else:
                out.append(call)
        ep.trace.append({'kind': 'joined', 'ids': [i for _, i in picks]})
        return out

    # -- before an ending ---------------------------------------------------------------------------------
    def before_buy(self, cid: str) -> tuple[str, str] | None:
        """Everything Best does or asks before a checkout goes out; a (cid, text) reply holds the call back."""
        ep = self.ep
        basket, judge, opener = ep.basket, ep.judge, ep.opener
        cart_text = '\nCart now: ' + basket.cart_text()
        missing = self.items_without_lines()
        if missing and ep.line_holds < 1 and ep.room() >= 2 * len(missing) + 1 and ep.time_left() > WRAP_UP_AT:
            ep.line_holds += 1
            names = ', '.join(ep.items[n - 1].name for n in missing)
            ep.trace.append({'kind': 'line_hold', 'missing': missing})
            return cid, f'Not run yet (no step used). The cart has no line for: {names}. Cart the best fitting listing for each of them first (search its category if none was read), then check out.' + cart_text
        wider = opener.wider_hold()
        if wider:
            return cid, wider + cart_text
        reread = judge.reread_offers()
        if ep.note_sold:
            judge.find_sold()
        viewer = ep.box.tool('cart_view')
        if ep.cart_touched and not basket.cart_fresh() and viewer and ep.steps_left() > 3:
            view_args, faults = ep.box.shape(viewer, {})
            view = ep.store.call(viewer, view_args) if not faults else None
            if view is not None:
                ep.cart_view = view.get('observation')
                ep.cart_seen_at = ep.store.step
                change = ep.observer.observe(viewer, ep.cart_view)
                if change:
                    ep.queued = []
                    return cid, 'Not run yet (no step used). Looking at the cart showed a change: ' + change + '\nCart now: ' + basket.cart_text() + '\nApply the change, fix the cart if needed, then check out.'
        swap = judge.swap_hint()
        if swap:
            return cid, 'Not run yet (no step used). ' + swap + cart_text
        if any(ep.catalog.sold_out(i) for i in basket.cart_ids()) and ep.steps_left() >= 3:
            # the model is checking out with a line it was told is gone: take it out by code and ask for a replacement
            cleaned = judge.drop_sold_lines()
            if cleaned:
                return cid, 'Not run yet. ' + cleaned + '. Add a replacement line for that item (the read listings that appear to fit were named above), then check out.' + '\nCart now: ' + basket.cart_text()
        chain = ep.chainer.resolve()
        pending = [rule.text for _, rule in ep.needs() if rule.points and not rule.resolved]
        if pending and ep.chain_blocks < 1 and ep.room() >= 4 and ep.time_left() > WRAP_UP_AT:
            ep.chain_blocks += 1
            return cid, ('Not run yet (no step used). ' + (chain + '\n' if chain else '') + 'These requirements take their value from another listing and that listing was not found yet: ' + ' | '.join(pending[:3]) +
                         ". Search for it by its shop or brand name together with its kind, open its page, and then check out; Best will read the value from the pages." + cart_text)
        misfit = basket.fit_faults() if ep.fit_holds < 2 and ep.steps_left() > ep.tail() + 3 else []
        if misfit:
            ep.fit_holds += 1
            ep.trace.append({'kind': 'fit_hold', 'faults': misfit[:4]})
            return cid, 'Not run yet (no step used). A part in the cart does not work with the device in the cart: ' + '; '.join(misfit[:3]) + '. Swap the part (or the device), then check out.' + cart_text
        faults = basket.line_faults() if ep.line_holds < 3 and ep.steps_left() > ep.tail() + 2 else []
        if faults and all(' is held at ' in f for f in faults):
            fixed = basket.fix_counts(basket.count_fixes())
            faults = basket.line_faults() if fixed else faults
            if fixed and not faults:
                ep.line_holds += 1
                return cid, "Not run yet (no step used). Best set the cart's quantities itself: " + '; '.join(fixed) + '. If the store refused any of these, fix those lines; then check out.' + cart_text
        if faults:
            ep.line_holds += 1
            ep.trace.append({'kind': 'line_hold', 'faults': faults[:6]})
            return cid, 'Not run yet (no step used). The cart does not match the request yet: ' + '; '.join(faults[:6]) + ". Fix those lines (the cart tool's own way to add or take off units), then check out." + cart_text
        read = opener.fill_facts(basket.cart_ids(), ep.tail() + 3) + opener.cheaper_rows()
        viewed = opener.complete_views(basket.cart_ids(), ep.tail() + 1)
        doubts = basket.doubtful_lines() if ep.doubt_opens < 1 and ep.room() >= 5 and ep.time_left() > WRAP_UP_AT + 20 else []
        if doubts:
            # the cart's evidence is shaky and no solid rival was read: read a few more rows of that kind first
            ep.doubt_opens += 1
            wanted = {w for _, n, prop in doubts for r in ep.items[n - 1].active() if Words.alike(r.prop, prop) for w in Words.content(r.resolved or r.value)}
            pool = []
            for _, number, _ in doubts:
                for ident in ep.kinds.item_ids(number):
                    if ident in ep.catalog.full or not ep.catalog.buyable(ident) or ident in pool:
                        continue
                    price = ep.catalog.price_of(ident)
                    if ep.pick == 'cheapest' and price is not None and all((ep.catalog.price_of(c) or 0) < price for c, n, _ in doubts if n == number):
                        continue
                    pool.append(ident)
            pool.sort(key=lambda i: (-len(wanted & Words.stems(ep.catalog.title_of(i))), ep.catalog.price_of(i) if ep.catalog.price_of(i) is not None else float('inf')))
            opened = opener.fill_facts(pool[:3], ep.tail() + 2)
            if opened:
                read += opened
                ep.trace.append({'kind': 'doubt_opens', 'doubts': [d[0] for d in doubts][:3], 'opened': len(opened)})
        shaky, stronger = basket.proof_faults() if ep.proof_holds < 2 and ep.steps_left() > ep.tail() + 2 else ([], [])
        if shaky:
            ep.proof_holds += 1
            ep.trace.append({'kind': 'proof_hold', 'faults': shaky[:4], 'better': stronger[:4]})
            swapped = judge.swap_to_better(stronger) if stronger and ep.swaps < 2 and ep.room() >= 3 else ''
            if swapped:
                ep.swaps += 1
                opener.complete_views(basket.cart_ids(), ep.tail() + 1)
                verdict = judge.check_cart()
                if not verdict:
                    return None
                return cid, 'Not run yet. ' + swapped + '\n' + verdict + cart_text
            return cid, ("Not run yet (no step used). The cart's evidence is not solid: " + '; '.join(shaky[:3]) +
                         ". A listing whose own key states a different value for a required property does not count as fitting; one whose keys disagree is doubtful. Swap in a listing whose keys state every requirement outright and agree (search by the requirement's value if none was read), then check out." + cart_text)
        verdict = judge.check_cart()
        if verdict and ep.better and ep.swaps < 2 and ep.room() >= 3:
            swapped = judge.swap_to_better(ep.better)
            if swapped:
                ep.swaps += 1
                opener.complete_views(basket.cart_ids(), ep.tail() + 1)
                again = judge.check_cart()
                if not again:
                    return None
                return cid, 'Not run yet. ' + swapped + '\n' + again + cart_text
        if verdict and ep.view_holds < 2:
            ep.view_holds += 1
            seen = "Best read these listings' views for the check (one step each):\n" + '\n'.join(read + viewed) + '\n' if read or viewed else ''
            return cid, 'Not run yet. ' + (chain + '\n' if chain else '') + seen + verdict + cart_text
        if reread and ep.brain.cost > SPEND_SOFT:
            return cid, "Not run yet (no step used). Best re-read the cart's listings just now:\n" + reread + '\nCheck out again when every line still shows stock and a price inside the budget; otherwise replace that line first.'
        return None

    def items_without_lines(self) -> list[int]:
        ep = self.ep
        key = ep.catalog.kind_key()
        if not key:
            return []
        ids = [i for i in ep.basket.cart_ids() if (ep.basket.held_units(i) or 0) > 0 and not ep.catalog.sold_out(i)]
        out = []
        for n, item in enumerate(ep.items, 1):
            known = any(ep.kinds.owns(item, str(r.get(key) or '')) for r in ep.catalog.rows.values())
            if known and not any(ep.kinds.owns(item, ep.catalog.label_of(i)) for i in ids):
                out.append(n)
        return out

    def before_refuse(self, cid: str) -> tuple[str, str] | None:
        """A decline is handed back unless the evidence truly supports it; otherwise Best buys the best candidates."""
        ep = self.ep
        judge = ep.judge
        if ep.steps_left() <= 2 or ep.time_left() < CLOCK_FLOOR + 20:
            return None
        if not ep.pushed and not ep.leads and ep.steps_left() > ep.max_steps // 3 and ep.time_left() > WRAP_UP_AT + 30:
            ep.pushed = True
            return cid, f"Not run yet (no step used). {ep.steps_left()} steps are still left: search in ways not tried yet (each item's category listing paged on, one-word value queries, the product's other names) and open the listings that might fit. End without a purchase only after that."
        review = judge.audit() if ep.steps_left() > 5 and (ep.audited == 0 or not ep.leads) else ''
        if review and ep.refuse_blocks == 0:
            ep.refuse_blocks = 1
            return cid, 'Not run yet (no step used). ' + review
        if judge.decline_allowed():
            ep.trace.append({'kind': 'decline_allowed', 'blocking': ep.blocking})
            return None
        picks = judge.forced_pick()
        if picks and ep.steps_left() >= len(picks) + 2:
            done = judge.buy_best(picks)
            if done:
                return cid, 'Not run (Best acted instead): no requirement the shopper stated outright rules out every listing, so ' + done + (' [episode finished]' if ep.done() else '')
        if ep.refuse_blocks < 2:
            ep.refuse_blocks += 1
            return cid, 'Not run yet (no step used). Ending without a purchase earns nothing unless a requirement the shopper stated has no buyable listing; an unverified chain, a missing page or a shortage of steps is not that. Cart the most likely fitting listing for each item and check out.'
        return None

    # -- running the model's calls -----------------------------------------------------------------------
    def run_calls(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        ep = self.ep
        box, catalog = ep.box, ep.catalog
        out: list[tuple[str, str]] = []
        calls = self.join_opens(calls)
        for call in calls:
            allowed = {t['function']['name'] for t in self.offered()}
            cid = call['id']
            if call.get('joined'):
                out.append((cid, 'Shown in the side-by-side result above: Best opened these listings together in one step.'))
                continue
            fn = call.get('function') or {}
            name = fn.get('name')
            raw = fn.get('arguments')
            if ep.done():
                out.append((cid, 'Not run: the episode is over.'))
                continue
            if name == box.tool('ask'):
                out.append((cid, 'Not run (no step used): Best sends every message to the shopper itself.'))
                continue
            if name not in box.specs:
                out.append((cid, f"Not run (no step used): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, 'Not run (no step used): too few steps are left for that; use: ' + ', '.join(sorted(allowed)) + '.'))
                continue
            args = raw if isinstance(raw, dict) else (first_json(raw) if (raw or '').strip() else {})
            if args is None:
                out.append((cid, 'Not run (no step used): the arguments were not one JSON object.'))
                continue
            args, faults = box.shape(name, args)
            if faults:
                out.append((cid, 'Not run (no step used): ' + '; '.join(faults[:6])))
                continue
            args, wrong = self.resolve_ids(name, args)
            if wrong:
                out.append((cid, 'Not run (no step used): ' + wrong + '.'))
                continue
            trimmed = ''
            many = box.many_target(name) if name == box.tool('compare') else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.compare_room() and ep.compare_bad:
                room = self.compare_room()
                trimmed = f"\n[Best] Sent the first {room} listings only: this store shows at most {room} side by side. Not shown yet: {', '.join(map(str, args[many][room:]))}."
                args = {**args, many: args[many][:room]}
            target = box.one_target(name)
            if target and name in (box.tool('open'), box.tool('offer')) and str(args.get(target)) in catalog.gone:
                out.append((cid, f'Not run (no step used): {args[target]} sold out at step {catalog.gone[str(args[target])]} and will not come back. Replace its cart line with another fitting listing.'))
                continue
            if name == box.tool('cart') and any((whole(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                view = compact(ep.cart_view) if ep.basket.cart_fresh() else ''
                absent = [v for v in args.values() if isinstance(v, str) and v in catalog.rows and v not in ep.cart_lines and v not in view and v not in ep.cart_said]
                if absent:
                    out.append((cid, f'Not run (no step used): {absent[0]} is not in the cart, so there is nothing to take off. Cart now: ' + ep.basket.cart_text()))
                    continue
            buying, refusing = self.is_buy(name, args), self.is_refuse(name, args)
            ending = buying or refusing
            if ending and len(calls) > 1:
                out.append((cid, 'Not run (no step used): send the finishing call on its own, after reading your other results.'))
                continue
            empty_cart = not ep.basket.cart_ids() or bool(self.items_without_lines())
            adding = name == box.tool('cart') and not any((whole(v) or 0) < 0 for v in args.values() if not isinstance(v, str)) and not any(isinstance(v, str) and v == box.remove_option(name) for v in args.values())
            if not ending and ep.steps_left() <= ep.tail() and box.endings() and not (adding and empty_cart and ep.steps_left() >= 2):
                out.append((cid, 'Not run (no step used): the last steps are kept for checking out (or for the no-purchase ending).'))
                continue
            if name == box.tool('cart') and adding:
                gone = [v for v in args.values() if isinstance(v, str) and v in catalog.rows and catalog.sold_out(v)]
                if gone:
                    out.append((cid, f'Not run (no step used): {gone[0]} is sold out and will not come back; add another listing instead.' + ('\nRead listings that appear to fit: ' + ', '.join(i for i in ep.swap_fits if not catalog.sold_out(i)) if ep.swap_fits else '')))
                    continue
            if name == box.tool('cart') and not adding and ep.steps_left() <= ep.tail() + 4 and ep.buys == 0:
                leaving = [v for v in args.values() if isinstance(v, str) and v in ep.basket.cart_ids() and not catalog.sold_out(v)]
                kept = [i for i in leaving if ep.judge.evidenced(i, ep.kinds.owner_of(catalog.label_of(i)) or 1) and i in catalog.full]
                if kept:
                    out.append((cid, f"Not run (no step used): too few steps are left for a swap, and {kept[0]} states every plain requirement. Check out with the cart as it is."))
                    continue
            mark = name + compact(args)
            if not ending and name != box.tool('cart') and mark in ep.recent:
                out.append((cid, 'Not run (no step used): you just made this same call and nothing changed; its result is above.'))
                continue
            readers = box.readers()
            cached = ep.read_cache.get(mark)
            if name in readers and cached and cached[0] == catalog.version:
                out.append((cid, f'Not run (no step used): the same call at step {cached[1]} returned this, and nothing has changed since:\n' + cached[2][:3000]))
                continue
            if target and name in readers and name != box.tool('compare') and catalog.fresh_view(str(args.get(target)), name):
                out.append((cid, f'Not run (no step used): {args[target]} has shown everything this view gives since the last change. Its record: ' + catalog.record_text(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not catalog.fresh_view(str(i), name)]
                if not keep:
                    out.append((cid, 'Not run (no step used): these listings have shown everything this view gives since the last change. Their records:\n' + '\n'.join(f'{i}: {catalog.record_text(str(i))}' for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Best] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + compact(args)
            if ending and ep.steps_left() > 2 and ep.time_left() > CLOCK_FLOOR + 25:
                held = self.before_buy(cid) if buying else self.before_refuse(cid)
                if held:
                    out.append(held)
                    continue
            elif buying and ep.steps_left() >= 2:
                cleaned = ep.judge.drop_sold_lines()
                if cleaned:
                    trimmed = '\n[Best] ' + cleaned + trimmed
            if refusing and ep.steps_left() <= 5 and not ep.late_held:
                lines = [i for i in ep.built if not catalog.sold_out(i)]
                missing = [i for i in lines if i not in ep.basket.cart_ids() or (i in ep.dropped and not ep.basket.cart_fresh())]
                if len(lines) >= len(ep.items) and len(missing) + 1 <= ep.steps_left():
                    ep.late_held = True
                    redo = f"Put {', '.join(missing)} back in the cart at the requested quantities, then check out." if missing else 'Check out the cart as it is.'
                    out.append((cid, 'Not run yet (no step used). Ending without a purchase earns nothing when the store has a fitting listing, and running short of steps does not show that it has none. ' + redo))
                    continue
            if refusing:
                args = self.quiet(name, args)
            if ending and ep.steps_left() >= 2 and ep.time_left() > CLOCK_FLOOR + 15:
                go, said = ep.judge.gate('order' if buying else 'end')
                if not go:
                    out.append((cid, 'Not run yet. ' + said + '\nCart now: ' + ep.basket.cart_text()))
                    continue
                if said:
                    trimmed = '\n[Best] ' + said + trimmed
            reply = ep.store.call(name, args)
            if reply is None:
                out.append((cid, 'The call failed (network or server). You may try again.' if not ep.done() else 'Not run: the episode is over.'))
                continue
            observation = reply.get('observation')
            out.append((cid, self.after_call(name, args, observation, reply, mark, many, trimmed, ending)))
        return out

    def after_call(self, name: str, args: dict[str, Any], observation: Any, reply: dict[str, Any], mark: str, many: str | None, trimmed: str, ending: bool) -> str:
        """Bookkeeping after one store call, and the text the model sees for it."""
        ep = self.ep
        box, catalog, observer = ep.box, ep.catalog, ep.observer
        if many and isinstance(args.get(many), list) and len(args[many]) > 1:
            shown = compact(observation)
            if not any(str(i) in shown for i in args[many]):
                ep.compare_bad = min(ep.compare_bad or 999, len(args[many]))
                trimmed += f'\n[Best] The side-by-side view showed nothing for {len(args[many])} listings: send at most {self.compare_room()} per call.'
            else:
                ep.compare_ok = max(ep.compare_ok, len(args[many]))
        ep.recent = (ep.recent + [mark])[-3:]
        for key, value in args.items():
            values = value if isinstance(value, list) else [value]
            if values and all(isinstance(x, str) and x in catalog.rows for x in values):
                box.id_slots.add((name, key))
        catalog.absorb(observation)
        if name == box.tool('compare'):
            rows = catalog.records_in(observation)
            if rows:
                ep.facts_from[name] = ep.facts_from.get(name, False) or any(facts_of(r) for _, r in rows)
            catalog.note_view(name, observation)
        taken = box.committers() | {box.tool('ask')}
        if box.one_target(name) and name not in taken:
            listing = catalog.ident(observation) if isinstance(observation, dict) else None
            if listing:
                catalog.note_view(name, observation)
                ep.facts_from[name] = ep.facts_from.get(name, False) or bool(facts_of(observation))
                if name == box.tool('open') and not ep.facts_from[name]:
                    other = next((t for t in box.specs if t != name and box.one_target(t) and ep.facts_from.get(t) is not False and t not in taken and t != box.tool('search')), None)
                    if other:
                        box.jobs['open'], box.jobs['offer'] = other, name
                if not ep.facts_from[name] and name != box.tool('open') and not box.tool('offer'):
                    box.jobs['offer'] = name
                if name == ep.offer_tool():
                    ep.offer_seen[listing] = ep.store.step
                observer.harvest_rules(observation)
        plain_tools = {tool for job, tool in box.jobs.items() if job != 'search'}
        paged = observer.more_pages(name, args, observation) if name not in plain_tools and not ending else ''
        if name == box.tool('cart'):
            ep.basket.note_cart(name, args)
            ep.cart_said = compact(observation)
            ep.recent = []
            counts = [whole(v) for v in args.values() if not isinstance(v, str) and whole(v) is not None]
            for value in args.values():
                if counts and isinstance(value, str) and value in catalog.rows:
                    if counts[0] <= 0 and not names_id(observation, value):
                        ep.dropped.add(value)
                    elif counts[0] > 0:
                        ep.dropped.discard(value)
            lines = [i for i in ep.basket.cart_ids() if not catalog.sold_out(i) and i not in ep.dropped]
            if len(lines) >= len(ep.built):
                ep.built = lines
        if name == box.tool('search') and catalog.records_in(observation):
            ep.searched = (ep.searched + [dict(args)])[-8:]
            observer.read_shape(name, observation)
        if name == box.tool('cart_view'):
            ep.cart_view = observation
            ep.cart_seen_at = ep.store.step
        buying = self.is_buy(name, args)
        if buying:
            ep.buys += 1
            if not reply.get('finished'):
                ep.refused_at = ep.store.step
        text = observer.show(observation) + paged + trimmed
        if name not in plain_tools and not ending:
            text += observer.cheapest_here([observation] + ep.paged_obs) + observer.cap_ignored(name, args, observation)
        base = text
        proof = observer.evidence_lines(observer.opened_in(observation))
        if proof:
            text += '\n' + proof
        if name == box.tool('cart') and any(whole(v) == 0 for v in args.values() if not isinstance(v, str)):
            if any(isinstance(v, str) and v in catalog.rows and names_id(observation, v) for v in args.values()):
                text += '\n[Best] A count of 0 left that line in the cart: this tool adds the count to the line, so a negative count takes units off.'
        ep.note_sold = False
        lines = observer.loose_lines(observation)
        if lines:
            meaning = observer.take_note(lines)
            if meaning:
                text += '\n[note read] ' + meaning + '. Apply this change to your picks before checking out.'
        refused = buying and not reply.get('finished')
        again = refused and any(line in Words.spoken(observation) for line in ep.sold_lines)
        loose = ep.note_sold and (refused or (not ep.note_named and not buying))
        if (loose or again) and ep.basket.cart_ids() and not ep.done():
            seen = ep.judge.find_sold()
            if seen:
                text += "\n[Best re-read the cart's lines after the refusal]\n" + seen
        ep.note_sold = False
        swap = ep.judge.swap_hint()
        if swap:
            text += '\n' + swap
        if ep.queued:
            text += "\n[notes seen on Best's own calls] " + ' | '.join(ep.queued)
            ep.queued = []
        if name in box.readers():
            ep.read_cache[mark] = (catalog.version, ep.store.step, base)
        if reply.get('finished'):
            text += ' [episode finished]'
        return text

    # -- status for the model ----------------------------------------------------------------------------
    def status(self) -> str:
        ep = self.ep
        left = ep.steps_left()
        text = f'[Best] step {ep.store.step} of {ep.max_steps} used, {left} left; {ep.msgs_free()} message(s) free.'
        if left <= ep.tail() + 1:
            text += ' FINAL STEPS: check out now when the cart holds a line for each item; the no-purchase ending is for a stated requirement nothing meets, never for a shortage of steps.'
        elif left <= 6 or (left <= len(ep.items) + 5 and ep.buys == 0):
            text += ' Few steps left: set the cart to your picks and check out now.'
        elif ep.buys == 0 and left <= 12:
            text += ' Steps are running low: settle on your picks, fill the cart and check out within the next few steps.'
        elif ep.buys == 0 and not ep.cart_touched and ep.store.step >= ep.max_steps // 2:
            text += " Half the steps are used and the cart is empty: cart each item's best fitting listing now."
        missing = self.items_without_lines()
        if missing and ep.cart_touched and left > 6:
            text += ' Items with no cart line yet: ' + ', '.join(ep.items[n - 1].name for n in missing) + '.'
        pending = [rule.text for _, rule in ep.needs() if rule.points and not rule.resolved]
        if pending and ep.buys == 0:
            text += '\n[chains] Best resolves these from the pages they point at: ' + ' | '.join(pending[:3])
        if ep.leads:
            text += '\n[review candidates] ' + ', '.join(ep.leads)
        plan = ep.basket.fit_plan() if ep.bound and ep.buys == 0 else ''
        if plan:
            text += '\n' + plan
        if ep.notes:
            text += '\n[changes so far] ' + ' | '.join(ep.notes[-4:])
        if ep.time_left() < WRAP_UP_AT + 45:
            text += f'\n[time] about {max(0, int(ep.time_left()))} s are left: settle and check out within the next few calls.'
        return text + '\n[ledger]\n' + ep.ledger()

    # -- ending by code ----------------------------------------------------------------------------------
    def finish(self, last: bool = False) -> str:
        """End the episode by code: take sold-out lines out, fill missing items from the known candidates, check
        out a cart with lines, and decline only when the evidence allows it or nothing at all can be bought.
        Returns the store's refusal text when an ending was refused (the task changed), else ''."""
        ep = self.ep
        if ep.done() or ep.store.broken or ep.steps_left() <= 0:
            return ''
        buy, refuse = ep.box.tool('buy'), ep.box.tool('refuse')
        basket, judge = ep.basket, ep.judge
        clock_ok = ep.time_left() > CLOCK_FLOOR + 20
        if buy and ep.steps_left() >= 3 and clock_ok:
            judge.drop_sold_lines()

        def holding() -> bool:
            return bool(basket.cart_ids()) if basket.cart_fresh() else bool(ep.cart_lines)
        missing = self.items_without_lines() if holding() else list(range(1, len(ep.items) + 1))
        if buy and missing and ep.steps_left() >= 3 and clock_ok:
            if not ep.blocking and ep.audited == 0 and ep.steps_left() > 6 and ep.time_left() > WRAP_UP_AT:
                judge.audit()
            if not judge.decline_allowed():
                picks = judge.forced_pick()
                if holding():
                    picks = {n: i for n, i in picks.items() if n in missing}
                if picks and ep.steps_left() >= len(picks) + 2:
                    done = judge.buy_best(picks)
                    if ep.done():
                        return ''
                    if done:
                        return 'the checkout was refused: ' + done
        if buy and holding() and ep.steps_left() >= 1:
            args, faults = ep.box.shape(buy, dict(ep.box.buy_args))
            if not faults:
                if ep.steps_left() >= 3 and clock_ok:
                    ep.opener.complete_views(basket.cart_ids(), 2)
                if ep.steps_left() >= 2 and ep.time_left() > CLOCK_FLOOR + 15:
                    judge.gate('order')
                if ep.done() or ep.steps_left() <= 0:
                    return ''
                reply = ep.store.call(buy, args)
                ep.trace.append({'kind': 'forced_buy', 'reply': shorten(reply, 200)})
                if ep.done() or ep.steps_left() <= 0 or reply is None:
                    return ''
                ep.buys += 1
                ep.refused_at = ep.store.step
                shown = compact(reply.get('observation'))
                if re.search(r'bare|empty|no (?:items|lines|products)|nothing (?:in|to)', shown, re.I):
                    ep.cart_lines, ep.cart_view, ep.cart_touched = {}, None, False
                ep.observer.observe(buy, reply.get('observation'))
                if ep.note_sold:
                    judge.find_sold()
                ep.note_sold = False
                return 'the checkout was refused: ' + ep.observer.show(reply.get('observation'))
        if refuse and last and buy and not holding() and not ep.blocking and ep.steps_left() >= 3:
            picks = judge.code_pick()
            if picks and judge.buy_best(picks) and ep.done():
                return ''
        if refuse and (last or not holding()):
            args, faults = ep.box.shape(refuse, self.quiet(refuse, dict(ep.box.refuse_args)))
            if not faults:
                if ep.steps_left() >= 2 and ep.time_left() > CLOCK_FLOOR + 15:
                    judge.gate('end')
                if ep.done() or ep.steps_left() <= 0:
                    return ''
                reply = ep.store.call(refuse, args)
                ep.trace.append({'kind': 'forced_refuse', 'reply': shorten(reply, 200)})
                if reply is not None and not ep.done():
                    ep.observer.observe(refuse, reply.get('observation'))
                    return 'ending without a purchase was refused: ' + ep.observer.show(reply.get('observation'))
        return ''

    def trim(self) -> None:
        ep = self.ep
        if sum(len(compact(m.get('content'))) for m in ep.chat) <= CHAT_CHARS:
            return
        tools = [i for i, m in enumerate(ep.chat) if m.get('role') == 'tool']
        for index in tools[:max(1, len(tools) // 2)]:
            content = ep.chat[index].get('content') or ''
            if isinstance(content, str) and len(content) > 600:
                ids = [i for i in dict.fromkeys(re.findall(r'"([A-Za-z0-9][A-Za-z0-9_\-:.]{5,40})"', content)) if i in ep.catalog.rows][:20]
                ep.chat[index]['content'] = content[:400] + f"...(older result shortened; listings in it: {', '.join(ids)})"

    def work(self, turn_cap: int, spend_cap: float) -> None:
        ep = self.ep
        idle = 0
        while not ep.done() and not ep.store.broken and ep.steps_left() > 0 and ep.turns < turn_cap and ep.brain.cost < spend_cap and ep.time_left() > WRAP_UP_AT:
            if ep.interviewer.ready():
                summary = ep.interviewer.run()
                if summary:
                    ep.chat.append({'role': 'user', 'content': summary + '\n' + self.status()})
                continue
            if ep.pick == 'cheapest' and (ep.interviewed or not ep.box.tool('ask') or ep.msgs_free() <= 0) and ep.catalog.rows and ep.room() >= 8:
                notes = [ep.opener.price_ladder(n) for n in range(1, len(ep.items) + 1) if not ep.ladder_done.get(n)]
                notes = [n for n in notes if n]
                if notes:
                    ep.chat.append({'role': 'user', 'content': '\n'.join(notes) + '\nCart the verified cheapest fit for each item (or read the cheaper unread rows first), then check out.\n' + self.status()})
                    continue
            pending = any(rule.points and not rule.resolved for _, rule in ep.needs())
            if pending and ep.chain_pass < 1 and (ep.interviewed or not ep.box.tool('ask') or ep.msgs_free() <= 0) and ep.room() >= 6 and ep.catalog.rows:
                ep.chain_pass += 1
                chained = ep.chainer.resolve()
                if chained:
                    ep.chat.append({'role': 'user', 'content': chained + '\n' + self.status()})
                    continue
            self.trim()
            ep.turns += 1
            message = ep.brain.complete(ep.chat, [FAST, FAST_ALT], 3000, self.offered())
            if message is None:
                idle += 1
                if ep.brain.broke or ep.brain.calls >= MODEL_CALLS or idle > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get('tool_calls') or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get('function') if isinstance(call.get('function'), dict) else {}
                raw = fn.get('arguments')
                parsed = raw if isinstance(raw, dict) else (first_json(raw) if isinstance(raw, str) and raw.strip() else {})
                calls.append({'id': str(call.get('id') or f'b{ep.turns}_{number}'), 'type': 'function', 'function': {'name': str(fn.get('name') or ''), 'arguments': compact(parsed if isinstance(parsed, dict) else {})}})
            content = message.get('content')
            if not isinstance(content, str):
                content = compact(content) if content else ''
            if not calls:
                idle += 1
                ep.chat.append({'role': 'assistant', 'content': content or '(no call)'})
                ep.chat.append({'role': 'user', 'content': 'Call one of the tools to continue.\n' + self.status()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = ep.box.committers()
            if not ep.interviewed and ep.box.tool('ask') and ep.msgs_free() > 0 and any(c['function']['name'] in committing for c in calls) and ep.steps_left() > ep.tail() + len(ep.items) + 2:
                ep.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
                ep.chat.extend({'role': 'tool', 'tool_call_id': c['id'], 'content': 'Not run (no step used): the shopper is asked first.'} for c in calls)
                summary = ep.interviewer.run()
                ep.chat.append({'role': 'user', 'content': (summary or 'Best had nothing to ask.') + '\n' + self.status()})
                continue
            ep.chat.append({'role': 'assistant', 'content': content or None, 'tool_calls': calls})
            ep.trace.append({'kind': 'turn', 'text': content[:300], 'calls': [c['function'] for c in calls]})
            replies = self.run_calls(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += '\n' + self.status()
                ep.chat.append({'role': 'tool', 'tool_call_id': cid, 'content': text})

    def run(self) -> list[dict[str, Any]]:
        ep = self.ep
        if ep.done():
            return ep.trace
        ep.read_request()
        ep.trace.append({'kind': 'brief', 'brief': shorten(ep.brief, 300), 'roles': ep.box.jobs, 'max_steps': ep.max_steps, 'messages': ep.msg_limit, 'items': [(i.name, i.qty, [r.text for r in i.active()]) for i in ep.items]})
        opening = ep.view.get('opening')
        if opening is not None:
            ep.catalog.absorb(opening)
        intro = "SHOPPER'S REQUEST:\n" + ep.request
        if opening is not None:
            intro += '\n\nWHAT THE STORE SHOWS FIRST:\n' + ep.observer.show(opening)
        intro += ('\n\nStart with one search per requested item (its category when a filter exists), then open the attributes of the '
                  'most promising rows per item, so Best can see which properties these listings carry and ask the shopper. '
                  'Cart a fitting listing for each item as soon as you have one.\n\n' + self.status())
        ep.chat = [{'role': 'system', 'content': GUIDE}, {'role': 'user', 'content': intro}]
        self.work(min(72, max(LOOP_TURNS, ep.max_steps)), SPEND_HARD)
        rounds = 0
        while not ep.done() and not ep.store.broken and ep.steps_left() > 0 and rounds < 3:
            rounds += 1
            turned = self.finish()
            if ep.done() or not turned:
                break
            if not ep.brain.broke and ep.brain.cost < SPEND_HARD and ep.steps_left() > 1 and ep.time_left() > WRAP_UP_AT:
                ep.chat.append({'role': 'user', 'content': '[Best] Best tried to end the episode and ' + turned + '\nApply what the store says, then finish.\n' + self.status()})
                self.work(ep.turns + 8, SPEND_HARD + ENDING_ROOM)
        if not ep.done() and not ep.store.broken and ep.steps_left() > 0:
            self.finish(last=True)
        ep.trace.append({'kind': 'end', 'finished': ep.done(), 'steps': ep.store.step, 'cost': round(ep.brain.cost, 5), 'calls': ep.brain.calls, 'seconds': round(ep.clock.used(), 1)})
        return ep.trace


# ------------------------------------------------------------------------------------------------------------
# wiring and the entry point
# ------------------------------------------------------------------------------------------------------------
class Agent:
    """One episode's crew: the episode state and the helpers that work on it."""

    def __init__(self, problem: dict[str, Any]):
        ep = Episode(problem)
        ep.observer = Observer(ep)
        ep.kinds = Kinds(ep)
        ep.interviewer = Interviewer(ep)
        ep.opener = Opener(ep)
        ep.chainer = Chainer(ep)
        ep.basket = Basket(ep)
        ep.judge = Judge(ep)
        ep.driver = Driver(ep)
        self.ep = ep

    def run(self) -> list[dict[str, Any]]:
        return self.ep.driver.run()


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        agent = Agent(problem_data)
    except Exception as exc:  # noqa: BLE001 - the harness must always get a result
        return [{'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]}]
    ep = agent.ep
    try:
        return agent.run()
    except Exception as exc:  # noqa: BLE001
        ep.trace.append({'kind': 'crash', 'error': f'{type(exc).__name__}: {exc}'[:400]})
        try:
            if not ep.done():
                ep.driver.finish()
        except Exception as again:  # noqa: BLE001
            ep.trace.append({'kind': 'crash', 'error': f'{type(again).__name__}: {again}'[:400]})
        return ep.trace


__all__ = ['agent_main']
