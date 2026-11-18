from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agent.proxy_client import ProxyClient

PARSE_MODEL = "anthropic/claude-sonnet-5.5"
PLAN_MODEL = "anthropic/claude-sonnet-5.5"
VERIFY_MODEL = "anthropic/claude-sonnet-5.5"
FAST_MODEL = "anthropic/claude-haiku-5.5"
BACKUP_MODEL = "openai/gpt-6-luna"

TASK_SECONDS = 265.0
WRAP_SECONDS = 45.0
FACT_CHARS = 2200
ROW_CHARS = 260
SOFT_SPEND = 0.03
SPEND_CAP = 0.05
REPLAN_EACH = True      # the question plan is made again after every reply, with the replies so far
PAIR_LAST = 0           # the last questions name two properties (off: shoppers ask for one subject at a time)
CHEAP_BY_PRICE = True   # under a cheapest rule, the lowest price among passing listings wins
CHECKED_ONLY = True     # an order goes only to a listing the final check passed (or a clean code read, when late)
REVEAL_RETRY = True     # a product with a revealed need is searched again before it is given up
BASKET_STEPS = True     # every carted listing is looked at again, and spare steps are shared across products
YES_READ = True         # the reply to the basket confirmation is read
READ_GROUNDED = True    # read values must be written in the request; the budget must be one of its figures
READ_CHECK = True       # a second strong read looks for stated product demands the first read missed
SHORT_QUERIES = True    # searches after a reveal use the bare product noun, not the long shelf name
NO_NEED_ROWS = True     # a product with no requirement left: any in-stock row on its shelf fits, opened or not
ARG_FILL = True         # a required cart argument the tool map left out is taken from the schema
NEAR_MISS = True        # no clean fit: a listing unsure only (never failed) goes to the final check before any decline
SPLIT_ROUTE = False     # one product and several products get their own step plans (off: thin pools hurt the multi-product picks)
REFUSAL_REPICK = True   # a turned-down decline always leads to another pick while steps remain
GATE_REPLY = True       # the reply to a procedure message is read for needs like any other reply
HUNT_MEMO = True        # a pointer hunted again skips the searches and pages already tried
NEG_RECHECK = True      # a reply read as an exclusion is read again by the strong reader
CLOSEST_RULE = True     # a request that wants the closest listing when nothing fits never ends in a decline
NAME_LABEL = True       # a shop section is matched against the product's name too, and a lone unmatched one goes to the label reader
_SERIES = re.compile(r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b")
REF_OPS = {"same as reference": "is", "different from reference": "is not", "counterpart of reference": "is",
           "at least reference": "at least", "at most reference": "at most"}

_llm = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_env = ProxyClient(timeout=90, max_retries=2, retry_delay=1)

# per world key, what earlier tasks in this process learned about the store
_WORLD_MEMORY: dict[str, Any] = {}

BRIEF_READING = """Role: you brief a buying agent before it works on one job in a simulated shop.
Input: the customer's request text, followed by the shop's tool list given as OpenAI function schemas.
Output: answer with exactly ONE JSON object and no surrounding text, laid out as below:

{
 "items": [  // add one entry for every separate product the customer intends to purchase; most often there is only one
   {"name": "short plain noun phrase for the product as worded in the request (inside a list, that entry's own words)",
    "category": "kind of goods or shop section being asked for, using the request's wording",
    "quantity": 1,
    "requirements": [ // all hard demands the text places on attributes of THIS entry; leave price, stock, quantity and budget out
      {"property": "attribute name the way an offer page would write it", "op": "is|is not|at least|at most|one of|contains|same as reference|different from reference|counterpart of reference|at least reference|at most reference",
       "value": "value, limit or list copied word for word from the request (for 'one of': alternatives joined with ' | ')",
       "reference": null  // for any op ending in 'reference': {"category": "...", "seller": "...", "attributes": {"property": "value", ...}, "property": "attribute to read off that other page", "count": null or how many offers the wording says qualify, "pick": null or the rule that selects among several (e.g. "whichever is wider")}
      }],
    "hints": ["attribute names the backstory implies the customer quietly cares about (a comment on who will wear it implies gender or age group; a comment on a computer's insides implies processor); empty list when nothing is implied"],
    "search_terms": ["2-4 search-box queries, most specific first. Since every query word must match, keep each one brief: the product noun plus one or two value words (a colour, a material, a number with its unit such as 5v or 16gb); never a lone attribute name and never words that only identify a referenced offer; finish with the bare product noun"]
   }],
 "total_cap": 12345 or null,        // spending ceiling for everything, a plain number in the request's unit (cents when the text says cents)
 "cheapest": false,                 // set true only when the text asks for the cheapest / lowest price / cheapest total
 "best": null,                      // if the text wants the top pick by some yardstick, write that yardstick; otherwise null
 "message_budget": null,            // integer count of messages the customer will accept, or null
 "call_budget": null,               // integer count of tool calls / steps allowed, or null
 "in_stock_only": true,             // true unless the text accepts listings that are sold out
 "ask_about": ["topics the text instructs the agent to raise with the customer, worded as in the text; empty when absent"],
 "fit_rules": [{"accessory": "item name", "host": "item name"}],   // instructions shaped like 'make sure X fits Y'; empty when absent
 "conditions": [{"if": {"item": "item name", "property": "...", "values": ["..."]}, "then": {"item": "item name", "property": "...", "op": "is|is not|at least|at most|one of", "value": "..."} or null, "otherwise": the same shape or null}],   // a demand on one entry that depends on the outcome of another entry; empty when absent
 "decline_rule": null,              // the text's instruction for when nothing qualifies (e.g. simply say so, no replacements), or null
 "no_fit": {                        // what the text wants done when no listing meets every demand; null when it says nothing about that
   "mode": "closest|decline",       // "closest": put forward or buy the listing missing the fewest demands anyway; "decline": buy nothing
   "counts": [],                    // things besides product demands the text counts as demands (e.g. "stock", "budget", or others, worded as in the text)
   "keep": [],                      // demands the text says the fallback may never miss, worded as in the text
   "choose": {"by": "price|other|none", "yardstick": null, "direction": "lowest|highest", "margin": null},   // how to choose among equally close listings: by price or by another yardstick (copy its wording), which end wins, and margin {"value": number written in the text, "unit": "percent|amount"} for how far from that end is still accepted
   "tell": false                    // true when the text wants the customer told that nothing fully fits
 },
 "procedure": [                     // demands on the WAY the job gets done (not product demands, budgets, message or call limits, nor the nothing-qualifies instruction, which is decline_rule): a sign-off or confirmation to obtain, something to show or report to the customer, a tool that must be used, a required sequence; empty when absent
   {"moment": "start|before_order|before_decline|before_finish", "kind": "message|tool", "tool": "tool name or null", "goal": "the thing that must occur, phrased as the text phrases it", "messages": 1}  // messages: customer messages this step consumes (0 for a tool); a check the agent performs in the shop on its own (stock, availability, some value) is kind "tool" with 0 messages, unless the text wants the customer told the result
 ],
 "tools": {
   "search_text": {"name": "tool", "query": "argument for the keywords", "page": "argument that moves to the next results (a page number, a row offset, or a token from the previous result), or null",
                   "page_kind": "number|offset|token or null", "mode": {"arg": "argument that selects keyword search", "value": "its value"} or null},
   "search_fields": {"name": "tool for searching by category/brand/price fields, or the same tool in its fields mode", "category": "arg or null", "brand": "arg or null", "price_max": "arg or null", "page": "arg or null", "page_kind": "number|offset|token or null", "mode": {"arg": "...", "value": "..."} or null} or null,
   "open": {"name": "tool that returns one listing in full", "id": "its id argument"},
   "facts": {"name": "separate tool that returns a listing's property table, if the store has one", "id": "arg"} or null,
   "compare": {"name": "tool returning several listings in full", "ids": "its list argument"} or null,
   "cart_add": {"name": "tool placing an offer into the cart", "id": "arg", "qty": "quantity arg or null",
                "mode": {"arg": "argument selecting add versus remove", "add": "value for adding", "remove": "value for removing"} or null},
   "cart_set": {"name": "tool that swaps the entire cart for a list of lines, if available", "lines": "its list argument", "id": "key of the offer id within a line", "qty": "key of the quantity within a line"} or null,
   "cart_view": {"name": "tool that displays the cart"} or null,
   "checkout": {"name": "tool that submits the order (final)", "args": {"argument": "value"} or null},
   "decline": {"name": "tool that closes with no purchase (final)", "remark": "text argument or null", "args": {"argument": "value"} or null},
   "message": {"name": "tool that writes to the customer", "text": "its text argument"},
   "status": {"name": "tool that tells the remaining allowances"} or null
 }
}

Rules:
* Only firm demands become requirements. Wishes (ideally, would be nice) and backstory (who uses it, where it is kept) are NOT; note what they imply under "hints". Words that merely name the product type or its section (men's, women's, kids' in "men's rain capes"; "gaming" in a section title) go into "name"/"category", never into "requirements". However, a value supplied for a named attribute ("in the <value> style", "with <value> as its <property>", "a <property> of <value>") IS a requirement on that attribute, even if it reads like a product type; the category remains the product the text names.
* "3 x gadgets where the colour is red and the size is not large" means a single entry, quantity 3, carrying two requirements (colour is red; size is not large).
* A count describing what one offer contains (a pack, set or box of N pieces, a pack holding N) is a requirement on pack or set size; "quantity" is how many offers to purchase.
* Wording that only restates a consequence of a requirement ("zip front only, so nothing with buttons or toggles") is ONE requirement (closure: zip front), not many; do not invent attributes for whatever it excludes.
* "A or B" on an attribute becomes op "one of", value "A | B". "not X" / "anything but X" becomes op "is not". "at least N units" becomes "at least", value "N units". Preserve units and spelling exactly.
* If a requirement refers to ANOTHER offer's page (pinned down by category, seller and a few attribute values): matching that page's value is op "same as reference"; anything but that page's value is op "different from reference"; the partner of that page's value (whatever that page or the shop couples with it, the thing that goes alongside) is op "counterpart of reference"; a value no smaller (or no bigger) than that page's is op "at least reference" (or "at most reference"). That page might itself only defer the attribute to some further offer: the agent follows such chains unaided, so describe the page the text names. Populate "reference" with category, seller, named attributes and the attribute to read. When the text says several offers from that seller qualify and supplies a way to select one (the larger, the cheaper, the one with more of something), store the way in "pick" and the number in "count"; keep "attributes" to plain identifying values, never the comparison. Identifying values describe THAT other offer only: never requirements on the wanted entry, never its search terms. Invented example: "for the colour, avoid whatever Fenwick's garden hose page (length 20 m, material rubber) states" gives one requirement {"property": "colour", "op": "different from reference", "value": "", "reference": {"category": "garden hoses", "seller": "Fenwick", "attributes": {"length": "20 m", "material": "rubber"}, "property": "colour"}} and none for length or material.
* A demand on one entry that applies only if another entry ends up a particular way ("should X's P turn out to be V, I need Y's Q to be W"; "Y's Q must be W, except when X's P is V") goes under "conditions" (first form: "then" applies when X's P is among the values; second form: "otherwise" applies when it is not), never under "requirements" or "procedure".
* Category means the type of goods; if the text names a shop section (e.g. "the category is garden tools"), reuse those exact words.
* Money: a total given in cents makes "total_cap" that figure in cents.
* Copy tool and argument names letter for letter from the schemas. If a SINGLE tool finishes the job both ways, with an argument choosing between buying the cart and walking away, then "checkout" and "decline" both point at that tool, each carrying "args" with the argument value that selects it (spelled as the schema lists it). A tool described as final that closes with no purchase is "decline"; the one paying for the cart is "checkout". When one search tool carries an enum argument flipping between keyword and field search, "search_text" and "search_fields" both point at it with their mode values."""

ROW_SCREENING = """Role: you triage result rows for a buying agent.
Input: the wanted product, its demands, and numbered rows (id, title, short text, category, price).
Rules:
* Per row, decide if the offer COULD pass every demand once its complete attribute page gets opened; rely only on what that row itself contains.
* Silence on a demand leaves the offer possible ("maybe").
* Row wording that plainly contradicts a demanded value means "no".
* A row that shows the demanded values itself means "yes".
* The shop's section label fixes the product type: a row filed under the wanted section is the correct type whatever its title implies, so the type of goods is never grounds for "no".
Output: just ONE JSON object: {"yes": [row numbers], "no": [row numbers]}; rows in neither list are "maybe"."""

LISTING_JUDGING = """Role: you rule on whether shop listings satisfy a shopper's requirements, using only each listing's own records.
Input: the wanted product, requirements numbered R1, R2, ..., optional STORE FIELDS, then the listings.
Rules:
* A requirement passes when the listing's own property for it carries the value. Keys come in many spellings (spacing, underscores, prefixes, short forms); any key naming that property qualifies, and one property can appear under several keys. Matching is by sense: a single element of a comma list is enough, singular equals plural, close synonyms and evident short forms qualify, letter case is irrelevant, and values with units match once units are converted (numbers compared numerically; a bound includes its edge; a range passes a bound only if it lies fully inside).
* The property table ranks first: when some property covering the requirement exists, its value rules (another value fails the requirement even if the title suggests otherwise). Only when NO key covers it does a value in the seller's notes, the options or variants field, the title or the description qualify. When nothing mentions it, the requirement is unaddressed ("unsure"). A value under a different though related property is irrelevant (a handle is not a lock type; a lining material is not the material).
* "is not" and "different from" requirements fail if the listing's property holds the excluded value; "one of" requirements require one listed value; "contains" requires the value among those present. A requirement for some feature to be absent (value none, no or without) passes when no records mentions that feature.
* The shop's section label fixes the product type: an listing labelled with the wanted section is the correct type whatever its title claims. Only an listing lacking any label may fail as an accessory, a spare part or a case instead of the product itself.
* When a requirement asks for compatibility with another product (named by its model names or codes), it is met only when this listing's own text lists one of those names among what it fits or works with. When the other product's page names a series or family of compatible part codes, a listing whose own part code (often in brackets in its title) starts with that series meets it. A sentence saying that some OTHER part fits a product applies to the listing whose title carries that part's code, not to the page it is written on; a listing's "related_pages" show such pages, and their statements about this listing's own part code count for it. A pointer to another page alone proves nothing.
* An listing's seller notes may delegate one property to ANOTHER listing (for that property, use or copy some other listing's value, not this one). In that case this listing's own value for it is void: a requirement on it is "unsure" unless a related page shown with the listing supplies it. Notes delegating an property no requirement mentions alter nothing.
* STORE FIELDS, when supplied, show per required property the keys and values this shop's opened listings of the section use. If some of them hold the exact required value in that field, a listing whose field holds another value fails, even a related or broader one; a value carried by no opened listing can still pass through a clear synonym.
* Rule only on the requirements given; never invent extra ones.
Output: ONE JSON object and nothing else (no comments): {"verdicts": [{"id": "...", "failed": ["R numbers the records clearly fails"], "unsure": ["R numbers no records addresses"], "weak": ["R numbers passed only via the title, the description or an options field, not via an property or a seller note"], "loose": ["R numbers passed via a synonym, a broader word or a conversion rather than the value's literal words (e.g. a value 'female' for a required 'girls')"]}, ...]} with an entry for every listing; drop any list that would be empty. An listing with neither "failed" nor "unsure" passes all requirements shown."""

QUESTION_PLANNING = """Role: you schedule, in order, the questions that a buying agent is going to send to its customer.
Input: the request, and for each wanted entry its name, demands known so far, backstory hints and an attribute menu: attributes its candidate offers truly carry (each with how many offers carry it and sample values).
Rules:
* The customer holds secret demands; every one is a value of ONE attribute of one wanted product, disclosed only when a message asks about precisely that attribute (by name or a plain synonym). A message is answered solely about the FIRST attribute it mentions; asking about something the customer is indifferent to yields "no preference", and the message is wasted.
* Choose exactly QUESTION_COUNT questions, putting the strongest one first and the weaker ones after it.
* Start with attributes the backstory or hints aim at (what the text says the customer will disclose if asked: who uses or wears it aims at age group or gender; "how it holds on" aims at closure; what is inside aims at processor; "how it looks" aims at colour, pattern or style). Map every single hint onto whichever attribute name in the menu comes closest to it.
* Continue with attributes buyers of such goods usually have opinions on, preferring ones carried by many candidates (the "listings" count) (material, appearance, size or fit, type or style, variant such as flavour, intended user, headline technical spec), present on many candidates, whose values differ across offers. An attribute with an identical value everywhere, a pseudo-attribute like an option or variant picker, an identifier, a model number, a warranty or delivery note, is a weak question.
* Skip any attribute a known demand already fixes, as well as price, stock or quantity, and never repeat an attribute (two spellings of one attribute are one). Divide questions among entries by how uncertain each is; an entry with strong hints takes the first question.
Output: JSON {"questions": [{"item": "entry name copied exactly", "property": "attribute name copied exactly from the menu (words with spaces)"}, ...]}."""

REPLY_READING = """Role: you decode a customer's answer to a buying agent that asked about one attribute of one product.
Input: the entry, the attribute asked about, the question, the KNOWN NEEDS and the answer text.
Output: ONE JSON object and nothing else:
{"kind": "need|repeat|free|split|unclear|other",
 "property": "attribute the customer refers to (their wording, or the asked attribute when that is what they intend)",
 "op": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart of reference|at least reference|at most reference",
 "value": "value, limit or list the customer supplied (list items joined with ' | ')",
 "reference": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "attribute to read off that other page", "count": null or how many offers it says qualify, "pick": null or the rule that selects among several},
 "budget": null or a number when the customer announces a new spending total,
 "answer": "" or, when the customer asked the agent something, a short plain reply,
 "steps": [] or demands the customer now places on how the job is done (a sign-off to obtain, something to show or report to them, a tool to use prior to ordering), each {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "the thing that must occur", "messages": 1}}
Rules:
* "need": the customer supplies a value, limit, exclusion or list they insist upon (however phrased: settled, firm, nothing but, only X will do, anything bar X, X or Y and nothing else). Decide the op by whether the customer accepts X or refuses it: "anything bar X will do" or "anything other than X" (X itself refused) is op "is not", value X; "nothing other than X", "will not settle for any P other than X", "only X" (X alone accepted) is op "is", value X. If they refer to ANOTHER offer's page (pinned down by category, seller and attribute values) and want that page's value, use op "same as reference" and fill "reference"; anything but that page's value is op "different from reference"; the value that page couples with its own (its partner, what goes alongside) is op "counterpart of reference"; a value no smaller or no bigger than that page's is op "at least reference" or "at most reference".
* "repeat": instead of addressing the asked attribute, the customer restates a demand already in KNOWN NEEDS (same attribute and value).
* "free": indifferent, leaves it open, anything is fine, never considered it.
* "split": wants the message broken apart, its topics raised one by one, or only its first part dealt with (the message mixed several topics).
* "unclear": did not grasp the message, or asks for it again or in simpler words.
* "other": everything remaining (a greeting, a counter-question, a budget shift).
* "property" is whatever attribute the answer itself mentions when it mentions one (their wording), possibly different from the asked one; only when it mentions none is it the asked attribute. When the message asked about two attributes ("X or Y") and the answer names neither, give the one of the two that its value describes.
* If the answer states a limit with explicit words like "at most N", "at least N", "no more than N" or "under N", those words decide op and value even when the sentence reads strangely (e.g. "nothing that falls short of at most 34 inches" is op "at most", value "34 inches")."""

EVENT_READING = """Role: a customer remark came inside a shop tool's output during a buying agent's job; say what changed.
Input: the remark lines and the wanted entries.
Output: ONE JSON object:
{"changes": [{"kind": "budget|drop|add|price|stock|step|other",
  "budget": null or new ceiling as a number,
  "property": null or attribute a removed or new demand is about,
  "op": "is|is not|at least|at most|one of", "value": null or value of the new demand,
  "item": null or the wanted product affected, "note": "one short sentence"}]}
Rules:
* "budget": a fresh amount to spend (raised or cut); copy the figure exactly as written, in its own unit (cents stay as cents: do not convert).
* "drop": the customer gives up demanding some attribute ("let X go", "drop X", "never mind X").
* "add": a new demand.
* "price": the customer reports the selected offer now costs something else.
* "stock": the customer reports the selected offer is no longer purchasable.
* "step": the customer demands a step in the way the job is done (a sign-off to obtain, something to show or report to them, a tool to use prior to ordering); place it under "step": {"moment": "before_order|before_decline|before_finish", "kind": "message|tool", "tool": null, "goal": "the thing that must occur", "messages": 1}."""

PAGE_FOLLOWING = """Role: you study one shop offer page for a buying agent tracing a reference chain.
Input: the attribute sought and the page.
Rules:
* The agent wants the sought attribute's value as this page presents it. It can sit in the attribute table or in seller remarks, or the remarks may say that attribute should come from ANOTHER offer (pinned down by category, seller and a few attribute values) rather than this one. Remarks may also tell which value of the attribute goes with which (a single couple for this page's value, or the shop's entire couple list).
* When remarks point the SOUGHT attribute to another offer, return the redirect and leave value empty; a remark pointing some different attribute away is not a redirect here.
* When remarks give the value plainly, return it. Failing that, take it from the attribute table under any key naming the attribute; without such a key, from the options field, title or description if they give it.
* A sentence that merely enumerates the shop's couples does not reveal this page's own value.
* "paired" must come from a coupling the page itself gives: inside a couple list, the couple that includes this page's own value (couples work in both directions); null if the page gives none or its own value is not known.
Output: ONE JSON object: {"value": "page's own value verbatim, or null", "paired": "value the page couples with it, or null", "redirect": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}"""

QUERY_WRITING = """Role: you invent search-box keyword strings for a buying agent.
Input: the product, its demands, the list of queries already used, and some of the offers seen (each marked with whether it passed).
Rules:
* Each query word must occur in an offer's title, brand, category or attribute values (stems match; very short words are dropped), and results come back in a fixed sequence a few rows at a time, so a query should be tight enough that most hits pass the demands.
* Give up to 5 fresh queries of 1-3 words each, strongest first: the product's own noun as shop titles phrase it plus a demanded value as offers would print it, or alternative words the shop uses for that product or value; never reuse an earlier query.
Output: only ONE JSON object: {"queries": ["...", ...]}."""

BRIEF_CHECK = """Role: you proofread the requirements another reader took from a shopper's request.
Input: the request and, per product, the requirements already taken.
Rules:
* List only demands on a product's own attributes that the request states outright and the list lacks or gets wrong (a missed value, a flipped exclusion, a wrong bound). Price, budget, stock, quantity, message and step limits, and the procedure are not product demands.
* Backstory that only hints at a wish (who uses it, where it goes) is not a stated demand: leave it out.
* Copy each value word for word from the request.
Output: ONE JSON object and nothing else: {"add": [{"item": "product name as listed", "property": "...", "op": "is|is not|at least|at most|one of|contains", "value": "..."}], "fix": [{"item": "...", "property": "...", "op": "...", "value": "..."}]}; empty lists when the reading is complete."""

BASKET_REPLY = """Role: a buying agent showed its customer the final basket and asked for a yes; you read the reply.
Input: the products, the message sent and the customer's reply.
Output: ONE JSON object and nothing else:
{"agree": "yes|no|unclear", "item": null or the product a requested change is about, "property": null or the attribute the customer now demands, "op": "is|is not|at least|at most|one of", "value": null or the value they demand, "budget": null or a new spending total as a number}
Rules:
* "no" only when the customer turns the basket down or asks for something different; a plain acknowledgement is "yes".
* Fill property and value only from words the customer wrote."""

ENDING_CHOICE = """Role: a shop tool that ends a buying job still needs an argument telling it how to end.
Input: the tool's description, the missing arguments (with the values each permits) and the goal.
Output: only ONE JSON object: {"<argument name>": "<the permitted value that meets the goal>", ...}."""

PROCESS_GATE = """Role: you audit a buying agent's procedure against the customer's text at a single point of the job.
Input: the text; the procedural demands it makes; the point reached (what the agent intends next); the current basket; remaining allowances; the shop's tools (those that finish the job or alter the cart are left out, since the agent runs those itself); and every earlier call with its output.
Rules:
* Work out which procedural demands fall due RIGHT NOW, ahead of the agent continuing, and are still undone for the present situation.
* A sign-off or confirmation only covers the basket in precisely its present form (one obtained for a previous basket is void); a demand already met in this situation is not due again; one tied to another point is not due.
* Any customer message must be understandable alone: for a basket, list each line (quantity, title, price) plus the total, and state clearly what you request.
* Never place a product-attribute question here, and never burn the final message on anything except a procedural demand.
* A check inside the shop (stock, availability, some value) uses the shop's reading tools, not a message, unless the text wants the customer told; the agent already opens every offer it purchases, so such a check is due only when the basket's offers were never opened.
Output: only ONE JSON object: {"actions": [{"tool": "exact tool name", "args": {"argument": "value", ...}, "why": "the procedural demand it serves"}]} in sequence, at most 2, or {"actions": []} when nothing falls due."""

BEST_RANKING = """Role: among offers that already pass all demands, a customer wants the top one by a stated yardstick (a value to maximise or minimise).
Input: the yardstick, the product and the offers.
Rules:
* Get each offer's yardstick value from its attributes, title or description, converting differing units so they are comparable.
* Include every offer id, best first; unreadable values go last; ties put the cheaper first.
Output: only ONE JSON object: {"order": ["id", ...], "values": {"id": "value as read"}}."""

SEARCH_AUDIT = """Role: you double-check a buying agent's hunt for one product before it gives that product up.
Input: product, demands, budget, queries already used and all offers met: OPENED ones (attributes, notes, demands the agent ruled failed or unaddressed) plus rows only glimpsed in results.
Rules: each of the three lists is allowed to come back empty when there is nothing to put in it.
Output: only ONE JSON object: {"fits_after_all": ["ids of OPENED offers that really pass every demand, judged on their own data: an attribute value rules, a conflicting value fails, an attribute no data gives counts as unmet"], "unopened": ["ids of unopened rows most likely to pass, strongest first, at most 4"], "queries": ["at most 2 fresh keyword queries of 1-3 words likely to surface passing offers, never one already used"]}."""

CONDITION_CHECK = """Role: a buying agent tests one offer against a customer's condition.
Input: the attribute, the allowed values and the offer.
Rules: from the offer's own data, decide if its value for that attribute is among those values (same sense; units converted); when no data gives the attribute, "holds" is false and "value" null.
Output: only ONE JSON object: {"holds": true or false, "value": "offer's value verbatim, or null"}."""

PAGE_MATCHING = """Role: a buying agent is trying to locate one particular offer described by a category, a seller and a few attribute values.
Input: the description and candidate offers with their data.
Rules:
* Seller must agree, the section label must name the described type, and each described attribute must agree with that offer's data (same sense; units converted).
* If an offer's attribute table is absent, its seller notes, title and description act as its data: an offer whose notes give the described values counts as a match.
* If a PICK rule appears, more than one offer may match: apply the rule across the matches and return its choice.
Output: only ONE JSON object (no reasoning): {"id": "id of the offer fitting the description, or null when none fits"}."""

FINAL_CHECK = """Role: you are the closing gatekeeper before a buying agent submits an order.
Input: wanted product with quantity, budget, demands and the offer.
Rules:
* Test the offer against EACH listed demand using its own data only.
* A demand passes when the offer's own attribute for it carries the value (any key naming the attribute qualifies; comma lists, synonyms, short forms, singular/plural and unit conversions qualify); with no covering key, seller notes, the options field, title or description may carry it; a demand nothing mentions fails; a value under another attribute is irrelevant.
* A compatibility demand passes when this offer names the other product among things it fits, or when the other product's page names a series of compatible part codes that this offer's own part code begins with.
* An attribute that the notes delegate to some other offer's value cannot pass via this offer's own value.
* Honour exclusions and closed lists; an offer labelled with the wanted section is the correct type whatever its title says.
* Price multiplied by quantity has to fit the budget (in the unit the prices use); if the data says it is not purchasable, it fails.
* Rule only on the listed demands; add none of your own.
Output: ONE JSON object: {"fits": true or false, "problems": ["every demand clearly unmet or unaddressed by the data"]}."""

NEAR_CHECK = """Role: you give an offer a second hearing after no offer passed the strict gatekeeper; the alternative is buying nothing.
Input: wanted product with quantity, budget, demands and the offer.
Rules:
* For each demand, look at the offer's own data: attribute keys, then notes, options, title and description.
* The demand is contradicted only when that data plainly states a different value, breaks an exclusion or a closed list, or sits outside a stated range.
* A demand the data says nothing about is "silent", not contradicted.
* A demand whose value is no concrete value (it points at another page, a note, a person or anything else this data cannot show) is "silent" as well.
* Price, budget and stock are checked elsewhere; leave them out.
* An offer filed under the wanted section is the correct type whatever its title says.
* Rule only on the listed demands; add none of your own.
Output: ONE JSON object: {"fits": true when nothing is contradicted, otherwise false, "problems": ["each contradicted demand and the value the data gives"], "silent": ["each silent demand"]}."""

LABEL_PICKING = """Role: a shop files offers under section labels; a customer wants one product type.
Input: the wanted product, a request excerpt and the labels seen.
Rules:
* Return each observed label that IS that type of product (the one whose sense matches the request's phrase as a whole, a broader label containing it included, like "sweaters and cardigans" for a cardigan).
* Exclude adjacent types (accessories, parts, cases for it, narrower or distinct types).
Output: only ONE JSON object: {"labels": ["label copied exactly", ...]} (empty when none matches)."""

KEY_MAPPING = """Role: a shop tool returned the JSON record shown; you identify which of its keys serve each role.
Input: the record.
Rules: give key names letter for letter, or null where no key serves.
Output: ONE JSON object:
{"id": "identifier of the offer", "title": "name shown for it", "text": "descriptive prose", "price": "cost", "stock": "availability flag", "category": "section label", "brand": "maker or brand", "seller": "vendor or shop", "facts": "object containing the attribute table", "notes": "vendor's free remarks on it", "total": "sum due for a cart", "lines": "array of cart lines", "next": "token or marker leading to more results", "count": "how many results match overall"}"""


PART_CODE = re.compile(r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b")


def plain(text: Any) -> str:
    lowered = str(text or "").lower()
    for old, new in (("&", " and "), ("_", " "), ("-", " ")):
        lowered = lowered.replace(old, new)
    return re.sub(r"[^a-z0-9 ]+", " ", lowered).strip()


def word_set(text: Any, keep_short: bool = False) -> set[str]:
    found: set[str] = set()
    for word in plain(text).split():
        if len(word) > 3 and word.endswith("s"):
            word = word[:-1]
        if keep_short or len(word) > 1:
            found.add(word)
    return found


def all_words(text: Any) -> set[str]:
    return word_set(text, keep_short=True)


def same_name(first: Any, second: Any) -> bool:
    if plain(first) == plain(second):
        return True
    words = word_set(first)
    return bool(words) and words == word_set(second)


def overlap_ratio(first: Any, second: Any) -> float:
    left, right = word_set(first), word_set(second)
    if left and right:
        return len(left & right) / len(left | right)
    return 0.0


def camel_spaced(key: Any) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(key))


def _balanced_json(body: str, opener: str, closer: str) -> Any:
    start = body.find(opener)
    if start < 0:
        return None
    depth, quoted, escaped = 0, False, False
    for pos in range(start, len(body)):
        ch = body[pos]
        if quoted:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                quoted = False
        elif ch == '"':
            quoted = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(body[start:pos + 1])
                except Exception:
                    return None
    return None


def first_json(reply: str) -> Any:
    if not reply:
        return None
    body = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    for opener, closer in ("{}", "[]"):
        found = _balanced_json(body, opener, closer)
        if found is not None:
            return found
    try:
        return json.loads(body)
    except Exception:
        return None


def squeeze(value: Any, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    hit = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(hit.group(0)) if hit else None


def whole_count(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def looks_like_sentence(value: Any) -> bool:
    return isinstance(value, str) and len(value) > 24 and " " in value.strip()


def largest_record_list(node: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    if isinstance(node, list):
        return node if node and all(isinstance(row, dict) for row in node) else None
    if not isinstance(node, dict) or depth >= 2:
        return None
    best = None
    for value in node.values():
        found = largest_record_list(value, depth + 1)
        if found and (best is None or len(found) > len(best)):
            best = found
    return best


def split_shopper_lines(obs: Any) -> tuple[Any, list[str]]:
    if not isinstance(obs, dict):
        return obs, []
    said: list[str] = []
    kept: dict[str, Any] = {}
    nested = sum(1 for value in obs.values() if isinstance(value, dict))
    # bare single-word strings mean the dict is a record itself, not a wrapper
    is_record = any(isinstance(value, str) and value.strip() and " " not in value.strip() for value in obs.values())
    wrapper = nested == 1 and len(obs) <= 3 and not is_record
    for key, value in obs.items():
        if isinstance(value, list) and value and all(looks_like_sentence(x) for x in value):
            said.extend(str(x) for x in value)
        elif looks_like_sentence(value) and wrapper:
            said.append(str(value))
        else:
            kept[key] = value
    if said and len(kept) == 1:
        only = next(iter(kept.values()))
        if isinstance(only, dict):
            kept = only
    return kept, said


def guarded(job: Any) -> Any:
    try:
        return job()
    except Exception:
        return None


def run_parallel(jobs: list, workers: int = 3) -> list:
    if len(jobs) < 2:
        return [guarded(job) for job in jobs]
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(guarded, jobs))


def tidy_property(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).replace("_", " ")).strip().lower()


def value_options(value: str) -> list[frozenset[str]]:
    return [frozenset(word_set(part)) for part in value.split("|") if word_set(part)]


def fact_elements(value: Any) -> list[str]:
    return re.split(r"[,;/|]", str(value))


# ---------------------------------------------------------------- records of the task


class Offer:
    def __init__(self, offer_id: str):
        self.id = offer_id
        self.title = ""
        self.text = ""
        self.price: float | None = None
        self.stock: bool | None = None
        self.category = ""
        self.brand = ""
        self.seller = ""
        self.facts: dict[str, Any] = {}
        self.notes = ""
        self.opened = False
        self.screen = "maybe"
        self.screened = False
        self.units: int | None = None

    def price_key(self) -> float:
        return self.price if self.price is not None else 1e18

    def card(self, limit: int = FACT_CHARS) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "title": self.title[:160]}
        for key, value in (("category", self.category), ("brand", self.brand), ("seller", self.seller)):
            if value:
                out[key] = value
        if self.price is not None:
            out["price"] = self.price
        if self.stock is not None:
            out["can_buy"] = self.stock
        if self.facts:
            out["properties"] = self.facts
        if self.notes:
            out["seller_notes"] = self.notes[:600]
        if self.text:
            out["description"] = self.text[:400] if self.facts else self.text[:500]
        size = len(json.dumps(out, ensure_ascii=False))
        if self.facts and size > limit:
            share = max(6, int(len(self.facts) * limit / size))
            out["properties"] = dict(list(self.facts.items())[:share])
        if out.get("description") and len(json.dumps(out, ensure_ascii=False)) > limit:
            out["description"] = out["description"][:200]
        return out

    def all_text(self) -> str:
        return " ".join([self.title or "", self.text or "", self.notes or "", self.category or "",
                         " ".join(f"{k} {v}" for k, v in self.facts.items())])


class Need:
    def __init__(self, prop: Any, op: Any, value: Any, reference: Any = None, origin: str = "request"):
        self.prop = str(prop or "").strip()
        self.op = str(op or "is").strip().lower()
        self.value = str(value or "").strip()
        self.reference = reference if isinstance(reference, dict) else None
        self.origin = origin

    def pending(self) -> bool:
        return bool(self.reference) and self.op in REF_OPS

    def describe(self) -> str:
        if self.op in REF_OPS and self.reference:
            return f"{self.prop}: {self.op} (not yet resolved)"
        return f"{self.prop}: {self.op} {self.value}"

    def key(self) -> str:
        return plain(self.prop)


class Wanted:
    def __init__(self, spec: dict[str, Any], index: int):
        self.index = index
        self.name = str(spec.get("name") or spec.get("category") or "the product").strip()
        self.category = str(spec.get("category") or self.name).strip()
        count = spec.get("quantity")
        self.quantity = int(count) if whole_count(count) and count >= 1 else 1
        self.needs: list[Need] = [Need(r.get("property"), r.get("op"), r.get("value"), r.get("reference"))
                                  for r in spec.get("requirements") or [] if isinstance(r, dict) and r.get("property")]
        self.hints = [str(h) for h in spec.get("hints") or [] if h]
        self.terms = [str(t) for t in spec.get("search_terms") or [] if t] or [self.category]
        self.noun = self.terms[-1]
        self.pool: dict[str, Offer] = {}
        self.verdicts: dict[str, dict[str, set[str]]] = {}
        self.judged_for: dict[str, set[str]] = {}
        self.asked: list[str] = []
        self.search_done = 0
        self.chosen: Offer | None = None
        self.labels_ok: set[str] | None = None
        self.label_heard = False
        self.closest = None
        self.cursors: dict[str, dict[str, Any]] = {}
        self.capped = False
        self.narrowed = False
        self.sweep_below: float | None = None
        self.need_search_done = False
        self.suggested = False

    def live_needs(self) -> list[Need]:
        return [n for n in self.needs if not n.pending()]

    def need_lines(self) -> list[str]:
        return [n.describe() for n in self.live_needs()]

    def plain_values(self, shopper: bool | None = None) -> list[str]:
        out = []
        for n in self.live_needs():
            if shopper is not None and (n.origin == "shopper") != shopper:
                continue
            if n.value and n.op in ("is", "one of"):
                out.append(n.value.split("|")[0].strip())
        return out



# ---------------------------------------------------------------- services


class Service:
    def link(self, **peers: Any) -> "Service":
        for name, peer in peers.items():
            setattr(self, name, peer)
        return self


class Clock(Service):
    def __init__(self):
        self.started = time.monotonic()

    def left(self) -> float:
        return TASK_SECONDS - (time.monotonic() - self.started)

    def stamp(self) -> float:
        return round(time.monotonic() - self.started, 1)


class Journal(Service):
    def __init__(self):
        self.entries: list[dict[str, Any]] = []

    def note(self, kind: str, **fields: Any) -> None:
        entry = {"kind": kind, "step": self.world.step, "t": self.clock.stamp()}
        entry.update(fields)
        self.entries.append(entry)


class LLMDesk(Service):
    def __init__(self):
        self.cost = 0.0
        self.calls = 0

    def over_cap(self) -> bool:
        return self.cost >= SPEND_CAP

    @staticmethod
    def _answered(data: Any) -> bool:
        return isinstance(data, dict) and bool(data.get("choices"))

    def text(self, model: str, system: str, user: str, max_tokens: int) -> str | None:
        if self.cost >= SOFT_SPEND and model in (PARSE_MODEL, PLAN_MODEL, VERIFY_MODEL):
            model = FAST_MODEL
        left = self.clock.left()
        if left < WRAP_SECONDS - 12:
            return None
        _llm.timeout = int(max(10, min(45, left - WRAP_SECONDS + 20)))
        system_part: Any = system
        if model.startswith("anthropic/"):
            system_part = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system_part}, {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        self.calls += 1
        route = "/inference/chat/completions"
        result = _llm.post_verbose(route, body)
        data, failure = result.data, result.error
        for wait in (2.0, 4.0):
            # only refusals (rate limit, gateway trouble) are worth another try
            if self._answered(data):
                break
            error = failure if isinstance(failure, dict) else {}
            if error.get("status") not in (429, 500, 502, 503, 504) or self.clock.left() < WRAP_SECONDS + 10:
                break
            time.sleep(wait)
            if wait < 4.0:
                result = _llm.post_verbose(route, body)
                data, failure = result.data, result.error
            else:
                # the last try: no decision follows it, so the plain reply is enough
                data, failure = _llm.post(route, body), "no reply after the last retry"
        if self._answered(data):
            spent = (data.get("usage") or {}).get("cost")
            if isinstance(spent, (int, float)):
                self.cost += float(spent)
            content = (data["choices"][0].get("message") or {}).get("content")
            if isinstance(content, list):
                content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
            if content:
                return str(content)
        self.journal.note("llm_failure", model=model, error=str(failure or "")[:200])
        return None

    def json(self, models: list[str], system: str, user: str, max_tokens: int) -> Any:
        for model in models:
            parsed = first_json(self.text(model, system, user, max_tokens) or "")
            if parsed is not None:
                return parsed
        return None


class ToolBook(Service):
    def __init__(self, tools: list[dict[str, Any]]):
        self.tools = tools
        self.schemas: dict[str, dict[str, Any]] = {}
        for fn in map(self._function, tools):
            if isinstance(fn, dict) and fn.get("name"):
                self.schemas[fn["name"]] = fn.get("parameters") or {"type": "object", "properties": {}}
        self.roles: dict[str, Any] = {}
        self.choice_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _function(tool: dict[str, Any]) -> Any:
        return tool.get("function") if isinstance(tool.get("function"), dict) else tool

    def listing(self) -> list[dict[str, Any]]:
        out = []
        for tool in self.tools:
            fn = self._function(tool)
            out.append({"name": fn.get("name"), "description": fn.get("description", ""), "parameters": fn.get("parameters")})
        return out

    def describe(self, name: str) -> str:
        for fn in map(self._function, self.tools):
            if isinstance(fn, dict) and fn.get("name") == name:
                return str(fn.get("description") or "")
        return ""

    def name_of(self, role: str) -> Any:
        return (self.roles.get(role) or {}).get("name")

    def names_of(self, *roles: str) -> set[Any]:
        return {self.name_of(role) for role in roles}

    def schema(self, role: str) -> dict[str, Any]:
        return self.schemas.get((self.roles.get(role) or {}).get("name") or "", {})

    def props(self, role: str) -> dict[str, Any]:
        return self.schema(role).get("properties") or {}

    def arg_limit(self, role: str, arg: Any, field: str) -> Any:
        return (self.props(role).get(arg) or {}).get(field)

    def page_mode(self, role: str) -> str | None:
        spec = self.roles.get(role) or {}
        page = spec.get("page")
        if not page:
            return None
        if self.arg_limit(role, page, "type") != "integer":
            return "token"
        kind = str(spec.get("page_kind") or "").lower()
        if kind == "offset":
            return "offset"
        if kind != "number" and any(mark in page.lower() for mark in ("offset", "start", "skip")):
            return "offset"
        return "index"

    def enum_value(self, role: str, arg: str, value: Any) -> Any:
        allowed = self.arg_limit(role, arg, "enum")
        if not isinstance(value, str) or not isinstance(allowed, list) or not allowed:
            return value
        spelled = {plain(e): e for e in allowed if isinstance(e, str)}
        if not spelled:
            return value
        if plain(value) in spelled:
            return spelled[plain(value)]
        closest = max(spelled, key=lambda e: overlap_ratio(e, value))
        return spelled[closest] if overlap_ratio(closest, value) >= 0.5 else None

    def trim(self, role: str, arg: str, value: str) -> str:
        limit = self.arg_limit(role, arg, "maxLength")
        if not isinstance(limit, int) or len(value) <= limit:
            return value
        head = value[:limit]
        return head.rsplit(" ", 1)[0] if " " in head else head

    @staticmethod
    def _typed_key(props: dict[str, Any], kind: str) -> Any:
        return next((k for k, v in props.items() if isinstance(v, dict) and v.get("type") == kind), None)

    def settle_roles(self) -> None:
        for role in ("cart_add", "cart_set"):
            spec = self.roles.get(role)
            if not spec:
                continue
            props = (self.schemas.get(spec["name"]) or {}).get("properties") or {}
            line_lists = [k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "array"
                          and isinstance(v.get("items"), dict) and v["items"].get("type") == "object"]
            if not line_lists:
                if role == "cart_add":
                    if spec.get("id") not in props:
                        spec["id"] = self._typed_key(props, "string")
                    if spec.get("qty") and spec["qty"] not in props:
                        spec["qty"] = self._typed_key(props, "integer")
                    if ARG_FILL and not spec.get("qty"):
                        # a count the schema demands is the quantity, whatever the map said
                        mode_arg = (spec.get("mode") or {}).get("arg") if isinstance(spec.get("mode"), dict) else None
                        counts = [k for k in (self.schemas.get(spec["name"]) or {}).get("required") or []
                                  if k not in (spec.get("id"), mode_arg)
                                  and (props.get(k) or {}).get("type") in ("integer", "number")]
                        if counts:
                            spec["qty"] = counts[0]
                    if not spec.get("id"):
                        self.roles.pop("cart_add", None)
                continue
            # a cart tool taking a list of lines replaces the whole cart
            lines_arg = line_lists[0]
            inner = props[lines_arg]["items"].get("properties") or {}
            keep_own = role == "cart_set"
            id_key = spec.get("id") if keep_own and spec.get("id") in inner else self._typed_key(inner, "string")
            qty_key = spec.get("qty") if keep_own and spec.get("qty") in inner else self._typed_key(inner, "integer")
            if id_key:
                self.roles["cart_set"] = {"name": spec["name"], "lines": lines_arg, "id": id_key, "qty": qty_key}
            if not keep_own:
                self.roles.pop("cart_add", None)
        for have, other in (("checkout", "decline"), ("decline", "checkout")):
            spec = self.roles.get(have)
            if not spec or other in self.roles:
                continue
            schema = self.schemas.get(spec["name"]) or {}
            props = schema.get("properties") or {}
            if any(isinstance((props.get(k) or {}).get("enum"), list) for k in schema.get("required") or []):
                self.roles[other] = {"name": spec["name"]}

    def ending_args(self, role: str) -> dict[str, Any]:
        spec = self.roles.get(role) or {}
        schema = self.schema(role)
        props = schema.get("properties") or {}
        chosen: dict[str, Any] = {}
        given = spec.get("args") if isinstance(spec.get("args"), dict) else {}
        for arg, value in given.items():
            if arg not in props or value in (None, ""):
                continue
            allowed = props[arg].get("enum")
            if isinstance(allowed, list) and allowed and value not in allowed:
                closest = max(allowed, key=lambda e: overlap_ratio(str(e), str(value)))
                if overlap_ratio(str(closest), str(value)) < 0.5:
                    continue
                value = closest
            chosen[arg] = value
        open_args = [k for k in schema.get("required") or []
                     if k not in chosen and isinstance((props.get(k) or {}).get("enum"), list)]
        if not open_args:
            return chosen
        picked = self.choice_cache.get(role)
        if picked is None:
            outcome = "buy what the cart holds now, as the order" if role == "checkout" else "close the task and buy nothing at all"
            user = (f"TOOL NAME: {spec.get('name')}\nWHAT IT DOES: {self.describe(spec.get('name') or '')}\n"
                    f"ARGUMENTS STILL NEEDED: {json.dumps({k: props[k] for k in open_args}, ensure_ascii=False)}\n"
                    f"WANTED OUTCOME: {outcome}")
            answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], ENDING_CHOICE, user, 300)
            picked = answer if isinstance(answer, dict) else {}
            if picked:
                self.choice_cache[role] = picked
        for arg in open_args:
            value = picked.get(arg)
            if value in props[arg]["enum"]:
                chosen[arg] = value
        return chosen


class WorldDesk(Service):
    def __init__(self, binding: dict[str, Any], max_steps: int, finished: bool):
        self.binding = binding
        self.max_steps = max_steps
        self.step = 0
        self.finished = finished
        self.dead = False
        self.pending_lines: list[str] = []
        self.handled_lines: set[str] = set()
        self.history: list[dict[str, Any]] = []

    def steps_left(self) -> int:
        return self.max_steps - self.step

    def _blocked(self, name: str) -> str | None:
        if self.finished or self.dead or name not in self.tools.schemas:
            return "unavailable"
        ending = name in self.tools.names_of("checkout", "decline")
        left = self.steps_left()
        if left <= 0 or (left <= 1 and not ending):
            return "no steps left"
        return None

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"obs": None, "lines": [], "error": None, "ok": False}
        refusal = self._blocked(name)
        if refusal:
            out["error"] = refusal
            return out
        payload: dict[str, Any] = dict(self.binding)
        payload["step"] = self.step + 1
        payload["action"] = {"name": name, "args": args}
        for _ in range(3):
            left = self.clock.left()
            if left < 3:
                out["error"] = "no time"
                return out
            _env.timeout = int(max(4, min(90, left - 1)))
            result = _env.post_verbose("/environment/call", payload)
            data = result.data
            if isinstance(data, dict) and ("observation" in data or "finished" in data):
                self._took_step(name, args, data, out)
                return out
            error = result.error or {}
            status = error.get("status") if isinstance(error, dict) else None
            body = str(error.get("body") if isinstance(error, dict) else error)[:300]
            self.journal.note("env_failure", status=status, body=body)
            if status == 402:
                self.dead = True
                out["error"] = "payment"
                return out
            if status in (403, 422):
                out["error"] = f"refused {status}"
                return out
            if status != 409:
                continue
            if "finished" in body:
                self.finished = True
                out["error"] = "finished"
                return out
            expected = re.search(r"expected (\d+)", body)
            if not expected:
                out["error"] = "conflict"
                return out
            self.step = int(expected.group(1)) - 1
            payload["step"] = self.step + 1
        out["error"] = "env failure"
        return out

    def _took_step(self, name: str, args: dict[str, Any], data: dict[str, Any], out: dict[str, Any]) -> None:
        self.step = data["step"] if isinstance(data.get("step"), int) else self.step + 1
        if data.get("finished"):
            self.finished = True
        obs, lines = split_shopper_lines(data.get("observation"))
        self.pending_lines.extend(line for line in lines if line not in self.handled_lines)
        out["obs"] = obs
        out["lines"] = lines
        if isinstance(obs, dict) and isinstance(obs.get("error"), str) and len(obs) <= 2:
            out["error"] = obs["error"]
        else:
            out["ok"] = True
        self.journal.note("call", tool=name, args=squeeze(args, 160), result=squeeze(obs, 200), lines=lines[:3])
        to_shopper = name == self.tools.name_of("message")
        record = {"step": self.step, "tool": name, "args": squeeze(args, 300 if to_shopper else 120),
                  "result": squeeze(obs, 160)}
        if lines:
            record["shopper_lines"] = [line[:200] for line in lines[:3]]
        if to_shopper:
            record["basket_then"] = [list(pair) for pair in self.cart.signature()]
        self.history.append(record)


class Catalogue(Service):
    def __init__(self, world_key: str):
        self.world_key = world_key
        learned = _WORLD_MEMORY.get(world_key) or {}
        self.maps: dict[str, dict[str, str]] = dict(learned.get("maps") or {})
        # several stores refuse more than 4 ids per compare call whatever the schema says
        self.compare_limit = int(learned.get("compare_limit") or 4)
        self.thin_compare = bool(learned.get("thin_compare"))
        self.compare_off = False
        self.compare_checked = False
        self.page_size = 0

    def remember(self) -> None:
        _WORLD_MEMORY[self.world_key] = {"maps": self.maps, "compare_limit": self.compare_limit,
                                         "thin_compare": self.thin_compare}

    def batch_open(self) -> bool:
        return bool(self.tools.roles.get("compare")) and not self.compare_off and not self.thin_compare

    @staticmethod
    def _shown(value: Any) -> Any:
        if isinstance(value, (dict, list)):
            value = squeeze(value, 160)
        if isinstance(value, str) and len(value) > 160:
            value = value[:160] + "…"
        return value

    def _borrowed(self, kind: str, sample: dict[str, Any]) -> dict[str, str] | None:
        for known_key, known in self.maps.items():
            if not known or not known_key.startswith(kind + ":"):
                continue
            hits = sum(1 for key in known.values() if key in sample)
            if hits >= max(2, int(0.6 * len(known))):
                return known
        return None

    def key_roles(self, kind: str, sample: dict[str, Any]) -> dict[str, str]:
        signature = kind + ":" + ",".join(sorted(sample.keys()))
        if signature in self.maps:
            return self.maps[signature]
        borrowed = self._borrowed(kind, sample)
        if borrowed is not None:
            self.maps[signature] = borrowed
            return borrowed
        shown = {key: self._shown(value) for key, value in sample.items()}
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], KEY_MAPPING, json.dumps(shown, ensure_ascii=False), 700)
        roles: dict[str, str] = {}
        if isinstance(answer, dict):
            roles = {role: key for role, key in answer.items() if isinstance(key, str) and key in sample}
        self._guess_missing(roles, sample)
        self.maps[signature] = roles
        self.remember()
        return roles

    @staticmethod
    def _guess_missing(roles: dict[str, str], sample: dict[str, Any]) -> None:
        tests = (
            ("facts", lambda k, v: isinstance(v, dict) and v
             and all(isinstance(x, (str, int, float)) or x is None for x in v.values())),
            ("id", lambda k, v: isinstance(v, str) and re.fullmatch(r"[A-Za-z]{0,3}-?[0-9a-f]{8,}", v)),
            ("price", lambda k, v: whole_count(v) and "price" in k.lower()),
        )
        for role, test in tests:
            if role in roles:
                continue
            hit = next((k for k, v in sample.items() if test(k, v)), None)
            if hit is not None:
                roles[role] = hit

    def offer_from(self, rec: dict[str, Any], kind: str) -> Offer | None:
        roles = self.key_roles(kind, rec)
        id_key = roles.get("id")
        if not id_key or not isinstance(rec.get(id_key), (str, int)):
            return None
        offer = Offer(str(rec[id_key]))
        for attr in ("title", "text", "category", "brand", "seller", "notes"):
            key = roles.get(attr)
            if key and isinstance(rec.get(key), str):
                setattr(offer, attr, rec[key])
        if roles.get("price"):
            offer.price = as_number(rec.get(roles["price"]))
        if roles.get("stock") and isinstance(rec.get(roles["stock"]), bool):
            offer.stock = rec[roles["stock"]]
        elif roles.get("stock") and whole_count(rec.get(roles["stock"])):
            # a count of units on hand: none left means it cannot be bought
            offer.units = int(rec[roles["stock"]])
            offer.stock = offer.units > 0
        if roles.get("facts") and isinstance(rec.get(roles["facts"]), dict):
            offer.facts = {str(k): v for k, v in rec[roles["facts"]].items() if v not in (None, "")}
        return offer

    def file_offer(self, item: Wanted, offer: Offer, opened: bool) -> Offer:
        known = item.pool.setdefault(offer.id, offer)
        if known is not offer:
            for attr in ("title", "text", "category", "brand", "seller", "notes"):
                fresh = getattr(offer, attr)
                if fresh and (opened or not getattr(known, attr)):
                    setattr(known, attr, fresh)
            if offer.price is not None:
                known.price = offer.price
            if offer.stock is not None:
                known.stock = offer.stock
            if offer.units is not None:
                known.units = offer.units
            if offer.facts:
                if offer.facts != known.facts:
                    # changed records are judged again
                    item.judged_for.pop(known.id, None)
                known.facts = offer.facts
        if opened:
            known.opened = True
        return known

    def category_ok(self, item: Wanted, offer: Offer) -> bool:
        if not offer.category or not item.category:
            return True
        label = plain(offer.category)
        if item.labels_ok:
            return label in item.labels_ok
        for wanted in {plain(item.category), plain(item.name)} if NAME_LABEL else {plain(item.category)}:
            if wanted and (label in wanted or wanted in label or overlap_ratio(label, wanted) >= 0.5):
                return True
        return False

    def settle_labels(self, item: Wanted) -> None:
        labels = sorted({o.category for o in item.pool.values() if o.category})
        if labels and item.labels_ok is not None:
            # a label seen later that names the asked category itself is always accepted
            wanted = plain(item.category)
            late = {plain(label) for label in labels if plain(label) not in item.labels_ok
                    and (plain(label) == wanted or same_name(label, item.category))}
            if late:
                item.labels_ok |= late
                self.journal.note("labels_added", item=item.name, labels=sorted(late))
            return
        if not labels:
            return
        by_plain = {plain(label): label for label in labels}
        if len(labels) == 1:
            only, wanted = plain(labels[0]), plain(item.category)
            if overlap_ratio(labels[0], item.category) >= 0.3 or wanted in only or only in wanted:
                item.labels_ok = set(by_plain)
                return
            if not NAME_LABEL:
                return
            # one label that shares no words with the category: the label reader rules on it once
            if item.label_heard:
                return
            item.label_heard = True
        user = (f"PRODUCT ASKED FOR: {item.name} (worded as a category: {item.category})\n"
                f"FROM THE REQUEST: {self.brief.request[:600]}\n"
                f"SECTION LABELS SEEN IN THE SHOP: {json.dumps(labels, ensure_ascii=False)}")
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], LABEL_PICKING, user, 500)
        picked: set[str] = set()
        if isinstance(answer, dict) and isinstance(answer.get("labels"), list):
            picked = {plain(label) for label in answer["labels"] if isinstance(label, str) and plain(label) in by_plain}
        if picked:
            item.labels_ok = picked
            self.journal.note("labels", item=item.name, labels=sorted(picked))

    def affordable(self, item: Wanted, offer: Offer) -> bool:
        cap = self.brief.cap
        return cap is None or offer.price is None or offer.price * item.quantity <= cap + 1e-6

    def candidates(self, item: Wanted) -> list[Offer]:
        keep = [o for o in item.pool.values()
                if not (o.stock is False and self.brief.in_stock_only)
                and self.affordable(item, o) and self.category_ok(item, o)]
        rank = {"yes": 0, "no": 2}
        keep.sort(key=lambda o: (rank.get(o.screen, 1), o.price_key()))
        return keep

    def unopened(self, item: Wanted) -> list[Offer]:
        return [o for o in self.candidates(item) if not o.opened]

    def opened(self, item: Wanted) -> list[Offer]:
        return [o for o in self.candidates(item) if o.opened]

    def screen(self, item: Wanted, offers: list[Offer]) -> None:
        todo = [o for o in offers if not o.opened and o.screen == "maybe" and not o.screened]
        if len(todo) < 8 or not item.need_lines():
            return
        batch = todo[:30]
        rows = [{"n": n, "title": o.title[:140], "text": (o.text or "")[:ROW_CHARS], "category": o.category, "price": o.price}
                for n, o in enumerate(batch, 1)]
        user = (f"PRODUCT KIND: {item.category}\nDEMANDS:\n- " + "\n- ".join(item.need_lines())
                + "\n\nNUMBERED ROWS:\n" + json.dumps(rows, ensure_ascii=False))
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], ROW_SCREENING, user, 1200)
        for offer in batch:
            offer.screened = True
        if not isinstance(answer, dict):
            return
        for verdict in ("no", "yes"):
            for n in answer.get(verdict) or []:
                if isinstance(n, (int, str)) and str(n).strip().isdigit() and 1 <= int(n) <= len(batch):
                    batch[int(n) - 1].screen = verdict


class Browser(Service):
    def shaped_query(self, item, role, arg, query):
        # value words lead (a store may read only the first words); category words follow
        kind_words = word_set(item.category) | word_set(item.name)
        values, rest = [], []
        for word in query.split():
            words = word_set(word)
            (rest if words and words <= kind_words else values).append(word)
        joined = " ".join(values + rest)
        limit = self.tools.arg_limit(role, arg, "maxLength")
        if not isinstance(limit, int) or len(joined) <= limit:
            return joined
        kept = values[:]
        for word in rest:
            if len(" ".join(kept + [word])) <= limit:
                kept.append(word)
        return self.tools.trim(role, arg, " ".join(kept) if kept else joined)

    def _field_args(self, role, spec, fields, args):
        for key, value in (fields or {}).items():
            arg = spec.get(key)
            if not arg or value in (None, ""):
                continue
            spelled = self.tools.enum_value(role, arg, value)
            if spelled is None:
                return False
            args[arg] = spelled if not isinstance(spelled, str) else self.tools.trim(role, arg, spelled)
        return True

    def _request(self, item, role, spec, query, fields):
        args = {}
        switch = spec.get("mode")
        if switch and isinstance(switch, dict) and switch.get("arg"):
            args[switch["arg"]] = switch.get("value")
        if query is None:
            return args if self._field_args(role, spec, fields, args) else "unlisted"
        if not spec.get("query"):
            return "no query"
        args[spec["query"]] = self.shaped_query(item, role, spec["query"], query)
        return args

    def _turn_page(self, item, role, spec, args, page, mode, token):
        if mode == "index":
            moved = page
        elif mode == "offset":
            moved = (page - 1) * max(self.book.page_size, 1)
        elif mode == "token" and token:
            moved = token
        else:
            return False
        args[spec["page"]] = moved
        top = self.tools.arg_limit(role, spec["page"], "maximum")
        if isinstance(moved, int) and isinstance(top, (int, float)) and moved > top:
            # the store shows no further pages: what it hides stays hidden
            item.capped = True
            return False
        return True

    @staticmethod
    def _read_result(obs):
        if not isinstance(obs, dict):
            return None, None
        words = [v for v in obs.values() if isinstance(v, str) and 0 < len(v) <= 20 and " " not in v]
        # the match count is the biggest number next to the rows
        numbers = [v for v in obs.values() if isinstance(v, int) and not isinstance(v, bool)]
        return (words[-1] if words else None), (max(numbers) if numbers else None)

    def _file_rows(self, item, rows):
        fresh = 0
        if rows:
            self.book.page_size = max(self.book.page_size, len(rows))
        for offer in map(lambda rec: self.book.offer_from(rec, "row"), rows):
            if offer is not None:
                fresh += offer.id not in item.pool
                self.book.file_offer(item, offer, opened=False)
        return fresh

    def _exhausted(self, item, rows, page, token, total, mode):
        if not rows or len(rows) < self.book.page_size:
            return True
        if total is not None and page * max(len(rows), 1) >= total:
            return True
        if mode != "token" or token:
            return False
        if total is not None and total > page * len(rows):
            item.capped = True
        return True

    def search(self, item, query=None, fields=None, max_pages=1, budget=None):
        role = "search_text" if query is not None else "search_fields"
        spec = self.tools.roles.get(role)
        if not spec:
            return 0
        mode = self.tools.page_mode(role)
        sig = role + ":" + (query if query is not None else json.dumps(fields or {}, sort_keys=True))
        cursor = item.cursors.get(sig) or {}
        if cursor.get("done"):
            return 0
        token = cursor.get("token")
        page = cursor.get("page", 0)
        if page >= 1 and mode is None:
            return 0
        request = self._request(item, role, spec, query, fields)
        last = page + max_pages
        added, done = 0, False
        while page < last:
            page += 1
            if self.world.steps_left() <= (budget or 0) or self.clock.left() < WRAP_SECONDS:
                page -= 1
                break
            if request == "no query":
                return 0
            if isinstance(request, str):
                item.cursors[sig] = {"page": 0, "token": None, "done": True}
                return added
            args = dict(request)
            if page > 1 and not self._turn_page(item, role, spec, args, page, mode, token):
                done = True
                page -= 1
                break
            res = self.world.call(spec["name"], args)
            if not res["ok"]:
                self.journal.note("search_error", error=res["error"])
                done = True
                break
            rows = largest_record_list(res["obs"]) or []
            token, total = self._read_result(res["obs"])
            added += self._file_rows(item, rows)
            item.search_done += 1
            if item.index != 99:
                self.book.settle_labels(item)
            if self._exhausted(item, rows, page, token, total, mode):
                done = True
                break
        item.cursors[sig] = {"page": max(page, 0), "token": token, "done": done or mode is None}
        return added

    def facts_of(self, offer_id):
        tool = self.tools.roles.get("facts")
        if not tool:
            return None
        res = self.world.call(tool["name"], {tool["id"]: offer_id})
        obs = res["obs"]
        if not res["ok"] or not isinstance(obs, dict):
            return None
        table_key = self.book.key_roles("facts", obs).get("facts")
        source = {}
        if table_key and isinstance(obs.get(table_key), dict):
            source = obs[table_key]
        elif all(v is None or isinstance(v, (str, int, float)) for v in obs.values()):
            source = obs
        return {str(k): v for k, v in source.items() if v not in (None, "")}

    def open_by_facts(self, item, offer_id):
        known = item.pool.get(offer_id)
        facts = self.facts_of(offer_id)
        if facts is None or (not facts and known is None):
            return None
        offer = Offer(offer_id)
        offer.facts = facts
        return self.book.file_offer(item, offer, True)

    def _short(self, floor, cost):
        return bool(floor) and self.world.steps_left() - cost < floor

    def _compare_round(self, item, chunk, compare, facts_tool, floor, opened):
        book = self.book
        res = self.world.call(compare["name"], {compare["ids"]: chunk})
        records = largest_record_list(res["obs"]) if res["ok"] else None
        if not records:
            if not (res["error"] or res["ok"]):
                return "single"
            # fewer ids per call, then none at all
            if book.compare_limit > 2 and len(chunk) > 2:
                book.compare_limit = 2 if book.compare_limit <= 4 else 4
                book.remember()
                return "retry"
            book.compare_off = True
            return "off"
        offers = [o for o in (book.offer_from(rec, "detail") for rec in records) if o is not None]
        thin = bool(facts_tool) and bool(offers) and not any(o.facts for o in offers)
        if thin and not book.thin_compare:
            book.thin_compare = True
            book.remember()
            self.journal.note("thin_compare", tool=compare["name"], facts_tool=facts_tool.get("name"))
        if not (thin and item is not None):
            for offer in offers:
                offer.opened = True
                opened.append(book.file_offer(item, offer, True) if item else offer)
            return "advance"
        for offer in offers:
            book.file_offer(item, offer, False)
        for offer in offers:
            if self._short(floor, 1) or self.clock.left() < WRAP_SECONDS:
                break
            got = self.open_by_facts(item, offer.id)
            if got is not None:
                opened.append(got)
        return "advance"

    def _open_single(self, item, open_tool, facts_tool, offer_id):
        res = self.world.call(open_tool["name"], {open_tool["id"]: offer_id})
        if not res["ok"] or not isinstance(res["obs"], dict):
            return None
        offer = self.book.offer_from(res["obs"], "detail") or Offer(offer_id)
        if not offer.facts and facts_tool and self.world.steps_left() > 0:
            offer.facts = self.facts_of(offer_id) or {}
        offer.id = offer_id
        offer.opened = True
        return self.book.file_offer(item, offer, True) if item else offer

    def open_many(self, item, ids, floor=0):
        queue = [i for i in ids if i]
        opened = []
        if not queue:
            return opened
        book, roles = self.book, self.tools.roles
        compare = roles.get("compare") if not book.compare_off else None
        facts_tool, open_tool = roles.get("facts"), roles.get("open")
        if compare and not book.compare_checked:
            book.compare_checked = True
            most = self.tools.arg_limit("compare", compare.get("ids") or "", "maxItems")
            if isinstance(most, int) and 2 <= most < book.compare_limit:
                book.compare_limit = most
        while queue and self.world.steps_left() > 0 and self.clock.left() > WRAP_SECONDS:
            if book.thin_compare and facts_tool and item is not None:
                # this store's compare rows lack the table: each table is read alone
                if self._short(floor, 1):
                    break
                got = self.open_by_facts(item, queue.pop(0))
            elif compare and len(queue) >= 2:
                if self._short(floor, 1):
                    break
                chunk = queue[:book.compare_limit]
                outcome = self._compare_round(item, chunk, compare, facts_tool, floor, opened)
                if outcome == "advance":
                    del queue[:len(chunk)]
                if outcome == "off":
                    compare = None
                if outcome != "single":
                    continue
                got = self._single_step(item, open_tool, facts_tool, queue, floor)
            else:
                got = self._single_step(item, open_tool, facts_tool, queue, floor)
            if got is False:
                break
            if got is not None:
                opened.append(got)
        return opened

    def _single_step(self, item, open_tool, facts_tool, queue, floor):
        if not open_tool or self._short(floor, 2 if facts_tool else 1):
            return False
        return self._open_single(item, open_tool, facts_tool, queue.pop(0))


class Assessor(Service):
    def __init__(self):
        self.verified = {}

    def _read_with_tables(self, item):
        return [o for o in item.pool.values() if o.opened and o.facts and self.book.category_ok(item, o)]

    def drop_label_restatements(self, item):
        # a value that only repeats the store's own label for the item, on a key no read listing has, is the category
        if not item.labels_ok:
            return
        label_words = set()
        for label in item.labels_ok:
            label_words |= word_set(label)
        read = self._read_with_tables(item)
        if len(read) < 3:
            return
        known_keys = [str(k) for o in read for k in o.facts]

        def restates(need):
            words = word_set(need.value)
            if need.origin != "request" or need.op != "is" or not words or not words <= label_words:
                return False
            return not any(same_name(k, need.prop) or overlap_ratio(k.replace("_", " "), need.prop) >= 0.5
                           for k in known_keys)

        kept = []
        for need in item.needs:
            if restates(need):
                self.journal.note("label_restatement", item=item.name, req=need.describe())
                continue
            kept.append(need)
        item.needs = kept

    @staticmethod
    def unstated(item, offer, need):
        if need.origin == "fit" or not need.value or need.op not in ("is", "one of"):
            return False
        name_words = all_words(item.name) | all_words(item.category)
        # short alphabetic joiners prove nothing; figures do
        options = [{w for w in all_words(part) - name_words if not w.isalpha() or len(w) > 3}
                   for part in need.value.split("|")]
        options = [o for o in options if o]
        if not options:
            return False
        present = all_words(offer.all_text())
        return not any(o & present for o in options)

    def field_table(self, item):
        read = self._read_with_tables(item)
        if len(read) < 3:
            return []
        rows = []
        for need in item.live_needs():
            plain_value = need.op in ("is", "one of") and need.value and not re.search(r"\d", need.value)
            wants = value_options(need.value) if plain_value else []
            row = self._field_row(need, wants, read) if wants else None
            if row is not None:
                rows.append(row)
        return rows

    @staticmethod
    def _field_row(need, wants, read):
        keys, writers, values = set(), set(), {}
        carriers = 0
        for offer in read:
            hits = [k for k in offer.facts if same_name(k, need.prop)]
            carriers += bool(hits)
            keys.update(hits)
            elements = [" ".join(e.lower().split()) for key in hits for e in fact_elements(offer.facts[key])]
            for element in filter(None, elements):
                values[element] = values.get(element, 0) + 1
                if any(w <= frozenset(word_set(element)) for w in wants):
                    writers.add(offer.id)
        if carriers == 0:
            return None
        return {"req": need.describe(), "prop": need.prop, "keys": sorted(keys), "carriers": carriers, "read": len(read),
                "values": values, "wants": wants, "writers": writers}

    def field_fails(self, item, offer, table=None):
        rows = table if table is not None else self.field_table(item)
        failing = set()
        for row in rows:
            own = [k for k in offer.facts if same_name(k, row["prop"])]
            if not own or not row["writers"] - {offer.id}:
                continue
            elements = [frozenset(word_set(e)) for k in own for e in fact_elements(offer.facts[k]) if word_set(e)]
            carried = any(w <= e for e in elements for w in row["wants"])
            if elements and not carried:
                failing.add(row["req"])
        return failing

    @staticmethod
    def field_lines(table):
        out = []
        for row in table:
            common = sorted(row["values"].items(), key=lambda pair: -pair[1])[:8]
            written = " | ".join("%s (%s)" % pair for pair in common)
            keys = ", ".join(map(repr, row["keys"][:3]))
            out.append(f"- {row['req']}: key {keys} filled on {row['carriers']} of {row['read']} opened offers; "
                       f"written values: {written}; {len(row['writers'])} offer(s) hold the demanded value as such")
        return "\n".join(out)

    def related_pages(self, item, offer, limit=2):
        own_codes = set(PART_CODE.findall(offer.title or ""))
        mentioned = set(PART_CODE.findall(" ".join([offer.text or "", offer.notes or ""]))) - own_codes
        found, seen = [], {offer.id}
        others = list(item.pool.values())
        for other_item in self.brief.items:
            if other_item is not item:
                others += list(other_item.pool.values())
        for other in others:
            if len(found) >= limit or other.id in seen:
                continue
            other_text = " ".join([other.text or "", other.notes or ""])
            linked = set(PART_CODE.findall(other.title or "")) & mentioned or own_codes & set(PART_CODE.findall(other_text))
            if linked:
                seen.add(other.id)
                found.append({"id": other.id, "title": other.title[:120], "text": other_text[:400]})
        return found

    @staticmethod
    def _closest_line(name, lines):
        if not lines:
            return name
        closest = max(lines, key=lambda line: overlap_ratio(line, name))
        return closest if overlap_ratio(closest, name) > 0.2 else name

    def _to_lines(self, names, lines):
        out = set()
        for name in names:
            numbered = re.fullmatch(r"\s*R?(\d+)\s*", name)
            index = int(numbered.group(1)) if numbered else 0
            out.add(lines[index - 1] if 1 <= index <= len(lines) else self._closest_line(name, lines))
        return out

    def _judge_prompt(self, item, lines, fields, batch):
        cards = []
        for offer in batch:
            card = offer.card()
            related = self.related_pages(item, offer)
            if related:
                card["related_pages"] = related
            cards.append(card)
        numbered = "\n".join("R%d: %s" % (n, line) for n, line in enumerate(lines, 1))
        return (f"PRODUCT KIND: {item.category}\nDEMANDS TO RULE ON (each one):\n" + numbered + fields
                + "\n\nOFFERS:\n" + json.dumps(cards, ensure_ascii=False))

    def drop_record_fields(self, item):
        # a "requirement" on the shop's own stock or price field is no product demand: the cart rules handle those
        keys = {key for roles in self.book.maps.values() for role, key in roles.items()
                if role in ("stock", "price", "total") and isinstance(key, str)}
        groups = [{w for w in word_set(camel_spaced(k).replace("_", " ")) if len(w) >= 4} for k in keys]
        groups = [g for g in groups if g]
        if not groups:
            return

        def names_field(need):
            words = {w for w in word_set(need.prop) if len(w) >= 4}
            return bool(words) and any(g <= words or words <= g for g in groups)

        gone = [n for n in item.needs if not n.reference and names_field(n)]
        if gone:
            item.needs = [n for n in item.needs if n not in gone]
            self.journal.note("record_field_needs", item=item.name, dropped=[n.describe() for n in gone])

    def judge(self, item, offers):
        self.drop_label_restatements(item)
        self.drop_record_fields(item)
        lines = item.need_lines()
        if not lines:
            # only the kind of product counts: every opened listing stands
            for offer in (o for o in offers if o.opened and o.id not in item.verdicts):
                item.verdicts[offer.id] = {"failed": set(), "unsure": set()}
                item.judged_for[offer.id] = set()
            return
        wanted = set(lines)
        todo = [o for o in offers if o.opened and not item.judged_for.get(o.id, set()) >= wanted]
        if not todo or self.clock.left() < WRAP_SECONDS:
            return
        table = self.field_table(item)
        fields = ""
        if table:
            fields = "\n\nSTORE FIELDS (how this shop's opened records write them so far):\n" + self.field_lines(table)
        batches = []
        while todo:
            batches.append(todo[:4])
            todo = todo[4:]
        jobs = []
        for batch in batches:
            prompt = self._judge_prompt(item, lines, fields, batch)
            jobs.append(lambda u=prompt: self.llm.json([FAST_MODEL, BACKUP_MODEL], LISTING_JUDGING, u, 2000))
        for batch, answer in zip(batches, run_parallel(jobs)):
            verdicts = answer.get("verdicts") if isinstance(answer, dict) else None
            by_id = {}
            for verdict in verdicts if isinstance(verdicts, list) else []:
                if isinstance(verdict, dict) and verdict.get("id") in item.pool:
                    by_id[verdict["id"]] = verdict
            for offer in batch:
                if offer.id in by_id:
                    self._record(item, offer, by_id[offer.id], lines, table)

    def _record(self, item, offer, verdict, lines, table):
        listed = {key: {str(x) for x in (verdict.get(key) or []) if x} for key in ("failed", "unsure", "weak", "loose")}
        ruling = {"failed": self._to_lines(listed["failed"], lines) | self.field_fails(item, offer, table),
                  "unsure": self._to_lines(listed["unsure"], lines),
                  "weak": self._to_lines(listed["weak"] | listed["loose"], lines),
                  "loose": self._to_lines(listed["loose"], lines)}
        item.verdicts[offer.id] = ruling
        item.judged_for[offer.id] = set(lines)
        settled = ruling["failed"] | ruling["unsure"] | ruling["loose"]
        for need in item.live_needs():
            line = need.describe()
            if line not in settled and self.unstated(item, offer, need):
                ruling["unsure"].add(line)

    def judge_opened(self, item):
        self.judge(item, self.book.opened(item))

    def judge_all(self):
        run_parallel([lambda it=it: self.judge(it, self.book.opened(it)) for it in self.brief.items])

    @staticmethod
    def exactness(item, offer):
        if not offer.facts:
            return 0
        elements = set()
        for value in offer.facts.values():
            elements.update(frozenset(word_set(e)) for e in fact_elements(value) if word_set(e))
        return sum(1 for need in item.live_needs()
                   if need.op in ("is", "one of") and need.value
                   and any(w in elements for w in value_options(need.value)))

    def fits(self, item, include_unsure=False, cheapest=False):
        wanted = set(item.need_lines())
        if NO_NEED_ROWS and not wanted and not any(n.pending() for n in item.needs):
            # nothing left to check on a page: the shelf, the stock flag and the price decide
            rows = [o for o in self.book.candidates(item) if o.stock is True and o.price is not None and o.category]
            if rows:
                rows.sort(key=lambda o: o.price_key() if cheapest else (not o.opened, o.price_key()))
                return rows
        sure, unsure = [], []
        for offer in (o for o in item.pool.values() if o.opened):
            ruling = item.verdicts.get(offer.id)
            if ruling is None or ruling["failed"] or not item.judged_for.get(offer.id, set()) >= wanted:
                continue
            if ruling["unsure"]:
                unsure.append(offer)
            else:
                sure.append(offer)
        # listings stating the values word for word lead; table evidence beats title evidence
        if cheapest and CHEAP_BY_PRICE:
            sure.sort(key=lambda o: (o.price_key(), -self.exactness(item, o)))
        elif cheapest:
            sure.sort(key=lambda o: (-self.exactness(item, o), o.price_key()))
        else:
            sure.sort(key=lambda o: (-self.exactness(item, o), len(item.verdicts[o.id].get("weak") or ()), o.price_key()))
        if sure or not include_unsure:
            return sure
        unsure.sort(key=lambda o: (len(item.verdicts[o.id]["unsure"]), o.price_key()))
        return unsure

    def code_clean(self, item, offer):
        # the listing judge passed every demand outright, and each wanted value is written in the listing itself
        if NO_NEED_ROWS and not item.need_lines() and offer.stock is True:
            return True
        ruling = item.verdicts.get(offer.id)
        if ruling is None or offer.stock is False or not offer.opened:
            return False
        if any(ruling.get(key) for key in ("failed", "unsure", "weak", "loose")):
            return False
        if self.field_fails(item, offer):
            return False
        return not any(self.unstated(item, offer, need) for need in item.live_needs())

    def buyable(self, item):
        table = self.field_table(item)
        fitting = self.fits(item, cheapest=self.brief.cheapest)
        return [o for o in fitting if o.stock is not False and not self.field_fails(item, o, table)
                and (o.units is None or o.units >= item.quantity)]

    def verify(self, item, offer):
        key = (item.index, offer.id, tuple(item.need_lines()), json.dumps(offer.facts, sort_keys=True, default=str))
        if key not in self.verified:
            self.verified[key] = self.final_check(item, offer)
        return self.verified[key]

    def near_misses(self, item):
        # opened listings with no failed demand and no field the code reads as wrong, fewest open questions first
        table = self.field_table(item)
        wanted = set(item.need_lines())
        rows = []
        for offer in (o for o in item.pool.values() if o.opened and self.book.category_ok(item, o)):
            ruling = item.verdicts.get(offer.id)
            if ruling is None or ruling.get("contradicted") or ruling["failed"] - ruling.get("final", set()) \
                    or not item.judged_for.get(offer.id, set()) >= wanted:
                continue
            if offer.stock is False or (offer.units is not None and offer.units < item.quantity):
                continue
            if self.field_fails(item, offer, table) or not self.book.affordable(item, offer):
                continue
            rows.append(offer)
        if self.brief.cheapest:
            rows.sort(key=lambda o: (len(item.verdicts[o.id]["unsure"]), o.price_key()))
        else:
            rows.sort(key=lambda o: (len(item.verdicts[o.id]["unsure"]), -self.exactness(item, o), o.price_key()))
        return rows

    def near_check(self, item, offer):
        # the second hearing: only a demand the listing's own data contradicts rules it out
        lines = item.need_lines()
        if not lines:
            return True
        cap = self.brief.cap
        user = (f"PRODUCT KIND: {item.category} (quantity {item.quantity})\n"
                f"BUDGET FOR ALL LINES TOGETHER: {'none given' if cap is None else cap}\n"
                "DEMANDS:\n- " + "\n- ".join(lines) + "\n\nOFFER:\n" + json.dumps(offer.card(3500), ensure_ascii=False))
        answer = self.llm.json([VERIFY_MODEL, FAST_MODEL], NEAR_CHECK, user, 800)
        if not isinstance(answer, dict):
            return True
        problems = list(map(str, answer.get("problems") or []))
        self.journal.note("near_check", id=offer.id, fits=answer.get("fits"), problems=problems[:4],
                          silent=list(map(str, answer.get("silent") or []))[:4])
        ruling = item.verdicts.setdefault(offer.id, {"failed": set(), "unsure": set()})
        ruling["silent"] = {self._closest_line(str(x), lines) for x in answer.get("silent") or []}
        if answer.get("fits") is not False or not problems:
            ruling["contradicted"] = set()
            return True
        ruling["contradicted"] = {self._closest_line(problem, lines) for problem in problems}
        ruling["failed"].update(ruling["contradicted"])
        return False

    @staticmethod
    def open_count(item, offer):
        # demands the listing leaves unanswered: the second hearing's count once heard, else the judge's
        ruling = item.verdicts.get(offer.id) or {}
        return len(ruling["silent"]) if "silent" in ruling else len(ruling.get("unsure") or ())

    def missed_lines(self, item, offer):
        ruling = item.verdicts.get(offer.id) or {"failed": set()}
        if "contradicted" in ruling:
            missed = set(ruling["contradicted"])
        else:
            missed = set(ruling["failed"]) - set(ruling.get("final") or ())
        return missed | self.field_fails(item, offer)

    def shortfalls(self, item, offer, rule):
        # demands the listing misses: contradictions once heard, else the judge's fails; stock and budget when the text counts them
        count = len(self.missed_lines(item, offer))
        if rule.get("stock") and (offer.stock is False or (offer.units is not None and offer.units < item.quantity)):
            count += 1
        if rule.get("budget") and not self.book.affordable(item, offer):
            count += 1
        return count

    def final_check(self, item, offer, strict=False, lenient=False):
        lines = item.need_lines()
        cap = self.brief.cap
        budget = "none given" if cap is None else cap
        listed = "\n- ".join(lines) if lines else "(nothing besides the product kind, the budget and availability)"
        user = (f"PRODUCT KIND: {item.category} (quantity {item.quantity})\nBUDGET FOR ALL LINES TOGETHER: {budget}\n"
                "DEMANDS:\n- " + listed + "\n\nOFFER:\n" + json.dumps(offer.card(3500), ensure_ascii=False))
        answer = self.llm.json([VERIFY_MODEL, FAST_MODEL], FINAL_CHECK, user, 800)
        if not isinstance(answer, dict):
            if lenient:
                return True
            if CHECKED_ONLY:
                # no ruling came back: only a listing the code reads as clean goes through
                return self.code_clean(item, offer)
            return not strict
        verdict = answer.get("fits")
        if strict and verdict is not True:
            return False
        problems = list(map(str, answer.get("problems") or []))
        self.journal.note("verify", id=offer.id, fits=verdict, problems=problems[:4])
        if verdict is not False or not problems:
            return True
        ruling = item.verdicts.setdefault(offer.id, {"failed": set(), "unsure": set()})
        missed = {self._closest_line(problem, lines) for problem in problems}
        ruling["failed"].update(missed)
        # kept apart: the strict check also fails what the data leaves unsaid, which a second hearing may allow
        ruling.setdefault("final", set()).update(missed)
        return False
class Brief(Service):
    def __init__(self, request, opening):
        self.request = request
        self.opening = opening
        self.items = []
        self.cap = None
        self.cheapest = False
        self.best = None
        self.message_budget = None
        self.dialog = []
        self.in_stock_only = True
        self.ask_about = []
        self.fit_rules = []
        self.conditions = []
        self.procedure = []
        # what the text wants when no listing meets everything: closest, its price margin and what counts
        self.no_fit = None
        # a message is owed to the shopper before the order
        self.confirm_first = False

    def messages_left(self, sent):
        if self.message_budget is None:
            return None
        return self.message_budget - sent


def tidy_step(step):
    step["moment"] = str(step.get("moment") or "before_finish").lower()
    is_message = str(step.get("kind") or "").lower() == "message"
    step["kind"] = "message" if is_message else "tool"
    return step


def owes_message(step):
    return step["kind"] == "message" and step["moment"] in ("before_order", "before_finish")


class RequestReader(Service):
    def _ask_reader(self):
        brief = self.brief
        sections = ["CUSTOMER REQUEST:\n" + brief.request]
        if brief.opening is not None:
            sections.append("FIRST OBSERVATION FROM THE SHOP:\n" + squeeze(brief.opening, 1500))
        sections.append("SHOP TOOLS:\n" + json.dumps(self.tools.listing(), ensure_ascii=False))
        user = "\n\n".join(sections)
        plan = self.llm.json([PARSE_MODEL, BACKUP_MODEL], BRIEF_READING, user, 2500)
        if not isinstance(plan, dict) or not isinstance(plan.get("tools"), dict):
            plan = self.llm.json([PARSE_MODEL, FAST_MODEL], BRIEF_READING, user, 2500)
        return plan

    def _take_roles(self, plan):
        tools = self.tools
        named = plan.get("tools")
        tools.roles = {}
        for role, spec in (named.items() if isinstance(named, dict) else []):
            if isinstance(spec, dict) and isinstance(spec.get("name"), str) and spec["name"] in tools.schemas:
                tools.roles[role] = spec
        tools.settle_roles()
        for role in ("search_text", "open", "checkout", "decline", "message"):
            if role not in tools.roles:
                self.journal.note("role_missing", role=role)

    def read(self):
        plan = self._ask_reader()
        if not isinstance(plan, dict):
            self.journal.note("setup_failed")
            return False
        brief = self.brief
        self._take_roles(plan)
        specs = plan.get("items")
        if isinstance(specs, list):
            brief.items = [Wanted(spec, n) for n, spec in enumerate(specs) if isinstance(spec, dict)]
        if not brief.items:
            brief.items = [Wanted({"name": "the product", "category": "", "search_terms": [brief.request[:60]]}, 0)]
        for item in brief.items:
            self._clean_item(item)
        self._read_limits(plan)
        if READ_CHECK:
            self._check_reading(brief)
        if READ_GROUNDED:
            self._ground(brief)
        brief.conditions = self.read_conditions(plan.get("conditions"))
        steps = [st for st in plan.get("procedure") or [] if isinstance(st, dict) and st.get("goal")]
        brief.procedure = list(map(tidy_step, steps[:6]))
        brief.confirm_first = any(map(owes_message, brief.procedure))
        self.journal.note("plan", items=[(i.name, i.quantity, i.need_lines(), i.hints) for i in brief.items], cap=brief.cap,
                          cheapest=brief.cheapest, messages=brief.message_budget, steps=self.world.max_steps,
                          roles={k: v.get("name") for k, v in self.tools.roles.items()})
        return True

    def _read_limits(self, plan):
        brief, world = self.brief, self.world
        brief.cap = as_number(plan.get("total_cap"))
        brief.cheapest = bool(plan.get("cheapest"))
        best = plan.get("best")
        brief.best = best if isinstance(best, str) and best else None
        budget = plan.get("message_budget")
        brief.message_budget = int(budget) if whole_count(budget) and budget >= 0 else None
        calls = plan.get("call_budget")
        if whole_count(calls) and 0 < calls < world.max_steps:
            world.max_steps = int(calls)
        brief.in_stock_only = bool(plan.get("in_stock_only", True))
        brief.ask_about = [str(a) for a in plan.get("ask_about") or [] if a]
        brief.fit_rules = [r for r in plan.get("fit_rules") or [] if isinstance(r, dict)]
        if CLOSEST_RULE:
            brief.no_fit = self._read_no_fit(plan.get("no_fit"))

    def _read_no_fit(self, raw):
        if not isinstance(raw, dict) or str(raw.get("mode") or "").lower() != "closest":
            return None
        figures = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", self.brief.request.replace(",", ""))}
        choose = raw.get("choose") if isinstance(raw.get("choose"), dict) else {}
        by = str(choose.get("by") or "none").lower()
        yardstick = str(choose.get("yardstick") or "").strip()
        if by not in ("price", "other") or (by == "other" and not yardstick):
            by = "none"
        margin = choose.get("margin") if isinstance(choose.get("margin"), dict) else {}
        size = as_number(margin.get("value"))
        unit = str(margin.get("unit") or "").lower()
        # a margin counts only when the request writes that figure
        if size is None or size <= 0 or size not in figures or unit not in ("percent", "amount"):
            size, unit = 0.0, "percent"
        if unit == "percent" and size > 100:
            size = 0.0
        counts = [plain(str(c)) for c in raw.get("counts") or [] if c]
        stock_words = ("stock", "availab", "ship", "order")
        return {"by": by, "yardstick": yardstick, "highest": str(choose.get("direction") or "").lower() == "highest",
                "margin": size / 100.0 if unit == "percent" else size, "unit": unit,
                "stock": any(any(w in c for w in stock_words) for c in counts),
                "budget": any("budget" in c or "price" in c or "spend" in c for c in counts),
                "keep": [str(k) for k in raw.get("keep") or [] if k], "tell": bool(raw.get("tell"))}

    @staticmethod
    def _written(value, words):
        # every word of the value is in the request, as written or sharing its first five letters
        wanted = word_set(value)
        if not wanted:
            return True
        for w in wanted:
            if w in words:
                continue
            if w.isalpha() and len(w) >= 5 and any(len(x) >= 5 and x[:5] == w[:5] for x in words):
                continue
            return False
        return True

    def _check_reading(self, brief):
        blocks = [{"product": it.name, "requirements": it.need_lines()} for it in brief.items]
        user = "REQUEST:\n" + brief.request + "\n\nREQUIREMENTS TAKEN:\n" + json.dumps(blocks, ensure_ascii=False)
        answer = self.llm.json([PARSE_MODEL, FAST_MODEL], BRIEF_CHECK, user, 700)
        if not isinstance(answer, dict):
            return
        words = word_set(brief.request)
        taken = []
        for kind in ("add", "fix"):
            for row in answer.get(kind) or []:
                if not isinstance(row, dict) or not row.get("property") or not row.get("value"):
                    continue
                value = str(row["value"])
                # only a value the request itself writes is taken over
                if not self._written(value, words) and not all(p.strip() and p.strip().lower() in brief.request.lower()
                                                               for p in value.split("|")):
                    continue
                item = self.item_named(row.get("item")) or (brief.items[0] if len(brief.items) == 1 else None)
                if item is None:
                    continue
                need = Need(row["property"], row.get("op") or "is", value)
                if kind == "fix":
                    item.needs = [n for n in item.needs if n.reference or not same_name(n.prop, need.prop)]
                elif any(same_name(n.prop, need.prop) and plain(n.value) == plain(need.value) for n in item.needs):
                    continue
                item.needs.append(need)
                taken.append(f"{kind} {item.name}: {need.describe()}")
        if taken:
            for item in brief.items:
                self._clean_item(item)
            self.journal.note("reading_checked", changes=taken[:6])

    def _ground(self, brief):
        words = word_set(brief.request)
        for item in brief.items:
            stated = [n for n in item.needs if not n.reference and n.value and not re.search(r"\d", n.value)]
            loose = [n for n in stated if not any(self._written(part, words) for part in n.value.split("|"))]
            if not loose or len(stated) < 2 or len(loose) > len(stated) / 2:
                continue
            # a value the request never writes came from the backstory: it is asked about, not required
            for need in loose:
                item.needs.remove(need)
                item.hints.append(f"{need.prop} (the story suggests {need.value})")
            self.journal.note("ungrounded", item=item.name, moved=[n.describe() for n in loose])
        figures = {as_number(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", brief.request)}
        cap = brief.cap
        if cap is not None and figures and cap not in figures:
            scaled = [x for x in (cap * 100, cap / 100) if x in figures]
            if scaled:
                self.journal.note("cap_regrounded", read=cap, used=scaled[0])
                brief.cap = scaled[0]

    @staticmethod
    def _clean_item(item):
        # values that only identify a referenced listing are neither needs nor search words for this item
        leaked = set()
        for need in item.needs:
            attrs = need.reference.get("attributes") if need.reference else None
            if isinstance(attrs, dict):
                leaked.update((plain(k), plain(v)) for k, v in attrs.items())
        if leaked:
            item.needs = [n for n in item.needs if n.reference or (plain(n.prop), plain(n.value)) not in leaked]
            leaked_values = [v for _, v in leaked]
            item.terms = [t for t in item.terms if not any(v and v in plain(t) for v in leaked_values)] or [item.category]
        kind_words = word_set(item.category)

        def only_kind_words(need):
            words = word_set(need.value)
            return bool(words) and words <= kind_words

        item.needs = [n for n in item.needs if n.reference or n.op != "is" or not only_kind_words(n)]
        # every stated value is a query of its own: fewer words find more
        for need in (n for n in item.needs if n.value and n.op in ("is", "one of", "contains")):
            for part in (p.strip() for p in need.value.split("|")):
                forms = [part, part.replace("-", " ")] if "-" in part else [part]
                for form in forms:
                    if form and len(form) <= 40 and all(plain(form) != plain(t) for t in item.terms):
                        item.terms.append(form)

    def item_named(self, name):
        name = str(name or "")
        items = self.brief.items
        if not name or not items:
            return None
        scored = [(max(overlap_ratio(it.name, name), overlap_ratio(it.category, name)), it) for it in items]
        best_score, best = scored[0]
        for score, it in scored[1:]:
            if score > best_score:
                best_score, best = score, it
        return best if best_score >= 0.2 else None

    def _branches(self, rule, src):
        out = []
        for key, when in (("then", True), ("otherwise", False)):
            branch = rule.get(key)
            if not isinstance(branch, dict) or not branch.get("property") or not branch.get("value"):
                continue
            dst = self.item_named(branch.get("item"))
            if dst is not None and dst is not src:
                out.append((when, dst, branch))
        return out

    def read_conditions(self, raw):
        rules = []
        for rule in raw if isinstance(raw, list) else []:
            cond = rule.get("if") if isinstance(rule, dict) else None
            if not isinstance(cond, dict) or not cond.get("property"):
                continue
            src = self.item_named(cond.get("item"))
            values = cond.get("values")
            if isinstance(values, list):
                values = [str(v) for v in values if v]
            else:
                values = [str(values)] if values else []
            branches = self._branches(rule, src)
            if src is None or not values or not branches:
                continue
            rules.append({"src": src, "prop": str(cond["property"]), "values": values, "branches": branches})
            for _, dst, branch in branches:
                # values a branch may demand get searched too
                for part in (p.strip() for p in str(branch["value"]).split("|")):
                    if part and len(part) <= 40 and all(plain(part) not in plain(t) for t in dst.terms):
                        dst.terms.append(f"{part} {dst.noun}"[:60])
        if rules:
            summary = [(c["src"].name, c["prop"], c["values"],
                        [(w, d.name, b.get("property"), b.get("value")) for w, d, b in c["branches"]]) for c in rules]
            self.journal.note("conditions", rules=summary)
        return rules


class Talker(Service):
    def __init__(self):
        self.sent = 0
        self.confirmed_sig = None
        self.turned_down = False

    def can_message(self):
        budget = self.brief.message_budget
        if "message" not in self.tools.roles or (budget is not None and self.sent >= budget):
            return False
        return self.world.steps_left() > 0

    def held(self):
        procedure = self.brief.procedure
        if "message" not in self.tools.roles or not procedure:
            return 0
        # a decline carries its own note: only the order keeps messages back
        kept = 0
        for st in filter(owes_message, procedure):
            kept += max(1, int(st.get("messages") or 1))
        left = self.brief.messages_left(self.sent)
        return kept if left is None else min(kept, max(0, left))

    def pending(self):
        shown = bool(self.cart.lines) and self.confirmed_sig == self.cart.signature()
        return 0 if shown else self.held()

    def can_ask(self):
        if not self.can_message():
            return False
        budget = self.brief.message_budget
        return budget is None or self.sent < budget - self.held()

    def _send(self, text):
        spec = self.tools.roles["message"]
        result = self.world.call(spec["name"], {spec["text"]: self.tools.trim("message", spec["text"], text)})
        self.sent += 1
        return result

    def _basket_parts(self):
        parts, total = [], 0.0
        for offer_id, qty in self.cart.signature():
            item = self.cart.owners.get(offer_id)
            offer = item.pool.get(offer_id) if item else None
            title = re.sub(r"\s+", " ", (offer.title if offer else offer_id) or offer_id).strip()[:70]
            line = f"{qty} x {title}"
            if offer and offer.price is not None:
                total += offer.price * qty
                line += f" ({offer.price:g} each)"
            parts.append(line)
        return parts, total

    def confirm_basket(self):
        if not self.can_message() or self.world.steps_left() < 2:
            return False
        parts, total = self._basket_parts()
        closing = f". Total {total:g}." if total else "."
        text = "Your final basket: " + "; ".join(parts) + closing + " Shall I check out this basket? Please confirm with a yes."
        result = self._send(text)
        self.confirmed_sig = self.cart.signature()
        self.journal.note("confirm_asked", basket=self.confirmed_sig, ok=result["ok"], obs=squeeze(result["obs"], 160))
        reply = self._reply_text(result["obs"]) if result["ok"] else None
        if YES_READ and reply:
            self.turned_down = self._take_basket_reply(text, reply)
        return True

    def _take_basket_reply(self, text, reply):
        items = self.brief.items
        user = (f"PRODUCTS: {json.dumps([it.name for it in items])}\nMESSAGE SENT: {text}\nCUSTOMER REPLY: {reply}")
        read = self.llm.json([FAST_MODEL, PLAN_MODEL], BASKET_REPLY, user, 300)
        self.journal.note("basket_reply", reply=reply[:200], read=squeeze(read, 200))
        if not isinstance(read, dict) or str(read.get("agree") or "").lower() != "no":
            return False
        changed = False
        budget = as_number(read.get("budget"))
        if budget is not None and budget > 0:
            self.brief.cap = budget
            changed = True
        if read.get("property") and read.get("value") and items:
            named = str(read.get("item") or "")
            item = max(items, key=lambda it: overlap_ratio(it.name, named or it.name))
            need = Need(read["property"], read.get("op") or "is", read["value"], None, origin="shopper")
            item.needs = [n for n in item.needs if n.origin != "shopper" or n.key() != need.key()] + [need]
            changed = True
        return changed

    @staticmethod
    def _examples(item, prop):
        seen = []
        for offer in item.pool.values():
            for key, value in offer.facts.items():
                if same_name(camel_spaced(key).replace("_", " "), prop) and str(value) not in seen:
                    seen.append(str(value)[:30])
        return " or ".join(seen[:2])

    @staticmethod
    def _reply_text(obs):
        if isinstance(obs, dict):
            strings = [v for v in obs.values() if isinstance(v, str)]
            return max(strings, key=len) if strings else squeeze(obs, 400)
        return obs if isinstance(obs, str) else None

    def ask(self, item, prop, narrow=False, extra=None):
        noun = item.name
        if extra and not narrow:
            text = f"Which {prop} or {extra} would you like the {noun} to have?"
        elif not narrow:
            text = f"Which {prop} would you like the {noun} to have?"
        else:
            examples = self._examples(item, prop)
            tail = f", for example {examples}?" if examples else "?"
            text = f"Just this one thing: for the {noun}, what {prop} do you need" + tail
        text = self.tools.trim("message", self.tools.roles["message"]["text"], text)
        result = self._send(text)
        if not narrow:
            item.asked.append(plain(prop))
            if extra:
                item.asked.append(plain(extra))
        reply = self._reply_text(result["obs"]) if result["ok"] else None
        self.brief.dialog.append({"item": noun, "asked": [prop] + ([extra] if extra else []), "reply": (reply or "")[:200]})
        if not reply:
            return None
        if extra:
            prop = f"{prop} or {extra}"
        known = [f"{n.prop}: {n.op} {n.value}" for n in item.needs if n.origin == "shopper" and n.value]
        user = (f"PRODUCT: {noun}\nATTRIBUTE ASKED ABOUT: {prop}\nMESSAGE SENT: {text}\n"
                f"KNOWN NEEDS: {json.dumps(known)}\nCUSTOMER REPLY: {reply}")
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL, PLAN_MODEL], REPLY_READING, user, 600)
        if NEG_RECHECK and isinstance(answer, dict) and str(answer.get("kind") or "").lower() == "need" \
                and str(answer.get("op") or "").lower() == "is not":
            # an exclusion flips the whole buy: the strong reader reads the reply once more
            second = self.llm.json([PLAN_MODEL, FAST_MODEL], REPLY_READING, user, 600)
            if isinstance(second, dict) and str(second.get("kind") or "").lower() == "need":
                self.journal.note("reply_recheck", first=squeeze(answer, 160), second=squeeze(second, 160))
                answer = second
        if isinstance(answer, dict) and str(answer.get("kind") or "").lower() == "need":
            asked = [prop, str(answer.get("property") or "")]
            moved = self.property_named(item, reply, asked, str(answer.get("value") or ""))
            if moved:
                # the shopper filed the need under another key the listings carry
                answer["property"] = moved
        self.journal.note("reply", item=noun, prop=prop, reply=reply[:200], read=squeeze(answer, 300))
        return answer if isinstance(answer, dict) else None

    @staticmethod
    def property_named(item, reply, asked, value):
        padded = " " + plain(reply) + " "
        if any(a and plain(a) and " " + plain(a) + " " in padded for a in asked):
            return None
        value_words = word_set(value)
        keys = set()
        for offer in item.pool.values():
            for raw in offer.facts:
                key = plain(camel_spaced(raw))
                if len(key) >= 4 and not word_set(key) & value_words and not any(same_name(key, a) for a in asked if a):
                    keys.add(key)
        hits = [k for k in keys if " " + k + " " in padded]
        # "network band" already holds "network": one mention
        hits = [k for k in hits if all(k == o or " " + k + " " not in " " + o + " " for o in hits)]
        return hits[0] if len(hits) == 1 else None

    def _need_from(self, answer, prop):
        need = Need(str(answer.get("property") or prop), answer.get("op") or "is", answer.get("value") or "",
                    answer.get("reference"), origin="shopper")
        if need.reference:
            need.reference["property"] = need.prop
        return need

    def take_reply(self, item, prop, answer):
        if not answer:
            return
        for step in answer.get("steps") or []:
            if isinstance(step, dict):
                self.watcher.add_step(step)
        kind = str(answer.get("kind") or "").lower()
        budget = as_number(answer.get("budget"))
        if budget is not None and budget > 0:
            self.brief.cap = budget
            self.journal.note("budget", cap=budget)
        if kind != "need" or not (answer.get("value") or answer.get("reference")):
            return
        need = self._need_from(answer, prop)
        repeat = need.value and any(n.origin == "shopper" and n.op == need.op and plain(n.value) == plain(need.value)
                                    for n in item.needs)
        if repeat:
            return
        item.needs = [n for n in item.needs if n.origin != "shopper" or n.key() != need.key()]
        item.needs.append(need)
        if need.pending():
            self.follower.resolve(item, need)

    def read_open_reply(self, item, text, obs):
        reply = self._reply_text(obs)
        if not reply:
            return
        known = [f"{n.prop}: {n.op} {n.value}" for n in item.needs if n.origin == "shopper" and n.value]
        user = (f"PRODUCT: {item.name}\nATTRIBUTE ASKED ABOUT: (none named; the message was not a question about one attribute)\n"
                f"MESSAGE SENT: {text[:400]}\nKNOWN NEEDS: {json.dumps(known)}\nCUSTOMER REPLY: {reply}")
        answer = self.llm.json([PLAN_MODEL, FAST_MODEL], REPLY_READING, user, 600)
        self.journal.note("open_reply", reply=reply[:200], read=squeeze(answer, 300))
        if isinstance(answer, dict) and str(answer.get("kind") or "").lower() == "need" and answer.get("property"):
            self.take_reply_keeping_steps(item, str(answer["property"]), answer)

    def take_reply_keeping_steps(self, item, prop, answer):
        left = self.brief.messages_left(self.sent)
        self.follower.floor = max(0, 2 if left is None else left) + self.allowance.end_reserve()
        try:
            self.take_reply(item, prop, answer)
        finally:
            self.follower.floor = 0


class ProcessGate(Service):
    def __init__(self):
        self.calls = {}

    def _basket(self):
        rows = []
        for offer_id, qty in self.cart.signature():
            item = self.cart.owners.get(offer_id)
            offer = item.pool.get(offer_id) if item else None
            rows.append({"id": offer_id, "quantity": qty, "title": (offer.title if offer else "")[:120],
                         "price": offer.price if offer else None, "for": item.name if item else None})
        return rows

    def _clean_args(self, name, act):
        schema = self.tools.schemas[name]
        props = schema.get("properties") or {}
        given = act.get("args")
        args = {}
        if isinstance(given, dict):
            args = {key: value for key, value in given.items() if key in props}
        for key in list(args):
            value, prop = args[key], props.get(key) or {}
            allowed = prop.get("enum")
            if isinstance(allowed, list) and allowed and value not in allowed:
                del args[key]
                continue
            limit = prop.get("maxLength")
            if isinstance(value, str) and isinstance(limit, int) and len(value) > limit:
                args[key] = value[:limit]
        missing = [k for k in schema.get("required") or [] if k not in args]
        return None if missing else args

    def _prompt(self, moment, intent, ending):
        world, brief = self.world, self.brief
        shown = [{"name": n, "description": self.tools.describe(n), "parameters": self.tools.schemas[n]}
                 for n in self.tools.schemas if n not in ending]
        basket = self._basket()
        total = sum((row["price"] or 0) * row["quantity"] for row in basket)
        left = brief.messages_left(self.talker.sent)
        parts = [
            "CUSTOMER REQUEST:\n" + brief.request,
            "\n\nPROCEDURE THE REQUEST DEMANDS:\n" + json.dumps(brief.procedure, ensure_ascii=False),
            f"\n\nMOMENT: {moment} (next the agent will {intent})",
            "\n\nCURRENT BASKET: " + json.dumps({"lines": basket, "total": total}, ensure_ascii=False),
            f"\nMESSAGES REMAINING: {'no limit' if left is None else left}; STEPS REMAINING: {world.steps_left()}",
            "\n\nAVAILABLE TOOLS:\n" + json.dumps(shown, ensure_ascii=False),
            "\n\nHISTORY (calls with their results, oldest first):\n" + json.dumps(world.history[-30:], ensure_ascii=False),
        ]
        return "".join(parts)

    def run(self, moment, intent):
        world, talker, journal = self.world, self.talker, self.journal
        used = self.calls.get(moment, 0)
        if used >= 3 or world.steps_left() < 2 or self.clock.left() < WRAP_SECONDS:
            return False
        self.calls[moment] = used + 1
        ending = self.tools.names_of("checkout", "decline", "cart_add", "cart_set") - {None}
        answer = self.llm.json([PLAN_MODEL, BACKUP_MODEL], PROCESS_GATE, self._prompt(moment, intent, ending), 900)
        if not isinstance(answer, dict):
            journal.note("gate_failed", moment=moment)
            return None
        acts = [a for a in (answer.get("actions") or [])[:2] if isinstance(a, dict)]
        message_tool = self.tools.name_of("message")
        ran = False
        for position, act in enumerate(acts):
            tool = act.get("tool")
            if tool not in self.tools.schemas or tool in ending:
                continue
            to_shopper = tool == message_tool
            if not to_shopper:
                due = max(sum(1 for a in acts[position + 1:] if a.get("tool") == message_tool), talker.pending())
                if world.steps_left() < 2 + due:
                    journal.note("gate_step_skipped", moment=moment, tool=tool, steps=world.steps_left(), due=due)
                    continue
            args = self._clean_args(tool, act)
            if args is None or (to_shopper and not talker.can_message()):
                continue
            if world.steps_left() < 2:
                break
            result = world.call(tool, args)
            if to_shopper:
                talker.sent += 1
                talker.confirmed_sig = self.cart.signature()
                if GATE_REPLY and result["ok"] and len(self.brief.items) == 1:
                    # whatever the shopper says back can carry a need, as any reply can
                    talker.read_open_reply(self.brief.items[0], str(args.get(self.tools.roles["message"]["text"]) or ""), result["obs"])
            journal.note("gate_step", moment=moment, tool=tool, why=str(act.get("why") or "")[:160], ok=result["ok"])
            ran = True
        if not ran:
            journal.note("gate_clear", moment=moment)
        return ran


class QuestionPlanner(Service):
    def __init__(self):
        self.planned = None

    @staticmethod
    def _menu_key(key):
        spaced = camel_spaced(key).replace("_", " ").replace(".", " ")
        return " ".join(spaced.split()).lower()

    def _tables(self, item):
        opened = [o for o in item.pool.values() if o.opened and self.book.category_ok(item, o)]
        alive = [o for o in opened if not (item.verdicts.get(o.id) or {}).get("failed")]
        chosen = alive if len(alive) >= 4 else opened
        tables = []
        for offer in chosen:
            table = {}
            for raw, value in offer.facts.items():
                key = self._menu_key(raw)
                if key:
                    table.setdefault(key, str(value).strip().lower()[:60])
            tables.append(table)
        return len(chosen), tables

    @staticmethod
    def _aliases(keys, tables):
        alias = {}
        for i, a in enumerate(keys):
            if a in alias:
                continue
            for b in (b for b in keys[i + 1:] if b not in alias):
                both = [t for t in tables if a in t and b in t]
                agree = sum(1 for t in both if t[a] == t[b])
                if len(both) >= 2 and agree >= 0.8 * len(both):
                    alias[b] = a
        return alias

    def property_menu(self, item):
        seen_count, tables = self._tables(item)
        counts = {}
        for table in tables:
            for key in table:
                counts[key] = counts.get(key, 0) + 1
        # on a tie the shorter name stands for its aliases
        keys = sorted(counts, key=lambda k: (-counts[k], len(k), k))
        alias = self._aliases(keys, tables)
        fixed = {n.key() for n in item.needs} | set(item.asked)

        def settled(name):
            return plain(name) in fixed or any(overlap_ratio(name, f) >= 0.67 for f in fixed)

        weighted = []
        for key in (k for k in keys if k not in alias):
            names = [key] + [b for b, a in alias.items() if a == key]
            if any(map(settled, names)):
                continue
            firsts = [next((t[n] for n in names if n in t), None) for t in tables]
            carriers = sum(1 for t in tables if any(n in t for n in names))
            values = list(dict.fromkeys(v for v in firsts if v))
            # one shared value says little; a value per listing (an id) says little too
            weight = carriers * (0.5 if len(values) == 1 and carriers >= 3 else 1.0)
            if carriers >= 4 and len(set(firsts) - {None}) >= carriers:
                weight *= 0.5
            entry = {"property": key, "listings": carriers, "of": seen_count, "values": values[:3]}
            if names[1:]:
                entry["same_as"] = names[1:4]
            weighted.append((-weight, key, entry))
        weighted.sort(key=lambda row: row[:2])
        return [row[2] for row in weighted[:40]]

    def decided_props(self, item):
        out = []
        for cond in self.brief.conditions:
            if cond["src"] is item:
                out.append(cond["prop"])
            for _, dst, branch in cond["branches"]:
                if dst is item:
                    out.append(str(branch["property"]))
        if any(acc is item for acc, _ in self.linker.fit_pairs()):
            out.append("compatibility")
        return list(dict.fromkeys(out))

    def taken(self, item, prop, plan):
        if any(same_name(prop, a) for a in item.asked):
            return True
        if any(same_name(prop, d) or overlap_ratio(prop, d) >= 0.5 for d in self.decided_props(item)):
            return True
        return any(same_name(prop, n.prop) for n in item.needs) or any(it is item and same_name(prop, p) for it, p in plan)

    def _ask_planner(self, count):
        brief = self.brief
        blocks = [{"item": it.name, "category": it.category, "known_requirements": it.need_lines(),
                   "decided_by_the_request": self.decided_props(it), "hints": it.hints, "already_asked": it.asked,
                   "property_menu": self.property_menu(it)} for it in brief.items]
        wanted = count if brief.message_budget is None else brief.message_budget
        user = ("CUSTOMER REQUEST:\n" + brief.request + "\n\nTOPICS THE CUSTOMER WANTS RAISED WITH THEM: "
                + json.dumps(brief.ask_about) + "\n\nPRODUCTS:\n" + json.dumps(blocks, ensure_ascii=False)
                + f"\n\nQUESTION_COUNT = {max(wanted, count) if not brief.dialog else count}")
        models = [PLAN_MODEL, FAST_MODEL]
        if brief.dialog:
            user += ("\n\nQUESTIONS ALREADY SENT AND THE CUSTOMER'S REPLIES (oldest first; a reply can hint at what "
                     "else they care about):\n" + json.dumps(brief.dialog, ensure_ascii=False))
            models = [FAST_MODEL, PLAN_MODEL]  # a re-plan is a small update of the first plan
        answer = self.llm.json(models, QUESTION_PLANNING, user, 700)
        questions = answer.get("questions") if isinstance(answer, dict) else None
        ordered = []
        for question in questions if isinstance(questions, list) else []:
            if not isinstance(question, dict) or not question.get("property"):
                continue
            named = question.get("item")
            target = max(brief.items, key=lambda it: overlap_ratio(it.name, named or it.name))
            prop = tidy_property(question["property"])
            if prop and all(it is not target or not same_name(prop, p) for it, p in ordered):
                ordered.append((target, prop))
        self.planned = ordered
        self.journal.note("plan_questions", questions=[(it.name, p) for it, p in ordered])

    def plan_extra(self, item, prop):
        for entry in self._usable_menu(item):
            other = entry["property"]
            if not same_name(other, prop) and not self.taken(item, other, [(item, prop)]):
                return other
        return None

    def _usable_menu(self, item):
        return [e for e in self.property_menu(item) if e["listings"] >= 2 or e["of"] <= 2]

    def plan(self, count):
        if count <= 0:
            return []
        if not self.planned:
            self._ask_planner(count)
        if len(self.brief.items) > 1:
            return self._take_turns(count)
        plan = []
        for item, prop in self.planned:
            if len(plan) >= count:
                break
            if not self.taken(item, prop, plan):
                plan.append((item, prop))
        if count >= 3:
            for item in self.brief.items:
                tops = [e["property"] for e in self.property_menu(item) if e["listings"] >= 3][:2]
                for k, prop in enumerate(tops):
                    if not self.taken(item, prop, plan):
                        plan.insert(min(len(plan), 2 * k + 1), (item, prop))
            del plan[count:]
        if len(plan) < count:
            self._pad(plan, count)
        return plan[:count]

    def _pad(self, plan, count):
        items = self.brief.items
        menus = {it.index: self._usable_menu(it) for it in items}
        order = sorted(items, key=lambda it: (sum(1 for h, _ in plan if h is it) + len(it.asked), it.index))
        progress = True
        while progress and len(plan) < count:
            progress = False
            for item in order:
                queue = menus[item.index]
                while queue:
                    prop = queue.pop(0)["property"]
                    if self.taken(item, prop, plan):
                        continue
                    plan.append((item, prop))
                    progress = True
                    break
                if len(plan) >= count:
                    break

    def _queues(self):
        items = self.brief.items
        queues = {it.index: [] for it in items}
        first = []
        for it, prop in self.planned:
            if it.index in queues and not self.taken(it, prop, []) \
                    and not any(same_name(prop, q) for q in queues[it.index]):
                queues[it.index].append(prop)
                if it.index not in first:
                    first.append(it.index)
        for it in items:
            menu = self._usable_menu(it)
            queue = queues[it.index]
            top = next((e["property"] for e in menu if e["listings"] >= 3), None)
            if top and not self.taken(it, top, []) and all(not same_name(top, x) for x in queue[:2]):
                queue[:] = [x for x in queue if not same_name(x, top)]
                queue.insert(min(1, len(queue)), top)
            for prop in (e["property"] for e in menu):
                if not self.taken(it, prop, []) and all(not same_name(prop, x) for x in queue):
                    queue.append(prop)
            if it.index not in first:
                first.append(it.index)
        return queues, first

    def _take_turns(self, count):
        items = self.brief.items
        queues, first = self._queues()
        told = {it.index for it in items if any(n.origin == "shopper" for n in it.needs)}
        by_index = {it.index: it for it in items}
        turn = sorted(first, key=lambda i: (i in told, len(by_index[i].asked), first.index(i)))
        plan = []
        while len(plan) < count and any(queues[i] for i in turn):
            for i in turn:
                if queues[i] and len(plan) < count:
                    plan.append((by_index[i], queues[i].pop(0)))
        return plan


class PageHunt:
    def __init__(self, reference, need):
        self.seller = str(reference.get("seller") or "").strip()
        self.category = str(reference.get("category") or "").strip()
        attrs = reference.get("attributes")
        self.attrs = attrs if isinstance(attrs, dict) else {}
        self.prop = str(reference.get("property") or need.prop).strip()
        self.attr_words = " ".join(str(v) for v in list(self.attrs.values())[:3] if v)
        self.pick = str(reference.get("pick") or "").strip()
        count = reference.get("count")
        self.count = count if isinstance(count, int) else None

    def by_rule(self):
        return bool(self.pick) or (self.count or 0) > 1

    def queries(self):
        if self.seller:
            middle = [self.seller + " " + self.category] if self.category else []
            return [(self.seller + " " + self.attr_words).strip()] + middle + [self.seller]
        return [(self.category + " " + self.attr_words).strip()] if self.category else []

    def seller_ok(self, offer):
        if not self.seller:
            return True
        if overlap_ratio(offer.seller, self.seller) >= 0.5 or overlap_ratio(offer.brand, self.seller) >= 0.5:
            return True
        return plain(self.seller) in plain(offer.title + " " + offer.text + " " + offer.notes)

    def overlap(self, offer):
        words = word_set(self.attr_words)
        if not words:
            return 0
        return len(words & word_set(offer.title + " " + offer.text + " " + offer.notes + " " + json.dumps(offer.facts)))

    def words_present(self, offer):
        if not self.attrs:
            return False
        table = " ".join(f"{k} {v}" for k, v in offer.facts.items())
        text = all_words(" ".join([offer.title or "", offer.notes or "", offer.text or "", table]))
        return all(all_words(v) and all_words(v) <= text for v in self.attrs.values() if str(v).strip())

    def table_matches(self, offer):
        if not self.attrs or not offer.opened:
            return False
        facts = [(plain(camel_spaced(k)), plain(v)) for k, v in offer.facts.items()]

        def stated(key, wanted):
            return any((same_name(fk, key) or overlap_ratio(fk, key) >= 0.5) and fv == wanted for fk, fv in facts)

        return all(plain(value) and stated(key, plain(value)) for key, value in self.attrs.items())

    def framing(self):
        return ("WANTED OFFER: category " + self.category + "; seller " + self.seller + "; attribute values "
                + json.dumps(self.attrs, ensure_ascii=False))


class ReferenceFollower(Service):
    def __init__(self):
        self.floor = 0
        self.hunts = {}

    def _ask_match(self, user):
        return self.llm.json([FAST_MODEL, BACKUP_MODEL], PAGE_MATCHING, user, 600)

    def _gather(self, hunt, scratch, done=None):
        found = []
        for query in hunt.queries():
            if self.world.steps_left() < 4:
                break
            if done is not None:
                if query in done:
                    continue
                done.add(query)
            for _ in range(2):
                if self.browser.search(scratch, query=query, max_pages=1, budget=3) == 0:
                    break
                found = [o for o in scratch.pool.values()
                         if hunt.seller_ok(o) and (not hunt.category or self.book.category_ok(scratch, o))]
                if found:
                    break
            if len(found) >= 2 or (found and hunt.overlap(found[0]) > 0):
                break
        return found

    @staticmethod
    def _opened_in(scratch, offers):
        return [scratch.pool[o.id] for o in offers if o.id in scratch.pool and scratch.pool[o.id].opened]

    def _keep_pages(self, scratch, ids):
        for offer in self.browser.open_many(None, ids):
            scratch.pool[offer.id] = offer

    def _choose_by_rule(self, hunt, scratch, found):
        # several of the seller's pages fit: all are read before the rule picks one
        book = self.book
        for start in range(0, min(len(found), 3 * book.compare_limit), book.compare_limit):
            if self.world.steps_left() < 3:
                break
            self._keep_pages(scratch, [x.id for x in found[start:start + book.compare_limit]])
        cards = [o.card(1200) for o in self._opened_in(scratch, found)]
        if not cards:
            return None
        user = hunt.framing()
        if hunt.pick:
            user += f"\nPICK: {hunt.pick}"
        if hunt.count:
            user += f"\nOFFERS THAT QUALIFY: {hunt.count}"
        match = self._ask_match(user + "\n\nCANDIDATE OFFERS:\n" + json.dumps(cards, ensure_ascii=False))
        if isinstance(match, dict) and match.get("id") in scratch.pool:
            return scratch.pool[match["id"]]
        return None

    def _walk(self, hunt, scratch, found):
        single = not self.tools.roles.get("compare") or self.book.compare_off
        size = 1 if single else self.book.compare_limit
        rounds_left = 5 if single else 3
        held = []
        while found and rounds_left > 0 and self.world.steps_left() >= 3:
            batch, found = found[:size], found[size:]
            rounds_left -= 1
            self._keep_pages(scratch, [o.id for o in batch])
            now = self._opened_in(scratch, batch)
            exact = [o for o in now if hunt.table_matches(o)]
            if exact:
                return exact[0]
            held += now
            if single and found and len(held) < 3:
                continue
            shown = held if single else now
            held = []
            if not shown:
                continue
            cards = [o.card(1600) for o in shown]
            match = self._ask_match(hunt.framing() + "\n\nCANDIDATE OFFERS:\n" + json.dumps(cards, ensure_ascii=False))
            worded = [o for o in shown if hunt.words_present(o)]
            if isinstance(match, dict) and match.get("id") in scratch.pool:
                return scratch.pool[match["id"]]
            if len(worded) == 1:
                return worded[0]
            if len(cards) == 1 and not found:
                return scratch.pool[cards[0]["id"]]
        return None

    def _read(self, page, prop):
        user = f"ATTRIBUTE NEEDED: {prop}\n\nOFFER PAGE:\n" + json.dumps(page.card(3000), ensure_ascii=False)
        return self.llm.json([FAST_MODEL, BACKUP_MODEL], PAGE_FOLLOWING, user, 600)

    def _table(self, page):
        page.facts = self.browser.facts_of(page.id) or {}

    def resolve(self, item, need, hop=0):
        reference = need.reference or {}
        if hop > 7 or self.world.steps_left() < max(4, self.floor) or self.clock.left() < WRAP_SECONDS + 20:
            return
        hunt = PageHunt(reference, need)
        memo = None
        if HUNT_MEMO:
            # the same pointer hunted again: what was searched and read before is not paid for twice
            key = json.dumps([plain(hunt.seller), plain(hunt.category), hunt.attrs, plain(hunt.prop), hunt.pick], sort_keys=True, default=str)
            memo = self.hunts.get(key)
            if memo is None:
                scratch = Wanted({"name": hunt.category or "listing", "category": hunt.category, "search_terms": []}, 99)
                memo = self.hunts[key] = {"scratch": scratch, "queries": set(), "read": set()}
            scratch = memo["scratch"]
            found = self._gather(hunt, scratch, memo["queries"])
            if not found:
                found = [o for o in scratch.pool.values()
                         if hunt.seller_ok(o) and (not hunt.category or self.book.category_ok(scratch, o))]
            found = [o for o in found if o.id not in memo["read"]]
        else:
            scratch = Wanted({"name": hunt.category or "listing", "category": hunt.category, "search_terms": []}, 99)
            found = self._gather(hunt, scratch)
        if not found:
            self.journal.note("reference_unfound", seller=hunt.seller, category=hunt.category)
            return
        found.sort(key=lambda o: (-hunt.overlap(o), overlap_ratio(o.seller, hunt.seller) < 0.5, o.price or 0))
        page = self._choose_by_rule(hunt, scratch, found) if hunt.by_rule() else None
        if page is None:
            page = self._walk(hunt, scratch, found)
        if memo is not None:
            memo["read"].update(o.id for o in scratch.pool.values() if o.opened)
        if page is None:
            self.journal.note("reference_unmatched", seller=hunt.seller, category=hunt.category)
            return
        self._settle(item, need, hunt, page, hop)

    def _first_read(self, page, prop, paired):
        world = self.world
        thin = self.book.thin_compare and not page.facts
        if thin and not page.notes and world.steps_left() >= 3:
            self._table(page)
            thin = False
        read = self._read(page, prop)
        if not thin or world.steps_left() < 3 or not isinstance(read, dict) or read.get("redirect"):
            return read, thin
        if read.get("value") and not paired:
            return read, thin
        self._table(page)
        return (self._read(page, prop) if page.facts else read), thin

    def _settle(self, item, need, hunt, page, hop):
        paired = need.op == "counterpart of reference"
        read, thin = self._first_read(page, hunt.prop, paired)
        self.journal.note("reference", hop=hop, page=page.id, read=squeeze(read, 200))
        redirect = read.get("redirect") if isinstance(read, dict) else None
        if isinstance(redirect, dict) and redirect:
            about = str(redirect.get("property") or hunt.prop)
            if not self._same_property(about, hunt.prop):
                redirect = None
                self._own_value(read, page, hunt.prop, thin)
                self.journal.note("reference_other_redirect", about=about, wanted=hunt.prop, value=read.get("value"))
        if isinstance(redirect, dict) and redirect:
            need.reference = redirect
            need.reference["property"] = hunt.prop
            self.resolve(item, need, hop + 1)
            return
        value = read.get("paired" if paired else "value") if isinstance(read, dict) else None
        if not value:
            if paired and isinstance(read, dict) and read.get("value"):
                self.journal.note("pair_unstated", page=page.id, value=read.get("value"))
            return
        need.value = str(value)
        need.op = REF_OPS.get(need.op, "is")
        need.reference = None
        fresh_term = need.op == "is" and len(need.value) <= 40 and item.terms
        if fresh_term and all(plain(need.value) not in plain(t) for t in item.terms):
            item.terms.insert(0, f"{need.value} {item.noun}"[:60])

    @staticmethod
    def _same_property(about, wanted):
        if same_name(about, wanted) or overlap_ratio(about, wanted) >= 0.5:
            return True
        return plain(wanted) in plain(about) or plain(about) in plain(wanted)

    def _own_value(self, read, page, prop, thin):
        if not read.get("value") and thin and not page.facts and self.world.steps_left() >= 3:
            self._table(page)
            again = self._read(page, prop) if page.facts else None
            if isinstance(again, dict):
                read["value"] = again.get("value") or read.get("value")
                read["paired"] = again.get("paired") or read.get("paired")
        if read.get("value"):
            return
        for key, value in page.facts.items():
            spaced = str(key).replace("_", " ")
            if same_name(spaced, prop) or overlap_ratio(spaced, prop) >= 0.5:
                if value not in (None, ""):
                    read["value"] = str(value)
                return


class EventWatcher(Service):
    def __init__(self):
        self.unread = []

    def _budget_from(self, change, lines):
        budget = as_number(change.get("budget"))
        stated = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", " ".join(lines))]
        if budget in stated:
            return budget
        scaled = [x for x in (budget * 100, budget / 100) if x in stated]
        if scaled:
            return scaled[0]
        cap = self.brief.cap
        near = [x for x in stated if cap / 20 <= x <= cap * 20] if stated and cap is not None else []
        return near[0] if near else budget

    def _target(self, change):
        named = change.get("item")
        if not named:
            return None
        items = self.brief.items
        target = max(items, key=lambda it: overlap_ratio(it.name, named))
        if len(items) > 1 and overlap_ratio(target.name, named) < 0.2:
            return None
        return target

    def handle_lines(self):
        world = self.world
        lines = [line for line in world.pending_lines if line not in world.handled_lines]
        world.pending_lines = []
        if not lines:
            return False
        world.handled_lines.update(lines)
        user = ("CUSTOMER LINES:\n- " + "\n- ".join(lines)
                + "\n\nPRODUCTS IN THE REQUEST: " + json.dumps([i.name for i in self.brief.items]))
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], EVENT_READING, user, 700)
        self.journal.note("event", lines=[line[:160] for line in lines], read=squeeze(answer, 300))
        if not isinstance(answer, dict):
            self.unread.extend(lines)  # news nobody could read: the closer re-reads the cart in code
            return False
        results = [self._apply(change, lines) for change in answer.get("changes") or [] if isinstance(change, dict)]
        return any(results)

    def _drop(self, targets, prop):
        dropped = False
        for item in targets:
            before = len(item.needs)
            item.needs = [n for n in item.needs if overlap_ratio(n.prop, prop) < 0.5]
            if len(item.needs) == before:
                continue
            dropped = True
            for ruling in item.verdicts.values():
                ruling["failed"] = {f for f in ruling["failed"] if overlap_ratio(f.split(":")[0], prop) < 0.5}
                if REFUSAL_REPICK:
                    ruling["unsure"] = {f for f in ruling["unsure"] if overlap_ratio(f.split(":")[0], prop) < 0.5}
                    ruling.pop("contradicted", None)
        return dropped

    def _apply(self, change, lines):
        kind = str(change.get("kind") or "").lower()
        target = self._target(change)
        items = self.brief.items
        if kind == "budget" and as_number(change.get("budget")):
            self.brief.cap = self._budget_from(change, lines)
            return True
        if kind == "drop" and change.get("property"):
            return self._drop([target] if target else items, change["property"])
        if kind == "add" and change.get("property") and change.get("value"):
            targets = [target] if target else items[:1]
            for item in targets:
                item.needs.append(Need(change["property"], change.get("op") or "is", change["value"], None, origin="shopper"))
            return bool(targets)
        if kind in ("price", "stock"):
            self.refresh_cart()
            return True
        if kind == "step" and isinstance(change.get("step"), dict):
            self.add_step(change["step"])
        return False

    def add_step(self, step):
        procedure = self.brief.procedure
        if not step.get("goal") or len(procedure) >= 8:
            return
        step = tidy_step(dict(step))
        if any(overlap_ratio(step["goal"], st.get("goal")) >= 0.6 for st in procedure):
            return
        procedure.append(step)
        if owes_message(step):
            self.brief.confirm_first = True
        self.journal.note("step_added", step=step)

    def refresh_cart(self):
        for offer_id, item in list(self.cart.owners.items()):
            if self.world.steps_left() <= 2:
                return
            self.browser.open_many(item, [offer_id])
            known = item.pool.get(offer_id)
            self.journal.note("refreshed", id=offer_id, price=known.price if known else None,
                              stock=known.stock if known else None)


class CartHand(Service):
    def __init__(self):
        self.lines = {}
        self.owners = {}
        self.decline_gated = False
        self.decline_refused = False
        self.decline_reason = ""
        self.viewed_sig = None

    def signature(self):
        return tuple(sorted((offer_id, int(q or 0)) for offer_id, q in self.lines.items() if q))

    def _whole_cart_tool(self):
        roles = self.tools.roles
        return not roles.get("cart_add") and bool(roles.get("cart_set"))

    def _replace(self, lines):
        spec = self.tools.roles.get("cart_set") or {}
        lines_schema = self.tools.props("cart_set").get(spec.get("lines") or "") or {}
        line_props = (lines_schema.get("items") or {}).get("properties") or {}
        maximum = (line_props.get(spec.get("qty") or "") or {}).get("maximum")
        rows = []
        for offer_id, qty in lines.items():
            row = {spec["id"]: offer_id}
            if spec.get("qty"):
                row[spec["qty"]] = qty if not isinstance(maximum, int) else min(qty, maximum)
            rows.append(row)
        result = self.world.call(spec["name"], {spec["lines"]: rows})
        if not result["ok"]:
            self.journal.note("cart_set_failed", error=result["error"], obs=squeeze(result["obs"], 200))
        return result["ok"]

    @staticmethod
    def _mode(spec):
        mode = spec.get("mode")
        return mode if isinstance(mode, dict) else None

    def _qty_limit(self, spec, field):
        return (self.tools.props("cart_add").get(spec["qty"]) or {}).get(field)

    def add(self, item, offer):
        if self._whole_cart_tool():
            lines = dict(self.lines)
            lines[offer.id] = item.quantity
            if not self._replace(lines):
                return False
            self.lines = lines
            self.owners[offer.id] = item
            return True
        spec = self.tools.roles.get("cart_add")
        if not spec:
            return False
        args = {spec["id"]: offer.id}
        qty = item.quantity
        if spec.get("qty"):
            maximum = self._qty_limit(spec, "maximum")
            if isinstance(maximum, int):
                qty = min(qty, maximum)
            args[spec["qty"]] = qty
        mode = self._mode(spec)
        if mode and mode.get("arg") and mode.get("add") is not None:
            args[mode["arg"]] = mode["add"]
        result = self.world.call(spec["name"], args)
        if not result["ok"]:
            self.journal.note("cart_add_failed", error=result["error"], obs=squeeze(result["obs"], 200))
            return False
        self.lines[offer.id] = qty
        self.owners[offer.id] = item
        return True

    def remove(self, offer_id):
        if self._whole_cart_tool():
            lines = {k: v for k, v in self.lines.items() if k != offer_id}
            if not self._replace(lines):
                return False
            self.lines = lines
            self.owners.pop(offer_id, None)
            return True
        spec = self.tools.roles.get("cart_add")
        if not spec:
            return False
        args = {spec["id"]: offer_id}
        mode = self._mode(spec)
        if mode and mode.get("arg") and mode.get("remove") is not None:
            args[mode["arg"]] = mode["remove"]
            if spec.get("qty"):
                args[spec["qty"]] = self.lines.get(offer_id, 1)
        elif spec.get("qty"):
            minimum = self._qty_limit(spec, "minimum")
            args[spec["qty"]] = minimum if isinstance(minimum, int) and minimum > 0 else 0
        if not self.world.call(spec["name"], args)["ok"]:
            return False
        self.lines.pop(offer_id, None)
        self.owners.pop(offer_id, None)
        return True

    def checkout(self):
        spec = self.tools.roles.get("checkout")
        if not spec:
            return False
        if self.brief.procedure and self.lines and not self._gate_before_order():
            return False
        result = self.world.call(spec["name"], self.tools.ending_args("checkout"))
        if not self.world.finished:
            self.journal.note("order_refused", obs=squeeze(result["obs"], 200), lines=result["lines"][:3])
            return False
        self.journal.note("ordered", cart=self.lines)
        return True

    def _gate_before_order(self):
        intent = "place the order for the basket shown below"
        talker, gate = self.talker, self.gate
        ran = gate.run("before_order", intent)
        if ran and talker.pending() and self.world.steps_left() >= 2:
            ran = gate.run("before_order", intent) or ran
        if ran is not False and self.brief.confirm_first and self.signature() != talker.confirmed_sig:
            ran = talker.confirm_basket() or ran
        if talker.turned_down:
            # the customer turned the basket down with a change: the caller picks again
            talker.turned_down = False
            self.journal.note("basket_turned_down")
            return False
        if not ran:
            return True
        if self.watcher.handle_lines() or self.closer.price_or_stock_changed():
            # something changed with the reply: the caller picks again
            self.journal.note("procedure_changed")
            return False
        return True

    def decline(self, reason=""):
        spec = self.tools.roles.get("decline")
        if not spec:
            return False
        world = self.world
        if self.brief.procedure and not self.decline_gated:
            self.decline_gated = True
            self.gate.run("before_decline", "close the task without any purchase")
        args = self.tools.ending_args("decline")
        remark = spec.get("remark")
        if remark and remark in (self.tools.schema("decline").get("required") or []):
            args[remark] = self.tools.trim("decline", remark, "Nothing available meets every requirement of the request.")
        result = world.call(spec["name"], args)
        self.journal.note("declined", reason=reason or self.decline_reason)
        if world.finished or self.decline_refused or world.steps_left() < 1:
            return world.finished
        self.decline_refused = True
        self.journal.note("decline_refused", obs=squeeze(result["obs"], 200), lines=result["lines"][:3])
        changed = self.watcher.handle_lines()
        if REFUSAL_REPICK and world.steps_left() >= len(self.brief.items) + 2:
            # a turned-down decline asks for a purchase: the closest listing is now acceptable
            self.chooser.after_refusal = True
            self.chooser.choose_and_finish()
        elif changed and world.steps_left() >= len(self.brief.items) + 2:
            self.chooser.choose_and_finish()
        if not world.finished:
            world.call(spec["name"], args)
        return world.finished


class Linker(Service):
    def __init__(self):
        self.held_cache = {}

    def fit_pairs(self):
        items = self.brief.items
        pairs = []
        if len(items) < 2:
            return pairs
        for rule in self.brief.fit_rules:
            accessory, host = str(rule.get("accessory") or ""), str(rule.get("host") or "")
            if not accessory or not host:
                continue
            a = max(items, key=lambda it: overlap_ratio(it.name, accessory))
            h = max(items, key=lambda it: overlap_ratio(it.name, host))
            usable = a is not h and overlap_ratio(a.name, accessory) >= 0.2 and overlap_ratio(h.name, host) >= 0.2
            if usable and (a, h) not in pairs:
                pairs.append((a, h))
        return pairs

    @staticmethod
    def in_series(offer, series):
        return any(head in series for head in _SERIES.findall(offer.title or ""))

    @staticmethod
    def _letters(code):
        return re.match(r"[A-Z]+", code).group(0)

    def _looked_up_codes(self, item):
        titled = {c for it in self.brief.items for o in it.pool.values() for c in PART_CODE.findall(o.title or "")}
        prefixes = {self._letters(c) for o in item.pool.values() for c in PART_CODE.findall(o.title or "")}
        wanted = []
        for offer in self.book.candidates(item):
            if not offer.opened or (item.verdicts.get(offer.id) or {}).get("failed"):
                continue
            for code in PART_CODE.findall(" ".join([offer.text or "", offer.notes or ""])):
                if code not in titled and code not in wanted and self._letters(code) in prefixes:
                    wanted.append(code)
        return wanted

    def _series(self, host_item, host):
        # a short code on the host page that starts accessory codes ("AB3" of "AB3-1234") names the fitting series
        host_text = " ".join([host.text or "", host.notes or "", json.dumps(host.facts, ensure_ascii=False)])
        heads = set()
        for it in self.brief.items:
            if it is not host_item:
                for o in it.pool.values():
                    heads.update(_SERIES.findall(o.title or ""))
        families = {self._letters(h) for h in heads}
        own_heads = _SERIES.findall(host.title or "")
        series = []
        for code in re.findall(r"\b[A-Z]{1,5}\d{1,3}\b", host_text):
            if code in series or code in own_heads:
                continue
            if code in heads or self._letters(code) in families:
                series.append(code)
        return series

    def bind_fit(self, item, host_item, host):
        world, book, reserve = self.world, self.book, self.allowance.end_reserve
        page = " ".join(x for x in [host.title[:120], (host.text or "")[:300], (host.notes or "")[:150]] if x)
        item.needs = [n for n in item.needs if n.origin != "fit"]
        item.needs.append(Need("compatibility", "is", f"works with the {host_item.name} chosen, whose page reads: {page}",
                               origin="fit"))
        wanted = self._looked_up_codes(item)
        series = self._series(host_item, host)
        for code in series[:2] + [w for w in wanted if w not in series][:3]:
            if world.steps_left() <= reserve() + 2 or self.clock.left() < WRAP_SECONDS + 10:
                break
            self.browser.search(item, query=f"{code} {item.category}" if code in series else code, max_pages=1, budget=reserve())
        if series and world.steps_left() > reserve() + 2:
            mine = [o for o in book.unopened(item) if self.in_series(o, series)]
            if mine:
                size = book.compare_limit if book.batch_open() else 3
                self.browser.open_many(item, [o.id for o in mine[:size]], floor=reserve() + 1)
        self.journal.note("fit_bound", item=item.name, host=host.id, series=series[:2], looked_up=wanted[:3])
        self.assessor.judge_opened(item)
        if series:
            self._credit_series(item, series)

    def _credit_series(self, item, series):
        fit_line = item.needs[-1].describe()
        for offer in self.book.candidates(item):
            ruling = item.verdicts.get(offer.id)
            if not ruling or not offer.opened or not self.in_series(offer, series):
                continue
            for key in ("unsure", "weak", "failed"):
                if key in ruling:
                    ruling[key].discard(fit_line)

    @staticmethod
    def _table_says(cond, offer):
        prop = cond["prop"]
        own = []
        for key, value in offer.facts.items():
            spaced = str(key).replace("_", " ")
            if same_name(spaced, prop) or overlap_ratio(spaced, prop) >= 0.5:
                own.append(str(value))
        if not own:
            return None
        text = all_words(" ".join(own))
        wants = [all_words(v) for v in cond["values"] if all_words(v)]
        if any(w <= text for w in wants):
            return True
        if any(w & text for w in wants):
            return None
        return False

    def condition_holds(self, cond, offer):
        key = (id(cond), offer.id, json.dumps(offer.facts, sort_keys=True, default=str))
        if key in self.held_cache:
            return self.held_cache[key]
        holds = self._table_says(cond, offer)
        if holds is None:
            user = (f"ATTRIBUTE: {cond['prop']}\nVALUES THAT SATISFY: {json.dumps(cond['values'], ensure_ascii=False)}\n\nOFFER:\n"
                    + json.dumps(offer.card(2500), ensure_ascii=False))
            answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], CONDITION_CHECK, user, 300)
            if isinstance(answer, dict) and isinstance(answer.get("holds"), bool):
                holds = answer["holds"]
        self.held_cache[key] = holds
        self.journal.note("condition", src=cond["src"].name, listing=offer.id, prop=cond["prop"], holds=holds)
        return holds

    def apply_conditions(self, item, chosen):
        mine = [c for c in self.brief.conditions if any(d is item for _, d, _ in c["branches"])]
        if not mine:
            return
        wanted = []
        for cond in mine:
            pick = chosen.get(cond["src"].index)
            holds = None if pick is None else self.condition_holds(cond, pick)
            if holds is None:
                continue
            for when, dst, branch in cond["branches"]:
                if dst is item and when == holds:
                    wanted.append(Need(branch["property"], branch.get("op") or "is", branch["value"], origin="condition"))
        if sorted(n.describe() for n in item.needs if n.origin == "condition") == sorted(n.describe() for n in wanted):
            return
        item.needs = [n for n in item.needs if n.origin != "condition"] + wanted
        self.journal.note("condition_applied", item=item.name, reqs=[n.describe() for n in wanted])
        self.assessor.judge_opened(item)
        spare = self.world.steps_left() - self.allowance.end_reserve()
        if not self.assessor.buyable(item) and spare >= 3 and self.clock.left() > WRAP_SECONDS + 20:
            self.explorer.explore(item, self.world.steps_left() - self.allowance.end_reserve())


class Allowance(Service):
    def __init__(self):
        self.override = None

    def end_reserve(self):
        # cart adds, a look at the carted listing, a swap, the checkout and one retry
        if self.override is not None:
            return self.override
        roles, brief = self.tools.roles, self.brief
        later_steps = [st for st in brief.procedure if st["moment"] != "start"]
        owed = min(4, len(later_steps) + int(bool(brief.confirm_first)))
        extra_reads = 2 if roles.get("facts") and not roles.get("compare") else 0
        return len(brief.items) + 6 + extra_reads + owed

    def lean(self):
        kept = self.talker.held()
        if self.brief.confirm_first:
            kept = max(kept, 1)
        return len(self.brief.items) + 3 + kept


class Explorer(Service):
    def _batch_size(self, otherwise):
        if self.book.batch_open():
            return self.book.compare_limit
        return 4 if self.book.thin_compare else otherwise

    def _may_go_on(self, floor, margin):
        return self.world.steps_left() > floor and self.clock.left() > WRAP_SECONDS + margin and not self.llm.over_cap()

    def _spare_rows(self, item):
        spare = [o for o in item.pool.values() if not o.opened and self.book.category_ok(item, o) and o.stock is not False]
        return sorted(spare, key=lambda o: (o.screen == "no", o.screen != "yes", o.price_key()))

    def warm_up(self, item, allowance):
        # open more of the item's kind so the property menu is representative
        world, reserve = self.world, self.allowance.end_reserve
        if world.steps_left() <= reserve() + 2 or "message" not in self.tools.roles:
            return
        floor = max(world.steps_left() - allowance, reserve() + 2)
        target = 5 if not self.book.batch_open() else 12
        if SPLIT_ROUTE and len(self.brief.items) > 1:
            # several products share the steps: two batched reads (or three single ones) per product
            target = 3 if not self.book.batch_open() else 2 * self.book.compare_limit
        while self._may_go_on(floor, 20):
            want = target - len([o for o in item.pool.values() if o.opened and self.book.category_ok(item, o)])
            if want <= 0:
                return
            todo = self._spare_rows(item)
            if len(todo) < 2 and item.search_done <= len(item.terms):
                size = len(item.pool)
                self.browser.search(item, query=item.category, max_pages=1, budget=floor)
                todo = self._spare_rows(item)
                if not todo and len(item.pool) == size:
                    return
            if not todo:
                return
            picked = todo[:min(want, self._batch_size(1))]
            if not self.browser.open_many(item, [o.id for o in picked], floor=floor):
                return
            self.assessor.judge_opened(item)

    def _cheaper_moves(self, item, terms, best, floor):
        world, browser = self.world, self.browser
        fields_role = self.tools.roles.get("search_fields")
        sweep = best < 1e17 and fields_role and fields_role.get("price_max") and item.sweep_below != best
        if sweep and world.steps_left() > floor() + 2:
            # a field search capped below the best price shows only cheaper listings
            item.sweep_below = best
            browser.search(item, fields={"category": item.category, "price_max": max(0.0, best - 0.01)},
                           max_pages=3, budget=floor())
            return "again"
        if item.capped and not item.narrowed and world.steps_left() > floor() + 2:
            # hidden matches: a query with every value lists fewer, all of them
            item.narrowed = True
            values = item.plain_values()
            if values:
                for query in [" ".join(values + [item.category]), " ".join(values)]:
                    if world.steps_left() <= floor() + 1:
                        break
                    browser.search(item, query=query, max_pages=2, budget=floor())
                return "again"
        open_cursors = [q for q in terms if item.cursors.get("search_text:" + q)
                        and not item.cursors["search_text:" + q].get("done")]
        if open_cursors and world.steps_left() > floor() + 2:
            stepped_from = world.step
            browser.search(item, query=open_cursors[0], max_pages=1, budget=floor() + 1)
            if world.step > stepped_from:
                return "again"
        if item.search_done >= min(len(terms), 2):
            return "stop"
        return "go"

    def explore(self, item, steps):
        world, brief = self.world, self.brief
        stop_at = world.steps_left() - steps

        def floor():
            return max(stop_at, self.allowance.end_reserve())

        terms = list(item.terms)
        position = [min(item.search_done, len(terms))]
        enough = 3 if brief.cheapest else (4 if brief.best else 2)
        while self._may_go_on(floor(), 10):
            self.watcher.handle_lines()
            self.book.screen(item, self.book.candidates(item))
            cands = self.book.candidates(item)
            self.assessor.judge(item, [o for o in cands if o.opened])
            unopened = [o for o in cands if not o.opened]
            fits = self.assessor.fits(item)
            if fits and not brief.cheapest:
                if len(fits) >= enough:
                    return
            elif fits:
                # the cheapest rule needs every cheaper unopened candidate seen
                best = fits[0].price_key()
                unopened = [o for o in unopened if o.price is not None and o.price < best]
                move = "go" if unopened else self._cheaper_moves(item, terms, best, floor)
                if move == "stop":
                    return
                if move == "again":
                    continue
            if unopened:
                if not self.browser.open_many(item, [o.id for o in unopened[:self._batch_size(2)]], floor=floor()):
                    return
                continue
            if not self._search_more(item, terms, position, floor):
                return

    def _search_more(self, item, terms, position, floor):
        world, browser, brief = self.world, self.browser, self.brief
        told, stated = item.plain_values(True), item.plain_values(False)
        if told and not item.need_search_done:
            # told values narrow a search well
            item.need_search_done = True
            noun = item.noun if SHORT_QUERIES and item.noun and len(item.noun) < len(item.category) else item.category
            queries = [f"{told[0]} {noun}" if SHORT_QUERIES else f"{item.category} {told[0]}", told[0]]
            if stated:
                queries.insert(0, f"{stated[0]} {told[0]} {noun}" if SHORT_QUERIES else f"{item.category} {stated[0]} {told[0]}")
            for query in queries:
                if world.steps_left() <= floor():
                    break
                browser.search(item, query=query, max_pages=3, budget=floor())
            return True
        if position[0] < len(terms):
            query = terms[position[0]]
            position[0] += 1
            browser.search(item, query=query, max_pages=3 if len(brief.items) != 1 else 6, budget=floor())
            return True
        if self.tools.roles.get("search_fields"):
            fields = {"category": item.category}
            if brief.cap is not None:
                fields["price_max"] = brief.cap / max(item.quantity, 1)
            cursor = item.cursors.get("search_fields:" + json.dumps(fields, sort_keys=True)) or {}
            if not cursor.get("done"):
                browser.search(item, fields=fields, max_pages=2, budget=floor())
                return True
        for query in terms[:2]:
            unfinished = not (item.cursors.get("search_text:" + query) or {}).get("done")
            if unfinished and browser.search(item, query=query, max_pages=2, budget=floor()) > 0:
                return True
        if item.suggested or world.steps_left() <= floor() + 1:
            return False
        item.suggested = True
        extra = self.suggest_queries(item)
        if not extra:
            return False
        item.terms += extra
        terms += extra
        return True

    @staticmethod
    def _tried(item):
        return sorted({key.split(":", 1)[1] for key in item.cursors if key.startswith("search_text:")})

    def suggest_queries(self, item):
        sample = sorted(item.pool.values(), key=lambda o: (not o.opened, o.id))[:14]
        fitting = {o.id for o in self.assessor.fits(item, include_unsure=True)}
        rows = [{"title": o.title[:110], "fits": o.id in fitting if o.opened else None} for o in sample]
        tried = self._tried(item)
        query_arg = (self.tools.roles.get("search_text") or {}).get("query") or ""
        limit = self.tools.props("search_text").get(query_arg, {}).get("maxLength")
        user = (f"PRODUCT: {item.name} (category: {item.category})\nDEMANDS:\n- " + ("\n- ".join(item.need_lines()) or "none")
                + f"\nQUERIES ALREADY RUN: {json.dumps(tried, ensure_ascii=False)}\nOFFERS SEEN: {json.dumps(rows, ensure_ascii=False)}")
        if limit:
            user += f"\nLONGEST QUERY ALLOWED: {limit} characters"
        answer = self.llm.json([FAST_MODEL, BACKUP_MODEL], QUERY_WRITING, user, 700)
        fresh = []
        proposed = answer.get("queries") if isinstance(answer, dict) else None
        for query in map(str, proposed or []):
            query = query.strip()
            known = {plain(t) for t in tried + item.terms + fresh}
            if query and len(query) <= 60 and plain(query) not in known:
                fresh.append(query)
        self.journal.note("suggested", item=item.name, queries=fresh[:5])
        return fresh[:5]

    def _audit_card(self, item, offer):
        card = {"id": offer.id, "title": offer.title[:110], "price": offer.price, "read": offer.opened}
        if not offer.opened:
            card["text"] = (offer.text or "")[:160]
            return card
        ruling = item.verdicts.get(offer.id) or {}
        for key in ("failed", "unsure"):
            card[key] = sorted(ruling.get(key) or [])[:3]
        card["properties"] = dict(list(offer.facts.items())[:16])
        if offer.notes:
            card["notes"] = offer.notes[:200]
        return card

    def audit(self, missing):
        learned = False
        self.allowance.override = self.allowance.lean()
        try:
            for item in missing:
                if self.clock.left() < WRAP_SECONDS + 20:
                    break
                if self._audit_one(item):
                    learned = True
        finally:
            self.allowance.override = None
        return learned

    def _audit_one(self, item):
        book = self.book
        pool = [o for o in item.pool.values() if book.category_ok(item, o) and o.stock is not False and book.affordable(item, o)]
        cards = [self._audit_card(item, o) for o in sorted(pool, key=lambda o: (not o.opened, o.price_key()))[:40]]
        if not cards:
            return False
        tried = self._tried(item)
        cap = self.brief.cap
        user = (f"PRODUCT: {item.name} (category: {item.category}; quantity {item.quantity})\nBUDGET FOR ALL LINES TOGETHER: "
                f"{'none given' if cap is None else cap}\nDEMANDS:\n- " + ("\n- ".join(item.need_lines()) or "none")
                + f"\nQUERIES ALREADY RUN: {json.dumps(tried, ensure_ascii=False)}\n\nOFFERS SEEN:\n" + squeeze(cards, 14000))
        answer = self.llm.json([PLAN_MODEL, FAST_MODEL], SEARCH_AUDIT, user, 1500)
        if not isinstance(answer, dict):
            return False
        self.journal.note("audit", item=item.name, answer=squeeze(answer, 300))
        learned = self._accept_after_all(item, answer)
        learned = self._follow_audit_leads(item, answer, tried) or learned
        self.assessor.judge_opened(item)
        return learned

    def _accept_after_all(self, item, answer):
        accepted = False
        for offer_id in [str(x) for x in answer.get("fits_after_all") or []][:2]:
            offer = item.pool.get(offer_id)
            if offer is None or not offer.opened or self.assessor.field_fails(item, offer):
                continue
            if self.assessor.final_check(item, offer, strict=True):
                item.verdicts[offer.id] = {"failed": set(), "unsure": set(), "weak": set(), "loose": set()}
                item.judged_for[offer.id] = set(item.need_lines())
                accepted = True
        return accepted

    def _follow_audit_leads(self, item, answer, tried):
        world, browser, book, reserve = self.world, self.browser, self.book, self.allowance.end_reserve
        learned = False
        unread = [str(x) for x in answer.get("unopened") or []]
        fresh = [x for x in unread if x in item.pool and not item.pool[x].opened][:4]
        if fresh and world.steps_left() - reserve() >= 1 and browser.open_many(item, fresh, floor=reserve()):
            learned = True
        tried_plain = {plain(t) for t in tried}
        for query in [str(x).strip() for x in answer.get("queries") or [] if str(x).strip()][:2]:
            if world.steps_left() - reserve() < 3 or self.clock.left() < WRAP_SECONDS + 15:
                break
            if plain(query) in tried_plain or browser.search(item, query=query[:60], max_pages=1, budget=reserve()) <= 0:
                continue
            book.screen(item, book.unopened(item))
            rows = [o for o in book.unopened(item) if o.screen != "no"][:2]
            if rows and browser.open_many(item, [o.id for o in rows], floor=reserve()):
                learned = True
        return learned


class Chooser(Service):
    def __init__(self):
        self.best_order = {}
        self.last_chance = False
        self.audited = False
        self.refs_retried = False
        self.after_refusal = False
        self.told = False

    def _rank_by_llm(self, item, cands, ids, measure=None):
        measure = measure or self.brief.best
        rows = [{"id": o.id, "title": o.title[:120], "price": o.price, "properties": o.facts,
                 "description": str(o.text or "")[:300]} for o in cands[:12]]
        answer = self.llm.json([PLAN_MODEL, BACKUP_MODEL], BEST_RANKING,
                               f"YARDSTICK: {measure}\nPRODUCT: {item.name}\nOFFERS: " + squeeze(rows, 9000), 600)
        order = []
        if isinstance(answer, dict):
            order = [str(x) for x in answer.get("order") or []]
        self.journal.note("best_rank", measure=measure, order=order[:12])
        if order and measure == self.brief.best:
            self.best_order[item.index] = (ids, order)
        return order

    def rank_best(self, item, cands):
        if not self.brief.best or len(cands) < 2:
            return cands
        ids = [o.id for o in cands]
        cached = self.best_order.get(item.index)
        order = cached[1] if cached is not None and cached[0] == ids else self._rank_by_llm(item, cands, ids)
        place = {offer_id: n for n, offer_id in enumerate(order)}
        return sorted(cands, key=lambda o: place.get(o.id, len(place) + ids.index(o.id)))

    def _ranked(self, item):
        return self.rank_best(item, self.assessor.buyable(item))

    def _late(self):
        return self.clock.left() < WRAP_SECONDS

    def checked_pick(self, item, cands, tries=3):
        checked = 0
        for cand in cands:
            if checked >= tries or self._late():
                break
            checked += 1
            if self.assessor.verify(item, cand):
                return cand
        if CHECKED_ONLY:
            # out of checks or time: only a listing the code reads as clean, never an unchecked one
            return next((c for c in cands[checked:] if self.assessor.code_clean(item, c)), None)
        return cands[0] if cands and (checked >= tries or self._late()) else None

    def _first_pick(self, item, chosen):
        pick = self.checked_pick(item, self._ranked(item), 4 if CHECKED_ONLY else 3)
        if pick is not None:
            chosen[item.index] = pick

    def _other_host(self, item, host, chosen):
        for other in self.assessor.buyable(host)[:3]:
            if chosen.get(host.index) is other or self._late():
                continue
            self.linker.bind_fit(item, host, other)
            got = self.assessor.buyable(item)
            if not got:
                continue
            chosen[host.index] = other
            chosen[item.index] = got[0]
            return

    def _try_source(self, item, cond, chosen):
        linker, assessor = self.linker, self.assessor
        src = cond["src"]
        outcome = linker.condition_holds(cond, chosen[src.index])
        for other in assessor.buyable(src)[:4]:
            if other is chosen[src.index] or linker.condition_holds(cond, other) == outcome:
                continue
            was = chosen[src.index]
            chosen[src.index] = other
            linker.apply_conditions(item, chosen)
            got = assessor.buyable(item)
            if got and assessor.verify(src, other):
                chosen[item.index] = got[0]
                return
            chosen[src.index] = was
            linker.apply_conditions(item, chosen)

    def _other_source(self, item, chosen, pairs):
        shaping = [c for c in self.brief.conditions
                   if c["src"].index in chosen and any(d is item for _, d, _ in c["branches"])]
        for cond in shaping:
            src = cond["src"]
            bound = any(h is src and a.index in chosen for a, h in pairs)
            if bound or self._late():
                continue
            self._try_source(item, cond, chosen)
            if item.index in chosen:
                return

    def _total(self, chosen):
        return sum((chosen[it.index].price or 0) * it.quantity for it in self.brief.items)

    def _within(self, chosen):
        return self._total(chosen) <= self.brief.cap + 1e-6

    def _fit_budget(self, chosen):
        moves = 0
        while moves < 10 and not self._within(chosen) and self._step_cheaper(chosen):
            moves += 1
        return self._within(chosen)

    def _step_cheaper(self, chosen):
        by_cost = sorted(self.brief.items, key=lambda i: -((chosen[i.index].price or 0) * i.quantity))
        for it in by_cost:
            fits = [f for f in self.assessor.fits(it, cheapest=True) if f.stock is not False]
            ids = [f.id for f in fits]
            current = chosen[it.index].id
            at = ids.index(current) if current in ids else -1
            if 0 <= at < len(fits) - 1:
                chosen[it.index] = fits[at + 1]
                return True
        return False

    def _pick_all(self):
        brief, assessor, linker = self.brief, self.assessor, self.linker
        chosen = {}
        pairs = linker.fit_pairs()
        hosts = {h.index for _, h in pairs}
        accessories = {a.index: h for a, h in pairs}
        sources = {c["src"].index for c in brief.conditions}
        dependents = {d.index for c in brief.conditions for _, d, _ in c["branches"]}
        independent = [it for it in brief.items if it.index not in accessories and it.index not in dependents]
        tops = [(it, (self._ranked(it) or [None])[0]) for it in independent]
        run_parallel([lambda it=it, c=c: assessor.verify(it, c) for it, c in tops if c is not None])

        def turn(it):
            waits = it.index in accessories or it.index in dependents
            return waits, it.index not in hosts and it.index not in sources, it.index

        for item in sorted(brief.items, key=turn):
            host = accessories.get(item.index)
            if host is not None and host.index in chosen:
                linker.bind_fit(item, host, chosen[host.index])
            if item.index in dependents:
                linker.apply_conditions(item, chosen)
            self._first_pick(item, chosen)
            if item.index in chosen:
                continue
            if host is not None:
                self._other_host(item, host, chosen)
            if item.index in dependents and item.index not in chosen:
                self._other_source(item, chosen, pairs)
        return chosen

    def choose_and_finish(self):
        brief, cart = self.brief, self.cart
        self.watcher.handle_lines()
        if REVEAL_RETRY and not self.refs_retried:
            # a need still pointing at another listing is not met by ignoring it
            self.refs_retried = True
            for item in brief.items:
                for need in [n for n in item.needs if n.pending()]:
                    if self.world.steps_left() > self.allowance.end_reserve() + 4 and not self._late():
                        self.follower.resolve(item, need)
        self.assessor.judge_all()
        chosen = self._pick_all()
        missing = [it for it in brief.items if it.index not in chosen]
        if missing and self._retry_missing(missing):
            self.choose_and_finish()
            return
        if missing and CLOSEST_RULE and brief.no_fit:
            short = []
            for item in list(missing):
                pick = self._closest(item)
                if pick is not None:
                    chosen[item.index] = pick
                    missing.remove(item)
                    short.append((item, pick))
            if short and not missing:
                self._tell_closest(short)
        if missing and NEAR_MISS:
            for item in list(missing):
                pick = self._near_miss(item)
                if pick is not None:
                    chosen[item.index] = pick
                    missing.remove(item)
        if missing:
            cart.decline_reason = "no fitting listing for " + ", ".join(it.name for it in missing)
            self.journal.note("no_fit", items=[it.name for it in missing], pools={it.name: len(it.pool) for it in missing})
            cart.decline(cart.decline_reason)
            return
        counted = bool(CLOSEST_RULE and brief.no_fit and brief.no_fit["budget"])
        if brief.cap is not None and not self._fit_budget(chosen) and not counted:
            cart.decline_reason = "the cheapest fitting combination is over the budget"
            self.journal.note("over_budget", total=self._total(chosen), cap=brief.cap)
            cart.decline(cart.decline_reason)
            return
        for item in brief.items:
            item.chosen = chosen[item.index]
        self.closer.place_order()

    def _tell_closest(self, short):
        # the text wants to hear that nothing fully fits: a free message says so, no question is held back for it
        talker = self.talker
        if not self.brief.no_fit.get("tell") or self.told or not talker.can_message():
            return
        if self.world.steps_left() <= self.allowance.end_reserve():
            return
        self.told = True
        parts = []
        for item, offer in short:
            missed = sorted(line.split(":")[0] for line in self.assessor.missed_lines(item, offer))
            gap = f" (it may fall short on {', '.join(missed[:3])})" if missed else ""
            parts.append(f"{offer.title[:90]} at {offer.price}{gap}")
        talker._send("No listing meets every one of your requirements, so I am putting forward the closest: "
                     + "; ".join(parts) + ".")
        self.watcher.handle_lines()

    def _closest(self, item):
        # the text wants the listing missing the fewest demands: count them, then take the cheapest group's best
        rule, assessor = self.brief.no_fit, self.assessor
        book = assessor.book
        usable = lambda o: o.opened and book.category_ok(item, o) and o.id in item.verdicts \
            and "cart refused" not in item.verdicts[o.id]["failed"]
        rows = [o for o in item.pool.values() if usable(o)]
        if not rows:
            spare = sorted((o for o in item.pool.values() if not o.opened and book.category_ok(item, o)),
                           key=lambda o: o.price_key())
            if spare and self.world.steps_left() > len(self.brief.items) + 3 and not self._late():
                self.explorer.browser.open_many(item, [o.id for o in spare[:2]])
                assessor.judge_opened(item)
                rows = [o for o in item.pool.values() if usable(o)]
        if not rows:
            return None
        key = lambda o: (assessor.shortfalls(item, o, rule), len(item.verdicts[o.id]["unsure"]), o.price_key())
        rows.sort(key=key)
        for offer in rows[:3]:
            # the strong check turns the judge's reading into plain contradictions
            if "contradicted" in item.verdicts[offer.id] or self._late() or self.llm.over_cap():
                continue
            assessor.near_check(item, offer)
        if rule["keep"]:
            # a demand the text never lets the fallback miss rules a listing out
            kept = [o for o in rows if not any(overlap_ratio(line.split(":")[0], k) >= 0.5 or overlap_ratio(line, k) >= 0.5
                                               for line in assessor.missed_lines(item, o) for k in rule["keep"])]
            rows = kept or rows
        rows.sort(key=key)
        fewest = key(rows[0])[0]
        group = [o for o in rows if key(o)[0] == fewest]
        group.sort(key=lambda o: (assessor.open_count(item, o), -assessor.exactness(item, o), o.price_key()))
        if rule["by"] == "price":
            priced = [o.price for o in group if o.price is not None]
            if priced:
                edge = max(priced) if rule["highest"] else min(priced)
                reach = edge * rule["margin"] if rule["unit"] == "percent" else rule["margin"]
                group = [o for o in group if o.price is None or abs(o.price - edge) <= reach + 1e-6] or group
        elif rule["by"] == "other" and len(group) > 1 and not self._late():
            # the text's own yardstick decides among equally close listings
            order = self._rank_by_llm(item, group, [o.id for o in group], rule["yardstick"])
            place = {offer_id: n for n, offer_id in enumerate(order)}
            group.sort(key=lambda o: place.get(o.id, len(place)))
        pick = group[0]
        item.closest = (pick.id, pick.stock, pick.price)
        self.journal.note("closest", item=item.name, id=pick.id, shortfalls=fewest, price=pick.price, group=len(group))
        return pick

    def _near_miss(self, item):
        # a decline is only right when the shop shows nothing close: a listing that failed nothing is checked first
        rows = self.assessor.near_misses(item)
        passed = []
        for offer in rows[:3]:
            if self._late() or self.llm.over_cap():
                break
            if self.assessor.near_check(item, offer):
                passed.append(offer)
                if not self.assessor.open_count(item, offer):
                    break
        if passed:
            # the listing leaving the fewest demands unanswered goes first
            offer = min(passed, key=lambda o: self.assessor.open_count(item, o))
            self.journal.note("near_miss", item=item.name, id=offer.id, open=self.assessor.open_count(item, offer))
            return offer
        rows = self.assessor.near_misses(item)
        if rows and (self.after_refusal or self._late()):
            # the customer turned the decline down, or no check time is left: the closest listing is bought
            self.journal.note("near_miss_forced", item=item.name, id=rows[0].id)
            return rows[0]
        return None

    def _rescue(self, item):
        # listings held back only because a revealed need is not spelled out: the strong check rules on them
        told = {n.describe() for n in item.live_needs() if n.origin == "shopper"}
        if not told:
            return False
        held = [o for o in self.assessor.fits(item, include_unsure=True, cheapest=self.brief.cheapest)
                if o.stock is not False and item.verdicts[o.id]["unsure"] and item.verdicts[o.id]["unsure"] <= told]
        for offer in held[:3]:
            if self._late() or self.llm.over_cap():
                break
            if self.assessor.final_check(item, offer, strict=True):
                item.verdicts[offer.id]["unsure"] = set()
                self.journal.note("rescued", item=item.name, id=offer.id)
                return True
        return False

    def _retry_missing(self, missing):
        world, allowance, assessor = self.world, self.allowance, self.assessor
        if not self.last_chance and self.clock.left() > WRAP_SECONDS + 20 and not self.llm.over_cap():
            # spend the ending's spare steps on one more search, keeping the cart, the yes and the checkout
            self.last_chance = True
            allowance.override = allowance.lean()
            try:
                for k, it in enumerate(missing):
                    if REVEAL_RETRY and any(n.origin == "shopper" for n in it.needs):
                        # a revealed need is searched for on its own words again, with fresh queries
                        it.need_search_done = False
                        it.suggested = False
                    spare = world.steps_left() - allowance.end_reserve()
                    if BASKET_STEPS:
                        spare //= max(1, len(missing) - k)
                    if spare >= 3:
                        self.explorer.explore(it, spare)
            finally:
                allowance.override = None
            if REVEAL_RETRY:
                for it in missing:
                    if not assessor.buyable(it):
                        self._rescue(it)
            if any(assessor.buyable(it) for it in missing):
                return True
        if self.audited or self.clock.left() <= WRAP_SECONDS + 25 or self.llm.over_cap():
            return False
        self.audited = True
        return self.explorer.audit(missing) and all(assessor.buyable(it) for it in missing)


class Closer(Service):
    def __init__(self):
        self.rounds = 0

    def _basket_total(self):
        return sum((it.chosen.price or 0) * it.quantity for it in self.brief.items if it.chosen)

    def price_or_stock_changed(self):
        cap = self.brief.cap
        rule = self.brief.no_fit if CLOSEST_RULE else None
        for item in self.brief.items:
            offer = item.chosen
            if offer is None:
                continue
            as_chosen = rule is not None and item.closest == (offer.id, offer.stock, offer.price)
            short = offer.stock is False or (offer.units is not None and offer.units < item.quantity)
            if short and not as_chosen:
                return True
            over = cap is not None and offer.price is not None and self._basket_total() > cap + 1e-6
            if over and not (as_chosen and rule["budget"]):
                return True
        return False

    def cart_faults(self):
        # the shop's own cart view must hold exactly the chosen listings
        spec = self.tools.roles.get("cart_view")
        if not spec or self.cart.viewed_sig == self.cart.signature():
            return []
        if self.world.steps_left() <= 3 + self.talker.pending():
            return []
        result = self.world.call(spec["name"], {})
        self.cart.viewed_sig = self.cart.signature()
        if not result["ok"]:
            return []
        known = {o for it in self.brief.items for o in it.pool}
        shown = set()

        def gather(node):
            if isinstance(node, dict):
                for value in node.values():
                    gather(value)
            elif isinstance(node, list):
                for value in node:
                    gather(value)
            elif isinstance(node, (str, int)) and str(node) in known:
                shown.add(str(node))

        gather(result["obs"])
        if not shown:
            return []  # a view that names no listing id proves nothing either way
        wanted = {it.chosen.id for it in self.brief.items if it.chosen}
        faults = [f"{offer_id} extra" for offer_id in sorted(shown - wanted)]
        faults += [f"{offer_id} missing" for offer_id in sorted(wanted - shown)]
        return faults

    def _sync_cart(self):
        cart, items = self.cart, self.brief.items
        wanted_ids = lambda: [it.chosen.id for it in items if it.chosen]
        for offer_id in list(cart.lines):
            if offer_id not in wanted_ids():
                cart.remove(offer_id)
        for item in items:
            if not item.chosen or item.chosen.id in cart.lines:
                continue
            if not cart.add(item, item.chosen):
                item.verdicts.setdefault(item.chosen.id, {"failed": set(), "unsure": set()})["failed"].add("cart refused")
                item.chosen = None

    def _recheck_chosen(self):
        changed = False
        for item in self.brief.items:
            pick = item.chosen
            if pick is None or pick.id in item.judged_for:
                continue
            self.assessor.judge(item, [pick])
            if (item.verdicts.get(pick.id) or {}).get("failed"):
                self.journal.note("chosen_failed", id=pick.id, failed=sorted(item.verdicts[pick.id]["failed"])[:3])
                changed = True
        return changed

    def place_order(self, attempt=0):
        self.rounds += 1
        world, items, watcher = self.world, self.brief.items, self.watcher
        if attempt > 2 or world.finished or self.rounds > 8:
            return
        self._sync_cart()
        if any(it.chosen is None for it in items):
            self.chooser.choose_and_finish()
            return
        first = items[0].chosen if items else None
        if BASKET_STEPS:
            # every carted listing is read again: stock and price can move while the basket fills
            for item in items:
                if item.chosen and world.steps_left() > 2 + self.talker.pending():
                    self.browser.open_many(item, [item.chosen.id])
        elif first and world.steps_left() > 2 + self.talker.pending():
            # a look at the carted listing: changes show up in the next result
            self.browser.open_many(items[0], [first.id])
        changed = watcher.handle_lines()
        if self._recheck_chosen():
            changed = True
        faults = self.cart_faults() if BASKET_STEPS else []
        if BASKET_STEPS and watcher.handle_lines():
            # the cart view can carry the shopper's or the shop's news
            changed = True
        if BASKET_STEPS and watcher.unread:
            # news that came too late to read: the carted listings are read again in code, no LLM needed
            watcher.unread = []
            spec = self.tools.roles.get("open")
            for item in items:
                if spec and item.chosen and world.steps_left() > 2 and self.clock.left() > 4:
                    self.browser._open_single(item, spec, None, item.chosen.id)
        if faults:
            # the view disagrees with the cart as recorded: bring the shop's cart to the chosen lines
            self.journal.note("cart_faults", faults=faults[:4])
            for fault in faults:
                offer_id, kind = fault.split()
                if kind == "extra":
                    self.cart.lines.setdefault(offer_id, 1)
                    self.cart.remove(offer_id)
                else:
                    self.cart.lines.pop(offer_id, None)
            self._sync_cart()
        if changed or self.price_or_stock_changed():
            self.journal.note("rechoose", cart=self.cart.lines)
            for item in items:
                self.assessor.judge_opened(item)
            self.rechoose()
        elif self.cart.checkout():
            return
        else:
            if not watcher.handle_lines():
                watcher.refresh_cart()
                watcher.handle_lines()
            self.rechoose()
        if not world.finished:
            self.place_order(attempt + 1)

    def _search_again(self, item):
        item.need_search_done = False
        item.sweep_below = None
        share = max(2, self.world.steps_left() - self.allowance.end_reserve())
        if share >= 3 and self.clock.left() > WRAP_SECONDS + 20:
            self.explorer.explore(item, share)

    def rechoose(self):
        brief, assessor = self.brief, self.assessor
        for item in brief.items:
            assessor.judge_opened(item)
            if brief.cheapest or not [f for f in assessor.fits(item) if f.stock is not False]:
                self._search_again(item)
        for item in brief.items:
            fits = [f for f in assessor.fits(item, cheapest=brief.cheapest) if f.stock is not False]
            keep = item.chosen is not None and not brief.cheapest and item.chosen.id in [f.id for f in fits]
            if not keep and CHECKED_ONLY:
                item.chosen = self.chooser.checked_pick(item, fits, 2)
            elif not keep:
                item.chosen = fits[0] if fits else None
        picks = {it.index: it.chosen for it in brief.items if it.chosen is not None}
        shaped = {}
        for cond in brief.conditions:
            for _, dst, _ in cond["branches"]:
                shaped[dst.index] = dst
        for dst in shaped.values():
            # a new source pick can change what the condition demands
            self.linker.apply_conditions(dst, picks)
            ok = [f.id for f in assessor.buyable(dst)]
            if dst.chosen is not None and dst.chosen.id not in ok:
                dst.chosen = dst.pool[ok[0]] if ok else None
        over = brief.cap is not None and all(it.chosen for it in brief.items) \
            and sum((it.chosen.price or 0) * it.quantity for it in brief.items) > brief.cap + 1e-6
        if any(it.chosen is None for it in brief.items) or over:
            self.chooser.choose_and_finish()

    def force_finish(self):
        world, items, cart = self.world, self.brief.items, self.cart
        if world.finished:
            return
        for item in items:
            if item.chosen is None or item.chosen.stock is False:
                fits = self.assessor.buyable(item)
                item.chosen = fits[0] if fits else None
                if item.chosen is None and CLOSEST_RULE and self.brief.no_fit:
                    item.chosen = self.chooser._closest(item)
        ready = bool(items) and all(it.chosen for it in items)
        if ready and self.brief.cap is not None and not (CLOSEST_RULE and self.brief.no_fit and self.brief.no_fit["budget"]):
            ready = sum((it.chosen.price or 0) * it.quantity for it in items) <= self.brief.cap + 1e-6
        if ready:
            chosen_ids = [it.chosen.id for it in items]
            stale = [offer_id for offer_id in cart.lines if offer_id not in chosen_ids]
            adds = [it for it in items if it.chosen.id not in cart.lines]
            if world.steps_left() >= len(stale) + len(adds) + 1:
                for offer_id in stale:
                    cart.remove(offer_id)
                for item in adds:
                    cart.add(item, item.chosen)
                if cart.checkout():
                    return
        if not world.finished:
            cart.decline("forced")


class Driver(Service):
    def run(self):
        journal = self.journal
        try:
            if not self.reader.read():
                self.closer.force_finish()
                return journal.entries
            self.work()
        except Exception as exc:
            journal.note("exception", error=repr(exc)[:300])
        try:
            if not self.world.finished:
                self.closer.force_finish()
        except Exception as exc:
            journal.note("exception", error=repr(exc)[:300])
        journal.note("done", cost=round(self.llm.cost, 4), llm_calls=self.llm.calls, steps=self.world.step)
        return journal.entries

    def _messages_free(self, fallback):
        budget = self.brief.message_budget
        return fallback if budget is None else budget - self.talker.sent - self.talker.held()

    def _hear(self, item, prop, narrow_on_split, extra=None):
        talker, reserve = self.talker, self.allowance.end_reserve
        answer = talker.ask(item, prop, extra=extra)
        split = isinstance(answer, dict) and str(answer.get("kind") or "").lower() == "split"
        if narrow_on_split and split and talker.can_ask() and self.world.steps_left() > reserve():
            # a reply asking for one subject at a time: re-ask with example values
            answer = talker.ask(item, prop, narrow=True)
        talker.take_reply_keeping_steps(item, prop, answer)
        self.watcher.handle_lines()
        # a plain "no preference" changes nothing worth a new plan
        return isinstance(answer, dict) and str(answer.get("kind") or "").lower() not in ("free", "")

    def _partner(self, item, prop, queue):
        # the last questions carry a second property of the same product
        if self._messages_free(len(queue) + 1) > PAIR_LAST:
            return None
        for k, (other, extra) in enumerate(queue):
            if other is item and not same_name(extra, prop):
                queue.pop(k)
                return extra
        spare = self.planner.plan_extra(item, prop)
        return spare

    def _question_round(self, questions):
        talker, world, reserve = self.talker, self.world, self.allowance.end_reserve
        queue = self.planner.plan(questions)
        while queue:
            item, prop = queue.pop(0)
            if not talker.can_ask() or world.steps_left() <= reserve():
                break
            if any(overlap_ratio(n.prop, prop) >= 0.67 for n in item.needs):
                continue
            known = len(item.needs)
            extra = self._partner(item, prop, queue) if PAIR_LAST and len(self.brief.items) == 1 else None
            said = self._hear(item, prop, True, extra)
            if queue and REPLAN_EACH and said and self.clock.left() > WRAP_SECONDS + 90:
                # every reply can change what is worth asking next
                self.planner.planned = None
                queue = self.planner.plan(min(len(queue), self._messages_free(len(queue))))
            elif queue and len(item.needs) != known:
                # a learned need: the rest of the questions skip its property
                queue = self.planner.plan(min(len(queue), self._messages_free(len(queue))))
        # leftover messages go to further properties
        while talker.can_ask() and world.steps_left() > reserve() + 1 and self.clock.left() > WRAP_SECONDS + 30:
            more = self.planner.plan(1)
            if not more:
                return
            extra = self._partner(more[0][0], more[0][1], []) if PAIR_LAST and len(self.brief.items) == 1 else None
            self._hear(more[0][0], more[0][1], False, extra)

    def _follow_all(self, floor):
        for item in self.brief.items:
            for need in list(item.needs):
                if not need.pending():
                    continue
                if floor is not None:
                    self.follower.floor = floor()
                self.follower.resolve(item, need)
                if floor is not None:
                    self.follower.floor = 0

    def work(self):
        brief, world, talker, reserve = self.brief, self.world, self.talker, self.allowance.end_reserve
        items = brief.items
        count = len(items)
        questions = 3 if brief.message_budget is None else brief.message_budget
        questions -= talker.held()
        if questions < 0 or "message" not in self.tools.roles:
            questions = 0
        spare_for_asking = questions + 1 if questions else 0
        # reference chains named in the request run first, with the asking budget held back
        self.follower.floor = spare_for_asking + reserve() + 2 * count
        self._follow_all(None)
        self.follower.floor = 0
        free = world.steps_left() - spare_for_asking - reserve()
        if SPLIT_ROUTE and count > 1:
            # several products: steps after the questions are kept for a search per product and the picks
            free -= 3 * count
        per_item = max(2, free // max(count, 1)) if free > 0 else 0
        if any(st["moment"] == "start" for st in brief.procedure):
            self.gate.run("start", "begin the task (searching the shop)")
            self.watcher.handle_lines()
            questions = max(0, min(questions, self._messages_free(questions)))
        for k, item in enumerate(items):
            share = per_item if k < count - 1 else max(per_item, world.steps_left() - questions - 1 - reserve())
            if SPLIT_ROUTE and count > 1:
                share = min(share, per_item)
            if share <= 0:
                break
            before = world.steps_left()
            self.explorer.explore(item, max(2, share // 2))
            used = before - world.steps_left()
            self.explorer.warm_up(item, max(0, share - used))
        if questions > 0 and talker.can_ask():
            self._question_round(questions)
        self._follow_all(lambda: reserve() + 2)
        # judge everything against what is known now; items without a fit search first
        self.assessor.judge_all()
        order = sorted(items, key=lambda it: (bool(self.assessor.fits(it)), -len(it.pool)))
        for k, item in enumerate(order):
            self.explorer.explore(item, max(2, (world.steps_left() - reserve()) // max(len(order) - k, 1)))
        self.chooser.choose_and_finish()


def assemble(problem_data):
    clock = Clock()
    environment = problem_data["environment"]
    binding = dict(environment["binding"])
    policy = environment["policy_view"]
    brief = Brief(str(policy.get("query") or ""), policy.get("opening"))
    tools = ToolBook([t for t in policy.get("tools") or [] if isinstance(t, dict)])
    world = WorldDesk(binding, int(policy.get("max_steps") or 30), bool(policy.get("finished")))
    book = Catalogue("|".join(sorted(tools.schemas)))
    journal, llm = Journal(), LLMDesk()
    browser, assessor, reader = Browser(), Assessor(), RequestReader()
    talker, gate, planner = Talker(), ProcessGate(), QuestionPlanner()
    follower, watcher, cart, linker = ReferenceFollower(), EventWatcher(), CartHand(), Linker()
    allowance, explorer, chooser, closer = Allowance(), Explorer(), Chooser(), Closer()
    driver = Driver()
    journal.link(clock=clock, world=world)
    llm.link(clock=clock, journal=journal)
    tools.link(llm=llm)
    world.link(clock=clock, journal=journal, tools=tools, cart=cart)
    book.link(llm=llm, journal=journal, tools=tools, brief=brief)
    browser.link(world=world, tools=tools, book=book, clock=clock, journal=journal)
    assessor.link(book=book, brief=brief, llm=llm, journal=journal, clock=clock)
    reader.link(brief=brief, tools=tools, llm=llm, journal=journal, world=world)
    talker.link(tools=tools, world=world, brief=brief, cart=cart, llm=llm, journal=journal,
                watcher=watcher, follower=follower, allowance=allowance)
    gate.link(tools=tools, world=world, brief=brief, cart=cart, llm=llm, journal=journal, clock=clock, talker=talker)
    planner.link(book=book, brief=brief, llm=llm, journal=journal, linker=linker)
    follower.link(world=world, clock=clock, browser=browser, book=book, tools=tools, llm=llm, journal=journal)
    watcher.link(world=world, brief=brief, llm=llm, journal=journal, cart=cart, browser=browser)
    cart.link(tools=tools, world=world, brief=brief, journal=journal, gate=gate, talker=talker,
              watcher=watcher, closer=closer, chooser=chooser)
    linker.link(brief=brief, book=book, world=world, clock=clock, browser=browser, journal=journal, llm=llm,
                assessor=assessor, allowance=allowance, explorer=explorer)
    allowance.link(tools=tools, brief=brief, talker=talker)
    explorer.link(world=world, brief=brief, tools=tools, book=book, browser=browser, assessor=assessor, clock=clock,
                  llm=llm, journal=journal, watcher=watcher, allowance=allowance)
    chooser.link(brief=brief, assessor=assessor, linker=linker, watcher=watcher, clock=clock, llm=llm, journal=journal,
                 world=world, allowance=allowance, explorer=explorer, cart=cart, closer=closer, follower=follower,
                 talker=talker)
    closer.link(world=world, brief=brief, cart=cart, assessor=assessor, journal=journal, chooser=chooser, talker=talker, tools=tools,
                browser=browser, watcher=watcher, allowance=allowance, explorer=explorer, clock=clock, linker=linker)
    driver.link(reader=reader, closer=closer, journal=journal, world=world, llm=llm, brief=brief, talker=talker,
                clock=clock, tools=tools, follower=follower, allowance=allowance, gate=gate, watcher=watcher,
                explorer=explorer, planner=planner, assessor=assessor, chooser=chooser)
    return driver


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    return assemble(problem_data).run()


__all__ = ["agent_main"]
