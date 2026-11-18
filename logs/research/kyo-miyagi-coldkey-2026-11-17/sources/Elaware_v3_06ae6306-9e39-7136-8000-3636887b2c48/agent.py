import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.agent.proxy_client import ProxyClient

# ---------------------------------------------------------------- A1 settings
M_READ = "anthropic/claude-sonnet-5.5"
M_PLAN = "anthropic/claude-sonnet-5.5"
M_VERIFY = "anthropic/claude-sonnet-5.5"
M_FAST = "anthropic/claude-haiku-5.5"
M_BACKUP = "openai/gpt-6-luna"
SONNET_ROLES = (M_READ, M_PLAN, M_VERIFY)
TASK_CLOCK = 255.0
MARGIN = 45.0
SUMMARY_CAP = 2200
ROW_TEXT_CAP = 260
SOFT_SPEND = 0.03
STANDIN_CAP = 4
HELPER_CAP = 3
HELPER_NEEDS = ('request', 'requirements', 'budget', 'quantity', 'candidate', 'shortlist', 'picked', 'cart')
HELPER_KINDS = ('number', 'flag', 'text', 'requirement')
SPEND_CAP = 0.05
TOL = 1e-6
SERIES_RE = re.compile(r"\b([A-Z]{1,5}[0-9]{1,3})-[0-9]{2,}\b")
CODE_RE = re.compile(r"\b[A-Z]{1,6}-?[0-9]{3,}[A-Z0-9]*\b")
SHORT_CODE_RE = re.compile(r"\b[A-Z]{1,5}[0-9]{1,3}\b")
SEP_RE = re.compile(r"[,;/|]")
REL_TABLE = {
    "same as reference": "is",
    "different from reference": "is not",
    "counterpart of reference": "is",
    "at least reference": "at least",
    "at most reference": "at most",
}
PLAIN_VALUE_OPS = ("is", "one of")
REPLAN_WHEN = ("never", "after_needs", "after_refs", "after_pick", "on_change")
SPEND_FLOOR = 0.05

# ---------------------------------------------------------------- A2 transports
_MODEL = ProxyClient(timeout=75, max_retries=1, retry_delay=1)
_ENV = ProxyClient(timeout=90, max_retries=2, retry_delay=1)
WORLD = {}
_WORLD_LOCK = threading.Lock()

# ---------------------------------------------------------------- prompts
P_READER = """You prepare a shopping assistant for one job in a simulated store. You receive what the shopper wrote and the store's tools as OpenAI-style function schemas. Answer with one JSON object and no text around it.

Fill in this template. The // remarks explain each field; do not copy them into your answer.
{
  "products": [                      // one entry per distinct product to be bought; usually there is exactly one
    {
      "name": "...",                 // a plain noun phrase, naming the product the way the request names it; for a request written as a list, that list's own words for this entry
      "category": "...",             // the store's category, or else the type of product wanted, worded as in the request
      "quantity": 1,                 // how many listings of it to buy (reading rules 2 and 3)
      "unit_cap": null,              // the most the shopper will pay for ONE unit of THIS product, when the request limits this product alone ("each item at 4900 or below", "no more than N for a X"); a bare number in the request's unit; null when the only limit is the order total
      "requirements": [              // every firm demand the request makes on this product's own properties; never budget, price, quantity or stock
        {
          "property": "...",         // the property name, spelled the way a listing would spell it
          "operator": "is",          // exactly one of: is | is not | at least | at most | one of | contains | same as reference | different from reference | counterpart of reference | at least reference | at most reference
          "value": "...",            // the value, bound or alternatives in the request's exact wording; for one of, the alternatives joined as "A | B"
          "reference": null          // null except with the five reference operators; then {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "the property whose value is read on that page", "count": null, "pick": null}; count is null or the number of listings the request says fit the description; pick is null or how to choose among several (for instance: take the wider one)
        }
      ],
      "hints": [],                   // names of properties the story implies matter privately to the shopper (saying who will wear it implies age group or gender; talking about a computer's internals implies the processor); empty if none
      "search_terms": ["..."]        // 2 to 4 keyword queries for the store's search, most specific first. A listing is returned only if it matches every word of a query, so keep each query short: the head noun plus one or two value words (a colour, a material, a figure with its unit such as a voltage or a memory size). A property name is never a query word on its own. A value that only serves to identify a referenced listing never appears (rule 6). The last query is the bare head noun.
    }
  ],
  "price_cap": 12345,                // the largest total price allowed, as a bare number in the request's own unit (in cents if the request counts in cents); null if none
  "cheapest": false,                 // true only if the request explicitly wants the cheapest product, the lowest price or the lowest total
  "best_measure": null,              // if the request wants the top product by some measure, that measure; otherwise null
  "message_allowance": null,         // how many messages the shopper is willing to receive; null if not stated
  "move_limit": null,                // how many tool calls or steps the request grants; null if not stated
  "in_stock_only": true,             // the shopper accepts only listings that can really be obtained
  "topics_to_raise": [],             // subjects the request tells the assistant to bring up with the shopper, worded as in the request; empty if none
  "fit_rules": [],                   // for instructions like "check that this one goes with that one": [{"accessory": "product name", "host": "product name"}]; empty if none
  "conditions": [],                  // a demand on one product that depends on how another product turns out; empty if none. Each: {"if": {"product": "name", "property": "...", "values": ["..."]}, "then": null or {"product": "name", "property": "...", "operator": "is | is not | at least | at most | one of", "value": "..."}, "otherwise": null or the same shape as then}
  "no_match_instruction": null,      // the request's own instruction for the case where nothing qualifies, copied complete and word for word, keeping any figure or percentage it states and any rule it gives for choosing the stand-in (shoppers variously want only to be told; want no replacement at all; or want the nearest listing named, sometimes defining nearness by how many demands it misses and allowing a margin over the lowest such price); else null
  "procedure": [                     // steps the request imposes on how the job is done: a sign-off or confirmation to collect, a tool to be used, something the shopper must be shown or told, a required order of actions. Not what the product must satisfy, not budgets, not limits on messages or steps, not the no-match instruction (it has its own field). Empty if none.
    {
      "moment": "pre-order",         // one of: start | pre-order | pre-decline | pre-finish
      "kind": "message",             // message or tool
      "tool": null,                  // a tool's name if one is meant, else null
      "goal": "...",                 // what has to happen, phrased as the request phrases it
      "messages": 1                  // how many shopper messages the step uses; 0 for a tool step. Checking something inside the store yourself (stock, availability, some value) is a tool step using 0 messages, unless the shopper expects to hear the outcome.
    }
  ],
  "tools": {                         // map each role to one of the store's tools; every name is copied exactly from the schemas (rule 10)
    "keyword_search": {"tool": "...", "query_arg": "...", "page_arg": null, "page_kind": null, "mode": null},
                                     // page_arg: the argument that moves on to further results, or null; page_kind: number (a page index) | offset (rows to skip) | token (a continuation token returned by the last answer), or null; mode: null or {"arg": "...", "value": "..."} choosing keyword mode
    "field_search": {"tool": "...", "category_arg": null, "brand_arg": null, "max_price_arg": null, "page_arg": null, "page_kind": null, "mode": null},
                                     // or null: a tool searching by category, brand and highest price, or the same search tool in its field mode
    "open": {"tool": "...", "id_arg": "..."},          // the tool showing one listing in full
    "facts": {"tool": "...", "id_arg": "..."},         // or null: a separate tool returning one listing's property table
    "compare": {"tool": "...", "ids_arg": "..."},      // or null: a tool showing several complete listings at once; ids_arg carries the list of identifiers
    "cart_add": {"tool": "...", "id_arg": "...", "quantity_arg": null, "mode": null},
                                     // adds one listing to the cart; mode: null or {"arg": "...", "add": "value that adds", "remove": "value that removes"}
    "cart_set": {"tool": "...", "lines_arg": "...", "line_id_key": "...", "line_quantity_key": "..."},
                                     // or null: a tool replacing the whole cart from a list of lines; the keys are those inside one line
    "cart_view": {"tool": "..."},    // or null: shows the cart
    "checkout": {"tool": "...", "fixed_args": null},   // places the order; nothing can follow it. fixed_args: null or {"argument": "value"}
    "decline": {"tool": "...", "remark_arg": null, "fixed_args": null},   // ends the task with no purchase; remark_arg: a free-text reason argument, or null
    "message": {"tool": "...", "text_arg": "..."},     // writes to the shopper
    "status": {"tool": "..."}        // or null: reports the allowances still left
  }
}

Reading rules:
1. Firm against soft.
   a. Only a firm demand of the request is a requirement.
   b. Wishes (signalled by words like ideally, or by saying something would be pleasant) and story details (the person who will use the product, the place where it will stay) are not requirements; whatever they imply goes into hints.
   c. Words that merely name the product type or its category, such as a gender or age word placed before the noun, or a qualifier that is part of a category's name, go into name or category and never into requirements.
   d. Exception: when the request supplies a value for a property it names (wanting it in the V style, with V as its P, a P of V), that property is constrained by V, even when V resembles a product type; the category still stays the product the shopper named.
2. A list line shaped like "N times some product, whose property A is x and whose property B is not y" is one product with quantity N and one requirement per clause (one is, one is not).
3. A size given for the contents of one listing (a box, set or pack holding N pieces) constrains the pack-size or set-size property; the quantity says how many such listings are bought.
4. A clause that only restates the consequence of another requirement (a wanted fastening type, followed by a remark that other fastening types are therefore out) is one requirement on the wanted value, not several; never make up properties for the alternatives it excludes.
5. Operators: two alternatives joined by "or" are one of, with the value written "A | B"; a negation (a plain not, or anything but) is is not; a minimum with a unit is at least, the value keeping both the number and the unit. Values keep their unit and spelling exactly as written.
6. References.
   a. A requirement aimed at a different listing's page (identified through category, seller and a few attribute values) is a reference requirement. Wanting the identical value: same as reference. Wanting any value but that one: different from reference. Wanting the value that partners it (paired with it by that page or by the store, the value that belongs with it): counterpart of reference. Wanting a value not below that page's: at least reference; not above it: at most reference.
   b. The named page may itself just redirect this property to yet another listing; the assistant traces such chains on its own, so describe only the page actually named.
   c. Fill the reference with four things: category, seller, attributes (the described values) and property (the wanted property).
   d. If the request says several of that seller's listings qualify and gives a selection rule (the bigger one, the cheaper one, whichever has more of some quantity), put the rule into pick and the number into count.
   e. attributes hold the plain values that pick out that listing; the comparison itself is never stored there.
   f. These identifying values describe the other listing only: they never become requirements on the product to buy, nor search words for it.
   g. Made-up example: "for the colour, avoid whatever seller Brightway's garden hose page (the 15 m rubber one) shows". Correct reading: exactly one requirement, property colour, operator different from reference, value "", reference {"category": "garden hose", "seller": "Brightway", "attributes": {"length": "15 m", "material": "rubber"}, "property": "colour", "count": null, "pick": null}; no requirement for the length or the material.
7. A demand on one product that applies only if another product turns out a certain way goes into conditions, never into requirements or procedure. "Should X's P be V, then Y's Q has to be W" fills then, which applies when X's P is among the listed values. "Y's Q has to be W, save when X's P is V" fills otherwise, which applies when X's P is none of them.
8. The category means the product type; if the request explicitly states the store category, keep its exact wording.
9. A total given in cents makes price_cap a number of cents.
10. Tools.
   a. Copy tool and argument names character for character from the schemas.
   b. If one single tool can end the task both ways, through an argument selecting either buying the cart or leaving with nothing, name it for both checkout and decline, each with fixed_args holding the value that selects its own ending, spelled exactly as the schema's list spells it.
   c. A tool described as final that closes without a purchase fills decline; the tool that buys the cart's contents fills checkout.
   d. If one search tool has an enumerated argument switching between keyword mode and field mode, point both search roles at it, each with its own mode value."""

P_SCREEN = """You screen search results for a shopping agent. Each row shows an identifying number, a title, some brief text, a category label and a price. For every row decide whether that listing could still meet the requirements once its full page has been read.
- Decide from what the row itself shows, nothing else.
- A row that says nothing about some requirement is still possible: leave it as maybe.
- A row whose wording clearly clashes with a wanted value is no.
- A row that itself spells out the wanted values is yes.
- The kind of product comes from the label the store files the row under. A row filed under the asked-for category is the right kind of product whatever its headline suggests, so the kind alone never justifies no.
- Rows are referred to by their numbers.
Reply with one JSON object and nothing around it: {"yes": [row numbers], "no": [row numbers]}. Rows in neither list are maybe."""

P_JUDGE = """You judge whether store listings meet a shopper's requirements, using only the evidence each listing carries.

1. A requirement is met when this very listing's property table has an entry for that property carrying the value.
   a. Key spellings vary (spacing, underscores, prefixes, shortened forms). Every key whose name means the property counts, and one property may sit under more than one key.
   b. Compare values by meaning: matching any single member of a comma list is enough; singular equals plural; close synonyms and evident short forms count; letter case is ignored; a figure with a unit matches once units are converted.
   c. Compare figures as numbers; a bound counts its own edge as inside; a range satisfies a bound only if all of it does.
2. Which evidence wins.
   a. The table first: if some entry covers the requirement, that entry's value settles it, and a differing value fails it even against a title that claims otherwise.
   b. Only if no key covers it may a value written elsewhere be used: in the seller notes, in a variant or option field, in the title or in the description.
   c. Nothing anywhere: the requirement is unsure.
   d. A value under a neighbouring but different property is no evidence (a handle entry is not the lock type; a lining's fabric is not the product's own fabric).
3. Operators.
   a. Exclusion (is not, different from) fails when the table holds the excluded value.
   b. A closed list (one of) needs a member of that list.
   c. contains needs that value among the listing's stated values.
   d. A wanted value meaning the feature is missing (none, without) is met when no evidence mentions the feature.
4. Kind of product: the store's label decides it. A listing labelled with the asked-for category is the right kind regardless of its headline. Only a listing with no label at all may be rejected as an add-on, a replacement piece or a protective cover instead of the product proper.
5. Compatibility with some other product (identified by model names or by codes).
   a. Met only if the text of this very listing includes one of those names in what it is said to suit or work with.
   b. If that other product's page gives a code series (a family of compatible parts), a listing whose own part code (often in brackets in its title) begins with that series meets it.
   c. A sentence claiming that some different part suits a product is about whichever listing has that part's code in its title, not about the page printing the sentence. The related_pages entry shows such pages; what they say about this listing's code counts for this listing.
   d. Merely pointing elsewhere proves nothing.
6. Redirected properties: seller notes may pass one property on to another listing (meaning: take its value from there, not from here). Then the value this listing itself shows for that property is ignored: a requirement touching it stays unsure, unless an attached related page gives it. Notes handing over a property that no requirement mentions change nothing.
7. Store fields: when that block is present it shows, per wanted property, the key names and written values used by the listings of this category read in this store. When some of them hold the exact wanted value under that key, any listing with some other value under that key fails, including a related or more general value; a value absent from every read listing can still count through a plain synonym.
8. Judge only the listed requirements; never make up others.
9. Requirements carry the tags R1, R2 and so on. Reply with one JSON object and no commentary:
{"verdicts": [{"id": "listing id", "failed": ["R1"], "unsure": ["R2"], "weak": ["R3"], "loose": ["R4"]}]}
with an entry for every listing; leave out empty lists.
   - failed: clearly failed by the evidence;
   - unsure: addressed by no evidence;
   - weak: met only through the headline, the description text or an option field, never through a table entry or a note;
   - loose: met through a synonym, a wider word or a conversion rather than the value's own wording (a female gender entry standing in for a requirement asking for girls).
10. A listing with neither failed nor unsure tags meets everything listed."""

P_PLANNER = """You plan which questions a shopping agent will put to a shopper.
- The shopper keeps hidden requirements. Each fixes a value for exactly one property of one wanted product, and it comes out only once a message names that exact property (its name or a plain synonym).
- Only the first property a message names gets answered. A question about something the shopper does not care about just draws an indifferent reply, and that message is wasted.
- You get the request and, per wanted product, its name, the requirements already known, hints, and a menu showing which properties the product's candidate listings really have, with how many listings carry each and sample values.
- Reply with exactly the number of questions the user message gives, best first, as one JSON object: {"questions": [{"product": "product name as spelled in the input", "property": "property name as spelled in the menu, words separated by spaces"}]}.
- Choose first the properties the story and the hints lead to, that is whatever the request announces the shopper would reveal on being asked: the wearer or user suggests age group or gender; fastening suggests closure; the hardware inside suggests processor; looks suggest style, colour or pattern. Map each hint to the closest menu name.
- Next, properties buyers of that kind of product usually care about, favouring those many candidates carry: material, look, size or fit, type or style, a variant such as flavour, intended user, the main technical figure; widely carried and with values that vary across listings.
- Weak questions: a property whose value never varies; a pseudo-property such as a variant or option picker; any identifier or model code; warranty or delivery remarks.
- Never a property a known requirement already fixes; never the price, the stock level or the quantity; never the same property twice (a second spelling still names one property).
- Share the questions among the products according to how uncertain each one is; the product with strong hints is asked first."""

P_WORKFLOW = """You schedule the work of a shopping agent that has several products to buy in one task. The agent can only work on one product at a time, and a local controller carries out your schedule exactly as you give it, so the order and the shares you choose are what will happen.

You receive the shopper's request, the products already read out of it, and the dependencies the agent has established in its own code. A dependency means that choosing one product fixes a requirement of another, so a dependent product must never be scheduled before the product it depends on.

Reply with one JSON object and nothing around it:
  "order": every product name exactly once, in the order the agent should work them.
  "slots": one entry per product, using the same names:
      "product": the name,
      "asks": how many of the shared question allowance this product may spend, a whole number, 0 when its requirements are already settled,
      "steps_share": this product's share of the store steps, a number from 0 to 1; the shares should total about 1,
      "spend_share": this product's share of the money, a number from 0 to 1, judged by what this product costs next to the others; the shares should total about 1,
      "needs_from": the product whose chosen listing fixes a requirement here, or null,
      "take": the property to carry over from that chosen listing, or null,
      "as": the name that property takes in this product, or null,
      "use": the names of the helpers this product needs, or an empty list.
  "helpers": the working-out this agent cannot already do for itself, as a list, empty when it needs none. The agent already judges a listing against a requirement, compares prices, converts units, ranks by a measure and checks stock and budget on its own, so never ask for those. Ask only for a calculation its code has no form for - combining figures across several chosen products, deriving a value one product implies for another, or a test that depends on more than one listing at once. Each entry is:
      "name": a short name,
      "purpose": one sentence telling the worker exactly what to work out and by what rule,
      "needs": which facts the agent should hand the worker, drawn only from: "request", "requirements", "budget", "quantity", "candidate", "shortlist", "picked", "cart",
      "returns": one of "number" (a figure), "flag" (true or false, used to accept or reject the listing in hand), "text" (a short phrase) or "requirement" (a "property: value" line that becomes a firm requirement of this product).
    Each call costs money and time, so declare at most two and only when the agent truly cannot do it.
  "replan_when": exactly one of "never", "after_needs", "after_refs", "after_pick", "on_change" - the single moment at which this schedule should be reconsidered. Nothing is spent by answering "never", so answer "never" unless information arriving at that moment could really change the order or the shares.
  "why": one short sentence.

Give the larger step share to the product that is harder to pin down, and no questions to a product whose requirements are already fixed. A product whose choice constrains another belongs early and usually needs the room to be chosen well."""


P_REPLY = """You read a shopper's answer to a shopping agent's question. The question concerned one property of one product. Reply with one JSON object and nothing around it:
{"kind": "need | repeat | free | split | unclear | other",
 "property": "the subject of the shopper's answer in their own wording, or the asked property when that is what they mean",
 "op": "is | is not | at least | at most | one of | same as reference | different from reference | counterpart of reference | at least reference | at most reference",
 "value": "whatever the shopper gave: one value, a bound, or a list with members joined as A | B",
 "reference": null,
 "budget": null,
 "answer": null,
 "steps": []}
- reference: null, or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "which property to read on that page", "count": null or the number of listings the shopper says qualify, "pick": null or a rule selecting one listing out of several}.
- budget: null, or a number when the shopper names a new spending total.
- answer: null, or a short direct answer when the shopper asked the agent something.
- steps: process steps the shopper newly insists on (getting an approval, being shown or told something, using some tool before ordering), each {"moment": "pre-order | pre-decline | pre-finish", "kind": "message | tool", "tool": null, "goal": "what has to happen", "messages": 1}; empty if none.
Kinds:
- need: the shopper insists on something (a value, a bound, an exclusion, a list), however it is phrased: stated as settled, implied by exclusivity ("only this will do"), or put as an exclusion or a closed choice ("this or that and nothing more").
  - "Anything except X" or "other than X": op is not, value X.
  - Pointing at some other listing's page (identified by category, seller and a few attribute values) and wanting whatever that page shows: same as reference, with the reference filled in.
  - Anything except the value on that page: different from reference.
  - Whatever value that page names as the partner of its own (its counterpart, the thing that belongs with it): counterpart of reference.
  - Nothing below, or nothing above, the page's value: at least reference or at most reference.
- repeat: instead of answering about the asked property, the shopper says again something the known needs already hold (same property and value).
- free: indifference: leaving it open, anything is fine, never thought about it.
- split: the shopper wants the question divided, its topics put one at a time, or answers only its opening part because it mixed several topics.
- unclear: the shopper did not follow, or asks for a repeat or simpler wording.
- other: everything else (greetings, a counter-question, a budget change).
Property: when the reply names a property, that name in their wording is the property even if it differs from the asked one; only a reply naming none falls back to the asked property.
Bounds: explicit bound wording (at most, under, no more than; at least) followed by a figure fixes the operator and the value even inside an oddly built sentence: "I can't see myself not taking one at most 30 inches long" is at most, value 30 inches."""

P_LINES = """While a shopping agent was working, a line from the shopper came back inside a store tool's result. Report what changed. Reply with one JSON object and nothing around it:
{"changes": [{"kind": "budget | drop | add | price | stock | step | other", "budget": null, "property": null, "op": "is", "value": null, "product": null, "note": "one short sentence", "step": null}]}
- budget: a new amount the shopper can spend, up or down. Copy the figure into budget exactly as written, in the line's own unit: cents remain cents, never convert.
- drop: the shopper no longer requires some property (let it go, drop it, never mind); give the property.
- add: a fresh requirement; give property, op (is | is not | at least | at most | one of) and value.
- price: the shopper reports a price movement on the picked listing.
- stock: the shopper reports that the chosen listing can no longer be bought.
- step: the shopper insists on a process step (an approval, something to be shown or told, some tool to be used before ordering); give step as {"moment": "pre-order | pre-decline | pre-finish", "kind": "message | tool", "tool": null, "goal": "what has to happen", "messages": 1}.
- other: anything else.
product: the wanted product concerned, or null."""

P_PAGE = """You read one listing page for a shopping agent that is following a reference: the agent needs the value this page gives for the wanted property.
- The value can be in the property table or among the seller notes; or those notes can tell the reader to take that property from a different listing instead (identified by category, seller and a few attribute values).
- The notes can also spell out pairings between values of that property: either the one partner of the value this page has, or a complete pairing list for the whole store.
Reply with one JSON object and nothing around it: {"value": "the value belonging to this page, verbatim, or empty", "paired": "its partner as the page states it, or empty", "redirect": null or {"category": "...", "seller": "...", "attributes": {"property": "value"}, "property": "..."}}.
- If the notes send the wanted property elsewhere: give the redirect and leave value empty. Notes sending a different property elsewhere are not a redirect of the wanted one.
- If the notes give the value directly: report it.
- Otherwise take it from the table, under whichever key means the property; failing any such key, from the option field, the title or the description, whichever of them states it.
- A sentence merely listing pairings for the store does not show which value belongs to this page.
- The paired value must come from a pairing the page itself states: from a pairing list, the pair that includes this page's own value (read in either direction). Leave it empty when the page gives no pairing or this page's own value is not known."""

P_QUERIES = """You write search-box keywords for a shopping agent.
- The store's search returns a listing only if every query word occurs somewhere in it (its title, brand, category or table values); stems match, and very short words are skipped.
- Hits come back in an unchanging order with few rows per page, so every query should be narrow enough that most hits already satisfy the requirements.
- You get the product with its requirements, the queries run so far, and a sample of listings seen, each marked as meeting the requirements or not.
Reply with one JSON object and nothing around it: {"queries": ["..."]}: at most 5 fresh queries, each 1 to 3 words, strongest first.
- A good query is the noun this store's titles use for the product plus one wanted value in the listings' own wording, or other words this store uses for the product or the value.
- Never repeat a query that was already run."""

P_HELPER = """You stand in for a function a shopping agent does not have. The agent's planner has written what this function must work out, and the agent has gathered the facts it needs. Do only that piece of working-out.

You are told the purpose, the facts, and the kind of answer wanted. Reply with one JSON object and nothing around it: {"value": ...}, where value is
 - a bare number when a number is wanted,
 - true or false when a flag is wanted,
 - a short phrase when text is wanted,
 - a single "property: value" line when a requirement is wanted, naming the property as a listing would and giving the value the product must have.
Work only from the facts given; where they do not settle it, answer with the most defensible reading rather than refusing. Add no explanation."""


P_STANDIN = """A shopping assistant has to call one of a store's tools, but its own code could not work out every argument that tool demands. You get the tool's name and description, its argument schema, the arguments already prepared, what the assistant is trying to do at this moment, and, when the store has already refused an attempt, the words of that refusal.

Reply with one JSON object and nothing around it: the complete argument object for this one call. Include the arguments already prepared unless the refusal shows one of them was wrong. Copy every argument name and every permitted value exactly as the schema spells it. Supply only arguments the schema lists and nothing else. Where the schema fixes a set of values, choose the one that serves the stated purpose. Where it asks for a number, give a bare number; where it asks for a list, give a list of the shape the schema describes. Guess as well as the schema allows rather than leaving a demanded argument out."""


P_CHOOSER = """A store tool closes the shopping task, and one of its arguments decides in which way it closes. You get the tool's description, the arguments still missing with each one's permitted values, and the goal. Reply with one JSON object and nothing around it, mapping each missing argument name to the permitted value that achieves the goal."""

P_GATE = """You audit, at one point of a shopping task, whether the agent's way of working obeys the shopper's request.
You get: the request; the process steps it demands; the moment (the agent's intended next action); the present basket; the allowances left; the store's tools (those that close the task or change the cart are withheld, because the agent runs them itself); and every earlier call together with what it returned.
- Find the process steps owed right now, ahead of the agent's next move, and still undone for the present state.
- Approvals and confirmations hold only for the present basket in its exact form; one obtained for an earlier basket is void.
- A step already carried out for the present state is not owed again.
- A step tied to a different moment is not owed now.
- Reply with one JSON object and nothing around it: {"actions": [{"tool": "exact tool name", "args": {"argument": "value"}, "why": "which process step it fulfils"}]}, at most 2 actions in order; an empty list when nothing is owed.
- A message to the shopper must stand on its own: for a basket it lists each line with its quantity, title and price, gives the overall sum, and says plainly what it asks of the shopper.
- No product-property question belongs in such a message, and the last remaining message may serve only a process step.
- Checks inside the store (stock, availability, some value) go through the store's read tools and never through a message, except where the request wants the shopper informed. The agent reads each listing before buying it anyway, so such a check is owed only for basket listings that were never read."""

P_RANK = """Every listing below already satisfies the shopper's requirements; the shopper wants the top one by a stated measure (a value to maximise or to minimise).
- For each listing read how it scores on the measure from its table, its title or its description; convert units where they differ so the values become comparable.
- Reply with one JSON object and nothing around it: {"order": ["every listing id, best first"], "values": {"listing id": "value read"}}.
- Listings whose value cannot be read go to the end; on equal values the lower price comes first."""

P_AUDIT = """You take a second look at how a shopping agent searched for one product, just before it abandons that product.
You get the product with its requirements, the budget, the queries already run, and all listings seen: those that were read (with table, notes, and the requirements the agent judged failed or open) and rows known only from search results.
Reply with one JSON object and nothing around it, holding three lists, each possibly empty:
{"fits_after_all": ["ids of already-read listings that in fact satisfy every requirement, judged on the evidence each carries: a table value decides, a differing value fails, and a property no evidence states counts as unmet"],
 "unopened": ["ids of never-read rows that most probably satisfy the requirements, strongest first, at most 4"],
 "queries": ["at most 2 fresh keyword queries of 1 to 3 words that would probably surface fitting listings, none already run"]}"""

P_COND = """A shopping agent tests one listing against a condition the shopper set. Decide whether this listing's value for the named property falls inside the supplied set of values (meaning counts; convert units), using only evidence on the listing itself.
Reply with one JSON object and nothing around it: {"holds": true or false, "value": "what the listing shows, verbatim, or empty"}.
If no evidence states the property: holds false, value empty."""

P_MATCH = """A shopping agent is hunting one specific listing, identified by category, seller and attributes. Candidates with their evidence follow.
Reply with one JSON object and no explanation: {"id": "the matching candidate's id"}, or {"id": null} when none matches.
- The seller must agree.
- Its category label must denote the described kind.
- Every described attribute must agree with that candidate's evidence (meaning counts; convert units).
- For a candidate shown without a property table, its notes, title and description serve as evidence, so notes that spell out the described values are enough to qualify.
- With a pick rule, more than one listing may qualify: apply the rule among the qualifying ones and return its choice."""

P_VERIFY = """You are the last check before a shopping agent orders a listing. Test the listing on each listed requirement, using only evidence the listing itself carries.
- A requirement is met when this listing's table entry carries the wanted value (any key whose name means the property; comma lists, synonyms, short forms, singular or plural, and unit conversions all count).
- Where no table key covers a requirement, it may be carried by the seller notes, an option field, the title or the description.
- A requirement nothing states is unmet.
- A value under some other property is no evidence.
- Compatibility: met if this listing names the other product as something it suits, or if that other product's page lists a code series for compatible parts and this listing's part code begins with it.
- When the notes defer a property's value to a different listing, the value this listing shows for it does not meet the requirement.
- Exclusions and closed lists of values have to be obeyed.
- A listing labelled with the asked-for category is the right kind of product, regardless of its headline.
- The price multiplied by the quantity has to fit the budget, in the unit of the prices shown.
- Evidence that it cannot be bought means failure.
- Test only the requirements given; never invent more.
Reply with one JSON object: {"fits": true or false, "problems": ["every requirement clearly unmet or addressed by no evidence"]}."""

P_LABELS = """A store files its listings under category labels, and the shopper wants one product type. Of the labels met so far, name every one that denotes that type: any label whose meaning fits the request's whole wording, a broader label that includes the type counting too (a combined knitwear label covers a wanted cardigan). Leave out adjacent types: add-ons, spare parts, protective covers for the product, and types that are narrower or different.
Reply with one JSON object and nothing around it: {"labels": ["each label spelled exactly as supplied"]}; an empty list if none fits."""

P_KEYMAP = """The record below came back from a store tool. For each role listed, name the key of the record that fills it: copy the key's name exactly as it is spelled in the record, or give null when no key fills it. Answer with key names only; never copy a value out of the record.
Reply with one JSON object over these fourteen roles, shaped like {"id": "<key name>", "title": "<key name>", "price": null, ...}:
- id: the key holding the listing's identifier
- title: the key holding its name
- text: the key holding its descriptive text
- price: the key holding its price
- stock: the key saying whether it can be bought
- category: the key holding the label it is filed under
- brand: the key holding its brand
- seller: the key holding the seller or shop
- facts: the key whose value is the nested object that is its property table
- notes: the key holding free-text remarks from the seller
- total: the key holding what a cart costs overall
- lines: the key holding a cart's list of lines
- next: the key holding a cursor (a token or some marker) for fetching more results
- count: the key holding how many results match overall"""


# ---------------------------------------------------------------- A3 text helpers
def norm(x):
    if not x:
        return ""
    s = str(x).lower()
    s = s.replace("&", " and ").replace("_", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return s.strip()


def _fold(p):
    if len(p) >= 4 and p.endswith("s"):
        return p[:-1]
    return p


def toks(x):
    return {_fold(p) for p in norm(x).split() if len(p) >= 2}


def words(x):
    return {_fold(p) for p in norm(x).split() if p}


def near(a, b):
    na, nb = norm(a), norm(b)
    if na == nb:
        return True
    ta, tb = toks(a), toks(b)
    return bool(ta) and ta == tb


def sim(a, b):
    ta, tb = toks(a), toks(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / float(len(ta | tb))


def txt(v):
    if v is None:
        return ""
    return str(v).strip()


def split_caps(k):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(k))


def lead_caps(code):
    m = re.match(r"[A-Z]+", code)
    return m.group(0) if m else ""


def best_by_sim(value, options, threshold):
    best, bs = None, -1.0
    for o in options:
        s = sim(value, o)
        if s > bs:
            best, bs = o, s
    if best is not None and bs >= threshold:
        return best
    return None


# ---------------------------------------------------------------- A4 readers
def _scan(t, open_c, close_c):
    i = t.find(open_c)
    if i < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for j in range(i, len(t)):
        ch = t[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_c:
            depth += 1
        elif ch == close_c:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(t[i:j + 1])
                except Exception:
                    return None
    return None


def extract_json(text):
    if not text:
        return None
    t = text.strip()
    if not t:
        return None
    m = re.search(r"```(?:json|JSON)?[ \t]*\n?(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    for o, c in (("{", "}"), ("[", "]")):
        r = _scan(t, o, c)
        if r is not None:
            return r
    try:
        return json.loads(t)
    except Exception:
        return None


def cj(v):
    return json.dumps(v, separators=(",", ":"), ensure_ascii=False, default=str)


def cut(v, limit):
    s = cj(v)
    if len(s) > limit:
        return s[:limit] + "…"
    return s


def num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v.replace(",", ""))
        if m:
            return float(m.group(0))
    return None


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def fmt_num(x):
    return "%g" % x


def is_sentence(v):
    return isinstance(v, str) and len(v) > 24 and " " in v.strip()


def find_records(obs):
    if isinstance(obs, list):
        if obs and all(isinstance(e, dict) for e in obs):
            return obs
        return None
    if not isinstance(obs, dict):
        return None
    found = []

    def walk(d, level):
        for v in d.values():
            if isinstance(v, list):
                if v and all(isinstance(e, dict) for e in v):
                    found.append(v)
            elif isinstance(v, dict) and level < 2:
                walk(v, level + 1)

    walk(obs, 1)
    best = None
    for f in found:
        if best is None or len(f) > len(best):
            best = f
    return best


def split_lines(obs):
    if not isinstance(obs, dict):
        return obs, []
    n_map = sum(1 for v in obs.values() if isinstance(v, dict))
    looks_record = any(isinstance(v, str) and v.strip() and " " not in v.strip() for v in obs.values())
    lines = []
    res = {}
    for k, v in obs.items():
        if isinstance(v, list) and v and all(is_sentence(e) for e in v):
            lines.extend(v)
        elif is_sentence(v) and n_map == 1 and len(obs) <= 3 and not looks_record:
            lines.append(v)
        else:
            res[k] = v
    if lines and len(res) == 1:
        only = list(res.values())[0]
        if isinstance(only, dict):
            res = only
    return res, lines


# ---------------------------------------------------------------- A5 parallel runner
def safe(job):
    try:
        return job()
    except Exception:
        return None


def parallel(jobs):
    if len(jobs) <= 1:
        return [safe(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=min(3, len(jobs))) as ex:
        futs = [ex.submit(safe, j) for j in jobs]
        return [f.result() for f in futs]


# ---------------------------------------------------------------- A6 model client
class LLM:
    def __init__(self, ep):
        self.ep = ep
        self.spend = 0.0
        self.calls = 0
        self.fails = 0
        self.lock = threading.Lock()

    def chat(self, model, system, user, cap):
        if self.spend >= SOFT_SPEND and model in SONNET_ROLES:
            model = M_FAST
        left = self.ep.time_left()
        if left < MARGIN - 12:
            return None
        timeout = int(max(10, min(45, left - 25)))
        if model.startswith("anthropic/"):
            sysmsg = {"role": "system", "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]}
        else:
            sysmsg = {"role": "system", "content": system}
        body = {"model": model, "messages": [sysmsg, {"role": "user", "content": user}],
                "temperature": 0, "max_tokens": cap}
        with self.lock:
            self.calls += 1
        data = None
        err = None
        for attempt in range(3):
            _MODEL.timeout = timeout
            r = _MODEL.post_verbose("/inference/chat/completions", body)
            d = r.data
            if isinstance(d, dict) and isinstance(d.get("choices"), list) and d["choices"]:
                data = d
                break
            err = r.error if r.error is not None else {"kind": "bad", "body": str(d)[:200]}
            st = (r.error or {}).get("status")
            if st in (429, 500, 502, 503, 504) and attempt < 2 and self.ep.time_left() >= MARGIN + 10:
                time.sleep(2 if attempt == 0 else 4)
                continue
            break
        text = ""
        if data is not None:
            usage = data.get("usage")
            cost = usage.get("cost") if isinstance(usage, dict) else None
            if isinstance(cost, (int, float)) and not isinstance(cost, bool):
                with self.lock:
                    self.spend += cost
            ch = data["choices"][0] if isinstance(data["choices"][0], dict) else {}
            msg = ch.get("message") if isinstance(ch.get("message"), dict) else {}
            c = msg.get("content")
            if isinstance(c, list):
                text = "".join(p.get("text") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
            elif isinstance(c, str):
                text = c
            if not text:
                err = {"kind": "empty"}
        if not text:
            with self.lock:
                self.fails += 1
            self.ep.note("model_fail", model=model, err=str(err)[:200])
            return None
        return text

    def json_call(self, models, system, user, cap):
        for m in models:
            t = self.chat(m, system, user, cap)
            j = extract_json(t or "")
            if j is not None:
                return j
        return None


# ---------------------------------------------------------------- A7 listing
class Listing:
    def __init__(self, lid):
        self.id = lid
        self.title = ""
        self.desc = ""
        self.price = None
        self.stock = None
        self.category = ""
        self.brand = ""
        self.seller = ""
        self.props = {}
        self.notes = ""
        self.opened = False
        self.page_read = False
        self.raw = None
        self.mark = "maybe"
        self.screened = False


def summary(L, limit=SUMMARY_CAP):
    d = {"id": L.id, "title": (L.title or "")[:160]}
    if L.category:
        d["category"] = L.category
    if L.brand:
        d["brand"] = L.brand
    if L.seller:
        d["seller"] = L.seller
    if L.price is not None:
        d["price"] = L.price
    if L.stock is not None:
        d["can_buy"] = L.stock
    if L.props:
        d["properties"] = dict(L.props)
    if L.notes:
        d["seller_notes"] = L.notes[:600]
    if L.desc:
        d["description"] = L.desc[:400] if L.props else L.desc[:500]
    s = cj(d)
    if len(s) > limit and L.props:
        n = max(6, int(len(L.props) * limit / float(len(s))))
        d["properties"] = dict(list(L.props.items())[:n])
        s = cj(d)
    if len(s) > limit and "description" in d:
        d["description"] = d["description"][:200]
    return d


def mark_rank(m):
    if m == "yes":
        return 0
    if m == "no":
        return 2
    return 1


def price_key(L):
    return (0, L.price) if L.price is not None else (1, 0)


# ---------------------------------------------------------------- A8 requirement
class Req:
    def __init__(self, prop, op="is", value="", ref=None, origin="request"):
        self.prop = txt(prop) if prop else ""
        o = txt(op).lower() if op else ""
        self.op = o or "is"
        self.value = str(value).strip() if value else ""
        self.ref = ref if isinstance(ref, dict) else None
        self.origin = origin

    def unresolved(self):
        return self.op in REL_TABLE and self.ref is not None

    def text(self):
        if self.unresolved():
            return "%s: %s (value still to be read from the referenced listing)" % (self.prop, self.op)
        return "%s: %s %s" % (self.prop, self.op, self.value)

    def key(self):
        return norm(self.prop)


OFFER_RX = re.compile(r"\b(?:forward|propose|suggest|offer|name|closest|nearest|fewest|short|next|instead)\b", re.I)
FORBID_RX = re.compile(r"\bno\s+(?:substitut\w*|alternativ\w*|replacement\w*|stand.?ins?)\b|\b(?:do not|don't|never)\s+(?:\w+\s+){0,2}(?:offer|suggest|name|propose|recommend)\b|\bwithout\s+(?:naming|offering|suggesting)\b|\bnothing\s+else\b", re.I)
COUNTBS_RX = re.compile(r"\b(?:count\w*|including)\b[^.;]{0,40}\b(?:stock|budget)\b", re.I)
PCT_RX = re.compile(r"(\d+(?:\.\d+)?)\s*%")


_J_NUM = re.compile(r'\d+(?:\.\d+)?')
_J_MOST = re.compile(r'\b(?:at (?:the )?(?:very )?most|no more than|maximum|max|up to|or less|or fewer)\b', re.I)
_J_LEAST = re.compile(r'\b(?:at (?:the )?(?:very )?least|no less than|minimum|min|or more)\b', re.I)
_J_UNDER = re.compile(r'\b(?:under|below|less than|fewer than)\b', re.I)
_J_OVER = re.compile(r'\b(?:over|above|more than|greater than)\b', re.I)
_J_RANGE = re.compile(r'\d[\d.,]*\s*[a-z]*\s*(?:~|--|-|to|and|through)\s*[a-z]*\s*\d')
_J_NOT = re.compile(r'\b(?:not|never|without|except|other than|excluding)\b', re.I)


def _j_norm(t):
    return re.sub(r'[^a-z0-9]+', ' ', str(t or '').lower()).strip()


def _j_nums(t):
    return [float(x) for x in _J_NUM.findall(str(t or ''))]


_J_FLOZ = re.compile(r'fl\.?\s*oz\.?', re.I)
_J_NUMU = re.compile(r'(\d+(?:\.\d+)?)\s*([a-z"\']+)?', re.I)
_J_PAREN = re.compile(r'\(([^()]*)\)')
_J_SPLIT = re.compile(r'\s*,\s*|\s+or\s+|\s*/\s*|\s*;\s*|\s*\|\s*', re.I)
_J_HASOR = re.compile(r'\bor\b|/', re.I)
_J_ONLY = re.compile(r'\b(?:nowt|nothing|none)\s+but\b|\bonly\b', re.I)
_J_ANYBUT = re.compile(r'\b(?:any|aught|anything|owt)\s+but\b', re.I)



def _j_has(hay, needle):
    if not needle:
        return False
    return re.search(r'(?<![a-z0-9])%s(?![a-z0-9])' % re.escape(needle), hay) is not None


def _j_units(t):
    s = _J_FLOZ.sub('floz', str(t or ''))
    out = []
    for m in _J_NUMU.finditer(s):
        u = (m.group(2) or '').lower().strip('."\'')
        if u and not u.isdigit():
            out.append(u)
    return out


def _j_unit_clash(have, val):
    hu, vu = set(_j_units(have)), set(_j_units(val))
    if not hu or not vu or (hu & vu):
        return False
    for a in hu:
        for b in vu:
            if a.startswith(b) or b.startswith(a):
                return False
    return True


def _j_alts(t, force=False):
    s = str(t or '')
    out = [s]
    if force or _J_HASOR.search(s):
        for x in _J_SPLIT.split(s):
            x = re.sub(r'^\s*(?:else|either)\s+', '', str(x or ''), flags=re.I).strip()
            if x:
                out.append(x)
    return out


def _j_fit_keys(val):
    out = []
    for m in _J_PAREN.finditer(str(val or '')):
        for part in m.group(1).split(';'):
            part = part.strip()
            if part and re.search(r'\d', part) and not part.lower().startswith('p-'):
                out.append(part)
    return out


def _j_id_key(rec):
    best = None
    for a, b in rec.items():
        if isinstance(b, bool) or not isinstance(b, (str, int)):
            continue
        na = _j_norm(a).replace(' ', '')
        if na == 'id':
            return a
        if best is None and na.endswith('id'):
            best = a
    if best is not None:
        return best
    for a, b in rec.items():
        if not isinstance(b, str) or isinstance(b, bool):
            continue
        t = b.strip()
        if 3 <= len(t) <= 64 and ' ' not in t and re.search(r'[-_0-9]', t):
            return a
    return None


def _j_field(facts, name):
    want = _j_norm(name)
    if not want:
        return None
    spare = None
    for k, v in (facts or {}).items():
        nk = _j_norm(k)
        if nk == want:
            return v
        if spare is None and (want in nk or nk in want):
            spare = v
    return spare


def _j_scalar(have, val):
    hs, vs = _j_norm(have), _j_norm(val)
    hn, vn = _j_nums(have), _j_nums(val)
    if vn and hn and _j_unit_clash(have, val):
        return None
    if vn and hn:
        if len(vn) >= 2 and _J_RANGE.search(str(val)):
            lo, hi = min(vn), max(vn)
            return any(lo - 1e-09 <= x <= hi + 1e-09 for x in hn)
        if _J_UNDER.search(str(val)) or _J_MOST.search(str(val)):
            return min(hn) <= max(vn) + 1e-09
        if _J_OVER.search(str(val)) or _J_LEAST.search(str(val)):
            return max(hn) >= min(vn) - 1e-09
        return any(abs(x - y) < 1e-09 for x in hn for y in vn) or _j_has(hs, vs)
    if not vs:
        return None
    return _j_has(hs, vs) or _j_has(vs, hs)


def _j_match(have, val):
    raw = str(val or '')
    neg = (_J_ANYBUT.search(raw) or _J_NOT.search(raw)) and not _J_ONLY.search(raw) \
        and not (_J_MOST.search(raw) or _J_LEAST.search(raw))
    if neg:
        bare = _J_ANYBUT.sub(' ', raw)
        bare = _J_NOT.sub(' ', bare)
        hs = _j_norm(have)
        forbidden = [_j_norm(x) for x in _j_alts(bare) if _j_norm(x)]
        if not forbidden:
            return None
        return not any(_j_has(hs, f) for f in forbidden)
    if _J_ONLY.search(raw):
        raw = _J_ONLY.sub(' ', raw)
    if _J_MOST.search(raw) or _J_LEAST.search(raw) or _J_UNDER.search(raw) \
            or _J_OVER.search(raw) or _J_RANGE.search(raw):
        return _j_scalar(have, raw)
    seen = False
    for w in _j_alts(raw):
        for h in _j_alts(have, True):
            r = _j_scalar(h, w)
            if r:
                return True
            if r is not None:
                seen = True
    return False if seen else None


def plain_holds(prop, val, facts, blob):
    have = _j_field(facts, prop)
    if have is None:
        if not _j_norm(val):
            return None
        return True if _j_has(_j_norm(blob), _j_norm(val)) else None
    return _j_match(have, val)


def read_fallback(text):
    t = str(text or "")
    if not t.strip():
        return None
    if FORBID_RX.search(t):
        return {"buy_near_miss": False, "tolerance": None, "count_stock_budget": False}
    if not OFFER_RX.search(t):
        return None
    m = PCT_RX.search(t)
    return {"buy_near_miss": True, "tolerance": float(m.group(1)) if m else None,
            "count_stock_budget": bool(COUNTBS_RX.search(t))}


def lax_fallback(fb):
    fb = fb or {}
    return fb.get("buy_near_miss") is not False and bool(fb.get("count_stock_budget"))


def empty_verdict():
    return {"failed": set(), "unsure": set(), "weak": set(), "loose": set()}


# ---------------------------------------------------------------- A9 product
class Product:
    def __init__(self, idx, entry):
        uc = num(entry.get("unit_cap")) if isinstance(entry, dict) else None
        self.unit_cap = uc if (uc is not None and uc > 0) else None
        self.idx = idx
        self.scratch = idx == 99
        self.name = (txt(entry.get("name")) or txt(entry.get("category")) or "the product").strip()
        self.category = txt(entry.get("category")) or self.name
        q = entry.get("quantity")
        self.qty = int(q) if (isinstance(q, (int, float)) and not isinstance(q, bool) and q >= 1) else 1
        self.reqs = []
        rl = entry.get("requirements")
        for r in rl if isinstance(rl, list) else []:
            if isinstance(r, dict) and txt(r.get("property")):
                self.reqs.append(Req(r.get("property"), r.get("operator"), r.get("value"), r.get("reference"), "request"))
        hs = entry.get("hints")
        self.hints = [str(h) for h in hs if h] if isinstance(hs, list) else []
        ts = entry.get("search_terms")
        self.terms = [str(t) for t in ts if t] if isinstance(ts, list) else []
        if not self.terms:
            self.terms = [self.category]
        self.noun = self.terms[-1]
        self.pool = {}
        self.verdicts = {}
        self.judged_for = {}
        self.asked = []
        self.dontcare = []
        self.pages_fetched = 0
        self.exhausted = False
        self.chosen = None
        self.labels = None
        self.cursors = {}
        self.capped = False
        self.swept = None
        self.narrowed = False
        self.need_done = False
        self.suggested = False
        self.series = []

    def active(self):
        return [r for r in self.reqs if not r.unresolved()]

    def texts(self):
        return [r.text() for r in self.active()]

    def fit_req(self):
        for r in self.reqs:
            if r.origin == "fit":
                return r
        return None

    def exactness(self, L):
        if not L.props:
            return 0
        elems = []
        for v in L.props.values():
            for piece in SEP_RE.split(str(v)):
                t = toks(piece)
                if t:
                    elems.append(t)
        n = 0
        for r in self.active():
            if r.op in PLAIN_VALUE_OPS and r.value:
                for alt in r.value.split("|"):
                    ta = toks(alt)
                    if ta and any(ta == e for e in elems):
                        n += 1
                        break
        return n

    def fits(self, include_unsure=False, cheapest=False):
        texts = set(self.texts())
        sure = []
        uns = []
        for i, L in enumerate(list(self.pool.values())):
            if not L.opened:
                continue
            v = self.verdicts.get(L.id)
            if v is None or v["failed"]:
                continue
            if not texts <= self.judged_for.get(L.id, set()):
                continue
            if v["unsure"]:
                uns.append((i, L, v))
            else:
                sure.append((i, L, v))
        if cheapest:
            sure.sort(key=lambda t: (-self.exactness(t[1]), price_key(t[1]), t[0]))
        else:
            sure.sort(key=lambda t: (-self.exactness(t[1]), len(t[2]["weak"]), price_key(t[1]), t[0]))
        uns.sort(key=lambda t: (len(t[2]["unsure"]), price_key(t[1]), t[0]))
        if sure or not include_unsure:
            return [t[1] for t in sure]
        return [t[1] for t in uns]


# ---------------------------------------------------------------- E1 episode
# ---- one product's scheduled slot, as the planner designed it and the controller carries it out
class Slot:

    def __init__(self, P, spec=None):
        spec = spec if isinstance(spec, dict) else {}
        self.P = P
        self.asks = spec.get("asks")
        self.steps_share = spec.get("steps_share")
        self.spend_share = spec.get("spend_share")
        self.needs_from = spec.get("needs_from")
        self.take = spec.get("take")
        self.as_prop = spec.get("as")
        self.state = "open"
        self.bound = False
        self.pick = None
        u = spec.get("use")
        self.use = [str(x).strip().lower() for x in u if str(x).strip()] if isinstance(u, list) else []
        self.asked = 0
        self.steps = 0
        self.ceiling = None

    def name(self):
        return self.P.name


class Episode:
    def __init__(self, problem_data):
        self.t0 = time.monotonic()
        env = (problem_data or {}).get("environment") if isinstance(problem_data, dict) else None
        env = env if isinstance(env, dict) else {}
        self.binding = dict(env.get("binding") or {})
        pv = env.get("policy_view")
        self.pv = pv if isinstance(pv, dict) else {}
        q = self.pv.get("query")
        self.request = str(q) if q is not None else ""
        self.opening = self.pv.get("opening")
        self.tools = []
        for t in self.pv.get("tools") or []:
            if isinstance(t, dict):
                f = t.get("function") if isinstance(t.get("function"), dict) else t
                self.tools.append(f)
        self.schemas = {}
        for f in self.tools:
            n = f.get("name")
            if n:
                p = f.get("parameters")
                self.schemas[n] = p if isinstance(p, dict) else {"type": "object", "properties": {}}
        try:
            ms = int(self.pv.get("max_steps") or 0)
        except Exception:
            ms = 0
        self.budget = ms or 30
        self.used = 0
        self.finished = bool(self.pv.get("finished"))
        self.dead = False
        self.world_key = "|".join(sorted(self.schemas))
        with _WORLD_LOCK:
            mem = WORLD.get(self.world_key) or {}
        self.keymaps = dict(mem.get("keymaps") or {})
        self.compare_limit = mem.get("limit") or 4
        self.thin = bool(mem.get("thin"))
        self.trace = []
        self.llm = LLM(self)
        self.roles = {}
        self.products = []
        self.cap = None
        self.cheapest = False
        self.best = None
        self.confirm_first = False
        self.procedure = []
        self.history = []
        self.gate_counts = {}
        self.confirmed = None
        self.allowance = None
        self.sent = 0
        self.in_stock = True
        self.topics = []
        self.fit_rules = []
        self.conditions = []
        self.fallback = None
        self.pending_lines = []
        self.handled = set()
        self.cart = {}
        self.carted = {}
        self.page_size = 0
        self.opened_pages = set()
        self.decline_reason = ""
        self.compare_off = False
        self.compare_checked = False
        self.choice_cache = {}
        self.enum_pick = {}
        self.enum_bad = set()
        self.standins = 0
        self.standin_cache = {}
        self.arg_patch = {}
        self.helpers = {}
        self.helper_calls = 0
        self.helper_cache = {}
        self.helper_said = {}
        self.helper_picks = {}
        self.reading = None
        self.decline_gated = False
        self.decline_retried = False
        self.standin_tried = False
        self.last_chance_done = False
        self.audit_done = False
        self.reduced = False
        self.ref_floor = 0
        self.rank_cache = {}
        self.planner_list = None
        self.order_rounds = 0
        self.slots = []
        self.slot_now = None
        self.slot_ceiling = None
        self.replan_at = ""
        self.replan_fired = set()
        self.replans = 0
        self.change_mark = None
        self.vcache = {}
        self.ccache = {}

    # ---- C2 clocks, steps, spend, notes, memory
    def time_left(self):
        return TASK_CLOCK - (time.monotonic() - self.t0)

    def steps_left(self):
        return self.budget - self.used

    def spent(self):
        return self.llm.spend >= SPEND_CAP

    def note(self, kind, **fields):
        e = {"kind": kind, "step": self.used, "t": round(time.monotonic() - self.t0, 1)}
        e.update(fields)
        self.trace.append(e)

    def batch_reading(self):
        return "compare" in self.roles and not self.compare_off and not self.thin

    def save_world(self):
        with _WORLD_LOCK:
            WORLD[self.world_key] = {"keymaps": self.keymaps, "limit": self.compare_limit, "thin": self.thin}

    def role_tool(self, role):
        r = self.roles.get(role)
        return r.get("tool") if r else None

    # ---- C3 environment call
    def read_helpers(self, ans):
        out = {}
        raw = ans.get("helpers") if isinstance(ans, dict) else None
        for e in raw if isinstance(raw, list) else []:
            if not isinstance(e, dict):
                continue
            nm = txt(e.get("name")).strip().lower()
            pur = txt(e.get("purpose")).strip()
            kind = txt(e.get("returns")).strip().lower()
            if not nm or not pur or kind not in HELPER_KINDS or nm in out:
                continue
            needs = [txt(x).strip().lower() for x in (e.get("needs") or []) if txt(x)]
            needs = [x for x in needs if x in HELPER_NEEDS]
            out[nm] = {"name": nm, "purpose": pur[:300], "needs": needs or ["request"],
                       "returns": kind}
            if len(out) >= 2:
                break
        return out

    def helper_card(self, L):
        if L is None:
            return {}
        return {"id": L.id, "title": (L.title or "")[:90], "price": L.price,
                "properties": dict(list((L.props or {}).items())[:14])}

    def helper_facts(self, names, P, candidate=None):
        f = {}
        for n in names:
            if n == "request":
                f["request"] = (self.request or "")[:900]
            elif n == "requirements":
                f["requirements"] = P.texts() if P is not None else []
            elif n == "budget":
                f["budget"] = self.cap
            elif n == "quantity":
                f["quantity"] = P.qty if P is not None else 1
            elif n == "candidate" and candidate is not None:
                f["candidate"] = self.helper_card(candidate)
            elif n == "shortlist" and P is not None:
                try:
                    f["shortlist"] = [self.helper_card(L) for L in self.rank(P, self.buyable(P))[:4]]
                except Exception:
                    f["shortlist"] = []
            elif n == "picked":
                got = []
                for Q in self.products:
                    L = self.helper_picks.get(Q.name) or Q.chosen
                    if L is not None and L is not candidate:
                        got.append(dict({"product": Q.name}, **self.helper_card(L)))
                f["picked"] = got
            elif n == "cart":
                f["cart"] = [{"id": k, "units": v} for k, v in self.cart.items()]
        return f

    def helper_run(self, name, P, candidate=None):
        h = self.helpers.get(name)
        if not h:
            return None
        facts = self.helper_facts(h["needs"], P, candidate)
        key = (name, json.dumps(facts, sort_keys=True, default=str)[:600])
        if key in self.helper_cache:
            return self.helper_cache[key]
        if self.helper_calls >= HELPER_CAP or self.spent() or self.time_left() < MARGIN:
            return None
        self.helper_calls += 1
        user = "Purpose: %s\nKind of answer wanted: %s\nFacts:\n%s" % (
            h["purpose"], h["returns"], json.dumps(facts, ensure_ascii=False, default=str)[:2000])
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_HELPER, user, 300)
        v = ans.get("value") if isinstance(ans, dict) else None
        out = None
        if h["returns"] == "number":
            out = num(v)
        elif h["returns"] == "flag":
            if isinstance(v, bool):
                out = v
            else:
                w = txt(v).strip().lower()
                out = True if w in ("true", "yes") else (False if w in ("false", "no") else None)
        elif h["returns"] == "text":
            out = txt(v)[:80] or None
        elif h["returns"] == "requirement":
            t = txt(v).strip()
            out = t[:80] if (":" in t and len(t) > 2) else None
        self.helper_cache[key] = out
        self.note("helper", name=name, returns=h["returns"], value=cut(out, 90),
                  calls=self.helper_calls, product=(P.name if P is not None else None))
        return out

    def helper_gate(self, P, L):
        for nm in self.helper_used(P):
            if (self.helpers.get(nm) or {}).get("returns") != "flag":
                continue
            v = self.helper_run(nm, P, candidate=L)
            if v is False:
                self.note("helper_rejected", name=nm, product=P.name, id=L.id)
                return False
        return True

    def helper_used(self, P):
        for sl in self.slots:
            if sl.P is P:
                return list(sl.use or [])
        return list(self.helpers) if len(self.products) < 2 else []

    def helper_prepare(self, slot):
        for nm in list(slot.use or []):
            h = self.helpers.get(nm)
            if not h or h["returns"] == "flag":
                continue
            v = self.helper_run(nm, slot.P)
            if v is None:
                continue
            if h["returns"] == "requirement":
                prop, _sep, val = str(v).partition(":")
                prop, val = prop.strip().lower(), val.strip()
                if prop and val and not any(near(r.prop, prop) for r in slot.P.reqs):
                    slot.P.reqs.append(Req(prop, "is", val, None, "helper"))
                    self.note("helper_requirement", name=nm, product=slot.P.name, req="%s: %s" % (prop, val))
            else:
                self.helper_said[(slot.P.name, nm)] = v

    def tool_props(self, tool):
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        return props, [k for k in req if isinstance(k, str)]

    def schema_gap(self, tool, args):
        props, req = self.tool_props(tool)
        return [k for k in req if k not in args and k in props]

    def fits_schema(self, spec, v):
        if not isinstance(spec, dict):
            return None
        en = spec.get("enum")
        if isinstance(en, list) and en:
            return v if v in en else None
        t = spec.get("type")
        if t == "string":
            if not isinstance(v, str):
                return None
            mx = spec.get("maxLength")
            return v[:int(mx)] if is_int(mx) else v
        if t in ("integer", "number"):
            n = num(v)
            if n is None:
                return None
            if t == "integer":
                n = int(n)
            mn, mx = num(spec.get("minimum")), num(spec.get("maximum"))
            if mn is not None and n < mn:
                n = mn
            if mx is not None and n > mx:
                n = mx
            return n
        if t == "boolean":
            return bool(v) if isinstance(v, bool) else None
        if t == "array":
            return v if isinstance(v, list) else None
        if t == "object":
            return v if isinstance(v, dict) else None
        return v

    def standin_goal(self, tool):
        said = {"keyword_search": "search the store for the wanted product by words",
                "field_search": "browse or narrow the store by category",
                "open": "read the full page of one listing",
                "facts": "read the property table of one listing",
                "compare": "look at several listings side by side",
                "cart_add": "put the chosen listing in the basket",
                "cart_set": "set the basket to exactly the chosen lines",
                "cart_view": "see what the basket holds",
                "checkout": "place the order for what the basket holds",
                "decline": "finish the task without buying anything",
                "message": "send the shopper one short question",
                "status": "read where the task stands"}
        for role, text in said.items():
            if self.role_tool(role) == tool:
                return text
        return "carry out the next step of the shopping task"

    def standin_args(self, tool, args, goal, err=None):
        gap = self.schema_gap(tool, args)
        if not gap and not err:
            return {}
        key = (tool, tuple(sorted(gap)), str(err or "")[:60])
        if key in self.standin_cache:
            return dict(self.standin_cache[key])
        if self.standins >= STANDIN_CAP or self.spent() or self.time_left() < MARGIN:
            return {}
        props, req = self.tool_props(tool)
        self.standins += 1
        user = "Tool: %s\nDescription: %s\nArgument schema: %s\nRequired: %s\nAlready prepared: %s\nPurpose: %s" % (
            tool, self.tool_desc(tool), json.dumps(props, ensure_ascii=False, default=str)[:1800],
            json.dumps(req), json.dumps(args, ensure_ascii=False, default=str)[:400], goal)
        if err:
            user += "\nThe store refused the last attempt with: %s" % str(err)[:200]
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_STANDIN, user, 400)
        out = {}
        if isinstance(ans, dict):
            for k, v in ans.items():
                if k not in props or v is None:
                    continue
                cv = self.fits_schema(props[k], v)
                if cv is not None:
                    out[k] = cv
        if not err:
            out = {k: v for k, v in out.items() if k not in args or k in gap}
        else:
            out = {k: v for k, v in out.items() if k not in args or args.get(k) != v}
        self.standin_cache[key] = dict(out)
        self.note("standin", tool=tool, gap=gap, gave=cut(out, 160), used=self.standins, after_error=bool(err))
        return dict(out)

    def call(self, tool, args):
        if self.finished or self.dead or tool not in self.schemas:
            return None, [], "unavailable", False
        fin = tool in (self.role_tool("checkout"), self.role_tool("decline"))
        left = self.steps_left()
        if left <= 0 or (left == 1 and not fin):
            return None, [], "no steps", False
        patch = self.arg_patch.get(tool)
        if patch:
            merged = dict(args)
            merged.update(patch)
            args = merged
        try:
            if self.schema_gap(tool, args):
                got = self.standin_args(tool, args, self.standin_goal(tool))
                if got:
                    merged = dict(args)
                    merged.update(got)
                    args = merged
        except Exception as exc:
            self.note("standin_error", tool=tool, error=str(exc)[:160])
        payload = dict(self.binding)
        payload["step"] = self.used + 1
        payload["action"] = {"name": tool, "args": args}
        for _attempt in range(3):
            tl = self.time_left()
            if tl < 3:
                return None, [], "no time", False
            _ENV.timeout = int(max(4, min(90, int(tl) - 1)))
            r = _ENV.post_verbose("/environment/call", payload)
            d = r.data
            if isinstance(d, dict) and ("observation" in d or "finished" in d):
                st = d.get("step")
                self.used = st if is_int(st) else self.used + 1
                if d.get("finished"):
                    self.finished = True
                obs, lines = split_lines(d.get("observation"))
                for ln in lines:
                    if ln not in self.handled:
                        self.pending_lines.append(ln)
                err = None
                ok = True
                if isinstance(obs, dict) and isinstance(obs.get("error"), str) and len(obs) <= 2:
                    err = obs["error"]
                    ok = False
                if not ok and err:
                    try:
                        fix = self.standin_args(tool, args, self.standin_goal(tool), err=err)
                        if fix:
                            cur = dict(self.arg_patch.get(tool) or {})
                            cur.update(fix)
                            self.arg_patch[tool] = cur
                            self.note("arg_patch", tool=tool, patch=cut(cur, 160))
                    except Exception as exc:
                        self.note("standin_error", tool=tool, error=str(exc)[:160])
                self.note("call", tool=tool, args=cut(args, 160), result=cut(obs, 200), lines=lines[:3])
                is_msg = tool == self.role_tool("message")
                h = {"step": self.used, "tool": tool, "args": cut(args, 300 if is_msg else 120), "result": cut(obs, 160)}
                if lines:
                    h["shopper_lines"] = [str(x)[:200] for x in lines[:3]]
                if is_msg:
                    h["basket"] = [list(x) for x in self.sig()]
                self.history.append(h)
                return obs, lines, err, ok
            e = r.error or {}
            status = e.get("status")
            body = str(e.get("body") or "")[:300]
            self.note("env_fail", tool=tool, status=status, body=body)
            if status == 402:
                self.dead = True
                return None, [], "payment", False
            if status == 409:
                if "finished" in body:
                    self.finished = True
                    return None, [], "finished", False
                m = re.search(r"expected (\d+)", body)
                if m:
                    n = int(m.group(1))
                    self.used = n - 1
                    payload["step"] = n
                    continue
                return None, [], "conflict", False
            if status in (403, 422):
                return None, [], "refused %s" % status, False
        return None, [], "environment failure", False

    # ---- C5 key map
    def keymap(self, kind, rec):
        sig = kind + "|" + "|".join(sorted(str(k) for k in rec.keys()))
        if sig in self.keymaps:
            return self.keymaps[sig]
        for s, m in list(self.keymaps.items()):
            if not s.startswith(kind + "|") or not m:
                continue
            present = sum(1 for k in m.values() if k in rec)
            if present >= max(2, int(0.6 * len(m))):
                self.keymaps[sig] = m
                return m
        shown = {}
        for k, v in rec.items():
            if isinstance(v, (dict, list)):
                shown[k] = cut(v, 160)
            elif isinstance(v, str) and len(v) > 160:
                shown[k] = v[:160] + "…"
            else:
                shown[k] = v
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_KEYMAP, json.dumps(shown, ensure_ascii=False, default=str), 700)
        m = {}
        if isinstance(ans, dict):
            for role, key in ans.items():
                if isinstance(key, str) and key in rec:
                    m[role] = key
        if "facts" not in m:
            for k, v in rec.items():
                if isinstance(v, dict) and v and all(isinstance(x, (str, int, float, bool)) or x is None for x in v.values()):
                    m["facts"] = k
                    break
        if "id" not in m:
            for k, v in rec.items():
                if isinstance(v, str) and re.fullmatch(r"[A-Za-z]{0,3}-?[0-9a-f]{8,}", v):
                    m["id"] = k
                    break
        if "price" not in m:
            for k, v in rec.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and "price" in str(k).lower():
                    m["price"] = k
                    break
        self.keymaps[sig] = m
        self.save_world()
        self.note("keymap", record=kind, map=m)
        return m

    # ---- C6 listing from record
    def build(self, kind, rec):
        if not isinstance(rec, dict):
            return None
        m = self.keymap(kind, rec)
        ik = m.get("id")
        v = rec.get(ik) if ik else None
        if isinstance(v, bool) or not isinstance(v, (str, int)):
            ik = _j_id_key(rec)
            if not ik:
                return None
            v = rec.get(ik)
            if isinstance(v, bool) or not isinstance(v, (str, int)):
                return None
            self.note("id_recovered", key=ik, value=str(v)[:40])
        L = Listing(str(v))
        L.raw = rec
        for role, attr in (("title", "title"), ("text", "desc"), ("category", "category"),
                           ("brand", "brand"), ("seller", "seller"), ("notes", "notes")):
            k = m.get(role)
            x = rec.get(k) if k else None
            if isinstance(x, str):
                setattr(L, attr, x)
        k = m.get("price")
        if k:
            L.price = num(rec.get(k))
        k = m.get("stock")
        if k and isinstance(rec.get(k), bool):
            L.stock = rec[k]
        k = m.get("facts")
        if k and isinstance(rec.get(k), dict):
            L.props = {str(a): b for a, b in rec[k].items() if b is not None and b != ""}
        if not L.props:
            cand = []
            for a, b in rec.items():
                if not isinstance(b, dict) or not b:
                    continue
                if all(not isinstance(x, (dict, list)) for x in b.values()):
                    cand.append((a, b))
            if len(cand) == 1:
                L.props = {str(a): b for a, b in cand[0][1].items() if b is not None and b != ""}
                self.note("props_recovered", id=L.id, key=cand[0][0], n=len(L.props))
        if L.price is None:
            for a, b in rec.items():
                al = str(a).lower()
                if num(b) is not None and ("price" in al or "cost" in al or "amount" in al):
                    L.price = num(b)
                    break
        if L.stock is None:
            bools = [b for b in rec.values() if isinstance(b, bool)]
            if len(bools) == 1:
                L.stock = bools[0]
        if not L.title:
            best = ""
            for a, b in rec.items():
                if not isinstance(b, str) or a == ik or len(b) <= len(best):
                    continue
                if b.strip() in (L.category or "", L.seller or "", L.brand or ""):
                    continue
                if " " in b.strip() and len(b) <= 160:
                    best = b
            if best:
                L.title = best
        return L

    # ---- C7 merge
    def merge(self, P, L, opened):
        old = P.pool.get(L.id)
        if old is None:
            P.pool[L.id] = L
            cur = L
        else:
            for a in ("title", "desc", "category", "brand", "seller", "notes"):
                nv = getattr(L, a)
                if nv and (opened or not getattr(old, a)):
                    setattr(old, a, nv)
            if L.price is not None:
                old.price = L.price
            if L.stock is not None:
                old.stock = L.stock
            if L.props:
                if L.props != old.props:
                    P.judged_for.pop(L.id, None)
                old.props = dict(L.props)
            old.page_read = old.page_read or L.page_read
            if L.raw:
                old.raw = L.raw
            cur = old
        if opened:
            cur.opened = True
            self.opened_pages.add(cur.id)
        return cur

    # ---- C8 setup
    def setup(self):
        tl = [{"name": f.get("name"), "description": f.get("description") or "", "parameters": f.get("parameters")}
              for f in self.tools]
        user = "SHOPPER REQUEST:\n" + self.request
        if self.opening is not None:
            user += "\n\nOPENING OBSERVATION:\n" + cut(self.opening, 1500)
        user += "\n\nSTORE TOOLS:\n" + json.dumps(tl, ensure_ascii=False, default=str)
        a = self.llm.json_call([M_READ, M_BACKUP], P_READER, user, 2500)
        if not (isinstance(a, dict) and isinstance(a.get("tools"), dict)):
            a = self.llm.json_call([M_READ, M_FAST], P_READER, user, 2500)
            if not isinstance(a, dict):
                self.note("setup_failed")
                return False
        self.reading = a
        self.roles = {}
        tm = a.get("tools")
        if isinstance(tm, dict):
            for role, d in tm.items():
                if isinstance(d, dict) and isinstance(d.get("tool"), str) and d["tool"] in self.schemas:
                    self.roles[role] = dict(d)
        self.role_check()
        for rn in ("keyword_search", "open", "checkout", "decline", "message"):
            if rn not in self.roles:
                self.note("role_missing", role=rn)
        self.products = []
        prods = a.get("products")
        if isinstance(prods, list):
            for i, e in enumerate(prods):
                if isinstance(e, dict):
                    self.products.append(Product(i, e))
        if not self.products:
            self.products = [Product(0, {"name": "the product", "category": "the product",
                                         "search_terms": [self.request[:60]]})]
        for P in self.products:
            self.clean_product(P)
        self.cap = num(a.get("price_cap"))
        self.cheapest = bool(a.get("cheapest"))
        bm = a.get("best_measure")
        self.best = bm if isinstance(bm, str) and bm.strip() else None
        ma = a.get("message_allowance")
        if isinstance(ma, (int, float)) and not isinstance(ma, bool) and ma >= 0:
            self.allowance = int(ma)
        ml = a.get("move_limit")
        if isinstance(ml, (int, float)) and not isinstance(ml, bool) and 0 < ml < self.budget:
            self.budget = int(ml)
        self.in_stock = bool(a.get("in_stock_only")) if "in_stock_only" in a else True
        tp = a.get("topics_to_raise")
        self.topics = [str(t) for t in tp if t] if isinstance(tp, list) else []
        fr = a.get("fit_rules")
        self.fit_rules = [e for e in fr if isinstance(e, dict)] if isinstance(fr, list) else []
        self.conditions = self.read_conditions(a.get("conditions"))
        self.fallback = read_fallback(a.get("no_match_instruction"))
        src = "reading"
        if self.fallback is None and FORBID_RX.search(self.request or ""):
            self.fallback, src = {"buy_near_miss": False, "tolerance": None,
                                  "count_stock_budget": False}, "request"
        from_req = None
        for sent in re.split(r"(?<=[.!?])\s+", self.request or ""):
            if re.search(r"\b(?:if|should|where|when)\b", sent, re.I) and \
                    re.search(r"\b(?:nowt|nothing|none|no listing|no item)\b", sent, re.I):
                got = read_fallback(sent)
                if got:
                    from_req = got
                    break
        if from_req is None and self.request:
            got = read_fallback(self.request)
            if got and got.get("buy_near_miss"):
                from_req = got
        if self.fallback is None:
            if from_req is not None:
                self.fallback, src = from_req, "request"
        elif from_req is not None and self.fallback.get("buy_near_miss") is not False:
            if not self.fallback.get("count_stock_budget") and from_req.get("count_stock_budget"):
                self.fallback["count_stock_budget"] = True
                src = src + "+request"
            if self.fallback.get("tolerance") is None and from_req.get("tolerance") is not None:
                self.fallback["tolerance"] = from_req["tolerance"]
                src = src + "+request" if "+request" not in src else src
        if self.fallback:
            self.note("fallback_rule", source=src, **self.fallback)
        self.procedure = []
        pr = a.get("procedure")
        if isinstance(pr, list):
            for e in [x for x in pr if isinstance(x, dict) and x.get("goal")][:6]:
                st = dict(e)
                st["moment"] = str(e.get("moment")).lower() if e.get("moment") else "pre-finish"
                st["kind"] = "message" if str(e.get("kind") or "").strip().lower() == "message" else "tool"
                self.procedure.append(st)
        self.confirm_first = any(s["kind"] == "message" and s["moment"] in ("pre-order", "pre-finish")
                                 for s in self.procedure)
        self.note("plan",
                  products=[{"name": P.name, "qty": P.qty, "reqs": [r.text() for r in P.reqs], "hints": P.hints,
                             "terms": P.terms} for P in self.products],
                  cap=self.cap, cheapest=self.cheapest, allowance=self.allowance, budget=self.budget,
                  procedure=self.procedure, roles={k: v.get("tool") for k, v in self.roles.items()})
        return True

    def clean_product(self, P):
        pairs = []
        for r in P.reqs:
            if r.ref is not None and isinstance(r.ref.get("attributes"), dict):
                for k, v in r.ref["attributes"].items():
                    pairs.append((norm(k), norm(v)))
        if pairs:
            ps = set(pairs)
            P.reqs = [r for r in P.reqs if r.ref is not None or (norm(r.prop), norm(r.value)) not in ps]
            vals = [v for _, v in pairs]
            P.terms = [t for t in P.terms if not any(v in norm(t) for v in vals)]
            if not P.terms:
                P.terms = [P.category]
        ct = toks(P.category)
        keep = []
        for r in P.reqs:
            tv = toks(r.value)
            if r.ref is None and r.op == "is" and tv and tv <= ct:
                continue
            keep.append(r)
        P.reqs = keep
        for r in P.reqs:
            if r.op in ("is", "one of", "contains") and r.value:
                for piece in r.value.split("|"):
                    piece = piece.strip()
                    cands = [piece]
                    if "-" in piece:
                        cands.append(piece.replace("-", " "))
                    for c in cands:
                        if c and len(c) <= 40 and norm(c) not in [norm(t) for t in P.terms]:
                            P.terms.append(c)

    # ---- C9 role check
    def role_check(self):
        for rn in ("cart_add", "cart_set"):
            r = self.roles.get(rn)
            if not r:
                continue
            sch = self.schemas.get(r.get("tool")) or {}
            props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
            ll = None
            for pn, ps in props.items():
                if isinstance(ps, dict) and ps.get("type") == "array" and isinstance(ps.get("items"), dict) \
                        and ps["items"].get("type") == "object":
                    ll = pn
                    break
            if ll:
                lp = props[ll]["items"].get("properties")
                lp = lp if isinstance(lp, dict) else {}
                strs = [k for k, s in lp.items() if isinstance(s, dict) and s.get("type") == "string"]
                ints = [k for k, s in lp.items() if isinstance(s, dict) and s.get("type") == "integer"]
                if rn == "cart_set" and r.get("line_id_key") in lp:
                    idk = r.get("line_id_key")
                else:
                    idk = strs[0] if strs else None
                if rn == "cart_set" and r.get("line_quantity_key") in lp:
                    qk = r.get("line_quantity_key")
                else:
                    qk = ints[0] if ints else None
                if idk:
                    self.roles["cart_set"] = {"tool": r["tool"], "lines_arg": ll, "line_id_key": idk,
                                              "line_quantity_key": qk}
                if rn == "cart_add":
                    self.roles.pop("cart_add", None)
            elif rn == "cart_add":
                strs = [k for k, s in props.items() if isinstance(s, dict) and s.get("type") == "string"]
                ints = [k for k, s in props.items() if isinstance(s, dict) and s.get("type") == "integer"]
                if r.get("id_arg") not in props:
                    r["id_arg"] = strs[0] if strs else None
                if r.get("quantity_arg") and r.get("quantity_arg") not in props:
                    r["quantity_arg"] = ints[0] if ints else None
                if not r.get("id_arg"):
                    self.roles.pop("cart_add", None)
        for have, miss in (("checkout", "decline"), ("decline", "checkout")):
            if have in self.roles and miss not in self.roles:
                sch = self.schemas.get(self.roles[have]["tool"]) or {}
                props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
                req = sch.get("required") if isinstance(sch.get("required"), list) else []
                if any(isinstance(props.get(a), dict) and props[a].get("enum") for a in req):
                    self.roles[miss] = {"tool": self.roles[have]["tool"]}

    # ---- C10-C15 argument helpers
    def tool_desc(self, tool):
        for f in self.tools:
            if f.get("name") == tool:
                return f.get("description") or ""
        return ""

    def role_schema(self, role):
        r = self.roles.get(role)
        if not r:
            return {}
        return self.schemas.get(r.get("tool")) or {}

    def arg_schema(self, role, arg):
        if not arg:
            return {}
        props = self.role_schema(role).get("properties")
        p = props.get(arg) if isinstance(props, dict) else None
        return p if isinstance(p, dict) else {}

    def arg_type(self, role, arg):
        return self.arg_schema(role, arg).get("type")

    def max_len(self, role, arg):
        mx = self.arg_schema(role, arg).get("maxLength")
        return mx if is_int(mx) else None

    def finish_args(self, role):
        r = self.roles.get(role) or {}
        tool = r.get("tool")
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        out = {}
        fx = r.get("fixed_args")
        if isinstance(fx, dict):
            for k, v in fx.items():
                if k not in props or v is None or v == "":
                    continue
                p = props[k] if isinstance(props[k], dict) else {}
                en = p.get("enum") if isinstance(p.get("enum"), list) else None
                if en and v not in en:
                    b = best_by_sim(str(v), en, 0.5)
                    if b is None:
                        continue
                    v = b
                out[k] = v
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        missing = [k for k in req if k not in out and isinstance(props.get(k), dict) and props[k].get("enum")]
        if missing:
            ans = self.choice_cache.get(role)
            if ans is None:
                goal = ("place the order for what the cart holds" if role == "checkout"
                        else "end the episode with nothing bought")
                user = "Tool: %s\nDescription: %s\nMissing arguments: %s\nGoal: %s" % (
                    tool, self.tool_desc(tool), json.dumps({k: props[k] for k in missing}, ensure_ascii=False), goal)
                ans = self.llm.json_call([M_FAST, M_BACKUP], P_CHOOSER, user, 300)
                if isinstance(ans, dict) and ans:
                    self.choice_cache[role] = ans
            if isinstance(ans, dict):
                for k in missing:
                    if k in ans and ans[k] in props[k]["enum"]:
                        out[k] = ans[k]
        return out

    def page_mode(self, role):
        r = self.roles.get(role) or {}
        pa = r.get("page_arg")
        if not pa:
            return None
        t = self.arg_type(role, pa)
        if t == "integer":
            kind = str(r.get("page_kind") or "").lower()
            if kind == "offset":
                return "offset"
            if kind != "number" and any(f in str(pa).lower() for f in ("offset", "start", "skip")):
                return "offset"
            return "index"
        return "token"

    def world_value(self, role, arg, value):
        en = self.arg_schema(role, arg).get("enum")
        if not isinstance(en, list) or not en or not isinstance(value, str):
            return value
        texts = [e for e in en if isinstance(e, str)]
        if not texts:
            return value
        nv = norm(value)
        for e in texts:
            if norm(e) == nv:
                return e
        return best_by_sim(value, texts, 0.5)

    def trim(self, text, mx):
        if not is_int(mx) or len(text) <= mx:
            return text
        c = text[:mx]
        if " " in c:
            c = c[:c.rfind(" ")]
        return c

    def fit_query(self, P, q):
        ws = q.split()
        cw = toks(P.category) | toks(P.name)
        vals = []
        oth = []
        for w in ws:
            tw = toks(w)
            if tw and tw <= cw:
                oth.append(w)
            else:
                vals.append(w)
        rebuilt = " ".join(vals + oth)
        r = self.roles.get("keyword_search") or {}
        mx = self.max_len("keyword_search", r.get("query_arg"))
        if mx is None or len(rebuilt) <= mx:
            return rebuilt
        kept = list(vals)
        for w in oth:
            if len(" ".join(kept + [w])) <= mx:
                kept.append(w)
        res = " ".join(kept) if kept else rebuilt
        return self.trim(res, mx)

    # ---------------------------------------------------------------- E2 search
    def need_enums(self, role, args, words, rival=()):
        sch = self.role_schema(role)
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        out = {}
        for k in req:
            if k in args or not isinstance(props.get(k), dict):
                continue
            en = props[k].get("enum")
            if not isinstance(en, list) or not en:
                continue
            opts = [str(v) for v in en if isinstance(v, (str, int, float))]
            opts = [v for v in opts if (role, k, v) not in self.enum_bad]
            if not opts:
                continue
            key = (role, k, tuple(words))
            pick = self.enum_pick.get(key)
            if pick is None or pick not in opts:
                best, bs = opts[0], -2.0
                for v in opts:
                    mine = max([sim(v, w) for w in words] + [0.0])
                    theirs = max([sim(v, w) for w in rival] + [0.0])
                    sc = mine - theirs
                    if sc > bs:
                        best, bs = v, sc
                pick = best
                self.enum_pick[key] = pick
            out[k] = pick
        return out

    KEYWORD_WORDS = ("text", "words", "keyword", "phrase", "query", "search", "terms")
    BROWSE_WORDS = ("browse", "shelf", "category", "department", "filter", "list", "fields",
                    "narrow", "refine", "facet", "field")
    PUT_WORDS = ("add", "put", "hold", "insert", "include", "plus", "keep", "set")
    DROP_WORDS = ("remove", "release", "drop", "delete", "minus", "discard", "take", "clear")

    def search(self, P, query=None, fields=None, pages=1, floor=0):
        role = "keyword_search" if query is not None else "field_search"
        r = self.roles.get(role)
        if not r:
            return 0
        if query is not None:
            sig = role + "|" + query
        else:
            sig = role + "|" + json.dumps(fields, sort_keys=True, default=str)
        cur = P.cursors.get(sig) or {"page": 0, "token": None, "done": False}
        if cur["done"]:
            return 0
        first = cur["page"] + 1
        token = cur["token"]
        mode = self.page_mode(role)
        if first > 1 and mode is None:
            return 0
        new = 0
        last = cur["page"]
        done = False
        for page in range(first, first + pages):
            if self.steps_left() <= floor or self.time_left() < MARGIN:
                break
            args = {}
            md = r.get("mode")
            if isinstance(md, dict) and md.get("arg"):
                args[md["arg"]] = md.get("value")
            if query is not None:
                qa = r.get("query_arg")
                if not qa:
                    return 0
                args[qa] = self.fit_query(P, query)
            else:
                for fname, fval in fields.items():
                    an = r.get(fname + "_arg")
                    if not an or fval is None or fval == "":
                        continue
                    v = self.world_value(role, an, fval)
                    if v is None:
                        P.cursors[sig] = {"page": 0, "token": token, "done": True}
                        return new
                    if isinstance(v, str):
                        v = self.trim(v, self.max_len(role, an))
                    args[an] = v
            if page > 1:
                pa = r.get("page_arg")
                if mode == "index":
                    args[pa] = page
                elif mode == "offset":
                    args[pa] = (page - 1) * max(1, self.page_size)
                elif mode == "token" and token:
                    args[pa] = token
                else:
                    done = True
                    break
                pv = args.get(pa)
                mx = self.arg_schema(role, pa).get("maximum")
                if is_int(pv) and isinstance(mx, (int, float)) and not isinstance(mx, bool) and pv > mx:
                    P.capped = True
                    done = True
                    break
            filled = {}
            try:
                filled = self.need_enums(role, args, self.KEYWORD_WORDS if query is not None else self.BROWSE_WORDS,
                                           self.BROWSE_WORDS if query is not None else self.KEYWORD_WORDS)
                if filled:
                    args.update(filled)
                    self.note("enum_filled", role=role, args=filled)
            except Exception as exc:
                self.note("enum_error", error=str(exc)[:160])
            obs, lines, err, ok = self.call(r["tool"], args)
            if not ok:
                for k, v in filled.items():
                    if str(k).lower() in str(err or "").lower() or str(v).lower() in str(err or "").lower():
                        self.enum_bad.add((role, k, v))
                        self.enum_pick.pop((role, k, self.KEYWORD_WORDS if query is not None else self.BROWSE_WORDS), None)
                        self.note("enum_rejected", role=role, arg=k, value=v)
                self.note("search_error", query=query, fields=fields, err=err)
                done = True
                last = page
                break
            rows = find_records(obs) or []
            tok = None
            total = None
            if isinstance(obs, dict):
                for k, v in obs.items():
                    if isinstance(v, str) and 1 <= len(v) <= 20 and " " not in v:
                        tok = v
                    if is_int(v):
                        total = v if total is None else max(total, v)
            if rows:
                self.page_size = max(self.page_size, len(rows))
            for rec in rows:
                L = self.build("row", rec)
                if L is None:
                    continue
                if L.id not in P.pool:
                    new += 1
                self.merge(P, L, False)
            P.pages_fetched += 1
            last = page
            if P.labels is None and not P.scratch:
                self.resolve_labels(P)
            if not rows or (total is not None and page * max(1, len(rows)) >= total) or len(rows) < self.page_size:
                done = True
            if mode == "token":
                if not tok:
                    if total is not None and total > page * len(rows):
                        P.capped = True
                    done = True
                else:
                    token = tok
            if done:
                break
        P.cursors[sig] = {"page": max(0, last), "token": token, "done": done or mode is None}
        return new

    # ---- E2.2 / E2.3 categories
    def cat_ok(self, P, L):
        if not L.category or not P.category:
            return True
        nl = norm(L.category)
        if P.labels is not None:
            return nl in P.labels
        nc = norm(P.category)
        return nl == nc or nl in nc or nc in nl or sim(L.category, P.category) >= 0.5

    def resolve_labels(self, P):
        if P.labels is not None:
            return
        labels = sorted({L.category for L in P.pool.values() if L.category})
        if not labels:
            return
        if len(labels) == 1:
            lb = labels[0]
            nl, nc = norm(lb), norm(P.category)
            if sim(lb, P.category) >= 0.3 or nl in nc or nc in nl:
                P.labels = {nl}
                self.note("labels", product=P.name, labels=sorted(P.labels))
            return
        user = "Wanted product: %s (%s)\nRequest excerpt: %s\nLabels seen: %s" % (
            P.name, P.category, self.request[:600], json.dumps(labels, ensure_ascii=False))
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_LABELS, user, 500)
        seen = {norm(x) for x in labels}
        kept = set()
        if isinstance(ans, dict) and isinstance(ans.get("labels"), list):
            for x in ans["labels"]:
                if isinstance(x, str) and norm(x) in seen:
                    kept.add(norm(x))
        if kept:
            P.labels = kept
            self.note("labels", product=P.name, labels=sorted(kept))

    # ---- E2.4 candidates
    def candidates(self, P, lax=False):
        out = []
        for i, L in enumerate(list(P.pool.values())):
            if self.in_stock and L.stock is False and not lax:
                continue
            if not lax and L.price is not None and P.unit_cap is not None \
                    and L.price > P.unit_cap + TOL:
                continue
            if not lax and L.price is not None and self.cap is not None \
                    and L.price * P.qty > self.cap + TOL:
                continue
            if not self.cat_ok(P, L):
                continue
            out.append((i, L))
        out.sort(key=lambda t: (mark_rank(t[1].mark), price_key(t[1]), t[0]))
        return [L for _, L in out]

    # ---- E2.5 screening
    def screen(self, P, listings):
        todo = [L for L in listings if not L.opened and L.mark == "maybe" and not L.screened]
        texts = P.texts()
        if len(todo) < 8 or not texts:
            return
        batch = todo[:30]
        rows = [{"n": i + 1, "title": (L.title or "")[:140], "text": (L.desc or "")[:ROW_TEXT_CAP],
                 "category": L.category, "price": L.price} for i, L in enumerate(batch)]
        for L in batch:
            L.screened = True
        user = "Wanted product: %s\nRequirements:\n%s\nRows:\n%s" % (
            P.category, "\n".join("- " + t for t in texts), json.dumps(rows, ensure_ascii=False))
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_SCREEN, user, 1200)
        if not isinstance(ans, dict):
            return
        for key in ("no", "yes"):
            lst = ans.get(key)
            if not isinstance(lst, list):
                continue
            for x in lst:
                n = None
                if is_int(x):
                    n = x
                elif isinstance(x, str) and x.strip().isdigit():
                    n = int(x.strip())
                if n is not None and 1 <= n <= len(batch):
                    batch[n - 1].mark = key

    # ---- E2.6 / E2.7 opening
    def open_batch(self, P, ids, floor=0):
        ids = [i for i in ids if i]
        if not ids:
            return []
        facts = self.roles.get("facts")
        opn = self.roles.get("open")
        comp = None if self.compare_off else self.roles.get("compare")
        if comp and not self.compare_checked:
            self.compare_checked = True
            mx = self.arg_schema("compare", comp.get("ids_arg")).get("maxItems")
            if is_int(mx) and 2 <= mx < self.compare_limit:
                self.compare_limit = mx
        out = []
        i = 0
        while i < len(ids) and self.steps_left() > 0 and self.time_left() > MARGIN:
            comp = None if self.compare_off else self.roles.get("compare")
            if self.thin and facts and P is not None:
                if floor and self.steps_left() - 1 < floor:
                    break
                L = self.read_by_table(P, ids[i])
                i += 1
                if L is not None:
                    out.append(L)
                continue
            if comp and len(ids) - i >= 2:
                if floor and self.steps_left() - 1 < floor:
                    break
                chunk = ids[i:i + self.compare_limit]
                obs, lines, err, ok = self.call(comp["tool"], {comp.get("ids_arg"): chunk})
                recs = find_records(obs) if ok else None
                if ok and recs:
                    Ls = [x for x in (self.build("detail", rec) for rec in recs) if x is not None]
                    thin = bool(facts) and len(Ls) > 0 and all(not x.props for x in Ls)
                    if thin and not self.thin:
                        self.thin = True
                        self.save_world()
                        self.note("thin_compare")
                    for x in Ls:
                        x.page_read = True
                    if thin and P is not None:
                        for x in Ls:
                            self.merge(P, x, False)
                        for x in Ls:
                            if floor and self.steps_left() - 1 < floor:
                                break
                            if self.time_left() < MARGIN:
                                break
                            y = self.read_by_table(P, x.id)
                            if y is not None:
                                out.append(y)
                    else:
                        for x in Ls:
                            x.opened = True
                            out.append(self.merge(P, x, True) if P is not None else x)
                    i += len(chunk)
                    continue
                if self.compare_limit > 2 and len(chunk) > 2:
                    self.compare_limit = 4 if self.compare_limit > 4 else 2
                    self.save_world()
                    continue
                self.compare_off = True
                self.note("compare_off", err=err)
                continue
            if not opn:
                break
            cost = 2 if facts else 1
            if floor and self.steps_left() - cost < floor:
                break
            lid = ids[i]
            i += 1
            obs, lines, err, ok = self.call(opn["tool"], {opn.get("id_arg"): lid})
            if not ok or not isinstance(obs, dict):
                continue
            L = self.build("detail", obs)
            if L is None:
                L = Listing(lid)
                L.raw = obs
            L.page_read = True
            if not L.props and facts and self.steps_left() > 0:
                t = self.fetch_table(lid)
                if t:
                    L.props = t
            L.id = lid
            L.opened = True
            out.append(self.merge(P, L, True) if P is not None else L)
        return out

    def fetch_table(self, lid):
        f = self.roles.get("facts")
        if not f:
            return None
        obs, lines, err, ok = self.call(f["tool"], {f.get("id_arg"): lid})
        if not ok or not isinstance(obs, dict):
            return None
        m = self.keymap("facts", obs)
        k = m.get("facts")
        if k and isinstance(obs.get(k), dict):
            src = obs[k]
        elif all(isinstance(v, (str, int, float, bool)) or v is None for v in obs.values()):
            src = obs
        else:
            return {}
        return {str(a): b for a, b in src.items() if b is not None and b != ""}

    def read_by_table(self, P, lid):
        t = self.fetch_table(lid)
        if t is None:
            return None
        if lid not in P.pool and not t:
            return None
        L = Listing(lid)
        L.props = t
        return self.merge(P, L, True)

    # ---- E2.8 label restatements
    def read_set(self, P):
        return [L for L in P.pool.values() if L.opened and L.props and self.cat_ok(P, L)]

    def drop_restatements(self, P):
        if P.labels is None:
            return
        lw = set()
        for lb in P.labels:
            lw |= toks(lb)
        op = self.read_set(P)
        if len(op) < 3:
            return
        keep = []
        for r in P.reqs:
            if r.origin == "request" and r.op == "is":
                tv = toks(r.value)
                if tv and tv <= lw and not any(near(k, r.prop) or sim(k, r.prop) >= 0.5 for L in op for k in L.props):
                    self.note("dropped_restatement", product=P.name, req=r.text())
                    continue
            keep.append(r)
        P.reqs = keep

    # ---- E2.9 field table
    def field_table(self, P):
        op = self.read_set(P)
        if len(op) < 3:
            return []
        rows = []
        for r in P.active():
            if r.op not in PLAIN_VALUE_OPS or not r.value or re.search(r"\d", r.value):
                continue
            wanted = [w for w in (toks(a) for a in r.value.split("|")) if w]
            if not wanted:
                continue
            keys = set()
            tally = {}
            carriers = 0
            writers = []
            for L in op:
                mk = [k for k in L.props if near(k, r.prop)]
                if not mk:
                    continue
                carriers += 1
                keys.update(mk)
                wrote = False
                for k in mk:
                    for piece in SEP_RE.split(str(L.props[k])):
                        pc = " ".join(piece.lower().split())
                        if not pc:
                            continue
                        tally[pc] = tally.get(pc, 0) + 1
                        pt = toks(pc)
                        if any(w <= pt for w in wanted):
                            wrote = True
                if wrote:
                    writers.append(L.id)
            if carriers:
                rows.append({"text": r.text(), "prop": r.prop, "keys": sorted(keys), "carriers": carriers,
                             "n": len(op), "tally": tally, "wanted": wanted, "writers": writers})
        return rows

    def field_failures(self, P, L, table=None):
        if table is None:
            table = self.field_table(P)
        out = set()
        for row in table:
            mk = [k for k in L.props if near(k, row["prop"])]
            if not mk:
                continue
            if not any(w != L.id for w in row["writers"]):
                continue
            pieces = [toks(p) for k in mk for p in SEP_RE.split(str(L.props[k]))]
            pieces = [p for p in pieces if p]
            if pieces and not any(w <= p for w in row["wanted"] for p in pieces):
                out.add(row["text"])
        return out

    def field_lines(self, table):
        out = []
        for row in table:
            top = sorted(row["tally"].items(), key=lambda kv: -kv[1])[:8]
            out.append("- %s | keys: %s | carried by %d of %d read listings | values: %s | %d listing(s) write the wanted value" % (
                row["text"], ", ".join("'%s'" % k for k in row["keys"][:3]), row["carriers"], row["n"],
                " | ".join("%s (%d)" % kv for kv in top), len(row["writers"])))
        return out

    # ---- E2.10 related pages
    def related(self, P, L):
        own = set(CODE_RE.findall(L.title or ""))
        ment = set(CODE_RE.findall((L.desc or "") + " " + (L.notes or ""))) - own
        if not own and not ment:
            return []
        out = []
        seen = {L.id}
        for Q in [P] + [x for x in self.products if x is not P]:
            for M in list(Q.pool.values()):
                if M.id in seen:
                    continue
                mt = set(CODE_RE.findall(M.title or ""))
                mtxt = set(CODE_RE.findall((M.desc or "") + " " + (M.notes or "")))
                if (mt & ment) or (mtxt & own):
                    seen.add(M.id)
                    out.append({"id": M.id, "title": (M.title or "")[:120],
                                "text": " ".join(x for x in (M.desc, M.notes) if x)[:400]})
                    if len(out) >= 2:
                        return out
        return out

    # ---- E2.11 judging
    def map_name(self, x, texts):
        m = re.fullmatch(r"\s*R?\s*(\d+)\s*", x)
        if m and 1 <= int(m.group(1)) <= len(texts):
            return texts[int(m.group(1)) - 1]
        best, bs = None, 0.2
        for t in texts:
            s = sim(x, t)
            if s > bs:
                best, bs = t, s
        return best if best is not None else x

    def judge(self, P, listings):
        self.drop_restatements(P)
        texts = P.texts()
        if not texts:
            for L in listings:
                if L.opened and L.id not in P.verdicts:
                    P.verdicts[L.id] = empty_verdict()
                    P.judged_for[L.id] = set()
            return
        ts = set(texts)
        todo = [L for L in listings if L.opened and not ts <= P.judged_for.get(L.id, set())]
        if not todo or self.time_left() < MARGIN:
            return
        table = self.field_table(P)
        fl = self.field_lines(table)
        head = "Wanted product: %s\nRequirements:\n%s" % (
            P.category, "\n".join("R%d. %s" % (i + 1, t) for i, t in enumerate(texts)))
        if fl:
            head += "\nStore fields (keys and values used by the listings of this category read in this store):\n" + "\n".join(fl)
        batches = [todo[i:i + 4] for i in range(0, len(todo), 4)]

        def job(b):
            recs = []
            for L in b:
                s = summary(L, SUMMARY_CAP)
                rp = self.related(P, L)
                if rp:
                    s["related_pages"] = rp
                recs.append(s)
            user = head + "\nListings:\n" + json.dumps(recs, ensure_ascii=False, default=str)
            return lambda: self.llm.json_call([M_FAST, M_BACKUP], P_JUDGE, user, 2000)

        answers = parallel([job(b) for b in batches])
        for b, ans in zip(batches, answers):
            ents = {}
            if isinstance(ans, dict) and isinstance(ans.get("verdicts"), list):
                for e in ans["verdicts"]:
                    if isinstance(e, dict) and e.get("id") is not None and str(e.get("id")) in P.pool:
                        ents[str(e["id"])] = e
            for L in b:
                e = ents.get(L.id)
                if e is None:
                    continue
                sets = {}
                for k in ("failed", "unsure", "weak", "loose"):
                    v = e.get(k)
                    items = [str(x) for x in v if str(x).strip()] if isinstance(v, list) else []
                    sets[k] = {self.map_name(x, texts) for x in items}
                sets["weak"] |= sets["loose"]
                sets["failed"] |= self.field_failures(P, L, table)
                for r in P.active():
                    t = r.text()
                    if t in sets["failed"] or t in sets["unsure"] or t in sets["loose"]:
                        continue
                    if self.unstated(P, L, r):
                        sets["unsure"].add(t)
                P.verdicts[L.id] = sets
                P.judged_for[L.id] = set(texts)

    # ---- E2.12 unstated values
    def unstated(self, P, L, r):
        if r.op not in PLAIN_VALUE_OPS or r.origin == "fit" or not r.value:
            return False
        pw = words(P.name) | words(P.category)
        alts = []
        for a in r.value.split("|"):
            w = {x for x in words(a) - pw if not (x.isalpha() and len(x) <= 3)}
            if w:
                alts.append(w)
        if not alts:
            return False
        lw = words(L.title) | words(L.desc) | words(L.notes) | words(L.category)
        for k, v in L.props.items():
            lw |= words("%s %s" % (k, v))
        return not any(a & lw for a in alts)

    # ---- E2.13 judging every product
    def judge_all(self):
        jobs = []
        for P in self.products:
            jobs.append(lambda P=P: self.judge(P, [L for L in self.candidates(P) if L.opened]))
        parallel(jobs)

    # ---- E2.14 final verification
    def verify(self, P, L, strict=False):
        texts = P.texts()
        key = (P.idx, L.id, tuple(texts), json.dumps(L.props, sort_keys=True, default=str))
        if not strict and key in self.vcache:
            return self.vcache[key]
        budget = ("Budget for all lines together: %s" % fmt_num(self.cap)) if self.cap is not None else "Budget: none stated"
        reqs = "\n".join("- " + t for t in texts) if texts else \
            "Nothing is required apart from the product type, the price limit and being in stock."
        user = "Wanted product: %s (quantity %d)\n%s\nRequirements:\n%s\nListing:\n%s" % (
            P.category, P.qty, budget, reqs, json.dumps(summary(L, 3500), ensure_ascii=False, default=str))
        ans = self.llm.json_call([M_VERIFY, M_FAST], P_VERIFY, user, 800)
        if not isinstance(ans, dict):
            res = not strict
        elif strict and ans.get("fits") is not True:
            return False
        else:
            probs = ans.get("problems") if isinstance(ans.get("problems"), list) else []
            self.note("verify", product=P.name, id=L.id, fits=ans.get("fits"), problems=[str(p)[:120] for p in probs[:4]])
            if ans.get("fits") is False and probs:
                v = P.verdicts.get(L.id)
                if v is None:
                    v = empty_verdict()
                    P.verdicts[L.id] = v
                for p in probs:
                    v["failed"].add(self.map_name_sim(str(p), texts))
                res = False
            else:
                res = True
        if not strict:
            self.vcache[key] = res
        return res

    def map_name_sim(self, x, texts):
        best, bs = None, 0.2
        for t in texts:
            s = sim(x, t)
            if s > bs:
                best, bs = t, s
        return best if best is not None else x


    # ---------------------------------------------------------------- E3 messages
    def msg_ok(self):
        if "message" not in self.roles:
            return False
        if self.allowance is not None and self.sent >= self.allowance:
            return False
        return self.steps_left() >= 1

    def held(self):
        if "message" not in self.roles or not self.procedure:
            return 0
        n = 0
        for s in self.procedure:
            if s["kind"] == "message" and s["moment"] in ("pre-order", "pre-finish"):
                c = s.get("messages")
                if not c:
                    c = 1
                c = int(c)
                if c < 1:
                    c = 1
                n += c
        if self.allowance is not None:
            n = min(n, max(0, self.allowance - self.sent))
        return n

    def sig(self):
        return tuple(sorted((lid, int(q)) for lid, q in self.cart.items() if q))

    def pending(self):
        if self.cart and self.sig() == self.confirmed:
            return 0
        return self.held()

    def q_ok(self):
        if not self.msg_ok():
            return False
        if self.allowance is not None and self.sent >= self.allowance - self.held():
            return False
        return True

    def msg_tool(self):
        r = self.roles.get("message") or {}
        return r.get("tool"), r.get("text_arg")

    PLAIN_WORDS = ("send", "ask", "tell", "say", "message", "text", "note", "open", "read", "view", "show")

    def send_message(self, text):
        tool, ta = self.msg_tool()
        text = self.trim(text, self.max_len("message", ta))
        args = {ta: text}
        try:
            extra = self.need_enums("message", args, self.PLAIN_WORDS)
            if extra:
                args.update(extra)
                self.note("enum_filled", role="message", args=extra)
        except Exception:
            pass
        res = self.call(tool, args)
        self.sent += 1
        return res, text

    # ---- C6 process gate
    def withheld(self):
        return {self.role_tool(r) for r in ("checkout", "decline", "cart_add", "cart_set")} - {None}

    def basket_view(self):
        lines = []
        total = 0.0
        for lid, q in self.sig():
            P = self.carted.get(lid)
            L = P.pool.get(lid) if P is not None else None
            price = L.price if L is not None else None
            lines.append({"id": lid, "quantity": q, "title": (L.title or "")[:120] if L is not None else "",
                          "price": price, "for": P.name if P is not None else None})
            total += (price or 0) * q
        return {"lines": lines, "total": total}

    def gate(self, moment, intent):
        if self.gate_counts.get(moment, 0) >= 3 or self.steps_left() < 2 or self.time_left() < MARGIN:
            return "nothing"
        self.gate_counts[moment] = self.gate_counts.get(moment, 0) + 1
        wh = self.withheld()
        tools = [{"name": f.get("name"), "description": f.get("description") or "", "parameters": f.get("parameters")}
                 for f in self.tools if f.get("name") and f.get("name") not in wh]
        left = ("%d" % max(0, self.allowance - self.sent)) if self.allowance is not None else "no limit"
        user = "\n\n".join([
            "REQUEST:\n" + self.request,
            "PROCESS STEPS DEMANDED:\n" + json.dumps(self.procedure, ensure_ascii=False, default=str),
            "MOMENT: %s (the agent is %s)" % (moment, intent),
            "BASKET NOW:\n" + json.dumps(self.basket_view(), ensure_ascii=False, default=str),
            "MESSAGES LEFT: %s\nSTEPS LEFT: %d" % (left, self.steps_left()),
            "TOOLS:\n" + json.dumps(tools, ensure_ascii=False, default=str),
            "HISTORY (oldest first):\n" + json.dumps(self.history[-30:], ensure_ascii=False, default=str),
        ])
        ans = self.llm.json_call([M_PLAN, M_BACKUP], P_GATE, user, 900)
        if not isinstance(ans, dict):
            self.note("gate", moment=moment, verdict="no answer")
            return "noanswer"
        acts = ans.get("actions")
        acts = [a for a in (acts[:2] if isinstance(acts, list) else []) if isinstance(a, dict)]
        mtool = self.role_tool("message")
        verdict = "nothing"
        for j, a in enumerate(acts):
            tool = a.get("tool")
            if not isinstance(tool, str) or tool not in self.schemas or tool in wh:
                continue
            is_msg = tool == mtool
            if not is_msg:
                due = max(sum(1 for b in acts[j + 1:] if b.get("tool") == mtool), self.pending())
                if self.steps_left() < 2 + due:
                    continue
            sch = self.schemas[tool]
            props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
            a0 = a.get("args") if isinstance(a.get("args"), dict) else {}
            args = {}
            for k, v in a0.items():
                if k not in props:
                    continue
                p = props[k] if isinstance(props[k], dict) else {}
                en = p.get("enum")
                if isinstance(en, list) and en and v not in en:
                    continue
                mx = p.get("maxLength")
                if isinstance(v, str) and is_int(mx) and len(v) > mx:
                    v = v[:mx]
                args[k] = v
            req = sch.get("required") if isinstance(sch.get("required"), list) else []
            if any(k not in args for k in req):
                continue
            if is_msg and not self.msg_ok():
                continue
            if self.steps_left() < 2:
                break
            self.call(tool, args)
            if is_msg:
                self.sent += 1
                self.confirmed = self.sig()
            verdict = "ran"
            self.note("gate_action", moment=moment, tool=tool, why=str(a.get("why"))[:160])
        self.note("gate", moment=moment, verdict=verdict)
        return verdict

    # ---- C7 basket confirmation
    def confirm_basket(self):
        if not self.msg_ok() or self.steps_left() < 2:
            return False
        parts = []
        total = 0.0
        for lid, q in self.sig():
            P = self.carted.get(lid)
            L = P.pool.get(lid) if P is not None else None
            title = " ".join(L.title.split())[:70] if (L is not None and L.title) else lid
            price = L.price if L is not None else None
            s = "%d x %s" % (q, title)
            if price is not None:
                s += " (%s each)" % fmt_num(price)
                total += price * q
            parts.append(s)
        text = "Final basket: " + "; ".join(parts)
        text += (". Total %s." % fmt_num(total)) if total else "."
        text += " Shall I proceed to checkout with these items? Please reply yes to confirm."
        self.send_message(text)
        self.confirmed = self.sig()
        self.note("basket_confirmation")
        return True

    # ---- C8 asking
    def ask(self, P, prop, narrowed=False):
        if narrowed:
            ex = []
            for L in list(P.pool.values()):
                for k, v in L.props.items():
                    kn = split_caps(k).replace("_", " ")
                    if near(kn, prop) and str(v) not in ex:
                        ex.append(str(v)[:30])
            ex = ex[:2]
            text = "Just one thing about the %s: what %s do you need" % (P.name, prop)
            text += (", for example %s?" % " or ".join(ex)) if ex else "?"
        else:
            text = "Which %s would you like the %s to have?" % (prop, P.name)
        (obs, lines, err, ok), sent_text = self.send_message(text)
        if not narrowed:
            P.asked.append(norm(prop))
        self.note("ask", product=P.name, prop=prop, narrowed=narrowed)
        if not ok:
            return None
        if isinstance(obs, str):
            rt = obs
        elif isinstance(obs, dict):
            strs = [v for v in obs.values() if isinstance(v, str)]
            rt = max(strs, key=len) if strs else cut(obs, 400)
        else:
            rt = ""
        if not rt.strip():
            return None
        known = [r.text() for r in P.reqs if r.origin == "shopper" and r.value]
        user = "Product: %s\nAsked property: %s\nQuestion sent: %s\nKnown needs: %s\nShopper reply: %s" % (
            P.name, prop, sent_text, json.dumps(known, ensure_ascii=False), rt)
        ans = self.llm.json_call([M_FAST, M_BACKUP, M_PLAN], P_REPLY, user, 600)
        if not isinstance(ans, dict):
            return None
        if str(ans.get("kind") or "").lower() == "need":
            nm = self.name_property(P, [prop, txt(ans.get("property"))], rt, txt(ans.get("value")))
            if nm:
                ans["property"] = nm
        self.note("reply", product=P.name, prop=prop, reading=cut(ans, 300))
        return ans

    # ---- C9 naming the property
    def name_property(self, P, asked, reply, value):
        r = " " + norm(reply) + " "
        for a in asked:
            na = norm(a)
            if na and (" " + na + " ") in r:
                return None
        vt = toks(value)
        cands = []
        for L in list(P.pool.values()):
            for k in L.props:
                c = norm(split_caps(k))
                if len(c) < 4 or c in cands:
                    continue
                if toks(c) & vt:
                    continue
                if any(near(c, a) for a in asked):
                    continue
                cands.append(c)
        hits = [c for c in cands if (" " + c + " ") in r]
        hits = [h for h in hits if not any(o != h and (" " + h + " ") in (" " + o + " ") for o in hits)]
        return hits[0] if len(hits) == 1 else None

    # ---- C10 applying a reply
    def apply_reply(self, P, prop, rd):
        if not isinstance(rd, dict):
            return
        st = rd.get("steps")
        if isinstance(st, list):
            for s in st:
                if isinstance(s, dict):
                    self.add_step(s)
        b = num(rd.get("budget"))
        if b is not None and b > 0:
            self.cap = b
            self.note("budget", cap=b, source="reply")
        kind = str(rd.get("kind") or "").lower()
        if kind == "need":
            val = txt(rd.get("value"))
            ref = rd.get("reference")
            if not (val or ref):
                return
            p = txt(rd.get("property")) or prop
            op = txt(rd.get("op")).lower() or "is"
            r = Req(p, op, val, ref if isinstance(ref, dict) else None, "shopper")
            if r.ref is not None:
                r.ref = dict(r.ref)
                r.ref["property"] = r.prop
            if r.value and any(x.origin == "shopper" and norm(x.value) == norm(r.value) and x.op == r.op for x in P.reqs):
                return
            P.reqs = [x for x in P.reqs if not (x.origin == "shopper" and norm(x.prop) == norm(r.prop))]
            P.reqs.append(r)
            self.note("need", product=P.name, req=r.text())
            if r.ref is not None and r.op in REL_TABLE:
                self.follow(P, r, 0)
        elif kind == "free":
            P.dontcare.append(norm(txt(rd.get("property")) or prop))

    def apply_kept(self, P, prop, rd):
        left = max(0, self.allowance - self.sent) if self.allowance is not None else 2
        self.ref_floor = left + self.reserve()
        try:
            self.apply_reply(P, prop, rd)
        finally:
            self.ref_floor = 0

    # ---- C12 property menu
    def menu(self, P):
        base = [L for L in P.pool.values() if L.opened and self.cat_ok(P, L)]
        live = [L for L in base if not (P.verdicts.get(L.id) or {}).get("failed")]
        S = live if len(live) >= 4 else base
        tables = []
        for L in S:
            t = {}
            for k, v in L.props.items():
                c = " ".join(split_caps(k).replace("_", " ").replace(".", " ").split()).lower()
                if not c or c in t:
                    continue
                t[c] = str(v).strip().lower()[:60]
            tables.append(t)
        count = {}
        for t in tables:
            for c in t:
                count[c] = count.get(c, 0) + 1
        names = sorted(count, key=lambda c: (-count[c], len(c), c))
        alias_of = {}
        heads = []
        aliases = {}
        for i, h in enumerate(names):
            if h in alias_of:
                continue
            heads.append(h)
            aliases[h] = []
            for o in names[i + 1:]:
                if o in alias_of or o in aliases:
                    continue
                both = [t for t in tables if h in t and o in t]
                if len(both) >= 2 and sum(1 for t in both if t[h] == t[o]) >= 0.8 * len(both):
                    alias_of[o] = h
                    aliases[h].append(o)
        fixed = [norm(r.prop) for r in P.reqs] + list(P.asked)

        def is_fixed(n):
            nn = norm(n)
            return any(nn == f or sim(n, f) >= 0.67 for f in fixed)

        entries = []
        for h in heads:
            group = [h] + aliases[h]
            if any(is_fixed(n) for n in group):
                continue
            nc = 0
            vals = []
            allv = set()
            for t in tables:
                for n in group:
                    if n in t:
                        nc += 1
                        v = t[n]
                        allv.add(v)
                        if v and v not in vals:
                            vals.append(v)
                        break
            w = float(nc)
            if len(vals) == 1 and nc >= 3:
                w /= 2
            if nc >= 4 and len(allv) >= nc:
                w /= 2
            e = {"property": h, "carried_by": nc, "of": len(S), "values": vals[:3]}
            if aliases[h]:
                e["also_called"] = aliases[h][:3]
            entries.append((w, h, e))
        entries.sort(key=lambda x: (-x[0], x[1]))
        return [e for _, _, e in entries[:40]], len(S)

    # ---- C14 decided properties
    def decided(self, P):
        out = []
        for c in self.conditions:
            if c["source"] is P:
                out.append(c["prop"])
            for b in c["branches"]:
                if b["target"] is P:
                    out.append(b["prop"])
        if any(a is P for a, h in self.fit_pairs()):
            out.append("compatibility")
        res = []
        for x in out:
            if x not in res:
                res.append(x)
        return res

    # ---- C15 planning
    def taken(self, P, prop, plan):
        if any(near(prop, a) for a in P.asked):
            return True
        for d in self.decided(P):
            if near(prop, d) or sim(prop, d) >= 0.5:
                return True
        if any(near(prop, r.prop) for r in P.reqs):
            return True
        if any(x is P and near(prop, y) for x, y in plan):
            return True
        return False

    def name_best(self, name):
        best, bs = None, -1.0
        for P in self.products:
            s = sim(name, P.name)
            if s > bs:
                best, bs = P, s
        return best

    def plan(self, count, model=M_PLAN):
        if count <= 0:
            return []
        if not self.planner_list:
            blocks = []
            for P in self.products:
                ents, _n = self.menu(P)
                blocks.append(json.dumps({"name": P.name, "category": P.category, "known_requirements": P.texts(),
                                          "fixed_by_rules": self.decided(P), "hints": P.hints,
                                          "already_asked": P.asked, "property_menu": ents},
                                         ensure_ascii=False, default=str))
            want = max(self.allowance if self.allowance is not None else count, count)
            user = "REQUEST:\n%s\n\nTopics to raise: %s\n\nPRODUCTS:\n%s\n\nNumber of questions wanted: %d" % (
                self.request, json.dumps(self.topics, ensure_ascii=False), "\n".join(blocks), want)
            models = [model, M_BACKUP if model == M_FAST else M_FAST]
            ans = self.llm.json_call(models, P_PLANNER, user, 700)
            lst = []
            if isinstance(ans, dict) and isinstance(ans.get("questions"), list):
                for q in ans["questions"]:
                    if isinstance(q, dict) and q.get("property"):
                        P = self.name_best(txt(q.get("product")))
                        prop = " ".join(str(q["property"]).replace("_", " ").split()).lower()
                        if any(x is P and near(y, prop) for x, y in lst):
                            continue
                        lst.append((P, prop))
            self.planner_list = lst
            self.note("planner", questions=[(P.name, p) for P, p in lst])
        if len(self.products) > 1:
            return self.multi_plan(count)
        P = self.products[0]
        plan = []
        for Q, prop in self.planner_list:
            if len(plan) >= count:
                break
            if not self.taken(Q, prop, plan):
                plan.append((Q, prop))
        if count >= 3:
            ents, _n = self.menu(P)
            tops = [e for e in ents if e["carried_by"] >= 3][:2]
            for j, e in enumerate(tops):
                if self.taken(P, e["property"], plan):
                    continue
                plan.insert(1 if j == 0 else 3, (P, e["property"]))
            plan = plan[:count]
        if len(plan) < count:
            ents, n = self.menu(P)
            pad = ents if n <= 2 else [e for e in ents if e["carried_by"] >= 2]
            for e in pad:
                if len(plan) >= count:
                    break
                if not self.taken(P, e["property"], plan):
                    plan.append((P, e["property"]))
        return plan[:count]

    # ---- C16 several products
    def multi_plan(self, count):
        queues = {id(P): [] for P in self.products}
        order = []
        for P, prop in self.planner_list:
            q = queues[id(P)]
            if not self.taken(P, prop, []) and not any(near(prop, x) for x in q):
                q.append(prop)
            if P not in order:
                order.append(P)
        for P in self.products:
            ents, n = self.menu(P)
            mnu = ents if n <= 2 else [e for e in ents if e["carried_by"] >= 2]
            q = queues[id(P)]
            top = next((e for e in mnu if e["carried_by"] >= 3), None)
            if top is not None:
                tp = top["property"]
                if not self.taken(P, tp, []) and not any(near(tp, x) for x in q[:2]):
                    q[:] = [x for x in q if not near(x, tp)]
                    q.insert(1 if q else 0, tp)
            for e in mnu:
                p = e["property"]
                if not self.taken(P, p, []) and not any(near(p, x) for x in q):
                    q.append(p)
            if P not in order:
                order.append(P)
        pos = {id(P): i for i, P in enumerate(order)}
        order.sort(key=lambda P: (1 if any(r.origin == "shopper" for r in P.reqs) else 0, len(P.asked), pos[id(P)]))
        out = []
        while len(out) < count and any(queues[id(P)] for P in order):
            for P in order:
                if len(out) >= count:
                    break
                if queues[id(P)]:
                    out.append((P, queues[id(P)].pop(0)))
        return out

    # ---------------------------------------------------------------- E4.1 references
    def follow(self, P, r, hop=0):
        if hop > 7 or self.steps_left() < max(4, self.ref_floor) or self.time_left() < MARGIN + 20:
            return False
        ref = r.ref if isinstance(r.ref, dict) else {}
        seller = txt(ref.get("seller"))
        cat = txt(ref.get("category"))
        attrs = ref.get("attributes") if isinstance(ref.get("attributes"), dict) else None
        wanted = txt(ref.get("property")) or r.prop
        S = Product(99, {"name": cat or "item", "category": cat or "item", "search_terms": ["item"]})
        S.category = cat
        S.terms = []
        vals = [txt(v) for v in (attrs or {}).values() if txt(v)][:3]
        vw = " ".join(vals)
        if seller:
            qs = [(seller + " " + vw).strip()]
            if cat:
                qs.append(seller + " " + cat)
            qs.append(seller)
        elif cat:
            qs = [(cat + " " + vw).strip()]
        else:
            self.note("reference", product=P.name, result="unfound", hop=hop)
            return False
        vtok = toks(vw)

        def overlap(L):
            lt = toks(L.title) | toks(L.desc) | toks(L.notes)
            for k, v in L.props.items():
                lt |= toks(k) | toks(v)
            return len(vtok & lt)

        def seller_ok(L):
            if not seller:
                return True
            if sim(L.seller, seller) >= 0.5 or sim(L.brand, seller) >= 0.5:
                return True
            return norm(seller) in norm(" ".join([L.title or "", L.desc or "", L.notes or ""]))

        def accepted():
            return [L for L in S.pool.values() if seller_ok(L) and self.cat_ok(S, L)]

        acc = []
        for q in qs:
            if self.steps_left() < 4:
                break
            for _t in range(2):
                n = self.search(S, query=q, pages=1, floor=3)
                if n == 0:
                    break
                acc = accepted()
                if acc:
                    break
            if len(acc) >= 2 or (acc and overlap(acc[0]) > 0):
                break
        if not acc:
            self.note("reference", product=P.name, result="unfound", hop=hop)
            return False
        acc = [L for _, L in sorted(enumerate(acc), key=lambda t: (
            -overlap(t[1]), 0 if sim(t[1].seller, seller) >= 0.5 else 1,
            t[1].price if t[1].price is not None else 0, t[0]))]
        count = ref.get("count")
        count = count if is_int(count) else None
        pick = ref.get("pick")
        desc = "Described listing: category %s; seller %s; attributes %s" % (
            json.dumps(cat), json.dumps(seller), json.dumps(attrs or {}, ensure_ascii=False, default=str))
        page = None
        if pick or (count is not None and count > 1):
            opened = []
            rest = list(acc)
            for _b in range(3):
                if not rest or self.steps_left() < 3:
                    break
                chunk, rest = rest[:self.compare_limit], rest[self.compare_limit:]
                opened += self.open_batch(None, [L.id for L in chunk])
            if opened:
                user = desc
                if pick:
                    user += "\nPick rule: %s" % txt(pick)
                if count is not None:
                    user += "\nNumber of qualifying listings: %d" % count
                user += "\nCandidates:\n" + json.dumps([summary(L, 1200) for L in opened], ensure_ascii=False, default=str)
                ans = self.llm.json_call([M_FAST, M_BACKUP], P_MATCH, user, 600)
                aid = ans.get("id") if isinstance(ans, dict) else None
                if isinstance(aid, (str, int)) and not isinstance(aid, bool) and str(aid) in S.pool:
                    page = next((L for L in opened if L.id == str(aid)), None) or S.pool[str(aid)]
        if page is None:
            single = "compare" not in self.roles or self.compare_off
            size = 1 if single else self.compare_limit
            maxr = 5 if single else 3
            rest = list(acc)
            collected = []
            for _rnd in range(maxr):
                if not rest or page is not None or self.steps_left() < 3:
                    break
                chunk, rest = rest[:size], rest[size:]
                pages = self.open_batch(None, [L.id for L in chunk])
                for pg in pages:
                    if self.exact_match(pg, attrs):
                        page = pg
                        break
                if page is not None:
                    break
                if single:
                    collected += pages
                    if len(collected) < 3 and rest:
                        continue
                    shown, collected = collected, []
                else:
                    shown = pages
                if not shown:
                    continue
                user = desc + "\nCandidates:\n" + json.dumps([summary(L, 1600) for L in shown], ensure_ascii=False, default=str)
                ans = self.llm.json_call([M_FAST, M_BACKUP], P_MATCH, user, 600)
                aid = ans.get("id") if isinstance(ans, dict) else None
                if isinstance(aid, (str, int)) and not isinstance(aid, bool) and str(aid) in S.pool:
                    page = next((L for L in shown if L.id == str(aid)), None) or S.pool[str(aid)]
                    break
                hits = [pg for pg in shown if self.carries_all(pg, attrs)]
                if len(hits) == 1:
                    page = hits[0]
                    break
                if len(shown) == 1 and not rest:
                    page = shown[0]
                    break
        if page is None:
            self.note("reference", product=P.name, result="unmatched", hop=hop)
            return False
        counter = r.op == "counterpart of reference"
        thin = self.thin and not page.props

        def read():
            user = "Wanted property: %s\nPage:\n%s" % (wanted, json.dumps(summary(page, 3000), ensure_ascii=False, default=str))
            return self.llm.json_call([M_FAST, M_BACKUP], P_PAGE, user, 600)

        if thin and not page.notes and self.steps_left() >= 3:
            t = self.fetch_table(page.id)
            if t:
                page.props = t
            thin = False
        ans = read()
        if thin and self.steps_left() >= 3 and isinstance(ans, dict) and not ans.get("redirect") and \
                (not txt(ans.get("value")) or counter):
            t = self.fetch_table(page.id)
            if t:
                page.props = t
                a2 = read()
                ans = a2
            thin = not page.props
        if not isinstance(ans, dict):
            self.note("reference", product=P.name, result="unread", hop=hop)
            return False
        val = txt(ans.get("value"))
        paired = txt(ans.get("paired"))
        red = ans.get("redirect")
        if isinstance(red, dict) and red:
            subj = txt(red.get("property")) or wanted
            ns, nw = norm(subj), norm(wanted)
            concerns = near(subj, wanted) or sim(subj, wanted) >= 0.5 or ns in nw or nw in ns
            if concerns:
                newref = dict(red)
                newref["property"] = wanted
                r.ref = newref
                self.note("reference", product=P.name, result="redirect", hop=hop)
                return self.follow(P, r, hop + 1)
            if not val and thin and not page.props and self.steps_left() >= 3:
                t = self.fetch_table(page.id)
                if t:
                    page.props = t
                    a2 = read()
                    if isinstance(a2, dict):
                        val = txt(a2.get("value")) or val
                        paired = txt(a2.get("paired")) or paired
            if not val:
                for k, v in page.props.items():
                    kk = str(k).replace("_", " ")
                    if (near(kk, wanted) or sim(kk, wanted) >= 0.5) and txt(v):
                        val = txt(v)
                        break
        if counter and val and not paired:
            self.note("reference", product=P.name, result="no paired value", hop=hop)
            return False
        use = paired if counter else val
        if not use:
            self.note("reference", product=P.name, result="no value", hop=hop)
            return False
        r.value = use
        r.op = REL_TABLE.get(r.op, "is")
        r.ref = None
        if r.op == "is" and len(use) <= 40 and P.terms and not any(norm(use) in norm(t) for t in P.terms):
            P.terms.insert(0, (use + " " + P.noun)[:60])
        self.note("reference", product=P.name, result="resolved", req=r.text(), hop=hop)
        return True

    def exact_match(self, pg, attrs):
        pairs = list((attrs or {}).items())
        for k, v in pairs:
            nv = norm(v)
            if not nv:
                return False
            found = False
            for pk, pv in pg.props.items():
                kk = split_caps(pk)
                if (near(kk, k) or sim(kk, k) >= 0.5) and norm(pv) == nv:
                    found = True
                    break
            if not found:
                return False
        return True

    def carries_all(self, pg, attrs):
        lw = words(pg.title) | words(pg.notes) | words(pg.desc)
        for k, v in pg.props.items():
            lw |= words("%s %s" % (k, v))
        for v in (attrs or {}).values():
            if not txt(v):
                continue
            if not words(v) <= lw:
                return False
        return True

    # ---------------------------------------------------------------- E4.2 shopper lines
    def handle_lines(self):
        taken = [x for x in self.pending_lines if x not in self.handled]
        self.pending_lines = []
        if not taken:
            return False
        for x in taken:
            self.handled.add(x)
        user = "Shopper lines:\n%s\nWanted products: %s" % (
            "\n".join("- " + str(x) for x in taken), json.dumps([P.name for P in self.products], ensure_ascii=False))
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_LINES, user, 700)
        if not isinstance(ans, dict):
            return False
        changed = False
        chs = ans.get("changes")
        for ch in chs if isinstance(chs, list) else []:
            if not isinstance(ch, dict):
                continue
            kind = str(ch.get("kind") or "").lower()
            tgt = self.line_target(txt(ch.get("product")))
            if kind == "budget":
                b = num(ch.get("budget"))
                if b is None or b == 0:
                    continue
                nums = []
                for x in taken:
                    for m in re.findall(r"\d[\d,]*(?:\.\d+)?", str(x)):
                        try:
                            nums.append(float(m.replace(",", "")))
                        except Exception:
                            pass

                def has(v):
                    return any(abs(v - n) < 1e-9 for n in nums)

                if has(b):
                    v = b
                elif has(b * 100):
                    v = b * 100
                elif has(b / 100.0):
                    v = b / 100.0
                elif nums and self.cap is not None:
                    v = next((n for n in nums if self.cap / 20.0 <= n <= self.cap * 20.0), b)
                else:
                    v = b
                self.cap = v
                changed = True
                self.note("budget", cap=v, source="line")
            elif kind == "drop":
                prop = txt(ch.get("property"))
                if not prop:
                    continue
                for Q in ([tgt] if tgt is not None else self.products):
                    before = len(Q.reqs)
                    Q.reqs = [r for r in Q.reqs if sim(r.prop, prop) < 0.5]
                    if len(Q.reqs) < before:
                        changed = True
                        for v in Q.verdicts.values():
                            v["failed"] = {t for t in v["failed"] if sim(t.split(":", 1)[0], prop) < 0.5}
                        self.note("drop", product=Q.name, prop=prop)
            elif kind == "add":
                prop = txt(ch.get("property"))
                val = txt(ch.get("value"))
                if not prop or not val:
                    continue
                Q = tgt if tgt is not None else self.products[0]
                Q.reqs.append(Req(prop, txt(ch.get("op")) or "is", val, None, "shopper"))
                changed = True
                self.note("add", product=Q.name, prop=prop, value=val)
            elif kind in ("price", "stock"):
                changed = True
                self.refresh_cart()
            elif kind == "step":
                if isinstance(ch.get("step"), dict):
                    self.add_step(ch["step"])
        return changed

    def line_target(self, name):
        if len(self.products) == 1:
            return self.products[0]
        if not name:
            return None
        best, bs = None, -1.0
        for P in self.products:
            s = sim(name, P.name)
            if s > bs:
                best, bs = P, s
        return best if bs >= 0.2 else None

    # ---- E4.3 procedure step
    def add_step(self, s):
        if not s.get("goal") or len(self.procedure) >= 8:
            return
        st = dict(s)
        st["moment"] = str(s.get("moment")).lower() if s.get("moment") else "pre-finish"
        st["kind"] = "message" if str(s.get("kind") or "").strip().lower() == "message" else "tool"
        if any(sim(str(x.get("goal")), str(st.get("goal"))) >= 0.6 for x in self.procedure):
            return
        self.procedure.append(st)
        if st["kind"] == "message" and st["moment"] in ("pre-order", "pre-finish"):
            self.confirm_first = True
        self.note("step_added", step=cut(st, 200))

    # ---- E4.4 - E4.7 cart
    def refresh_cart(self):
        for lid, P in list(self.carted.items()):
            if self.steps_left() <= 2:
                break
            self.open_batch(P, [lid])

    def set_cart(self, wanted):
        r = self.roles["cart_set"]
        tool = r.get("tool")
        la = r.get("lines_arg")
        sch = self.schemas.get(tool) or {}
        props = sch.get("properties") if isinstance(sch.get("properties"), dict) else {}
        items = (props.get(la) or {}).get("items") if isinstance(props.get(la), dict) else None
        lp = items.get("properties") if isinstance(items, dict) and isinstance(items.get("properties"), dict) else {}
        idk = r.get("line_id_key") or "id"
        qk = r.get("line_quantity_key")
        mx = lp.get(qk, {}).get("maximum") if (qk and isinstance(lp.get(qk), dict)) else None
        lines = []
        for lid, q in wanted:
            d = {idk: lid}
            if qk:
                d[qk] = min(q, mx) if is_int(mx) else q
            lines.append(d)
        args = {la: lines}
        filled = {}
        try:
            filled = self.need_enums("cart_set", args, self.PUT_WORDS, self.DROP_WORDS)
            if filled:
                args.update(filled)
                self.note("enum_filled", role="cart_set", args=filled)
        except Exception:
            pass
        obs, _l, err, ok = self.call(tool, args)
        if not ok:
            for k, v in filled.items():
                if str(k).lower() in str(err or "").lower() or str(v).lower() in str(err or "").lower():
                    self.enum_bad.add(("cart_set", k, v))
                    self.enum_pick.pop(("cart_set", k, self.PUT_WORDS), None)
                    self.note("enum_rejected", role="cart_set", arg=k, value=v)
            self.note("cart_set_failed", err=err)
        return ok

    def cart_add(self, P, L):
        if "cart_add" not in self.roles and "cart_set" in self.roles:
            want = list(self.cart.items()) + [(L.id, P.qty)]
            if not self.set_cart(want):
                return False
            self.cart = dict(want)
            self.carted[L.id] = P
            return True
        r = self.roles.get("cart_add")
        if not r:
            return False
        args = {r["id_arg"]: L.id}
        q = P.qty
        qa = r.get("quantity_arg")
        if qa:
            mx = self.arg_schema("cart_add", qa).get("maximum")
            if is_int(mx) and mx < q:
                q = mx
            args[qa] = q
        md = r.get("mode")
        if isinstance(md, dict) and md.get("arg") and md.get("add") is not None:
            args[md["arg"]] = md["add"]
        filled = {}
        try:
            filled = self.need_enums("cart_add", args, self.PUT_WORDS, self.DROP_WORDS)
            if filled:
                args.update(filled)
                self.note("enum_filled", role="cart_add", args=filled)
        except Exception as exc:
            self.note("enum_error", error=str(exc)[:160])
        obs, _l, err, ok = self.call(r["tool"], args)
        if not ok:
            for k, v in filled.items():
                if str(k).lower() in str(err or "").lower() or str(v).lower() in str(err or "").lower():
                    self.enum_bad.add(("cart_add", k, v))
                    self.enum_pick.pop(("cart_add", k, self.PUT_WORDS), None)
                    self.note("enum_rejected", role="cart_add", arg=k, value=v)
        if ok:
            self.cart[L.id] = q
            self.carted[L.id] = P
            return True
        self.note("cart_add_failed", id=L.id, err=err)
        return False

    def cart_remove(self, lid):
        if "cart_add" not in self.roles and "cart_set" in self.roles:
            want = [(x, q) for x, q in self.cart.items() if x != lid]
            if not self.set_cart(want):
                return False
            self.cart.pop(lid, None)
            self.carted.pop(lid, None)
            return True
        r = self.roles.get("cart_add")
        if not r:
            return False
        args = {r["id_arg"]: lid}
        qa = r.get("quantity_arg")
        md = r.get("mode")
        try:
            dropped = self.need_enums("cart_add", args, self.DROP_WORDS, self.PUT_WORDS)
            if dropped:
                args.update(dropped)
                self.note("enum_filled", role="cart_remove", args=dropped)
        except Exception:
            pass
        if isinstance(md, dict) and md.get("arg") and md.get("remove") is not None:
            args[md["arg"]] = md["remove"]
            if qa:
                args[qa] = self.cart.get(lid) or 1
        elif qa:
            mn = self.arg_schema("cart_add", qa).get("minimum")
            args[qa] = mn if (is_int(mn) and mn > 0) else 0
        obs, _l, err, ok = self.call(r["tool"], args)
        if ok:
            self.cart.pop(lid, None)
            self.carted.pop(lid, None)
        return ok

    # ---- E4.8 finishing an order
    def finish_order(self):
        co = self.roles.get("checkout")
        if not co:
            return False
        if self.procedure and self.cart:
            v = self.gate("pre-order", "about to buy the basket shown")
            if v == "ran" and self.pending() > 0 and self.steps_left() >= 2:
                self.gate("pre-order", "about to buy the basket shown")
            if v in ("ran", "noanswer") and self.confirm_first and self.sig() != self.confirmed:
                if self.confirm_basket():
                    v = "ran"
            if v == "ran":
                if self.handle_lines() or self.price_stock_change():
                    self.note("order_abandoned", reason="change after gate")
                    return False
        args = self.finish_args("checkout")
        obs, lines, err, ok = self.call(co["tool"], args)
        if self.finished:
            self.note("ordered", basket=[list(x) for x in self.sig()])
            return True
        self.note("checkout_refused", result=cut(obs, 300), lines=[str(x)[:200] for x in lines[:3]], err=err)
        return False

    # ---- E4.9 declining
    def standin_order(self, reason=""):
        got = {}
        for P in self.products:
            L = P.chosen
            if L is None:
                L = self.near_miss_pick(P)
            if L is None:
                L = self.plain_pick(P)
            if L is not None:
                got[P] = L
        self.note("standin_attempt", reason=str(reason)[:70], products=len(self.products),
                  found=len(got), pools=[len(P.pool) for P in self.products],
                  priced=[sum(1 for x in P.pool.values() if x.price is not None)
                          for P in self.products])
        if not got or len(got) != len(self.products):
            return False
        keep = {L.id for L in got.values()}
        stale = [lid for lid in self.cart if lid not in keep]
        adds = [P for P, L in got.items() if L.id not in self.cart]
        if self.steps_left() < len(stale) + len(adds) + 1:
            return False
        for lid in stale:
            self.cart_remove(lid)
        for P, L in got.items():
            P.chosen = L
            if L.id not in self.cart:
                self.cart_add(P, L)
        if not self.cart:
            return False
        ok = self.finish_order()
        self.note("standin_result", ordered=bool(ok), ids=sorted(self.cart))
        return ok

    def decline(self, reason=""):
        d = self.roles.get("decline")
        if not d:
            return False
        if not self.standin_tried and self.products and self.roles.get("checkout") \
                and not (bool(self.fallback) and self.fallback.get("buy_near_miss") is False) \
                and not FORBID_RX.search(self.request or ""):
            self.standin_tried = True
            try:
                if self.standin_order(reason):
                    return True
            except Exception as exc:
                self.note("standin_error", error=str(exc)[:160])
        if not self.decline_gated and self.procedure:
            self.decline_gated = True
            self.gate("pre-decline", "about to finish with no purchase")
        args = self.finish_args("decline")
        ra = d.get("remark_arg")
        sch = self.schemas.get(d["tool"]) or {}
        req = sch.get("required") if isinstance(sch.get("required"), list) else []
        if ra and ra in req:
            mx = self.arg_schema("decline", ra).get("maxLength")
            args[ra] = self.trim("Nothing on offer satisfies all of the requirements in the request.", mx if is_int(mx) else None)
        self.note("decline", reason=reason or self.decline_reason)
        self.call(d["tool"], args)
        if not self.finished and not self.decline_retried and self.steps_left() >= 1:
            self.decline_retried = True
            self.note("decline_refused")
            if self.handle_lines() and self.steps_left() >= len(self.products) + 2:
                self.choose_and_finish()
            if not self.finished:
                self.call(d["tool"], args)
        return self.finished


    # ---------------------------------------------------------------- E5 run and loop
    def reserve(self):
        if self.reduced:
            return len(self.products) + 3 + max(self.held(), 1 if self.confirm_first else 0)
        two = 2 if ("facts" in self.roles and "compare" not in self.roles) else 0
        proc = min(4, sum(1 for s in self.procedure if s["moment"] != "start") + (1 if self.confirm_first else 0))
        return len(self.products) + 6 + two + proc

    # ---- W1 the dependencies the code already holds: dependent -> provider
    def dep_graph(self):
        need = {}
        for a, h in self.fit_pairs():
            if a not in need:
                need[a] = h
        for c in self.conditions:
            S = c.get("source")
            for b in c.get("branches") or []:
                t = b.get("target")
                if t is not None and t is not S and t not in need:
                    need[t] = S
        return need

    # ---- W2 the planner designs the schedule
    def workflow_plan(self, again=False):
        if len(self.products) < 2 or self.time_left() < MARGIN + 25 or self.spent():
            return None
        need = self.dep_graph()
        blocks = []
        for P in self.products:
            blocks.append({"product": P.name, "quantity": P.qty, "category": P.category,
                           "requirements": P.texts(), "open_hints": P.hints,
                           "depends_on": need[P].name if P in need else None})
        user = "REQUEST:\n%s\n\nPRODUCTS:\n%s\n\nSteps available: %d\nQuestions available: %s\nMoney for all lines together: %s" % (
            self.request, json.dumps(blocks, ensure_ascii=False, default=str),
            max(0, self.steps_left() - self.reserve()),
            "none stated" if self.allowance is None else str(max(0, self.allowance - self.sent - self.held())),
            "none stated" if self.cap is None else fmt_num(self.cap))
        if again:
            worked = [s.name() for s in self.slots if s.state in ("done", "skipped")]
            user += "\n\nAlready worked and not to be scheduled again: %s" % json.dumps(worked, ensure_ascii=False)
        return self.llm.json_call([M_PLAN, M_FAST], P_WORKFLOW, user, 700)

    # ---- W3 dependency order is settled in code, the planner may only order within it
    def dep_sort(self, seq, need):
        out = []
        pend = list(seq)
        guard = 0
        while pend and guard <= len(seq) + 2:
            guard += 1
            moved = False
            for P in list(pend):
                H = need.get(P)
                if H is None or H not in pend:
                    out.append(P)
                    pend.remove(P)
                    moved = True
            if not moved:
                break
        return out + pend

    def build_slots(self, again=False):
        need = self.dep_graph()
        spec = {}
        order = []
        ans = self.workflow_plan(again=again)
        if isinstance(ans, dict):
            if isinstance(ans.get("slots"), list):
                for e in ans["slots"]:
                    if isinstance(e, dict):
                        P = self.name_best(txt(e.get("product")))
                        if P is not None and id(P) not in spec:
                            spec[id(P)] = e
            if isinstance(ans.get("order"), list):
                for nm in ans["order"]:
                    P = self.name_best(txt(nm))
                    if P is not None and P not in order:
                        order.append(P)
            w = txt(ans.get("replan_when")).lower()
            if not self.replan_at and w in REPLAN_WHEN:
                self.replan_at = w
            try:
                got = self.read_helpers(ans)
                if got:
                    self.helpers.update(got)
                    self.note("helpers", declared=[(h["name"], h["returns"], h["needs"])
                                                   for h in got.values()])
            except Exception as exc:
                self.note("helper_error", error=str(exc)[:160])
        seq = [P for P in order if P in self.products]
        for P in self.products:
            if P not in seq:
                seq.append(P)
        seq = self.dep_sort(seq, need)
        keep = {}
        for sl in self.slots:
            if sl.state in ("done", "skipped"):
                keep[id(sl.P)] = sl
        fresh = []
        for P in seq:
            if id(P) in keep:
                fresh.append(keep[id(P)])
                continue
            sl = Slot(P, spec.get(id(P)))
            if P in need and not txt(sl.needs_from):
                sl.needs_from = need[P].name
            fresh.append(sl)
        self.slots = fresh
        self.share_asks()
        self.note("workflow", order=[sl.name() for sl in self.slots],
                  asks=[sl.asks for sl in self.slots],
                  steps_share=[sl.steps_share for sl in self.slots],
                  spend_share=[sl.spend_share for sl in self.slots],
                  deps=[(sl.name(), sl.needs_from, sl.take, sl.as_prop) for sl in self.slots if sl.needs_from],
                  replan_when=self.replan_at or "never", planned=bool(ans), again=again)

    # ---- W4 the question allowance is divided between the open slots
    def share_asks(self):
        if self.allowance is not None:
            left = max(0, self.allowance - self.sent - self.held())
        elif "message" in self.roles:
            left = max(0, 3 - self.sent)
        else:
            left = 0
        openers = [sl for sl in self.slots if sl.state == "open"]
        if not openers:
            return
        named = []
        for sl in openers:
            v = sl.asks
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0:
                sl.asks = int(v)
                named.append(sl)
            else:
                sl.asks = None
        tot = sum(sl.asks for sl in named)
        if named and tot > 0:
            for sl in named:
                sl.asks = int(round(left * (sl.asks / float(tot))))
        elif named:
            for sl in named:
                sl.asks = 0
        blank = [sl for sl in openers if sl.asks is None]
        if blank:
            spread = max(0, left - sum(sl.asks for sl in openers if sl.asks is not None))
            base = spread // len(blank)
            extra = spread - base * len(blank)
            for i, sl in enumerate(blank):
                sl.asks = base + (1 if i < extra else 0)
        over = sum(sl.asks for sl in openers) - left
        guard = 0
        while over > 0 and guard <= 60:
            guard += 1
            for sl in sorted(openers, key=lambda x: -x.asks):
                if over <= 0:
                    break
                if sl.asks > 0:
                    sl.asks -= 1
                    over -= 1

    # ---- W5 the step share for one slot, unspent room falling through to the rest
    def slot_steps(self, slot):
        free = self.steps_left() - self.reserve()
        if self.allowance is not None:
            free -= max(0, self.allowance - self.sent - self.held())
        elif "message" in self.roles:
            free -= max(0, 3 - self.sent)
        if free <= 0:
            return 0
        openers = [sl for sl in self.slots if sl.state == "open"]
        if len(openers) <= 1:
            return max(2, free)
        sh = slot.steps_share
        ok = isinstance(sh, (int, float)) and not isinstance(sh, bool) and 0 < sh <= 1
        if ok:
            rest = 0.0
            for sl in openers:
                v = sl.steps_share
                if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1:
                    rest += float(v)
            if rest > 0:
                want = int(free * (float(sh) / rest))
                return max(2, min(want, free - 2 * (len(openers) - 1)))
        return max(2, int(free / float(len(openers))))

    # ---- W6 the money cut: what this slot may still spend once the committed picks are paid for
    def set_ceiling(self, slot):
        self.slot_ceiling = None
        slot.ceiling = None
        if self.cap is None:
            return
        committed = 0.0
        for sl in self.slots:
            if sl.state == "done" and sl.pick is not None and sl.pick.price is not None:
                committed += (sl.pick.price or 0) * max(1, sl.P.qty)
        others = [sl for sl in self.slots if sl.state == "open" and sl is not slot]
        floor = 0.0
        for sl in others:
            v = sl.spend_share
            if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v < 1:
                floor += self.cap * float(v) * 0.5
            else:
                floor += self.cap * SPEND_FLOOR
        room = self.cap - committed - floor
        v = slot.spend_share
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1 and room > 0:
            room = min(room, self.cap * float(v) * 1.5)
        if room > 0:
            self.slot_ceiling = room / float(max(1, slot.P.qty))
            slot.ceiling = self.slot_ceiling

    def provisional(self, P):
        bl = self.rank(P, self.buyable(P))
        return bl[0] if bl else None

    # ---- W7 the provider's choice becomes this slot's requirement, before the slot is worked
    def inject_dep(self, slot):
        if slot.bound or not txt(slot.needs_from):
            return
        H = self.name_best(txt(slot.needs_from))
        if H is None or H is slot.P:
            return
        hs = next((sl for sl in self.slots if sl.P is H), None)
        if hs is None or hs.state != "done":
            return
        HL = hs.pick if hs.pick is not None else self.provisional(H)
        if HL is None:
            return
        if not slot.P.pool and self.steps_left() > self.reserve() + 2 and self.time_left() > MARGIN + 20:
            self.search(slot.P, query=slot.P.terms[0] if slot.P.terms else slot.P.category,
                        pages=1, floor=self.reserve())
        before = len(slot.P.reqs)
        if any(a is slot.P and h is H for a, h in self.fit_pairs()):
            self.bind_fit(slot.P, H, HL)
        if any(any(b.get("target") is slot.P for b in (c.get("branches") or [])) for c in self.conditions):
            picks = {}
            for sl in self.slots:
                if sl.state == "done" and sl.pick is not None:
                    picks[sl.P] = sl.pick
            picks[H] = HL
            self.apply_conditions(slot.P, picks)
        carried = None
        want = " ".join(txt(slot.take).replace("_", " ").split()).lower()
        if want:
            keys = list(HL.props)
            k = best_by_sim(want, keys, 0.5) if keys else None
            val = txt(HL.props.get(k)) if k is not None else ""
            if not val and near(want, "price") and HL.price is not None:
                val = fmt_num(HL.price)
            if val:
                nm = " ".join((txt(slot.as_prop) or want).replace("_", " ").split()).lower()
                if not any(near(r.prop, nm) for r in slot.P.reqs):
                    slot.P.reqs.append(Req(nm, "is", val, None, "dep"))
                    carried = (nm, val)
        slot.bound = True
        self.note("slot_dep", product=slot.P.name, provider=H.name, id=HL.id, take=slot.take,
                  given=slot.as_prop, carried=carried, reqs_before=before, reqs_after=len(slot.P.reqs))

    # ---- W8 this slot's questions, charged to this slot
    def slot_questions(self, slot):
        k = slot.asks if isinstance(slot.asks, int) else 0
        P = slot.P
        while slot.asked < k and self.q_ok() and self.steps_left() > self.reserve() and self.time_left() > 75:
            pl = [(Q, prop) for Q, prop in self.plan(max(1, k - slot.asked)) if Q is P]
            if not pl:
                break
            prop = pl[0][1]
            if any(sim(r.prop, prop) >= 0.67 for r in P.reqs):
                break
            slot.asked += 1
            n0 = len(P.reqs)
            rd = self.ask(P, prop)
            if isinstance(rd, dict) and str(rd.get("kind") or "").lower() == "split" and self.q_ok() \
                    and self.steps_left() > self.reserve():
                rd = self.ask(P, prop, narrowed=True)
            self.apply_kept(P, prop, rd)
            self.handle_lines()
            if len(P.reqs) != n0 and slot.asked < k:
                self.planner_list = None

    # ---- W9 one slot, worked under the cuts the controller handed it
    def run_slot(self, slot):
        P = slot.P
        self.inject_dep(slot)
        try:
            self.helper_prepare(slot)
        except Exception as exc:
            self.note("helper_error", error=str(exc)[:160])
        steps = self.slot_steps(slot)
        if steps <= 0:
            return False
        if slot.bound and self.provisional(P) is not None:
            steps = max(2, steps // 3)
        self.slot_now = slot
        self.set_ceiling(slot)
        before = self.used
        try:
            self.explore(P, max(2, steps // 2))
            used = self.used - before
            self.warm_up(P, max(0, steps - used))
            self.slot_questions(slot)
            if slot.asked:
                self.judge(P, [L for L in self.candidates(P) if L.opened])
        finally:
            slot.steps = self.used - before
            slot.pick = self.provisional(P)
            self.slot_now = None
            self.slot_ceiling = None
        self.note("slot_done", product=P.name, steps=slot.steps, asked=slot.asked, ceiling=slot.ceiling,
                  pick=(slot.pick.id if slot.pick is not None else None), pool=len(P.pool))
        return True

    # ---- W10 the planner names the moment to think again; the controller detects it
    def state_mark(self):
        return (self.cap, self.allowance, len(self.products), len(self.conditions), len(self.fit_rules))

    def replan_ready(self):
        if self.change_mark is not None and self.state_mark() != self.change_mark and "change" not in self.replan_fired:
            self.replan_fired.add("change")
            return "on_change"
        w = self.replan_at
        if not w or w == "never" or w in self.replan_fired:
            return ""
        if w == "after_needs":
            if any(r.unresolved() for P in self.products for r in P.reqs):
                return ""
            if any(sl.P.hints for sl in self.slots if sl.state == "open"):
                return ""
        elif w == "after_refs":
            refs = [r for P in self.products for r in P.reqs if r.ref is not None]
            if not refs or any(r.unresolved() for r in refs):
                return ""
        elif w == "after_pick":
            if not any(sl.state == "done" and sl.pick is not None for sl in self.slots):
                return ""
        else:
            return ""
        self.replan_fired.add(w)
        return w

    def replan(self, why):
        if self.replans >= 2 or len(self.products) < 2 or self.spent():
            return
        if self.time_left() < MARGIN + 30:
            return
        if not any(sl.state == "open" for sl in self.slots):
            return
        self.replans += 1
        self.note("replan", why=why, done=[sl.name() for sl in self.slots if sl.state == "done"],
                  open=[sl.name() for sl in self.slots if sl.state == "open"])
        self.build_slots(again=True)

    # ---- W11 the controller
    def controller(self):
        self.build_slots()
        self.change_mark = self.state_mark()
        guard = 0
        limit = 2 * len(self.products) + 4
        while guard <= limit:
            guard += 1
            nxt = next((sl for sl in self.slots if sl.state == "open"), None)
            if nxt is None:
                break
            if self.steps_left() <= self.reserve() or self.time_left() < MARGIN + 10:
                cut = [sl.name() for sl in self.slots if sl.state == "open"]
                for sl in self.slots:
                    if sl.state == "open":
                        sl.state = "skipped"
                self.note("slots_cut", left=cut)
                break
            nxt.state = "doing"
            try:
                self.run_slot(nxt)
            finally:
                nxt.state = "done"
            sig = self.replan_ready()
            if sig:
                self.replan(sig)
            self.change_mark = self.state_mark()

    def run(self):
        try:
            if not self.setup():
                try:
                    self.forced_finish()
                except Exception as e:
                    self.note("error", err=repr(e)[:300])
                return self.trace
            self.work()
        except Exception as e:
            self.note("error", err=repr(e)[:300])
        if not self.finished:
            try:
                self.forced_finish()
            except Exception as e:
                self.note("error", err=repr(e)[:300])
        self.note("done", spend=round(self.llm.spend, 4), calls=self.llm.calls, steps=self.used)
        return self.trace

    def work(self):
        n = len(self.products)
        Q = (self.allowance if self.allowance is not None else 3) - self.held()
        if "message" not in self.roles or Q < 0:
            Q = 0
        self.ref_floor = Q + (1 if Q > 0 else 0) + self.reserve() + 2 * n
        try:
            for P in self.products:
                for r in list(P.reqs):
                    if r.unresolved():
                        self.follow(P, r, 0)
        finally:
            self.ref_floor = 0
        if any(s["moment"] == "start" for s in self.procedure):
            self.gate("start", "about to begin, searching the store first")
            self.handle_lines()
            if self.allowance is not None:
                Q = max(0, min(Q, self.allowance - self.sent - self.held()))
        self.controller()
        if Q > 0 and self.q_ok():
            plan = self.plan(Q)
            while plan:
                P, prop = plan.pop(0)
                if not self.q_ok() or self.steps_left() <= self.reserve():
                    break
                if any(sim(r.prop, prop) >= 0.67 for r in P.reqs):
                    continue
                n0 = len(P.reqs)
                rd = self.ask(P, prop)
                if isinstance(rd, dict) and str(rd.get("kind") or "").lower() == "split" and self.q_ok() \
                        and self.steps_left() > self.reserve():
                    rd = self.ask(P, prop, narrowed=True)
                self.apply_kept(P, prop, rd)
                self.handle_lines()
                if len(P.reqs) != n0 and plan:
                    if self.allowance is None:
                        k = len(plan)
                    else:
                        k = min(len(plan), self.allowance - self.sent - self.held())
                    plan = self.plan(k)
            while self.q_ok() and self.steps_left() > self.reserve() + 1 and self.time_left() > 75:
                pl = self.plan(1)
                if not pl:
                    break
                P, prop = pl[0]
                rd = self.ask(P, prop)
                self.apply_kept(P, prop, rd)
                self.handle_lines()
        for P in self.products:
            for r in list(P.reqs):
                if r.unresolved():
                    self.ref_floor = self.reserve() + 2
                    try:
                        self.follow(P, r, 0)
                    finally:
                        self.ref_floor = 0
        self.judge_all()
        order = sorted(enumerate(self.products),
                       key=lambda t: (1 if t[1].fits() else 0, -len(t[1].pool), t[0]))
        for k, (_i, P) in enumerate(order):
            room = self.steps_left() - self.reserve()
            if room <= 0:
                break
            allow = max(2, int(room / float(n - k)))
            if self.provisional(P) is not None and not any(r.unresolved() for r in P.reqs):
                allow = min(allow, 3)
            self.explore(P, allow, final=True)
        self.choose_and_finish()

    # ---- C4 warm-up
    def warm_up(self, P, allowance):
        if self.steps_left() <= self.reserve() + 2 or "message" not in self.roles:
            return
        floor = max((self.steps_left() - allowance) if allowance else 0, self.reserve() + 2)
        target = 12 if self.batch_reading() else 5
        def spares():
            out = [(i, L) for i, L in enumerate(list(P.pool.values()))
                   if not L.opened and self.cat_ok(P, L) and L.stock is not False]
            out.sort(key=lambda t: (mark_rank(t[1].mark), price_key(t[1]), t[0]))
            return [L for _, L in out]

        while self.steps_left() > floor and self.time_left() > 65 and not self.spent():
            opened_n = sum(1 for L in P.pool.values() if L.opened and self.cat_ok(P, L))
            want = target - opened_n
            if want <= 0:
                break
            sp = spares()
            if len(sp) < 2 and P.pages_fetched < len(P.terms) + 1:
                n0 = len(P.pool)
                self.search(P, query=P.category, pages=1, floor=floor)
                sp = spares()
                if len(P.pool) == n0 and not sp:
                    break
            if not sp:
                break
            if self.batch_reading():
                chunk = min(want, self.compare_limit)
            elif self.thin:
                chunk = min(want, 4)
            else:
                chunk = min(want, 1)
            got = self.open_batch(P, [L.id for L in sp[:chunk]], floor=floor)
            if not got:
                break
            self.judge(P, [L for L in self.candidates(P) if L.opened])

    def value_queries(self, P):
        out = []
        for r in P.active():
            if r.op not in PLAIN_VALUE_OPS or not r.value or r.unresolved():
                continue
            v = str(r.value).split("|")[0].strip()
            if not v or len(v.split()) > 4:
                continue
            if _J_NOT.search(v) or _J_ANYBUT.search(v):
                continue
            if not re.search(r"[a-z]", v, re.I):
                continue
            for q in ("%s %s" % (P.category, v), v):
                q = " ".join(q.split())
                if q and q not in out:
                    out.append(q)
        return out

    # ---- C5 exploring
    def explore(self, P, allowance, final=False):
        stop = self.steps_left() - allowance

        def wf():
            return max(stop, self.reserve())

        queries = list(P.terms)
        for q in self.value_queries(P):
            if q not in queries:
                queries.append(q)
        qi = min(P.pages_fetched, len(queries))
        want = 4 if self.best else 2
        kw = "keyword_search|"
        while self.steps_left() > wf() and self.time_left() > 55 and not self.spent():
            self.handle_lines()
            self.screen(P, self.candidates(P))
            cands = self.candidates(P)
            unopened = [L for L in cands if not L.opened]
            self.judge(P, [L for L in cands if L.opened])
            sure = P.fits()
            if not self.cheapest and len(sure) >= want:
                return
            if self.cheapest and sure:
                best = sure[0].price
                cheaper = [L for L in unopened if L.price is not None and (best is None or L.price < best)]
                if not cheaper:
                    fs = self.roles.get("field_search")
                    if best is not None and fs and fs.get("max_price_arg") and P.swept != best \
                            and self.steps_left() > wf() + 2:
                        P.swept = best
                        self.search(P, fields={"category": P.category, "max_price": max(0, best - 0.01)},
                                    pages=3, floor=wf())
                        continue
                    if P.capped and not P.narrowed and self.steps_left() > wf() + 2:
                        P.narrowed = True
                        vals = [r.value.split("|")[0].strip() for r in P.active()
                                if r.op in PLAIN_VALUE_OPS and r.value]
                        if vals:
                            for q in (" ".join(vals + [P.category]), " ".join(vals)):
                                if self.steps_left() <= wf() + 1:
                                    break
                                self.search(P, query=q, pages=2, floor=wf())
                            continue
                    stepped = False
                    for q in queries:
                        c = P.cursors.get(kw + q)
                        if c and not c["done"]:
                            if self.steps_left() > wf() + 2:
                                u0 = self.used
                                self.search(P, query=q, pages=1, floor=wf() + 1)
                                stepped = self.used > u0
                            break
                    if stepped:
                        continue
                    if P.pages_fetched >= min(len(queries), 2):
                        return
                unopened = cheaper
            if unopened:
                if self.batch_reading():
                    nb = self.compare_limit
                elif self.thin:
                    nb = 4
                else:
                    nb = 2
                got = self.open_batch(P, [L.id for L in unopened[:nb]], floor=wf())
                if not got:
                    return
                continue
            needs = [r for r in P.active() if r.origin == "shopper" and r.op in PLAIN_VALUE_OPS and r.value]
            if needs and not P.need_done:
                P.need_done = True
                first_need = needs[0].value.split("|")[0].strip()
                others = [r for r in P.active() if r.origin != "shopper" and r.op in PLAIN_VALUE_OPS and r.value]
                qs = []
                if others:
                    qs.append("%s %s %s" % (P.category, others[0].value.split("|")[0].strip(), first_need))
                qs.append("%s %s" % (P.category, first_need))
                qs.append(first_need)
                for q in qs:
                    if self.steps_left() <= wf():
                        break
                    self.search(P, query=q, pages=3, floor=wf())
                continue
            if qi < len(queries):
                q = queries[qi]
                qi += 1
                got_new = self.search(P, query=q, pages=6 if len(self.products) == 1 else 3, floor=wf())
                if got_new == 0 and qi >= len(queries):
                    P.exhausted = True
                continue
            fs = self.roles.get("field_search")
            if fs:
                fields = {"category": P.category}
                ceil = self.slot_ceiling if (self.slot_ceiling is not None
                                             and self.slot_now is not None and self.slot_now.P is P) else None
                if ceil is not None:
                    fields["max_price"] = ceil
                elif self.cap is not None:
                    fields["max_price"] = self.cap / float(max(P.qty, 1))
                c = P.cursors.get("field_search|" + json.dumps(fields, sort_keys=True, default=str))
                if not c or not c["done"]:
                    self.search(P, fields=fields, pages=2, floor=wf())
                    continue
            progressed = False
            for q in queries[:2]:
                c = P.cursors.get(kw + q)
                if not c or not c["done"]:
                    if self.search(P, query=q, pages=2, floor=wf()) > 0:
                        progressed = True
                        break
            if progressed:
                continue
            if not P.suggested and self.steps_left() > wf() + 1:
                P.suggested = True
                new = self.suggest(P)
                if new:
                    P.terms.extend(new)
                    queries.extend(new)
                    continue
            P.exhausted = True
            return

    # ---- C6 query suggestions
    def shortfalls(self, P, L):
        v = P.verdicts.get(L.id)
        if v:
            n = len(v.get("failed") or ()) + len(v.get("unsure") or ())
        else:
            n = self.plain_shortfalls(P, L)
        if self.fallback and self.fallback.get("count_stock_budget"):
            if L.stock is False:
                n += 1
            if self.cap is not None and L.price is not None and (L.price * P.qty > self.cap + TOL):
                n += 1
            if P.unit_cap is not None and L.price is not None and L.price > P.unit_cap + TOL:
                n += 1
        return n

    def near_miss_pick(self, P):
        rows = list(self.candidates(P, lax=lax_fallback(getattr(self, 'fallback', None))))
        if not rows:
            rows = list(P.pool.values())
        scored = [(self.shortfalls(P, L), L.price, L.id, L) for L in rows if L.price is not None]
        if not scored:
            return None
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        best = scored[0][0]
        same = [t for t in scored if t[0] == best]
        tol = (self.fallback or {}).get("tolerance")
        room = same
        if tol:
            ceil = same[0][1] * (1.0 + float(tol) / 100.0) + TOL
            room = [t for t in same if t[1] <= ceil] or same
        self.note("near_miss", product=P.name, id=room[0][3].id, price=room[0][1],
                  short=best, at_rank=len(same), within_tolerance=len(room), tolerance=tol)
        return room[0][3]

    def plain_misses(self, P, L):
        facts = dict(L.props or {})
        blob = " ".join(str(x or "") for x in ((L.title or ""), (L.desc or "")[:400], (L.notes or "")[:200]))
        bad = 0
        for r in P.reqs:
            if r.ref is not None and r.unresolved():
                continue
            val = str(r.value or "")
            if r.op and r.op not in ("is",):
                val = ("%s %s" % (r.op, val)).strip()
            got = plain_holds(r.prop, val, facts, blob)
            if got is False:
                bad += 1
        return bad

    def plain_shortfalls(self, P, L):
        facts = dict(L.props or {})
        blob = " ".join(str(x or "") for x in ((L.title or ""), (L.desc or "")[:400], (L.notes or "")[:200]))
        bad = 0
        for r in P.reqs:
            if r.ref is not None and r.unresolved():
                continue
            val = str(r.value or "")
            if r.op and r.op not in ("is",):
                val = ("%s %s" % (r.op, val)).strip()
            if plain_holds(r.prop, val, facts, blob) is not True:
                bad += 1
        return bad

    def plain_pick(self, P):
        rows = [L for L in self.candidates(P, lax=lax_fallback(getattr(self, 'fallback', None))) if L.price is not None]
        if not rows:
            rows = [L for L in P.pool.values() if L.price is not None]
        if not rows:
            return None
        scored = []
        for L in rows:
            gone = 1 if L.stock is False else 0
            over = 0
            if P.unit_cap is not None and L.price > P.unit_cap + TOL:
                over = 1
            if self.cap is not None and L.price * P.qty > self.cap + TOL:
                over = 1
            scored.append((self.plain_misses(P, L) + gone + over, L.price, L.id, L))
        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        best = scored[0]
        self.note("plain_pick", product=P.name, id=best[3].id, price=best[1], misses=best[0],
                  of=len(scored))
        return best[3]

    def say_near_miss(self, got):
        if not self.msg_ok():
            return
        bits = []
        for P, L in got.items():
            who = (L.title or L.id)[:60]
            bits.append("%s: %s%s" % (P.name, who, "" if L.price is None else " at %s" % fmt_num(L.price)))
        self.send_message("Nothing in the shop meets every requirement for %s. The closest I can get is %s."
                          % (", ".join(P.name for P in got), "; ".join(bits)))

    def suggest(self, P):
        shown = sorted(P.pool.values(), key=lambda L: (0 if L.opened else 1, L.id))[:14]
        fids = {L.id for L in P.fits(include_unsure=True)}
        items = [{"title": (L.title or "")[:110], "fits": (L.id in fids) if L.opened else None} for L in shown]
        tried = sorted({s[len("keyword_search|"):] for s in P.cursors if s.startswith("keyword_search|")})
        r = self.roles.get("keyword_search") or {}
        mx = self.max_len("keyword_search", r.get("query_arg"))
        texts = P.texts()
        user = "Product: %s (category: %s)\nRequirements:\n%s\nQueries already run: %s\nListings seen: %s" % (
            P.name, P.category, "\n".join("- " + t for t in texts) if texts else "none",
            json.dumps(tried, ensure_ascii=False), json.dumps(items, ensure_ascii=False))
        if mx is not None:
            user += "\nQuery length limit: %d characters" % mx
        ans = self.llm.json_call([M_FAST, M_BACKUP], P_QUERIES, user, 700)
        out = []
        if isinstance(ans, dict) and isinstance(ans.get("queries"), list):
            seen = {norm(q) for q in tried} | {norm(t) for t in P.terms}
            for q in ans["queries"]:
                s = txt(q)
                if not s or len(s) > 60 or norm(s) in seen:
                    continue
                seen.add(norm(s))
                out.append(s)
                if len(out) >= 5:
                    break
        self.note("suggested", product=P.name, queries=out)
        return out

    # ---- C7 ranking by the best measure
    def rank(self, P, lst):
        if not self.best or len(lst) < 2:
            return lst
        ids = [L.id for L in lst]
        c = self.rank_cache.get(P.idx)
        if c is not None and c[0] == ids:
            order = c[1]
        else:
            rows = [{"id": L.id, "title": (L.title or "")[:120], "price": L.price, "properties": L.props,
                     "description": (L.desc or "")[:300]} for L in lst[:12]]
            user = "Measure: %s\nProduct: %s\nListings: %s" % (self.best, P.name, cut(rows, 9000))
            ans = self.llm.json_call([M_PLAN, M_BACKUP], P_RANK, user, 600)
            order = [str(x) for x in ans.get("order")] if isinstance(ans, dict) and isinstance(ans.get("order"), list) else []
            self.note("ranked", product=P.name, order=order[:12])
            if order:
                self.rank_cache[P.idx] = (ids, order)
        pos = {}
        for i, x in enumerate(order):
            pos[x] = i
        named = sorted([L for L in lst if L.id in pos], key=lambda L: pos[L.id])
        return named + [L for L in lst if L.id not in pos]

    # ---------------------------------------------------------------- E6 choosing and finishing
    def buyable(self, P, table=None):
        fl = [L for L in P.fits(cheapest=self.cheapest) if L.stock is not False]
        if table is None:
            table = self.field_table(P)
        return [L for L in fl if not self.field_failures(P, L, table)]

    def name_product(self, name):
        if not name or not self.products:
            return None
        best, bs = None, -1.0
        for P in self.products:
            s = max(sim(name, P.name), sim(name, P.category))
            if s > bs:
                best, bs = P, s
        return best if bs >= 0.2 else None

    def fit_pairs(self):
        out = []
        if len(self.products) < 2:
            return out
        for fr in self.fit_rules:
            an = txt(fr.get("accessory"))
            hn = txt(fr.get("host"))
            if not an or not hn:
                continue
            a = h = None
            sa = sh = -1.0
            for P in self.products:
                s = sim(an, P.name)
                if s > sa:
                    a, sa = P, s
                s = sim(hn, P.name)
                if s > sh:
                    h, sh = P, s
            if a is not h and sa >= 0.2 and sh >= 0.2 and not any(x is a and y is h for x, y in out):
                out.append((a, h))
        return out

    # ---- E6.4 binding a fit
    def bind_fit(self, A, H, hostL):
        if getattr(A, "bound_host", None) == hostL.id and any(r.origin == "fit" for r in A.reqs):
            self.note("fit_cached", accessory=A.name, host=hostL.id)
            return
        A.bound_host = hostL.id
        htxt = " ".join(x for x in ((hostL.title or "")[:120], (hostL.desc or "")[:300], (hostL.notes or "")[:150]) if x)
        A.reqs = [r for r in A.reqs if r.origin != "fit"]
        A.reqs.append(Req("compatibility", "is",
                          "must suit the listing picked for the %s: %s" % (H.name, htxt), None, "fit"))
        titled = set()
        prefixes = set()
        for Q in self.products:
            for L in list(Q.pool.values()):
                cs = CODE_RE.findall(L.title or "")
                titled.update(cs)
                if Q is A:
                    prefixes.update(lead_caps(c) for c in cs)
        look = []
        for L in self.candidates(A):
            if not L.opened or (A.verdicts.get(L.id) or {}).get("failed"):
                continue
            for c in CODE_RE.findall((L.desc or "") + " " + (L.notes or "")):
                if c not in titled and c not in look and lead_caps(c) in prefixes:
                    look.append(c)
        heads = set()
        for Q in self.products:
            if Q is H:
                continue
            for L in list(Q.pool.values()):
                heads.update(SERIES_RE.findall(L.title or ""))
        own = set(SERIES_RE.findall(hostL.title or ""))
        host_txt = " ".join([hostL.desc or "", hostL.notes or ""] + ["%s %s" % (k, v) for k, v in hostL.props.items()])
        series = []
        hlead = {lead_caps(h) for h in heads}
        for w in SHORT_CODE_RE.findall(host_txt):
            if w in own or w in series:
                continue
            if w in heads or lead_caps(w) in hlead:
                series.append(w)
        lk = series[:2] + [c for c in look if c not in series][:3]
        for c in lk:
            if self.steps_left() <= self.reserve() + 2 or self.time_left() < 55:
                break
            q = ("%s %s" % (c, A.category)) if c in series[:2] else c
            self.search(A, query=q, pages=1, floor=self.reserve())
        A.series = series[:2]
        if series and self.steps_left() > self.reserve() + 2:
            un = [L for L in self.candidates(A) if not L.opened and any(h in series for h in SERIES_RE.findall(L.title or ""))]
            lim = self.compare_limit if self.batch_reading() else 3
            if un:
                self.open_batch(A, [L.id for L in un[:lim]], floor=self.reserve() + 1)
        self.judge(A, [L for L in self.candidates(A) if L.opened])
        if series:
            fr = A.fit_req()
            ft = fr.text() if fr is not None else None
            for L in self.candidates(A):
                v = A.verdicts.get(L.id)
                if L.opened and v is not None and ft and any(h in series for h in SERIES_RE.findall(L.title or "")):
                    for k in ("failed", "unsure", "weak"):
                        v[k].discard(ft)
        self.note("fit_bound", accessory=A.name, host=hostL.id, series=series, lookups=lk)

    # ---- E6.5 reading conditions
    def read_conditions(self, raw):
        out = []
        if not isinstance(raw, list):
            return out
        for e in raw:
            if not isinstance(e, dict):
                continue
            ip = e.get("if")
            if not isinstance(ip, dict) or not ip.get("property"):
                continue
            src = self.name_product(txt(ip.get("product")))
            v = ip.get("values")
            if isinstance(v, list):
                vals = [str(x) for x in v if x is not None and str(x).strip()]
            elif isinstance(v, (str, int, float)) and not isinstance(v, bool) and str(v).strip():
                vals = [str(v)]
            else:
                vals = []
            brs = []
            for bk in ("then", "otherwise"):
                b = e.get(bk)
                if isinstance(b, dict) and b.get("property") and b.get("value"):
                    tgt = self.name_product(txt(b.get("product")))
                    if tgt is not None and tgt is not src:
                        brs.append({"when": bk == "then", "target": tgt, "prop": txt(b.get("property")),
                                    "op": txt(b.get("operator")) or "is", "value": txt(b.get("value"))})
            if src is None or not vals or not brs:
                continue
            out.append({"source": src, "prop": txt(ip.get("property")), "values": vals, "branches": brs})
            for b in brs:
                T = b["target"]
                for part in b["value"].split("|"):
                    part = part.strip()
                    if part and len(part) <= 40 and not any(norm(part) in norm(t) for t in T.terms):
                        T.terms.append((part + " " + T.noun)[:60])
        if out:
            self.note("conditions", conditions=[{"source": c["source"].name, "prop": c["prop"], "values": c["values"],
                                                 "branches": [(b["when"], b["target"].name, b["prop"], b["op"], b["value"])
                                                              for b in c["branches"]]} for c in out])
        return out

    # ---- E6.6 testing a condition
    def cond_test(self, c, L):
        key = (id(c), L.id, json.dumps(L.props, sort_keys=True, default=str))
        if key in self.ccache:
            return self.ccache[key]
        own = [str(v) for k, v in L.props.items()
               if near(str(k).replace("_", " "), c["prop"]) or sim(str(k).replace("_", " "), c["prop"]) >= 0.5]
        res = None
        if own:
            ow = words(" ".join(own))
            vws = [w for w in (words(v) for v in c["values"]) if w]
            if any(w <= ow for w in vws):
                res = True
            elif not any(w & ow for w in vws):
                res = False
        if res is None:
            user = "Property: %s\nValues: %s\nListing:\n%s" % (
                c["prop"], json.dumps(c["values"], ensure_ascii=False),
                json.dumps(summary(L, 2500), ensure_ascii=False, default=str))
            ans = self.llm.json_call([M_FAST, M_BACKUP], P_COND, user, 300)
            if isinstance(ans, dict) and isinstance(ans.get("holds"), bool):
                res = ans["holds"]
        self.ccache[key] = res
        self.note("condition_test", prop=c["prop"], id=L.id, result=res)
        return res

    # ---- E6.7 applying conditions
    def apply_conditions(self, D, picks):
        conds = [c for c in self.conditions if any(b["target"] is D for b in c["branches"])]
        if not conds:
            return
        new = []
        for c in conds:
            pk = picks.get(c["source"])
            if pk is None:
                continue
            res = self.cond_test(c, pk)
            if res is None:
                continue
            for b in c["branches"]:
                if b["target"] is D and b["when"] == res:
                    new.append(Req(b["prop"], b["op"] or "is", b["value"], None, "condition"))
        cur = sorted(r.text() for r in D.reqs if r.origin == "condition")
        if sorted(r.text() for r in new) == cur:
            return
        D.reqs = [r for r in D.reqs if r.origin != "condition"] + new
        self.note("conditions_applied", product=D.name, reqs=[r.text() for r in new])
        self.judge(D, [L for L in self.candidates(D) if L.opened])
        if not self.buyable(D) and self.steps_left() - self.reserve() >= 3 and self.time_left() > 65:
            D.exhausted = False
            self.explore(D, self.steps_left() - self.reserve(), final=True)

    # ---- E6.8 choosing and finishing
    def choose_and_finish(self):
        if self.finished:
            return
        self.handle_lines()
        self.judge_all()
        picks = {}
        pairs = self.fit_pairs()
        acc_host = {}
        for a, h in pairs:
            if a not in acc_host:
                acc_host[a] = h
        hosts = [h for _a, h in pairs]
        sources = [c["source"] for c in self.conditions]
        deps = []
        for c in self.conditions:
            for b in c["branches"]:
                if b["target"] not in deps:
                    deps.append(b["target"])
        free = [P for P in self.products if P not in acc_host and P not in deps]
        firsts = []
        for P in free:
            bl = self.rank(P, self.buyable(P))
            if bl:
                firsts.append((P, bl[0]))
        parallel([(lambda P=P, L=L: self.verify(P, L)) for P, L in firsts])
        order = sorted(self.products, key=lambda P: (0 if P in free else 1,
                                                     0 if (P in hosts or P in sources) else 1,
                                                     self.products.index(P)))
        for P in order:
            if P in acc_host and acc_host[P] in picks:
                self.bind_fit(P, acc_host[P], picks[acc_host[P]])
            if P in deps:
                self.apply_conditions(P, picks)
            tries = 0
            for L in list(self.rank(P, self.buyable(P))):
                if tries >= 3 or self.time_left() < MARGIN:
                    break
                tries += 1
                try:
                    if not self.helper_gate(P, L):
                        continue
                except Exception as exc:
                    self.note("helper_error", error=str(exc)[:160])
                if self.verify(P, L):
                    picks[P] = L
                    self.helper_picks[P.name] = L
                    break
            if P not in picks:
                bl = self.rank(P, self.buyable(P))
                if bl and (tries >= 3 or self.time_left() < MARGIN):
                    picks[P] = bl[0]
                    self.note("unverified_pick", product=P.name, id=bl[0].id)
            if P not in picks and P in acc_host:
                H = acc_host[P]
                for HL in self.buyable(H)[:3]:
                    if (picks.get(H) is not None and HL.id == picks[H].id) or self.time_left() < MARGIN:
                        continue
                    self.bind_fit(P, H, HL)
                    bl = self.buyable(P)
                    if bl:
                        picks[H] = HL
                        picks[P] = bl[0]
                        break
            if P not in picks and P in deps:
                for c in self.conditions:
                    if P in picks:
                        break
                    S = c["source"]
                    if S not in picks or not any(b["target"] is P for b in c["branches"]):
                        continue
                    if any(h is S and a in picks for a, h in pairs) or self.time_left() < MARGIN:
                        continue
                    cur = self.cond_test(c, picks[S])
                    prev = picks[S]
                    for SL in self.buyable(S)[:4]:
                        if SL.id == prev.id:
                            continue
                        if self.cond_test(c, SL) == cur:
                            continue
                        picks[S] = SL
                        self.apply_conditions(P, picks)
                        bl = self.buyable(P)
                        if bl and self.verify(S, SL):
                            picks[P] = bl[0]
                            break
                        picks[S] = prev
                        self.apply_conditions(P, picks)
        missing = [P for P in self.products if P not in picks]
        if missing and not self.last_chance_done and self.time_left() > 65 and not self.spent():
            self.last_chance_done = True
            self.reduced = True
            try:
                for P in missing:
                    if self.steps_left() - self.reserve() >= 3:
                        P.exhausted = False
                        self.explore(P, self.steps_left() - self.reserve(), final=True)
            finally:
                self.reduced = False
            if any(self.buyable(P) for P in missing):
                return self.choose_and_finish()
        if missing and not self.audit_done and self.time_left() > 70 and not self.spent():
            self.audit_done = True
            if self.audit(missing) and all(self.buyable(P) for P in missing):
                return self.choose_and_finish()
        forbid = bool(self.fallback) and self.fallback.get("buy_near_miss") is False
        if missing and not forbid and self.time_left() > MARGIN:
            got = {}
            for P in missing:
                if lax_fallback(getattr(self, 'fallback', None)):
                    L = self.near_miss_pick(P)
                    if L is None:
                        L = self.plain_pick(P)
                else:
                    L = self.plain_pick(P)
                    if L is None:
                        L = self.near_miss_pick(P)
                if L is not None:
                    got[P] = L
            if len(got) == len(missing):
                for P, L in got.items():
                    picks[P] = L
                try:
                    self.say_near_miss(got)
                except Exception as exc:
                    self.note("near_miss_say_error", error=str(exc)[:200])
                self.note("near_miss_filled", products=[P.name for P in got],
                          ids=[L.id for L in got.values()])
                missing = []
        if missing:
            self.decline_reason = "no pick for " + ", ".join("%s (pool %d)" % (P.name, len(P.pool)) for P in missing)
            self.note("no_pick", reason=self.decline_reason)
            self.decline(self.decline_reason)
            return
        if self.cap is not None:
            def total():
                return sum((picks[P].price or 0) * P.qty for P in self.products)
            moves = 0
            while total() > self.cap + TOL and moves < 10:
                moved = False
                for P in sorted(self.products, key=lambda P: -((picks[P].price or 0) * P.qty)):
                    fl = [L for L in P.fits(cheapest=True) if L.stock is not False]
                    if P.unit_cap is not None:
                        fl = [L for L in fl if L.price is None or L.price <= P.unit_cap + TOL]
                    cur = picks[P].price
                    cheaper = [L for L in fl if L.price is not None
                               and (cur is None or L.price < cur - TOL)]
                    if cheaper:
                        cheaper.sort(key=lambda L: L.price)
                        picks[P] = cheaper[0]
                        moves += 1
                        moved = True
                        self.note("stepped_down", product=P.name, id=cheaper[0].id,
                                  price=cheaper[0].price, was=cur, total=total(), cap=self.cap)
                        break
                if not moved:
                    break
            if total() > self.cap + TOL:
                self.note("over_budget", total=total(), cap=self.cap)
                swapped = False
                if not (bool(self.fallback) and self.fallback.get("buy_near_miss") is False):
                    for P in self.products:
                        alt = self.plain_pick(P)
                        if alt is not None and alt.price is not None and \
                                (picks[P].price is None or alt.price < picks[P].price - TOL):
                            picks[P] = alt
                            swapped = True
                    self.note("over_budget_best_effort", total=total(), cap=self.cap, swapped=swapped)
                if total() > self.cap + TOL and not swapped:
                    fb = self.fallback or {}
                    if fb.get("buy_near_miss") is not False and fb.get("count_stock_budget"):
                        self.note("over_budget_allowed", total=total(), cap=self.cap,
                                  tolerance=fb.get("tolerance"))
                    else:
                        self.decline_reason = "even the cheapest fitting set costs more than the cap"
                        self.decline(self.decline_reason)
                        return
        for P in self.products:
            P.chosen = picks[P]
        self.note("picks", picks=[(P.name, P.chosen.id, P.chosen.price) for P in self.products])
        self.place_order(0)

    # ---- E6.9 pre-decline audit
    def audit(self, missing):
        learned = False
        self.reduced = True
        try:
            for P in missing:
                if self.time_left() < 65:
                    break
                src = []
                for i, L in enumerate(list(P.pool.values())):
                    if not self.cat_ok(P, L) or L.stock is False:
                        continue
                    if P.unit_cap is not None and L.price is not None and L.price > P.unit_cap + TOL:
                        continue
                    if self.cap is not None and L.price is not None and L.price * P.qty > self.cap + TOL:
                        continue
                    src.append((i, L))
                src.sort(key=lambda t: (0 if t[1].opened else 1, price_key(t[1]), t[0]))
                cards = []
                for _i, L in src[:40]:
                    c = {"id": L.id, "title": (L.title or "")[:110], "price": L.price, "read": L.opened}
                    if L.opened:
                        v = P.verdicts.get(L.id)
                        if v:
                            c["failed"] = sorted(v["failed"])[:3]
                            c["unsure"] = sorted(v["unsure"])[:3]
                        c["properties"] = dict(list(L.props.items())[:16])
                        if L.notes:
                            c["notes"] = L.notes[:200]
                    else:
                        c["text"] = (L.desc or "")[:160]
                    cards.append(c)
                if not cards:
                    continue
                tried = sorted({s[len("keyword_search|"):] for s in P.cursors if s.startswith("keyword_search|")})
                texts = P.texts()
                budget = ("Budget for all lines together: %s" % fmt_num(self.cap)) if self.cap is not None else "Budget: none stated"
                user = "Product: %s (category: %s, quantity %d)\n%s\nRequirements:\n%s\nQueries already run: %s\nListings seen:\n%s" % (
                    P.name, P.category, P.qty, budget, "\n".join("- " + t for t in texts) if texts else "none",
                    json.dumps(tried, ensure_ascii=False), cut(cards, 14000))
                ans = self.llm.json_call([M_PLAN, M_FAST], P_AUDIT, user, 1500)
                if not isinstance(ans, dict):
                    continue
                self.note("audit", product=P.name, answer=cut(ans, 300))
                fa = ans.get("fits_after_all")
                for x in ([str(i) for i in fa] if isinstance(fa, list) else [])[:2]:
                    L = P.pool.get(x)
                    if L is None or not L.opened or self.field_failures(P, L):
                        continue
                    if self.verify(P, L, strict=True):
                        P.verdicts[L.id] = empty_verdict()
                        P.judged_for[L.id] = set(texts)
                        learned = True
                un = ans.get("unopened")
                un = [str(i) for i in un] if isinstance(un, list) else []
                un = [i for i in un if i in P.pool and not P.pool[i].opened][:4]
                if un and self.steps_left() - self.reserve() >= 1:
                    if self.open_batch(P, un, floor=self.reserve()):
                        learned = True
                qs = ans.get("queries")
                qs = [txt(q) for q in qs if txt(q)] if isinstance(qs, list) else []
                tn = {norm(t) for t in tried}
                for q in qs[:2]:
                    if self.steps_left() - self.reserve() < 3 or self.time_left() < 60:
                        break
                    if norm(q) in tn:
                        continue
                    if self.search(P, query=q[:60], pages=1, floor=self.reserve()) == 0:
                        continue
                    self.screen(P, self.candidates(P))
                    ids = [L.id for L in self.candidates(P) if not L.opened and L.mark != "no"][:2]
                    if ids and self.open_batch(P, ids, floor=self.reserve()):
                        learned = True
                self.judge(P, [L for L in self.candidates(P) if L.opened])
        finally:
            self.reduced = False
        return learned

    # ---- E6.10 placing the order
    def place_order(self, attempt):
        self.order_rounds += 1
        if attempt > 2 or self.finished or self.order_rounds > 8:
            return
        pick_ids = {P.chosen.id for P in self.products if P.chosen is not None}
        for lid in list(self.cart):
            if lid not in pick_ids:
                self.cart_remove(lid)
        for P in self.products:
            L = P.chosen
            if L is not None and L.id not in self.cart:
                if not self.cart_add(P, L):
                    v = P.verdicts.get(L.id)
                    if v is None:
                        v = empty_verdict()
                        P.verdicts[L.id] = v
                    v["failed"].add("cart: the store refused to add this listing")
                    P.chosen = None
        if any(P.chosen is None for P in self.products):
            return self.choose_and_finish()
        P0 = self.products[0]
        if self.steps_left() > 2 + self.pending() and P0.chosen is not None:
            self.open_batch(P0, [P0.chosen.id])
        changed = self.handle_lines()
        for P in self.products:
            L = P.chosen
            if L is not None and L.id not in P.judged_for:
                self.judge(P, [L])
                if (P.verdicts.get(L.id) or {}).get("failed"):
                    self.note("pick_failed_after_look", product=P.name, id=L.id)
                    changed = True
        if changed or self.price_stock_change():
            self.note("change_before_checkout")
            self.judge_all()
            self.choose_again()
            if self.finished:
                return
            return self.place_order(attempt + 1)
        self.finish_order()
        if self.finished:
            return
        if not self.handle_lines():
            self.refresh_cart()
            self.handle_lines()
        self.choose_again()
        if not self.finished:
            self.place_order(attempt + 1)

    # ---- E6.11 price or stock change
    def price_stock_change(self):
        lax = lax_fallback(getattr(self, 'fallback', None))
        picks = [(P, P.chosen) for P in self.products if P.chosen is not None]
        for P, L in picks:
            if L.stock is False and not lax:
                return True
            if P.unit_cap is not None and L.price is not None and L.price > P.unit_cap + TOL:
                return True
            if self.cap is not None and L.price is not None and not lax:
                tot = sum((M.price or 0) * Q.qty for Q, M in picks)
                if tot > self.cap + TOL:
                    return True
        return False

    # ---- E6.12 choosing again
    def choose_again(self):
        for P in self.products:
            self.judge(P, [L for L in self.candidates(P) if L.opened])
            sure = [L for L in P.fits() if L.stock is not False]
            if not sure or self.cheapest:
                P.need_done = False
                P.swept = None
                allow = max(2, self.steps_left() - self.reserve())
                if allow >= 3 and self.time_left() > 65:
                    self.explore(P, allow, final=True)
        for P in self.products:
            fl = [L for L in P.fits(cheapest=self.cheapest) if L.stock is not False]
            if not fl:
                P.chosen = None
            elif self.cheapest:
                P.chosen = fl[0]
            elif not (P.chosen is not None and P.chosen.id in {L.id for L in fl}):
                P.chosen = fl[0]
        pm = {P: P.chosen for P in self.products if P.chosen is not None}
        deps = []
        for c in self.conditions:
            for b in c["branches"]:
                if b["target"] not in deps:
                    deps.append(b["target"])
        for D in deps:
            self.apply_conditions(D, pm)
            bl = self.buyable(D)
            if D.chosen is None or D.chosen.id not in {L.id for L in bl}:
                D.chosen = bl[0] if bl else None
        if any(P.chosen is None for P in self.products):
            return self.choose_and_finish()
        if self.cap is not None and sum((P.chosen.price or 0) * P.qty for P in self.products) > self.cap + TOL:
            return self.choose_and_finish()

    # ---- E6.13 forced finish
    def forced_finish(self):
        if self.finished:
            return
        lax = lax_fallback(getattr(self, 'fallback', None))
        for P in self.products:
            if P.chosen is None or (P.chosen.stock is False and not lax):
                bl = self.buyable(P)
                if bl:
                    P.chosen = bl[0]
                elif lax:
                    try:
                        P.chosen = P.chosen or self.near_miss_pick(P)
                    except Exception:
                        pass
                else:
                    P.chosen = None
        ready = bool(self.products) and all(P.chosen is not None for P in self.products)
        if ready and not lax and self.cap is not None and \
                sum((P.chosen.price or 0) * P.qty for P in self.products) > self.cap + TOL:
            ready = False
        if ready:
            pick_ids = {P.chosen.id for P in self.products}
            stale = [lid for lid in self.cart if lid not in pick_ids]
            adds = [P for P in self.products if P.chosen.id not in self.cart]
            if self.steps_left() >= len(stale) + len(adds) + 1:
                for lid in stale:
                    self.cart_remove(lid)
                for P in adds:
                    self.cart_add(P, P.chosen)
                if self.finish_order():
                    return
        if not self.finished:
            self.decline("forced")


# ---------------------------------------------------------------- A10 entry point
def agent_main(problem_data):
    ep = Episode(problem_data)
    return ep.run()


__all__ = ["agent_main"]
