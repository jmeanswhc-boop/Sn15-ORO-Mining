"""Wren: buys for a single client within an ORO merchant-site episode.

A small model walks the site using its tools. Python keeps the sheet (wants,
demands, cap, articles seen), plans what to ask the client, and vetoes wasted
moves. A larger model reads the client's note twice (an intake pass, then a
proof pass auditing it against the note's own words), judges the trolley ahead
of payment, and weighs the evidence ahead of a bow-out.
"""
from __future__ import annotations

import json
import random
import re
import time
from typing import Any

from src.agent.proxy_client import ProxyClient

SMALL = "anthropic/claude-haiku-5.5"
BIG = "anthropic/claude-sonnet-5.5"
BACKUP = "openai/gpt-6-luna"

TURN_CAP = 42
CALL_CAP = 62
COST_SOFT = 0.055
COST_HARD = 0.08
BIG_MARGIN = 0.02
BACKOFF = (4, 10, 20, 35)      # pauses before resending a failed model call; the episode has 300 s, so at most 45 s sleep
BACKOFF_SUM = 45
TIME_SOFT = 195              # seconds after which the walker stops and Wren forces a finish (the sandbox kills the run at 300)
TIME_HARD = 240              # seconds after which no further model call is made
TIME_KILL = 300              # the sandbox ends the run here
LIST_SIZE = 3                # properties one note may name while the client tolerates lists
MIX_SHARE = 1                # properties named per want inside a note that spans several wants
EARLY_CHAINS = 3             # chain rounds the post-interview follow may spend
HUNT_STEPS = 8               # searches Wren runs by itself before a cheapest payment or a bow-out
QUIET_WORD = 4
LINK_ROWS = 3
LINK_FAMILIES = 12
LINK_SEARCHES = 3
LATE_STEPS = 4               # moves that must remain for a question put right before paying
SHOW_LEN = 450
OBS_CAP = 15000
LOG_CAP = 150000
CHORE_ROUNDS = 3               # shop calls the chore gate may spend before one checkout or ending
SELF_PAGES = 6                 # later search pages Wren reads by itself for a cheapest / max / min pick
CHEAP_PEEKS = 3                # cheaper rows of a basket line's kind whose attributes Wren reads before a cheapest checkout
RANK_KEYS = 3
RANK_SHARE = 0.0
RANK_GAP = 0.34
PEEK_EACH = 6
JUNK_KEY_RX = r"\b(?:condition|photos?|images?|links?|url|video|warrant\w*|barcodes?|ean|upc|asin|codes?|ids?|sku|models?|variants?|options?)\b"
GRAB_BAG_RX = r"^(?!.*\b(?:shapes?|kinds?|styles?|types?)\b).*\b(?:misc\w*|extras?|accessor\w*|includ\w*|descriptions?|specifications?|specs?|highlights?|functions?|features?)\b"
MODEL_CODE_RX = r"[A-Z]{1}[A-Za-z]{,5}[ -]?(?:[0-9]{3,6}|[0-9]{1,4}[xX]{1,2})"
PART_RX = r"[A-Z]{1,5}[0-9]{0,3}-?[0-9]{3,7}"
SERIES_RX = r"\b([A-Za-z]{1,5}[0-9]{1,3})-[0-9]{2,7}\b"
ASK_WORDS_RX = r"\b(?:one|single|each|first|at a time|many|several|much|time|lists?|once|separately|split|shorter|questions?|asking|ask|lot|turn|all|too|slow\w*|again|repeat\w*)\b"
COLOUR_RX = r"\b(black|white|red|blue|green|yellow|orange|purple|pink|brown|gr[ae]y|silver|gold|beige|navy|teal|turquoise|ivory|cream|tan|maroon|burgundy|violet|lilac|lavender|olive|khaki|charcoal|bronze|copper|clear|transparent|multicolou?r|rainbow|natural)\b"
BOUND_RX = r"\b(?:except|or|at least|up to|within|between|more|less|above|below|over|under|at most|unlike|other than|no|not)\b"

# one try per call (consult() resends): a hung call costs 50 s, not minutes, out of the 300 s
_brain = ProxyClient(timeout=45, max_retries=1, retry_delay=1)
_shop = ProxyClient(timeout=95, max_retries=3, retry_delay=1)


CHARTER = """Your name is Wren. You buy on behalf of one client inside a merchant site, and tool calls are your only hands: nothing happens unless you call a site tool. Grading is binary. Credit comes solely from a correct finish: a paid order holding every want's article at that want's unit count, each article passing all demands standing at paying time, total inside the spending cap; or, when the site truly stocks nothing that passes everything, a bow-out through the no-buy tool phrased the way the client wanted.

HOW THE SITE BEHAVES
- One tool call = one move, and moves are rationed; thinking costs none. Tool names, parameter names and reply layouts are peculiar to this site, so study each tool's description plus schema and send parameters spelled exactly as declared, with permitted values only.
- Searching may be crude: a department filter, keyword matching which looks at the leading words alone, five-hit pages, a cursor or offset for more. Check the declared hit total against what you have actualy seen and fetch further pages (with the exact cursor the site returned) whenever completeness matters. Short value-first keyword strings do best (wanted value, then the article noun). Zero hits from a filtered query usually mean a wrong filter value or wrong filter parameter: resend with the tag spelled like the hits spell it, or through another parameter.
- Hits are abridged. Sites carve up an article's record in their own fashion: one view may hold the fields (properties), another just price, stock and blurb, and a multi-article view may hold either. Wren's sheet names the field-bearing tool once known. Decide every property demand from those fields, so read them on every article you might buy; a single call showing several articles saves moves whenever it shows what matters. Never re-open an article if nothing changed after your last read of it.
- Notes to the client are sent by Wren's interviewer on your behalf; you never contact the client directly. Replies land in this conversation and are copied onto the sheet.
- The client may also demand errands en route: an approval, an announcement, something to read or verify beforehand. The sheet lists them. Errands due right before paying or before bowing out are done by Wren at the moment you call the pay tool or the no-buy tool, so do not wait for them and do not bow out because of them; any other errand you perform with site tools (questions about articles go via the interviewer).

WHEN AN ARTICLE PASSES A DEMAND
- The article kind is fixed by its department tag: only a tag matching the want's article as a whole qualifies, never an adjacent one.
- A property demand passes when the article's field under that property carries the value. Spelling of keys is irrelevent (spaces, underscores, camelCase, prefixes), and one entry inside a field value joined by commas or slashes suffices. Values are compared word by word: case, plural form and everyday synonyms do not matter, and a field value containing the demanded words passes (a darker shade still carries the colour word; an added qualifier still carries the base word). A field value built from other words fails (an acronym or code differs from what it abbreviates; two colour names sharing a word remain two colours). Convert units before comparing numbers, bounds are inclusive, and a stated range passes only if it sits wholly inside the bound.
- A field with a key naming the demanded property (same name or everyday synonym) rules: if it states something else, the article fails no matter what the title or other keys say. A key naming some other property, even a close relative (the colour of a component is not the article's colour), rules nothing. Wren appends [proof] lines after each opened article, listing the keys that name a demanded property.
- Lacking a field for the property, the demanded value written under a different field (a features or styles list including it), within the title, or within the blurb passes. A number measuring something else does not (mass is not volume), while a bare figure lacking any word that says what it measures passes nothing.
- Should the department tag itself carry the demanded value (an audience or season inside the tag), the tag satisfies that demand.
- A demand may cite another article instead of a value ("an X matching what <shop> sells with <property>", "X at most N away from the X of ..."). Settle it before anything else: track that article down (search its department, brand, shop, or key words), open its full detail view via the single-article or multi-article tool (a hit alone will not do), and if that view forwards you to yet another article for the property, track down and open it too; the final view carries the value. The purchase is valid solely once each view of that chain has been opened. Where the demand wants some value differing from the cited article's, it bars only articles carrying precisely that value, and on its own it never justifies a bow-out.
- If an article you mean to buy forwards you to another view for more detail, read it before buying.
- When one want must suit another (an add-on for a host device), choose both together from the articles' own text (a compatibility list, a series or model designation printed on both views). If no add-on passing its own demands suits the host you chose, switch to a different host that passes its demands, which such an add-on does suit; never bow out until that switch was tried.
- A demand that the article suit a named model code (a device the client already owns) passes only when the article's own text lists that code as something it suits or works alongside; Wren flags a trolley line whose text never mentions the code.
- A per-unit price ceiling set for one article kind binds every unit of that kind. A conditional demand binds only when its trigger holds on the article chosen for the other want: read that article's field first, then decide.
- Honour exclusions and closed value lists. Wishes (phrased "ideally", "would be lovely") merely break ties.
- Never buy an article shown sold out; the order total (unit price times count, summed over lines) must not pass the cap, equality is fine.

SELECTING
- When the client wants the cheapest option (the smallest sum, the lowest tag), the order has to carry the lowest-priced article among ALL articles that pass. Collect the entire passing set (paging through it: hits rarely arrive in price order, and Wren pulls the rest of a short result's pages unaided), test candidates from cheapest upward, keep the first that passes fully. When a search accepts a maximum price, exploit it: list the department beneath your current best candidate's price so that only cheaper ones remain.
- When the client wants a measure's extreme (its most or its least), compare all passing articles across that measure and keep the extreme; an article silent on the measure is out of the running.
- Otherwise whichever article passes everything is right: no moves spent hunting for a nicer one.
- Trolley each want as soon as an opened article passes all its demands (under a cheapest / max / min rule: once the passing set is compared), then move on. Sites tend to sell out a trolleyed line shortly after, and an early trolley leaves moves for a replacement.
- Several wants need one trolley line each, at that want's count.

CHANGES MID-RUN
- Client or site may alter things: a fresh cap, a demand added or withdrawn, a price shift, a sell-out. They surface inside tool replies, often as a refused payment; a notice counts whichever tool's reply displays it, and the sheet carries the demands currently standing. After a refusal the run goes on: apply each change, mend the trolley, pay again.
- Re-read the record of each chosen article shortly before paying, since price and stock drift silently.

FINISHING
- Make the trolley contain only the chosen lines, nothing beyond them (take out anything else), view it once, then pay with a call sent on its own.
- Bow out only after a genuine search turned up no article passing everything. Being short of moves proves no such thing: in that case pay for the best trolley you have. Word the bow-out as the client asked; if a remark is called for, keep it terse and name an article only if the client asked to be offered the nearest alternative.
- Every reply must contain a tool call, or several. Prose stays terse: what is known, then your coming move."""


INTAKE = """You convert a client's shopping note into a work order for a buying agent. The note is chatty: anecdotes, metaphors, place names, people, deadlines, moods and jokes sit between the genuine demands. Keep only what a merchant can be held to, and hang each demand on the correct article. Read the note together with the site's tool list, then give back a single compact JSON object, unindented whose keys are:

"wants": an entry for every distinct article the client wants bought, each {"what": the article kind in the client's wording, "label": the site's department tag for it if a tool enumerates tags (verbatim), failing that, whichever department the note assigns this article ("a coat from the rainwear section" -> "rainwear"), else "", "count": units of this single article (integer; 1 when unsaid; a pair, set or tub sold as one article is 1), "needs": firm demands on this article, each written "property: constraint", carrying the precise value, bound, exclusion or closed option list, "cap_each": a price ceiling the note puts on each unit of this article kind alone ("keep each <kind> at or under N", "no more than N on each <kind>"), numeric, else null}.
  How a demand is named: take the client's own word for the property ("meant for a <place> occasion" -> "occasion: <place>"; "a 30 cm <part>" -> "<part> length: 30 cm"); never substitute a neighbouring property for it. A value folded into the article phrase ("two <colour> <articles>", "a <flavour>-flavoured <article>", "<level> <measure> <article>") is a demand on that property all the same. Hang a demand only on the article whose phrase holds it; where several articles share a sentence, a demand belongs to the article named right before it, and a sentence shaped "whatever <article> you choose, the <other article> must ..." is a compatibility pair rather than a demand.
  A demand whose value has to be read off another article's detail view is ONE demand carrying the full description of that other article ("<property>: same as the <property> on the article <shop> sells with <property> <value> plus <property> <value>, or whichever view that one forwards to"); the words that single out that other article (shop, kind, the "with ..." properties) remain inside this single demand and never become demands on the article being bought. A demand that the value be anything but that view's value is shaped the same way, using "unlike".
  Phrasings to decode (all of them demands): "pick the <property> from <A> and <B> only" and "<property> is <A> or <B>" are closed lists ("<property>: <A> or <B>"); "give <A> a wide berth for the <property>", "mustn't be <A>", "can't be <A>", "isn't <A>", "anything but <A>" are exclusions ("<property>: not <A>"); "<measure> of N or lower / at most N / no higher than N" and "N or bigger / at least N / N and up" are bounds; "has to fit the <thing> I already have, a <code>" is a demand "compatibility: must suit <thing> model <code>" on that article, not a pair. A phrase "<A> or <B>" that merely spells out one department's two nouns (the way the site names the department) is ONE want rather than a choice.
  Price, the spending cap, stock and the unit count never go into needs. Chatter is not a demand: who uses it, where it stands, what broke earlier, the occasion, weather, deadlines, craft or workshop metaphors, measuring habits, flatmates, towns, delivery wishes and views on past buys are context, unless a sentence states a property value or bound outright. Words forming part of the article's name (a season, audience or use welded onto the noun) never become demands; they live in "what". An occasion or use that the note turns into one of the article's own properties ("for the <activity> occasion", "the occasion is <activity>") IS a demand on that property, unless the department tag already carries it; the event, person or place the article is bought for stays context. An article named by one phrase with "and" or "&" in it (a single "I want ..." carrying one count, often a department name) is ONE want; split only where the note gives each article its own count or its own demands.
  Wants arise solely from the ordering sentences ("fetch me", "could you get", "N x <department>", "first ..., third ..., last, 1 ..."), wich may sit far apart in the note. A thing spoken of as owned already, borrowed, broken, being replaced, standing in the room, used alongside the purchase, or to be bought from some other shop is never a want, and its traits (material, colour, size, brand, age, where it sits) never turn into demands on the article actualy being bought. Figures inside chatter - guests, ages, years, dates, clock times, how many share a home, a room or gap measured without any bound attached - are neither counts nor limits; the unit count is only the figure standing right before the article phrase. A sentence repeating a demand given elsewhere adds nothing. Metaphors borrowing material, vessel, craft, garden or weather words carry no property value at all.
"limit": {"amount": the top sum the client will pay, numeric, in the note's own unit, else null, "unit": that unit as a word, else null, "scope": "order" when one sum covers everything, "each" for a per-article sum, null without a cap}.
"stocked_only": false only when the note plainly tolerates unavailable articles, else true ("let it drift on by", "steer clear of", "skip" said of sold-out stock all mean true).
"choose": "cheapest" when the note wants the lowest tag or the smallest total ("cheapest one wins", "the combo with the smallest total", "a smaller final total wins", "best bargain wins"), "max" or "min" when it wants the extreme of a measure ("of those that match, pick whichever tops the lot for <measure>" is "max"), otherwise "any". Moaning about money is no selection rule.
"choose_by": the measure behind "max" / "min", else "".
"message_cap": the number of notes the client lets the agent send ("you've N messages", "message me N times at most"), integer, null if unsaid.
"step_cap": the number of moves (tool calls) the note permits ("you've N tool calls to play with", "tool calls are capped at N"), integer, null if unsaid.
"pointers": the needs (verbatim) whose value has to come off another article's view before anything else, else [].
"pairs": [{"part": want number, "host": want number}] for each want the note says must be compatible with, suit or work with ANOTHER want (whichever one the agent selects); [] if none.
"conditional": [{"item": want number, "mode": "only_if" or "except_if", "about": the want number whose chosen article decides, "test": "property: value or values", "need": "property: constraint"}] covering each demand that binds only if, or except if, some property on another chosen article carries a given value; [] if none. A conditional demand is NOT repeated inside that want's "needs".
"fallback": {"announce": true when the note says to declare that nothing matches, "offer_nearest": true when it wants the nearest article named or offered if nothing passes (say, the article failing the fewest checks), "nearest_rule": the note's wording for how that nearest article is chosen, else "", "slack_pct": a percentage of price leeway granted over the cheapest such article, else null, "stay_quiet": true when naming alternatives is forbidden, false where it is asked for, null when unsaid}.
"open": [{"item": want number, "property": the property}] for every property the note itself declares as not mattering ("any way of storing it works", "I can make any size work", "not fussed about the colour", "it could end up in any room"); such a sentence is the reverse of a hint and never appears under "hints"; [] if none.
"hints": [{"item": want number, "words": at most 8 words of the note hinting at a preference the client holds but has not stated (a person whose taste decides, something left to you, looks or setting left open, who wears or uses it, a worry about how it travels, stows or is sized, "I'd rather be asked than guess", "ask me whatever you need", "I'll tell you straight", "she'd tell me which kind once I got to that bit"), "property": the property it most probably concerns, else ""}]; [] if none. Such a hint usually means a firm unwritten preference exists: never drop it.
"check_first": matters the note explicitly tells the agent to ask the client about before selecting ("ask me which <property> to look through", "ask me about the <part>"); [] if none.
"chores": [{"what": an errand the note demands beyond finding and buying (an approval, announcing or showing something, verifying with the client, reading something beforehand, using a named tool), under 15 words, "stage": "pick" (before articles are selected), "order" (right before paying), "end" (right before bowing out), "messages": the client notes it costs, 0 if none}]; [] if no such errand. Demands, the cap, stock, note and move limits, and the manner of ending when nothing passes are not errands.
"nice": what the client would merely like; [].
"manner": each instruction on working method or on the manner of ending, under 12 words apiece; [].
"tools": per job, the name of the tool that does it, verbatim, else null if none does: "find" (search), "page" (one article's fields or properties; given two single-article tools, pick the one promising fields, specifications or properties in its description), "deal" (one article's live price and stock, where that is a separate tool from "page"), "side" (several articles in full at once), "talk" (note to the client), "basket" (alters the trolley), "basket_view" (displays the trolley), "checkout" (pays), "walk_away" (ends with no purchase).
"checkout_args": the fixed parameters under which the "checkout" tool pays (for a mode-driven tool, that mode), else {}.
"walk_away_args": the fixed parameters under which the "walk_away" tool ends with no purchase, else {}.
"browse": how a whole department gets listed, if any tool manages that: {"tool": that tool's name verbatim, "category": its parameter for the department tag, "max_price": its parameter for a price ceiling, else null, "fixed": whatever further parameters put it into listing mode, else {}}; null where no tool lists departments.
"""


SIFT = """Sort the sentences of a client's shopping note into three bins, so a buying agent reads only what it can act on. A sentence extends until the following full stop, question mark or exclamation mark; copy each one verbatim, punctuation included, and place it in exactly one bin. Give back a single JSON object and no other text:
{"orders": [sentences that order an article ("fetch me", "could you get", "N x ..."), or state a demand on one (a property value, a bound, an exclusion, a closed option list, a unit count, a per-unit ceiling, a compatibility or conditional rule, a value to copy off some other article's page), or fix the spending cap, the stock rule, or the selection rule (cheapest, most, least)],
"process": [sentences on how the agent must work: how many notes or moves it may spend, what to ask or verify beforehand, what to announce, show or approve, the manner of closing if no article passes, which property the client leaves open, which property the client hints at yet leaves unsaid ("I'd rather be asked", "she decides the look")],
"chatter": [every other sentence: the client's job, home, people, guests, ages, occasions, dates, clock times, weather, moods, apologies, thanks, metaphors, things owned already, broken, replaced or bought elsewhere, how or where the article will be used]}
A sentence mixing an order with chatter goes under "orders"; one mixing working method with chatter goes under "process". Figures inside chatter stay chatter, whatever they count. Leave no sentence out and invent none."""


PROOF = """A buying agent drafted a work order from a client's note. Audit that draft against the note, phrase by phrase. The note blends real demands with anecdotes, metaphors and chatter, so the draft may have (a) invented a demand from story text, (b) copied part of the article's name as a demand, (c) hung a demand on the wrong article, (d) misread a value, bound or unit, or (e) overlooked a demand, a per-unit price ceiling, a count, a conditional rule or a compatibility pair, or (f) listed as a want something the note never orders. Give back a single JSON object and no other text:
{"wants": [{"item": want number, "needs": [{"need": the drafted demand verbatim, "evidence": the note's exact words stating it, else "", "ruling": "keep" | "story" (no sentence states it as a property of this article) | "name" (it merely echoes the article's name) | "other_item" (belongs to a different want) | "misread" (value, bound or unit wrong)}], "fix": [{"need": corrected form of a "misread" or "other_item" demand, "item": the want it belongs to}], "missing": [{"evidence": exact words of the note, "need": "property: constraint"}], "count": the unit count the note gives this want (the figure attached to the article phrase; guests, ages, years, days, clock times and room measurements are no count; 1 when none), "cap_each": a per-unit price ceiling the note puts on this article kind, else null}], "limit": the overall spending sum, numeric, else null, "pairs": [{"part": n, "host": m}] for each compatibility demand between two wants, "conditional": [{"item": n, "mode": "only_if" | "except_if", "about": m, "test": "property: values", "need": "property: constraint"}], "spurious": [numbers of wants the note never asks to buy: an article merely owned, broken, replaced, borrowed, bought elsewhere or named inside a metaphor], "verdict": "sound" or "flawed"}
Rules: a value folded into the article phrase is a genuine demand. Wording like "the <property> on them is <value>", "<article> where the <property> is <value>", "<article> in <value>", "running <value>", "made of <value>", "with a <property> of <value>" is a genuine demand. Sentences on the client's life, room, weather, craft, flatmate or deadline is never a demand unless it names a value for some property the article has; neither are the traits of an article the client owns or is replacing. A demand that cites another article's view stays one demand with its whole description. "evidence" has to be verbatim from the note; if you cannot quote it, rule "story"."""


ASKPLAN = """You decide which questions a buying agent puts to its client. Clients hold a handful of firm yet unwritten preferences, each on one property of a single wanted article. A question reveals such a preference only if it names precisely that property; a property on which the client holds no view earns a shrug, one they never considered earns a baffled reply, so each named property eats into a note.

You get the client's note, the wants together with what the note already fixes per want, plus an enumerated list of field keys observed on opened articles of each want's department, with the count of opened articles carrying each key plus sample values. Some unstated preference might concern a property absent from every listed key.

Priority:
0. Whatever the note itself tells the agent to ask the client (listed as ASK FIRST if present), each named by its property.
1. Properties the chatter hints at. The chatter frequently points toward some unstated preference yet withholds its value: for instance a sentence on something the client cannot judge or hands to you, a person whose taste decides, how the article looks or where it goes, who wears or uses it, what it will hold, "I'd rather be quizzed". For every such sentence, give the single property it points to and the want concerned, following the note's order.
Express a hinted property through one of that want's listed keys whenever one matches the hint (wearer hints -> the gender or audience key; looks hints -> whichever key for colour, for pattern or for finish most of its articles carry); a property absent from all the want's articles ranks after that want's three most frequent keys.
2. Next, listed keys, ranked: prefer properties that shoppers for that article kind settle first (type or style, form or fit, size, capacity, dimensions, material, colour, pattern: whichever governs purchases of such articles) plus keys present on most of the want's articles; after five or more opened articles, pass over keys present on merely one or two.
Leave out any property listed under LEFT OPEN (the note says the client is fine with whatever) and a property the note already fixes for that want (a demand names the property; a key with values that just repeat a demanded value is NOT fixed), an alias of a property chosen already, keys with values that bundle several purchase options together, identifiers, model designations, warranty terms and logistics. Pack sizes, unit counts and per-pack quantities are ordinary properties, and a key flagged [not the stated ...] denotes a property distinct from that demand, whatever words they share. Rank every want separately: hinted properties first, listed keys after.
"label" holds the plain property name: when it comes from a listed key, every word of that key spelled out plainly (words seperated, nothing abbreviated; never shortened to a single generic word); the bare name, free of sample values, brackets or an extra property.
Select up to COUNT per want, best first. Give back a single JSON object and no other text: {"ask": [{"item": want number, "label": "...", "hint": "up to 8 words of the note it arises from, else empty"}]}, each want's picks ranked"""


LISTEN = """A buying agent put some questions to its client. Read every reply, then give back a single JSON object and no other text:
{"limit": a fresh overall spending sum, numeric, else null, "crowded": [numbers of those questions whose reply grumbles that too much was asked at once, or demands questions singly], "puzzled": [numbers of those questions whose reply calls the question unclear or wants it said once more], "free": [{"item": want number, "property": "..."}], "drop": [{"item": want number, "property": a demand the client now releases}], "needs": [{"item": want number, "points": true if the value must be read off another article's view, false otherwise, "need": "property: constraint"}]}
- "needs": each firm preference a reply states, with its precise value or bound plus unit ("no less than 30 cm", "round", "not <material>"). Name it after whichever property the reply brings up, even where that differs from the one the note asked on; several replies on one property make a single demand. Copy the value verbatim as the reply says it, nouns and units included ("at least 3 <parts>" rather than "at least 3").
- A reply can convey its value through a reference to another article (a value equal, unequal or close to what some shop's article shows, that article singled out through a few properties) - that counts as ONE demand on whatever property the reply mentions, with "points" set true, the value being the reply's own wording copied exactly, starting at the comparison words and running until that description ends, plus anything said about a view that article might forward to; never abridge or reword. The properties singling out that other article are not demands on this want.
- "free": the property behind any question the client shrugged off as not mattering ("not bothered", "nowt in mind", "open book", "whichever way", "choose as you like" and the like).
- Dialect, slang and nautical or kitchen metaphors change nothing: read the value underneath them.
- A note may put a property of each of several wants in one breath; hang each preference on the want the reply itself names (by the article's words) and, when it names none, on the first want asked.
- Folksy or indirect wording counts as stating a value whenever it names one. Phrasing that will not haggle over a value (nothing will loosen, swap or trade it) fixes that value as the wanted one; use "not X" only where the reply rejects X as such.
- The value holds nothing but what an article's field could show, starting from the value wanted, or from its bound word, and ending with the phrase's final word, stripped of whatever the reply wraps around it ("oval" out of "matching oval" or "settled, with oval"; "no less than 40 cm" out of "landing at no less than 40 cm"; "large" out of "go with size large"). A value pointing at another article's view is the exception: copy it whole."""


NOTICE = """During a buying agent's run, the site displayed the notice below (from the client or the site). Give back a single JSON object and no other text:
{"kind": "change" if the notice alters the task (cap, demands, prices, availability, a refused payment), "error" if it merely reports a malformed or rejected call, "other" otherwise, "limit": the fresh overall spending sum, numeric, in the note's unit, else null, "gone": [ids of articles declared unobtainable], "sold_out": true if it says some chosen article has vanished, sold out, or is no longer purchasable (even unnamed), "price_moved": true if some price changed, "summary": a single brief sentence, "drop": [{"item": want number, "need": the demand no longer wanted}], "add": [{"need": "property: constraint", "item": want number}]}
A replaced demand is one drop plus one add. Report every change the notice holds."""


MATCH_RULES = """an article's field rules a property when its key carries that property's name (key spelling irrelevant, one list element suffices, synonyms and converted units pass, a field value containing the demanded words passes, bounds inclusive); a key naming some other property, even a near relative (a component's colour differs from the article's colour), rules nothing; lacking a field for the property, the demanded value written under another field, within the title, or within the blurb passes (a lone figure that never says what it measures passes nothing); a demand the department tag itself carries is fulfilled by that tag; the department tag must fit the want; an unlike-another-article demand passes unless some record shows that very value; sold-out articles and articles breaking the cap or a per-unit ceiling fail; a conditional demand is tested only after the trigger article's field was read; a demand to suit a named model code passes only when the article's text lists that code among what it works with"""


VERDICT = """You are the final gate before a buying agent pays. Going by the records shown alone, decide whether this trolley is exactly right.
Matching rules: """ + MATCH_RULES + """. Every want needs a trolley line of its own at the wanted count, the trolley may hold nothing extra, no line may be sold out, each line has to honour any per-unit ceiling for its kind, and price times count summed across the lines has to stay inside the cap. An article's newest record prevails. For a conditional demand, read the trigger article's field in the records; when the trigger holds (or, for an "except if" rule, fails), the demand binds.
Under a cheapest, max or min rule, also scan the other articles shown: if one plainly passes every demand and beats the trolley's choice under that rule, the trolley is wrong.
Give back a single JSON object and no other text: {"ok": true/false, "problems": ["brief, concrete issue", ...], "better": [ids of articles fit to replace a trolley line, strongest first]}. Set ok=false solely for an issue the records plainly show or a demand no record speaks to."""


REVIEW = """A buying agent wants to bow out without buying, believing nothing passes the demands. Test the article records shown against them, under these matching rules: """ + MATCH_RULES + """.
A record marked ROW was seen just as a search hit: its detail view (the fields) stays unread, so it can neither pass nor fail any property the hit leaves out.
Give back a single JSON object and no other text: {"fits": {"want number": [ids of articles that appear to pass every demand, cheapest first, four at most]}, "open": [ids of those ROW articles of a wanted kind, inside the cap, whose view might show they pass, most promising first, six at most], "not_firm": [{"item": want number, "need": a demand (verbatim) that the note never states as a property value of that article, merely story text misread as a demand}], "corrected": [{"item": want number, "need": a demand (verbatim) whose value, bound or unit was misread, "value": "property: constraint" as the note or reply truly states it}], "search": [{"item": want number, "query": some search words (a demanded value plus the article noun, or a synonym of the article) untried so far that might surface a passing article}], "reason": a single sentence}
A demand that a client's reply stated stays firm (it can be corrected, never set aside); a demand borrowing a value from another article's view is left alone."""


SPLIT = """A client's note was parsed into firm demands on an article. Some demands borrow a value from some other article the note cites, and the words singling out that other article (its kind, the shop, the properties after "with ...") are at times misread as demands on the article being bought. Judging by the note's wording, mark those OTHER DEMANDS below that merely single out that other article. Give back a single JSON object and no other text: {"describe_other": [those demands verbatim; empty where each demand genuinely concerns the article itself]}"""


CHAIN = """A buying agent is on the point of paying. Some demands borrow a value from some other article rather than naming it, and an article's detail view may send whoever reads it onward to another article for that same property. The purchase is valid solely once the agent has fully opened every article along each chain of that sort. Trace each demand's chain from its own description, link after link, through the views' own wording, based on the records listed further down (OPENED: that view was opened by the agent; ROW: seen just as a hit, view unread).
Give back a single JSON object and no other text: {"chains": [{"need": the demand verbatim, "links": [{"id": the article id if some record below is this link, else "", "about": the article as described by the demand or the preceding view, "find": for an unfound link, search words (the shop plus the values named in its description)}], "done": true if the final link is a view marked OPENED that states this property's value outright (likewise when the demand wants a value differing from or unlike it: that view's value is the yardstick), false otherwise, "value": this property's value exactly as the last view gives it, else ""}]}
A record counts as a link only if it fits the link's entire description: shop or brand, article kind AND each property value named in the description (a description naming a pack count and a form excludes the same shop's article with another count or form); where none fits it entirely, that link remains unfound, even if that shop sells other articles of the kind. Supply "find" for an unfound link, namely the shop's name plus the description's values, in a handful of search terms.
List the links of each chain in order and halt at the earliest link whose view is unopened or unfound, because its wording is still unknown."""


STAGEGATE = """A buying agent is on the point of STAGE. The client's note demands the errands below en route. From the numbered DIARY of the agent's doings, decide whether each errand that falls due ahead of this STAGE is complete for the trolley as it currently stands. An errand counts as complete only if a DIARY line shows it done and, for an errand concerning the trolley or concerning the order, such a line must postdate the trolley's latest change and its reply approves; questions on article properties complete no errand.
For an outstanding errand, give the single site call that performs it next, namely one tool out of TOOLS plus its parameters. A note to the client travels via the messaging tool as a single brief sentence (or question) concerning that errand alone, plainly put, naming no article property or value the errand does not require. Give no call if every due errand is complete or if the remaining notes or moves cannot perform the outstanding one.
Give back a single JSON object and no other text: {"steps": [{"step": the errand, "done": true/false, "lines": [DIARY line numbers proving it]}], "call": null or {"tool": "...", "args": {...}}, "hold": whatever change a reply requests before going on, else "", "needs": [{"need": "property: constraint", "item": want number} per firm demand stated in such a reply]}"""


RESCUE = """An article sitting in a buying agent's trolley has sold out. Find stand-ins for that line among the remaining article records shown, under these matching rules: """ + MATCH_RULES + """.
Give back a single JSON object and no other text: {"fits": [{"id": article id, "price": numeric price as the record gives it} for each article that appears to pass every demand of the want behind the sold-out line, six at most], "reason": a single sentence}"""


LAYOUT = """A site tool produced a reply, shown further down. Name its parts using the exact keys found in the reply and in the tool's parameters. Give back a single JSON object and no other text: {"rows": the key under which the article list sits, "count": the key carrying the total number of matching articles (not the number displayed), else null, "next": whichever key carries the value that fetches the subsequent page, else null, "next_arg": which tool parameter accepts that value, else null, "price": inside a row, the key of an article's price, else null, "stock": inside a row, the key telling whether the article is buyable right now, else null}"""


CLOSEST = """A buying agent found no article passing every demand, and the client asked to be told so in that case and to be offered the nearest article. Score each article record below against the demands of the want it belongs to, under these matching rules: """ + MATCH_RULES + """.
Count the checks an article fails: each demand it plainly fails or on which its records stay silent, plus one when sold out, plus one when its price (times the wanted count) breaks the cap or a per-unit ceiling. A ROW record (view unread) fails a property only where the hit contradicts it. Give back a single JSON object and no other text: {"ranked": [{"id": article id, "item": want number, "price": its price, numeric, else null, "misses": the number of failed checks, "short_on": [what it lacks, under 6 words apiece]}]}, eight entries at most, fewest misses first."""


def _pack(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _trim(value: Any, limit: int, depth: int = 0) -> Any:
    if depth > 7:
        return "..."
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "..."
    if isinstance(value, dict):
        return {k: _trim(v, limit, depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_trim(v, limit, depth + 1) for v in value]
    return value


def _json_of(text: Any) -> dict[str, Any] | None:
    """Extracts the leading JSON object from a model reply, fenced or not; None if the reply holds none."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    fenced = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", text, re.S)
    if fenced:
        text = fenced.group(1)
    try:
        got = json.loads(text)
        return got if isinstance(got, dict) else None
    except ValueError:
        pass
    head, tail = text.find("{"), text.rfind("}")
    if 0 <= head < tail:
        try:
            got = json.loads(text[head:tail + 1])
            return got if isinstance(got, dict) else None
        except ValueError:
            return None
    return None


def _plain(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(key))
    return re.sub(r"[\s_.\-]+", " ", key).strip().lower()


def _bag(text: str) -> frozenset[str]:
    return frozenset(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in _plain(text).split())


def _cores(text: str) -> set[str]:
    out = set()
    for w in _plain(text).split():
        w = w[:-2] if re.search(r"(?:ch|sh|ss|x|z)es$", w) else w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w
        if len(w) >= 3 and w not in ("and", "for", "the", "with"):
            out.add(w)
    return out


def _same_prop(prop: frozenset[str], key: frozenset[str]) -> bool:
    if not prop or not key:
        return False
    return prop == key or prop <= key or (len(key) >= 2 and key <= prop)


def _kin(one: str, other: str) -> bool:
    """Whether two property names mean the same thing: equal word bags, one inside the other, or two long words sharing their first four letters."""
    a, b = _bag(one), _bag(other)
    if _same_prop(a, b) or _same_prop(b, a):
        return True
    return any(len(x) >= 5 and len(y) >= 5 and x[:4] == y[:4] for x in _cores(one) for y in _cores(other))


def _types(schema: dict[str, Any]) -> list[str]:
    kind = schema.get("type")
    if isinstance(kind, str):
        return [kind]
    return [k for k in kind or [] if isinstance(k, str)]


def _is_type(value: Any, kind: str) -> bool:
    if kind == "string":
        return isinstance(value, str)
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "array":
        return isinstance(value, list)
    if kind == "object":
        return isinstance(value, dict)
    if kind == "null":
        return value is None
    return True


def _conform(value: Any, schema: Any, where: str, faults: list[str]) -> Any:
    """Bend a value to fit its JSON schema; note whatever cannot be mended."""
    if not isinstance(schema, dict):
        return value
    kinds = _types(schema)
    if kinds and not any(_is_type(value, k) for k in kinds):
        for kind in kinds:
            if kind == "integer" and isinstance(value, float) and value.is_integer():
                value = int(value)
                break
            if kind in ("integer", "number") and isinstance(value, str) and re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", value):
                number = float(value)
                value = int(number) if kind == "integer" or number.is_integer() else number
                break
            if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
                value = str(value)
                break
            if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
                break
            if kind == "array" and not isinstance(value, list) and value is not None:
                value = [value]
                break
        if not any(_is_type(value, k) for k in kinds):
            faults.append(f"{where} should be {'/'.join(kinds)}")
            return value
    choices = schema.get("enum")
    if isinstance(choices, list) and value not in choices:
        if isinstance(value, str):
            low = value.strip().lower()
            hit = [c for c in choices if isinstance(c, str) and c.lower() == low]
            if hit:
                value = hit[0]
        if value not in choices:
            faults.append(f"{where}: allowed values are {choices}")
    if isinstance(value, str) and isinstance(schema.get("maxLength"), int) and len(value) > schema["maxLength"]:
        value = value[: schema["maxLength"]]
    if isinstance(value, str) and isinstance(schema.get("minLength"), int) and len(value) < schema["minLength"]:
        faults.append(f"{where} is too short")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(schema.get("minimum"), (int, float)) and value < schema["minimum"]:
            faults.append(f"{where} is below {schema['minimum']}")
        if isinstance(schema.get("maximum"), (int, float)) and value > schema["maximum"]:
            faults.append(f"{where} is above {schema['maximum']}")
    if isinstance(value, list):
        if isinstance(schema.get("maxItems"), int) and len(value) > schema["maxItems"]:
            faults.append(f"{where} takes at most {schema['maxItems']} entries")
        if isinstance(schema.get("minItems"), int) and len(value) < schema["minItems"]:
            faults.append(f"{where} wants at least {schema['minItems']} entries")
        if isinstance(schema.get("items"), dict):
            value = [_conform(v, schema["items"], f"{where}[{i}]", faults) for i, v in enumerate(value)]
    if isinstance(value, dict):
        props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        if schema.get("additionalProperties") is False:
            for name in [n for n in value if n not in props]:
                faults.append(f"{where}.{name} is not an accepted argument")
        for name in schema.get("required") or []:
            if name not in value:
                faults.append(f"{where}.{name} is mandatory")
        value = {k: _conform(v, props.get(k), f"{where}.{k}", faults) if k in props else v for k, v in value.items()}
    return value


def _listed(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, (str, dict)) and value:
        return [value]
    return []


def _whole(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.0+)?\s*", value):
        return int(float(value))
    return None


def _money(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value) and float(value) > 0:
        return float(value)
    return None


def _looks_id(value: Any, named: bool = False) -> bool:
    if not isinstance(value, str) or not 3 <= len(value.strip()) <= 64 or " " in value.strip():
        return False
    return named or bool(re.search(r"\d", value))


def _id_field(rows: list[dict[str, Any]]) -> str | None:
    """The row-identifying field: space-free strings carrying digits, distinct across rows, longest on average."""
    best = None
    for key in rows[0]:
        values = [row.get(key) for row in rows]
        if not all(_looks_id(v, "id" in str(key).lower()) for v in values):
            continue
        values = [v.strip() for v in values]
        if len(rows) > 1 and len(set(values)) < len(values):
            continue
        score = ("id" in str(key).lower(), sum(len(v) for v in values) / len(values))
        if best is None or score > best[0]:
            best = (score, key)
    return best[1] if best else None


def _row_holds(node: Any, ident: str, depth: int = 0) -> bool:
    """Whether some list inside the result carries a record with this id (a basket view still listing the line)."""
    if depth > 5:
        return False
    if isinstance(node, list):
        return any(isinstance(v, dict) and ident in [str(x).strip() for x in v.values() if isinstance(x, str)] for v in node) or any(_row_holds(v, ident, depth + 1) for v in node)
    if isinstance(node, dict):
        return any(_row_holds(v, ident, depth + 1) for v in node.values())
    return False


def _spec_block(node: Any) -> dict[str, Any] | None:
    """The biggest nested dict of scalars inside a record: where a shop keeps a product's attributes."""
    if not isinstance(node, dict):
        return None
    best = None
    for value in node.values():
        if isinstance(value, dict) and len(value) >= 2:
            scalars = {k: v for k, v in value.items() if isinstance(v, (str, int, float)) and not isinstance(v, bool)}
            if len(scalars) >= 2 and (best is None or len(scalars) > len(best)):
                best = scalars
    return best


def _model_codes(text: Any) -> list[str]:
    return [re.sub(r"\s+", " ", c).strip().lower() for c in re.findall(r"\b" + MODEL_CODE_RX + r"\b", str(text or ""))]


def _code_hit(pattern: str, code: str) -> bool:
    """A code an add-on names covers a host's code when letters and digits agree, blanks and hyphens aside; a run of
    trailing x's stands for any digits of that length ("net 37xx" covers "net 3750")."""
    wide, plain = re.sub(r"[\s-]+", "", pattern.lower()), re.sub(r"[\s-]+", "", code.lower())
    stem = wide.rstrip("x")
    if stem != wide and stem[-1:].isdigit():
        return plain.startswith(stem) and len(plain) == len(wide)
    return wide == plain


def _series_marks(text: str) -> tuple[list[str], list[str]]:
    """The series an article's own part codes open with ("VI3" out of "VI3-3480"), plus whichever bare series words the text
    mentions elsewhere (a host "takes part series VI1, VI3"), clauses that exclude a series left out."""
    fam = {hit.upper() for hit in re.findall(SERIES_RX, text)}
    rest = re.sub(SERIES_RX.replace("(", "(?:", 1), " ", text)
    rest = re.sub(r"(?i)\b(?:excluding|except|does not (?:suit|take|fit)|not compatible with|not for|but not)\b[^.;]*", " ", rest)
    named = {hit.upper() for hit in re.findall(r"\b[A-Za-z]{1,5}[0-9]{1,3}\b", rest)}
    return sorted(fam), sorted(named)


def _compat_marks(text: Any) -> dict[str, Any]:
    """Model codes a page carries, codes it says it works with (and those it rules out), codes a page says a part also
    suits, plus the part series it belongs to or names."""
    text = str(text or "")
    own = re.search(r"[\[(]\s*(?:part\s+)?(" + PART_RX + r")\s*[\])]", text, re.I)
    codes: list[str] = []
    for hit in re.finditer(r"\bmodel\s*(?:#|number|no\.?)?\s*[.:]?\s*(?!(?:number|no)\b)(" + MODEL_CODE_RX + r")\b", text, re.I):
        codes += _model_codes(hit.group(1))
    for hit in re.finditer(r"\b(?:also known as|a\.?k\.?a\.?|sold also as|also sold as)\s+([^;.]+)", text, re.I):
        codes += _model_codes(hit.group(1))
    works: list[str] = []
    bars: list[str] = []
    for hit in re.finditer(r"\b(?<!never )(?<!not )(?:fits|compatible with|works with|goes with(?: every| all| any)?|suits)\s*:?\s+([^;.]+)", text, re.I):
        if not re.search(r"\bpart\s+" + PART_RX + r"\s+also\s+$", text[max(0, hit.start() - 40):hit.start()], re.I | re.S):
            cut = re.split(r"(?i)\b(?:excluding|but not|except)\b", hit.group(1), maxsplit=1)
            works += _model_codes(cut[0])
            bars += _model_codes(cut[1]) if len(cut) > 1 else []
    also = [(hit.group(1).upper(), _model_codes(hit.group(2))) for hit in re.finditer(r"\bpart\s+(" + PART_RX + r")\s+also\s+fit?s\s*:?\s*([^;.]+)", text, re.I)]
    more = re.search(r"\b(?:read|see|details|more)\s+(?:about|under|at|on)?\s*(?:the\s*)?part\s+(" + PART_RX + r")", text, re.I)
    fam, named = _series_marks(text)
    return {"own": own.group(1).upper() if own else "", "codes": codes, "works": works, "bars": bars, "also": also, "more": more.group(1).upper() if more else "", "fam": fam, "named": named}


def _tokens(text: str) -> set[str]:
    """Lower-cased words three letters long or longer, plus each number, for evidence checks against the customer's message."""
    return {w for w in re.findall(r"[a-z0-9][a-z0-9.'/\"-]*", str(text).lower()) if len(w) >= 3 or w.isdigit() or re.search(r"\d", w)}


def _in_text(fragment: str, text: str, share: float = 0.6) -> bool:
    """Whether most of a fragment's words occur in the text (a loose 'is this quoted from it' test)."""
    want = _tokens(fragment)
    if not want:
        return False
    have = _tokens(text)
    stems = {w.rstrip("s") for w in have}
    hit = sum(1 for w in want if w in have or w.rstrip("s") in stems)
    return hit >= share * len(want)


RECORD_KEY_RX = re.compile(r"^([A-Za-z_][A-Za-z0-9_ ]{0,40}): ")


def _record_scalar(text: str) -> Any:
    value = text.strip()
    if value in ("True", "False"):
        return value == "True"
    if value in ("None", "null", ""):
        return None
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value


def _is_record(text: Any) -> bool:
    """A listing the site rendered as one string: 'key: value | key: value | ...' with two keyed parts at least."""
    return isinstance(text, str) and " | " in text and bool(RECORD_KEY_RX.match(text)) and sum(1 for seg in text.split(" | ") if RECORD_KEY_RX.match(seg)) >= 2


def _record_dict(text: str) -> dict[str, Any]:
    """Such a string becomes a dict; a part shaped 'facts: a=b; c=d' becomes a nested dict of fields."""
    fields: list[tuple[str, str]] = []
    for segment in text.split(" | "):
        m = RECORD_KEY_RX.match(segment)
        if m:
            fields.append((m.group(1).strip(), segment[m.end():]))
        elif fields:
            key, prior = fields[-1]
            fields[-1] = (key, prior + " | " + segment)
    out: dict[str, Any] = {}
    for key, value in fields:
        if ("=" in value and ";" in value) or (key.lower() in ("facts", "attributes", "specs", "fields", "properties", "details") and "=" in value):
            pairs: dict[str, Any] = {}
            for part in value.split(";"):
                name, sep, val = part.partition("=")
                if name.strip() and sep:
                    pairs[name.strip()] = _record_scalar(val)
            out[key] = pairs if pairs else value
        else:
            out[key] = _record_scalar(value)
    return out


def _unpack_records(payload: Any, depth: int = 0) -> Any:
    """Every record string inside a site reply, however deep, replaced by its dict; all else untouched."""
    if depth > 6:
        return payload
    if isinstance(payload, list):
        return [_record_dict(x) if _is_record(x) else _unpack_records(x, depth + 1) for x in payload]
    if isinstance(payload, dict):
        return {k: _record_dict(v) if _is_record(v) else _unpack_records(v, depth + 1) for k, v in payload.items()}
    return _record_dict(payload) if _is_record(payload) else payload


NUMBER_WORDS = {1: "one|a single", 2: "two|a pair of|a couple of", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}
ORDER_RX = re.compile(r"\b(?:fetch|find|get|grab|send|bring|sort|order|buy|pick up|want|wanting|need|looking for|could you|can you|\d+ x|(?:first|second|third|fourth|fifth|last)(?:ly)?(?:,| one| up| off)|also|plus|and \d)\b", re.I)


def _count_backed(count: int, phrases: list[str], text: str) -> bool:
    """Whether a unit count above one is written right before the article phrase (digits or a number word, no sentence break between)."""
    if count <= 1:
        return True
    words = {w for phrase in phrases for w in _cores(phrase) if len(w) >= 4}
    if not words:
        return False
    number = r"(?:" + str(count) + (r"|" + NUMBER_WORDS[count] if count in NUMBER_WORDS else "") + r")"
    for word in words:
        if re.search(r"\b" + number + r"\b[^.?!;]{0,45}?\b" + re.escape(word), text, re.I):
            return True
    return False


def _ordered(phrases: list[str], text: str) -> bool:
    """Whether some ordering sentence of the message names the article (some word from its kind or shop label)."""
    words = {w for phrase in phrases for w in _cores(phrase) if len(w) >= 4}
    for sentence in re.split(r"(?<=[.?!])\s+", text):
        low = sentence.lower()
        if ORDER_RX.search(sentence) and any(w in low for w in words):
            return True
    return False


class Want:
    """One product the customer wants: its kind, count, firm needs and what the interview added."""

    def __init__(self, what: str, label: str, count: int, needs: list[str], cap: float | None = None):
        self.what = what
        self.label = label
        self.count = count
        self.needs = needs
        self.cap = cap
        self.told: list[str] = []
        self.dropped: list[str] = []
        self.free: list[str] = []


class Wren:
    def __init__(self, problem: dict[str, Any]):
        world = problem["environment"]
        self.link = dict(world["binding"])
        self.scene = world["policy_view"]
        self.ask_text = str(self.scene.get("query") or "")
        self.toolset = [t for t in self.scene.get("tools") or [] if isinstance(t, dict) and isinstance(t.get("function"), dict) and isinstance(t["function"].get("name"), str)]
        self.specs = {t["function"]["name"]: t["function"].get("parameters") if isinstance(t["function"].get("parameters"), dict) else {"type": "object", "properties": {}} for t in self.toolset}
        cap = _whole(self.scene.get("max_steps"))
        if cap is None and isinstance(self.scene.get("max_steps"), float):
            cap = int(self.scene["max_steps"])
        self.step_cap = cap if cap and cap > 0 else 30
        self.born = time.monotonic()
        self.at = 0
        self.over = bool(self.scene.get("finished"))
        self.env_dead = False
        self.dry = False
        self.slept = 0
        self.blanks = 0
        self.spent = 0.0
        self.calls = 0
        self.turns = 0
        self.log: list[dict[str, Any]] = []
        self.intake: dict[str, Any] = {}
        self.sifted: dict[str, list[str]] = {}
        self.jobs: dict[str, str] = {}
        self.wants: list[Want] = []
        self.limit: Any = None
        self.talk_cap = 0
        self.talked = 0
        self.chores: list[dict[str, Any]] = []
        self.held_back = 0
        self.chore_talk = 0
        self.chores_done: set[tuple[str, str]] = set()
        self.diary: list[str] = []
        self.diary_steps: list[int] = []
        self.interviewed = False
        self.cards: dict[str, dict[str, Any]] = {}
        self.pages: dict[str, dict[str, Any]] = {}
        self.arrival: list[str] = []
        self.noted_lines: set[str] = set()
        self.changes: list[str] = []
        self.epoch = 0
        self.basket_moved = False
        self.basket_at = -1
        self.basket_seen = -1
        self.basket_snap: Any = None
        self.flagged_named = False
        self.basket: dict[str, int] = {}
        self.judged: set[str] = set()
        self.reviewed = False
        self.deal_seen: dict[str, int] = {}
        self.vanished: dict[str, int] = {}
        self.unbuyable: set[str] = set()
        self.soldout_lines: set[str] = set()
        self.flagged_sold = False
        self.stalls = 0
        self.id_slots: set[tuple[str, str]] = set()
        self.basket_reply = ""
        self.bounced_at = -1
        self.leads: list[str] = []
        self.exit_blocks = 0
        self.nudged_once = False
        self.built: list[str] = []
        self.removed: set[str] = set()
        self.late_block = False
        self.lead_map: dict[str, list[str]] = {}
        self.rulings: dict[str, bool] = {}
        self.vouched: str | None = None
        self.autofilled = False
        self.id_name: str | None = None
        self.specs_by: dict[str, bool] = {}
        self.pending_notes: list[str] = []
        self.checkouts = 0
        self.just_did: list[str] = []
        self.thread: list[dict[str, Any]] = []
        self.visited: set[str] = set()
        self.pointers: list[tuple[int, str]] = []
        self.pointed_values: dict[str, str] = {}
        self.chain_runs = 0
        self.chain_cap: int | None = None
        self.refuse_trailed = False
        self.heard_once = False
        self.hunt_done = False
        self.hunt_tried: set[str] = set()
        self.chain_blocks = 0
        self.side_ok = 0
        self.side_bad = 0
        self.view_fields: dict[str, set[str]] = {}
        self.card_fields: dict[str, tuple[int, set[str]]] = {}
        self.replay: dict[str, tuple[int, int, str]] = {}
        self.self_paged = 0
        self.layouts: dict[str, dict[str, Any] | None] = {}
        self.price_field: str | None = None
        self.stock_field: str | None = None
        self.peeked_marks: set[str] = set()
        self.wider_block = False
        self.paged_seen: list[Any] = []
        self.hint_order: list[int] = []
        self.listing = True
        self.lanes: dict[int, list[str]] = {}
        self.lane_found: dict[int, int] = {}
        self.lane_spent: dict[int, int] = {}
        self.lane_used: dict[int, set[str]] = {}
        self.follow: list[tuple[int, list[str]]] = []
        self.late_asked = False
        self.browse: dict[str, Any] | None = None
        self.browse_misses = 0
        self.listed_below: set[tuple[str, float]] = set()
        self.line_blocks = 0
        self.pairs: list[tuple[int, int]] = []
        self.pair_blocks = 0
        self.queries: list[dict[str, Any]] = []
        self.link_hunts = 0
        self.link_found: dict[str, list[str]] = {}
        self.conditionals: list[dict[str, Any]] = []
        self.fallback: dict[str, Any] = {}
        self.hints: list[dict[str, Any]] = []
        self.cap_blocks = 0
        self.nearest: dict[str, Any] | None = None

    # ---------- small helpers ----------
    def room(self) -> int:
        return self.step_cap - self.at

    def elapsed(self) -> float:
        return time.monotonic() - self.born

    def late(self) -> bool:
        return self.elapsed() >= TIME_SOFT

    def talk_room(self) -> int:
        """Notes the interviewer may still spend; those needed by chores at payment or bow-out time stay reserved."""
        return self.talk_cap - self.talked - max(0, self.held_back - self.chore_talk)

    def basket_key(self) -> str:
        return _pack(sorted(self.basket.items()))

    def page_views(self) -> set[str]:
        """Tools showing an article's detail view: the deal view where the site has one beside the field view
        (a bare field list is no detail view), otherwise the single-article view; plus the multi-article view."""
        page = self.jobs.get("deal") or self.jobs.get("page")
        return {t for t in (page, self.jobs.get("side")) if t}

    def spec_view(self) -> str | None:
        """The single-article tool carrying fields: one that already showed them, otherwise the intake's field view."""
        proven = next((t for t, ok in self.specs_by.items() if ok and self.id_slot(t)), None)
        if proven:
            return proven
        tool = self.jobs.get("page")
        return tool if self.specs_by.get(tool or "") is not False and self.id_slot(tool) else None

    def side_specs(self) -> bool | None:
        """Does the multi-article view carry fields? True / False once observed, None until then."""
        return self.specs_by.get(self.jobs.get("side") or "")

    def side_blind(self) -> bool:
        """Reading fields through the multi-article view is pointless: it carried none, or (untried) the site
        keeps a separate field view (one that carried them, or one distinct from the deal view)."""
        seen = self.side_specs()
        return seen is False or (seen is None and (self.specs_by.get(self.spec_view() or "") is True or bool(self.jobs.get("deal"))))

    def chore_due(self, stage: str) -> bool:
        return any(c["stage"] == stage for c in self.chores) and (stage, self.basket_key() + f"#{self.epoch}") not in self.chores_done

    def keep_back(self) -> int:
        """Moves held in reserve for finishing: the payment, and another while a payment-time chore still needs a note."""
        talk = any(c["stage"] == "order" and c["messages"] > 0 for c in self.chores)
        return 2 if talk and self.talked < self.talk_cap and self.chore_due("order") else 1

    def judge_models(self, limit: float = COST_SOFT) -> list[str]:
        if self.spent <= limit and self.spent + BIG_MARGIN <= COST_HARD and self.elapsed() < TIME_SOFT - 60:
            return [BIG, SMALL]
        return [SMALL] if self.spent <= COST_HARD else []

    # ---------- model and shop calls ----------
    def consult(self, messages: list[dict[str, Any]], models: list[str], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
        for model in models:
            if self.calls >= CALL_CAP:
                return None
            self.calls += 1
            tries = 0
            while True:
                message, status = self.one_call(model, messages, max_tokens, tools)
                if message is not None:
                    return message
                # no credit, rate limits, timeouts and server errors are retried after a pause
                soft = status is None or status in (402, 408, 425, 429) or (isinstance(status, int) and status >= 500)
                if self.late():
                    break
                # a timeout already burnt the client's 50 s, so it is resent once at most
                if not soft or tries >= (1 if status in (None, 408) else len(BACKOFF)) or self.slept + BACKOFF[tries] > BACKOFF_SUM:
                    break
                pause = BACKOFF[tries] * (0.8 + 0.4 * random.random())
                self.slept += BACKOFF[tries]
                tries += 1
                time.sleep(pause)
            if status == 402:
                self.dry = True
                return None
        return None

    def one_call(self, model: str, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None) -> tuple[dict[str, Any] | None, Any]:
        """One inference request: the reply, or None together with the failure status (0 for an empty reply)."""
        sent = messages
        if model.startswith("anthropic/") and len(messages) > 1:
            sent = [dict(m) for m in messages]
            marks = [0] + ([len(sent) - 1] if len(sent) > 1 else [])
            for index in marks:
                content = sent[index].get("content")
                if isinstance(content, str) and content:
                    sent[index]["content"] = [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        body: dict[str, Any] = {"model": model, "messages": sent, "max_tokens": max_tokens, "temperature": 0}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        _brain.timeout = max(8, min(45, int(TIME_KILL - 12 - self.elapsed())))
        result = _brain.post_verbose("/inference/chat/completions", body)
        data = result.data
        if isinstance(data, dict) and data.get("choices"):
            usage = data.get("usage") or {}
            if isinstance(usage.get("cost"), (int, float)):
                self.spent += float(usage["cost"])
            message = data["choices"][0].get("message") or {}
            if message.get("content") or message.get("tool_calls"):
                return message, 200
            self.log.append({"kind": "llm_fail", "model": model, "status": 0})
            return None, 0
        error = result.error or {}
        status = error.get("status")
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            # an error wrapped inside a 200 reply, carrying the provider's own code
            status = _whole(data["error"].get("code"))
            error = {"body": data["error"].get("message")}
        self.log.append({"kind": "llm_fail", "model": model, "status": status, "body": str(error.get("body") or "")[:200]})
        return None, status

    def json_call(self, prompt: str, models: list[str], max_tokens: int = 1800) -> dict[str, Any] | None:
        if self.elapsed() >= TIME_HARD:
            return None
        for model in models:
            message = self.consult([{"role": "user", "content": prompt}], [model], max_tokens)
            got = _json_of((message or {}).get("content"))
            if got is not None:
                return got
            if self.env_dead or self.dry or self.calls >= CALL_CAP:
                return None
        return None

    def do_step(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        payload = dict(self.link)
        payload["step"] = self.at + 1
        payload["action"] = {"name": name, "args": args}
        _shop.timeout = max(10, min(95, int(TIME_KILL - 6 - self.elapsed())))
        _shop.max_retries = 3 if self.elapsed() < TIME_SOFT else 1
        for _ in range(4):
            result = _shop.post_verbose("/environment/call", payload)
            if isinstance(result.data, dict):
                data = result.data
                if "observation" in data:
                    data["observation"] = _unpack_records(data["observation"])
                self.at = data["step"] if isinstance(data.get("step"), int) else self.at + 1
                if data.get("finished"):
                    self.over = True
                seen = data.get("observation")
                if name in self.page_views() and not (isinstance(seen, dict) and seen.get("error")):
                    # products whose full page was opened (a search row does not count)
                    for value in args.values():
                        self.visited.update(v.strip() for v in (value if isinstance(value, list) else [value]) if isinstance(v, str))
                return data
            error = result.error or {}
            status = error.get("status")
            body = str(error.get("body") or "")
            self.log.append({"kind": "env_fail", "status": status, "body": body[:200]})
            if status == 409:
                low = body.lower()
                if "finish" in low or "quarantin" in low or "finaliz" in low:
                    self.over = True
                    return None
                found = re.search(r"expected (\d+)", body)
                if found:
                    self.at = int(found.group(1)) - 1
                    payload["step"] = self.at + 1
                    continue
                if "reuse" in low or "different action" in low:
                    self.at += 1
                    payload["step"] = self.at + 1
                    continue
                return None
            if status in (400, 401, 402, 403, 404, 422):
                if status == 402:
                    self.env_dead = True
                return None
        return None

    # ---------- reading the customer's message ----------
    def tool_lines(self) -> list[str]:
        return [f"- {t['function']['name']}: {t['function'].get('description', '')} | parameters: {_pack(t['function'].get('parameters') or {})}" for t in self.toolset]

    def sift_pass(self) -> None:
        """Sort the note's sentences into orders, working method and chatter; only sentences found verbatim in the note count."""
        self.sifted = {}
        if self.dry or len(self.ask_text.split()) < 12:
            return
        got = self.json_call(SIFT + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text, [SMALL, BIG], 3000) or {}
        bins: dict[str, list[str]] = {}
        for key in ("orders", "process", "chatter"):
            seen: list[str] = []
            for item in _listed(got.get(key)):
                line = re.sub(r"\s+", " ", str(item)).strip()
                if line and len(line) <= 600 and _in_text(line, self.ask_text, 0.9) and line not in seen:
                    seen.append(line)
            bins[key] = seen
        kept = bins["orders"] + bins["process"]
        # the sift is trusted only when it kept something and set aside a fair share of the note
        if bins["orders"] and sum(len(x) for x in bins["chatter"]) >= 0.15 * len(self.ask_text) and sum(len(x) for x in kept) >= 0.2 * len(self.ask_text):
            self.sifted = bins
            self.log.append({"kind": "sift", "orders": len(bins["orders"]), "process": len(bins["process"]), "chatter": len(bins["chatter"])})

    def note_core(self) -> str:
        """The note with its chatter set aside: the ordering and working-method sentences in the note's own order."""
        if not self.sifted:
            return self.ask_text
        lines = self.sifted["orders"] + self.sifted["process"]
        lines.sort(key=lambda line: self.ask_text.find(line[:40]) if self.ask_text.find(line[:40]) >= 0 else len(self.ask_text))
        return " ".join(lines)

    def intake_pass(self) -> None:
        self.sift_pass()
        prompt = INTAKE + "\nTOOLS:\n" + "\n".join(self.tool_lines()) + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text
        if self.sifted:
            prompt += ("\n\nSIFTED BEFOREHAND (a first reading sorted the sentences; the ones below carry the order and the working method, the rest is chatter. "
                       "A want or a demand drawn from any other sentence needs that sentence to state a property value outright):\nORDER SENTENCES:\n- "
                       + "\n- ".join(self.sifted["orders"]) + "\nWORKING-METHOD SENTENCES:\n- " + ("\n- ".join(self.sifted["process"]) or "(none)"))
        if self.scene.get("opening") is not None:
            prompt += "\n\nWHAT THE SHOP SHOWED FIRST:\n" + _pack(_trim(self.scene.get("opening"), 1500))
        # the small model reads first; the big one is called when that reading names no want or no checkout tool
        draft = self.json_call(prompt, [SMALL], 4000) or {}
        if not self.dry and (not _listed(draft.get("wants")) or not isinstance(draft.get("tools"), dict) or not draft["tools"].get("checkout")):
            draft = self.json_call(prompt, [BIG, SMALL], 2500) or draft
        elif not self.dry and self.twin_labels(draft):
            again = self.json_call(prompt, [BIG], 2500) or {}
            if _listed(again.get("wants")) and isinstance(again.get("tools"), dict) and again["tools"].get("checkout"):
                draft = again
        self.intake = draft
        jobs = draft.get("tools") if isinstance(draft.get("tools"), dict) else {}
        self.jobs = {k: v for k, v in jobs.items() if isinstance(v, str) and v in self.specs}
        for key in ("checkout_args", "walk_away_args"):
            if not isinstance(draft.get(key), dict):
                draft[key] = {}
        self.guess_jobs(draft)
        self.read_browse(draft.get("browse"))
        buy, quit_ = self.jobs.get("checkout"), self.jobs.get("walk_away")
        if buy and buy == quit_ and (not draft["checkout_args"] or not draft["walk_away_args"] or draft["checkout_args"] == draft["walk_away_args"]):
            ask = (f"The shop tool {buy} both places the order and ends with no purchase, depending on its arguments. Schema: "
                   + _pack(self.specs.get(buy)) + "\nGive back a single JSON object and no other text: {\"checkout_args\": the fixed parameters under which it pays, \"walk_away_args\": the fixed parameters under which it ends with no purchase} (just the mode-selecting parameters).")
            fixed = self.json_call(ask, [SMALL, BIG], 1500) or {}
            if isinstance(fixed.get("checkout_args"), dict) and isinstance(fixed.get("walk_away_args"), dict) and fixed["checkout_args"] != fixed["walk_away_args"]:
                draft["checkout_args"], draft["walk_away_args"] = fixed["checkout_args"], fixed["walk_away_args"]
        for entry in _listed(draft.get("wants")):
            if not isinstance(entry, dict):
                continue
            count = _whole(entry.get("count"))
            needs = [str(n).strip() for n in _listed(entry.get("needs")) if str(n).strip()]
            want = Want(str(entry.get("what") or "the product"), str(entry.get("label") or ""), count if count and count > 0 else 1, needs, _money(entry.get("cap_each")))
            if want.count > 1 and self.ask_text.strip() and not _count_backed(want.count, [want.what, want.label], self.ask_text):
                # a figure from the chatter (guests, years, ages) taken for a unit count
                self.log.append({"kind": "count_unbacked", "want": want.what, "count": want.count})
                want.count = 1
            self.wants.append(want)
        renumber = self.merge_twins()
        if not self.wants:
            self.wants = [Want("the wanted product", "", 1, [])]
        for pair in _listed(draft.get("pairs")):
            self.add_pair(pair, renumber)
        for rule in _listed(draft.get("conditional")):
            self.add_conditional(rule, renumber)
        for pointer in _listed(draft.get("pointers")):
            self.add_pointer(str(pointer).strip())
        for hint in _listed(draft.get("hints")):
            if isinstance(hint, dict) and str(hint.get("words") or "").strip():
                number = renumber.get(_whole(hint.get("item")) or 1, _whole(hint.get("item")) or 1)
                number = min(max(1, number), len(self.wants))
                self.hints.append({"item": number, "words": str(hint["words"]).strip()[:120], "property": str(hint.get("property") or "").strip()})
        for entry in _listed(draft.get("open")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                number = renumber.get(_whole(entry.get("item")) or 1, _whole(entry.get("item")) or 1)
                number = min(max(1, number), len(self.wants))
                prop = str(entry["property"]).strip()[:60]
                if prop not in self.wants[number - 1].free:
                    self.wants[number - 1].free.append(prop)
                # a property the note leaves open was no hint either
                self.hints = [h for h in self.hints if not (h["item"] == number and h["property"] and _kin(h["property"], prop))]
        self.proof_pass()
        self.noise_guard()
        self.count_guard()
        self.colour_guard()
        self.split_pointer()
        amount = (draft.get("limit") or {}).get("amount") if isinstance(draft.get("limit"), dict) else draft.get("limit")
        money = _money(amount)
        if money is not None:
            self.limit = money
        cap = _whole(draft.get("message_cap"))
        if cap is not None and cap >= 0:
            self.talk_cap = cap
        elif self.jobs.get("talk"):
            self.talk_cap = 3
        steps = _whole(draft.get("step_cap"))
        if steps is not None and 10 <= steps < self.step_cap:
            self.step_cap = steps
        for entry in _listed(draft.get("chores")):
            if isinstance(entry, dict) and str(entry.get("what") or "").strip():
                stage = str(entry.get("stage") or "").strip().lower()
                self.chores.append({"what": str(entry["what"]).strip(), "stage": stage if stage in ("pick", "order", "end") else "order",
                                    "messages": max(0, _whole(entry.get("messages")) or 0)})
        if self.jobs.get("talk"):
            # messages the chores at the order and the ending consume: at least the larger of the two
            talk = [c["messages"] for c in self.chores if c["stage"] in ("order", "end") and c["messages"]]
            self.held_back = min(max(talk), self.talk_cap) if talk else 0
        fallback = draft.get("fallback") if isinstance(draft.get("fallback"), dict) else {}
        self.fallback = {"announce": fallback.get("announce") is True, "offer_nearest": fallback.get("offer_nearest") is True,
                         "rule": str(fallback.get("nearest_rule") or "").strip(), "slack": _money(fallback.get("slack_pct")),
                         "quiet": fallback.get("stay_quiet") if isinstance(fallback.get("stay_quiet"), bool) else None}
        if self.fallback["offer_nearest"] and self.fallback["quiet"] is None:
            self.fallback["quiet"] = False

    def guess_jobs(self, draft: dict[str, Any]) -> None:
        """Fill the tool jobs the intake left blank out of what the tools call themselves and their schemas (a race renames every tool)."""
        before = dict(self.jobs)
        taken = set(self.jobs.values())

        def params(name: str) -> dict[str, Any]:
            return (self.specs.get(name) or {}).get("properties") or {}

        def kinds(name: str) -> tuple[list[str], list[str], list[str], list[str]]:
            texts, enums, numbers, arrays = [], [], [], []
            for key, spec in params(name).items():
                if not isinstance(spec, dict):
                    continue
                if isinstance(spec.get("enum"), list):
                    enums.append(key)
                elif "array" in _types(spec):
                    arrays.append(key)
                elif "integer" in _types(spec) or "number" in _types(spec):
                    numbers.append(key)
                elif "string" in _types(spec):
                    texts.append(key)
            return texts, enums, numbers, arrays

        def finish_modes(name: str) -> tuple[str, Any, Any] | None:
            for key, spec in params(name).items():
                if isinstance(spec, dict) and isinstance(spec.get("enum"), list):
                    buy = next((v for v in spec["enum"] if isinstance(v, str) and re.search(r"check|order|buy|purchas|confirm|commit|place|submit|pay|complete", v, re.I)), None)
                    quit_ = next((v for v in spec["enum"] if isinstance(v, str) and v != buy and re.search(r"cancel|declin|abort|skip|quit|give|none|stop|walk|reject|abandon|no", v, re.I)), None)
                    if buy and quit_:
                        return key, buy, quit_
            return None
        names = [n for n in self.specs if n not in taken]
        # one tool closing the episode either way, told apart by a mode value
        if not self.jobs.get("checkout") or not self.jobs.get("walk_away"):
            for name in list(self.specs):
                modes = finish_modes(name)
                if modes and (re.search(r"wrap|check|close|finish|final|complet|order|purchas|pay|done|end|submit|commit|conclud|settle", name, re.I) or not kinds(name)[0]):
                    key, buy, quit_ = modes
                    if not self.jobs.get("checkout"):
                        self.jobs["checkout"] = name
                        draft["checkout_args"] = {key: buy}
                    if not self.jobs.get("walk_away"):
                        self.jobs["walk_away"] = name
                        draft["walk_away_args"] = {key: quit_}
                    break
        if not self.jobs.get("checkout"):
            pick = next((n for n in names if re.search(r"checkout|check_out|purchase|pay|place.?order|buy|confirm|commit", n, re.I) and not kinds(n)[3] and not re.search(r"cancel|declin|quit|abandon|give", n, re.I)), None)
            if pick:
                self.jobs["checkout"] = pick
        if not self.jobs.get("walk_away"):
            pick = next((n for n in names if re.search(r"cancel|declin|quit|abandon|give.?up|walk|no.?purchase|skip|abort|reject", n, re.I) and n != self.jobs.get("checkout")), None)
            if pick:
                self.jobs["walk_away"] = pick
        taken = set(self.jobs.values())
        names = [n for n in self.specs if n not in taken]
        if not self.jobs.get("talk"):
            pick = next((n for n in names if re.search(r"message|ask|buyer|shopper|chat|say|talk|contact|client|customer|user|tell|question", n, re.I) and len(kinds(n)[0]) >= 1 and not kinds(n)[3]), None)
            if pick:
                self.jobs["talk"] = pick
        if not self.jobs.get("basket"):
            pick = next((n for n in names if re.search(r"add|stash|put|cart|basket|trolley|bag|select|pick", n, re.I) and kinds(n)[2] and kinds(n)[0] and n != self.jobs.get("talk")), None) \
                or next((n for n in names if kinds(n)[3] and re.search(r"add|stash|put|cart|basket|trolley|bag|select", n, re.I)), None)
            if pick:
                self.jobs["basket"] = pick
        if not self.jobs.get("basket_view"):
            pick = next((n for n in names if not params(n) and re.search(r"cart|basket|trolley|bag|summary|items|review|view", n, re.I) and n != self.jobs.get("basket")), None)
            if pick:
                self.jobs["basket_view"] = pick
        taken = set(self.jobs.values())
        names = [n for n in self.specs if n not in taken]
        seekers = [n for n in names if re.search(r"search|seek|find|lookup|look_up|query|browse|list|range|catalog|explore|filter", n, re.I)]
        if not self.jobs.get("find"):
            pick = next((n for n in seekers if any(k not in ((self.specs.get(n) or {}).get("required") or []) or True for k in kinds(n)[0]) and kinds(n)[0]), None) or (seekers[0] if seekers else None)
            if pick:
                self.jobs["find"] = pick
        names = [n for n in names if n not in seekers]
        if not self.jobs.get("side"):
            pick = next((n for n in names if kinds(n)[3] and n not in (self.jobs.get("basket"), self.jobs.get("find")) and re.search(r"compar|weigh|side|several|multi|batch|many|bulk", n, re.I)), None)
            if pick:
                self.jobs["side"] = pick
        if not self.jobs.get("page"):
            singles = [n for n in names if len(kinds(n)[0]) == 1 and not kinds(n)[1] and not kinds(n)[2] and not kinds(n)[3] and n not in self.jobs.values()]
            pick = next((n for n in singles if re.search(r"spec|detail|full|field|attribute|propert|fact", n, re.I)), None) or next((n for n in singles if re.search(r"show|page|view|describe|open|listing|offer|item|product", n, re.I)), None)
            if pick:
                self.jobs["page"] = pick
                other = next((n for n in singles if n != pick and re.search(r"show|page|view|describe|open|listing|offer|deal|price", n, re.I)), None)
                if other and not self.jobs.get("deal"):
                    self.jobs["deal"] = other
        if self.jobs != before:
            self.log.append({"kind": "jobs_guessed", "jobs": dict(self.jobs)})

    def read_browse(self, spec: Any) -> None:
        """The way one department gets listed by a tool, price ceiling included where it accepts one, from the intake or the schemas."""
        if isinstance(spec, dict) and spec.get("tool") in self.specs:
            props = (self.specs[spec["tool"]] or {}).get("properties") or {}
            category, price = spec.get("category"), spec.get("max_price")
            if category in props:
                fixed = spec.get("fixed") if isinstance(spec.get("fixed"), dict) else {}
                self.browse = {"tool": spec["tool"], "category": category, "price": price if price in props else None,
                               "fixed": {k: v for k, v in fixed.items() if k in props and k not in (category, price)}}
                return
        for name, schema in self.specs.items():
            props = (schema or {}).get("properties") or {}
            shelves = [k for k, v in props.items() if isinstance(v, dict) and re.search(r"categor|section|shelf|department|aisle|label|group|range", _plain(k)) and ("enum" in v or "string" in _types(v))]
            caps = [k for k, v in props.items() if isinstance(v, dict) and ({"integer", "number"} & set(_types(v))) and re.search(r"cap|max|limit|ceiling|budget|under|price|up to", _plain(k)) and not re.search(r"page|size|count|step|num", _plain(k))]
            if len(shelves) == 1 and name not in (self.jobs.get("talk"), self.jobs.get("basket"), self.jobs.get("checkout")):
                self.browse = {"tool": name, "category": shelves[0], "price": caps[0] if len(caps) == 1 else None, "fixed": {}}
                return

    def shelf_of(self, number: int) -> str:
        """The department tag of a want: whichever its rows carry most, else the intake's label."""
        kind_field = self.kind_field()
        want = self.wants[number - 1]
        labels = [str(r.get(kind_field) or "") for r in self.cards.values()] if kind_field else []
        labels = [label for label in labels if label and self.owned_by(want, label)]
        return max(set(labels), key=labels.count) if labels else want.label

    def list_category(self, number: int, cap: float | None, more: int = 0) -> list[str] | None:
        """What a department listing yields for one want, below a price ceiling where one exists, reading at most
        `more` extra pages; None where no tool lists departments or its call cannot be formed."""
        spec = self.browse
        label = self.shelf_of(number)
        if not spec or not label or self.browse_misses >= 2 or (cap is not None and not spec["price"]) or self.over:
            return None
        args: dict[str, Any] = {**spec["fixed"], spec["category"]: label}
        if cap is not None:
            args[spec["price"]] = int(cap) if float(cap).is_integer() else cap
        schema = self.specs.get(spec["tool"]) or {}
        props = schema.get("properties") or {}
        for key in schema.get("required") or []:
            # a required parameter the spec left out: the department-listing mode of an enum, else the tag itself
            if key in args or not isinstance(props.get(key), dict):
                continue
            options = props[key].get("enum")
            if isinstance(options, list) and options:
                used = [a[key] for a in self.queries if key in a and spec["category"] in a]
                mode = next((o for o in options if isinstance(o, str) and re.search(r"categor|brows|list|section|group|filter|department|shelf", o, re.I)), None)
                args[key] = used[-1] if used else mode if mode is not None else options[0]
            elif "string" in _types(props[key]):
                args[key] = label
            elif {"integer", "number"} & set(_types(props[key])):
                args[key] = 1
        faults: list[str] = []
        args = _conform(args, schema, "args", faults)
        if faults:
            self.browse_misses += 1
            return None
        reply = self.do_step(spec["tool"], args)
        if reply is None:
            self.browse_misses += 1
            return None
        seen = reply.get("observation")
        self.digest(spec["tool"], seen)
        found = [i for i, _ in self.cards_in(seen)]
        if isinstance(seen, dict) and seen.get("error"):
            self.browse_misses += 1
            return None
        for _ in range(more):
            plan = self.page_plan(spec["tool"], seen)
            if not plan or plan[3] is None or self.over:
                break
            faults = []
            call = _conform({**args, plan[2]: plan[3]}, self.specs.get(spec["tool"]), "args", faults)
            reply = self.do_step(spec["tool"], call) if not faults else None
            if reply is None:
                break
            seen = reply.get("observation")
            self.digest(spec["tool"], seen)
            found += [i for i, _ in self.cards_in(seen)]
        return found

    def list_below(self) -> None:
        """When the rule is cheapest: list the department of each trolley line below that line's price, so cheaper rows surface."""
        if self.choose() != "cheapest" or not self.browse or not self.browse.get("price"):
            return
        kind_field = self.kind_field()
        for line in self.basket_ids():
            price = self.price(line)
            label = str((self.cards.get(line) or {}).get(kind_field) or "") if kind_field else ""
            number = next((n for n, w in enumerate(self.wants, 1) if label and self.owned_by(w, label)), 0)
            if price is None or price < 2 or not number or (line, price) in self.listed_below:
                continue
            if self.room() <= self.keep_back() + 5 or self.over or self.late():
                return
            self.listed_below.add((line, price))
            self.list_category(number, price - 1 if float(price).is_integer() else price, 1 if self.room() > self.keep_back() + 8 else 0)
            self.log.append({"kind": "list_below", "line": line, "price": price})

    def take_review(self, got: dict[str, Any] | None) -> None:
        """Apply what the bow-out audit faulted in the sheet: story demands set aside, misread values mended.
        A demand the client stated in a reply may be mended, never set aside; a pointer demand is left alone."""
        pointed = {need for _, need in self.pointers}
        changed: list[str] = []
        aside: set[int] = set()
        for entry in _listed((got or {}).get("not_firm")):
            if not isinstance(entry, dict):
                continue
            number, need = _whole(entry.get("item")) or 0, str(entry.get("need") or "").strip()
            if 1 <= number <= len(self.wants) and number not in aside:
                want = self.wants[number - 1]
                if need in want.needs and need not in want.dropped and need not in pointed:
                    aside.add(number)
                    want.dropped.append(need)
                    changed.append(f"set aside {need}")
        for entry in _listed((got or {}).get("corrected")):
            if not isinstance(entry, dict):
                continue
            number, need, value = _whole(entry.get("item")) or 0, str(entry.get("need") or "").strip(), str(entry.get("value") or "").strip()
            if 1 <= number <= len(self.wants) and ":" in value and value != need:
                want = self.wants[number - 1]
                if need in want.needs + want.told and need not in pointed and need not in want.dropped and _in_text(value.partition(":")[2], self.ask_text + " " + " ".join(want.told), 0.5):
                    want.dropped.append(need)
                    want.told.append(value)
                    changed.append(f"{need} read as {value}")
        if changed:
            self.epoch += 1
            self.log.append({"kind": "review_applied", "changed": changed})
            self.changes.append("the note's re-read: " + "; ".join(changed))

    def recovery_search(self, got: dict[str, Any] | None, missing: list[int]) -> bool:
        """Ahead of a bow-out: list the department of every want still without a fit, run whatever searches the audit
        suggests, then open the likeliest fresh rows, giving a second audit new evidence."""
        keep = self.keep_back() + len(self.wants) + 2
        room = min(HUNT_STEPS + 2, self.room() - keep)
        if room < 3 or self.late():
            return False
        before = len(self.pages)
        last = self.at + room
        searching = self.at + max(2, room * 2 // 3)
        plan = self.query_pattern()
        queries: list[tuple[int, str]] = []
        for entry in _listed((got or {}).get("search")):
            if isinstance(entry, dict) and str(entry.get("query") or "").strip():
                queries.append((min(max(1, _whole(entry.get("item")) or 1), len(self.wants)), str(entry["query"]).strip()[:80]))
        queries = [q for q in queries if q[0] in missing]
        for n in missing:
            queries += [(n, q) for q in self.hunt_queries(n) if q not in [x[1].lower() for x in queries]][:4]
        found: dict[int, list[str]] = {n: [] for n in missing}
        cap = self.limit if isinstance(self.limit, (int, float)) and self.browse and self.browse.get("price") else None
        for number in missing[:3]:
            if self.at < searching:
                found[number] += self.list_category(number, cap) or []
        for number, words in queries[:HUNT_STEPS]:
            if plan and self.at < searching and not self.over and not self.late():
                tool, base, field = plan
                faults: list[str] = []
                args = _conform({**base, field: words}, self.specs.get(tool), "args", faults)
                reply = None if faults else self.do_step(tool, args)
                if reply is not None:
                    self.digest(tool, reply.get("observation"))
                    self.queries = (self.queries + [dict(args)])[-8:]
                    found.setdefault(number, [])
                    found[number] += [i for i, _ in self.cards_in(reply.get("observation"))]
        kind_field = self.kind_field()
        for number, rows in found.items():
            want = self.wants[number - 1]
            wanted = {w for w in re.findall(r"[a-z0-9]{3,}", " ".join(self.need_values(number)).lower())}
            pool = [i for i in dict.fromkeys(rows) if i not in self.pages and self.in_stock(i) and (not kind_field or self.owned_by(want, str((self.cards.get(i) or {}).get(kind_field) or "")))]
            pool.sort(key=lambda i: (-len(wanted & set(re.findall(r"[a-z0-9]{3,}", _pack(self.cards.get(i) or {}).lower()))), self.price(i) or 0))
            self.open_specs(pool[:max(0, min(2, last - self.at))], keep)
        self.log.append({"kind": "recovery", "missing": missing, "queries": queries[:4], "opened": len(self.pages) - before})
        return len(self.pages) > before

    def states_needs(self, ident: str, number: int) -> bool:
        """Whether an article's fields carry each of the want's demands word for word (bounds and exclusions count as not shown)."""
        block = _spec_block(self.pages.get(ident) or {}) or {}
        for owner, need in self.live_needs():
            if owner != number:
                continue
            prop, _, value = need.partition(":")
            words = set(re.sub(r"[,/;|]", " ", _plain(value)).split())
            if not words or words & {"not", "no", "least", "most", "under", "over", "unlike", "different", "min", "max", "minimum", "maximum", "same", "page", "view"}:
                return False
            if not any(_same_prop(_bag(prop), _bag(str(k))) and words <= set(re.sub(r"[,/;|]", " ", _plain(str(v))).split()) for k, v in block.items()):
                return False
        return True

    def fill_short_cart(self) -> bool:
        """Before a forced payment with several wants: trolley, for each want lacking a line, the lowest-priced opened, in-stock
        article of its kind whose fields state its demands. False when a want stays uncovered (a short order earns nothing)."""
        basket, kind_field = self.jobs.get("basket") or "", self.kind_field()
        if not basket or not kind_field or len(self.wants) < 2:
            return True
        held = [str((self.cards.get(i) or {}).get(kind_field) or "") for i in self.basket_ids()]
        bound = {n for pair in self.pairs for n in pair}
        for number, want in enumerate(self.wants, 1):
            if any(self.owned_by(want, label) for label in held):
                continue
            fits = [i for i in self.pages if i not in self.basket and self.in_stock(i) and self.price(i) is not None
                    and self.owned_by(want, str((self.cards.get(i) or {}).get(kind_field) or "")) and self.states_needs(i, number)]
            fits.sort(key=lambda i: self.price(i) or 0.0)
            pick = fits[0] if fits and number not in bound else None
            if pick and isinstance(self.limit, (int, float)):
                scope = (self.intake.get("limit") or {}).get("scope") if isinstance(self.intake.get("limit"), dict) else None
                spent = sum((self.price(i) or 0.0) * self.units_of(i) for i in self.basket_ids())
                cost = (self.price(pick) or 0.0) * (1 if scope == "each" else want.count)
                if cost + (0.0 if scope == "each" else spent) > self.limit + 1e-9:
                    pick = None
            calls = self.basket_calls({**{i: self.basket.get(i, 1) for i in self.basket_ids()}, pick: want.count}, []) if pick else None
            if not calls or len(calls) + 1 > self.room():
                return False
            for args in calls:
                faults: list[str] = []
                args = _conform(args, self.specs.get(basket), "args", faults)
                if faults:
                    return False
                reply = self.do_step(basket, args)
                if reply is None:
                    return False
                self.track_basket(basket, args)
                self.digest(basket, reply.get("observation"))
            self.log.append({"kind": "filled_line", "want": number, "line": pick})
        return True

    def drop_want(self, number: int) -> None:
        """Remove a want the note never ordered, shifting every reference to later wants down by one."""
        if len(self.wants) < 2 or not 1 <= number <= len(self.wants):
            return
        gone = self.wants.pop(number - 1)
        shift = lambda n: n - 1 if n > number else n
        self.pairs = [(shift(p), shift(h)) for p, h in self.pairs if number not in (p, h)]
        self.conditionals = [dict(c, item=shift(c["item"]), about=shift(c["about"])) for c in self.conditionals if number not in (c["item"], c["about"])]
        self.pointers = [(shift(n), p) for n, p in self.pointers if n != number]
        self.hints = [dict(h, item=shift(h["item"])) for h in self.hints if h["item"] != number]
        self.log.append({"kind": "want_dropped", "what": gone.what, "needs": gone.needs})

    def merge_twins(self) -> dict[int, int]:
        """Two wants on the same shop label with no clashing needs are one product listed twice; fold them."""
        kept: list[Want] = []
        became: dict[int, int] = {}
        for old, want in enumerate(self.wants, 1):
            twin = next((k for k in kept if want.label.strip() and _plain(k.label) == _plain(want.label)
                         and not any(_plain(a.split(":")[0]) == _plain(b.split(":")[0]) and _plain(a.partition(":")[2]) != _plain(b.partition(":")[2]) for a in k.needs for b in want.needs)
                         and k.cap == want.cap), None)
            if twin is None:
                kept.append(want)
                became[old] = len(kept)
            else:
                became[old] = kept.index(twin) + 1
                twin.needs += [n for n in want.needs if n not in twin.needs]
                self.log.append({"kind": "wants_merged", "into": twin.what, "dropped": want.what})
        self.wants = kept
        return became

    def add_pair(self, pair: Any, renumber: dict[int, int] | None = None) -> None:
        if not isinstance(pair, dict):
            return
        part, host = _whole(pair.get("part")) or 0, _whole(pair.get("host")) or 0
        if renumber:
            part, host = renumber.get(part, 0), renumber.get(host, 0)
        if part and host and part != host and 1 <= part <= len(self.wants) and 1 <= host <= len(self.wants) and (part, host) not in self.pairs:
            self.pairs.append((part, host))

    def add_conditional(self, rule: Any, renumber: dict[int, int] | None = None) -> None:
        if not isinstance(rule, dict) or not str(rule.get("need") or "").strip() or not str(rule.get("test") or "").strip():
            return
        item, about = _whole(rule.get("item")) or 0, _whole(rule.get("about")) or 0
        if renumber:
            item, about = renumber.get(item, 0), renumber.get(about, 0)
        if not (1 <= item <= len(self.wants)) or not (1 <= about <= len(self.wants)):
            return
        mode = "except_if" if str(rule.get("mode") or "").strip().lower() in ("except_if", "unless", "except") else "only_if"
        entry = {"item": item, "need": str(rule["need"]).strip(), "mode": mode, "about": about, "test": str(rule["test"]).strip(), "state": None}
        if not any(c["item"] == entry["item"] and c["need"] == entry["need"] and c["test"] == entry["test"] for c in self.conditionals):
            self.conditionals.append(entry)
            # a conditional need listed as a plain need as well would be enforced unconditionally
            want = self.wants[item - 1]
            want.needs = [n for n in want.needs if _plain(n) != _plain(entry["need"])]

    def add_pointer(self, text: str) -> None:
        if not text:
            return
        if not any(text in w.needs for w in self.wants):
            # the pointer copied loosely: the need on the same property stands in for it
            prop = _plain(text.split(":")[0])
            text = next((n for w in self.wants for n in w.needs if _plain(n.split(":")[0]) == prop), text)
        number = next((i for i, w in enumerate(self.wants, 1) if text in w.needs), 1)
        if (number, text) not in self.pointers:
            self.pointers.append((number, text))

    def proof_pass(self) -> None:
        """A second model audits the draft needs against the message's own words; unsupported needs are dropped,
        misread ones mended, missing ones (with evidence in the message) added."""
        if self.dry or not self.ask_text.strip() or self.elapsed() > 75:
            self.log.append({"kind": "proof_skipped", "elapsed": round(self.elapsed())}) if not self.dry else None
            return
        drafted = []
        for number, want in enumerate(self.wants, 1):
            drafted.append(f"want {number}: {want.what} (count {want.count}" + (f", per-unit ceiling {want.cap:g}" if want.cap else "") + ")\n  needs: " + ("; ".join(want.needs) or "none"))
        extra = ""
        if self.pairs:
            extra += "\npairs: " + "; ".join(f"want {p} must suit want {h}" for p, h in self.pairs)
        if self.conditionals:
            extra += "\nconditional: " + "; ".join(f"want {c['item']}: {c['need']} ({c['mode'].replace('_', ' ')} want {c['about']} has {c['test']})" for c in self.conditionals)
        limit = (self.intake.get("limit") or {}).get("amount") if isinstance(self.intake.get("limit"), dict) else None
        prompt = PROOF + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + "\n\nDRAFT:\n" + "\n".join(drafted) + f"\nlimit: {limit}" + extra
        got = self.json_call(prompt, [BIG, SMALL], 3500) or {}
        self.log.append({"kind": "proof", "got": _trim(got, 160)})
        applied: list[str] = []
        for entry in _listed(got.get("wants")):
            if not isinstance(entry, dict):
                continue
            number = _whole(entry.get("item")) or 0
            if not 1 <= number <= len(self.wants):
                continue
            want = self.wants[number - 1]
            for ruling in _listed(entry.get("needs")):
                if not isinstance(ruling, dict):
                    continue
                need = str(ruling.get("need") or "").strip()
                verdict = str(ruling.get("ruling") or "keep").strip().lower()
                evidence = str(ruling.get("evidence") or "").strip()
                if need not in want.needs or verdict == "keep":
                    continue
                pointed = need in [p for _, p in self.pointers]
                if verdict in ("story", "name") and not pointed:
                    # the judge saw the whole message; a quoted evidence string that is not in the message is ignored
                    if evidence and not _in_text(evidence, self.ask_text, 0.8) and verdict == "story":
                        continue
                    want.needs.remove(need)
                    applied.append(f"dropped ({verdict}) want {number}: {need}")
                elif verdict in ("other_item", "misread"):
                    want.needs.remove(need)
                    self.pointers = [(n, p) for n, p in self.pointers if not (n == number and p == need)]
                    applied.append(f"removed ({verdict}) want {number}: {need}")
            for fix in _listed(entry.get("fix")):
                if not isinstance(fix, dict) or not str(fix.get("need") or "").strip():
                    continue
                target = _whole(fix.get("item")) or number
                if 1 <= target <= len(self.wants):
                    need = str(fix["need"]).strip()
                    if need not in self.wants[target - 1].needs and _in_text(need.partition(":")[2] or need, self.ask_text, 0.5):
                        self.wants[target - 1].needs.append(need)
                        applied.append(f"mended want {target}: {need}")
            for miss in _listed(entry.get("missing")):
                if not isinstance(miss, dict) or not str(miss.get("need") or "").strip():
                    continue
                need = str(miss["need"]).strip()
                evidence = str(miss.get("evidence") or "").strip()
                prop = _plain(need.split(":")[0])
                if need in want.needs or any(_plain(n.split(":")[0]) == prop for n in want.needs):
                    continue
                if evidence and _in_text(evidence, self.ask_text, 0.8) and _in_text(need.partition(":")[2] or need, self.ask_text, 0.6):
                    want.needs.append(need)
                    applied.append(f"added want {number}: {need}")
                    if re.search(r"\b(?:page|listing|same as|unlike|whatever)\b", need, re.I):
                        self.add_pointer(need)
            count = _whole(entry.get("count"))
            if count and count > 0 and count != want.count and _count_backed(count, [want.what, want.label], self.ask_text):
                applied.append(f"count of want {number}: {want.count} -> {count}")
                want.count = count
            cap = _money(entry.get("cap_each"))
            if cap is not None and want.cap is None and re.search(r"\b" + re.escape(f"{cap:g}") + r"\b", self.ask_text):
                want.cap = cap
                applied.append(f"per-unit ceiling of want {number}: {cap:g}")
        amount = _money(got.get("limit"))
        if amount is not None and limit is None and re.search(r"\b" + re.escape(f"{amount:g}") + r"\b", self.ask_text):
            self.intake["limit"] = {"amount": amount, "unit": (self.intake.get("limit") or {}).get("unit") if isinstance(self.intake.get("limit"), dict) else None, "scope": "order"}
            applied.append(f"limit: {amount:g}")
        for pair in _listed(got.get("pairs")):
            before = len(self.pairs)
            self.add_pair(pair)
            if len(self.pairs) > before:
                applied.append(f"pair added: {self.pairs[-1]}")
        for rule in _listed(got.get("conditional")):
            before = len(self.conditionals)
            self.add_conditional(rule)
            if len(self.conditionals) > before:
                applied.append(f"conditional added: {self.conditionals[-1]['need']}")
        # a want the auditor calls spurious goes only when no ordering sentence names it either
        for number in sorted({_whole(x) or 0 for x in _listed(got.get("spurious"))}, reverse=True):
            if 1 <= number <= len(self.wants) and len(self.wants) > 1:
                want = self.wants[number - 1]
                if not _ordered([want.what, want.label], self.ask_text):
                    applied.append(f"dropped want {number} ({want.what}): never ordered")
                    self.drop_want(number)
        if applied:
            self.log.append({"kind": "proof_applied", "changes": applied})

    def order_sentences(self) -> list[str]:
        if self.sifted and self.sifted.get("orders"):
            return list(self.sifted["orders"])
        return [x for x in re.split(r"(?<=[.?!])\s+", self.ask_text) if ORDER_RX.search(x)]

    def want_named(self, phrase: str) -> int:
        """Which want a phrase from the note points at (its kind words or shop label), 0 when none or several."""
        words = {w for w in _cores(phrase) if len(w) >= 3}
        if not words:
            return 0
        hits = [n for n, w in enumerate(self.wants, 1) if words & {x for x in _cores(w.what) | _cores(w.label or "") if len(x) >= 3}]
        return hits[0] if len(hits) == 1 else 0

    def count_guard(self) -> None:
        """An explicit "N x <article>" in an ordering sentence fixes that want's unit count (race 17: a draft read
        "2 x cpu" as one unit and the order fell short)."""
        rx = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\s*(?:x|×)\s+([a-z][a-z &'/-]{2,40}?)(?=\s+(?:where|with|in|whose|that|which|at|from|for|of|and)\b|[,;.!?]|$)", re.I)
        names = {alt.strip(): k for k, v in NUMBER_WORDS.items() for alt in str(v).split("|")}
        for sentence in self.order_sentences():
            for hit in rx.finditer(sentence):
                raw = hit.group(1).lower()
                count = int(raw) if raw.isdigit() else names.get(raw)
                number = self.want_named(hit.group(2))
                if not count or not number or self.wants[number - 1].count == count:
                    continue
                self.log.append({"kind": "count_fixed", "want": number, "from": self.wants[number - 1].count, "to": count, "words": hit.group(0)[:60]})
                self.wants[number - 1].count = count

    def colour_guard(self) -> None:
        """"<article> in <colour>" or "<colour> <article>" inside an ordering sentence is a colour demand on that want
        when the draft holds none (race 17: "3 x fishing lines in black" lost its colour and the order failed)."""
        spelling = "colour" if re.search(r"\bcolour", self.ask_text, re.I) else "color"
        for sentence in self.order_sentences():
            for hit in re.finditer(r"([a-z][a-z &'/-]{2,40}?)\s+in\s+" + COLOUR_RX + r"\b|" + COLOUR_RX + r"\s+([a-z][a-z &'/-]{2,40}?)(?=[,;.!?]|\s+(?:where|with|whose|that|which|at|from|for|and)\b)", sentence, re.I):
                phrase = hit.group(1) or hit.group(4) or ""
                value = (hit.group(2) or hit.group(3) or "").lower()
                before = sentence[max(0, hit.start() - 14):hit.start()].lower()
                if not value or re.search(r"\b(?:never|no|not|unlike|except|than)\s*$", before):
                    continue
                number = self.want_named(phrase)
                if not number:
                    continue
                want = self.wants[number - 1]
                if any(re.search(r"colou?r", need.split(":", 1)[0], re.I) for need in want.needs) or any(re.search(r"colou?r", f, re.I) for f in want.free):
                    continue
                want.needs.append(f"{spelling}: {value}")
                self.log.append({"kind": "colour_added", "want": number, "value": value, "words": hit.group(0)[:60]})

    def noise_guard(self) -> None:
        """A need with value words found nowhere in the message was invented; a need on a want's own name is noise."""
        for number, want in enumerate(self.wants, 1):
            keep = []
            for need in want.needs:
                prop, _, value = need.partition(":")
                pointed = need in [p for _, p in self.pointers]
                if not value.strip():
                    keep.append(need)
                    continue
                if not pointed and not _in_text(value, self.ask_text, 0.5):
                    self.log.append({"kind": "noise_dropped", "want": number, "need": need})
                    continue
                keep.append(need)
            want.needs = keep

    @staticmethod
    def twin_labels(draft: dict[str, Any]) -> bool:
        labels = [_plain(str(e.get("label") or "")) for e in _listed(draft.get("wants")) if isinstance(e, dict)]
        labels = [s for s in labels if s]
        return len(labels) != len(set(labels))

    def split_pointer(self) -> None:
        """Demands that merely single out the article a pointer cites (misread as demands on the want) fold into that pointer."""
        for number, want in enumerate(self.wants, 1):
            refs = [need for n, need in self.pointers if n == number and need in want.needs]
            rest = [n for n in want.needs if n not in refs]
            if not refs or not rest or self.elapsed() > 90:
                continue
            prompt = (SPLIT + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + f"\n\nPRODUCT: {want.what}\nREQUIREMENTS REFERRING TO ANOTHER PRODUCT:\n"
                      + "\n".join(refs) + "\nOTHER REQUIREMENTS:\n" + "\n".join(rest))
            got = self.json_call(prompt, [BIG, SMALL], 1500) or {}
            moved = [n for n in rest if n in [str(x).strip() for x in _listed(got.get("describe_other"))]]
            if not moved:
                continue
            want.needs = [n for n in want.needs if n not in moved]
            joined = refs[0] + " (that product: " + "; ".join(moved) + ")"
            want.needs = [joined if n == refs[0] else n for n in want.needs]
            self.pointers = [(n, joined if (n, need) == (number, refs[0]) else need) for n, need in self.pointers]
            self.log.append({"kind": "split", "want": number, "moved": moved})

    def story_only(self, number: int, need: str) -> bool:
        """A long worded need on a property no opened product of that kind names: the customer's description, not a test."""
        if need in [p for _, p in self.pointers] or need in [str(x) for x in _listed(self.intake.get("pointers"))]:
            return False
        prop, _, value = need.partition(":")
        if not prop.strip() or len(re.findall(r"[a-z0-9]+", value.lower())) < 4:
            return False
        if re.search(r"\b(?:same as|unlike|other than|shown|page|listing|view)\b", value, re.I):
            return False
        want = self.wants[number - 1]
        if any({w for w in _bag(other.what) if len(w) >= 4} & _bag(value) for other in self.wants if other is not want):
            return False
        kind_field = self.kind_field()
        keys: set[str] = set()
        read = 0
        for record in self.pages.values():
            if kind_field and not self.owned_by(want, str(record.get(kind_field) or "")):
                continue
            block = _spec_block(record) or {}
            if block:
                read += 1
                keys |= {_plain(k) for k in block}
        return read >= 3 and not any(_bag(k) & _bag(prop) for k in keys)

    # ---------- the sheet shown to the walker ----------
    def sheet(self) -> str:
        parts = []
        many = len(self.wants) > 1
        for number, want in enumerate(self.wants, 1):
            live = [n for n in want.needs + want.told if n not in want.dropped]
            worded = [n for n in live if self.story_only(number, n)]
            needs = [n + (f" (its page gives: {self.pointed_values[n]})" if n in self.pointed_values else "") for n in live if n not in worded]
            head = f"want {number}: {want.what}" if many else f"want: {want.what}"
            if want.label:
                head += f" (shop label: {want.label})"
            head += f", count {want.count}"
            if want.cap is not None:
                head += f", at most {want.cap:g} per unit"
            text = head + "; needs: " + ("; ".join(needs) if needs else "none beyond the product kind")
            if worded:
                text += "; the customer's own description, stated by no product attribute (decides nothing): " + "; ".join(worded)
            if want.free:
                text += "; customer does not mind: " + ", ".join(want.free)
            parts.append(text)
        for rule in self.conditionals:
            when = "only when" if rule["mode"] == "only_if" else "except when"
            state = {True: " [applies: trigger holds on the basket's product]", False: " [does not apply: trigger fails on the basket's product]"}.get(rule["state"], " [undecided until that product is read]")
            parts.append(f"conditional need for want {rule['item']}: {rule['need']}, {when} want {rule['about']}'s product has {rule['test']}{state}")
        for part, host in self.pairs:
            parts.append(f"want {part} must be compatible with the want {host} you pick (read both pages' own words)")
        draft = self.intake
        if self.limit is not None:
            box = draft.get("limit") if isinstance(draft.get("limit"), dict) else {}
            scope = "for each product" if box.get("scope") == "each" else "for the whole order"
            parts.append(f"limit: at most {self.limit:g} {box.get('unit') or ''} {scope}".replace("  ", " "))
        choose = self.choose()
        if choose == "cheapest":
            parts.append("choose: the cheapest product that satisfies everything (lowest price)")
        elif choose in ("max", "min"):
            parts.append(f"choose: the {'most' if choose == 'max' else 'least'} {draft.get('choose_by') or ''} among satisfying products".replace("  ", " "))
        else:
            parts.append("choose: any product that satisfies everything")
        if draft.get("stocked_only", True) is not False:
            parts.append("in-stock products only")
        for nice in _listed(draft.get("nice")):
            parts.append(f"nice-to-have (tie-breaker only): {nice}")
        for rule in _listed(draft.get("manner")):
            parts.append(f"instruction from the customer: {rule}")
        if self.fallback.get("quiet") is True:
            parts.append("if nothing passes: bow out while naming or suggesting no article")
        elif self.fallback.get("offer_nearest"):
            parts.append("if nothing satisfies: say so and offer the nearest product" + (f" ({self.fallback['rule']})" if self.fallback.get("rule") else "") + "; Wren words that remark itself")
        for chore in self.chores:
            when = {"pick": "before selecting", "order": "at payment time (Wren performs it as you pay)", "end": "at bow-out time (Wren performs it then)"}[chore["stage"]]
            parts.append(f"chore the customer asks for, {when}: {chore['what']}")
        view = self.spec_view()
        if view and (self.specs_by.get(view) is True or self.jobs.get("deal")):
            bare = sorted(t for t in self.page_views() | {self.jobs.get("deal") or ""} if t and self.specs_by.get(t) is False)
            pages = sorted(t for t in self.page_views() if t != view)
            parts.append(f"product attributes come from {view}" + (f"; {', '.join(bare)} show none" if bare else "") + (f"; a product's page (what a chain of pages or a page check means) is shown by {' or '.join(pages)}" if pages else ""))
        return "\n".join("- " + p for p in parts)

    def choose(self) -> str:
        mode = str(self.intake.get("choose") or "").strip().lower()
        if mode in ("cheapest", "lowest", "min_price"):
            return "cheapest"
        if mode in ("max", "most"):
            return "max"
        if mode in ("min", "least"):
            return "min"
        return "any"

    def extreme_pick(self) -> bool:
        return self.choose() in ("cheapest", "max", "min")

    def stamp(self) -> str:
        left = self.room()
        text = f"[Wren] step {self.at} of {self.step_cap} used, {left} left."
        if left <= self.keep_back() + 1:
            text += " FINAL MOVES: pay now if the trolley holds a line per want; bow out only where your searching proved some want has no passing article (exhausted moves prove nothing)."
        elif left <= 6 or (left <= len(self.wants) + 5 and self.checkouts == 0):
            text += " Few moves remain: make the trolley hold your chosen lines and pay now."
        elif self.checkouts == 0 and left <= 12:
            text += " Moves are getting scarce: settle your picks, load the trolley and pay inside the next few moves."
        elif self.checkouts == 0 and not self.basket_moved and self.at >= self.step_cap // 2:
            text += " Half your moves are spent and the trolley is empty: trolley each want's passing article now (the site may yet sell one out)."
        live = {need for _, need in self.live_needs()}
        pending = [need for _, need in self.pointers if need in live and need not in self.pointed_values]
        if self.elapsed() >= TIME_SOFT - 50:
            text += f"\n[time] about {max(0, int(TIME_SOFT - self.elapsed()))} s of this episode remain for searching: settle your picks, load the trolley and pay within the next few moves (a run cut off by the clock earns nothing)."
        if pending and self.checkouts == 0:
            text += "\n[settle first: open the article each of these cites, plus every article its view forwards to; the last view carries the value] " + " | ".join(pending[:3])
        if self.leads:
            text += "\n[review candidates] " + ", ".join(self.leads)
        plan = self.pair_plan() if self.pairs and self.checkouts == 0 else ""
        if plan:
            text += "\n" + plan
        if self.changes:
            text += "\n[changes so far] " + " | ".join(self.changes[-4:])
        text += "\n[sheet]\n" + self.sheet()
        return text

    # ---------- records of products ----------
    def id_of(self, node: Any, in_list: bool = False, depth: int = 0) -> str | None:
        if not isinstance(node, dict) or len(node) < 3:
            return None
        if self.id_name and _looks_id(node.get(self.id_name), True):
            return node[self.id_name].strip()
        nested = any(isinstance(v, dict) for v in node.values())
        if not (in_list or depth == 0 or nested):
            return None
        found = _id_field([node])
        if found and ("id" in str(found).lower() or node[found].strip() in self.cards):
            return node[found].strip()
        return None

    def learn_id(self, node: list[Any]) -> bool:
        rows = [n for n in node if isinstance(n, dict) and len(n) >= 3]
        if not rows or len(rows) < len(node) / 2:
            return False
        if self.id_name and all(_looks_id(r.get(self.id_name), True) for r in rows):
            return True
        found = _id_field(rows)
        if found:
            self.id_name = found
            return True
        return False

    def ingest(self, observation: Any) -> None:
        """Fold every product record inside a result into the cards (rows) and pages (records with attributes)."""
        def walk(node: Any, depth: int, in_list: bool = False) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.id_of(node, in_list, depth)
                if ident:
                    if ident not in self.cards:
                        self.arrival.append(ident)
                        self.cards[ident] = {}
                    old = self.cards[ident]
                    if any(old.get(k) is True and v is False for k, v in node.items()):
                        self.unbuyable.add(ident)
                    elif any(old.get(k) is False and v is True for k, v in node.items()):
                        self.unbuyable.discard(ident)
                    self.cards[ident].update(node)
                    if _spec_block(node):
                        self.pages[ident] = dict(self.cards[ident])
                    elif ident in self.pages:
                        self.pages[ident].update(node)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                rows = self.learn_id(node)
                for value in node:
                    walk(value, depth + 1, rows)
        walk(observation, 0)

    def enum_labels(self) -> set[str]:
        found: set[str] = set()

        def walk(schema: Any, depth: int) -> None:
            if depth > 6 or not isinstance(schema, dict):
                return
            for option in schema.get("enum") or []:
                if isinstance(option, str):
                    found.add(option.strip().lower())
            for value in schema.values():
                if isinstance(value, dict):
                    walk(value, depth + 1)
        for schema in self.specs.values():
            walk(schema, 0)
        return found

    def kind_field(self) -> str | None:
        """The row field that holds the category label: short strings, present on most rows, few distinct values."""
        counts: dict[str, set[str]] = {}
        present: dict[str, int] = {}
        for row in self.cards.values():
            for key, value in row.items():
                if isinstance(value, str) and 0 < len(value) <= 40 and not re.search(r"\d{3}", value):
                    counts.setdefault(key, set()).add(value)
                    present[key] = present.get(key, 0) + 1
        total = max(1, len(self.cards))
        known = self.enum_labels()
        best = None
        for key, values in counts.items():
            if present[key] < 0.8 * total or len(values) > 40:
                continue
            listed = sum(1 for v in values if v.strip().lower() in known) / len(values)
            mean = sum(len(v) for v in values) / len(values)
            score = (-listed, len(values), -present[key], -mean)
            if best is None or score < best[0]:
                best = (score, key)
        return best[1] if best else None

    def field_table(self) -> list[tuple[str, str, int, list[str]]]:
        """Per department tag: field keys observed on opened articles, how many carry each, sample values."""
        kind_field = self.kind_field()
        groups: dict[str, dict[str, Any]] = {}
        for ident in self.arrival:
            record = self.pages.get(ident)
            if not record:
                continue
            block = _spec_block(record) or {}
            label = str(record.get(kind_field) or "products") if kind_field else "products"
            group = groups.setdefault(label, {"n": 0, "keys": {}})
            group["n"] += 1
            done: set[str] = set()
            for key, value in block.items():
                name = _plain(key)
                if not name or name in done:
                    continue
                done.add(name)
                slot = group["keys"].setdefault(name, {"n": 0, "values": [], "per": {}})
                slot["n"] += 1
                text = str(value).strip().lower()
                slot["per"][ident] = text
                if text not in slot["values"]:
                    slot["values"].append(text)
        wanted = [(_plain(w.label or ""), _plain(w.what), {x for x in _bag(w.label or "") | _bag(w.what) if len(x) >= 4}) for w in self.wants]
        matched = {label for label in groups if any(_plain(label) in (lab, what) or {x for x in _bag(label) if len(x) >= 4} & long for lab, what, long in wanted) or self.owner(label)}
        if matched:
            groups = {label: group for label, group in groups.items() if label in matched}
        table = []
        for label, group in sorted(groups.items(), key=lambda kv: -kv[1]["n"]):
            kept: list[tuple[str, dict[str, Any]]] = []
            for name, slot in sorted(group["keys"].items(), key=lambda kv: (-kv[1]["n"], len(kv[0]))):
                values = list(slot["per"].values())
                if not values:
                    continue
                mean = sum(len(v) for v in values) / len(values)
                if mean > 40 or sum(1 for v in values if " / " in v) * 2 >= len(values):
                    continue
                recur = sum(len(set(re.findall(r"[a-z]{3,}", v)) & set(re.findall(r"[a-z]{3,}", " ".join(o for o in values if o is not v)))) > 0 for v in values)
                if len(values) >= 4 and len(set(values)) == len(values) and mean > 12 and recur * 2 < len(values):
                    continue
                twin = next((spec for other, spec in kept if self.alias_pair(other, spec, name, slot)), None)
                if twin is not None:
                    for ident, text in slot["per"].items():
                        twin["per"].setdefault(ident, text)
                        if text not in twin["values"]:
                            twin["values"].append(text)
                    twin["n"] = len(twin["per"])
                    continue
                kept.append((name, slot))
            kept.sort(key=lambda kv: -kv[1]["n"])
            for name, slot in kept[:14]:
                table.append((label, name, slot["n"], [v[:40] for v in slot["values"][:4]]))
        return table

    def flag_other(self, label: str, name: str, values: list[str]) -> str:
        """Mark a key that shares words with a stated need but shows other values: a different property."""
        number = self.owner(label)
        if not number:
            return ""
        for need in self.wants[number - 1].needs:
            prop, _, value = need.partition(":")
            if re.search(BOUND_RX, value, re.I):
                continue
            words = set(re.findall(r"[a-z0-9]{3,}", value.lower()))
            shown = set(re.findall(r"[a-z0-9]{3,}", " ".join(values).lower()))
            if prop.strip() and _plain(prop) != name and _bag(prop) & _bag(name) and words and not words <= shown:
                return f" [not the stated {prop.strip()}: another property]"
        return ""

    @staticmethod
    def alias_pair(name: str, spec: dict[str, Any], other: str, slot: dict[str, Any]) -> bool:
        """Two keys that spell a single property differently (their values agree on shared products)."""
        a, b = set(name.split()), set(other.split())
        if not a or not b or not a & b:
            return False

        def agree(x: str, y: str) -> bool:
            wx, wy = re.findall(r"[a-z0-9]+", x.lower()), re.findall(r"[a-z0-9]+", y.lower())
            short, long = (wx, wy) if len(wx) <= len(wy) else (wy, wx)
            return x == y or (bool(short) and set(short) <= set(long) and len(long) <= 2 * len(short) + 2)
        both = [i for i in spec["per"] if i in slot["per"] and spec["per"][i].strip() and slot["per"][i].strip()]
        same = sum(1 for i in both if agree(spec["per"][i], slot["per"][i]))
        if not (a <= b or b <= a):
            return len(both) >= 2 and same == len(both)
        if both:
            return same * 2 > len(both)
        return bool((set(spec["values"]) & set(slot["values"])) - {""})

    def want_groups(self) -> list[list[str]]:
        kind_field = self.kind_field()
        groups: dict[str, list[str]] = {}
        for ident in self.arrival:
            label = str(self.cards[ident].get(kind_field) or "") if kind_field else ""
            groups.setdefault(label, []).append(ident)
        wanted = [(_plain(w.label or ""), _plain(w.what), {x for x in _bag(w.label or "") | _bag(w.what) if len(x) >= 4}) for w in self.wants]
        matched = [ids for label, ids in groups.items() if any(_plain(label) in (lab, what) or {x for x in _bag(label) if len(x) >= 4} & long for lab, what, long in wanted) or self.owner(label)]
        pool = matched or list(groups.values())
        return sorted(pool, key=lambda ids: -len(ids))[:max(1, len(self.wants))]

    def id_slot(self, tool: str | None) -> str | None:
        """The single free-text parameter of a one-product tool (where the product id goes), if the tool has exactly that."""
        schema = self.specs.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _types(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        if not tool or not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def guess_field(self, pattern: str, rows: int = 0) -> str | None:
        """A row key matching the pattern, present on most cards: the price or stock key when no layout was learnt."""
        counts: dict[str, int] = {}
        for row in self.cards.values():
            for key in row:
                if re.search(pattern, _plain(str(key)), re.I):
                    counts[key] = counts.get(key, 0) + 1
        best = max(counts, key=lambda k: counts[k], default=None)
        return best if best and counts[best] * 2 >= max(1, len(self.cards)) else None

    def price(self, ident: str) -> float | None:
        field = self.price_field or self.guess_field(r"^(?:price|cost|amount|unit price|price cents|price usd)$|price")
        value = (self.cards.get(ident) or {}).get(field) if field else None
        if isinstance(value, dict):
            value = next((v for v in value.values() if isinstance(v, (int, float)) and not isinstance(v, bool)), None)
        if isinstance(value, str) and re.fullmatch(r"\s*\d+(\.\d+)?\s*", value):
            value = float(value)
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    def in_stock(self, ident: str) -> bool:
        if self.gone_now(ident):
            return False
        field = self.stock_field or self.guess_field(r"stock|on hand|available|availab|in store|purchasable|buyable")
        value = (self.cards.get(ident) or {}).get(field) if field else None
        if isinstance(value, str):
            low = value.strip().lower()
            if low in ("false", "no", "out of stock", "sold out", "unavailable", "none", "0"):
                return False
            return True
        return not (value is False or (isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0))

    def cards_in(self, observation: Any) -> list[tuple[str, dict[str, Any]]]:
        if isinstance(observation, dict):
            ident = self.id_of(observation)
            if ident:
                return [(ident, observation)]
            found: list[tuple[str, dict[str, Any]]] = []
            for value in observation.values():
                if isinstance(value, list):
                    found += [(i, r) for r in value if isinstance(r, dict) for i in [self.id_of(r, True)] if i]
                elif isinstance(value, dict):
                    found += [(i, r) for v in value.values() if isinstance(v, list) for r in v if isinstance(r, dict) for i in [self.id_of(r, True)] if i]
            return found
        if isinstance(observation, list):
            return [(i, r) for r in observation if isinstance(r, dict) for i in [self.id_of(r, True)] if i]
        return []

    def mark_view(self, name: str, observation: Any) -> None:
        for ident, record in self.cards_in(observation):
            keys = set(record)
            self.view_fields.setdefault(name, set()).update(keys)
            epoch, have = self.card_fields.get(ident, (-1, set()))
            self.card_fields[ident] = (self.epoch, (have if epoch == self.epoch else set()) | keys)

    def already_shown(self, name: str, ident: str) -> bool:
        epoch, have = self.card_fields.get(ident, (-1, set()))
        keys = self.view_fields.get(name)
        return bool(keys) and epoch == self.epoch and keys <= have and ident not in self.vanished

    def layout_of(self, name: str, observation: Any) -> dict[str, Any] | None:
        if name in self.layouts:
            return self.layouts[name]
        if not isinstance(observation, dict):
            return None
        lists = [k for k, v in observation.items() if isinstance(v, list) and len(v) >= 2 and all(isinstance(r, dict) for r in v)]
        if not lists:
            return None
        self.layouts[name] = None
        sample = {k: v[:3] if k in lists else v for k, v in observation.items()}
        prompt = LAYOUT + "\n\nTOOL: " + name + " | parameters: " + _pack(self.specs.get(name) or {}) + "\n\nREPLY (opening rows only):\n" + _pack(_trim(sample, 120))[:4000]
        got = self.json_call(prompt, [SMALL, BIG], 2500) or {}
        rows = got.get("rows") if got.get("rows") in lists else None
        if not rows:
            return None
        first = observation[rows][0]
        props = (self.specs.get(name) or {}).get("properties") or {}

        def named(key: Any, where: dict[str, Any]) -> str | None:
            return key if isinstance(key, str) and key in where else None
        layout = {"rows": rows, "count": named(got.get("count"), observation), "next": named(got.get("next"), observation), "next_arg": named(got.get("next_arg"), props), "price": named(got.get("price"), first), "stock": named(got.get("stock"), first)}
        if layout["count"] and (isinstance(observation[layout["count"]], bool) or not isinstance(observation[layout["count"]], int)):
            layout["count"] = None
        self.layouts[name] = layout
        self.price_field = self.price_field or layout["price"]
        self.stock_field = self.stock_field or layout["stock"]
        self.log.append({"kind": "layout", "tool": name, "layout": layout})
        return layout

    def page_plan(self, name: str, observation: Any) -> tuple[int, int, str, Any] | None:
        layout = self.layout_of(name, observation)
        if not layout or not layout["count"] or not layout["next"] or not layout["next_arg"] or not isinstance(observation, dict):
            return None
        rows, count, value = observation.get(layout["rows"]), observation.get(layout["count"]), observation.get(layout["next"])
        if not isinstance(rows, list) or not rows or isinstance(count, bool) or not isinstance(count, int):
            return None
        if value is not None and (isinstance(value, bool) or not isinstance(value, (str, int))):
            return None
        return count, len(rows), layout["next_arg"], value

    def ascending(self, observation: Any) -> bool:
        prices = [self.price(i) for i, _ in self.cards_in(observation)]
        return len(prices) >= 3 and None not in prices and all(a <= b for a, b in zip(prices, prices[1:]))

    def auto_page(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Under an extreme selection rule, pull a short result's further pages without help (a move per page)."""
        self.paged_seen = []
        plan = self.page_plan(name, observation) if self.extreme_pick() else None
        if not plan:
            return ""
        count, shown, param, value = plan
        given = args.get(param)
        later = given not in (None, 0, 1) if isinstance(given, int) and not isinstance(given, bool) else bool(str(given or "").strip())
        if value is None or later or count <= shown:
            return ""
        if self.choose() == "cheapest" and self.ascending(observation):
            return ""
        pages = -(-(count - shown) // shown)
        if pages > 2 and self.self_paged + 2 <= SELF_PAGES and self.room() - self.keep_back() - 2 >= 12:
            pages = 2
        if pages > 2 or self.self_paged + pages > SELF_PAGES or self.room() - self.keep_back() - pages < 12:
            return f"\n[Wren] This search matched {count} products and showed {shown}, in no price order. The choosing rule counts every satisfying product: search with narrower words (the need's value) or page on with {param}={_pack(value)}."
        texts = []
        for _ in range(pages):
            faults: list[str] = []
            call = _conform({**args, param: value}, self.specs.get(name), "args", faults)
            if faults:
                break
            reply = self.do_step(name, call)
            if reply is None or self.over:
                break
            self.self_paged += 1
            seen = reply.get("observation")
            self.digest(name, seen)
            self.paged_seen.append(seen)
            shown_text = self.render(seen)
            self.replay[name + _pack(call)] = (self.epoch, self.at, shown_text)
            texts.append(f"{param}={_pack(value)}: {shown_text}")
            after = self.page_plan(name, seen)
            value = after[3] if after else None
            if value is None:
                break
        self.log.append({"kind": "pages", "tool": name, "read": len(texts)})
        rest = f"\n[Wren] {count} products matched and Wren has read {shown * (len(texts) + 1)} rows of them; narrow the search with the need's value or page on to see the rest." if value is not None and texts else ""
        return "\n[Wren fetched this search's remaining pages unaided, a move apiece]\n" + "\n".join(texts) + rest if texts else ""

    def cap_check(self, name: str, args: dict[str, Any], observation: Any) -> str:
        """Warn when a search's price ceiling argument was plainly ignored by the shop."""
        props = (self.specs.get(name) or {}).get("properties") or {}

        def priced(key: str) -> bool:
            spec = props.get(key) if isinstance(props.get(key), dict) else {}
            return bool(re.search(r"money|spend|budget|amount|cost|price", key + " " + str(spec.get("description") or ""), re.I))
        caps = [k for k, v in args.items() if k in props and isinstance(v, (int, float)) and not isinstance(v, bool) and re.search(r"up_to|upto|cap|limit|below|under|most|max", k, re.I) and priced(k)]
        if not caps or not self.price_field:
            return ""
        caps.sort(key=lambda k: args[k])
        cap = args[caps[0]]
        over = [i for i, _ in self.cards_in(observation) if (self.price(i) or 0) > cap]
        if not over:
            return ""
        modes = {k: v for k, v in args.items() if isinstance(v, str) and isinstance((props.get(k) or {}).get("enum"), list)}
        others = {k: [o for o in props[k]["enum"] if o != v] for k, v in modes.items()}
        hint = "; browse the department using " + ", ".join(f"{k}={_pack(o[0])}" for k, o in others.items() if o) + f" plus the same {caps[0]} so only cheaper articles show" if any(others.values()) else ""
        return f"\n[Wren] This search did not apply {caps[0]}={cap:g}: {len(over)} of its rows cost more{hint}."

    def cheap_rows(self, observations: list[Any]) -> str:
        if self.choose() != "cheapest" or not self.price_field:
            return ""
        found = {i for seen in observations for i, _ in self.cards_in(seen)}
        rows = [(self.price(i), i) for i in found if self.in_stock(i) and self.price(i) is not None]
        if len(rows) < 2:
            return ""
        return "\n[Wren] in-stock hits here, cheapest leading: " + ", ".join(f"{i} ({p:g})" for p, i in sorted(rows)[:6])

    def noun_of(self, number: int) -> str:
        """The want's searchable noun: its department tag when seen, else the last words of its name."""
        want = self.wants[number - 1]
        words = [w for w in re.findall(r"[A-Za-z][A-Za-z'&-]+", want.label or "") if len(w) >= 3]
        if not words:
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z'&-]+", want.what) if len(w) >= 3 and not re.fullmatch(r"(?i)x|of|the|pack|packs|pair|pairs|set|sets", w)]
        return " ".join(words[-2:])

    def hunt_queries(self, number: int) -> list[str]:
        """Queries one want deserves a move for, likeliest first: a series it must belong to with its leading value,
        all its demand values with the noun, each value with the noun, each value alone, the noun alone."""
        noun = self.noun_of(number)
        values = self.need_values(number)[:3]
        tries: list[list[str]] = [[fam, values[0] if values else "", noun] for fam in self.series_for(number)[:2]]
        tries += [values[:2] + [noun]] + [[v, noun] for v in values[:2]] + [[v] for v in values[:2]] + ([[noun]] if noun else [])
        out: list[str] = []
        for words in tries:
            query = " ".join(w for w in words if w).strip()
            if query and query.lower() not in out and query.lower() not in self.hunt_tried:
                out.append(query.lower())
        return out

    def run_hunt(self, queries: list[tuple[int, str]], room: int) -> dict[int, list[str]]:
        """Issue at most `room` searches (a move apiece) with the search tool in its usual shape; the fresh rows per want."""
        plan = self.query_pattern()
        found: dict[int, list[str]] = {}
        if not plan or room <= 0:
            return found
        tool, base, field = plan
        spent = 0
        for number, words in queries:
            if spent >= room or self.over or self.late():
                break
            faults: list[str] = []
            args = _conform({**base, field: words}, self.specs.get(tool), "args", faults)
            if faults:
                continue
            self.hunt_tried.add(words.lower())
            reply = self.do_step(tool, args)
            spent += 1
            if reply is None:
                break
            self.digest(tool, reply.get("observation"))
            self.queries = (self.queries + [dict(args)])[-8:]
            found.setdefault(number, [])
            found[number] += [i for i, _ in self.cards_in(reply.get("observation")) if i not in found[number]]
        if spent:
            self.log.append({"kind": "hunt", "searches": spent, "rows": sum(len(v) for v in found.values())})
        return found

    def wider_search(self) -> str:
        """Before a cheapest payment: search on by itself (demand values with the noun, then each value alone), because
        an opening search shows a single page, and a kind's lowest-priced row often sits on a later one."""
        if self.wider_block or self.choose() != "cheapest" or self.room() < 14 + 2 * len(self.wants) or self.late():
            return ""
        self.wider_block = True
        before = len(self.cards)
        room = min(HUNT_STEPS, self.room() - self.keep_back() - 6 - 2 * len(self.wants))
        queries: list[tuple[int, str]] = []
        for number in range(1, len(self.wants) + 1):
            queries += [(number, q) for q in self.hunt_queries(number)[:3]]
        self.run_hunt(queries, room)
        fresh = len(self.cards) - before
        lines = [f"{i} at {self.price(i):g}" for i in self.basket_ids() if self.price(i) is not None]
        return (f"Held back (no step charged). The rule is cheapest and {self.room()} moves remain. Wren ran further searches by itself (a move apiece) and they surfaced {fresh} rows not seen before. Searches match an article's title, so a demanded value (often present only among the fields) surfaces few of the passing articles. Compare the trolley lines"
                + (f" ({'; '.join(lines)})" if lines else "") + " against every cheaper row of the same kind now listed; inspect the fields of the lowest-priced such rows, substitute whichever passes, then pay.")

    def open_specs(self, ids: list[str], keep: int) -> list[str]:
        """Open the attribute view of each unopened product given, while steps stay above `keep`."""
        tool = self.spec_view()
        target = self.id_slot(tool)
        if not tool or not target:
            return []
        texts = []
        for ident in [i for i in ids if i in self.cards and i not in self.pages and not self.gone_now(i)]:
            if self.room() <= keep or self.over or self.specs_by.get(tool) is False:
                break
            faults: list[str] = []
            args = _conform({target: ident}, self.specs.get(tool), "args", faults)
            if faults:
                break
            reply = self.do_step(tool, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.digest(tool, seen)
            self.mark_view(tool, seen)
            if isinstance(seen, dict) and self.id_of(seen):
                self.specs_by[tool] = self.specs_by.get(tool, False) or bool(_spec_block(seen))
            texts.append(f"{ident}: " + self.render(seen)[:1500])
        if texts:
            self.log.append({"kind": "open_specs", "n": len(texts)})
        return texts

    def peek_cheaper(self) -> list[str]:
        """Ahead of a cheapest payment, read the fields on a few lower-priced hits of each trolley line's kind."""
        if self.choose() != "cheapest":
            return []
        mark = self.basket_key() + f"#{self.epoch}"
        kind_field = self.kind_field()
        if mark in self.peeked_marks or not kind_field:
            return []
        self.peeked_marks.add(mark)
        held = self.basket_ids()
        picks: list[str] = []
        for line in held:
            price = self.price(line)
            kind = str((self.cards.get(line) or {}).get(kind_field) or "").strip().lower()
            if price is None or not kind:
                continue
            pool = [i for i in self.arrival if i not in held and i not in self.pages and self.in_stock(i)
                    and str((self.cards.get(i) or {}).get(kind_field) or "").strip().lower() == kind
                    and (self.price(i) if self.price(i) is not None else float("inf")) < price]
            pool.sort(key=lambda i: self.price(i) or 0.0)
            picks += [i for i in pool[:2] if i not in picks]
        return self.open_specs(picks[:CHEAP_PEEKS], self.keep_back() + 4)

    def digest(self, name: str, observation: Any) -> str:
        """Ingest records and read any loose prose in a result as a possible notice."""
        self.ingest(observation)
        lines = self.free_text(observation)
        if not lines:
            return ""
        meaning = self.read_notice(lines)
        if meaning:
            self.pending_notes.append(f"(seen on {name}) {meaning}")
        return meaning

    def open_batch(self) -> None:
        """Before the interview, open several products per want so the planner sees which attribute keys exist."""
        tool = self.jobs.get("side")
        schema = self.specs.get(tool or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _types(v)), None)
        if not tool or not target or any(k != target for k in schema.get("required") or []) or self.side_blind():
            self.open_one_by_one()
            return
        if self.room() <= 12:
            return
        cap = props[target].get("maxItems") if isinstance(props[target].get("maxItems"), int) else 4
        ranked = self.want_groups()
        share = max(1, cap // max(1, len(ranked)))
        picks: list[str] = []
        for ids in ranked:
            picks += [i for i in ids if i not in self.pages][:share]
        picks = picks[:cap]
        if len(picks) < 2:
            return
        faults: list[str] = []
        args = _conform({target: picks}, schema, "args", faults)
        if faults:
            return
        before = len(self.pages)
        reply = self.do_step(tool, args)
        if reply is not None:
            seen = reply.get("observation")
            self.digest(tool, seen)
            self.mark_view(tool, seen)
            rows = self.cards_in(seen)
            if rows:
                self.specs_by[tool] = self.specs_by.get(tool, False) or any(_spec_block(r) for _, r in rows)
            self.log.append({"kind": "open_batch", "ids": picks, "opened": len(self.pages) - before})
            if rows and self.specs_by.get(tool) is False:
                self.open_one_by_one()
                return
            if len(self.pages) == before and len(picks) > 2 and self.room() > 12:
                faults = []
                args = _conform({target: picks[:2]}, schema, "args", faults)
                if not faults:
                    reply = self.do_step(tool, args)
                    if reply is not None:
                        self.digest(tool, reply.get("observation"))

    def open_one_by_one(self) -> None:
        tool = self.spec_view()
        target = self.id_slot(tool)
        if not tool or not target:
            return
        wanted = {w for _, need in self.live_needs() for w in re.findall(r"[a-z0-9]{3,}", need.split(":", 1)[-1].lower())}
        each = PEEK_EACH if len(self.wants) <= 2 else 4
        lanes: list[list[str]] = []
        heads: list[int] = []
        for ids in self.want_groups():
            opened = sum(1 for i in ids if i in self.pages)
            pool = [i for i in ids if i not in self.pages and self.in_stock(i)]
            if self.choose() == "cheapest":
                pool.sort(key=lambda i: self.price(i) if self.price(i) is not None else float("inf"))
            pool.sort(key=lambda i: -len(wanted & set(re.findall(r"[a-z0-9]{3,}", _pack(self.cards.get(i) or {}).lower()))))
            lanes.append(pool[:max(0, each - opened)])
            heads.append(max(0, 4 - opened))
        first = [lane[k] for k in range(4) for lane, head in zip(lanes, heads) if k < min(head, len(lane))]
        more = [lane[k] for k in range(each) for lane, head in zip(lanes, heads) if head <= k < len(lane)]
        cap = max(5, 2 * len(lanes) + 2)
        read = self.open_specs(first[:cap][:8], 22)
        self.open_specs((first[cap:] + more)[:max(0, 10 - len(read))], 30)

    # ---------- the interview ----------
    def time_to_ask(self) -> bool:
        if self.interviewed or not self.jobs.get("talk") or self.talk_room() <= 0:
            return False
        if self.room() <= 8:
            return False
        opened = len(self.pages)
        if opened >= 3 * len(self.wants) + 1:
            return True
        return self.at >= max(8, self.step_cap // 4)

    def talk_args(self, text: str) -> dict[str, Any] | None:
        name = self.jobs.get("talk")
        schema = self.specs.get(name or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _types(v) and "enum" not in v]
        target = next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if texts else None)
        if not target:
            return None
        args: dict[str, Any] = {target: text}
        for key in schema.get("required") or []:
            if key not in args:
                spec = props.get(key) or {}
                if isinstance(spec.get("enum"), list) and spec["enum"]:
                    args[key] = spec["enum"][0]
                else:
                    return None
        faults: list[str] = []
        args = _conform(args, schema, "args", faults)
        return None if faults else args

    def interview(self) -> str:
        """Put a property (or a short list) to the customer per note, best property first, until messages run out."""
        self.interviewed = True
        if self.talk_room() <= 0:
            return ""
        self.open_batch()
        count = self.talk_room()
        self.lanes = self.plan_questions(max(count, LIST_SIZE) if self.listing else count)
        self.lane_found = {n: 0 for n in self.lanes}
        self.lane_spent = {n: 0 for n in self.lanes}
        self.lane_used = {n: set() for n in self.lanes}
        self.follow = []
        sent: list[tuple[int, str, str]] = []
        told: list[str] = []
        # A note names a short list of properties while the client bears lists (the reply concerns the first listed
        # property the client holds a view on); a "one thing at a time" reply ends lists for the episode. A baffled or
        # shrugging reply means no view on the property asked, so a note is never repeated, reworded or given examples.
        groups = self.groups_for(self.talk_room()) if self.listing else []
        while self.talk_room() > 0 and not self.over and self.room() > 6 and not self.late():
            batch: list[tuple[int, str, str]] = []
            while groups and self.listing and not batch:
                batch = self.ask_group(groups.pop(0), told)
            batch = batch or self.ask_next(told)
            if not batch:
                break
            sent += batch
        found_items = [n for n in self.lane_found if self.lane_found[n] > 0]
        chain = self.early_chain()
        searched = self.need_search(found_items) if found_items else ""
        if told:
            self.epoch += 1
        lines = [f"Q: {q}\nA: {a}" for _, q, a in sent]
        summary = "Wren's interviewer asked the customer (one step per question):\n" + "\n".join(lines)
        summary += "\nNew firm needs from these replies: " + ("; ".join(told) if told else "none")
        summary += ". Fold them into every pick (re-search using the fresh values where needed)."
        pointed = [need for _, need in self.pointers if any(need in t for t in told)]
        if pointed:
            summary += " These borrow a value from some other article: " + " | ".join(pointed) + ". Track that article down, open its full view, then open every article its view forwards to; the last view carries the value, while the purchase is valid solely once all were opened."
        if chain:
            summary += "\n" + chain
        if searched:
            summary += "\n" + searched
        if self.pending_notes:
            summary += "\n[notices seen meanwhile] " + " | ".join(self.pending_notes)
            self.pending_notes = []
        return summary

    def query_pattern(self) -> tuple[str, dict[str, Any], str] | None:
        """From past searches: the search tool, the fixed arguments to keep, and the free-text field to vary."""
        tool = self.jobs.get("find")
        schema = self.specs.get(tool or "") or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _types(v) and "enum" not in v]
        named = [k for k in texts if re.search(r"(?:^|_)(?:words?|terms?|keywords?|phrase|text|search|query|q)(?:_|$)|keyword|phrase|query", k, re.I)]
        for args in reversed(self.queries):
            fields = [k for k in named or texts if isinstance(args.get(k), str) and re.search(r"[a-z]", args[k], re.I)]
            if not named:
                labels = {str(r.get(self.kind_field()) or "").strip().lower() for r in self.cards.values()} - {""} if self.kind_field() else set()
                fields = [k for k in fields if len({str(a.get(k)) for a in self.queries if a.get(k)}) >= 2 and any(len(str(a.get(k)).split()) >= 2 for a in self.queries) and not any(str(a.get(k) or "").strip().lower() in labels for a in self.queries)]
            if fields and tool:
                field = fields[0]
                keep = {k: v for k, v in args.items() if k != field and (isinstance((props.get(k) or {}).get("enum"), list) or k in (schema.get("required") or []))}
                kind_field = self.kind_field()
                shelves = {_plain(w.label or "") for w in self.wants} | {_plain(w.what) for w in self.wants}
                shelves |= {_plain(str(row.get(kind_field) or "")) for row in self.cards.values()} if kind_field else set()
                keep = {k: v for k, v in keep.items() if k in (schema.get("required") or []) or not isinstance(v, str) or (_plain(v) not in shelves and not any({x for x in _bag(v) if len(x) >= 4} & {x for x in _bag(s) if len(x) >= 4} for s in shelves if s))}
                worded = [a for a in self.queries if isinstance(a.get(field), str) and re.search(r"[a-z]", a[field], re.I)]
                for k in list(keep):
                    if isinstance((props.get(k) or {}).get("enum"), list):
                        used = [_pack(a[k]) for a in worded if k in a]
                        if used:
                            best = max(used, key=used.count)
                            keep[k] = next(a[k] for a in worded if k in a and _pack(a[k]) == best)
                return tool, keep, field
        return None

    def need_values(self, number: int) -> list[str]:
        out = []
        for n, need in self.live_needs():
            if n != number or need in [p for _, p in self.pointers] or ":" not in need:
                continue
            value = need.split(":", 1)[1].strip()
            if not value or re.search(BOUND_RX + r"|\b(?:view|shown|page|listing|alongside)\b", value, re.I):
                continue
            out.append(value)
        return out

    def need_search(self, numbers: list[int]) -> str:
        """After the interview: search each want with its need values together and open its kind's leading rows."""
        plan = self.query_pattern()
        if not plan or self.room() < 22:
            return ""
        tool, base, field = plan
        seen: list[str] = []
        allowance = 8
        for number in numbers:
            values = self.need_values(number)
            want = self.wants[number - 1]
            fresh = [v for v in values if any(":" in t and t.split(":", 1)[1].strip() == v for t in want.told)]
            if not fresh or allowance <= 2:
                continue
            stated = [i for i, v in enumerate(values) if v not in fresh]
            drop = stated[-1] if stated else 0
            tries = [values, values[:drop] + values[drop + 1:]] if len(values) > 1 else [values]
            for words in tries[:2]:
                if not words:
                    continue
                faults: list[str] = []
                args = _conform({**base, field: " ".join(words)}, self.specs.get(tool), "args", faults)
                if faults or self.over:
                    break
                reply = self.do_step(tool, args)
                allowance -= 1
                if reply is None:
                    break
                observation = reply.get("observation")
                self.digest(tool, observation)
                kind_field = self.kind_field()
                found = [i for i, _ in self.cards_in(observation)]
                cap = self.limit if isinstance(self.limit, (int, float)) else None
                prices = [self.price(i) for i in found]
                if cap is not None and prices and all(p is not None and p > 10 * cap for p in prices):
                    cap = None
                rows = [i for i in found if self.in_stock(i) and (not kind_field or self.owned_by(want, str((self.cards.get(i) or {}).get(kind_field) or ""))) and (cap is None or (self.price(i) or 0) <= cap)]
                seen.append(f"{_pack(args.get(field))}: " + (", ".join(rows[:6]) or "no row of that kind"))
                if rows:
                    opened = self.open_specs([i for i in rows if i not in self.pages][:min(3, allowance)], self.keep_back() + 16)
                    allowance -= len(opened)
                    if opened:
                        seen.append("opened: " + " | ".join(opened)[:2400])
                    break
        if not seen:
            return ""
        self.log.append({"kind": "need_search", "seen": [s[:200] for s in seen]})
        return "[Wren ran a combined search on each such want's demand values and opened the leading hits of the right kind] " + "\n".join(seen)

    def pick_question(self, lanes: dict[int, list[str]], found: dict[int, int], spent: dict[int, int], used: dict[int, set[str]]) -> tuple[int, list[str]]:
        first = {n: i for i, n in enumerate(self.hint_order)}
        for number in sorted(lanes, key=lambda n: (found.get(n, 0) > 0, spent.get(n, 0), first.get(n, len(first) + n))):
            left = [label for label in lanes[number] if _plain(label) not in used[number] and not self.told_about(number, label)]
            if left:
                return number, left[:LIST_SIZE if self.listing else 1]
        return 0, []

    def told_about(self, number: int, label: str) -> bool:
        """Whether a reply has already pinned down the want's property (asking again would waste a note)."""
        return any(_kin(label, need.split(":", 1)[0]) for need in self.wants[number - 1].told)

    def item_said(self, number: int) -> str:
        """How a note calls the article: a name that joins two nouns with "and" or "&" is said with "or", since the
        client reads "A and B" as two topics in one note and asks for them singly."""
        return re.sub(r"\s*&\s*|\s+and\s+", " or ", self.wants[number - 1].what, flags=re.I)

    def ask_next(self, told: list[str], floor: int = 6) -> list[tuple[int, str, str]]:
        """Dispatch the next note (a queued follow-up ahead of the labels planned next), fold its reply into the wants,
        settle the named labels. Gives the dispatched batch, [] if nothing was sent."""
        if self.follow:
            number, labels = self.follow.pop(0)
            labels = [label for label in labels if not self.told_about(number, label)]
            if not labels:
                return self.ask_next(told, floor)
        else:
            number, labels = self.pick_question(self.lanes, self.lane_found, self.lane_spent, self.lane_used)
        if not labels:
            return []
        text = self.phrase(number, labels)
        if not self.talk_args(text):
            labels, text = labels[:1], self.phrase(number, labels[:1])
        batch = self.deliver([(number, labels[0], text)], floor)
        if not batch:
            return []
        self.finish_ask(number, labels, batch, told)
        return batch

    def finish_ask(self, number: int, labels: list[str], batch: list[tuple[int, str, str]], told: list[str]) -> None:
        """Fold a dispatched note's reply into the wants, then settle its labels."""
        self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(w.told) for w in self.wants]
        heard = {} if self.quiet_reply(batch) else self.listen(batch, (SMALL, BIG))
        if len(labels) > 1:
            heard.pop("free", None)
        self.apply_answers(heard, told)
        new = [need for want, size in zip(self.wants, had) for need in want.told[size:]]
        for n, (want, size) in enumerate(zip(self.wants, had), 1):
            if len(want.told) > size:
                self.lane_found[n] = self.lane_found.get(n, 0) + 1
        crowded = any(_whole(x) == 1 for x in _listed(heard.get("crowded")))
        self.settle(number, labels, new, crowded)

    def quiet_reply(self, sent: list[tuple[int, str, str]]) -> bool:
        """Whether a reply can go unread: nothing in it echoes the notes, the wants, the field keys or opened
        values, it holds no figure, and it raises no complaint on the asking. The episode's first reply is always read, since it
        shows whether this client bears lists. Spares a model call (several seconds of the 300) on the plain shrugs."""
        if not self.heard_once:
            self.heard_once = True
            return False
        known: set[str] = set()
        for want in self.wants:
            known |= _cores(want.what) | _cores(want.label or "")
            for need in want.needs + want.told:
                known |= _cores(need.split(":", 1)[0])
        for label, name, _, values in self.field_table():
            known |= _cores(label) | _cores(name)
            for v in values:
                known |= {w for w in _cores(v) if len(w) >= QUIET_WORD}
        for record in self.pages.values():
            for value in (_spec_block(record) or {}).values():
                known |= {w for w in _cores(str(value)) if len(w) >= QUIET_WORD}
        for _, question, answer in sent:
            asked = re.sub(r"\b(?:for|the|which|what|do|you|prefer|would|like|or|and)\b", " ", question, flags=re.I)
            mine = known | _cores(asked)
            said = _cores(answer)
            if re.search(r"\d", answer) or said & mine or re.search(ASK_WORDS_RX, answer, re.I):
                return False
        self.log.append({"kind": "quiet_reply", "a": sent[-1][2][:120] if sent else ""})
        return True

    def groups_for(self, free: int) -> list[list[int]]:
        """Which wants each free note covers when the wants outnumber the notes: adjacent runs of near-equal
        size, so that no want goes unasked (race 17: two notes for three wants in most tasks, and the hidden need sat
        on the third want in a third of our misses)."""
        numbers = [n for n in self.hint_order if n in self.lanes] + [n for n in self.lanes if n not in self.hint_order]
        numbers = [n for n in numbers if self.lanes.get(n)]
        if free <= 0 or len(numbers) <= free:
            return []
        out: list[list[int]] = []
        start = 0
        for slot in range(free):
            size = (len(numbers) - start + (free - slot) - 1) // (free - slot)
            out.append(numbers[start:start + size])
            start += size
        return out

    def ask_group(self, group: list[int], told: list[str], floor: int = 6) -> list[tuple[int, str, str]]:
        """One note naming a property of each want in the group. The client speaks to the earliest named property
        it cares about, on whichever want; a reply revealing nothing closes the opening want's label alone."""
        parts: list[tuple[int, list[str]]] = []
        for number in group:
            left = [label for label in self.lanes.get(number) or [] if _plain(label) not in self.lane_used.setdefault(number, set()) and not self.told_about(number, label)]
            if left:
                parts.append((number, left[:MIX_SHARE]))
        if not parts:
            return []
        if len(parts) == 1:
            number, _ = parts[0]
            left = [label for label in self.lanes.get(number) or [] if _plain(label) not in self.lane_used[number] and not self.told_about(number, label)]
            labels = left[:LIST_SIZE]
            text = self.phrase(number, labels)
            if not self.talk_args(text):
                labels, text = labels[:1], self.phrase(number, labels[:1])
            batch = self.deliver([(number, labels[0], text)], floor)
            if batch:
                self.finish_ask(number, labels, batch, told)
            return batch
        text = ""
        while parts:
            pieces = [f"for the {self.item_said(number)}, which {labels[0]}" for number, labels in parts]
            text = "F" + pieces[0][1:] + " do you prefer, and " + ", and ".join(pieces[1:]) + "?"
            if self.talk_args(text):
                break
            parts = parts[:-1]
        if len(parts) < 2:
            return self.ask_group([n for n, _ in parts], told, floor) if parts else []
        first = parts[0][0]
        batch = self.deliver([(first, parts[0][1][0], text)], floor)
        if not batch:
            return []
        for number, _ in parts:
            self.lane_spent[number] = self.lane_spent.get(number, 0) + 1
        had = [len(w.told) for w in self.wants]
        heard = {} if self.quiet_reply(batch) else self.listen(batch, (SMALL, BIG), [[n for n, _ in parts]])
        heard.pop("free", None)
        self.apply_answers(heard, told)
        answered = False
        for number, labels in parts:
            new = self.wants[number - 1].told[had[number - 1]:]
            if new:
                answered = True
                self.lane_found[number] = self.lane_found.get(number, 0) + 1
                self.settle(number, labels, list(new), False)
        crowded = any(_whole(x) == 1 for x in _listed(heard.get("crowded")))
        if crowded and not answered:
            self.listing = False
            self.log.append({"kind": "lists_off", "want": first})
            return batch
        if not answered:
            self.lane_used.setdefault(first, set()).add(_plain(parts[0][1][0]))
        self.log.append({"kind": "mixed_ask", "parts": parts, "answered": answered})
        return batch

    def settle(self, number: int, labels: list[str], new: list[str], crowded: bool) -> None:
        """Record what a reply taught about the labels in its note. A lone label counts as settled whatever came back
        (a shrug, bafflement or a pacing complaint all mean no view on it). For a list: a stated preference closes its
        own label and the ones ahead of it, with the rest leading the want's next note; a shrug closes the leading label
        alone (race 17: the client speaks to the first property named, so the shrug says nothing of the others); a
        pacing complaint switches lists off until the episode ends, with the leading label going out by itself."""
        used = self.lane_used.setdefault(number, set())
        want = self.wants[number - 1]
        if len(labels) == 1:
            used.add(_plain(labels[0]))
            if not new and not any(_kin(labels[0], f) for f in want.free):
                want.free.append(labels[0])
            return
        if crowded and not new:
            self.listing = False
            self.log.append({"kind": "lists_off", "want": number})
            self.follow.insert(0, (number, labels[:1]))
            return
        hit = -1
        for need in new:
            prop = need.split(":", 1)[0]
            scores = [i for i, label in enumerate(labels) if _kin(label, prop)]
            if scores:
                hit = scores[0] if hit < 0 else min(hit, scores[0])
        settled = labels[:hit + 1] if hit >= 0 else labels[:1]
        used.update(_plain(label) for label in settled)
        if hit < 0:
            for label in settled:
                if not any(_kin(label, f) for f in want.free):
                    want.free.append(label)
        ahead = [label for label in labels if _plain(label) not in used]
        rest = [label for label in self.lanes.get(number) or [] if label not in labels]
        self.lanes[number] = ahead + rest
        self.log.append({"kind": "list_settled", "want": number, "done": settled, "next": ahead[:4]})

    def late_question(self) -> str:
        """Right before paying, spend a note the order can spare on one further planned question: notes held
        back for a bow-out chore are free once the agent buys. Returns a summary for the walker when the reply adds a
        demand, else ''."""
        if self.late_asked or not self.jobs.get("talk") or self.talked >= self.talk_cap or self.late():
            return ""
        if any(c["stage"] == "order" and c["messages"] > 0 for c in self.chores) or self.room() < LATE_STEPS:
            return ""
        self.late_asked = True
        if not self.lanes:
            self.lanes = self.plan_questions(LIST_SIZE if self.listing else 1)
            self.lane_found = {n: 0 for n in self.lanes}
            self.lane_spent = {n: 0 for n in self.lanes}
            self.lane_used = {n: set() for n in self.lanes}
        told: list[str] = []
        batch = self.ask_next(told, LATE_STEPS - 2)
        if not batch:
            return ""
        number, question, answer = batch[0]
        self.log.append({"kind": "late_ask", "q": question, "a": answer[:200], "told": told})
        if not told:
            return ""
        self.epoch += 1
        summary = f"Wren's interviewer put one more question to the client before paying (one step):\nQ: {question}\nA: {answer}\nNew firm needs from this reply: " + "; ".join(told) + ". The payment was held: test each trolley line against them (re-search with the fresh values where needed), mend the trolley, then pay."
        pointed = [need for _, need in self.pointers if any(need in t for t in told)]
        if pointed:
            summary += " These borrow a value from some other article: " + " | ".join(pointed) + ". Track that article down, open its full view, then every article its view forwards to; the last view carries the value."
        searched = self.need_search([number])
        return summary + ("\n" + searched if searched else "")

    def phrase(self, number: int, labels: list[str], shown: bool = False) -> str:
        if len(labels) >= 2:
            return f"For the {self.item_said(number)}, which {', '.join(labels[:-1])} or {labels[-1]} do you prefer?"
        return self.wording(number, labels[0], self.examples(number, labels[0]) if shown else None)

    def lane_fields(self, number: int, table: list[tuple[str, str, int, list[str]]]) -> list[str]:
        want = self.wants[number - 1]
        settled = [_bag(n.split(":", 1)[0]) for n in want.needs]
        rows = [(label, key, carried, len(values)) for label, key, carried, values in table if self.owner(label) == number and not any(_kin(key, f) for f in want.free)]
        measured = {key: sum(1 for v in values if re.match(r"^\s*[0-9.,/x -]+\s*[a-z]{0,6}\s*$", v)) * 2 > len(values) for _, key, _, values in table}
        if len(self.wants) == 1 and not rows:
            rows = [(label, key, carried, len(values)) for label, key, carried, values in table]
        rows.sort(key=lambda r: (-r[2], measured.get(r[1], False), -r[3]))
        keys = [key for _, key, _, _ in rows if not any(_same_prop(prop, _bag(key)) for prop in settled) and not re.search(JUNK_KEY_RX, key)]
        return [key for key in keys if not re.search(GRAB_BAG_RX, key)] + [key for key in keys if re.search(GRAB_BAG_RX, key)]

    def owner(self, label: str) -> int:
        """Which want a category label belongs to (0 when none fits)."""
        if not label:
            return 0
        words = _plain(label)
        long = _cores(label)
        best, top = 0, 0.0
        for number, want in enumerate(self.wants, 1):
            if words == _plain(want.label or ""):
                score = 3.0
            elif words == _plain(want.what):
                score = 2.0
            else:
                mine = _cores(want.label or "") | _cores(want.what)
                score = len(long & mine) / max(1, len(long | mine))
            if score > top:
                best, top = number, score
        return best

    def owned_by(self, want: Want, label: str) -> bool:
        return bool(label) and want in self.wants and self.owner(label) == self.wants.index(want) + 1

    def pages_of(self, number: int) -> int:
        kind_field = self.kind_field()
        want = self.wants[number - 1]
        return sum(1 for record in self.pages.values() if kind_field and self.owned_by(want, str(record.get(kind_field) or "")) and _spec_block(record))

    def plan_questions(self, count: int) -> dict[int, list[str]]:
        """Ask the planner model for the properties worth a question, then guard its picks with the key statistics."""
        table = self.field_table()
        settled = []
        for number, want in enumerate(self.wants, 1):
            settled.append(f"want {number}: {want.what} - fixed by the message: " + ("; ".join(want.needs) or "nothing"))
        listing = []
        last = None
        for index, (label, name, carried, values) in enumerate(table, 1):
            if label != last:
                listing.append(f"category {label!r}:")
                last = label
            listing.append(f"  {index}. {name} (on {carried}): " + " | ".join(values) + self.flag_other(label, name, values) + (" [a grab-bag of features: ask last]" if re.search(GRAB_BAG_RX, name) else ""))
        prompt = ASKPLAN.replace("COUNT", str(max(1, count))) + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + "\n\nWANTS:\n" + "\n".join(settled)
        first = [str(x) for x in _listed(self.intake.get("check_first")) if str(x).strip()]
        if first:
            prompt += "\n\nASK FIRST (the message says to ask about these):\n" + "\n".join(first)
        if self.hints:
            prompt += "\n\nHINTS NOTICED AT INTAKE:\n" + "\n".join(f"- want {h['item']}: \"{h['words']}\"" + (f" -> {h['property']}" if h["property"] else "") for h in self.hints)
        left_open = [f"- want {n}: {', '.join(w.free)}" for n, w in enumerate(self.wants, 1) if w.free]
        if left_open:
            prompt += "\n\nLEFT OPEN (the client said these do not matter; never ask about them):\n" + "\n".join(left_open)
        prompt += "\n\nATTRIBUTE KEYS SEEN:\n" + ("\n".join(listing) or "(no full product records yet)")
        plan = self.json_call(prompt, [BIG, SMALL], 3500) or {}
        lanes: dict[int, list[str]] = {n: [] for n in range(1, len(self.wants) + 1)}
        hinted: dict[int, set[str]] = {n: set() for n in lanes}
        for pick in _listed(plan.get("ask")):
            if not isinstance(pick, dict):
                continue
            number = min(max(1, _whole(pick.get("item") or 1) or 1), len(self.wants))
            label = re.split(r"\s+(?:or|and)\s+|[(;,/]", str(pick.get("label") or ""))[0].strip().rstrip("?.")
            if not label or len(label.split()) > 5:
                continue
            if any(_kin(label, f) for f in self.wants[number - 1].free):
                continue
            if all(_plain(label) != _plain(other) for other in lanes[number]):
                lanes[number].append(label)
            if str(pick.get("hint") or "").strip():
                hinted[number].add(_plain(label))
                if number not in self.hint_order:
                    self.hint_order.append(number)
        for number in lanes:
            keys = self.lane_fields(number, table)
            read = max(1, self.pages_of(number))
            own: dict[str, float] = {}
            for label, key, carried, _ in table:
                if self.owner(label) == number:
                    own[key] = max(own.get(key, 0.0), carried / read)

            def spelled(label: str) -> str:
                plain = lambda text: re.sub(r"our\b", "or", re.sub(r"(?<=[a-z]{3})s\b", "", _plain(text)))
                return next((k for k in own if plain(k) == plain(label)), label)

            def share_of(label: str) -> float:
                if _plain(label) in own:
                    return own[_plain(label)]
                near = [s for k, s in own.items() if _same_prop(_bag(label), _bag(k)) or _same_prop(_bag(k), _bag(label))]
                return max(near) if near else 0.0
            hinted[number] = {_plain(spelled(h)) for h in hinted[number]}
            lane = []
            for pick in lanes[number]:
                if all(_plain(spelled(pick)) != _plain(other) for other in lane):
                    lane.append(spelled(pick))
            for key in [k for k in keys if own.get(k, 0.0) >= RANK_SHARE and not re.search(GRAB_BAG_RX, k)][:RANK_KEYS]:
                if any(_plain(key) == _plain(other) for other in lane[:3]):
                    continue
                weak = next((i for i, other in enumerate(lane[:3]) if _plain(other) not in hinted[number] and share_of(other) + RANK_GAP <= own.get(key, 0.0)), None)
                if weak is None and len(lane) >= 3:
                    continue
                placed = [other for other in lane if _plain(other) != _plain(key)]
                if weak is not None and len(self.wants) > 1:
                    moved = placed[weak]
                    placed[weak] = key
                    placed.insert(min(3, len(placed)), moved)
                else:
                    placed.insert(min(len(placed), weak if weak is not None else len(placed)), key)
                if any(_plain(x) in hinted[number] and all(_plain(x) != _plain(y) for y in placed[:3]) for x in lane[:3]):
                    continue
                lane = placed
            lanes[number] = lane
            for key in keys:
                if all(_plain(key) != _plain(other) for other in lanes[number]):
                    lanes[number].append(key)
            lanes[number] = [x for x in lanes[number] if not re.search(GRAB_BAG_RX, _plain(x))] + [x for x in lanes[number] if re.search(GRAB_BAG_RX, _plain(x))]
            if count <= 3 and read >= 3:
                # few notes: properties hinted at, or carried by half the opened articles, go first
                sure = [x for x in lanes[number] if _plain(x) in hinted[number] or share_of(x) >= 0.5]
                lanes[number] = sure + [x for x in lanes[number] if x not in sure]
        self.log.append({"kind": "plan", "ask": _trim(plan.get("ask"), 120), "lanes": {n: v[:8] for n, v in lanes.items()}})
        return lanes

    def wording(self, number: int, label: str, samples: list[str] | None = None) -> str:
        text = f"For the {self.item_said(number)}, which {label} do you prefer"
        return text + (f" (say, {samples[0]} or {samples[1]})?" if samples else "?")

    def examples(self, number: int, label: str) -> list[str]:
        want = _plain(label)
        fixed = " ".join(self.wants[number - 1].needs).lower()
        for _, name, _, values in self.field_table():
            if name == want:
                return [v for v in values if v and v not in fixed and len(v) <= 30][:2]
        return []

    def deliver(self, questions: list[tuple[int, str, str]], floor: int = 6) -> list[tuple[int, str, str]]:
        tool = self.jobs["talk"]
        sent: list[tuple[int, str, str]] = []
        for number, _, text in questions:
            if self.over or self.room() <= floor or self.talked >= self.talk_cap:
                break
            args = self.talk_args(text)
            if not args:
                break
            reply = self.do_step(tool, args)
            if reply is None:
                break
            self.talked += 1
            answer = reply.get("observation")
            words = self.voice(answer)
            sent.append((number, text, words))
            self.diary.append(f"step {self.at}: message to the customer {text!r} -> reply {words[:300]!r} (basket then: {self.basket_key()})")
            self.diary_steps.append(self.at)
            self.log.append({"kind": "ask", "q": text, "a": words[:300]})
            if isinstance(answer, dict) and answer.get("error"):
                break
        return sent

    def apply_answers(self, heard: dict[str, Any], told: list[str]) -> None:
        many = len(self.wants) > 1
        for entry in _listed(heard.get("needs")):
            if not isinstance(entry, dict) or not str(entry.get("need") or "").strip():
                continue
            number = min(max(1, _whole(entry.get("item")) or 1), len(self.wants))
            need = str(entry["need"]).strip()
            if need not in self.wants[number - 1].told:
                self.wants[number - 1].told.append(need)
                told.append(f"want {number}: {need}" if many else need)
            if entry.get("points") is True and (number, need) not in self.pointers:
                self.pointers.append((number, need))
        for entry in _listed(heard.get("drop")):
            if isinstance(entry, dict) and str(entry.get("property") or "").strip():
                want = self.wants[min(max(1, _whole(entry.get("item")) or 1), len(self.wants)) - 1]
                gone = _plain(str(entry["property"]).split(":")[0])
                hit = [n for n in want.needs + want.told if n not in want.dropped and _plain(n.split(":")[0]) == gone]
                want.dropped.extend(hit)
                told.extend(f"dropped {n}" for n in hit)
        for entry in _listed(heard.get("free")):
            if isinstance(entry, dict) and entry.get("property"):
                number = min(max(1, _whole(entry.get("item")) or 1), len(self.wants))
                self.wants[number - 1].free.append(str(entry["property"]))
        money = _money(heard.get("limit"))
        if money is not None:
            self.limit = money

    def listen(self, sent: list[tuple[int, str, str]], models: tuple[str, ...] = (BIG, SMALL), groups: list[list[int]] | None = None) -> dict[str, Any]:
        prompt = LISTEN + "\n\nWANTS:\n" + "\n".join(f"want {n}: {w.what}" for n, w in enumerate(self.wants, 1))

        def about(i: int, n: int) -> str:
            wants = groups[i - 1] if groups and i - 1 < len(groups) and groups[i - 1] else [n]
            return f"about want {wants[0]}" if len(wants) == 1 else "about wants " + " and ".join(str(x) for x in wants) + ", the reply naming the one it concerns"
        prompt += "\n\nQUESTIONS AND REPLIES:\n" + "\n".join(f"{i}. ({about(i, n)}) Q: {q}\nA: {a}" for i, (n, q, a) in enumerate(sent, 1))
        return self.json_call(prompt, list(models), 2500) or {}

    # ---------- chores the customer asked for ----------
    def chore_gate(self, stage: str) -> tuple[bool, str]:
        """Ahead of payment or bow-out: a model performs the chores falling due at that stage (approvals, notices)."""
        due = [c for c in self.chores if c["stage"] == stage]
        if not due or self.over or not self.chore_due(stage):
            return True, ""
        closers = {self.jobs.get("checkout"), self.jobs.get("walk_away")} - {None}
        tools = [line for line in self.tool_lines() if line[2:].split(":")[0] not in closers]
        said: list[str] = []
        nudged = False
        for _ in range(CHORE_ROUNDS):
            prompt = (STAGEGATE.replace("STAGE", "paying" if stage == "order" else "bowing out") + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text
                      + "\n\nCHORES DUE NOW:\n" + "\n".join(f"- {c['what']}" for c in due) + "\n\nSHEET:\n" + self.sheet() + "\n\nBASKET NOW: " + self.basket_text()
                      + "\n\nDIARY (messages and chore calls so far):\n" + ("\n".join(f"{i}. {line}" for i, line in enumerate(self.diary, 1)) or "(nothing yet)")
                      + ("\n\nSHOP NOTICES SEEN:\n" + "\n".join(self.changes[-4:]) if self.changes else "")
                      + f"\n\nMESSAGES LEFT: {max(0, self.talk_cap - self.talked)}   STEPS LEFT: {self.room()} (the {'order' if stage == 'order' else 'ending'} itself takes one)"
                      + "\n\nTOOLS:\n" + "\n".join(tools))
            got = self.json_call(prompt, [BIG, SMALL] if nudged else [SMALL, BIG], 3500)
            if got is None:
                break
            told: list[str] = []
            hold = ""
            if said:
                self.apply_answers({"needs": got.get("needs")}, told)
                hold = str(got.get("hold") or "").strip()
            call = got.get("call") if isinstance(got.get("call"), dict) else None
            self.log.append({"kind": "chore_gate", "stage": stage, "call": _trim(call, 200), "steps": _trim(got.get("steps"), 120), "hold": hold[:200], "told": told})
            if told or hold:
                self.epoch += 1
                return False, (("Before going on, Wren did the chores the customer asked for:\n" + "\n".join(said) + "\n") if said else "") + "The client's reply alters the task: " + "; ".join(told + ([hold] if hold else [])) + ". Mend the trolley accordingly, then pay once more."
            name = str((call or {}).get("tool") or "")
            shown = [e for e in _listed(got.get("steps")) if isinstance(e, dict)]
            talk = stage == "order" and any(c["messages"] > 0 for c in due)
            numbers = range(1, len(self.diary) + 1)
            proof = {i for i in numbers if not talk or (i <= len(self.diary_steps) and self.diary_steps[i - 1] > self.basket_at)}
            unproven = [e for e in shown if e.get("done") is not True or not any((_whole(n) or 0) in proof for n in _listed(e.get("lines")))]
            if not call and unproven and not said and not nudged and self.jobs.get("talk") and self.talked < self.talk_cap and self.room() >= 2:
                nudged = True
                self.log.append({"kind": "chore_unproven", "steps": _trim(unproven, 120)})
                self.diary.append(f"(Wren: no line since the basket last changed at step {self.basket_at} shows " + "; ".join(str(e.get("step") or "") for e in unproven)[:200] + f" done, and {self.talk_cap - self.talked} message(s) and {self.room()} steps remain: give the call that does it)")
                self.diary_steps.append(-1)
                continue
            if not call or name not in self.specs or name in closers:
                if not call:
                    self.chores_done.add((stage, self.basket_key() + f"#{self.epoch}"))
                break
            if self.room() < 2 or (name == self.jobs.get("talk") and self.talked >= self.talk_cap):
                break
            faults: list[str] = []
            args = _conform(call.get("args") if isinstance(call.get("args"), dict) else {}, self.specs[name], "args", faults)
            if faults:
                break
            if name == self.jobs.get("talk"):
                text = next((v for v in args.values() if isinstance(v, str)), "")
                batch = self.deliver([(1, "", text)], 1)
                if not batch:
                    break
                self.chore_talk += 1
                said.append(f"Q: {text}\nA: {batch[0][2]}")
                continue
            reply = self.do_step(name, args)
            if reply is None:
                break
            seen = reply.get("observation")
            self.ingest(seen)
            self.diary.append(f"step {self.at}: {name} {_pack(args)} -> {self.render(seen)[:300]} (basket then: {self.basket_key()})")
            self.diary_steps.append(self.at)
            said.append(f"{name} {_pack(args)} -> {self.render(seen)[:400]}")
        return True, ("Before going on, Wren did the chores the customer asked for:\n" + "\n".join(said)) if said else ""

    # ---------- notices and prose inside results ----------
    def voice(self, observation: Any) -> str:
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
        return " ".join(found)[:800] if found else _pack(observation)[:400]

    def free_text(self, observation: Any) -> list[str]:
        lines: list[str] = []
        if isinstance(observation, dict) and not self.id_of(observation):
            for key, value in observation.items():
                if isinstance(value, str) and len(value.split()) >= 5:
                    lines.append(value)
                elif isinstance(value, list) and value and all(isinstance(v, str) for v in value):
                    lines.extend(v for v in value if len(v.split()) >= 5)
                elif isinstance(value, dict) and not self.id_of(value, False, 1):
                    for inner in value.values():
                        if isinstance(inner, str) and len(inner.split()) >= 6:
                            lines.append(inner)
        return [line for line in lines if line not in self.noted_lines]

    def read_notice(self, lines: list[str]) -> str:
        self.noted_lines.update(lines)
        prompt = NOTICE + "\n\nWANTS AND NEEDS NOW:\n" + self.sheet() + "\n\nTHE NOTICE:\n" + "\n".join(lines)
        got = self.json_call(prompt, [SMALL, BACKUP], 2500) or {}
        gone = [str(g) for g in _listed(got.get("gone")) if isinstance(g, (str, int))]
        self.unbuyable.update(g for g in gone if g in self.cards)
        self.flagged_sold = bool(gone) or got.get("sold_out") is True
        self.flagged_named = any(g in self.cards for g in gone)
        if self.flagged_sold and len(lines) == 1:
            self.soldout_lines.update(lines)
        if got.get("kind") not in (None, "change"):
            return ""
        changed = []
        money = _money(got.get("limit"))
        if money is not None and money != self.limit:
            self.limit = money
            changed.append(f"limit now {money:g}")
        drops = [e for e in _listed(got.get("drop")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        adds = [e for e in _listed(got.get("add")) if isinstance(e, dict) and str(e.get("need") or "").strip()]
        for entry in drops:
            want = self.wants[min(max(1, _whole(entry.get("item")) or 1), len(self.wants)) - 1]
            target = str(entry["need"]).strip()
            live = [n for n in want.needs + want.told if n not in want.dropped]
            hit = [n for n in live if n == target]
            if not hit:
                hit = [n for n in live if _plain(n.split(":")[0]) == _plain(target.split(":")[0])][:1]
            want.dropped.extend(hit or [target])
            changed.append(f"dropped {target}")
        for entry in adds:
            want = self.wants[min(max(1, _whole(entry.get("item")) or 1), len(self.wants)) - 1]
            need = str(entry["need"]).strip()
            if need in want.dropped:
                want.dropped.remove(need)
            if need not in want.needs + want.told:
                want.told.append(need)
            changed.append(f"added {need}")
        if gone:
            changed.append("unavailable: " + ", ".join(gone))
        if got.get("price_moved") is True:
            changed.append("a price moved")
        summary = str(got.get("summary") or "").strip()
        if self.choose() == "cheapest" and (drops or gone or self.flagged_sold or any(c.startswith("limit now") for c in changed)):
            self.wider_block = False
            self.peeked_marks = set()
        if changed or summary:
            self.epoch += 1
            self.just_did = []
            self.changes.append(summary or "; ".join(changed))
        return (summary + (" (" + "; ".join(changed) + ")" if changed else "")).strip()

    def live_needs(self) -> list[tuple[int, str]]:
        out = []
        for number, want in enumerate(self.wants, 1):
            for need in want.needs + want.told:
                if need not in want.dropped:
                    out.append((number, need))
        for rule in self.conditionals:
            if rule["state"] is True and (rule["item"], rule["need"]) not in out:
                out.append((rule["item"], rule["need"]))
        return out

    def proof_lines(self, observation: Any) -> str:
        """After an article was opened: the keys named after a demanded property, with their values."""
        ids: list[str] = []

        def walk(node: Any, depth: int) -> None:
            if depth > 6:
                return
            if isinstance(node, dict):
                ident = self.id_of(node, False, depth)
                if ident and _spec_block(node) and ident not in ids:
                    ids.append(ident)
                    return
                for value in node.values():
                    walk(value, depth + 1)
            elif isinstance(node, list):
                for value in node:
                    walk(value, depth + 1)
        walk(observation, 0)
        needs = self.live_needs()
        if not ids or not needs:
            return ""
        many = len(self.wants) > 1
        lines = []
        for ident in ids[:12]:
            block = _spec_block(self.pages.get(ident) or {}) or {}
            parts = []
            for number, need in needs[:8]:
                prop = need.split(":", 1)[0].strip()
                hits = [f"{k}={str(v)[:50]}" for k, v in block.items() if _same_prop(_bag(prop), _bag(k))]
                tag = f"(want {number}) " if many else ""
                if hits:
                    parts.append(f"{tag}{prop} -> " + ", ".join(hits[:3]))
            if parts:
                lines.append(f"{ident}: " + "; ".join(parts))
        if not lines:
            return ""
        return "[proof: keys named after a demanded property; a key naming some other property (a separate part or aspect) rules nothing, while a differently named key can still carry that same property] " + " | ".join(lines)

    # ---------- the basket ----------
    def basket_text(self) -> str:
        if self.basket_snap is not None and self.basket_seen >= self.basket_at:
            return _pack(_trim(self.basket_snap, 300))[:3000]
        return "lines by Wren's record of the basket calls (product: units): " + _pack(self.basket)

    def units_of(self, ident: str) -> int:
        if self.basket_snap is not None and self.basket_seen >= self.basket_at:
            stack: list[Any] = [self.basket_snap]
            while stack:
                node = stack.pop()
                if isinstance(node, dict):
                    if ident in [v.strip() for v in node.values() if isinstance(v, str)]:
                        counts = [_whole(v) for k, v in node.items() if not isinstance(v, str) and _whole(v) is not None and re.search(r"\b(?:units?|count|quantity|qty)\b", _plain(k))]
                        if len(counts) == 1:
                            return counts[0]
                    stack += [v for v in node.values() if isinstance(v, (dict, list))]
                elif isinstance(node, list):
                    stack += node
        return self.basket.get(ident, 1)

    def basket_ids(self) -> list[str]:
        if self.basket_snap is not None and self.basket_seen >= self.basket_at:
            text = _pack(self.basket_snap)
            return [i for i in self.cards if i in text]
        return list(self.basket)

    def card_text(self, ident: str) -> str:
        record = self.pages.get(ident) or self.cards.get(ident)
        return _pack(_trim(record, 400))[:3000] if record else "(no record seen)"

    def other_cards(self, skip: list[str], limit: int = 30) -> str:
        lines = []
        order = list(reversed(self.arrival))
        if self.choose() == "cheapest":
            order.sort(key=lambda i: (i not in self.pages, self.price(i) if self.price(i) is not None else float("inf")))
        for ident in order:
            if ident in skip or self.gone_now(ident):
                continue
            record = self.pages.get(ident) or self.cards.get(ident)
            lines.append(_pack(_trim(record, 160))[:900])
            if len(lines) >= limit:
                break
        return "\n".join(lines)

    def title_of(self, ident: str) -> str:
        row = self.cards.get(ident) or {}
        for key, value in row.items():
            if isinstance(value, str) and re.search(r"title|name", str(key), re.I) and 3 <= len(value.strip()) <= 120:
                return value.strip()
        return ""

    # ---------- compatibility between wants ----------
    def compat_index(self) -> dict[str, Any]:
        marks: dict[str, dict[str, Any]] = {}
        notes: dict[str, set[str]] = {}
        for ident in self.arrival:
            record = self.pages.get(ident) or self.cards.get(ident) or {}
            text = " ".join(str(v) for v in record.values() if isinstance(v, str))
            text += " " + ". ".join(f"{k} {v}" for block in record.values() if isinstance(block, dict) for k, v in block.items() if isinstance(v, str))
            marks[ident] = _compat_marks(text)
            for part, codes in marks[ident]["also"]:
                notes.setdefault(part, set()).update(codes)
        return {"marks": marks, "notes": notes}

    def compat_set(self, ident: str, index: dict[str, Any]) -> set[str]:
        mark = index["marks"].get(ident) or {}
        found = set(mark.get("works") or [])
        if mark.get("own"):
            found |= index["notes"].get(mark["own"], set())
        return found

    def bars_of(self, ident: str, index: dict[str, Any]) -> set[str]:
        return set((index["marks"].get(ident) or {}).get("bars") or [])

    def want_ids(self, number: int) -> list[str]:
        kind_field = self.kind_field()
        want = self.wants[number - 1]
        return [i for i in self.arrival if kind_field and self.owned_by(want, str((self.cards.get(i) or {}).get(kind_field) or ""))]

    def link_system(self, part: int, host: int, index: dict[str, Any]) -> tuple[int, int, set[str]] | None:
        """(the want whose rows bear series codes, the want whose rows name series, those series) for two wants tied by a
        part-series scheme; None while the rows seen show no such scheme. The one want's rows must mostly bear codes from
        few series, while some row of the other names any of those series (race 17: bulbs "(part VI3-3480)", lamps "takes part
        series VI1, VI3")."""
        marks = index["marks"]
        for carrier, namer in ((part, host), (host, part)):
            rows = self.want_ids(carrier)
            tagged = [i for i in rows if (marks.get(i) or {}).get("fam")]
            families = {f for i in tagged for f in marks[i]["fam"]}
            if len(tagged) < LINK_ROWS or 2 * len(tagged) < len(rows) or len(families) > LINK_FAMILIES:
                continue
            if any(f in families for i in self.want_ids(namer) for f in (marks.get(i) or {}).get("named") or []):
                return carrier, namer, families
        return None

    def link_holds(self, carried: str, namer: str, families: set[str], index: dict[str, Any]) -> bool | None:
        """True when the row's series figures among the ones named by its partner row, None if either side is silent."""
        own = set((index["marks"].get(carried) or {}).get("fam") or []) & families
        named = set((index["marks"].get(namer) or {}).get("named") or []) & families
        return bool(own & named) if own and named else None

    def link_faults(self, part: int, host: int, device: str, lines: list[str], index: dict[str, Any]) -> list[str]:
        system = self.link_system(part, host, index)
        if not system:
            return []
        carrier, _, families = system
        marks = index["marks"]
        faults = []
        for line in lines:
            carried, namer = (line, device) if carrier == part else (device, line)
            if self.link_holds(carried, namer, families, index) is not False:
                continue
            wanted = set(marks[namer].get("named") or []) & families
            good = [i for i in self.want_ids(carrier) if set(marks[i].get("fam") or []) & wanted and not self.gone_now(i) and i != carried][:4]
            faults.append(f"{carried} belongs to series {', '.join(sorted(set(marks[carried].get('fam') or []) & families))}, whereas {namer} takes {', '.join(sorted(wanted))}"
                          + (f"; articles of that series seen so far: {', '.join(good)}" if good else f"; search that series together with the article's words, or take another {self.wants[host - 1].what} whose series the wanted lines carry"))
        return faults

    def series_for(self, number: int) -> list[str]:
        """Series under which a tied want gets searched, handiest first: the ones its trolley lines belong to, after them the ones
        that recur most across the tied wants' rows. [] without a series system."""
        if not self.pairs:
            return []
        index = self.compat_index()
        marks = index["marks"]
        counts: dict[str, int] = {}
        held: list[str] = []
        basket = self.basket_ids()
        for part, host in self.pairs:
            system = self.link_system(part, host, index) if number in (part, host) else None
            if not system:
                continue
            for member in (part, host):
                for i in self.want_ids(member):
                    tags = set((marks.get(i) or {}).get("fam") or []) | set((marks.get(i) or {}).get("named") or [])
                    for family in tags & system[2]:
                        counts[family] = counts.get(family, 0) + 1
                        if i in basket and family not in held:
                            held.append(family)
        return (held + [f for f in sorted(counts, key=lambda f: (-counts[f], f)) if f not in held])[:LINK_SEARCHES]

    def pair_faults(self) -> list[str]:
        if not self.pairs:
            return []
        index = self.compat_index()
        held = self.basket_ids()
        faults = []
        for part, host in self.pairs:
            devices = [i for i in held if i in self.want_ids(host)]
            parts = [i for i in held if i in self.want_ids(part)]
            if len(devices) != 1 or not parts:
                continue
            linked = self.link_faults(part, host, devices[0], parts, index)
            if linked:
                faults += linked
                continue
            codes = set(index["marks"].get(devices[0], {}).get("codes") or [])
            if not codes:
                continue
            covers = lambda i: any(_code_hit(p, c) for p in self.compat_set(i, index) for c in codes) and not any(_code_hit(b, c) for b in self.bars_of(i, index) for c in codes)
            for line in parts:
                suits = self.compat_set(line, index)
                if suits and not covers(line):
                    good = [i for i in self.want_ids(part) if covers(i) and not self.gone_now(i) and i not in held][:4]
                    faults.append(f"{line} ({self.wants[part - 1].what}) works with {', '.join(sorted(suits)[:4])}, not with {devices[0]} ({', '.join(sorted(codes))})"
                                  + (f"; add-ons of that kind seen to suit it: {', '.join(good)}" if good else "; no add-on of that kind seen so far suits it, so hunt for one or change the host: " + self.pair_plan(short=True)))
        return faults

    def pair_plan(self, short: bool = False) -> str:
        if not self.pairs:
            return ""
        index = self.compat_index()
        marks = index["marks"]
        lines = []
        for host in sorted({h for _, h in self.pairs}):
            parts = [p for p, h in self.pairs if h == host]
            tally: dict[str, dict[int, int]] = {}
            for p in parts:
                for i in self.want_ids(p):
                    if self.gone_now(i):
                        continue
                    for code in self.compat_set(i, index):
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
                system = self.link_system(p, host, index)
                for i in self.want_ids(p) if system and system[0] == p else []:
                    for code in set((marks.get(i) or {}).get("fam") or []) & system[2] if not self.gone_now(i) else ():
                        tally.setdefault(code, {}).setdefault(p, 0)
                        tally[code][p] += 1
            ranked = sorted(tally.items(), key=lambda kv: (-len(kv[1]), -sum(kv[1].values())))[:6]
            devices = {code: [i for i in self.want_ids(host) if code in ((marks.get(i) or {}).get("named") or []) or any(_code_hit(code, c) for c in marks.get(i, {}).get("codes") or [])][:2] for code, _ in ranked}
            text = "; ".join(f"{code}: " + ", ".join(f"{n} {self.wants[p - 1].what}" for p, n in sorted(per.items())) + (f" (host seen: {', '.join(devices[code])})" if devices[code] else " (host unseen so far)") for code, per in ranked)
            if text:
                lines.append(f"codes or series named by the parts of the {self.wants[host - 1].what} candidates -> {text}")
        if not lines:
            return ""
        return " | ".join(lines) if short else "[pair plan: a host suits an add-on once the add-on's text cites the host's model code or an alias it sells under, or once the host's text cites the series that opens the add-on's part code; run a single search pairing the article's words with a code or a series; when it comes back empty, browse the host's department and note the codes its hits carry] " + " | ".join(lines)

    def attribute_of(self, ident: str, prop: str) -> str | None:
        """Value of the opened article's field named after this property, or None."""
        block = _spec_block(self.pages.get(ident) or {}) or {}
        for key, value in block.items():
            if _same_prop(_bag(prop), _bag(key)):
                return str(value)
        return None

    @staticmethod
    def value_names(value: str, wanted: str) -> bool:
        """Whether an attribute value carries any wanted value (word containment, any of an 'or' list)."""
        have = set(re.findall(r"[a-z0-9]+", str(value).lower()))
        for option in re.split(r"\s*(?:,|/|\bor\b)\s*", wanted.lower()):
            words = set(re.findall(r"[a-z0-9]+", option)) - {"the", "a", "an"}
            if words and words <= have:
                return True
        return False

    def decide_conditionals(self) -> list[str]:
        """Read the trigger product in the basket for each conditional need; return the needs now in force."""
        notes = []
        for rule in self.conditionals:
            lines = [i for i in self.basket_ids() if i in self.want_ids(rule["about"])]
            prop, _, values = rule["test"].partition(":")
            if len(lines) != 1 or not prop.strip() or not values.strip():
                rule["state"] = None
                continue
            seen = self.attribute_of(lines[0], prop)
            if seen is None:
                rule["state"] = None
                continue
            holds = self.value_names(seen, values)
            rule["state"] = holds if rule["mode"] == "only_if" else not holds
            if rule["state"]:
                notes.append(f"want {rule['item']}: {rule['need']} (trigger {prop.strip()}={seen[:40]} on {lines[0]})")
        return notes

    def condition_faults(self) -> list[str]:
        faults = []
        for rule in self.conditionals:
            if rule["state"] is not True:
                continue
            prop, _, value = rule["need"].partition(":")
            if not value.strip() or re.search(BOUND_RX, value, re.I):
                continue
            for line in [i for i in self.basket_ids() if i in self.want_ids(rule["item"])]:
                seen = self.attribute_of(line, prop)
                if seen is not None and not self.value_names(seen, value):
                    faults.append(f"conditional need '{rule['need']}' now applies to want {rule['item']}, but {line} shows {prop.strip()}={seen[:40]}")
        return faults

    # ---------- basket faults ----------
    def basket_faults(self) -> list[str]:
        kind_field = self.kind_field()
        ids = [i for i in self.basket_ids() if (self.units_of(i) or 0) > 0]
        if not ids or not kind_field:
            return []
        label = {i: str((self.cards.get(i) or {}).get(kind_field) or "") for i in ids}
        known = {n for n, w in enumerate(self.wants, 1) if any(self.owned_by(w, str(r.get(kind_field) or "")) for r in self.cards.values())}
        owner = {i: [n for n, w in enumerate(self.wants, 1) if self.owned_by(w, label[i])] for i in ids}
        faults: list[str] = []
        for n, want in enumerate(self.wants, 1):
            if n not in known:
                continue
            lines = [i for i in ids if n in owner[i]]
            if not lines:
                faults.append(f"want {n} ({want.what}) has no line in the basket")
            elif len([i for i in lines if owner[i] == [n]]) > 1:
                faults.append(f"want {n} ({want.what}) has {len(lines)} lines ({', '.join(lines)}); keep one")
            for i in lines:
                have = self.units_of(i)
                if owner[i] == [n] and have != want.count:
                    faults.append(f"{i} is held at {have} unit(s); want {n} ({want.what}) asks for {want.count}")
                price = self.price(i)
                if owner[i] == [n] and want.cap is not None and price is not None and want.cap < price <= 10 * want.cap:
                    faults.append(f"{i} costs {price:g} a unit, over the {want.cap:g} ceiling for each {want.what}")
        if len(known) == len(self.wants):
            for i in ids:
                if not owner[i]:
                    faults.append(f"{i} ({label[i] or 'unknown kind'}) is not a kind the customer asked for: remove it")
        for i in ids:
            if self.gone_now(i):
                faults.append(f"{i} is sold out")
        prices = {i: self.price(i) for i in ids}
        if isinstance(self.limit, (int, float)) and all(p is not None for p in prices.values()):
            scope = (self.intake.get("limit") or {}).get("scope") if isinstance(self.intake.get("limit"), dict) else None
            if scope == "each":
                over = [i for i in ids if self.limit < prices[i] <= 10 * self.limit]
                faults += [f"{i} costs {prices[i]:g}, over the {self.limit:g} limit per product" for i in over]
            else:
                total = sum(prices[i] * self.units_of(i) for i in ids)
                if self.limit + 1e-9 < total <= 10 * self.limit:
                    faults.append(f"the total is {total:g}, over the limit of {self.limit:g}")
        self.decide_conditionals()
        faults += self.condition_faults()
        faults += self.code_faults()
        return faults

    def code_faults(self) -> list[str]:
        """A demand naming a model code the article must suit: the trolley line's own text has to carry that code."""
        faults = []
        for number, need in self.live_needs():
            if not re.search(r"\b(?:suit|fit|work|compatib)", need, re.I):
                continue
            codes = {c.lower() for c in re.findall(r"\b[A-Za-z]{1,5}-?\d{2,6}[A-Za-z]{0,3}\b", need.partition(":")[2])}
            if not codes:
                continue
            for line in [i for i in self.basket_ids() if i in self.want_ids(number)]:
                text = _pack(self.pages.get(line) or self.cards.get(line) or {}).lower()
                missing = [c for c in codes if c not in text and c.replace("-", "") not in text.replace("-", "")]
                if missing and line in self.pages:
                    faults.append(f"{line}'s own text never mentions {', '.join(missing)}, which want {number} must suit ({need})")
        return faults

    def count_gaps(self) -> list[tuple[str, int, int]]:
        kind_field = self.kind_field()
        if not kind_field:
            return []
        out = []
        for ident in self.basket_ids():
            have = self.units_of(ident) or 0
            owners = [n for n, w in enumerate(self.wants, 1) if self.owned_by(w, str((self.cards.get(ident) or {}).get(kind_field) or ""))]
            mine = [i for i in self.basket_ids() if (self.basket.get(i, 1) or 0) > 0 and len(owners) == 1 and self.owned_by(self.wants[owners[0] - 1], str((self.cards.get(i) or {}).get(kind_field) or ""))]
            if have > 0 and len(owners) == 1 and len(mine) == 1 and have != self.wants[owners[0] - 1].count and not self.gone_now(ident):
                out.append((ident, have, self.wants[owners[0] - 1].count))
        return out

    def mend_counts(self, wrong: list[tuple[str, int, int]]) -> list[str]:
        name = self.jobs.get("basket") or ""
        schema = self.specs.get(name) or {}
        props = schema.get("properties") or {}
        texts = [k for k, v in props.items() if isinstance(v, dict) and "string" in _types(v) and "enum" not in v]
        target = next((k for t, k in self.id_slots if t == name and k in texts), None) or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, v in props.items() if isinstance(v, dict) and "integer" in _types(v)), None)
        mode = next((k for k, v in props.items() if isinstance(v, dict) and isinstance(v.get("enum"), list) and len(v["enum"]) >= 2), None)
        if not name or not target or not count:
            return []
        calls: list[dict[str, Any]] = []
        for ident, have, want in wrong:
            if self.basket_absolute(name):
                calls.append({target: ident, count: want})
            elif mode:
                options = [str(o) for o in props[mode]["enum"]]
                off = next((o for o in options if re.search(r"take|clear|drop|delet|remov", o, re.I)), options[1])
                on = next((o for o in options if o != off), options[0])
                calls += [{target: ident, mode: off, count: have}, {target: ident, mode: on, count: want}]
            else:
                return []
        if len(calls) + 3 > self.room():
            return []
        done = []
        for args in calls:
            faults: list[str] = []
            args = _conform(args, schema, "args", faults)
            if faults:
                break
            reply = self.do_step(name, args)
            if reply is None or self.over:
                break
            self.track_basket(name, args)
            self.basket_reply = _pack(reply.get("observation"))
            self.digest(name, reply.get("observation"))
            done.append(_pack(args) + " -> " + self.render(reply.get("observation"))[:300])
        self.log.append({"kind": "count_mend", "calls": done})
        return done

    def ruling_key(self, ids: list[str]) -> str:
        return "|".join(sorted(ids)) + f"#{self.epoch}"

    def judge_basket(self, ids: list[str] | None = None, limit: float = COST_SOFT) -> str:
        """The big model's verdict on the basket (or on a planned set of products) against the sheet."""
        planned = ids is not None
        ids = ids if planned else self.basket_ids()
        key = self.ruling_key(ids)
        if key in self.judged or not self.judge_models(limit) or not ids:
            return ""
        self.judged.add(key)
        self.decide_conditionals()
        prompt = VERDICT + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.sheet()
        if planned:
            counts = "; ".join(f"want {n}: count {w.count}" for n, w in enumerate(self.wants, 1))
            prompt += "\n\nBASKET (planned; judge these products as if basketed at the wanted counts: " + counts + "):\n" + ", ".join(ids)
        else:
            prompt += "\n\nBASKET:\n" + self.basket_text()
        prompt += "\n\nRECORDS OF THE BASKET'S PRODUCTS:\n" + "\n".join(f"{i}: " + ("SOLD OUT (per the site, no longer purchasable). " if self.gone_now(i) else "") + self.card_text(i) for i in ids)
        if self.extreme_pick():
            prompt += "\n\nFURTHER ARTICLES (latest first):\n" + self.other_cards(ids, 15)
        judges = self.judge_models(limit)
        got = None
        if len(judges) > 1 and not planned and not self.extreme_pick() and not self.leads:
            got = self.json_call(prompt, [SMALL], 3000)
            if got:
                judges = [BIG] if got.get("ok") is False else []
        if judges and not self.dry:
            got = self.json_call(prompt, judges, 3000) or got
        self.log.append({"kind": "verdict", "basket": ids, "got": got})
        self.rulings[key] = bool(got) and got.get("ok") is not False
        if not got or got.get("ok") is not False:
            return ""
        self.leads = [i for i in self.leads if i not in ids]
        problems = [str(p) for p in _listed(got.get("problems"))][:5]
        better = [str(b) for b in _listed(got.get("better")) if str(b) in self.cards and not self.gone_now(str(b))][:4]
        text = "A careful review finds this trolley not right yet: " + "; ".join(problems or ["see the requirements"])
        if better:
            text += ". Articles looking better: " + ", ".join(better)
        return text + ". Mend the trolley (or refute the review through a fresh look at the records), then pay."

    # ---------- needs that refer to another product's page ----------
    def follow_chain(self) -> tuple[str, bool]:
        live = {need for _, need in self.live_needs()}
        pending = [need for _, need in self.pointers if need in live and need not in self.pointed_values]
        tool = self.jobs.get("deal") or self.jobs.get("page")
        target = self.id_slot(tool)
        seen: list[str] = []
        lost: list[str] = []
        while pending and self.chain_runs < min(6 if self.room() > 15 else 4, self.chain_cap or 6) and self.spent < COST_HARD and not self.late():
            self.chain_runs += 1
            records = []
            for ident in reversed(self.arrival):
                record = self.pages.get(ident) or self.cards.get(ident) if ident in self.visited else None
                if record and len(records) < 45:
                    records.append("OPENED " + _pack(_trim(record, 300))[:1400])
            for ident in reversed(self.arrival):
                if len(records) >= 60:
                    break
                if ident not in self.visited or not (self.pages.get(ident) or self.cards.get(ident)):
                    records.append("ROW " + _pack(_trim(self.cards.get(ident), 80))[:300])
            prompt = CHAIN + "\n\nREQUIREMENTS WITH THEIR VALUE ON ANOTHER PRODUCT:\n" + "\n".join(pending)
            prompt += "\n\nRECORDS (latest first):\n" + "\n".join(records)
            got = self.json_call(prompt, [SMALL] + ([BIG] if self.spent + BIG_MARGIN <= COST_HARD else []), 4500) or {}
            self.log.append({"kind": "chain", "pending": pending, "got": _trim(got, 150)})
            reach: list[str] = []
            lost = []
            gaps: list[tuple[str, str]] = []
            for chain in _listed(got.get("chains")):
                if not isinstance(chain, dict):
                    continue
                need = next((p for p in pending if p == str(chain.get("need") or "").strip()), None) or (pending[0] if len(pending) == 1 else None)
                if not need:
                    continue
                links = [l for l in _listed(chain.get("links")) if isinstance(l, dict)]
                ids = [str(l.get("id") or "").strip() for l in links]
                if chain.get("done") is True and links and all(i in self.visited for i in ids) and str(chain.get("value") or "").strip():
                    self.pointed_values[need] = str(chain["value"]).strip()
                    self.epoch += 1
                    continue
                if links and all(i in self.visited for i in ids) and not str(links[-1].get("find") or "").strip():
                    self.pointed_values[need] = str(chain.get("value") or "").strip() or f"as stated on {ids[-1]}"
                    self.epoch += 1
                    continue
                unopened = [i for i in ids if i in self.cards and i not in self.visited]
                if unopened:
                    reach += [i for i in unopened if i not in reach]
                else:
                    gap = next((l for l, i in zip(links, ids) if i not in self.cards), {})
                    about = str(gap.get("about") or "") or "the product it refers to"
                    words = str(gap.get("find") or "").strip()
                    lost.append(f"{need} -> find {about}" + (f" (search: {words})" if words else ""))
                    gaps.append((about, words))
            pending = [p for p in pending if p not in self.pointed_values]
            if not reach and gaps and tool and target and self.room() > 12 and self.link_hunts < 3:
                reach = self.hunt_link(*gaps[0])
                if self.log and self.log[-1].get("kind") == "link_known":
                    seen.append(f"{self.log[-1]['id']}: an opened page of the shop this chain names that shows the values its description gives; read the value from it")
                    lost = lost[1:]
            if not reach or not tool or not target or self.room() < len(reach) + 4 or self.over:
                break
            for ident in reach[:max(0, min(3, 6 - len(seen)))]:
                faults: list[str] = []
                args = _conform({target: ident}, self.specs.get(tool), "args", faults)
                if faults:
                    break
                reply = self.do_step(tool, args)
                if reply is None or self.over:
                    break
                self.digest(tool, reply.get("observation"))
                seen.append(f"{ident}: " + self.render(reply.get("observation"))[:1500])
        text = ""
        if seen:
            text += "[Wren opened the views cited by a demand]\n" + "\n".join(seen) + "\n"
        if self.pointed_values:
            text += "[values read off the views that demands cite] " + "; ".join(f"{k} -> {v}" for k, v in self.pointed_values.items()) + "\n"
        hold = bool(lost) and self.chain_blocks < (3 if self.room() > 15 else 2) and self.room() > 5
        if hold:
            self.chain_blocks += 1
            text += ("The purchase is valid solely once each view cited by a demand was fully opened (a hit alone will not do), following every view forwarding to a further article. Still to track down and open: " + " | ".join(lost[:3])
                     + ". Search using that shop's name plus the values from its description (that shop's other articles of the kind are different articles), open that shop's hit whose view carries those values, use the value that view states, then pay.")
        return text, hold

    def early_chain(self) -> str:
        """Right after the interview, follow the demands that cite another article, a few rounds only, while enough
        moves remain to use what they reveal (Cairn: the checkout-time follow alone came too late in a third of its chains)."""
        live = {need for _, need in self.live_needs()}
        if self.room() < 15 or self.late() or not any(need in live and need not in self.pointed_values for _, need in self.pointers):
            return ""
        kept = (self.chain_runs, self.chain_blocks)
        self.chain_runs, self.chain_cap = 0, EARLY_CHAINS
        try:
            text, _ = self.follow_chain()
        finally:
            (self.chain_runs, self.chain_blocks), self.chain_cap = kept, None
        self.log.append({"kind": "early_chain", "room": self.room(), "values": list(self.pointed_values.values())[:3]})
        return text

    def refuse_trail(self) -> str:
        """Once, before a bow-out with moves left: chase whatever a demand refers to; returns what turned up, '' if nothing did."""
        live = {need for _, need in self.live_needs()}
        if self.refuse_trailed or self.room() < 15 or self.late() or not any(need in live and need not in self.pointed_values for _, need in self.pointers):
            return ""
        self.refuse_trailed = True
        text, _ = self.follow_chain()
        return text + "Search those values at once and buy whatever passes; bow out solely when nothing does." if text else ""

    def seller_rows(self, about: str, ids: list[str]) -> list[str]:
        low = about.lower()
        out = []
        for ident in ids:
            row = self.cards.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(r"merchant|maker|brand|vendor|store|shop|seller", str(k), re.I)]
            if ident not in self.visited and any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low) for n in names):
                out.append(ident)
        return out

    def seller_pages(self, about: str) -> list[str]:
        low = about.lower()
        out = []
        for ident in self.visited:
            row = self.cards.get(ident) or {}
            names = [str(v).strip() for k, v in row.items() if isinstance(v, str) and 2 < len(v.strip()) <= 40 and re.search(r"merchant|maker|brand|vendor|store|shop|seller", str(k), re.I)]
            if any(re.search(r"(?<![a-z0-9])" + re.escape(n.lower()) + r"(?![a-z0-9])", low) for n in names):
                out.append(ident)
        return out

    def hunt_link(self, about: str, words: str) -> list[str]:
        """Hunt the product that a chain link depicts, by the shop's name plus the description's values."""
        known: list[str] = []
        named = {w for ident in self.seller_rows(about, list(self.cards)) + self.seller_pages(about) for k, v in (self.cards.get(ident) or {}).items() if isinstance(v, str) and len(v.strip()) <= 40 and re.search(r"merchant|maker|brand|vendor|store|shop|seller", str(k), re.I) for w in re.findall(r"[a-z0-9]{3,}", v.lower())}
        stop = {"the", "and", "with", "from", "that", "comes", "listing", "page", "its", "for", "of", "product"}
        values = {w for w in re.findall(r"[a-z0-9]+", (words or about).lower()) if len(w) >= 3 or w.isdigit()} - named - stop
        for ident in self.seller_pages(about):
            stack, leaves = [self.pages.get(ident) or self.cards.get(ident) or {}], []
            while stack:
                node = stack.pop()
                if isinstance(node, (dict, list)):
                    stack += list(node.values()) if isinstance(node, dict) else node
                else:
                    leaves.append(str(node))
            text = " ".join(leaves).lower()
            if values and all(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", text) for v in values):
                self.log.append({"kind": "link_known", "id": ident})
                return []
        plan = self.query_pattern()
        memo = " ".join(words.lower().split())
        if memo and memo in self.link_found:
            self.link_hunts += 1
            known = self.seller_rows(about, self.link_found[memo])
        elif plan and words:
            tool, base, field = plan
            faults: list[str] = []
            args = _conform({**base, field: words}, self.specs.get(tool), "args", faults)
            if not faults:
                self.link_hunts += 1
                reply = self.do_step(tool, args)
                if reply is not None:
                    observation = reply.get("observation")
                    self.digest(tool, observation)
                    self.link_found[memo] = [i for i, _ in self.cards_in(observation)]
                    known = self.seller_rows(about, self.link_found[memo])
                    self.log.append({"kind": "link_hunt", "words": words[:80], "rows": known[:4]})
        kind = {w for w in re.findall(r"[a-z0-9]{3,}", (words or about).lower())} - named - stop

        def of_kind(ids: list[str]) -> list[str]:
            return [i for i in ids if not kind or kind & set(re.findall(r"[a-z0-9]{3,}", _pack(self.cards.get(i) or {}).lower()))]
        return (of_kind(known) or of_kind(self.seller_rows(about, list(reversed(self.arrival)))))[:2]

    # ---------- stock and price refreshes ----------
    def deal_view(self) -> str | None:
        return self.jobs.get("deal") or self.jobs.get("page")

    def refresh_deals(self) -> str:
        tool = self.deal_view()
        target = self.id_slot(tool)
        if not tool or not target:
            return ""
        since = max(self.at - 2, self.bounced_at)
        stale = [i for i in self.basket_ids() if self.deal_seen.get(i, -99) < since and i not in self.vanished]
        if not stale or self.room() < len(stale) + 3:
            return ""
        lines = []
        for ident in stale:
            faults: list[str] = []
            args = _conform({target: ident}, self.specs.get(tool), "args", faults)
            if faults:
                break
            before = dict(self.cards.get(ident) or {})
            reply = self.do_step(tool, args)
            if reply is None or self.over:
                break
            observation = reply.get("observation")
            self.deal_seen[ident] = self.at
            self.digest(tool, observation)
            after = self.cards.get(ident) or {}
            moved = [k for k, v in after.items() if k in before and before[k] != v and not isinstance(v, (dict, list))]
            if moved:
                lines.append(f"{ident} changed ({', '.join(moved[:4])}): {_pack(_trim(observation, 300))[:900]}")
        self.log.append({"kind": "refresh", "ids": stale, "changed": len(lines)})
        if lines:
            self.epoch += 1
        return "\n".join(lines)

    def gone_now(self, ident: str) -> bool:
        return ident in self.unbuyable or ident in self.vanished

    def swap_note(self) -> str:
        lost = [i for i in self.basket_ids() if i not in self.vanished and self.gone_now(i)]
        if not lost:
            return ""
        for ident in lost:
            self.vanished[ident] = self.at
        self.leads = [i for i in self.leads if i not in lost]
        self.epoch += 1
        picks = self.stand_ins(lost)
        text = f"[Wren] Sold out now: {', '.join(lost)} (per the site, unbuyable). It stays gone, so never open it again."
        if picks and self.room() <= 6:
            return text + f" Few moves remain: this turn, add {picks[0]} at the wanted count and drop the sold-out line (two calls), then pay. Further opened articles that seem to pass: {', '.join(picks[1:]) or 'none'}."
        if picks and self.choose() == "cheapest":
            return text + f" Opened articles that seem to pass for that want instead, cheapest leading: {', '.join(picks)}. The rule is cheapest, so unless the whole passing set is already seen, run a search capped under {picks[0]}'s price (via the search's maximum-price parameter, if any) before swapping the sold-out line for the cheapest passing article."
        if picks:
            return text + f" Opened articles that seem to pass for that want instead: {', '.join(picks)}. This turn, drop the sold-out line and add {picks[0]} at the wanted count, then pay."
        return text + " Drop that line and trolley the best alternative article that passes; search for one when no opened article does."

    def spot_sold(self) -> str:
        ids = [i for i in self.basket_ids() if i not in self.vanished]
        if any(self.gone_now(i) for i in ids):
            return ""
        if len(ids) == 1:
            self.unbuyable.add(ids[0])
            return ""
        return self.refresh_deals() if ids else ""

    def stand_ins(self, lost: list[str]) -> list[str]:
        if not self.judge_models():
            return []
        records = []
        for ident in reversed(self.arrival):
            if ident in lost or ident in self.vanished or self.gone_now(ident):
                continue
            records.append(_pack(_trim(self.pages.get(ident) or self.cards.get(ident), 200))[:1200])
            if len(records) >= 25:
                break
        if not records:
            return []
        prompt = RESCUE + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.sheet()
        prompt += "\n\nSOLD-OUT BASKET LINE:\n" + "\n".join(f"{i}: {self.card_text(i)}" for i in lost) + "\n\nBASKET NOW:\n" + self.basket_text()
        prompt += "\n\nFURTHER ARTICLE RECORDS (latest first):\n" + "\n".join(records)
        got = self.json_call(prompt, self.judge_models(), 3500)
        fits: list[tuple[float, str]] = []
        for entry in _listed((got or {}).get("fits")):
            ident = str(entry.get("id") if isinstance(entry, dict) else entry)
            price = entry.get("price") if isinstance(entry, dict) else None
            if ident in self.cards and ident not in lost and ident not in self.vanished and ident not in self.basket_ids() and ident not in [i for _, i in fits]:
                fits.append((float(price) if isinstance(price, (int, float)) and not isinstance(price, bool) else float("inf"), ident))
        ranked = [i for _, i in sorted(fits, key=lambda f: f[0])] if self.choose() == "cheapest" else [i for _, i in fits]
        self.log.append({"kind": "stand_ins", "lost": lost, "fits": fits})
        return ranked[:4]

    # ---------- before a walk-away ----------
    def review(self) -> str:
        if self.reviewed or not self.judge_models():
            return ""
        self.reviewed = True
        got = self.review_call()
        self.take_review(got)
        named = self.review_fits(got)
        missing = [n for n in range(1, len(self.wants) + 1) if not named.get(str(n))]
        if missing:
            rows = [str(i) for i in _listed((got or {}).get("open")) if str(i) in self.cards and str(i) not in self.visited and not self.gone_now(str(i))]
            opened = bool(rows) and self.open_leads(rows)
            if self.recovery_search(got, missing) or opened:
                got = self.review_call()
                self.take_review(got)
                named = self.review_fits(got) or named
        if not named:
            return ""
        self.lead_map = named
        self.leads = [i for ids in named.values() for i in ids]
        return "A careful review suspects these articles pass: " + "; ".join(f"want {k}: {', '.join(v)}" for k, v in named.items()) + (f" ({got.get('reason')})" if got.get("reason") else "") + ". Open them, buy whichever truly passes, and bow out only once every one has failed."

    def seen_records(self, cap: int = 30) -> list[str]:
        money = self.limit if isinstance(self.limit, (int, float)) else None
        kind_field = self.kind_field()
        for strict in (True, False):
            opened: list[str] = []
            rows: list[str] = []
            for ident in reversed(self.arrival):
                label = str((self.cards.get(ident) or {}).get(kind_field) or "") if kind_field else ""
                if self.gone_now(ident) or (strict and ((money is not None and (self.price(ident) or 0) > money) or (label and not any(self.owned_by(w, label) for w in self.wants)))):
                    continue
                record = self.pages.get(ident) if ident in self.pages else None
                if record and len(opened) < cap * 2 // 3:
                    opened.append("OPENED " + _pack(_trim(record, 200))[:1000])
                elif not record and len(rows) < cap // 2:
                    rows.append("ROW " + _pack(_trim(self.cards.get(ident), 120))[:500])
            if opened or rows:
                return opened + rows
        return []

    def review_call(self) -> dict[str, Any] | None:
        records = self.seen_records()
        if not records:
            return None
        prompt = REVIEW + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.sheet()
        answers = [line for line in self.diary if "message to the customer" in line]
        if answers:
            prompt += "\n\nREPLIES FROM THE CLIENT:\n" + "\n".join(answers)
        tried = [str(v) for args in self.queries for v in args.values() if isinstance(v, str)]
        if tried:
            prompt += "\n\nSEARCHES TRIED SO FAR:\n" + " | ".join(dict.fromkeys(tried))
        prompt += "\n\nARTICLE RECORDS (latest first):\n" + "\n".join(records)
        got = self.json_call(prompt, self.judge_models(), 4000)
        self.log.append({"kind": "review", "got": got})
        return got

    def review_fits(self, got: dict[str, Any] | None) -> dict[str, list[str]]:
        fits = (got or {}).get("fits")
        if not isinstance(fits, dict):
            return {}
        named = {str(k): [str(i) for i in _listed(v) if str(i) in self.cards and not self.gone_now(str(i))][:4] for k, v in fits.items()}
        return {k: v for k, v in named.items() if v}

    def open_leads(self, rows: list[str]) -> bool:
        many, joiner = self.side_slot(), self.jobs.get("side")
        tool = self.spec_view()
        target = self.id_slot(tool)
        if many and joiner and not self.side_blind():
            plan = [(joiner, {many: rows[:self.side_cap()]})]
        elif tool and target:
            plan = [(tool, {target: i}) for i in rows[:2]]
        else:
            return False
        if self.room() < len(plan) + 5:
            return False
        before = len(self.pages)
        for name, args in plan:
            faults: list[str] = []
            args = _conform(args, self.specs.get(name), "args", faults)
            if faults:
                return False
            reply = self.do_step(name, args)
            if reply is None or self.over:
                break
            self.digest(name, reply.get("observation"))
        self.log.append({"kind": "review_open", "rows": rows, "opened": len(self.pages) - before})
        return len(self.pages) > before

    def nearest_miss(self) -> dict[str, Any] | None:
        """When the customer wants the closest product named on a walk-away: rank seen products by failed checks,
        then take the cheapest among those with the fewest misses (the message's price slack permits dearer ones,
        but the cheapest of them is always acceptable)."""
        if self.nearest is not None:
            return self.nearest or None
        self.nearest = {}
        models = self.judge_models(COST_HARD)
        records = self.seen_records(40)
        if not models or not records:
            return None
        prompt = CLOSEST + "\n\nCUSTOMER'S MESSAGE:\n" + self.ask_text + "\n\nREQUIREMENTS IN FORCE:\n" + self.sheet()
        if self.fallback.get("rule"):
            prompt += "\n\nTHE CUSTOMER'S OWN RULE FOR THE CLOSEST PRODUCT: " + self.fallback["rule"]
        prompt += "\n\nARTICLE RECORDS (latest first):\n" + "\n".join(records)
        got = self.json_call(prompt, models, 3000) or {}
        ranked = []
        for entry in _listed(got.get("ranked")):
            if not isinstance(entry, dict):
                continue
            ident = str(entry.get("id") or "").strip()
            misses = _whole(entry.get("misses"))
            if ident not in self.cards or misses is None:
                continue
            price = _money(entry.get("price"))
            if price is None:
                price = self.price(ident)
            ranked.append({"id": ident, "item": _whole(entry.get("item")) or 0, "price": price, "misses": misses, "short_on": [str(s) for s in _listed(entry.get("short_on"))][:4]})
        self.log.append({"kind": "nearest", "ranked": _trim(ranked, 80)})
        if not ranked:
            return None
        fewest = min(r["misses"] for r in ranked)
        pool = [r for r in ranked if r["misses"] == fewest]
        priced = [r for r in pool if r["price"] is not None]
        best = min(priced, key=lambda r: r["price"]) if priced else pool[0]
        self.nearest = best
        return best

    def exit_words(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Fill the walk-away tool's remark the way the customer asked: quiet, or naming the nearest product."""
        schema = self.specs.get(name) or {}
        props = schema.get("properties") or {}
        out = dict(args)
        plain = "Sorry, the site holds no article meeting all of your demands, so nothing will be bought."
        offered = ""
        if self.fallback.get("offer_nearest") and self.fallback.get("quiet") is not True:
            best = self.nearest_miss()
            if best:
                title = self.title_of(best["id"])
                offered = (f"Nothing on the site meets every demand, so nothing was bought. The nearest fit is {best['id']}" + (f" ({title})" if title else "")
                           + (f" at {best['price']:g}" if best.get("price") is not None else "") + (", short on: " + "; ".join(best["short_on"]) if best["short_on"] else "") + ".")
        for key, spec in props.items():
            if not isinstance(spec, dict) or "string" not in _types(spec) or "enum" in spec:
                continue
            value = out.get(key)
            if value is None and key not in (schema.get("required") or []):
                continue
            text = str(value or "")
            cap = spec.get("maxLength") if isinstance(spec.get("maxLength"), int) else 400
            if offered:
                out[key] = offered[:cap]
                continue
            names_one = any(ident in text for ident in self.cards)
            if not text or (names_one and self.fallback.get("quiet") is not False):
                out[key] = plain[:cap]
        return out

    # ---------- basket calls ----------
    def basket_calls(self, lines: dict[str, int], drop: list[str]) -> list[dict[str, Any]] | None:
        name = self.jobs.get("basket") or ""
        schema = self.specs.get(name) or {}
        props = schema.get("properties") or {}
        for key, spec in props.items():
            shape = spec.get("items") if isinstance(spec, dict) and "array" in _types(spec) else None
            if isinstance(shape, dict) and isinstance(shape.get("properties"), dict):
                inner = shape["properties"]
                id_key = next((k for k, s in inner.items() if isinstance(s, dict) and "string" in _types(s) and "enum" not in s), None)
                count_key = next((k for k, s in inner.items() if isinstance(s, dict) and "integer" in _types(s)), None)
                if not id_key:
                    return None
                return [{key: [{id_key: i, **({count_key: n} if count_key else {})} for i, n in lines.items()]}]
        texts = [k for k, s in props.items() if isinstance(s, dict) and "string" in _types(s) and "enum" not in s]
        target = next((k for t, k in self.id_slots if t == name and k in texts), None) or next((k for k in schema.get("required") or [] if k in texts), None) or (texts[0] if len(texts) == 1 else None)
        count = next((k for k, s in props.items() if isinstance(s, dict) and "integer" in _types(s)), None)
        mode = next((k for k, s in props.items() if isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2), None)
        if not target:
            return None
        calls: list[dict[str, Any]] = []
        for ident in drop:
            have = self.basket.get(ident, 1)
            if mode:
                calls.append({target: ident, mode: props[mode]["enum"][1], **({count: have} if count else {})})
            elif count and self.basket_absolute(name):
                calls.append({target: ident, count: 0})
            elif count and isinstance(props[count].get("minimum"), int) and props[count]["minimum"] <= -have:
                calls.append({target: ident, count: -have})
            else:
                return None
        for ident, want in lines.items():
            have = self.basket.get(ident, 0) if ident in self.basket_ids() else 0
            if have == want:
                continue
            if have and not self.basket_absolute(name):
                return None
            calls.append({target: ident, **({mode: props[mode]["enum"][0]} if mode else {}), **({count: want} if count else {})})
        return calls

    def finish_basket(self) -> str:
        """When the walker insists on quitting although a reviewed product was vouched for: basket it and check out."""
        basket, buy = self.jobs.get("basket"), self.jobs.get("checkout")
        if self.autofilled or not self.vouched or self.gone_now(self.vouched) or not basket or not buy:
            return ""
        self.autofilled = True
        kind_field = self.kind_field()

        def kind(ident: str) -> str:
            return str((self.cards.get(ident) or {}).get(kind_field) or "").strip().lower() if kind_field else ident
        plan: dict[str, int] = {}
        for number, ids in self.lead_map.items():
            index = _whole(number)
            pick = next((i for i in ids if not self.gone_now(i)), None)
            if pick and index and 1 <= index <= len(self.wants) and kind(pick) not in {kind(p) for p in plan}:
                plan[pick] = self.wants[index - 1].count
        if self.vouched not in plan:
            return ""
        now = self.basket_ids()
        keep = {i: self.basket.get(i, 1) for i in now if i not in plan and not self.gone_now(i) and kind(i) not in {kind(p) for p in plan}}
        drop = [i for i in now if i not in plan and i not in keep]
        if len(plan) + len({kind(i) for i in keep}) < len(self.wants):
            return ""
        lines = {**keep, **plan}
        calls = self.basket_calls(lines, drop)
        if not calls or len(calls) + 1 > self.room():
            return ""
        self.open_specs(list(lines), len(calls) + 3)
        self.judge_basket(list(lines), COST_HARD)
        if not self.rulings.get(self.ruling_key(list(lines))):
            return ""
        done = []
        for args in calls:
            faults: list[str] = []
            args = _conform(args, self.specs.get(basket), "args", faults)
            if faults:
                return ""
            reply = self.do_step(basket, args)
            if reply is None or self.over:
                return ""
            self.track_basket(basket, args)
            self.digest(basket, reply.get("observation"))
            done.append(_pack(args))
        faults = []
        args = _conform(dict(self.intake.get("checkout_args") or {}), self.specs.get(buy), "args", faults)
        if faults:
            return ""
        replies = []
        for attempt in range(2):
            if attempt:
                self.refresh_deals()
                self.judge_basket(list(lines), COST_HARD)
                if not self.rulings.get(self.ruling_key(list(lines))) or self.room() < 1 or self.over or any(self.gone_now(i) for i in lines):
                    break
            go, said = self.chore_gate("order")
            if said:
                replies.append(said)
            if not go:
                break
            reply = self.do_step(buy, args)
            self.log.append({"kind": "autofilled", "lines": lines, "drop": drop, "reply": _trim(reply, 200)})
            if reply is None:
                break
            self.checkouts += 1
            if not reply.get("finished"):
                self.bounced_at = self.at
            self.digest(buy, reply.get("observation"))
            replies.append(self.render(reply.get("observation")))
            if self.over or reply.get("finished"):
                break
        if not replies:
            return ""
        return "Skipped (Wren acted instead): an independent check vouched for " + self.vouched + ", so Wren loaded the trolley (" + "; ".join(done) + ") and paid. Site reply: " + " | then: ".join(replies)

    def is_checkout(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.jobs.get("checkout"):
            return False
        fixed = self.intake.get("checkout_args") if isinstance(self.intake.get("checkout_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def is_exit(self, name: str, args: dict[str, Any]) -> bool:
        if name != self.jobs.get("walk_away"):
            return False
        fixed = self.intake.get("walk_away_args") if isinstance(self.intake.get("walk_away_args"), dict) else {}
        return all(args.get(k) == v for k, v in fixed.items())

    def menu(self) -> list[dict[str, Any]]:
        """The tools offered to the walker this turn: narrowed to closing tools when steps are nearly gone."""
        left = self.room()
        talk = self.jobs.get("talk")
        keep: set[str] | None = None
        closers = {self.jobs.get("checkout"), self.jobs.get("walk_away")} - {None}
        if left <= self.keep_back():
            keep = set(closers)
        elif left <= self.keep_back() + 3:
            keep = set(closers) | {self.jobs.get("basket"), self.jobs.get("basket_view")}
        elif left <= len(self.wants) + 5 and self.checkouts == 0:
            keep = set(closers) | {self.jobs.get("basket"), self.jobs.get("basket_view"), self.jobs.get("deal")}
        tools = [t for t in self.toolset if t["function"]["name"] != talk]
        if keep is not None and closers:
            narrowed = [t for t in tools if t["function"]["name"] in keep]
            tools = narrowed or tools
        return tools

    def fix_ids(self, name: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
        """Swap a title typed where a product id belongs for the id, when exactly one product carries that title."""
        if name not in {self.jobs.get(r) for r in ("basket", "page", "deal", "side")} - {None} or not self.id_name:
            return args, ""
        props = (self.specs.get(name) or {}).get("properties") or {}
        free = [k for k, v in props.items() if isinstance(v, dict) and ("string" in _types(v) or "array" in _types(v)) and "enum" not in v]
        if len(free) != 1 or (name != self.jobs.get("basket") and (name, free[0]) not in self.id_slots):
            return args, ""

        def mend(value: Any) -> tuple[Any, str]:
            if isinstance(value, str) and value.strip() in self.cards:
                return value.strip(), ""
            if not isinstance(value, str) or _looks_id(value, True) or len(value.strip()) < 8:
                return value, ""
            low = value.strip().lower()
            hits = [i for i, row in self.cards.items() if any(k != self.id_name and isinstance(v, str) and v.strip().lower().startswith(low) for k, v in row.items())]
            if len(hits) == 1:
                return hits[0], ""
            if hits:
                return value, f"several products carry the text {value!r} ({', '.join(hits[:6])}); pass one product id"
            return value, f"{value!r} is not a product id; pass the id from the result rows (their {self.id_name} field)"
        out = dict(args)
        for key in free:
            if key not in args:
                continue
            fixed = [mend(v) for v in args[key]] if isinstance(args[key], list) else [mend(args[key])]
            trouble = next((p for _, p in fixed if p), "")
            if trouble:
                return args, trouble
            out[key] = [v for v, _ in fixed] if isinstance(args[key], list) else fixed[0][0]
        return out, ""

    def basket_absolute(self, name: str) -> bool:
        """Whether the basket tool sets a line's count outright (a count argument whose minimum is 0, no mode)."""
        props = self.specs.get(name, {}).get("properties") or {}
        if any(isinstance(s, dict) and isinstance(s.get("enum"), list) and len(s["enum"]) >= 2 for s in props.values()):
            return False
        return any(isinstance(s, dict) and "integer" in _types(s) and s.get("minimum") == 0 for s in props.values())

    def track_basket(self, name: str, args: dict[str, Any]) -> None:
        self.basket_moved = True
        self.basket_at = self.at
        spec = self.specs.get(name, {}).get("properties") or {}
        removal = False
        for key, value in args.items():
            options = (spec.get(key) or {}).get("enum")
            if isinstance(options, list) and len(options) >= 2 and value in options[1:]:
                removal = True
        loose_count = next((_whole(v) for v in args.values() if _whole(v) is not None and not isinstance(v, str)), None)
        lines: list[tuple[str, int | None]] = []
        whole = False
        for key, value in args.items():
            if isinstance(value, list):
                shape = (spec.get(key) or {}).get("items") or {}
                if not value and isinstance(shape, dict) and (shape.get("type") == "object" or "properties" in shape):
                    whole = True
                for line in value:
                    if isinstance(line, dict):
                        whole = True
                        ident = next((v for v in line.values() if isinstance(v, str) and v in self.cards), None)
                        count = next((_whole(v) for v in line.values() if _whole(v) is not None and not isinstance(v, str)), 1)
                        if ident:
                            lines.append((ident, count))
                    elif isinstance(line, str) and line in self.cards:
                        lines.append((line, loose_count))
            elif isinstance(value, str) and value in self.cards:
                lines.append((value, loose_count))
        if whole:
            self.basket = {i: c for i, c in lines if c and c > 0}
            return
        absolute = self.basket_absolute(name)
        for ident, count in lines:
            have = self.basket.get(ident, 0)
            if removal:
                have = have - count if count else 0
            elif absolute:
                have = count if count is not None else 1
            else:
                have = have + (count if count is not None else 1)
            if have > 0:
                self.basket[ident] = have
            else:
                self.basket.pop(ident, None)

    def render(self, observation: Any) -> str:
        text = _pack(_trim(observation, SHOW_LEN))
        if len(text) > OBS_CAP:
            text = text[:OBS_CAP] + "...(cut)"
        return text

    def side_slot(self) -> str | None:
        schema = self.specs.get(self.jobs.get("side") or "") or {}
        props = schema.get("properties") or {}
        target = next((k for k, v in props.items() if isinstance(v, dict) and "array" in _types(v)), None)
        if not target or any(k != target for k in schema.get("required") or []):
            return None
        return target

    def side_cap(self) -> int:
        spec = ((self.specs.get(self.jobs.get("side") or "") or {}).get("properties") or {}).get(self.side_slot() or "") or {}
        if isinstance(spec.get("maxItems"), int):
            return spec["maxItems"]
        if self.side_bad:
            return max(2, min(self.side_bad - 1, max(self.side_ok, 2)))
        return max(4, self.side_ok)

    def merge_opens(self, calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Several single-article opens within a turn collapse into one multi-article call where the site allows."""
        opener, joiner, target = self.jobs.get("page"), self.jobs.get("side"), self.id_slot(self.jobs.get("page"))
        many = self.side_slot()
        if not opener or not joiner or not target or not many or opener == joiner:
            return calls
        if self.side_blind():
            return calls
        picks: list[tuple[int, str]] = []
        for index, call in enumerate(calls):
            fn = call.get("function") or {}
            args = _json_of(fn.get("arguments")) if isinstance(fn.get("arguments"), str) else None
            ident = str((args or {}).get(target) or "").strip()
            if fn.get("name") == opener and args and set(args) == {target} and ident in self.cards and ident not in self.vanished:
                picks.append((index, ident))
        picks = picks[:self.side_cap()]
        if len(picks) < 2:
            return calls
        first = picks[0][0]
        out = []
        for index, call in enumerate(calls):
            if index == first:
                out.append({"id": call["id"], "type": "function", "function": {"name": joiner, "arguments": _pack({many: [i for _, i in picks]})}})
            elif index in {i for i, _ in picks}:
                out.append({"id": call["id"], "merged": True})
            else:
                out.append(call)
        self.log.append({"kind": "merged", "ids": [i for _, i in picks]})
        return out

    # ---------- running the walker's tool calls ----------
    def execute(self, calls: list[dict[str, Any]]) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        calls = self.merge_opens(calls)
        for index, call in enumerate(calls):
            allowed = {t["function"]["name"] for t in self.menu()}
            cid = call["id"]
            if call.get("merged"):
                out.append((cid, "See the multi-article reply above: Wren opened these articles jointly in a single move."))
                continue
            fn = call.get("function") or {}
            name = fn.get("name")
            raw = fn.get("arguments")
            if self.over:
                out.append((cid, "Skipped: the episode is already over."))
                continue
            if name == self.jobs.get("talk"):
                out.append((cid, "Skipped (no step charged): notes to the client go through Wren's interviewer alone."))
                continue
            if name not in self.specs:
                out.append((cid, f"Skipped (no step charged): unknown tool. Tools: {', '.join(sorted(allowed))}."))
                continue
            if name not in allowed:
                out.append((cid, "Skipped (no step charged): too few moves remain for that; use: " + ", ".join(sorted(allowed)) + "."))
                continue
            if isinstance(raw, dict):
                args = raw
            else:
                args = _json_of(raw) if (raw or "").strip() else {}
                if args is None:
                    out.append((cid, "Skipped (no step charged): the parameters did not form a single JSON object."))
                    continue
            faults: list[str] = []
            args = _conform(args, self.specs[name], "args", faults)
            if faults:
                out.append((cid, "Skipped (no step charged): " + "; ".join(faults[:6])))
                continue
            args, wrong = self.fix_ids(name, args)
            if wrong:
                out.append((cid, "Skipped (no step charged): " + wrong + "."))
                continue
            trimmed = ""
            many = self.side_slot() if name == self.jobs.get("side") else None
            if many and isinstance(args.get(many), list) and len(args[many]) > self.side_cap() and self.side_bad:
                cap = self.side_cap()
                trimmed = f"\n[Wren] Sent the first {cap} products only: this shop shows at most {cap} side by side. Not shown yet: {', '.join(map(str, args[many][cap:]))}."
                args = {**args, many: args[many][:cap]}
            target = self.id_slot(name)
            if target and name in (self.jobs.get("page"), self.jobs.get("deal")) and str(args.get(target)) in self.vanished:
                out.append((cid, f"Skipped (no step charged): {args[target]} sold out at move {self.vanished[str(args[target])]} and stays gone. Exchange its trolley line for another passing article."))
                continue
            if name == self.jobs.get("basket") and any((_whole(v) or 0) < 0 for v in args.values() if not isinstance(v, str)):
                snap = _pack(self.basket_snap) if self.basket_seen >= self.basket_at else ""
                absent = [v for v in args.values() if isinstance(v, str) and v in self.cards and v not in self.basket and v not in snap and v not in self.basket_reply]
                if absent:
                    out.append((cid, f"Skipped (no step charged): {absent[0]} is absent from the trolley, so nothing can be removed. Trolley now: " + self.basket_text()))
                    continue
            closing = self.is_checkout(name, args) or self.is_exit(name, args)
            if closing and len(calls) > 1:
                out.append((cid, "Skipped (no step charged): the finishing call goes alone, once your other replies are read."))
                continue
            if not closing and self.room() <= self.keep_back() and (self.jobs.get("checkout") or self.jobs.get("walk_away")):
                out.append((cid, "Skipped (no step charged): the final moves stay reserved for paying (with the client's approval of the trolley where due) or for bowing out."))
                continue
            mark = name + _pack(args)
            if not closing and name != self.jobs.get("basket") and mark in self.just_did:
                out.append((cid, "Skipped (no step charged): the same call just ran and the state is unchanged; see its reply above."))
                continue
            readers = {self.jobs.get(r) for r in ("find", "page", "deal", "side")} - {None}
            cached = self.replay.get(mark)
            if name in readers and cached and cached[0] == self.epoch:
                out.append((cid, f"Skipped (no step charged): this identical call at move {cached[1]} gave the following, unchanged since:\n" + cached[2][:3000]))
                continue
            if target and name in readers and name != self.jobs.get("side") and self.already_shown(name, str(args.get(target))):
                out.append((cid, f"Skipped (no step charged): {args[target]} already showed everything this view offers since the latest change. Its record: " + self.card_text(str(args[target]))))
                continue
            if many and isinstance(args.get(many), list) and args[many]:
                keep = [i for i in args[many] if not self.already_shown(name, str(i))]
                if not keep:
                    out.append((cid, "Skipped (no step charged): these articles already showed everything this view offers since the latest change. Their records:\n" + "\n".join(f"{i}: {self.card_text(str(i))}" for i in args[many])[:4000]))
                    continue
                if len(keep) < len(args[many]):
                    trimmed += f"\n[Wren] Left out {', '.join(str(i) for i in args[many] if i not in keep)}: shown in full already, nothing changed since."
                    args = {**args, many: keep}
                    mark = name + _pack(args)
            if closing and self.room() > 2:
                if self.is_checkout(name, args):
                    wider = self.wider_search()
                    if wider:
                        out.append((cid, wider + "\nTrolley now: " + self.basket_text()))
                        continue
                    late = self.late_question()
                    if late:
                        out.append((cid, "Held back (no step charged). " + late + "\nTrolley now: " + self.basket_text()))
                        continue
                    refreshed = self.refresh_deals()
                    if self.flagged_sold:
                        self.spot_sold()
                    if self.basket_moved and self.basket_seen < self.basket_at and self.jobs.get("basket_view") and self.room() > 3:
                        faults = []
                        view_args = _conform({}, self.specs.get(self.jobs["basket_view"]), "args", faults)
                        view = self.do_step(self.jobs["basket_view"], view_args) if not faults else None
                        if view is not None:
                            self.basket_snap = view.get("observation")
                            self.basket_seen = self.at
                            change = self.digest(self.jobs["basket_view"], self.basket_snap)
                            if change:
                                self.pending_notes = []
                                out.append((cid, "Held back (no step charged). Viewing the trolley revealed a change: " + change + "\nTrolley now: " + self.basket_text() + "\nFold the change in, mend the trolley where needed, then pay."))
                                continue
                    swap = self.swap_note()
                    stuck = [i for i in self.basket_ids() if i in self.vanished] if self.stalls < 2 else []
                    if swap or stuck:
                        self.stalls += 0 if swap else 1
                        out.append((cid, "Held back (no step charged). " + (swap or f"{', '.join(stuck)} in the trolley is sold out: exchange that line first.") + "\nTrolley now: " + self.basket_text()))
                        continue
                    chain, hold = self.follow_chain()
                    if hold:
                        out.append((cid, "Held back (no step charged). " + chain + "\nTrolley now: " + self.basket_text()))
                        continue
                    misfit = self.pair_faults() if self.pair_blocks < 2 and self.room() > self.keep_back() + 3 else []
                    if misfit:
                        self.pair_blocks += 1
                        self.log.append({"kind": "pair_hold", "faults": misfit[:4]})
                        out.append((cid, "Held back (no step charged). An add-on in the trolley does not suit the host in the trolley: " + "; ".join(misfit[:3]) + ". Exchange the add-on (or the host), then pay.\nTrolley now: " + self.basket_text()))
                        continue
                    faults = self.basket_faults() if self.line_blocks < 2 and self.room() > self.keep_back() + 2 else []
                    mended = []
                    if faults and all(" is held at " in f for f in faults):
                        mended = self.mend_counts(self.count_gaps())
                        faults = self.basket_faults() if mended else faults
                        if mended and not faults:
                            self.line_blocks += 1
                            out.append((cid, "Held back (no step charged). Wren corrected the trolley counts itself: " + "; ".join(mended) + ". Should the site have refused any, mend those lines; then pay.\nTrolley now: " + self.basket_text()))
                            continue
                    if faults:
                        self.line_blocks += 1
                        self.log.append({"kind": "line_hold", "faults": faults[:6]})
                        out.append((cid, "Held back (no step charged). The trolley still disagrees with the client's note: " + "; ".join(faults[:6]) + ". Mend those lines (using the trolley tool's own manner of adding or removing units), then pay.\nTrolley now: " + self.basket_text()))
                        continue
                    self.list_below()
                    read = self.open_specs(self.basket_ids(), self.keep_back() + 3) + self.peek_cheaper()
                    ruling = self.judge_basket()
                    if ruling:
                        seen = "Wren opened these articles' fields for the verdict (a move apiece):\n" + "\n".join(read) + "\n" if read else ""
                        out.append((cid, "Held back. " + chain + seen + ruling + "\nTrolley now: " + self.basket_text()))
                        continue
                    if refreshed and self.spent > COST_SOFT:
                        out.append((cid, "Held back (no step charged). Wren just refreshed the trolley's articles:\n" + refreshed + "\nPay again once each line still carries stock plus a price within the cap; otherwise exchange that line first."))
                        continue
                elif self.room() > 5:
                    trail = self.refuse_trail()
                    if trail:
                        out.append((cid, "Held back (no step charged). " + trail + "\nTrolley now: " + self.basket_text()))
                        continue
                    verdict = self.review()
                    if verdict:
                        out.append((cid, "Held back (no step charged). " + verdict))
                        continue
                    if self.leads and self.room() > 4 and self.exit_blocks == 0:
                        self.exit_blocks = 1
                        self.log.append({"kind": "exit_blocked", "leads": list(self.leads)})
                        out.append((cid, "Held back (no step charged). The review turned up articles that seem to pass: " + ", ".join(self.leads) + ". Open them where needed, trolley the best passing one (honouring the selection rule) and pay. Bow out only where each of them plainly breaks a demand; state which demand in your prose, then resend the bow-out call."))
                        continue
                    if self.leads and self.room() > 4 and self.exit_blocks == 1:
                        self.exit_blocks = 2
                        first = self.leads[0]
                        self.judge_basket([first], COST_HARD)
                        if self.rulings.get(self.ruling_key([first])) and first in self.leads:
                            self.vouched = first
                            self.log.append({"kind": "exit_blocked", "vouched": first})
                            out.append((cid, f"Held back (no step charged). An independent check of {first}'s records finds it passing every demand. Trolley it at the wanted count and pay; bow out only where its record plainly breaks a demand."))
                            continue
            if self.is_exit(name, args) and self.room() <= 5 and not self.late_block:
                lines = [i for i in self.built if not self.gone_now(i)]
                fresh = self.basket_snap is not None and self.basket_seen >= self.basket_at
                missing = [i for i in lines if i not in self.basket_ids() or (i in self.removed and not fresh)]
                if len(lines) >= len(self.wants) and len(missing) + 1 <= self.room():
                    self.late_block = True
                    self.log.append({"kind": "late_exit_held", "lines": lines, "missing": missing})
                    redo = f"Return {', '.join(missing)} to the trolley at the wanted counts, then pay." if missing else "Pay for the trolley as it stands."
                    out.append((cid, "Held back (no step charged). Bowing out earns nothing while the site holds a passing article, and scarce moves are no proof that it holds none. " + redo + " Bow out only where your searching proved that some want has no article meeting its demands; then resend the bow-out call."))
                    continue
            if self.is_exit(name, args) and not self.leads and not self.nudged_once and self.room() > self.step_cap // 3:
                self.nudged_once = True
                out.append((cid, f"Held back (no step charged). {self.room()} moves remain, so search in untried ways before finishing: browse each want's department and page through it with the cursor or offset the site returns, try single-word value queries plus the property's other names, and open whichever articles might pass. Bow out only afterwards."))
                continue
            if self.is_exit(name, args):
                done = self.finish_basket()
                if done:
                    out.append((cid, done + (" [episode finished]" if self.over else "")))
                    continue
                args = self.exit_words(name, args)
            if closing and self.room() >= 2:
                go, said = self.chore_gate("order" if self.is_checkout(name, args) else "end")
                if not go:
                    out.append((cid, "Held back. " + said + "\nTrolley now: " + self.basket_text()))
                    continue
                if said:
                    trimmed = "\n[Wren] " + said + trimmed
            reply = self.do_step(name, args)
            if reply is None:
                out.append((cid, "That call did not go through (network or server trouble); feel free to resend it."))
                continue
            observation = reply.get("observation")
            if many and isinstance(args.get(many), list) and len(args[many]) > 1:
                shown = _pack(observation)
                if not any(str(i) in shown for i in args[many]):
                    self.side_bad = min(self.side_bad or 999, len(args[many]))
                    trimmed += f"\n[Wren] The side-by-side view showed nothing for {len(args[many])} products: send at most {self.side_cap()} per call."
                else:
                    self.side_ok = max(self.side_ok, len(args[many]))
            self.just_did = (self.just_did + [mark])[-3:]
            self.id_slots.update((name, k) for k, v in args.items() if (v if isinstance(v, list) else [v]) and all(isinstance(x, str) and x in self.cards for x in (v if isinstance(v, list) else [v])))
            self.ingest(observation)
            if name == self.jobs.get("side"):
                rows = self.cards_in(observation)
                if rows:
                    self.specs_by[name] = self.specs_by.get(name, False) or any(_spec_block(r) for _, r in rows)
            if name in readers - {self.jobs.get("find")}:
                self.mark_view(name, observation)
            known = {tool for job, tool in self.jobs.items() if job != "find"}
            paged = self.auto_page(name, args, observation) if name not in known and not closing else ""
            if self.id_slot(name) and name not in (self.jobs.get("basket"), self.jobs.get("checkout"), self.jobs.get("walk_away"), self.jobs.get("talk")):
                product = self.id_of(observation) if isinstance(observation, dict) else None
                if product:
                    self.specs_by[name] = self.specs_by.get(name, False) or bool(_spec_block(observation))
                    if name == self.jobs.get("page") and not self.specs_by[name]:
                        other = next((t for t in self.specs if t != name and self.id_slot(t) and self.specs_by.get(t) is not False and t not in (self.jobs.get("basket"), self.jobs.get("checkout"), self.jobs.get("walk_away"), self.jobs.get("talk"), self.jobs.get("find"))), None)
                        if other:
                            self.jobs["page"] = other
                    if not self.specs_by[name] and name != self.jobs.get("page") and not self.jobs.get("deal"):
                        self.jobs["deal"] = name
                    if name == self.deal_view():
                        self.deal_seen[product] = self.at
            if name == self.jobs.get("basket"):
                self.track_basket(name, args)
                self.basket_reply = _pack(observation)
                self.just_did = []
                counts = [_whole(v) for v in args.values() if not isinstance(v, str) and _whole(v) is not None]
                for value in args.values():
                    if counts and isinstance(value, str) and value in self.cards:
                        if counts[0] <= 0 and not _row_holds(observation, value):
                            self.removed.add(value)
                        elif counts[0] > 0:
                            self.removed.discard(value)
                lines = [i for i in self.basket_ids() if not self.gone_now(i) and i not in self.removed]
                if len(lines) >= len(self.built):
                    self.built = lines
            if name == self.jobs.get("find") and self.cards_in(observation):
                self.queries = (self.queries + [dict(args)])[-8:]
            if name == self.jobs.get("basket_view"):
                self.basket_snap = observation
                self.basket_seen = self.at
            if self.is_checkout(name, args):
                self.checkouts += 1
                if not reply.get("finished"):
                    self.bounced_at = self.at
            text = self.render(observation) + paged + trimmed
            if name not in known and not closing:
                text += self.cheap_rows([observation] + self.paged_seen) + self.cap_check(name, args, observation)
            base = text
            proof = self.proof_lines(observation)
            if proof:
                text += "\n" + proof
            if name == self.jobs.get("basket") and any(_whole(v) == 0 for v in args.values() if not isinstance(v, str)):
                if any(isinstance(v, str) and v in self.cards and _row_holds(observation, v) for v in args.values()):
                    text += "\n[Wren] Count 0 kept the line trolleyed: this tool adds its count onto the line, hence a negative count removes units."
            self.flagged_sold = False
            lines = self.free_text(observation)
            if lines:
                meaning = self.read_notice(lines)
                if meaning:
                    text += "\n[notice read] " + meaning + ". Fold this change into your picks ahead of paying."
            bounced = self.is_checkout(name, args) and not reply.get("finished")
            again = bounced and any(line in self.voice(observation) for line in self.soldout_lines)
            loose = self.flagged_sold and (bounced or (not self.flagged_named and not self.is_checkout(name, args)))
            if (loose or again) and self.basket_ids() and not self.over:
                seen = self.spot_sold()
                if seen:
                    text += "\n[Wren refreshed the trolley lines following the refusal]\n" + seen
            self.flagged_sold = False
            swap = self.swap_note()
            if swap:
                text += "\n" + swap
            if self.pending_notes:
                text += "\n[notices seen on Wren's own calls] " + " | ".join(self.pending_notes)
                self.pending_notes = []
            if name in readers:
                self.replay[mark] = (self.epoch, self.at, base)
            if reply.get("finished"):
                text += " [episode finished]"
            out.append((cid, text))
        return out

    # ---------- closing the episode ----------
    def wrap_up(self, again: bool = False) -> str:
        """Force a finish the walker never made: pay for a held trolley, otherwise bow out."""
        if self.over or self.env_dead or self.room() <= 0:
            return ""
        buy, quit_ = self.jobs.get("checkout"), self.jobs.get("walk_away")
        fresh = self.basket_snap is not None and self.basket_seen >= self.basket_at
        holding = bool(self.basket_ids()) if fresh else bool(self.basket)
        whole = True
        if holding and len(self.wants) >= 2 and self.room() >= 2:
            whole = self.fill_short_cart() or not quit_
            if not whole:
                self.log.append({"kind": "short_cart_refused", "lines": self.basket_ids()})
        if self.over or self.room() <= 0:
            return ""
        if whole and buy and (holding or (self.basket_moved and not fresh and self.room() >= 2)):
            faults: list[str] = []
            args = _conform(dict(self.intake.get("checkout_args") or {}), self.specs.get(buy), "args", faults)
            if not faults:
                if self.room() >= 2:
                    self.chore_gate("order")
                if self.over or self.room() <= 0:
                    return ""
                reply = self.do_step(buy, args)
                self.log.append({"kind": "forced_checkout", "reply": _trim(reply, 200)})
                if self.over or self.room() <= 0:
                    return ""
                if reply is not None and again:
                    self.checkouts += 1
                    self.bounced_at = self.at
                    self.digest(buy, reply.get("observation"))
                    return "the payment was turned down: " + self.render(reply.get("observation"))
        if quit_:
            faults = []
            args = _conform(self.exit_words(quit_, dict(self.intake.get("walk_away_args") or {})), self.specs.get(quit_), "args", faults)
            if not faults:
                if self.room() >= 2:
                    self.chore_gate("end")
                if self.over or self.room() <= 0:
                    return ""
                reply = self.do_step(quit_, args)
                self.log.append({"kind": "forced_exit", "reply": _trim(reply, 200)})
                if reply is not None and not self.over:
                    self.digest(quit_, reply.get("observation"))
                    return "the bow-out was turned down: " + self.render(reply.get("observation"))
        return ""

    def shrink(self) -> None:
        size = sum(len(_pack(m.get("content"))) for m in self.thread)
        if size <= LOG_CAP:
            return
        tools = [i for i, m in enumerate(self.thread) if m.get("role") == "tool"]
        for index in tools[:max(1, len(tools) // 2)]:
            content = self.thread[index].get("content") or ""
            if len(content) > 600:
                ids = re.findall(r"\"([0-9A-Za-z][0-9A-Za-z_.:\-]{5,40})\"", content)
                ids = [i for i in dict.fromkeys(ids) if i in self.cards][:20]
                self.thread[index]["content"] = content[:400] + f"...(older result shortened; products in it: {', '.join(ids)})"

    def play(self) -> list[dict[str, Any]]:
        if self.over:
            return self.log
        self.intake_pass()
        self.log.append({"kind": "intake", "intake": self.intake, "wants": [{"what": w.what, "label": w.label, "count": w.count, "needs": w.needs, "cap": w.cap} for w in self.wants],
                         "pairs": self.pairs, "conditional": self.conditionals, "fallback": self.fallback, "hints": self.hints})
        opening = _unpack_records(self.scene.get("opening"))
        if opening is not None:
            self.ingest(opening)
        intro = ("CLIENT'S NOTE (chatter about the client's life set aside; Wren's sheet below is the binding reading of it):\n" + self.note_core()) if self.sifted else "CLIENT'S NOTE:\n" + self.ask_text
        if opening is not None:
            intro += "\n\nTHE SITE'S OPENING SCREEN:\n" + self.render(opening)
        intro += "\n\nOpen by searching every want within its department and opening a few articles in full, giving the interviewer a view of the properties such articles carry.\n\n" + self.stamp()
        self.thread = [{"role": "system", "content": CHARTER}, {"role": "user", "content": intro}]
        self.loop(TURN_CAP, COST_HARD)
        if not self.over:
            turned = self.wrap_up(again=True)
            if turned and not self.over and not self.env_dead and not self.dry and self.room() > 1 and not self.late():
                self.thread.append({"role": "user", "content": "[Wren] Wren attempted a finish and " + turned + "\nFold in what the site says, then finish.\n" + self.stamp()})
                self.loop(self.turns + 10, COST_HARD + 0.01)
            if not self.over:
                self.wrap_up()
        self.log.append({"kind": "end", "finished": self.over, "steps": self.at, "cost": round(self.spent, 5), "calls": self.calls})
        return self.log

    def loop(self, turn_cap: int, cost_cap: float) -> None:
        idle = 0
        while not self.over and not self.env_dead and self.room() > 0 and self.turns < turn_cap and self.spent < cost_cap and not self.late():
            if self.time_to_ask():
                summary = self.interview()
                if summary:
                    self.thread.append({"role": "user", "content": summary + "\n" + self.stamp()})
                continue
            self.shrink()
            self.turns += 1
            message = self.consult(self.thread, [SMALL, BACKUP], 3000, self.menu())
            if message is None:
                self.blanks += 1
                if self.dry or self.calls >= CALL_CAP or self.blanks > 2:
                    break
                continue
            calls = []
            for number, call in enumerate(message.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") if isinstance(call.get("function"), dict) else {}
                raw = fn.get("arguments")
                parsed = raw if isinstance(raw, dict) else _json_of(raw) if isinstance(raw, str) and raw.strip() else {}
                calls.append({"id": str(call.get("id") or f"w{self.turns}_{number}"), "type": "function", "function": {"name": str(fn.get("name") or ""), "arguments": _pack(parsed if isinstance(parsed, dict) else {})}})
            content = message.get("content")
            if not isinstance(content, str):
                content = _pack(content) if content else ""
            if not calls:
                idle += 1
                self.thread.append({"role": "assistant", "content": content or "(no call)"})
                self.thread.append({"role": "user", "content": "Go on by calling a tool.\n" + self.stamp()})
                if idle >= 3:
                    break
                continue
            idle = 0
            committing = {self.jobs.get("basket"), self.jobs.get("checkout"), self.jobs.get("walk_away")} - {None}
            if not self.interviewed and self.jobs.get("talk") and self.talk_room() > 0 and any(c["function"]["name"] in committing for c in calls) and self.room() > 8:
                self.thread.append({"role": "assistant", "content": content or None, "tool_calls": calls})
                for call in calls:
                    self.thread.append({"role": "tool", "tool_call_id": call["id"], "content": "Skipped (no step charged): the client gets asked first."})
                summary = self.interview()
                self.thread.append({"role": "user", "content": (summary or "The interviewer found nothing worth asking.") + "\n" + self.stamp()})
                continue
            self.thread.append({"role": "assistant", "content": content or None, "tool_calls": calls})
            self.log.append({"kind": "turn", "text": content[:300], "calls": [c["function"] for c in calls]})
            replies = self.execute(calls)
            for index, (cid, text) in enumerate(replies):
                if index == len(replies) - 1:
                    text += "\n" + self.stamp()
                self.thread.append({"role": "tool", "tool_call_id": cid, "content": text})


def agent_main(problem_data: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        episode = Wren(problem_data)
    except Exception as exc:
        return [{"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]}]
    try:
        return episode.play()
    except Exception as exc:
        episode.log.append({"kind": "crash", "error": f"{type(exc).__name__}: {exc}"[:400]})
        try:
            if not episode.over:
                episode.wrap_up()
        except Exception as again:
            episode.log.append({"kind": "crash", "error": f"{type(again).__name__}: {again}"[:400]})
        return episode.log


__all__ = ["agent_main"]
