import json
import re
import threading
from difflib import SequenceMatcher as Similarity
from os import getenv as env_value
from time import monotonic as clock
from time import sleep as pause
import src.agent.proxy_client as proxy_module

PROVIDER_MODELS = dict(
    chutes=("zai-org/GLM-5.2-TEE", "moonshotai/Kimi-K2.6-TEE"), openrouter=("z-ai/glm-5.2", "moonshotai/kimi-k2.6")
)
USD_CAP = 0.05
GUESS_READER_FIRST = dict(openrouter="openai/gpt-6-luna")   # read first, before the two answer models
GUESS_WAIT = 25.0            # seconds the opening readings may take together
ASK_ROUNDS = 9               # attribute questions allowed, one attribute each (plus one per extra expected detail)
ASK_ROUNDS_AFTER_PAIR = 4
OVER_LIMIT_SPAN = 1.35      # a fitting listing at most this far over the limit is named when abstaining
PURPOSES_ASKED = 2           # purposes of the request asked about by name
AFTER_PAIR_ATTRS = 1         # attribute questions once a side is picked and no purpose brought a detail out
REFUSED_AS_MANY = re.compile(r"one (?:by one|at a time)|in sequence|separately|one thing|point by point", re.I)
AFTER_FIND = 2               # attribute questions after the first told detail, once the purposes are used up
AFTER_FIND_TWO = 6           # the same when the request itself gives two purposes
ASIDE_CALLS = 3              # readings that sort attribute names into the item's own and the rest
SALE_TALK = re.compile(r"warrant|\bcondition|\bmodel\b|\bseries\b|\bpack|travel|recommended|\bage\b|gender|skin type|hair type|concern|deliver|shipping|polic")
SHELF_READS = 3              # readings of the shelf's fact labels that choose what to ask about
ADDED_TALK = re.compile(r"\b(?:also|as well|in addition|one more thing)\b", re.I)
SWAP_TALK = re.compile(r"\b(?:instead|forget|no longer|rather than|replac\w*|swap\w*|switch\w*)\b", re.I)
MORE_AFTER_FIND = 6          # questions in all after the first told detail
ASK_LIMIT = 10               # questions put to the shopper in an episode
ASK_TURN_FLOOR = 10          # turns kept free of questions for the cart, the changes and the order
DECOY_TRIES = 14             # turns that look for a listing to cart first, as the request was written
DECOY_PLAIN_FROM = 4         # from this try a plain statement of the request is enough
DECOY_FIT_FROM = 6           # from this try any fitting listing opens the cart
DECOY_TURN_FLOOR = 14
BEST_ROUNDS = 3              # search rounds for fitting listings that stand better by what decides
CHANGE_WORDS = re.compile(
    r"\b(?:instead of|change of plan|change (?:my|the|it|that|to)|changing|changed|switch(?:ing)? (?:to|it|the)"
    r"|swap\w*|forget|no longer|rather than|replace (?:the|it|that|this)|everything else|the rest stays|keep the rest)\b",
    re.I,
)
OPEN_REACH = 1.6             # rows this far over the limit are still read (below in-budget rows that look as good)
RANKING_TALK = re.compile(r"\b(?:most|highest|largest|biggest|greatest|longest|lowest|least|fewest|cheapest|priority|budget|price)\b", re.I)
CALLS_MAX = 16
IDS_PER_COMPARE = 5
READ_CAP = 250
SEARCH_CAP = 80
CARDS_PER_JUDGE = 18
CARDS_CAP = 120
WALL_SECONDS = 180.0
RESERVED_TURNS = 6
SEARCH_TEXT_MAX = 200
THINK_EXTRA = 3000
TOKENS_FLOOR = 900
UNIT_CHOICES = 3
LOSS_ROOM = 24
BRAND_FILTER_CAP = 6
LABELS_SHOWN = 30
STEP_SHARES = (0.7, 0.45, 0.25, 0.12, 0.85, 0.55, 0.35, 0.18, 0.08, 0.04)
NO_MAKER = {"none", "no brand", "oem"}
PURCHASE_TERMS = re.compile("categor|budget|price|\\bstock|availab")
FIRST_PICKS = re.compile("colou?r|material|fabric|warrant")
GROUP_NAMES = re.compile("\\b(?:features|specifications|specs|details)$")
PLEASE_REPLY = re.compile(
    "\\b(?:let me know|please confirm|can you confirm|could you confirm|should i|would you like|do you want)\\b", re.I
)
_client = proxy_module.ProxyClient(timeout=90, max_retries=2)


def flatten_lower(txt: object) -> str:
    return " ".join(str(txt or "").lower().split())


def lexemes(txt: object) -> list[str]:
    return re.findall("[a-z0-9]+(?:\\.[0-9]+)?", flatten_lower(txt))


def root_word(tok: str) -> str:
    if len(tok) > 4:
        if tok.endswith("ies"):
            return tok[:-3] + "y"
        if tok.endswith(("ches", "shes", "sses", "xes")):
            return tok[:-2]
    if len(tok) > 3 and tok[-1] == "s" and (tok[-2:] != "ss"):
        tok = tok[:-1]
    if len(tok) > 4 and tok[-2:] == "ie":
        tok = tok[:-2] + "y"
    return tok


def unit_root(fig: str, unit_txt: str) -> str:
    return re.sub("^[\\d.]+", "", root_word(fig + unit_txt))


def tight_units(txt: object) -> str:
    return re.sub("(\\d)\\s+(?=[a-z]{1,4}\\b)", "\\1", flatten_lower(txt))


def same_root(turn_one: str, second: str) -> bool:
    if turn_one == second:
        return True
    shorter, longer = (turn_one, second) if len(turn_one) <= len(second) else (second, turn_one)
    return (
        len(shorter) >= 4
        and (not shorter[0].isdigit())
        and longer.startswith(shorter)
        and (len(longer) - len(shorter) <= 2)
    )


def root_set(txt: object) -> set[str]:
    return {root_word(wd) for wd in lexemes(tight_units(txt))}


def shelf_key(txt: object) -> str:
    return " ".join((root_word(wd) for wd in lexemes(str(txt).replace("&", " and ")) if wd != "and"))


def positive_amount(val: object) -> float | None:
    try:
        amount = float(val)
    except (TypeError, ValueError, OverflowError):
        return None
    if not amount > 0 or amount == float("inf"):
        return None
    return amount


def fn_schema(ident: str, about: str, fields: dict, level_required: list[str]) -> dict:
    schema = {"type": "object", "properties": fields, "required": level_required}
    return {"type": "function", "function": {"name": ident, "description": about, "parameters": schema}}


def str_array(about: str) -> dict:
    return {"type": "array", "items": dict(type="string"), "description": about}


def env_act(ident: str, **args: object) -> dict:
    return {"name": ident, "args": {ref: val for ref, val in args.items() if val is not None}}


def as_list(val: object, split: str = "[\\n;,]+") -> list:
    if isinstance(val, str):
        return re.split(split, val)
    return val if isinstance(val, list) else []


NEED_FIELDS = {
    "attribute": {
        "type": "string",
        "description": "What the detail is called, in the shopper's own words (colour, material, capacity, style ...).",
    },
    "value": {
        "type": "string",
        "description": "The detail as the shopper gives it, unit included, as briefly as possible.",
    },
    "spellings": {
        "type": "array",
        "items": dict(type="string"),
        "maxItems": 6,
        "description": "This exact value written differently: another spelling, or the same quantity in another unit. Empty when none.",
    },
    "synonyms": {
        "type": "array",
        "items": dict(type="string"),
        "maxItems": 6,
        "description": "Words sellers put for this same value; never a part of it or a broader class. Empty when none.",
    },
}
NEED_ITEM = {"type": "object", "properties": NEED_FIELDS, "required": ["attribute", "value"]}
BRIEF_TOOL = fn_schema(
    "brief",
    "The shopper's request turned into a shopping brief.",
    {
        "kind": {
            "type": "string",
            "description": "The request's own words for the kind of thing wanted, verbatim, minus the article and the details.",
        },
        "shelf": {
            "type": "string",
            "description": "A store category - only if the shopper calls something a category. Otherwise empty.",
        },
        "needs": {
            "type": "array",
            "items": NEED_ITEM,
            "description": "Every detail of the item itself that the request insists on.",
        },
        "max_price": {"type": "number", "description": "The spending limit; 0 when the request sets none."},
        "maker": {"type": "string", "description": "A brand the shopper insists on; empty otherwise."},
        "maker_wish": {"type": "string", "description": "A brand the shopper would like if one is to be had, without insisting; empty otherwise."},
        "either_or": {
            "type": "array",
            "items": dict(type="string"),
            "maxItems": 2,
            "description": "If the shopper wishes for two things and doubts one listing has both: the two wishes, each a value with what it describes ('oak wood material', 'the ACME brand'). Otherwise empty.",
        },
        "uses": {
            "type": "array",
            "items": dict(type="string"),
            "maxItems": 3,
            "description": "Each use or recipient the request gives for the item, one per entry, in the shopper's own words and written so that it can follow 'for' (with its article where the request has one). Two uses given in one sentence are two entries; one use is never split in two. Empty if none.",
        },
        "look_at": {
            "type": "integer",
            "description": "When the request tells the assistant to open, compare or check at least some number of fitting options before choosing: that number; otherwise 0.",
        },
        "rank_searches": str_array(
            "When something decides between qualifying items: up to four more searches aimed at the listings that stand best by it; otherwise empty."
        ),
        "decides": {
            "type": "string",
            "description": "What the shopper says should pick between items that all qualify (the cheapest, the longest ...). Empty if nothing.",
        },
        "searches": str_array(
            "Six store searches of two to five words likely to show the item: its words with the insisted values, adding a synonym or another spelling of a value where one exists."
        ),
    },
    ["kind", "needs", "max_price", "either_or", "uses", "searches"],
)
LISTEN_TOOL = fn_schema(
    "listen",
    "What the shopper's newest line changes.",
    {
        "new_needs": {
            "type": "array",
            "items": NEED_ITEM,
            "description": "Each detail of the item that the line insists on and that was not known yet.",
        },
        "withdrawn": str_array("Attributes whose earlier detail the line takes back or swaps for another."),
        "max_price": {"type": "number", "description": "A new spending limit the line sets; 0 when it sets none."},
        "shelf": {
            "type": "string",
            "description": "A store category the line names for the item, copied as written; empty if none.",
        },
        "decides": {
            "type": "string",
            "description": "What the line says should pick between qualifying items; empty if nothing.",
        },
        "expects_reply": {
            "type": "boolean",
            "description": "True when the line asks the assistant something or waits for an answer.",
        },
        "market_reaction": {
            "type": "boolean",
            "description": "True when the line only reacts to the chosen item's stock or price having moved.",
        },
        "searches": str_array(
            "Up to six short store searches using what the line added (another spelling or a synonym of the value too); empty when nothing was added."
        ),
    },
    ["new_needs", "withdrawn", "max_price", "expects_reply", "searches"],
)
CHECK_TOOL = fn_schema(
    "check",
    "Which listings carry every requirement.",
    {
        "verdicts": str_array(
            "One entry per listing, in the order shown: its number, K when it is the kind of thing wanted or X when not, then per requirement in order Y when the listing's own words give that value or N when they do not. Two requirements, listing 3: '3 K YN'."
        ),
        "evidence": {
            "type": "array",
            "description": "Only for listings that are K with every requirement Y: where each value is given.",
            "items": {
                "type": "object",
                "properties": {
                    "index": {"type": "integer", "description": "The listing's number."},
                    "places": str_array(
                        "Per requirement in order: title, options, or the fact label exactly as shown."
                    ),
                    "quotes": str_array("Per requirement in order: the listing's exact words giving the value."),
                    "figure": {
                        "type": "number",
                        "description": "The listing's figure for the shopper's deciding point when there is one, else 0.",
                    },
                },
                "required": ["index", "places", "quotes"],
            },
        },
        "searches": str_array(
            "When no listing carries every requirement: up to five other short searches worth trying. Otherwise empty."
        ),
    },
    ["verdicts"],
)
NAMING_TOOL = fn_schema(
    "naming",
    "What each text says about one attribute.",
    {
        "marks": {
            "type": "array",
            "items": dict(type="string"),
            "description": "One per text, in order: '<number> <only|among|other|none>'.",
        }
    },
    ["marks"],
)
SPELLINGS_TOOL = fn_schema(
    "spellings",
    "Other words sellers write for one value.",
    {
        "words": {
            "type": "array",
            "items": dict(type="string"),
            "description": "At most six, the commonest first; empty if there are none.",
        }
    },
    ["words"],
)
FIGURES_TOOL = fn_schema(
    "figures",
    "Each listing's figure for what decides between qualifying items.",
    {
        "per_listing": {
            "type": "array",
            "items": dict(type="string"),
            "description": "One per listing, in order: '<number> <figure>' as a bare number in one unit for all listings, or '<number> none' when its words give none.",
        },
        "prefer": {
            "type": "string",
            "enum": ["higher", "lower"],
            "description": "Whether the bigger or the smaller figure wins.",
        },
        "unit": {
            "type": "string",
            "description": "The one unit every figure is written in, as listings write it (ml, cm, pcs, w ...); empty for a bare count.",
        },
    },
    ["per_listing", "prefer"],
)
UNITS_TOOL = fn_schema(
    "unit_names",
    "The attribute each unit measures for this kind of item.",
    {
        "pairs": str_array(
            "One per unit, in order: '<unit> = <attribute>' with the attribute as a shopper says it (one to three words), or '<unit> = none'."
        )
    },
    ["pairs"],
)
SEARCHES_TOOL = fn_schema(
    "new_searches",
    "Fresh store searches.",
    {"searches": str_array("Six searches of two to five words, none of them among those already run.")},
    ["searches"],
)
PROPS_TOOL = fn_schema(
    "properties",
    "Attributes of the item worth asking the shopper about, likeliest first.",
    {
        "names": str_array(
            "Up to eight attribute names of one to four words, each in the words of its fact label, likeliest first."
        )
    },
    ["names"],
)
ASIDE_TOOL = fn_schema(
    "label_marks",
    "Whether each attribute name tells about the item itself.",
    {"marks": str_array("One per name, in order: '<number> item' or '<number> other'.")},
    ["marks"],
)
GUESS_TOOL = fn_schema(
    "likely_attributes",
    "Where the shopper's unstated requirement most likely lies.",
    {
        "attributes": {
            "type": "array",
            "items": dict(type="string"),
            "maxItems": 4,
            "description": "Three or four attributes of the item, likeliest first, each named the way listings label it (one to three plain words, no value).",
        },
        "reasons": {
            "type": "array",
            "items": dict(type="string"),
            "maxItems": 3,
            "description": "The separate purposes the request gives for buying, one entry per purpose. What belongs to one purpose stays in its entry: who it is for, what that person said or wants, why it matters. Only a different purpose (another use, occasion or person) is a new entry. Empty if none.",
        },
    },
    ["attributes", "reasons"],
)
ANSWER_TOOL = fn_schema(
    "answer",
    "A short reply to the shopper.",
    {
        "text": {
            "type": "string",
            "description": "One or two plain statements that respond to the shopper's line from the facts given. Never a question.",
        }
    },
    ["text"],
)
BRIEF_RULES = """You turn a shopper's request into a brief for the assistant who will shop for it. Fill the brief tool from the request's words and nothing else.
- kind: the request's words for the kind of thing, complete and unchanged - keep a leading word like 'other' and keep '&' or '/' (the store's categories carry such names); never swap in a broader or narrower kind.
- needs: each detail of the item that the request insists on - material, colour, an amount with its unit (capacity, size, weight, length, number of pieces, power), pattern, shape, style, type, an occasion or use joined to the item, the model or thing it must fit, and so on. Almost every request carries at least one; read it again before leaving needs empty. Call each attribute what the shopper calls it, or by the plain name of that kind of detail when only the value is given. Keep the value short - the word or amount alone, units spelled out, without the attribute's name or the item. One phrase gives one need; never split it. The budget, the store category and availability are not needs, and neither is the reason for buying, which stands in a sentence of its own.
- maker_wish: a brand the shopper would like if one is to be had ('I'd prefer X', 'if there is a suitable one from X'), named without insisting; maker stays empty then.
- either_or: when the shopper hopes for two things and suspects no single listing has both, in whatever words, give the two wishes; the shopper will pick one when asked. Leave it empty otherwise.
- uses: every job the item will do, every person it is for, and whatever the purchase has to match, follow or be picked from (a list someone wrote down, things already owned, a setup already planned); each may hide a need the shopper has not said yet. Two uses joined by 'and', 'also' or 'more specifically' are two entries; what belongs to one use (who it is for and what that person said) stays in its entry.
- spellings: the same value in another form (grey: gray; 1 kg: 1000g, 1 kilo). synonyms: words sellers use for the same value (navy: dark blue; sofa: couch), never a part of it or a broader class.
- searches: what a shopper would type, two to five words, from the kind and the insisted values; write amounts as listings do, figure and unit joined."""
LISTEN_RULES = """You follow the conversation between a shopper and a shopping assistant. From what is known so far and the shopper's newest line, fill the listen tool with what that line changes.
- new_needs: a detail of the item the line insists on (an attribute with its value). Name the attribute as the shopper does and keep the value to the word or amount alone. A purpose by itself is not an attribute; the attribute it calls for is. The budget, the store category and availability are not attributes. spellings: the same value written differently (another spelling; the same amount in another unit: 1 kg is 1000g or 1 kilo). synonyms: words sellers use for the same value (navy: dark blue; sofa: couch), never a part of it or a broader class.
- withdrawn: each attribute whose earlier detail the line takes back, or replaces with another value.
- The line may hold several sentences or lines; read them in order, and for one attribute the later statement wins.
- Putting one value in place of another, taking one back or adding one changes what the shopper wants even when it is phrased as a question or a check ('..., right?'): withdraw the old attribute and add the new value.
- max_price: only when the line sets a new limit.
- shelf: the store category the line names for the item, copied as written; empty if none.
- expects_reply: true when the line asks the assistant something or waits for an answer; false for a remark that needs none.
- market_reaction: true when the line reacts to the chosen item's stock or price having just moved (the assistant answers by choosing again). If the same line also changes a detail, fill new_needs and withdrawn as well.
- searches: two to five words joining the item's words with the value just added."""
CHECK_RULES = """You compare store listings with a shopper's requirements. For each listing shown, say whether it is the kind of thing wanted and, requirement by requirement, whether the listing's own words give the value.
- A value is given when the listing's own words carry it: the same word, a plain synonym or another spelling, or the same amount in another unit.
- Where to look: a colour or a size is what the options say, since the options describe the exact variant sold - options naming another colour or amount leave the requirement unmet, whatever the title says. A material or an amount may be given by the title, the options or a fact. Any other attribute the store keeps as its own fact is given by the fact whose label names it.
- When a listing shows 'own fact for requirement N', that fact decides requirement N: met when it gives the value, unmet when it gives something else - unless the options give the value.
- A fact giving the value settles it even if the title hints otherwise. A value that is missing is unmet; never assume it.
- An amount counts when it is one of several measures written for the variant (40*30*5cm gives 5 cm; a range of 4-6cm includes 5 cm). An amount under a fact for a different quantity does not count.
- verdicts: one entry per listing in order, '<number> <K or X> <Y or N per requirement>'.
- evidence: only for listings that are K with every requirement Y - the place (title, options, or the fact label exactly as shown) and the listing's exact words for each value; figure is the listing's number for the shopper's deciding point when one is given.
Give a verdict for every listing, in order."""
NAMING_RULES = """You read what a listing's words say about one attribute of the item - its colour, or the material it is made of. Answer each numbered text with one word:
- only: the words name the wanted value (the same word, or another word sellers use for it) and no other value of that attribute.
- among: the words name the wanted value next to another value of the same attribute (two colours, two materials), so which one the item is stays open.
- other: the words name a value of that attribute, but not the wanted one.
A shade or tint of the wanted colour (light, dark, off-, or a describing word in front) is the wanted colour; a name made of two different colours is a different colour.
What the variant's own words name is what is actually sold: a material named there is the item's material, regardless of the title.
- none: the words name no value of that attribute.
Sizes, codes, counts and other details are not values of the attribute. Mark every text, in order."""
UNITS_RULES = """Listings of one kind of item write figures with the units below. For each unit, give the one attribute of the item that a figure in that unit measures, as a short noun a shopper would use (volume, capacity, weight, piece count, length, size, voltage, power, storage ...); the number of pieces sold together is its piece count. A unit of distance always measures its length, whichever part of the item the figure describes. Write none only when the unit measures nothing about the item itself (a code, a size letter, a resolution mark, a delivery or warranty period)."""
SPELLINGS_RULES = """A shopper wants one value of an attribute. List the other words or spellings sellers write in listing titles and variant names for exactly that value: synonyms and other spellings of a word, or the same amount in other units and notations. Nothing that is only a part, a shade or a broader class of it."""
FIGURES_RULES = """A shopper said what makes one qualifying item better than another. For each listing, find in its own words (options first, then facts, then title) the figure that this is about, and write it as a bare number in one unit shared by all listings, converting where needed (an amount of another kind - a weight where a volume decides - is no figure). Write none when the listing gives no such figure; never guess. Each entry is the listing's number, a space, then the figure or none."""
SEARCHES_RULES = """A shopping assistant has not yet found a listing that meets all of a shopper's requirements. From the request, the requirements, the searches already run and the titles already seen, write new short store searches likely to reach a listing that meets them all: combine the required values with each other and with words such listings carry in their titles, use other words sellers write for a value, and try the values without the item's words. Never repeat a search already run."""
PROPS_RULES = """A shopper asked a shopping assistant for an item. Beyond what the request says, the shopper holds back one more requirement for each purpose named, and tells it only when asked about that very attribute, one attribute per question. From what the store's listings of this kind show, choose the attributes to ask about, likeliest first.
- Colour first, unless the request or an earlier answer already settles it.
- Then the fact labels that tell what the item is, does, contains or looks like - its kind or type, key ingredient or ingredient preference, benefit, form or formulation, finish, style, pattern, scent, what it plugs into or fits - and its material. Prefer labels whose values differ between listings, the widest filled first.
- Then one measure the request leaves open and listings give in figures: every distance (a length, width, height, diameter, thickness, the size of a part) is named 'length'; otherwise weight, volume, capacity, voltage or piece count.
- Last, and only when little else is left: whom or what the item suits (a skin or hair type, an age group, a gender), a problem it treats, and the terms it is sold on (warranty, model code, condition, how it is packed, travel size).
- Leave out what the request already fixes, the quantity qualifying items are ranked by, the brand, price, stock and store category, and anything already asked about - also under another label that means the same thing.
- Name each attribute by its fact label's own words, written apart with spaces (never joined by underscores), complete and in the label's order ('finish lipstick', not 'finish'; a label 'type of x' reads 'x type'), singular, without a value. One attribute per name."""
ASIDE_RULES = """Listings of one kind of item are described by the attribute names below. Mark each name: item when it tells something about the item itself - what it is, what it does or contains, how it looks, what it measures or connects to; other when it tells whom or what the item suits (a skin or hair type, an age group, a gender), a problem it treats, or the terms it is sold on (warranty, model code, condition, delivery, how it is packed, travel size). One mark per name, in order."""
GUESS_RULES = """A shopper has asked an assistant for one item. Beyond what the request says, the shopper keeps one more requirement in mind for each reason they give for buying, and names it only when asked about that exact attribute, one attribute per question. Guess the attributes that requirement most likely concerns.
- Leave out anything the request already settles, the price, stock, the store shelf, any brand it names, and the quantity the shopper wants most or least of.
- When the request names no colour, put colour first - unless the shopper is stocking up or supplying many people: then the number of pieces first and colour second.
- Let each reason point the way: looks or fitting in with something (a room, an outfit, a set they own, a theme, hiding marks) suggests colour; stocking up suggests the number of pieces; joining an existing setup or standing in for a broken one suggests what it plugs into or runs on (plug type, voltage) when it is powered, its type or size when not; rough, wet or outdoor use suggests material; heavy or long use suggests capacity; carrying it about suggests weight or volume; a job bound to a space or a body part suggests size or length.
- After those, the attribute that splits this item into kinds, written '<noun> type' with one noun (for a name joined with '&', its last noun), then material, pattern or style.
- Name each the way listings label it: 'colour', 'material', 'plug type', 'capacity', 'piece count', 'weight', 'pattern'."""
BETTER_TOOL = fn_schema(
    "better_searches",
    "Store searches that may reach fitting listings standing better by what decides.",
    {"searches": {"type": "array", "items": dict(type="string"), "maxItems": 8, "description": "Up to eight new searches of two to six words each, none already sent."}},
    ["searches"],
)
BETTER_RULES = """A shopper wants, of the listings that meet every requirement, the one that does best on one measure (the most of something, the longest, the largest...). Some fitting listings are already found. Write store searches that could reach fitting listings doing better still on that measure:
- the item's words with the required values and a figure beyond the best one found, a few steps further, one figure per search; write some figures joined to the unit and some apart;
- for a found title that lists several amounts, one of them better, that title's main words with the better amount and the required values;
- one search with the words sellers use for the high (or low) end of that measure;
- no search already sent."""
ANSWER_RULES = """You are the shopping assistant replying to the shopper's lines, each of which changed what the shopper wants. Answer them in turn with plain statements: confirm each change in the shopper's own terms - yes, the new value now instead of the old one (or the detail added or dropped) - and that everything else stays as it was. Name no item and no price, and do not repeat the shopper's other details. Ask nothing and never end on a question; never ask the shopper to confirm, reply or let you know anything."""


class ModelGateway:

    def __init__(self):
        self.model_list = PROVIDER_MODELS.get(env_value("INFERENCE_PROVIDER", "chutes"), PROVIDER_MODELS["chutes"])
        self.spend = 0.0
        self.calls_made = 0

    def guess_readers(self) -> list[str]:
        first = GUESS_READER_FIRST.get(env_value("INFERENCE_PROVIDER", "chutes"))
        return ([first] if first else []) + [m for m in self.model_list if m != first]

    def over_spend(self) -> bool:
        return self.spend >= USD_CAP

    def query_model(
        self, rules: str, txt: str, tool_obj: dict, tokens_out: int, fallback: bool = False, deep: bool = False,
        only: str = "",
    ) -> dict | None:
        sought = tool_obj["function"]["name"]
        for model in [only] if only else self.model_list[1:] if fallback else self.model_list:
            ask_text = {
                "model": model,
                "temperature": 0,
                "max_tokens": max(tokens_out, TOKENS_FLOOR) + (THINK_EXTRA if deep else 0),
                "messages": [{"role": "system", "content": rules}, {"role": "user", "content": txt}],
                "tools": [tool_obj],
                "tool_choice": {"type": "function", "function": {"name": sought}},
                "reasoning": {"enabled": deep} if re.search("glm|kimi", model, re.I) else {"effort": "low"},
            }
            resp = _client.post("/inference/chat/completions", json_data=ask_text)
            self.calls_made += 1
            if not isinstance(resp, dict):
                continue
            self.spend += self._usage_cost(resp.get("usage") or {})
            located = self._args_from(resp, sought)
            if located is not None:
                return located
        return None

    @staticmethod
    def _usage_cost(use_info: dict) -> float:
        reported = use_info.get("cost")
        if isinstance(reported, (int, float)):
            return float(reported)
        return (int(use_info.get("prompt_tokens") or 0) + 4 * int(use_info.get("completion_tokens") or 0)) * 1e-06

    @staticmethod
    def _close_json(raw_args: str) -> dict | None:
        ends = [pos + 1 for pos, char_now in enumerate(raw_args) if char_now in '"]}']
        for end in list(reversed(ends))[:400]:
            prefix_text = raw_args[:end].rstrip().rstrip(",")
            closers: list[str] = []
            all_words_in = escape = False
            for char_now in prefix_text:
                if escape:
                    escape = False
                    continue
                if char_now == "\\":
                    escape = True
                elif char_now == '"':
                    all_words_in = not all_words_in
                elif not all_words_in and char_now == "{":
                    closers.append("}")
                elif not all_words_in and char_now == "[":
                    closers.append("]")
                elif not all_words_in and char_now in "]}" and closers:
                    closers.pop()
            if all_words_in:
                continue
            try:
                val = json.loads(prefix_text + "".join(closers[::-1]))
            except ValueError:
                continue
            if isinstance(val, dict):
                return val
        return None

    @classmethod
    def _args_from(cls, resp: dict, sought: str) -> dict | None:
        try:
            reply_msg = resp["choices"][0]["message"]
        except (IndexError, KeyError, TypeError):
            return None
        msg = reply_msg
        for tcall in msg.get("tool_calls") or []:
            raw_args = (tcall.get("function") or {}).get("arguments")
            if isinstance(raw_args, dict):
                return raw_args
            if isinstance(raw_args, str):
                try:
                    val = json.loads(raw_args)
                except ValueError:
                    val = cls._close_json(raw_args)
                if isinstance(val, dict):
                    return val
        content_words = msg.get("content")
        if not isinstance(content_words, str) or "{" not in content_words:
            return None
        try:
            val = json.loads(content_words[content_words.index("{") : content_words.rindex("}") + 1])
        except ValueError:
            return None
        if not isinstance(val, dict):
            return None
        return val[sought] if isinstance(val.get(sought), dict) else val


class ShopChannel:

    def __init__(self, problem_in: dict) -> None:
        env_in = problem_in["environment"]
        pview = env_in["policy_view"]
        self.bind_info = dict(env_in["binding"])
        self.ask_text = str(pview.get("query") or "")
        self.tool_specs = {}
        for fn_spec in pview.get("tools") or []:
            if isinstance(fn_spec, dict) and "function" in fn_spec:
                self.tool_specs[fn_spec["function"]["name"]] = fn_spec["function"]
        self.turn_cap = int(pview.get("max_steps") or 20)
        self.per_turn = max(1, min(CALLS_MAX, int(pview.get("max_calls_per_turn") or 1)))
        self.task_ref = str(problem_in.get("problem_id") or problem_in.get("id") or "task")
        self.turns_used = 0
        self.ended_flag = False
        self.trace_log: list[dict] = []
        self.found_rows: dict[tuple[str, str], dict] = {}
        self.rows_in_order: list[tuple[str, str]] = []
        self.filtered_rows: dict[tuple[str, str], str] = {}
        self.read_pages: dict[str, dict] = {}
        self.stock_seen: dict[tuple[str, str], dict] = {}
        self.basket: list[tuple[str, str]] = []
        self.unheard: list[str] = []

    def turns_free(self) -> int:
        return self.turn_cap - self.turns_used

    def has_tool(self, tool_obj: str) -> bool:
        return tool_obj in self.tool_specs

    def send_turn(self, actions: list[dict]) -> list[dict]:
        actions = [a for a in actions if a["name"] in self.tool_specs][: self.per_turn]
        if not actions or self.ended_flag or self.turns_free() <= 0:
            return []
        self.turns_used += 1
        marker = f"{self.task_ref}-turn-{self.turns_used}"
        envelope = dict(self.bind_info)
        envelope.update(
            call_id=marker,
            idempotency_key=marker,
            turn=self.turns_used,
            calls=[
                {"call_id": f"{marker}-{n}", "action": {"name": a["name"], "args": a["args"]}}
                for n, a in enumerate(actions, start=1)
            ],
        )
        result = _client.post("/environment/call", json_data=envelope)
        valid = isinstance(result, dict) and isinstance(result.get("calls"), list)
        self.trace_log.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": actions,
                "environment_result": result if valid else None,
            }
        )
        if not valid:
            self.ended_flag = True
            return []
        last_read = []
        for action, outcome_row in zip(actions, result["calls"]):
            stage = outcome_row.get("observation") if isinstance(outcome_row, dict) else None
            stage = stage if isinstance(stage, dict) else {}
            body = stage.get("observation")
            body = body if isinstance(body, dict) else {}
            if stage.get("done") is True:
                self.ended_flag = True
            self._take_in(action, body)
            last_read.append(body)
        msg = result.get("user_message")
        if isinstance(msg, dict):
            txt = str(msg.get("content") or "").strip()
            if txt:
                self.unheard.append(txt)
        return last_read

    def _record_stock(self, product: object, sku: object, cost_now: object, stock_flag: object) -> None:
        if product is None or sku is None:
            return
        piece_txt = self.stock_seen.setdefault((str(product), str(sku)), {})
        piece_txt["turn"] = self.turns_used
        if isinstance(cost_now, (int, float)) and (not isinstance(cost_now, bool)):
            piece_txt["price"] = float(cost_now)
        if isinstance(stock_flag, bool):
            piece_txt["in_stock"] = stock_flag

    def _take_in(self, action: dict, body: dict) -> None:
        sort_of = action["name"]
        if sort_of in ("search", "filter"):
            shelf = action["args"].get("category") if sort_of == "filter" else None
            for hit_row in body.get("results") or []:
                if not (isinstance(hit_row, dict) and hit_row.get("product_id") and hit_row.get("sku")):
                    continue
                ref = (str(hit_row["product_id"]), str(hit_row["sku"]))
                if ref not in self.found_rows:
                    self.found_rows[ref] = hit_row
                    self.rows_in_order.append(ref)
                if shelf:
                    self.filtered_rows.setdefault(ref, str(shelf))
                self._record_stock(ref[0], ref[1], hit_row.get("price"), hit_row.get("in_stock"))
        elif sort_of in ("view", "compare"):
            read_pages = [body] if sort_of == "view" else body.get("compared") or []
            for doc in read_pages:
                if not (isinstance(doc, dict) and doc.get("product_id")):
                    continue
                self.read_pages[str(doc["product_id"])] = doc
                for variant_info in doc.get("variants") or []:
                    if isinstance(variant_info, dict):
                        self._record_stock(
                            doc["product_id"],
                            variant_info.get("sku"),
                            variant_info.get("price"),
                            variant_info.get("in_stock"),
                        )
        elif sort_of == "inspect_stock":
            self._record_stock(body.get("product_id"), body.get("sku"), body.get("price"), body.get("in_stock"))
        elif sort_of == "remove_from_cart" and isinstance(body.get("cart"), list):
            self.basket = [
                (str(it.get("product_id")), str(it.get("sku"))) for it in body["cart"] if isinstance(it, dict)
            ]
        elif sort_of == "remove_from_cart" and not body.get("error"):
            gone = (str(action["args"].get("product_id")), str(action["args"].get("sku")))
            self.basket = [k for k in self.basket if k != gone]
        elif sort_of == "add_to_cart" and isinstance(body.get("added"), dict):
            added = body["added"]
            ref = (str(added.get("product_id")), str(added.get("sku")))
            if ref not in self.basket:
                self.basket.append(ref)
            self._record_stock(ref[0], ref[1], added.get("price"), added.get("in_stock"))


def want_from(raw_args: dict) -> dict:

    def strings_of(*names: str) -> list[str]:
        for ident in names:
            if isinstance(raw_args.get(ident), list):
                return [str(it).strip() for it in raw_args[ident] if str(it).strip()][:6]
        return []

    val = re.sub("(\\d)\\s*(?:\\\"|”|″|'')", "\\1 inch", str(raw_args.get("value") or "").strip())
    prop = str(raw_args.get("attribute") or raw_args.get("property") or "").strip()
    filed_set = not re.search("\\d", val) and (not re.search("colou?r|material", flatten_lower(prop)))
    return {
        "prop": prop,
        "value": val,
        "spell": strings_of("spellings", "written"),
        "syn": strings_of("synonyms", "alike"),
        "filed": filed_set,
    }


class EpisodeCore:

    def __init__(self, problem_in: dict) -> None:
        self.store = ShopChannel(problem_in)
        self.infer = ModelGateway()
        self.t0 = clock()
        self.item_phrase = ""
        self.named_category = ""
        self.reqs: list[dict] = []
        self.spend_limit: float | None = None
        self.limit_cut = False
        self.brand_required = ""
        self.brand_wished = ""
        self.open_wishes: list[dict] = []
        self.option_words_done: set[str] = set()
        self.stood_in = False
        self.wish_settled = False
        self.pair_sides: list[str] = []
        self.request_skus: list[tuple[str, str]] = []
        self.decider = ""
        self.decider_dir = ""
        self.decider_unit = ""
        self.purposes_list: list[str] = []
        self.pair_ask = ""
        self.pair_wishes: list[dict] = []
        self.messages_asked = 0
        self.need_questions = 0
        self.asked_attrs: list[str] = []
        self.needs_found_n = 0
        self.first_found_at = 0
        self.reply_awaited = False
        self.asked_twice: set[str] = set()
        self.unit_names_map: dict[str, str] = {}
        self.search_queue: list[str] = []
        self.sent_tags: set[str] = set()
        self.pending_filter_calls: list[dict] = []
        self.shelf_words_taken = 0
        self.kin_titles_done: set[str] = set()
        self.variant_probes_done: set[str] = set()
        self.search_allow = SEARCH_CAP
        self.read_allow = READ_CAP
        self.judge_allow = CARDS_CAP
        self.rethink_count = 0
        self.sweep_count = 0
        self.widened_once = False
        self.rulings: dict[tuple[str, str], dict] = {}
        self.rulings_key = ""
        self.cards_sent = 0
        self.lenient_mode = False
        self.memo_store: dict = {}
        self.naming_marks: dict[tuple, str] = {}
        self.naming_count = 0
        self.figure_calls = 0
        self.shelf_leaf_memo: tuple[int, str] = (-1, "")
        self.dropped: set[tuple[str, str]] = set()
        self.dropped_titles: set[str] = set()
        self.dropped_ids: set[str] = set()
        self.dropped_stems: set[str] = set()
        self.carted_fit = False
        self.reaction_owed_flag = False
        self.question_pending = ""
        self.changes_unanswered: list[str] = []
        self.change_clauses: dict[str, str] = {}
        self.reply_context: list[str] = []
        self.jump_note = ""

    def log_note(self, txt: str) -> None:
        self.store.trace_log.append({"role": "assistant", "content": f"note: {txt}"})

    def over_time(self) -> bool:
        return clock() - self.t0 > WALL_SECONDS

    def model_ok(self) -> bool:
        return not self.infer.over_spend()

    @staticmethod
    def content_words(txt: str) -> set[str]:
        return {root_word(wd) for wd in lexemes(txt) if len(wd) >= 4}

    def all_words_in(self, inner: str, outer: str) -> bool:
        toks = self.content_words(inner)
        return bool(toks) and toks <= self.content_words(outer)

    def pair_is_misread(self, duo: list[str], purposes_list: list[str]) -> bool:
        spoken = [self.content_words(u) for u in purposes_list]
        ranking = self.content_words(self.decider)
        return any(
            (
                any((self.content_words(wd) & s for s in spoken)) or len(self.content_words(wd) & ranking) >= 2
                for wd in duo
            )
        )

    def take_brief(self) -> str:
        ask_text = self.store.ask_text
        prompt = f"Request from the shopper:\n{ask_text}"
        brief = self.infer.query_model(BRIEF_RULES, prompt, BRIEF_TOOL, 700, deep=True) or {}
        if not [wd for wd in brief.get("needs") or [] if isinstance(wd, dict) and str(wd.get("value") or "").strip()]:
            retry = self.infer.query_model(BRIEF_RULES, prompt, BRIEF_TOOL, 700, fallback=True) or {}
            if retry.get("needs"):
                brief = {**brief, **{k: vt for k, vt in retry.items() if vt}} if brief else retry
        self.item_phrase = str(brief.get("kind") or "").strip()
        self.named_category = str(brief.get("shelf") or "").strip()
        rds = [
            want_from(wd)
            for wd in brief.get("needs") or []
            if isinstance(wd, dict) and str(wd.get("value") or "").strip()
        ]
        self.reqs = [
            dict(wd, stated=True)
            for i, wd in enumerate(rds)
            if not PURCHASE_TERMS.search(flatten_lower(f"{wd['prop']} {wd['value']}"))
            and not any(flatten_lower(wd["value"]) == flatten_lower(prev["value"]) for prev in rds[:i])
        ]
        self.spend_limit = positive_amount(brief.get("max_price"))
        if self.spend_limit is None:
            line_figs = [float(it.replace(",", "")) for it in re.findall("(\\d[\\d,]*\\.\\d{2})", ask_text)]
            self.spend_limit = max(line_figs) if line_figs else None
        self.brand_required = str(brief.get("maker") or "").strip()
        self.brand_wished = str(brief.get("maker_wish") or "").strip()
        if not self.brand_wished and not self.brand_required:
            said = re.search(r"\b(?:prefer|one from|made by)\s+(?:an?\s+|the\s+)?([A-Z0-9][\w&'-]*(?:\s+[A-Z0-9][\w&'-]*){0,2})", ask_text)
            if said and re.search(r"\bprefer", ask_text, re.I):
                self.brand_wished = said.group(1).strip()
        if flatten_lower(self.brand_wished) in NO_MAKER or flatten_lower(self.brand_wished) == flatten_lower(self.brand_required):
            self.brand_wished = ""
        self.request_skus = [(mo.group(1), mo.group(2)) for mo in re.finditer("(\\d{6,})::([\\w-]+)", ask_text)]
        self.decider = str(brief.get("decides") or "").strip()
        if not self.decider:
            ranks_by = re.search(
                r"\b(?:most|highest|largest|greatest|biggest|longest|lowest|least|fewest|maximum|minimum)\s+"
                r"(?:number of\s+|amount of\s+)?(?!i\b|you\b|can\b|that\b)[a-z]{3,}(?:\s+(?:count|capacity|rating))?",
                ask_text, re.I,
            )
            self.decider = ranks_by.group(0) if ranks_by else ""
        self.queue_searches(self.value_searches() + list(brief.get("searches") or []))
        if self.decider:
            self.queue_searches([str(x) for x in as_list(brief.get("rank_searches"))][:4])
        try:
            self.look_at = max(0, min(8, int(brief.get("look_at") or 0)))
        except (ValueError, TypeError, OverflowError):
            self.look_at = 0
        duo = [" ".join(str(it).split()) for it in brief.get("either_or") or [] if str(it).strip()][:2]
        purposes_list = [" ".join(str(it).split()) for it in brief.get("uses") or [] if str(it).strip()][:3]
        self.log_note(
            f"brief kind={self.item_phrase!r} wants={[(wd['prop'], wd['value']) for wd in self.reqs]} cap={self.spend_limit} pair={duo} uses={purposes_list} rank_by={self.decider!r}"
        )
        if not self.search_queue:
            self.queue_searches([" ".join(lexemes(ask_text)[:5])])
        self.brief_uses = list(purposes_list)
        self.purposes_list = [
            u
            for i, u in enumerate(purposes_list)
            if not any(
                (
                    i != j and self.all_words_in(u, vt) and (not self.all_words_in(vt, u) or j < i)
                    for j, vt in enumerate(purposes_list)
                )
            )
        ]
        # a value said only inside a reason for buying describes the reason, not the item
        self.reqs = [
            wd for wd in self.reqs if not self.only_in_purpose(wd["value"]) and not self.reason_echo(wd["value"], self.purposes_list)
        ]
        if (
            len(duo) == 2 and len(" ".join(duo).split()) <= 16 and (not self.pair_is_misread(duo, self.purposes_list))
            and not any(RANKING_TALK.search(side) for side in duo)
        ):
            wished = flatten_lower(" ".join(duo))
            self.pair_wishes = [wd for wd in self.reqs if flatten_lower(wd["value"]) in wished]
            self.reqs = [wd for wd in self.reqs if flatten_lower(wd["value"]) not in wished]
            sides = [self.wish_said(side) for side in duo]
            self.pair_sides = sides
            if self.brand_wished and flatten_lower(self.brand_wished) in wished:
                self.brand_wished = ""        # the brand is one side of the doubted pair
            if self.brand_required and flatten_lower(self.brand_required) in wished:
                self.brand_required = ""
            self.pair_ask = f"Which is more important to you: {sides[0]} or {sides[1]}?"
            return str(self.pair_ask)
        return ""

    def wish_said(self, side: str) -> str:
        in_request = {root_word(w) for w in lexemes(self.store.ask_text)}
        words = side.split()
        kept = [w for w in words if root_word(flatten_lower(w)) in in_request or flatten_lower(w) == "brand"]
        if not [w for w in kept if flatten_lower(w) != "brand"] or len(kept) == len(words):
            return side
        lead = words[0] if flatten_lower(words[0]) in ("the", "a", "an") and len(kept) > 1 else ""
        return " ".join(([lead] if lead and kept[0] != lead else []) + kept)

    @staticmethod
    def reason_echo(said_value: str, reasons: list[str]) -> bool:
        def strong(text: str) -> set[str]:
            return {root_word(w) for w in lexemes(text) if len(w) >= 5}

        spoken = strong(said_value)
        return bool(spoken) and any(spoken <= strong(reason) for reason in reasons)

    def only_in_purpose(self, said_value: str) -> bool:
        spoken = flatten_lower(said_value)
        remainder = flatten_lower(self.store.ask_text)
        if not spoken or not self.purposes_list or spoken not in remainder:
            return False
        for use in self.purposes_list:
            remainder = remainder.replace(flatten_lower(use), " ")
        return spoken not in remainder

    def known_state(self) -> str:
        details = "; ".join((f"{wd['prop']} = {wd['value']}" for wd in self.reqs)) or "(none)"
        state_lines = [
            f"Shopper's request: {self.store.ask_text}",
            f"Item wanted: {self.item_phrase}",
            f"Details insisted on: {details}",
            f"Spending limit: {(self.spend_limit if self.spend_limit is not None else 'none')}",
        ]
        if self.decider:
            state_lines.append(f"Deciding point between qualifying items: {self.decider}")
        return "\n".join(state_lines)


class Searching(EpisodeCore):

    def queue_searches(self, qlist: list, front: bool = False) -> None:
        new_ones: list[str] = []
        for qtext in qlist:
            txt = flatten_lower(" ".join(str(qtext or "").split())[:SEARCH_TEXT_MAX])
            if txt and txt not in self.sent_tags and (txt not in self.search_queue) and (txt not in new_ones):
                new_ones.append(txt)
        self.search_queue = new_ones + self.search_queue if front else self.search_queue + new_ones

    def value_word_list(self) -> list[str]:
        terms = []
        for req in self.reqs:
            called = self.name_words_set(req["prop"])
            toks = lexemes(tight_units(req["value"]))
            kept_words = [wd for wd in toks if root_word(wd) not in called]
            if not [wd for wd in kept_words if len(wd) > 2 or wd[0].isdigit()]:
                kept_words = toks         # 'no stone' stays whole: 'no' alone searches for nothing
            terms.append(" ".join(kept_words))
        return terms

    def item_word_list(self) -> list[str]:
        return [
            wd
            for wd in lexemes(self.named_category or self.item_phrase)
            if len(wd) >= 3 and wd not in {"and", "the", "for", "with", "other"}
        ]

    def value_searches(self) -> list[str]:
        distinct_vals = self.value_word_list()
        item = self.named_category or self.item_phrase
        fact_name = self.category_label()
        if fact_name and distinct_vals and (flatten_lower(fact_name) != flatten_lower(item)):
            self.queue_searches([f"{' '.join(distinct_vals)} {fact_name}", f"{distinct_vals[-1]} {fact_name}"])
        if not item:
            return []
        if not distinct_vals:
            return [item]
        joined, latest = (" ".join(distinct_vals), distinct_vals[-1])
        located = [f"{joined} {item}", f"{latest} {item}", f"{distinct_vals[0]} {item}"]
        for req in self.reqs:
            for other in (req.get("spell") or [])[:3]:
                if tight_units(other) != tight_units(req["value"]):
                    segs = [tight_units(other if wd is req else wd["value"]) for wd in self.reqs]
                    located.append(f"{' '.join(segs)} {item}")
        heads = self.item_word_list()
        for req in self.reqs:
            for other in (req.get("spell") or [])[:3]:
                if not re.search(r"\d", req["value"]) and other.lower() != req["value"].lower() \
                        and Similarity(None, flatten_lower(other), flatten_lower(req["value"])).ratio() >= 0.75:
                    spelled = " ".join(other if wd is req else str(wd["value"]) for wd in self.reqs)
                    located += [f"{spelled} {hw}" for hw in heads[:3]]
        located += [f"{joined} {hw}" for hw in heads[:3]]
        if len(distinct_vals) > 1:
            located.append(joined)
        located += [f"{latest} {hw}" for hw in heads[:2]]
        return located

    def shelf_word_searches(self, calls_made: int) -> list[str]:
        link = self.store
        headings = [link.found_rows[k].get("title") for k in link.filtered_rows if k in link.found_rows]
        headings += [p.get("title") for p in link.read_pages.values() if self.in_shelf(p)]
        if len(headings) < 3 or not self.reqs:
            return []
        known_set = set(self.item_word_list()) | {
            wd for vt in self.value_word_list() for wd in lexemes(tight_units(vt))
        }
        freq: dict[str, int] = {}
        for heading in headings:
            for tok in set(lexemes(heading)):
                if len(tok) >= 4 and (not tok.isdigit()) and (tok not in known_set):
                    freq[tok] = freq.get(tok, 0) + 1
        common = [wd for wd, n in sorted(freq.items(), key=lambda duo: (-duo[1], duo[0])) if n >= 2]
        picked = common[self.shelf_words_taken : self.shelf_words_taken + calls_made]
        self.shelf_words_taken += len(picked)
        distinct_vals = self.value_word_list()
        joined, latest = (" ".join(distinct_vals), distinct_vals[-1])
        prefix_text = (self.item_word_list() or [""])[-1]
        half = picked[: calls_made // 2]
        return (
            [f"{wd} {joined}" for wd in picked]
            + [f"{wd} {joined} {prefix_text}" for wd in half]
            + [f"{wd} {latest}" for wd in half]
        )

    def option_word_searches(self, calls_made: int) -> list[str]:
        shelf = [p for p in self.store.read_pages.values() if self.in_shelf(p)]
        if len(shelf) < 3 or not self.reqs:
            return []
        known_set = set(self.item_word_list()) | {
            wd for vt in self.value_word_list() for wd in lexemes(tight_units(vt))
        }
        freq: dict[str, int] = {}
        for doc in shelf:
            spoken = [str(vt.get("options") or "") for vt in doc.get("variants") or [] if isinstance(vt, dict)]
            spoken += [str(val) for val in (doc.get("facts") or {}).values() if len(str(val).split()) <= 2]
            for tok in {t for txt in spoken for t in lexemes(txt)}:
                if len(tok) >= 3 and not tok.isdigit() and tok not in known_set:
                    freq[tok] = freq.get(tok, 0) + 1
        common = [wd for wd, n in sorted(freq.items(), key=lambda duo: (-duo[1], duo[0])) if n >= 2]
        picked = [wd for wd in common if wd not in self.option_words_done][:calls_made]
        self.option_words_done |= set(picked)
        joined = " ".join(self.value_word_list())
        return [f"{wd} {joined}" for wd in picked]

    def titles_fitting(self) -> list[str]:
        return list(
            dict.fromkeys(
                (
                    flatten_lower((self.store.read_pages.get(k[0]) or {}).get("title"))
                    for k in self.rulings
                    if self.words_fit(k)
                )
            )
        )

    def kin_pending_any(self) -> bool:
        return len(self.kin_titles_done) < 4 and any((t not in self.kin_titles_done for t in self.titles_fitting()))

    def kin_searches(self) -> list[str]:
        joined = " ".join(self.value_word_list())
        link = self.store
        located: list[str] = []
        for ref in [k for k in self.rulings if self.words_fit(k)]:
            doc = link.read_pages.get(ref[0]) or {}
            heading = flatten_lower(doc.get("title"))
            if heading in self.kin_titles_done or len(self.kin_titles_done) >= 4:
                continue
            self.kin_titles_done.add(heading)
            toks = lexemes(heading)
            if len(toks) < 2:
                continue
            located += [
                f"{joined} {' '.join(toks[:10])}",
                f"{joined} {' '.join(toks[:5])}",
                f"{joined} {self.item_phrase} {' '.join(toks[:3])}",
            ]
            if len(toks) > 10:
                located.append(f"{joined} {' '.join(toks)}")
            opts = " ".join(lexemes(self.variant_info(ref).get("options")))
            if opts:
                located.append(f"{opts} {' '.join(toks[:10])}")
            brand_required = flatten_lower(doc.get("brand"))
            if brand_required and brand_required not in NO_MAKER:
                located.append(f"{joined} {brand_required} {self.item_phrase}")
            located.append(f"{joined} {' '.join(toks[:2])}")
            alt_forms = {
                wd
                for hit_row in link.found_rows.values()
                if flatten_lower(hit_row.get("title")) != heading
                for wd in lexemes(hit_row.get("title"))
            }
            unique = sorted(
                (
                    wd
                    for wd in dict.fromkeys(toks)
                    if len(wd) >= 5 and (not wd[0].isdigit()) and (wd not in alt_forms)
                ),
                key=len,
                reverse=True,
            )
            if unique:
                located.append(unique[0])
        return located

    def variant_probe_searches(self) -> list[str]:
        located: list[str] = []
        for ref, ruling in list(self.rulings.items()):
            if len(self.variant_probes_done) >= 4:
                break
            if not ruling.get("kind_ok") or ruling.get("by") != "words" or (not self.reqs):
                continue
            grades = [self.plain_level(ref, wd) for wd in self.reqs]
            missing = [wd for wd, g in zip(self.reqs, grades) if g < 1]
            if not missing or any((self.req_kind(wd) not in ("colour", "amount") for wd in missing)):
                continue
            heading = flatten_lower((self.store.read_pages.get(ref[0]) or {}).get("title"))
            if not heading or heading in self.variant_probes_done:
                continue
            self.variant_probes_done.add(heading)
            prefix_text = " ".join(lexemes(heading)[:10])
            for req in missing:
                for tok in [tight_units(req["value"])] + list(req.get("syn") or [])[:2]:
                    located.append(f"{tok} {prefix_text}")
        return located

    def shelf_name_guesses(self) -> list[str]:
        base = self.named_category or self.item_phrase
        if not base:
            return []

        def pluralized(tok: str) -> str:
            if tok.endswith("s"):
                return tok
            if re.search("(?:x|ch|sh)$", tok):
                return tok + "es"
            if re.search("[^aeiou]y$", tok):
                return tok[:-1] + "ies"
            return tok + "s"

        name_tries: list[str] = []
        toks = base.split()
        for skip in range(min(3, len(toks))):
            wording = " ".join(toks[skip:])
            if skip and (not re.match("[A-Za-z]{3}", wording)):
                continue
            last = re.sub("([A-Za-z]+)(\\W*)$", lambda mo: pluralized(mo.group(1)) + mo.group(2), wording)
            if skip:
                name_tries.append(last)
                continue
            every = re.sub(
                "([A-Za-z]+)(?=\\s*(?:&|/|,|\\band\\b)|\\W*$)", lambda mo: pluralized(mo.group(1)), wording
            )
            name_tries += [wording, last, every]
        return list(dict.fromkeys(name_tries))

    def category_label(self) -> str:
        called = self.shelf_leaf()
        if called:
            return called
        if self.store.filtered_rows:
            return next(iter(self.store.filtered_rows.values()))
        own = {shelf_key(self.named_category), shelf_key(self.item_phrase)} - {""}
        for doc in self.store.read_pages.values():
            for fact_name in doc.get("category_path") or []:
                if shelf_key(fact_name) in own:
                    return str(fact_name)
        return ""

    def price_step_filters(self, calls_made: int = 4) -> list[dict]:
        fact_name = self.category_label()
        if not fact_name or not self.store.has_tool("filter") or self.spend_limit is None:
            return []
        calls_out = []
        for share in STEP_SHARES:
            marker = f"filter|{flatten_lower(fact_name)}|{share}"
            if (
                len(calls_out) < calls_made
                and marker not in self.sent_tags
                and (len(self.sent_tags) < self.search_allow)
            ):
                self.sent_tags.add(marker)
                calls_out.append(
                    env_act(
                        "filter",
                        category=fact_name,
                        max_price=round(self.spend_limit * share, 2),
                        in_stock=True,
                        k=10,
                    )
                )
        return calls_out

    def brand_filters_for(self, calls_made: int) -> list[dict]:
        fact_name = self.category_label()
        if calls_made <= 0 or not fact_name or (not self.store.has_tool("filter")) or (self.spend_limit is None):
            return []
        freq: dict[str, int] = {}
        for doc in self.store.read_pages.values():
            brand_required = " ".join(str(doc.get("brand") or "").split())
            if brand_required and flatten_lower(brand_required) not in NO_MAKER and self.in_shelf(doc):
                freq[brand_required] = freq.get(brand_required, 0) + 1
        prefix = f"filter|{flatten_lower(fact_name)}|{self.spend_limit}|brand|"
        done = sum((1 for marker in self.sent_tags if marker.startswith(prefix)))
        calls_out: list[dict] = []
        for brand_required in sorted(freq, key=lambda mo: (-freq[mo], mo)):
            if len(calls_out) >= min(calls_made, BRAND_FILTER_CAP - done) or len(self.sent_tags) >= self.search_allow:
                break
            marker = prefix + flatten_lower(brand_required)
            if marker not in self.sent_tags:
                self.sent_tags.add(marker)
                calls_out.append(
                    env_act(
                        "filter",
                        category=fact_name,
                        brand=brand_required,
                        max_price=self.spend_limit,
                        in_stock=True,
                        k=10,
                    )
                )
        return calls_out

    def plan_shelf_filters(self) -> None:
        brand_required = next((wd["value"] for wd in self.reqs if "brand" in flatten_lower(wd["prop"])), None)
        fact_name = self.category_label()
        for shelf in [fact_name] if fact_name else self.shelf_name_guesses()[:2]:
            self.pending_filter_calls.append(
                env_act(
                    "filter", category=shelf, brand=brand_required, max_price=self.spend_limit, in_stock=True, k=10
                )
            )
        if brand_required:
            self.pending_filter_calls.append(
                env_act("filter", brand=brand_required, max_price=self.spend_limit, in_stock=True, k=10)
            )

    def next_search_calls(self, room: int) -> list[dict]:
        if room <= 0:
            return []
        per_turn = self.pending_filter_calls[:room] if self.store.has_tool("filter") else []
        self.pending_filter_calls = self.pending_filter_calls[len(per_turn) :] if per_turn else []
        while self.search_queue and len(per_turn) < room and (len(self.sent_tags) < self.search_allow):
            qtext = self.search_queue.pop(0)
            if qtext not in self.sent_tags:
                self.sent_tags.add(qtext)
                per_turn.append(env_act("search", query=qtext, k=10, in_stock=True))
        return per_turn


def terms_and_figures(txt: object) -> tuple[set[str], set[str]]:
    txt = tight_units(txt).replace("gray", "grey")
    toks = {root_word(wd) for wd in lexemes(txt)}
    return ({"color" if wd == "colour" else wd for wd in toks}, set(re.findall("\\d+(?:\\.\\d+)?", txt)))


def said_in_line(said_line: str, reading_add: dict) -> bool:
    toks, line_figs = terms_and_figures(said_line)
    frms = (
        [reading_add.get("value")]
        + list(reading_add.get("spellings") or reading_add.get("written") or [])
        + list(reading_add.get("synonyms") or reading_add.get("alike") or [])
    )
    for frm in frms:
        own_words, form_figs = terms_and_figures(frm)
        if form_figs:
            if form_figs <= line_figs:
                return True
        elif any((wd in toks for wd in own_words if len(wd) > 2)) or (own_words and own_words <= toks):
            return True                   # a short value (us, uk, xl) counts when every word of it is in the line
    return False


class Hearing(Searching):

    @staticmethod
    def fix_lists(rd: dict | None) -> dict:
        rd = dict(rd or {})
        for field in ("new_needs", "withdrawn", "searches"):
            val = rd.get(field)
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except ValueError:
                    val = []
            rd[field] = val if isinstance(val, list) else []
        return rd

    @staticmethod
    def readings_added(rd: dict) -> list[dict]:
        return [a for a in rd.get("new_needs") or [] if isinstance(a, dict) and str(a.get("value") or "").strip()]

    def withdrawal_spoken(self, said_line: str, ident: str, adds: list[dict]) -> bool:
        said, _ = terms_and_figures(said_line)
        for req in self.reqs:
            if self.name_words_set(req["prop"]) == self.name_words_set(ident):
                toks = self.name_words_set(req["prop"]) | {root_word(wd) for wd in lexemes(req["value"])}
                if toks & said:
                    return True
        return any((self.name_words_set(a.get("attribute") or a.get("property")) == self.name_words_set(ident) for a in adds))

    def hear_waiting(self) -> None:
        if self.store.unheard:
            self.hear_lines()

    def hear_lines(self) -> tuple[bool, bool]:
        link = self.store
        changed, expects = False, False
        while link.unheard:
            said_line = link.unheard.pop(0)
            segs = [seg.strip() for seg in said_line.split("\n") if seg.strip()]
            if len(segs) > 1:
                link.unheard[:0] = segs
                continue
            prior = [(wd["prop"], wd["value"]) for wd in self.reqs]
            cap_before = self.spend_limit
            txt = f"{self.known_state()}\n\nThe shopper's newest line:\n{said_line}"
            rd = self.fix_lists(
                self.infer.query_model(LISTEN_RULES, txt, LISTEN_TOOL, 1500) if self.model_ok() else None
            )
            if self.reply_awaited and (not self.readings_added(rd)) and self.model_ok():
                second = self.fix_lists(self.infer.query_model(LISTEN_RULES, txt, LISTEN_TOOL, 1500, fallback=True))
                if self.readings_added(second):
                    second["shelf"] = second.get("shelf") or rd.get("shelf")
                    rd = second
            answering, self.reply_awaited = (self.reply_awaited, False)
            if rd.get("market_reaction") is True:
                rd["max_price"] = 0
                self.market_lines += 1
            adds = [a for a in self.readings_added(rd) if said_in_line(said_line, a) and not dropped_in_line(said_line, a)]
            new_cap = positive_amount(rd.get("max_price"))
            cap_line = new_cap is not None and new_cap != self.spend_limit
            swapped = {flatten_lower(a.get("attribute") or a.get("property")) for a in adds}
            change_said = bool(CHANGE_WORDS.search(said_line))
            withdrew = False
            adds_only = bool(ADDED_TALK.search(said_line)) and not SWAP_TALK.search(said_line)
            let_go: list[dict] = []
            for ident in rd.get("withdrawn") or []:
                if adds_only or (cap_line and flatten_lower(ident) not in swapped):
                    continue
                if not self.withdrawal_spoken(said_line, str(ident), adds):
                    continue
                kept_words = [wd for wd in self.reqs if self.name_words_set(wd["prop"]) != self.name_words_set(ident)]
                let_go += [wd for wd in self.reqs if all(wd is not k for k in kept_words)]
                changed = changed or len(kept_words) != len(self.reqs)
                withdrew = withdrew or len(kept_words) != len(self.reqs)
                self.reqs = kept_words
            answers_pair = answering and bool(self.pair_ask) and (self.messages_asked == 1)
            if answering and change_said and not answers_pair:
                withdrew = True               # a line that changes what is wanted answers no question
            shelf_said = " ".join(str(rd.get("shelf") or "").split())
            if not (shelf_said and flatten_lower(shelf_said) in flatten_lower(said_line)):
                shelf_said = (
                    self.named_category
                    if self.named_category and flatten_lower(self.named_category) in flatten_lower(said_line)
                    else ""
                )
            known_set = {wd for req in self.reqs for wd in self.name_words_set(req["prop"])} - {"type", "kind"}
            fresh_props: list[str] = []
            added_now: list[dict] = []
            replaced_now = withdrew
            change_said = bool(CHANGE_WORDS.search(said_line))
            for reading_add in adds:
                if shelf_said and set(shelf_key(reading_add.get("value")).split()) <= set(
                    shelf_key(shelf_said).split()
                ):
                    continue
                fresh_req = want_from(reading_add)
                if PURCHASE_TERMS.search(flatten_lower(f"{fresh_req['prop']} {fresh_req['value']}")):
                    continue
                turned = re.search(
                    r"\bthe ([a-z][\w -]{1,40}?) (?:must|has to|needs to) be (?:of )?(?:the |an? )?([a-z][\w -]{1,40}?) (type|kind)\b",
                    flatten_lower(said_line),
                )
                read_words = self.content_words(fresh_req["value"])
                if (
                    turned and read_words and read_words <= self.content_words(self.item_phrase)
                    and read_words <= self.content_words(turned.group(2))
                    and not self.content_words(turned.group(1)) <= self.content_words(self.item_phrase)
                ):
                    # the item's own kind was read as the value: the line names the value first
                    fresh_req = dict(fresh_req, prop=f"{turned.group(2)} {turned.group(3)}", value=turned.group(1), spell=[], syn=[])
                if self.echoes_use(fresh_req["value"]):
                    continue                  # the reason for buying said again, not a detail of the item
                wish = next(
                    (wd for wd in self.open_wishes if flatten_lower(wd["value"]) == flatten_lower(fresh_req["value"])), None
                )
                if wish is not None and answering and any(wd is wish for wd in self.reqs):
                    # one side of the doubted pair named as a must: the other side is no requirement any more
                    wish["chosen"] = True
                    left = [wd for wd in self.open_wishes if wd is not wish]
                    self.reqs = [wd for wd in self.reqs if all(wd is not other for other in left)]
                    if any("brand" in flatten_lower(other["prop"]) for other in left):
                        self.brand_required = ""
                    self.open_wishes = []
                    self.wish_settled = True
                    changed = True
                    continue
                existing = next(
                    (wd for wd in self.reqs if self.name_words_set(wd["prop"]) == self.name_words_set(fresh_req["prop"])), None
                )
                former = existing or next(
                    (wd for wd in let_go if self.name_words_set(wd["prop"]) == self.name_words_set(fresh_req["prop"])), None
                )
                bare_unit = re.search(r"\d\s*([a-z]+)\b", flatten_lower(former["value"])) if former is not None else None
                if bare_unit and re.fullmatch(r"\d+(?:\.\d+)?", str(fresh_req["value"]).strip()):
                    fresh_req["value"] = f"{str(fresh_req['value']).strip()} {bare_unit.group(1)}"
                    fresh_req["spell"], fresh_req["syn"] = [], []
                if existing is not None and flatten_lower(existing["value"]) == flatten_lower(fresh_req["value"]):
                    continue
                if existing is not None and ADDED_TALK.search(said_line) and not SWAP_TALK.search(said_line):
                    existing = None           # 'it also needs ...': a second value beside the first
                # an answer that gives the request's own attribute another value, with no word of change: the request
                # was read wrong there, and the answer is the detail that was held back
                old_words, new_words = set(lexemes(existing["value"])) if existing else set(), set(lexemes(fresh_req["value"]))
                misread = (
                    answering and existing is not None and bool(existing.get("stated"))
                    and not change_said and not answers_pair
                    and not (old_words <= new_words or new_words <= old_words)
                )
                if existing is not None:
                    self.reqs.remove(existing)
                    if not misread:
                        replaced_now = True
                        self.shopper_moved = True
                elif change_said and answering and not answers_pair:
                    self.shopper_moved = True
                if answers_pair:
                    fresh_req["chosen"] = True
                    if "brand" in flatten_lower(fresh_req["prop"]):
                        self.brand_required = str(fresh_req["value"]).strip()
                if misread:
                    fresh_req["stated"] = False
                    fresh_props.append(fresh_req["prop"])
                elif existing is None and (not self.name_words_set(fresh_req["prop"]) - {"type", "kind"} & known_set):
                    fresh_props.append(fresh_req["prop"])
                self.reqs.append(fresh_req)
                added_now.append(fresh_req)
                changed = True
                if "brand" in flatten_lower(fresh_req["prop"]):
                    self.plan_shelf_filters()
                    self.queue_searches([f"{fresh_req['value']} {self.item_phrase}", fresh_req["value"]], front=True)
            if change_said and added_now and not replaced_now and not answers_pair:
                # 'forget X, it has to be Y': the requirement whose value the line names goes, whatever it was called
                line_words = set(lexemes(tight_units(said_line)))
                named_old = [
                    wd for wd in self.reqs
                    if all(wd is not n for n in added_now) and lexemes(tight_units(wd["value"]))
                    and set(lexemes(tight_units(wd["value"]))) <= line_words
                    and not set(lexemes(tight_units(wd["value"]))) <= {t for n in added_now for t in lexemes(tight_units(n["value"]))}
                ]
                if len(named_old) == 1:
                    self.reqs = [wd for wd in self.reqs if wd is not named_old[0]]
                    replaced_now, self.shopper_moved = True, True
            if answering and fresh_props and (not answers_pair) and not replaced_now:
                self.note_reason_use(said_line)
                self.found_lines.append(said_line)
                self.needs_found_n += 1
                self.first_found_at = self.first_found_at or self.need_questions
                self.latest_found = fresh_props
            if shelf_said and shelf_key(shelf_said) != shelf_key(self.named_category):
                self.named_category, self.shelf_leaf_memo = (shelf_said, (-1, ""))
                self.plan_shelf_filters()
            if new_cap is not None and any(
                (
                    abs(new_cap - float((link.stock_seen.get(k) or {}).get("price") or -1.0)) < 0.005
                    for k in link.basket
                )
            ):
                new_cap = None
            if new_cap is not None and new_cap != self.spend_limit:
                self.shopper_moved = True
                lower = self.spend_limit is None or new_cap < self.spend_limit
                self.limit_cut = self.limit_cut or (self.spend_limit is not None and new_cap < self.spend_limit)
                self.spend_limit, changed = (new_cap, True)
                if lower:
                    self.plan_shelf_filters()
            decides = str(rd.get("decides") or "").strip()
            if decides and flatten_lower(decides) != flatten_lower(self.decider):
                self.decider, changed = (decides, True)
            if withdrew:
                self.shopper_moved = True
            if (not answering or replaced_now) and [(wd["prop"], wd["value"]) for wd in self.reqs] != prior:
                self.shopper_moved = self.shopper_moved or not answering
                self.changes_unanswered.append(said_line)
                self.change_clauses[said_line] = self.change_summary(prior, cap_before)
            withdrawn_values = [v for p, v in prior if (p, v) not in [(wd["prop"], wd["value"]) for wd in self.reqs]]
            for old in withdrawn_values:
                stale = flatten_lower(tight_units(old))
                if stale:
                    self.search_queue = [q for q in self.search_queue if stale not in q]
                    # the value just replaced is never another word for its replacement ('plain instead of solid')
                    for wd in self.reqs:
                        for field in ("spell", "syn"):
                            wd[field] = [x for x in wd.get(field) or [] if stale not in flatten_lower(tight_units(x))]
            if changed:
                self.queue_searches(self.value_searches() + list(rd.get("searches") or []), front=True)
                self.search_allow = max(self.search_allow, len(self.sent_tags) + 16)
                self.read_allow = max(self.read_allow, len(link.read_pages) + 40)
                self.judge_allow = max(self.judge_allow, self.cards_sent + 36)
            self.log_note(
                f"heard {said_line[:80]!r}: adds={[(a.get('attribute'), a.get('value')) for a in adds]} withdrawn={rd.get('withdrawn')} cap={self.spend_limit} reply={rd.get('expects_reply')}"
            )
            if rd.get("expects_reply") and (not rd.get("market_reaction")):
                if self.reaction_owed_flag:
                    self.reaction_owed_flag = False
                else:
                    expects = True
                    self.question_pending = said_line
            elif rd.get("market_reaction"):
                self.reaction_owed_flag = False
        return (changed, expects)

    def echoes_use(self, said_value: str) -> bool:
        spoken = self.content_words(said_value)
        return bool(spoken) and any(spoken <= self.content_words(use) for use in self.purposes_list)

    def note_reason_use(self, said_line: str) -> None:
        if len(self.purposes_list) >= 2 or not self.purposes_list:
            return
        heard = self.content_words(said_line)
        if any(self.content_words(use) & heard for use in self.purposes_list):
            return
        reason = re.search(r"\b(?:because|since|as)\b(.*?)(?:,|$)", said_line, re.I)
        self.purposes_list.append(" ".join((reason.group(1) if reason else said_line).split())[:120])
        self.log_note(f"a use the brief missed: {self.purposes_list[-1]!r}")

    def change_summary(self, prior: list[tuple[str, str]], cap_before: float | None) -> str:
        after = [(wd["prop"], wd["value"]) for wd in self.reqs]
        gone = [it for it in prior if it not in after]
        came = [it for it in after if it not in prior]
        clause_list, paired = ([], set())
        for prop, val in came:
            previous = next(((p, vt) for p, vt in gone if flatten_lower(p) == flatten_lower(prop)), None)
            if previous is None and len(gone) == 1 and (len(came) == 1):
                previous = gone[0]
            if previous is None:
                clause_list.append(f"it now also needs {val} as its {prop}")
            else:
                paired.add(previous)
                clause_list.append(f"the {prop} is now {val} instead of {previous[1]}")
        clause_list += [f"it no longer needs {val} as its {prop}" for prop, val in gone if (prop, val) not in paired]
        if self.spend_limit is not None and self.spend_limit != cap_before:
            clause_list.append(f"the most you will spend is now {self.spend_limit:.2f}")
        return "; ".join(clause_list)


def dropped_in_line(said_line: str, reading_add: dict) -> bool:
    # a value the line itself lets go of ('forget about silver', 'instead of silver') is no new requirement
    value = " ".join(lexemes(reading_add.get("value")))
    if not value:
        return False
    heard = " ".join(lexemes(said_line))
    return bool(re.search(
        r"\b(?:forget(?: about)?|instead of|rather than|no longer|drop|not)\s+(?:\w+\s+){0,3}?" + re.escape(value) + r"\b", heard
    ))


class Questioning(Hearing):

    def needs_expected_n(self) -> int:
        lone_purpose = len(self.purpose_phrases()) <= 1 and not self.purposes_gave and self.needs_found_n >= 1
        if sum(1 for k in self.guess_reason_counts if k >= 2) >= 2 and not lone_purpose:
            return 2
        if self.expect_two or re.search(r"\bmore specifically\b", self.store.ask_text, re.I):
            return 2
        if lone_purpose:
            return 1
        uses = [u for u in self.brief_uses if not RANKING_TALK.search(str(u))]
        return max(1, min(2, max(len(self.purpose_phrases()), len(uses))))

    def question_limit(self) -> int:
        return (ASK_ROUNDS_AFTER_PAIR if self.pair_ask else ASK_ROUNDS) + self.needs_expected_n() - 1

    def shelf_summary_text(self) -> str:
        link = self.store
        read_pages = [p for p in link.read_pages.values() if self.in_shelf(p)] or list(link.read_pages.values())
        freq: dict[str, int] = {}
        examples_by_fact: dict[str, list[str]] = {}
        for doc in read_pages:
            for fact_name, val in (doc.get("facts") or {}).items():
                fact_name = str(fact_name)
                freq[fact_name] = freq.get(fact_name, 0) + 1
                kept_words = examples_by_fact.setdefault(fact_name, [])
                txt = " ".join(str(val).split())[:40]
                if txt and txt not in kept_words and (len(kept_words) < 4):
                    kept_words.append(txt)
        labels = sorted(freq, key=lambda ident: (-freq[ident], ident))[:LABELS_SHOWN]
        variants = list(
            dict.fromkeys(
                (
                    " ".join(str(vt.get("options") or "").split())[:50]
                    for doc in read_pages
                    for vt in doc.get("variants") or []
                    if isinstance(vt, dict) and str(vt.get("options") or "").strip()
                )
            )
        )
        in_list_fact = (
            "\n".join((f"- {ident} ({freq[ident]}): {'; '.join(examples_by_fact[ident])}" for ident in labels))
            or "(nothing read yet)"
        )
        return f"Listings read: {len(read_pages)}\nFact labels:\n{in_list_fact}\nVariant names seen: " + (
            " | ".join(variants[:15]) or "(none)"
        )

    def already_covered(self, ident: str) -> bool:
        got_words = self.name_words_set(ident)
        if not got_words:
            return True
        known_set = [self.name_words_set(wd["prop"]) for wd in self.reqs + self.pair_wishes] + [
            self.name_words_set(p) for p in self.asked_attrs
        ]
        return any((k and (k <= got_words or got_words <= k) for k in known_set)) or bool(
            got_words & self.content_words(self.decider)
        )

    def name_open(self, ident: str) -> bool:
        low = flatten_lower(ident)
        return not (
            GROUP_NAMES.search(low) or PURCHASE_TERMS.search(low) or "brand" in low
            or (self.just_the_item(ident) and not re.search(r"\b(?:type|kind)\b", low)) or self.already_covered(ident)
        )

    def label_picks_open(self) -> list[str]:
        left = [n for n in self.shelf_picks if self.name_open(n)]
        shelved = sum(1 for doc in self.store.read_pages.values() if self.in_shelf(doc))
        if left or self.label_reads >= SHELF_READS or shelved < 3 or not self.model_ok() or self.over_time():
            return left
        self.label_reads += 1
        txt = (
            f"{self.known_state()}\nPurposes: "
            + ("; ".join(self.purposes_list) or "(none given)")
            + "\nAlready asked about: "
            + (", ".join(self.asked_attrs) or "nothing")
            + f"\n\nWhat the store's listings of this kind show:\n{self.shelf_summary_text()}"
        )
        picks = as_list((self.infer.query_model(PROPS_RULES, txt, PROPS_TOOL, 300) or {}).get("names"))
        self.shelf_picks = [n for n in (self.clean_attribute(p) for p in picks) if n]
        self.log_note(f"shelf reading {self.label_reads}: {self.shelf_picks}")
        return [n for n in self.shelf_picks if self.name_open(n)]

    def just_the_item(self, ident: str) -> bool:
        own = self.name_words_set(self.item_phrase) | self.name_words_set(self.named_category)
        said = self.name_words_set(ident) - {"and", "type", "kind"}
        return bool(said) and said <= own

    def clean_attribute(self, raw: object) -> str:
        name = " ".join(str(raw or "").lower().replace("_", " ").split()).strip(" .,;:")
        name = re.split(r"\s*(?:&|,|/|\band\b)\s*", name)[-1]
        if re.fullmatch(r"colou?rs?(?: family)?", name):
            name = "colour"
        if not name or len(name.split()) > 4:
            return ""
        if re.fullmatch(r"(?:the\s+)?(?:price|budget|cost|stock|availability|category|shelf|item|product|quantity|purpose|use|type|kind|brand|model|make)", name):
            return ""
        if self.content_words(name) & self.content_words(self.decider):
            return ""
        return name

    def opening_guess(self) -> list[str]:
        if self.guess_pool is not None:
            return self.guess_pool
        ask_txt = f"Request from the shopper:\n{self.store.ask_text}"
        models = self.infer.guess_readers() if self.model_ok() else []
        answers: list = [None] * len(models)
        stagger = (sum(map(ord, self.store.task_ref)) % 21) / 10.0   # spreads parallel tasks' first calls apart

        def read_with(slot: int, model: str) -> None:
            if slot == 0 and model == GUESS_READER_FIRST.get(env_value("INFERENCE_PROVIDER", "chutes")):
                pause(stagger)
            answers[slot] = self.infer.query_model(GUESS_RULES, ask_txt, GUESS_TOOL, 400, only=model)

        workers = [threading.Thread(target=read_with, args=(i, m), daemon=True) for i, m in enumerate(models)]
        for w in workers:
            w.start()
        until = clock() + GUESS_WAIT
        for w in workers:
            w.join(timeout=max(0.0, until - clock()))
        readings = [a for a in answers if isinstance(a, dict)]
        if not readings and self.model_ok() and not self.over_time():
            again = self.infer.query_model(GUESS_RULES, ask_txt, GUESS_TOOL, 400)
            readings = [again] if isinstance(again, dict) else []
        for ans in readings:
            given = [" ".join(str(x).split()) for x in as_list(ans.get("reasons"), "[\\n;]+") if str(x).strip()]
            self.guess_reason_counts.append(len(given))
            self.guess_reasons.append(given)
        points: dict[str, int] = {}
        first_seen: dict[str, tuple[int, int]] = {}
        for order, ans in enumerate(readings):
            for place, raw in enumerate(as_list(ans.get("attributes"))[:4]):
                name = self.clean_attribute(raw)
                if name:
                    points[name] = points.get(name, 0) + 4 - place
                    first_seen.setdefault(name, (order, place))
        pool = [p for p in sorted(points, key=lambda p: (-points[p], first_seen[p])) if p != "colour"]
        if not self.already_covered("colour"):
            stocking = re.search(r"\b(?:restock\w*|stock(?:ing)? up|suppl(?:y|ies|ying))\b", self.store.ask_text, re.I)
            pool.insert(1 if pool[:1] == ["piece count"] and stocking else 0, "colour")
        self.guess_pool = pool
        self.log_note(f"pooled guess: {pool}")
        return pool

    def fact_attribute(self, key: object, val: object = None) -> str:
        k = str(key).split(":", 1)[-1].lower()
        if re.match(r"colou?rs?(?:_|$)", k):
            return "colour"
        kind = re.match(r"type_of_(.+)", k)
        words = (kind.group(1) + "_type" if kind else k).replace("_", " ").split()
        while words[1:] and len(words[-1]) < 3:
            del words[-1]
        if val is not None and words[1:] and words[-1] in self.figure_units(val):
            del words[-1]
        if not words or (len(words) == 1 and len(words[0]) <= 2):
            return ""
        return self.singular_name(" ".join(words))

    @staticmethod
    def figure_units(txt: object) -> set[str]:
        return {m.group(1) for m in re.finditer(r"\d(?:[.,]?\d)*\s*([a-z]{1,6})\b", flatten_lower(txt))}

    def learn_unit_meanings(self, units: list[str]) -> None:
        unnamed = [u for u in units if u not in self.unit_names_map]
        if not unnamed or not self.model_ok():
            return
        ans = self.infer.query_model(UNITS_RULES, f"Item: {self.item_phrase}\nUnits: " + ", ".join(unnamed), UNITS_TOOL, 200) or {}
        for duo in as_list(ans.get("pairs"), "[\\n;]+"):
            unit_txt, _, ident = str(duo).partition("=")
            unit_txt, ident = flatten_lower(unit_txt), " ".join(lexemes(ident)[:4])
            if unit_txt in unnamed:
                self.unit_names_map[unit_txt] = "" if ident in ("", "none") else ident
        for u in unnamed:
            self.unit_names_map.setdefault(u, "")

    def ranked_away(self) -> set[str]:
        return self.content_words(self.decider) - {"type", "number", "amount", "total"}

    def shelf_attribute_rank(self) -> tuple[list[str], list[str]]:
        docs = [d for d in self.store.read_pages.values() if self.in_shelf(d)]
        if not docs:
            return [], []
        filled: dict[str, int] = {}
        seen_vals: dict[str, list[str]] = {}
        unit_labels: dict[str, dict[str, int]] = {}
        for doc in docs:
            facts = doc.get("facts") or {}
            names = []
            for key, val in facts.items():
                name, text = self.fact_attribute(key, val), flatten_lower(val)
                if not name:
                    continue
                seen_vals.setdefault(name, []).append(text[:60])
                if re.match(r"(?:brand|model)\b", name) or text in ("yes", "no", "true", "false", "n/a") \
                        or text == flatten_lower(str(key).replace("_", " ")) or re.match(r"(?:non|not|no)\b", text):
                    continue
                if name not in names:
                    names.append(name)
                units = self.figure_units(val)
                tail = str(key).lower().rsplit("_", 1)
                if not units and tail[1:] and re.fullmatch(r"[a-z]{1,3}", tail[1]) and re.fullmatch(r"\s*\d+(?:\.\d+)?\s*", str(val)):
                    units = {tail[1]}
                for u in units:
                    unit_labels.setdefault(u, {}).setdefault(name, 0)
                    unit_labels[u][name] += 1
            for name in names:
                filled[name] = filled.get(name, 0) + 1

        def bland(name: str) -> bool:
            vals = seen_vals.get(name) or []
            top = max((vals.count(v) for v in set(vals)), default=0)
            return (len(vals) >= 3 and top >= 0.8 * len(vals)) or (bool(vals) and sum(map(len, vals)) > 40 * len(vals)) \
                or any(re.search(r"<|http|\d{4}-\d\d", v) for v in vals)

        meaning: dict[str, str] = {}
        for u, labels in unit_labels.items():
            fit = {n: c for n, c in labels.items() if n in filled and not bland(n)}
            if fit:
                meaning[u] = max(fit, key=fit.get)

        def printed_text(doc: dict) -> str:
            return " ".join([str(doc.get("title") or "")] + [str(v.get("options") or "") for v in doc.get("variants") or [] if isinstance(v, dict)])

        printed: dict[str, int] = {}
        for doc in docs:
            for m in re.finditer(r"\d(?:[.,]?\d)*\s*([a-z]{1,6})\b", flatten_lower(printed_text(doc))):
                printed[m.group(1)] = printed.get(m.group(1), 0) + 1
        common = [u for u in sorted(printed, key=lambda u: -printed[u]) if printed[u] >= 2]
        self.learn_unit_meanings(common)      # the model's word for every printed unit wins over a fact label
        for u in common:
            ident = self.unit_names_map.get(u) or ""
            if ident and not re.fullmatch(r"(?:the\s+)?(?:price|budget|cost|stock|item|product|quantity|type|kind)", ident) \
                    and not self.name_words_set(ident) <= self.name_words_set(self.item_phrase):
                meaning[u] = ident
        by_title: dict[str, int] = {}
        for doc in docs:
            for ident in {meaning[u] for u in self.figure_units(printed_text(doc)) if u in meaning}:
                by_title[ident] = by_title.get(ident, 0) + 1
        filled = {n: c for n, c in filled.items() if not bland(n)}
        total = len(docs)
        away = self.ranked_away()
        request_words = set(lexemes(self.store.ask_text))
        given = {meaning[u] for u in self.figure_units(self.store.ask_text) if u in meaning}

        def allowed(name: str) -> bool:
            return not (self.content_words(name) & away) and not (len(name.split()) == 1 and name in request_words)

        amounts = [n for n in sorted(by_title, key=lambda n: -by_title[n]) if n not in given and by_title[n] >= 2 and allowed(n)]
        score = {n: c / total * (2.0 if n == "type" or n.endswith(" type") else 1.0) for n, c in filled.items()}
        for label, hits in by_title.items():
            if label not in given:
                score[label] = max(score.get(label, 0.0), hits / total * 1.5)
        ranked = [n for n in sorted(score, key=lambda n: -score[n])
                  if max(filled.get(n, 0), by_title.get(n, 0)) / total >= 0.15 and allowed(n)]
        return ranked, amounts

    def ask_candidates(self) -> list[str]:
        guess = self.opening_guess()
        ranked, amounts = self.shelf_attribute_rank()
        picked_side = {flatten_lower(wd["prop"]) for wd in self.reqs if wd.get("chosen")}
        other_side = [
            flatten_lower(wd["prop"]) for wd in self.pair_wishes
            if picked_side and flatten_lower(wd["prop"]) not in picked_side and "brand" not in flatten_lower(wd["prop"])
        ][:1]
        held = {flatten_lower(wd["prop"]) for wd in self.reqs + self.pair_wishes} - set(other_side)
        held_words = [self.name_words_set(p) for p in held if self.name_words_set(p) - {"type", "kind"}]
        away = self.ranked_away()
        picked: list = []

        def core(name: str) -> list[str]:
            return [root_word(w) for w in lexemes(name) if w not in ("of", "the", "and", "for")]

        def near_copy(name: str) -> bool:
            mine = core(name)
            for other in picked + self.asked_attrs:
                theirs = core(other)
                if mine and theirs and mine[-1] == theirs[-1] and (set(mine) <= set(theirs) or set(theirs) <= set(mine)) \
                        and len(set(mine) ^ set(theirs)) <= 1:
                    return True
            return False

        plain_words = {"type", "kind", "filter", "general", "number", "count", "family"}

        def said_before(name: str) -> bool:
            mine = core(name)
            bare = set(mine) - plain_words
            tails = [core(other) for other in picked + self.asked_attrs if core(other)]
            if any(bare and bare == set(theirs) - plain_words for theirs in tails):
                return True                   # 'material filter' after 'material', 'number of piece' after 'piece count'
            same_tail = [theirs for theirs in tails if mine and theirs[-1] == mine[-1]]
            return any(len(set(mine) & set(theirs)) >= 2 for theirs in same_tail)

        read_names = self.label_picks_open()
        for name in guess[:1] + other_side + read_names[:3] + amounts[:1] + read_names[3:] + guess[1:3] + ranked + guess[3:]:
            if not name or name in picked or name in self.asked_attrs or name in held:
                continue
            if said_before(name):
                continue
            if any(hw and hw <= self.name_words_set(name) for hw in held_words):
                continue                      # 'volume in ml' while the volume is held
            if (self.content_words(name) & away) or (self.brand_required and "brand" in name):
                continue
            if name in ranked and name not in guess and near_copy(name):
                continue
            picked.append(name)
        return self.item_names_first(picked, set(other_side) | set(amounts[:1]) | set(self.shelf_picks))

    def item_names_first(self, names: list[str], kept: set[str]) -> list[str]:
        for n in names:
            if n not in self.aside_marks and SALE_TALK.search(n):
                self.aside_marks[n] = True
        unread = [n for n in names if n not in self.aside_marks and n != "colour" and n not in kept]
        if unread and self.model_ok() and self.aside_calls < ASIDE_CALLS and not self.over_time():
            self.aside_calls += 1
            shown = "\n".join(f"{i}. {n}" for i, n in enumerate(unread, 1))
            ans = self.infer.query_model(ASIDE_RULES, f"Item: {self.item_phrase}\nNames:\n{shown}", ASIDE_TOOL, 250) or {}
            for mark in as_list(ans.get("marks"), "[\\n;,]+"):
                hit = re.match(r"\s*\[?(\d+)\]?\W+(item|other)\b", str(mark).lower())
                if hit and 1 <= int(hit.group(1)) <= len(unread):
                    self.aside_marks[unread[int(hit.group(1)) - 1]] = hit.group(2) == "other"
            for n in unread:
                self.aside_marks.setdefault(n, False)
            self.log_note(f"names not about the item itself, asked last: {[n for n in unread if self.aside_marks[n]]}")
        return [n for n in names if not self.aside_marks.get(n)] + [n for n in names if self.aside_marks.get(n)]

    def shelf_units(self) -> list[str]:
        read_pages = [p for p in self.store.read_pages.values() if self.in_shelf(p)] or list(
            self.store.read_pages.values()
        )
        given = set()
        for req in self.reqs:
            for tok in lexemes(tight_units(req["value"])):
                mobj = re.fullmatch("[\\d.]+([a-z]+)", tok)
                if mobj:
                    given.add(mobj.group(1))
        freq: dict[str, int] = {}
        for doc in read_pages:
            texts = [str(doc.get("title") or "")] + [
                str(vt.get("options") or "") for vt in doc.get("variants") or [] if isinstance(vt, dict)
            ]
            texts += [str(it) for it in (doc.get("facts") or {}).values()]
            for unit_txt in {
                mo.group(1)
                for mo in re.finditer("\\d(?:[.,]?\\d)*\\s*([a-z]{1,5})\\b", flatten_lower(" ".join(texts)))
            }:
                freq[unit_txt] = freq.get(unit_txt, 0) + 1
        return [u for u in sorted(freq, key=lambda u: (-freq[u], u)) if freq[u] >= 3 and u not in given]

    def amount_attribute(self) -> str:
        unit_set = self.shelf_units()[: UNIT_CHOICES + 2]
        if not unit_set:
            return ""
        unnamed = [u for u in unit_set if u not in self.unit_names_map]
        if unnamed and self.model_ok():
            ans = (
                self.infer.query_model(
                    UNITS_RULES, f"Item: {self.item_phrase}\nUnits: " + ", ".join(unnamed), UNITS_TOOL, 200
                )
                or {}
            )
            for duo in as_list(ans.get("pairs"), "[\\n;]+"):
                unit_txt, _, ident = str(duo).partition("=")
                unit_txt, ident = (flatten_lower(unit_txt), " ".join(lexemes(ident)[:4]))
                if unit_txt in unnamed:
                    self.unit_names_map[unit_txt] = "" if ident in ("", "none") else ident
        named = [self.unit_names_map.get(unit_txt, f"amount in {unit_txt}") for unit_txt in unit_set]
        # a count of pieces is the amount kept back most often: it goes first whenever the shelf states one
        named.sort(key=lambda ident: not re.search(r"\bpiece", ident))
        for ident in named:
            if ident and (not self.already_covered(ident)):
                return ident
        return ""

    def varied_fact(self, type_only: bool = False) -> str:
        coverage: dict[str, int] = {}
        distinct: dict[str, set[str]] = {}
        flat_hits: dict[str, int] = {}
        maker_hits: dict[str, int] = {}
        entry_hits: dict[str, dict[str, int]] = {}
        for doc in self.store.read_pages.values():
            if not self.in_shelf(doc):
                continue
            brand_required = flatten_lower(doc.get("brand"))
            for fact_name, val in (doc.get("facts") or {}).items():
                ident = " ".join(lexemes(str(fact_name).replace("_", " ")))
                txt = flatten_lower(val)
                coverage[ident] = coverage.get(ident, 0) + 1
                distinct.setdefault(ident, set()).add(txt)
                if re.fullmatch("(?:yes|no|true|false|none|n/?a)", txt):
                    flat_hits[ident] = flat_hits.get(ident, 0) + 1
                for piece in {p.strip() for p in re.split(r"[,;/|]+", txt) if p.strip()}:
                    entry_hits.setdefault(ident, {})[piece] = entry_hits.get(ident, {}).get(piece, 0) + 1
                if brand_required and txt == brand_required:
                    maker_hits[ident] = maker_hits.get(ident, 0) + 1

        def varies(ident: str) -> bool:
            n = coverage[ident]
            return (
                n >= 2
                and len(distinct[ident]) >= 2
                and (2 * flat_hits.get(ident, 0) <= n)
                and (2 * maker_hits.get(ident, 0) <= n)
                and (not (n >= 3 and max(entry_hits.get(ident, {"": 0}).values()) < 2) or re.search(r"\b(?:type|kind|form)s?\b", ident))
                and (not re.search("\\d", ident))
                and (not FIRST_PICKS.search(ident))
                and (not PURCHASE_TERMS.search(ident))
            )

        ranked = sorted(
            (n for n in coverage if n and varies(n)),
            key=lambda n: (not re.search("\\b(?:type|kind|form)s?\\b", n), -coverage[n], n),
        )
        asked = [self.name_words_set(p) for p in self.asked_attrs]
        for ident in ranked:
            if GROUP_NAMES.search(ident) or self.already_covered(ident):
                continue
            if any((len(self.name_words_set(ident) & toks) >= 2 for toks in asked)):
                continue
            if type_only and not re.search(r"\b(?:type|kind|form)s?\b", ident):
                continue
            return ident
        return ""

    @staticmethod
    def singular_name(prop: str) -> str:
        words = str(prop).split()
        if words and not re.search(r"(?:series|species|ss|us|is|as|ns)$", words[-1]):
            words[-1] = root_word(words[-1])
        return " ".join(words)

    def question_text(self) -> str:
        if self.reask_attr:
            prop, self.reask_attr = (self.reask_attr, "")
        else:
            prop = next(iter(self.ask_candidates()), "")
        if not prop:
            return ""
        if prop not in self.asked_attrs:
            self.asked_attrs.append(prop)
        self.need_questions += 1
        return f"Which {self.singular_name(prop)} do you require?"

    @staticmethod
    def to_shopper(text: str) -> str:
        swaps = (
            (r"\bI'm\b", "you're"), (r"\bI've\b", "you've"), (r"\bI'll\b", "you'll"), (r"\bI'd\b", "you'd"),
            (r"\bI am\b", "you are"), (r"\bI\b", "you"), (r"\b[Mm]e\b", "you"), (r"\bmy\b", "your"), (r"\bMy\b", "Your"),
            (r"\bmine\b", "yours"), (r"\bmyself\b", "yourself"), (r"\bwe\b", "you"), (r"\bWe\b", "You"), (r"\bus\b", "you"),
            (r"\bour\b", "your"), (r"\bOur\b", "Your"), (r"\bours\b", "yours"),
            (r"\bthe shopper's\b", "your"), (r"\bthe shopper\b", "you"),
        )
        for pattern, said in swaps:
            text = re.sub(pattern, said, text)
        return text

    @staticmethod
    def purpose_words(text: str) -> set[str]:
        long_words = {root_word(w) for w in lexemes(text) if len(w) >= 4}
        return long_words or {w for w in lexemes(text) if len(w) == 3}

    def add_read_purposes(self) -> None:
        if len(self.brief_uses) >= 2 or self.said_purposes is not None:
            return
        self.opening_guess()

        def alike(one: str, other: str) -> bool:
            a, b = self.purpose_words(one), self.purpose_words(other)
            return bool(a and b) and len(a & b) >= 0.6 * min(len(a), len(b))

        for order, given in enumerate(self.guess_reasons):
            for reason in given:
                agreed = sum(1 for n, other in enumerate(self.guess_reasons) if n != order and any(alike(reason, x) for x in other))
                if agreed >= 1 and not RANKING_TALK.search(reason) and not any(alike(reason, use) for use in self.brief_uses):
                    self.brief_uses.append(reason)
                    self.log_note(f"a purpose two readers give and the brief missed: {reason!r}")
        self.brief_uses = self.brief_uses[:3]

    def purpose_phrases(self) -> list[tuple[str, str]]:
        if self.said_purposes is not None:
            return self.said_purposes
        ask_text = " ".join(self.store.ask_text.replace("’", "'").split())
        low_text = ask_text.lower()
        request_words = self.purpose_words(ask_text)
        out: list[tuple[str, str]] = []
        for use in self.brief_uses[:3]:
            phrase = " ".join(str(use).replace("’", "'").split()).strip(" .;:!?\"'")
            wanted = self.purpose_words(phrase)
            if not wanted or RANKING_TALK.search(phrase) or len(wanted & request_words) < 0.7 * len(wanted):
                continue                      # what decides between items, or not the shopper's own words
            link_word = "for"
            lead = re.match(r"^(?:(?:it|this|that|i)\b[^,]{0,30}?\b)?(for|as|to|on|from)\s+", phrase, re.I)
            if lead:
                link_word, phrase = ("to" if lead.group(1).lower() == "to" else "for"), phrase[lead.end():]
            if re.match(r"^(?:you|i)\s+(?:want|need|would|prefer)|^(?:i'd|you'd)\b", phrase, re.I):
                continue                      # a wish for the item, not a purpose
            at = low_text.find(" ".join(phrase.lower().split()[:3]))
            before = low_text[:at].split()[-1:] if at > 0 else []
            if at >= 0:
                phrase = ask_text[at : at + 1] + phrase[1:]     # its first letter as the request writes it
            if before and before[0] == "to":
                link_word = "to"
            if link_word == "for" and before and before[0] in ("a", "an", "the", "my", "our"):
                phrase = f"{before[0]} {phrase}"   # the article the request has before it
            head = phrase.split(",")[0].strip()
            if len(head.split()) >= 3:
                phrase = head
            words = phrase.split()[:16]
            if len(words) < 2:
                continue
            said = self.to_shopper(" ".join(words))
            mine = self.purpose_words(said)
            if any(len(mine & self.purpose_words(p)) >= 0.6 * max(1, min(len(mine), len(self.purpose_words(p)))) for _, p in out):
                continue                      # the same purpose said twice
            out.append(("meant to" if link_word == "to" else "for", said))
        self.said_purposes = out[:PURPOSES_ASKED]
        return self.said_purposes

    def item_named(self) -> str:
        item = self.item_phrase
        plain = bool(item) and len(item.split()) <= 3 and not re.search(r"[&/,+]|\b(?:and|or)\b", item, re.I)
        return f"the {item}" if plain else "it"

    def purpose_question(self, index: int, again: bool = False) -> str:
        link_word, phrase = self.purpose_phrases()[index]
        if again:
            return f"Since it's {link_word} {phrase}, what is the one other thing {self.item_named()} must have?"
        return f"Since it's {link_word} {phrase}, what must {self.item_named()} have for that use?"

    def open_purpose(self) -> tuple[int, bool] | None:
        count = len(self.purpose_phrases())
        for i in range(count):
            if i not in self.purposes_met and i not in self.purposes_asked:
                return i, False
        for i in sorted(self.purpose_again | self.purposes_gave):
            if i < count and i not in self.purposes_twice and i not in self.purposes_dead:
                return i, True                # a purpose whose question brought a detail out gives the next one too
        return None

    def next_need_question(self) -> tuple[str, str]:
        if not self.reask_attr:
            nxt = self.open_purpose()
            if nxt is not None:
                index, again = nxt
                (self.purposes_twice if again else self.purposes_asked).add(index)
                self.purpose_again.discard(index)
                self.purpose_now = index
                return "purpose", self.purpose_question(index, again)
        self.purpose_now = None
        ask_line = self.question_text()
        if ask_line:
            self.property_asks += 1
        return ("attribute", ask_line) if ask_line else ("", "")

    def chosen_side(self) -> bool:
        return any(wd.get("chosen") for wd in self.reqs)

    def still_asking(self) -> bool:
        link = self.store
        found = self.needs_found_n
        if found >= self.needs_expected_n():
            return False
        if found and self.first_found_q is not None and self.questions_out >= self.first_found_q + MORE_AFTER_FIND:
            return False
        purpose_left = self.open_purpose() is not None
        if self.chosen_side() and not found and not purpose_left and self.property_asks >= AFTER_PAIR_ATTRS:
            return False
        if found and not purpose_left:
            room = AFTER_FIND_TWO if len(self.purpose_phrases()) >= 2 else AFTER_FIND
            if self.property_asks - self.attrs_at_find >= room:
                return False
        if self.questions_out >= ASK_LIMIT or link.ended_flag or not link.has_tool("message"):
            return False
        if link.turns_free() <= ASK_TURN_FLOOR or self.over_time():
            return False
        return not any(k != self.decoy_ref for k in link.basket)

    def after_reply(self, kind: str, found_before: int, missed: bool, silent: bool = False) -> None:
        phrases = self.purpose_phrases()
        asked_for = self.purpose_now if kind == "purpose" else None
        if self.wish_settled:
            self.wish_settled = False
            if asked_for is not None and self.needs_found_n == found_before and ("purpose", asked_for) not in self.repeat_done:
                self.repeat_done.add(("purpose", asked_for))
                self.purposes_asked.discard(asked_for)   # the answer chose between the two wishes instead
                return
        if self.needs_found_n > found_before:
            if self.first_found_q is None:
                self.first_found_q = self.questions_out
                self.attrs_at_find = self.property_asks
            if asked_for is not None:
                self.purposes_gave.add(asked_for)
            heard = self.purpose_words(" ".join(self.found_lines[found_before:]))
            shared = {i: len(self.purpose_words(p) & heard) for i, (_, p) in enumerate(phrases) if i not in self.purposes_met}
            best = max(shared, key=lambda i: (shared[i], i == asked_for), default=None)
            if (
                asked_for is not None and best is not None and best != asked_for
                and shared[best] >= 0.6 * max(1, len(self.purpose_words(phrases[best][1])))
                and shared.get(asked_for, 0) < 0.6 * max(1, len(self.purpose_words(phrases[asked_for][1])))
                and asked_for not in self.purposes_twice
            ):
                # the answer serves another purpose: that one is met, the one asked about is asked again
                self.purposes_met.add(best)
                self.purpose_again.add(asked_for)
                self.expect_two = True
            elif best is not None and shared[best] >= 1:
                self.purposes_met.add(best)
            elif asked_for is not None:
                self.purposes_met.add(asked_for)
        elif kind == "purpose" and not missed:
            self.purpose_empty = True
            if asked_for is not None:
                self.purposes_dead.add(asked_for)
        if missed and asked_for is not None and ("purpose", asked_for) not in self.repeat_done:
            self.repeat_done.add(("purpose", asked_for))
            self.purposes_asked.discard(asked_for)   # a change or a new limit came instead of an answer

    def plainness(self, ref: tuple[str, str], reqs: list[dict]) -> int:
        return sum(self.plain_level(ref, wd) + self.exact_score(ref, wd) for wd in reqs)

    def decoy_ok(self, ref: tuple[str, str], firm: bool = True, floor: int = 0) -> bool:
        link = self.store
        doc = link.read_pages.get(ref[0]) or {}
        ruling = self.rulings.get(ref) or {}
        stated = [wd for wd in self.reqs if wd.get("stated")]
        maker_ok = not self.brand_required or set(lexemes(self.brand_required)) <= set(lexemes(doc.get("brand")))
        shelved = self.in_shelf(doc) or not any(self.in_shelf(p) for p in link.read_pages.values())
        return (
            bool(ruling.get("kind_ok"))
            and ref not in self.dropped
            and self.affordable_now(ref)
            and (link.stock_seen.get(ref) or {}).get("in_stock") is not False
            and maker_ok
            and shelved
            and all(self.plain_level(ref, wd, own_words=True) >= (3 if firm else floor or self.level_required(wd)) for wd in stated)
            and (not firm or all(self.req_kind(wd) != "fact" or self.fact_is_value(ref, wd) for wd in stated))
        )

    def fact_is_value(self, ref: tuple[str, str], req: dict) -> bool:
        doc = self.store.read_pages.get(ref[0]) or {}
        for fact_name, val in self.own_facts(doc, req):
            pieces_txt = self.fact_entries(val)
            if len(pieces_txt) == 1 and self.entry_is_value(pieces_txt[0], req, fact_name):
                return True                   # the listing's own fact is the value and nothing more
        return False

    def of_wished_brand(self, ref: tuple[str, str]) -> bool:
        doc = self.store.read_pages.get(ref[0]) or {}
        return bool(self.brand_wished) and set(lexemes(self.brand_wished)) <= set(lexemes(doc.get("brand")))

    def usual_on_shelf(self) -> dict:
        title_counts: dict[str, int] = {}
        for doc in self.store.read_pages.values():
            if self.in_shelf(doc):
                for wd in self.content_words(str(doc.get("title") or "")):
                    title_counts[wd] = title_counts.get(wd, 0) + 1
        return title_counts

    def title_of(self, ref: tuple[str, str]) -> str:
        return flatten_lower((self.store.read_pages.get(ref[0]) or {}).get("title"))

    def place_decoy(self) -> bool:
        link = self.store
        if (
            self.decoy_ref
            or link.basket
            or self.decoy_tries >= DECOY_TRIES
            or link.ended_flag
            or not link.has_tool("add_to_cart")
            or self.pair_ask
            or self.chosen_side()
            or link.turns_free() < DECOY_TURN_FLOOR
            or self.over_time()
            or not link.read_pages
        ):
            return False
        self.decoy_tries += 1
        self.grade_candidates()
        stated = [wd for wd in self.reqs if wd.get("stated")]
        fitting = set(self.ranked_fits())
        seen = self.usual_on_shelf()

        def usual(ref: tuple[str, str]) -> float:
            words = self.content_words(self.title_of(ref))
            return round(sum(seen.get(w, 0) for w in words) / max(1, len(words)), 1)

        pool = [k for k in self.rulings if self.of_wished_brand(k) and self.decoy_ok(k, firm=False, floor=1)]
        if not pool and self.brand_wished and self.decoy_tries < 2:
            return False                      # a listing of the brand the shopper would like may still turn up
        pool = pool or [k for k in self.rulings if self.decoy_ok(k)]
        if not pool and self.decoy_tries >= DECOY_PLAIN_FROM:
            pool = [k for k in self.rulings if self.decoy_ok(k, firm=False)]
        if not pool and self.decoy_tries >= DECOY_FIT_FROM:
            fits = [k for k in fitting if k not in self.dropped]
            pool = [k for k in fits if self.keeps_request_text(k)] or fits
        if not pool:
            return False
        # the plainest statement of the request; a listing that is no candidate for the choice takes the store's event
        pool.sort(key=lambda k: (-self.plainness(k, stated), k in fitting, -usual(k), (link.stock_seen.get(k) or {}).get("price") or 0.0))
        self.reply_awaited = False
        self.decoy_ref = pool[0]
        self.log_note(f"decoy cart {self.decoy_ref}: the shopper's changes and the store's events come out now")
        self.cart_with_read(self.decoy_ref, self.next_search_calls(link.per_turn - 2))
        if self.decoy_ref not in link.basket:
            self.dropped.add(self.decoy_ref)
            self.decoy_ref = None
            self.hear_waiting()
            return True
        self.hear_lines()
        return True

    def settle_decoy(self) -> None:
        link = self.store
        for _ in range(3):
            if (
                self.decoy_ref or link.basket or self.decoy_tries >= 3 or link.ended_flag
                or link.turns_free() < DECOY_TURN_FLOOR or self.over_time() or self.pair_ask or self.chosen_side()
                or not link.has_tool("add_to_cart")
            ):
                return
            if self.decoy_tries or not any(self.in_shelf(p) for p in link.read_pages.values()):
                work = self.pages_to_read(link.per_turn * IDS_PER_COMPARE)[: link.per_turn] or self.next_search_calls(link.per_turn)
                if not work:
                    return
                link.send_turn(work)
                self.hear_waiting()
            self.place_decoy()

    def late_stand_in(self) -> None:
        link = self.store
        if link.basket or link.ended_flag or not link.has_tool("add_to_cart") or link.turns_free() < 9 or self.over_time():
            return
        self.grade_candidates()
        self.read_rank_figures(every=True)
        ranked = [k for k in self.ranked_fits() if k not in self.dropped]
        stated = [wd for wd in self.reqs if wd.get("stated")]

        def plainest(pool: list) -> tuple[str, str] | None:
            pool.sort(key=lambda k: (not self.keeps_pair_choice(k), -self.plainness(k, stated), (link.stock_seen.get(k) or {}).get("price") or 0.0))
            return pool[0] if pool else None

        if not ranked:
            # nothing meets everything: the cart still opens with a listing that states the request as first written
            ref = plainest([k for k in self.rulings if self.decoy_ok(k)]) or plainest([k for k in self.rulings if self.decoy_ok(k, firm=False)])
        else:
            best = ranked[0]
            others = [k for k in ranked[1:] if self.title_of(k) != self.title_of(best) and self.decoy_ok(k)]
            ref = others[-1] if others else None      # the fitting listing with least to lose
            if ref is None and not (self.decoy_ok(best) and not self.shopper_moved):
                ref = plainest([k for k in self.rulings if k not in ranked and self.title_of(k) != self.title_of(best) and self.decoy_ok(k)])
            if ref is None:
                spare = [k for k in ranked[1:] if self.title_of(k) != self.title_of(best) and self.keeps_request_text(k)]
                ref = spare[-1] if spare else best
        if ref is None:
            return
        self.decoy_ref = ref
        self.log_note(f"stand-in cart {ref} before the choice")
        self.cart_with_read(ref)
        if ref not in link.basket:
            self.decoy_ref = None
            self.hear_waiting()
            return
        changed, _ = self.hear_lines()
        if changed and link.turns_free() > 6 and not self.over_time():
            self.look_around(4, 2)

    def ask_hidden_needs(self) -> None:
        link = self.store
        held = 0
        while self.still_asking():
            if not any((self.in_shelf(p) for p in link.read_pages.values())) and held < 2:
                reads = self.pages_to_read(link.per_turn * IDS_PER_COMPARE)
                if reads:
                    held += 1
                    link.send_turn(reads[: link.per_turn])
                    self.hear_waiting()
                    continue
            if self.place_decoy():
                continue
            kind, ask_line = self.next_need_question()
            if not ask_line:
                break
            asked = (self.asked_attrs[-1] if self.asked_attrs else "") if kind == "attribute" else ""
            bundle = [env_act("message", content=ask_line[:300])] + self.pages_to_read(2 * IDS_PER_COMPARE)
            bundle += self.next_search_calls(link.per_turn - len(bundle))
            self.reply_awaited = True
            self.messages_asked += 1
            self.questions_out += 1
            found_before = self.needs_found_n
            cap_before, changes_before, market_before = self.spend_limit, len(self.changes_unanswered), self.market_lines
            self.latest_found = []
            link.send_turn(bundle)
            silent = not link.unheard
            plain_no = any(re.search(r"\bcategor", said, re.I) for said in link.unheard)   # the shopper's 'nothing to add'
            self.hear_lines()
            self.log_note(f"question {self.questions_out}: {ask_line!r} -> found {self.needs_found_n}/{self.needs_expected_n()}")
            missed = self.needs_found_n == found_before and not plain_no and (
                self.spend_limit != cap_before or len(self.changes_unanswered) > changes_before
                or self.market_lines > market_before
            )
            if missed and asked and asked not in self.repeat_done:
                self.repeat_done.add(asked)
                self.reask_attr = asked       # the reply was about something else: the question is still open
            self.after_reply(kind, found_before, missed, silent)
        self.reply_awaited = False
        self.settle_decoy()

    def same_attribute(self, one: str, other: str) -> bool:
        first, second = self.name_words_set(one), self.name_words_set(other)
        core_first, core_second = first - {"type", "kind"}, second - {"type", "kind"}
        if core_first and core_second:
            return bool(core_first & core_second)
        return bool(first & second)


FIGURE_UNIT = re.compile("(\\d+(?:\\.\\d+)?)\\s*([a-z]+)")
LACK_WORDS = re.compile("\\bfree\\b|\\bnon\\b|\\bno\\b|\\bwithout\\b|\\b0\\s*%")


def colour_bits(seg: str) -> set[str]:
    located: set[str] = set()
    for rd in (flatten_lower(seg), tight_units(seg)):
        for piece in re.split("[^\\w.\\-]+|_", rd):
            if not piece:
                continue
            bits = [b for b in piece.split("-") if b]
            if len(bits) > 1 and any((re.search("\\d", b) for b in bits)):
                located |= {root_word(b) for b in bits}
            else:
                located.add(root_word(piece))
    return located


class Grading(Questioning):

    def affordable_now(self, ref: tuple[str, str]) -> bool:
        last_read = self.store.stock_seen.get(ref) or {}
        if last_read.get("in_stock") is False or ref in self.dropped:
            return False
        cost_now = last_read.get("price")
        return self.spend_limit is None or not isinstance(cost_now, float) or cost_now <= self.spend_limit + 1e-09

    def shelf_leaf(self) -> str:
        read_pages = self.store.read_pages
        if self.shelf_leaf_memo[0] == len(read_pages):
            return self.shelf_leaf_memo[1]
        spoken = f" {shelf_key(self.store.ask_text)} {shelf_key(self.named_category)} "
        leaf_counts: dict[str, int] = {}
        for doc in read_pages.values():
            trail = [str(fact_name) for fact_name in doc.get("category_path") or []]
            if trail and shelf_key(trail[-1]) and (f" {shelf_key(trail[-1])} " in spoken):
                leaf_counts[trail[-1]] = leaf_counts.get(trail[-1], 0) + 1
        own = {shelf_key(self.named_category), shelf_key(self.item_phrase)} - {""}
        best = max(
            leaf_counts,
            key=lambda leaf: (len(shelf_key(leaf).split()), shelf_key(leaf) in own, leaf_counts[leaf]),
            default="",
        )
        self.shelf_leaf_memo = (len(read_pages), best)
        return best

    def in_shelf(self, doc: dict) -> bool:
        called = self.shelf_leaf()
        sought = (
            {shelf_key(called)} if called else {shelf_key(self.named_category), shelf_key(self.item_phrase)} - {""}
        )
        return bool(sought & {shelf_key(fact_name) for fact_name in doc.get("category_path") or []})

    def variant_info(self, ref: tuple[str, str]) -> dict:
        for option in (self.store.read_pages.get(ref[0]) or {}).get("variants") or []:
            if str(option.get("sku")) == ref[1]:
                return option
        return {}

    @staticmethod
    def name_words_set(txt: object) -> set[str]:
        toks = {root_word(wd) for wd in lexemes(str(txt).replace("_", " "))} - {"of"}
        return {"color" if wd == "colour" else wd for wd in toks}

    @staticmethod
    def req_kind(req: dict) -> str:
        if re.search("\\d", req["value"]):
            return "amount"
        prop = flatten_lower(req["prop"])
        if re.search("colou?r", prop):
            return "colour"
        return "fact" if "material" not in prop else "material"

    def value_forms(self, req: dict, lenient_mode: bool = False) -> set[str]:
        val = req["value"]
        other = list(req.get("spell") or [])
        similar = list(req.get("syn") or [])
        if req.get("filed"):
            frms = other + similar if lenient_mode else [val]
        elif re.search("\\d", val):
            if re.search("\\d\\s*[a-z]", flatten_lower(val)):
                stated = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]+)", flatten_lower(val))
                other = [
                    fm
                    for fm in other
                    if not re.fullmatch("[\\d.,\\s]+", str(fm).strip())
                    and (not re.search("\\d\\s*[^\\w\\s.,]", str(fm)))
                    and not self.unit_cut_short(fm, stated)
                ]
            frms = similar if lenient_mode else [val] + other
        else:
            close_facts = [
                fm for fm in other if Similarity(None, flatten_lower(fm), flatten_lower(val)).ratio() >= 0.75
            ]
            frms = [fm for fm in other if fm not in close_facts] + similar if lenient_mode else [val] + close_facts
        return {" ".join((root_word(wd) for wd in lexemes(tight_units(fm)))) for fm in frms} - {""}

    @staticmethod
    def unit_cut_short(form: object, stated: re.Match | None) -> bool:
        # the same figure with a unit cut short from the stated one names another unit ('2g' for '2gb')
        said = re.search(r"(\d+(?:\.\d+)?)\s*([a-z]+)", flatten_lower(form))
        return bool(
            stated and said and float(said.group(1)) == float(stated.group(1))
            and said.group(2) != stated.group(2) and stated.group(2).startswith(said.group(2))
        )

    @staticmethod
    def fact_entries(txt: object) -> list[str]:
        segs = re.split("[,;/|]+", flatten_lower(txt))
        return [ent for ent in (" ".join((root_word(wd) for wd in lexemes(tight_units(seg)))) for seg in segs) if ent]

    def memoized(self, ref: tuple, stamp: tuple, build):
        held = self.memo_store.get(ref)
        if held and held[0] == stamp:
            return held[1]
        val = build()
        self.memo_store[ref] = (stamp, val)
        return val

    def words_under_label(self, fact_name: str) -> set[str]:

        def build() -> set[str]:
            return {
                wd
                for doc in self.store.read_pages.values()
                if self.in_shelf(doc)
                for piece_txt in self.fact_entries((doc.get("facts") or {}).get(fact_name) or "")
                for wd in piece_txt.split()
            }

        return self.memoized(("label", fact_name), (len(self.store.read_pages), self.shelf_leaf()), build)

    @staticmethod
    def without_lacks(txt: object) -> str:
        low = re.sub("\\b[\\w%]+(?:[\\s-]+[\\w%]+)?[\\s-]free\\b", " ", flatten_lower(txt))
        return re.sub("(?:\\b0\\s*%|\\bno\\b|\\bnon\\b|\\bwithout\\b)[\\s-]+[\\w-]+", " ", low)

    def forms_under_label(self, req: dict, fact_name: str) -> set[str]:
        called = self.name_words_set(fact_name)
        filed_set = self.words_under_label(fact_name)

        def linking(tok: str) -> bool:
            return (
                2 <= len(tok) <= 3
                and (not re.search("\\d", tok))
                and (tok not in filed_set)
                and (self.without_lacks(f"{tok} x") == f"{tok} x")
            )

        frms = set(self.value_forms(req))
        for frm in list(frms):
            toks = frm.split()
            kept_words = [wd for wd in toks if not any((same_root(wd, n) for n in called)) and (not linking(wd))]
            if (
                kept_words
                and len(kept_words) < len(toks)
                and (not filed_set or any((wd in filed_set for wd in kept_words)))
            ):
                frms.add(" ".join(kept_words))
        return frms

    def entry_is_value(self, piece_txt: str, req: dict, fact_name: str) -> bool:
        called = self.name_words_set(fact_name)
        trimmed = (
            " ".join((wd for wd in piece_txt.split() if not any((same_root(wd, n) for n in called)))) or piece_txt
        )
        frms = self.forms_under_label(req, fact_name)
        return piece_txt in frms or trimmed in frms

    def own_label_sets(self, req: dict) -> tuple[set[str], set[str], set[str]]:
        read_pages = self.store.read_pages
        stamp = (req["prop"], req["value"], len(req.get("spell") or []), len(read_pages), self.shelf_leaf())

        def build() -> tuple[set[str], set[str], set[str]]:
            sought = self.name_words_set(req["prop"])
            sort_of = self.req_kind(req)
            shelf = [p for p in read_pages.values() if self.in_shelf(p)] or list(read_pages.values())
            spoken = " ".join(lexemes(req["prop"]))
            level_tiers: list[set[str]] = [set(), set(), set(), set(), set()]
            shelf_roots = {root_word(wd) for wd in self.item_word_list()}
            unit_set = {mo.group(2) for frm in self.value_forms(req) for mo in FIGURE_UNIT.finditer(frm)}
            exact_hits: dict[str, int] = {}
            for doc in shelf:
                for fact_name, val in (doc.get("facts") or {}).items():
                    fact_name = str(fact_name)
                    got_words = self.name_words_set(fact_name)
                    if sought and got_words:
                        if sought < got_words and got_words - sought <= shelf_roots:
                            level_tiers[0].add(fact_name)      # '<shelf words> <attribute>': the shelf's own label for it
                        elif " ".join(lexemes(fact_name.replace("_", " "))) == spoken:
                            level_tiers[1].add(fact_name)
                        elif got_words == sought:
                            level_tiers[2].add(fact_name)
                        elif got_words <= sought or sought <= got_words:
                            level_tiers[3 if got_words ^ sought <= unit_set else 4].add(fact_name)
                    pieces_txt = self.fact_entries(val)
                    if len(pieces_txt) == 1 and self.entry_is_value(pieces_txt[0], req, fact_name):
                        exact_hits[fact_name] = exact_hits.get(fact_name, 0) + 1
            linked = next((level_tier for level_tier in level_tiers if level_tier), set())
            top = max(exact_hits.values(), default=0)
            deduced: set[str] = set()
            if sort_of != "colour" and (not any((exact_hits.get(fact_name) for fact_name in linked))):
                deduced = {fact_name for fact_name, n in exact_hits.items() if n >= 2 and 3 * n >= top}
            vocabulary: set[str] = set()
            for doc in shelf:
                for fact_name, val in (doc.get("facts") or {}).items():
                    if str(fact_name) in linked or str(fact_name) in deduced:
                        vocabulary |= {ent for ent in self.fact_entries(val) if " " not in ent}
                if sort_of == "colour":
                    for option in doc.get("variants") or []:
                        vocabulary |= {
                            ent for ent in self.fact_entries(option.get("options")) if re.fullmatch("[a-z]{3,}", ent)
                        }
            return (linked, deduced, vocabulary)

        return self.memoized(("own", req["prop"]), stamp, build)

    def own_facts(self, doc: dict, req: dict) -> list[tuple[str, str]]:
        linked, deduced, _ = self.own_label_sets(req)
        return [
            (str(fact_name), str(val))
            for fact_name, val in (doc.get("facts") or {}).items()
            if str(fact_name) in linked or str(fact_name) in deduced
        ]

    def own_fact_words(self, req: dict) -> set[str]:

        def build() -> set[str]:
            linked, deduced, _ = self.own_label_sets(req)
            toks: set[str] = set()
            for doc in self.store.read_pages.values():
                if self.in_shelf(doc):
                    for fact_name, val in (doc.get("facts") or {}).items():
                        if str(fact_name) in linked or str(fact_name) in deduced:
                            for piece_txt in self.fact_entries(val):
                                toks |= set(piece_txt.split())
            return toks

        return self.memoized(
            ("vocab", req["prop"], req["value"]), (len(self.store.read_pages), self.shelf_leaf()), build
        )

    def entry_within_value(self, piece_txt: str, req: dict, fact_name: str) -> bool:
        got_words = piece_txt.split()
        if not got_words:
            return False
        vocabulary = self.own_fact_words(req)
        for frm in self.forms_under_label(req, fact_name):
            toks = frm.split()
            if not all((any((same_root(hw, wd) for wd in toks)) for hw in got_words)):
                continue
            extra = [wd for wd in toks if not any((same_root(wd, hw) for hw in got_words))]
            if extra and (
                not any(
                    (
                        len(wd) > 3 or re.search("\\d", wd) or any((same_root(wd, vt) for vt in vocabulary))
                        for wd in extra
                    )
                )
            ):
                return True
        return False

    def entry_states(self, piece_txt: str, req: dict, fact_name: str, lenient_mode: bool = False) -> bool:
        frms = self.value_forms(req, lenient_mode=True) if lenient_mode else self.forms_under_label(req, fact_name)
        got_words = piece_txt.split()
        if any((all((any((same_root(n, hw) for hw in got_words)) for n in frm.split())) for frm in frms)):
            return True
        return not lenient_mode and self.entry_within_value(piece_txt, req, fact_name)

    def exactly_states(self, txt: object, req: dict, fact_name: str = "") -> bool:
        pieces_txt = set()
        for seg in re.split("[,;/|]+", flatten_lower(txt)):
            for rd in (flatten_lower(seg), tight_units(seg)):
                pieces_txt.add(" ".join((root_word(wd) for wd in lexemes(rd))))
        if not fact_name:
            return bool(pieces_txt & self.value_forms(req))
        called = self.name_words_set(fact_name)
        for piece_txt in list(pieces_txt):
            kept_words = [wd for wd in piece_txt.split() if not any((same_root(wd, n) for n in called))]
            if kept_words:
                pieces_txt.add(" ".join(kept_words))
        return bool(pieces_txt & self.forms_under_label(req, fact_name))

    def text_states(self, txt: object, req: dict, lenient_mode: bool = False) -> bool:
        frms = self.value_forms(req, lenient_mode)
        if not any((LACK_WORDS.search(frm) for frm in frms)):
            txt = self.without_lacks(txt)
        got_words = {root_word(wd) for rd in (flatten_lower(txt), tight_units(txt)) for wd in lexemes(rd)}
        return any((all((any((same_root(n, hw) for hw in got_words)) for n in frm.split())) for frm in frms))

    def amounts_of(self, txt: object, req: dict) -> tuple[set, set]:
        sought = {
            (float(mo.group(1)), unit_root(mo.group(1), mo.group(2)))
            for frm in self.value_forms(req)
            for mo in FIGURE_UNIT.finditer(frm)
        }
        unit_set = {unit_txt for _, unit_txt in sought}
        located = {
            (float(mo.group(1)), unit_root(mo.group(1), mo.group(2)))
            for mo in re.finditer("(?<![\\w.])(\\d+(?:\\.\\d+)?)\\s*([a-z]+)", flatten_lower(txt))
        }
        located = {duo for duo in located if duo[1] in unit_set}
        return (located & sought, located - sought)

    def amount_in_a_row(self, txt: object, req: dict) -> bool:
        rows = [[root_word(wd) for wd in lexemes(rd)] for rd in (flatten_lower(txt), tight_units(txt))]
        for frm in self.value_forms(req):
            toks = frm.split()
            for row in rows:
                if toks and any(
                    all(same_root(tok, row[at + step]) for step, tok in enumerate(toks))
                    for at in range(len(row) - len(toks) + 1)
                ):
                    return True
        return False

    def has_wanted_amount(self, txt: object, req: dict) -> bool:
        if self.amounts_of(txt, req)[0]:
            return True
        if re.search("\\d", req["value"]):
            return self.amount_in_a_row(txt, req)   # '50*23mm (1 piece)' does not say 50 pieces
        return self.text_states(txt, req)

    def has_other_amount(self, txt: object, req: dict) -> bool:
        existing, alt_forms = self.amounts_of(txt, req)
        return bool(alt_forms) and (not existing) and (not self.has_wanted_amount(txt, req))

    def label_has_unit(self, fact_name: str, req: dict) -> bool:
        unit_set = {mo.group(2) for frm in self.value_forms(req) for mo in FIGURE_UNIT.finditer(frm)}
        return bool(unit_set & self.name_words_set(fact_name))

    @staticmethod
    def option_plain_words(toks: set[str]) -> set[str]:
        return {wd for wd in toks if re.fullmatch("[a-z]{3,}", wd) and (not re.fullmatch("x*[sml]|x+l", wd))}

    def plain_level(self, ref: tuple[str, str], req: dict, own_words: bool = False) -> int:
        doc = self.store.read_pages.get(ref[0]) or {}
        opts = str(self.variant_info(ref).get("options") or "")
        heading = doc.get("title")
        sort_of = self.req_kind(req)
        linked, deduced, vocabulary = self.own_label_sets(req)
        own = [(fact_name, self.fact_entries(val)) for fact_name, val in self.own_facts(doc, req)]
        singles = [(fact_name, pieces_txt[0]) for fact_name, pieces_txt in own if len(pieces_txt) == 1]
        in_list_fact = any(
            (
                self.entry_states(piece_txt, req, fact_name)
                for fact_name, pieces_txt in own
                if len(pieces_txt) > 1
                for piece_txt in pieces_txt
            )
        )
        stating = [fact_name for fact_name, piece_txt in singles if self.entry_states(piece_txt, req, fact_name)]
        disagreeing = [fact_name for fact_name, piece_txt in singles if fact_name not in stating]
        value_words = {wd for frm in self.value_forms(req) for wd in frm.split()}
        any_fact_says = any((self.text_states(val, req) for val in (doc.get("facts") or {}).values()))
        if sort_of == "amount":
            existing, alt_forms = self.amounts_of(opts, req)
            if self.has_other_amount(opts, req) or (existing and alt_forms):
                return -1
            if self.has_wanted_amount(opts, req):
                return 3
            in_title, other_in_title = self.amounts_of(heading, req)
            if other_in_title and (not in_title):
                return -1
            fig = re.search("\\d+(?:\\.\\d+)?", req["value"])
            bare = [
                fact_name
                for fact_name, piece_txt in singles
                if fig
                and re.fullmatch("\\d+(?:\\.\\d+)?", piece_txt)
                and (float(piece_txt) == float(fig.group(0)))
            ]
            if any((self.label_has_unit(fact_name, req) for fact_name in bare)) or any(
                (self.has_wanted_amount(piece_txt, req) for _, piece_txt in singles)
            ):
                return 3
            if bare:
                return 2 if self.has_wanted_amount(heading, req) else 1
            if any((re.search("\\d", piece_txt) for _, piece_txt in singles)):
                return -1
            if fig and any(
                self.label_has_unit(str(fact_name), req)
                and re.fullmatch("\\d+(?:\\.\\d+)?", flatten_lower(val).strip())
                and float(flatten_lower(val)) == float(fig.group(0))
                for fact_name, val in (doc.get("facts") or {}).items()
            ):
                return 2
            if self.has_wanted_amount(heading, req):
                return 1 if self.amounts_of(heading, req)[1] else 2
            return (
                1
                if in_list_fact
                or any((self.has_wanted_amount(val, req) for val in (doc.get("facts") or {}).values()))
                else 0
            )
        naming = None if own_words else self.naming_marks.get((ref, req["prop"], req["value"]))
        if sort_of == "colour":
            sought = self.value_forms(req)
            segs = [colour_bits(seg) for seg in re.split("[/,;|]+", opts)]
            hits = [seg for seg in segs if any((set(frm.split()) <= seg for frm in sought))]
            if hits:
                alone = any(not (seg & vocabulary) - value_words for seg in hits)
                if not alone and naming == "among":
                    return -1
                return 3 if alone and naming not in ("among", "other") else 1
            if any((frm.replace(" ", "") in flatten_lower(opts).replace(" ", "") for frm in sought)):
                return -1
            if stating:
                if disagreeing or opts.strip():
                    return 1
                return 3
            loose_hit = any(
                (set(frm.split()) <= seg for frm in self.value_forms(req, lenient_mode=True) for seg in segs)
            )
            if loose_hit or naming in ("only", "among"):
                return 1
            if opts.strip():
                return -1
            return 1 if in_list_fact else 0
        if sort_of == "material":
            if stating:
                return 3
            if naming in ("among", "other"):
                return 1 if naming == "among" else -1
            other_material = bool(singles)
            if naming == "only":
                return 1 if other_material else 2
            if self.text_states(heading, req):
                beside = ({root_word(wd) for wd in lexemes(heading)} & vocabulary) - value_words
                return 1 if beside or other_material else 2
            if in_list_fact or self.text_states(heading, req, lenient_mode=True):
                return 1
            if other_material:
                return -1
            return 1 if any_fact_says else 0
        named = [(fact_name, pieces_txt) for fact_name, pieces_txt in own if fact_name in linked]
        if named:
            said = [
                fact_name for fact_name, pieces_txt in named
                if len(pieces_txt) == 1 and self.entry_states(pieces_txt[0], req, fact_name)
            ]
            if said:
                clash = (root_set(opts) & vocabulary) - value_words
                if clash and not self.text_states(opts, req):
                    return 1                  # the variant's own name gives another value of this attribute
                return 3 if len(said) == len(named) else 1
            listed = any(
                len(pieces_txt) > 1 and any(self.entry_states(piece_txt, req, fact_name) for piece_txt in pieces_txt)
                for fact_name, pieces_txt in named
            )
            alike = any(
                len(pieces_txt) == 1 and self.entry_states(pieces_txt[0], req, fact_name, lenient_mode=True)
                for fact_name, pieces_txt in named
            )
            elsewhere = any(
                str(fact_name) not in linked and len(self.fact_entries(val)) == 1
                and self.entry_is_value(self.fact_entries(val)[0], req, str(fact_name))
                for fact_name, val in (doc.get("facts") or {}).items()
            )
            return 1 if listed or alike or elsewhere else -1
        if stating:
            return 1 if disagreeing else 3
        if in_list_fact or any(
            (self.entry_states(piece_txt, req, fact_name, lenient_mode=True) for fact_name, piece_txt in singles)
        ):
            return 1
        if not own and any(
            len(self.fact_entries(val)) == 1 and self.entry_is_value(self.fact_entries(val)[0], req, str(fact_name))
            for fact_name, val in (doc.get("facts") or {}).items()
        ):
            return 1
        if own or linked:
            return -1
        if deduced:
            return 0
        return 1 if any_fact_says or self.text_states(heading, req) else 0

    def exact_score(self, ref: tuple[str, str], req: dict) -> int:
        doc = self.store.read_pages.get(ref[0]) or {}
        score = 0
        for fact_name, val in self.own_facts(doc, req):
            pieces_txt = self.fact_entries(val)
            if len(pieces_txt) == 1:
                if self.entry_is_value(pieces_txt[0], req, fact_name):
                    score += 1
                elif not self.entry_states(pieces_txt[0], req, fact_name):
                    score -= 1
        return score + int(self.exactly_states(self.variant_info(ref).get("options") or "", req))

    def level_required(self, req: dict) -> int:
        return 2 if self.req_kind(req) in ("amount", "material") else 3


class Choosing(Grading):

    def row_context_sets(self) -> tuple[set[str], set[str]]:
        headings = set(self.titles_fitting()) | self.variant_probes_done
        stems = {k[1].rsplit("-", 1)[0] for k in self.rulings if "-" in k[1] and self.words_fit(k)}
        return (headings, stems)

    def row_score_of(self, ref: tuple[str, str], context: tuple[set[str], set[str]]) -> float:
        hit_row = self.store.found_rows.get(ref) or {}
        heading = root_set(hit_row.get("title"))
        score = 4.0 if ref in self.store.filtered_rows else 0.0
        for req in self.reqs:
            need = root_set(req["value"])
            alt_forms = [root_set(fm) for fm in list(req.get("spell") or []) + list(req.get("syn") or [])]
            if need and need <= heading or any((o and o <= heading for o in alt_forms)):
                score += 3
            elif need & heading:
                score += 1
        score += sum((1 for wd in {root_word(wd) for wd in self.item_word_list()} if wd in heading))
        headings, stems = context
        if flatten_lower(hit_row.get("title")) in headings:
            score += 20
        elif "-" in ref[1] and ref[1].rsplit("-", 1)[0] in stems:
            score += 20
        return score + (50 if self.request_skus and ref in self.request_skus else 0)

    def pages_to_read(self, limit: int, allowance: int | None = None) -> list[dict]:
        link = self.store
        allowance = self.read_allow if allowance is None else allowance
        def reachable(k: tuple[str, str]) -> bool:
            if self.affordable_now(k):
                return True
            info = link.stock_seen.get(k) or {}
            cost = info.get("price")
            return (
                self.spend_limit is not None and info.get("in_stock") is not False and isinstance(cost, float)
                and self.spend_limit < cost <= OPEN_REACH * self.spend_limit
            )

        unread = [
            k for k in link.rows_in_order
            if k[0] not in link.read_pages and self.read_tries.get(k[0], 0) < 2 and reachable(k)
        ]
        if self.request_skus:
            unread = [k for k in self.request_skus if k[0] not in link.read_pages] + [
                k for k in unread if k in self.request_skus
            ]
        context = self.row_context_sets()
        ranked = sorted(
            dict.fromkeys(unread), key=lambda k: -(self.row_score_of(k, context) - (0 if self.affordable_now(k) else 2))
        )
        ids = list(dict.fromkeys((k[0] for k in ranked)))[: max(0, min(limit, allowance - len(link.read_pages)))]
        for prod_id in ids:
            self.read_tries[prod_id] = self.read_tries.get(prod_id, 0) + 1
        if link.has_tool("compare"):
            return [
                env_act("compare", product_ids=ids[i : i + IDS_PER_COMPARE])
                for i in range(0, len(ids), IDS_PER_COMPARE)
            ]
        return [env_act("view", product_id=prod_id) for prod_id in ids]

    def candidate_refs(self) -> list[tuple[str, str]]:
        link = self.store
        any_shelf = any((self.in_shelf(p) for p in link.read_pages.values()))
        to_score = []
        for prod_id, doc in link.read_pages.items():
            if any_shelf and (not self.in_shelf(doc)):
                continue
            for option in doc.get("variants") or []:
                ref = (prod_id, str(option.get("sku")))
                if self.affordable_now(ref) and (not self.request_skus or ref in self.request_skus):
                    to_score.append(ref)
        return to_score

    def listing_card(self, number: int, ref: tuple[str, str]) -> str:
        doc = self.store.read_pages[ref[0]]
        names = {wd for req in self.reqs for wd in self.name_words_set(req["prop"])}
        distinct_vals = {wd for req in self.reqs for wd in root_set(req["value"])}
        close_facts, other_facts = ([], [])
        for fact_name, val in (doc.get("facts") or {}).items():
            if self.name_words_set(fact_name) & names or root_set(val) & distinct_vals:
                close_facts.append(f"{fact_name} = {str(val)[:110]}")
            else:
                other_facts.append(f"{fact_name} = {str(val)[:36]}")
        facts = "; ".join(close_facts + other_facts[:5])
        for position, req in enumerate(self.reqs, start=1):
            own = self.own_facts(doc, req)
            if own:
                facts += f" || own fact for requirement {position}: " + "; ".join(
                    (f"{fact_name} = {val[:110]}" for fact_name, val in own)
                )
        cost_now = (self.store.stock_seen.get(ref) or {}).get("price")
        trail = " > ".join((str(it) for it in (doc.get("category_path") or [])[-2:]))
        return f"[{number}] title: {str(doc.get('title') or '')[:150]} | options: {self.variant_info(ref).get('options') or '-'} | price: {cost_now} | category: {trail} | brand: {doc.get('brand') or '-'} | facts: {facts or '-'}"

    def grade_candidates(self) -> None:
        shelf_words = set(shelf_key(self.shelf_leaf()).split())
        self.reqs = [
            wd
            for wd in self.reqs
            if not (wd.get("stated") and shelf_words and (set(shelf_key(wd["value"]).split()) <= shelf_words))
            or self.filed_on_shelf(wd)
        ]
        self.naming_check()
        stamp = json.dumps([self.reqs, self.decider, self.lenient_mode, len(self.naming_marks)], sort_keys=True)
        if stamp != self.rulings_key:
            self.rulings, self.rulings_key = ({}, stamp)
        for ref in self.candidate_refs():
            if ref not in self.rulings:
                self.rulings[ref] = {
                    "kind_ok": self.in_shelf(self.store.read_pages[ref[0]]),
                    "met": [self.plain_level(ref, wd) >= 1 for wd in self.reqs],
                    "by": "words",
                }

    def labels_silent(self, ref: tuple[str, str], req: dict) -> int:
        def build() -> set[str]:
            return {
                str(fact_name)
                for doc in self.store.read_pages.values() if self.in_shelf(doc)
                for fact_name, val in (doc.get("facts") or {}).items()
                if len(self.fact_entries(val)) == 1 and self.entry_is_value(self.fact_entries(val)[0], req, str(fact_name))
            }

        filing = self.memoized(("filing", req["prop"], req["value"]), (len(self.store.read_pages), self.shelf_leaf()), build)
        facts = (self.store.read_pages.get(ref[0]) or {}).get("facts") or {}
        # every label the shelf files this value under should say it here too
        return sum(
            1 for fact_name in filing
            if not (len(self.fact_entries(facts.get(fact_name))) == 1 and self.entry_is_value(self.fact_entries(facts.get(fact_name))[0], req, fact_name))
        )

    def filed_on_shelf(self, req: dict) -> bool:
        linked = self.own_label_sets(req)[0]
        return any(
            self.entry_states(piece_txt, req, str(fact_name))
            for doc in self.store.read_pages.values() if self.in_shelf(doc)
            for fact_name, val in (doc.get("facts") or {}).items() if str(fact_name) in linked
            for piece_txt in self.fact_entries(val)
        )

    def model_check(self) -> None:
        open_keys = [
            k
            for k, vt in self.rulings.items()
            if self.affordable_now(k)
            and vt.get("by") == "words"
            and vt.get("kind_ok")
            and (not self.words_fit(k))
            and (not any((self.plain_level(k, wd) == -1 for wd in self.reqs)))
        ]
        open_keys.sort(key=lambda k: -sum((self.plain_level(k, wd) for wd in self.reqs)))
        per_turn = open_keys[:CARDS_PER_JUDGE]
        if not per_turn or not self.model_ok() or self.cards_sent >= self.judge_allow:
            return
        self.cards_sent += len(per_turn)
        listing = (
            "\n".join((f"{i}. {wd['prop']}: {wd['value']}" for i, wd in enumerate(self.reqs, start=1)))
            or "(nothing beyond the kind of thing)"
        )
        txt = (
            f"Item wanted: {self.item_phrase}\nRequirements, in order:\n{listing}\n"
            + (f"Deciding point between qualifying items: {self.decider}\n" if self.decider else "")
            + "\nListings:\n"
            + "\n".join((self.listing_card(i, ref) for i, ref in enumerate(per_turn, start=1)))
        )
        ans = self.infer.query_model(CHECK_RULES, txt, CHECK_TOOL, 1400) or {}
        ticks = as_list(ans.get("verdicts"))
        for tick in ticks:
            located = re.match("\\s*\\[?(\\d+)\\]?\\W+([KkXx])\\W*([YyNn\\s,]*)", str(tick))
            if not located or not 1 <= int(located.group(1)) <= len(per_turn):
                continue
            ref = per_turn[int(located.group(1)) - 1]
            said = [it in "Yy" for it in re.findall("[YyNn]", located.group(3))]
            said += [False] * (len(self.reqs) - len(said))
            met_flags = [
                self.plain_level(ref, wd) >= 1
                or (said[i] and self.plain_level(ref, wd) != -1 and (not re.search("\\d", wd["value"])))
                for i, wd in enumerate(self.reqs)
            ]
            self.rulings[ref] = {"kind_ok": located.group(2) in "Kk", "met": met_flags, "by": "model"}
        for ref in per_turn:
            if self.rulings[ref].get("by") == "words":
                self.rulings[ref]["by"] = "model"
        for proof in ans.get("evidence") or []:
            if (
                isinstance(proof, dict)
                and isinstance(proof.get("index"), int)
                and (1 <= proof["index"] <= len(per_turn))
            ):
                self.rulings[per_turn[proof["index"] - 1]].update(
                    places=proof.get("places"), quotes=proof.get("quotes"), figure=proof.get("figure")
                )
        self.log_note(f"model check on {len(per_turn)}: " + " | ".join((str(mo)[:12] for mo in ticks))[:260])
        self.queue_searches(as_list(ans.get("searches")))

    def naming_check(self) -> None:
        for req in self.reqs:
            sort_of = self.req_kind(req)
            if sort_of not in ("colour", "material") or not self.model_ok() or self.naming_count >= 10:
                continue
            alt_forms = [wd for wd in self.reqs if wd is not req]
            queued_texts = []
            for ref in self.candidate_refs():
                marker = (ref, req["prop"], req["value"])
                if marker in self.naming_marks or any((self.plain_level(ref, wd) < 1 for wd in alt_forms)):
                    continue
                doc = self.store.read_pages.get(ref[0]) or {}
                opts = str(self.variant_info(ref).get("options") or "").strip()
                heading = str(doc.get("title") or "")[:150]
                if sort_of == "colour":
                    txt = f"variant: {opts}" if opts else f"title: {heading}"
                else:
                    txt = f"title: {heading}" + (f" | variant: {opts}" if opts else "")
                queued_texts.append((marker, txt))
            queued_texts = queued_texts[:12]
            if not queued_texts:
                continue
            self.naming_count += 1
            prompt = (
                f"Item: {self.item_phrase}\nAttribute: {req['prop']}\nWanted value: {req['value']}\n\nTexts:\n"
                + "\n".join((f"[{i}] {txt}" for i, (_, txt) in enumerate(queued_texts, start=1)))
            )
            ticks = as_list((self.infer.query_model(NAMING_RULES, prompt, NAMING_TOOL, 500) or {}).get("marks"))
            numbered = all((re.match("\\s*\\[?\\d+", str(mo)) for mo in ticks))
            if numbered or len(ticks) == len(queued_texts):
                for position, tick in enumerate(ticks, start=1):
                    tok = re.search("\\b(only|among|other|none)\\b", flatten_lower(tick))
                    digits = re.match("\\s*\\[?(\\d+)", str(tick))
                    at = int(digits.group(1)) if numbered else position
                    if tok and 1 <= at <= len(queued_texts):
                        self.naming_marks[queued_texts[at - 1][0]] = tok.group(1)
            for marker, _ in queued_texts:
                self.naming_marks.setdefault(marker, "unread")
            self.log_note(
                f"naming {req['value']}: "
                + " ".join(
                    (f"{i}:{self.naming_marks[marker]}" for i, (marker, _) in enumerate(queued_texts, start=1))
                )[:240]
            )

    def code_proven_fit(self, ref: tuple[str, str]) -> bool:
        return (self.rulings.get(ref) or {}).get("by") == "words" and all(
            (self.plain_level(ref, wd, own_words=True) >= 1 for wd in self.reqs)
        )

    def words_fit(self, ref: tuple[str, str]) -> bool:
        ruling = self.rulings.get(ref) or {}
        met_flags = ruling.get("met") or []
        return bool(ruling.get("kind_ok")) and len(met_flags) >= len(self.reqs) and all(met_flags[: len(self.reqs)])

    def fits_now(self, ref: tuple[str, str]) -> bool:
        if not self.words_fit(ref) or not self.affordable_now(ref):
            return False
        doc = self.store.read_pages.get(ref[0]) or {}
        shelved = self.in_shelf(doc) or not any((self.in_shelf(p) for p in self.store.read_pages.values()))
        maker_ok = not self.brand_required or set(lexemes(self.brand_required)) <= set(lexemes(doc.get("brand")))
        return shelved and maker_ok

    def stated_solid(self, ref: tuple[str, str]) -> bool:
        return (self.rulings.get(ref) or {}).get("by") == "words" and all(
            (self.plain_level(ref, wd) >= 2 for wd in self.reqs)
        )

    def stated_firm(self, ref: tuple[str, str]) -> bool:
        return (self.rulings.get(ref) or {}).get("by") == "words" and all(
            (self.plain_level(ref, wd) >= self.level_required(wd) for wd in self.reqs)
        )

    def rank_is_price(self) -> bool:
        return bool(re.search("\\b(?:cheap|price|cost|expensive|afford)", flatten_lower(self.decider)))

    def ranked_fits(self) -> list[tuple[str, str]]:
        if self.decider_dir in ("higher", "lower"):
            sign = -1.0 if self.decider_dir == "higher" else 1.0
        else:
            sign = -1.0 if re.search("\\b(?:high|larg|big|long|most|max|best)", flatten_lower(self.decider)) else 1.0
        by_price = self.rank_is_price()

        def position(ref: tuple[str, str]) -> tuple:
            ruling = self.rulings[ref]
            cost_now = (self.store.stock_seen.get(ref) or {}).get("price") or 0.0
            fig = cost_now if by_price else ruling.get("figure", self.figure_cache.get((ref, self.decider)))
            known_set = isinstance(fig, (int, float)) and (not isinstance(fig, bool))
            plainness = sum((self.plain_level(ref, wd) + self.exact_score(ref, wd) for wd in self.reqs))
            return (
                sum(1 for wd in self.reqs if not self.said_in_own_words(ref, wd)),
                sum(1 for wd in self.reqs if self.req_kind(wd) == "fact" and self.own_label_sets(wd)[0] and not self.fact_is_value(ref, wd)),
                sum(self.labels_silent(ref, wd) for wd in self.reqs if self.req_kind(wd) == "fact"),
                0 if self.code_proven_fit(ref) else 1,
                0 if self.twin_of_dropped(ref) else 1,
                sum(1 for wd in self.reqs if self.req_kind(wd) == "amount" and self.plain_level(ref, wd) < 2),
                0 if known_set or not self.decider else 1,
                sign * float(fig) if known_set and self.decider else 0.0,
                0 if self.figure_in_options(ref) else 1,
                0 if self.stated_firm(ref) else 1,
                0 if self.stated_solid(ref) else 1,
                -plainness,
                cost_now,
            )

        return sorted((k for k in self.rulings if self.fits_now(k)), key=position)

    def keeps_request_text(self, ref: tuple[str, str]) -> bool:
        stated = [wd for wd in self.reqs if wd.get("stated")]
        return bool((self.rulings.get(ref) or {}).get("kind_ok")) and all(
            (self.plain_level(ref, wd, own_words=True) >= self.level_required(wd) for wd in stated)
        )

    def keeps_pair_choice(self, ref: tuple[str, str]) -> bool:
        met_flags = ((self.rulings.get(ref) or {}).get("met") or [])[: len(self.reqs)]
        ticks = [it for it, wd in zip(met_flags, self.reqs) if wd.get("chosen")]
        return bool(ticks) and all(ticks)

    def read_rank_figures(self, every: bool = False) -> None:
        for _batch in range(4 if every else 1):
            if not self.read_figure_batch(every):
                return

    def read_figure_batch(self, every: bool) -> bool:
        if not self.decider or self.rank_is_price() or (not self.model_ok()) or (self.figure_calls >= 8):
            return False
        qualifying = self.ranked_fits()
        for k in qualifying:
            if "figure" not in self.rulings[k] and (k, self.decider) in self.figure_cache:
                self.rulings[k]["figure"] = self.figure_cache[(k, self.decider)]
        to_score = [k for k in qualifying if "figure" not in self.rulings[k]][:CARDS_PER_JUDGE]
        if len(qualifying) < (1 if every else 2) or not to_score:
            return False
        self.figure_calls = self.figure_calls + 1
        txt = f"Item: {self.item_phrase}\nWhat decides: {self.decider}\n\nListings:\n" + "\n".join(
            (self.listing_card(i, ref) for i, ref in enumerate(to_score, start=1))
        )
        ans = self.infer.query_model(FIGURES_RULES, txt, FIGURES_TOOL, 400) or {}
        if str(ans.get("prefer") or "").lower() in ("higher", "lower"):
            self.decider_dir = str(ans["prefer"]).lower()
        unit_said = flatten_lower(ans.get("unit"))
        if re.fullmatch(r"[a-z]{1,8}", unit_said) and not self.decider_unit:
            self.decider_unit = unit_said
        pieces_txt = as_list(ans.get("per_listing"), "[\\n;]+")
        numbered = [re.match(r"\s*\[?(\d+)\]?(?:[\s:)\-]+|\.\s+)(\S.*)$", str(p)) for p in pieces_txt]
        by_number = bool(pieces_txt) and all(numbered)
        if by_number and sorted(int(m.group(1)) for m in numbered) != list(range(1, len(numbered) + 1)):
            by_number = False             # the leading numbers are no listing numbers: the entries stand in order
        for position, piece_txt in enumerate(pieces_txt, start=1):
            m = numbered[position - 1] if by_number else None
            at, rest = (int(m.group(1)), m.group(2)) if m else (position, str(piece_txt))
            figure = re.search(r"-?[0-9]+(?:\.[0-9]+)?", rest)
            if figure and 1 <= at <= len(to_score):
                self.rulings[to_score[at - 1]]["figure"] = float(figure.group(0))
        for ref in to_score:
            self.rulings[ref].setdefault("figure", None)
            settled, own_figure = self.page_figure(ref)
            if settled:
                self.rulings[ref]["figure"] = own_figure     # the page's own number in the unit counts
            self.figure_cache[(ref, self.decider)] = self.rulings[ref]["figure"]
        self.log_note(f"figures for {self.decider!r}: " + " ".join((str(ent)[:14] for ent in pieces_txt))[:200])
        return True

    @staticmethod
    def unit_scale(written: str, unit: str) -> float | None:
        a, b = written.rstrip("s") if len(written) > 2 else written, unit.rstrip("s") if len(unit) > 2 else unit
        if a == b:
            return 1.0
        for long, short in ((a, b), (b, a)):
            if len(long) == len(short) + 1 and long.endswith(short) and len(short) <= 3:
                lead = long[0]
                step = 1000.0 if lead == "k" else 0.01 if lead == "c" else 0.001 if lead == "m" else 0.0
                if step:
                    return step if long == a else 1.0 / step
        return None

    def page_figure(self, ref: tuple[str, str]) -> tuple[bool, float | None]:
        unit = self.decider_unit
        if not unit:
            return False, None
        doc = self.store.read_pages.get(ref[0]) or {}
        for text in (self.variant_info(ref).get("options"), doc.get("title")):
            found: set[float] = set()
            written = flatten_lower(text)
            for m in re.finditer(r"(?<![a-z0-9.,])(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", written):
                scale = self.unit_scale(m.group(2), unit)
                if scale:
                    times = re.match(r"\s*[*x×]\s*(\d+)(?!\s*[a-z\d.])", written[m.end():])
                    found.add(round(float(m.group(1)) * scale * (int(times.group(1)) if times else 1), 6))
            if found:
                # the first place that writes such an amount settles it; several different ones settle none
                return True, (next(iter(found)) if len(found) == 1 else None)
        return False, None

    def worse_by_decider(self, ref: tuple[str, str], than: tuple[str, str]) -> bool:
        if not self.decider or self.rank_is_price():
            return False
        mine = (self.rulings.get(ref) or {}).get("figure", self.figure_cache.get((ref, self.decider)))
        theirs = (self.rulings.get(than) or {}).get("figure", self.figure_cache.get((than, self.decider)))
        if not isinstance(theirs, (int, float)) or isinstance(theirs, bool):
            return False
        if not isinstance(mine, (int, float)) or isinstance(mine, bool):
            return True
        return mine < theirs if self.higher_better() else mine > theirs

    def figure_in_options(self, ref: tuple[str, str]) -> bool:
        unit = self.decider_unit
        opts = flatten_lower(self.variant_info(ref).get("options"))
        return bool(unit) and any(
            self.unit_scale(m.group(2), unit) for m in re.finditer(r"(\d+(?:\.\d+)?)\s?([a-z]+)(?![a-z0-9])", opts)
        )

    def said_in_own_words(self, ref: tuple[str, str], req: dict) -> bool:
        if re.search(r"\d", req["value"]):
            return True                   # amounts are weighed by their level
        doc = self.store.read_pages.get(ref[0]) or {}
        texts = [doc.get("title"), self.variant_info(ref).get("options")] + list((doc.get("facts") or {}).values())
        have = set()
        for text in texts:
            have |= root_set(text)
        return any(set(root_set(form)) <= have for form in self.value_forms(req) if root_set(form))

    def higher_better(self) -> bool:
        if self.decider_dir in ("higher", "lower"):
            return self.decider_dir == "higher"
        return not re.search(r"\b(?:low|least|small|short|few|min|light|cheap)", flatten_lower(self.decider))

    def hunt_better(self) -> None:
        link = self.store
        if not self.decider or self.rank_is_price() or not link.has_tool("search") or not self.model_ok():
            return
        if not self.ranked_fits():
            return                    # nothing fits yet: there is no best to better
        best_seen: float | None = None
        for round_no in range(BEST_ROUNDS):
            if (
                link.ended_flag or self.over_time() or not self.model_ok() or link.turns_free() <= RESERVED_TURNS + 4
                or any(k != self.decoy_ref for k in link.basket)
            ):
                break
            self.read_rank_figures(every=True)
            higher = self.higher_better()
            scored = [
                (float(self.rulings[k]["figure"]), k) for k in self.ranked_fits()
                if isinstance(self.rulings[k].get("figure"), (int, float)) and not isinstance(self.rulings[k].get("figure"), bool)
            ]
            scored.sort(key=lambda x: -x[0] if higher else x[0])
            best = scored[0][0] if scored else None
            if round_no and (best is None or (best_seen is not None and not (best > best_seen if higher else best < best_seen))):
                break                 # the last round turned up nothing better
            best_seen = best
            shown = "\n".join(
                f"- {fig:g}: {self.title_of(k)[:90]} | {self.variant_info(k).get('options') or '-'}" for fig, k in scored[:6]
            ) or "(no fitting listing states a figure yet)"
            txt = (
                f"{self.known_state()}\nWhat decides: {self.decider} ({'higher' if higher else 'lower'} is better)\n"
                f"Best fitting listings so far:\n{shown}\nSearches already sent:\n" + "\n".join(sorted(self.sent_tags)[-30:])
            )
            ans = self.infer.query_model(BETTER_RULES, txt, BETTER_TOOL, 300) or {}
            queries = [str(q) for q in as_list(ans.get("searches")) if str(q).strip()][:8]
            kin = self.kin_searches()
            if not queries and not kin:
                break
            self.search_allow = max(self.search_allow, len(self.sent_tags) + len(queries) + len(kin) + 4)
            self.queue_searches(kin + queries, front=True)
            calls = self.next_search_calls(link.per_turn)
            if not calls:
                break
            link.send_turn(calls)
            self.hear_waiting()
            if link.turns_free() > RESERVED_TURNS + 2:
                self.read_allow = max(self.read_allow, len(link.read_pages) + 60)
                reads = self.pages_to_read(link.per_turn * IDS_PER_COMPARE)
                if reads:
                    link.send_turn(reads[: link.per_turn])
                    self.hear_waiting()
            self.grade_candidates()
        self.read_rank_figures(every=True)

    def near_misses_list(self) -> list[tuple[str, str]]:

        def position(ref: tuple[str, str]) -> tuple:
            ruling = self.rulings[ref]
            met_flags = (ruling.get("met") or [])[: len(self.reqs)]
            chosen = sum((1 for it, wd in zip(met_flags, self.reqs) if it and wd.get("chosen")))
            stated = sum((1 for it, wd in zip(met_flags, self.reqs) if it and wd.get("stated")))
            return (
                -int(bool(ruling.get("kind_ok"))),
                -chosen,
                -stated,
                -sum((1 for it in met_flags if it)),
                (self.store.stock_seen.get(ref) or {}).get("price") or 0.0,
            )

        return sorted((k for k in self.rulings if self.affordable_now(k)), key=position)

    def fresh_searches(self) -> bool:
        if (
            self.rethink_count >= 2
            or not self.model_ok()
            or self.over_time()
            or (self.store.turns_free() <= RESERVED_TURNS + 2)
        ):
            return False
        self.rethink_count += 1
        headings = list(dict.fromkeys((str(p.get("title") or "")[:90] for p in self.store.read_pages.values())))[-25:]
        txt = (
            f"{self.known_state()}\n\nSearches already run:\n"
            + "\n".join(sorted(self.sent_tags)[-25:])
            + "\n\nTitles seen so far (none meets every requirement):\n"
            + "\n".join(headings)
        )
        ans = self.fix_lists(self.infer.query_model(SEARCHES_RULES, txt, SEARCHES_TOOL, 400))
        qlist = [q for q in ans.get("searches") or [] if isinstance(q, str) and q.strip()][:6]
        self.log_note(f"new searches: {qlist}")
        if not qlist:
            return False
        self.queue_searches(qlist, front=True)
        self.search_allow = max(self.search_allow, len(self.sent_tags) + len(qlist) + 8)
        self.look_around(3, 1)
        return True

    def shelf_sweep(self) -> bool:
        if self.sweep_count >= 2 or self.over_time() or self.store.turns_free() <= RESERVED_TURNS + 4:
            return False
        self.sweep_count += 1
        self.search_allow = max(self.search_allow, len(self.sent_tags) + 24)
        self.read_allow = max(self.read_allow, len(self.store.read_pages) + 40)
        self.pending_filter_calls += self.price_step_filters(4) + self.brand_filters_for(4)
        self.queue_searches(self.shelf_word_searches(6), front=True)
        prior = len(self.store.read_pages)
        self.look_around(2, 1)
        return len(self.store.read_pages) > prior

    def learn_synonyms(self) -> None:
        self.widened_once = True
        for req in self.reqs:
            if not self.model_ok():
                break
            ans = (
                self.infer.query_model(
                    SPELLINGS_RULES,
                    f"Item: {self.item_phrase}\nAttribute: {req['prop']}\nValue: {req['value']}",
                    SPELLINGS_TOOL,
                    200,
                )
                or {}
            )
            toks = [str(wd).strip() for wd in as_list(ans.get("words")) if str(wd).strip()][:6]
            field = "spell" if re.search("\\d", req["value"]) else "syn"
            req[field] = list(dict.fromkeys(list(req.get(field) or []) + toks))
            other_facts = " ".join((wd["value"] for wd in self.reqs if wd is not req))
            self.queue_searches([f"{tok} {other_facts} {self.item_phrase}" for tok in toks[:3]], front=True)
        self.log_note(f"widened: {[(wd['value'], wd.get('spell'), wd.get('syn')) for wd in self.reqs]}")

    def look_around(self, rounds: int, enough: int = 2) -> None:
        link = self.store
        for round_no in range(int(rounds)):
            if link.ended_flag or link.turns_free() <= RESERVED_TURNS or self.over_time():
                break
            reads = self.pages_to_read(link.per_turn * IDS_PER_COMPARE)
            if reads:
                link.send_turn(reads[: link.per_turn])
                self.hear_waiting()
            self.grade_candidates()
            if not link.basket and self.place_decoy():
                self.grade_candidates()
            good = [k for k in self.ranked_fits() if self.stated_solid(k)]
            if len([k for k in good if self.stated_firm(k)]) >= enough:
                return
            if len(good) >= enough and (not self.kin_pending_any()) and (round_no >= 2):
                return
            self.queue_searches(self.kin_searches(), front=True)
            if len(good) < enough:
                self.queue_searches(self.variant_probe_searches(), front=True)
                if not self.ranked_fits():
                    self.queue_searches(self.option_word_searches(3))
            if len(self.search_queue) < link.per_turn // 2:
                self.queue_searches(self.shelf_word_searches(6))
            searches = self.price_step_filters(link.per_turn // 2) + (
                [] if self.ranked_fits() else self.brand_filters_for(2)
            )
            searches += self.next_search_calls(link.per_turn - len(searches))
            if not searches and (not reads):
                break
            if searches:
                link.send_turn(searches)
                self.hear_waiting()
        reads = self.pages_to_read(link.per_turn * IDS_PER_COMPARE)
        if reads and (not link.ended_flag) and (link.turns_free() > RESERVED_TURNS):
            link.send_turn(reads[: link.per_turn])
            self.hear_waiting()
        self.grade_candidates()
        if len(self.ranked_fits()) < enough and (not self.widened_once) and (not link.ended_flag):
            self.learn_synonyms()
            return self.look_around(3, enough)
        for _ in range(2):
            if len(self.ranked_fits()) >= enough:
                return
            self.model_check()
        if len(self.ranked_fits()) < enough and (not self.lenient_mode):
            self.lenient_mode = True
            self.grade_candidates()


class Closing(Choosing):

    def twin_of_dropped(self, ref: tuple[str, str]) -> bool:
        if not self.dropped_titles:
            return False
        return (
            ref[0] in self.dropped_ids
            or ref[1].rsplit("-", 1)[0] in self.dropped_stems
            or flatten_lower((self.store.read_pages.get(ref[0]) or {}).get("title")) in self.dropped_titles
        )

    def stock_and_price_ok(self, ref: tuple[str, str]) -> bool:
        last_read = self.store.stock_seen.get(ref) or {}
        cost_now = last_read.get("price")
        return last_read.get("in_stock") is not False and (
            self.spend_limit is None or not isinstance(cost_now, float) or cost_now <= self.spend_limit + 1e-09
        )

    def cart_with_read(self, ref: tuple[str, str], extra: list[dict] | None = None, swap: list[dict] | None = None) -> None:
        bundle = [env_act("add_to_cart", product_id=ref[0], sku=ref[1])] + (swap or [])
        if self.store.has_tool("inspect_stock"):
            bundle.append(env_act("inspect_stock", product_id=ref[0], sku=ref[1]))
        self.store.send_turn(bundle + (extra or []))

    def recart_next(self) -> tuple[str, str] | None:
        for ref in [k for k in self.ranked_fits() if k not in self.dropped][:2]:
            if self.store.ended_flag or self.store.turns_free() < 3:
                break
            self.cart_with_read(ref)
            self.hear_lines()
            if self.stock_and_price_ok(ref):
                return ref
            self.dropped.add(ref)
        return None

    @staticmethod
    def as_statement(txt: str) -> str:
        txt = " ".join(txt.split()).replace("?", ".")
        kept_words = [s for s in re.split("(?<=[.!])\\s+", txt) if not PLEASE_REPLY.search(s)]
        return " ".join(kept_words).strip() or txt

    def answer_line(self, said_line: str) -> None:
        link = self.store
        if link.ended_flag or not link.has_tool("message") or link.turns_free() <= 2:
            return
        if said_line in self.changes_unanswered:
            self.changes_unanswered.remove(said_line)
        facts = "\n".join(self.reply_context[-6:]) or "(nothing new)"
        resp = (
            self.infer.query_model(
                ANSWER_RULES, f"Facts:\n{facts}\n\nThe shopper's line or lines:\n{said_line}", ANSWER_TOOL, 300
            )
            if self.model_ok()
            else None
        )
        txt = self.as_statement(str((resp or {}).get("text") or ""))
        if txt:
            link.send_turn([env_act("message", content=txt[:400])])

    def confirm_changes(self, selected: tuple[str, str]) -> None:
        last_read = self.store.stock_seen.get(selected) or {}
        heading = str((self.store.read_pages.get(selected[0]) or {}).get("title") or "")[:80]
        state = "out of stock" if last_read.get("in_stock") is False else "in stock"
        self.reply_context.append(f"The item being ordered: {heading}, price {last_read.get('price')}, {state}.")
        state_lines, self.changes_unanswered = (self.changes_unanswered[-4:], [])
        clause_list = [self.change_clauses.get(said_line, "") for said_line in state_lines]
        if state_lines and all(clause_list) and self.store.has_tool("message") and (self.store.turns_free() > 2):
            segs = list(dict.fromkeys((seg for clause in clause_list for seg in clause.split("; "))))
            self.store.send_turn(
                [
                    env_act(
                        "message",
                        content=(
                            "Understood: " + "; ".join(segs) + "; everything else you asked for stays the same."
                        )[:400],
                    )
                ]
            )
            return
        if state_lines:
            self.answer_line("\n".join((f"- {said_line}" for said_line in state_lines)))

    def note_price_jump(self, prior: object, now: object, over: bool) -> None:
        if over and isinstance(prior, float) and isinstance(now, float) and (now > prior + 1e-09):
            self.jump_note = f"The {self.item_phrase or 'item'} I had chosen went up from {prior:.2f} to {now:.2f}, over your limit of {self.spend_limit:.2f}, so I am picking a different one."

    def pending_notice(self) -> list[dict]:
        if not self.jump_note or not self.store.has_tool("message"):
            return []
        txt, self.jump_note = (self.jump_note, "")
        return [env_act("message", content=txt)]

    def refill_after_loss(self, dropped: tuple[str, str], wide: bool = True) -> None:
        link = self.store
        doc = link.read_pages.get(dropped[0]) or {}
        toks = lexemes(doc.get("title"))
        if len(toks) < 2 or not link.has_tool("search"):
            return
        joined = " ".join(self.value_word_list())
        own = " ".join(
            (wd for wd in lexemes(self.variant_info(dropped).get("options")) if not wd.replace(".", "").isdigit())
        )
        qlist = [" ".join(toks), " ".join(toks[:10])]
        if own:
            qlist.append(f"{own} {' '.join(toks[:10])}")
        qlist += [
            f"{joined} {' '.join(toks[:10])}",
            f"{joined} {' '.join(toks[:5])}",
            f"{joined} {self.item_phrase} {' '.join(toks[:3])}",
        ]
        if wide:
            qlist += [f"{joined} {' '.join(toks)}"] + self.shelf_word_searches(6)
        texts = list(dict.fromkeys((flatten_lower(" ".join(q.split())[:SEARCH_TEXT_MAX]) for q in qlist)))
        per_turn = [env_act("search", query=t, k=10, in_stock=True) for t in texts]
        fact_name, cost_now = (self.category_label(), (link.stock_seen.get(dropped) or {}).get("price"))
        if wide and fact_name and isinstance(cost_now, float) and link.has_tool("filter"):
            ceiling = round(min(self.spend_limit or cost_now * 1.1, cost_now * 1.1), 2)
            per_turn.append(env_act("filter", category=fact_name, max_price=ceiling, in_stock=True, k=10))
            brand_required = str(doc.get("brand") or "").strip()
            if brand_required and flatten_lower(brand_required) not in NO_MAKER:
                per_turn.append(
                    env_act(
                        "filter", category=fact_name, brand=brand_required, max_price=ceiling, in_stock=True, k=10
                    )
                )
        link.send_turn((self.pending_notice() + per_turn)[: link.per_turn])
        self.hear_waiting()
        reads = self.pages_to_read(4 * IDS_PER_COMPARE, allowance=len(link.read_pages) + 4 * IDS_PER_COMPARE)
        if reads and (not link.ended_flag):
            link.send_turn(reads)
            self.hear_waiting()
        self.grade_candidates()

    def after_loss_of(self, dropped: tuple[str, str]) -> None:
        link = self.store
        self.dropped.add(dropped)
        self.dropped_titles.add(flatten_lower((link.read_pages.get(dropped[0]) or {}).get("title")))
        self.dropped_ids.add(dropped[0])
        self.dropped_stems.add(dropped[1].rsplit("-", 1)[0])
        self.reaction_owed_flag = True
        if link.turns_free() <= RESERVED_TURNS or self.over_time():
            return
        if len([k for k in self.ranked_fits() if k not in self.dropped and self.stated_firm(k)]) < 2:
            self.search_allow = max(self.search_allow, len(self.sent_tags) + LOSS_ROOM)
            self.read_allow = max(self.read_allow, len(link.read_pages) + 40)
            self.judge_allow = max(self.judge_allow, self.cards_sent + 36)
        self.refill_after_loss(
            dropped, wide=not any((self.stated_firm(k) for k in self.ranked_fits() if k not in self.dropped))
        )
        remaining = [k for k in self.ranked_fits() if k not in self.dropped]
        if (
            len([k for k in remaining if self.stated_firm(k)]) < 2
            and link.turns_free() > RESERVED_TURNS
            and (not self.over_time())
        ):
            self.kin_titles_done.clear()
            self.queue_searches(self.kin_searches() + self.shelf_word_searches(6), front=True)
            self.look_around(3 if remaining else 5, 2 if remaining else 1)
        if (
            not [k for k in self.ranked_fits() if k not in self.dropped]
            and link.turns_free() > RESERVED_TURNS
            and (not self.over_time())
        ):
            self.pending_filter_calls += self.price_step_filters(4) + self.brand_filters_for(4)
            self.look_around(2, 1)

    def after_change_of_mind(self) -> None:
        if self.store.turns_free() <= RESERVED_TURNS or self.over_time():
            return
        remaining = [k for k in self.ranked_fits() if k not in self.dropped]
        if len([k for k in remaining if self.stated_firm(k)]) >= 2:
            return
        self.look_around(2 if remaining else 4, 2 if remaining else 1)

    def cite_target(self, doc: dict) -> tuple[str, str] | None:
        located = re.search(
            r"\b(?:cite|citing|justif\w+|quot\w+|back(?:ed|ing)?\b[^.:]{0,30}\bwith|support\w*\b[^.:]{0,30}\bwith|facts you (?:saw|read|found))\b([^.]*)",
            self.store.ask_text, re.I,
        )
        if not located:
            return None
        order = [
            wd
            for wd in (root_word(it) for it in lexemes(located.group(1)) if len(it) > 3)
            if wd not in {"price", "stock", "brand", "product", "category"}
        ]
        sought = set(order)
        best = None
        for fact_name, val in (doc.get("facts") or {}).items():
            shared = {root_word(wd) for wd in lexemes(str(fact_name).replace("_", " "))} & sought
            if not shared:
                continue
            score = (len(shared), -min((order.index(wd) for wd in shared)), bool(re.search("\\d", str(val))))
            if best is None or score > best[0]:
                best = (score, str(fact_name), str(val))
        return (best[1], best[2]) if best else None

    def justification_for(self, ref: tuple[str, str]) -> dict:
        last_read = self.store.stock_seen.get(ref) or {}
        doc = self.store.read_pages.get(ref[0]) or {}
        places = {"title": str(doc.get("title") or ""), "options": str(self.variant_info(ref).get("options") or "")}
        places.update({str(fact_name): str(val) for fact_name, val in (doc.get("facts") or {}).items()})
        claim_list: list[dict] = []
        if isinstance(last_read.get("price"), float):
            claim_list.append({"field": "price", "value": last_read["price"]})
        claim_list.append({"field": "in_stock", "value": True})

        def taken(field: str) -> bool:
            return any((c["field"] == field for c in claim_list))

        ruling = self.rulings.get(ref) or {}
        for where, quote in zip(ruling.get("places") or [], ruling.get("quotes") or []):
            where = re.sub("^facts?\\s*[:.]\\s*", "", str(where or ""))
            field = next(
                (fact_name for fact_name in places if flatten_lower(fact_name) == flatten_lower(where)), None
            )
            said = " ".join(lexemes(quote))
            if field and said and (f" {said} " in f" {' '.join(lexemes(places[field]))} ") and (not taken(field)):
                claim_list.append({"field": field, "value": str(quote).strip()})
        for req in self.reqs:
            toks = lexemes(tight_units(req["value"]))
            if not toks:
                continue
            for field, txt in places.items():
                if f" {' '.join(toks)} " in f" {' '.join(lexemes(tight_units(txt)))} " and (not taken(field)):
                    hit = re.search("\\W+".join((re.escape(wd) for wd in toks)), flatten_lower(txt))
                    if hit:
                        claim_list.append({"field": field, "value": hit.group(0)})
                        break
        cited = self.cite_target(doc)
        if cited:
            claim_list = [c for c in claim_list if c["field"] in ("price", "in_stock", "brand")] + [
                {"field": cited[0], "value": cited[1]}
            ]
        else:
            one = [c for c in claim_list if c["field"] not in ("price", "in_stock")][:1]
            claim_list = [c for c in claim_list if c["field"] in ("price", "in_stock")] + one
        return {"claims": claim_list}

    def pre_order_check(self, selected: tuple[str, str]) -> tuple[str, str]:

        def passes(ref: tuple[str, str]) -> bool:
            plain = all(
                (
                    self.plain_level(ref, wd) >= self.level_required(wd)
                    for wd in self.reqs
                    if self.req_kind(wd) in ("colour", "material")
                )
            )
            return self.stock_and_price_ok(ref) and plain and self.fits_now(ref)

        if passes(selected) or self.store.turns_free() < 4 or self.over_time():
            return selected
        for ref in [k for k in self.ranked_fits() if k != selected and k not in self.dropped][:3]:
            if passes(ref) and not self.worse_by_decider(ref, selected):
                self.log_note(f"checklist: {selected} -> {ref}")
                self.cart_with_read(ref)
                self.hear_lines()
                return ref if self.stock_and_price_ok(ref) else selected
        return selected

    def order_tool_abstains(self) -> bool:
        fn_spec = self.store.tool_specs.get("place_test_order") or {}
        return "abstain" in ((fn_spec.get("parameters") or {}).get("properties") or {})

    def over_limit_match(self) -> tuple[str, str] | None:
        link = self.store
        if self.spend_limit is None:
            return None
        above = [
            k
            for k in dict.fromkeys(link.rows_in_order)
            if k[0] not in link.read_pages
            and k not in self.dropped
            and ((link.stock_seen.get(k) or {}).get("in_stock") is not False)
            and isinstance((link.stock_seen.get(k) or {}).get("price"), float)
            and (self.spend_limit + 1e-09 < link.stock_seen[k]["price"] <= OVER_LIMIT_SPAN * self.spend_limit)
        ]
        context = self.row_context_sets()
        near = sorted((k for k in above if self.row_score_of(k, context) >= 1), key=lambda k: link.stock_seen[k]["price"])
        ids = list(dict.fromkeys(k[0] for k in near))[: 3 * IDS_PER_COMPARE]
        if ids and link.has_tool("compare") and (link.turns_free() > 2):
            link.send_turn(
                [
                    env_act("compare", product_ids=ids[i : i + IDS_PER_COMPARE])
                    for i in range(0, len(ids), IDS_PER_COMPARE)
                ]
            )
            self.hear_waiting()
        located = []
        any_shelf = any(self.in_shelf(p) for p in link.read_pages.values())
        for prod_id, doc in link.read_pages.items():
            if any_shelf and not self.in_shelf(doc):
                continue
            if self.brand_required and not set(lexemes(self.brand_required)) <= set(lexemes(doc.get("brand"))):
                continue
            for option in doc.get("variants") or []:
                ref = (prod_id, str(option.get("sku")))
                last_read = link.stock_seen.get(ref) or {}
                cost_now = last_read.get("price")
                if (
                    ref in self.dropped
                    or last_read.get("in_stock") is False
                    or (not isinstance(cost_now, float))
                    or (cost_now <= self.spend_limit + 1e-09)
                    or cost_now > OVER_LIMIT_SPAN * self.spend_limit
                ):
                    continue
                if self.reqs and all((self.plain_level(ref, wd) >= 1 for wd in self.reqs)):
                    located.append((cost_now, ref))
        return min(located)[1] if located else None

    def closest_miss(self, selected: tuple[str, str] | None) -> tuple[str, str] | None:
        stock_seen = self.store.stock_seen

        def position(ref: tuple[str, str]) -> tuple:
            met_flags = ((self.rulings.get(ref) or {}).get("met") or [])[: len(self.reqs)]
            return (
                -sum((1 for it in met_flags if it)),
                not self.keeps_request_text(ref),
                not self.keeps_pair_choice(ref),
                (stock_seen.get(ref) or {}).get("price") or 0.0,
            )

        candidate_refs = [
            k
            for k, vt in self.rulings.items()
            if k not in self.dropped and vt.get("kind_ok") and self.affordable_now(k)
        ]
        if candidate_refs:
            return min(candidate_refs, key=position)
        return selected if selected is not None and selected not in self.dropped else None

    def abstain_pick(self, selected: tuple[str, str] | None) -> tuple[str, str] | None:
        if (selected is not None and self.fits_now(selected)) or not self.abstain_ready():
            return None
        target = self.over_limit_found
        self.log_note(f"abstain: {target} meets everything but the price")
        return target

    def abstain_ready(self) -> bool:
        if (
            not self.order_tool_abstains()
            or self.store.ended_flag
            or self.limit_cut
            or self.shopper_moved
            or self.carted_fit
            or any(k not in self.dropped for k in self.ranked_fits())
        ):
            return False
        if self.over_limit_found is None and (
            not self.over_limit_tried or (self.over_limit_tries < 3 and len(self.store.read_pages) > self.over_limit_seen + 4)
        ):
            self.over_limit_tried = True
            self.over_limit_tries += 1
            self.over_limit_seen = len(self.store.read_pages)
            self.over_limit_found = self.over_limit_match()
        return self.over_limit_found is not None

    def shortfall_claims(self, ref: tuple[str, str]) -> dict:
        last_read = self.store.stock_seen.get(ref) or {}
        doc = self.store.read_pages.get(ref[0]) or {}
        claim_list: list[dict] = []
        if isinstance(last_read.get("price"), float):
            claim_list.append({"field": "price", "value": last_read["price"]})
        if isinstance(last_read.get("in_stock"), bool):
            claim_list.append({"field": "in_stock", "value": last_read["in_stock"]})
        met_flags = (self.rulings.get(ref) or {}).get("met") or []
        for i, req in enumerate(self.reqs):
            if i < len(met_flags) and met_flags[i]:
                continue
            for fact_name, val in self.own_facts(doc, req)[:1]:
                if not any((c["field"] == fact_name for c in claim_list)):
                    claim_list.append({"field": fact_name, "value": val})
        return {"claims": claim_list}

    def order_anyway(self) -> None:
        link = self.store
        if link.ended_flag or link.turns_free() < 1 or (not link.has_tool("place_test_order")):
            return
        alive = [k for k in link.basket if k not in self.dropped and (k != self.decoy_ref or self.fits_now(k))]
        ref = alive[-1] if alive else next((k for k in self.ranked_fits() if k not in self.dropped), None)
        if ref is None:
            return
        if ref not in link.basket:
            if link.turns_free() < 2:
                return
            link.send_turn([env_act("add_to_cart", product_id=ref[0], sku=ref[1])])
        link.send_turn(
            [env_act("place_test_order", product_id=ref[0], sku=ref[1], justification=self.justification_for(ref))]
        )


class ShopperEpisode(Closing):

    def __init__(self, problem_in: dict) -> None:
        super().__init__(problem_in)
        self.reask_attr = ""
        self.guess_pool: list[str] | None = None   # pooled opening guess of the hidden requirement's attribute
        self.guess_reason_counts: list[int] = []    # reasons each opening reading found
        self.guess_reasons: list[list[str]] = []
        self.repeat_done: set[object] = set()       # questions asked again after a reply that answered nothing
        self.market_lines = 0
        self.said_purposes: list[str] | None = None  # each purpose as said back to the shopper
        self.purposes_asked: set[int] = set()
        self.purposes_met: set[int] = set()
        self.purpose_now: int | None = None
        self.questions_out = 0                      # every question to the shopper, the pair question included
        self.property_asks = 0
        self.first_found_q: int | None = None
        self.purpose_empty = False                  # a purpose question drew nothing
        self.purpose_again: set[int] = set()
        self.purposes_dead: set[int] = set()        # purposes whose question brought no detail out
        self.purposes_gave: set[int] = set()        # purposes whose question brought one out
        self.attrs_at_find = 0
        self.aside_marks: dict[str, bool] = {}
        self.shelf_picks: list[str] = []
        self.label_reads = 0
        self.aside_calls = 0
        self.purposes_twice: set[int] = set()
        self.expect_two = False
        self.shopper_moved = False                  # the shopper changed something at some point
        self.look_at = 0
        self.brief_uses: list[str] = []
        self.figure_cache: dict = {}
        self.read_tries: dict[str, int] = {}
        self.found_lines: list[str] = []
        self.decoy_ref: tuple[str, str] | None = None
        self.decoy_tries = 0
        self.over_limit_found: tuple[str, str] | None = None
        self.over_limit_tried = False
        self.over_limit_tries = 0
        self.over_limit_seen = 0
        self.latest_found: list[str] = []

    def turn_one_plan(self) -> None:
        link = self.store
        ask_line = self.take_brief()
        kind = "pair" if ask_line else ""
        if link.has_tool("message"):
            self.add_read_purposes()
        if not ask_line and link.has_tool("message"):
            kind, ask_line = self.next_need_question()
        self.questions_out += bool(ask_line and link.has_tool("message"))
        turn_one: list[dict] = []
        if ask_line and link.has_tool("message"):
            turn_one.append(env_act("message", content=ask_line[:300]))
            self.messages_asked += 1
        if self.request_skus and link.has_tool("compare"):
            turn_one.append(
                env_act(
                    "compare", product_ids=list(dict.fromkeys((k[0] for k in self.request_skus)))[:IDS_PER_COMPARE]
                )
            )
        name_tries = (
            [
                env_act("filter", category=g, max_price=self.spend_limit, in_stock=True, k=10)
                for g in self.shelf_name_guesses()[:5]
            ]
            if link.has_tool("filter")
            else []
        )
        turn_one += self.next_search_calls(link.per_turn - len(turn_one) - len(name_tries))
        self.reply_awaited = bool(ask_line)
        known_set, found_before = (len(self.reqs), self.needs_found_n)
        cap_before, changes_before = self.spend_limit, len(self.changes_unanswered)
        link.send_turn(turn_one + name_tries)
        silent = not link.unheard
        refused = kind == "pair" and any(REFUSED_AS_MANY.search(x) for x in link.unheard)
        self.hear_lines()
        if refused and self.pair_ask and len(self.reqs) == known_set and link.has_tool("message") and not link.ended_flag:
            self.reply_awaited = True
            self.questions_out += 1
            bare = [str(wd["value"]) for wd in self.pair_wishes][:2]
            first, second = bare if len(bare) == 2 else self.pair_sides[:2]
            again = f"If only one of the two can be had, which one should it be: {first} or {second}?"
            link.send_turn([env_act("message", content=again[:300])] + self.next_search_calls(link.per_turn - 1))
            self.hear_lines()
        if self.pair_ask and len(self.reqs) == known_set:
            self.reqs += self.pair_wishes
            self.open_wishes = list(self.pair_wishes)
            self.queue_searches(self.value_searches(), front=True)
            self.pair_ask = ""
        if kind in ("purpose", "attribute"):
            missed = self.needs_found_n == found_before and (
                self.spend_limit != cap_before or len(self.changes_unanswered) > changes_before
            )
            self.after_reply(kind, found_before, missed, silent)

    def cart_before_naming(self) -> bool:
        link = self.store
        if link.basket or link.ended_flag or not link.has_tool("add_to_cart") or link.turns_free() < 3 or self.stood_in:
            return False
        self.stood_in = True
        stated = [wd for wd in self.reqs if wd.get("stated")]
        pool = [k for k in self.rulings if self.decoy_ok(k)] or [k for k in self.rulings if self.decoy_ok(k, firm=False)]
        if not pool:
            return False
        pool.sort(key=lambda k: (-self.plainness(k, stated), (link.stock_seen.get(k) or {}).get("price") or 0.0))
        self.decoy_ref = pool[0]
        self.cart_with_read(self.decoy_ref)
        if self.decoy_ref not in link.basket:
            self.decoy_ref = None
            self.hear_waiting()
            return True
        self.hear_lines()
        return True

    def commit_loop(self) -> tuple[str, str] | None:
        link = self.store
        commits = 0
        selected: tuple[str, str] | None = None
        while not link.ended_flag and link.turns_free() >= 2 and (commits < 5):
            self.read_rank_figures()
            open_keys = [k for k in self.ranked_fits() if k not in self.dropped]
            real_cart = [k for k in link.basket if k != self.decoy_ref]
            stuck = not open_keys or (not real_cart and (not any((self.keeps_request_text(k) for k in open_keys)))) \
                or not any(all(self.said_in_own_words(k, wd) for wd in self.reqs) for k in open_keys)
            if stuck and (self.fresh_searches() or (not real_cart and self.shelf_sweep())):
                continue
            order = open_keys or [k for k in self.near_misses_list() if k not in self.dropped]
            if not order:
                break
            if not open_keys and self.abstain_ready():
                if self.cart_before_naming():
                    continue
                selected = None
                self.log_note("nothing fits within the limit; a fitting listing just over it is named instead")
                break
            if not [k for k in link.basket if k != self.decoy_ref]:
                order.sort(key=lambda k: (not self.keeps_request_text(k), not self.keeps_pair_choice(k)))
            selected = order[0]
            commits += 1
            heading = str((link.read_pages.get(selected[0]) or {}).get("title") or "")[:50]
            self.log_note(
                f"commit {commits}: {selected} {heading!r} qualifies={self.fits_now(selected)} turn={link.turns_used} pages={len(link.read_pages)} cost={self.infer.spend:.4f} grades={[self.plain_level(selected, wd) for wd in self.reqs]}"
            )
            if self.fits_now(selected):
                self.carted_fit = True
            prior = (link.stock_seen.get(selected) or {}).get("price")
            swap_out = (
                [env_act("remove_from_cart", product_id=self.decoy_ref[0], sku=self.decoy_ref[1])]
                if self.decoy_ref and self.decoy_ref in link.basket and selected != self.decoy_ref
                else []
            )
            if selected == self.decoy_ref and selected in link.basket:
                link.send_turn(
                    ([env_act("inspect_stock", product_id=selected[0], sku=selected[1])] if link.has_tool("inspect_stock") else [])
                    + self.pending_notice()
                )
            else:
                self.cart_with_read(selected, self.pending_notice(), swap_out)
            cost_now = (link.stock_seen.get(selected) or {}).get("price")
            over = (
                self.spend_limit is not None and isinstance(cost_now, float) and (cost_now > self.spend_limit + 1e-09)
            )
            self.note_price_jump(prior, cost_now, over)
            changed, expects = self.hear_lines()
            if not self.stock_and_price_ok(selected):
                self.question_pending = ""
                self.after_loss_of(selected)
                selected = None
                continue
            if changed:
                self.grade_candidates()
                if not self.fits_now(selected):
                    self.dropped.add(selected)
                    self.after_change_of_mind()
                    selected = None
                    continue
            if expects and self.question_pending:
                self.reply_context.append(f"The listing now in the cart: {heading}, {cost_now}, in stock.")
                self.answer_line(self.question_pending)
                self.question_pending = ""
                if self.hear_lines()[0]:
                    continue
            if (
                link.turns_used > int((link.stock_seen.get(selected) or {}).get("turn") or 0)
                and link.has_tool("inspect_stock")
                and (link.turns_free() >= 3)
            ):
                prior = (link.stock_seen.get(selected) or {}).get("price")
                link.send_turn([env_act("inspect_stock", product_id=selected[0], sku=selected[1])])
                moved, _ = self.hear_lines()
                now = (link.stock_seen.get(selected) or {}).get("price")
                self.note_price_jump(
                    prior,
                    now,
                    self.spend_limit is not None and isinstance(now, float) and (now > self.spend_limit + 1e-09),
                )
                if not self.stock_and_price_ok(selected):
                    self.after_loss_of(selected)
                    selected = None
                    continue
                if moved:
                    self.grade_candidates()
                    if not self.fits_now(selected):
                        self.dropped.add(selected)
                        self.after_change_of_mind()
                        selected = None
                        continue
            if self.changes_unanswered and link.has_tool("message") and (link.turns_free() >= 3):
                self.confirm_changes(selected)
                if self.hear_lines()[0]:
                    self.grade_candidates()
                    if not self.fits_now(selected):
                        self.dropped.add(selected)
                        selected = None
                        continue
            break
        if selected is None and (not link.ended_flag) and not self.abstain_ready():
            alive = [k for k in link.basket if k not in self.dropped and (k != self.decoy_ref or self.fits_now(k))]
            alive = alive or [k for k in link.basket if k not in self.dropped]
            selected = alive[-1] if alive else None
            if selected is None and link.turns_free() >= 3:
                nearest = self.closest_miss(None)
                if nearest:
                    self.cart_with_read(nearest)
                    self.hear_lines()
                    selected = nearest if nearest in link.basket else None
        return selected

    def place_order(self, selected: tuple[str, str] | None) -> None:
        link = self.store
        if selected is not None and (not link.ended_flag):
            selected = self.pre_order_check(selected)
        abstain_to = self.abstain_pick(selected)
        target = abstain_to or selected
        if target is not None and not abstain_to and self.look_at and link.has_tool("compare") and link.turns_free() >= 3:
            ids = list(dict.fromkeys([target[0]] + [k[0] for k in self.ranked_fits() if k[0] != target[0]]))[: self.look_at + 2]
            link.send_turn([env_act("compare", product_ids=ids[i : i + IDS_PER_COMPARE]) for i in range(0, len(ids), IDS_PER_COMPARE)])
            changed_now = self.hear_lines()[0]
            if changed_now:
                self.grade_candidates()
            if (changed_now and not self.fits_now(target)) or not self.stock_and_price_ok(target):
                self.dropped.add(target)
                target = self.recart_next() or target
        for _ in range(3):
            if target is None or link.ended_flag or link.turns_free() < 1:
                return
            if self.changes_unanswered and link.has_tool("message") and (link.turns_free() >= 2):
                self.confirm_changes(target)
            abstaining = abstain_to is not None and target == abstain_to
            sent = link.send_turn(
                [
                    env_act(
                        "place_test_order",
                        product_id=target[0],
                        sku=target[1],
                        abstain=True if abstaining else None,
                        justification=self.justification_for(target),
                    )
                ]
            )
            if link.ended_flag:
                return
            changed, expects = self.hear_lines()
            if expects and self.question_pending and (link.turns_free() >= 2):
                self.answer_line(self.question_pending)
                self.question_pending = ""
                changed = self.hear_lines()[0] or changed
            if changed and link.turns_free() >= 3 and (not abstaining):
                self.grade_candidates()
                if not self.fits_now(target):
                    self.dropped.add(target)
                    target = self.recart_next()
            if not sent:
                return

    def run_episode(self) -> list[dict]:
        self.turn_one_plan()
        if self.store.has_tool("message"):
            self.ask_hidden_needs()
        else:
            self.settle_decoy()
        self.look_around(9, 3 if self.decider else 2)
        self.late_stand_in()
        self.hunt_better()
        self.place_order(self.commit_loop())
        return self.store.trace_log


def agent_main(problem_data: dict[str, object]) -> list[dict[str, object]]:
    episode = ShopperEpisode(problem_data)
    try:
        return episode.run_episode()
    except Exception as failure:
        episode.store.trace_log.append(
            {"role": "assistant", "content": f"agent error: {type(failure).__name__}: {failure}"}
        )
        try:
            episode.order_anyway()
        except Exception as second_failure:
            episode.store.trace_log.append(
                {
                    "role": "assistant",
                    "content": f"agent error at the last order: {type(second_failure).__name__}: {second_failure}",
                }
            )
        return episode.store.trace_log


__all__ = ("agent_main",)
