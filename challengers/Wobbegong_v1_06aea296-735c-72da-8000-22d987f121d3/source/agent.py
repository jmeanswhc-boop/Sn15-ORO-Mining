from __future__ import annotations

import json, re, time
from concurrent.futures import ThreadPoolExecutor
from src.agent.proxy_client import ProxyClient


def _dial_group(title, /, **values):
    return type(title, (), values)


_WRAP = 45.0
_SONNET, _HAIKU = "anthropic/claude-sonnet-5.5", "anthropic/claude-haiku-5.5"
_LUNA = "openai/gpt-6-luna"


class Dials:

    Model = _dial_group(
        "Model", strong=_SONNET, light=_HAIKU, spare=_LUNA,
        reader=_SONNET, planner=_SONNET, checker=_SONNET, downgradable=frozenset({_SONNET}),
        cached_prefix="anthropic/", temperature=0, workers=3,
        reader_spare=(_SONNET, _LUNA), reader_light=(_SONNET, _HAIKU), light_spare=(_HAIKU, _LUNA),
        checker_light=(_SONNET, _HAIKU), planner_spare=(_SONNET, _LUNA), planner_light=(_SONNET, _HAIKU),
        reply_chain=(_HAIKU, _LUNA, _SONNET))

    Tokens = _dial_group(
        "Tokens", intake=2500, key_roles=700, arg_fix=400, end_choice=300, kind_labels=500, sift=1200,
        assess=2000, final_check=800, search_words=700, measure_order=600, second_look=1500, condition_check=300,
        process_check=900, question_order=700, answer=600, aside=700, page_value=600, find_page=600)

    Timing = _dial_group(
        "Timing", task=255.0, wrap=_WRAP, model_floor=_WRAP - 12,
        model_wait_low=10, model_wait_high=45, model_wait_add=20, retry_until=_WRAP + 10,
        store_last=3, store_wait_low=4, store_wait_high=90, store_wait_less=1,
        search_until=_WRAP, open_until=_WRAP, verdicts_until=_WRAP, process_until=_WRAP, choose_until=_WRAP,
        refs_until=_WRAP + 20, more_questions_over=_WRAP + 30, early_cart_until=_WRAP + 30,
        menu_reads_over=_WRAP + 20, browse_over=_WRAP + 10, runner_up_until=_WRAP + 20,
        extra_search_over=_WRAP + 20, recheck_over=_WRAP + 25, recheck_cards_until=_WRAP + 20,
        recheck_query_until=_WRAP + 15, conditions_over=_WRAP + 20, fit_until=_WRAP + 10,
        pick_again_over=_WRAP + 20)

    Cost = _dial_group("Cost", soft=0.03, stop=0.05)

    Net = _dial_group(
        "Net", model_timeout=75, model_client_tries=1, model_client_gap=1,
        store_timeout=90, store_client_tries=2, store_client_gap=1,
        chat_path="/inference/chat/completions", step_path="/environment/call",
        model_retries=2, model_pause=2.0, retry_codes=(429, 500, 502, 503, 504), step_attempts=3,
        no_credit=402, out_of_step=409, refused_codes=(403, 422), body_cut=300, error_keys=2)

    History = _dial_group("History", message_args=300, other_args=120, result=160, lines=3, line_cut=200,
                          shown=30)

    Args = _dial_group("Args", refusal_cut=400, repairs_per_tool=2, repair_steps=2, blank_text="n/a",
                       blank_number=1)

    Keys = _dial_group("Keys", reuse_min=2, reuse_share=0.6, fresh_text_over=25, sample_cut=160)

    Request = _dial_group("Request", opening_cut=1500, procedure_kept=6, min_quantity=1, request_term_cut=60)

    Allowance = _dial_group(
        "Allowance", default_steps=30, finish_kept=1, reserve_base=6, reserve_looks_cap=3,
        reserve_looks_plain=2, reserve_procedure_cap=4, reserve_confirm=1, reserve_cheapest=1,
        min_share=2, first_browse_div=2)

    Query = _dial_group(
        "Query", token_max=20, full_rows_min=2, count_ratio_top=3, count_ratio_base=10, more_words=2,
        group_alts_max=4, group_alt_chars=24, noun_chars=22, alts_per_group=3, phrases_per_query=3,
        value_queries_enough=4, first_value_pages_one=6, first_value_pages_many=3, value_pages_all_words=3,
        value_pages_any_word=2, shorter_pages=2, value_spare=1, query_cut=60, value_term_max=40)

    Screen = _dial_group("Screen", min_rows=8, batch=30)

    Read = _dial_group("Read", compare_default=4, compare_min=2, compare_shrink=2, page_cost_with_table=2,
                       page_cost=1)

    Browse = _dial_group(
        "Browse", enough_fits=2, enough_fits_cheapest=3, enough_fits_best=4,
        own_term_pages_one=6, own_term_pages_many=3, field_pages=2, resume_terms=2, resume_pages=2,
        sweep_pages=3, narrow_pages=2, main_pages=1, step_slack=2, step_spare=1, give_up_pages=2,
        open_batch=2, open_batch_wide=4, wide_items=3,
        menu_target_batch=12, menu_target=5, menu_spare=2, menu_rows_min=2, menu_term_extra=1, menu_pages=1,
        menu_chunk_thin=4, menu_chunk=1, need_pages=3, suggest_rows=14, suggest_title=110, suggest_max=5)

    Verdict = _dial_group("Verdict", read_min=3, batch=4, filler_len=3, stem=4, field_values_shown=8,
                          field_keys_shown=3, related_max=2)

    Show = _dial_group(
        "Show", card_default=2200, card_title=160, card_notes=600, card_text=500, card_text_with_facts=400,
        card_min_props=6, card_text_tight=200, check_card=3500, page_card=3000, pick_card=1200,
        match_card=1600, condition_card=2500, screen_title=140, screen_text=260, related_title=120,
        related_text=400, basket_title=120, label_excerpt=600)

    Like = _dial_group("Like", same=0.5, covered=0.67, loose=0.2, label=0.3, goal=0.6, alias=0.8)

    Shopper = _dial_group(
        "Shopper", default_questions=3, reask_spare=1, pair_width=1, step_messages_min=1, assumed_left=2,
        extra_ask_spare=1, process_calls=3, process_steps=2, process_actions=2, procedure_max=8,
        confirm_steps=2, confirm_title=70, example_chars=30, examples=2, reply_cut=400, told_words=2,
        named_key_min=4, alive_min=4, menu_value_cut=60, alias_shared_min=2, one_value_carriers=3,
        ident_carriers=4, weight_cut=0.5, menu_values=3, menu_aliases=3, menu_size=40,
        skip_words=frozenset({"option", "variant"}),
        tops_from=3, tops_one_item=2, tops_many_items=1, top_listings=3, top_first_pos=1, top_gap=2,
        pad_listings=2, pad_of=2, top_clash_window=2)

    Lookup = _dial_group(
        "Lookup", scratch_index=99, max_hops=7, floor_min=4, first_pass_per_item=2, second_pass_spare=2,
        short_value_words=3, long_value_words=4, query_values=3, pages_per_query=3, search_floor=3,
        loose_seller_factor=2, squeezed_min=3, table_steps=3, pick_reach_factor=3, pick_reach_min=4,
        pick_count_factor=2, pick_count_default=2, match_steps=3, single_tries=5, batch_tries=3,
        singles_per_look=3)

    Notice = _dial_group("Notice", cents=100, budget_span=20)

    Choose = _dial_group(
        "Choose", check_tries=3, host_options=3, source_options=4, measure_min=2, measure_rows=12,
        measure_title=120, measure_text=300, measure_cut=9000, override_extra=3, override_confirm=1,
        extra_search_steps=3, repair_moves=10, runner_up_steps=1, condition_steps=3)

    Recheck = _dial_group(
        "Recheck", cards=40, title=110, text=160, notes=200, reasons=3, props=16, cut=14000, fits_taken=2,
        unopened=4, queries=2, open_steps=1, query_steps=3, query_pages=1, rows_per_query=2)

    Fit = _dial_group("Fit", items_min=2, host_title=120, host_text=300, host_notes=150, series_used=2,
                      codes_looked=3, step_spare=2, single_reads=3, read_floor_spare=1)

    Ending = _dial_group(
        "Ending", early_cart_steps=3, refresh_stop=2, regate_steps=2, look_steps=2, parts_steps=2, code_min=4,
        codes_shown=5, checkout_tries=2, checkout_rounds=8, add_steps=2, finish_step=1, near_retries=2,
        rechoice_extra=2, pick_again_steps=3)

    Money = _dial_group("Money", slack=1e-6, unknown_price=1e18, known_below=1e17, sweep_step=0.01,
                        sweep_floor=0.0)

    Trace = _dial_group("Trace", short=80, mid=160, long=200, full=300, few=3, some=4, many=12, cost_digits=4)

    Wording = _dial_group("Wording", short_word=1, plural_over=3, remark_over=24, rows_depth=2,
                          wrapper_dicts=1, wrapper_keys=3, cut_mark="…")

    Ops = _dial_group(
        "Ops",
        borrowed=dict(zip(
            ("same as reference", "different from reference", "counterpart of reference",
             "at least reference", "at most reference", "within reference"),
            ("is", "is not", "is", "at least", "at most", "at least"))),
        with_value=("is", "one of", "contains"), exact=("is", "one of"))


"""Text and number helpers shared by every part: plain forms of words, word overlap, JSON pulled from model
replies, values made to fit an argument schema, and the code-shape patterns. Pure functions, no state."""

UNFIT = object()

PART_CODE, SERIES_CODE, TITLE_CODE, SHORT_SERIES = (re.compile(source) for source in (
    r"\b[A-Za-z]{2,}-?\d{3,}[A-Za-z]?\b|\b[A-Za-z]{2,}\d*-\d{2,}\b",
    r"\b([A-Z]{1,5}\d{1,3})-\d{2,}\b",
    r"\b[A-Z]{1,6}-?\d{3,}[A-Z0-9]*\b",
    r"\b[A-Z]{1,5}\d{1,3}\b",
))
POINTS_AWAY = re.compile(
    r"\b(?:see|read|look|more|details?|info(?:rmation)?)\b"
    r"[^.;:!?]{0,24}?"
    r"\b(?:code|page|part|item|model|product|listing)"
    r"\s*(?:no\.?|number|#)?\s*"
    r"([A-Za-z0-9][A-Za-z0-9-]{2,})",
    re.I,
)

_RX = {name: re.compile(*how) for name, *how in (
    ("off_alphabet", r"[^a-z0-9 ]+"),
    ("fenced", r"```(?:json)?\s*(.*?)```", re.S),
    ("json_piece", r'"(?:[^"\\]|\\.)*"?|[{}\[\]]', re.S),
    ("figure", r"-?\d+(?:\.\d+)?"),
    ("figures", r"\d[\d,]*(?:\.\d+)?"),
    ("leading_figure", r"^[\s\d.,+-]+"),
    ("case_step", r"(?<=[a-z0-9])(?=[A-Z])"),
    ("separators", r"[,;/|]"),
    ("digit", r"\d"),
    ("letter_run", r"[A-Za-z][A-Za-z-]{3,}"),
    ("capitals", r"[A-Z]+"),
    ("joined_kinds", r"\s+and\s+"),
    ("word_run", r"\w+"),
    ("step_wanted", r"expected (\d+)"),
    ("id_shape", r"[A-Za-z]{0,3}-?[0-9a-f]{8,}"),
    ("arg_faults", r"invalid|malformed|unexpected", re.I),
    ("arg_topics", r"missing|required|argument|parameter|schema|validation", re.I),
)}
_FOLD = str.maketrans({"&": " and ", "_": " ", "-": " "})
_KEY_GAPS = str.maketrans("_.", "  ")
_TIGHT = (",", ":")
_ID_EDGE = "[A-Za-z0-9_]"
_CODE_EDGE = "[A-Za-z0-9]"
_DONE_WORDS = frozenset({"ok", "okay", "added", "done", "accepted", "applied", "updated"})
_HOLDER_WORDS = ("cart", "basket")
_HOLDER_LEADS = ("the", "your")


def simplify(text):
    folded = str(text or "").lower().translate(_FOLD)
    return _RX["off_alphabet"].sub(" ", folded).strip()


def _word_forms(text, keep_short):
    shortest = 0 if keep_short else Dials.Wording.short_word
    for word in simplify(text).split():
        if len(word) <= shortest:
            continue
        plural = len(word) > Dials.Wording.plural_over and word[-1] == "s"
        yield word[:-1] if plural else word


def wordset(text):
    return set(_word_forms(text, False))


def wordset_full(text):
    return set(_word_forms(text, True))


def same_name(a, b):
    return simplify(a) == simplify(b) or _same_words(a, b)


def _same_words(a, b):
    left = wordset(a)
    return len(left) > 0 and left == wordset(b)


def likeness(a, b):
    left = wordset(a)
    right = wordset(b) if left else set()
    if left and right:
        common = len(left & right)
        return common / (len(left) + len(right) - common)
    return 0.0


def names_match(key, prop, floor=Dials.Like.same):
    name = str(key)
    return same_name(name, prop) or likeness(name, prop) >= floor


def heads_of(words, size=Dials.Verdict.stem):
    return {word[:size] for word in words}


def _balanced(body, opener, closer):
    start = body.find(opener)
    if start < 0:
        return None
    depth = 0
    for piece in _RX["json_piece"].finditer(body, start):
        mark = piece.group()
        if mark == opener:
            depth += 1
        elif mark == closer:
            depth -= 1
            if not depth:
                return body[start:piece.end()]
    return None


def pull_json(text):
    if not text:
        return None
    body = text.strip()
    fenced = _RX["fenced"].search(body)
    if fenced:
        body = fenced.group(1).strip()
    spans = [_balanced(body, pair[0], pair[1]) for pair in ("{}", "[]")]
    for chunk in [s for s in spans if s is not None] + [body]:
        try:
            return json.loads(chunk)
        except Exception:
            pass
    return None


def shorten(text, limit):
    return text[:limit] + Dials.Wording.cut_mark if len(text) > limit else text


def json_cut(value, limit):
    return shorten(json.dumps(value, separators=_TIGHT, ensure_ascii=False), limit)


def to_json(value):
    return json.dumps(value, ensure_ascii=False)


def ascii_json(value):
    return json.dumps(value)


def keyed_json(value):
    return json.dumps(value, default=str, sort_keys=True)


def read_number(value):
    if isinstance(value, str):
        found = _RX["figure"].search(value.replace(",", ""))
        return float(found.group(0)) if found else None
    numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
    return float(value) if numeric else None


def numbers_in(text):
    return [read_number(raw) for raw in _RX["figures"].findall(text)]


def unit_part(value):
    return _RX["leading_figure"].sub("", str(value)).strip()


def over_limit(amount, limit):
    known = amount is not None and limit is not None
    return known and amount > limit + Dials.Money.slack


def price_rank(price):
    return Dials.Money.unknown_price if price is None else price


def schema_kind(spec):
    kind = spec.get("type")
    if not isinstance(kind, list):
        return kind
    named = [k for k in kind if k != "null"]
    return named[0] if named else None


def closest_option(options, value, floor=Dials.Like.same):
    if not isinstance(value, str):
        return None
    by_form = dict((simplify(o), o) for o in options if isinstance(o, str))
    if simplify(value) in by_form:
        return by_form[simplify(value)]
    scores = {form: likeness(form, value) for form in by_form}
    if not scores:
        return None
    best = max(scores, key=scores.get)
    return by_form[best] if scores[best] >= floor else None


def cut_at_word(text, limit):
    if len(text) > limit:
        head = text[:limit]
        space = head.rfind(" ")
        return head[:space] if space >= 0 else head
    return text


def _enum_of(spec):
    listed = spec.get("enum")
    return listed if isinstance(listed, list) and listed else None


def _from_list(listed, value):
    if value in listed:
        return value
    option = closest_option(listed, value)
    return UNFIT if option is None else option


def _fit_figure(spec, value, whole):
    figure = read_number(value)
    if figure is None:
        return UNFIT
    for key, pick in (("minimum", max), ("maximum", min)):
        bound = spec.get(key)
        if isinstance(bound, (int, float)):
            figure = pick(figure, bound)
    return int(round(figure)) if whole else figure


def _fit_text(spec, value):
    if isinstance(value, str):
        text = value
    elif isinstance(value, list) and value:
        text = str(value[0])
    else:
        text = str(value)
    longest, shortest = spec.get("maxLength"), spec.get("minLength")
    if isinstance(longest, int):
        text = cut_at_word(text, longest)
    too_short = isinstance(shortest, int) and len(text) < shortest
    return UNFIT if too_short else text


def _fit_flag(value):
    if isinstance(value, bool):
        return value
    said = str(value).strip().lower()
    return said == "true" if said in ("true", "false") else UNFIT


def _fit_list(spec, value):
    items = [value] if not isinstance(value, list) else value
    most = spec.get("maxItems")
    return items[:most] if isinstance(most, int) else items


def conform(spec, value):
    listed = _enum_of(spec)
    if listed is not None:
        return _from_list(listed, value)
    kind = schema_kind(spec)
    if kind in ("integer", "number"):
        return _fit_figure(spec, value, whole=kind == "integer")
    if kind == "string":
        return _fit_text(spec, value)
    if kind == "boolean":
        return _fit_flag(value)
    if kind == "array":
        return _fit_list(spec, value)
    return value


def simplest(spec):
    listed = _enum_of(spec)
    if listed is not None:
        return listed[0]
    kind = schema_kind(spec)
    if kind in ("integer", "number", "string"):
        blank = Dials.Args.blank_text if kind == "string" else Dials.Args.blank_number
        return conform(spec, blank)
    if kind == "boolean":
        return False
    return [] if kind == "array" and not spec.get("minItems") else UNFIT


def sounds_like_bad_args(text):
    text = str(text)
    if _RX["arg_faults"].search(text) or _RX["arg_topics"].search(text):
        return True
    return "must be" in text.lower()


def _whole_token(needle, edge, flags=0):
    return re.compile("(?<!%s)%s(?!%s)" % (edge, re.escape(needle), edge), flags)


def mentions_arg(text, name):
    return _whole_token(name, _ID_EDGE).search(text) is not None


def names_code(text, code):
    return _whole_token(code, _CODE_EDGE, re.I).search(text or "") is not None


def _held_in_basket(runs, at, text):
    def gap_before(i):
        return text[runs[i - 1][2]:runs[i][1]]
    if at < 1 or gap_before(at) != " ":
        return False
    before = runs[at - 1][0]
    if before == "in":
        return True
    two_back = at >= 2 and gap_before(at - 1) == " " and runs[at - 2][0] == "in"
    return before in _HOLDER_LEADS and two_back


def _reads_as_success(text):
    runs = [(m.group().lower(), m.start(), m.end()) for m in _RX["word_run"].finditer(text)]
    for at, run in enumerate(runs):
        word = run[0]
        if word in _DONE_WORDS or word.startswith("success"):
            return True
        if word in _HOLDER_WORDS and _held_in_basket(runs, at, text):
            return True
    return False


def cart_said_no(obs):
    lone = list(obs.values()) if isinstance(obs, dict) else []
    if len(lone) == 1 and isinstance(lone[0], str):
        return not _reads_as_success(lone[0])
    return False


def is_remark(value):
    if not isinstance(value, str) or len(value) <= Dials.Wording.remark_over:
        return False
    return " " in value.strip()


def _all_records(rows):
    return len(rows) > 0 and all(isinstance(row, dict) for row in rows)


def row_list(obs, depth=0):
    if isinstance(obs, list):
        return obs if _all_records(obs) else None
    if not isinstance(obs, dict) or depth >= Dials.Wording.rows_depth:
        return None
    found = [rows for rows in (row_list(inner, depth + 1) for inner in obs.values()) if rows]
    return max(found, key=len) if found else None


def _id_word(value):
    return isinstance(value, str) and bool(value.strip()) and " " not in value.strip()


def _bare_wrapper(obs):
    dict_values = [v for v in obs.values() if isinstance(v, dict)]
    if len(dict_values) != Dials.Wording.wrapper_dicts or len(obs) > Dials.Wording.wrapper_keys:
        return False
    return not any(_id_word(v) for v in obs.values())


def _remark_list(value):
    if isinstance(value, list) and value and all(is_remark(x) for x in value):
        return value
    return None


def peel(obs):
    if not isinstance(obs, dict):
        return obs, []
    lone_line_ok = _bare_wrapper(obs)
    said, rest = [], {}
    for key, val in obs.items():
        lines = _remark_list(val)
        if lines is None and lone_line_ok and is_remark(val):
            lines = [val]
        if lines is None:
            rest[key] = val
        else:
            said += [str(line) for line in lines]
    only = next(iter(rest.values())) if len(rest) == 1 else None
    if said and isinstance(only, dict):
        return only, said
    return rest, said


def idlike(value):
    return bool(isinstance(value, str) and _RX["id_shape"].fullmatch(value))


def expected_step(body):
    found = _RX["step_wanted"].search(body)
    return None if found is None else int(found.group(1))


def page_identity(link):
    if isinstance(link, dict):
        values = link.get("identifying_values")
        named = sorted(simplify(v) for v in (values.values() if isinstance(values, dict) else ()) if v)
        return simplify(link.get("sold_by")), simplify(link.get("of_category")), tuple(named)
    return ()


def split_camel(key):
    return _RX["case_step"].sub(" ", str(key))


def key_plain(key):
    return simplify(split_camel(key))


def facet_name(key):
    return " ".join(split_camel(key).translate(_KEY_GAPS).split()).lower()


def prop_form(text):
    return " ".join(str(text).replace("_", " ").split()).lower()


def single_space(text):
    return " ".join(str(text).split())


def value_bits(value):
    return _RX["separators"].split(str(value))


def or_parts(value):
    return [part.strip() for part in str(value).split("|")]


def has_digit(text):
    return _RX["digit"].search(str(text)) is not None


def lettered_words(text):
    return _RX["letter_run"].findall(str(text))


def letter_head(code):
    found = _RX["capitals"].match(code)
    return found.group(0) if found else ""


def head_noun(category):
    if not category:
        return ""
    words = _RX["joined_kinds"].split(category)[0].split()
    return words[-1] if words else ""


def tag_number(text, tag):
    found = re.fullmatch(r"\s*(?:%s)?(\d+)\s*" % re.escape(tag), text)
    return int(found.group(1)) if found else None


def has_part_code(text):
    return PART_CODE.search(text) is not None


def series_heads(text):
    return SERIES_CODE.findall(text or "")


def pointed_codes(text):
    return [m.group(1) for m in POINTS_AWAY.finditer(text)]


def title_codes(text):
    return TITLE_CODE.findall(text or "")


def short_series(text):
    return SHORT_SERIES.findall(text or "")


"""Voice: every text a model or the shopper reads. One brief_<job>() per model job returns (system, user); the
system part never varies for a job (so it can be cached) and all data rides in the user part. Builders shape the
listing cards, rows and diary entries shown to models; the shopper texts are the questions, the basket check, the
closing remark and the compatibility need."""

NEED_TAG = "N"
LINKED_KEY = "linked_pages"


def _say(*parts):
    return "\n".join(parts)


def _dashes(items, when_empty=""):
    return "- " + ("\n- ".join(items) or when_empty)


def _money_or(limit):
    return "not given" if limit is None else str(limit)


def _arg_text(value):
    return json.dumps(value, ensure_ascii=False, default=str)


_PLAN_FIELDS = ("product", "category", "known_needs", "settled_by_request", "story_hints", "asked_before", "menu")
_MENU_FIELDS = ("property", "carried_by", "out_of", "examples")


def _pairs_dict(names, values):
    return dict(zip(names, values))


def _deduped_traits(traits):
    first = {}
    for pair in traits.items():
        first.setdefault((key_plain(pair[0]), keyed_json(pair[1])), pair)
    return dict(first.values())


def _card_extras(offer, traits, blurb):
    show = Dials.Show
    yield "category", offer.shelf, bool(offer.shelf)
    yield "brand", offer.maker, bool(offer.maker)
    yield "vendor", offer.vendor, bool(offer.vendor)
    yield "price", offer.cost, offer.cost is not None
    yield "buyable", offer.in_stock, offer.in_stock is not None
    yield "attributes", traits and _deduped_traits(traits), bool(traits)
    yield "vendor_notes", (offer.remarks or "")[:show.card_notes], bool(offer.remarks)
    text_cut = show.card_text_with_facts if traits else show.card_text
    yield "description", blurb[:text_cut], bool(blurb)


def _shrink_card(card, limit, has_sheet):
    show = Dials.Show
    size = len(to_json(card))
    if has_sheet and size > limit:
        listed = list(card["attributes"].items())
        card["attributes"] = dict(listed[:max(show.card_min_props, int(len(listed) * limit / size))])
    if card.get("description") and len(to_json(card)) > limit:
        card["description"] = card["description"][:show.card_text_tight]
    return card


def offer_card(offer, limit=Dials.Show.card_default):
    traits = offer.traits or {}
    card = dict(id=offer.oid, title=(offer.headline or "")[:Dials.Show.card_title])
    card.update((key, val) for key, val, shown in _card_extras(offer, traits, offer.blurb or "") if shown)
    return _shrink_card(card, limit, bool(traits))


def call_record(step, tool, args, obs, lines, basket, to_shopper):
    hist = Dials.History
    args_cut = hist.message_args if to_shopper else hist.other_args
    entry = dict(step=step, tool=tool, arguments=json_cut(args, args_cut), outcome=json_cut(obs, hist.result))
    said = [line[:hist.line_cut] for line in (lines or [])[:hist.lines]]
    if said:
        entry["shopper_said"] = said
    if to_shopper:
        entry["basket"] = list(map(list, basket or []))
    return entry


def plan_block(product, category, known, settled, hints, asked, menu):
    lists = tuple(map(list, (known, settled, hints, asked, menu)))
    return _pairs_dict(_PLAN_FIELDS, (product, category) + lists)


def menu_entry(prop, carried_by, out_of, examples, aliases):
    shopper = Dials.Shopper
    entry = _pairs_dict(_MENU_FIELDS, (prop, carried_by, out_of, list(examples[:shopper.menu_values])))
    if aliases:
        entry.update(also_called=list(aliases[:shopper.menu_aliases]))
    return entry


def _wording_line(row):
    verdict = Dials.Verdict
    tally = row["tally"]
    common = sorted(tally, key=lambda val: -tally[val])[:verdict.field_values_shown]
    written = " | ".join("{} ({})".format(val, tally[val]) for val in common)
    keys = ", ".join(map(repr, row["keys"][:verdict.field_keys_shown]))
    return ("- {}: key {} found on {} of {} listings read; values seen: {}; {} listing(s) carry the required value "
            "itself").format(row["need"], keys, row["holders"], row["seen"], written, len(row["exact"]))


def _linked_entry(page):
    show = Dials.Show
    text = (page.blurb or "") + " " + (page.remarks or "")
    return dict(id=page.oid, title=(page.headline or "")[:show.related_title], snippet=text[:show.related_text])


def _sample_value(val):
    cut = Dials.Keys.sample_cut
    if isinstance(val, (dict, list)):
        val = json_cut(val, cut)
    return shorten(val, cut) if isinstance(val, str) else val


def _review_card(offer, missed, doubtful):
    check = Dials.Recheck
    head = [("id", offer.oid), ("title", (offer.headline or "")[:check.title]), ("price", offer.cost),
            ("seen", offer.read_full)]
    if not offer.read_full:
        return dict(head + [("body", (offer.blurb or "")[:check.text])])
    head += [("missed", sorted(missed or [])[:check.reasons]), ("doubts", sorted(doubtful or [])[:check.reasons]),
             ("attributes", dict(list((offer.traits or {}).items())[:check.props]))]
    if offer.remarks:
        head.append(("memos", offer.remarks[:check.notes]))
    return dict(head)


def _measure_row(offer):
    choose = Dials.Choose
    return dict(id=offer.oid, title=(offer.headline or "")[:choose.measure_title], price=offer.cost,
                attributes=offer.traits, description=str(offer.blurb or "")[:choose.measure_text])


def _sift_row(number, offer):
    show = Dials.Show
    return dict(row=number, title=(offer.headline or "")[:show.screen_title],
                snippet=(offer.blurb or "")[:show.screen_text], category=offer.shelf, price=offer.cost)


def _tagged(need_texts):
    return "\n".join("%s%d: %s" % (NEED_TAG, at + 1, line) for at, line in enumerate(need_texts))


def brief_intake(request, opening, tools):
    system = _say(
        "Turn one shopping task into a working plan for an assistant in a simulated store. You receive what the "
        "shopper asked for and the list of store tools (OpenAI function schemas). Answer with ONE JSON object and "
        "no other text, in the shape below; everything after // is guidance, not part of the JSON.",
        "",
        "{",
        ' "products": [   // a separate entry per product the shopper will buy (normally a single one)',
        '   {"product_name": "how the request names the product, as a plain noun phrase; inside a list, the words of '
        'that line",',
        '    "kind": "the type of product or store category that is wanted, in the vocabulary of the request",',
        '    "units": 1,',
        '    "needs": [   // firm demands on the own attributes of THIS product only; price, stock, quantity and '
        "budget never go here",
        '      {"property_name": "attribute name as a listing would print it",',
        '       "comparison": "is|is not|at least|at most|one of|contains|same as reference|different from '
        'reference|counterpart of reference|at least reference|at most reference|within reference",',
        '       "required_value": "value, limit or list in the exact wording of the request (one of: alternatives '
        "separated by ' | ')\",",
        '       "other_page": null   // filled only for the comparisons that end in reference: {"of_category": '
        '"...", "sold_by": "...", "identifying_values": {"attribute": "value", ...}, "read_property": "which '
        'attribute to take from that page", "fit_count": null or how many listings the description is said to '
        'match, "pick_rule": null or the tie-break among several (say "whichever holds more"), "relation_words": '
        '"the phrase of the request that links the wanted value to the value on that page", "leeway": null or, '
        "for within reference, the allowance together with its unit}",
        "      }],",
        '    "unit_price_limit": null,   // a ceiling on the price of a single unit of this product (not the order '
        "total) as a plain number in the currency unit of the request; null when there is none",
        '    "hint_properties": ["attribute names the story implies the shopper privately minds about (a note on '
        "who will get it implies gender or age group; a note on what drives a laptop implies the processor); "
        '[] when there are none"],',
        '    "queries": ["2 to 4 search-box queries, narrowest first. The store insists on every word matching, so '
        "each stays short: the product noun with one or two words of value (colour, material, or a number and its "
        "unit like 60w or 2tb). No bare attribute names, and no words that merely identify some other listing. The "
        'last entry is the product noun alone."]',
        "   }],",
        ' "total_limit": null,   // ceiling on the whole order as a plain number, in the unit of the request (cents '
        "if it counts cents)",
        ' "want_cheapest": false,   // true only if the request wants the lowest-priced option or the lowest total '
        "among the fitting ones; a ranking rule for near misses when nothing fits is something else",
        ' "best_by": null,   // the yardstick when the request wants the top listing on some measure, else null',
        ' "messages_allowed": null,   // integer limit on messages to the shopper, or null',
        ' "steps_allowed": null,   // integer limit on tool calls or steps, or null',
        ' "buyable_only": true,   // only purchasable listings are acceptable',
        ' "raise_topics": ["topics the request instructs the assistant to discuss with the shopper, quoted from the '
        'request; [] when none"],',
        ' "must_fit": [{"add_on": "product name", "main": "product name"}],   // rules of the kind X must fit Y; '
        "[] when none",
        ' "cross_needs": [{"when": {"product": "product name", "property_name": "...", "any_of": ["..."]}, '
        '"then_need": {"product": "product name", "property_name": "...", "comparison": "is|is not|at least|at '
        'most|one of", "required_value": "..."} or null, "else_need": same shape or null}],   // a need on one '
        "product that depends on what another product turns out to be; [] when none",
        ' "no_fit_words": null,   // the instruction of the request for the no-match case (e.g. simply say so, no '
        "alternatives), or null",
        ' "no_fit_action": null,   // "nearest" if, when no listing meets every need, the request wants the closest '
        'one bought, proposed or recommended; "decline" if it wants the task closed with no purchase then; null if '
        "it says nothing",
        ' "process": [   // duties about the WAY the work is done: approval or confirmation to get, information to '
        "show or tell the shopper, a tool that must be used, a required order of steps. Product needs, budgets "
        "and message or step limits are not duties, and neither is the no-match instruction (saying so, offering "
        "a near miss), which goes to no_fit_words / no_fit_action. [] when none",
        '   {"at": "start|before_order|before_decline|before_finish", "via": "message|tool", "tool_name": "a tool '
        'name or null", "aim": "the duty in the terms of the request", "message_count": 1}   // message_count = '
        "shopper messages the duty consumes (0 for a tool); a check the assistant runs inside the store (stock, "
        'availability, some value) is via "tool" with 0 messages unless the request wants the outcome reported',
        " ],",
        ' "tool_roles": {',
        '   "keyword_search": {"tool": "...", "words_arg": "argument taking the search words", "page_arg": '
        '"argument for later results (page number, row offset, or a token returned by the previous call), or '
        'null", "paging": "number|offset|token or null", "switch": {"arg_name": "argument that picks keyword '
        'mode", "arg_value": "the value for it"} or null},',
        '   "field_search": {"tool": "tool filtering by category, brand or price fields (possibly the same tool in '
        'a field mode)", "category_arg": "argument or null", "brand_arg": "argument or null", "max_price_arg": '
        '"argument or null", "page_arg": "argument or null", "paging": "number|offset|token or null", "switch": '
        '{"arg_name": "...", "arg_value": "..."} or null} or null,',
        '   "open_one": {"tool": "tool giving one full listing", "id_arg": "argument for the listing id"},',
        '   "spec_sheet": {"tool": "a distinct tool giving just the attribute table of a listing, if the store has '
        'one", "id_arg": "argument"} or null,',
        '   "open_several": {"tool": "tool giving several full listings at once", "ids_arg": "argument for the id '
        'list"} or null,',
        '   "add_to_cart": {"tool": "tool placing a listing in the cart", "id_arg": "argument", "count_arg": '
        '"quantity argument or null", "switch": {"arg_name": "argument choosing add or remove", "add_value": '
        '"value for adding", "remove_value": "value for removing"} or null},',
        '   "replace_cart": {"tool": "tool overwriting the whole cart with a list of lines, if any", "lines_arg": '
        '"argument holding the lines", "line_id_key": "listing-id key within one line", "line_count_key": '
        '"quantity key within one line"} or null,',
        '   "show_cart": {"tool": "tool displaying the cart"} or null,',
        '   "buy_cart": {"tool": "final tool that completes the purchase", "preset_args": {"argument": "value"} or '
        "null},",
        '   "end_unbought": {"tool": "final tool that closes with no purchase", "remark_arg": "its free-text '
        'argument or null", "preset_args": {"argument": "value"} or null},',
        '   "tell_shopper": {"tool": "tool for messaging the shopper", "text_arg": "argument for the message '
        'text"},',
        '   "allowances": {"tool": "tool reporting what allowance remains"} or null',
        " }",
        "}",
        "",
        "Reading rules:",
        '1. Only firm demands become needs. Wishes ("ideally", "if possible") and background (who receives it, '
        "where it goes) are not needs; record what they suggest in hint_properties.",
        "2. Words that only say what type of product it is, or name its category, belong in product_name and kind "
        "and never become needs (\"kids'\" in \"kids' scooters\"; \"outdoor\" inside a category name such as "
        '"outdoor lighting"). The exception: a value the request supplies for a named attribute ("styled as '
        '<value>", "whose <property> is <value>", "with a <value> <property>") is a need on that attribute even '
        "when the value sounds like a product type, and kind remains the product the request names.",
        '3. A list line like "2 x planters where the glaze is matte and the base is not terracotta" yields one '
        "product with units 2 and the needs glaze: is matte and base: is not terracotta.",
        "4. A count describing the contents of ONE listing (a pack, set or box with N pieces) is a need on pack or "
        "set size; units says how many listings to buy.",
        '5. When wording merely spells out a consequence of a need ("screw-top only, so no flip or snap lids"), '
        "record ONE need (lid type: screw-top) and create no attributes for the excluded things.",
        '6. Comparisons: "A or B" becomes one of with "A | B"; "not X" or "anything except X" becomes is not; '
        '"at least N units" becomes at least with "N units". Values keep the units and spelling of the request.',
        "7. Needs tied to the page of ANOTHER listing (that listing being identified by category, seller and a few "
        "attribute values) use the reference comparisons. The same value as stated there: same as reference. Any "
        "value but the one stated there: different from reference. The value matched or paired with the value of "
        "that page instead of the value itself (whatever the page or the store pairs with it, or what suits, fits "
        "or is compatible with it): counterpart of reference. Not lower, or not higher, than the value of that "
        "page: at least reference / at most reference. Within some margin of the figure of that page: within "
        "reference, with the margin in leeway.",
        "   Describe the page the request itself names, even when that page merely refers on to a further listing "
        "for the attribute; the assistant chases such chains on its own. other_page carries category, seller, the "
        "identifying values named and the attribute to read. If the request says several listings of that seller "
        "answer the description and gives a way to choose (the bigger one, the cheaper one, the one with more of "
        "something), the rule goes to pick_rule and the number to fit_count; identifying_values hold the plain "
        "identifying values only, never the comparison. Identifying values describe THAT other listing alone: "
        "never turn them into needs or search words for the product being bought.",
        "   Invented illustration: \"for the strap colour, avoid whatever <seller>'s vacuum flask page (volume "
        '750 ml, coating powder) states" yields exactly one need {"property_name": "strap colour", "comparison": '
        '"different from reference", "required_value": "", "other_page": {"of_category": "vacuum flasks", '
        '"sold_by": "<seller>", "identifying_values": {"volume": "750 ml", "coating": "powder"}, '
        '"read_property": "strap colour"}} and no needs about volume or coating.',
        "8. Needs that hang on how another product comes out (\"if X ends up with P = V, then Y needs Q = W\"; \"Y "
        "needs Q = W unless X's P is V\") go to cross_needs: in the first form then_need applies when the P of X "
        "is among the values, in the second else_need applies when it is not. They never go to needs or process.",
        "9. kind is the product type; if the request names a store category outright (e.g. \"the category is "
        'kitchen storage"), copy that wording.',
        "10. Money: a total stated in cents makes total_limit that many cents. All amounts (total_limit, "
        "unit_price_limit) use the single unit the request counts in; if that unit is cents, convert any amount "
        'given in dollars or spelled out ("nine dollars and five cents" becomes 905).',
        "11. Copy tool and argument names exactly from the schemas. If a SINGLE tool finishes the task both ways, "
        "through an argument that decides between buying the cart and leaving empty-handed, then buy_cart and "
        "end_unbought both point to it, each with preset_args set to the argument value selecting that outcome, "
        "spelled as in the schema. A tool described as final that closes with no purchase is end_unbought; the "
        "one that checks the cart out is buy_cart.",
        "12. If one search tool toggles between keyword and field search through an enum argument, keyword_search "
        "and field_search both point to it, each with its own switch value.",
    )
    user = "SHOPPER REQUEST:\n" + request
    if opening is not None:
        user += "\n\nWHAT THE STORE SHOWED FIRST:\n" + json_cut(opening, Dials.Request.opening_cut)
    user += "\n\nSTORE TOOLS:\n" + to_json(tools)
    return system, user


def brief_key_roles(record):
    system = _say(
        "Label the parts of a store record. The message is one JSON record produced by a store tool. Map every "
        "role listed here to the record key that fills it, spelled exactly as the key appears, or null if nothing "
        "fills it. Answer with ONE JSON object only:",
        '{"listing_id": "identifier of the listing", "listing_name": "its name or title", "description": "its '
        'descriptive copy", "unit_price": "what it costs", "in_stock": "whether it is purchasable", '
        '"category_label": "its category", "brand_name": "its brand", "seller_name": "the shop or seller behind '
        'it", "property_table": "the nested object with its attributes", "notes_text": "free-form remarks from the '
        'seller", "cart_total": "the price total of a cart", "cart_lines": "the line list of a cart", "next_page": '
        '"token or marker for fetching the following page", "result_count": "how many results match overall"}',
    )
    shown = dict(zip(record, map(_sample_value, record.values())))
    return system, to_json(shown)


def brief_arg_fix(tool, description, schema, have, uses, refusal=None):
    system = _say(
        "Repair the arguments of a store tool call. An assistant called a store tool and the call fell short: an "
        "argument it needs is absent, or the store rejected it. Given are the tool name and its description, the "
        "parameter schema, the arguments that were sent, what the tool is being used for, and the rejection text "
        "if there was one.",
        "Arguments already present stay as they are, except where the rejection says one is wrong. Fill each gap "
        "with the value that suits what the tool is doing here: quantity 1 unless the arguments say otherwise, "
        "the first results page, the choice that matches that purpose. Only names and values the schema permits.",
        'Answer with ONE JSON object only: {"arguments": {every argument of the corrected call}}.',
    )
    user = (f"TOOL NAME: {tool}\nWHAT IT DOES: {description}\nPARAMETER SCHEMA: {to_json(schema or {})}"
            f"\nCURRENT ARGUMENTS: {_arg_text(have)}\nROLE IN THIS TASK: {', '.join(uses) or 'not known'}")
    if refusal is not None:
        user += "\nSTORE'S REFUSAL: " + json_cut(refusal, Dials.Args.refusal_cut)
    return system, user


def brief_end_choice(tool, description, needed, buying):
    system = _say(
        "Pick the closing option for a store's finishing tool. This tool finishes a shopping task, and certain "
        "arguments decide the way it finishes. Given are the description of the tool, those arguments with their "
        "permitted values, and the aim.",
        "For each argument choose the permitted value that achieves the aim. Answer with ONE JSON object only, one "
        'key per argument: {"<argument name>": "<permitted value>", ...}.',
    )
    aim = "check out and buy what the cart holds" if buying else "close the task with no purchase"
    user = (f"TOOL NAME: {tool}\nWHAT IT DOES: {description}\nARGUMENTS STILL NEEDED: {to_json(needed)}"
            f"\nAIM: {aim}")
    return system, user


def brief_kind_labels(product, kind, request, labels):
    system = _say(
        "Decide which of a store's category labels are the product a shopper asked for. Listings in this store sit "
        "under category labels, and the shopper is after a single type of product.",
        "List each label that names exactly that type: its meaning fits the wording of the request taken as a "
        'whole, and a wider label that includes the type qualifies too (say, a "pens and pencils" label when a '
        "pen is wanted). Exclude related but different things: accessories, spare parts, cases made for the "
        "product, sub-types or other types.",
        'Answer with ONE JSON object only: {"matching_labels": [labels copied exactly]}, with an empty list if no '
        "label qualifies.",
    )
    user = (f"PRODUCT WANTED: {product} (category words: {kind})\n"
            f"FROM THE REQUEST: {request[:Dials.Show.label_excerpt]}\n"
            f"CATEGORY LABELS IN THIS STORE: {to_json(labels)}")
    return system, user


def brief_sift(kind, need_texts, offers):
    system = _say(
        "Sort search-result rows for a shopping assistant before any page is opened. Every row is numbered and "
        "shows a title, a snippet, a category and a price. Mark, per row, whether that listing MIGHT satisfy the "
        "needs once its complete page is examined.",
        "Use only what the row shows. Silence about a need keeps a row possible: that is a maybe, which belongs in "
        "neither list. A row whose own text plainly conflicts with a required value goes to ruled_out. A row "
        "that shows the required values itself goes to could_fit.",
        "The category label settles what type of product a row is. Rows filed under the requested category are "
        "the right type, whatever the title seems to say, so the product type is never a reason to rule a row "
        "out.",
        'Answer with ONE JSON object only, using row numbers: {"could_fit": [...], "ruled_out": [...]}; rows in '
        "neither list are maybes.",
    )
    rows = list(map(_sift_row, range(1, len(offers) + 1), offers))
    user = f"PRODUCT WANTED: {kind}\nNEEDS:\n" + _dashes(need_texts) + "\n\nSEARCH ROWS:\n" + to_json(rows)
    return system, user


def brief_assess(kind, need_texts, wording_rows, batch):
    system = _say(
        "Check store listings, need by need, against what a shopper requires, using only each listing's own "
        'records. In a record, "attributes" is the spec table of the listing and "vendor_notes" holds the '
        "remarks of the seller.",
        "When a need counts as met:",
        "- A listing's own spec entry for that need gives the required value. Spec keys come in many "
        "spellings (spacing, underscores, prefixes, short forms); any key naming the attribute will do, and one "
        "attribute may appear under more than one key.",
        "- Matching goes by meaning. One entry of a comma list suffices; singular equals plural; near-synonyms and "
        "obvious short forms count; capitalisation is irrelevant; a stored value consisting of the required "
        'words plus extra qualifiers satisfies it ("satin nickel" satisfies "nickel", "engineered wood" '
        'satisfies "wood"). Figures with units are compared after unit conversion and as numbers; limits include '
        "the boundary; a range satisfies a limit only if all of it does.",
        "Which evidence decides:",
        "- The spec table outranks everything. If an entry covers the need, its value settles it, and a "
        "conflicting value is a failure whatever the title claims.",
        "- Only when no spec key covers the need can the description, the title, the vendor notes or an "
        "options/variants field supply the value.",
        "- If no part of the record mentions it, the need is unaddressed.",
        "- A value filed under a neighbouring but different attribute is no evidence (a strap length says nothing "
        "about length; a lid colour says nothing about colour).",
        'Comparison types: "is not" and "different from" fail when the spec shows the forbidden value; "one of" '
        'requires one of the listed values; "contains" requires that value among the ones listed. A need for '
        "something to be absent (value none, no or without) is met when nothing in the record mentions it.",
        "Product type: the category label is decisive, so a listing labelled with the requested category is the "
        "right type regardless of its title. Being an accessory, spare part or case instead of the product itself "
        "can only sink a listing that has no label.",
        "Compatibility with another product (given by model names or codes): satisfied only if this listing's own text "
        "names one of them among the things it fits or works with. If the page of the other product "
        "lists a family or series of part codes that fit, then a listing whose own part code (often in brackets "
        "in the title) starts with that series satisfies it. When a page says some OTHER part fits a product, "
        "that statement is about the listing carrying the code of that part in its title, not about the page "
        "it appears on; "
        f'pages shown under "{LINKED_KEY}" are such pages, and their statements about this listing\'s own part code '
        "do count. A mere reference to another page is not proof.",
        "Handed-over attributes: vendor notes may say that for some attribute the value of another listing "
        "applies (look to, or use, that other listing). This listing's own value for that attribute is then "
        "void, and a need on it is unaddressed unless a linked page given with the listing supplies it. If no "
        "need names the handed-over attribute, nothing changes.",
        "STORE WORDING (when present): for each required attribute, the keys and values used by the listings of "
        "this category that were read so far in this store. If some of those listings hold exactly the required "
        "value in that field, a listing holding a different value there fails, even a broader or related one "
        "(required words with added qualifiers are not a different value). A value none of the read listings "
        "uses may still be satisfied by a clear synonym.",
        "Assess only the needs listed and invent no others.",
        f"Needs carry the tags {NEED_TAG}1, {NEED_TAG}2, ...; refer to them by tag. Answer with ONE JSON object "
        'only, no remarks: {"results": [{"listing_id": "...", "fails": [tags the record clearly fails], '
        '"unaddressed": [tags nothing in the record speaks to], "text_only": [tags satisfied only through the '
        "title, the description or an options field, never through a spec entry or a vendor note], "
        '"by_other_word": [tags satisfied through a synonym or a wider term instead of the value\'s own '
        "words, e.g. 'citrus' for a required 'lemon', or a gender word standing in for a required age-group word; "
        "unit conversion, or a series covering a code, does not count as this]}, ...]}. Give one entry per "
        'listing and omit empty lists. A listing with neither "fails" nor "unaddressed" meets every need given.',
    )
    cards = []
    for offer, linked in batch:
        card = offer_card(offer)
        if linked:
            card[LINKED_KEY] = [_linked_entry(page) for page in linked]
        cards.append(card)
    user = f"PRODUCT WANTED: {kind}\nNEEDS TO CHECK:\n"
    user += _tagged(need_texts)
    if wording_rows:
        user += "\n\nSTORE WORDING (from this store's listings read so far):\n" + "\n".join(map(_wording_line, wording_rows))
    user += "\n\nLISTINGS:\n" + to_json(cards)
    return system, user


def brief_final_check(kind, quantity, total_limit, need_texts, offer):
    system = _say(
        "Last look before an order is placed: does this one listing satisfy everything? Test it against ALL the "
        'needs listed, using its own record only ("attributes" = spec table, "vendor_notes" = remarks of the '
        "seller):",
        "1. A need passes when the listing's own spec entry for it gives the value. Any key naming the "
        "attribute is acceptable; comma lists, synonyms, short forms, singular or plural and converted units are "
        "all acceptable.",
        "2. With no spec key for the need, the vendor notes, options field, title or description may supply it.",
        "3. A need the record says nothing about FAILS. A value under some other attribute is no evidence.",
        "4. Compatibility passes if the other product appears among the things this listing says it fits, or if the "
        "page of the other product gives a compatible part-code series and this listing's own part code "
        "begins with it.",
        "5. If the notes say the value of an attribute comes from another listing, this listing's own value "
        "for it does not satisfy the need.",
        "6. Exclusions and closed lists are binding.",
        "7. A listing labelled with the requested category is the right type of product regardless of its title.",
        "8. Price multiplied by quantity has to fit inside the budget (same currency unit as the prices).",
        "9. If the record shows it cannot be purchased, it fails.",
        "Check only the needs listed and add none of your own. Answer with ONE JSON object only: "
        '{"meets_all": true or false, "unmet": [each need that clearly fails or that nothing in the record '
        "speaks to]}.",
    )
    user = (f"PRODUCT WANTED: {kind} (quantity {quantity})\nTOTAL BUDGET FOR ALL LINES: {_money_or(total_limit)}\n"
            "NEEDS:\n" + _dashes(need_texts, "(no needs beyond the kind of product, the budget and availability)")
            + "\n\nLISTING:\n" + to_json(offer_card(offer, Dials.Show.check_card)))
    return system, user


def brief_search_words(product, kind, need_texts, tried, seen, max_len):
    system = _say(
        "Suggest search-box wording for a store whose search is literal. A listing turns up only if each word of "
        "the query occurs in its title, brand, category or attribute values (stems match, very short words are "
        "skipped). Results arrive in a fixed order with few rows per page, so a good query is narrow enough that "
        "most hits satisfy the needs.",
        "Given: the product, its needs, the queries used already, and some listings seen with a flag saying "
        "whether each met the needs (null = not opened).",
        "Propose at most 5 fresh queries of 1 to 3 words, strongest first: the noun this store's titles use for "
        "the product combined with a required value written the way listings write it, or other words the store "
        "uses for the product or the value. Never repeat a query used already.",
        'Answer with ONE JSON object only: {"new_queries": [...]}.',
    )
    cut = Dials.Browse.suggest_title
    rows = [{"title": (o.headline or "")[:cut], "meets_needs": flag} for o, flag in seen]
    user = (f"PRODUCT: {product} (category: {kind})\nNEEDS:\n" + _dashes(need_texts, "none")
            + f"\nQUERIES ALREADY SENT: {to_json(tried)}\nLISTINGS SEEN: {to_json(rows)}")
    if max_len:
        user += f"\nMAX QUERY LENGTH: {max_len} characters"
    return system, user


def brief_measure_order(measure, product, offers):
    system = _say(
        "Order listings by a shopper's measure of best. All listings below already meet the needs; the shopper "
        "wants whichever is best on the stated measure (a quantity to maximise or to minimise).",
        "Take the figure of each listing for the measure from its attributes, title or description, converting "
        "units where needed so the figures are comparable.",
        'Answer with ONE JSON object only: {"ranked_ids": [every listing id, best first], "read_values": {"id": '
        '"figure as read"}}. Listings with no readable figure come last; on a tie the lower price goes first.',
    )
    rows = [_measure_row(offer) for offer in offers]
    user = f"MEASURE: {measure}\nPRODUCT: {product}\nLISTINGS: " + json_cut(rows, Dials.Choose.measure_cut)
    return system, user


def brief_second_look(product, kind, quantity, total_limit, need_texts, tried,
                      seen):
    system = _say(
        "Second opinion on a product search that is about to end empty-handed. An assistant is close to "
        "abandoning one product. Given: the product, its needs, the budget, the queries used so far, and every "
        'listing met: fully opened ones ("seen": true, with attributes, memos, and the needs the assistant marked '
        "missed or in doubt) and rows known only from search results.",
        "Answer with ONE JSON object only; any list may be empty:",
        '{"fit_ids": [ids of OPENED listings that in fact satisfy every need, judged on their own records: a spec '
        "value is decisive, a conflicting value fails, an attribute no record gives counts as unmet],",
        ' "open_next": [ids of unopened rows likeliest to satisfy the needs, strongest first, no more than 4],',
        ' "new_queries": [no more than 2 fresh keyword queries of 1 to 3 words that would likely surface fitting '
        "listings; none used already]}",
    )
    cards = [_review_card(*entry) for entry in seen]
    user = (f"PRODUCT: {product} (category: {kind}; quantity {quantity})\n"
            f"TOTAL BUDGET FOR ALL LINES: {_money_or(total_limit)}\nNEEDS:\n" + _dashes(need_texts, "none")
            + f"\nQUERIES ALREADY SENT: {to_json(tried)}\n\nLISTINGS SEEN:\n" + json_cut(cards, Dials.Recheck.cut))
    return system, user


def brief_condition_check(property, values, offer):
    system = _say(
        "Read one value off one listing to settle a shopper's if-then rule. Determine from the listing's own "
        "record whether its value for the named property is among the accepted values (equal meaning counts; "
        "convert units).",
        'Answer with ONE JSON object only: {"matches": true or false, "value_found": "the value exactly as the '
        'listing writes it, or null"}. If the record does not give that property at all, answer "matches": false '
        'with "value_found": null.',
    )
    user = (f"PROPERTY: {property}\nACCEPTED VALUES: {to_json(values)}\n\nLISTING:\n"
            + to_json(offer_card(offer, Dials.Show.condition_card)))
    return system, user


def _next_move(moment):
    if moment == "start":
        return "begin the task by searching the store"
    if moment == "before_order":
        return "place the order for the basket shown below"
    if moment == "before_decline":
        return "finish without buying anything"
    return moment


def brief_process_check(request, steps, moment, basket, messages_left, steps_left,
                        tools, calls):
    system = _say(
        "Process duty check at one point of a shopping task. A shopper's request may impose duties on HOW the job "
        "is done; find out which of them the assistant has to perform right now.",
        "Given: the request; the duties it demands; the point reached (what the assistant does next); the current "
        "basket; the remaining allowances; the store tools apart from those that alter the cart or end the task "
        "(the assistant uses those itself); and a log of every call so far with its outcome.",
        "A duty is due only if it belongs to this point, has to happen before the assistant carries on, and has not "
        "yet been fulfilled for the present situation. An approval or confirmation holds only for the basket in "
        "exactly its current form; one obtained for a different basket does not count. A duty already fulfilled "
        "for this situation is not due again, and a duty that belongs to another point is not due.",
        'Answer with ONE JSON object only: {"do_now": [{"tool_name": "exact tool name", "arguments": {"argument": '
        '"value", ...}, "step_served": "which demanded duty this performs"}]}, in execution order, two entries at '
        'most; {"do_now": []} when nothing is due.',
        "Any message to the shopper has to stand by itself: for a basket it lists each line (quantity, title, "
        "price) plus the total, and it states clearly what you want from the shopper. Do not use this check to "
        "ask about product properties, and do not use up the final message on anything other than a demanded "
        "duty. Verifying a fact in the store (stock, availability, a value) means using the read tools of the "
        "store, not messaging, unless the request wants to be informed; since the assistant reads every "
        "listing it buys anyway, such a check is due only if the listings in the basket were never read.",
    )
    cut = Dials.Show.basket_title
    lines = [dict(id=line[0], qty=line[1], title=(line[2] or "")[:cut], price=line[3], product=line[4])
             for line in basket]
    total = sum(qty * (price or 0) for _, qty, _, price, _ in basket)
    left = "unlimited" if messages_left is None else messages_left
    user = ("SHOPPER REQUEST:\n" + request
            + "\n\nPROCESS STEPS THE REQUEST DEMANDS:\n" + to_json(steps)
            + f"\n\nPOINT IN THE TASK: {moment} (next, the assistant will {_next_move(moment)})"
            + "\n\nBASKET AT THIS POINT: " + to_json({"lines": lines, "total": total})
            + f"\nMESSAGES LEFT: {left}; STEPS LEFT: {steps_left}"
            + "\n\nAVAILABLE TOOLS:\n" + to_json(tools)
            + "\n\nCALLS SO FAR (oldest first, with results):\n" + to_json(calls))
    return system, user


def brief_question_order(request, raise_topics, blocks, ask_count):
    system = _say(
        "Choose which property questions a shopping assistant sends, and in what order.",
        "How the shopper behaves: each hidden need is a value for ONE property of one requested product. It comes "
        "out only when a message asks about exactly that property (by its name or a plain synonym). Only the "
        "FIRST property a message mentions gets an answer. Asking about a property the shopper has no opinion on "
        'returns "no preference" and wastes the message.',
        "Given: the request, and per product its name, the needs known so far, hints from the story, and a menu "
        'of properties found on its candidate listings, each with "carried_by" (how many listings have it) and '
        "sample values.",
        'Return exactly ASK_COUNT questions, strongest first: {"asks": [{"product": "the product name as given", '
        '"property_name": "the property spelled as on the menu (words separated by spaces)"}, ...]}.',
        "Priorities:",
        "1. Properties the story or the hints signal come first: what the request implies the shopper will answer "
        "about. For example the person it is meant for leads to age group or gender; how it stays on or fastens "
        "leads to closure; what runs inside leads to processor; its appearance leads to colour, pattern or style. "
        "Translate each hint into the nearest property on the menu.",
        "2. After that, the properties buyers of this product type usually care about: material, look, size or "
        "fit, type or style, a variant like flavour or scent, the intended user, the headline technical figure. "
        'Favour properties with a high "carried_by" whose values vary across listings. Poor picks: a property '
        "with one value everywhere, an option or variant selector posing as a property, identifiers, model "
        "numbers, warranty or shipping remarks.",
        "3. Never ask about a property a known need already settles, about price, stock or quantity, or about one "
        "property twice (different spellings of one property are one property). Divide the questions among the "
        "products by how uncertain each one is; the product with the strongest hints is asked first.",
        "Answer with ONE JSON object only.",
    )
    user = ("SHOPPER REQUEST:\n" + request
            + "\n\nTOPICS THE REQUEST SAYS TO RAISE: " + to_json(raise_topics)
            + "\n\nPRODUCTS:\n" + to_json(blocks)
            + f"\n\nASK_COUNT = {ask_count}")
    return system, user


def brief_answer(product, prop, also, message, known, reply):
    system = _say(
        "Classify a shopper's answer to a property question. The question concerned a single property of one "
        "product, now and then two. Answer with ONE JSON object only:",
        '{"reading": "need|repeat|free|split|unclear|other",',
        ' "property_name": "the property the reply is about (wording of the shopper, or the asked property when '
        'that is what they mean)",',
        ' "comparison": "is|is not|at least|at most|one of|same as reference|different from reference|counterpart '
        'of reference|at least reference|at most reference|within reference",',
        ' "required_value": "the stated value, limit or list (list items separated by \' | \')",',
        ' "other_page": null or {"of_category": "...", "sold_by": "...", "identifying_values": {"attribute": '
        '"value"}, "read_property": "which property to take from that page", "fit_count": null or the number of '
        'listings said to match, "pick_rule": null or how to choose among several, "relation_words": "the phrase '
        'of the shopper linking the wanted value to the value on that page", "leeway": null or the margin allowed '
        "above or below the figure on that page},",
        ' "new_total_limit": null or a number if the shopper gives a new overall spending limit,',
        ' "short_answer": "" or, if the shopper put a question to the assistant, a brief plain reply,',
        ' "process": [] or duties the shopper now imposes on how the job is done (an approval to collect, '
        'something to show or say to them, a tool to run before ordering), each {"at": "before_order|before_decline|'
        'before_finish", "via": "message|tool", "tool_name": null, "aim": "the duty", "message_count": 1}}',
        "Reading types:",
        "- need: a value, limit, exclusion or list the shopper holds to, however it is phrased (it has to be X; "
        "I'm set on X; X and nothing else; either X or Y, no others; anything except X). \"anything except X\" "
        'and "other than X" mean is not X. If the shopper refers to the page of ANOTHER listing (identified by '
        "category, seller and attribute values): wanting the value shown there is same as reference, with "
        "other_page filled; anything but that value is different from reference; the partner value the page "
        "matches with its own (its counterpart, whatever goes with it) is counterpart of reference; not lower, or not higher, "
        "than the value of that page is at least reference / at most reference.",
        "- repeat: instead of answering about the asked property, the shopper restates something already listed "
        "under NEEDS THE SHOPPER ALREADY GAVE (same property and value).",
        "- free: indifferent, leaves it open, anything is fine, has not thought about it.",
        "- split: asks for the message to be split up, or for its subjects one at a time, or answers only its "
        "first part (because it covered more than one subject).",
        "- unclear: did not understand the message, or wants it repeated or simplified.",
        "- other: everything else (greetings, a counter-question, a budget change).",
        "property_name: when the reply itself names a property, use that, in the wording of the shopper, even if "
        "it differs from the one asked; only if the reply names none, use the asked property (with two asked, "
        "the one the value belongs to).",
        'Explicit limit words ("at most N", "at least N", "no more than N", "under N") fix the comparison and the '
        "value even when the sentence around them is oddly put: \"I could never accept one that isn't at most 9 "
        'kilos" gives at most with "9 kilos".',
        "Without limit words, an insisted-on value is is, whatever the figure of speech: declining to look past "
        'a value, or to accept anything else, asks for that very value, figures included (is, "250 ml").',
    )
    asked = prop + " or " + also if also else prop
    user = (f"PRODUCT: {product}\nPROPERTY ASKED: {asked}\nMESSAGE SENT: {message}\n"
            f"NEEDS THE SHOPPER ALREADY GAVE: {to_json(known)}\nSHOPPER'S REPLY: {reply}")
    return system, user


def brief_aside(lines, products):
    system = _say(
        "Shopper remarks turned up inside a store tool's output while a shopping assistant was working. Work out "
        "what they change. Answer with ONE JSON object only:",
        '{"updates": [{"change": "budget|drop|add|price|stock|step|other", "new_total_limit": null or the new top '
        'total as a number, "property_name": null or the property of a need being dropped or added, "comparison": '
        '"is|is not|at least|at most|one of", "required_value": null or the value of the added need, "product": '
        'null or the requested product concerned, "summary": "a brief sentence"}]}',
        "Change types:",
        "- budget: the spending limit of the shopper moved (up or down). Copy the number as the line gives it, in "
        "the unit of that line; cents remain cents, with no conversion.",
        "- drop: a property is no longer required (forget X; X can go; X no longer matters).",
        "- add: an extra need.",
        "- price: the chosen listing is said to cost something different now.",
        "- stock: the chosen listing is said to be unavailable now.",
        "- step: a new duty on how the job is done (an approval to collect, something to show or say to them, a "
        'tool to run before ordering), given in "process_step": {"at": "before_order|before_decline|before_finish", '
        '"via": "message|tool", "tool_name": null, "aim": "the duty", "message_count": 1}.',
    )
    user = "SHOPPER LINES:\n" + _dashes(lines) + "\n\nPRODUCTS IN THE REQUEST: " + to_json(products)
    return system, user


def brief_page_value(prop, relation, offer):
    system = _say(
        "Read a property value off a listing page that a reference points to. The assistant wants the value this "
        "page gives for one property. Possible situations: the value sits in the spec table (\"attributes\") or "
        "in the remarks of the seller (\"vendor_notes\"); or the remarks say the property should be taken from "
        "ANOTHER listing (described by its category, its seller and a few attribute values) rather than from this one. The "
        "remarks may also state which values of the property go together: a single pair involving the value of "
        "this page, or the full table of pairs of the store.",
        'Answer with ONE JSON object only: {"own_value": "the value of this page as written, or null", '
        '"paired_value": "the value this page pairs with its own, or null", "relation_asks": "the value the '
        "relation of the shopper calls for: the paired value if the relation wants what pairs with, suits or "
        "works alongside the value of the page and the page gives a pairing; the page's own value if it "
        'wants the same value", "look_elsewhere": null or {"of_category": "null unless the remarks name one", '
        '"sold_by": "...", "identifying_values": {"attribute": "value"}, "read_property": "..."}}',
        "If the remarks send the REQUESTED property to another listing, fill look_elsewhere and give no value; "
        "remarks that send a different property elsewhere are no redirect for this one. A value stated plainly in "
        "the remarks is the answer. Failing that, read the spec table under any key naming the property; with no "
        "such key, use the options field, title or description if they give it. A sentence that merely lists the "
        "pairs of the store does not reveal which one is this page's own value. paired_value has to come "
        "from a pairing the page gives: from a list of pairs, take the pair containing this page's own value "
        "(pairs work in both directions); null if the page gives no pairing or if its own value is not known.",
    )
    user = f"PROPERTY TO READ: {prop}\n"
    if relation:
        user += f"HOW THE SHOPPER RELATES IT: {relation}\n"
    user += "\nPAGE:\n" + to_json(offer_card(offer, Dials.Show.page_card))
    return system, user


def brief_find_page(category, seller, values, offers, pick="", count=None,
                    choosing=False):
    system = _say(
        "Identify one described listing among candidate pages. The assistant is hunting for a specific listing "
        "known only by category, seller and a few identifying values. The candidate listings follow, each with its "
        "record.",
        'Answer with ONE JSON object only, with no explanation: {"listing_id": "id of the listing fitting the '
        'description, or null if none fits"}.',
        "Conditions for a fit: the seller is the same; the category label names the described type; every "
        "identifying value agrees with the record of that listing (equal meaning; units converted). For a listing "
        "shown without its spec table (\"attributes\"), the remarks of its seller (\"vendor_notes\"), its title "
        "and its description make up its record, and remarks stating the described values make it a fit. A "
        "listing whose record omits any described value, or gives a different one, is no fit: answer null "
        "instead of the nearest candidate, and the assistant will keep searching. With a PICK RULE, more than one "
        "listing can fit: apply the rule among those that fit and return its choice.",
    )
    limit = Dials.Show.pick_card if choosing else Dials.Show.match_card
    user = f"LOOKING FOR: category {category}; seller {seller}; identifying values {to_json(values)}"
    if choosing and pick:
        user += f"\nPICK RULE: {pick}"
    if choosing and count:
        user += f"\nHOW MANY FIT: {count}"
    user += "\n\nCANDIDATES:\n" + to_json([offer_card(o, limit) for o in offers])
    return system, user


_NAME_BREAK = re.compile(r"\s*(?:,|&|/|\band\b)\s*", re.I)


def short_name(product, request=""):
    text = single_space(str(product or ""))
    parts = [part for part in _NAME_BREAK.split(text) if part]
    if len(parts) < 2:
        return text
    said = str(request or "").lower()
    pick = next((part for part in parts if part.lower() in said), parts[0])
    if len(pick.split()) > 1 or pick.lower().endswith("s"):
        return pick
    return parts[-1].split()[-1]


def question_single(prop, product, request=""):
    return f"What {prop} should the {short_name(product, request)} have?"


def question_pair(prop, also, product, request=""):
    return f"Which {prop} and which {also} do you want for the {short_name(product, request)}?"


def question_focused(prop, product, examples, request=""):
    shown = " or ".join(examples[:Dials.Shopper.examples])
    ask = f"One thing only: what {prop} do you need for the {short_name(product, request)}"
    return ask + (f" (for example {shown})?" if shown else "?")


def basket_confirmation(lines):
    parts = []
    total = 0.0
    for qty, title, price in lines:
        part = f"{qty} x {single_space(title)[:Dials.Shopper.confirm_title]}"
        if price is not None:
            total += price * qty
            part += f" ({price:g} each)"
        parts.append(part)
    closing = f". Total: {total:g}." if total else "."
    return ("Here is the basket I am about to order: " + "; ".join(parts) + closing
            + " Do you approve checking out this basket? A yes is all I need.")


def no_fit_remark():
    return "None of the store's listings meets all of the request's requirements."


def fit_need_value(host_product, host):
    fit = Dials.Fit
    pieces = [(host.headline or "")[:fit.host_title], (host.blurb or "")[:fit.host_text],
              (host.remarks or "")[:fit.host_notes]]
    page = " ".join(piece for piece in pieces if piece)
    return f"fits the {host_product} picked for this order; that listing's page says: {page}"


"""The model gateway: every language-model call of the agent passes through one Oracle."""

_MODEL_LINE = ProxyClient(retry_delay=Dials.Net.model_client_gap, max_retries=Dials.Net.model_client_tries,
                          timeout=Dials.Net.model_timeout)


def _clamp_int(value, low, high):
    return int(max(low, min(high, value)))


def _plain_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _content_text(content):
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict):
                pieces.append(str(part.get("text") or ""))
        content = "".join(pieces)
    if content:
        return str(content)
    return None


class Oracle:

    def __init__(self, errand):
        self.e = errand
        self.spend, self.tries, self.misses = 0.0, 0, 0

    def post(self, path, body):
        return _MODEL_LINE.post_verbose(path, body)

    def over_spend(self):
        return Dials.Cost.stop <= self.spend

    def ask_json(self, models, brief, limit):
        for model in models:
            found = pull_json(self.talk(model, *brief, limit) or "")
            if found is not None:
                return found
        return None

    def talk(self, model, system, user, limit):
        model = self._model_for(model)
        left = self.e.seconds_left()
        if left < Dials.Timing.model_floor:
            return None
        clock = Dials.Timing
        _MODEL_LINE.timeout = _clamp_int(left - clock.wrap + clock.model_wait_add, clock.model_wait_low,
                                         clock.model_wait_high)
        body = dict(model=model, messages=[self._system_part(model, system), dict(role="user", content=user)],
                    temperature=Dials.Model.temperature, max_tokens=limit)
        self.tries += 1
        result = self.post(Dials.Net.chat_path, body)
        reposts = 0
        while reposts < Dials.Net.model_retries and self._repost_due(result):
            reposts += 1
            time.sleep(Dials.Net.model_pause * reposts)
            result = self.post(Dials.Net.chat_path, body)
        text = self._reply_text(result)
        if text is None:
            self.misses += 1
            why = str(result.error)[:Dials.Trace.long]
            self.e.log("model_miss", error=why, model=model)
        return text

    def _model_for(self, model):
        if model != Dials.Model.light and model in Dials.Model.downgradable and Dials.Cost.soft <= self.spend:
            return Dials.Model.light
        return model

    @staticmethod
    def _system_part(model, system):
        if not model.startswith(Dials.Model.cached_prefix):
            return dict(role="system", content=system)
        block = dict(type="text", text=system, cache_control=dict(type="ephemeral"))
        return dict(role="system", content=[block])

    @staticmethod
    def _answered(result):
        data = result.data
        return isinstance(data, dict) and bool(data.get("choices"))

    def _repost_due(self, result):
        if self._answered(result):
            return False
        status = result.error.get("status") if isinstance(result.error, dict) else None
        return status in Dials.Net.retry_codes and self.e.seconds_left() >= Dials.Timing.retry_until

    def _reply_text(self, result):
        if not self._answered(result):
            return None
        data = result.data
        cost = (data.get("usage") or {}).get("cost")
        if isinstance(cost, (int, float)):
            self.spend += float(cost)
        first = data["choices"][0]
        return _content_text((first.get("message") or {}).get("content"))

    @staticmethod
    def run_many(jobs, width=Dials.Model.workers):
        if len(jobs) < 2:
            return list(map(Oracle.shield, jobs))
        size = min(width, len(jobs))
        with ThreadPoolExecutor(max_workers=size) as pool:
            done = pool.map(Oracle.shield, jobs)
            return list(done)

    @staticmethod
    def shield(job):
        outcome = None
        try:
            outcome = job()
        except Exception:
            pass
        return outcome


"""The store gateway: tool roles, argument fitting and repair, one environment step, record keys."""

_STORE_LINE = ProxyClient(retry_delay=Dials.Net.store_client_gap, max_retries=Dials.Net.store_client_tries,
                          timeout=Dials.Net.store_timeout)
_FINISHERS = ("buy_cart", "end_unbought")
_NO_CHOICE = object()


class Outcome:

    def __init__(self):
        self.ok, self.body, self.said = False, None, []
        self.fault, self.args_bad = None, False


def _tool_body(entry):
    wrapped = entry.get("function")
    if isinstance(wrapped, dict):
        return wrapped
    return entry


def _params(form):
    return form.get("properties") or {}


def _needed(form):
    return form.get("required") or []


def _kind_is(spec, *kinds):
    return isinstance(spec, dict) and spec.get("type") in kinds


def _listed_enum(props, name):
    spec = props.get(name) or {}
    return spec.get("enum")


def _first(names, keep, default=None):
    for name in names:
        if keep(name):
            return name
    return default


def _holds_lines(spec):
    if not _kind_is(spec, "array"):
        return False
    inner = spec.get("items")
    return isinstance(inner, dict) and inner.get("type") == "object"


def _json_type_class(spec):
    kind = schema_kind(spec)
    if kind == "string":
        return str
    if kind in ("integer", "number"):
        return (int, float)
    if kind == "array":
        return list
    if kind == "boolean":
        return bool
    return object


def _scalar_table(key, value):
    return isinstance(value, dict) and bool(value) and \
        all(x is None or isinstance(x, (str, int, float)) for x in value.values())


def _id_value(key, value):
    return idlike(value)


def _price_value(key, value):
    return _plain_number(value) and "price" in key.lower()


def _text_value(record, field):
    value = record.get(field) if field else None
    return value if isinstance(value, str) else ""


def _took_step(data):
    return isinstance(data, dict) and any(mark in data for mark in ("observation", "finished"))


class Counter:

    def __init__(self, errand):
        self.e = errand
        view = errand.task_view
        self.tool_list = [entry for entry in view.get("tools") or () if isinstance(entry, dict)]
        self.forms = {}
        for body in map(_tool_body, self.tool_list):
            name = body.get("name")
            if name:
                self.forms[name] = body.get("parameters") or dict(type="object", properties={})
        self.allowance, self.used = int(view.get("max_steps") or Dials.Allowance.default_steps), 0
        self.closed, self.broke = bool(view.get("finished")), False
        self.ticket = dict(errand.task_binding)
        self.jobs, self.diary, self.inbox, self.heard = {}, [], [], set()
        self.key_maps = dict(errand.lore.get("key_maps") or {})
        self.filled, self.mends, self.picked, self.rekeyed = {}, {}, {}, set()

    def steps_free(self):
        return -self.used + self.allowance

    def cap_steps(self, value):
        if _plain_number(value) and 0 < value < self.allowance:
            self.allowance = int(value)

    def drain_inbox(self):
        fresh = list(filter(lambda line: line not in self.heard, self.inbox))
        self.heard |= set(fresh)
        self.inbox = []
        return fresh

    def diary_tail(self, n):
        return self.diary[-n:]

    def known_maps(self):
        return self.key_maps

    def job(self, role):
        return self.jobs.get(role)

    def job_field(self, role, key):
        spec = self.jobs.get(role)
        return spec.get(key) if spec else None

    def tool_for(self, role):
        return self.job_field(role, "tool")

    def form_of(self, role):
        tool = self.tool_for(role)
        return self.forms.get(tool, {}) if tool else {}

    def arg_form(self, role, arg):
        return _params(self.form_of(role)).get(arg) or {}

    def arg_kind(self, role, arg):
        return self.arg_form(role, arg).get("type") if arg else None

    def paging(self, role):
        page_arg = self.job_field(role, "page_arg")
        if not page_arg:
            return None
        if self.arg_kind(role, page_arg) != "integer":
            return "token"
        said = str(self.job_field(role, "paging") or "").lower()
        if said == "offset":
            return "offset"
        named = any(part in page_arg.lower() for part in ("offset", "start", "skip"))
        return "offset" if said != "number" and named else "number"

    def listed_value(self, role, arg, value):
        options = self.arg_form(role, arg).get("enum")
        texts = isinstance(value, str) and isinstance(options, list) and any(isinstance(o, str) for o in options)
        return closest_option(options, value) if texts else value

    def clip_arg(self, role, arg, text):
        most = self.arg_form(role, arg).get("maxLength")
        return cut_at_word(text, most) if isinstance(most, int) else text

    def describe(self, tool):
        for body in map(_tool_body, self.tool_list):
            if body.get("name") == tool:
                return str(body.get("description") or "")
        return ""

    def tool_cards(self):
        return [dict(name=body.get("name"), description=body.get("description", ""), parameters=body.get("parameters"))
                for body in map(_tool_body, self.tool_list)]

    def tool_names(self):
        return list(self.forms)

    def tool_form(self, tool):
        return self.forms.get(tool) if isinstance(tool, str) else None

    def jobs_of(self, tool):
        return [role for role in self.jobs if self.job_field(role, "tool") == tool]

    def _form_named(self, spec):
        return self.forms.get(spec["tool"]) or {}

    def _names_tool(self, spec):
        if not isinstance(spec, dict):
            return False
        tool = spec.get("tool")
        return isinstance(tool, str) and tool in self.forms

    def adopt_jobs(self, raw):
        table = raw if isinstance(raw, dict) else {}
        self.jobs = {role: spec for role, spec in table.items() if self._names_tool(spec)}
        self.square_jobs()
        absent = [role for role in ("keyword_search", "open_one", "buy_cart", "end_unbought", "tell_shopper")
                  if role not in self.jobs]
        for role in absent:
            self.e.log("job_missing", role=role)
        self.e.log("jobs", roles={role: self.tool_for(role) for role in self.jobs})

    def square_jobs(self):
        self._square_cart("add_to_cart")
        self._square_cart("replace_cart")
        self._name_arg("tell_shopper", "text_arg", "string")
        self._name_arg("open_one", "id_arg", "string")
        self._name_arg("spec_sheet", "id_arg", "string")
        self._name_arg("open_several", "ids_arg", "array")
        self._name_arg("keyword_search", "words_arg", "string")
        self._share_ending("buy_cart", "end_unbought")
        self._share_ending("end_unbought", "buy_cart")

    @staticmethod
    def _switch_arg(spec):
        switch = spec.get("switch")
        if isinstance(switch, dict):
            return switch.get("arg_name")
        return None

    def _square_cart(self, role):
        spec = self.jobs.get(role)
        if not spec:
            return
        form = self._form_named(spec)
        props = _params(form)
        lines_arg = _first(props, lambda name: _holds_lines(props[name]))
        if lines_arg is None:
            if role == "add_to_cart":
                self._square_adder(spec, form, props)
            return
        row = _params(props[lines_arg]["items"])
        keep = role == "replace_cart"
        id_key = spec["line_id_key"] if keep and spec.get("line_id_key") in row else \
            _first(row, lambda name: _kind_is(row[name], "string"))
        count_key = spec["line_count_key"] if keep and spec.get("line_count_key") in row else \
            _first(row, lambda name: _kind_is(row[name], "integer"))
        if id_key:
            self.jobs["replace_cart"] = dict(tool=spec["tool"], lines_arg=lines_arg, line_id_key=id_key,
                                             line_count_key=count_key)
        if role == "add_to_cart":
            del self.jobs["add_to_cart"]

    def _square_adder(self, spec, form, props):
        needed = _needed(form)
        if spec.get("id_arg") not in props:
            plain = [name for name in props if _kind_is(props[name], "string") and not props[name].get("enum")]
            spec["id_arg"] = _first(needed, lambda name: name in plain, plain[0] if plain else None)
        if spec.get("count_arg") and spec["count_arg"] not in props:
            spec["count_arg"] = _first(props, lambda name: _kind_is(props[name], "integer"))
        if not spec.get("count_arg"):
            skip = (spec.get("id_arg"), self._switch_arg(spec))
            spec["count_arg"] = _first(list(needed) + list(props),
                                       lambda name: name not in skip and _kind_is(props.get(name), "integer", "number"))
        if not spec.get("id_arg"):
            self.jobs.pop("add_to_cart", None)

    def _name_arg(self, role, field, kind):
        spec = self.jobs.get(role)
        if not spec:
            return
        form = self._form_named(spec)
        props = _params(form)
        if not props or spec.get(field) in props:
            return
        switch_arg = self._switch_arg(spec)
        pool = [name for name in props
                if name != switch_arg and _kind_is(props[name], kind) and not props[name].get("enum")]
        spec[field] = _first(_needed(form), lambda name: name in pool, pool[0] if pool else None)

    def _share_ending(self, have, other):
        spec = self.jobs.get(have)
        if not spec or other in self.jobs:
            return
        form = self._form_named(spec)
        props = _params(form)
        if _first(_needed(form), lambda name: isinstance(_listed_enum(props, name), list), _NO_CHOICE) is not _NO_CHOICE:
            self.jobs[other] = dict(tool=spec["tool"])

    def ending_args(self, role):
        spec = self.jobs.get(role) or {}
        form = self.form_of(role)
        props = _params(form)
        chosen = self._preset_choices(spec.get("preset_args"), props)
        still = [name for name in _needed(form) if name not in chosen and isinstance(_listed_enum(props, name), list)]
        if still:
            answer = self._choice_answer(role, spec.get("tool"), props, still)
            for name in still:
                value = answer.get(name)
                if value in props[name]["enum"]:
                    chosen[name] = value
        return chosen

    @staticmethod
    def _preset_choices(preset, props):
        chosen = {}
        for name, value in (preset.items() if isinstance(preset, dict) else ()):
            if name in props and value not in (None, ""):
                value = Counter._listed_or_near(props[name].get("enum"), value)
                if value is not _NO_CHOICE:
                    chosen[name] = value
        return chosen

    @staticmethod
    def _listed_or_near(options, value):
        if not (isinstance(options, list) and options) or value in options:
            return value
        best, score = None, -1.0
        for option in options:
            close = likeness(str(option), str(value))
            if close > score:
                best, score = option, close
        return best if score >= Dials.Like.same else _NO_CHOICE

    def _choice_answer(self, role, tool, props, still):
        cached = self.picked.get(role)
        if cached is not None:
            return cached
        brief = brief_end_choice(tool=tool, description=self.describe(tool or ""),
                                 needed={name: props[name] for name in still}, buying=role == "buy_cart")
        got = self.e.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.end_choice)
        answer = got if isinstance(got, dict) else {}
        if answer:
            self.picked[role] = answer
        return answer

    def act(self, tool, args):
        fitted = self.shape_args(tool, args)
        first = self.transmit(tool, fitted)
        if not first.fault or not self.looks_arg_refusal(first):
            return first
        fixed = self.mend_args(tool, fitted, first.body)
        retry = fixed is not None and fixed != fitted
        final = self.transmit(tool, fixed) if retry else first
        final.args_bad = bool(final.fault) and self.looks_arg_refusal(final)
        return final

    @staticmethod
    def looks_arg_refusal(out):
        return sounds_like_bad_args(" ".join((str(out.fault), json_cut(out.body, Dials.Args.refusal_cut))))

    def _blocked(self, tool):
        if self.closed or self.broke or tool not in self.forms:
            return "cannot call"
        free = self.steps_free()
        if free <= 0:
            return "steps used up"
        if free <= Dials.Allowance.finish_kept and tool not in map(self.tool_for, _FINISHERS):
            return "steps used up"
        return None

    def transmit(self, tool, args):
        out = Outcome()
        out.fault = self._blocked(tool)
        if out.fault:
            return out
        payload = dict(self.ticket, step=self.used + 1, action=dict(name=tool, args=args))
        for _ in range(Dials.Net.step_attempts):
            left = self.e.seconds_left()
            if left < Dials.Timing.store_last:
                out.fault = "time used up"
                return out
            clock = Dials.Timing
            _STORE_LINE.timeout = _clamp_int(left - clock.store_wait_less, clock.store_wait_low, clock.store_wait_high)
            result = _STORE_LINE.post_verbose(Dials.Net.step_path, payload)
            if _took_step(result.data):
                self._record_step(tool, args, result.data, out)
                return out
            out.fault = self._step_failed(result, payload)
            if out.fault is not None:
                return out
        out.fault = "store unreachable"
        return out

    def _record_step(self, tool, args, data, out):
        number = data.get("step")
        self.used = number if isinstance(number, int) else self.used + 1
        self.closed = self.closed or bool(data.get("finished"))
        body, lines = peel(data.get("observation"))
        self.inbox += [line for line in lines if line not in self.heard]
        out.body, out.said = body, lines
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, str) and len(body) <= Dials.Net.error_keys:
            out.fault = error
        else:
            out.ok = True
        self.e.log("tool", lines=lines[:Dials.Trace.few], result=json_cut(body, Dials.Trace.long),
                   args=json_cut(args, Dials.Trace.mid), tool=tool)
        to_shopper = tool == self.tool_for("tell_shopper")
        basket = list(map(list, self.e.parley.basket_key())) if to_shopper else None
        self.diary.append(call_record(step=self.used, tool=tool, args=args, obs=body, lines=lines, basket=basket,
                                      to_shopper=to_shopper))

    def _step_failed(self, result, payload):
        problem = result.error or {}
        known = isinstance(problem, dict)
        status = problem.get("status") if known else None
        text = str(problem.get("body") if known else problem)[:Dials.Net.body_cut]
        self.e.log("store_error", body=text, status=status)
        if status == Dials.Net.no_credit:
            self.broke = True
            return "key out of credit"
        if status in Dials.Net.refused_codes:
            return "store refused %s" % status
        return self._resync(text, payload) if status == Dials.Net.out_of_step else None

    def _resync(self, text, payload):
        if "finished" in text:
            self.closed = True
            return "task already over"
        wanted = expected_step(text)
        if wanted is None:
            return "step out of sync"
        self.used = wanted - 1
        payload["step"] = wanted
        return None

    def shape_args(self, tool, args):
        form = self.forms.get(tool) or {}
        props = _params(form)
        if not props:
            return args
        fitted, strays = {}, {}
        for key, value in args.items():
            if value is None:
                continue
            if key not in props:
                strays[key] = value
            elif (shaped := conform(props[key], value)) is not UNFIT:
                fitted[key] = shaped
        lacking = [key for key in _needed(form) if key not in fitted]
        unmet = [key for key in lacking if not self._place_stray(key, props.get(key) or {}, strays, fitted)]
        if unmet:
            fitted.update(self.supply_args(tool, fitted, unmet))
        return fitted

    @staticmethod
    def _place_stray(key, spec, strays, fitted):
        kind = _json_type_class(spec)
        order = sorted(strays, key=lambda name: (not isinstance(strays[name], kind), -likeness(str(name), key)))
        for name in order:
            shaped = conform(spec, strays[name])
            if shaped is UNFIT:
                continue
            fitted[key] = shaped
            strays.pop(name)
            return True
        return False

    def supply_args(self, tool, have, missing, refusal=None):
        props = _params(self.forms.get(tool) or {})
        ending = not set(_FINISHERS).isdisjoint(self.jobs_of(tool))
        given, unknown = {}, []
        for key in missing:
            spec = props.get(key) or {}
            if (tool, key) in self.filled:
                given[key] = self.filled[(tool, key)]
            elif "default" in spec:
                given[key] = spec["default"]
            elif not (ending and isinstance(spec.get("enum"), list)):
                unknown.append(key)
        if unknown:
            given.update(self._guess_args(tool, have, refusal, props, unknown))
            self.e.log("args_guessed", args=json_cut(given, Dials.Trace.long), tool=tool)
        return given

    def _guess_args(self, tool, have, refusal, props, unknown):
        reply = (self.consult_args(tool, have, refusal) or {}).get("arguments")
        reply = reply if isinstance(reply, dict) else {}
        guessed = {}
        for key in unknown:
            spec = props.get(key) or {}
            value = UNFIT if reply.get(key) is None else conform(spec, reply[key])
            if value is UNFIT:
                value = simplest(spec)
            if value is not UNFIT:
                guessed[key] = self.filled[(tool, key)] = value
        return guessed

    def consult_args(self, tool, have, refusal):
        brief = brief_arg_fix(tool=tool, description=self.describe(tool), schema=self.forms.get(tool) or {},
                              have=have, uses=self.jobs_of(tool), refusal=refusal)
        answer = self.e.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.arg_fix)
        return answer if isinstance(answer, dict) else None

    def _may_mend(self, tool):
        done = self.mends.get(tool, 0)
        if done >= Dials.Args.repairs_per_tool or self.steps_free() < Dials.Args.repair_steps:
            return False
        self.mends[tool] = done + 1
        return True

    def mend_args(self, tool, args, refusal):
        if not self._may_mend(tool):
            return None
        said = json_cut(refusal, Dials.Args.refusal_cut)
        named = [key for key in _params(self.forms.get(tool) or {}) if key not in args and mentions_arg(said, key)]
        self.e.log("args_rejected", refusal=said[:Dials.Trace.long], args=json_cut(args, Dials.Trace.long), tool=tool)
        if named:
            return {**args, **self.supply_args(tool, args, named, refusal)}
        answer = (self.consult_args(tool, args, refusal) or {}).get("arguments")
        return self.shape_args(tool, answer) if isinstance(answer, dict) and answer else None

    def learn_keys(self, kind, record):
        key = "%s:%s" % (kind, ",".join(sorted(record)))
        known = self.key_maps.get(key)
        if known is None:
            known = self._borrow_map(kind, key, record)
        if known is not None and not self._second_look(known, key, record):
            return known
        roles = dict(known or {})
        self._take_answer(roles, record)
        self._fill_by_shape(roles, record)
        self.key_maps[key] = roles
        self.e.keep_lore()
        return roles

    def _second_look(self, known, key, record):
        mapped = set(known.values())
        fresh = [field for field, value in record.items()
                 if field not in mapped and (field, key) not in self.rekeyed and self._holds_text(value)]
        if not fresh or all(role in known for role in ("notes_text", "description", "property_table")):
            return False
        self.rekeyed.update((field, key) for field in fresh)
        return True

    def _borrow_map(self, kind, key, record):
        prefix = kind + ":"
        for other_key, other in self.key_maps.items():
            if not other or not other_key.startswith(prefix):
                continue
            present = sum(field in record for field in other.values())
            if present >= max(Dials.Keys.reuse_min, int(Dials.Keys.reuse_share * len(other))):
                self.key_maps[key] = other
                return other
        return None

    @staticmethod
    def _holds_text(value):
        if isinstance(value, str):
            return len(value) > Dials.Keys.fresh_text_over
        return isinstance(value, dict) and bool(value)

    def _take_answer(self, roles, record):
        answer = self.e.oracle.ask_json(Dials.Model.light_spare, brief_key_roles(record), Dials.Tokens.key_roles)
        if not isinstance(answer, dict):
            return
        for role, field in answer.items():
            if role not in roles and isinstance(field, str) and field in record:
                roles[role] = field

    @staticmethod
    def _fill_by_shape(roles, record):
        for role, test in (("property_table", _scalar_table), ("listing_id", _id_value), ("unit_price", _price_value)):
            if role in roles:
                continue
            found = _first(record, lambda field: test(field, record[field]))
            if found is not None:
                roles[role] = found

    def offer_from(self, record, kind):
        roles = self.learn_keys(kind, record)
        id_key = roles.get("listing_id")
        ident = record.get(id_key) if id_key else None
        if not isinstance(ident, (str, int)):
            return None
        offer = Offer(str(ident))
        offer.blob = record
        text = lambda role: _text_value(record, roles.get(role))
        offer.headline, offer.blurb, offer.shelf = text("listing_name"), text("description"), text("category_label")
        offer.maker, offer.vendor, offer.remarks = text("brand_name"), text("seller_name"), text("notes_text")
        if roles.get("unit_price"):
            offer.cost = read_number(record.get(roles["unit_price"]))
        stock = record.get(roles["in_stock"]) if roles.get("in_stock") else None
        if isinstance(stock, bool):
            offer.in_stock = stock
        table = record.get(roles["property_table"]) if roles.get("property_table") else None
        if isinstance(table, dict):
            offer.traits = {str(name): value for name, value in table.items() if value not in (None, "")}
        return offer


"""The request: what the shopper wants (Wanted), each firm wish (Need), each listing seen (Offer), and Order,
which reads the request once and holds its settings."""

FROM_REQUEST, FROM_SHOPPER, FROM_RULE, FROM_FIT = "stated", "heard", "rule", "pairing"
DEFAULT_PHRASE = "the item"
_JUDGE_ONLY = ("faint", "indirect")


def blank_ruling(full=False):
    ruling = dict(missed=set(), doubtful=set())
    for part in _JUDGE_ONLY if full else ():
        ruling[part] = set()
    return ruling


def _clean(value, fallback=""):
    return str(value or fallback).strip()


class Need:

    def __init__(self, trait, rel, want, link=None, source=FROM_REQUEST):
        self.trait, self.want = _clean(trait), _clean(want)
        self.rel = _clean(rel, "is").lower()
        self.link = link if isinstance(link, dict) else None
        self.source, self.renamed = source, False
        self.link_first = self.link_last = None

    def waiting(self):
        return self.rel in Dials.Ops.borrowed and bool(self.link)

    def line(self):
        tail = "(linked value unread)" if self.waiting() else self.want
        return "%s: %s %s" % (self.trait, self.rel, tail)

    def slug(self):
        return simplify(self.trait)

    def move_to(self, prop):
        self.trait, self.renamed = prop, True

    def pin_first(self, mark):
        self.link_first = self.link_first or mark

    def relink(self, link):
        self.link = link

    def settle(self, rel, want, mark):
        self.rel, self.want, self.link, self.link_last = rel, want, None, mark

    def set_rel(self, rel):
        self.rel = rel


class Offer:

    def __init__(self, oid):
        self.oid = oid
        self.headline = self.blurb = self.shelf = self.maker = self.vendor = self.remarks = ""
        self.cost = self.in_stock = None
        self.traits, self.blob = {}, {}
        self.read_full = self.page_seen = False
        self.sift_mark = "maybe"


def _alt_wordsets(value):
    found = []
    for alt in or_parts(value):
        words = wordset(alt)
        if words:
            found.append(frozenset(words))
    return found


def _listed_wordsets(values):
    return {frozenset(words) for words in (wordset(bit) for value in values for bit in value_bits(value)) if words}


def _unit_count(units):
    if _plain_number(units) and units >= Dials.Request.min_quantity:
        return int(units)
    return Dials.Request.min_quantity


def _needs_from(raw):
    for item in raw or ():
        if isinstance(item, dict) and item.get("property_name"):
            yield Need(trait=item.get("property_name"), rel=item.get("comparison"), want=item.get("required_value"),
                       link=item.get("other_page"))


def _value_forms(value):
    for alt in or_parts(value):
        yield alt
        if "-" in alt:
            yield alt.replace("-", " ")


class Wanted:

    def __init__(self, slot, phrase="", kind="", queries=None):
        self.slot = slot
        self.phrase = str(phrase or kind or DEFAULT_PHRASE).strip()
        self.kind = str(kind or self.phrase).strip()
        listed = [str(q) for q in queries or () if q]
        self.queries = listed if listed else [self.kind]
        self.headword = self.queries[-1]
        self.count, self.needs, self.clues, self.unit_cap = 1, [], [], None
        self.found, self.rulings, self.judged = {}, {}, {}
        self.pages_run = 0

    @classmethod
    def from_entry(cls, entry, slot):
        made = cls(slot, phrase=entry.get("product_name"), kind=entry.get("kind"), queries=entry.get("queries"))
        made.count = _unit_count(entry.get("units"))
        made.needs = list(_needs_from(entry.get("needs")))
        made.clues = [str(h) for h in entry.get("hint_properties") or () if h]
        cap = read_number(entry.get("unit_price_limit"))
        made.unit_cap = cap if cap is None or not cap <= 0 else None
        return made

    def is_scratch(self):
        return self.slot == Dials.Lookup.scratch_index

    def count_page(self):
        self.pages_run += 1

    def absorb(self, offer, opened):
        held = self.found.setdefault(offer.oid, offer)
        if held is not offer:
            self._fold(held, offer, opened)
        if opened:
            held.read_full = True
        return held

    def _fold(self, held, new, opened):
        pick = (lambda old, fresh: fresh if fresh and (opened or not old) else old)
        held.headline, held.blurb = pick(held.headline, new.headline), pick(held.blurb, new.blurb)
        held.shelf, held.maker = pick(held.shelf, new.shelf), pick(held.maker, new.maker)
        held.vendor, held.remarks = pick(held.vendor, new.vendor), pick(held.remarks, new.remarks)
        held.cost = held.cost if new.cost is None else new.cost
        held.in_stock = held.in_stock if new.in_stock is None else new.in_stock
        if new.traits:
            if held.traits != new.traits:
                self.forget_judged(held.oid)
            held.traits = new.traits
        held.page_seen, held.blob = held.page_seen or new.page_seen, new.blob or held.blob

    def queue_query(self, q):
        if q and simplify(q) not in {simplify(t) for t in self.queries}:
            spot = min(self.pages_run, len(self.queries))
            self.queries.insert(spot, q[:Dials.Query.query_cut])

    def append_query(self, q):
        self.queries.append(q)

    def live_needs(self):
        return list(filter(lambda need: not need.waiting(), self.needs))

    def need_lines(self):
        return list(map(Need.line, self.live_needs()))

    def add_need(self, need):
        self.needs.append(need)

    def drop_needs(self, test):
        kept = []
        for need in self.needs:
            if not test(need):
                kept.append(need)
        self.needs = kept

    def swap_needs(self, source, new):
        self.drop_needs(lambda need: need.source == source)
        self.needs.extend(new)

    def _plain_value_needs(self):
        return [need for need in self.live_needs() if need.want and need.rel in Dials.Ops.exact]

    def ruling(self, oid):
        return self.rulings.get(oid)

    def judged_lines(self, oid):
        return self.judged.get(oid)

    def set_ruling(self, oid, ruling, against=None):
        self.rulings[oid] = ruling
        if against is None:
            return
        self.judged[oid] = set(against)

    def flag_miss(self, oid, line):
        target = self.rulings.get(oid)
        if target is None:
            target = self.rulings[oid] = blank_ruling()
        target["missed"].add(line)

    def unflag(self, oid, line):
        ruling = self.rulings.get(oid) or {}
        for part in ("doubtful", "faint", "missed"):
            ruling.get(part, set()).discard(line)

    def drop_misses(self, test):
        for ruling in self.rulings.values():
            ruling["missed"] = set(filter(lambda line: not test(line), ruling["missed"]))

    def forget_judged(self, oid):
        self.judged.pop(oid, None)

    def too_dear(self, offer):
        return over_limit(offer.cost, self.unit_cap)

    @staticmethod
    def _own_values(offer, need):
        return [value for key, value in offer.traits.items() if names_match(key, need.trait)]

    def _buried(self, offer, need):
        forms = _alt_wordsets(need.want)
        own = self._own_values(offer, need)
        return bool(forms) and bool(own) and _listed_wordsets(own).isdisjoint(forms)

    def buried_count(self, offer):
        return sum(1 for need in self._plain_value_needs() if self._buried(offer, need))

    def exact_hits(self, offer):
        if not offer.traits:
            return 0
        pieces = _listed_wordsets(offer.traits.values())
        return sum(1 for need in self._plain_value_needs() if not pieces.isdisjoint(_alt_wordsets(need.want)))

    @staticmethod
    def own_word_match(offer, need):
        words = set()
        for value in Wanted._own_values(offer, need):
            words.update(wordset(str(value)))
        return any(words & wordset(alt) for alt in or_parts(need.want))

    def _by_other_word(self, need, indirect, offer):
        if need.line() not in indirect or need.rel not in Dials.Ops.exact or not need.want:
            return False
        if has_digit(need.want) or need.source == FROM_FIT:
            return False
        return offer is None or not self.own_word_match(offer, need)

    def synonym_lines(self, oid):
        indirect = set((self.rulings.get(oid) or {}).get("indirect") or ())
        if not indirect:
            return []
        offer = self.found.get(oid)
        return [need.line() for need in self.live_needs() if self._by_other_word(need, indirect, offer)]

    def _counts(self, offer, ruling, firm_only):
        if not offer.read_full or ruling is None or ruling["missed"] or self.too_dear(offer):
            return False
        return not (firm_only and ruling.get("faint"))

    def _split_fits(self, firm_only):
        lines = set(self.need_lines())
        sure, doubtful = [], []
        for oid, offer in self.found.items():
            ruling = self.rulings.get(oid)
            if self._counts(offer, ruling, firm_only) and lines <= self.judged.get(oid, set()):
                unsure = ruling["doubtful"] or self.synonym_lines(oid)
                (doubtful if unsure else sure).append(offer)
        return sure, doubtful

    def _cheap_rank(self, offer):
        return self.buried_count(offer) > 0, price_rank(offer.cost)

    def _evidence_rank(self, offer):
        faint = self.rulings[offer.oid].get("faint") or ()
        return -self.exact_hits(offer), len(faint), price_rank(offer.cost)

    def _doubt_rank(self, offer):
        doubts = len(self.rulings[offer.oid]["doubtful"]) + len(self.synonym_lines(offer.oid))
        return doubts, price_rank(offer.cost)

    def fitting(self, with_doubts=False, cheapest_first=False, firm_only=False):
        sure, doubtful = self._split_fits(firm_only)
        if with_doubts and not sure:
            return sorted(doubtful, key=self._doubt_rank)
        return sorted(sure, key=self._cheap_rank if cheapest_first else self._evidence_rank)


class Order:

    def __init__(self, errand):
        self.e = errand
        view = errand.task_view
        self.text, self.first_view = str(view.get("query") or ""), view.get("opening")
        self.wants, self.raise_topics, self.match_rules = [], [], []
        self.budget = self.measure = None
        self.lowest_price = self.closest_ok = self.says_decline = False
        self.buyable_only = True

    def set_budget(self, value):
        self.budget = value

    def read_request(self):
        got = self._ask_reader()
        if not isinstance(got, dict):
            self.e.log("read_failed")
            return False
        self.e.counter.adopt_jobs(got.get("tool_roles"))
        self.wants = self._wants_from(got.get("products"))
        for want in self.wants:
            for tidy in (self._strip_link_leaks, self._strip_kind_values, self._add_value_queries):
                tidy(want)
        messages = self._take_settings(got)
        self._log_reading(messages)
        return True

    def _log_reading(self, messages):
        rows = []
        for want in self.wants:
            row = (want.phrase, want.count, want.need_lines(), want.clues)
            if want.unit_cap is not None:
                row += (want.unit_cap,)
            rows.append(row)
        self.e.log("reading", steps=self.e.counter.steps_free(), messages=messages, nearest=self.closest_ok,
                   cheapest=self.lowest_price, budget=self.budget, wants=rows)

    def _ask_reader(self):
        oracle = self.e.oracle
        brief = brief_intake(request=self.text, opening=self.first_view, tools=self.e.counter.tool_cards())
        first = oracle.ask_json(Dials.Model.reader_spare, brief, Dials.Tokens.intake)
        roles = first.get("tool_roles") if isinstance(first, dict) else None
        if isinstance(roles, dict):
            return first
        return oracle.ask_json(Dials.Model.reader_light, brief, Dials.Tokens.intake)

    def _wants_from(self, raw):
        made = []
        for at, entry in enumerate(raw if isinstance(raw, list) else ()):
            if isinstance(entry, dict):
                made.append(Wanted.from_entry(entry, at))
        return made or [Wanted(0, DEFAULT_PHRASE, "", [self.text[:Dials.Request.request_term_cut]])]

    @staticmethod
    def _strip_link_leaks(want):
        leaked = set()
        for need in want.needs:
            values = need.link.get("identifying_values") if need.link else None
            for name, value in (values.items() if isinstance(values, dict) else ()):
                leaked.add((simplify(name), simplify(value)))
        if not leaked:
            return
        want.drop_needs(lambda n: not n.link and (simplify(n.trait), simplify(n.want)) in leaked)
        clues = {value for _, value in leaked if value}
        kept = [q for q in want.queries if not any(clue in simplify(q) for clue in clues)]
        want.queries = kept or [want.kind]

    @staticmethod
    def _strip_kind_values(want):
        kind_words = wordset(want.kind)

        def names_kind(need):
            words = wordset(need.want)
            return not need.link and need.rel == "is" and bool(words) and words <= kind_words

        want.drop_needs(names_kind)

    @staticmethod
    def _add_value_queries(want):
        for need in want.needs:
            if not need.want or need.rel not in Dials.Ops.with_value:
                continue
            for form in _value_forms(need.want):
                fits = form and len(form) <= Dials.Query.value_term_max
                if fits and simplify(form) not in {simplify(q) for q in want.queries}:
                    want.append_query(form)

    def _take_settings(self, got):
        action = str(got.get("no_fit_action") or "").strip().lower()
        self.budget = read_number(got.get("total_limit"))
        self.lowest_price = bool(got.get("want_cheapest"))
        self.closest_ok, self.says_decline = action == "nearest", action == "decline" or bool(got.get("no_fit_words"))
        best = got.get("best_by")
        self.measure = best if best and isinstance(best, str) else None
        allowed = got.get("messages_allowed")
        messages = int(allowed) if _plain_number(allowed) and allowed >= 0 else None
        self.e.counter.cap_steps(got.get("steps_allowed"))
        self.buyable_only = bool(got.get("buyable_only", True))
        self.raise_topics = [str(t) for t in got.get("raise_topics") or () if t]
        self.match_rules = list(filter(lambda r: isinstance(r, dict), got.get("must_fit") or ()))
        self.e.picker.load_rules(got.get("cross_needs"))
        self.e.parley.adopt(messages, got.get("process"))
        return messages


"""Scout: searching the store, value queries, kind labels, the shortlist and the cheap row sift."""

_WORDS_JOB = "keyword_search"
_FIELDS_JOB = "field_search"


def _reply_items(value):
    return [] if not value or isinstance(value, (int, float)) else value


def _report(errand, kind, fields):
    errand.log(kind, **fields)


def _shape(text):
    return frozenset(wordset_full(text))


class _Spot:
    __slots__ = ("at", "token", "ended")

    def __init__(self, at, token, ended):
        self.at, self.token, self.ended = at, token, ended


_NEW_SPOT = _Spot(0, None, False)


class _Ledger:

    def __init__(self, owner):
        self.owner = owner
        self.text_spots, self.field_spots, self.shown_by = {}, {}, {}
        self.value_seen, self.triaged = set(), set()
        self.value_tried = []
        self.values_done = self.page_capped = False
        self.kind_labels = None


class Scout:

    def __init__(self, errand):
        self.e = errand
        self.every_word = errand.lore.get("every_word")
        self.match_counts, self.ledgers = {}, {}
        self.widest_page, self.full_rows = 0, False

    def _ledger(self, w):
        book = self.ledgers.get(id(w))
        if book is not None and book.owner is w:
            return book
        self.ledgers[id(w)] = fresh = _Ledger(w)
        return fresh

    def seek(self, w, text=None, fields=None, pages=1, floor=None):
        counter = self.e.counter
        role = _FIELDS_JOB if text is None else _WORDS_JOB
        if not counter.job(role):
            return 0
        book = self._ledger(w)
        if text is None:
            spots, key = book.field_spots, keyed_json(fields or {})
        else:
            spots, key = book.text_spots, text
        spot = spots.get(key, _NEW_SPOT)
        how = counter.paging(role)
        if spot.ended or (spot.at > 0 and how is None):
            return 0
        words_arg = None if text is None else counter.job_field(role, "words_arg")
        page, token = spot.at, spot.token
        fresh, ended = 0, False
        while page < spot.at + pages:
            if self._halted(floor):
                break
            if text is not None and not words_arg:
                return 0
            args = self._call_args(w, role, text, fields, words_arg)
            if args is None:
                spots[key] = _Spot(0, None, True)
                return fresh
            ahead = page + 1
            if ahead > 1 and not self._put_page(role, how, ahead, token, args, book):
                ended = True
                break
            outcome = counter.act(counter.tool_for(role), args)
            page = ahead
            if not outcome.ok:
                _report(self.e, "seek_failed", {"error": outcome.fault})
                ended = True
                break
            rows, token, total = self._page_facts(outcome.body)
            if text is not None and page == 1:
                self._count_first_page(args[words_arg], rows, total)
            fresh += self._absorb_page(w, book, rows, text)
            ended = self._page_ends(book, how, page, rows, token, total)
            if ended:
                break
        spots[key] = _Spot(max(page, 0), token, ended or how is None)
        return fresh

    def _halted(self, floor):
        return self.e.counter.steps_free() <= (floor or 0) or self.e.seconds_left() < Dials.Timing.search_until

    def _call_args(self, w, role, text, fields, words_arg):
        counter = self.e.counter
        args = {}
        switch = counter.job_field(role, "switch")
        name = switch.get("arg_name") if isinstance(switch, dict) else None
        if name:
            args[name] = switch.get("arg_value")
        if text is not None:
            args[words_arg] = self._sent_form(w, role, words_arg, text)
            return args
        for slot in fields or {}:
            value, arg = fields[slot], counter.job_field(role, slot)
            if value in (None, "") or not arg:
                continue
            shown = counter.listed_value(role, arg, value)
            if shown is None:
                return None
            if isinstance(shown, str):
                shown = counter.clip_arg(role, arg, shown)
            args[arg] = shown
        return args

    def _put_page(self, role, how, page, token, args, book):
        counter = self.e.counter
        pick = {"number": lambda: page, "offset": lambda: max(1, self.widest_page) * (page - 1)}
        if how in pick:
            value = pick[how]()
        elif how == "token" and token:
            value = token
        else:
            return False
        page_arg = counter.job_field(role, "page_arg")
        args[page_arg] = value
        top = counter.arg_form(role, page_arg).get("maximum")
        if isinstance(top, (int, float)) and isinstance(value, int) and top < value:
            book.page_capped = True
            return False
        return True

    @staticmethod
    def _page_facts(body):
        token, counts = None, []
        for value in body.values() if isinstance(body, dict) else ():
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                counts.append(value)
            elif isinstance(value, str) and " " not in value and 0 < len(value) <= Dials.Query.token_max:
                token = value
        return row_list(body) or [], token, max(counts) if counts else None

    def _absorb_page(self, w, book, rows, text):
        if len(rows) > self.widest_page:
            self.widest_page = len(rows)
        listed = list(filter(None, map(lambda rec: self.e.counter.offer_from(rec, "row"), rows)))
        fresh = 0
        for offer in listed:
            fresh += offer.oid not in w.found
            w.absorb(offer, False)
        if text is not None and listed:
            book.shown_by.setdefault(text, set()).update(o.oid for o in listed)
        if len(listed) >= Dials.Query.full_rows_min:
            self.full_rows = not any(o.cost is None or o.in_stock is None for o in listed)
        w.count_page()
        if book.kind_labels is None and not w.is_scratch():
            self._settle_labels(w)
        return fresh

    def _count_first_page(self, sent, rows, total):
        if total is None and rows:
            return
        self._learn_rule(sent, 0 if total is None or not rows else total)

    def _page_ends(self, book, how, page, rows, token, total):
        if not rows or len(rows) < self.widest_page:
            return True
        if total is not None and total <= page * len(rows):
            return True
        if how != "token" or token:
            return False
        if total is not None and page * len(rows) < total:
            book.page_capped = True
        return True

    @staticmethod
    def _pair_rule(mine, n_mine, other, n_other):
        if len(other) < len(mine):
            (small, n_small), (big, n_big) = (other, n_other), (mine, n_mine)
        else:
            (small, n_small), (big, n_big) = (mine, n_mine), (other, n_other)
        if len(small) == len(big) or not small & big:
            return None, n_small, n_big
        inside = small < big
        if inside and n_big > n_small:
            return False, n_small, n_big
        far_fewer = n_big * Dials.Query.count_ratio_base < n_small * Dials.Query.count_ratio_top
        wider = inside or len(big) >= len(small) + Dials.Query.more_words
        return (True if far_fewer and wider else None), n_small, n_big

    def _learn_rule(self, sent, count):
        mine = _shape(sent)
        if not mine:
            return
        for other in list(self.match_counts):
            rule, n_small, n_big = self._pair_rule(mine, count, _shape(other), self.match_counts[other])
            if rule is None or rule is self.every_word:
                continue
            self.every_word = rule
            _report(self.e, "match_rule", {"every_word": rule, "short_count": n_small, "long_count": n_big})
        self.match_counts[sent] = count

    def _sent_form(self, w, role, arg, query):
        kind_words = wordset(w.kind) | wordset(w.phrase)
        values, kinds = [], []
        for word in query.split():
            own = wordset(word)
            (kinds if own and own <= kind_words else values).append(word)
        ordered = " ".join(values + kinds)
        limit = self.e.counter.arg_form(role, arg).get("maxLength")
        if isinstance(limit, int) and len(ordered) > limit:
            kept = values
            for word in kinds:
                trial = kept + [word]
                if len(" ".join(trial)) <= limit:
                    kept = trial
            return self.e.counter.clip_arg(role, arg, " ".join(kept) or ordered)
        return ordered

    def _sent(self, w, query):
        arg = self.e.counter.job_field(_WORDS_JOB, "words_arg") or ""
        return self._sent_form(w, _WORDS_JOB, arg, query)

    @staticmethod
    def value_choices(w):
        heard, others = [], []
        for need in w.live_needs():
            if not need.want or need.rel not in Dials.Ops.with_value:
                continue
            options = [p for p in or_parts(need.want) if wordset(p) and not has_part_code(p)]
            fits = 0 < len(options) <= Dials.Query.group_alts_max
            if fits and max(map(len, options)) <= Dials.Query.group_alt_chars:
                (heard if need.source == FROM_SHOPPER else others).append(options)
        return heard + others

    @staticmethod
    def values_carried(o, choices):
        head = " ".join((o.headline, o.blurb or "", o.shelf, o.maker, o.vendor))
        table = " ".join(map("%s %s".__mod__, o.traits.items()))
        carried = wordset_full(head + " " + table)
        return sum(any(wordset_full(p) <= carried for p in options) for options in choices)

    def _value_query_plan(self, w):
        choices = self.value_choices(w)
        if not choices:
            return []
        noun = w.kind if len(w.headword) > Dials.Query.noun_chars else w.headword
        tail = [noun] if self.every_word is not False else []
        leads = [first for first, *_ in choices]
        plan = {}
        for at, options in enumerate(choices):
            company = leads[:at] + leads[at + 1:]
            for option in options[:Dials.Query.alts_per_group]:
                query = " ".join(([option] + company)[:Dials.Query.phrases_per_query] + tail)
                plan.setdefault(_shape(query), query)
            if len(plan) >= Dials.Query.value_queries_enough:
                break
        return list(plan.values())

    def _unsent_value_queries(self, w):
        book = self._ledger(w)
        sent = [_shape(q) for q in (*book.text_spots, *book.value_tried)]
        return [q for q in self._value_query_plan(w) if all(not _shape(q).issubset(s) for s in sent)]

    def _value_pages(self, book):
        if book.value_tried:
            return Dials.Query.value_pages_all_words if self.every_word else Dials.Query.value_pages_any_word
        if len(self.e.order.wants) == 1:
            return Dials.Query.first_value_pages_one
        return Dials.Query.first_value_pages_many

    def try_value_query(self, w, floor):
        counter = self.e.counter
        spare = floor + Dials.Query.value_spare
        todo = self._unsent_value_queries(w)
        if counter.steps_free() <= spare or not todo:
            return False
        book, query = self._ledger(w), todo[0]
        pages = self._value_pages(book)
        book.value_tried.append(query)
        _report(self.e, "value_probe", {"item": w.phrase, "query": query})
        used_before = counter.used
        self.seek(w, query, None, pages, floor)
        self._note_value_rows(w, query)
        if self.match_counts.get(self._sent(w, query)) == 0 and counter.steps_free() > spare:
            self._shorter_tries(w, book, query, floor, spare)
        return counter.used > used_before

    def _shorter_tries(self, w, book, query, floor, spare):
        counter = self.e.counter
        noun_words = wordset(w.headword)
        bare = " ".join(x for x in query.split() if not (wordset(x) and wordset(x) <= noun_words))
        tries = (bare, self.value_choices(w)[0][0] + " " + w.headword)
        for shorter in tries:
            usable = shorter and simplify(shorter) != simplify(query) and counter.steps_free() > spare
            if not usable or self._sent(w, shorter) in self.match_counts:
                continue
            book.value_tried.append(shorter)
            self.seek(w, shorter, None, Dials.Query.shorter_pages, floor)
            self._note_value_rows(w, shorter)
            if self.match_counts.get(self._sent(w, shorter)):
                return

    def _note_value_rows(self, w, query):
        book = self._ledger(w)
        book.value_seen |= book.shown_by.get(query) or set()
        noun_words = wordset_full(w.headword)

        def whole(q):
            sent = self._sent(w, q)
            count = self.match_counts.get(sent)
            spot = book.text_spots.get(q)
            if not count or spot is None or not spot.ended:
                return False
            return len(book.shown_by.get(q, ())) >= count and wordset_full(sent) >= wordset_full(q) - noun_words

        plan = self._value_query_plan(w)
        if plan and all(map(whole, plan)):
            book.values_done = True

    def right_kind(self, w, o):
        if not (o.shelf and w.kind):
            return True
        label, accepted = simplify(o.shelf), self._ledger(w).kind_labels
        if accepted:
            return label in accepted
        want = simplify(w.kind)
        return label in want or want in label or likeness(label, want) >= Dials.Like.same

    def _settle_labels(self, w):
        book = self._ledger(w)
        if book.kind_labels is not None:
            return
        labels = sorted(set(filter(None, (o.shelf for o in w.found.values()))))
        if not labels:
            return
        plain = set(map(simplify, labels))
        if len(labels) > 1:
            self._ask_labels(w, book, labels, plain)
            return
        only, want = simplify(labels[0]), simplify(w.kind)
        if want in only or only in want or likeness(labels[0], w.kind) >= Dials.Like.label:
            book.kind_labels = plain

    def _ask_labels(self, w, book, labels, plain):
        brief = brief_kind_labels(w.phrase, w.kind, self.e.order.text, labels)
        answer = self.e.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.kind_labels)
        named = answer.get("matching_labels") if isinstance(answer, dict) else ()
        if isinstance(named, list):
            chosen = set(simplify(x) for x in named if isinstance(x, str)) & plain
            if chosen:
                book.kind_labels = chosen
                _report(self.e, "kind_labels", {"item": w.phrase, "labels": sorted(chosen)})

    def _listable(self, w, o):
        order = self.e.order
        if order.buyable_only and o.in_stock is False:
            return False
        if o.cost is not None and over_limit(o.cost * w.count, order.budget):
            return False
        return self.right_kind(w, o)

    def shortlist(self, w):
        return sorted(filter(lambda o: self._listable(w, o), w.found.values()),
                      key=lambda o: (o.sift_mark != "yes", o.sift_mark == "no", price_rank(o.cost)))

    @staticmethod
    def _row_number(entry, size):
        if not isinstance(entry, (int, str)) or not str(entry).strip().isdigit():
            return None
        try:
            number = int(entry)
        except ValueError:
            return None
        return number if 1 <= number <= size else None

    def triage(self, w, offers):
        book = self._ledger(w)
        fresh = [o for o in offers if o.sift_mark == "maybe" and o.oid not in book.triaged and not o.read_full]
        lines = w.need_lines()
        if not lines or len(fresh) < Dials.Screen.min_rows:
            return
        sample = fresh[:Dials.Screen.batch]
        answer = self.e.oracle.ask_json(Dials.Model.light_spare, brief_sift(w.kind, lines, sample), Dials.Tokens.sift)
        book.triaged.update(map(lambda o: o.oid, sample))
        if isinstance(answer, dict):
            for mark, key in (("no", "ruled_out"), ("yes", "could_fit")):
                for number in map(lambda x: self._row_number(x, len(sample)), _reply_items(answer.get(key))):
                    if number is not None:
                        sample[number - 1].sift_mark = mark

    def cursor_state(self, w, text=None, fields=None):
        book = self._ledger(w)
        found = book.field_spots.get(keyed_json(fields or {})) if text is None else book.text_spots.get(text)
        return None if found is None else ("ended" if found.ended else "open")

    def ran_queries(self, w):
        return list(self._ledger(w).text_spots)

    def queries_showing(self, w, oid):
        shown = self._ledger(w).shown_by
        return list(filter(lambda q: oid in shown[q], shown))

    def in_value_rows(self, w, oid):
        return oid in self._ledger(w).value_seen

    def values_complete(self, w):
        return self._ledger(w).values_done

    def forget_values(self, w):
        book = self._ledger(w)
        book.value_seen, book.values_done = set(), False

    def page_limit_hit(self, w):
        return self._ledger(w).page_capped

    def accepted_labels(self, w):
        held = self._ledger(w).kind_labels
        return frozenset(held) if held is not None else None

    def match_rule(self):
        return self.every_word

    def rows_priced(self):
        return self.full_rows


"""Assessor: reading listings in full, judging them against the needs, the store's own wording, the final check."""

_SHEET_JOB = "spec_sheet"
_PAGE_JOB = "open_one"
_SIDE_JOB = "open_several"
_STOP, _NEXT, _FALL = "stop", "next", "fall"


def _answer_items(value):
    return [] if not value or isinstance(value, (int, float)) else value


def _nearest_line(name, lines):
    if not lines:
        return name
    score, _, line = max((likeness(text, name), -at, text) for at, text in enumerate(lines))
    return line if score > Dials.Like.loose else name


def _sheet_offer(oid, traits):
    offer = Offer(oid)
    offer.traits = traits
    return offer


def _forms_of(value):
    return list(filter(None, map(lambda part: frozenset(wordset(part)), or_parts(value))))


def _bit_sets(value):
    return list(filter(None, map(lambda bit: frozenset(wordset(bit)), value_bits(value))))


def _codes_in(*texts):
    return set(title_codes(" ".join(t or "" for t in texts)))


class Assessor:

    def __init__(self, errand):
        self.e = errand
        lore = errand.lore
        self.side_limit, self.side_thin = int(lore.get("batch_size") or Dials.Read.compare_default), \
            bool(lore.get("thin_batch"))
        self.side_checked = self.side_off = False
        self.approvals = {}

    def compare_size(self):
        return self.side_limit

    def side_rows_thin(self):
        return self.side_thin

    def side_usable(self):
        return not self.side_off and bool(self.e.counter.job(_SIDE_JOB))

    def batch_reads(self):
        return not self.side_thin and self.side_usable()

    def table_first(self):
        counter = self.e.counter
        both = counter.job(_SHEET_JOB) and counter.job(_PAGE_JOB)
        return bool(both) and self.e.scout.rows_priced() and not self.side_usable()

    def _table_only(self, w):
        if w is None or not self.table_first():
            return False
        if any(w is addon or w is host for addon, host in self.e.picker.pairings()):
            return False
        return all(not need.link for need in w.needs)

    def _short(self, floor, cost):
        return bool(floor) and self.e.counter.steps_free() - cost < floor

    def read_listings(self, w, ids, floor=0, own_page=False):
        todo = list(filter(None, ids))
        found = []
        if not todo:
            return found
        counter = self.e.counter
        side = counter.job(_SIDE_JOB) if not self.side_off else None
        if side and not self.side_checked:
            self._learn_side_cap(side)
        sheet, page_job = counter.job(_SHEET_JOB), counter.job(_PAGE_JOB)
        tables_only = False if own_page else self._table_only(w)
        sent_back = set()
        while todo and counter.steps_free() > 0 and self.e.seconds_left() > Dials.Timing.open_until:
            if tables_only or (self.side_thin and sheet and w is not None):
                if self._short(floor, Dials.Read.page_cost):
                    break
                self._keep(found, self._read_by_table(w, todo.pop(0)))
                continue
            if side and len(todo) >= Dials.Read.compare_min:
                step = self._side_round(w, side, sheet, todo, floor, found, sent_back)
                if step == _STOP:
                    break
                if step == _NEXT:
                    side = None if self.side_off else side
                    continue
            if not page_job or "id_arg" not in page_job:
                break
            if self._short(floor, Dials.Read.page_cost_with_table if sheet else Dials.Read.page_cost):
                break
            self._keep(found, self._read_page(w, page_job, sheet, todo.pop(0)))
        return found

    @staticmethod
    def _keep(found, offer):
        if offer is not None:
            found.append(offer)

    def _learn_side_cap(self, side):
        self.side_checked = True
        most = self.e.counter.arg_form(_SIDE_JOB, side.get("ids_arg") or "").get("maxItems")
        if isinstance(most, int) and Dials.Read.compare_min <= most < self.side_limit:
            self.side_limit = most

    def _side_round(self, w, side, sheet, todo, floor, found, sent_back):
        if "ids_arg" not in side:
            self.side_off = True
            return _NEXT
        if self._short(floor, Dials.Read.page_cost):
            return _STOP
        chunk = todo[:self.side_limit]
        result, rows = self._side_call(side, chunk)
        if result == "odd":
            return _FALL
        if result == "failed":
            self._side_failed(len(chunk))
            return _NEXT
        found.extend(self._keep_side_rows(w, rows, sheet, floor))
        del todo[:len(chunk)]
        self._send_back(todo, chunk, rows, sent_back)
        return _NEXT

    def _side_failed(self, sent):
        floor_size = Dials.Read.compare_min
        if sent > floor_size and self.side_limit > floor_size:
            self.side_limit = max(floor_size, min(self.side_limit, sent) // Dials.Read.compare_shrink)
            self.e.keep_lore()
        else:
            self.side_off = True

    def _send_back(self, todo, chunk, rows, sent_back):
        answered = set(map(lambda o: o.oid, rows))
        if not rows or len(answered) >= len(chunk):
            return
        missing = [x for x in chunk if not (x in answered or x in sent_back)]
        if not missing:
            return
        if Dials.Read.compare_min <= len(answered) < self.side_limit:
            self.side_limit = len(answered)
            self.e.keep_lore()
        sent_back.update(missing)
        todo[0:0] = missing

    def _side_call(self, side, chunk):
        counter = self.e.counter
        outcome = counter.act(counter.tool_for(_SIDE_JOB), {side["ids_arg"]: chunk})
        records = row_list(outcome.body) if outcome.ok else None
        if not records:
            return ("failed" if outcome.fault or outcome.ok else "odd"), []
        return "rows", list(filter(None, map(lambda rec: counter.offer_from(rec, "page"), records)))

    def _keep_side_rows(self, w, rows, sheet, floor):
        thin = bool(sheet) and bool(rows) and all(not o.traits for o in rows)
        if thin and not self.side_thin:
            self._mark_thin()
        for offer in rows:
            offer.page_seen = True
        if thin and w is not None:
            for offer in rows:
                w.absorb(offer, False)
            return self._sheets_for(w, rows, floor)
        out = []
        for offer in rows:
            offer.read_full = True
            out.append(offer if w is None else w.absorb(offer, True))
        return out

    def _mark_thin(self):
        counter = self.e.counter
        self.side_thin = True
        self.e.keep_lore()
        self.e.log("side_rows_thin", **{"tool": counter.tool_for(_SIDE_JOB), "table_tool": counter.tool_for(_SHEET_JOB)})

    def _sheets_for(self, w, rows, floor):
        out = []
        for offer in rows:
            if self.e.seconds_left() < Dials.Timing.open_until or self._short(floor, Dials.Read.page_cost):
                break
            self._keep(out, self._read_by_table(w, offer.oid))
        return out

    def _wants_sheet(self, w, oid, offer, sheet):
        if offer.traits or not sheet or self.e.counter.steps_free() <= 0:
            return False
        held = None if w is None else w.found.get(oid)
        return held is None or not held.traits

    def _read_page(self, w, page_job, sheet, oid):
        counter = self.e.counter
        outcome = counter.act(counter.tool_for(_PAGE_JOB), {page_job["id_arg"]: oid})
        body = outcome.body
        if not (outcome.ok and isinstance(body, dict)):
            return None
        offer = counter.offer_from(body, "page")
        if offer is None:
            offer = Offer(oid)
            offer.blob = body
        offer.page_seen = True
        if self._wants_sheet(w, oid, offer, sheet):
            offer.traits = self.table_for(oid) or {}
        offer.oid, offer.read_full = oid, True
        return offer if w is None else w.absorb(offer, True)

    def table_for(self, oid):
        counter = self.e.counter
        sheet = counter.job(_SHEET_JOB)
        if not sheet or "id_arg" not in sheet:
            return None
        outcome = counter.act(counter.tool_for(_SHEET_JOB), {sheet["id_arg"]: oid})
        body = outcome.body
        if not (outcome.ok and isinstance(body, dict)):
            return None
        table_key = counter.learn_keys("sheet", body).get("property_table")
        held = body.get(table_key) if table_key else None
        if not isinstance(held, dict):
            flat = all(v is None or isinstance(v, (str, int, float)) for v in body.values())
            if not flat:
                return {}
            held = body
        return dict((str(k), v) for k, v in held.items() if v not in (None, ""))

    def _read_by_table(self, w, oid):
        was_known = oid in w.found
        traits = self.table_for(oid)
        if traits is None or not (traits or was_known):
            return None
        return w.absorb(_sheet_offer(oid, traits), True)

    def _read_ones(self, w):
        kind_ok = self.e.scout.right_kind
        return list(filter(lambda o: o.read_full and o.traits and kind_ok(w, o), w.found.values()))

    @staticmethod
    def _key_named(read, trait):
        return any(names_match(key, trait) for offer in read for key in offer.traits)

    def _drop_kind_echoes(self, w):
        labels = self.e.scout.accepted_labels(w)
        if not labels:
            return
        label_words = set().union(*map(wordset, labels))
        read = self._read_ones(w)
        if len(read) < Dials.Verdict.read_min:
            return

        def echoes(need):
            words = wordset(need.want)
            plain_is = need.rel == "is" and need.source == FROM_REQUEST
            return plain_is and bool(words) and words <= label_words and not self._key_named(read, need.trait)

        doomed = list(filter(echoes, w.needs))
        for need in doomed:
            self.e.log("need_is_label", **{"item": w.phrase, "need": need.line()})
        if doomed:
            w.drop_needs(lambda need: any(need is d for d in doomed))

    @staticmethod
    def _value_keys(offer, forms):
        return {key_plain(k) for k, v in offer.traits.items()
                if any(f <= bit for bit in _bit_sets(v) for f in forms)}

    def _adopt_store_keys(self, w):
        read = self._read_ones(w)
        if len(read) < Dials.Verdict.read_min:
            return
        for need in w.live_needs():
            plain = need.rel in Dials.Ops.with_value and need.want and not has_digit(need.want)
            if not plain or need.renamed or self._key_named(read, need.trait):
                continue
            forms = _forms_of(need.want)
            if not forms:
                continue
            tally = {}
            for offer in read:
                for key in self._value_keys(offer, forms):
                    tally[key] = tally.get(key, 0) + 1
            counts = sorted(tally.values(), reverse=True)
            if not counts or (len(counts) > 1 and counts[0] == counts[1]):
                continue
            best = max(tally, key=tally.get)
            self.e.log("need_renamed", **dict(item=w.phrase, was=need.trait, now=best, value=need.want))
            need.move_to(best)

    def field_usage(self, w):
        read = self._read_ones(w)
        if len(read) < Dials.Verdict.read_min:
            return []
        rows = map(lambda need: self._usage_row(need, read), w.live_needs())
        return [row for row in rows if row is not None]

    @staticmethod
    def _usage_row(need, read):
        if not need.want or need.rel not in Dials.Ops.exact or has_digit(need.want):
            return None
        forms = _forms_of(need.want)
        if not forms:
            return None
        keys, tally, exact, holders = set(), {}, set(), 0
        for offer in read:
            named = list(filter(lambda k: same_name(k, need.trait), offer.traits))
            if not named:
                continue
            holders += 1
            keys.update(named)
            for bit in (single_space(b.lower()) for k in named for b in value_bits(offer.traits[k])):
                if not bit:
                    continue
                tally[bit] = tally.get(bit, 0) + 1
                if any(f <= frozenset(wordset(bit)) for f in forms):
                    exact.add(offer.oid)
        if not holders:
            return None
        return dict(need=need.line(), trait=need.trait, keys=sorted(keys), holders=holders, seen=len(read),
                    tally=tally, forms=forms, exact=exact)

    def plain_misses(self, w, o, usage=None):
        rows = usage if usage is not None else self.field_usage(w)
        return {row["need"] for row in rows if self._misses_row(o, row)}

    @staticmethod
    def _misses_row(o, row):
        if not row["exact"] - {o.oid}:
            return False
        own = list(filter(lambda k: same_name(k, row["trait"]), o.traits))
        bits = [b for k in own for b in _bit_sets(o.traits[k])]
        if not bits:
            return False
        heads = [heads_of(f) for f in row["forms"]]
        return all(not head <= heads_of(b) for b in bits for head in heads)

    @staticmethod
    def _no_value_words(w, o, need):
        if not need.want or need.source == FROM_FIT or need.rel not in Dials.Ops.exact:
            return False
        if any(names_match(k, need.trait) for k in o.traits):
            return False
        common = wordset_full(w.phrase) | wordset_full(w.kind)
        short = Dials.Verdict.filler_len

        def telling(part):
            return {x for x in wordset_full(part) - common if not x.isalpha() or len(x) > short}
        options = list(filter(None, map(telling, or_parts(need.want))))
        if not options:
            return False
        table_text = " ".join(map("%s %s".__mod__, o.traits.items()))
        seen = heads_of(wordset_full(" ".join((o.headline or "", o.blurb or "", o.remarks or "", o.shelf or "",
                                               table_text))))
        return all(not heads_of(option) & seen for option in options)

    def _linked_pages(self, w, o, limit=Dials.Verdict.related_max):
        own = _codes_in(o.headline)
        named = _codes_in(o.blurb, o.remarks) - own
        picked, taken = [], {o.oid}
        others = (x.found.values() for x in self.e.order.wants if x is not w)
        for pool in (w.found.values(), *others):
            for other in pool:
                if len(picked) >= limit:
                    return picked
                if other.oid in taken:
                    continue
                if _codes_in(other.headline) & named or own & _codes_in(other.blurb, other.remarks):
                    taken.add(other.oid)
                    picked.append(other)
        return picked

    def reassess(self, w):
        self.assess(w, [o for o in self.e.scout.shortlist(w) if o.read_full])

    def assess_all(self):
        self.e.oracle.run_many([lambda x=x: self.reassess(x) for x in self.e.order.wants])

    def assess(self, w, offers):
        self._drop_kind_echoes(w)
        self._adopt_store_keys(w)
        lines = w.need_lines()
        if lines:
            self._assess_against(w, offers, lines)
            return
        for o in filter(lambda x: x.read_full and w.ruling(x.oid) is None, offers):
            w.set_ruling(o.oid, blank_ruling(), set())

    def _still_open(self, w, o, wanted, lines):
        done = w.judged_lines(o.oid)
        if done is None:
            return True
        if done >= wanted:
            return False
        old = w.ruling(o.oid)
        if old and wanted & set(old.get("missed") or ()):
            w.set_ruling(o.oid, old, lines)
            return False
        return True

    def _assess_against(self, w, offers, lines):
        wanted = set(lines)
        pending = [o for o in offers if o.read_full and self._still_open(w, o, wanted, lines)]
        if not pending or self.e.seconds_left() < Dials.Timing.verdicts_until:
            return
        usage = self.field_usage(w)
        step = Dials.Verdict.batch
        groups = list(map(lambda at: pending[at:at + step], range(0, len(pending), step)))
        oracle = self.e.oracle
        jobs = []
        for group in groups:
            brief = brief_assess(w.kind, lines, usage, [(o, self._linked_pages(w, o)) for o in group])
            jobs.append(lambda brief=brief: oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.assess))
        for group, answer in zip(groups, oracle.run_many(jobs)):
            entries = self._entries_by_id(w, answer)
            for o in group:
                if o.oid in entries:
                    w.set_ruling(o.oid, self._ruling_from(w, o, entries[o.oid], lines, usage), lines)

    @staticmethod
    def _entries_by_id(w, answer):
        listed = answer.get("results") if isinstance(answer, dict) else None
        if not isinstance(listed, list):
            return {}
        entries = {}
        for entry in filter(lambda x: isinstance(x, dict), listed):
            ref = entry.get("listing_id")
            if not isinstance(ref, (list, dict)) and ref in w.found:
                entries[ref] = entry
        return entries

    def _ruling_from(self, w, o, entry, lines, usage):
        names = {key: {str(x) for x in _answer_items(entry.get(key)) if x}
                 for key in ("fails", "unaddressed", "text_only", "by_other_word")}
        mapped = {key: self._to_need_texts(raw, lines) for key, raw in names.items()}
        ruling = dict(missed=mapped["fails"] | self.plain_misses(w, o, usage), doubtful=mapped["unaddressed"],
                      faint=self._to_need_texts(names["text_only"] | names["by_other_word"], lines),
                      indirect=mapped["by_other_word"])
        shielded = ruling["missed"] | ruling["doubtful"] | ruling["indirect"]
        for need in w.live_needs():
            if need.line() not in shielded and self._no_value_words(w, o, need):
                ruling["doubtful"].add(need.line())
        return ruling

    @staticmethod
    def _to_need_texts(names, lines):
        def one(name):
            number = tag_number(name, NEED_TAG)
            return lines[number - 1] if number is not None and 0 < number <= len(lines) else _nearest_line(name, lines)
        return set(map(one, names))

    def approve(self, w, o):
        key = (w.slot, o.oid, keyed_json(o.traits), tuple(w.need_lines()))
        held = self.approvals.get(key)
        if held is None:
            held = self.approvals[key] = self.approve_now(w, o)
        return held

    def approve_now(self, w, o, strict=False):
        lines = w.need_lines()
        brief = brief_final_check(w.kind, w.count, self.e.order.budget, lines, o)
        answer = self.e.oracle.ask_json(Dials.Model.checker_light, brief, Dials.Tokens.final_check)
        if not isinstance(answer, dict):
            return not strict
        verdict = answer.get("meets_all")
        if strict and verdict is not True:
            return False
        unmet = list(map(str, _answer_items(answer.get("unmet"))))
        self.e.log("final_check", **{"id": o.oid, "fits": verdict, "problems": unmet[:Dials.Trace.some]})
        vetoed = verdict is False and bool(unmet)
        for problem in unmet if vetoed else ():
            w.flag_miss(o.oid, _nearest_line(problem, lines))
        return not vetoed


_YES_MOMENTS = ("before_order", "before_finish")
_ASKING_AIM = re.compile(r"^\W*(ask|discuss|clarify|chat|inquire|enquire)\b", re.I)
_YES_WORDS = re.compile(r"approv|confirm|permission|consent|okay|\bok\b|go-ahead|\byes\b|sign", re.I)
_CLOSING_ROLES = ("buy_cart", "end_unbought", "add_to_cart", "replace_cart")
_HALT = object()


def _near_any(prop, names):
    return any(same_name(prop, other) for other in names)


class Parley:
    def __init__(self, errand):
        self.e = errand
        self.msg_limit, self.msgs_out = None, 0
        self.duties, self.yes_needed = [], False
        self.shown_key, self.due_tries = None, {}
        self.planner_pick, self.raised, self.need_replies = None, {}, {}

    @staticmethod
    def _tidy(step):
        via = str(step.get("via") or "").lower()
        step.update(at=str(step.get("at") or "before_finish").lower(), via=via if via == "message" else "tool")
        return step

    @staticmethod
    def _usable_step(step):
        if not (isinstance(step, dict) and step.get("aim")):
            return False
        aim = str(step["aim"])
        only_asks = str(step.get("via") or "").lower() == "message" and bool(_ASKING_AIM.search(aim))
        return not only_asks or bool(_YES_WORDS.search(aim))

    @staticmethod
    def _asks_yes(step):
        return step["at"] in _YES_MOMENTS and step["via"] == "message"

    def adopt(self, msg_limit, raw_steps):
        self.msg_limit = msg_limit
        chosen = list(filter(self._usable_step, raw_steps or []))[:Dials.Request.procedure_kept]
        self.duties = list(map(self._tidy, chosen))
        self.yes_needed = any(map(self._asks_yes, self.duties))

    def add_duty(self, step):
        if len(self.duties) >= Dials.Shopper.procedure_max or not self._usable_step(step):
            return
        duty = self._tidy(dict(step))
        aims = [known.get("aim") for known in self.duties]
        if max((likeness(duty["aim"], aim) for aim in aims), default=0.0) >= Dials.Like.goal:
            return
        self.duties.append(duty)
        self.yes_needed = self.yes_needed or self._asks_yes(duty)
        self.e.log("duty_added", step=duty)

    def has_duties(self, moment=None):
        moments = [st["at"] for st in self.duties]
        return bool(moments) if moment is None else moment in moments

    def late_duty_count(self):
        return len([st for st in self.duties if st["at"] != "start"])

    def needs_yes(self):
        return self.yes_needed

    @staticmethod
    def _message_count(step):
        least = Dials.Shopper.step_messages_min
        try:
            told = int(step.get("message_count") or least)
        except (TypeError, ValueError, OverflowError):
            told = least
        return max(least, told)

    def _limit_left(self):
        return None if self.msg_limit is None else self.msg_limit - self.msgs_out

    def _has_channel(self):
        return bool(self.e.counter.job("tell_shopper"))

    def can_send(self):
        left = self._limit_left()
        return self._has_channel() and (left is None or left > 0) and self.e.counter.steps_free() > 0

    def reserved_msgs(self):
        if not (self.duties and self._has_channel()):
            return 0
        held = sum(map(self._message_count, filter(self._asks_yes, self.duties)))
        left = self._limit_left()
        return held if left is None else min(held, max(0, left))

    def owed(self):
        if self.e.till.has_lines() and self.shown_key == self.basket_key():
            return 0
        return self.reserved_msgs()

    def can_question(self):
        left = self._limit_left()
        return self.can_send() and (left is None or left > self.reserved_msgs())

    def room_after_held(self):
        left = self._limit_left()
        return left if left is None else left - self.reserved_msgs()

    def first_question_count(self):
        planned = (Dials.Shopper.default_questions if self.msg_limit is None else self.msg_limit) - self.reserved_msgs()
        return planned if planned >= 0 and self._has_channel() else 0

    def _send(self, text):
        c = self.e.counter
        arg = c.job_field("tell_shopper", "text_arg")
        text = c.clip_arg("tell_shopper", arg, text)
        outcome = c.act(c.tool_for("tell_shopper"), {arg: text})
        self.msgs_out += 1
        return text, outcome

    def _mark_shown(self):
        self.msgs_out += 1
        self.shown_key = self.basket_key()

    def basket_key(self):
        lines = [(oid, int(qty or 0)) for oid, qty in self.e.till.basket().items() if qty]
        return tuple(sorted(lines))

    def _basket_rows(self):
        till = self.e.till
        rows = []
        for oid, qty in self.basket_key():
            owner = till.wanted_for(oid)
            rows.append((oid, qty, owner, None if owner is None else owner.found.get(oid)))
        return rows

    def show_basket(self):
        if self.e.counter.steps_free() < Dials.Shopper.confirm_steps or not self.can_send():
            return False
        lines = [(qty, getattr(offer, "headline", oid) or oid, getattr(offer, "cost", None))
                 for oid, qty, _, offer in self._basket_rows()]
        outcome = self._send(basket_confirmation(lines))[1]
        self.shown_key = self.basket_key()
        self.e.log("basket_shown", basket=self.shown_key, ok=outcome.ok, body=json_cut(outcome.body, Dials.Trace.mid))
        return True

    def before_checkout(self):
        if not (self.duties and self.e.till.has_lines()):
            return False
        ran = self.run_due("before_order")
        again = ran and self.owed() and self.e.counter.steps_free() >= Dials.Ending.regate_steps
        if again:
            ran = self.run_due("before_order") or ran
        unconfirmed = self.yes_needed and self.shown_key != self.basket_key()
        if unconfirmed and ran is not False:
            ran = self.show_basket() or ran
        return ran

    def _gate_brief(self, moment, closing):
        c = self.e.counter
        tools = [dict(name=t, description=c.describe(t), parameters=c.tool_form(t))
                 for t in c.tool_names() if t not in closing]
        basket = [(oid, qty, getattr(offer, "headline", ""), getattr(offer, "cost", None), getattr(owner, "phrase", None))
                  for oid, qty, owner, offer in self._basket_rows()]
        return brief_process_check(request=self.e.order.text, steps=self.duties, moment=moment, basket=basket,
                                   messages_left=self._limit_left(), steps_left=c.steps_free(), tools=tools,
                                   calls=c.diary_tail(Dials.History.shown))

    @staticmethod
    def _hard_cut(spec, value):
        cut = spec.get("maxLength")
        if not isinstance(value, str) or not isinstance(cut, int) or len(value) <= cut:
            return value
        return value[:cut]

    def _gate_args(self, raw, props):
        if not isinstance(raw, dict):
            return {}

        def allowed(key):
            options = (props.get(key) or {}).get("enum")
            return not (isinstance(options, list) and options) or raw[key] in options

        return {key: self._hard_cut(props.get(key) or {}, raw[key]) for key in raw if key in props and allowed(key)}

    @staticmethod
    def _gate_actions(listed):
        if not isinstance(listed, list):
            return []
        return [act for act in listed[:Dials.Shopper.process_actions] if isinstance(act, dict)]

    def _gate_step(self, moment, act, later, closing):
        c = self.e.counter
        tool = act.get("tool_name")
        form = c.tool_form(tool)
        if form is None or tool in closing:
            return False
        talk = c.tool_for("tell_shopper")
        to_shopper = tool == talk
        if not to_shopper:
            due = max(len([x for x in later if x.get("tool_name") == talk]), self.owed())
            if c.steps_free() < due + Dials.Shopper.process_steps:
                self.e.log("duty_skipped", moment=moment, tool=tool, steps=c.steps_free(), due=due)
                return False
        args = self._gate_args(act.get("arguments"), form.get("properties") or {})
        if not all(name in args for name in form.get("required") or []):
            return False
        if to_shopper and not self.can_send():
            return False
        if c.steps_free() < Dials.Shopper.process_steps:
            return _HALT
        outcome = c.act(tool, args)
        if to_shopper:
            self._mark_shown()
        self.e.log("duty_ran", moment=moment, tool=tool, why=str(act.get("step_served") or "")[:Dials.Trace.mid],
                   ok=outcome.ok)
        return True

    def run_due(self, moment):
        c = self.e.counter
        tries = self.due_tries.get(moment, 0)
        if tries >= Dials.Shopper.process_calls or c.steps_free() < Dials.Shopper.process_steps:
            return False
        if self.e.seconds_left() < Dials.Timing.process_until:
            return False
        self.due_tries[moment] = tries + 1
        closing = {tool for tool in map(c.tool_for, _CLOSING_ROLES) if tool is not None}
        answer = self.e.oracle.ask_json(Dials.Model.planner_spare, self._gate_brief(moment, closing),
                                        Dials.Tokens.process_check)
        if not isinstance(answer, dict):
            self.e.log("duty_no_reply", moment=moment)
            return None
        acts = self._gate_actions(answer.get("do_now"))
        ran = False
        for at, act in enumerate(acts):
            done = self._gate_step(moment, act, acts[at + 1:], closing)
            if done is _HALT:
                break
            ran = ran or done
        if not ran:
            self.e.log("duty_none", moment=moment)
        return ran

    @staticmethod
    def _facet_table(offer):
        table = {}
        for key, val in offer.traits.items():
            table.setdefault(facet_name(key), str(val).strip().lower()[:Dials.Shopper.menu_value_cut])
        table.pop("", None)
        return table

    @staticmethod
    def _agree(tables, one, two):
        both = [t for t in tables if one in t and two in t]
        if len(both) < Dials.Shopper.alias_shared_min:
            return False
        return len([t for t in both if t[one] == t[two]]) >= Dials.Like.alias * len(both)

    @classmethod
    def _alias_map(cls, order, tables):
        alias_of = {}
        for at, key in enumerate(order):
            head = next((h for h in order[:at] if h not in alias_of and cls._agree(tables, h, key)), None)
            if head is not None:
                alias_of[key] = head
        return alias_of

    @staticmethod
    def _settled_name(name, fixed):
        return simplify(name) in fixed or max((likeness(name, f) for f in fixed), default=0.0) >= Dials.Like.covered

    @staticmethod
    def _menu_line(names, tables, out_of):
        held = [v for v in (next((t[nm] for nm in names if nm in t), None) for t in tables) if v is not None]
        examples = list(dict.fromkeys(v for v in held if v))
        carriers = len(held)
        weight = carriers * 1.0
        if carriers >= Dials.Shopper.one_value_carriers and len(examples) == 1:
            weight *= Dials.Shopper.weight_cut
        if len(set(held)) >= carriers >= Dials.Shopper.ident_carriers:
            weight *= Dials.Shopper.weight_cut
        entry = dict(prop=names[0], carriers=carriers, out_of=out_of, examples=examples[:Dials.Shopper.menu_values],
                     aliases=names[1:1 + Dials.Shopper.menu_aliases])
        return weight, entry

    def menu_for(self, w):
        scout = self.e.scout
        read = [o for o in w.found.values() if o.read_full and scout.right_kind(w, o)]
        clean = [o for o in read if not (w.ruling(o.oid) or {}).get("missed")]
        basis = clean if len(clean) >= Dials.Shopper.alive_min else read
        tables = list(map(self._facet_table, basis))
        spotted = [name for table in tables for name in table]
        tally = {name: spotted.count(name) for name in set(spotted)}
        order = sorted(tally, key=lambda k: (-tally[k], len(k), k))
        alias_of = self._alias_map(order, tables)
        fixed = {n.slug() for n in w.needs}.union(self.raised.get(w.slot, []))
        ranked = []
        for head in (k for k in order if k not in alias_of):
            names = [head, *(b for b, a in alias_of.items() if a == head)]
            if wordset(head) & Dials.Shopper.skip_words or any(self._settled_name(nm, fixed) for nm in names):
                continue
            ranked.append(self._menu_line(names, tables, len(basis)))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["prop"]))
        return [entry for _, entry in ranked][:Dials.Shopper.menu_size]

    def _taken(self, w, prop, plan):
        known = [*self.raised.get(w.slot, []), *(n.trait for n in w.needs), *(p for x, p in plan if x is w)]
        if _near_any(prop, known) or self.heard_before(w, prop):
            return True
        return any(same_name(prop, s) or likeness(prop, s) >= Dials.Like.same for s in self.e.picker.rule_props(w))

    def _plan_block(self, w):
        menu = [menu_entry(prop=m["prop"], carried_by=m["carriers"], out_of=m["out_of"], examples=m["examples"],
                           aliases=m["aliases"]) for m in self.menu_for(w)]
        return plan_block(product=w.phrase, category=w.kind, known=w.need_lines(), settled=self.e.picker.rule_props(w),
                          hints=w.clues, asked=list(self.raised.get(w.slot, [])), menu=menu)

    @staticmethod
    def _read_asks(answer, wants):
        asks = answer.get("asks") if isinstance(answer, dict) else None
        picks = []
        usable = [a for a in asks if isinstance(a, dict) and a.get("property_name")] if isinstance(asks, list) else []
        for ask in usable:
            prop = prop_form(ask["property_name"])
            target = max(wants, key=lambda x: likeness(x.phrase, ask.get("product") or x.phrase))
            if prop and not _near_any(prop, [p for x, p in picks if x is target]):
                picks.append((target, prop))
        return picks

    def _ask_planner(self, count):
        wants = self.e.order.wants
        whole = count if self.msg_limit is None else self.msg_limit
        brief = brief_question_order(request=self.e.order.text, raise_topics=self.e.order.raise_topics,
                                     blocks=[self._plan_block(w) for w in wants], ask_count=max(whole, count))
        self.planner_pick = self._read_asks(
            self.e.oracle.ask_json(Dials.Model.planner_light, brief, Dials.Tokens.question_order), wants)
        self.e.log("question_order", questions=[(x.phrase, p) for x, p in self.planner_pick])

    @staticmethod
    def _pad_menu(menu):
        return [m for m in menu if m["out_of"] <= Dials.Shopper.pad_of or m["carriers"] >= Dials.Shopper.pad_listings]

    def question_plan(self, count):
        if count <= 0:
            return []
        if not self.planner_pick:
            self._ask_planner(count)
        many = len(self.e.order.wants) > 1
        return self._rotate(count) if many else self._single_order(count)

    def _next_free(self, w, queue, plan):
        while queue:
            prop = queue.pop(0)["prop"]
            if not self._taken(w, prop, plan):
                return prop
        return None

    def _single_order(self, count):
        wants = self.e.order.wants
        plan = []
        for pick in self.planner_pick:
            if len(plan) == count:
                break
            if not self._taken(*pick, plan):
                plan.append(pick)
        menus = {w.slot: self.menu_for(w) for w in wants}
        if count >= Dials.Shopper.tops_from:
            for w in wants:
                tops = [m["prop"] for m in menus[w.slot] if m["carriers"] >= Dials.Shopper.top_listings]
                for k, prop in enumerate(tops[:Dials.Shopper.tops_one_item]):
                    if self._taken(w, prop, plan):
                        continue
                    plan.insert(min(Dials.Shopper.top_first_pos + k * Dials.Shopper.top_gap, len(plan)), (w, prop))
            del plan[count:]
        if len(plan) < count:
            self._pad(plan, count, wants, menus)
        return plan[:count]

    def _pad(self, plan, count, wants, menus):
        queues = {w.slot: self._pad_menu(menus[w.slot]) for w in wants}
        turn = sorted(wants, key=lambda w: (len([1 for x, _ in plan if x is w]) + len(self.raised.get(w.slot, [])),
                                            w.slot))
        while len(plan) < count:
            added = 0
            for w in turn:
                prop = self._next_free(w, queues[w.slot], plan)
                if prop is not None:
                    plan.append((w, prop))
                    added += 1
                if len(plan) >= count:
                    break
            if not added:
                break

    def _fill_queue(self, w, line):
        menu = self._pad_menu(self.menu_for(w))
        top = next((m["prop"] for m in menu if m["carriers"] >= Dials.Shopper.top_listings), None)
        if top and not self._taken(w, top, []) and not _near_any(top, line[:Dials.Shopper.top_clash_window]):
            line[:] = [q for q in line if not same_name(q, top)]
            line.insert(min(len(line), Dials.Shopper.top_first_pos), top)
        for prop in [m["prop"] for m in menu]:
            if not (self._taken(w, prop, []) or _near_any(prop, line)):
                line.append(prop)

    @staticmethod
    def _deal(turn, queues, by_slot, count):
        plan = []
        depth = 0
        while len(plan) < count:
            row = [(by_slot[s], queues[s][depth]) for s in turn if depth < len(queues[s])]
            if not row:
                break
            plan.extend(row)
            depth += 1
        return plan[:count]

    def _rotate(self, count):
        wants = self.e.order.wants
        queues = {w.slot: [] for w in wants}
        rank = {}
        for owner, prop in self.planner_pick:
            line = queues.get(owner.slot)
            if line is None or self._taken(owner, prop, []) or _near_any(prop, line):
                continue
            line.append(prop)
            rank.setdefault(owner.slot, len(rank))
        for w in wants:
            self._fill_queue(w, queues[w.slot])
            rank.setdefault(w.slot, len(rank))
        told = {w.slot for w in wants if any(n.source == FROM_SHOPPER for n in w.needs)}
        turn = sorted(rank, key=lambda s: (s in told, len(self.raised.get(s, [])), rank[s]))
        return self._deal(turn, queues, {w.slot: w for w in wants}, count)

    def pose(self, w, prop, also=None):
        said = self.e.order.text
        sent, outcome = self._send(question_pair(prop, also, w.phrase, said) if also else
                                   question_single(prop, w.phrase, said))
        self.raised.setdefault(w.slot, []).extend([simplify(prop)] + ([simplify(also)] if also else []))
        return self._hear(w, prop, also, sent, outcome)

    def pose_narrow(self, w, prop):
        examples = []
        pairs = ((key, str(val)) for offer in w.found.values() for key, val in offer.traits.items())
        for key, text in pairs:
            if text not in examples and same_name(split_camel(key), prop):
                examples.append(text[:Dials.Shopper.example_chars])
        sent, outcome = self._send(question_focused(prop, w.phrase, examples, self.e.order.text))
        return self._hear(w, prop, None, sent, outcome)

    @staticmethod
    def _reply_text(body):
        if isinstance(body, dict):
            longest = None
            for val in body.values():
                if isinstance(val, str) and (longest is None or len(val) > len(longest)):
                    longest = val
            return json_cut(body, Dials.Shopper.reply_cut) if longest is None else longest
        return body if isinstance(body, str) else None

    def _rename_need(self, w, reply, reading, asked):
        spoken = asked + [str(reading.get("property_name") or "")]
        named = self._named_prop(w, reply, spoken, str(reading.get("required_value") or ""))
        if named:
            reading["property_name"] = named

    def _hear(self, w, prop, also, sent, outcome):
        reply = self._reply_text(outcome.body) if outcome.ok else None
        if not reply:
            return None
        known = [f"{n.trait}: {n.rel} {n.want}" for n in w.needs if n.want and n.source == FROM_SHOPPER]
        brief = brief_answer(product=w.phrase, prop=prop, also=also, message=sent, known=known, reply=reply)
        reading = self.e.oracle.ask_json(Dials.Model.reply_chain, brief, Dials.Tokens.answer)
        kind = self.reply_kind(reading)
        if kind == "need":
            self._rename_need(w, reply, reading, [prop, also or ""])
        self.e.log("answer_read", item=w.phrase, prop=prop, reply=reply[:Dials.Trace.long],
                   read=json_cut(reading, Dials.Trace.full))
        if kind in ("need", "repeat"):
            self.need_replies.setdefault(w.slot, []).append(f" {simplify(reply)} ")
        return reading if isinstance(reading, dict) else None

    @staticmethod
    def reply_kind(answer):
        return str(answer.get("reading") or "").lower() if isinstance(answer, dict) else ""

    def _named_prop(self, w, reply, spoken, value):
        padded = f" {simplify(reply)} "
        if any(form and f" {form} " in padded for form in (simplify(s) for s in spoken if s)):
            return None
        value_words = wordset(value)
        names = {key_plain(k) for offer in w.found.values() for k in offer.traits}
        hits = [nm for nm in names
                if f" {nm} " in padded and len(nm) >= Dials.Shopper.named_key_min and not wordset(nm) & value_words
                and not any(same_name(nm, s) for s in spoken if s)]
        alone = [nm for nm in hits if all(nm == other or f" {nm} " not in f" {other} " for other in hits)]
        return alone[0] if len(alone) == 1 else None

    def heard_before(self, w, prop):
        name = simplify(prop)
        phrase = f" {name} "
        return len(name.split()) >= Dials.Shopper.told_words and any(phrase in ln for ln in self.need_replies.get(w.slot, []))

    def covered(self, w, prop):
        if self.heard_before(w, prop):
            return True
        return max((likeness(n.trait, prop) for n in w.needs), default=0.0) >= Dials.Like.covered

    def take_reply(self, w, prop, answer):
        left = self._limit_left()
        spare = Dials.Shopper.assumed_left if left is None else left
        self._apply(w, prop, answer, floor=self.e.ending_room() + max(0, spare))

    def _apply(self, w, prop, answer, floor):
        if not answer:
            return
        steps = answer.get("process")
        if isinstance(steps, list):
            for step in filter(lambda s: isinstance(s, dict), steps):
                self.add_duty(step)
        budget = read_number(answer.get("new_total_limit"))
        if (budget or 0) > 0:
            self.e.order.set_budget(budget)
            self.e.log("cap_from_reply", cap=budget)
        told = answer.get("required_value") or answer.get("other_page")
        if told and self.reply_kind(answer) == "need":
            self._learn_need(w, prop, answer, floor)

    @staticmethod
    def _repeat_page(w, mark):
        for old in w.needs:
            if mark == old.link_last or mark == old.link_first:
                return True
            if old.link and page_identity(old.link) == mark:
                return True
        return False

    def _learn_need(self, w, prop, answer, floor):
        need = Need(trait=answer.get("property_name") or prop, rel=answer.get("comparison") or "is",
                    want=answer.get("required_value") or "", link=answer.get("other_page"), source=FROM_SHOPPER)
        if need.link:
            need.link["read_property"] = need.trait
            mark = page_identity(need.link)
            if mark and self._repeat_page(w, mark):
                self.e.log("link_repeat", item=w.phrase, prop=need.trait)
                return
        same_value = simplify(need.want)
        if need.want and any((old.source, old.rel) == (FROM_SHOPPER, need.rel) and simplify(old.want) == same_value
                             for old in w.needs):
            return
        w.drop_needs(lambda old: (old.slug(), old.source) == (need.slug(), FROM_SHOPPER))
        w.add_need(need)
        plain_value = need.want and not need.link and need.rel in Dials.Ops.exact
        if plain_value and len(need.want) <= Dials.Query.value_term_max:
            w.queue_query(" ".join([or_parts(need.want)[0], w.headword]))
        if need.waiting():
            self.e.trail.follow(w, need, floor)

    def _line_target(self, name):
        if not name:
            return None
        wants = self.e.order.wants
        scores = [likeness(w.phrase, name) for w in wants]
        best = max(range(len(wants)), key=scores.__getitem__)
        return None if len(wants) > 1 and scores[best] < Dials.Like.loose else wants[best]

    def _reconcile_budget(self, value, lines):
        stated = numbers_in(" ".join(lines))
        for guess in (value, value * Dials.Notice.cents, value / Dials.Notice.cents):
            if guess in stated:
                return guess
        cap = self.e.order.budget
        if cap is None or not stated:
            return value
        low, high = cap / Dials.Notice.budget_span, cap * Dials.Notice.budget_span
        return next((x for x in stated if low <= x <= high), value)

    def _on_budget(self, upd, lines):
        value = read_number(upd.get("new_total_limit"))
        if value:
            self.e.order.set_budget(self._reconcile_budget(value, lines))
        return bool(value)

    def _on_cart_news(self, upd, lines):
        self.e.till.recheck_lines()
        return True

    def _on_step(self, upd, lines):
        step = upd.get("process_step")
        if isinstance(step, dict):
            self.add_duty(step)
        return False

    def _on_drop(self, upd, lines):
        prop = upd.get("property_name")
        if not prop:
            return False
        target = self._line_target(upd.get("product"))
        hit = False
        for w in self.e.order.wants if target is None else [target]:
            count = len(w.needs)
            w.drop_needs(lambda n: likeness(n.trait, prop) >= Dials.Like.same)
            if count == len(w.needs):
                continue
            hit = True
            w.drop_misses(lambda line: likeness(line.split(":")[0], prop) >= Dials.Like.same)
        return hit

    def _on_add(self, upd, lines):
        prop, value = upd.get("property_name"), upd.get("required_value")
        if not (prop and value):
            return False
        target = self._line_target(upd.get("product"))
        hosts = self.e.order.wants[:1] if target is None else [target]
        for w in hosts:
            w.add_need(Need(prop, upd.get("comparison") or "is", value, None, FROM_SHOPPER))
        return len(hosts) > 0

    def absorb_lines(self):
        lines = self.e.counter.drain_inbox()
        if not lines:
            return False
        answer = self.e.oracle.ask_json(Dials.Model.light_spare,
                                        brief_aside(lines=lines, products=[w.phrase for w in self.e.order.wants]),
                                        Dials.Tokens.aside)
        self.e.log("lines_read", lines=[ln[:Dials.Trace.mid] for ln in lines], read=json_cut(answer, Dials.Trace.full))
        updates = answer.get("updates") if isinstance(answer, dict) else None
        if not isinstance(updates, list):
            return False
        handlers = {"budget": self._on_budget, "price": self._on_cart_news, "stock": self._on_cart_news,
                    "step": self._on_step, "drop": self._on_drop, "add": self._on_add}
        changed = False
        for upd in filter(lambda u: isinstance(u, dict), updates):
            handler = handlers.get(str(upd.get("change") or "").lower())
            if handler is not None and handler(upd, lines):
                changed = True
        return changed


def _field_text(link, key):
    return str(link.get(key) or "").strip()


class _Clue:

    def __init__(self, link):
        self.link = link
        names = ("sold_by", "of_category", "pick_rule", "relation_words")
        self.seller, self.category, self.pick, self.relation = (_field_text(link, k) for k in names)
        given, count = link.get("identifying_values"), link.get("fit_count")
        self.attrs = given if isinstance(given, dict) else {}
        self.count = count if isinstance(count, int) else None
        self.values = self._query_values()
        self.words = " ".join(self.values[:Dials.Lookup.query_values])
        self.noun = (self.category.split() or [""])[-1]

    def _query_values(self):
        limit = Dials.Lookup.short_value_words
        texts = [str(v) for v in self.attrs.values()]
        picked = [t.strip() for t in texts if t.strip() and len(t.split()) <= limit]
        avoid = simplify(self.seller)
        for text in (t for t in texts if len(t.split()) > limit):
            extra = [wd for wd in lettered_words(text) if wd not in picked and simplify(wd) != avoid]
            picked += extra[:Dials.Lookup.long_value_words]
        return sorted(picked, key=has_digit)

    def key(self):
        described = sorted(map(lambda kv: (simplify(kv[0]), simplify(kv[1])), self.attrs.items()))
        return ascii_json([*map(simplify, (self.seller, self.category)), described])

    def queries(self):
        found = []
        if self.seller:
            nouns = [n for n in dict.fromkeys((self.noun, head_noun(self.category))) if n] if self.category else []
            found += [f"{self.seller} {self.words}".strip(), *(f"{self.seller} {n}" for n in nouns), self.seller]
        if self.words and self.category:
            found.append(f"{self.words} {self.noun}".strip())
        return [q for q in dict.fromkeys(found) if q]

    def seller_fits(self, offer):
        if not self.seller:
            return True
        close = max(likeness(offer.vendor, self.seller), likeness(offer.maker, self.seller))
        own_text = " ".join((offer.headline, offer.blurb, offer.remarks))
        return close >= Dials.Like.same or simplify(self.seller) in simplify(own_text)

    def overlap(self, offer):
        wanted = wordset(self.words)
        if not wanted:
            return 0
        raw = " ".join((offer.headline, offer.blurb, offer.remarks, ascii_json(offer.traits)))
        have = wordset(raw)
        squeezed = simplify(raw).replace(" ", "")

        def glued(value):
            if not has_digit(value):
                return False
            own = wordset(value)
            if own and own <= have:
                return False
            flat = simplify(value).replace(" ", "")
            return len(flat) >= Dials.Lookup.squeezed_min and flat in squeezed

        return len(wanted & have) + len([v for v in self.values if glued(v)])

    def words_in_text(self, offer):
        if not self.attrs:
            return False
        parts = [offer.headline or "", offer.remarks or "", offer.blurb or ""]
        parts += [f"{k} {v}" for k, v in offer.traits.items()]
        text = wordset_full(" ".join(parts))
        needed = [wordset_full(v) for v in self.attrs.values() if str(v).strip()]
        return all(need and need <= text for need in needed)

    def props_match(self, offer):
        if not (self.attrs and offer.read_full):
            return False
        facts = [(key_plain(k), simplify(v)) for k, v in offer.traits.items()]

        def stated(name, val):
            want = simplify(val)
            return bool(want) and any(fv == want and names_match(fk, name) for fk, fv in facts)

        return all(stated(name, val) for name, val in self.attrs.items())


class Trail:
    def __init__(self, errand):
        self.e = errand
        self.hunts = {}

    def follow(self, w, n, floor=0):
        self._walk(w, n, floor, 0)

    def _short_of(self, least):
        return self.e.counter.steps_free() < least

    def _walk(self, w, n, floor, hop):
        link = n.link or {}
        n.pin_first(page_identity(link))
        least = max(floor, Dials.Lookup.floor_min)
        if hop > Dials.Lookup.max_hops or self._short_of(least) or self.e.seconds_left() < Dials.Timing.refs_until:
            return
        clue = _Clue(link)
        page = self._hunt(clue, least)
        if page is None:
            return
        prop = str(link.get("read_property") or n.trait).strip()
        paired = n.rel == "counterpart of reference"
        reading, thin = self._read_page(page, prop, clue.relation, paired)
        self.e.log("page_reading", hop=hop, page=page.oid, read=json_cut(reading, Dials.Trace.long))
        onward = self._onward(reading, prop, page, clue.relation, thin)
        if onward is None:
            self._settle(w, n, clue, reading, page, paired)
            return
        carried = {"read_property": prop}
        if clue.relation:
            carried["relation_words"] = clue.relation
        if link.get("leeway"):
            carried["leeway"] = link["leeway"]
        onward.update(carried)
        n.relink(onward)
        self._walk(w, n, floor, hop + 1)

    def _onward(self, reading, prop, page, relation, thin):
        target = reading.get("look_elsewhere") if isinstance(reading, dict) else None
        if not (isinstance(target, dict) and target):
            return None
        about = str(target.get("read_property") or prop)
        if self._same_property(about, prop):
            return target
        self._own_value_instead(page, prop, relation, reading, thin, about)
        return None

    @staticmethod
    def _same_property(about, prop):
        one, two = simplify(about), simplify(prop)
        return same_name(about, prop) or likeness(about, prop) >= Dials.Like.same or two in one or one in two

    def _read(self, page, prop, relation):
        return self.e.oracle.ask_json(Dials.Model.light_spare, brief_page_value(prop=prop, relation=relation, offer=page),
                                      Dials.Tokens.page_value)

    def _fetch_table(self, page):
        page.traits = self.e.assessor.table_for(page.oid) or {}

    def _room_for_table(self):
        return not self._short_of(Dials.Lookup.table_steps)

    def _read_page(self, page, prop, relation, paired):
        thin = self.e.assessor.side_rows_thin() and not page.traits
        early = thin and not page.remarks and self._room_for_table()
        if early:
            self._fetch_table(page)
        thin = thin and not early
        reading = self._read(page, prop, relation)
        if not (thin and isinstance(reading, dict) and self._room_for_table()):
            return reading, thin
        if reading.get("look_elsewhere") or not (paired or not reading.get("own_value")):
            return reading, thin
        self._fetch_table(page)
        return (self._read(page, prop, relation) if page.traits else reading), thin

    def _own_value_instead(self, page, prop, relation, reading, thin, about):
        if thin and not page.traits and not reading.get("own_value") and self._room_for_table():
            self._fetch_table(page)
            again = self._read(page, prop, relation) if page.traits else None
            if isinstance(again, dict):
                for key in ("own_value", "paired_value"):
                    reading[key] = again.get(key) or reading.get(key)
        if not reading.get("own_value"):
            own = next((v for k, v in page.traits.items() if names_match(k, prop)), None)
            if own is not None and own != "":
                reading["own_value"] = str(own)
        self.e.log("link_elsewhere", about=about, wanted=prop, value=reading.get("own_value"))

    def _chosen_value(self, n, got, paired, clue, page):
        own, partner, asks = got.get("own_value"), got.get("paired_value"), got.get("relation_asks")
        value = partner if paired else own
        if n.rel != "same as reference" or not (partner and asks):
            return value
        if simplify(asks) != simplify(partner) or simplify(partner) == simplify(own):
            return value
        self.e.log("paired_value", page=page.oid, relation=clue.relation[:Dials.Trace.short], value=partner)
        return partner

    def _as_range(self, w, n, value, clue, mark):
        centre, spread = read_number(value), read_number(_field_text(clue.link, "leeway"))
        if centre is None or spread is None:
            return False
        unit = unit_part(value)
        low, high = [f"{x:g} {unit}".strip() for x in (centre - spread, centre + spread)]
        n.settle("at least", low, mark)
        w.add_need(Need(trait=n.trait, rel="at most", want=high, source=n.source))
        self.e.log("link_range", prop=n.trait, low=low, high=high)
        return True

    @staticmethod
    def _as_plain(w, n, value, mark):
        n.settle(Dials.Ops.borrowed.get(n.rel, "is"), str(value), mark)
        if n.rel != "is" or len(n.want) > Dials.Query.value_term_max or not w.queries:
            return
        want = simplify(n.want)
        if all(want not in simplify(q) for q in w.queries):
            w.queue_query(f"{n.want} {w.headword}")

    def _settle(self, w, n, clue, reading, page, paired):
        got = reading if isinstance(reading, dict) else {}
        value = self._chosen_value(n, got, paired, clue, page)
        mark = page_identity(clue.link)
        if value and n.rel == "within reference":
            if self._as_range(w, n, value, clue, mark):
                return
            n.set_rel("same as reference")
        if value:
            self._as_plain(w, n, value, mark)
            return
        own = got.get("own_value")
        if paired and own:
            self.e.log("pair_missing", page=page.oid, value=own)
            n.settle("is", str(own), mark)

    def _scratch(self, clue):
        key = clue.key()
        state = self.hunts.get(key)
        if state is None:
            scratch = Wanted(Dials.Lookup.scratch_index, phrase=clue.category or "listing", kind=clue.category, queries=[])
            state = self.hunts[key] = (scratch, set())
        return state

    def _page_through(self, clue, scratch, checked, query, least):
        for _ in range(Dials.Lookup.pages_per_query):
            if self._short_of(least):
                return
            if not self.e.scout.seek(scratch, text=query, pages=1, floor=Dials.Lookup.search_floor):
                return
            if self._candidates(clue, scratch, checked):
                return

    def _hunt(self, clue, least):
        scratch, checked = self._scratch(clue)
        page = None
        for query in clue.queries():
            if page is not None or self._short_of(least):
                break
            self._page_through(clue, scratch, checked, query, least)
            found = self._candidates(clue, scratch, checked)
            if not found:
                continue
            checked.update(o.oid for o in found)
            page = self._match(clue, scratch, found)
        if page is None:
            self.e.log("page_no_match" if checked else "page_not_found", seller=clue.seller, category=clue.category)
        return page

    def _candidates(self, clue, scratch, checked):
        place = dict(zip(scratch.found, range(len(scratch.found))))
        fresh = [o for o in scratch.found.values() if o.oid not in checked and clue.seller_fits(o)]
        kept = [o for o in fresh if self.e.scout.right_kind(scratch, o)] if clue.category else fresh
        if not kept and clue.seller and len(fresh) <= Dials.Lookup.loose_seller_factor * self.e.assessor.compare_size():
            kept = fresh

        def rank(o):
            return -clue.overlap(o), likeness(o.vendor, clue.seller) < Dials.Like.same, place.get(o.oid, 0)

        return sorted(kept, key=rank)

    @staticmethod
    def _is_read(scratch, oid):
        held = scratch.found.get(oid)
        return held is not None and held.read_full

    @staticmethod
    def _keep_read(scratch, offers):
        scratch.found.update((offer.oid, offer) for offer in offers)

    def _named_in(self, answer, scratch):
        oid = answer.get("listing_id") if isinstance(answer, dict) else None
        return scratch.found.get(oid) if isinstance(oid, str) else None

    def _match(self, clue, scratch, found):
        if clue.pick or (clue.count or 0) > 1:
            page, found = self._pick_among(clue, scratch, found)
            if page is not None:
                return page
        return self._match_rounds(clue, scratch, found)

    def _pick_among(self, clue, scratch, found):
        look = Dials.Lookup
        size = self.e.assessor.compare_size()
        reach = min(len(found), look.pick_reach_factor * size,
                    max(look.pick_reach_min, look.pick_count_factor * (clue.count or look.pick_count_default)))
        for start in range(0, reach, size):
            if self._short_of(look.match_steps):
                break
            end = min(start + self.e.assessor.compare_size(), reach)
            todo = [o.oid for o in found[start:end] if not self._is_read(scratch, o.oid)]
            self._keep_read(scratch, self.e.assessor.read_listings(None, todo))
        read = [scratch.found[o.oid] for o in found if self._is_read(scratch, o.oid)]
        page = None
        if read:
            brief = brief_find_page(clue.category, clue.seller, clue.attrs, read, pick=clue.pick, count=clue.count,
                                    choosing=True)
            page = self._named_in(self.e.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.find_page), scratch)
        if page is not None:
            return page, []
        return None, [o for o in found if not self._is_read(scratch, o.oid)]

    def _match_rounds(self, clue, scratch, found):
        look = Dials.Lookup
        single = not self.e.assessor.side_usable()
        size = 1 if single else self.e.assessor.compare_size()
        rounds = look.single_tries if single else look.batch_tries
        gathered = []
        tries = 0
        while found and tries < rounds and not self._short_of(look.match_steps):
            batch, found = found[:size], found[size:]
            tries += 1
            self._keep_read(scratch, self.e.assessor.read_listings(None, [o.oid for o in batch]))
            now = [scratch.found[o.oid] for o in batch if self._is_read(scratch, o.oid)]
            exact = next((o for o in now if clue.props_match(o)), None)
            if exact is not None:
                return exact
            gathered += now
            if single and found and len(gathered) < look.singles_per_look:
                continue
            shown = gathered if single else now
            gathered = []
            if not shown:
                continue
            brief = brief_find_page(clue.category, clue.seller, clue.attrs, shown)
            answer = self.e.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.find_page)
            worded = [o for o in shown if clue.words_in_text(o)]
            named = self._named_in(answer, scratch)
            if named is not None:
                return named
            if isinstance(answer, dict) and "listing_id" in answer:
                continue
            if len(worded) == 1:
                return worded[0]
            if len(shown) == 1 and not found:
                return shown[0]
        return None


CART_REFUSED = "cart refused"


class Till:

    def __init__(self, errand):
        self.e = errand
        self.cart_qty = {}
        self.cart_for = {}
        self.selection = {}
        self.fallback = None
        self.closest_used = self.unclear_used = False
        self.closest_refusals = 0
        self.walk_gate_done = self.walk_retry_used = False
        self.codes_handled = set()
        self.submits = 0
        self.add_args_bad = False
        self.why_none = ""

    def _wants(self):
        return self.e.order.wants

    def _free(self):
        return self.e.counter.steps_free()

    def _too_much(self, amount):
        return over_limit(amount, self.e.order.budget)

    def can_cart(self):
        return any(self.e.counter.job(role) for role in ("add_to_cart", "replace_cart"))

    def has_lines(self):
        return len(self.cart_qty) > 0

    def basket(self):
        return dict(self.cart_qty)

    def wanted_for(self, oid):
        return self.cart_for.get(oid)

    def pick_of(self, w):
        return self.selection.get(w.slot)

    def sum_of(self, chosen):
        total = 0.0
        for w in self._wants():
            pick = chosen.get(w.slot)
            total += 0.0 if pick is None else w.count * (pick.cost or 0)
        return total

    def _picked_pairs(self):
        return [(w, self.selection[w.slot]) for w in self._wants() if w.slot in self.selection]

    def _all_picked(self):
        return all(w.slot in self.selection for w in self._wants())

    def _cart_mode(self):
        jobs = self.e.counter
        if jobs.job("add_to_cart"):
            return "add"
        return "set" if jobs.job("replace_cart") else ""

    def _replace_all(self, lines):
        jobs = self.e.counter
        spec = jobs.job("replace_cart") or {}
        list_arg = spec.get("lines_arg")
        id_key = spec.get("line_id_key")
        if not list_arg or (lines and not id_key):
            self.e.log("cart_replace_failed", said=None, error="replace_cart spec lacks its line keys")
            return False
        count_key = spec.get("line_count_key")
        item_form = jobs.arg_form("replace_cart", list_arg).get("items") or {}
        top = ((item_form.get("properties") or {}).get(count_key or "") or {}).get("maximum")
        clamp = isinstance(top, int)

        def line_row(oid, qty):
            row = dict.fromkeys([id_key], oid)
            if count_key:
                row[count_key] = min(qty, top) if clamp else qty
            return row

        out = jobs.act(spec.get("tool"), {list_arg: list(map(line_row, lines, lines.values()))})
        if not out.ok:
            self.e.log("cart_replace_failed", said=json_cut(out.body, Dials.Trace.long), error=out.fault)
        return out.ok

    def _adopt(self, planned, oid, w=None):
        if not self._replace_all(planned):
            return False
        self.cart_qty = planned
        if w is None:
            self.cart_for.pop(oid, None)
        else:
            self.cart_for[oid] = w
        return True

    def _switch_pair(self, which):
        sw = self.e.counter.job_field("add_to_cart", "switch")
        if not isinstance(sw, dict) or not sw.get("arg_name") or sw.get(which) is None:
            return None
        return sw["arg_name"], sw[which]

    def _add_args(self, spec, oid, count):
        args = {spec["id_arg"]: oid}
        count_arg = spec.get("count_arg")
        if count_arg:
            top = self.e.counter.arg_form("add_to_cart", count_arg).get("maximum")
            count = min(count, top) if isinstance(top, int) else count
            args[count_arg] = count
        pair = self._switch_pair("add_value")
        if pair:
            args.update([pair])
        return count, args

    def _forget(self, oid):
        for book in (self.cart_qty, self.cart_for):
            book.pop(oid, None)

    def add_line(self, w, o):
        mode = self._cart_mode()
        if mode == "set":
            return self._adopt({**self.cart_qty, o.oid: w.count}, o.oid, w)
        if mode != "add":
            return False
        spec = self.e.counter.job("add_to_cart")
        qty, args = self._add_args(spec, o.oid, w.count)
        out = self.e.counter.act(spec["tool"], args)
        if not out.ok or cart_said_no(out.body):
            self.add_args_bad = bool(out.args_bad)
            self.e.log("line_add_failed", said=json_cut(out.body, Dials.Trace.long), error=out.fault)
            return False
        self.cart_for[o.oid] = w
        self.cart_qty[o.oid] = qty
        return True

    def _drop_count(self, oid, switched):
        count_arg = self.e.counter.job_field("add_to_cart", "count_arg")
        if not count_arg:
            return None
        if switched:
            return self.cart_qty.get(oid, Dials.Request.min_quantity)
        low = self.e.counter.arg_form("add_to_cart", count_arg).get("minimum")
        return low if isinstance(low, int) and low > 0 else 0

    def drop_line(self, oid):
        mode = self._cart_mode()
        if mode == "set":
            return self._adopt({k: q for k, q in self.cart_qty.items() if k != oid}, oid)
        if mode != "add":
            return False
        spec = self.e.counter.job("add_to_cart")
        args = {spec["id_arg"]: oid}
        pair = self._switch_pair("remove_value")
        if pair:
            args.update([pair])
        count = self._drop_count(oid, pair is not None)
        if count is not None:
            args[spec["count_arg"]] = count
        done = self.e.counter.act(spec["tool"], args).ok
        if done:
            self._forget(oid)
        return done

    def recheck_lines(self):
        queue = list(self.cart_for.items())
        while queue and self._free() > Dials.Ending.refresh_stop:
            oid, w = queue.pop(0)
            self.e.assessor.read_listings(w, [oid], own_page=True)
            now = w.found.get(oid)
            price, stock = (None, None) if now is None else (now.cost, now.in_stock)
            self.e.log("line_rechecked", stock=stock, price=price, id=oid)

    def remember_near(self, chosen):
        self.fallback = dict(chosen)

    def order_these(self, chosen):
        self.selection = {w.slot: chosen[w.slot] for w in self._wants()}
        self.submit()

    def picks_broken(self):
        over = self._too_much(self.sum_of(self.selection))
        return any(o.in_stock is False or (over and o.cost is not None) for _, o in self._picked_pairs())

    def _fill_cart(self):
        keep = {o.oid for o in self.selection.values()}
        for oid in [k for k in self.cart_qty if k not in keep]:
            self.drop_line(oid)
        for w, pick in self._picked_pairs():
            if pick.oid in self.cart_qty or self.add_line(w, pick):
                continue
            if self.add_args_bad:
                self.e.log("cart_args_rejected")
                self.walk_away("the cart tool keeps rejecting its arguments")
                return False
            w.flag_miss(pick.oid, CART_REFUSED)
            self.selection.pop(w.slot)
        return True

    def _look_after_add(self):
        reader = self.e.assessor

        def room():
            return self._free() > Dials.Ending.look_steps + self.e.parley.owed()

        if reader.table_first():
            for w, pick in self._picked_pairs():
                if pick.page_seen or not room():
                    continue
                reader.read_listings(w, [pick.oid], own_page=True)
            return
        wants = self._wants()
        if room() and wants and wants[0].slot in self.selection:
            reader.read_listings(wants[0], [self.selection[wants[0].slot].oid])

    def _rejudge_picks(self):
        failing = False
        stale = [(w, o) for w, o in self._picked_pairs() if w.judged_lines(o.oid) is None]
        for w, o in stale:
            self.e.assessor.assess(w, [o])
            missed = sorted((w.ruling(o.oid) or {}).get("missed") or ())
            if missed:
                failing = True
                self.e.log("pick_fails_now", failed=missed[:Dials.Trace.few], id=o.oid)
        return failing

    def submit(self, attempt=0):
        self.submits += 1
        if self.submits > Dials.Ending.checkout_rounds or attempt > Dials.Ending.checkout_tries:
            return
        if self.e.counter.closed or not self._fill_cart():
            return
        if not self._all_picked():
            self.e.picker.pick_all()
            return
        self._look_after_add()
        heard = self.e.parley.absorb_lines()
        moved = self._rejudge_picks() or heard
        if not moved and not self.picks_broken():
            if self.ring_up():
                return
            self._read_refusal()
        else:
            self.e.log("repick_after_change", cart=dict(self.cart_qty))
            for w in self._wants():
                self.e.assessor.reassess(w)
        self.repick()
        if self.e.counter.closed:
            return
        self.submit(attempt=attempt + 1)

    def _read_refusal(self):
        talk = self.e.parley
        if talk.absorb_lines():
            return
        self.recheck_lines()
        talk.absorb_lines()

    @staticmethod
    def _sellable(offers):
        return [o for o in offers if o.in_stock is not False]

    def _search_again(self, w, cheap):
        e = self.e
        e.assessor.reassess(w)
        if self._sellable(w.fitting()) and not cheap:
            return
        e.rearm_hunt(w)
        share = max(self._free() - e.ending_room(), Dials.Allowance.min_share)
        if e.seconds_left() > Dials.Timing.pick_again_over and share >= Dials.Ending.pick_again_steps:
            e.hunt(w, share, True)

    def _pick_from_fits(self, w, cheap):
        fits = self._sellable(w.fitting(cheapest_first=cheap))
        if not fits:
            self.selection.pop(w.slot, None)
            return
        held = self.selection.get(w.slot)
        keep = held is not None and not cheap and any(f.oid == held.oid for f in fits)
        if not keep:
            self.selection[w.slot] = fits[0]

    def repick(self):
        cheap = self.e.order.lowest_price
        for w in self._wants():
            self._search_again(w, cheap)
        for w in self._wants():
            self._pick_from_fits(w, cheap)
        self._rules_again()
        if not self._all_picked() or self._too_much(self.sum_of(self.selection)):
            self.e.picker.pick_all()

    def _rules_again(self):
        picker = self.e.picker
        frozen = dict((w.slot, o) for w, o in self._picked_pairs())
        targets = {}
        for rule in picker.rules():
            targets.update((arm.dest.slot, arm.dest) for arm in rule.arms)
        for dest in targets.values():
            picker.impose_rules(dest, frozen)
            ok_ids = [o.oid for o in picker.ready_fits(dest)]
            held = self.selection.get(dest.slot)
            if held is None or held.oid in ok_ids:
                continue
            if not ok_ids:
                del self.selection[dest.slot]
                continue
            self.selection[dest.slot] = dest.found[ok_ids[0]]

    def ring_up(self):
        jobs = self.e.counter
        tool = jobs.tool_for("buy_cart")
        if not tool:
            return False
        if self.e.parley.before_checkout() and self._changed_by_steps():
            self.e.log("basket_changed_at_steps")
            return False
        self._show_linked_codes()
        out = jobs.act(tool, jobs.ending_args("buy_cart"))
        if not jobs.closed:
            self.e.log("order_turned_down", lines=out.said[:Dials.Trace.few], said=json_cut(out.body, Dials.Trace.long))
            return False
        self.e.log("order_placed", cart=dict(self.cart_qty))
        return True

    def _changed_by_steps(self):
        return self.e.parley.absorb_lines() or self.picks_broken()

    def _home_of(self, oid):
        return next((w.found[oid] for w in self._wants() if oid in w.found), None)

    def _code_kept(self, code, codes, home, every_title):
        if len(code) < Dials.Ending.code_min or not has_digit(code):
            return False
        if code in codes or code.lower() in self.codes_handled or names_code(home.headline or "", code):
            return False
        seen = names_code(every_title, code)
        if seen:
            self.codes_handled.add(code.lower())
        return not seen

    def _pointed_codes(self):
        every_title = " ".join(o.headline or "" for w in self._wants() for o in w.found.values())
        codes = []
        for home in filter(None, map(self._home_of, list(self.cart_qty))):
            text = (home.blurb or "") + " " + (home.remarks or "")
            for raw in pointed_codes(text):
                code = raw.strip("-")
                if self._code_kept(code, codes, home, every_title):
                    codes.append(code)
        return codes

    def _show_linked_codes(self):
        jobs = self.e.counter
        spec = jobs.job("keyword_search")
        if not self.cart_qty or not spec or not spec.get("words_arg"):
            return
        codes = self._pointed_codes()
        if not codes or self._free() < self.e.parley.owed() + Dials.Ending.parts_steps:
            return
        sent = codes[:Dials.Ending.codes_shown]
        words_arg = spec["words_arg"]
        sw = spec.get("switch")
        args = {sw["arg_name"]: sw.get("arg_value")} if isinstance(sw, dict) and sw.get("arg_name") else {}
        args[words_arg] = jobs.clip_arg("keyword_search", words_arg, " ".join(sent))
        body = jobs.act(spec["tool"], args).body
        self.codes_handled.update(map(str.lower, sent))
        self.e.log("linked_codes", rows=len(row_list(body) or []), codes=sent)

    def _refill_unsellable(self):
        lacking = [w for w in self._wants() if getattr(self.selection.get(w.slot), "in_stock", False) is False]
        for w in lacking:
            ready = self.e.picker.ready_fits(w)
            if ready:
                self.selection[w.slot] = ready[0]
            else:
                self.selection.pop(w.slot, None)

    def close_out(self):
        if self.e.counter.closed:
            return
        self._refill_unsellable()
        wants = self._wants()
        if wants and self._all_picked() and not self._too_much(self.sum_of(self.selection)):
            if self._sync_and_buy(wants):
                return
        if not self.e.counter.closed:
            self.walk_away("forced")

    def _sync_and_buy(self, wants):
        picked = {o.oid for o in self.selection.values()}
        stale = [oid for oid in self.cart_qty if oid not in picked]
        adds = [w for w in wants if self.selection[w.slot].oid not in self.cart_qty]
        if len(stale) + len(adds) + Dials.Ending.finish_step > self._free():
            return False
        for oid in stale:
            self.drop_line(oid)
        for w in adds:
            self.add_line(w, self.selection[w.slot])
        return self.ring_up()

    def _other_order(self):
        order = self.e.order
        if order.closest_ok:
            return self.buy_closest()
        return not order.says_decline and self.buy_unclear()

    def _decline_args(self):
        jobs = self.e.counter
        args = dict(jobs.ending_args("end_unbought"))
        remark = jobs.job_field("end_unbought", "remark_arg")
        needed = jobs.form_of("end_unbought").get("required") or []
        if remark and remark in needed:
            args[remark] = jobs.clip_arg("end_unbought", remark, no_fit_remark())
        return args

    def walk_away(self, reason):
        if reason:
            self.why_none = reason
        jobs = self.e.counter
        if self._other_order() or jobs.closed:
            return True
        tool = jobs.tool_for("end_unbought")
        if not tool:
            return False
        if not self.walk_gate_done and self.e.parley.has_duties():
            self.walk_gate_done = True
            self.e.parley.run_due("before_decline")
        args = self._decline_args()
        out = jobs.act(tool, args)
        self.e.log("walked_away", reason=self.why_none)
        if not (jobs.closed or self.walk_retry_used or self._free() < Dials.Ending.finish_step):
            self._walk_again(tool, args, out)
        return jobs.closed

    def _walk_again(self, tool, args, out):
        self.walk_retry_used = True
        self.e.log("walk_refused", lines=out.said[:Dials.Trace.few], said=json_cut(out.body, Dials.Trace.long))
        need = len(self._wants()) + Dials.Ending.rechoice_extra
        if self.e.parley.absorb_lines() and self._free() >= need:
            self.e.picker.pick_all()
        if self.e.counter.closed:
            return
        for oid in list(self.cart_qty):
            if Dials.Ending.finish_step < self._free():
                self.drop_line(oid)
        self.e.counter.act(tool, args)

    def _place_plan(self, plan, wants):
        ids = {o.oid for o in plan.values()}
        stale = [oid for oid in self.cart_qty if oid not in ids]
        adds = [w for w in wants if w.slot in plan and plan[w.slot].oid not in self.cart_qty]
        self.e.log("closest_order", picks=dict((w.phrase, plan[w.slot].oid) for w in wants if w.slot in plan))
        for oid in stale:
            if len(adds) + Dials.Ending.finish_step < self._free():
                self.drop_line(oid)
        for w in adds:
            if Dials.Ending.add_steps > self._free():
                break
            self.selection[w.slot] = plan[w.slot]
            self.add_line(w, self.selection[w.slot])
        return bool(self.cart_qty) and self._free() >= Dials.Ending.finish_step

    def buy_closest(self):
        wants = self._wants()
        jobs = self.e.counter
        if not wants or self.closest_used or not jobs.job("buy_cart"):
            return False
        self.closest_used = True
        plan = self._closest_plan(wants)
        if not plan or not self._place_plan(plan, wants):
            return False
        if self.ring_up() or jobs.closed:
            return True
        self.closest_refusals += 1
        heard = self.e.parley.absorb_lines()
        if not heard or self.closest_refusals > Dials.Ending.near_retries:
            return False
        if self._free() < len(wants) + Dials.Ending.finish_step:
            return False
        self.closest_used, self.fallback = False, None
        self.e.picker.pick_all()
        return jobs.closed

    def _best_or_miss(self, w):
        ranked = self.e.picker.best_ready(w)
        return ranked[0] if ranked else self.closest_miss(w)

    def _closest_plan(self, wants):
        kept = self.fallback or {}
        plan = {}
        for w in wants:
            pick = kept.get(w.slot) or self._best_or_miss(w)
            if pick is not None:
                plan[w.slot] = pick
        return plan

    def _miss_score(self, w, o, ruling, usage):
        missed = set(ruling.get("missed") or ())
        gaps = (o.in_stock is False, w.too_dear(o), o.cost is not None and over_limit(o.cost * w.count, self.e.order.budget))
        short = sum(map(bool, gaps)) + len(missed | self.e.assessor.plain_misses(w, o, usage))
        return short, len(ruling.get("doubtful") or ()), price_rank(o.cost)

    def closest_miss(self, w):
        usage = self.e.assessor.field_usage(w)
        scored = []
        for o in filter(lambda x: x.read_full, w.found.values()):
            ruling = w.ruling(o.oid)
            if ruling is None or not self.e.scout.right_kind(w, o):
                continue
            if CART_REFUSED in (ruling.get("missed") or ()):
                continue
            scored.append((self._miss_score(w, o, ruling, usage), o))
        return min(scored, key=lambda pair: pair[0])[1] if scored else None

    def _clear_enough(self, w, o, usage):
        if o.in_stock is False or self.e.assessor.plain_misses(w, o, usage):
            return False
        return not w.synonym_lines(o.oid) and CART_REFUSED not in (w.ruling(o.oid) or {}).get("missed", ())

    def _unclear_pick(self, w):
        pool = self.e.picker.best_ready(w)
        if pool:
            return pool[0]
        usage = self.e.assessor.field_usage(w)
        doubtful = w.fitting(cheapest_first=self.e.order.lowest_price, with_doubts=True)
        return next((o for o in doubtful if self._clear_enough(w, o, usage)), None)

    def buy_unclear(self):
        if self.closest_used or self.unclear_used or not self.e.counter.job("buy_cart"):
            return False
        self.unclear_used = True
        plan = {}
        for w in self._wants():
            pick = self._unclear_pick(w)
            if pick is None:
                return False
            plan[w.slot] = pick
        if self._too_much(self.sum_of(plan)):
            return False
        self.e.log("unclear_order", picks=dict((w.phrase, plan[w.slot].oid) for w in self._wants()))
        self.fallback = plan
        return self.buy_closest()


FIT_PROP = "compatibility"


class Arm:

    def __init__(self, holds_side, dest, prop, op, value):
        self.holds_side, self.dest = holds_side, dest
        self.prop, self.op, self.value = prop, op, value

    def need(self):
        return Need(self.prop, self.op or "is", self.value, source=FROM_RULE)


class IfRule:

    def __init__(self, watch, prop, values, arms):
        self.watch, self.prop, self.values, self.arms = watch, prop, values, arms

    def steers(self, w):
        return any(a.dest is w for a in self.arms)


def in_family(o, heads):
    return any(h in heads for h in series_heads(o.headline or ""))


class Picker:

    def __init__(self, errand):
        self.e = errand
        self.measure_memo = {}
        self.spares_read = self.wide_used = self.opinion_used = False
        self.pinned = None
        self.rule_memo = {}
        self.if_rules = []

    def rules(self):
        return self.if_rules

    def reserve_pin(self):
        return self.pinned

    def spares_done(self):
        return self.spares_read

    def _wants(self):
        return self.e.order.wants

    def _room(self):
        return self.e.counter.steps_free() - self.e.ending_room()

    def _pin(self):
        talk = self.e.parley
        yes = Dials.Choose.override_confirm if talk.needs_yes() else 0
        self.pinned = max(talk.reserved_msgs(), yes) + Dials.Choose.override_extra + len(self._wants())

    def _some_ready(self, wants):
        return next((True for w in wants if self.ready_fits(w)), False)

    def _all_ready(self, wants):
        return next((False for w in wants if not self.ready_fits(w)), True)

    def ready_fits(self, w):
        reader = self.e.assessor
        usage = reader.field_usage(w)

        def sound(o):
            return not reader.plain_misses(w, o, usage) and o.in_stock is not False

        return list(filter(sound, w.fitting(cheapest_first=self.e.order.lowest_price)))

    def _ask_measure(self, w, measure, offers, ids):
        brief = brief_measure_order(measure, w.phrase, offers[:Dials.Choose.measure_rows])
        got = self.e.oracle.ask_json(Dials.Model.planner_spare, brief, Dials.Tokens.measure_order)
        order = list(map(str, got.get("ranked_ids") or [])) if isinstance(got, dict) else []
        self.e.log("measure_order", order=order[:Dials.Trace.many], measure=measure)
        if order:
            self.measure_memo[w.slot] = (ids, order)
        return order

    def by_measure(self, w, offers):
        measure = self.e.order.measure
        if len(offers) < Dials.Choose.measure_min or not measure:
            return offers
        ids = [o.oid for o in offers]
        known = self.measure_memo.get(w.slot)
        order = known[1] if known and known[0] == ids else self._ask_measure(w, measure, offers, ids)
        spot = {oid: at for at, oid in enumerate(order)}
        return sorted(offers, key=lambda o: (o.oid not in spot, spot.get(o.oid, 0), ids.index(o.oid)))

    def best_ready(self, w):
        return self.by_measure(w, self.ready_fits(w))

    def _spare_rows(self, w, bound):
        budget = self.e.order.budget

        def worth(o):
            if o.read_full or o.in_stock is False or o.sift_mark == "no" or o.cost is None:
                return False
            return (bound is None or bound >= o.cost) and not over_limit(o.cost * w.count, budget)

        return sorted(filter(worth, self.e.scout.shortlist(w)), key=lambda o: o.cost)

    def read_spares(self):
        self.spares_read = True
        e = self.e
        if not self._all_ready(self._wants()):
            return
        for w in self._wants():
            if e.seconds_left() < Dials.Timing.runner_up_until or self._room() < Dials.Choose.runner_up_steps:
                break
            fits = self.ready_fits(w)
            bound = None if not fits else (fits[1] if len(fits) > 1 else fits[0]).cost
            rows = self._spare_rows(w, bound)
            if not rows:
                continue
            ids = [o.oid for o in rows][:e.assessor.compare_size()]
            got = e.assessor.read_listings(w, ids, floor=e.ending_room())
            e.log("spares", bound=bound, read=[o.oid for o in got], item=w.phrase)
            if got:
                e.assessor.assess(w, got)

    def _layout(self):
        pairs = self.pairings()
        host_of = dict((acc.slot, host) for acc, host in pairs)
        hosts = set(host.slot for _, host in pairs)
        watched = set(r.watch.slot for r in self.if_rules)
        steered = set(a.dest.slot for r in self.if_rules for a in r.arms)
        return pairs, host_of, hosts | watched, steered

    def pick_all(self):
        e = self.e
        e.parley.absorb_lines()
        if not self.spares_read and e.order.lowest_price:
            self.read_spares()
        e.assessor.assess_all()
        pairs, host_of, leading, steered = self._layout()
        self._verify_tops(steered | set(host_of))
        chosen = {}

        def turn(w):
            return (w.slot in host_of or w.slot in steered, w.slot not in leading, w.slot)

        for w in sorted(self._wants(), key=turn):
            self._choose_one(w, chosen, host_of.get(w.slot), w.slot in steered, pairs)
        missing = [w for w in self._wants() if w.slot not in chosen]
        if missing and (self._last_search(missing) or self._try_opinion(missing)):
            self.pick_all()
            return
        e.till.remember_near(chosen)
        if missing:
            self._give_up(missing)
        elif self._fit_budget(chosen):
            e.till.order_these(chosen)

    def _give_up(self, missing):
        names = [w.phrase for w in missing]
        self.e.log("nothing_fits", pools=dict((w.phrase, len(w.found)) for w in missing), items=names)
        self.e.till.walk_away("no fitting offer for " + ", ".join(names))

    def _verify_tops(self, waiting):
        tops = [(w, self.best_ready(w)) for w in self._wants() if w.slot not in waiting]
        approve = self.e.assessor.approve
        self.e.oracle.run_many([lambda w=w, o=top[0]: approve(w, o) for w, top in tops if top])

    def _first_approved(self, w):
        e = self.e
        ladder = iter(self.best_ready(w))
        tries = 0
        while tries < Dials.Choose.check_tries and e.seconds_left() >= Dials.Timing.choose_until:
            o = next(ladder, None)
            if o is None:
                break
            tries += 1
            if e.assessor.approve(w, o):
                return o, tries
        return None, tries

    def _choose_one(self, w, chosen, host, steered, pairs):
        e = self.e
        if host is not None and host.slot in chosen:
            self.tie_to_host(w, host, chosen[host.slot])
        if steered:
            self.impose_rules(w, chosen)
        pick, tries = self._first_approved(w)
        if pick is None:
            rest = self.best_ready(w)
            out_of_tries = tries >= Dials.Choose.check_tries or e.seconds_left() < Dials.Timing.choose_until
            pick = rest[0] if rest and out_of_tries else None
        if pick is not None:
            chosen[w.slot] = pick
            return
        if host is not None:
            self._other_host(w, host, chosen)
        if steered and w.slot not in chosen:
            self._other_source(w, chosen, pairs)

    def _other_host(self, w, host, chosen):
        for alt in self.ready_fits(host)[:Dials.Choose.host_options]:
            if self.e.seconds_left() < Dials.Timing.choose_until or chosen.get(host.slot) is alt:
                continue
            self.tie_to_host(w, host, alt)
            got = self.ready_fits(w)
            if not got:
                continue
            chosen.update({host.slot: alt, w.slot: got[0]})
            return

    def _with_source(self, w, chosen, src, pick):
        chosen[src.slot] = pick
        self.impose_rules(w, chosen)
        return self.ready_fits(w)

    def _flip_source(self, w, rule, chosen):
        src = rule.watch
        base = chosen[src.slot]
        now = self.rule_met(rule, base)
        for alt in self.ready_fits(src)[:Dials.Choose.source_options]:
            if alt is base or now == self.rule_met(rule, alt):
                continue
            got = self._with_source(w, chosen, src, alt)
            if got and self.e.assessor.approve(src, alt):
                chosen[w.slot] = got[0]
                return True
            self._with_source(w, chosen, src, base)
        return False

    def _other_source(self, w, chosen, pairs):
        mine = [r for r in self.if_rules if r.watch.slot in chosen and r.steers(w)]
        for rule in mine:
            tied = any(acc.slot in chosen and h is rule.watch for acc, h in pairs)
            if tied or self.e.seconds_left() < Dials.Timing.choose_until:
                continue
            if self._flip_source(w, rule, chosen):
                return

    def _last_search(self, missing):
        e = self.e
        if self.wide_used or e.seconds_left() <= Dials.Timing.extra_search_over or e.oracle.over_spend():
            return False
        self.wide_used = True
        self._pin()
        try:
            for w in missing:
                room = self._room()
                if room >= Dials.Choose.extra_search_steps:
                    e.hunt(w, room, True)
        finally:
            self.pinned = None
        return self._some_ready(missing)

    def _try_opinion(self, missing):
        e = self.e
        if self.opinion_used or e.seconds_left() <= Dials.Timing.recheck_over or e.oracle.over_spend():
            return False
        self.opinion_used = True
        return self.second_opinion(missing) and self._all_ready(missing)

    def _fit_budget(self, chosen):
        budget = self.e.order.budget
        if budget is None:
            return True
        till = self.e.till
        for _ in range(Dials.Choose.repair_moves):
            if not over_limit(till.sum_of(chosen), budget) or not self._step_along(chosen):
                break
        total = till.sum_of(chosen)
        if not over_limit(total, budget):
            return True
        self.e.log("total_too_high", cap=budget, total=total)
        till.walk_away("even the cheapest fitting set is over the budget")
        return False

    def _step_along(self, chosen):
        def line(w):
            return w.count * (chosen[w.slot].cost or 0)

        for w in sorted(self._wants(), key=line, reverse=True):
            ladder = list(filter(lambda o: o.in_stock is not False, w.fitting(cheapest_first=True)))
            at = next((i for i, o in enumerate(ladder) if o.oid == chosen[w.slot].oid), None)
            if at is not None and at + 1 < len(ladder):
                chosen[w.slot] = ladder[at + 1]
                return True
        return False

    def second_opinion(self, missing):
        learned = False
        self._pin()
        try:
            for w in missing:
                if self.e.seconds_left() < Dials.Timing.recheck_cards_until:
                    break
                learned = self._look_again(w) or learned
        finally:
            self.pinned = None
        return learned

    def _worth_showing(self, w, o, budget):
        if not self.e.scout.right_kind(w, o) or o.in_stock is False:
            return False
        return o.cost is None or not over_limit(o.cost * w.count, budget)

    def _seen_rows(self, w):
        budget = self.e.order.budget
        seen = sorted((o for o in w.found.values() if self._worth_showing(w, o, budget)),
                      key=lambda o: (not o.read_full, price_rank(o.cost)))

        def row(o):
            ruling = (w.ruling(o.oid) or {}) if o.read_full else {}
            return o, list(ruling.get("missed") or []), list(ruling.get("doubtful") or [])

        return list(map(row, seen[:Dials.Recheck.cards]))

    def _look_again(self, w):
        rows = self._seen_rows(w)
        if not rows:
            return False
        e = self.e
        tried = sorted(set(e.scout.ran_queries(w)))
        brief = brief_second_look(w.phrase, w.kind, w.count, e.order.budget, w.need_lines(), tried, rows)
        got = e.oracle.ask_json(Dials.Model.planner_light, brief, Dials.Tokens.second_look)
        if not isinstance(got, dict):
            return False
        e.log("second_opinion", reply=json_cut(got, Dials.Trace.full), item=w.phrase)
        steps = (self._take_fits(w, got), self._open_named(w, got), self._run_named(w, got, tried))
        e.assessor.reassess(w)
        return any(steps)

    def _take_fits(self, w, got):
        reader = self.e.assessor
        named = list(map(str, got.get("fit_ids") or []))[:Dials.Recheck.fits_taken]
        learned = False
        for o in filter(None, map(w.found.get, named)):
            if o.read_full and not reader.plain_misses(w, o) and reader.approve_now(w, o, strict=True):
                w.set_ruling(o.oid, blank_ruling(full=True), against=w.need_lines())
                learned = True
        return learned

    def _open_named(self, w, got):
        named = list(map(str, got.get("open_next") or []))
        fresh = [x for x in named if x in w.found and not w.found[x].read_full][:Dials.Recheck.unopened]
        if not fresh or self._room() < Dials.Recheck.open_steps:
            return False
        return len(self.e.assessor.read_listings(w, fresh, floor=self.e.ending_room())) > 0

    def _query_pays(self, w, q):
        e = self.e
        if e.scout.seek(w, text=q[:Dials.Query.query_cut], floor=e.ending_room(), pages=Dials.Recheck.query_pages) <= 0:
            return False
        e.scout.triage(w, [o for o in e.scout.shortlist(w) if not o.read_full])
        unread = [o.oid for o in e.scout.shortlist(w) if o.sift_mark != "no" and not o.read_full]
        unread = unread[:Dials.Recheck.rows_per_query]
        return bool(unread) and bool(e.assessor.read_listings(w, unread, floor=e.ending_room()))

    def _run_named(self, w, got, tried):
        e = self.e
        known = set(map(simplify, tried))
        queries = [q for q in (str(x).strip() for x in got.get("new_queries") or []) if q][:Dials.Recheck.queries]
        learned = False
        for q in queries:
            if e.seconds_left() < Dials.Timing.recheck_query_until or self._room() < Dials.Recheck.query_steps:
                break
            if simplify(q) not in known and self._query_pays(w, q):
                learned = True
        return learned

    def wanted_called(self, name):
        text = str(name or "")
        if not text or not self._wants():
            return None

        def score(w):
            return max(likeness(text, w.phrase), likeness(text, w.kind))

        best = max(self._wants(), key=score)
        return None if score(best) < Dials.Like.loose else best

    @staticmethod
    def _value_list(raw):
        if not isinstance(raw, list):
            return [str(raw)] if raw else []
        return list(map(str, filter(None, raw)))

    def _arm(self, side, spec, watch):
        if not isinstance(spec, dict):
            return None
        prop, value = spec.get("property_name"), spec.get("required_value")
        if not prop or not value:
            return None
        dest = self.wanted_called(spec.get("product"))
        if dest is None or dest is watch:
            return None
        return Arm(side, dest, prop, spec.get("comparison"), value)

    def _seed_terms(self, arm):
        dest = arm.dest
        for piece in filter(None, or_parts(arm.value)):
            if len(piece) > Dials.Query.value_term_max:
                continue
            form = simplify(piece)
            if any(form in simplify(q) for q in dest.queries):
                continue
            dest.append_query((piece + " " + dest.headword)[:Dials.Query.query_cut])

    def _rule_from(self, entry):
        when = entry.get("when") if isinstance(entry, dict) else None
        if not isinstance(when, dict) or not when.get("property_name"):
            return None
        watch = self.wanted_called(when.get("product"))
        values = self._value_list(when.get("any_of"))
        sides = ((True, entry.get("then_need")), (False, entry.get("else_need")))
        arms = [a for a in (self._arm(side, spec, watch) for side, spec in sides) if a is not None]
        if watch is None or not values or not arms:
            return None
        return IfRule(watch, str(when["property_name"]), values, arms)

    def load_rules(self, raw):
        kept = []
        for entry in raw if isinstance(raw, list) else []:
            rule = self._rule_from(entry)
            if rule is None:
                continue
            kept.append(rule)
            for arm in rule.arms:
                self._seed_terms(arm)
        self.if_rules = kept
        if kept:
            shown = [(r.watch.phrase, r.prop, r.values, [(a.holds_side, a.dest.phrase, a.prop, a.value) for a in r.arms])
                     for r in kept]
            self.e.log("rules_read", rules=shown)
        return kept

    @staticmethod
    def _table_says(rule, o):
        own = [str(o.traits[k]) for k in o.traits if names_match(k, rule.prop)]
        if not own:
            return None
        have = wordset_full(" ".join(own))
        wanted = list(filter(None, map(wordset_full, rule.values)))
        if any(have >= ws for ws in wanted):
            return True
        return None if any(have & ws for ws in wanted) else False

    def rule_met(self, rule, o):
        key = (id(rule), o.oid, keyed_json(o.traits))
        if key not in self.rule_memo:
            self.rule_memo[key] = self._decide_rule(rule, o)
            self.e.log("rule_checked", holds=self.rule_memo[key], prop=rule.prop, offer=o.oid, watch=rule.watch.phrase)
        return self.rule_memo[key]

    def _decide_rule(self, rule, o):
        verdict = self._table_says(rule, o)
        if verdict is not None:
            return verdict
        brief = brief_condition_check(rule.prop, rule.values, o)
        got = self.e.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.condition_check)
        said = got.get("matches") if isinstance(got, dict) else None
        return said if isinstance(said, bool) else None

    def _arms_now(self, rule, w, chosen):
        pick = chosen.get(rule.watch.slot)
        side = None if pick is None else self.rule_met(rule, pick)
        return [] if side is None else [a for a in rule.arms if a.holds_side == side and a.dest is w]

    def impose_rules(self, w, chosen):
        mine = [r for r in self.if_rules if r.steers(w)]
        if not mine:
            return
        fresh = [a.need() for rule in mine for a in self._arms_now(rule, w, chosen)]
        new_lines = [n.line() for n in fresh]
        if sorted(new_lines) == sorted(n.line() for n in w.needs if n.source == FROM_RULE):
            return
        w.swap_needs(FROM_RULE, fresh)
        self.e.log("rule_needs", needs=new_lines, item=w.phrase)
        self.e.assessor.reassess(w)
        e = self.e
        if self._room() < Dials.Choose.condition_steps or e.seconds_left() <= Dials.Timing.conditions_over:
            return
        if not self.ready_fits(w):
            e.hunt(w, self._room(), True)

    def rule_props(self, w):
        props = []
        for rule in self.if_rules:
            props.extend(([rule.prop] if rule.watch is w else []) + [str(a.prop) for a in rule.arms if a.dest is w])
        if any(acc is w for acc, _ in self.pairings()):
            props.append(FIT_PROP)
        return list(dict.fromkeys(props))

    @staticmethod
    def _likeliest(wants, name):
        best = max(wants, key=lambda w: likeness(w.phrase, name))
        return best, likeness(best.phrase, name)

    def pairings(self):
        wants = self._wants()
        pairs = []
        if len(wants) < Dials.Fit.items_min:
            return pairs
        for ask in filter(lambda x: isinstance(x, dict), self.e.order.match_rules):
            add_on, main = str(ask.get("add_on") or ""), str(ask.get("main") or "")
            if not (add_on and main):
                continue
            (acc, near_acc), (host, near_host) = self._likeliest(wants, add_on), self._likeliest(wants, main)
            if acc is host or min(near_acc, near_host) < Dials.Like.loose or (acc, host) in pairs:
                continue
            pairs.append((acc, host))
        return pairs

    def _headlines(self, wants):
        return (o.headline or "" for x in wants for o in x.found.values())

    def _codes_to_look_up(self, w):
        titled = set(c for t in self._headlines(self._wants()) for c in title_codes(t))
        prefixes = set(map(letter_head, (c for t in self._headlines([w]) for c in title_codes(t))))
        found = []
        sound = [o for o in self.e.scout.shortlist(w) if o.read_full and not (w.ruling(o.oid) or {}).get("missed")]
        for text in ((o.blurb or "") + " " + (o.remarks or "") for o in sound):
            for c in title_codes(text):
                if c not in found and c not in titled and letter_head(c) in prefixes:
                    found.append(c)
        return found

    def _series_named_by(self, host_w, host):
        page = (host.blurb or "") + " " + (host.remarks or "") + " " + to_json(host.traits)
        others = [x for x in self._wants() if x is not host_w]
        heads = set(h for t in self._headlines(others) for h in series_heads(t))
        families = set(map(letter_head, heads))
        own = series_heads(host.headline or "")
        named = []
        for code in short_series(page):
            fits = code in heads or letter_head(code) in families
            if fits and code not in named and code not in own:
                named.append(code)
        return named

    def _look_up(self, w, codes, series):
        e = self.e
        firsts = series[:Dials.Fit.series_used] + [c for c in codes if c not in series][:Dials.Fit.codes_looked]
        for code in firsts:
            if e.seconds_left() < Dials.Timing.fit_until or e.counter.steps_free() <= e.ending_room() + Dials.Fit.step_spare:
                return
            e.scout.seek(w, floor=e.ending_room(), text=(code + " " + w.kind) if code in series else code)

    def _read_family(self, w, series):
        e = self.e
        if e.ending_room() + Dials.Fit.step_spare >= e.counter.steps_free():
            return
        kin = [o.oid for o in e.scout.shortlist(w) if not o.read_full and in_family(o, series)]
        if not kin:
            return
        take = Dials.Fit.single_reads if not e.assessor.batch_reads() else e.assessor.compare_size()
        e.assessor.read_listings(w, kin[:take], floor=Dials.Fit.read_floor_spare + e.ending_room())

    def tie_to_host(self, w, host_w, host):
        e = self.e
        w.swap_needs(FROM_FIT, [Need(FIT_PROP, "is", fit_need_value(host_w.phrase, host), source=FROM_FIT)])
        codes = self._codes_to_look_up(w)
        series = self._series_named_by(host_w, host)
        self._look_up(w, codes, series)
        if series:
            self._read_family(w, series)
        e.log("host_tied", looked_up=codes[:Dials.Fit.codes_looked], series=series[:Dials.Fit.series_used],
              host=host.oid, item=w.phrase)
        e.assessor.reassess(w)
        if series:
            fit_line = w.needs[-1].line()
            credited = [o for o in e.scout.shortlist(w) if o.read_full and w.ruling(o.oid) and in_family(o, series)]
            for o in credited:
                w.unflag(o.oid, fit_line)


"""c13: the errand. One shopping session: it builds the parts, runs the phases in order and guards the end."""

STORE_LORE = {}


class Errand:

    def __init__(self, problem_data):
        self.began = time.monotonic()
        self.task_binding, self.task_view = self._inputs(problem_data)
        self.log_rows = []
        self.counter = None
        self.store_key = "|".join(sorted({name for name in map(self._tool_title, self.task_view.get("tools") or [])
                                          if name}))
        self.lore = self._recall()
        self.questions_only = False
        self.early_cart_wanted = self.early_cart_tried = False
        self.group_marks, self.swept_at = {}, {}
        self.narrowed_ix, self.model_asked_ix, self.heard_ix = set(), set(), set()
        for attr, part in (("oracle", Oracle), ("counter", Counter), ("order", Order), ("scout", Scout),
                           ("assessor", Assessor), ("parley", Parley), ("trail", Trail), ("till", Till),
                           ("picker", Picker)):
            setattr(self, attr, part(self))

    @staticmethod
    def _inputs(problem_data):
        env = problem_data["environment"]
        return env["binding"], env["policy_view"]

    @staticmethod
    def _tool_title(entry):
        if not isinstance(entry, dict):
            return None
        inner = entry.get("function")
        return (entry if not isinstance(inner, dict) else inner).get("name")

    def _recall(self):
        found = STORE_LORE.get(self.store_key)
        return found if found else {}

    def seconds_left(self):
        spent = time.monotonic() - self.began
        return Dials.Timing.task - spent

    def log(self, kind, **fields):
        stamp = 0 if self.counter is None else self.counter.used
        self.log_rows.append(dict({"kind": kind, "step": stamp, "t": self._elapsed()}, **fields))

    def _elapsed(self):
        return round(time.monotonic() - self.began, 1)

    def keep_lore(self):
        STORE_LORE[self.store_key] = dict(key_maps=self.counter.known_maps(),
                                          batch_size=self.assessor.compare_size(),
                                          thin_batch=self.assessor.side_rows_thin(),
                                          every_word=self.scout.match_rule())

    def rearm_hunt(self, w):
        self.heard_ix.discard(w.slot)
        self.swept_at.pop(w.slot, None)

    def ending_room(self):
        pinned = self.picker.reserve_pin()
        if pinned is not None:
            return pinned
        count = len(self.order.wants)
        room = count + Dials.Allowance.reserve_base
        no_side = not self.counter.job("open_several")
        if no_side and self.counter.job("spec_sheet"):
            room += min(count, Dials.Allowance.reserve_looks_cap) if self.assessor.table_first() \
                else Dials.Allowance.reserve_looks_plain
        later = self.parley.late_duty_count()
        if self.parley.needs_yes():
            later += Dials.Allowance.reserve_confirm
        room += min(Dials.Allowance.reserve_procedure_cap, later)
        if self.order.lowest_price and not self.picker.spares_done():
            room += Dials.Allowance.reserve_cheapest
        return room

    def carry_out(self):
        if self._guarded(self._read_then_work):
            return self.log_rows
        self._guarded(self._close_if_open)
        tally = {"cost": round(self.oracle.spend, Dials.Trace.cost_digits), "llm_calls": self.oracle.tries,
                 "steps": self.counter.used}
        self.log("done", **tally)
        return self.log_rows

    def _guarded(self, job):
        try:
            return job()
        except Exception as exc:
            self._note_crash(exc)
        return False

    def _note_crash(self, exc):
        shown = repr(exc)
        self.log("exception", error=shown[:Dials.Trace.full])

    def _read_then_work(self):
        if self.order.read_request():
            self.run_phases()
            return False
        self.till.close_out()
        return True

    def _close_if_open(self):
        if not self.counter.closed:
            self.till.close_out()

    def run_phases(self):
        wants = self.order.wants
        quota = self.parley.first_question_count()
        spare_one = Dials.Shopper.reask_spare if quota else 0
        first_floor = quota + spare_one + self.ending_room() + Dials.Lookup.first_pass_per_item * len(wants)
        self._follow_links(lambda: first_floor)
        spare_steps = self.counter.steps_free() - quota - spare_one - self.ending_room()
        look_share = max(Dials.Allowance.min_share, spare_steps // max(len(wants), 1)) if spare_steps > 0 else 0
        quota = self._start_gate(quota)
        self._first_looks(quota, look_share)
        if quota > 0 and self.parley.can_question():
            self._ask_round(quota)
        self._follow_links(lambda: self.ending_room() + Dials.Lookup.second_pass_spare)
        if not self.questions_only:
            self._close_in()

    def _close_in(self):
        self.early_cart_wanted = True
        self.cart_early()
        self._final_hunts()
        self.picker.pick_all()

    def _follow_links(self, floor_of):
        pending = ((w, need) for w in self.order.wants for need in list(w.needs))
        for w, need in pending:
            if need.waiting():
                self.trail.follow(w, need, floor_of())

    def _start_gate(self, quota):
        if self.parley.has_duties("start"):
            self.parley.run_due("start")
            self.parley.absorb_lines()
            room = self.parley.room_after_held()
            if room is not None:
                quota = max(0, min(quota, room))
        return quota

    def _first_looks(self, quota, look_share):
        wants = self.order.wants
        for at, w in enumerate(wants):
            portion = look_share
            if at == len(wants) - 1:
                rest = self.counter.steps_free() - quota - Dials.Shopper.reask_spare - self.ending_room()
                portion = max(look_share, rest)
            if portion <= 0:
                break
            steps_then = self.counter.steps_free()
            self.hunt(w, max(Dials.Allowance.min_share, portion // Dials.Allowance.first_browse_div))
            used = steps_then - self.counter.steps_free()
            self.widen_menu(w, allowance=max(0, portion - used))

    def _final_hunts(self):
        self.assessor.assess_all()
        lined = sorted(self.order.wants, key=self._final_rank)
        for left, w in zip(range(len(lined), 0, -1), lined):
            portion = (self.counter.steps_free() - self.ending_room()) // left
            self.hunt(w, max(Dials.Allowance.min_share, portion), final=True)

    @staticmethod
    def _final_rank(w):
        has_fit = bool(w.fitting())
        return has_fit, -len(w.found)

    def _ask_round(self, quota):
        per_msg = Dials.Shopper.pair_width if len(self.order.wants) > 1 else 1
        queue = self.parley.question_plan(quota * per_msg)
        singles = []
        while queue or singles:
            if not self.parley.can_question() or self.counter.steps_free() <= self.ending_room():
                break
            second = None
            if singles:
                w, prop = singles.pop(0)
            else:
                w, prop = queue.pop(0)
                at = self._partner_at(w, prop, queue, per_msg)
                if at is not None and queue[at][1]:
                    second = queue.pop(at)[1]
            kept = self._uncovered(w, prop, second)
            if kept is None:
                continue
            prop, second = kept
            if self._put_question(w, prop, second, singles) and queue:
                room = self.parley.room_after_held()
                left = len(queue) if room is None else min(len(queue), room)
                queue = self.parley.question_plan(left * per_msg)
        self._ask_leftovers(per_msg)

    def _ask_leftovers(self, per_msg):
        ask_for = per_msg * len(self.order.wants) if per_msg > 1 else 1
        spill = []
        while self.parley.can_question() and \
                self.counter.steps_free() > self.ending_room() + Dials.Shopper.extra_ask_spare and \
                self.seconds_left() > Dials.Timing.more_questions_over:
            options = self.parley.question_plan(ask_for)
            if not options:
                break
            w, prop = options[0]
            rest = options[1:]
            at = self._partner_at(w, prop, rest, per_msg)
            self._put_question(w, prop, rest[at][1] if at is not None else None, spill)

    def _uncovered(self, w, prop, second):
        if second and self.parley.covered(w, second):
            second = None
        if not self.parley.covered(w, prop):
            return prop, second
        return (second, None) if second else None

    def _partner_at(self, w, prop, rest, per_msg):
        if per_msg <= 1 or not self.parley.raised.get(w.slot):
            return None
        fits_pair = (lambda who, other: who is w and likeness(other, prop) < Dials.Like.same
                     and not same_name(other, prop))
        return next((at for at, entry in enumerate(rest) if fits_pair(*entry)), None)

    def _put_question(self, w, prop, second, singles):
        count_then = len(w.needs)
        answer = self.parley.pose(w, prop, second)
        if self.parley.reply_kind(answer) == "split":
            if second:
                singles.insert(0, (w, second))
                singles.insert(0, (w, prop))
            elif self.parley.can_question() and self.counter.steps_free() > self.ending_room():
                answer = self.parley.pose_narrow(w, prop)
        self.parley.take_reply(w, prop, answer)
        self.parley.absorb_lines()
        return len(w.needs) != count_then

    def cart_early(self):
        if self.early_cart_tried or self.till.has_lines() or not self.till.can_cart():
            return
        if self.counter.steps_free() - self.ending_room() < Dials.Ending.early_cart_steps or \
                self.seconds_left() < Dials.Timing.early_cart_until:
            self.early_cart_tried = True
            return
        self.assessor.assess_all()
        chosen = next(self._early_picks(), None)
        if chosen is None:
            return
        self.early_cart_tried = True
        w, pick = chosen
        if self.till.add_line(w, pick):
            note = dict(item=w.phrase, id=pick.oid)
            self.log("provisional_cart", **note)

    def _early_picks(self):
        for w in self.order.wants:
            top = next(iter(self.picker.best_ready(w)), None)
            if top is not None and (top.cost is None or not over_limit(top.cost * w.count, self.order.budget)):
                yield w, top

    def widen_menu(self, w, allowance=None):
        if self.counter.steps_free() <= self.ending_room() + Dials.Browse.menu_spare or \
                not self.counter.job("tell_shopper"):
            return
        free_now = self.counter.steps_free()
        stop_line = free_now - (allowance if allowance is not None else free_now)
        keep_line = max(stop_line, self.ending_room() + Dials.Browse.menu_spare)
        aim = Dials.Browse.menu_target_batch if self.assessor.batch_reads() else Dials.Browse.menu_target
        while self.counter.steps_free() > keep_line and self.seconds_left() > Dials.Timing.menu_reads_over and \
                not self.oracle.over_spend():
            missing = aim - sum(1 for o in w.found.values() if self.scout.right_kind(w, o) and o.read_full)
            if missing <= 0:
                return
            to_open = self._menu_rows(w, keep_line)
            if not to_open:
                return
            if self.assessor.batch_reads():
                chunk = self.assessor.compare_size()
            else:
                chunk = Dials.Browse.menu_chunk_thin if self.assessor.side_rows_thin() else Dials.Browse.menu_chunk
            batch = to_open[:min(missing, chunk)]
            if not self.assessor.read_listings(w, floor=keep_line, ids=[o.oid for o in batch]):
                return
            self.assessor.reassess(w)

    def _menu_rows(self, w, keep_line):
        rows = self._unread_rows(w)
        if len(rows) >= Dials.Browse.menu_rows_min or w.pages_run >= len(w.queries) + Dials.Browse.menu_term_extra:
            return rows
        self.scout.seek(w, text=w.kind, pages=Dials.Browse.menu_pages, floor=keep_line)
        return self._unread_rows(w)

    def _unread_rows(self, w):
        keep = (lambda o: o.in_stock is not False and not o.read_full and self.scout.right_kind(w, o))
        return sorted(filter(keep, w.found.values()), key=self._menu_rank)

    @staticmethod
    def _menu_rank(o):
        mark = o.sift_mark
        return mark == "no", mark != "yes", price_rank(o.cost)

    def hunt(self, w, steps, final=False):
        terms = w.queries[:]
        tour = {"stop": self.counter.steps_free() - steps, "terms": terms,
                "next": min(w.pages_run, len(terms)),
                "enough": self._enough_fits()}
        while self.counter.steps_free() > self._line(tour) and self.seconds_left() > Dials.Timing.browse_over and \
                not self.oracle.over_spend():
            if not self._hunt_round(w, tour, final):
                break

    def _enough_fits(self):
        if self.order.lowest_price:
            return Dials.Browse.enough_fits_cheapest
        if self.order.measure:
            return Dials.Browse.enough_fits_best
        return Dials.Browse.enough_fits

    def _line(self, tour):
        return max(tour["stop"], self.ending_room())

    def _hunt_round(self, w, tour, final):
        self.parley.absorb_lines()
        self.scout.triage(w, self.scout.shortlist(w))
        by_read = {True: [], False: []}
        for o in self.scout.shortlist(w):
            by_read[bool(o.read_full)].append(o)
        unread = by_read[False]
        self.assessor.assess(w, by_read[True])
        fits = w.fitting()
        if final and fits and self.early_cart_wanted and not self.till.has_lines():
            self.cart_early()
        cheapest = self.order.lowest_price
        if fits and not cheapest and len(fits) >= tour["enough"]:
            return False
        value_sets = self._current_value_sets(w)
        if value_sets:
            hits = self._order_by_values(w, unread, value_sets)
            due = self._cheaper_than(unread, price_rank(fits[0].cost)) if fits and cheapest else unread
            if all(hits[o.oid] != len(value_sets) for o in due) and \
                    self.scout.try_value_query(w, self._line(tour)):
                return True
        covered = bool(value_sets) and self.scout.match_rule() and self.scout.values_complete(w)
        if fits and cheapest:
            unread, verdict = self._cheaper_follow_up(w, tour, fits, unread, covered)
            if verdict is not None:
                return verdict
        if unread:
            return self._open_some(w, tour, unread)
        return self._search_more(w, tour)

    def _current_value_sets(self, w):
        value_sets = self.scout.value_choices(w)
        mark = repr(value_sets)
        if self.group_marks.get(w.slot) != mark:
            self.group_marks[w.slot] = mark
            self.scout.forget_values(w)
        return value_sets

    def _order_by_values(self, w, unread, value_sets):
        hits = {o.oid: self._carried(w, o, value_sets) for o in unread}

        def rank(o):
            mark = o.sift_mark
            return mark == "no", -hits[o.oid], mark != "yes", price_rank(o.cost)
        unread.sort(key=rank)
        return hits

    @staticmethod
    def _cheaper_than(rows, ceiling):
        return list(filter(lambda o: o.cost is not None and o.cost < ceiling, rows))

    def _carried(self, w, offer, value_sets):
        whole = len(value_sets)
        if self.scout.in_value_rows(w, offer.oid):
            return whole
        count = self.scout.values_carried(offer, value_sets)
        if self.scout.match_rule() and count < whole:
            shown_for = [wordset_full(q) for q in self.scout.queries_showing(w, offer.oid)]
            by_query = 0
            for group in value_sets:
                alts = [wordset_full(alt) for alt in group]
                if any(words.issuperset(one) for words in shown_for for one in alts):
                    by_query += 1
            count = max(count, by_query)
        return count

    def _cheaper_follow_up(self, w, tour, fits, unread, covered):
        ceiling = price_rank(fits[0].cost)
        cheaper = self._cheaper_than(unread, ceiling)
        if covered:
            cheaper = [o for o in cheaper if o.sift_mark == "yes" or self.scout.in_value_rows(w, o.oid)]
        if cheaper:
            return cheaper, None
        if self.search_heard(w, self._line(tour)):
            return cheaper, True
        if not covered and self._sweep_below(w, tour, ceiling):
            return cheaper, True
        if not covered and self._narrow_capped(w, tour):
            return cheaper, True
        if self._page_on_lead(w, tour):
            return cheaper, True
        if w.pages_run >= min(len(tour["terms"]), Dials.Browse.give_up_pages):
            return cheaper, False
        return cheaper, None

    def _sweep_below(self, w, tour, ceiling):
        if ceiling >= Dials.Money.known_below or not self.counter.job_field("field_search", "max_price_arg"):
            return False
        if self.swept_at.get(w.slot) == ceiling or \
                self.counter.steps_free() <= self._line(tour) + Dials.Browse.step_slack:
            return False
        self.swept_at[w.slot] = ceiling
        fields = {"category_arg": w.kind,
                  "max_price_arg": max(Dials.Money.sweep_floor, ceiling - Dials.Money.sweep_step)}
        self.scout.seek(w, fields=fields, pages=Dials.Browse.sweep_pages, floor=self._line(tour))
        return True

    def _narrow_capped(self, w, tour):
        if not self.scout.page_limit_hit(w) or w.slot in self.narrowed_ix or \
                self.counter.steps_free() <= self._line(tour) + Dials.Browse.step_slack:
            return False
        self.narrowed_ix.add(w.slot)
        values = [or_parts(n.want)[0] for n in w.live_needs() if n.rel in Dials.Ops.exact and n.want]
        if not values:
            return False
        joined = " ".join(values)
        for text in (f"{joined} {w.kind}", joined):
            if self.counter.steps_free() > self._line(tour) + Dials.Browse.step_spare:
                self.scout.seek(w, text=text, pages=Dials.Browse.narrow_pages, floor=self._line(tour))
            else:
                break
        return True

    def _page_on_lead(self, w, tour):
        lead = next((q for q in tour["terms"] if self.scout.cursor_state(w, text=q) == "open"), None)
        if not lead or self.counter.steps_free() <= self._line(tour) + Dials.Browse.step_slack:
            return False
        used_then = self.counter.used
        self.scout.seek(w, text=lead, pages=Dials.Browse.main_pages, floor=self._line(tour) + Dials.Browse.step_spare)
        return self.counter.used > used_then

    def _open_some(self, w, tour, unread):
        if self.assessor.batch_reads():
            size = self.assessor.compare_size()
        elif self.assessor.side_rows_thin() or len(self.order.wants) >= Dials.Browse.wide_items:
            size = Dials.Browse.open_batch_wide
        else:
            size = Dials.Browse.open_batch
        batch = unread[:size]
        return bool(self.assessor.read_listings(w, floor=self._line(tour), ids=[o.oid for o in batch]))

    def _search_more(self, w, tour):
        if self.search_heard(w, self._line(tour)):
            return True
        terms = tour["terms"]
        if tour["next"] < len(terms):
            text = terms[tour["next"]]
            tour["next"] += 1
            pages = Dials.Browse.own_term_pages_one if len(self.order.wants) == 1 else Dials.Browse.own_term_pages_many
            self.scout.seek(w, text=text, pages=pages, floor=self._line(tour))
            return True
        if self.counter.job("field_search"):
            fields = self._kind_fields(w)
            if self.scout.cursor_state(w, fields=fields) != "ended":
                self.scout.seek(w, fields=fields, pages=Dials.Browse.field_pages, floor=self._line(tour))
                return True
        for text in terms[:Dials.Browse.resume_terms]:
            if self.scout.cursor_state(w, text=text) == "ended":
                continue
            if self.scout.seek(w, text=text, pages=Dials.Browse.resume_pages, floor=self._line(tour)) > 0:
                return True
        if w.slot not in self.model_asked_ix and \
                self.counter.steps_free() > self._line(tour) + Dials.Browse.step_spare:
            self.model_asked_ix.add(w.slot)
            added = self.fresh_queries(w)
            if added:
                for text in added:
                    w.append_query(text)
                terms.extend(added)
                return True
        return False

    def _kind_fields(self, w):
        caps = [] if self.order.budget is None else [self.order.budget / max(w.count, 1)]
        if w.unit_cap is not None:
            caps.append(w.unit_cap)
        fields = {"category_arg": w.kind}
        if caps:
            fields["max_price_arg"] = min(caps)
        return fields

    def search_heard(self, w, floor):
        heard, given = [], []
        for need in w.live_needs():
            if need.want and need.rel in Dials.Ops.exact:
                (heard if need.source == FROM_SHOPPER else given).append(or_parts(need.want)[0])
        if not heard or w.slot in self.heard_ix or self.scout.match_rule() is True:
            return False
        self.heard_ix.add(w.slot)
        lead = heard[0]
        tries = [f"{w.kind} {lead}", lead]
        if given:
            tries[:0] = [" ".join((w.kind, given[0], lead))]
        sent = {frozenset(wordset_full(q)) for q in self.scout.ran_queries(w)}
        ran = 0
        for text in [t for t in tries if frozenset(wordset_full(t)) not in sent]:
            if self.counter.steps_free() <= floor:
                break
            self.scout.seek(w, text=text, pages=Dials.Browse.need_pages, floor=floor)
            ran += 1
        return ran > 0

    @staticmethod
    def _shown_rank(o):
        return not o.read_full, o.oid

    def fresh_queries(self, w):
        fit_ids = {o.oid for o in w.fitting(with_doubts=True)}
        ordered = sorted(w.found.values(), key=self._shown_rank)
        seen = [(o, (o.oid in fit_ids) if o.read_full else None) for o in ordered[:Dials.Browse.suggest_rows]]
        used = sorted(set(self.scout.ran_queries(w)))
        words_arg = self.counter.job_field("keyword_search", "words_arg")
        max_len = self.counter.arg_form("keyword_search", words_arg).get("maxLength")
        brief = brief_search_words(w.phrase, w.kind, w.need_lines(), used, seen, max_len)
        reply = self.oracle.ask_json(Dials.Model.light_spare, brief, Dials.Tokens.search_words)
        picked = self._new_queries(reply, used + list(w.queries))[:Dials.Browse.suggest_max]
        self.log("suggested", **{"item": w.phrase, "queries": picked})
        return list(picked)

    @staticmethod
    def _new_queries(reply, known_queries):
        if not isinstance(reply, dict):
            return []
        known = set(map(simplify, known_queries))
        kept = []
        for text in map(lambda raw: str(raw).strip(), reply.get("new_queries") or []):
            form = simplify(text)
            if not text or len(text) > Dials.Query.query_cut or form in known:
                continue
            kept.append(text)
            known.add(form)
        return kept

def agent_main(problem_data: dict) -> list:
    return Errand(problem_data).carry_out()


__all__ = ["agent_main"]
