import json
import random

import pytest

from config import N_GRPO, N_SFT, STOP_TOKEN
from data import (align_english, arg_schema, bfcl_items, build_item, candidates, convert, eval_items, fn_name,
                  functions_json, held_out_items, judged, load_massive, slots, split_train, stratified,
                  why_dropped)
from generate import pending, read_preds
from parse import parse
from prompts import FORMAT, bfcl_zero_shot, custom_one_shot, grpo_prompt, target, training_messages
from rewards import reward, reward_parts
from score import norm, score_item
from train_sft import encode

TOOLS = [
    {"name": "alarm.set", "description": "Set an alarm", "parameters": {
        "type": "object",
        "properties": {"time": {"type": "string"}, "count": {"type": "integer"}},
        "required": ["time"]}},
    {"name": "email.query", "description": "Read emails",
     "parameters": {"type": "object", "properties": {}, "required": []}},
]


def item(gold, tools=TOOLS, query="wake me at five"):
    return {"id": "1", "lang": "en-US", "query": query, "tools": tools, "gold": gold}


ALARM = [{"name": "alarm.set", "arguments": {"time": ["five am"]}}]


# ── parse ─────────────────────────────────────────────────────────────────────
def test_strict_json_list():
    p = parse('[{"tool_name": "alarm.set", "parameters": {"time": "five am"}}]')
    assert p == {"calls": [{"name": "alarm.set", "arguments": {"time": "five am"}}],
                 "parsed": True, "strict": True}


def test_empty_list_is_a_valid_no_call():
    assert parse("[]") == {"calls": [], "parsed": True, "strict": True}


def test_plain_text_is_not_parsed():
    assert parse("Sorry, I can't do that.") == {"calls": [], "parsed": False, "strict": False}


def test_malformed_json_is_not_parsed():
    assert parse('[{"tool_name": "alarm.set", "parameters": {"time": ')["parsed"] is False


def test_extra_keys_in_a_call_are_not_strict():
    p = parse('[{"tool_name": "a.b", "parameters": {}, "note": "x"}]')
    assert p["strict"] is False


def test_prose_around_json_parses_but_is_not_strict():
    p = parse('Sure! [{"tool_name": "a.b", "parameters": {"x": 1}}]')
    assert p["calls"] == [{"name": "a.b", "arguments": {"x": 1}}]
    assert (p["parsed"], p["strict"]) == (True, False)


def test_pythonic_bfcl_list_with_dotted_name():
    p = parse('[alarm.set(time="nine am", count=2), email.query()]')
    assert p["calls"] == [{"name": "alarm.set", "arguments": {"time": "nine am", "count": 2}},
                          {"name": "email.query", "arguments": {}}]
    assert p["strict"] is True


def test_pythonic_action_prefix():
    assert parse('Action: alarm.set(time="nine am")')["calls"][0]["name"] == "alarm.set"


def test_pythonic_non_literal_argument_is_rejected():
    assert parse("[alarm.set(time=now())]")["parsed"] is False


def test_pythonic_positional_argument_is_rejected():
    assert parse('[alarm.set("nine am")]')["parsed"] is False


# ── score ─────────────────────────────────────────────────────────────────────
def score(gold, text, **kw):
    return score_item(item(gold, **kw), text)


def call(**args):
    return json.dumps([{"tool_name": "alarm.set", "parameters": args}], ensure_ascii=False)


def test_exact_match_passes_every_level():
    s = score(ALARM, call(time="five am"))
    assert (s["fsa"], s["ast_verbatim"], s["ast_strict"], s["ast_lenient"]) == (1, 1, 1, 1)
    assert s["error"] == "correct"


def test_case_difference_is_strict_but_not_verbatim():
    s = score(ALARM, call(time="Five AM"))
    assert (s["ast_verbatim"], s["ast_strict"]) == (0, 1)


def test_lenient_catches_what_strict_misses():
    s = score([{"name": "alarm.set", "arguments": {"time": ["5 a.m."]}}], call(time="5:00 AM"))
    assert (s["ast_strict"], s["ast_lenient"]) == (0, 1)


def test_wrong_value_fails_all_levels_but_keeps_fsa():
    s = score(ALARM, call(time="six am"))
    assert (s["fsa"], s["ast_strict"], s["ast_lenient"], s["error"]) == (1, 0, 0, "invalid_value")


def test_accepted_value_lists():
    gold = [{"name": "alarm.set", "arguments": {"time": ["five am", "5 am"]}}]
    assert score(gold, call(time="5 am"))["ast_strict"] == 1


def test_optional_arg_may_be_omitted():
    gold = [{"name": "alarm.set", "arguments": {"time": ["five am"], "count": [2, ""]}}]
    assert score(gold, call(time="five am"))["ast_strict"] == 1


def test_missing_required_arg():
    gold = [{"name": "alarm.set", "arguments": {"time": ["five am"], "count": [2]}}]
    s = score(gold, call(time="five am"))
    assert (s["ast_strict"], s["error"]) == (0, "missing_param")


def test_hallucinated_key_fails():
    s = score(ALARM, call(time="five am", place="home"))
    assert (s["ast_strict"], s["error"]) == (0, "unexpected_param")


def test_float_for_integer_schema_fails_strict():
    gold = [{"name": "alarm.set", "arguments": {"time": ["five am"], "count": [2]}}]
    assert score(gold, call(time="five am", count=2.0))["ast_strict"] == 0


def test_int_for_number_schema_passes():
    tools = [{"name": "alarm.set", "parameters": {
        "type": "object", "properties": {"v": {"type": "number"}}, "required": ["v"]}}]
    gold = [{"name": "alarm.set", "arguments": {"v": [2.0]}}]
    assert score(gold, call(v=2), tools=tools)["ast_strict"] == 1


def test_wrong_function():
    s = score(ALARM, '[{"tool_name": "email.query", "parameters": {}}]')
    assert (s["fsa"], s["ast_strict"], s["error"]) == (0, 0, "wrong_function")


def test_wrong_number_of_calls():
    text = json.dumps([{"tool_name": "alarm.set", "parameters": {"time": "five am"}}] * 2)
    s = score(ALARM, text)
    assert (s["fsa"], s["ast_strict"], s["error"]) == (0, 0, "wrong_count")


def test_unparsed_output_on_a_call_item_is_a_syntax_error():
    s = score(ALARM, "I set your alarm.")
    assert (s["format_valid"], s["error"]) == (0, "invalid_syntax")


def test_no_call_correct_for_empty_list_and_plain_text():
    assert score([], "[]")["no_call_correct"] == 1
    assert score([], "No tool fits.")["no_call_correct"] == 1


def test_no_call_item_with_a_call_is_over_calling():
    s = score([], call(time="five am"))
    assert (s["no_call_correct"], s["error"]) == (0, "over_call")


def test_call_metrics_are_absent_on_no_call_items():
    s = score([], "[]")
    assert s["fsa"] is None and s["ast_strict"] is None and s["no_arg_bucket"] is None


def test_argument_level_counts():
    gold = [{"name": "alarm.set", "arguments": {"time": ["five am"], "count": [2]}}]
    s = score(gold, call(time="six am", place="home"))
    assert (s["key_tp"], s["key_fp"], s["key_fn"]) == (1, 1, 1)
    assert (s["val_tp"], s["val_fp"], s["val_fn"]) == (0, 2, 2)


def test_difficulty_bucket():
    assert score(ALARM, "[]")["no_arg_bucket"] == "1"
    two = [{"name": "alarm.set", "arguments": {"time": ["a"], "count": [1]}}]
    assert score(two, "[]")["no_arg_bucket"] == "2+"
    zero = [{"name": "email.query", "arguments": {}}]
    assert score(zero, "[]")["no_arg_bucket"] == "0"


def test_language_mismatch_flags_non_english_value():
    gold = [{"name": "alarm.set", "arguments": {"time": ["today"]}}]
    assert score(gold, call(time="आज"))["lang_mismatch"] == 1
    assert score(gold, call(time="tomorrow"))["lang_mismatch"] == 0


def test_language_mismatch_ignores_a_correct_value():
    gold = [{"name": "alarm.set", "arguments": {"time": ["आज"]}}]
    assert score(gold, call(time="आज"))["lang_mismatch"] == 0


def test_language_mismatch_in_latin_script_igbo_uses_copying_from_the_query():
    gold = [{"name": "alarm.set", "arguments": {"time": ["today"]}}]
    it = {**item(gold, query="tinye mkpu n'ututu"), "lang": "ig"}
    assert score_item(it, call(time="ututu"))["lang_mismatch"] == 1
    assert score_item(it, call(time="tomorrow"))["lang_mismatch"] == 0


def test_norm_converts_digits_from_other_scripts():
    assert norm("٥ صباحا") == norm("5 صباحا")
    assert norm("५") == "5"


def test_norm_keeps_hindi_combining_marks():
    assert norm("मुंबई") != norm("मबई")


def test_norm_maps_english_number_words():
    assert norm("Five") == "5"


# ── rewards ───────────────────────────────────────────────────────────────────
def parts(gold, text):
    return reward_parts(text, item(gold))


def test_perfect_call_earns_full_reward():
    assert reward(call(time="five am"), item(ALARM)) == pytest.approx(1.0)


def test_normalised_only_value_earns_half_value_credit():
    gold = [{"name": "alarm.set", "arguments": {"time": ["5 a.m."]}}]
    assert parts(gold, call(time="5:00 AM"))["values"] == 0.5


def test_value_reward_is_gated_on_function():
    p = parts(ALARM, '[{"tool_name": "email.query", "parameters": {"time": "five am"}}]')
    assert (p["function"], p["keys"], p["values"]) == (0, 0, 0)


def test_value_reward_is_gated_on_exact_keys():
    p = parts(ALARM, call(time="five am", place="home"))
    assert p["keys"] < 1 and p["values"] == 0


def test_empty_list_on_a_call_item_earns_nothing():
    assert reward("[]", item(ALARM)) == 0


def test_prose_wrapped_call_earns_nothing():
    assert reward("Sure! " + call(time="five am"), item(ALARM)) == 0


def test_correct_no_call_earns_full_reward():
    assert reward("[]", item([])) == pytest.approx(1.0)


def test_plain_text_no_call_loses_only_the_format_part():
    assert reward("No tool fits.", item([])) == pytest.approx(0.9)


def test_over_calling_earns_at_most_the_format_part():
    assert reward(call(time="five am"), item([])) == pytest.approx(0.1)


def test_no_arg_call_earns_full_reward():
    gold = [{"name": "email.query", "arguments": {}}]
    assert reward('[{"tool_name": "email.query", "parameters": {}}]', item(gold)) == pytest.approx(1.0)


# ── prompts ───────────────────────────────────────────────────────────────────
def test_grpo_prompt_composes_into_the_training_prompt():
    # agentic rollout: system = f"{system}\n\n{tools_fallback_prompt.format(tools=...)}"
    it = {**item(ALARM), "policy": "verbatim"}
    system, user = grpo_prompt(it)
    composed = f"{system['content']}\n\n{FORMAT.format(tools='ignored')}"
    assert [{"role": "system", "content": composed}, user] == training_messages(it)


def test_training_prompt_offers_only_the_item_tools():
    it = {**item(ALARM, tools=TOOLS[:1]), "policy": "verbatim"}
    system = training_messages(it)[0]["content"]
    assert "alarm.set" in system and "email.query" not in system


def test_policy_line_changes_only_the_policy():
    base = {**item(ALARM), "policy": None}
    assert "verbatim" not in training_messages(base)[0]["content"].lower()
    assert "exactly as written" in training_messages({**base, "policy": "verbatim"})[0]["content"]
    assert "in English" in training_messages({**base, "policy": "english"})[0]["content"]


def test_bfcl_prompt_keeps_the_paper_wording():
    text = bfcl_zero_shot(item(ALARM))[0]["content"]
    assert text.startswith("You are an expert in composing functions.")
    assert "User: wake me at five Bot:" in text


def test_one_shot_prompt_keeps_the_paper_example():
    text = custom_one_shot(item(ALARM))[0]["content"]
    assert 'Bot: Action: alarm.set(time="nine am", date="this week")' in text
    assert text.endswith("User: wake me at five Bot:")


def test_compact_catalogue_lists_one_tool_per_line_without_types():
    it = {**item(ALARM), "policy": "verbatim", "compact": True}
    system = training_messages(it)[0]["content"]
    assert '{"name": "alarm.set", "description": "Set an alarm", "args": ["time", "count"]}' in system
    assert '"type"' not in system


# ── data ──────────────────────────────────────────────────────────────────────
def row(**kw):
    return {"id": "7", "locale": "hi-IN", "scenario": "weather", "intent": "weather_query",
            "utt": "x", "annot_utt": "x", **kw}


def test_fn_name_replaces_only_the_first_underscore():
    assert fn_name("alarm_set") == "alarm.set"
    assert fn_name("audio_volume_up") == "audio.volume_up"


def test_slots_parse_unicode_and_inner_colon():
    assert slots("मौसम [place_name : मुंबई] [date : आज]") == [("place_name", "मुंबई"), ("date", "आज")]
    assert slots("at [time : 5:30 am]") == [("time", "5:30 am")]


def test_convert_maps_slots_to_arguments():
    c = convert(row(annot_utt="[place_name : मुंबई] मौसम"))
    assert (c["name"], c["args"]) == ("weather.query", {"place_name": "मुंबई"})


def test_convert_drops_catch_all_intents():
    assert convert(row(intent="general_quirky")) is None


def test_convert_drops_duplicate_slots():
    assert convert(row(annot_utt="[food_type : a] and [food_type : b]")) is None


def test_convert_uses_the_whole_utterance_for_qa():
    c = convert(row(intent="qa_factoid", utt="how large is alaska",
                    annot_utt="how large is [place_name : alaska]"))
    assert (c["name"], c["args"]) == ("qa.factoid", {"query": "how large is alaska"})


def test_align_english_needs_the_same_keys():
    assert align_english({"date": "आज"}, {"date": "today"}) == {"date": "today"}
    assert align_english({"date": "आज"}, {"time": "today"}) is None


NAMES = {"a.1": "a", "a.2": "a", "a.3": "a", "a.4": "a", "b.1": "b", "b.2": "b", "c.1": "c"}


def test_eval_candidates_are_correct_plus_three_same_scenario_plus_one_random():
    picks = candidates("a.1", NAMES, 5, random.Random("x"))
    assert len(picks) == 5 and "a.1" in picks
    assert sum(NAMES[p] == "a" for p in picks) == 4


def test_candidates_fill_from_other_scenarios_when_the_scenario_is_small():
    picks = candidates("c.1", NAMES, 5, random.Random("x"))
    assert len(set(picks)) == 5 and "c.1" in picks


def test_build_item_uses_identical_candidates_across_languages():
    fns = {n: {"name": n, "description": n, "parameters": {}} for n in NAMES}
    en = build_item({"id": "9", "lang": "en-US", "query": "q", "name": "a.1", "args": {}},
                    fns, NAMES, k=5, no_call=False, policy="verbatim", seed=1)
    hi = build_item({"id": "9", "lang": "hi-IN", "query": "क", "name": "a.1", "args": {}},
                    fns, NAMES, k=5, no_call=False, policy="verbatim", seed=1)
    assert [t["name"] for t in en["tools"]] == [t["name"] for t in hi["tools"]]


def test_no_call_item_removes_the_correct_tool_and_keeps_the_list_size():
    fns = {n: {"name": n, "description": n, "parameters": {}} for n in NAMES}
    it = build_item({"id": "9", "lang": "en-US", "query": "q", "name": "a.1", "args": {"x": "y"}},
                    fns, NAMES, k=5, no_call=True, policy="verbatim", seed=1)
    assert it["gold"] == [] and len(it["tools"]) == 5
    assert "a.1" not in [t["name"] for t in it["tools"]]


def test_build_item_wraps_gold_values_as_accepted_lists():
    fns = {n: {"name": n, "description": n, "parameters": {}} for n in NAMES}
    it = build_item({"id": "9", "lang": "en-US", "query": "q", "name": "a.1", "args": {"x": "y"}},
                    fns, NAMES, k=2, no_call=False, policy="verbatim", seed=1)
    assert it["gold"] == [{"name": "a.1", "arguments": {"x": ["y"]}}]


# ── data filters ──────────────────────────────────────────────────────────────
def vote(intent=1, slots_score=1, grammar=4, spelling=2, lang="target"):
    return {"intent_score": intent, "slots_score": slots_score, "grammar_score": grammar,
            "spelling_score": spelling, "language_identification": lang}


def judged_row(votes, locale="fr-FR"):
    return {"locale": locale, "judgments": votes}


def test_judged_passes_no_slot_rows_and_reasonable_intents():
    assert judged(judged_row([vote(slots_score=2)] * 3))
    assert judged(judged_row([vote(intent=2)] * 3))


def test_judged_needs_a_majority():
    assert not judged(judged_row([vote(intent=0), vote(intent=0), vote()]))
    assert judged(judged_row([vote(slots_score=0), vote(), vote()]))


def test_judged_uses_median_scores():
    assert not judged(judged_row([vote(grammar=2), vote(grammar=2), vote(grammar=4)]))
    assert not judged(judged_row([vote(spelling=0), vote(spelling=0), vote()]))


def test_judged_keeps_mixed_language_but_needs_the_target():
    assert judged(judged_row([vote(lang="target & english")] * 3))
    assert not judged(judged_row([vote(lang="english")] * 3))


def test_judged_without_votes():
    assert judged({"locale": "en-US"})
    assert not judged(judged_row([]))


def mrow(intent, annot, partition="train"):
    return {"id": "1", "locale": "en-US", "partition": partition, "scenario": "s",
            "intent": intent, "utt": annot, "annot_utt": annot}


def test_arg_schema_cut_ignores_dev_and_test_rows():
    rows = [mrow("alarm_set", "[time : five]")] * 99 + [mrow("alarm_set", "[date : monday]")]
    rows += [mrow("alarm_set", "[relation : mom]", "test")] * 500
    assert arg_schema(rows, cut=0.02) == {"alarm_set": ["time"]}


def test_arg_schema_whole_query_intents_take_only_query():
    assert arg_schema([mrow("qa_factoid", "how large is [place_name : alaska]")]) == {"qa_factoid": ["query"]}


def test_why_dropped_names_each_filter():
    schema = {"alarm_set": ["time"]}
    assert why_dropped(mrow("general_greet", "hi"), schema) == "open_ended_intent"
    assert why_dropped(mrow("alarm_set", "[time : a] [time : b]"), schema) == "duplicate_argument"
    assert why_dropped(mrow("alarm_set", "[relation : mom]"), schema) == "rare_argument"
    assert why_dropped(mrow("alarm_set", "[time : five]"), schema) is None
    assert why_dropped({**mrow("alarm_set", "[time : five]"), "locale": "fr-FR", "judgments": []}, schema) == "judgment"


def test_functions_json_only_requires_query():
    from schemas import FN_DESC
    assert len(FN_DESC) == 55
    specs = functions_json({"alarm_set": ["time"], "qa_factoid": ["query"]})
    assert specs[0]["parameters"]["required"] == [] and specs[1]["parameters"]["required"] == ["query"]


def conv(row_id, name, args):
    return {"id": str(row_id), "lang": "en-US", "query": "q", "name": name, "args": args}


def test_stratified_is_even_across_functions_and_reproducible():
    convs = [conv(i, "a.big", {"x": "1"}) for i in range(100)] + [conv(100 + i, "b.small", {}) for i in range(3)]
    picked = stratified(convs, 6, seed=1)
    assert sum(c["name"] == "b.small" for c in picked) == 3
    assert [c["id"] for c in picked] == [c["id"] for c in stratified(convs, 6, seed=1)]
    assert len({c["id"] for c in picked}) == 6


def test_stratified_stops_when_rows_run_out():
    assert len(stratified([conv(1, "a.x", {})], 5, seed=1)) == 1


GOOD_VOTES = [{"intent_score": 1, "slots_score": 1, "grammar_score": 4, "spelling_score": 2,
               "language_identification": "target"}] * 3
SCENARIOS = {"weather.query": "weather", "weather.a": "weather", "weather.b": "weather", "weather.c": "weather",
             "other.a": "other", "other.b": "other", "other.c": "other", "other.d": "other"}
FUNCTIONS = {name: {"name": name, "description": "", "parameters": {}} for name in SCENARIOS}


def test_split_train_gives_grpo_the_rare_functions_that_sampling_uses_up_first():
    items = [conv(i, "a.rare", {}) for i in range(N_SFT)] + [conv(N_SFT + i, "b.common", {}) for i in range(N_GRPO)]
    sft, grpo = split_train(items, seed=1)
    assert (len(sft), len(grpo)) == (N_SFT, N_GRPO)
    assert "a.rare" in {c["name"] for c in grpo}
    assert not {c["id"] for c in sft} & {c["id"] for c in grpo}
    assert sft == split_train(items, seed=1)[0]


def test_eval_items_skips_a_missing_row_and_keeps_no_call_tied_to_the_id():
    convs = {i: conv(i, "weather.query", {}) for i in ("1", "3", "4")}
    items = eval_items(["1", "2", "3", "4"], 2, convs, FUNCTIONS, SCENARIOS, seed=1)
    assert [i["id"] for i in items] == ["1", "3", "4"]
    assert [bool(i["gold"]) for i in items] == [True, False, False]


def test_held_out_items_applies_the_same_filters_and_matches_the_english_tools():
    schema = {"weather_query": ["place_name"]}
    rows = [row(id="1", annot_utt="[place_name : x]", judgments=GOOD_VOTES),
            row(id="2", annot_utt="[color_type : x]", judgments=GOOD_VOTES),
            row(id="3", annot_utt="[place_name : y]", judgments=[])]
    test_ids = ["1", "2", "3"] + [str(i) for i in range(4, 100)]
    items = held_out_items(rows, schema, test_ids, FUNCTIONS, SCENARIOS, seed=1)
    assert [(i["id"], i["lang"]) for i in items] == [("1", "hi-IN")]
    english = build_item(conv("1", "weather.query", {}), FUNCTIONS, SCENARIOS, 5, False, "verbatim", 1)
    assert items[0]["tools"] == english["tools"]


def test_load_massive_reads_only_the_requested_locales(tmp_path):
    (tmp_path / "sw-KE.jsonl").write_text('{"id": "1"}\n', encoding="utf-8")
    assert load_massive(tmp_path, ("sw-KE",)) == {"sw-KE": [{"id": "1"}]}


# ── generate ──────────────────────────────────────────────────────────────────
def test_pending_skips_finished_ids_and_survives_a_torn_last_line(tmp_path):
    log = tmp_path / "preds.jsonl"
    log.write_text('{"id": "1", "reply": "[]"}\n\n{"id": "2", "rep', encoding="utf-8")
    items = [{"id": "1"}, {"id": "2"}, {"id": "3"}]
    assert [i["id"] for i in pending(items, log)] == ["2", "3"]


def test_pending_returns_everything_when_there_is_no_log(tmp_path):
    assert pending([{"id": "1"}], tmp_path / "none.jsonl") == [{"id": "1"}]


def test_read_preds_returns_each_finished_row_by_id(tmp_path):
    log = tmp_path / "preds.jsonl"
    log.write_text('{"id": "1", "reply": "a"}\n{"id": "2", "rep\n{"id": "3", "reply": "c"}\n', encoding="utf-8")
    assert {k: v["reply"] for k, v in read_preds(log).items()} == {"1": "a", "3": "c"}


# ── repeated function names, as in BFCL's parallel items ──────────────────────
SPOTIFY = [{"name": "spotify.play", "description": "", "parameters": {"type": "object", "properties": {
    "artist": {"type": "string"}, "duration": {"type": "integer"}}, "required": ["artist", "duration"]}}]
TWO_CALLS = [{"name": "spotify.play", "arguments": {"artist": ["Taylor Swift"], "duration": [20]}},
             {"name": "spotify.play", "arguments": {"artist": ["Maroon 5"], "duration": [15]}}]


def spotify_reply(*pairs):
    return json.dumps([{"tool_name": "spotify.play", "parameters": {"artist": a, "duration": d}} for a, d in pairs])


def test_same_named_calls_score_correct_in_either_order():
    gold_item = item(TWO_CALLS, tools=SPOTIFY)
    assert score_item(gold_item, spotify_reply(("Taylor Swift", 20), ("Maroon 5", 15)))["ast_strict"] == 1
    assert score_item(gold_item, spotify_reply(("Maroon 5", 15), ("Taylor Swift", 20)))["ast_strict"] == 1


def test_same_named_calls_with_crossed_values_stay_wrong():
    assert score_item(item(TWO_CALLS, tools=SPOTIFY), spotify_reply(("Taylor Swift", 15), ("Maroon 5", 20)))["ast_strict"] == 0


# ── BFCL items ────────────────────────────────────────────────────────────────
def test_bfcl_items_unwraps_tools_and_takes_the_single_turn():
    from types import SimpleNamespace
    tool = {"name": "add", "description": "", "parameters": {}}
    call = {"name": "add", "arguments": {"a": [1]}}
    samples = [SimpleNamespace(id="simple_0", input_text="q", tools=[{"type": "function", "function": tool}],
                               expected_tool_calls=[[call]]),
               SimpleNamespace(id="irrelevance_0", input_text="r", tools=[{"type": "function", "function": tool}],
                               expected_tool_calls=[])]
    first, second = bfcl_items(samples)
    assert (first["id"], first["lang"], first["query"], first["tools"], first["gold"]) == ("simple_0", "en-US", "q", [tool], [call])
    assert first["policy"] == "verbatim"
    assert second["gold"] == []


# ── BFCL checker parity, from the gorilla cross-check ─────────────────────────
def one_tool(props, required=()):
    return [{"name": "f", "description": "", "parameters": {"type": "object", "properties": props, "required": list(required)}}]


def f_reply(*calls):
    return json.dumps([{"tool_name": "f", "parameters": args} for args in calls])


FLOATS = one_tool({"xs": {"type": "array", "items": {"type": "number"}}})


def test_ints_in_a_float_array_fail_as_in_bfcl():
    gold = [{"name": "f", "arguments": {"xs": [[1.0, 3.0]]}}]
    assert score_item(item(gold, tools=FLOATS), f_reply({"xs": [1, 3]}))["ast_strict"] == 0
    assert score_item(item(gold, tools=FLOATS), f_reply({"xs": [1.0, 3.0]}))["ast_strict"] == 1


def test_ints_in_a_float_array_pass_when_gold_lists_ints():
    gold = [{"name": "f", "arguments": {"xs": [[1, 3]]}}]
    assert score_item(item(gold, tools=FLOATS), f_reply({"xs": [1, 3]}))["ast_strict"] == 1


def test_an_argument_the_schema_requires_cannot_be_dropped_even_if_gold_marks_it_optional():
    tools = one_tool({"a": {"type": "integer"}, "b": {"type": "integer"}}, required=["a", "b"])
    gold = [{"name": "f", "arguments": {"a": [1], "b": [2, ""]}}]
    assert score_item(item(gold, tools=tools), f_reply({"a": 1}))["ast_strict"] == 0
    assert score_item(item(gold, tools=tools), f_reply({"a": 1, "b": 2}))["ast_strict"] == 1


def test_each_gold_call_takes_the_first_call_that_fully_passes():
    tools = one_tool({"a": {"type": "integer"}, "b": {"type": "integer"}})
    gold = [{"name": "f", "arguments": {"a": [1], "b": ["", 2]}}, {"name": "f", "arguments": {"a": [1], "b": [2]}}]
    assert score_item(item(gold, tools=tools), f_reply({"a": 1}, {"a": 1, "b": 2}))["ast_strict"] == 1


def test_nested_dicts_match_their_accepted_values_not_the_literal_gold():
    tools = one_tool({"c": {"type": "object"}})
    gold = [{"name": "f", "arguments": {"c": [{"dept": ["Science"], "school": ["BHS", ""]}]}}]
    assert score_item(item(gold, tools=tools), f_reply({"c": {"dept": "Science"}}))["ast_strict"] == 1
    assert score_item(item(gold, tools=tools), f_reply({"c": {"dept": ["Science"], "school": ["BHS", ""]}}))["ast_strict"] == 0


# ── SFT ───────────────────────────────────────────────────────────────────────
class CharTokenizer:
    """One token per character, enough to check what encode() masks."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "".join(m["content"] for m in messages) + "|"

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def test_target_is_exactly_the_format_the_prompt_asks_for():
    assert target(item(ALARM)) == '[{"tool_name": "alarm.set", "parameters": {"time": "five am"}}]'
    assert target(item([])) == "[]"


def test_target_keeps_non_latin_values_readable():
    gold = [{"name": "alarm.set", "arguments": {"time": ["सुबह"]}}]
    assert "सुबह" in target(item(gold))


def test_encode_masks_the_prompt_and_ends_at_the_stop_token():
    gold_item = {**item(ALARM), "policy": "verbatim"}
    row = encode(gold_item, CharTokenizer())
    reply = [ord(c) for c in target(gold_item) + STOP_TOKEN]
    n_prompt = len("".join(m["content"] for m in training_messages(gold_item))) + 1
    assert row["input_ids"][n_prompt:] == reply
    assert row["labels"] == [-100] * n_prompt + reply


def test_target_scores_perfectly_and_earns_full_reward():
    gold_item = {**item(ALARM), "policy": "verbatim"}
    assert score_item(gold_item, target(gold_item))["ast_verbatim"] == 1
    assert reward(target(gold_item), gold_item) == pytest.approx(1.0)
