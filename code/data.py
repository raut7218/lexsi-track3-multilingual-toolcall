"""MASSIVE rows to tool-calling items, following the MASSIVE-Agents paper's stated rules
(Sections 2.3 and 2.4). Its manual filter list isn't published, so this is a rebuild.
"""
import collections
import gzip
import json
import random
import re
import statistics
from pathlib import Path

from config import (ARG_CUT, ENGLISH_RATE, LOCALES, N_DEV, N_DEV_NO_CALL, N_GRPO, N_SFT, N_TEST,
                    N_TEST_NO_CALL, NO_CALL_RATE, SEED)
from schemas import ARG_DESC, FN_DESC

DROP_INTENTS = {"general_greet", "general_quirky", "cooking_query", "audio_volume_other", "music_settings"}
WHOLE_QUERY_INTENTS = {"qa_factoid", "qa_maths", "qa_currency", "datetime_convert"}

_SLOT = re.compile(r"\[\s*([^\[\]:]+?)\s*:\s*([^\[\]]*?)\s*\]")


def fn_name(intent):
    return intent.replace("_", ".", 1)


def slots(annot_utt):
    return _SLOT.findall(annot_utt)


def convert(row):
    """One MASSIVE row as {id, lang, query, scenario, name, args}, or None if filtered out."""
    if row["intent"] in DROP_INTENTS:
        return None
    if row["intent"] in WHOLE_QUERY_INTENTS:
        args = {"query": row["utt"]}
    else:
        pairs = slots(row["annot_utt"])
        args = dict(pairs)
        if len(args) != len(pairs):
            return None
    return {"id": row["id"], "lang": row["locale"], "query": row["utt"],
            "scenario": row["scenario"], "name": fn_name(row["intent"]), "args": args}


def judged(row):
    """Light filter on MASSIVE's judgments, using the scales in its README. en-US has none.

    slots_score 2 means "no slots" and intent_score 2 "a reasonable interpretation", so both pass.
    """
    if row["locale"] == "en-US":
        return True
    votes = row.get("judgments") or []
    need = len(votes) // 2 + 1
    return bool(votes) and (
        sum(v["intent_score"] in (1, 2) for v in votes) >= need
        and sum(v["slots_score"] in (1, 2) for v in votes) >= need
        and statistics.median(v["grammar_score"] for v in votes) >= 3
        and statistics.median(v["spelling_score"] for v in votes) >= 1
        and sum("target" in v["language_identification"] for v in votes) >= need)


def arg_schema(rows, cut=ARG_CUT):
    """Arguments per intent: slots in at least `cut` of its train rows. Train only, so no leakage."""
    total, used = collections.Counter(), collections.defaultdict(collections.Counter)
    for row in rows:
        if row["partition"] == "train" and row["intent"] not in DROP_INTENTS:
            total[row["intent"]] += 1
            used[row["intent"]].update({key for key, _ in slots(row["annot_utt"])})
    return {intent: ["query"] if intent in WHOLE_QUERY_INTENTS else
            sorted(key for key, n in used[intent].items() if n / total[intent] >= cut)
            for intent in total}


def why_dropped(row, schema):
    """The filter that removes this row, or None if it stays."""
    if row["intent"] in DROP_INTENTS:
        return "open_ended_intent"
    conv = convert(row)
    if conv is None:
        return "duplicate_argument"
    if not set(conv["args"]) <= set(schema[row["intent"]]):
        return "rare_argument"
    return None if judged(row) else "judgment"


def functions_json(schema):
    """One English schema per function. Only `query` is required: no-argument calls are valid."""
    out = []
    for intent in sorted(schema):
        props = {key: {"type": "string", "description": ARG_DESC[key]} for key in schema[intent]}
        out.append({"name": fn_name(intent), "description": FN_DESC[intent],
                    "parameters": {"type": "object", "properties": props,
                                   "required": ["query"] if intent in WHOLE_QUERY_INTENTS else []}})
    return out


def stratified(convs, n, seed):
    """Paper Section 2.6: cycle through shuffled functions, taking a random argument tuple from each."""
    rng = random.Random(seed)
    pools = collections.defaultdict(lambda: collections.defaultdict(list))
    for conv in sorted(convs, key=lambda c: int(c["id"])):
        pools[conv["name"]][tuple(sorted(conv["args"]))].append(conv)
    for groups in pools.values():
        for group in groups.values():
            rng.shuffle(group)
    picked, names = [], sorted(pools)
    while names and len(picked) < n:
        rng.shuffle(names)
        for name in list(names):
            if len(picked) == n:
                break
            key = rng.choice(sorted(pools[name]))
            picked.append(pools[name][key].pop())
            if not pools[name][key]:
                del pools[name][key]
            if not pools[name]:
                names.remove(name)
    return picked


def align_english(args, en_args):
    """Gold values for the English policy: the aligned en-US row's, if its slots match."""
    return en_args if set(args) == set(en_args) else None


def candidates(gold, scenarios, k, rng):
    """gold + same-scenario distractors + one random, topped up if the scenario is small."""
    same = sorted(n for n, s in scenarios.items() if s == scenarios[gold] and n != gold)
    other = sorted(n for n, s in scenarios.items() if s != scenarios[gold])
    n_same = min(k - 1 - (1 if k >= 3 else 0), len(same))
    picks = [gold, *rng.sample(same, n_same), *rng.sample(other, k - 1 - n_same)]
    rng.shuffle(picks)
    return picks


def build_item(conv, functions, scenarios, k, no_call, policy, seed):
    # Seeded by id alone so parallel rows get the same tools in every language.
    rng = random.Random(f"{seed}:{conv['id']}")
    names = candidates(conv["name"], scenarios, k + no_call, rng)
    if no_call:
        names.remove(conv["name"])
    gold = [] if no_call else [{"name": conv["name"],
                                "arguments": {key: [value] for key, value in conv["args"].items()}}]
    return {"id": conv["id"], "lang": conv["lang"], "query": conv["query"],
            "tools": [functions[n] for n in names], "gold": gold, "policy": policy}


def load_massive(raw_dir):
    return {locale: [json.loads(line) for line in (Path(raw_dir) / f"{locale}.jsonl").open(encoding="utf-8")]
            for locale in LOCALES}


def train_item(conv, en_conv, functions, scenarios, seed):
    """k, no-call and policy are seeded by id, so parallel rows match across languages."""
    rng = random.Random(f"{seed}:train:{conv['id']}")
    k, no_call, wants_english = rng.randint(2, 8), rng.random() < NO_CALL_RATE, rng.random() < ENGLISH_RATE
    en_args = align_english(conv["args"], en_conv["args"]) if en_conv else None
    if wants_english and en_args is not None:
        return build_item({**conv, "args": en_args}, functions, scenarios, k, no_call, "english", seed)
    return build_item(conv, functions, scenarios, k, no_call, "verbatim", seed)


def eval_items(id_list, n_no_call, convs, functions, scenarios, seed):
    """The last n_no_call ids become no-call items, so every language gets the same split."""
    cut = len(id_list) - n_no_call
    return [build_item(convs[row_id], functions, scenarios, 5, i >= cut, "verbatim", seed)
            for i, row_id in enumerate(id_list)]


def write_items(path, items):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_all(raw_dir, out_dir, seed=SEED):
    """Writes functions.json and one items file per split and language; returns counts for the report."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_massive(raw_dir)
    schema = arg_schema(row for locale in LOCALES for row in rows[locale])
    specs = functions_json(schema)
    functions = {spec["name"]: spec for spec in specs}
    (out / "functions.json").write_text(json.dumps(specs, ensure_ascii=False, indent=1), encoding="utf-8")
    scenarios = {fn_name(row["intent"]): row["scenario"] for row in rows["en-US"] if row["intent"] in schema}

    counts, kept = {}, {}
    for locale in LOCALES:
        reasons = {row["id"]: why_dropped(row, schema) for row in rows[locale]}
        kept[locale] = {row["id"]: (convert(row), row["partition"]) for row in rows[locale]
                        if reasons[row["id"]] is None}
        counts[locale] = {"rows": len(rows[locale]),
                          "removed": dict(collections.Counter(r for r in reasons.values() if r)),
                          "kept": dict(collections.Counter(part for _, part in kept[locale].values()))}

    common = set.intersection(*(set(kept[locale]) for locale in LOCALES))

    def shared(part):
        return [kept["en-US"][i][0] for i in sorted(common, key=int) if kept["en-US"][i][1] == part]

    test_ids = [c["id"] for c in stratified(shared("test"), N_TEST, seed)]
    dev_ids = [c["id"] for c in stratified(shared("dev"), N_DEV, seed)]
    if len(test_ids) < N_TEST or len(dev_ids) < N_DEV:
        raise ValueError(f"too few shared ids: test {len(test_ids)}, dev {len(dev_ids)}")

    english = {i: conv for i, (conv, _) in kept["en-US"].items()}
    for locale in LOCALES:
        convs = {i: conv for i, (conv, _) in kept[locale].items()}
        train = stratified([c for c, part in kept[locale].values() if part == "train"], N_SFT + N_GRPO, seed)
        if len(train) < N_SFT + N_GRPO:
            raise ValueError(f"{locale}: only {len(train)} train rows")
        items = [train_item(c, english.get(c["id"]), functions, scenarios, seed) for c in train]
        write_items(out / f"sft_{locale}.jsonl.gz", items[:N_SFT])
        write_items(out / f"grpo_{locale}.jsonl.gz", items[N_SFT:])
        write_items(out / f"dev_{locale}.jsonl.gz",
                    eval_items(dev_ids, N_DEV_NO_CALL, convs, functions, scenarios, seed))
        write_items(out / f"test_{locale}.jsonl.gz",
                    eval_items(test_ids, N_TEST_NO_CALL, convs, functions, scenarios, seed))
        counts[locale].update(
            aligned=sum(c["id"] in english and align_english(c["args"], english[c["id"]]["args"]) is not None
                        for c in train),
            train_used=len(train), english_policy=sum(i["policy"] == "english" for i in items),
            no_call=sum(not i["gold"] for i in items))
    (out / "eval_ids.json").write_text(json.dumps({"test": test_ids, "dev": dev_ids}), encoding="utf-8")
    return {"counts": counts, "shared": {part: len(shared(part)) for part in ("train", "dev", "test")},
            "n_functions": len(specs), "n_args": sum(len(v) for v in schema.values()),
            "arg_names": len({key for args in schema.values() for key in args})}
