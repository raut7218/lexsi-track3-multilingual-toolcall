"""MASSIVE rows to tool-calling items, following the MASSIVE-Agents paper's stated rules
(Sections 2.3 and 2.4). Its manual filter list isn't published, so this is a rebuild.
"""
import random
import re

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
