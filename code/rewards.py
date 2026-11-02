"""GRPO reward: format, function, keys, values, each gated on the one before it."""
from score import score_item

WEIGHTS = {"format": 0.1, "function": 0.3, "keys": 0.3, "values": 0.3}


def reward_parts(text, item):
    s = score_item(item, text)
    called = s["n_calls"] > 0
    if not item["gold"]:
        correct = s["no_call_correct"]
        return {"format": s["strict_format"], "function": correct, "keys": correct, "values": correct}

    # "[]" is valid syntax but no call, so it can't collect the format part on a call item.
    fmt = int(s["strict_format"] and called)
    function = fmt * s["fsa"]
    keys = function * s["key_f1"]
    return {"format": fmt, "function": function, "keys": keys,
            "values": s["value_credit"] if function and keys == 1 else 0}


def reward(text, item):
    return sum(WEIGHTS[k] * v for k, v in reward_parts(text, item).items())
