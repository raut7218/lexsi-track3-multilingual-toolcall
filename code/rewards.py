"""GRPO reward: format, function, keys, values, each gated on the one before it."""
import json
from pathlib import Path

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


def make_reward(log_path):
    """GRPO's reward function. It logs every reply and its parts, for the spot checks and the audit."""
    def tool_call_reward(completions, item, **_):
        scores = []
        with Path(log_path).open("a", encoding="utf-8") as log:
            for completion, row in zip(completions, item):
                text = completion if isinstance(completion, str) else completion[0]["content"]
                parsed = json.loads(row)
                parts = reward_parts(text, parsed)
                scores.append(sum(WEIGHTS[k] * v for k, v in parts.items()))
                log.write(json.dumps({"id": parsed["id"], "lang": parsed["lang"], "reply": text, "parts": parts,
                                      "reward": scores[-1]}, ensure_ascii=False) + "\n")
        return scores
    return tool_call_reward
