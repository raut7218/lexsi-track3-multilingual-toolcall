"""Score one reply against one item. Item gold is [{"name", "arguments": {key: [accepted, ...]}}];
"" in an accepted list marks an optional argument, and an empty gold list means no tool fits.

Value levels:
  VERBATIM  exact equality
  STRICT    BFCL's checker, the same rule the MASSIVE-Agents paper scores with
  LENIENT   after norm()
"""
import re
import unicodedata

from auditkit.bfcl import _value_eq  # pinned commit; BFCL's comparison, kept faithful
from parse import parse

WRONG, LENIENT, STRICT, VERBATIM = 0, 1, 2, 3

# ponytail: English number words only; other languages' words need per-language tables.
_NUMBER_WORDS = dict(zip(
    "zero one two three four five six seven eight nine ten eleven twelve".split(),
    map(str, range(13))))


def norm(value):
    s = unicodedata.normalize("NFKC", str(value)).casefold()
    s = "".join(str(unicodedata.digit(c)) if c.isdigit() else c for c in s)
    s = re.sub(r"(\d{1,2}):00\b", r"\1", s).replace(".", "")
    s = "".join(" " if unicodedata.category(c)[0] in "PSZ" else c for c in s)
    return " ".join(_NUMBER_WORDS.get(w, w) for w in s.split())


def value_level(pred, alts, spec=None):
    if any(type(pred) is type(a) and pred == a for a in alts):
        return VERBATIM
    float_for_int = (spec or {}).get("type") == "integer" and isinstance(pred, float)
    if not float_for_int and any(_value_eq(pred, a) for a in alts):
        return STRICT
    n = norm(pred)
    return LENIENT if n and any(n == norm(a) for a in alts) else WRONG


def _properties(item, name):
    for tool in item["tools"]:
        fn = tool.get("function", tool)
        if fn["name"] == name:
            return (fn.get("parameters") or {}).get("properties") or {}
    return {}


def _lang_mismatch(item, pred, level):
    """Value wrong, and in the user's language where English was expected.

    ponytail: flags presence only. MLCL's three subtypes need manual labels; Igbo is
    Latin-script, so copying from the query stands in for a script check there.
    """
    if level >= STRICT or not isinstance(pred, str):
        return False
    if item["lang"] == "ig":
        return pred.casefold() in item["query"].casefold()
    return any(ord(c) > 127 and c.isalpha() for c in pred)


def _required(gold_call):
    return {k for k, alts in gold_call["arguments"].items() if "" not in alts}


def _bucket(gold):
    n = max(len(g["arguments"]) for g in gold)
    return "2+" if n >= 2 else str(n)


def _f1(tp, fp, fn):
    return 1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)


def _no_call_result(item, parsed):
    correct = not parsed["calls"]
    return {
        "id": item["id"], "lang": item["lang"], "n_calls": len(parsed["calls"]),
        "no_call_correct": int(correct), "error": "correct" if correct else "over_call",
        "format_valid": None, "strict_format": int(parsed["strict"]), "fsa": None,
        "ast_verbatim": None, "ast_strict": None, "ast_lenient": None, "no_arg_bucket": None,
        "key_tp": 0, "key_fp": 0, "key_fn": 0, "val_tp": 0, "val_fp": 0, "val_fn": 0,
        "lang_mismatch": 0, "key_f1": None, "value_credit": None,
    }


def _pair(item, gold, calls):
    """Order doesn't matter, as in BFCL's no-order check: each gold call takes its best same-named call."""
    rest, pairs = list(calls), []
    for g in gold:
        props = _properties(item, g["name"])

        def matches(call):
            args = call["arguments"]
            right = sum(k in args and value_level(args[k], alts, props.get(k)) >= STRICT
                        for k, alts in g["arguments"].items())
            return right - len(set(args) - set(g["arguments"]))

        best = max((c for c in rest if c["name"] == g["name"]), key=matches)
        rest.remove(best)
        pairs.append((g, best))
    return pairs


def score_item(item, text):
    gold, parsed = item["gold"], parse(text)
    if not gold:
        return _no_call_result(item, parsed)

    calls = parsed["calls"]
    by_name = lambda call: call["name"]  # noqa: E731
    same_names = [c["name"] for c in sorted(calls, key=by_name)] == [g["name"] for g in sorted(gold, key=by_name)]

    key_tp = key_fp = key_fn = val_tp = val_fp = val_fn = 0
    levels, credits, extra, missing, mismatch = [], [], 0, 0, 0
    if same_names:
        for g, c in _pair(item, gold, calls):
            props = _properties(item, g["name"])
            pred, ref = c["arguments"], g["arguments"]
            need = _required(g)
            common = set(pred) & set(ref)
            extra_keys, missing_keys = set(pred) - set(ref), need - set(pred)
            extra += len(extra_keys)
            missing += len(missing_keys)
            key_tp += len(common)
            key_fp += len(extra_keys)
            key_fn += len(missing_keys)
            arg_levels = {k: value_level(pred[k], ref[k], props.get(k)) for k in common}
            levels += arg_levels.values()
            credits += [{VERBATIM: 1.0, STRICT: 1.0, LENIENT: 0.5, WRONG: 0.0}[v]
                        for v in arg_levels.values()] + [0.0] * (len(extra_keys) + len(missing_keys))
            wrong = {k for k, v in arg_levels.items() if v < STRICT}
            val_tp += len(common) - len(wrong)
            val_fp += len(wrong) + len(extra_keys)
            val_fn += len(missing_keys) + len(wrong & need)
            mismatch += any(_lang_mismatch(item, pred[k], arg_levels[k]) for k in wrong)
    else:
        n_required = sum(len(_required(g)) for g in gold)
        n_pred = sum(len(c["arguments"]) for c in calls)
        key_fn, key_fp, val_fn, val_fp = n_required, n_pred, n_required, n_pred

    shape_ok = same_names and not extra and not missing
    lowest = min(levels, default=VERBATIM)

    if not parsed["parsed"]:
        error = "invalid_syntax"
    elif len(calls) != len(gold):
        error = "wrong_count"
    elif not same_names:
        error = "wrong_function"
    elif extra:
        error = "unexpected_param"
    elif missing:
        error = "missing_param"
    elif lowest < STRICT:
        error = "invalid_value"
    else:
        error = "correct"

    return {
        "id": item["id"], "lang": item["lang"], "n_calls": len(calls),
        "no_call_correct": None, "error": error,
        "format_valid": int(parsed["parsed"]), "strict_format": int(parsed["strict"]),
        "fsa": int(same_names),
        "ast_verbatim": int(shape_ok and lowest >= VERBATIM),
        "ast_strict": int(shape_ok and lowest >= STRICT),
        "ast_lenient": int(shape_ok and lowest >= LENIENT),
        "no_arg_bucket": _bucket(gold),
        "key_tp": key_tp, "key_fp": key_fp, "key_fn": key_fn,
        "val_tp": val_tp, "val_fp": val_fp, "val_fn": val_fn,
        "lang_mismatch": int(mismatch > 0),
        "key_f1": _f1(key_tp, key_fp, key_fn) if same_names else 0.0,
        "value_credit": sum(credits) / len(credits) if credits else float(same_names),
    }
