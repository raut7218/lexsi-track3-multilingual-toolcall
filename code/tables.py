"""Markdown tables from a stage's saved score files: python tables.py base. Needs no GPU."""
import collections
import json
import math
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "GauravRaut/lexsi-track3-workspace"
LOCALES = ("en-US", "fr-FR", "es-ES", "zh-CN", "ja-JP", "ar-SA", "hi-IN", "id-ID", "vi-VN", "tr-TR")
HELD_OUT = ("sw-KE", "am-ET", "te-IN", "km-KH")
PAPER = Path(__file__).parent.parent / "admin/paper_rows.json"


def load(stage):
    prefix = f"03_preds/{stage}/"
    passes = collections.defaultdict(dict)
    for rel in HfApi().list_repo_files(REPO, repo_type="dataset"):
        if rel.startswith(prefix) and rel.endswith(".json"):
            name, group = rel.removeprefix(prefix).removesuffix(".json").split("/")
            path = hf_hub_download(REPO, rel, repo_type="dataset")
            passes[name][group] = json.loads(Path(path).read_text(encoding="utf-8"))
    return passes


def pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}"


def mean(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return sum(values) / len(values) if values else None


def wilson(p, n, z=1.96):
    """95% interval for one proportion, in percent."""
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return f"{100 * (centre - half):.1f}–{100 * (centre + half):.1f}"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    print("\n".join(lines) + "\n")


def by_pass(passes):
    rows = []
    for name, groups in sorted(passes.items()):
        r = list(groups.values())
        ast = [x["ast_strict"] for x in r if x.get("ast_strict") is not None]
        rows.append((name, len(r), sum(x["n_call"] for x in r), sum(x["n_no_call"] for x in r),
                     pct(mean(r, "ast_strict")), f"{pct(min(ast))}–{pct(max(ast))}", pct(mean(r, "fsa")),
                     pct(mean(r, "format_valid")), pct(mean(r, "strict_format")),
                     pct(mean(r, "no_call_correct")), pct(mean(r, "hit_token_cap"))))
    table(("Pass", "Files", "Call rows", "No-call rows", "AST strict", "AST range", "Right function",
           "Format", "Strict format", "No-call", "Hit cap"), rows)


def by_group(groups, order):
    table(("Group", "Call rows", "AST strict", "95% CI", "AST verbatim", "AST lenient", "Right function",
           "No-call", "Hit cap"),
          [(g, s["n_call"], pct(s["ast_strict"]),
            wilson(s["ast_strict"], s["n_call"]) if s["n_call"] else "n/a", pct(s["ast_verbatim"]),
            pct(s["ast_lenient"]), pct(s["fsa"]), pct(s["no_call_correct"]), pct(s["hit_token_cap"]))
           for g, s in ((g, groups[g]) for g in order if g in groups)])


def errors(passes, names):
    kinds = sorted({k for n in names for s in passes[n].values() for k in s["errors"]})
    rows = []
    for name in names:
        total = collections.Counter()
        for s in passes[name].values():
            total.update(s["errors"])
        n = sum(total.values())
        rows.append((name, n, *(f"{100 * total[k] / n:.1f}" for k in kinds)))
    table(("Pass", "Rows", *kinds), rows)


def paper(passes):
    tables = json.loads(PAPER.read_text(encoding="utf-8"))["tables"]
    for mode in ("zero_shot", "one_shot"):
        rows = tables[mode]
        ours = passes.get(f"catalogue_{mode}", {})
        ranked = sorted(rows, key=lambda m: -sum(rows[m][L] for L in LOCALES))
        picks = ["Llama 3.2 3B", ranked[0]]
        head = ("Language", "Tiny Aya catalogue", *picks)
        body = [(L, pct(ours[L]["ast_strict"]), *(f"{rows[m][L]:.1f}" for m in picks))
                for L in LOCALES if L in ours]
        body.append(("Mean of these 10", pct(mean([ours[L] for L in LOCALES if L in ours], "ast_strict")),
                     *(f"{sum(rows[m][L] for L in LOCALES) / len(LOCALES):.1f}" for m in picks)))
        print(f"Paper comparison, {mode}. Highest of 21 on these 10 languages: {ranked[0]}.\n")
        table(head, body)


def main(stage):
    passes = load(stage)
    by_pass(passes)
    for name in sorted(passes):
        if name.startswith("shortlist"):
            print(f"{name}\n")
            by_group(passes[name], LOCALES)
    for name in sorted(passes):
        if name.startswith("heldout"):
            print(f"{name}\n")
            by_group(passes[name], HELD_OUT)
    for name in sorted(passes):
        if name.startswith("bfcl"):
            print(f"{name}\n")
            by_group(passes[name], ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance"))
    errors(passes, [n for n in sorted(passes) if not n.startswith("bfcl")])
    if stage == "base":
        paper(passes)


if __name__ == "__main__":
    main(*sys.argv[1:])
