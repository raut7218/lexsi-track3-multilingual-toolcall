"""The eval matrix for one stage. Finished files live in the workspace dataset, so a rerun skips them.

    python run_eval.py base
    python run_eval.py sft <adapter>
"""
import json
import os
import sys
import time
from pathlib import Path

import config
import data
import generate
import prompts
from score import summarize

REPO = "GauravRaut/lexsi-track3-workspace"
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
BFCL_REVISION = "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda"
BFCL_CATEGORIES = ("simple", "multiple", "parallel", "parallel_multiple", "irrelevance")
DATA, OUT = Path("/tmp/workspace"), Path("/tmp/out")
TOKEN = os.environ["HF_TOKEN"]


def load(name):
    return data.read_items(DATA / "01_data/items" / f"{name}.jsonl.gz")


def bfcl():
    import auditkit as ak
    from huggingface_hub import snapshot_download
    root = snapshot_download(BFCL_REPO, repo_type="dataset", revision=BFCL_REVISION, token=TOKEN,
                             local_dir=DATA / "bfcl", allow_patterns=["possible_answer/*", "BFCL_v3_*.json"])
    return {c: data.bfcl_items(ak.load_bfcl(f"{root}/BFCL_v3_{c}.json", revision=BFCL_REVISION[:7]))
            for c in BFCL_CATEGORIES}


def matrix(stage):
    """(pass, group, items, render) for every file the stage writes: the plan's matrix, cheapest passes early."""
    training = {"training": prompts.training_messages}
    shortlist = {**training, "zero_shot": prompts.bfcl_zero_shot, "one_shot": prompts.custom_one_shot}
    ood = {**training, "zero_shot": prompts.bfcl_zero_shot}
    if stage != "base":
        shortlist = ood = training

    test = {locale: load(f"test_{locale}") for locale in config.LOCALES}
    functions = json.loads((DATA / "01_data/items/functions.json").read_text(encoding="utf-8"))
    catalogue = data.sample_positions(config.N_CATALOGUE, 0)
    english = data.sample_positions(config.N_ENGLISH_CALL, config.N_ENGLISH_NO_CALL)
    bfcl_sets = bfcl()

    rows = []
    for name, render in shortlist.items():
        rows += [(f"shortlist_{name}", L, test[L], render) for L in config.LOCALES]
    rows += [("english_training", L, data.english_items(test[L], test["en-US"], english), prompts.training_messages)
             for L in config.LOCALES]
    for name, render in ood.items():
        rows += [(f"bfcl_{name}", c, bfcl_sets[c], render) for c in BFCL_CATEGORIES]
        rows += [(f"heldout_{name}", L, load(f"heldout_{L}"), render) for L in config.HELD_OUT]
    for name, render in shortlist.items():
        rows += [(f"catalogue_{name}", L, data.catalogue_items(test[L], functions, catalogue), render)
                 for L in config.LOCALES]
    return rows


def commit(api, message, paths):
    """Five tries: one Hub hiccup shouldn't end a run that takes hours."""
    from huggingface_hub import CommitOperationAdd
    for attempt in range(5):
        try:
            return api.create_commit(REPO, repo_type="dataset", commit_message=message,
                                     operations=[CommitOperationAdd(rel, str(OUT / rel)) for rel in paths])
        except Exception as error:
            if attempt == 4:
                raise
            print("hub retry:", type(error).__name__, flush=True)
            time.sleep(60)


def main(stage, adapter=None):
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=TOKEN)
    snapshot_download(REPO, repo_type="dataset", local_dir=DATA, token=TOKEN, allow_patterns=["01_data/items/*"])
    rows = matrix(stage)
    done = set(api.list_repo_files(REPO, repo_type="dataset"))
    todo = [row for row in rows if f"03_preds/{stage}/{row[0]}/{row[1]}.jsonl" not in done]
    print(f"{len(rows)} files, {sum(len(r[2]) for r in rows)} items | to run: {len(todo)} files, "
          f"{sum(len(r[2]) for r in todo)} items", flush=True)

    model, tk = generate.load_model(adapter, TOKEN)
    for name, group, items, render in todo:
        rel = f"03_preds/{stage}/{name}/{group}"
        path = OUT / f"{rel}.jsonl"
        timing = generate.run(model, tk, items, render, path)
        preds = generate.read_preds(path)
        if set(preds) != {item["id"] for item in items}:
            raise ValueError(f"{rel}: {len(preds)} replies for {len(items)} items")
        summary = {**timing, **summarize(items, preds)}
        (OUT / f"{rel}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        commit(api, rel, [f"{rel}.jsonl", f"{rel}.json"])
        print(rel, "items/s", round(timing["items"] / timing["seconds"], 2), "ast_strict", summary["ast_strict"],
              flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
