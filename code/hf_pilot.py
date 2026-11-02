"""Does HF's t4-small match the Colab T4? Same slices and settings as Step 2. Each stage runs in its
own process, so it starts clean, and results reach the Hub after each one.
"""
import gzip
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = "GauravRaut/lexsi-track3-workspace"
OUT = Path("/tmp/out")
DATA = Path("/tmp/workspace")
TOKEN = os.environ["HF_TOKEN"]
SFT_STEPS, SFT_BATCH, SFT_ACCUM, GRPO_STEPS = 8, 2, 8, 4


def load(name):
    return [json.loads(line) for line in gzip.open(DATA / "01_data/items" / f"{name}.jsonl.gz", "rt", encoding="utf-8")]


def peaks():
    import torch
    return {"peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "reserved_gpu_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "peak_ram_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)}


def stage_eval():
    import config
    import generate
    import prompts
    model, tk = generate.load_model(token=TOKEN)
    items = [dict(i, id=f'{i["lang"]}:{i["id"]}') for L in config.LOCALES for i in load(f"test_{L}")[:6]]
    timing = generate.run(model, tk, items, prompts.training_messages, OUT / "eval.jsonl")
    new, old = generate.read_preds(OUT / "eval.jsonl"), generate.read_preds(DATA / "03_preds/pilot/bs4.jsonl")
    return {**timing, "items_per_s": round(timing["items"] / timing["seconds"], 2), "colab_items_per_s": 0.82,
            "same_as_colab": sum(new[k]["reply"] == old[k]["reply"] for k in old), "of": len(old), **peaks()}


def stage_sft():
    import config
    import train_sft
    pool = [i for L in config.LOCALES for i in load(f"sft_{L}")[:30]]
    trainer = train_sft.build(pool, str(OUT / "sft"), token=TOKEN, batch_size=SFT_BATCH, grad_accum=SFT_ACCUM,
                              max_steps=SFT_STEPS, logging_steps=SFT_STEPS, save_strategy="no")
    started = time.time()
    result = trainer.train()
    seconds = time.time() - started
    trainer.save_model(str(OUT / "sft_adapter"))
    return {"seconds": round(seconds, 1), "examples_per_s": round(SFT_STEPS * SFT_BATCH * SFT_ACCUM / seconds, 2),
            "colab_examples_per_s": 0.36, "loss": round(result.training_loss, 4), **peaks()}


def stage_grpo():
    import config
    import train_grpo
    pool = [i for L in config.LOCALES for i in load(f"grpo_{L}")[:6]]
    trainer = train_grpo.build(pool, str(OUT / "grpo"), str(OUT / "rewards.jsonl"), adapter=str(OUT / "sft_adapter"),
                               token=TOKEN, max_steps=GRPO_STEPS, logging_steps=1, save_strategy="no")
    started = time.time()
    trainer.train()
    seconds = time.time() - started
    return {"seconds": round(seconds, 1), "s_per_step": round(seconds / GRPO_STEPS, 1), "colab_s_per_step": 31.8,
            **peaks()}


def publish(api):
    api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(OUT), ignore_patterns=["sft*/*", "grpo/*"],
                      path_in_repo=f'03_preds/hf_pilot/{os.environ.get("JOB_ID", "local")}')


def main():
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi(token=TOKEN)
    snapshot_download(REPO, repo_type="dataset", local_dir=DATA, token=TOKEN,
                      allow_patterns=["01_data/items/*", "03_preds/pilot/bs4.jsonl"])
    gpu = subprocess.run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader", shell=True,
                         capture_output=True, text=True).stdout.strip()
    env = {"flavor": os.environ.get("ACCELERATOR"), "gpu": gpu, "memory": os.environ.get("MEMORY"),
           "cores": os.environ.get("CPU_CORES"), "python": platform.python_version()}
    (OUT / "env.json").write_text(json.dumps(env, indent=1))
    print("ENV", json.dumps(env), flush=True)
    failed = 0
    for stage in ("eval", "sft", "grpo"):
        failed += subprocess.run([sys.executable, __file__, stage]).returncode != 0
        path = OUT / f"{stage}.json"
        print(stage.upper(), path.read_text() if path.exists() else "FAILED, no result", flush=True)
        publish(api)
    sys.exit(failed)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        result = {"eval": stage_eval, "sft": stage_sft, "grpo": stage_grpo}[sys.argv[1]]()
        (OUT / f"{sys.argv[1]}.json").write_text(json.dumps(result, indent=1))
    else:
        main()
