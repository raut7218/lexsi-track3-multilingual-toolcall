"""Step 4 SFT on HF Jobs: 1,000 train rows per language, a checkpoint every 2,000 examples.

Each checkpoint goes to the workspace dataset as soon as it is written, because job disks don't persist,
and a rerun resumes from the newest one. SMOKE=1 runs 4 steps into 04_models/sft_smoke instead.
"""
import gzip
import json
import os
import re
import time
from pathlib import Path

REPO = "GauravRaut/lexsi-track3-workspace"
DATA, OUT = Path("/tmp/workspace"), Path("/tmp/out")
TOKEN = os.environ["HF_TOKEN"]
SMOKE = os.environ.get("SMOKE") == "1"
PREFIX = "04_models/sft_smoke" if SMOKE else "04_models/sft"
PER_LANGUAGE = 4 if SMOKE else 1000
# Constant LR so late checkpoints aren't flattened by decay: dev can then say whether more data still helps.
SCHEDULE = dict(lr_scheduler_type="constant_with_warmup", warmup_steps=1 if SMOKE else 20, logging_steps=1 if SMOKE else 10,
                **(dict(max_steps=4, save_steps=2) if SMOKE else dict(num_train_epochs=1, save_steps=125)))


def upload(api, folder, path_in_repo):
    """Five tries: one Hub hiccup shouldn't end a run that takes hours."""
    for attempt in range(5):
        try:
            return api.upload_folder(repo_id=REPO, repo_type="dataset", folder_path=str(folder),
                                     path_in_repo=path_in_repo, commit_message=path_in_repo)
        except Exception as error:
            if attempt == 4:
                raise
            print("hub retry:", type(error).__name__, flush=True)
            time.sleep(60)


def latest(files):
    """Newest checkpoint folder under PREFIX, or None."""
    steps = {int(m.group(1)) for f in files if (m := re.match(rf"{PREFIX}/checkpoint-(\d+)/", f))}
    return f"{PREFIX}/checkpoint-{max(steps)}" if steps else None


def main():
    import config
    import torch
    import train_sft
    from huggingface_hub import HfApi, snapshot_download
    from transformers import TrainerCallback

    api = HfApi(token=TOKEN)
    resume = latest(api.list_repo_files(REPO, repo_type="dataset"))
    snapshot_download(REPO, repo_type="dataset", local_dir=DATA, token=TOKEN,
                      allow_patterns=["01_data/items/sft_*"] + ([f"{resume}/*"] if resume else []))
    pool = []
    for L in config.LOCALES:
        with gzip.open(DATA / "01_data/items" / f"sft_{L}.jsonl.gz", "rt", encoding="utf-8") as f:
            pool += [json.loads(line) for line in f][:PER_LANGUAGE]
    print(f"{len(pool)} examples | resume from: {resume}", flush=True)

    class Upload(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            name = f"checkpoint-{state.global_step}"
            upload(api, Path(args.output_dir) / name, f"{PREFIX}/{name}")
            print("uploaded", name, flush=True)

    trainer = train_sft.build(pool, str(OUT / "sft"), token=TOKEN, batch_size=2, grad_accum=8, **SCHEDULE)
    trainer.add_callback(Upload())
    started = time.time()
    result = trainer.train(resume_from_checkpoint=str(DATA / resume) if resume else None)
    summary = {"examples": len(pool), "steps": trainer.state.global_step, "resumed_from": resume,
               "seconds_this_run": round(time.time() - started, 1), "train_loss": result.training_loss,
               "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    api.upload_file(path_or_fileobj=str(OUT / "summary.json"), path_in_repo=f"{PREFIX}/summary.json",
                    repo_id=REPO, repo_type="dataset")
    print("SUMMARY", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
