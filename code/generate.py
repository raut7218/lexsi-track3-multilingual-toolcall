"""Batched greedy eval. One JSON line per item, appended as it goes, so a new session skips finished ids."""
import json
import time
from pathlib import Path

import torch

from config import MAX_CONTEXT, MAX_NEW_TOKENS, MODEL, STOP_TOKEN


def read_preds(path):
    """Finished rows by id. A crash can tear a line, and that row just counts as not done."""
    rows = {}
    if Path(path).exists():
        for line in Path(path).open(encoding="utf-8"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[row["id"]] = row
    return rows


def pending(items, path):
    done = read_preds(path)
    return [item for item in items if item["id"] not in done]


def load_model(adapter=None, token=None, trainable=False):
    """fp16 base, eos set to what Tiny Aya really emits, left padding for batched generation."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tk = AutoTokenizer.from_pretrained(MODEL, token=token, padding_side="left")
    tk.eos_token = STOP_TOKEN
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=token).to("cuda")
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    return (model.train() if trainable else model.eval()), tk


def run(model, tk, items, render, path, batch_size=4, max_new_tokens=MAX_NEW_TOKENS):
    """render(item) gives the messages. Returns timing; rows go to path as they finish."""
    todo = pending(items, path)
    texts = [tk.apply_chat_template(render(i), tokenize=False, add_generation_prompt=True) for i in todo]
    order = sorted(range(len(todo)), key=lambda n: -len(texts[n]))  # longest first: an OOM shows up at once
    started, n_new = time.time(), 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as log:
        log.write("\n")  # a torn last line must not swallow the next row
        for start in range(0, len(order), batch_size):
            picks = order[start:start + batch_size]
            enc = tk([texts[n] for n in picks], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            width = enc["input_ids"].shape[1]
            if width + max_new_tokens > MAX_CONTEXT:
                raise ValueError(f"prompt of {width} tokens leaves no room for {max_new_tokens} new ones")
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tk.pad_token_id, eos_token_id=tk.eos_token_id)
            for n, row in zip(picks, out[:, width:].tolist()):
                stopped = tk.eos_token_id in row
                row = row[:row.index(tk.eos_token_id)] if stopped else row
                n_new += len(row)
                log.write(json.dumps({"id": todo[n]["id"], "reply": tk.decode(row, skip_special_tokens=True),
                                      "n_new": len(row), "stopped": stopped}, ensure_ascii=False) + "\n")
            log.flush()
    return {"items": len(todo), "seconds": round(time.time() - started, 1), "new_tokens": n_new}
