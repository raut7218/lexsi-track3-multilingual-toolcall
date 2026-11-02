"""GRPO through AgentTune, continuing the SFT adapter or starting from base.

No tools are passed, which keeps AgentTune on TRL's batched single-turn path. With tools, its rollout
put a second BOS on every prompt and generated a follow-up turn after every call.
"""
import json

from config import MAX_NEW_TOKENS
from generate import load_model
from prompts import training_messages
from rewards import make_reward
from train_sft import LORA


def grpo_row(item):
    return {"prompt": training_messages(item), "item": json.dumps(item, ensure_ascii=False)}


def build(items, output_dir, log_path, adapter=None, token=None, **args):
    """Returns AgentTune's trainer. Extra keyword arguments go to it, such as max_steps or learning_rate."""
    from agenttune.core.backend_factory import create_agentic_trainer
    from datasets import Dataset

    model, tk = load_model(adapter, token, trainable=True)
    if adapter is None:
        args["peft_config"] = dict(LORA)
    return create_agentic_trainer(
        algorithm="grpo", model=model, processing_class=tk, reward_funcs=[make_reward(log_path)],
        train_dataset=Dataset.from_list([grpo_row(item) for item in items]), output_dir=output_dir,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=4,
        max_completion_length=MAX_NEW_TOKENS, use_vllm=False, report_to="none", fp16=True, bf16=False, **args)
