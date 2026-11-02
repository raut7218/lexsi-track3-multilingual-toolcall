"""SFT on the 10-language train split. Prompts render exactly as in eval, and only the reply earns loss."""
from config import MAX_CONTEXT, SEED, STOP_TOKEN
from generate import load_model
from prompts import target, training_messages

LORA = dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")


def encode(item, tk):
    prompt = tk.apply_chat_template(training_messages(item), tokenize=False, add_generation_prompt=True)
    prompt_ids = tk(prompt, add_special_tokens=False)["input_ids"]
    reply_ids = tk(target(item) + STOP_TOKEN, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt_ids + reply_ids, "labels": [-100] * len(prompt_ids) + reply_ids}


def build(items, output_dir, token=None, batch_size=2, grad_accum=8, learning_rate=2e-4, **args):
    """Returns the trainer. Extra keyword arguments go to SFTConfig, such as max_steps or save_steps."""
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer

    model, tk = load_model(token=token)
    tk.padding_side = "right"
    model = get_peft_model(model, LoraConfig(**LORA))
    model.enable_input_require_grads()
    config = SFTConfig(output_dir=output_dir, per_device_train_batch_size=batch_size,
                       gradient_accumulation_steps=grad_accum, learning_rate=learning_rate, fp16=True, bf16=False,
                       gradient_checkpointing=True, report_to="none", seed=SEED,
                       max_length=MAX_CONTEXT, **args)  # TRL's default of 1024 would cut every reply off
    return SFTTrainer(model=model, args=config, processing_class=tk,
                      train_dataset=Dataset.from_list([encode(item, tk) for item in items]))
