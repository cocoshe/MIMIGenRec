from typing import Optional, Union

import fire
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)
from trl import GRPOTrainer

from MIMIGenRec import MIMIGenRec, get_grpo_config
from rewards.ranking_reward import get_ndcg_rule_reward, rule_reward
from util import build_constrained_logits_processor


def main(
    model: str = "saves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz0",
    index_path: str = "data/Industrial_and_Scientific/Industrial_and_Scientific.index.json",
    prefix: Optional[str] = None,
    num_beams: int = 16,
    # num_beams: int = 2,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 50,
    data_dir: str = "data/Industrial_and_Scientific/rl",
    output_dir: str = "rl_outputs/qwen2.5-0.5b-instruct-grpo",
    # output_dir: str = "rl_outputs/MiniOneRec-MiniMind2-grpo",
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 2,
    num_train_epochs: int = 2,
    learning_rate: float = 1e-5,
    logging_steps: int = 1,
    eval_step: int = 20,
    eval_strategy: str = "steps",
    save_strategy: str = "steps",
    save_steps: Union[int, float] = 0.1,
    save_total_limit: int = 3,
    warmup_ratio: float = 0.03,
    max_grad_norm: float = 0.3,
    optim: str = "paged_adamw_32bit",
    lr_scheduler_type: str = "cosine",
    max_completion_length: int = 128,
    beta: float = 1e-3,
    repetition_penalty: float = 1.0,
    do_sample: bool = False,
    bf16: bool = True,
    deepspeed: Optional[str] = None,
    report_to: Optional[str] = None,
):
    # load dataset
    dataset = load_dataset(
        "json",
        data_files={
            "train": f"{data_dir}/train.json",
            "valid": f"{data_dir}/valid.json",
            "test": f"{data_dir}/test.json",
        },
    )
    train_dataset = dataset["train"]
    eval_dataset = dataset["valid"]
    test_dataset = dataset["test"]  # noqa: F841

    training_args = get_grpo_config(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_steps=eval_step,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        beta=beta,
        num_generations=num_beams,
        bf16=bf16,
        deepspeed=deepspeed,
        report_to=report_to,
    )

    tokenizer = AutoTokenizer.from_pretrained(model)
    logits_processor = build_constrained_logits_processor(index_path, tokenizer, prefix=prefix, num_beams=num_beams)

    model = MIMIGenRec.from_pretrained(
        model,
        logits_processor=logits_processor,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        max_completion_length=max_completion_length,
        repetition_penalty=repetition_penalty,
        do_sample=do_sample,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=[rule_reward, get_ndcg_rule_reward(num_beams)],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)


# from datasets import load_dataset
# from trl import GRPOTrainer

# dataset = load_dataset("trl-lib/tldr", split="train")

# # Dummy reward function: count the number of unique characters in the completions
# def reward_num_unique_chars(completions, **kwargs):
#     return [len(set(c)) for c in completions]

# from trl.trainer.grpo_config import GRPOConfig
# args = GRPOConfig(
#     output_dir="rl_outputs/Qwen2.5-0.5B-Instruct-grpo",
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     num_train_epochs=1,
#     learning_rate=5e-6,
#     logging_steps=10,
#     num_generations=2,
#     top_k=50,
#     top_p=1.0,
#     max_completion_length=128,
# )
# from transformers import AutoModelForCausalLM
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# trainer = GRPOTrainer(
#     model=model,
#     reward_funcs=reward_num_unique_chars,
#     train_dataset=dataset,
#     args=args,
# )
# trainer.train()
