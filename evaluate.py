import json
import math
import os
import random
from collections.abc import Callable
from typing import Optional

import fire
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList

from logit_processor import ConstrainedLogitsProcessor


try:
    from llamafactory.extras.trie import Trie
except ImportError:
    import sys

    _root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_root, "src"))
    from llamafactory.extras.trie import Trie


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_assistant_prefix_from_tokenizer(tokenizer: AutoTokenizer) -> str:
    try:
        messages = [{"role": "user", "content": "x"}]
        with_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        without_gen = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        return with_gen[len(without_gen) :]
    except Exception:
        raise ValueError("Failed to get assistant prefix from tokenizer.")


def build_trie_from_index(
    index_path: str,
    tokenizer: AutoTokenizer,
    prefix: Optional[str] = None,
) -> tuple[Trie, list[int], int]:
    with open(index_path, encoding="utf-8") as f:
        index_data = json.load(f)

    item_strs = set()
    for tokens in index_data.values():
        item_str = "".join(tokens)
        item_strs.add(item_str)

    if prefix is None:
        prefix = get_assistant_prefix_from_tokenizer(tokenizer)

    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    prefix_index = len(prefix_ids)
    prompt_suffix_ids = prefix_ids

    eos_id = tokenizer.eos_token_id
    sequences = []
    for item_str in item_strs:
        full_ids = tokenizer(prefix + item_str, add_special_tokens=False).input_ids
        item_ids = full_ids[len(prefix_ids) :]
        sequences.append(prefix_ids + item_ids + [eos_id])

    trie = Trie(sequences)
    return trie, prompt_suffix_ids, prefix_index


def create_prefix_allowed_tokens_fn(trie: Trie, prompt_suffix_ids: list[int]) -> Callable[[int, list[int]], list[int]]:
    def prefix_allowed_tokens_fn(input_ids: list[int]) -> list[int]:
        # input_ids: [prefix_ids, item_ids, ...] -> trie -> next allowed token
        # lookup_key = prompt_suffix_ids + input_ids
        return trie.get(input_ids)

    return prefix_allowed_tokens_fn


def format_prompt(
    tokenizer: AutoTokenizer,
    system: str,
    instruction: str,
    input_text: str,
) -> str:
    user_content = f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )


def compute_metrics(
    test_data: list,
    topk_list: list = None,
    save_path: Optional[str] = None,
) -> dict:
    if topk_list is None:
        topk_list = [1, 3, 5, 10, 20, 50]

    n = len(test_data)
    if n == 0:
        return {}

    # n_beam from first sample's predict length
    first_predict = test_data[0].get("predict", [])
    if isinstance(first_predict, str):
        n_beam = 1 if first_predict else 0
    else:
        n_beam = len(first_predict) if first_predict else 0
    valid_topk = [k for k in topk_list if k <= n_beam] if n_beam > 0 else topk_list

    all_ndcg = dict.fromkeys(valid_topk, 0.0)
    all_hr = dict.fromkeys(valid_topk, 0)

    for sample in test_data:
        predict = sample.get("predict", "")
        target = sample.get("output", "")

        if isinstance(predict, str):
            pred_items = [predict.strip('"\n').strip()] if predict else [""]
        else:
            pred_items = [str(p).strip('"\n').strip() for p in predict]

        if isinstance(target, list):
            target_item = target[0].strip('"\n').strip(" ") if target else ""
        else:
            target_item = str(target).strip(' \n"')

        min_id = len(pred_items)
        for i, pred in enumerate(pred_items):
            if pred == target_item:
                min_id = i
                break

        for topk in valid_topk:
            if topk > len(pred_items):
                continue
            if min_id < topk:
                all_hr[topk] = all_hr.get(topk, 0) + 1
                # DCG = 1/ln(rank+2), same as MiniOneRec calc.py
                all_ndcg[topk] = all_ndcg.get(topk, 0) + (1.0 / math.log(min_id + 2))

    metrics = {}
    for k in valid_topk:
        if k in all_hr:
            metrics[f"HR@{k}"] = all_hr[k] / n
            metrics[f"NDCG@{k}"] = all_ndcg[k] / n / (1.0 / math.log(2))

    if metrics and save_path:

        def _round4(x):
            try:
                return round(float(x), 4)
            except Exception:
                return x

        metrics_to_save = {k: _round4(v) for k, v in metrics.items()}
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics_to_save, f, indent=2, ensure_ascii=False)

        tsv_path = os.path.splitext(save_path)[0] + ".tsv"
        tsv_columns = [f"HR@{k}" for k in valid_topk] + [f"NDCG@{k}" for k in valid_topk]
        row_path = os.path.dirname(save_path) or "."
        row_name = row_path.split("/", 1)[1] if "/" in row_path else row_path
        row_data = {c: metrics_to_save.get(c) for c in tsv_columns}
        df = pd.DataFrame([row_data], index=[row_name])
        df.index.name = "path"
        df.to_csv(tsv_path, sep="\t", encoding="utf-8")

    return metrics


def main(
    model_name_or_path: str = "saves/qwen2.5-1.5b/full/industry_rec_sft",
    test_data_path: str = "data/industry_rec_test.json",
    result_json_path: str = "temp/eval-industry_rec_sft/result.json",
    index_path: str = "data/Industrial_and_Scientific.index.json",
    batch_size: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 1.0,
    do_sample: bool = True,
    num_beams: int = 50,
    length_penalty: float = 0.0,
    repetition_penalty: float = 1.1,
    seed: int = 42,
    compute_metrics_flag: bool = True,
    metrics_only: bool = False,
    num_return_sequences: int = None,
    top_p: float = None,
    top_k: int = None,
    gen_config_path: str = None,
    prefix: Optional[str] = None,
):
    if metrics_only:
        with open(result_json_path, encoding="utf-8") as f:
            test_data = json.load(f)
        metrics_path = os.path.join(os.path.dirname(result_json_path), "metrics.json")
        metrics = compute_metrics(test_data, save_path=metrics_path)
        if metrics:
            print("\n=== Metrics ===")
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"NDCG@{k}"
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"HR@{k}"
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
            tsv_path = os.path.splitext(metrics_path)[0] + ".tsv"
            print(f"Metrics saved to: {metrics_path}, {tsv_path}")
        return

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(test_data_path, encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"Loading test data: {test_data_path}, samples: {len(test_data)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    add_tokens_path = os.path.join(script_dir, "data", "new_tokens.json")
    if os.path.exists(add_tokens_path):
        with open(add_tokens_path) as f:
            new_tokens = json.load(f)
        tokenizer.add_tokens(new_tokens)

    # Build Trie from index
    print(f"Building Trie from {index_path}...")
    trie, prompt_suffix_ids, prefix_index = build_trie_from_index(index_path, tokenizer, prefix=prefix)
    print(f"Trie built: prefix_index={prefix_index}, num_items={len(trie)}")

    prefix_allowed_tokens_fn = create_prefix_allowed_tokens_fn(trie, prompt_suffix_ids)
    # import pdb; pdb.set_trace()
    logits_processor = LogitsProcessorList(
        [
            ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                num_beams=num_beams,
                prefix_index=prefix_index,
                eos_token_id=tokenizer.eos_token_id,
            )
        ]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    prompts = []

    for sample in test_data:
        system = sample.get("system", "You are a helpful assistant.")
        instruction = sample.get("instruction", "Can you predict the next possible item that the user may expect?")
        input_text = sample.get("input", "")
        prompts.append(format_prompt(tokenizer, system, instruction, input_text))

    encodings = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )

    input_ids = encodings["input_ids"].to(model.device)
    attention_mask = encodings["attention_mask"].to(model.device)
    print("-" * 30)
    print(f"Example test tokens: {tokenizer.decode(input_ids[0])}")
    print(f"Prompt suffix tokens: {tokenizer.decode(prompt_suffix_ids)}")
    print("-" * 30)
    try:
        gen_config = getattr(model, "generation_config", None)
        if gen_config is None:
            gen_config = GenerationConfig.from_model_config(model.config)
    except Exception:
        gen_config = GenerationConfig()

    # Build user overrides as kwargs for generate() (highest priority)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences if num_return_sequences is not None else num_beams,
        length_penalty=length_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        temperature=temperature if do_sample else 1.0,
        do_sample=do_sample,
        top_p=None,
        top_k=None,
    )

    outputs = []
    num_batches = (len(prompts) + batch_size - 1) // batch_size
    max_len = input_ids.shape[1]

    for i in tqdm(range(num_batches), desc="Inference"):
        start, end = i * batch_size, min((i + 1) * batch_size, len(prompts))
        batch_input_ids = input_ids[start:end]
        batch_attention_mask = attention_mask[start:end]

        # Reset logits processor count per batch (processor is stateful)
        for proc in logits_processor:
            if hasattr(proc, "count"):
                proc.count = 0

        with torch.no_grad():
            generated = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                logits_processor=logits_processor,
                return_dict_in_generate=True,
                output_scores=True,
                **gen_kwargs,
            )
        batch_completions = generated.sequences[:, max_len:]
        batch_outputs = tokenizer.batch_decode(
            batch_completions, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # group by num_beams
        for j in range(end - start):
            beam_outputs = batch_outputs[j * num_beams : (j + 1) * num_beams]
            outputs.append(beam_outputs)

    for i, sample in enumerate(test_data):
        sample["predict"] = outputs[i] if i < len(outputs) else []

    os.makedirs(os.path.dirname(result_json_path) or ".", exist_ok=True)
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print(f"Results saved to: {result_json_path}")

    if compute_metrics_flag:
        metrics_path = os.path.join(os.path.dirname(result_json_path), "metrics.json")
        metrics = compute_metrics(test_data, save_path=metrics_path)
        if metrics:
            print("\n=== Metrics ===")
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"NDCG@{k}"
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
            for k in [1, 3, 5, 10, 20, 50]:
                key = f"HR@{k}"
                if key in metrics:
                    print(f"  {key}: {metrics[key]:.4f}")
            tsv_path = os.path.splitext(metrics_path)[0] + ".tsv"
            print(f"Metrics saved to: {metrics_path}, {tsv_path}")


if __name__ == "__main__":
    fire.Fire(main)
