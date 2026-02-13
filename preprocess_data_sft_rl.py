import json
import os
import random

import fire


SYSTEM_PROMPT = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."


def load_items(item_json_path: str) -> dict[str, str]:
    """Load item.json, return item_id -> title mapping."""
    with open(item_json_path, encoding="utf-8") as f:
        items = json.load(f)
    return {str(k): v.get("title", "") for k, v in items.items() if v.get("title")}


def load_items_full(item_json_path: str) -> dict[str, str]:
    """Load item.json, return item_id -> title mapping (fallback to Item_id when title missing, for title2sid task)."""
    with open(item_json_path, encoding="utf-8") as f:
        items = json.load(f)
    return {str(k): v.get("title", f"Item_{k}") for k, v in items.items()}


def load_items_with_desc(item_json_path: str) -> dict[str, dict]:
    """Load full item.json, return item_id -> {title, description} for RLTitle2SidDataset format."""
    with open(item_json_path, encoding="utf-8") as f:
        return json.load(f)


def load_index(index_path: str) -> dict[str, str]:
    """Load index.json, return item_id -> combined_sid mapping (e.g. <a_20><b_188><c_134>)."""
    with open(index_path, encoding="utf-8") as f:
        indices = json.load(f)
    id2sid = {}
    for item_id, sids in indices.items():
        if len(sids) >= 3:
            id2sid[str(item_id)] = sids[0] + sids[1] + sids[2]
    return id2sid


def load_index_raw(index_path: str) -> dict[str, list[str]]:
    """Load index.json raw structure: item_id -> [sid1, sid2, sid3]."""
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def extract_new_tokens(index_path: str) -> list[str]:
    """Extract all semantic ID tokens (e.g. <a_20>, <b_188>) from index.json, consistent with sft.py TokenExtender."""
    with open(index_path, encoding="utf-8") as f:
        indices = json.load(f)
    tokens = set()
    for sids in indices.values():
        for token in sids:
            tokens.add(token)
    return sorted(tokens)


def build_sid_trie(indices: dict[str, list[str]]) -> dict:
    """Build trie from index.json item_id -> [sid1, sid2, sid3].

    Each path is a token sequence, leaf nodes store item_id list (supports multiple items per semantic ID).
    Returns JSON-serializable nested dict.
    """
    trie: dict = {}
    for item_id, sids in indices.items():
        if len(sids) < 3:
            continue
        node = trie
        for i, token in enumerate(sids[:3]):  # use first 3 levels only
            if token not in node:
                node[token] = {}
            node = node[token]
        if "_item_ids" not in node:
            node["_item_ids"] = []
        node["_item_ids"].append(str(item_id))
    return trie


def parse_inter_file(inter_path: str) -> list[tuple]:
    """Parse .inter file, format: user_id:token  item_id_list:token_seq  item_id:token.

    Returns [(history_item_ids, target_item_id), ...].
    """
    rows = []
    with open(inter_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        user_id, item_id_list, item_id = parts
        history_ids = [x.strip() for x in item_id_list.split()] if item_id_list.strip() else []
        target_id = item_id.strip()
        rows.append((history_ids, target_id))
    return rows


def to_rl_format(
    sample: dict,
    data_source: str,
    ability: str,
    split: str,
    idx: int,
    task_name: str = "",
) -> dict:
    """Each sample must contain the following.

    - data_source: dataset name, used to index the reward function in RewardModel
    - prompt: prompt in Hugging Face chat_template format
    - ability: task category
    - reward_model: includes ground_truth, used for evaluation
    - extra_info: auxiliary metadata.
    """
    input_text = sample.get("input", "")
    user_content = input_text
    system = sample.get("system", SYSTEM_PROMPT)

    # prompt in Hugging Face chat_template format (include system to avoid tokenizer default "You are a helpful assistant.")
    prompt = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]

    # ground_truth: the output (target sid), strip trailing newlines
    ground_truth = sample.get("output", "").strip()

    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "task": task_name,
        },
    }


def build_seq_samples(
    rows: list[tuple],
    id2sid: dict[str, str],
    category: str = "",
    sample: int = -1,
    seed: int = 42,
    dedup: bool = False,
    is_test: bool = False,
) -> list[dict]:
    """Build sequential recommendation samples: user history (semantic ID) -> predict next semantic ID.

    Format consistent with sft.py SidSFTDataset.
    """
    samples = []
    for history_ids, target_id in rows:
        if target_id not in id2sid:
            continue
        target_sid = id2sid[target_id]
        history_sids = [id2sid.get(i) for i in history_ids]
        history_sids = [s for s in history_sids if s]
        if not history_sids:
            continue
        if dedup and history_sids and target_sid == history_sids[-1]:
            continue
        history_str = ", ".join(history_sids)
        if not is_test:  # train dataset
            input_text = f"The user has interacted with items {history_str} in chronological order. Can you predict the next possible item that the user may expect?"
        else:  # test dataset
            input_text = f"Can you predict the next possible item the user may expect, given the following chronological interaction history: {history_str}"
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Can you predict the next possible item that the user may expect?",
                "input": input_text,
                "output": target_sid,
            }
        )
    if sample > 0 and len(samples) > sample:
        random.seed(seed)
        samples = random.sample(samples, sample)
    return samples


def build_fusion_seq_samples(
    rows: list[tuple],
    id2sid: dict[str, str],
    id2title: dict[str, str],
    sample: int = -1,
    seed: int = 42,
    dedup: bool = False,
) -> list[dict]:
    """Build FusionSeqRecDataset samples: user history (semantic ID) -> predict next item title.

    Format consistent with sft.py FusionSeqRecDataset.
    """
    samples = []
    for history_ids, target_id in rows:
        if target_id not in id2sid or target_id not in id2title:
            continue
        target_sid = id2sid[target_id]
        target_title = id2title[target_id]
        history_sids = [id2sid.get(i) for i in history_ids]
        history_sids = [s for s in history_sids if s]
        if not history_sids:
            continue
        if dedup and history_sids and target_sid == history_sids[-1]:
            continue
        history_str = ", ".join(history_sids)
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Can you recommend the next item for the user based on their interaction history?",
                "input": f"The user has sequentially interacted with items {history_str}. Can you recommend the next item for him? Tell me the title of the item",
                "output": target_title,
            }
        )
    if sample > 0 and len(samples) > sample:
        random.seed(seed)
        samples = random.sample(samples, sample)
    return samples


def build_hisTitle2sid_seq_samples(
    rows: list[tuple],
    id2sid: dict[str, str],
    id2title: dict[str, str],
    sample: int = -1,
    seed: int = 42,
    dedup: bool = False,
) -> list[dict]:
    """Build RLSeqTitle2SidDataset format samples: input history title sequence, output sid.

    Consistent with data.py RLSeqTitle2SidDataset: Given the title sequence of user historical interactive items: "title1", "title2", ... -> sid.
    """
    samples = []
    for history_ids, target_id in rows:
        if target_id not in id2sid:
            continue
        target_sid = id2sid[target_id]
        # convert history item_ids to titles
        history_titles = []
        for i in history_ids:
            title = id2title.get(i, "")
            if title:
                history_titles.append(title)
        if not history_titles:
            continue
        # dedup: skip if target equals last item in history (consistent with RLSeqTitle2SidDataset)
        if dedup and history_ids and target_id == history_ids[-1]:
            continue
        # format consistent with RLSeqTitle2SidDataset: ", ".join([f'"{title}"' for title in history_item_title])
        inter_titles = ", ".join([f'"{t}"' for t in history_titles])
        input_text = f"Given the title sequence of user historical interactive items: {inter_titles}, can you recommend a suitable next item for the user?"
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Can you recommend a suitable next item for the user based on the title sequence of their historical interactive items?",
                "input": input_text,
                "output": target_sid,
            }
        )
    if sample > 0 and len(samples) > sample:
        random.seed(seed)
        samples = random.sample(samples, sample)
    return samples


def build_title_desc2sid_samples(
    items: dict[str, dict],
    id2sid: dict[str, str],
    sample: int = -1,
    seed: int = 42,
) -> list[dict]:
    r"""Build RLTitle2SidDataset format samples: title2sid and description2sid as separate tasks.

    Consistent with data.py RLTitle2SidDataset:
      - title2sid: prompt "Which item has the title: {title}?" -> sid
      - description2sid: prompt "An item can be described as follows: \"{desc}\". Which item is it describing?" -> sid.
    """
    title2sid = {}
    description2sid = {}

    for item_id, sid in id2sid.items():
        if item_id not in items:
            continue
        feat = items[item_id]
        title = feat.get("title", "")
        description = feat.get("description", "")

        # Handle description format (consistent with data.py RLTitle2SidDataset)
        if isinstance(description, str) and description.startswith("['") and description.endswith("']"):
            try:
                desc_list = eval(description)
                description = desc_list[0] if desc_list else description
            except Exception:
                pass
        elif isinstance(description, list):
            description = description[0] if description else ""
        elif not isinstance(description, str):
            description = ""

        # Match data.py RLTitle2SidDataset exactly: add all, no length filter
        title2sid[title or ""] = sid
        description2sid[description or ""] = sid

    samples = []

    # title2sid samples (consistent with RLTitle2SidDataset.generate_prompt)
    for title, sid in title2sid.items():
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Answer the question about item identification.",
                "input": f"Which item has the title: {title}?",
                "output": sid,
            }
        )

    # description2sid samples (consistent with RLTitle2SidDataset.generate_prompt)
    for description, sid in description2sid.items():
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Answer the question about item identification.",
                "input": f'An item can be described as follows: "{description}". Which item is it describing?',
                "output": sid,
            }
        )

    if sample > 0 and len(samples) > sample:
        random.seed(seed)
        samples = random.sample(samples, sample)
    return samples


def build_item_qa_samples(
    id2sid: dict[str, str],
    id2title: dict[str, str],
    sample: int = -1,
    seed: int = 42,
) -> list[dict]:
    """Build sid2title and title2sid QA samples, consistent with SidItemFeatDataset (title2sid dedup by title)."""
    sid2title = {}
    title2sid = {}
    for item_id, sid in id2sid.items():
        title = id2title.get(item_id, "")
        if not title or len(title) < 3:
            continue
        sid2title[sid] = title
        title2sid[title] = sid  # keep one per title, consistent with data.py
    samples = []
    for sid, title in sid2title.items():
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Answer the question about item identification.",
                "input": f'What is the title of item "{sid}"?',
                "output": title,
            }
        )
    for title, sid in title2sid.items():
        samples.append(
            {
                "system": SYSTEM_PROMPT,
                "instruction": "Answer the question about item identification.",
                "input": f"Which item has the title: {title}?",
                "output": sid,
            }
        )
    if sample > 0 and len(samples) > sample:
        random.seed(seed)
        samples = random.sample(samples, sample)
    return samples


def main(
    data_dir: str = "./data/Amazon18",
    category: str = "Industrial_and_Scientific",
    output_dir: str = "data/amazon_industry",
    only_task1: bool = False,
    only_task2: bool = False,
    only_task3: bool = False,
    only_task4: bool = False,
    only_task5: bool = False,
    seq_sample: int = 10000,
    seed: int = 42,
    data_source: str = "",
):
    """Build LlamaFactory dataset (with semantic ID) from data_dir/category (e.g. Amazon18/Industrial_and_Scientific)."""
    if not data_source:
        data_source = category
    # Task logic: all enabled by default; if any only_taskN is True, only that task runs
    only_specified = only_task1 or only_task2 or only_task3 or only_task4 or only_task5
    task1_sid_sft = only_task1 if only_specified else True
    task2_sid_item_feat = only_task2 if only_specified else True
    task3_fusion_seq = only_task3 if only_specified else True
    task4_hisTitle2sid = only_task4 if only_specified else True
    task5_title_desc2sid = only_task5 if only_specified else True

    category_dir = os.path.join(os.path.abspath(data_dir), category)
    item_path = os.path.join(category_dir, f"{category}.item.json")
    train_path = os.path.join(category_dir, f"{category}.train.inter")
    valid_path = os.path.join(category_dir, f"{category}.valid.inter")
    test_path = os.path.join(category_dir, f"{category}.test.inter")

    for p in [item_path, train_path, valid_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    index_path = os.path.join(category_dir, f"{category}.index.json")
    id2title = load_items(item_path)
    id2title_full = load_items_full(item_path)  # with fallback, for task4 title2sid
    items_with_desc = load_items_with_desc(item_path)  # with title+description, for task5
    id2sid = load_index(index_path)
    index_raw = load_index_raw(index_path)  # item_id -> ["<a_*>", "<b_*>", "<c_*>"]
    print(f"Loaded {len(id2title)} items, {len(id2sid)} semantic ID mappings")

    new_tokens = extract_new_tokens(index_path)
    os.makedirs(output_dir, exist_ok=True)
    new_tokens_path = os.path.join(output_dir, "new_tokens.json")
    with open(new_tokens_path, "w", encoding="utf-8") as f:
        json.dump(new_tokens, f, indent=2)
    print(f"Saved {len(new_tokens)} new tokens to {new_tokens_path}")

    id2sid_path = os.path.join(output_dir, "id2sid.json")
    with open(id2sid_path, "w", encoding="utf-8") as f:
        json.dump(index_raw, f, ensure_ascii=False, indent=4)
    print(f"Saved id2sid (item_id -> [sid1, sid2, sid3]) to {id2sid_path}")

    train_rows = parse_inter_file(train_path)
    valid_rows = parse_inter_file(valid_path)
    test_rows = parse_inter_file(test_path)
    print(f"Parsed train: {len(train_rows)}, valid: {len(valid_rows)}, test: {len(test_rows)}")

    # sft: task1, task2, task3 -> amazon_industry/sft
    sft_train, sft_valid, sft_test = [], [], []
    # rl: task1, task4, task5 -> amazon_industry/rl (format: data_source, prompt, ability, reward_model, extra_info)
    rl_train, rl_valid, rl_test = [], [], []

    if task1_sid_sft:
        train_seq = build_seq_samples(train_rows, id2sid, sample=-1, seed=seed)
        valid_seq = build_seq_samples(valid_rows, id2sid, sample=-1, seed=seed)
        test_seq = build_seq_samples(test_rows, id2sid, sample=-1, seed=seed, is_test=True)
        sft_train.extend(train_seq)
        sft_valid.extend(valid_seq)
        sft_test.extend(test_seq)
        # RL format
        for i, s in enumerate(train_seq):
            rl_train.append(to_rl_format(s, data_source, "seq_rec", "train", i, "task1_sid_sft"))
        for i, s in enumerate(valid_seq):
            rl_valid.append(to_rl_format(s, data_source, "seq_rec", "valid", i, "task1_sid_sft"))
        for i, s in enumerate(test_seq):
            rl_test.append(to_rl_format(s, data_source, "seq_rec", "test", i, "task1_sid_sft"))
        print(
            f"Task1 SidSFT: train={len(train_seq)}, valid={len(valid_seq)}, test={len(test_seq)}, (sft, rl, train, valid, test)"
        )

    if task2_sid_item_feat:
        qa = build_item_qa_samples(id2sid, id2title, sample=-1, seed=seed)
        sft_train.extend(qa)
        print(f"Task2 SidItemFeat: {len(qa)} samples (sft, train only)")

    if task3_fusion_seq:
        # FusionSeqRecDataset: history sids -> predict next title (from train sequences)
        fusion_train = build_fusion_seq_samples(train_rows, id2sid, id2title, sample=-1, seed=seed)
        sft_train.extend(fusion_train)
        print(f"Task3 FusionSeqRec: train={len(fusion_train)} (sft, history_sids->title)")

    if task4_hisTitle2sid:
        # RLSeqTitle2SidDataset: input history title, output sid
        hisTitle2sid_train = build_hisTitle2sid_seq_samples(
            train_rows, id2sid, id2title_full, sample=seq_sample, seed=seed
        )
        for i, s in enumerate(hisTitle2sid_train):
            rl_train.append(to_rl_format(s, data_source, "seq_title2sid", "train", i, "task4_hisTitle2sid"))
        print(f"Task4 Title2Sid: train={len(hisTitle2sid_train)} (rl, train only)")

    if task5_title_desc2sid:
        # RLTitle2SidDataset: title2sid + description2sid (separate tasks), output sid
        title_desc2sid = build_title_desc2sid_samples(items_with_desc, id2sid, sample=-1, seed=seed)
        for i, s in enumerate(title_desc2sid):
            rl_train.append(to_rl_format(s, data_source, "title_desc2sid", "train", i, "task5_title_desc2sid"))
        print(f"Task5 TitleDesc2Sid: {len(title_desc2sid)} samples (rl, title2sid+desc2sid)")

    # Save sft: task1, task2, task3 -> {output_dir}/sft/
    sft_dir = os.path.join(output_dir, "sft")
    if sft_train or sft_valid or sft_test:
        random.seed(seed)
        random.shuffle(sft_train)
        random.shuffle(sft_valid)
        if sft_test:
            random.shuffle(sft_test)
        os.makedirs(sft_dir, exist_ok=True)
        for name, data in [("train", sft_train), ("valid", sft_valid), ("test", sft_test)]:
            if data:
                out = os.path.join(sft_dir, f"{name}.json")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Saved {len(data)} samples to {out}")

    # Save rl: task1, task4, task5 -> {output_dir}/rl/
    rl_dir = os.path.join(output_dir, "rl")
    if rl_train or rl_valid or rl_test:
        random.seed(seed)
        random.shuffle(rl_train)
        random.shuffle(rl_valid)
        if rl_test:
            random.shuffle(rl_test)
        os.makedirs(rl_dir, exist_ok=True)
        for name, data in [("train", rl_train), ("valid", rl_valid), ("test", rl_test)]:
            if data:
                out = os.path.join(rl_dir, f"{name}.json")
                with open(out, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"Saved {len(data)} samples to {out}")

    print(f"\nDone! SFT dataset -> {sft_dir}, RL dataset -> {rl_dir}")


if __name__ == "__main__":
    fire.Fire(main)
