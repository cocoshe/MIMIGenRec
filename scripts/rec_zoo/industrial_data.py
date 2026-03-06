"""Load Industrial_and_Scientific for SASRec."""

import json
import os
import re


SID_PATTERN = re.compile(r"<a_\d+><b_\d+><c_\d+>")


def load_id2sid(id2sid_path: str) -> dict[str, list[str]]:
    with open(id2sid_path, encoding="utf-8") as f:
        return json.load(f)


def build_sid2id(id2sid_raw: dict[str, list[str]]) -> dict[str, int]:
    sid2id = {}
    for item_id, tokens in id2sid_raw.items():
        if len(tokens) >= 3:
            sid = "".join(tokens[:3])
            sid2id[sid] = int(item_id)
    return sid2id


def extract_sids_from_text(text: str) -> list[str]:
    return SID_PATTERN.findall(text)


def parse_rl_sample(item: dict, sid2id: dict[str, int], pad_id: int, seq_size: int):
    if item.get("ability") != "seq_rec":
        return None
    prompt = item.get("prompt") or []
    content = ""
    for p in prompt:
        if p.get("role") == "user":
            content = p.get("content", "")
            break
    ground_truth = (item.get("reward_model") or {}).get("ground_truth", "").strip()
    if not content or not ground_truth:
        return None
    sids = extract_sids_from_text(content)
    if not sids:
        return None
    next_sid = SID_PATTERN.findall(ground_truth)
    if not next_sid:
        return None
    next_sid = next_sid[0]
    next_id = sid2id.get(next_sid)
    if next_id is None:
        return None
    seq_ids = [sid2id[s] for s in sids if sid2id.get(s) is not None]
    if not seq_ids:
        return None
    if len(seq_ids) > seq_size:
        seq_ids = seq_ids[-seq_size:]
    else:
        seq_ids = seq_ids + [pad_id] * (seq_size - len(seq_ids))
    return (seq_ids, next_id)


def load_rl_json(json_path: str, sid2id: dict[str, int], item_num: int, seq_size: int):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pad_id = item_num
    return [t for item in data for t in [parse_rl_sample(item, sid2id, pad_id, seq_size)] if t is not None]


def load_sft_json(json_path: str, sid2id: dict[str, int], item_num: int, seq_size: int):
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    pad_id = item_num
    samples = []
    for item in data:
        inp = item.get("input", "")
        out = (item.get("output") or "").strip()
        sids = extract_sids_from_text(inp)
        next_sid = SID_PATTERN.findall(out)
        if not sids or not next_sid:
            continue
        next_id = sid2id.get(next_sid[0])
        if next_id is None:
            continue
        seq_ids = [sid2id[s] for s in sids if sid2id.get(s) is not None]
        if not seq_ids:
            continue
        if len(seq_ids) > seq_size:
            seq_ids = seq_ids[-seq_size:]
        else:
            seq_ids = seq_ids + [pad_id] * (seq_size - len(seq_ids))
        samples.append((seq_ids, next_id))
    return samples


def build_id2sid(id2sid_raw: dict[str, list[str]]) -> dict[int, str]:
    """Build numeric id -> full sid string (e.g. '<a_20><b_188><c_134>')."""
    return {int(item_id): "".join(tokens[:3]) for item_id, tokens in id2sid_raw.items() if len(tokens) >= 3}


def load_industrial_scientific(data_dir: str, seq_size: int = 10, split: str = "rl"):
    id2sid_path = os.path.join(data_dir, "id2sid.json")
    id2sid_raw = load_id2sid(id2sid_path)
    sid2id = build_sid2id(id2sid_raw)
    id2sid = build_id2sid(id2sid_raw)
    item_num = len(id2sid_raw)
    if split == "rl":
        loader = load_rl_json
        train_path = os.path.join(data_dir, "rl", "train.json")
        valid_path = os.path.join(data_dir, "rl", "valid.json")
        test_path = os.path.join(data_dir, "rl", "test.json")
    else:
        loader = load_sft_json
        train_path = os.path.join(data_dir, "sft", "train.json")
        valid_path = os.path.join(data_dir, "sft", "valid.json")
        test_path = os.path.join(data_dir, "sft", "test.json")
    return (
        loader(train_path, sid2id, item_num, seq_size),
        loader(valid_path, sid2id, item_num, seq_size),
        loader(test_path, sid2id, item_num, seq_size),
        item_num,
        id2sid,
    )
