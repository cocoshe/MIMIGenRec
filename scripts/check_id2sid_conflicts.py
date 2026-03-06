#!/usr/bin/env python3
"""Check for SID conflicts in id2sid.json.

Definition here:
- id2sid.json: { "item_id": ["<a_x>", "<b_y>", "<c_z>", ...], ... }
- We define sid = "<a_x><b_y><c_z>" (first three tokens joined).
- Conflict = the same sid maps to more than one item_id.
"""

import argparse
import json
import os
from collections import defaultdict


def main(dataset: str, data_dir: str = "data") -> None:
    """Check id2sid conflicts for the given dataset.

    Args:
        dataset: Dataset name (e.g. Office_Products, Industrial_and_Scientific).
        data_dir: Base directory containing dataset folders (default: data).
    """
    file_path = os.path.join(data_dir, dataset, "id2sid.json")

    if not os.path.isfile(file_path):
        print(f"File not found: {file_path}", flush=True)
        return

    with open(file_path, encoding="utf-8") as f:
        id2sid_raw = json.load(f)

    # sid -> list[item_id]
    sid_to_ids = defaultdict(list)

    for item_id, tokens in id2sid_raw.items():
        if not isinstance(tokens, list) or len(tokens) < 3:
            continue
        sid = "".join(tokens[:3])
        sid_to_ids[sid].append(item_id)

    conflicts = {sid: ids for sid, ids in sid_to_ids.items() if len(set(ids)) > 1}

    total_items = len(id2sid_raw)
    total_sids = len(sid_to_ids)
    num_conflict_sids = len(conflicts)

    # Conflict rate: by sid (what fraction of sids have conflict)
    sid_conflict_rate = (num_conflict_sids / total_sids * 100) if total_sids else 0.0
    # Items involved in conflict (item_id whose sid maps to >1 item)
    items_in_conflict = set()
    for ids in conflicts.values():
        items_in_conflict.update(ids)
    item_conflict_rate = (len(items_in_conflict) / total_items * 100) if total_items else 0.0

    def out(msg: str = "") -> None:
        print(msg, flush=True)

    out(f"File: {file_path}")
    out(f"Total items (keys in id2sid): {total_items}")
    out(f"Unique sids (by <a_><b_><c_>): {total_sids}")
    out(f"Sids with conflicts (mapped to >1 item_id): {num_conflict_sids}")
    out(f"Conflict rate (by sid): {sid_conflict_rate:.2f}%")
    out(f"Conflict rate (by item): {item_conflict_rate:.2f}%  (items involved in conflict: {len(items_in_conflict)})")
    out()

    if not conflicts:
        out("No sid conflicts found.")
        return

    out("Conflict details (sid -> item_ids):")
    out("-" * 80)
    # Print at most first 100 conflict entries to avoid huge output
    for idx, (sid, ids) in enumerate(sorted(conflicts.items()), start=1):
        if idx > 100:
            out(f"... ({len(conflicts) - 100} more conflicts not shown)")
            break
        uniq_ids = sorted(set(ids), key=lambda x: int(x))
        out(f"{idx}. sid: {sid}")
        out(f"   item_ids: {uniq_ids}")
        out()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check for SID conflicts in id2sid.json for a dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Industrial_and_Scientific",
        help="Dataset name (e.g. Office_Products, Industrial_and_Scientific).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base directory containing dataset folders (default: data).",
    )
    args = parser.parse_args()
    main(dataset=args.dataset, data_dir=args.data_dir)
