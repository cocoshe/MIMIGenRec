"""Split JSON test data by GPU count for multi-GPU parallel evaluation."""

import json
import os

import fire


def split(input_path: str, output_path: str, cuda_list: str):
    """Split JSON test data.

    Args:
        input_path: Input JSON file path
        output_path: Output directory
        cuda_list: GPU list, e.g. "0,1,2,3" or "0 1 2 3"
    """
    if isinstance(cuda_list, str):
        cuda_list = [x.strip() for x in cuda_list.replace(",", " ").split() if x.strip()]

    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(output_path, exist_ok=True)
    n = len(data)
    num_parts = len(cuda_list)

    for i, gpu_id in enumerate(cuda_list):
        start = i * n // num_parts
        end = (i + 1) * n // num_parts
        part = data[start:end]
        out_file = os.path.join(output_path, f"{gpu_id}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(part, f, indent=4, ensure_ascii=False)
        print(f"GPU {gpu_id}: {len(part)} samples -> {out_file}")


if __name__ == "__main__":
    fire.Fire(split)
