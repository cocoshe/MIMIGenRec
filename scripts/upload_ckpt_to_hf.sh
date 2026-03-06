#!/usr/bin/env bash
# Before: hf auth login
# Set user: export HF_USER=your_username

set -e
HF_USER="${HF_USER:-}"

if [[ -z "$HF_USER" ]]; then
  echo "Please set HF_USER (e.g. export HF_USER=your_username)"
  exit 1
fi


# ----------------------- SFT -----------------------
# [Industrial_and_Scientific] qwen2.5-0.5b-industrial-scientific-sft-dsz0 SFT
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "saves/qwen2.5-0.5b/full/Industrial_and_Scientific-sft-dsz0" \
  --repo_id "${HF_USER}/qwen2.5-0.5b-industrial-scientific-sft-dsz0"

# [Office_Products] qwen2.5-0.5b-office-products-sft-dsz0 SFT
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "saves/qwen2.5-0.5b/full/Office_Products-sft-dsz0/checkpoint-420" \
  --repo_id "${HF_USER}/qwen2.5-0.5b-office-products-sft-dsz0"

# [Toys_and_Games] Toys_and_Games-qwen2.5-0.5b-sft-dsz0 SFT
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "saves/qwen2.5-0.5b/full/Toys_and_Games-sft-dsz0" \
  --repo_id "${HF_USER}/Toys_and_Games-qwen2.5-0.5b-sft-dsz0"

# [Toys_and_Games] Toys_and_Games-qwen2.5-1.5b-sft-dsz2 SFT
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "saves/qwen2.5-1.5b/full/Toys_and_Games-sft-dsz2" \
  --repo_id "${HF_USER}/Toys_and_Games-qwen2.5-1.5b-sft-dsz2"

# ----------------------- GRPO -----------------------
# [Industrial_and_Scientific] Industrial_and_Scientific-qwen2.5-0.5b-instruct-grpo GRPO
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "rl_outputs/Industrial_and_Scientific-qwen2.5-0.5b-instruct-grpo" \
  --repo_id "${HF_USER}/Industrial_and_Scientific-qwen2.5-0.5b-instruct-grpo"

# [Office_Products] Office_Products-qwen2.5-0.5b-instruct-grpo GRPO
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "rl_outputs/Office_Products-qwen2.5-0.5b-instruct-grpo" \
  --repo_id "${HF_USER}/Office_Products-qwen2.5-0.5b-instruct-grpo"

# [Toys_and_Games] Toys_and_Games-qwen2.5-0.5b-instruct-grpo GRPO
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "rl_outputs/Toys_and_Games-qwen2.5-0.5b-instruct-grpo" \
  --repo_id "${HF_USER}/Toys_and_Games-qwen2.5-0.5b-instruct-grpo"

# [Toys_and_Games] Toys_and_Games-qwen2.5-1.5b-instruct-grpo GRPO
python scripts/upload_ckpt_to_hf.py --user "$HF_USER" \
  --ckpt "rl_outputs/Toys_and_Games-qwen2.5-1.5b-instruct-grpo" \
  --repo_id "${HF_USER}/Toys_and_Games-qwen2.5-1.5b-instruct-grpo"
