# -------- [Industrial_and_Scientific] --------
# accelerate launch --num_processes 8 amazon_text2emb.py \
#     --dataset Industrial_and_Scientific \
#     --root data/Amazon18/Industrial_and_Scientific \
#     --plm_checkpoint Qwen/Qwen3-Embedding-4B

# -------- [Office_Products] --------
# accelerate launch --num_processes 8 rq/text2emb/amazon_text2emb.py \
#     --dataset Office_Products \
#     --root data/Amazon18/Office_Products \
#     --plm_checkpoint Qwen/Qwen3-Embedding-4B

# -------- [Toys_and_Games] --------
accelerate launch --num_processes 8 rq/text2emb/amazon_text2emb.py \
    --dataset Toys_and_Games \
    --root data/Amazon18/Toys_and_Games \
    --plm_checkpoint Qwen/Qwen3-Embedding-4B
