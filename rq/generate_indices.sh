# -------- [Industrial_and_Scientific] --------
# python rq/generate_indices.py \
#     --dataset Industrial_and_Scientific \
#     --ckpt_path output/Industrial_and_Scientific/best_collision_model.pth \
#     --output_dir data/Amazon18/Industrial_and_Scientific \
#     --device cuda:0

# -------- [Office_Products] --------
# python rq/generate_indices.py \
#     --dataset Office_Products \
#     --ckpt_path output/Office_Products/best_collision_model.pth \
#     --output_dir data/Amazon18/Office_Products \
#     --device cuda:0

# -------- [Toys_and_Games] --------
python rq/generate_indices.py \
    --dataset Toys_and_Games \
    --ckpt_path output/Toys_and_Games/best_collision_model.pth \
    --output_dir data/Amazon18/Toys_and_Games \
    --device cuda:0
