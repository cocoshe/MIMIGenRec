# -------- [Industrial_and_Scientific] --------
# python rq/rqvae.py \
#       --data_path data/Amazon18/Industrial_and_Scientific/Office_Products.emb-qwen-td.npy \
#       --ckpt_dir ./output/Industrial_and_Scientific \
#       --lr 1e-3 \
#       --epochs 10000 \
#       --batch_size 20480

# -------- [Office_Products] --------
# python rq/rqvae.py \
#       --data_path data/Amazon18/Office_Products/Office_Products.emb-qwen-td.npy \
#       --ckpt_dir ./output/Office_Products \
#       --lr 1e-3 \
#       --epochs 10000 \
#       --batch_size 20480

# -------- [Toys_and_Games] --------
python rq/rqvae.py \
      --data_path data/Amazon18/Toys_and_Games/Toys_and_Games.emb-qwen-td.npy \
      --ckpt_dir ./output/Toys_and_Games \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480
