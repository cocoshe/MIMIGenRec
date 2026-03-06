import argparse
import copy
import json
import os
import random
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .caser import Caser
from .dataset import RecDataset
from .gru import GRU
from .industrial_data import load_industrial_scientific
from .sasrec import SASRec


def parse_args():
    p = argparse.ArgumentParser(description="Rec zoo: sequential recommendation (SASRec, GRU, Caser)")
    p.add_argument("--data_dir", type=str, default="data/Industrial_and_Scientific")
    p.add_argument("--split", type=str, default="sft", choices=["rl", "sft"])
    p.add_argument("--seq_size", type=int, default=10)
    p.add_argument("--epoch", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden_factor", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--l2_decay", type=float, default=1e-5)
    p.add_argument("--dropout_rate", type=float, default=0.3)
    p.add_argument("--loss_type", type=str, default="bce", choices=["bce", "ce"])
    p.add_argument("--model", type=str, default="SASRec", choices=["SASRec", "GRU", "Caser"])
    p.add_argument("--num_filters", type=int, default=16)
    p.add_argument("--filter_sizes", type=str, default="[2,3,4]")
    p.add_argument("--early_stop", type=int, default=20)
    p.add_argument("--eval_num", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--save_dir", type=str, default="")
    p.add_argument("--result_json", type=str, default=None)
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, samples, device, topk, pad_id):
    model.eval()
    hit_all = [0.0] * len(topk)
    ndcg_all = [0.0] * len(topk)
    bs = 1024
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        seq = torch.LongTensor([s[0] for s in batch]).to(device)
        len_seq = torch.LongTensor([max(1, sum(1 for x in s[0] if x != pad_id)) for s in batch]).to(device)
        target = torch.LongTensor([s[1] for s in batch]).to(device)
        with torch.no_grad():
            pred = model.forward_eval(seq, len_seq)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        rank_list = pred.shape[1] - 1 - torch.argsort(torch.argsort(pred))
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1)
        ndcg_full = 1.0 / torch.log2(target_rank + 2)
        for j, k in enumerate(topk):
            mask = (target_rank < k).float()
            hit_all[j] += mask.sum().cpu().item()
            ndcg_all[j] += (ndcg_full * mask).sum().cpu().item()
    n = len(samples)
    return [hit_all[j] / n for j in range(len(topk))], [ndcg_all[j] / n for j in range(len(topk))]


def format_metrics(hr_list, ndcg_list, topk, prefix=""):
    """Format HR/NDCG for each topk as a single print string."""
    parts = []
    for i, k in enumerate(topk):
        parts.append(f"HR@{k}={hr_list[i]:.4f}")
    for i, k in enumerate(topk):
        parts.append(f"NDCG@{k}={ndcg_list[i]:.4f}")
    return (prefix + " " if prefix else "") + "  ".join(parts)


def evaluate_with_predictions(model, samples, device, topk, pad_id, id2sid, top_predict=50):
    """Run model on test set, compute HR/NDCG for topk, and build per-sample predict list for final_result.json."""
    model.eval()
    hit_all = [0.0] * len(topk)
    ndcg_all = [0.0] * len(topk)
    final_result_list = []
    bs = 1024
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        seq = torch.LongTensor([s[0] for s in batch]).to(device)
        len_seq = torch.LongTensor([max(1, sum(1 for x in s[0] if x != pad_id)) for s in batch]).to(device)
        target = torch.LongTensor([s[1] for s in batch]).to(device)
        with torch.no_grad():
            pred = model.forward_eval(seq, len_seq)
        if pred.dim() == 1:
            pred = pred.unsqueeze(0)
        rank_list = pred.shape[1] - 1 - torch.argsort(torch.argsort(pred), dim=1)
        target_rank = torch.gather(rank_list, 1, target.view(-1, 1)).view(-1)
        ndcg_full = 1.0 / torch.log2(target_rank + 2)
        for j, k in enumerate(topk):
            mask = (target_rank < k).float()
            hit_all[j] += mask.sum().cpu().item()
            ndcg_all[j] += (ndcg_full * mask).sum().cpu().item()
        for b in range(len(batch)):
            seq_ids, next_id = batch[b][0], batch[b][1]
            order = torch.argsort(pred[b], descending=True).cpu().tolist()
            predict_sids = [id2sid.get(idx, "") for idx in order[:top_predict]]
            history_sids = [id2sid[x] for x in seq_ids if x != pad_id and x in id2sid]
            input_str = ", ".join(history_sids) if history_sids else ""
            final_result_list.append(
                {
                    "input": input_str,
                    "output": id2sid.get(next_id, ""),
                    "predict": predict_sids,
                }
            )
    n = len(samples)
    hr_list = [hit_all[j] / n for j in range(len(topk))]
    ndcg_list = [ndcg_all[j] / n for j in range(len(topk))]
    return hr_list, ndcg_list, final_result_list


def main():
    args = parse_args()
    if not args.save_dir:
        data_name = os.path.basename(os.path.normpath(args.data_dir))
        args.save_dir = f"experiments/{args.model}_{data_name}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    setup_seed(args.seed)

    if args.result_json is None:
        args.result_json = os.path.join(args.save_dir, "result.json")

    train_samples, valid_samples, test_samples, item_num, id2sid = load_industrial_scientific(
        args.data_dir, seq_size=args.seq_size, split=args.split
    )
    seq_size = args.seq_size
    pad_id = item_num
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"item_num={item_num} train={len(train_samples)} valid={len(valid_samples)} test={len(test_samples)}")

    if args.model == "SASRec":
        model = SASRec(args.hidden_factor, item_num, seq_size, args.dropout_rate, device)
    elif args.model == "GRU":
        model = GRU(args.hidden_factor, item_num, seq_size)
    else:
        model = Caser(
            args.hidden_factor,
            item_num,
            seq_size,
            args.num_filters,
            args.filter_sizes,
            args.dropout_rate,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-8, weight_decay=args.l2_decay)
    criterion = nn.BCEWithLogitsLoss() if args.loss_type == "bce" else nn.CrossEntropyLoss()
    model.to(device)
    train_dataset = RecDataset(train_samples, pad_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    freq = Counter(s[1] for s in train_samples)
    ps = np.array([freq.get(i, 0) + 1 for i in range(item_num)], dtype=np.float64)
    ps = np.power(ps / np.sum(ps), 0.05)
    ps = torch.tensor(ps, dtype=torch.float32).to(device)

    topk = [1, 3, 5, 10, 20]
    num_batches = len(train_loader)
    ndcg_max, best_epoch, early_stop = 0.0, 0, 0
    best_model, best_hr, best_ndcg = None, None, None
    train_log = []

    for epoch in range(args.epoch):
        model.train()
        for batch_idx, (seq, len_seq, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            target_neg = []
            for t in target:
                neg = np.random.randint(item_num)
                while neg == t.item():
                    neg = np.random.randint(item_num)
                target_neg.append(neg)

            seq = seq.to(device)
            len_seq = len_seq.to(device)
            target = target.to(device)
            target_neg = torch.LongTensor(target_neg).to(device)

            if args.model == "GRU":
                len_seq = len_seq.cpu()

            optimizer.zero_grad()
            out = model.forward(seq, len_seq)

            target_1 = target.view(-1, 1)
            target_neg_1 = target_neg.view(-1, 1)
            pos_scores = torch.gather(out, 1, target_1)
            neg_scores = torch.gather(out, 1, target_neg_1)
            scores = torch.cat((pos_scores, neg_scores), 0)
            labels = torch.cat((torch.ones_like(pos_scores), torch.zeros_like(neg_scores)), 0)

            if args.loss_type == "bce":
                loss = criterion(scores, labels)
            else:
                loss = criterion(out, target)

            loss.backward()
            optimizer.step()

            if (batch_idx + 1) % max(1, num_batches * args.eval_num) == 0 or (batch_idx + 1) == num_batches:
                val_hr, val_ndcg = evaluate(model, valid_samples, device, topk, pad_id)
                test_hr, test_ndcg = evaluate(model, test_samples, device, topk, pad_id)
                ndcg_last = val_ndcg[-1]
                print("  Val: " + format_metrics(val_hr, val_ndcg, topk))
                log_entry = {
                    "epoch": epoch,
                    "val_HR": {k: val_hr[i] for i, k in enumerate(topk)},
                    "val_NDCG": {k: val_ndcg[i] for i, k in enumerate(topk)},
                }
                if ndcg_last > ndcg_max:
                    ndcg_max, best_epoch, early_stop = ndcg_last, epoch, 0
                    best_hr, best_ndcg = test_hr, test_ndcg
                    best_model = copy.deepcopy(model)
                    log_entry["is_best"] = True
                    log_entry["best_val_NDCG20"] = ndcg_last
                else:
                    early_stop += 1
                    log_entry["is_best"] = False
                    if early_stop >= args.early_stop:
                        print(f"Early stop at epoch {epoch}")
                        break
                train_log.append(log_entry)
                print(f"  best_epoch={best_epoch} early_stop={early_stop}")

        if early_stop >= args.early_stop:
            break

    if best_model is None:
        best_model = model
        best_hr, best_ndcg = evaluate(best_model, test_samples, device, topk, pad_id)
        best_epoch = args.epoch - 1

    topk_tsv = [1, 3, 5, 10, 20, 50]
    hr_tsv, ndcg_tsv, final_result_list = evaluate_with_predictions(
        best_model, test_samples, device, topk_tsv, pad_id, id2sid, top_predict=50
    )

    os.makedirs(os.path.dirname(args.result_json) or ".", exist_ok=True)
    result = {
        "best_epoch": best_epoch,
        "best_from_val_NDCG20": ndcg_max,
        "NDCG": {k: ndcg_tsv[i] for i, k in enumerate(topk_tsv)},
        "HR": {k: hr_tsv[i] for i, k in enumerate(topk_tsv)},
    }
    with open(args.result_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "train_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2, ensure_ascii=False)
    metrics_tsv_path = os.path.join(args.save_dir, "metrics.tsv")
    path_name = os.path.basename(os.path.normpath(args.save_dir))
    tsv_header = "path\t" + "\t".join(f"HR@{k}" for k in topk_tsv) + "\t" + "\t".join(f"NDCG@{k}" for k in topk_tsv)
    tsv_row = (
        path_name
        + "\t"
        + "\t".join(f"{hr_tsv[j]:.4f}" for j in range(len(topk_tsv)))
        + "\t"
        + "\t".join(f"{ndcg_tsv[j]:.4f}" for j in range(len(topk_tsv)))
    )
    with open(metrics_tsv_path, "w", encoding="utf-8") as f:
        f.write(tsv_header + "\n")
        f.write(tsv_row + "\n")
    final_result_path = os.path.join(args.save_dir, "final_result.json")
    with open(final_result_path, "w", encoding="utf-8") as f:
        json.dump(final_result_list, f, indent=4, ensure_ascii=False)
    ckpt_path = os.path.join(args.save_dir, "best_state.pth")
    torch.save(best_model.state_dict(), ckpt_path)
    meta_path = os.path.join(args.save_dir, "best_ckpt_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_val_NDCG20": ndcg_max,
                "test_HR": result["HR"],
                "test_NDCG": result["NDCG"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {args.save_dir}")
    print(f"metrics.tsv: {metrics_tsv_path}")
    print(f"final_result.json: {final_result_path}")
    print(f"Best ckpt from epoch {best_epoch} (val NDCG@20={ndcg_max:.4f})")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
