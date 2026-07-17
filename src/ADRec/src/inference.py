"""Evaluate a saved final ADRec checkpoint without retraining."""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR.parents[1]))

from experiment_tools.experiment_tracking import (  # noqa: E402
    ExperimentTracker,
    checkpoint_path,
    recommendation_popularity,
    save_dataset_popularity,
)
from evaluate_topk_dp import compute_all_metrics  # noqa: E402
from trainer import choose_model, downvote_seen_items  # noqa: E402
from utils import Data_Test, fix_random_seed_as  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m", choices=["ml-1m", "baby", "toys"])
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--metric_ks", nargs="+", type=int, default=[10, 20, 100])
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def final_train_sequences(data_raw: dict) -> list[list[int]]:
    sequences = []
    for uid, train_sequence in data_raw["train_dict"].items():
        sequence = list(train_sequence)
        if uid in data_raw["val_seq_dict"] and uid in data_raw["val_tgt_dict"]:
            sequence += list(data_raw["val_seq_dict"][uid])
            sequence.append(data_raw["val_tgt_dict"][uid])
        sequences.append(sequence)
    return sequences


def main() -> None:
    cli = parse_args()
    device = cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = (cli.checkpoint or checkpoint_path(
        "ADRec", cli.dataset, cli.max_len, cli.random_seed, ".pth"
    )).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    payload = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload or "args" not in payload:
        raise ValueError(f"Unsupported ADRec checkpoint format: {model_path}")

    args = argparse.Namespace(**payload["args"])
    args.dataset = cli.dataset
    args.max_len = cli.max_len
    args.random_seed = cli.random_seed
    args.metric_ks = cli.metric_ks
    args.device = device
    args.mask_seen = True
    args.pretrained = False
    args.freeze_emb = False
    fix_random_seed_as(args.random_seed)

    os.chdir(SOURCE_DIR)
    with (SOURCE_DIR.parent / "datasets" / "data" / args.dataset / "dataset.pkl").open("rb") as handle:
        data_raw = pickle.load(handle)

    args.item_num = data_raw["item_count"]
    train_combined = final_train_sequences(data_raw)
    args.coverage_candidate_items = {
        item for sequence in train_combined for item in sequence
    }
    args.train_item_popularity = dict(
        Counter(item for sequence in train_combined for item in sequence)
    )
    save_dataset_popularity(args.dataset, args.train_item_popularity)

    model = choose_model(args)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    test_data = Data_Test(
        data_raw["test_seq"],
        [[] for _ in data_raw["test_tgt"]],
        data_raw["test_tgt"],
        args,
    )
    test_loader = test_data.get_pytorch_dataloaders()

    tracker = ExperimentTracker(args.dataset, "ADRec")
    all_actual, all_predicted = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="ADRec inference"):
            batch = [tensor.to(device) for tensor in batch]
            _, last_item, *_ = model(batch[0], batch[1], train_flag=False)
            scores = model.calculate_score(last_item)
            if args.mask_seen:
                scores = downvote_seen_items(scores, batch[0])
            topk = torch.topk(scores, k=max(args.metric_ks), dim=-1).indices
            for index in range(len(batch[1])):
                all_actual.append([batch[1][index, -1].item()])
                all_predicted.append(topk[index].cpu().tolist())
    inference_seconds = time.perf_counter() - started

    _, recalls, ndcgs, mrrs, coverages = compute_all_metrics(
        all_actual,
        all_predicted,
        args.metric_ks,
        len(args.coverage_candidate_items),
        candidate_items=args.coverage_candidate_items,
    )
    metrics = {
        k: {"recall": recall, "ndcg": ndcg, "mrr": mrr, "coverage": coverage}
        for k, recall, ndcg, mrr, coverage in zip(
            args.metric_ks, recalls, ndcgs, mrrs, coverages
        )
    }
    pd.DataFrame({
        "user_id": range(len(all_predicted)),
        "recommendations": all_predicted,
    }).to_csv(tracker.run_dir / "recommendations.csv", index=False)
    tracker.log_final_metrics(
        metrics,
        split="global_temporal_70_10_20",
        mask_seen=True,
        seed=args.random_seed,
        inference_total_sec=inference_seconds,
        n_users=len(all_actual),
        maxlen=args.max_len,
        checkpoint=str(model_path),
        popularity_bias=recommendation_popularity(
            all_predicted, args.train_item_popularity, args.metric_ks
        ),
    )
    tracker.close()

    print(f"Loaded checkpoint: {model_path}")
    print(f"Inference time: {inference_seconds:.2f}s")
    for k in args.metric_ks:
        values = metrics[k]
        print(
            f"k={k}: Recall={values['recall']:.6f}, NDCG={values['ndcg']:.6f}, "
            f"MRR={values['mrr']:.6f}, Coverage={values['coverage']:.6f}"
        )
    print(f"Results saved to: {tracker.run_dir}")


if __name__ == "__main__":
    main()
