from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR.parent))

from experiment_tools.experiment_tracking import (  # noqa: E402
    ExperimentTracker,
    checkpoint_path,
    recommendation_popularity,
    save_dataset_popularity,
)
from research_buckets.evaluate_buckets import (  # noqa: E402
    evaluate_bucketed_hr,
    print_bucketed_hr,
)
from research_buckets.popularity_buckets import build_popularity_buckets  # noqa: E402
from load_evaluate_pipeline import (  # noqa: E402
    prepare_data_and_description,
    run_inference_pipeline,
)
from model import load_sasrec_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='ml-1m',
        choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty',
                 'amazon_Sports_and_Outdoors', 'amazon_Toys_and_Games'],
    )
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20, 100])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--checkpoint', type=Path, default=None)
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return requested


def main() -> None:
    cli = parse_args()
    random.seed(cli.random_seed)
    np.random.seed(cli.random_seed)
    torch.manual_seed(cli.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.random_seed)
    torch.backends.cudnn.deterministic = True

    device = select_device(cli.device)
    model_path = (
        cli.checkpoint
        or checkpoint_path('SASRec', cli.dataset, cli.maxlen, cli.random_seed)
    ).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')

    os.chdir(SOURCE_DIR)
    model, config, _, _ = load_sasrec_model(model_path, device=device)
    if int(config['maxlen']) != cli.maxlen:
        raise ValueError(
            f'Checkpoint maxlen={config["maxlen"]} does not match --maxlen={cli.maxlen}'
        )

    (_, _, _, test_examples, _, data_description, userid_col, itemid_col,
     time_col, _, train_val_data) = prepare_data_and_description(cli.dataset)
    train_item_popularity = train_val_data[itemid_col].value_counts().to_dict()
    save_dataset_popularity(cli.dataset, train_item_popularity)

    recs, users, metrics, inference_seconds = run_inference_pipeline(
        model,
        history_data=train_val_data,
        train_data=train_val_data,
        test_examples=test_examples,
        data_description=data_description,
        userid_col=userid_col,
        itemid_col=itemid_col,
        time_col=time_col,
        val_seq_dict={},
        topn=max(cli.metric_ks),
        metric_ks=cli.metric_ks,
    )
    if not users:
        raise ValueError('No eligible warm-start test examples remain')

    precisions, recalls, ndcgs, mrrs, coverages = metrics
    target_by_user = test_examples.set_index(userid_col)[itemid_col].to_dict()
    targets = [[target_by_user[user]] for user in users]
    bucket_by_item = build_popularity_buckets(
        train_item_popularity, train_val_data[itemid_col].unique().tolist()
    )
    bucket_metrics = evaluate_bucketed_hr(
        targets, recs.tolist(), bucket_by_item, cli.metric_ks
    )
    print_bucketed_hr(bucket_metrics)
    tracker = ExperimentTracker(
        cli.dataset, 'SASRec', maxlen=cli.maxlen, run_type='inference'
    )
    pd.DataFrame({
        'user_id': users,
        'recommendations': [list(map(int, rec)) for rec in recs],
    }).to_csv(tracker.run_dir / 'recommendations.csv', index=False)
    tracker.log_final_metrics(
        {k: {'recall': recall, 'ndcg': ndcg, 'mrr': mrr, 'coverage': coverage}
         for k, recall, ndcg, mrr, coverage in zip(
             cli.metric_ks, recalls, ndcgs, mrrs, coverages
         )},
        split='global_temporal_70_10_20',
        mask_seen=True,
        seed=cli.random_seed,
        inference_total_sec=inference_seconds,
        n_users=len(users),
        maxlen=cli.maxlen,
        checkpoint=str(model_path),
        ranking_protocol='warm_start_known_catalog_v2',
        popularity_bias=recommendation_popularity(
            recs.tolist(), train_item_popularity, cli.metric_ks
        ),
    )
    tracker.close()

    print(f'Loaded checkpoint: {model_path}')
    print(f'Inference time: {inference_seconds:.4f}s')
    for k, recall, ndcg, mrr, coverage in zip(
        cli.metric_ks, recalls, ndcgs, mrrs, coverages
    ):
        print(
            f'k={k}: Recall={recall:.6f}, NDCG={ndcg:.6f}, '
            f'MRR={mrr:.6f}, Coverage={coverage:.6f}'
        )
    print(f'Results saved to: {tracker.run_dir}')


if __name__ == '__main__':
    main()
