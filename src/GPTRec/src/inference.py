from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR.parents[1]))

from experiment_tools.experiment_tracking import (  # noqa: E402
    ExperimentTracker,
    checkpoint_path,
    recommendation_popularity,
    save_dataset_popularity,
)
from modules import SeqRecHuggingface  # noqa: E402
from preprocess import add_time_idx  # noqa: E402
from run_train_predict import create_model, evaluate, predict, prepare_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', default='ml-1m',
        choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty',
                 'amazon_Sports_and_Outdoors', 'amazon_Toys_and_Games'],
    )
    parser.add_argument('--max_len', type=int, default=50)
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
    device = select_device(cli.device)

    model_path = (
        cli.checkpoint
        or checkpoint_path('GPTRec', cli.dataset, cli.max_len, cli.random_seed)
    ).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')
    config_path = model_path.with_suffix('.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f'Checkpoint config not found: {config_path}')

    config = OmegaConf.load(config_path)
    saved_max_len = int(config.dataset.max_length)
    if saved_max_len != cli.max_len:
        raise ValueError(
            f'Checkpoint max_length={saved_max_len} does not match --max_len={cli.max_len}'
        )
    config.dataset_name = cli.dataset
    config.evaluator.top_k = list(cli.metric_ks)
    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    os.chdir(SOURCE_DIR)
    train, validation, test, item_count = prepare_data(config)
    history_before_test = add_time_idx(
        pd.concat([train, validation], ignore_index=True)
    )
    train_item_popularity = history_before_test.item_id.value_counts().to_dict()
    save_dataset_popularity(cli.dataset, train_item_popularity)

    model = create_model(config, item_count=item_count)
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    if str(config.model) != 'GPT-2':
        raise ValueError(f'Unsupported GPTRec checkpoint model: {config.model}')
    seqrec_module = SeqRecHuggingface(
        model,
        **config.seqrec_module,
        candidate_items=history_before_test.item_id.unique().tolist(),
    )
    if config.generation:
        seqrec_module.set_predict_mode(
            generate=True,
            mode=config.mode,
            **config.generation_params,
        )
    else:
        seqrec_module.set_predict_mode(generate=False)

    trainer = pl.Trainer(
        callbacks=[TQDMProgressBar(refresh_rate=100)],
        enable_checkpointing=False,
        logger=False,
        accelerator='gpu' if device == 'cuda' else 'cpu',
        devices=1,
    )
    started = time.perf_counter()
    recs = predict(
        trainer,
        seqrec_module,
        history_before_test,
        config,
        test_data=test,
        last_evaluation=True,
    )
    inference_seconds = time.perf_counter() - started
    if recs.empty:
        raise ValueError('No eligible warm-start test examples remain')

    test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
    metrics = evaluate(
        recs, test_last, history_before_test, config, prefix='test_last'
    )
    recs.to_csv('recommendations.csv', index=False)

    tracker = ExperimentTracker(cli.dataset, str(config.model))
    tracker.log_final_metrics(
        {k: {
            'recall': metrics.get(f'test_last_recall@{k}', 0.0),
            'ndcg': metrics.get(f'test_last_ndcg@{k}', 0.0),
            'mrr': metrics.get(f'test_last_mrr@{k}', 0.0),
            'coverage': metrics.get(f'test_last_coverage@{k}', 0.0),
        } for k in cli.metric_ks},
        split='global_temporal_70_10_20',
        mask_seen=True,
        seed=cli.random_seed,
        inference_total_sec=inference_seconds,
        n_users=int(recs['user_id'].nunique()),
        maxlen=cli.max_len,
        checkpoint=str(model_path),
        ranking_protocol='warm_start_known_catalog_v2',
        popularity_bias=recommendation_popularity(
            recs.groupby('user_id')['item_id'].apply(list).tolist(),
            train_item_popularity,
            cli.metric_ks,
        ),
    )
    tracker.close()

    print(f'Loaded checkpoint: {model_path}')
    print(f'Inference time: {inference_seconds:.4f}s')
    for k in cli.metric_ks:
        print(
            f'k={k}: Recall={metrics.get(f"test_last_recall@{k}", 0.0):.6f}, '
            f'NDCG={metrics.get(f"test_last_ndcg@{k}", 0.0):.6f}, '
            f'MRR={metrics.get(f"test_last_mrr@{k}", 0.0):.6f}, '
            f'Coverage={metrics.get(f"test_last_coverage@{k}", 0.0):.6f}'
        )
    print(f'Recommendations saved to: {SOURCE_DIR / "recommendations.csv"}')
    print(f'Results saved to: {tracker.run_dir}')


if __name__ == '__main__':
    main()
