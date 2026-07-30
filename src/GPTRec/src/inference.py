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
from research_buckets.evaluate_buckets import (  # noqa: E402
    BUCKET_NAMES,
    evaluate_bucketed_hr,
    print_bucketed_hr,
)
from research_buckets.popularity_buckets import build_popularity_buckets  # noqa: E402
from modules import SeqRecHuggingface  # noqa: E402
from preprocess import add_time_idx  # noqa: E402
from run_train_predict import (  # noqa: E402
    create_model,
    evaluate,
    is_relevance_aggregation,
    predict,
    prepare_data,
    run_relevance_aggregation_by_k,
)


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
    parser.add_argument(
        '--decoding_strategy',
        choices=['saved', 'top_k', 'relevance_aggregation'],
        default='saved',
    )
    parser.add_argument('--ra_temperature', type=float, default=None)
    parser.add_argument('--ra_num_sequences', type=int, default=30)
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return requested


def evaluate_popularity_buckets(
    test_last,
    recs,
    train_item_popularity,
    candidate_items,
    ks,
    recommendations_by_k=None,
):
    bucket_by_item = build_popularity_buckets(
        train_item_popularity, candidate_items
    )
    target_by_user = (
        test_last[test_last['item_id'].isin(candidate_items)]
        .set_index('user_id')['item_id']
        .to_dict()
    )

    def evaluate_frame(frame, frame_ks):
        predicted_by_user = frame.groupby('user_id')['item_id'].apply(list).to_dict()
        users = sorted(set(target_by_user) & set(predicted_by_user))
        return evaluate_bucketed_hr(
            [[target_by_user[user]] for user in users],
            [predicted_by_user[user] for user in users],
            bucket_by_item,
            frame_ks,
        )

    if recommendations_by_k is None:
        return evaluate_frame(recs, ks)

    combined = {
        bucket: {'num_cases': None, 'hr': {}}
        for bucket in BUCKET_NAMES
    }
    for k in ks:
        current = evaluate_frame(recommendations_by_k[k], [k])
        for bucket in BUCKET_NAMES:
            num_cases = current[bucket]['num_cases']
            if combined[bucket]['num_cases'] not in (None, num_cases):
                raise ValueError('GPTRec bucket populations differ between K values')
            combined[bucket]['num_cases'] = num_cases
            combined[bucket]['hr'][k] = current[bucket]['hr'][k]
    return combined


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
    if cli.decoding_strategy == 'top_k':
        config.generation = False
    elif cli.decoding_strategy == 'relevance_aggregation':
        if cli.ra_temperature is None:
            raise ValueError(
                '--ra_temperature must be the value selected on validation'
            )
        if cli.ra_temperature <= 0:
            raise ValueError('--ra_temperature must be greater than zero')
        if cli.ra_num_sequences < 1:
            raise ValueError('--ra_num_sequences must be at least one')
        config.generation = True
        config.mode = 'relevance_aggregation'
        config.generation_params = {
            'num_return_sequences': cli.ra_num_sequences,
            'do_sample': True,
            'temperature': cli.ra_temperature,
            'top_k': 0,
        }
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
    test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
    tracker = ExperimentTracker(
        cli.dataset,
        str(config.model),
        maxlen=cli.max_len,
        run_type='inference',
    )
    recommendations_by_k = None
    if is_relevance_aggregation(config):
        recs, recommendations_by_k, metrics, inference_seconds = (
            run_relevance_aggregation_by_k(
                trainer, seqrec_module, history_before_test, test, test_last,
                history_before_test, config, 'test_last',
                output_dir=tracker.run_dir,
            )
        )
    else:
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
        metrics = evaluate(
            recs, test_last, history_before_test, config,
            prefix='test_last', output_dir=tracker.run_dir,
        )
    if recs.empty:
        raise ValueError('No eligible warm-start test examples remain')
    bucket_metrics = evaluate_popularity_buckets(
        test_last,
        recs,
        train_item_popularity,
        set(history_before_test.item_id.unique().tolist()),
        cli.metric_ks,
        recommendations_by_k=recommendations_by_k,
    )
    print_bucketed_hr(bucket_metrics)
    recommendations_path = tracker.run_dir / 'recommendations.csv'
    recs.to_csv(recommendations_path, index=False)
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
        decoding_strategy=str(config.mode) if config.generation else 'top_k',
        ra_temperature=(
            float(config.generation_params.temperature)
            if is_relevance_aggregation(config) else None
        ),
        ra_num_sequences=(
            int(config.generation_params.num_return_sequences)
            if is_relevance_aggregation(config) else None
        ),
        popularity_bias=(
            {
                k: recommendation_popularity(
                    recommendations_by_k[k].groupby('user_id')['item_id']
                    .apply(list).tolist(),
                    train_item_popularity,
                    [k],
                )[k]
                for k in cli.metric_ks
            }
            if recommendations_by_k is not None
            else recommendation_popularity(
                recs.groupby('user_id')['item_id'].apply(list).tolist(),
                train_item_popularity,
                cli.metric_ks,
            )
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
    print(f'Recommendations saved to: {recommendations_path}')
    print(f'Results saved to: {tracker.run_dir}')


if __name__ == '__main__':
    main()
