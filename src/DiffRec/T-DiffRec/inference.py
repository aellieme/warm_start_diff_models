from __future__ import annotations

import argparse
import ast
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR.parents[1]))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--data_path', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--topN', default='[10, 20, 100]')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--checkpoint', type=Path, default=None)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == 'auto':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return torch.device('cuda:0' if requested == 'cuda' else 'cpu')


def main() -> None:
    cli = parse_args()
    topn = ast.literal_eval(cli.topN)
    if not isinstance(topn, list) or not topn or not all(
        isinstance(k, int) and k > 0 for k in topn
    ):
        raise ValueError('--topN must be a non-empty list of positive integers')

    random.seed(cli.random_seed)
    np.random.seed(cli.random_seed)
    torch.manual_seed(cli.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cli.random_seed)
    device = select_device(cli.device)

    model_path = (
        cli.checkpoint
        or checkpoint_path('T-DiffRec', cli.dataset, seed=cli.random_seed, extension='.pth')
    ).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')
    payload = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or 'model_state_dict' not in payload:
        raise ValueError('Legacy T-DiffRec checkpoints must be retrained for protocol v2')
    saved_args = payload.get('args', {})
    saved_dataset = saved_args.get('dataset')
    if saved_dataset is not None and saved_dataset != cli.dataset:
        raise ValueError(
            f'Checkpoint dataset={saved_dataset} does not match --dataset={cli.dataset}'
        )

    os.chdir(SOURCE_DIR)
    data_dir = cli.data_path or Path(saved_args.get('data_path', '../../data'))
    if data_dir.name != cli.dataset:
        data_dir = data_dir / cli.dataset
    protocol = data_utils.load_warm_start_data(
        str(data_dir),
        float(saved_args.get('w_min', 0.1)),
        float(saved_args.get('w_max', 1.0)),
    )
    inputs, history, targets, users = data_utils.select_eligible_rows(
        protocol['test_input'],
        protocol['test_mask'],
        protocol['test_targets'],
        protocol['train_val_candidates'],
    )
    loader = DataLoader(
        data_utils.DataDiffusion(torch.FloatTensor(inputs.toarray())),
        batch_size=cli.batch_size,
        shuffle=False,
    )

    model = DNN(**payload['model_kwargs']).to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()
    mean_type = (
        gd.ModelMeanType.START_X
        if saved_args.get('mean_type', 'x0') == 'x0'
        else gd.ModelMeanType.EPSILON
    )
    steps = int(saved_args.get('steps', 100))
    diffusion = gd.GaussianDiffusion(
        mean_type,
        saved_args.get('noise_schedule', 'linear-var'),
        float(saved_args.get('noise_scale', 0.1)),
        float(saved_args.get('noise_min', 0.0001)),
        float(saved_args.get('noise_max', 0.02)),
        steps,
        device,
    ).to(device)
    sampling_steps = int(saved_args.get('sampling_steps') or steps)
    sampling_noise = bool(saved_args.get('sampling_noise', False))

    candidate_mask = torch.as_tensor(
        protocol['train_val_candidates'], dtype=torch.bool, device=device
    )
    if int(candidate_mask.sum().item()) < max(topn):
        raise ValueError('Candidate catalogue is smaller than the largest metric K')

    predictions = []
    offset = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch_size = len(batch)
            scores = diffusion.p_sample(
                model, batch.to(device), sampling_steps, sampling_noise
            )
            scores.masked_fill_(~candidate_mask.unsqueeze(0), -torch.inf)
            seen = torch.as_tensor(
                history[offset:offset + batch_size].toarray() > 0,
                dtype=torch.bool,
                device=device,
            )
            scores.masked_fill_(seen, -torch.inf)
            predictions.extend(
                torch.topk(scores, max(topn), dim=-1).indices.cpu().tolist()
            )
            offset += batch_size
    inference_seconds = time.perf_counter() - started
    if not predictions:
        raise ValueError('No eligible warm-start test examples remain')

    actual = [[int(target)] for target in targets]
    candidates = set(np.flatnonzero(protocol['train_val_candidates']).tolist())
    _, recalls, ndcgs, mrrs, coverages = eval_metrics.compute_all_metrics(
        actual, predictions, topn, len(candidates), candidate_items=candidates
    )
    popularity_matrix = protocol['train_binary'] + protocol['valid_binary']
    item_counts = np.asarray(popularity_matrix.sum(axis=0)).ravel()
    train_item_popularity = {
        index: int(count) for index, count in enumerate(item_counts) if count > 0
    }
    save_dataset_popularity(cli.dataset, train_item_popularity)
    bucket_by_item = build_popularity_buckets(
        train_item_popularity, candidates
    )
    bucket_metrics = evaluate_bucketed_hr(
        actual, predictions, bucket_by_item, topn
    )
    print_bucketed_hr(bucket_metrics)

    tracker = ExperimentTracker(cli.dataset, 'T-DiffRec', run_type='inference')
    pd.DataFrame({
        'user_id': users,
        'recommendations': predictions,
    }).to_csv(tracker.run_dir / 'recommendations.csv', index=False)
    tracker.log_final_metrics(
        {k: {'recall': recall, 'ndcg': ndcg, 'mrr': mrr, 'coverage': coverage}
         for k, recall, ndcg, mrr, coverage in zip(
             topn, recalls, ndcgs, mrrs, coverages
         )},
        split='global_temporal_70_10_20',
        mask_seen=True,
        seed=cli.random_seed,
        inference_total_sec=inference_seconds,
        n_users=len(users),
        maxlen=None,
        checkpoint=str(model_path),
        ranking_protocol='warm_start_known_catalog_v2',
        popularity_bias=recommendation_popularity(
            predictions, train_item_popularity, topn
        ),
    )
    tracker.close()

    print(f'Loaded checkpoint: {model_path}')
    print(f'Eligible users: {len(users)}')
    print(f'Inference time: {inference_seconds:.4f}s')
    for k, recall, ndcg, mrr, coverage in zip(
        topn, recalls, ndcgs, mrrs, coverages
    ):
        print(
            f'k={k}: Recall={recall:.6f}, NDCG={ndcg:.6f}, '
            f'MRR={mrr:.6f}, Coverage={coverage:.6f}'
        )
    print(f'Results saved to: {tracker.run_dir}')


if __name__ == '__main__':
    main()
