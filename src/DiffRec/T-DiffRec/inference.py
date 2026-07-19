"""Evaluate a final T-DiffRec checkpoint under the shared warm-start protocol."""

import argparse
import ast
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import checkpoint_path  # noqa: E402


SEED = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ml-1m')
    parser.add_argument('--data_path', type=Path, default=Path('../../data'))
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--topN', default='[10, 20, 50, 100]')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--w_min', type=float, default=0.1)
    parser.add_argument('--w_max', type=float, default=1.0)
    parser.add_argument('--mean_type', choices=['x0', 'eps'], default='x0')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--noise_schedule', default='linear-var')
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--noise_min', type=float, default=0.0001)
    parser.add_argument('--noise_max', type=float, default=0.02)
    parser.add_argument('--sampling_noise', action='store_true')
    parser.add_argument('--sampling_steps', type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    topn = ast.literal_eval(args.topN)
    sampling_steps = args.sampling_steps or args.steps

    data_dir = args.data_path
    if data_dir.name != args.dataset:
        data_dir = data_dir / args.dataset
    protocol = data_utils.load_warm_start_data(
        str(data_dir), args.w_min, args.w_max
    )
    inputs, history, targets, users = data_utils.select_eligible_rows(
        protocol['test_input'], protocol['test_mask'], protocol['test_targets'],
        protocol['train_val_candidates'],
    )
    loader = DataLoader(
        data_utils.DataDiffusion(torch.FloatTensor(inputs.toarray())),
        batch_size=args.batch_size,
        shuffle=False,
    )

    model_path = args.checkpoint or checkpoint_path(
        'T-DiffRec', args.dataset, seed=SEED, extension='.pth'
    )
    if not model_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')
    payload = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or 'model_state_dict' not in payload:
        raise ValueError('Legacy T-DiffRec checkpoints must be retrained for protocol v2')
    model = DNN(**payload['model_kwargs']).to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    mean_type = (
        gd.ModelMeanType.START_X if args.mean_type == 'x0'
        else gd.ModelMeanType.EPSILON
    )
    diffusion = gd.GaussianDiffusion(
        mean_type, args.noise_schedule, args.noise_scale, args.noise_min,
        args.noise_max, args.steps, device,
    ).to(device)
    candidate_mask = torch.as_tensor(
        protocol['train_val_candidates'], dtype=torch.bool, device=device
    )
    if int(candidate_mask.sum().item()) < max(topn):
        raise ValueError('Candidate catalogue is smaller than the largest metric K')

    predictions = []
    offset = 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch)
            scores = diffusion.p_sample(
                model, batch.to(device), sampling_steps, args.sampling_noise
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
    elapsed = time.perf_counter() - started

    actual = [[int(target)] for target in targets]
    candidates = set(np.flatnonzero(protocol['train_val_candidates']).tolist())
    _, recalls, ndcgs, mrrs, coverages = eval_metrics.compute_all_metrics(
        actual, predictions, topn, len(candidates), candidate_items=candidates
    )
    print(f'Loaded checkpoint: {model_path}')
    print(f'Eligible users: {len(users)}')
    print(f'Inference time: {elapsed:.4f}s')
    for k, recall, ndcg, mrr, coverage in zip(
        topn, recalls, ndcgs, mrrs, coverages
    ):
        print(
            f'k={k}: Recall={recall:.6f}, NDCG={ndcg:.6f}, '
            f'MRR={mrr:.6f}, Coverage={coverage:.6f}'
        )


if __name__ == '__main__':
    main()
