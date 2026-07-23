from __future__ import annotations

import argparse
import logging
import os
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pandas as pd
import torch


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR.parents[1]))

from experiment_tools.experiment_tracking import (  # noqa: E402
    ExperimentTracker,
    checkpoint_path,
    save_dataset_popularity,
)
from main import fix_random_seed_as, item_num_create, load_and_split_gts  # noqa: E402
from model import Att_Diffuse_model, create_model_diffu  # noqa: E402
from trainer import evaluate_and_print  # noqa: E402
from utils import Data_Test  # noqa: E402


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
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()


def select_device(requested: str) -> str:
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available')
    return requested


def main() -> None:
    cli = parse_args()
    device = select_device(cli.device)
    model_path = (
        cli.checkpoint
        or checkpoint_path('DiffuRec', cli.dataset, cli.max_len, cli.random_seed)
    ).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')

    payload = torch.load(model_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or 'model_state_dict' not in payload or 'args' not in payload:
        raise ValueError(f'Unsupported DiffuRec checkpoint format: {model_path}')
    if payload['args'].get('item_id_offset') != 1:
        raise ValueError('Legacy DiffuRec checkpoint is incompatible with warm_start_known_catalog_v2')
    saved_dataset = payload['args'].get('dataset')
    saved_max_len = payload['args'].get('max_len')
    if saved_dataset is not None and saved_dataset != cli.dataset:
        raise ValueError(
            f'Checkpoint dataset={saved_dataset} does not match --dataset={cli.dataset}'
        )
    if saved_max_len is not None and int(saved_max_len) != cli.max_len:
        raise ValueError(
            f'Checkpoint max_len={saved_max_len} does not match --max_len={cli.max_len}'
        )

    args = Namespace(**payload['args'])
    args.dataset = cli.dataset
    args.max_len = cli.max_len
    args.random_seed = cli.random_seed
    args.metric_ks = cli.metric_ks
    args.device = device
    args.amp = cli.amp
    args.ranking_protocol = 'warm_start_known_catalog_v2'
    fix_random_seed_as(args.random_seed)

    os.chdir(SOURCE_DIR)
    data_raw = load_and_split_gts(quantiles=(0.7, 0.8), dataset_name=args.dataset)
    args = item_num_create(args, len(data_raw['smap']))

    merged_train = {uid: list(sequence) for uid, sequence in data_raw['train'].items()}
    for uid, sequence in data_raw['val_seq'].items():
        merged_train[uid] = list(sequence) + list(data_raw['val'].get(uid, []))
    args.coverage_candidate_items = {
        item for sequence in merged_train.values() for item in sequence
    }
    args.train_item_popularity = dict(Counter(
        item for sequence in merged_train.values() for item in sequence
    ))
    save_dataset_popularity(args.dataset, args.train_item_popularity)

    test_data = Data_Test(
        data_raw['test_seq'],
        {uid: [] for uid in data_raw['test_seq']},
        data_raw['test'],
        args,
    )
    test_loader = test_data.get_pytorch_dataloaders()

    model = Att_Diffuse_model(create_model_diffu(args), args).to(device)
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    tracker = ExperimentTracker(args.dataset, 'DiffuRec')
    args.experiment_tracker = tracker
    result = evaluate_and_print(
        model, test_loader, args, logging.getLogger(__name__), description='test'
    )
    pd.DataFrame({
        'user_id': range(len(result['canonical_predicted'])),
        'recommendations': result['canonical_predicted'],
    }).to_csv(tracker.run_dir / 'recommendations.csv', index=False)

    print(f'Loaded checkpoint: {model_path}')
    print(f'Results saved to: {tracker.run_dir}')


if __name__ == '__main__':
    main()
