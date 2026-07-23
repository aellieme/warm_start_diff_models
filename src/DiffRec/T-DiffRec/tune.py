"""Tune T-DiffRec on validation, then train once on train+validation."""

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from visualization.plotting import TrainingPlotter


DATASET = 'ml-1m'
DATA_PATH = Path('../../data') / DATASET
BATCH_SIZE = 400
EPOCHS = 1000
FINAL_EPOCHS = 200
TOP_N = [10, 20, 50, 100]
SEED = 42
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def seed_everything():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def make_loader(matrix, shuffle):
    if shuffle:
        active = np.flatnonzero(matrix.getnnz(axis=1) > 0)
        matrix = matrix[active]
    return DataLoader(
        data_utils.DataDiffusion(torch.FloatTensor(matrix.toarray())),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True,
    )


def evaluate(model, diffusion, loader, targets, history, candidates, sampling_steps):
    model.eval()
    candidate_mask = torch.as_tensor(candidates, dtype=torch.bool, device=DEVICE)
    predictions = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch_size = len(batch)
            scores = diffusion.p_sample(
                model, batch.to(DEVICE), sampling_steps, sampling_noise=False
            )
            scores.masked_fill_(~candidate_mask.unsqueeze(0), -torch.inf)
            seen = torch.as_tensor(
                history[offset:offset + batch_size].toarray() > 0,
                dtype=torch.bool,
                device=DEVICE,
            )
            scores.masked_fill_(seen, -torch.inf)
            predictions.extend(
                torch.topk(scores, max(TOP_N), dim=-1).indices.cpu().tolist()
            )
            offset += batch_size
    actual = [[int(target)] for target in targets]
    candidate_items = set(np.flatnonzero(candidates).tolist())
    return eval_metrics.compute_all_metrics(
        actual, predictions, TOP_N, len(candidate_items),
        candidate_items=candidate_items,
    )


def build_components(params, n_item):
    diffusion = gd.GaussianDiffusion(
        gd.ModelMeanType.START_X, 'linear-var', params['noise_scale'],
        params['noise_min'], params['noise_max'], params['steps'], DEVICE,
    ).to(DEVICE)
    out_dims = [1000, n_item]
    in_dims = out_dims[::-1]
    model = DNN(
        in_dims, out_dims, params['emb_size'], time_type='cat', norm=False
    ).to(DEVICE)
    optimizer = optim.AdamW(
        model.parameters(), lr=params['lr'], weight_decay=params['weight_decay']
    )
    return model, diffusion, optimizer, in_dims, out_dims


def train_and_evaluate(trial):
    seed_everything()
    params = {
        'lr': trial.suggest_float('lr', 1e-5, 1e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True),
        'steps': trial.suggest_int('steps', 5, 100),
        'noise_scale': trial.suggest_float('noise_scale', 0.0001, 0.5, log=True),
        'noise_min': trial.suggest_float('noise_min', 0.0001, 0.001, log=True),
        'noise_max': trial.suggest_float('noise_max', 0.0011, 0.1, log=True),
        'w_min': trial.suggest_float('w_min', 0.0, 0.5),
        'w_max': trial.suggest_float('w_max', 0.5, 1.0),
        'emb_size': trial.suggest_int('emb_size', 8, 32),
    }
    protocol = data_utils.load_warm_start_data(
        str(DATA_PATH), params['w_min'], params['w_max'], include_test=False
    )
    train_loader = make_loader(protocol['train_weighted'], shuffle=True)
    valid_input, valid_history, targets, _ = data_utils.select_eligible_rows(
        protocol['valid_input'], protocol['valid_mask'], protocol['valid_targets'],
        protocol['train_candidates'],
    )
    valid_loader = make_loader(valid_input, shuffle=False)
    model, diffusion, optimizer, _, _ = build_components(params, protocol['n_item'])
    candidate_mask = torch.as_tensor(
        protocol['train_candidates'], dtype=torch.bool, device=DEVICE
    )

    best_recall = -1.0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        if epoch - best_epoch >= 25:
            break
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            losses = diffusion.training_losses(
                model, batch.to(DEVICE), reweight=True,
                candidate_mask=candidate_mask,
            )
            loss = losses['loss'].mean()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            metrics = evaluate(
                model, diffusion, valid_loader, targets, valid_history,
                protocol['train_candidates'], params['steps'],
            )
            recall10 = metrics[1][TOP_N.index(10)]
            if recall10 > best_recall:
                best_recall = recall10
                best_epoch = epoch
        trial.report(best_recall, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_recall


def main():
    seed_everything()
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(train_and_evaluate, n_trials=25, timeout=5500)
    params = dict(study.best_params)
    print('Best hyperparameters:', params)

    protocol = data_utils.load_warm_start_data(
        str(DATA_PATH), params['w_min'], params['w_max']
    )
    train_loader = make_loader(protocol['train_val_weighted'], shuffle=True)
    model, diffusion, optimizer, in_dims, out_dims = build_components(
        params, protocol['n_item']
    )
    candidate_mask = torch.as_tensor(
        protocol['train_val_candidates'], dtype=torch.bool, device=DEVICE
    )
    plotter = TrainingPlotter(
        save_dir='./log/' + DATASET,
        model_name=f"T-DiffRec_final_{time.strftime('%Y%m%d_%H%M%S')}",
        metrics=['loss'],
    )
    for epoch in range(1, FINAL_EPOCHS + 1):
        model.train()
        losses_epoch = []
        for batch in train_loader:
            optimizer.zero_grad()
            losses = diffusion.training_losses(
                model, batch.to(DEVICE), reweight=True,
                candidate_mask=candidate_mask,
            )
            loss = losses['loss'].mean()
            loss.backward()
            optimizer.step()
            losses_epoch.append(loss.item())
        plotter.update(epoch=epoch, loss=float(np.mean(losses_epoch)))
    plotter.plot(save=True, show=False, suffix='_final')

    os.makedirs('./saved_models', exist_ok=True)
    model_path = Path('./saved_models/final_model_trainval.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_kwargs': {
            'in_dims': in_dims, 'out_dims': out_dims,
            'emb_size': params['emb_size'], 'time_type': 'cat', 'norm': False,
        },
        'args': {**params, 'ranking_protocol': 'warm_start_known_catalog_v2'},
    }, model_path)
    print(f'Final model saved to {model_path}')

    test_input, test_history, targets, users = data_utils.select_eligible_rows(
        protocol['test_input'], protocol['test_mask'], protocol['test_targets'],
        protocol['train_val_candidates'],
    )
    metrics = evaluate(
        model, diffusion, make_loader(test_input, shuffle=False), targets,
        test_history, protocol['train_val_candidates'], params['steps'],
    )
    print(f'Eligible test users: {len(users)}')
    print('Final test metrics:', metrics)


if __name__ == '__main__':
    main()
