import optuna
import torch
import numpy as np
import random
from copy import deepcopy
import sys
import time
from tensorboardX import SummaryWriter

import world
import utils
import dataloader
import diffusion as gd
import register
from parse import parse_args
from os.path import join

import Procedure
from model import LightGCN
from utils import BPRLoss

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)

def objective(trial, base_args, dataset, device):
    """
    обучаем модель и возвращаем лучший recall@10 на валидации
    """
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    diff_lr = trial.suggest_float('diff_lr', 1e-4, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 0.05, 0.5)
    decay = trial.suggest_float('decay', 1e-6, 1e-3, log=True)
    layer = trial.suggest_int('layer', 1, 4)
    steps = trial.suggest_int('steps', 2, 50)
    sampling_steps = trial.suggest_int('sampling_steps', 1, steps)
    noise_scale = trial.suggest_float('noise_scale', 1e-4, 1e-1, log=True)
    noise_min = trial.suggest_float('noise_min', 1e-4, 1e-2, log=True)
    noise_max = trial.suggest_float('noise_max', 1e-3, 1e-1, log=True)
    emb_size = trial.suggest_int('emb_size', 5, 50)

    # клонируем базовую конфигурацию и обновляем предложенными значениями
    config = deepcopy(base_args.__dict__)
    config.update({
        'lr': lr,
        'diff_lr': diff_lr,
        'alpha': alpha,
        'decay': decay,
        'layer': layer,
        'steps': steps,
        'sampling_steps': sampling_steps,
        'noise_scale': noise_scale,
        'noise_min': noise_min,
        'noise_max': noise_max,
        'emb_size': emb_size,
        # остальные парам как в base_args
    })
    # Обновляем world.config 
    world.config.update(config)
    world.TRAIN_epochs = 100  

    out_dims = eval(base_args.dims) + [base_args.recdim]
    in_dims = out_dims[::-1]

    Recmodel = LightGCN(world.config, dataset).to(device)
    user_reverse_model = register.DIFF_MODELS['dnn'](
        in_dims, out_dims, config['emb_size'], time_type="cat", norm=base_args.norm
    ).to(device)
    item_reverse_model = register.DIFF_MODELS['dnn'](
        in_dims, out_dims, config['emb_size'], time_type="cat", norm=base_args.norm
    ).to(device)

    mean_type = gd.ModelMeanType.START_X if base_args.mean_type == 'x0' else gd.ModelMeanType.EPSILON
    diffusion = gd.GaussianDiffusion(
        world.config, mean_type, base_args.noise_schedule,
        config['noise_scale'], config['noise_min'], config['noise_max'],
        config['steps'], device
    ).to(device)

    bpr = BPRLoss(Recmodel, user_reverse_model, item_reverse_model, diffusion, world.config)

    best_recall = 0.0
    cnt = 0
    iter_count = 0
    for epoch in range(world.TRAIN_epochs):
        Recmodel.train()
        user_reverse_model.train()
        item_reverse_model.train()
        dataset.get_pair_bpr()
        aver_loss = 0.0
        idx = 0
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=base_args.batch_size, shuffle=True, num_workers=0
        )
        for batch_users, batch_pos, batch_neg in train_loader:
            batch_users = batch_users.to(device)
            batch_pos = batch_pos.to(device)
            batch_neg = batch_neg.to(device)
            loss = bpr.call_bpr(batch_users, batch_pos, batch_neg, iter_count)
            aver_loss += loss
            idx += 1
            iter_count += 1
        aver_loss = aver_loss / idx

        if (epoch + 1) % 5 == 0:
            results = Procedure.Test(
                dataset, Recmodel, user_reverse_model, item_reverse_model,
                diffusion, epoch, w=None, multicore=world.config['multicore']
            )
            # results: (valid_precision, valid_recall, valid_ndcg, valid_mrr,
            #           test_precision, test_recall, test_ndcg, test_mrr)
            recall_at_10 = results[1][0]  # recall@10 на валидации
            if recall_at_10 > best_recall:
                best_recall = recall_at_10

            # Ранняя остановка 
            if epoch > 30:
                if recall_at_10 < best_recall:
                    cnt += 1
                else:
                    cnt = 1
                if cnt >= 6:
                    break

    return best_recall

def main_tune():
    base_args = parse_args()
    base_args.epochs = 100  
    base_args.batch_size = 2048
    base_args.recdim = 64
    base_args.dropout = 0
    base_args.keepprob = 0.6
    base_args.dims = '[200,600]'
    base_args.emb_size = 10  
    base_args.mean_type = 'x0'
    base_args.noise_schedule = 'linear-var'
    base_args.norm = False
    base_args.act = 'relu'
    base_args.num_ng = 4
    base_args.multicore = 0
    base_args.data_path = '../data/ml-1m'  

    dataset = dataloader.DiffData(path=base_args.data_path)
    device = world.device

    study = optuna.create_study(direction='maximize', study_name='lightgcn_diff_tuning')
    study.optimize(
        lambda trial: objective(trial, base_args, dataset, device),
        n_trials=100,  # количество экспериментов
        timeout=None  # можно ограничить по времени
    )

    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value (best recall@10): {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    import json
    with open('best_params.json', 'w') as f:
        json.dump(best_trial.params, f, indent=4)

    print("\n--- Retraining final model with best parameters ---")
    # Обновляем base_args лучшими параметрами
    for key, value in best_trial.params.items():
        setattr(base_args, key, value)
    # Запускаем финальное обучение (можно импортировать и вызвать функцию из main.py,
    # но проще запустить main.py как подпроцесс с аргументами)
    import subprocess
    cmd = f"python main.py --lr={best_trial.params['lr']} --diff_lr={best_trial.params['diff_lr']} --alpha={best_trial.params['alpha']} --decay={best_trial.params['decay']} --layer={best_trial.params['layer']} --steps={best_trial.params['steps']} --sampling_steps={best_trial.params['sampling_steps']} --noise_scale={best_trial.params['noise_scale']} --noise_min={best_trial.params['noise_min']} --noise_max={best_trial.params['noise_max']} --emb_size={best_trial.params['emb_size']} --epochs=1000 --data_path={base_args.data_path}"
    print(f"Running: {cmd}")
    subprocess.run(cmd, shell=True)

if __name__ == '__main__':
    main_tune()