# tune.py
import optuna
import torch
import numpy as np
import copy
import time
import logging
from argparse import Namespace
from functools import partial

from main import load_and_split_gts, item_num_create, fix_random_seed_as
from model import create_model_diffu, Att_Diffuse_model
from utils import Data_Train, Data_Val, Data_Test
from trainer import model_train, evaluate_and_print
from evaluate_topk_dp import compute_all_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_recall10_on_validation(model, val_loader, args):
    """recall@10 на валидации """
    device = args.device
    model.eval()
    all_actual = []
    all_predicted = []
    with torch.no_grad():
        for batch in val_loader:
            batch = [x.to(device) for x in batch]
            _, rep_diffu, _, _, _, _ = model(batch[0], batch[1], train_flag=False)
            scores = model.diffu_rep_pre(rep_diffu)
            _, topk = torch.topk(scores, k=10, dim=-1)
            for i in range(len(batch[1])):
                all_actual.append([batch[1][i].item()])
                all_predicted.append(topk[i].cpu().tolist())
    _, recalls, _, _, _ = compute_all_metrics(all_actual, all_predicted, [10], args.item_num)
    return recalls[0]

def objective(trial, base_args, data_raw):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])
    hidden_size = trial.suggest_categorical('hidden_size', [64, 128, 256])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    emb_dropout = trial.suggest_float('emb_dropout', 0.1, 0.5)
    num_blocks = trial.suggest_int('num_blocks', 2, 6)
    diffusion_steps = trial.suggest_categorical('diffusion_steps', [16, 32, 64])
    lambda_uncertainty = trial.suggest_float('lambda_uncertainty', 0.0001, 0.01, log=True)
    noise_schedule = trial.suggest_categorical('noise_schedule', ['trunc_lin', 'linear', 'cosine'])
    schedule_sampler_name = trial.suggest_categorical('schedule_sampler_name', ['uniform', 'lossaware'])

    args = Namespace(**vars(base_args))
    args.lr = lr
    args.batch_size = batch_size
    args.hidden_size = hidden_size
    args.dropout = dropout
    args.emb_dropout = emb_dropout
    args.num_blocks = num_blocks
    args.diffusion_steps = diffusion_steps
    args.lambda_uncertainty = lambda_uncertainty
    args.noise_schedule = noise_schedule
    args.schedule_sampler_name = schedule_sampler_name
    args.epochs = 7
    args.eval_interval = 5
    args.patience = 3

    fix_random_seed_as(args.random_seed)

    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw['val_seq'], data_raw['val'], args)
    test_data = Data_Test(data_raw['test_seq'], {uid: [] for uid in data_raw['test_seq']}, data_raw['test'], args)
    tra_loader = tra_data.get_pytorch_dataloaders()
    val_loader = val_data.get_pytorch_dataloaders()
    test_loader = test_data.get_pytorch_dataloaders()

    diffu_rec = create_model_diffu(args)
    model = Att_Diffuse_model(diffu_rec, args)
    model = model.to(args.device)

    best_model, _ = model_train(tra_loader, val_loader, test_loader, model, args, logger)

    recall10 = compute_recall10_on_validation(best_model, val_loader, args)
    return recall10

def main():
    base_args = Namespace(
        dataset='ml-1m',
        random_seed=42,
        max_len=50,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_gpu=1,
        batch_size=512,
        hidden_size=128,
        dropout=0.1,
        emb_dropout=0.3,
        num_blocks=4,
        diffusion_steps=32,
        lambda_uncertainty=0.001,
        noise_schedule='trunc_lin',
        schedule_sampler_name='lossaware',
        optimizer='Adam',
        lr=0.001,
        weight_decay=0,
        momentum=None,
        rescale_timesteps=True,
        metric_ks=[10],
        eval_interval=20,
        patience=5,
        epochs=60,
        description='Diffu_norm_score',
        long_head=False,
        diversity_measure=False,
        epoch_time_avg=False,
        log_file='log/',
        decay_step=100,          
        gamma=0.1,               
        loss_lambda=0.001,      
        hidden_act='gelu',       
    )
    fix_random_seed_as(base_args.random_seed)
    data_raw = load_and_split_gts(quantiles=(0.7, 0.8, 0.9))
    base_args = item_num_create(base_args, len(data_raw['smap']))

    study = optuna.create_study(direction='maximize', study_name='diffurec_tuning')
    objective_partial = partial(objective, base_args=base_args, data_raw=data_raw)
    study.optimize(objective_partial, n_trials=15, timeout=3600)  # 15 trials или 1 час

    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    final_args = Namespace(**vars(base_args))
    for key, value in best_params.items():
        setattr(final_args, key, value)
    final_args.epochs = 60
    final_args.eval_interval = 10
    final_args.patience = 5

    tra_data_final = Data_Train(data_raw['train'], final_args)
    val_data_final = Data_Val(data_raw['val_seq'], data_raw['val'], final_args)
    test_data_final = Data_Test(data_raw['test_seq'], {uid: [] for uid in data_raw['test_seq']}, data_raw['test'], final_args)

    tra_loader_final = tra_data_final.get_pytorch_dataloaders()
    val_loader_final = val_data_final.get_pytorch_dataloaders()
    test_loader_final = test_data_final.get_pytorch_dataloaders()

    diffu_rec_final = create_model_diffu(final_args)
    model_final = Att_Diffuse_model(diffu_rec_final, final_args)
    model_final = model_final.to(final_args.device)

    best_model_final, _ = model_train(tra_loader_final, val_loader_final, test_loader_final, model_final, final_args, logger)

    print("\nFinal evaluation ")
    baseline_test_seq = {}
    for uid in data_raw['test_seq'].keys():
        full_seq = data_raw['test_seq'][uid]
        adapt_items = set(data_raw.get('adapt_seq', {}).get(uid, []))
        baseline_seq = [item for item in full_seq if item not in adapt_items]
        baseline_test_seq[uid] = baseline_seq
    baseline_test_data = Data_Test(baseline_test_seq, {uid: [] for uid in baseline_test_seq}, data_raw['test'], final_args)
    baseline_loader = baseline_test_data.get_pytorch_dataloaders()

    evaluate_and_print(best_model_final, baseline_loader, final_args, logger, description="baseline")
    evaluate_and_print(best_model_final, test_loader_final, final_args, logger, description="adaptation")

if __name__ == '__main__':
    main()