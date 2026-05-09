import optuna
import torch
import numpy as np
import time
import logging
import os
import json
import argparse
from argparse import Namespace
from functools import partial

from utils import load_and_split_gts, fix_random_seed_as, Data_Train, Data_Val, Data_Test
from trainer import item_num_create, choose_model, model_train, evaluate_and_print

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEARCH_SPACE = {
    'lr': [5e-4, 1e-3, 2e-3],
    'batch_size': [256, 512],
    'hidden_size': [64, 128, 256],
    'dropout': [0.1, 0.2, 0.3],
    'emb_dropout': [0.1, 0.2, 0.3],
    'num_blocks': [2, 3, 4],
    'diffusion_steps': [32, 64],
    'lambda_uncertainty': [1e-4, 1e-3, 1e-2],
    'noise_schedule': ['trunc_lin', 'linear', 'cosine'],
    'schedule_sampler_name': ['uniform', 'lossaware'],
}


def objective(trial, base_args, data_raw):
    trial_args = Namespace(**vars(base_args))
    trial_args.lr = trial.suggest_categorical('lr', SEARCH_SPACE['lr'])
    trial_args.batch_size = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size'])
    trial_args.hidden_size = trial.suggest_categorical('hidden_size', SEARCH_SPACE['hidden_size'])
    trial_args.dropout = trial.suggest_categorical('dropout', SEARCH_SPACE['dropout'])
    trial_args.emb_dropout = trial.suggest_categorical('emb_dropout', SEARCH_SPACE['emb_dropout'])
    trial_args.num_blocks = trial.suggest_categorical('num_blocks', SEARCH_SPACE['num_blocks'])
    trial_args.diffusion_steps = trial.suggest_categorical('diffusion_steps', SEARCH_SPACE['diffusion_steps'])
    trial_args.lambda_uncertainty = trial.suggest_categorical('lambda_uncertainty', SEARCH_SPACE['lambda_uncertainty'])
    trial_args.noise_schedule = trial.suggest_categorical('noise_schedule', SEARCH_SPACE['noise_schedule'])
    trial_args.schedule_sampler_name = trial.suggest_categorical('schedule_sampler_name', SEARCH_SPACE['schedule_sampler_name'])

    trial_args.epochs = 7
    trial_args.eval_interval = 7
    trial_args.patience = 3

    fix_random_seed_as(trial_args.random_seed)

    pretrain_path = os.path.join('saved', 'pretrain', trial_args.dataset, 'pretrain.pth')
    if not os.path.exists(pretrain_path):
        trial_args.pretrained = False
        trial_args.freeze_emb = False

    tra_data = Data_Train(data_raw['train'], trial_args)
    val_data = Data_Val(data_raw['val_seq'], data_raw['val_tgt'], trial_args)
    tra_loader = tra_data.get_pytorch_dataloaders()
    val_loader = val_data.get_pytorch_dataloaders()

    # Модель 
    model = choose_model(trial_args)
    # pretrain_path = os.path.join('saved', 'pretrain', trial_args.dataset, 'pretrain.pth')
    # if not os.path.exists(pretrain_path):
    #     trial_args.pretrained = False
    #     trial_args.freeze_emb = False

    # Обучение без теста
    best_model, _ = model_train(model, tra_loader, val_loader, None, trial_args, logger, train_time="tune")

    # Валидационная метрика
    recall10 = evaluate_and_print(best_model, val_loader, trial_args, logger, description="Tune Validation")
    return recall10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='toys',
                        help='Dataset name (toys, beauty, sports, ml-1m, yelp, etc.)')
    args_cli = parser.parse_args()

    # настройки, которые не перебираются
    base_args = Namespace(
        dataset=args_cli.dataset,
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
        weight_decay=0.0,
        momentum=None,
        rescale_timesteps=True,
        metric_ks=[5, 10, 20],
        eval_interval=20,
        patience=5,
        epochs=80,
        description='Diffu_norm_score',
        long_head=False,
        diversity_measure=False,
        epoch_time_avg=False,
        log_file='log/',
        decay_step=100,
        gamma=0.1,
        loss='mse',                    
        loss_scale=1.0,
        loss_lambda=0.001,
        hidden_act='gelu',
        model='adrec',                  # фиксированная модель
        independent=True,
        pcgrad=False,
        geodesic=False,
        pretrained=False,               # будет переопределено в choose_model
        freeze_emb=False,
        parallel_ag=False,
        split_onebyone=False,
        is_causal=False,
        dif_decoder='att',
        dif_objective='pred_x0',
        lambda_schedule=False,
        lambda_beta_a=0.0,
        lambda_beta_b=0.0,
        beta_a=0.3,
        beta_b=10.0,
        cfg_scale=1.0,
        mask_seen=False,                # во время тюнинга маскирование выключено
    )

    fix_random_seed_as(base_args.random_seed)
    
    data_raw = load_and_split_gts(quantiles=(0.7, 0.8))
    base_args = item_num_create(base_args) 

    study = optuna.create_study(
        direction='maximize',
        study_name='adrec_tuning',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        partial(objective, base_args=base_args, data_raw=data_raw),
        n_trials=20,
        timeout=3600,
        show_progress_bar=True
    )

    best_params = study.best_params
    logger.info(f"Best hyperparameters: {best_params}")

    #  Финальное обучение на train+val
    final_args = Namespace(**vars(base_args))
    for k, v in best_params.items():
        setattr(final_args, k, v)
    final_args.epochs = 80
    final_args.mask_seen = True   # маскирование просмотренных айтемов
    
    pretrain_path = os.path.join('saved', 'pretrain', final_args.dataset, 'pretrain.pth')
    if not os.path.exists(pretrain_path):
        final_args.pretrained = False
        final_args.freeze_emb = False
    

    # Объединяем train и val_seq+val_tgt
    train_combined = []
    for uid in sorted(data_raw['train_dict'].keys()):
        if uid in data_raw['val_seq_dict'] and uid in data_raw['val_tgt_dict']:
            combined_seq = (data_raw['train_dict'][uid] +
                            data_raw['val_seq_dict'][uid] +
                            [data_raw['val_tgt_dict'][uid]])
            train_combined.append(combined_seq)

    tra_data_final = Data_Train(train_combined, final_args)
    test_data_final = Data_Test(data_raw['test_seq'],
                                [[] for _ in data_raw['test_tgt']],
                                data_raw['test_tgt'], final_args)

    tra_loader_final = tra_data_final.get_pytorch_dataloaders()
    test_loader_final = test_data_final.get_pytorch_dataloaders()

    fix_random_seed_as(final_args.random_seed)

    model_final = choose_model(final_args)

    best_model_final, test_metrics = model_train(
        model_final,
        tra_loader_final,
        None,                # без валидации
        test_loader_final,
        final_args,
        logger,
        train_time="final_trainval"
    )

    # Сохраняем модель и конфиг
    os.makedirs('best_models', exist_ok=True)
    torch.save(best_model_final.state_dict(), 'best_models/final_trainval_model.pth')
    with open('best_models/final_trainval_args.json', 'w') as f:
        json.dump(vars(final_args), f, indent=2)

    logger.info("Final evaluation completed.")
    logger.info("Best hyperparameters:")
    for key in sorted(best_params.keys()):
        logger.info(f"  {key}: {best_params[key]}")


if __name__ == '__main__':
    main()