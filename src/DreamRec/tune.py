import optuna
import pandas as pd
import json
import torch
import numpy as np
import random
import os
import time as Time
from collections import defaultdict
from utility import extract_axis_1
from Modules_ori import MultiHeadAttention, PositionwiseFeedForward
from DreamRec import Tenc, diffusion, evaluate, setup_seed
from load_and_split import load_and_preprocess_ml1m, global_temporal_split, prepare_dreamrec_data
from load_and_split import build_sequences, pad_and_format
import pandas as pd
from plotting import TrainingPlotter

FIXED_ARGS = {
    'epoch': 5,                
    'data': 'ml-1m',
    'random_seed': 100,
    'beta_end': 0.02,
    'beta_start': 0.0001,
    'beta_sche': 'exp',
    'optimizer': 'adam',
    'l2_decay': 0,
    'cuda': 0,
    'report_epoch': False,
    'layers': 1,
    'max_seq_len': 50,
}

_data_cache = None

def get_data():
    global _data_cache
    if _data_cache is None:
        all_data, data_index, n_users, n_items, userid_col, itemid_col, time_col = load_and_preprocess_ml1m()
        train_raw, val_raw, test_raw, T_train, T_val = global_temporal_split(all_data, time_col)
        item_num = n_items
        train_data, val_data, test_data, pad_token = prepare_dreamrec_data(
            train_raw, val_raw, test_raw, userid_col, itemid_col, time_col,
            max_seq_len=FIXED_ARGS['max_seq_len'], pad_item=item_num
        )
        _data_cache = (train_data, val_data, test_data, item_num, FIXED_ARGS['max_seq_len'])
    return _data_cache

def objective(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    w = trial.suggest_float('w', 0.5, 4.0)
    p = trial.suggest_float('p', 0.0, 0.5)
    hidden_factor = trial.suggest_categorical('hidden_factor', [64])
    batch_size = trial.suggest_categorical('batch_size', [128, 256, 512])
    timesteps = trial.suggest_categorical('timesteps', [50, 100, 200])
    diffuser_type = trial.suggest_categorical('diffuser_type', ['mlp1', 'mlp2'])

    train_data, val_data, test_data, item_num, seq_size = get_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    setup_seed(FIXED_ARGS['random_seed'])

    model = Tenc(hidden_factor, item_num, seq_size, dropout_rate, diffuser_type, device)
    diff = diffusion(timesteps, FIXED_ARGS['beta_start'], FIXED_ARGS['beta_end'], w)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-8, weight_decay=FIXED_ARGS['l2_decay'])
    model.to(device)

    epochs = FIXED_ARGS['epoch']
    num_batches = int(len(train_data) / batch_size)

    best_hr = 0.0
    for epoch in range(epochs):
        model.train()
        for j in range(num_batches):
            batch = train_data.sample(n=batch_size).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())

            optimizer.zero_grad()
            seq = torch.LongTensor(seq).to(device)
            len_seq_tensor = torch.LongTensor(len_seq).to(device)
            target = torch.LongTensor(target).to(device)

            x_start = model.cacu_x(target)
            h = model.cacu_h(seq, len_seq_tensor, p)
            n = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss, _ = diff.p_losses(model, x_start, h, n, loss_type='l2')
            loss.backward()
            optimizer.step()

        # Валидация
        hr = evaluate(model, val_data, diff, device, item_num)
        if hr > best_hr:
            best_hr = hr

        trial.report(hr, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_hr

if __name__ == '__main__':
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=42),
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=10, timeout=3600)

    print('Лучшие параметры:')
    for k, v in study.best_trial.params.items():
        print(f'  {k}: {v}')
    print(f'Лучший HR@20 на валидации: {study.best_trial.value:.6f}')

    with open('best_params.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)
    print('Параметры сохранены в best_params.json')
    
    
    print("\nФинальное обучение на train+val с лучшими параметрами")
    best = study.best_params
    final_epochs = 100         
    
    all_data, data_index, n_users, n_items, userid_col, itemid_col, time_col = load_and_preprocess_ml1m()
    train_raw, val_raw, test_raw, _, _ = global_temporal_split(all_data, time_col)

    train_final_raw = pd.concat([train_raw, val_raw]).sort_values([userid_col, time_col])

    max_seq_len = FIXED_ARGS['max_seq_len']
    item_num = n_items

    train_seq_final = build_sequences(train_final_raw, userid_col, itemid_col, time_col,
                                      max_seq_len, keep_last_only=False)
    train_df_final = pad_and_format(train_seq_final, max_seq_len, pad_item=item_num)

    # Тестовые последовательности остаются как раньше (из prepare_dreamrec_data)
    _, _, test_data, _, _ = get_data()   # test_data уже готово

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(FIXED_ARGS['random_seed'])

    # Создаём модель и диффузор с лучшими параметрами
    model = Tenc(hidden_size=best['hidden_factor'],
                 item_num=item_num,
                 state_size=max_seq_len,
                 dropout=best['dropout_rate'],
                 diffuser_type=best['diffuser_type'],
                 device=device)
    diff = diffusion(timesteps=best['timesteps'],
                     beta_start=FIXED_ARGS['beta_start'],
                     beta_end=FIXED_ARGS['beta_end'],
                     w=best['w'])

    optimizer = torch.optim.Adam(model.parameters(), lr=best['lr'], eps=1e-8,
                                 weight_decay=FIXED_ARGS['l2_decay'])
    model.to(device)

    final_plotter = TrainingPlotter(save_dir='./logs', model_name='DreamRec_final',
                                    metrics=['loss'])

    batch_size_final = best['batch_size']
    num_batches = int(len(train_df_final) / batch_size_final)

    print(f"Начало обучения ({final_epochs} эпох, batch_size={batch_size_final})...")
    for epoch in range(final_epochs):
        model.train()
        epoch_loss = 0.0
        for j in range(num_batches):
            batch = train_df_final.sample(n=batch_size_final).to_dict()
            seq = list(batch['seq'].values())
            len_seq = list(batch['len_seq'].values())
            target = list(batch['next'].values())

            optimizer.zero_grad()
            seq = torch.LongTensor(seq).to(device)
            len_seq_tensor = torch.LongTensor(len_seq).to(device)
            target = torch.LongTensor(target).to(device)

            x_start = model.cacu_x(target)
            h = model.cacu_h(seq, len_seq_tensor, best['p'])
            n = torch.randint(0, best['timesteps'], (batch_size_final,), device=device).long()
            loss, _ = diff.p_losses(model, x_start, h, n, loss_type='l2')
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, num_batches)
        final_plotter.update(epoch=epoch, loss=avg_loss)
        print(f"Epoch {epoch+1:3d}/{final_epochs} - loss: {avg_loss:.6f}")

    final_plotter.plot(show=False)
    print("График сохранён в ./logs/DreamRec_final_training_curves.png")

    print("\n Финальная оценка на тесте ")
    hr = evaluate(model, test_data, diff, device, item_num)
    print(f"Test HR@20: {hr:.6f}")