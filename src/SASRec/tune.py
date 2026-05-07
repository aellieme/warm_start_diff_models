import optuna
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True



from load_evaluate_pipeline import prepare_data_and_description
from training import (
    prepare_sasrec_model,
    train_sasrec_epoch,
    validate_last_item,
)
from model import save_sasrec_model, get_model_path, generate_model_name
from load_evaluate_pipeline import run_inference_pipeline
from training import build_sasrec_model
from training import build_final_sasrec_model

BASE_CONFIG = {
    'num_epochs': 500,
    'tune_epochs': 30,
    'patience': 3,
    'maxlen': 200,
    'hidden_units': 128,
    'dropout_rate': 0.5,
    'num_blocks': 2,
    'num_heads': 2,
    'batch_size': 128,
    'learning_rate': 1e-3,
    'l2_emb': 0.0,
    'sampler_seed': 99,
    'manual_seed': 111,
}


SEARCH_SPACE = {
    'maxlen': [200],                     
    'hidden_units': [256, 384, 512],     # чуть выше, чтобы проверить, не лучше ли
    'dropout_rate': [0.1, 0.2, 0.3],     # вокруг 0.2
    'num_blocks': [2, 3],                # 2 было лучшим, проверим 3
    'num_heads': [2, 4],                 # 2 было лучшим, проверим 4
    'batch_size': [128, 256],            # 128 лучше, но проверим 256
    'learning_rate': [5e-4, 1e-3, 2e-3], # вокруг 1e-3
    'l2_emb': [5e-5, 1e-4, 5e-4],        # вокруг 1e-4
}



def train_and_evaluate(config, train_data, val_data, data_description, max_epochs, patience):
    """Обучает модель, возвращает лучший HR@10 на валидации."""
    model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(
        config, train_data, data_description
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_hr = 0.0
    no_improve = 0
    val_every = 2

    for epoch in range(max_epochs):
        train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizer, criterion, device
        )
        hr, _ = validate_last_item(model, val_data, train_data, data_description, topn=10)

        if hr > best_hr:
            best_hr = hr
            no_improve = 0
        else:
            no_improve += 1
            if no_improve * val_every >= patience:
                break
    return best_hr


def objective(trial, train_data, val_data, data_description):
    config = deepcopy(BASE_CONFIG)
    config['maxlen'] = trial.suggest_categorical('maxlen', SEARCH_SPACE['maxlen'])
    config['hidden_units'] = trial.suggest_categorical('hidden_units', SEARCH_SPACE['hidden_units'])
    config['dropout_rate'] = trial.suggest_categorical('dropout_rate', SEARCH_SPACE['dropout_rate'])
    config['num_blocks'] = trial.suggest_categorical('num_blocks', SEARCH_SPACE['num_blocks'])
    config['num_heads'] = trial.suggest_categorical('num_heads', SEARCH_SPACE['num_heads'])
    config['batch_size'] = trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size'])
    config['learning_rate'] = trial.suggest_categorical('learning_rate', SEARCH_SPACE['learning_rate'])
    config['l2_emb'] = trial.suggest_categorical('l2_emb', SEARCH_SPACE['l2_emb'])

    hr = train_and_evaluate(
        config, train_data, val_data, data_description,
        max_epochs=config['tune_epochs'], patience=config['patience']
    )
    return -hr   # тк оптуна минимизирует


def main():
    (train_data, val_data, test_data, test_examples, data_index, data_description, userid_col, itemid_col, time_col, val_seq_dict, train_val_data) = prepare_data_and_description()
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")

    # study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    study.enqueue_trial({
        'maxlen': 200,
        'hidden_units': 256,
        'dropout_rate': 0.2,
        'num_blocks': 2,
        'num_heads': 2,
        'batch_size': 128,
        'learning_rate': 0.001,
        'l2_emb': 0.0001,
    })
    
    print("Оптимизация")
    study.optimize(
        lambda trial: objective(trial, train_data, val_data, data_description),
        timeout=3600,
        n_trials=50,
        show_progress_bar=True
    )

    best_params = study.best_params
    best_hr = -study.best_value
    print("\nЛучшие гиперпараметры")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Лучший HR@10: {best_hr:.4f}")

    
    print("\nФинальное обучение на train+val ")
    final_config = deepcopy(BASE_CONFIG)
    final_config.update(best_params)
    final_config['num_epochs'] = 250       
    final_model = build_final_sasrec_model(final_config, train_val_data, data_description)

    model_filename = generate_model_name(final_config, suffix='tuned')
    model_path = get_model_path(model_filename)
    save_sasrec_model(final_model, final_config, data_description, data_index, model_path)
    print(f"Модель сохранена: {model_path}")


    print("\n Оценка финальной модели на тесте")

    recs_baseline, users_baseline, metrics_baseline, time_base = run_inference_pipeline(
        final_model, train_data, train_data, test_examples,
        data_description, userid_col, itemid_col, time_col, val_seq_dict, topn=10
    )
    prec, rec, ndcg, mrr, cov = metrics_baseline
    print(f"Инференс: Recall@10={rec[0]:.4f}, MRR={mrr[0]:.4f}, NDCG={ndcg[0]:.4f}, Cov={cov[0]:.4f}, Latency = {time_base:.4f}")


if __name__ == "__main__":
    main()