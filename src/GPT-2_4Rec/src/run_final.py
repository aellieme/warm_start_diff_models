import json
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import optuna
from omegaconf import OmegaConf


from run_train_predict import (
    prepare_data,
    create_model,
    create_dataloaders,
    training,
    final_training,
    predict,
    evaluate,
)
from datasets import CausalLMDataset, PaddingCollateFn
from preprocess import add_time_idx


base_cfg = OmegaConf.load("configs/GPT_Optuna.yaml")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

def objective(trial):
    n_embd = trial.suggest_categorical("n_embd", [32, 64, 128, 256])
    n_layer = trial.suggest_int("n_layer", 1, 3)
    n_head  = trial.suggest_categorical("n_head", [1, 2, 4])
    lr      = trial.suggest_float("lr", 0.0003, 0.003, log=True)
    pdrop   = trial.suggest_categorical("pdrop", [0.1, 0.3, 0.5])

    cfg = OmegaConf.create(base_cfg)  
    cfg.model_params = {
        "n_positions": 128,
        "n_embd": n_embd,
        "n_layer": n_layer,
        "n_head": n_head,
        "embd_pdrop": pdrop,
        "attn_pdrop": pdrop,
    }
    cfg.seqrec_module.lr = lr

    train, validation, test, item_count = prepare_data(cfg)
    train_loader, eval_loader = create_dataloaders(train, validation, cfg)
    model = create_model(cfg, item_count=item_count)
    trainer, seqrec_module = training(model, train_loader, eval_loader, cfg)

    val_last = validation.sort_values('time_idx').groupby('user_id').last().reset_index()
    recs = predict(
        trainer, seqrec_module,
        train[train.user_id.isin(validation.user_id.unique())],
        cfg,
        test_data=validation,
        last_evaluation=True,
    )
    metrics = evaluate(recs, val_last, train, cfg, prefix='val_last')
    return metrics.get('val_last_ndcg@10', 0.0)


study = optuna.create_study(
    direction="maximize",
    study_name=base_cfg.task_name,
    storage=f"sqlite:///{base_cfg.task_name}.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=base_cfg.hydra.sweeper.n_trials)

best_params = study.best_params
best_params["n_positions"] = 128  
with open("best_params.json", "w") as f:
    json.dump(best_params, f, indent=2)
print("Лучшие параметры сохранены в best_params.json")

final_cfg = OmegaConf.create(base_cfg)

final_cfg.model_params = {
    "n_positions": 128,
    "n_embd": best_params["n_embd"],
    "n_layer": best_params["n_layer"],
    "n_head": best_params["n_head"],
    "embd_pdrop": best_params["pdrop"],
    "attn_pdrop": best_params["pdrop"],
}
final_cfg.seqrec_module.lr = best_params["lr"]

final_cfg.dataset.shift_labels = True

if "optuna_metrics" in final_cfg:
    del final_cfg.optuna_metrics

final_cfg.final_train = True
final_cfg.final_epochs = 80

train, validation, test, item_count = prepare_data(final_cfg)

train_val = pd.concat([train, validation], ignore_index=True)
train_val = add_time_idx(train_val)

train_dataset = CausalLMDataset(train_val, **final_cfg.dataset)
train_loader = DataLoader(
    train_dataset,
    batch_size=final_cfg.dataloader.batch_size,
    shuffle=True,
    num_workers=final_cfg.dataloader.num_workers,
    collate_fn=PaddingCollateFn(),
)

model = create_model(final_cfg, item_count=item_count)

seqrec_module, trainer = final_training(model, train_loader, final_cfg)
torch.save(seqrec_module.model.state_dict(), "final_model.pt")

history_before_test = pd.concat([train, validation], ignore_index=True)
history_before_test = add_time_idx(history_before_test)

recs = predict(
    trainer, seqrec_module,
    history_before_test,
    final_cfg,
    test_data=test,
    last_evaluation=True,
)

test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
metrics = evaluate(recs, test_last, train, final_cfg, prefix='test_last')
print("Final test metrics:", metrics)

summary = {
    'Recall@10': metrics.get('test_last_recall@10', 0),
    'NDCG@10': metrics.get('test_last_ndcg@10', 0),
    'Coverage': metrics.get('test_last_coverage@10', 0),
    'MRR': metrics.get('test_last_mrr@10', 0),
}
print(pd.DataFrame([summary]).to_string(index=False))