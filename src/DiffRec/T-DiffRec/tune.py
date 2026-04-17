import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
import argparse
import warnings
warnings.filterwarnings("ignore")

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN

DATASET = 'ml-1m'                     
DATA_PATH = f'../../data/{DATASET}/'
BATCH_SIZE = 400
EPOCHS = 1000                          
TOP_N = [10, 20, 50, 100]
TST_W_VAL = True  
CUDA = True
GPU = '0'
SAVE_PATH = './saved_models/'
LOG_NAME = 'tune'
ROUND = 1

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
device = torch.device("cuda:0" if CUDA else "cpu")

train_path = DATA_PATH + 'train_list.npy'
valid_path = DATA_PATH + 'valid_list.npy'
test_path = DATA_PATH + 'test_list.npy'

train_data, train_data_ori, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
    train_path, valid_path, test_path, w_min=0.1, w_max=1.0   # временные w_min/w_max, будут заменены при тюнинге
)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=4)
test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
if TST_W_VAL:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()) + torch.FloatTensor(valid_y_data.toarray()))
    test_twv_loader = DataLoader(tv_dataset, batch_size=BATCH_SIZE, shuffle=False)
mask_tv = train_data_ori + valid_y_data

print("Data loaded. n_user={}, n_item={}".format(n_user, n_item))

# def evaluate(model, diffusion, data_loader, data_te, mask_his, topN):
def evaluate(model, diffusion, data_loader, data_te, mask_his, topN, sampling_steps, sampling_noise=False):
    """Оценка модели (скопировано из main.py)"""
    model.eval()
    e_idxlist = list(range(mask_his.shape[0]))
    e_N = mask_his.shape[0]
    predict_items = []
    target_items = []
    for i in range(e_N):
        target_items.append(data_te[i, :].nonzero()[1].tolist())
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            his_data = mask_his[e_idxlist[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+len(batch)]]
            batch = batch.to(device)
            # prediction = diffusion.p_sample(model, batch, args_sampling_steps, args_sampling_noise)
            prediction = diffusion.p_sample(model, batch, sampling_steps, sampling_noise)
            prediction[his_data.nonzero()] = -np.inf
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
            
    precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(target_items, predict_items, topN, n_item)
    return precisions, recalls, ndcgs, mrrs, covs

def train_and_evaluate(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    # weight_decay = trial.suggest_float('weight_decay', 0.0, 1e-4, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
    steps = trial.suggest_int('steps', 5, 100)
    noise_scale = trial.suggest_float('noise_scale', 0.0001, 0.5, log=True)
    noise_min = trial.suggest_float('noise_min', 0.0001, 0.001, log=True)
    noise_max = trial.suggest_float('noise_max', 0.0011, 0.1, log=True)
    w_min = trial.suggest_float('w_min', 0.0, 0.5)
    w_max = trial.suggest_float('w_max', 0.5, 1.0)
    emb_size = trial.suggest_int('emb_size', 8, 32)
    dims = [1000]

    global train_data, train_dataset, train_loader
    train_data, _, _, _, _, _ = data_utils.data_load(train_path, valid_path, test_path, w_min, w_max)
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=4)

    # диффузия
    mean_type = gd.ModelMeanType.START_X
    diffusion = gd.GaussianDiffusion(mean_type, 'linear-var', noise_scale, noise_min, noise_max, steps, device)
    diffusion.to(device)
    # DNN
    out_dims = dims + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, emb_size, time_type="cat", norm=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_recall = -100
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        if epoch - best_epoch >= 25:
            break
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, reweight=True)
            loss = losses["loss"].mean()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        # Оценка каждые 5 эпох
        if epoch % 5 == 0:
            # используем train_data_ori (бинарную) для маски
            # precisions, recalls, ndcgs, mrrs, covs = evaluate(model, diffusion, test_loader, valid_y_data, train_data_ori, TOP_N)
            precisions, recalls, ndcgs, mrrs, covs = evaluate(model, diffusion, test_loader, valid_y_data, train_data_ori, TOP_N, sampling_steps=steps, sampling_noise=False)
            recall10 = recalls[0]   # recall@10
            if recall10 > best_recall:
                best_recall = recall10
                best_epoch = epoch
                # сохраняем лучшую модель в память (не на диск)
                best_model_state = model.state_dict()
        # Логирование
        trial.report(best_recall, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    # Возвращаем лучший recall@10
    return best_recall

def main():
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(train_and_evaluate, n_trials=25, timeout=5500)  # 20 попыток или 1 час

    # Лучшие параметры
    best_params = study.best_params
    steps = best_params['steps']
    print("Best hyperparameters:", best_params)

    #  Обучение финальной модели с лучшими параметрами 
    print("\nTraining final model with best parameters...")
    # Перезагружаем данные с лучшими w_min/w_max
    w_min = best_params['w_min']
    w_max = best_params['w_max']
    train_data, train_data_ori, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
        train_path, valid_path, test_path, w_min, w_max)
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=4)
    # диффузия
    diffusion = gd.GaussianDiffusion(gd.ModelMeanType.START_X, 'linear-var',
                                     best_params['noise_scale'], best_params['noise_min'], best_params['noise_max'],
                                     best_params['steps'], device)
    diffusion.to(device)
    # DNN
    out_dims = [1000] + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, best_params['emb_size'], time_type="cat", norm=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    best_recall = -100
    best_epoch = 0
    best_model_state = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, reweight=True)
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0:
            # precisions, recalls, _, _, _ = evaluate(model, diffusion, test_loader, valid_y_data, train_data_ori, TOP_N)
            precisions, recalls, _, _, _ = evaluate(model, diffusion, test_loader, valid_y_data, train_data_ori, TOP_N, sampling_steps=steps, sampling_noise=False)
            recall10 = recalls[0]
            if recall10 > best_recall:
                best_recall = recall10
                best_epoch = epoch
                best_model_state = model.state_dict()
            print(f"Epoch {epoch:03d}, val recall@10={recall10:.4f}, best={best_recall:.4f}")
            if epoch - best_epoch >= 25:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with recall {best_recall:.4f}")
                break
    # Загружаем лучшую модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    else:
        print("Warning: No best model state found!")
    # model.load_state_dict(best_model_state)
    # Сохраняем модель
    os.makedirs(SAVE_PATH, exist_ok=True)
    model_path = f"{SAVE_PATH}/best_tuned_model.pth"
    torch.save(model, model_path)
    print(f"Final model saved to {model_path}")

    # Инференс базовый 
    print("\n Inference without adaptation ")
    # Загружаем модель (можно из сохранённой, но у нас уже есть в памяти)
    model.eval()
    # Для базового теста используем test_loader и test_y_data, маска mask_tv = train_data_ori + valid_y_data
    # precisions, recalls, ndcgs, mrrs, covs = evaluate(model, diffusion, test_loader, test_y_data, mask_tv, TOP_N)
    precisions, recalls, ndcgs, mrrs, covs = evaluate(model, diffusion, test_loader, test_y_data, mask_tv, TOP_N, sampling_steps=steps, sampling_noise=False)
    print("Base test results:")
    # print(f"  Precision@{TOP_N}: {precisions}")
    print(f"  Recall@{TOP_N}:    {recalls}")
    print(f"  NDCG@{TOP_N}:      {ndcgs}")
    print(f"  MRR@{TOP_N}:       {mrrs}")
    print(f"  Coverage:          {covs}")

    # Инференс с адаптацией
    adapt_path = DATA_PATH + 'adapt_list.npy'
    if os.path.exists(adapt_path):
        print("\n Warm-start adaptation ")
        adapt_list = np.load(adapt_path, allow_pickle=True)
        # Объединяем train + adapt
        train_list_full = np.load(train_path, allow_pickle=True)
        combined_list = np.vstack([train_list_full, adapt_list])
        user_items = {}
        for uid, iid in combined_list:
            user_items.setdefault(int(uid), []).append(int(iid))
        rows, cols, weights = [], [], []
        for uid, items in user_items.items():
            w = np.linspace(w_min, w_max, len(items))
            for i, iid in enumerate(items):
                rows.append(uid)
                cols.append(iid)
                weights.append(w[i])
        from scipy.sparse import csr_matrix
        train_data_adapt = csr_matrix((weights, (rows, cols)), shape=(n_user, n_item))
        mask_adapt = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_user, n_item))
        if TST_W_VAL:
            mask_adapt += valid_y_data
        adapt_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data_adapt.toarray()))
        adapt_loader = DataLoader(adapt_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        start_time = time.perf_counter()
        # Оценка
        # precisions_a, recalls_a, ndcgs_a, mrrs_a, covs_a = evaluate(model, diffusion, adapt_loader, test_y_data, mask_adapt, TOP_N)
        precisions_a, recalls_a, ndcgs_a, mrrs_a, covs_a = evaluate(model, diffusion, adapt_loader, test_y_data, mask_adapt, TOP_N, sampling_steps=steps, sampling_noise=False)
        warm_latency = time.perf_counter() - start_time
        print("Warm-start test results:")
        # print(f"  Precision@{TOP_N}: {precisions_a}")
        print(f"  Recall@{TOP_N}:    {recalls_a}")
        print(f"  NDCG@{TOP_N}:      {ndcgs_a}")
        print(f"  MRR@{TOP_N}:       {mrrs_a}")
        print(f"  Coverage:          {covs_a}")
        print(f"Warm-start inference latency: {warm_latency:.4f} sec")
    else:
        print("adapt_list.npy not found, skipping warm-start.")

if __name__ == '__main__':
    main()