import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
from optuna.trial import TrialState
import argparse
import scipy.sparse as sp
import warnings
warnings.filterwarnings("ignore")

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN

import time
from plotting import TrainingPlotter

import random
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True

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
    train_path, valid_path, test_path, w_min=0.1, w_max=1.0  
)
train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=2)
test_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
if TST_W_VAL:
    tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()) + torch.FloatTensor(valid_y_data.toarray()))
    test_twv_loader = DataLoader(tv_dataset, batch_size=BATCH_SIZE, shuffle=False)
mask_tv = train_data_ori + valid_y_data

print("Data loaded. n_user={}, n_item={}".format(n_user, n_item))

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
            prediction = diffusion.p_sample(model, batch, sampling_steps, sampling_noise)
            prediction[his_data.nonzero()] = -np.inf
            _, indices = torch.topk(prediction, topN[-1])
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
            
    precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(target_items, predict_items, topN, n_item)
    return precisions, recalls, ndcgs, mrrs, covs

def train_and_evaluate(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True, num_workers=2)

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
            precisions, recalls, ndcgs, mrrs, covs = evaluate(model, diffusion, test_loader, valid_y_data, train_data_ori, TOP_N, sampling_steps=steps, sampling_noise=False)
            recall10 = recalls[0]   # recall@10
            if recall10 > best_recall:
                best_recall = recall10
                best_epoch = epoch
                best_model_state = model.state_dict()
        # Логирование
        trial.report(best_recall, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_recall

def main():
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(train_and_evaluate, n_trials=25, timeout=5500)  
    
    best_params = study.best_params
    w_min = best_params['w_min']
    w_max = best_params['w_max']
    steps = best_params['steps']
    print("Best hyperparameters:", best_params)

    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    combined_list = np.vstack([train_list, valid_list])

    # группировка по пользователям с сохранением хронологии
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

    train_val_data = sp.csr_matrix((weights, (rows, cols)), shape=(n_user, n_item))
    train_val_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_val_data.toarray()))
    train_val_loader = DataLoader(train_val_dataset, batch_size=BATCH_SIZE,
                                  pin_memory=True, shuffle=True, num_workers=2)

    diffusion = gd.GaussianDiffusion(gd.ModelMeanType.START_X, 'linear-var',
                                     best_params['noise_scale'], best_params['noise_min'],
                                     best_params['noise_max'], best_params['steps'], device)
    diffusion.to(device)

    out_dims = [1000] + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, best_params['emb_size'], time_type="cat", norm=False).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=best_params['lr'],
                            weight_decay=best_params['weight_decay'])

    print("\nTraining final model with best parameters on train+val")
    # Инициализация плоттера
    plotter = TrainingPlotter(
        save_dir='./log/' + DATASET,
        model_name=f"T-DiffRec_final_trainval_{time.strftime('%Y%m%d_%H%M%S')}",
        metrics=['loss']
    )

    FIXED_EPOCHS = 200
    for epoch in range(1, FIXED_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_val_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, reweight=True)
            loss = losses["loss"].mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_val_loader)
        print(f"Epoch {epoch:03d} average loss: {avg_loss:.4f}")
        plotter.update(epoch=epoch, loss=avg_loss)

        # Сохраняем график каждые 10 эпох
        if epoch % 10 == 0:
            plotter.plot(save=True, show=False, suffix=f'_epoch{epoch}')

    # Финальный график и сохранение модели
    plotter.plot(save=True, show=False, suffix='_final')
    os.makedirs(SAVE_PATH, exist_ok=True)
    model_path = f"{SAVE_PATH}/final_model_trainval.pth"
    torch.save(model, model_path)
    print(f"Final model (train+val) saved to {model_path}")

    # Инференс на тесте
    test_loader_combined = DataLoader(train_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    start_time = time.perf_counter()
    precisions, recalls, ndcgs, mrrs, covs = evaluate(
        model, diffusion, test_loader_combined, test_y_data, mask_tv,
        TOP_N, sampling_steps=steps, sampling_noise=False)
    latency = time.perf_counter() - start_time
    print("Test Results on train+val model:")
    print(f"  Recall@{TOP_N}:    {recalls}")
    print(f"  NDCG@{TOP_N}:      {ndcgs}")
    print(f"  MRR@{TOP_N}:       {mrrs}")
    print(f"  Coverage:          {covs}")
    print(f"  Inference latency: {latency:.4f} sec")

if __name__ == '__main__':
    main()