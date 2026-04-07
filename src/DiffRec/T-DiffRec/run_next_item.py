# run_next_item.py
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from collections import defaultdict

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN
from next_item_eval import evaluate_last_target, evaluate_successive_target  # созданный ранее

DATASET = 'ml-1m'
DATA_PATH = f'../../data/{DATASET}/'
BATCH_SIZE = 400
TOP_N = [1]
CUDA = True
GPU = '0'
MODEL_PATH = './saved_models/best_tuned_model.pth'   # модель из tune.py
DEVICE = torch.device(f"cuda:{GPU}" if CUDA else "cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU

print("Loading data...")
train_path = DATA_PATH + 'train_list.npy'
valid_path = DATA_PATH + 'valid_list.npy'
test_path = DATA_PATH + 'test_list.npy'

# Для получения n_user, n_item и бинарных матриц используем data_load (с временными w_min/w_max, они будут перезаписаны позже, но нам нужны только размеры и матрицы)
_, train_data_ori, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(
    train_path, valid_path, test_path, w_min=0.1, w_max=1.0
)

# Загружаем test_list.npy для восстановления хронологических последовательностей (нужно для Last и Successive)
test_list = np.load(test_path, allow_pickle=True)
user_test_items = defaultdict(list)
for uid, iid in test_list:
    user_test_items[int(uid)].append(int(iid))
# Преобразуем в обычный словарь
user_test_items = dict(user_test_items)

# Маска для базового сценария: train + valid (чтобы не рекомендовать уже известное)
mask_base = train_data_ori + valid_y_data

print(f"Loading best model from {MODEL_PATH}")
# model = torch.load(MODEL_PATH, map_location=DEVICE)
model = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model.to(DEVICE)
model.eval()


best_params = {
    'lr': 0.0009897796742466861,
    'weight_decay': 2.20305238899952e-07,
    'steps': 62,
    'noise_scale': 0.02371057175242013,
    'noise_min': 0.0002025810998725498,
    'noise_max': 0.0011174720713008645,
    'w_min': 0.34544333918849457,
    'w_max': 0.7698128829362187,
    'emb_size': 16
}
steps = best_params['steps']
# Создаём diffusion
diffusion = gd.GaussianDiffusion(
    gd.ModelMeanType.START_X, 'linear-var',
    best_params['noise_scale'], best_params['noise_min'], best_params['noise_max'],
    steps, DEVICE
)
diffusion.to(DEVICE)

#baseline
history_dense = torch.FloatTensor(train_data_ori.toarray())
history_dataset = data_utils.DataDiffusion(history_dense)
history_loader = DataLoader(history_dataset, batch_size=BATCH_SIZE, shuffle=False)

#basic last target
# print("\n" + "="*60)
print("last target baseline")
start_time = time.perf_counter()
last_prec, last_rec, last_ndcg, last_mrr, last_cov = evaluate_last_target(
    model, diffusion, history_loader, mask_base, user_test_items, TOP_N,
    DEVICE, sampling_steps=steps, sampling_noise=False
)
last_latency = time.perf_counter() - start_time

print(f"LAST Target latency: {last_latency:.2f} sec")
print(f"Precision@{TOP_N}: {[round(x,4) for x in last_prec]}")
print(f"Recall@{TOP_N}:    {[round(x,4) for x in last_rec]}")
print(f"NDCG@{TOP_N}:      {[round(x,4) for x in last_ndcg]}")
print(f"MRR@{TOP_N}:       {[round(x,4) for x in last_mrr]}")
print(f"Coverage:          {[round(x,4) for x in last_cov]}")


#basic succsessive target
# print("succsessive target baseline")
# start_time = time.perf_counter()
# succ_prec, succ_rec, succ_ndcg, succ_mrr, succ_cov = evaluate_successive_target(
#     model, diffusion, history_loader, mask_base, user_test_items, TOP_N,
#     DEVICE, sampling_steps=steps, sampling_noise=False
# )
# succ_latency = time.perf_counter() - start_time

# print(f"succsessive target latency: {succ_latency:.2f} sec")
# print(f"Precision@{TOP_N}: {[round(x,4) for x in succ_prec]}")
# print(f"Recall@{TOP_N}:    {[round(x,4) for x in succ_rec]}")
# print(f"NDCG@{TOP_N}:      {[round(x,4) for x in succ_ndcg]}")
# print(f"MRR@{TOP_N}:       {[round(x,4) for x in succ_mrr]}")
# print(f"Coverage:          {[round(x,4) for x in succ_cov]}")



#warm start
adapt_path = DATA_PATH + 'adapt_list.npy'
if os.path.exists(adapt_path):
    print("\n" + "="*60)
    print("Подготовка warm‑start ")
    
    adapt_list = np.load(adapt_path, allow_pickle=True)
    train_list_full = np.load(train_path, allow_pickle=True)
    combined_list = np.vstack([train_list_full, adapt_list])
    
    user_items_adapt = defaultdict(list)
    for uid, iid in combined_list:
        user_items_adapt[int(uid)].append(int(iid))
    rows, cols, weights = [], [], []
    for uid, items in user_items_adapt.items():
        w = np.linspace(best_params['w_min'], best_params['w_max'], len(items))
        for i, iid in enumerate(items):
            rows.append(uid)
            cols.append(iid)
            weights.append(w[i])
    train_data_adapt = csr_matrix((weights, (rows, cols)), shape=(n_user, n_item))
    mask_adapt = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_user, n_item))
    # Добавляем валидацию 
    mask_adapt = mask_adapt + valid_y_data   # бинарная маска: train+adapt+valid
    
    adapt_dense = torch.FloatTensor(train_data_adapt.toarray())
    adapt_dataset = data_utils.DataDiffusion(adapt_dense)
    adapt_loader = DataLoader(adapt_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print("\nlast target + warm‑start")
    start_time = time.perf_counter()
    last_prec_w, last_rec_w, last_ndcg_w, last_mrr_w, last_cov_w = evaluate_last_target(
        model, diffusion, adapt_loader, mask_adapt, user_test_items, TOP_N,
        DEVICE, sampling_steps=steps, sampling_noise=False
    )
    last_latency_w = time.perf_counter() - start_time
    print(f"LAST (warm) latency: {last_latency_w:.2f} sec")
    print(f"Recall@{TOP_N}: {[round(x,4) for x in last_rec_w]}")
    print(f"Precision@{TOP_N}: {[round(x,4) for x in last_prec_w]}")
    print(f"Recall@{TOP_N}:    {[round(x,4) for x in last_rec_w]}")
    print(f"NDCG@{TOP_N}:      {[round(x,4) for x in last_ndcg_w]}")
    print(f"MRR@{TOP_N}:       {[round(x,4) for x in last_mrr_w]}")
    print(f"Coverage:          {[round(x,4) for x in last_cov_w]}")
    
    # print("\nsuccsessive target + warm‑start")
    # start_time = time.perf_counter()
    # succ_prec_w, succ_rec_w, succ_ndcg_w, succ_mrr_w, succ_cov_w = evaluate_successive_target(
    #     model, diffusion, adapt_loader, mask_adapt, user_test_items, TOP_N,
    #     DEVICE, sampling_steps=steps, sampling_noise=False
    # )
    # succ_latency_w = time.perf_counter() - start_time
    # print(f"SUCCESSIVE (warm) latency: {succ_latency_w:.2f} sec")
    # print(f"Recall@{TOP_N}: {[round(x,4) for x in succ_rec_w]}")
else:
    print("adapt_list.npy not found ")
