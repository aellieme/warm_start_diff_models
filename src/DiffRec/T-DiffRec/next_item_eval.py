import numpy as np
import torch
from torch.utils.data import DataLoader
import time
from collections import defaultdict
from tqdm import tqdm

import data_utils
import evaluate_topk_dp as eval_metrics
import models.gaussian_diffusion as gd
from models.DNN import DNN

def _get_user_sequences(test_list_path, n_user, n_item):
    """
    Загружает test_list.npy и возвращает два словаря:
    - user_test_items: {uid: [iid1, iid2, ...]} в хронологическом порядке
    - user_test_mask: бинарная матрица (csr) всех тестовых взаимодействий (для совместимости)
    """
    test_list = np.load(test_list_path, allow_pickle=True)
    user_items = defaultdict(list)
    for uid, iid in test_list:
        user_items[int(uid)].append(int(iid))
    # Преобразуем в обычные списки
    user_test_items = {uid: items for uid, items in user_items.items()}
    # Создаём csr-матрицу для совместимости с существующим кодом (если нужно)
    rows, cols = [], []
    for uid, items in user_items.items():
        for iid in items:
            rows.append(uid)
            cols.append(iid)
    from scipy.sparse import csr_matrix
    test_y_data = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_user, n_item))
    return user_test_items, test_y_data

def evaluate_last_target(model, diffusion, history_loader, history_mask, user_test_items, topN_list, device, sampling_steps, sampling_noise=False):
    """
    Оценка LAST target: предсказываем только последний айтем из тестовой последовательности каждого пользователя.
    
    Args:
        model: обученная DNN
        diffusion: объект GaussianDiffusion
        history_loader: DataLoader с данными истории (train_data_ori или train_data_adapt) – должен содержать только историю, не тест!
        history_mask: csr-матрица маски (история) – для зануления предсказаний
        user_test_items: словарь {uid: [item1, item2, ...]} – тестовые последовательности
        topN_list: список K для метрик
        device: torch device
        sampling_steps: количество шагов диффузии при сэмплировании
        sampling_noise: добавлять ли шум при сэмплировании
    Returns:
        (precisions, recalls, ndcgs, mrrs, covs) – каждый список длины len(topN_list)
    """
    model.eval()
    
    n_user = history_mask.shape[0]
    # target_last = [None] * n_user
    target_last = [[] for _ in range(n_user)]
    for uid, items in user_test_items.items():
        if items:
            target_last[uid] = [items[-1]]   # только последний
        else:
            target_last[uid] = []
    
    # Собираем предсказания
    predict_items = []  # список списков предсказанных айтемов (topK)
    with torch.no_grad():
        for batch_idx, batch in enumerate(history_loader):
            # batch – тензор (batch_size, n_item) – вход для модели
            batch = batch.to(device)
            his_data = history_mask[batch_idx * history_loader.batch_size : batch_idx * history_loader.batch_size + len(batch)]
            # Предсказание
            pred = diffusion.p_sample(model, batch, sampling_steps, sampling_noise)
            # Маскируем историю
            pred[his_data.nonzero()] = -np.inf
            _, indices = torch.topk(pred, max(topN_list))
            indices = indices.cpu().numpy().tolist()
            predict_items.extend(indices)
    
    
    target_items_ordered = [target_last[i] for i in range(n_user)]
    
    # Вычисляем метрики
    precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(
        target_items_ordered, predict_items, topN_list, n_items=history_mask.shape[1]
    )
    return precisions, recalls, ndcgs, mrrs, covs

def evaluate_successive_target(model, diffusion, history_loader, history_mask, user_test_items, topN_list, device, sampling_steps, sampling_noise=False):
    """
    Оценка succsessive target: предсказываем каждый следующий айтем в тестовой последовательности,
    инкрементально расширяя историю.
    
    Возвращает средние метрики по всем предсказанным шагам.
    """
    model.eval()
    n_user = history_mask.shape[0]
    n_item = history_mask.shape[1]
    
    # Получаем историю каждого пользователя из csr-матрицы
    user_history_items = {}
    for uid in range(n_user):
        row = history_mask[uid]
        user_history_items[uid] = row.nonzero()[1].tolist()
    
    all_targets = []
    all_predictions = []
    
    # Прогресс-бар по пользователям
    for uid in tqdm(range(n_user), desc="Successive evaluation"):
        test_items = user_test_items.get(uid, [])
        if not test_items:
            continue
        current_history = user_history_items[uid].copy()
        for target_item in test_items:
            input_vec = torch.zeros(n_item).to(device)
            for it in current_history:
                input_vec[it] = 1.0
            input_batch = input_vec.unsqueeze(0)
            mask = torch.zeros(n_item).to(device)
            mask[current_history] = 1.0
            pred = diffusion.p_sample(model, input_batch, sampling_steps, sampling_noise).squeeze(0)
            pred[mask.bool()] = -np.inf
            _, topk = torch.topk(pred, max(topN_list))
            topk = topk.cpu().numpy().tolist()
            all_predictions.append(topk)
            all_targets.append([target_item])
            current_history.append(target_item)
    
    # Вычисляем метрики
    precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(
        all_targets, all_predictions, topN_list, n_items=n_item
    )
    return precisions, recalls, ndcgs, mrrs, covs

# def evaluate_successive_target(model, diffusion, history_loader, history_mask, user_test_items, topN_list, device, sampling_steps, sampling_noise=False):
#     """
#     Оценка SUCCESSIVE target: предсказываем каждый следующий айтем в тестовой последовательности,
#     инкрементально расширяя историю.
    
#     Возвращает средние метрики по всем предсказанным шагам.
#     """
#     model.eval()
#     n_user = history_mask.shape[0]
#     n_item = history_mask.shape[1]
    
#     user_history_items = {}
#     for uid in range(n_user):
#         row = history_mask[uid]
#         user_history_items[uid] = row.nonzero()[1].tolist()
    
#     # Собираем все предсказания и цели для каждого шага
#     all_targets = []   # список истинных айтемов для каждого шага (в порядке обхода)
#     all_predictions = []  # список топ-K списков для каждого шага
    
#     # Проходим по каждому пользователю отдельно (для successive нужен последовательный инференс)
#     for uid in range(n_user):
#         test_items = user_test_items.get(uid, [])
#         if not test_items:
#             continue
#         # Текущая история (копия)
#         current_history = user_history_items[uid].copy()
#         # Для каждого тестового айтема (по порядку)
#         for t_idx, target_item in enumerate(test_items):
            
#             input_vec = torch.zeros(n_item).to(device)
#             for it in current_history:
#                 input_vec[it] = 1.0   # бинарно
#             # Добавляем размерность батча
#             input_batch = input_vec.unsqueeze(0)
#             # Маска – текущая история (чтобы не рекомендовать уже виденное)
#             mask = torch.zeros(n_item).to(device)
#             mask[current_history] = 1.0
#             # Предсказание
#             pred = diffusion.p_sample(model, input_batch, sampling_steps, sampling_noise).squeeze(0)
#             pred[mask.bool()] = -np.inf
#             _, topk = torch.topk(pred, max(topN_list))
#             topk = topk.cpu().numpy().tolist()
#             # Сохраняем
#             all_predictions.append(topk)
#             all_targets.append([target_item])
#             # Обновляем историю: добавляем текущий целевой айтем (для следующего шага)
#             current_history.append(target_item)
    
#     # Теперь all_targets и all_predictions – списки одинаковой длины (общее число шагов)
#     # Вычисляем метрики
#     precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(
#       all_targets, all_predictions, topN_list, n_items=n_item
#     )
    # return precisions, recalls, ndcgs, mrrs, covs

if __name__ == "__main__":
    
    pass