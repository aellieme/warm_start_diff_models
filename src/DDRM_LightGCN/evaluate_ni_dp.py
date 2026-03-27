# src\evaluate\evaluate_ni.py

import numpy as np

def calculate_recall_at_k_ni(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    в задаче Next-Item эквивалентен Hit Rate@K; определяет долю случаев, когда истинный айтем попал в список топ-K.
    actual: (n_users,) — массив ID истинных айтемов.
    predicted: (n_users, max_k) — матрица предсказаний модели.
    k: порог отсечения.
    """
    top_k = predicted[:, :k] # oграничиваем предсказания до k
    hits = (top_k == actual.reshape(-1, 1)).any(axis=1) #сравниваем каждый столбец с правильным ответом
    return float(np.mean(hits))

def calculate_mrr_at_k_ni(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    учитывает позицию истинного айтема, если айтема нет в топ-K, вклад равен 0
    actual: (n_users,)
    predicted: (n_users, max_k)
    """
    top_k = predicted[:, :k]
    hits_indices = np.where(top_k == actual.reshape(-1, 1))
    reciprocal_ranks = np.zeros(len(actual))
    # hits_indices[0] -  индексы пользователей, hits_indices[1] - позиции айтемов 
    # только первое вхождение для каждого пользователя
    _, first_indices = np.unique(hits_indices[0], return_index=True)
    user_idx = hits_indices[0][first_indices]
    item_pos = hits_indices[1][first_indices]
    
    reciprocal_ranks[user_idx] = 1.0 / (item_pos + 1)
    return float(np.mean(reciprocal_ranks))

def calculate_ndcg_at_k_ni(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    """
    для одного релевантного объекта IDCG всегда равен 1
    формула: 1 / log2(rank + 1)
    actual: (n_users,)
    predicted: (n_users, max_k)
    """
    top_k = predicted[:, :k]
    hits_indices = np.where(top_k == actual.reshape(-1, 1))
    
    ndcg_scores = np.zeros(len(actual))
    
    _, first_indices = np.unique(hits_indices[0], return_index=True)
    user_idx = hits_indices[0][first_indices]
    item_pos = hits_indices[1][first_indices]
    
    ndcg_scores[user_idx] = 1.0 / np.log2(item_pos + 2)
    return float(np.mean(ndcg_scores))

def calculate_coverage_at_k_ni(predicted: np.ndarray, total_catalog_items: list, k: int) -> float:
    """
    доля уникальных айтемов из каталога, которые модель рекомендует хотя бы раз в топ-K
    predicted: (n_users, max_k)
    total_catalog_items: список всех уникальных ID из обучающей выборки.
    """
    top_k = predicted[:, :k]
    unique_recommended = np.unique(top_k)
    
    total_unique_items = len(set(total_catalog_items))#set для уникальности
    
    if total_unique_items == 0:
        return 0.0
        
    return float(len(unique_recommended) / total_unique_items)

