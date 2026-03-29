# src\evaluate\evaluate_topk_dp.py

import numpy as np

def precision_at_k(actual: list, predicted: list, k: int) -> float:
    """
    какая доля из рекомендованных k айтемов реально релевантна
    actual: list of lists (у каждого пользователя свой список правильных id)
    predicted: np.ndarray [n_users, max_k]
    """
    precisions = []
    for i in range(len(actual)):
        gt = set(actual[i])
        if not gt:
            continue
        top_k = predicted[i][:k]          
        hits = len(set(top_k) & gt)
        precisions.append(hits / k)
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(actual: list, predicted: np.ndarray, k: int) -> float:
    """
    какую долю из всех релевантных айтемов мы смогли найти в топ-k
    actual: list of lists
    predicted: np.ndarray [n_users, max_k]
    """
    recalls = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
            
        top_k = predicted[i][:k]
        hits = len(set(top_k) & ground_truth)
        recalls.append(hits / len(ground_truth))
        
    return float(np.mean(recalls)) if recalls else 0.0

def mrr(actual: list, predicted: np.ndarray, k: int) -> float:
    """
    срзнач, обратное рангу первого найденного релевантного айтема
    """
    rr_scores = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
            
        top_k = predicted[i][:k]
        found = False
        for rank, item in enumerate(top_k, 1):
            if item in ground_truth:
                rr_scores.append(1.0 / rank)
                found = True
                break
        if not found:
            rr_scores.append(0.0)
            
    return float(np.mean(rr_scores)) if rr_scores else 0.0

def ndcg_at_k(actual: list, predicted: np.ndarray, k: int) -> float:
    ndcg_scores = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
            
        top_k = predicted[i][:k]
        
        # DCG
        dcg = 0.0
        for rank, item in enumerate(top_k, 1):
            if item in ground_truth:
                dcg += 1.0 / np.log2(rank + 1)
        
        # IDCG (когда все наши попадания стоят на первых местах)
        n_relevant = len(ground_truth)
        n_hits = min(n_relevant, k) 
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, n_hits + 1))
        
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
            
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

# def catalog_coverage(predicted: np.ndarray, total_items: set, k: int) -> float:
#     """
#     процент айтемов из каталога, которые были рекомендованы хотя бы раз
#     """
#     top_k = predicted[:, :k]
#     unique_recommended = set(top_k.flatten())
    
#     if not total_items:
#         return 0.0
        
#     return len(unique_recommended) / len(total_items)

def calculate_catalog_coverage(predicted: list, total_items: set, k: int) -> float:
    """
    predicted: list of lists (рекомендованные id для каждого пользователя)
    total_items: set всех id предметов в каталоге
    """
    unique_recommended = set()
    for user_recs in predicted:
        unique_recommended.update(user_recs[:k])
    if not total_items:
        return 0.0
    return len(unique_recommended) / len(total_items)