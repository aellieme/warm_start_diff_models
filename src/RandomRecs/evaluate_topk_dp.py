# evaluate_topk_dp.py

import numpy as np

def precision_at_k(actual: list, predicted: list, k: int) -> float:
    precisions = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
        pred_row = predicted[i]
        if hasattr(pred_row, 'tolist'):
            pred_row = pred_row.tolist()
        top_k = pred_row[:k]
        hits = len(set(top_k) & ground_truth)
        precisions.append(hits / k)
    return float(np.mean(precisions)) if precisions else 0.0

def recall_at_k(actual: list, predicted: list, k: int) -> float:
    recalls = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
        pred_row = predicted[i]
        if hasattr(pred_row, 'tolist'):
            pred_row = pred_row.tolist()
        top_k = pred_row[:k]
        hits = len(set(top_k) & ground_truth)
        recalls.append(hits / len(ground_truth))
    return float(np.mean(recalls)) if recalls else 0.0

def mrr(actual: list, predicted: list, k: int) -> float:
    rr_scores = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
        pred_row = predicted[i]
        if hasattr(pred_row, 'tolist'):
            pred_row = pred_row.tolist()
        top_k = pred_row[:k]
        found = False
        for rank, item in enumerate(top_k, 1):
            if item in ground_truth:
                rr_scores.append(1.0 / rank)
                found = True
                break
        if not found:
            rr_scores.append(0.0)
    return float(np.mean(rr_scores)) if rr_scores else 0.0

def ndcg_at_k(actual: list, predicted: list, k: int) -> float:
    ndcg_scores = []
    for i in range(len(actual)):
        ground_truth = set(actual[i])
        if not ground_truth:
            continue
        pred_row = predicted[i]
        if hasattr(pred_row, 'tolist'):
            pred_row = pred_row.tolist()
        top_k = pred_row[:k]
        dcg = 0.0
        for rank, item in enumerate(top_k, 1):
            if item in ground_truth:
                dcg += 1.0 / np.log2(rank + 1)
        n_hits = min(len(ground_truth), k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, n_hits + 1))
        if idcg > 0:
            ndcg_scores.append(dcg / idcg)
        else:
            ndcg_scores.append(0.0)
    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0

def coverage(predicted: list, n_items: int, pad_token=-1, candidate_items=None) -> float:
    allowed_items = set(candidate_items) if candidate_items is not None else None
    all_recommended = set()
    for user_recs in predicted:
        if hasattr(user_recs, 'tolist'):
            user_recs = user_recs.tolist()
        for item in user_recs:
            if item != pad_token:
                if allowed_items is not None and item not in allowed_items:
                    continue
                all_recommended.add(item)
    denominator = len(allowed_items) if allowed_items is not None else n_items
    return len(all_recommended) / denominator if denominator > 0 else 0.0

def compute_all_metrics(actual: list, predicted: list, topN_list: list, n_items: int, candidate_items=None):
    precisions = []
    recalls = []
    ndcgs = []
    mrrs = []
    covs = []
    for k in topN_list:
        precisions.append(precision_at_k(actual, predicted, k))
        recalls.append(recall_at_k(actual, predicted, k))
        ndcgs.append(ndcg_at_k(actual, predicted, k))
        mrrs.append(mrr(actual, predicted, k))
        predicted_k = [rec[:k] for rec in predicted]
        covs.append(coverage(predicted_k, n_items, pad_token=-1, candidate_items=candidate_items))
    return precisions, recalls, ndcgs, mrrs, covs
