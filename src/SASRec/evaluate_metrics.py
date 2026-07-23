import numpy as np
import pandas as pd

def drop_invalid_items(history, candidate_items):
    allowed = set(candidate_items)
    return [int(item) for item in history if int(item) in allowed]

def mask_invalid_items(scores, history, candidate_items, pad_token):
    allowed = np.zeros(len(scores), dtype=bool)
    valid_candidates = [
        int(item) for item in candidate_items
        if 0 <= int(item) < len(scores) and int(item) != pad_token
    ]
    allowed[valid_candidates] = True
    scores[~allowed] = -np.inf
    seen = [int(item) for item in history if 0 <= int(item) < len(scores)]
    scores[seen] = -np.inf
    return scores

def downvote_seen_items(scores, data, data_description):
    userid = data_description['users']
    itemid = data_description['items']
    user_idx = data[userid].values
    item_idx = data[itemid].values
    user_idx, _ = pd.factorize(user_idx, sort=True)
    seen_idx_flat = np.ravel_multi_index((user_idx, item_idx), scores.shape)
    np.put(scores, seen_idx_flat, -np.inf)

def topk_recs_selection(scores, topn=10):
    recommendations = np.full((len(scores), topn), -1, dtype=np.int64)
    for row_index, row in enumerate(scores):
        finite = np.flatnonzero(np.isfinite(row))
        if finite.size == 0:
            continue
        take = min(topn, finite.size)
        local = np.argpartition(row[finite], -take)[-take:]
        ranked = finite[local[np.argsort(-row[finite][local])]]
        recommendations[row_index, :take] = ranked
    return recommendations

def model_evaluate(recommended_items, holdout, holdout_description, topn=10):
    itemid = holdout_description['items']
    holdout_items = holdout[itemid].values
    assert recommended_items.shape[0] == len(holdout_items)
    hits_mask = recommended_items[:, :topn] == holdout_items.reshape(-1, 1)
    hr = np.mean(hits_mask.any(axis=1))
    n_test_users = recommended_items.shape[0]
    hit_rank = np.where(hits_mask)[1] + 1.0
    mrr = np.sum(1 / hit_rank) / n_test_users
    n_items = holdout_description['n_items']
    cov = np.unique(recommended_items).size / n_items
    return hr, mrr, cov
