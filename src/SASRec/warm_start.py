import numpy as np


def filter_history_to_candidates(history, candidate_items):
    """Keep chronological history events that belong to the known catalogue."""
    allowed = set(candidate_items)
    return [int(item) for item in history if int(item) in allowed]


def is_eligible_warm_start_example(history, target, candidate_items):
    """Require a known target and at least one known history event."""
    allowed = set(candidate_items)
    return bool(history) and int(target) in allowed


def mask_ranking_scores(scores, history, candidate_items, pad_token):
    """Mask padding, unavailable items and observed history in-place."""
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


def topn_from_masked_scores(scores, topn):
    """Return only finite candidates and pad short rows with sentinel -1."""
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
