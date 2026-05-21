import pandas as pd
import numpy as np
import random
from polara import get_movielens_data
from evaluate_topk_dp import compute_all_metrics
import time

random.seed(42)
np.random.seed(42)

def get_top_k_recommendations(user_histories, popular_items, k=20):
    """
    Генерирует top-k рекомендаций на основе глобальной популярности.
    Исключает уже просмотренные пользователем items.
    """
    recommendations = []
    for history in user_histories:
        seen_items = set(history)
        recs = []
        for item in popular_items:
            if item not in seen_items:
                recs.append(item)
            if len(recs) == k:
                break
        # Если не набрали k, дополняем -1
        while len(recs) < k:
            recs.append(-1)
        recommendations.append(recs)
    return recommendations

def print_results(title, topN_list, recalls, ndcgs, mrrs, covs, latencies=None):
    print(f"\nResults for {title}:")
    header = f"{'K':<5} | {'Recall@K':<10} | {'NDCG@K':<10} | {'MRR@K':<10} | {'Coverage':<10}"
    if latencies is not None:
        header += f" | {'Latency (s)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, k in enumerate(topN_list):
        row = f"{k:<5} | {recalls[i]:.6f} | {ndcgs[i]:.6f} | {mrrs[i]:.6f} | {covs[i]:.6f}"
        if latencies is not None:
            row += f" | {latencies[i]:.6f}"
        print(row)

df = get_movielens_data(include_time=True)
df = df.sort_values('timestamp').reset_index(drop=True)

train_cutoff = df['timestamp'].quantile(0.7)   # 70%
val_cutoff   = df['timestamp'].quantile(0.8)   # 10% validation
# test — оставшиеся 20% (от 0.8 до 1.0)

train_df = df[df['timestamp'] <= train_cutoff].copy()
val_df   = df[(df['timestamp'] > train_cutoff) & (df['timestamp'] <= val_cutoff)].copy()
test_df  = df[df['timestamp'] > val_cutoff].copy()

# Проверка непересекаемости временных интервалов
assert train_df['timestamp'].max() <= val_df['timestamp'].min(), "Train и Val пересекаются по времени"
assert val_df['timestamp'].max() <= test_df['timestamp'].min(), "Val и Test пересекаются по времени"

print(f"Разбиение: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Train time: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
print(f"Test time:  {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")

all_items_count = df['movieid'].nunique()  # общее количество фильмов во всём датасете

# popularity_scores = train_df['movieid'].value_counts().index.tolist()
popularity_scores = pd.concat([train_df, val_df])['movieid'].value_counts().index.tolist()

test_users = test_df['userid'].unique()
test_grouped = test_df.groupby('userid')['movieid'].apply(list).reindex(test_users).tolist()

history_baseline = pd.concat([train_df, val_df]).groupby('userid')['movieid'].apply(list)
users_hist_baseline = [history_baseline.get(u, []) for u in test_users]

TOP_K_LIST = [1, 10, 20, 50, 100]
print("\n Inference (train+val → test)")

MMAX_K = max(TOP_K_LIST)
start_time = time.perf_counter()
preds_full = get_top_k_recommendations(users_hist_baseline, popularity_scores, k=MAX_K)
total_latency = time.perf_counter() - start_time

(_, recalls, ndcgs, mrrs, covs) = compute_all_metrics(
    test_grouped, preds_full, TOP_K_LIST, all_items_count
)
latencies = [total_latency] * len(TOP_K_LIST)

print_results("Top-Popular", TOP_K_LIST,
              recalls, ndcgs, mrrs, covs,
              latencies=latencies)