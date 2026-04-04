import pandas as pd
import numpy as np
from polara import get_movielens_data
from evaluate_topk_dp import compute_all_metrics
import time

def get_top_k_recommendations(user_histories, popular_items, k=20):
    recommendations = []
    for history in user_histories:
        seen_items = set(history)
        recs = []
        for item in popular_items:
            if item not in seen_items:
                recs.append(item)
            if len(recs) == k:
                break
        # Если не набрали k , дополняем нулями или популярными
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

time_train = df['timestamp'].quantile(0.7)
time_val   = df['timestamp'].quantile(0.8)
time_adapt = df['timestamp'].quantile(0.9)

train_df = df[df['timestamp'] <= time_train].copy()
val_df   = df[(df['timestamp'] > time_train) & (df['timestamp'] <= time_val)].copy()
adapt_df = df[(df['timestamp'] > time_val) & (df['timestamp'] <= time_adapt)].copy()
test_df  = df[df['timestamp'] > time_adapt].copy()

assert train_df['timestamp'].max() <= val_df['timestamp'].min(), "Train и Val пересекаются"
assert val_df['timestamp'].max() <= adapt_df['timestamp'].min(), "Val и Adapt пересекаются"
assert adapt_df['timestamp'].max() <= test_df['timestamp'].min(), "Adapt и Test пересекаются"

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Adapt: {len(adapt_df)}, Test: {len(test_df)}")
print(f"Train time: {train_df['timestamp'].min()} -> {train_df['timestamp'].max()}")
print(f"Test time:  {test_df['timestamp'].min()} -> {test_df['timestamp'].max()}")

all_items_count = df['movieid'].nunique()  # общее количество фильмов во всём датасете

# Популярность тольео на train_df
popularity_scores = train_df['movieid'].value_counts().index.tolist()

test_users = test_df['userid'].unique()
test_grouped = test_df.groupby('userid')['movieid'].apply(list).reindex(test_users).tolist()

full_history_before_adapt = pd.concat([train_df, val_df])
history_baseline = full_history_before_adapt.groupby('userid')['movieid'].apply(list)

full_history_with_adapt = pd.concat([train_df, val_df, adapt_df])
history_adapt = full_history_with_adapt.groupby('userid')['movieid'].apply(list)

users_hist_baseline = [history_baseline.get(u, []) for u in test_users]
users_hist_adapt = [history_adapt.get(u, []) for u in test_users]

K_MAX = 50
TOP_K_LIST = [1, 10, 20, 50]

# BASELINE 
print("\nBaseline Inference")
baseline_latencies = []
for k in TOP_K_LIST:
    start_time = time.perf_counter()
    preds_baseline = get_top_k_recommendations(users_hist_baseline, popularity_scores, k=k)
    end_time = time.perf_counter()
    baseline_latencies.append(end_time - start_time)
    
    # Для последнего K сохраняем предсказания для метрик
    if k == max(TOP_K_LIST):
        preds_baseline_final = preds_baseline

(_, recalls_baseline, ndcgs_baseline, mrrs_baseline, covs_baseline) = compute_all_metrics(test_grouped, preds_baseline_final, TOP_K_LIST, all_items_count)

# ADAPT
print("\nAdapt Inference")
adapt_latencies = []
for k in TOP_K_LIST:
    start_time = time.perf_counter()
    preds_adapt = get_top_k_recommendations(users_hist_adapt, popularity_scores, k=k)
    end_time = time.perf_counter()
    adapt_latencies.append(end_time - start_time)
    
    if k == max(TOP_K_LIST):
        preds_adapt_final = preds_adapt

(_, recalls_adapt, ndcgs_adapt, mrrs_adapt, covs_adapt) = compute_all_metrics(test_grouped, preds_adapt_final, TOP_K_LIST, all_items_count)

print_results("BASELINE", TOP_K_LIST, recalls_baseline, ndcgs_baseline, mrrs_baseline, covs_baseline, latencies=baseline_latencies)
print_results("ADAPT", TOP_K_LIST, recalls_adapt, ndcgs_adapt, mrrs_adapt, covs_adapt, latencies=adapt_latencies)
