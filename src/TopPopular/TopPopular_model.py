import pandas as pd
import numpy as np
from polara import get_movielens_data
from evaluate_topk_dp import compute_all_metrics
import time

df = get_movielens_data()

n_rows = len(df)
train_end = int(n_rows * 0.7)
val_end   = int(n_rows * 0.8)
adapt_end = int(n_rows * 0.9)

train_df = df.iloc[:train_end].copy()
val_df   = df.iloc[train_end:val_end].copy()
adapt_df = df.iloc[val_end:adapt_end].copy()
test_df  = df.iloc[adapt_end:].copy()

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Adapt: {len(adapt_df)}, Test: {len(test_df)}")

all_items_count = df['movieid'].nunique()

# считаем популярность только на train_df
popularity_scores = train_df['movieid'].value_counts().index.tolist()

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


test_users = test_df['userid'].unique()

test_grouped = test_df.groupby('userid')['movieid'].apply(list).reindex(test_users).tolist()

full_history_before_adapt = pd.concat([train_df, val_df])
history_baseline = full_history_before_adapt.groupby('userid')['movieid'].apply(list)

full_history_with_adapt = pd.concat([train_df, val_df, adapt_df])
history_adapt = full_history_with_adapt.groupby('userid')['movieid'].apply(list)

users_hist_baseline = [history_baseline.get(u, []) for u in test_users]
users_hist_adapt = [history_adapt.get(u, []) for u in test_users]

K_MAX = 20
TOP_K_LIST = [1, 10, 20, 50]


print("\n Baseline Inference")
preds_baseline = get_top_k_recommendations(users_hist_baseline, popularity_scores, k=K_MAX)

(precisions_baseline, recalls_baseline, ndcgs_baseline, mrrs_baseline, covs_baseline) = compute_all_metrics(test_grouped, preds_baseline, TOP_K_LIST, all_items_count)

print("\n Adapt Inference")
start_time = time.perf_counter()
preds_adapt = get_top_k_recommendations(users_hist_adapt, popularity_scores, k=K_MAX)
end_time = time.perf_counter()
total_latency = end_time - start_time

(precisions_adapt, recalls_adapt, ndcgs_adapt, mrrs_adapt, covs_adapt) = compute_all_metrics(test_grouped, preds_adapt, TOP_K_LIST, all_items_count)


def print_results(title, topN_list, recalls, ndcgs, mrrs, covs, latency=None):
    print(f"\nResults for {title}:")
    header = f"{'K':<5} | {'Recall@K':<10} | {'NDCG@K':<10} | {'MRR@K':<10} | {'Coverage':<10}"
    if latency is not None:
        header += f" | {'Latency (s)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, k in enumerate(topN_list):
        row = f"{k:<5} | {recalls[i]:.6f} | {ndcgs[i]:.6f} | {mrrs[i]:.6f} | {covs[i]:.6f}"
        if latency is not None:
            if i == 0:
                row += f" | {latency:.6f}"
            else:
                row += f" | {'—':<12}"
        print(row)

print_results("BASELINE", TOP_K_LIST,recalls_baseline, ndcgs_baseline,mrrs_baseline, covs_baseline)
print_results("ADAPT", TOP_K_LIST,recalls_adapt, ndcgs_adapt, mrrs_adapt, covs_adapt, latency=total_latency)


#TODO: calculate latency for each k