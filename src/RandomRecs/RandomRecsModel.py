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

print(f"Разбиение: Train: {len(train_df)}, Val: {len(val_df)}, Adapt: {len(adapt_df)}, Test: {len(test_df)}")

all_items_count = df['movieid'].nunique()

# список всех уникальных айтемов, которые были в трейне
item_catalog = train_df['movieid'].unique().tolist()

def get_random_recommendations(user_histories, item_catalog, k=20):
    recommendations = []
    rng = np.random.default_rng() 
    
    for history in user_histories:
        seen_items = set(history)
        # случайная перестановкa каталога
        shuffled_items = rng.permutation(item_catalog)
        recs = []
        for item in shuffled_items:
            if item not in seen_items:
                recs.append(item)
            if len(recs) == k:
                break
        
        while len(recs) < k:
            recs.append(-1)
        recommendations.append(recs)
    return recommendations


test_users = test_df['userid'].unique()
test_grouped = test_df.groupby('userid')['movieid'].apply(list).reindex(test_users).tolist()

history_baseline = pd.concat([train_df, val_df]).groupby('userid')['movieid'].apply(list)
history_adapt = pd.concat([train_df, val_df, adapt_df]).groupby('userid')['movieid'].apply(list)

users_hist_baseline = [history_baseline.get(u, []) for u in test_users]
users_hist_adapt = [history_adapt.get(u, []) for u in test_users]

TOP_K_LIST = [1, 10, 20, 50]


def run_experiment(histories, catalog, k_list, title):
    print(f"\n--- Running {title} Inference ---")
    
    # Словари для хранения результатов для каждого K
    results = {
        'recalls': [], 'ndcgs': [], 'mrrs': [], 'covs': [], 'latencies': []
    }
    
    for k in k_list:
        # Замер latency конкретно для этого K
        start_time = time.perf_counter()
        preds = get_random_recommendations(histories, catalog, k=k)
        end_time = time.perf_counter()
        
        latency = end_time - start_time
        
        (_, recalls, ndcgs, mrrs, covs) = compute_all_metrics(
            test_grouped, preds, [k], all_items_count
        )
        
        results['recalls'].append(recalls[0])
        results['ndcgs'].append(ndcgs[0])
        results['mrrs'].append(mrrs[0])
        results['covs'].append(covs[0])
        results['latencies'].append(latency)
        
        print(f"K={k} processed. Latency: {latency:.6f}s")
        
    return results

results_baseline = run_experiment(users_hist_baseline, item_catalog, TOP_K_LIST, "BASELINE")
results_adapt = run_experiment(users_hist_adapt, item_catalog, TOP_K_LIST, "ADAPT")


def print_final_results(title, topN_list, res):
    print(f"\nFINAL RESULTS FOR {title}:")
    header = f"{'K':<5} | {'Recall@K':<10} | {'NDCG@K':<10} | {'MRR@K':<10} | {'Coverage':<10} | {'Latency (s)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, k in enumerate(topN_list):
        print(f"{k:<5} | {res['recalls'][i]:.6f} | {res['ndcgs'][i]:.6f} | {res['mrrs'][i]:.6f} | {res['covs'][i]:.6f} | {res['latencies'][i]:.6f}")

print_final_results("RANDOM BASELINE", TOP_K_LIST, results_baseline)
print_final_results("RANDOM ADAPT", TOP_K_LIST, results_adapt)