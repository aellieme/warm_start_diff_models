import pandas as pd
import numpy as np
from polara import get_movielens_data
from evaluate_topk_dp import compute_all_metrics
import time

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

def print_final_results(title, topN_list, res):
    print(f"\nFINAL RESULTS FOR {title}:")
    header = f"{'K':<5} | {'Recall@K':<10} | {'NDCG@K':<10} | {'MRR@K':<10} | {'Coverage':<10} | {'Latency (s)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i, k in enumerate(topN_list):
        print(f"{k:<5} | {res['recalls'][i]:.6f} | {res['ndcgs'][i]:.6f} | {res['mrrs'][i]:.6f} | {res['covs'][i]:.6f} | {res['latencies'][i]:.6f}")



df = get_movielens_data(include_time=True)
df = df.sort_values('timestamp').reset_index(drop=True)

n_rows = len(df)
time_train = df['timestamp'].quantile(0.7)
time_val   = df['timestamp'].quantile(0.8)
time_adapt = df['timestamp'].quantile(0.9)

train_df = df[df['timestamp'] <= time_train].copy()
val_df   = df[(df['timestamp'] > time_train) & (df['timestamp'] <= time_val)].copy()
adapt_df = df[(df['timestamp'] > time_val) & (df['timestamp'] <= time_adapt)].copy()
test_df  = df[df['timestamp'] > time_adapt].copy()

# Проверка непересекаемости временных интервалов
assert train_df['timestamp'].max() <= val_df['timestamp'].min(), "Train и Val пересекаются по времени"
assert val_df['timestamp'].max() <= adapt_df['timestamp'].min(), "Val и Adapt пересекаются по времени"
assert adapt_df['timestamp'].max() <= test_df['timestamp'].min(), "Adapt и Test пересекаются по времени"

print("интервалы не пересекаются")

print(f"разбиение: Train: {len(train_df)}, Val: {len(val_df)}, Adapt: {len(adapt_df)}, Test: {len(test_df)}")

all_items_count = df['movieid'].nunique()

# список всех уникальных айтемов, которые были в трейне
item_catalog = train_df['movieid'].unique().tolist()

test_users = test_df['userid'].unique()
test_grouped = test_df.groupby('userid')['movieid'].apply(list).reindex(test_users).tolist()

history_baseline = pd.concat([train_df, val_df]).groupby('userid')['movieid'].apply(list)
history_adapt = pd.concat([train_df, val_df, adapt_df]).groupby('userid')['movieid'].apply(list)

users_hist_baseline = [history_baseline.get(u, []) for u in test_users]
users_hist_adapt = [history_adapt.get(u, []) for u in test_users]

TOP_K_LIST = [1, 10, 20, 50, 100]

results_baseline = run_experiment(users_hist_baseline, item_catalog, TOP_K_LIST, "BASELINE")
results_adapt = run_experiment(users_hist_adapt, item_catalog, TOP_K_LIST, "ADAPT")

print_final_results("RANDOM BASELINE", TOP_K_LIST, results_baseline)
print_final_results("RANDOM ADAPT", TOP_K_LIST, results_adapt)


# Импортируем новые модули
from evaluate_last import evaluate_last
from evaluate_successive import evaluate_successive

# Фиксируем генератор для воспроизводимости
rng = np.random.default_rng(42)

print("\n LAST (цель - последний элемент в тестовом периоде)")
last_base = evaluate_last(test_grouped, users_hist_baseline, item_catalog, rng)
last_adapt = evaluate_last(test_grouped, users_hist_adapt, item_catalog, rng)

print("\nBaseline (LAST):")
for metric, value in last_base.items():
    print(f"  {metric}: {value:.6f}")
print("\nAdapt (LAST):")
for metric, value in last_adapt.items():
    print(f"  {metric}: {value:.6f}")

print("\n" + "-"*40)
print(">>> SUCCESSIVE (каждый элемент - отдельная цель)")
succ_base = evaluate_successive(test_grouped, users_hist_baseline, item_catalog, rng)
succ_adapt = evaluate_successive(test_grouped, users_hist_adapt, item_catalog, rng)

print("\nBaseline (SUCCESSIVE):")
for metric, value in succ_base.items():
    print(f"  {metric}: {value:.6f}")
print("\nAdapt (SUCCESSIVE):")
for metric, value in succ_adapt.items():
    print(f"  {metric}: {value:.6f}")