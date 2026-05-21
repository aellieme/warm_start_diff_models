import pandas as pd
import numpy as np
import random
from polara import get_movielens_data
from evaluate_topk_dp import compute_all_metrics
import time

# ========== 1. Фиксация случайного seed = 42 ==========
random.seed(42)
np.random.seed(42)

def get_random_recommendations(user_histories, item_catalog, k=20, rng=None):
    """Генерация случайных рекомендаций с фиксированным seed."""
    if rng is None:
        rng = np.random.default_rng(42)
    recommendations = []
    for history in user_histories:
        seen_items = set(history)
        # случайная перестановка каталога
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

def run_experiment(histories, catalog, k_list, title, rng):
    max_k = max(k_list)
    start_time = time.perf_counter()
    preds_full = get_random_recommendations(histories, catalog, k=max_k, rng=rng)
    total_latency = time.perf_counter() - start_time

    results = {'recalls': [], 'ndcgs': [], 'mrrs': [], 'covs': [], 'latencies': []}
    for k in k_list:
        preds_k = [rec[:k] for rec in preds_full]   # обрезаем до k
        (_, recalls, ndcgs, mrrs, covs) = compute_all_metrics(test_grouped, preds_k, [k], all_items_count)
        results['recalls'].append(recalls[0])
        results['ndcgs'].append(ndcgs[0])
        results['mrrs'].append(mrrs[0])
        results['covs'].append(covs[0])
        results['latencies'].append(total_latency)   
    return results

def print_final_results(title, topN_list, res):
    print(f"\nFINAL RESULTS FOR {title}:")
    header = f"{'K':<5} | {'Recall@K':<10} | {'NDCG@K':<10} | {'MRR@K':<10} | {'Coverage':<10} | {'Latency (s)':<12}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for i, k in enumerate(topN_list):
        print(f"{k:<5} | {res['recalls'][i]:.6f} | {res['ndcgs'][i]:.6f} | {res['mrrs'][i]:.6f} | {res['covs'][i]:.6f} | {res['latencies'][i]:.6f}")

# ========== 2. Загрузка данных и GTS-сплит 70/10/20 (без адаптации) ==========
df = get_movielens_data(include_time=True)
df = df.sort_values('timestamp').reset_index(drop=True)

n_rows = len(df)
train_cutoff = df['timestamp'].quantile(0.7)   # 70%
val_cutoff   = df['timestamp'].quantile(0.8)   # 10% validation
# test — оставшиеся 20% (от 0.8 до 1.0)

train_df = df[df['timestamp'] <= train_cutoff].copy()
val_df   = df[(df['timestamp'] > train_cutoff) & (df['timestamp'] <= val_cutoff)].copy()
test_df  = df[df['timestamp'] > val_cutoff].copy()

# Проверка непересекаемости временных интервалов
assert train_df['timestamp'].max() <= val_df['timestamp'].min(), "Train и Val пересекаются по времени"
assert val_df['timestamp'].max() <= test_df['timestamp'].min(), "Val и Test пересекаются по времени"

print("Интервалы не пересекаются")
print(f"Разбиение: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

all_items_count = df['movieid'].nunique()
# item_catalog = train_df['movieid'].unique().tolist()
# item_catalog = set(train_df['movieid'].unique()) | set(val_df['movieid'].unique())
# item_catalog = list(item_catalog)
all_items = df['movieid'].unique().tolist()   # все фильмы из всего датасета
item_catalog = all_items
# Список пользователей теста (только те, у которых есть взаимодействия в test_df)
test_users = test_df['userid'].unique()
test_grouped = test_df.groupby('userid')['movieid'].apply(list).reindex(test_users).tolist()

# История для baseline: train + validation (без адаптации!)
history_baseline = pd.concat([train_df, val_df]).groupby('userid')['movieid'].apply(list)
users_hist_baseline = [history_baseline.get(u, []) for u in test_users]

# ========== 3. Запуск экспериментов (только baseline) ==========
TOP_K_LIST = [1, 10, 20, 50, 100]
rng = np.random.default_rng(42)   # фиксированный seed для воспроизводимости

results_baseline = run_experiment(users_hist_baseline, item_catalog, TOP_K_LIST, "BASELINE", rng)
print_final_results("RANDOM BASELINE", TOP_K_LIST, results_baseline)


