import os
import argparse
import random
import time
import json
import gzip
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from evaluate_topk_dp import compute_all_metrics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tools.experiment_tracking import ExperimentTracker, recommendation_popularity, save_dataset_popularity
from experiment_tools.warm_start import build_last_item_examples

random.seed(42)
np.random.seed(42)

def load_movielens_local(data_dir='../data/info'):
    ratings_path = os.path.join(data_dir, 'ratings.dat')
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.dat not found at {ratings_path}")
    df = pd.read_csv(ratings_path, sep='::', engine='python',
                     names=['userid', 'movieid', 'rating', 'timestamp'])
    df = df[['userid', 'movieid', 'timestamp']]
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['userid'] = user_enc.fit_transform(df['userid'])
    df['movieid'] = item_enc.fit_transform(df['movieid']) + 1
    return df

def load_amazon(dataset_name, data_dir='../data/amazon'):
    file_map = {
        'baby':   'reviews_Baby_5.json',
        'beauty': 'reviews_Beauty_5.json',
        'sports': 'reviews_Sports_and_Outdoors_5.json',
        'toys':   'reviews_Toys_and_Games_5.json'
    }
    fname = file_map.get(dataset_name)
    if fname is None:
        raise ValueError(f"Unknown Amazon dataset: {dataset_name}")
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Amazon data not found: {path}")

    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df = df.rename(columns={
        'reviewerID': 'userid',
        'asin': 'movieid',
        'unixReviewTime': 'timestamp'
    })
    df = df[['userid', 'movieid', 'timestamp']]
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['userid'] = user_enc.fit_transform(df['userid'])
    df['movieid'] = item_enc.fit_transform(df['movieid']) + 1
    return df

def global_temporal_split(df, train_ratio=0.7, val_ratio=0.1):
    df = df.sort_values('timestamp').reset_index(drop=True)
    total = len(df)
    train_cutoff = df['timestamp'].quantile(train_ratio)
    val_cutoff   = df['timestamp'].quantile(train_ratio + val_ratio)

    train_df = df[df['timestamp'] <= train_cutoff].copy()
    val_df   = df[(df['timestamp'] > train_cutoff) & (df['timestamp'] <= val_cutoff)].copy()
    test_df  = df[df['timestamp'] > val_cutoff].copy()

    print(f"Split: Train {len(train_df)} ({len(train_df)/total:.1%}), "
          f"Val {len(val_df)} ({len(val_df)/total:.1%}), "
          f"Test {len(test_df)} ({len(test_df)/total:.1%})")
    all_items = df['movieid'].unique().tolist()
    return train_df, val_df, test_df, all_items

def get_random_recommendations(user_histories, item_catalog, k=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    recommendations = []
    for history in user_histories:
        seen = set(history)
        shuffled = rng.permutation(item_catalog)
        recs = []
        for item in shuffled:
            if item not in seen:
                recs.append(item)
            if len(recs) == k:
                break
        while len(recs) < k:
            recs.append(-1)
        recommendations.append(recs)
    return recommendations

def run_experiment(histories, ground_truth, catalog, k_list, rng):
    max_k = max(k_list)
    start_time = time.perf_counter()
    preds_full = get_random_recommendations(histories, catalog, k=max_k, rng=rng)
    total_latency = time.perf_counter() - start_time

    results = {'recalls': [], 'ndcgs': [], 'mrrs': [], 'covs': [], 'latencies': []}
    for k in k_list:
        preds_k = [rec[:k] for rec in preds_full]
        (_, recalls, ndcgs, mrrs, covs) = compute_all_metrics(
            ground_truth, preds_k, [k], len(catalog), candidate_items=catalog
        )
        results['recalls'].append(recalls[0])
        results['ndcgs'].append(ndcgs[0])
        results['mrrs'].append(mrrs[0])
        results['covs'].append(covs[0])
        results['latencies'].append(total_latency)
    results['predictions'] = preds_full
    return results

def print_results(title, topN_list, res):
    header = f"{'K':<5} | {'Recall@K':<10} | {'NDCG@K':<10} | {'MRR@K':<10} | {'Coverage':<10} | {'Latency (s)':<12}"
    print(header)
    print("-" * len(header))
    for i, k in enumerate(topN_list):
        print(f"{k:<5} | {res['recalls'][i]:.6f} | {res['ndcgs'][i]:.6f} | {res['mrrs'][i]:.6f} | {res['covs'][i]:.6f} | {res['latencies'][i]:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        choices=['ml-1m', 'baby', 'beauty', 'sports', 'toys'])
    parser.add_argument('--topk_list', nargs='+', type=int, default=[10,20,100])
    args = parser.parse_args()

    if args.dataset == 'ml-1m':
        print("Loading MovieLens-1M...")
        data = load_movielens_local(data_dir='../data/info')
    else:
        print(f"Loading Amazon {args.dataset}...")
        data = load_amazon(args.dataset, data_dir='../data/amazon')

    train_df, val_df, test_df, _ = global_temporal_split(data)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    item_catalog = train_val_df['movieid'].unique().tolist()

    test_users, user_histories, ground_truth = build_last_item_examples(
        train_val_df, test_df, 'userid', 'movieid', 'timestamp', item_catalog
    )

    rng = np.random.default_rng(42)
    results = run_experiment(
        user_histories, ground_truth, item_catalog, args.topk_list, rng
    )
    print_results(f"Random ({args.dataset})", args.topk_list, results)
    dataset_name = {"baby": "amazon_Baby", "toys": "amazon_Toys_and_Games"}.get(args.dataset, args.dataset)
    popularity = train_val_df['movieid'].value_counts().to_dict()
    save_dataset_popularity(dataset_name, popularity)
    tracker = ExperimentTracker(dataset_name, "RandomRecs")
    tracker.log_final_metrics(
        {k: {"recall": results['recalls'][i], "ndcg": results['ndcgs'][i],
             "mrr": results['mrrs'][i], "coverage": results['covs'][i]}
         for i, k in enumerate(args.topk_list)},
        split="global_temporal_70_10_20", mask_seen=True, seed=42,
        inference_total_sec=results['latencies'][0],
        n_users=len(test_users), maxlen=None,
        ranking_protocol="warm_start_known_catalog_v2",
        popularity_bias=recommendation_popularity(results['predictions'], popularity, args.topk_list),
    )
    tracker.close()
