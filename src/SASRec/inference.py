import pandas as pd
import numpy as np
import torch
from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex
import time

from data_utils import (
    transform_indices, split_per_user_leave_k, split_future_for_eval
)
from evaluate_metrics import downvote_seen_items, topn_recommendations, model_evaluate
from training import sasrec_model_scoring   
from model import load_sasrec_model, get_latest_model_path 
from evaluate_topk_dp import compute_all_metrics

def main():
    #  Загрузка и подготовка данных 
    mldata = get_movielens_data(include_time=True)
    userid_col = 'userid'
    itemid_col = 'movieid'
    time_col = 'timestamp'

    all_data, data_index = transform_indices(mldata.copy(), userid_col, itemid_col)
    all_data_sorted = all_data.sort_values(time_col).reset_index(drop=True)
    N = len(all_data_sorted)

    train_end = int(0.7 * N)
    val_end   = int(0.8 * N)
    adapt_end = int(0.9 * N)

    train_data = all_data_sorted.iloc[:train_end].copy()
    adapt_data = all_data_sorted.iloc[val_end:adapt_end].copy()
    test_data  = all_data_sorted.iloc[adapt_end:].copy()
    test_last = test_data.sort_values([userid_col, time_col]).groupby(userid_col).last().reset_index()

    data_description = dict(
        users=data_index['users'].name,
        items=data_index['items'].name,
        order=time_col,
        n_users=len(data_index['users']),
        n_items=len(data_index['items']),
        )

    #  Загрузка сохранённой модели 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model, config, _, _ = load_sasrec_model('sasrec_checkpoint.pt', device=device)
    # model, config, _, _ = load_sasrec_model('sasrec_checkpoint.pt')  
    
     # Загрузка последней сохранённой модели
    from model import load_sasrec_model, get_latest_model_path
    model_path = get_latest_model_path()
    print(f"Loading latest model from {model_path}")
    model, config, _, _ = load_sasrec_model(model_path)
    print(f"Loaded config: {config}")
    
    
    print("Model loaded from sasrec_checkpoint.pt")
    print(f"Loaded config: {config}")


    # ========== BASELINE: только train_data ==========
    print("\nbaseline")
    history_baseline = train_data.sort_values([userid_col, time_col])

    start_time = time.perf_counter()
    sasrec_scores, user_order = sasrec_model_scoring(model, history_baseline, data_description)
    downvote_seen_items(sasrec_scores, history_baseline, data_description)

    train_users = set(train_data[userid_col].unique())
    valid_users = set(test_last[userid_col].unique()).intersection(train_users)

    # valid_users = set(test_last[userid_col].unique())
    valid_indices = [i for i, u in enumerate(user_order) if u in valid_users]
    sasrec_scores = sasrec_scores[valid_indices]
    filtered_user_order = [user_order[i] for i in valid_indices]

    holdout_ordered = test_last.set_index(userid_col).loc[filtered_user_order].reset_index()
    sasrec_recs = topn_recommendations(sasrec_scores, topn=10)
    inference_time = time.perf_counter() - start_time

    print(f"Total inference time: {inference_time:.4f} sec")
    actual = [[row] for row in holdout_ordered[itemid_col].values]
    predicted = sasrec_recs.tolist()
    precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(actual, predicted, [10], data_description['n_items'])
    print(f"Evaluated users: {len(filtered_user_order)}")
    for k, p, r, ndcg, mrr, cov in zip([10], precisions, recalls, ndcgs, mrrs, covs):
        print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")

    
    # Adapt инференс 
    print("\nadaptation")
    inference_history = pd.concat([train_data, adapt_data], ignore_index=True)
    inference_history = inference_history.sort_values([userid_col, time_col])

    start_time = time.perf_counter()
    sasrec_scores, user_order = sasrec_model_scoring(model, inference_history, data_description)

    # Убираем просмотренные айтемы
    downvote_seen_items(sasrec_scores, inference_history, data_description)

    # Фильтруем только пользователей, присутствующих в holdout
    # valid_users = set(test_last[userid_col].unique())
    train_users = set(train_data[userid_col].unique())
    valid_users = set(test_last[userid_col].unique()).intersection(train_users)
    valid_indices = [i for i, u in enumerate(user_order) if u in valid_users]
    sasrec_scores = sasrec_scores[valid_indices]
    filtered_user_order = [user_order[i] for i in valid_indices]

    holdout_ordered = test_last.set_index(userid_col).loc[filtered_user_order].reset_index()

    # Рекомендации top‑10 и оценка
    sasrec_recs = topn_recommendations(sasrec_scores, topn=10)
    inference_time = time.perf_counter() - start_time
    print(f"Total inference time: {inference_time:.4f} sec")
    
    # holdout_ordered — DataFrame с колонкой itemid (целевые айтемы)
    actual = [[row] for row in holdout_ordered[itemid_col].values]   # список списков с одним айтемом
    predicted = sasrec_recs.tolist()                                 # список списков (top‑10)
    n_items = data_description['n_items']
    topN_list = [10]   # какие k вас интересуют
    precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(actual, predicted, topN_list, n_items)
    print(f"Evaluated users: {len(filtered_user_order)}")
    for k, p, r, ndcg, mrr, cov in zip(topN_list, precisions, recalls, ndcgs, mrrs, covs):
        print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")
    
    # hr, mrr, cov = model_evaluate(sasrec_recs, holdout_ordered, data_description, topn=10)
    # print(f"Evaluated users: {len(filtered_user_order)}")
    # print(f"HR: {hr:.4f}, MRR: {mrr:.4f}, Coverage: {cov:.4f}")

    # Пример для одного юзера
    import os
    if not os.path.exists('../data/info'):
        os.system('wget -O ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip')
        os.system('unzip -o ml-1m.zip')
    movies_meta = pd.read_csv(
        '../data/info/movies.dat', sep='::', engine='python',
        encoding='ISO-8859-1', names=['movieId', 'title', 'genres']
    )
    movie_dict = dict(zip(movies_meta.movieId, movies_meta.title))

    def decode_items(item_series, data_index, movie_dict=None):
        index_to_item = {i: movieid for i, movieid in enumerate(data_index['items'])}
        real_items = [index_to_item[i] for i in item_series]
        if movie_dict is not None:
            return [movie_dict[i] for i in real_items]
        return real_items

    example_user = filtered_user_order[1]
    user_train = train_data[train_data[userid_col] == example_user].sort_values(time_col)
    user_adapt = adapt_data[adapt_data[userid_col] == example_user].sort_values(time_col)
    
    user_holdout = test_last[test_last[userid_col] == example_user]
    
    # user_holdout = holdout_data[holdout_data[userid_col] == example_user]
    user_position = filtered_user_order.index(example_user)
    user_recs = sasrec_recs[user_position]

    train_movies = decode_items(user_train[itemid_col], data_index, movie_dict)
    adapt_movies = decode_items(user_adapt[itemid_col], data_index, movie_dict)
    rec_movies = decode_items(user_recs, data_index, movie_dict)
    # holdout_movie = decode_items(user_holdout[itemid_col], data_index, movie_dict)[0]
    holdout_movie = decode_items(user_holdout[itemid_col], data_index, movie_dict)[0]

    print("\n--- Example user ---")
    print(f"User ID: {example_user}")
    print("Last 10 train movies:", train_movies[-10:])
    print("Adaptation (warm‑start) movies:", adapt_movies)
    print("Recommended next movies:", rec_movies[:10])
    print("True next movie:", holdout_movie)

if __name__ == "__main__":
    main()