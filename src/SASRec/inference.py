import pandas as pd
import numpy as np
import torch
from polara import get_movielens_data
from polara.preprocessing.dataframes import reindex

from data_utils import (
    transform_indices, split_per_user_leave_k, split_future_for_eval
)
from evaluate_metrics import downvote_seen_items, topn_recommendations, model_evaluate
from training import sasrec_model_scoring   
from model import load_sasrec_model, get_latest_model_path 


def main():
    #  Загрузка и подготовка данных 
    mldata = get_movielens_data(include_time=True)
    userid_col = 'userid'
    itemid_col = 'movieid'
    time_col = 'timestamp'

    train_raw, future_raw = split_per_user_leave_k(mldata, user_col=userid_col, time_col=time_col, k=3)

    train_data, data_index = transform_indices(train_raw.copy(), userid_col, itemid_col)
    future_data = reindex(future_raw, data_index['items'])
    future_data = future_data[future_data[userid_col].isin(data_index['users'])]

    adapt_data, holdout_data = split_future_for_eval(future_data, user_col=userid_col, time_col=time_col)

    data_description = dict(
        users=data_index['users'].name,
        items=data_index['items'].name,
        order=time_col,
        n_users=len(data_index['users']),
        n_items=len(data_index['items']),
    )
    print("Data description:", data_description)

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

    # Adapt инференс 
    inference_history = pd.concat([train_data, adapt_data], ignore_index=True)
    inference_history = inference_history.sort_values([userid_col, time_col])

    sasrec_scores, user_order = sasrec_model_scoring(model, inference_history, data_description)

    # Убираем просмотренные айтемы
    downvote_seen_items(sasrec_scores, inference_history, data_description)

    # Фильтруем только пользователей, присутствующих в holdout
    valid_users = set(holdout_data[userid_col].unique())
    valid_indices = [i for i, u in enumerate(user_order) if u in valid_users]
    sasrec_scores = sasrec_scores[valid_indices]
    filtered_user_order = [user_order[i] for i in valid_indices]

    holdout_ordered = (
        holdout_data.set_index(userid_col)
        .loc[filtered_user_order]
        .reset_index()
    )

    # Рекомендации top‑10 и оценка
    sasrec_recs = topn_recommendations(sasrec_scores, topn=10)
    hr, mrr, cov = model_evaluate(sasrec_recs, holdout_ordered, data_description, topn=10)
    print(f"Evaluated users: {len(filtered_user_order)}")
    print(f"HR: {hr:.4f}, MRR: {mrr:.4f}, Coverage: {cov:.4f}")

    # Пример для одного юзера
    import os
    if not os.path.exists('ml-1m'):
        os.system('wget -O ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip')
        os.system('unzip -o ml-1m.zip')
    movies_meta = pd.read_csv(
        'ml-1m/movies.dat', sep='::', engine='python',
        encoding='ISO-8859-1', names=['movieId', 'title', 'genres']
    )
    movie_dict = dict(zip(movies_meta.movieId, movies_meta.title))

    def decode_items(item_series, data_index, movie_dict=None):
        index_to_item = {i: movieid for i, movieid in enumerate(data_index['items'])}
        real_items = [index_to_item[i] for i in item_series]
        if movie_dict is not None:
            return [movie_dict[i] for i in real_items]
        return real_items

    example_user = filtered_user_order[0]
    user_train = train_data[train_data[userid_col] == example_user].sort_values(time_col)
    user_adapt = adapt_data[adapt_data[userid_col] == example_user].sort_values(time_col)
    user_holdout = holdout_data[holdout_data[userid_col] == example_user]
    user_position = filtered_user_order.index(example_user)
    user_recs = sasrec_recs[user_position]

    train_movies = decode_items(user_train[itemid_col], data_index, movie_dict)
    adapt_movies = decode_items(user_adapt[itemid_col], data_index, movie_dict)
    rec_movies = decode_items(user_recs, data_index, movie_dict)
    holdout_movie = decode_items(user_holdout[itemid_col], data_index, movie_dict)[0]

    print("\n--- Example user ---")
    print(f"User ID: {example_user}")
    print("Last 10 train movies:", train_movies[-10:])
    print("Adaptation (warm‑start) movies:", adapt_movies)
    print("Recommended next movies:", rec_movies[:10])
    print("True next movie:", holdout_movie)

if __name__ == "__main__":
    main()