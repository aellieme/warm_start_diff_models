import os
import time
import pandas as pd
import numpy as np
import torch
from polara import get_movielens_data

from data_utils import transform_indices, data_to_sequences
from evaluate_metrics import downvote_seen_items, topn_recommendations
from evaluate_topk_dp import compute_all_metrics
from training import sasrec_model_scoring

def load_amazon(dataset_name, data_dir='../data/amazon'):
    file_map = {
        'amazon_Baby':               'reviews_Baby_5.json',
        'amazon_Beauty':             'reviews_Beauty_5.json',
        'amazon_Sports_and_Outdoors':'reviews_Sports_and_Outdoors_5.json',
        'amazon_Toys_and_Games':     'reviews_Toys_and_Games_5.json'
    }
    fname = file_map[dataset_name]
    path = os.path.join(data_dir, fname)

    df = pd.read_json(path, lines=True)
    df = df.rename(columns={
        'reviewerID': 'userid',
        'asin': 'itemid',
        'unixReviewTime': 'timestamp'
    })
    df = df[['userid', 'itemid', 'timestamp']]
    return df

def prepare_data_and_description(dataset):
    if dataset == 'ml-1m':
        raw_data = get_movielens_data(include_time=True)
        userid_col = 'userid'
        itemid_col = 'movieid'
        time_col = 'timestamp'
    else:  # Amazon
        raw_data = load_amazon(dataset)
        userid_col = 'userid'
        itemid_col = 'itemid'
        time_col = 'timestamp'

    all_data, data_index = transform_indices(raw_data.copy(), userid_col, itemid_col)
    all_data_sorted = all_data.sort_values(time_col).reset_index(drop=True)

    T_valid = all_data_sorted[time_col].quantile(0.70)
    T_test  = all_data_sorted[time_col].quantile(0.80)

    train_data = all_data_sorted[all_data_sorted[time_col] <= T_valid].copy()
    future_data = all_data_sorted[all_data_sorted[time_col] > T_valid].copy()

    # Validation (T_valid < ts <= T_test)
    val_window = future_data[future_data[time_col] <= T_test].copy()
    val_seq_dict = (
        val_window.sort_values([userid_col, time_col])
        .groupby(userid_col)[itemid_col].apply(list)
        .to_dict()
    )

    val_inputs, val_targets, val_users = [], [], []
    for uid, user_future in future_data.groupby(userid_col):
        user_future = user_future.sort_values(time_col)
        items = user_future[itemid_col].tolist()
        times = user_future[time_col].tolist()
        # последний элемент до T_test
        target_idx = -1
        for i, t in enumerate(times):
            if t <= T_test:
                target_idx = i
        if target_idx == -1:
            continue
        target_item = items[target_idx]
        input_seq = items[:target_idx]
        val_inputs.append(input_seq)
        val_targets.append(target_item)
        val_users.append(uid)

    val_data = pd.DataFrame({
        userid_col: val_users,
        itemid_col: val_targets,
        'history': val_inputs
    })

    # Test (ts > T_test)
    test_data = all_data_sorted[all_data_sorted[time_col] > T_test].copy()
    train_val_data = all_data_sorted[all_data_sorted[time_col] <= T_test].copy()
    test_examples = []
    for uid, user_test in test_data.groupby(userid_col):
        user_test = user_test.sort_values(time_col)
        items = user_test[itemid_col].tolist()
        if len(items) == 0:
            continue
        target = items[-1]
        history = items[:-1]
        test_examples.append({
            userid_col: uid,
            itemid_col: target,
            'history': history
        })

    test_examples_df = pd.DataFrame(test_examples)

    data_description = {
        'users': data_index['users'].name,
        'items': data_index['items'].name,
        'order': time_col,
        'n_users': len(data_index['users']),
        'n_items': len(data_index['items']),
        'T_valid': T_valid,
        'T_test': T_test,
    }

    return (train_data, val_data, test_data, test_examples_df,
            data_index, data_description, userid_col, itemid_col, time_col,
            val_seq_dict, train_val_data)
 
 
def run_inference_pipeline(
    model,
    history_data,
    train_data,         
    test_examples,
    data_description,
    userid_col,
    itemid_col,
    time_col,
    val_seq_dict,     
    topn=10
):
    start_time = time.perf_counter()   
    train_users = set(train_data[userid_col].unique())
    history_sorted = history_data.sort_values([userid_col, time_col])
    # Получаем последовательности из history_data (train+adapt) в виде словаря {user: list}
    train_seq_dict = data_to_sequences(history_sorted, data_description)

    model.eval()
    device = next(model.parameters()).device
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    scores_list = []
    user_order = []
    targets_list = []

    with torch.no_grad():
        for _, row in test_examples.iterrows():
            uid = row[userid_col]
            if uid not in train_users:   # исключаем новых пользователей
                continue
            target = row[itemid_col]
            test_history = row['history']   
            # полная история = train/adapt  + тест до таргета
            # Полная история = train/adapt (из history_data) + валидационные + тестовые (до цели)
            train_history = train_seq_dict.get(uid, [])
            val_history = val_seq_dict.get(uid, [])
            full_history = train_history + val_history + test_history
            # full_history = train_seq_dict.get(uid, []) + test_history
            if len(full_history) == 0:
                continue
            seq_tensor = tensor(full_history)
            scores = model.score(seq_tensor).cpu().numpy()
            if scores.ndim == 2:
                scores = scores[0]
            # маскирую все просмотренные 
            seen = set(full_history)
            for it in seen:
                if it < len(scores):
                    scores[it] = -np.inf
            scores_list.append(scores)
            user_order.append(uid)
            targets_list.append(target)

    if not scores_list:
        return np.array([]), [], ([], [], [], [], []), 0.0

    scores = np.stack(scores_list)
    recs = topn_recommendations(scores, topn=topn)

    actual = [[t] for t in targets_list]
    predicted = recs.tolist()
    candidate_items = set(history_data[itemid_col].unique().tolist())
    # topN_list = [topn]
    topN_list = [10, 20, topn] if topn >= 20 else [topn]
    precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
        actual, predicted, topN_list, len(candidate_items), candidate_items=candidate_items
    )
    inference_time = time.perf_counter() - start_time
    print(f"DEBUG: inference_time = {inference_time:.10f}")
    return recs, user_order, (precisions, recalls, ndcgs, mrrs, covs), inference_time  


def load_movies_metadata(dataset, data_dir='../data/info'):
    """
    Загружает метаданные (базово - MovieLens-1M) из указанной папки,
    если папка отсутствует, автоматически скачивает и распаковывает архив.

    Параметры: data_dir : str - путь к папке, в которой лежит movies.dat
    Возвращает: movie_dict : dict - словарь movieId: title
    """
    if dataset != 'ml-1m':
        return None
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        os.system('wget -O ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip')
        os.system('unzip -o ml-1m.zip -d ../data/')
        # Перемещаем movies.dat в ../data/info, если требуется
        if not os.path.exists(os.path.join(data_dir, 'movies.dat')):
            os.system('mv ../data/ml-1m/movies.dat ' + data_dir)

    movies_meta = pd.read_csv(
        os.path.join(data_dir, 'movies.dat'),
        sep='::', engine='python',
        encoding='ISO-8859-1',
        names=['movieId', 'title', 'genres']
    )
    return dict(zip(movies_meta.movieId, movies_meta.title))


def decode_items(item_series, data_index, movie_dict=None):
    """
    Преобразование последовательности внутренних индексов модели в оригинальные id айтемов (базово - movieId)
    и, если передан файл индексы - названия (movie_dict), то преобразует в названия айтемов (фильмов)

    параметры: item_series : iterable - список внутренних индексов айтемов (int)
    data_index : dict - содержит 'items' — pandas.Index с оригинальными id (movieId)
    movie_dict : dict, optional - словарь movieId: title с названиями

    возвращает list - список оригинальных id или названий (если есть словарь id-название)
    """
    index_to_item = {i: movieid for i, movieid in enumerate(data_index['items'])}
    real_items = [index_to_item[i] for i in item_series]
    if movie_dict is not None:
        return [movie_dict[i] for i in real_items]
    return real_items


def print_example_user(
    example_user,
    filtered_user_order,
    sasrec_recs,
    train_data,
    test_examples,
    data_index,
    data_description,
    userid_col,
    itemid_col,
    time_col,
    dataset
):
    """
    Печатает рекомендации для одного пользователя
    Параметры:
    example_user : int - id пользоватеоя
    filtered_user_order : list - список пользователей, соответствующий строкам sasrec_recs
    sasrec_recs : np.ndarray - матрица рекомендаций
    train_data, adapt_data, test_last : pd.DataFrame - срезы
    data_index : dict
    data_description : dict
    userid_col, itemid_col, time_col : str - названия колонок
    """
    movie_dict = load_movies_metadata(dataset)
    # movie_dict = load_movies_metadata()

    user_train = train_data[train_data[userid_col] == example_user].sort_values(time_col)
    # user_adapt = adapt_data[adapt_data[userid_col] == example_user].sort_values(time_col)
    user_holdout = test_examples[test_examples[userid_col] == example_user]

    user_position = filtered_user_order.index(example_user)
    user_recs = sasrec_recs[user_position]

    train_movies = decode_items(user_train[itemid_col], data_index, movie_dict)
    # adapt_movies = decode_items(user_adapt[itemid_col], data_index, movie_dict)
    rec_movies = decode_items(user_recs, data_index, movie_dict)
    holdout_movie = decode_items(user_holdout[itemid_col], data_index, movie_dict)[0]

    print("\n Example user")
    print(f"User ID: {example_user}")
    print("Last 10 train movies:", train_movies[-10:])
    # print("Adaptation (warm‑start) movies:", adapt_movies)
    print("Recommended next movies:", rec_movies[:10])
    print("True next movie:", holdout_movie)
