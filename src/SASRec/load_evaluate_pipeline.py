import os
import time
import pandas as pd
import numpy as np
import torch
from polara import get_movielens_data

from data_utils import transform_indices
from evaluate_metrics import downvote_seen_items, topn_recommendations
from evaluate_topk_dp import compute_all_metrics
from training import sasrec_model_scoring

def prepare_data_and_description():
    """
    Загружает данные MovieLens-1M, проводит сквозную переиндексацию пользователей и айтемов,
    сортирует все взаимодействия по времени, выполняет хронологическое разбиение на train, val, adapt, test.

    Returns
    -------
    train_data : pd.DataFrame
    val_data : pd.DataFrame
    adapt_data : pd.DataFrame
    test_data : pd.DataFrame
    test_last : pd.DataFrame
        Последнее взаимодействие каждого пользователя из test_data.
    data_index : dict
        Словарь с отображениями 'users' и 'items' (pandas.Index).
    data_description : dict
        Метаинформация о данных: названия колонок, количество уникальных пользователей и айтемов.
    userid_col : str
    itemid_col : str
    time_col : str
    """
    mldata = get_movielens_data(include_time=True)
    userid_col = 'userid'
    itemid_col = 'movieid'
    time_col = 'timestamp'

    all_data, data_index = transform_indices(mldata.copy(), userid_col, itemid_col)
    all_data_sorted = all_data.sort_values(time_col).reset_index(drop=True)

    # Ваши квантили
    quantile_valid = 0.70   # T_valid
    quantile_adapt = 0.80   # граница между validation и adapt
    quantile_test  = 0.90   # T_test

    T_valid = all_data_sorted[time_col].quantile(quantile_valid)
    T_adapt = all_data_sorted[time_col].quantile(quantile_adapt)
    T_test  = all_data_sorted[time_col].quantile(quantile_test)

    assert T_valid < T_adapt < T_test

    #  Обучающая выборка (до T_valid) 
    train_data = all_data_sorted[all_data_sorted[time_col] <= T_valid].copy()

    #  Данные после T_valid (для валидации, адаптации, теста) 
    future_data = all_data_sorted[all_data_sorted[time_col] > T_valid].copy()

    # Валидационные примеры (цель — последнее взаимодействие до T_adapt)
    val_inputs, val_targets, val_users = [], [], []
    for uid, user_future in future_data.groupby(userid_col):
        user_future = user_future.sort_values(time_col)
        items = user_future[itemid_col].tolist()
        times = user_future[time_col].tolist()

        # Находим последний элемент с timestamp <= T_adapt
        target_idx = -1
        for i, t in enumerate(times):
            if t <= T_adapt:
                target_idx = i
        if target_idx == -1:
            continue   # нет цели в валидационном окне

        target_item = items[target_idx]
        # Входная последовательность — все элементы до target (включая из того же окна, но до цели)
        input_seq = items[:target_idx]
        if len(input_seq) == 0:
            continue

        val_inputs.append(input_seq)
        val_targets.append(target_item)
        val_users.append(uid)

    val_data = pd.DataFrame({
        userid_col: val_users,
        itemid_col: val_targets,
        'history': val_inputs   # список индексов item_id
    })

    #  Адаптационная выборка (строго между T_adapt и T_test) 
    adapt_data = all_data_sorted[
        (all_data_sorted[time_col] > T_adapt) & (all_data_sorted[time_col] <= T_test)
    ].copy()

    #  Тестовые примеры (последнее взаимодействие после T_test) 
    test_data = all_data_sorted[all_data_sorted[time_col] > T_test].copy()
    test_last = (
        test_data
        .sort_values([userid_col, time_col])
        .groupby(userid_col)
        .last()
        .reset_index()
    )

    data_description = {
        'users': data_index['users'].name,
        'items': data_index['items'].name,
        'order': time_col,
        'n_users': len(data_index['users']),
        'n_items': len(data_index['items']),
        'T_valid': T_valid,
        'T_adapt': T_adapt,
        'T_test': T_test,
    }

    return (train_data, val_data, adapt_data, test_data, test_last,
            data_index, data_description, userid_col, itemid_col, time_col)
 


def run_inference_pipeline(
    model,
    history_data,
    train_data,
    test_last,
    data_description,
    userid_col,
    itemid_col,
    time_col,
    topn=10
):
    """
    Выполняет инференс и оценкe для заданной исторической последовательности.

    Параметры
    ---------
    model : torch.nn.Module
        Обученная модель SASRec.
    history_data : pd.DataFrame
        DataFrame с историей взаимодействий, используемый для построения предсказаний.
        Должен содержать колонки userid_col, itemid_col, time_col.
    train_data : pd.DataFrame
        Обучающая выборка (необходима для определения множества пользователей, присутствовавших в обучении).
    test_last : pd.DataFrame
        DataFrame с целевыми (последними) взаимодействиями для каждого пользователя.
    data_description : dict
        Метаданные датасета (из prepare_data_and_description).
    userid_col : str
    itemid_col : str
    time_col : str
    topn : int, default=10
        Размер списка рекомендаций.

    Returns
    -------
    sasrec_recs : np.ndarray
        Матрица рекомендаций (num_users, topn) с индексами айтемов.
    filtered_user_order : list
        Список идентификаторов пользователей в порядке, соответствующем строкам sasrec_recs.
    metrics : tuple
        Кортеж (precisions, recalls, ndcgs, mrrs, covs), где каждый элемент — список метрик для topn=[10].
    inference_time : float
        Время выполнения инференса (в секундах) без учёта вычисления метрик.
    """
    # Сортируем историю для корректной работы модели
    history_sorted = history_data.sort_values([userid_col, time_col])

    start_time = time.perf_counter()

    # Получение скоров от модели
    scores, user_order = sasrec_model_scoring(model, history_sorted, data_description)

    # Подавление уже просмотренных айтемов
    downvote_seen_items(scores, history_sorted, data_description)

    # Определяем пользователей, которые есть и в обучении, и в целевом наборе
    train_users = set(train_data[userid_col].unique())
    valid_users = set(test_last[userid_col].unique()).intersection(train_users)

    # Индексы пользователей в порядке user_order, удовлетворяющие условию
    valid_indices = [i for i, u in enumerate(user_order) if u in valid_users]
    scores = scores[valid_indices]
    filtered_user_order = [user_order[i] for i in valid_indices]

    # Упорядочиваем целевые взаимодействия в том же порядке
    holdout_ordered = test_last.set_index(userid_col).loc[filtered_user_order].reset_index()

    # Формируем top-n рекомендации
    recs = topn_recommendations(scores, topn=topn)

    inference_time = time.perf_counter() - start_time

    # Подготовка данных для метрик
    actual = [[row] for row in holdout_ordered[itemid_col].values]
    predicted = recs.tolist()
    topN_list = [topn]

    precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
        actual, predicted, topN_list, data_description['n_items']
    )

    return recs, filtered_user_order, (precisions, recalls, ndcgs, mrrs, covs), inference_time


def load_movies_metadata(data_dir='../data/info'):
    """
    Загружает метаданные фильмов MovieLens-1M из указанной директории.
    Если директория отсутствует, автоматически скачивает и распаковывает архив.

    Parameters
    ----------
    data_dir : str
        Путь к папке, содержащей movies.dat.

    Returns
    -------
    movie_dict : dict
        Словарь {movieId: title}.
    """
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
    Преобразует последовательность внутренних индексов модели в оригинальные movieId
    и, если передан movie_dict, в названия фильмов.

    Parameters
    ----------
    item_series : iterable
        Список внутренних индексов айтемов (int).
    data_index : dict
        Содержит 'items' — pandas.Index с оригинальными movieId.
    movie_dict : dict, optional
        Словарь {movieId: title}.

    Returns
    -------
    list
        Список оригинальных идентификаторов или названий.
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
    adapt_data,
    test_last,
    data_index,
    data_description,
    userid_col,
    itemid_col,
    time_col
):
    """
    Печатает подробную информацию о рекомендациях для одного пользователя.

    Parameters
    ----------
    example_user : int
        Идентификатор пользователя (внутренний индекс).
    filtered_user_order : list
        Список пользователей, соответствующий строкам sasrec_recs.
    sasrec_recs : np.ndarray
        Матрица рекомендаций.
    train_data, adapt_data, test_last : pd.DataFrame
        Соответствующие срезы данных.
    data_index : dict
        Индексные отображения.
    data_description : dict
        Метаинформация.
    userid_col, itemid_col, time_col : str
        Названия колонок.
    """
    movie_dict = load_movies_metadata()

    user_train = train_data[train_data[userid_col] == example_user].sort_values(time_col)
    user_adapt = adapt_data[adapt_data[userid_col] == example_user].sort_values(time_col)
    user_holdout = test_last[test_last[userid_col] == example_user]

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