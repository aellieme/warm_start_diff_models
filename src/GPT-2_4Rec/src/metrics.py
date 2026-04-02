"""
Metrics.
"""

import pandas as pd
# from recommenders.evaluation import python_evaluation
from evaluate_topk_dp import compute_all_metrics


DEFAULT_METRICS_OVERALL = ['ndcg_at_k', 'recall_at_k', 'mrr_at_k', 'precision_at_k']
DFAULT_METRICS_BEYOND_ACCURACY = ['catalog_coverage']
DEFAULT_METRICS_BY_TIME_IDX = ['ndcg_at_k', 'recall_at_k']

# DEFAULT_METRICS_OVERALL = ['ndcg_at_k', 'recall_at_k', 'mrr_at_k', 'precision_at_k']
# DFAULT_METRICS_BEYOND_ACCURACY = ['catalog_coverage']

METRIC_NAMES = {
    # 'map_at_k': 'map',
    'ndcg_at_k': 'ndcg',
    'recall_at_k': 'recall',
    'precision_at_k': 'precision',
    'catalog_coverage': 'coverage',
    'mrr_at_k': 'mrr',
    # 'distributional_coverage': 'entropy'
}

class Evaluator:
    def __init__(self, metrics_overall=DEFAULT_METRICS_OVERALL,
                 metrics_by_time_idx=DEFAULT_METRICS_OVERALL,   # можно использовать те же
                 metrics_beyond_accuracy=DFAULT_METRICS_BEYOND_ACCURACY,
                 top_k=[10], col_user='user_id', col_item='item_id',
                 col_prediction='prediction', col_rating='rating',
                 col_time_idx='time_idx'):

        self.metrics_overall = metrics_overall
        self.metrics_by_time_idx = metrics_by_time_idx
        self.metrics_beyond_accuracy = metrics_beyond_accuracy
        self.top_k = top_k
        self.col_user = col_user
        self.col_item = col_item
        self.col_prediction = col_prediction
        self.col_rating = col_rating
        self.col_time_idx = col_time_idx

    def _prepare_lists(self, test, recs):
        """Преобразует DataFrame в списки ground truth и предсказаний для каждого пользователя."""
        actual = test.groupby(self.col_user)[self.col_item].apply(list).to_dict()
        predicted = recs.groupby(self.col_user)[self.col_item].apply(list).to_dict()
        # Берём только пользователей, присутствующих в обоих
        users = sorted(set(actual.keys()) & set(predicted.keys()))
        actual_list = [actual[u] for u in users]
        predicted_list = [predicted[u] for u in users]
        return actual_list, predicted_list

    def compute_metrics(self, test, recs, train=None):
        # Подготовка данных
        actual_list, predicted_list = self._prepare_lists(test, recs)

        # Общее количество предметов (для coverage)
        n_items = train[self.col_item].nunique() if train is not None else 0

        # Вычисляем все метрики сразу для всех top_k
        precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
            actual_list, predicted_list, self.top_k, n_items
        )

        result = {}
        for i, k in enumerate(self.top_k):
            # Добавляем только те метрики, которые запрошены в self.metrics_overall
            if 'precision_at_k' in self.metrics_overall:
                result[f'precision@{k}'] = precisions[i]
            if 'recall_at_k' in self.metrics_overall:
                result[f'recall@{k}'] = recalls[i]
            if 'ndcg_at_k' in self.metrics_overall:
                result[f'ndcg@{k}'] = ndcgs[i]
            if 'mrr_at_k' in self.metrics_overall:
                result[f'mrr@{k}'] = mrrs[i]
            if 'catalog_coverage' in self.metrics_beyond_accuracy and train is not None:
                result[f'coverage@{k}'] = covs[i]

        return result

    def compute_metrics_by_time_idx(self, test, recs, top_k_gt: bool = False) -> pd.DataFrame:
        """
        Вычисляет метрики для каждого временного индекса.
        top_k_gt: если True – используются все элементы до текущего включительно,
                  если False – только текущий элемент.
        """
        results = {}
        for time_idx in test[self.col_time_idx].unique():
            condition = test[self.col_time_idx] <= time_idx if top_k_gt else test[self.col_time_idx] == time_idx
            # Для метрик beyond accuracy train не передаём (coverage не будет вычислен)
            results[time_idx] = self.compute_metrics(test[condition], recs, train=None)

        return pd.DataFrame(results).T


# class Evaluator:

#     def __init__(self, metrics_overall=DEFAULT_METRICS_OVERALL,
#                  metrics_by_time_idx=DEFAULT_METRICS_BY_TIME_IDX,
#                  metrics_beyond_accuracy=DFAULT_METRICS_BEYOND_ACCURACY,
#                  top_k=[10], col_user='user_id', col_item='item_id',
#                  col_prediction='prediction', col_rating='rating',
#                  col_time_idx='time_idx'):

#         self.metrics_overall = metrics_overall
#         self.metrics_by_time_idx = metrics_by_time_idx
#         self.metrics_beyond_accuracy = metrics_beyond_accuracy
#         self.top_k = top_k
#         self.col_user = col_user
#         self.col_item = col_item
#         self.col_prediction = col_prediction
#         self.col_rating = col_rating
#         self.col_time_idx = col_time_idx

#     def compute_metrics(self, test, recs, train=None):

#         if not hasattr(test, 'rating'):
#             test = test.assign(rating=1)

#         result = {}
#         for k in self.top_k:
#             for metric in self.metrics_overall:
#                 metric_obj = getattr(python_evaluation, metric)
#                 metric_name = METRIC_NAMES.get(metric) or metric
#                 result[f'{metric_name}@{k}'] = metric_obj(
#                     test, recs,  k=k, col_user=self.col_user, col_item=self.col_item,
#                     col_prediction=self.col_prediction, col_rating=self.col_rating)
#             if train is not None:
#                 for metric in self.metrics_beyond_accuracy:
#                     metric_obj = getattr(python_evaluation, metric)
#                     metric_name = METRIC_NAMES.get(metric) or metric
#                     try:
#                         result[f'{metric_name}@{k}'] = metric_obj(
#                             train, recs, col_user=self.col_user, col_item=self.col_item)
#                     except:
#                         pass

#         return result

#     def compute_metrics_by_time_idx(self, test, recs, top_k_gt: bool = False) -> pd.DataFrame:
#         """
#         Compute metric for each item of ground truth sequence
#         or all items before and on the current position of an item ground truth sequence.

#         :param test: ground truth in ms_recommenders format
#         :param recs: recommendations in ms_recommenders format
#         :param top_k_gt: True: Leave all items before and on the current position in ground truth items (test);
#             False: Leave only one item on the current position in ground truth items (test);
#         """
#         result = {}
#         for time_idx in test[self.col_time_idx].unique():
#             condition = test[self.col_time_idx] <= time_idx if top_k_gt else test[self.col_time_idx] == time_idx
#             result[time_idx] = self.compute_metrics(
#                 test[condition], recs)

#         result = pd.DataFrame(result).T

#         return result
