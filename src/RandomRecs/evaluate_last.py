import numpy as np
from tqdm import tqdm

def evaluate_last(test_sequences, user_histories, item_catalog, rng=None):
    """
    Параметры:
    test_sequences: список списков, где каждый внутренний список содержит элементы 
                    пользователя в тестовом периоде (упорядочены по времени).
    user_histories: список списков, где каждый внутренний список содержит элементы 
                    в истории пользователя до тестового периода.
    item_catalog: список всех уникальных элементов.
    rng: numpy генератор случайных чисел.
    
    Возвращает:
    dict с метриками: recall@1, precision@1, ndcg@1, mrr@1, coverage@1
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_users = len(test_sequences)
    hits = 0
    precisions = []
    reciprocal_ranks = []
    ndcgs = []
    recommended_items = set()
    
    for i in tqdm(range(n_users), desc="Оценка Last (top-1)"):
        test_seq = test_sequences[i]
        if not test_seq:
            continue
            
        # 1. Целевой элемент - последний в тестовой последовательности
        target_item = test_seq[-1]
        
        # 2. История = все взаимодействия до тестового периода + все тестовые,
        #    кроме последнего
        full_history = user_histories[i] + test_seq[:-1]
        seen_items = set(full_history)
        
        # 3. Генерация случайной рекомендации (top-1)
        shuffled = rng.permutation(item_catalog)
        rec_item = None
        for item in shuffled:
            if item not in seen_items:
                rec_item = item
                break
        if rec_item is None:
            rec_item = shuffled[0]
        
        recommended_items.add(rec_item)
        
        # 4. Расчет метрик для данного пользователя
        hit = 1 if rec_item == target_item else 0
        hits += hit
        precisions.append(hit)       # precision@1 = hit
        reciprocal_ranks.append(hit) # MRR@1 = hit
        ndcgs.append(hit)            # NDCG@1 = hit
    
    if not precisions:
        return {
            'recall@1': 0.0,
            'precision@1': 0.0,
            'ndcg@1': 0.0,
            'mrr@1': 0.0,
            'coverage@1': 0.0
        }
    
    # Усреднение по пользователям
    recall = hits / n_users
    precision = np.mean(precisions)
    ndcg = np.mean(ndcgs)
    mrr = np.mean(reciprocal_ranks)
    coverage = len(recommended_items) / len(item_catalog)
    
    return {
        'recall@1': recall,
        'precision@1': precision,
        'ndcg@1': ndcg,
        'mrr@1': mrr,
        'coverage@1': coverage
    }