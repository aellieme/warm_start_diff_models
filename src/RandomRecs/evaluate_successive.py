import numpy as np
from tqdm import tqdm

def evaluate_successive(test_sequences, user_histories, item_catalog, rng=None):
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
    
    all_hits = []
    all_precisions = []
    all_reciprocal_ranks = []
    all_ndcgs = []
    recommended_items = set()
    total_targets = 0
    
    for i in tqdm(range(len(test_sequences)), desc="Оценка Successive (top-1)"):
        test_seq = test_sequences[i]
        if len(test_seq) < 2:
            continue
            
        base_history = user_histories[i]
        
        # Проходим по каждому элементу тестовой последовательности, начиная со второго
        for j in range(1, len(test_seq)):
            target_item = test_seq[j]
            
            # 1. История = базовая + предыдущие элементы из тестовой последовательности
            current_history = base_history + test_seq[:j]
            seen_items = set(current_history)
            
            # 2. Генерация случайной рекомендации (top-1)
            shuffled = rng.permutation(item_catalog)
            rec_item = None
            for item in shuffled:
                if item not in seen_items:
                    rec_item = item
                    break
            if rec_item is None:
                rec_item = shuffled[0]
            
            recommended_items.add(rec_item)
            
            # 3. Расчет метрик для данной пары (история -> цель)
            hit = 1 if rec_item == target_item else 0
            all_hits.append(hit)
            all_precisions.append(hit)
            all_reciprocal_ranks.append(hit)
            all_ndcgs.append(hit)
            total_targets += 1
    
    if total_targets == 0:
        return {
            'recall@1': 0.0,
            'precision@1': 0.0,
            'ndcg@1': 0.0,
            'mrr@1': 0.0,
            'coverage@1': 0.0
        }
    
    # Усреднение по всем сгенерированным целям
    recall = np.mean(all_hits)
    precision = np.mean(all_precisions)
    ndcg = np.mean(all_ndcgs)
    mrr = np.mean(all_reciprocal_ranks)
    coverage = len(recommended_items) / len(item_catalog)
    
    return {
        'recall@1': recall,
        'precision@1': precision,
        'ndcg@1': ndcg,
        'mrr@1': mrr,
        'coverage@1': coverage
    }