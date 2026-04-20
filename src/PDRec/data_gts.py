import numpy as np
import pandas as pd
from collections import defaultdict
import polara
from polara.datasets.movielens import get_movielens_data

class MovieLensGTSProcessor:
    """
    Processor for MovieLens-1M with GTS (Global Temporal Split) protocol.
    Implements validation/adaptation/test splits as described in "Time to Split".
    """
    def __init__(self, valid_quantile=0.7, adapt_quantile=0.8, test_quantile=0.9, maxlen=200):
        self.valid_quantile = valid_quantile
        self.adapt_quantile = adapt_quantile
        self.test_quantile = test_quantile
        self.maxlen = maxlen
        self.item_decoder = None  # to map internal id -> original movieid
        self.user_decoder = None

    def load_and_process(self):
        raw_data = get_movielens_data(include_time=True)  # returns DataFrame with 'userid','movieid','rating','timestamp'
        raw_data = raw_data[['userid', 'movieid', 'timestamp']].drop_duplicates(['userid','movieid'])
        
        # Reindex users and items to continuous integers 1..N, 1..M (0 reserved for padding)
        unique_users = raw_data['userid'].unique()
        unique_items = raw_data['movieid'].unique()
        
        self.user_encoder = {orig: i+1 for i, orig in enumerate(unique_users)}
        self.item_encoder = {orig: i+1 for i, orig in enumerate(unique_items)}
        self.item_decoder = {i+1: orig for i, orig in enumerate(unique_items)}
        self.user_decoder = {i+1: orig for i, orig in enumerate(unique_users)}
        
        data = raw_data.copy()
        data['userid'] = data['userid'].map(self.user_encoder)
        data['movieid'] = data['movieid'].map(self.item_encoder)
        
        # 3. Sort globally by timestamp
        data = data.sort_values('timestamp')
        timestamps = data['timestamp'].values
        
        # 4. Compute global temporal cutoffs (quantiles)
        T_valid = np.quantile(timestamps, self.valid_quantile)
        T_adapt = np.quantile(timestamps, self.adapt_quantile)
        T_test  = np.quantile(timestamps, self.test_quantile)
        print(f"Global cutoffs: T_valid={T_valid}, T_adapt={T_adapt}, T_test={T_test}")
        
        # 5. Split data
        train_data = data[data['timestamp'] <= T_valid]
        future_data = data[data['timestamp'] > T_valid]
        adapt_data = data[(data['timestamp'] > T_adapt) & (data['timestamp'] <= T_test)]
        
        # 6. Build validation and test examples (input sequence + target)
        val_examples = self._build_examples(future_data, T_adapt, mode='valid')
        test_examples = self._build_examples(data[data['timestamp'] > T_test], T_test, mode='test')
        
        # 7. Build user histories in PDRec format
        user_train = defaultdict(list)
        for uid, group in train_data.groupby('userid'):
            user_train[uid] = group['movieid'].tolist()
        
        user_adapt = defaultdict(list)
        for uid, group in adapt_data.groupby('userid'):
            user_adapt[uid] = group['movieid'].tolist()
        
        # Validation targets and input sequences (to be concatenated with train history during evaluation)
        user_valid_target = {}
        user_valid_seq = defaultdict(list)
        for uid, seq, target in val_examples:
            user_valid_target[uid] = [target]
            user_valid_seq[uid] = seq
        
        user_test_target = {}
        user_test_seq = defaultdict(list)
        for uid, seq, target in test_examples:
            user_test_target[uid] = [target]
            user_test_seq[uid] = seq
        
        usernum = len(self.user_encoder)
        itemnum = len(self.item_encoder)
        dataset_dict = {
            'user_train': user_train,
            'user_valid_target': user_valid_target,
            'user_test_target': user_test_target,
            'user_adapt': user_adapt,
            'user_valid_seq': user_valid_seq,
            'user_test_seq': user_test_seq,
            'usernum': usernum,
            'itemnum': itemnum,
            'item_decoder': self.item_decoder,
            'user_decoder': self.user_decoder,
        }
        user_train_source = defaultdict(list)
        user_train_ti_source = defaultdict(list)
        user_train_mix = user_train 
        user_train_ti_mix = {}      

        user_train_ti_target = {}
        for u, seq in user_train.items():
            user_train_ti_target[u] = list(range(1, len(seq)+1))
            user_train_ti_mix[u] = user_train_ti_target[u]  # mix = только target

        user_train_mix_sequence_for_target = {}
        user_train_source_sequence_for_target = {}
        for u in user_train:
            seq_len = len(user_train[u])
            user_train_mix_sequence_for_target[u] = list(range(-seq_len, 0))
            user_train_source_sequence_for_target[u] = list(range(-seq_len, 0))  

        user_valid_ti_target = {u: [0] for u in user_valid_target}
        user_test_ti_target = {u: [0] for u in user_test_target}

        dataset_dict.update({
            'user_train_source': user_train_source,
            'user_train_ti_mix': user_train_ti_mix,
            'user_train_ti_source': user_train_ti_source,
            'user_train_ti_target': user_train_ti_target,
            'user_valid_ti_target': user_valid_ti_target,
            'user_test_ti_target': user_test_ti_target,
            'user_train_mix': user_train_mix,
            'user_train_mix_sequence_for_target': user_train_mix_sequence_for_target,
            'user_train_source_sequence_for_target': user_train_source_sequence_for_target,
            'interval': itemnum,  
        })
        return dataset_dict
    
    def _build_examples(self, data, cutoff_time, mode):
        examples = []
        for uid, group in data.groupby('userid'):
            sorted_group = group.sort_values('timestamp')
            items = sorted_group['movieid'].tolist()
            timestamps = sorted_group['timestamp'].tolist()
            
            if mode == 'valid':
                # Find last item with timestamp <= cutoff_time
                target_idx = None
                for i, ts in enumerate(timestamps):
                    if ts > cutoff_time:
                        target_idx = i - 1
                        break
                if target_idx is None or target_idx < 0:
                    continue
                input_seq = items[:target_idx]
                target = items[target_idx]
            else:  # mode == 'test'
                if len(items) < 1:
                    continue
                input_seq = items[:-1]
                target = items[-1]
            
            # Truncate to maxlen
            input_seq = input_seq[-self.maxlen:]
            examples.append((uid, input_seq, target))
        return examples

# import polara
# import pandas as pd
# import numpy as np
# from polara.datasets.movielens import get_movielens_data
# from collections import defaultdict

# class MovieLensGTSDataProcessor:
#     def __init__(self, valid_quantile=0.7, adapt_quantile=0.8, test_quantile=0.9, maxlen=200):
#         self.valid_quantile = valid_quantile
#         self.adapt_quantile = adapt_quantile
#         self.test_quantile = test_quantile
#         self.maxlen = maxlen
        
#     def load_and_transform(self):
#         # 1. Загрузка данных через polara
#         raw_data = get_movielens_data(include_time=True)
        
#         # 2. Сквозная переиндексация (transform_indices из polara)
#         self.user_encoder = {uid: i+1 for i, uid in enumerate(raw_data['userid'].unique())}
#         self.item_encoder = {iid: i+1 for i, iid in enumerate(raw_data['movieid'].unique())}
#         self.item_decoder = {v: k for k, v in self.item_encoder.items()}  # для обратного декодирования
        
#         data = raw_data.copy()
#         data['userid'] = data['userid'].map(self.user_encoder)
#         data['movieid'] = data['movieid'].map(self.item_encoder)
        
#         # 3. Сортировка по timestamp глобально
#         data = data.sort_values('timestamp')
#         timestamps = data['timestamp'].values
        
#         # 4. Глобальные временные отсечки (квантили)
#         T_valid = np.quantile(timestamps, self.valid_quantile)
#         T_adapt = np.quantile(timestamps, self.adapt_quantile)
#         T_test = np.quantile(timestamps, self.test_quantile)
        
#         # 5. Разделение данных
#         train_data = data[data['timestamp'] <= T_valid]
#         future_data = data[data['timestamp'] > T_valid]
#         adapt_data = data[(data['timestamp'] > T_adapt) & (data['timestamp'] <= T_test)]
        
#         # 6. Формирование валидационной и тестовой выборок (формат PDRec)
#         val_data = self._build_sequences(future_data, T_adapt, 'valid')
#         test_data = self._build_sequences(data[data['timestamp'] > T_test], T_test, 'test')
        
#         # 7. Преобразование в формат, совместимый с PDRec
#         return self._to_pdrec_format(train_data, val_data, test_data, adapt_data)
    
#     def _build_sequences(self, data, cutoff_time, stage):
#         sequences = defaultdict(list)
#         targets = {}
#         for user_id, group in data.groupby('userid'):
#             sorted_group = group.sort_values('timestamp')
#             items = sorted_group['movieid'].tolist()
#             timestamps = sorted_group['timestamp'].tolist()
            
#             if stage == 'valid':
#                 # Найти последний элемент с timestamp <= cutoff_time
#                 target_idx = next((i for i, ts in enumerate(timestamps) if ts > cutoff_time), len(items)) - 1
#                 if target_idx < 0:
#                     continue
#                 input_seq = items[:target_idx]
#                 target = items[target_idx]
#             else:  # test
#                 input_seq = items[:-1]
#                 target = items[-1]
            
#             # Обрезаем до maxlen
#             input_seq = input_seq[-self.maxlen:]
#             sequences[user_id] = input_seq
#             targets[user_id] = target
            
#         return {'sequences': sequences, 'targets': targets}
    
#     def _to_pdrec_format(self, train_data, val_data, test_data, adapt_data):
#         user_train = defaultdict(list)
#         user_valid = defaultdict(list)
#         user_test = defaultdict(list)
#         user_adapt = defaultdict(list)
        
#         for user_id, group in train_data.groupby('userid'):
#             user_train[user_id] = group['movieid'].tolist()
            
#         for user_id in val_data['sequences']:
#             user_valid[user_id] = [val_data['targets'][user_id]]

#         for user_id in test_data['sequences']:
#             user_test[user_id] = [test_data['targets'][user_id]]
            
#         for user_id, group in adapt_data.groupby('userid'):
#             user_adapt[user_id] = group['movieid'].tolist()
            
#         usernum = len(self.user_encoder)
#         itemnum = len(self.item_encoder)
        
#         return (user_train, user_valid, user_test, user_adapt, 
#                 val_data['sequences'], test_data['sequences'],
#                 usernum, itemnum)