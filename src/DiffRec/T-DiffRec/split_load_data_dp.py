import os
import sys
import json
import numpy as np
import pandas as pd
from polara import get_movielens_data
import argparse

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# DATA_DIR = '../../data/ml-1m/'
T_VALUES = [10, 20, 50, 100]      
# def ensure_dirs():
#     for d in [DATA_DIR]:
#         os.makedirs(d, exist_ok=True)
def load_amazon(dataset_name, data_dir='../../data/amazon/'):
    file_map = {
        'amazon_Baby':               'reviews_Baby_5.json',
        'amazon_Beauty':             'reviews_Beauty_5.json',
        'amazon_Sports_and_Outdoors':'reviews_Sports_and_Outdoors_5.json',
        'amazon_Toys_and_Games':     'reviews_Toys_and_Games_5.json'
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
        'reviewerID': 'user_id',
        'asin': 'item_id',
        'unixReviewTime': 'timestamp'
    })
    df = df[['user_id', 'item_id', 'timestamp']]
    # преобразование в последовательные индексы (item_id с 1)
    df['user_id'] = pd.Categorical(df['user_id']).codes
    df['item_id'] = pd.Categorical(df['item_id']).codes + 1
    df = df.rename(columns={'user_id': 'userid', 'item_id': 'movieid'})
    return df

class GlobalTemporalSplitter:
    """
    Global Temporal Split на основе квантилей времени (GTS).
    
    Разделение:
      1. Train   (0% – train_p)
      2. Valid   (train_p – train_p+val_p)
      4. Test    (оставшиеся)
    """
    def __init__(self, df, user_col='userid', item_col='movieid', time_col='timestamp'):
        # сортировка по времени (глобально)
        self.df = df.sort_values(by=time_col).reset_index(drop=True)
        self.u_col = user_col
        self.i_col = item_col
        self.time_col = time_col

        print(f"[Data Prep] Total interactions: {len(self.df)}")
        print(f"[Data Prep] Time range: {self.df[time_col].min()} to {self.df[time_col].max()}")

    def split(self, train_p=0.7, val_p=0.1):
        """
        Разделение по квантилям временной метки.
        Возвращает словарь с массивами пар (uid, iid) для каждого сплита,
        отмапленными на индексы, построенные по тренировочному множеству
        """
        n = len(self.df)
        time_values = self.df[self.time_col]

        # квантили
        train_cutoff = time_values.quantile(train_p)
        val_cutoff   = time_values.quantile(train_p + val_p)

        print(f"[Cutoffs] Train ≤ {train_cutoff}, Val ≤ {val_cutoff}")

        train_df = self.df[self.df[self.time_col] <= train_cutoff].copy()
        val_df   = self.df[(self.df[self.time_col] > train_cutoff) & (self.df[self.time_col] <= val_cutoff)].copy()
        test_df  = self.df[self.df[self.time_col] > val_cutoff].copy()

        print(f"[Split] Train: {len(train_df)} ({len(train_df)/n:.2%})")
        print(f"[Split] Valid: {len(val_df)} ({len(val_df)/n:.2%})")
        print(f"[Split] Test: {len(test_df)} ({len(test_df)/n:.2%})")

        # Маппинг только по тренировочным пользователям и айтемам
        unique_users = train_df[self.u_col].unique()
        unique_items = train_df[self.i_col].unique()

        u_map = {uid: i for i, uid in enumerate(unique_users)}
        i_map = {iid: i for i, iid in enumerate(unique_items)}

        print(f"[Mapping] Unique Users (Train): {len(u_map)}")
        print(f"[Mapping] Unique Items (Train): {len(i_map)}")

        # Преобразование всех сплитов в индексы
        train_data = self._map(train_df, u_map, i_map)
        val_data   = self._map(val_df,   u_map, i_map)
        test_data  = self._map(test_df,  u_map, i_map)

        print(f"[Filter] Train interactions kept: {len(train_data)}")
        print(f"[Filter] Valid interactions kept: {len(val_data)} (dropped {len(val_df) - len(val_data)})")
        print(f"[Filter] Test interactions kept: {len(test_data)} (dropped {len(test_df) - len(test_data)})")

        return {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'maps': (u_map, i_map)
        }

    def _map(self, df, u_map, i_map):
        # "Преобразование колонок user/item в индексы с отбрасыванием отсутствующих
        mapped = df.copy()
        mapped['uid'] = mapped[self.u_col].map(u_map)
        mapped['iid'] = mapped[self.i_col].map(i_map)
        cleaned = mapped.dropna(subset=['uid', 'iid'])
        return cleaned[['uid', 'iid']].values.astype(int)

    def save_splits(self, data_dict, output_path):
        # сохраняю сплиты в .нпу файлы
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"[IO] Created directory: {output_path}")

        files = {
            'train_list.npy': data_dict['train'],
            'valid_list.npy': data_dict['val'],
            'test_list.npy': data_dict['test']
        }

        for filename, array in files.items():
            filepath = os.path.join(output_path, filename)
            np.save(filepath, array)
            print(f"[IO] Saved {filename}: shape {array.shape} -> {filepath}")

        self._verify_no_leakage(data_dict)

    def _verify_no_leakage(self, data_dict):
        # Проверка, что взаимодействия не пересекаются между сплитами
        train_set = set(map(tuple, data_dict['train']))
        val_set   = set(map(tuple, data_dict['val']))
        test_set  = set(map(tuple, data_dict['test']))

        assert len(train_set & val_set) == 0, "Leakage: Train & Valid overlap!"
        assert len(train_set & test_set) == 0, "Leakage: Train & Test overlap!"
        assert len(val_set & test_set) == 0, "Leakage: Valid & Test overlap!"

        print("[Verification] PASSED: No data leakage detected between splits.")

def initialize_data(dataset='ml-1m'):
    data_dir = f'../../data/{dataset}/'
    os.makedirs(data_dir, exist_ok=True)
    if dataset == 'ml-1m':
        df = get_movielens_data(include_time=True)
    else:
        df = load_amazon(dataset)
    print('Dataset loaded')
    print(df.head(3))
    print("Preparing data splits using Global Temporal Split")
    print('GTS')

    splitter = GlobalTemporalSplitter(df)
    data = splitter.split(train_p=0.7, val_p=0.1)
    splitter.save_splits(data, data_dir)  
    # splitter.save_splits(data, DATA_DIR)
    print('Dataset prepared, splits saved to', data_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty',
                                 'amazon_Sports_and_Outdoors', 'amazon_Toys_and_Games'])
    args = parser.parse_args()
    initialize_data(dataset=args.dataset)

