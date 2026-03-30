# src\data_split\data_load.py
import os
import torch
import sys
import os
import numpy as np

from polara import get_movielens_data

current_file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_file_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

DATA_DIR = '../data2/ml-1m/'

T_VALUES = [10, 20, 50, 100]      # количество шагов диффузии для экспериментов
# EPOCHS = 50                        # максимальное число эпох
# PATIENCE = 5                       # для ранней остановки
# MODE = 'both'                      

# создание каталогов, если их нет
def ensure_dirs():
    for d in [DATA_DIR]:
        os.makedirs(d, exist_ok=True)

class GlobalTemporalSplitter:
    """
    Global Temporal Split 
    
    1. Train (0% - 70%)   
    2. Validation (70% - 80%) : Early Stopping и подбор гиперпараметров 
    3. Adaptation (80% - 90%) : данные, доступные только на инференсе
    4. Test (90% - 100%)  
    """
    def __init__(self, df, user_col='userid', item_col='movieid', time_col='timestamp'):
        # сортировка по времени 
        self.df = df.sort_values(by=time_col).reset_index(drop=True)
        self.u_col = user_col
        self.i_col = item_col
        self.time_col = time_col
        
        print(f"[Data Prep] Total interactions: {len(self.df)}")
        print(f"[Data Prep] Time range: {self.df[time_col].min()} to {self.df[time_col].max()}")

    def split(self, train_p=0.7, val_p=0.1, adapt_p=0.1):
        n = len(self.df)
        
        # расчет индексов границ
        t_idx = int(n * train_p)
        v_idx = int(n * (train_p + val_p))
        a_idx = int(n * (train_p + val_p + adapt_p))
        
        # сплит датафрейма
        train_df = self.df.iloc[:t_idx].copy()
        val_df   = self.df.iloc[t_idx:v_idx].copy()
        adapt_df = self.df.iloc[v_idx:a_idx].copy()
        test_df  = self.df.iloc[a_idx:].copy()
        
        print(f"[Split] Train: {len(train_df)} ({len(train_df)/n:.2%})")
        print(f"[Split] Valid: {len(val_df)} ({len(val_df)/n:.2%})")
        print(f"[Split] Adapt: {len(adapt_df)} ({len(adapt_df)/n:.2%})")
        print(f"[Split] Test: {len(test_df)} ({len(test_df)/n:.2%})")

        unique_users = train_df[self.u_col].unique()
        unique_items = train_df[self.i_col].unique()
        
        u_map = {uid: i for i, uid in enumerate(unique_users)}
        i_map = {iid: i for i, iid in enumerate(unique_items)}
        
        print(f"[Mapping] Unique Users (Train): {len(u_map)}")
        print(f"[Mapping] Unique Items (Train): {len(i_map)}")

        # маппинг и фильтрация (удалим пользователей/айтемы, которых не было в трейне)
        train_data = self._map(train_df, u_map, i_map)
        val_data   = self._map(val_df, u_map, i_map)
        adapt_data = self._map(adapt_df, u_map, i_map)
        test_data  = self._map(test_df, u_map, i_map)
        
        print(f"[Filter] Train interactions kept: {len(train_data)}")
        print(f"[Filter] Valid interactions kept: {len(val_data)} (dropped {len(val_df) - len(val_data)})")
        print(f"[Filter] Adapt interactions kept: {len(adapt_data)} (dropped {len(adapt_df) - len(adapt_data)})")
        print(f"[Filter] Test interactions kept: {len(test_data)} (dropped {len(test_df) - len(test_data)})")

        return {
            'train': train_data,
            'val': val_data,
            'adapt': adapt_data,
            'test': test_data,
            'maps': (u_map, i_map)
        }

    def _map(self, df, u_map, i_map):
        mapped = df.copy()
        mapped['uid'] = mapped[self.u_col].map(u_map)
        mapped['iid'] = mapped[self.i_col].map(i_map)
        # удаляем строки, где пользователь или айтем не встретились в трейне
        cleaned = mapped.dropna(subset=['uid', 'iid'])
        return cleaned[['uid', 'iid']].values.astype(int)

    def save_splits(self, data_dict, output_path):
        """
        сохраняет сплиты в .npy файлы для DiffData и adaptation data
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            print(f"[IO] Created directory: {output_path}")
        
        files = {
            'train_list.npy': data_dict['train'],
            'valid_list.npy': data_dict['val'],    # Для Early Stopping
            'adapt_list.npy': data_dict['adapt'],  # Для инференса 
            'test_list.npy': data_dict['test'] 
        }
        
        for filename, array in files.items():
            filepath = os.path.join(output_path, filename)
            np.save(filepath, array)
            print(f"[IO] Saved {filename}: shape {array.shape} -> {filepath}")
            
        self._verify_no_leakage(data_dict)

    def _verify_no_leakage(self, data_dict):
        """
        проврка что взаимодействия не пересекаются между сплитами
        """
        train_set = set(map(tuple, data_dict['train']))
        val_set = set(map(tuple, data_dict['val']))
        adapt_set = set(map(tuple, data_dict['adapt']))
        test_set = set(map(tuple, data_dict['test']))
        
        assert len(train_set & val_set) == 0, "Leakage: Train & Valid overlap!"
        assert len(train_set & adapt_set) == 0, "Leakage: Train & Adapt overlap!"
        assert len(train_set & test_set) == 0, "Leakage: Train & Test overlap!"
        assert len(val_set & adapt_set) == 0, "Leakage: Valid & Adapt overlap!"
        assert len(val_set & test_set) == 0, "Leakage: Valid & Test overlap!"
        assert len(adapt_set & test_set) == 0, "Leakage: Adapt & Test overlap!"
        
        print("[Verification] PASSED: No data leakage detected between splits.")


def initialize_data():
    # если данные уже есть, не перезаписываем 
    # if os.path.exists(os.path.join(DATA_DIR, 'train.pkl')): 
    #     print(f"Data found in {DATA_DIR}, skipping preparation.")
    #     return
        
    print("Preparing data splits...")
    df = get_movielens_data(include_time=True) #загружаем датасет
    print('Загрузили датасет')
    print(df.head(3))
    splitter = GlobalTemporalSplitter(df)
    data = splitter.split(train_p=0.7, val_p=0.1, adapt_p=0.1) #делим с помощью GTS как 0.7/0.1/0.1/0.1
    splitter.save_splits(data, DATA_DIR) #сохраняем в папку data_dir
    print('dataset prepared, splits saved to', DATA_DIR)

initialize_data()

