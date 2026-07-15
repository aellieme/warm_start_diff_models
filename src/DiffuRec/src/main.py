import os
import random
import argparse
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import (ExperimentTracker, checkpoint_due, checkpoint_path,
                                                  save_dataset_popularity, save_torch_checkpoint)
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
import pickle
from utils import Data_Train, Data_Val, Data_Test, Data_CHLS
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train, LSHT_inference
from collections import Counter
import polara
from polara.datasets.movielens import get_movielens_data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import json
import torch.optim as optim
from tqdm import tqdm

from trainer import evaluate_and_print
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-1m',
                    choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty',
                             'amazon_Sports_and_Outdoors', 'amazon_Toys_and_Games'])
# parser.add_argument('--dataset', default='ml-1m', help='Dataset name: toys, amazon_beauty, steam, ml-1m')
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--final_train', action='store_true', help='Train on train+val without validation')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
parser.add_argument('--max_len', type=int, default=50, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=1024, help='Batch Size')  
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=60, help='Number of epochs for training')  ## 500
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=0.001, help='loss weight for diffusion')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=20, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=5, help='the number of epoch to wait before early stop')
parser.add_argument('--description', type=str, default='Diffu_norm_score', help='Model brief introduction')
parser.add_argument('--long_head', default=False, help='Long and short sequence, head and long-tail items')
parser.add_argument('--diversity_measure', default=False, help='Measure the diversity of recommendation results')
parser.add_argument('--epoch_time_avg', default=False, help='Calculate the average time of one epoch training')
args = parser.parse_args()

print(args)

if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.dataset):
    os.makedirs(args.log_file + args.dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)
logger.info(args)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

def load_amazon(dataset_name, data_dir='../../data/amazon'):
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
        'reviewerID': 'userid',
        'asin': 'itemid',
        'unixReviewTime': 'timestamp'
    })
    df = df[['userid', 'itemid', 'timestamp']]
    # кодирование
    from sklearn.preprocessing import LabelEncoder
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['userid'] = user_enc.fit_transform(df['userid'])
    df['itemid'] = item_enc.fit_transform(df['itemid'])
    smap = {idx: original for idx, original in enumerate(item_enc.classes_)}
    return df, smap
def item_num_create(args, item_num):
    args.item_num = item_num
    return args
def load_movielens_local(data_dir='../../data/info/'):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    ratings_path = os.path.join(data_dir, 'ratings.dat')
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.dat not found at {ratings_path}")
    df = pd.read_csv(ratings_path, sep='::', engine='python',
                     names=['userid', 'movieid', 'rating', 'timestamp'])
    df = df[['userid', 'movieid', 'timestamp']]
    user_enc = LabelEncoder()
    item_enc = LabelEncoder()
    df['userid'] = user_enc.fit_transform(df['userid'])
    df['movieid'] = item_enc.fit_transform(df['movieid'])
    smap = {idx: original for idx, original in enumerate(item_enc.classes_)}
    df = df.rename(columns={'movieid': 'itemid'})
    return df, smap

def cold_hot_long_short(data_raw, dataset_name):
    item_list = []
    len_list = []
    target_item = []

    for id_temp in data_raw['train']:
        temp_list = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        len_list.append(len(temp_list))
        target_item.append(data_raw['test'][id_temp][0])
        item_list += temp_list
    item_num_count = Counter(item_list)
    split_num = np.percentile(list(item_num_count.values()), 80)
    cold_item, hot_item = [], []
    for item_num_temp in item_num_count.items():
        if item_num_temp[1] < split_num:
            cold_item.append(item_num_temp[0])
        else:
            hot_item.append(item_num_temp[0])
    cold_ids, hot_ids = [], []
    cold_list, hot_list = [], []
    for id_temp, item_temp in enumerate(data_raw['test'].values()):
        if item_temp[0] in hot_item:
            hot_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                hot_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                hot_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
        else:
            cold_ids.append(id_temp)
            if dataset_name == 'ml-1m':
                cold_list.append(data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1])
            else:
                cold_list.append(data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp])
    cold_hot_dict = {'hot': hot_list, 'cold': cold_list}

    len_short = np.percentile(len_list, 20)
    len_midshort = np.percentile(len_list, 40)
    len_midlong = np.percentile(len_list, 60)
    len_long = np.percentile(len_list, 80)
    
    len_seq_dict = {'short': [], 'mid_short': [], 'mid': [], 'mid_long': [], 'long': []}
    for id_temp, len_temp in enumerate(len_list):
        if dataset_name == 'ml-1m':
            temp_seq = data_raw['train'][id_temp+1] + data_raw['val'][id_temp+1] + data_raw['test'][id_temp+1]
        else:
            temp_seq = data_raw['train'][id_temp] + data_raw['val'][id_temp] + data_raw['test'][id_temp]
        if len_temp <= len_short:
            len_seq_dict['short'].append(temp_seq)
        elif len_short < len_temp <= len_midshort:
            len_seq_dict['mid_short'].append(temp_seq)
        elif len_midshort < len_temp <= len_midlong:
            len_seq_dict['mid'].append(temp_seq)
        elif len_midlong < len_temp <= len_long:
            len_seq_dict['mid_long'].append(temp_seq)
        else:
            len_seq_dict['long'].append(temp_seq)
    return cold_hot_dict, len_seq_dict, split_num, [len_short, len_midshort, len_midlong, len_long], len_list, list(item_num_count.values())

def load_and_split_gts(quantiles=(0.7, 0.8), dataset_name='ml-1m'):
    """
    Global Temporal Split 
    quantiles: (validation_quantile, test_quantile)
    Default: 70% train, 10% validation (70-80), 20% test (>80)
    """
    # if dataset_name == 'ml-1m':
    #     df = get_movielens_data(include_time=True)
    #     user_enc = LabelEncoder()
    #     item_enc = LabelEncoder()
    #     df['userid'] = user_enc.fit_transform(df['userid'])
    #     df['movieid'] = item_enc.fit_transform(df['movieid'])
    #     smap = {idx: original for idx, original in enumerate(item_enc.classes_)}
    #     df = df.rename(columns={'movieid': 'itemid'})  
    if dataset_name == 'ml-1m':
        df, smap = load_movielens_local(data_dir='../../data/info/')
        if 'itemid' not in df.columns:
            df = df.rename(columns={'movieid': 'itemid'})
    else:
        df, smap = load_amazon(dataset_name)
        # убедимся, что колонка называется 'itemid'
        if 'itemid' not in df.columns:
            df = df.rename(columns={'item_id': 'itemid'})
    # # 1. Загрузка MovieLens-1M с временными метками
    # df = get_movielens_data(include_time=True)
    
    # # 2. Сквозная переиндексация пользователей и айтемов
    # user_enc = LabelEncoder()
    # item_enc = LabelEncoder()
    # df['userid'] = user_enc.fit_transform(df['userid'])
    # df['movieid'] = item_enc.fit_transform(df['movieid'])
    # smap = {idx: original for idx, original in enumerate(item_enc.classes_)}
    
    # 3. Глобальная сортировка по времени
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 4. Вычисление глобальных квантилей
    T_valid = df['timestamp'].quantile(quantiles[0])   # 70%
    T_test  = df['timestamp'].quantile(quantiles[1])   # 80%
    
    # 5. Разделение по пользователям
    train_data = {}
    val_examples = {}
    test_examples = {}
    
    for uid, group in df.groupby('userid'):
        group = group.sort_values('timestamp')
        # items = group['movieid'].tolist()
        items = group['itemid'].tolist()
        timestamps = group['timestamp'].tolist()
        
        # Обучающая история (всё до T_valid)
        train_items = [item for item, ts in zip(items, timestamps) if ts <= T_valid]
        
        # Валидационное окно (T_valid < ts <= T_test)
        val_window = [(item, ts) for item, ts in zip(items, timestamps) if T_valid < ts <= T_test]
        if val_window:
            target_item = val_window[-1][0]
            val_history = [item for item, _ in val_window[:-1]]
            full_input = train_items + val_history
            val_examples[uid] = (full_input, [target_item])
        
        # Тестовое окно (ts > T_test)
        test_window = [(item, ts) for item, ts in zip(items, timestamps) if ts > T_test]
        if test_window:
            target_item = test_window[-1][0]
            test_history = [item for item, _ in test_window[:-1]]
            # Полная входная последовательность: обучение + валидационная история + тестовая история
            val_history_for_test = [item for item, _ in val_window]  # вся история валидации (без таргета)
            full_input = train_items + val_history_for_test + test_history
            test_examples[uid] = (full_input, [target_item])
        
        if train_items:
            train_data[uid] = train_items
    
    # Преобразование в формат, ожидаемый классами Data_Val и Data_Test
    val_u2seq = {uid: val_examples[uid][0] for uid in val_examples}
    val_u2answer = {uid: val_examples[uid][1] for uid in val_examples}
    
    test_u2seq = {uid: test_examples[uid][0] for uid in test_examples}
    test_u2answer = {uid: test_examples[uid][1] for uid in test_examples}
    
    data_raw = {
        'train': train_data,
        'val': val_u2answer,
        'test': test_u2answer,
        'smap': smap,
        'val_seq': val_u2seq,
        'test_seq': test_u2seq,
    }
    return data_raw


def main(args):    
    fix_random_seed_as(args.random_seed)
    torch.backends.cudnn.benchmark = True
    # data_raw = load_and_split_gts(quantiles=(0.7, 0.8))
    data_raw = load_and_split_gts(quantiles=(0.7, 0.8), dataset_name=args.dataset)
    
    print("Train users:", len(data_raw['train']))
    print("Item vocab size:", len(data_raw['smap']))
    
    args = item_num_create(args, len(data_raw['smap']))
    
    # Подготовка данных для валидации
    data_raw_for_val = {
        'train': data_raw['val_seq'],
        'val': data_raw['val'],
        'test': data_raw['test'],
        'smap': data_raw['smap']
    }
    
    # Подготовка данных для теста
    data_raw_for_test = {
        'train': data_raw['test_seq'],
        'val': {uid: [] for uid in data_raw['test_seq']},
        'test': data_raw['test'],
        'smap': data_raw['smap']
    }
    
    tra_data = Data_Train(data_raw['train'], args)
    val_data = Data_Val(data_raw_for_val['train'], data_raw_for_val['val'], args)
    test_data = Data_Test(data_raw_for_test['train'], data_raw_for_test['val'], data_raw_for_test['test'], args)
    
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    args.coverage_candidate_items = {
        item for sequence in data_raw['train'].values() for item in sequence
    }
    args.train_item_popularity = Counter(
        item for sequence in data_raw['train'].values() for item in sequence
    )
    save_dataset_popularity(args.dataset, args.train_item_popularity)
    
    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)
    # if args.final_train:
        
    if args.final_train:
        merged_train = {**data_raw['train']}
        for uid, seq in data_raw['val_seq'].items():
            if uid in merged_train:
                merged_train[uid] = merged_train[uid] + seq   # присоединяем последовательность валидации (без таргета)
            else:
                merged_train[uid] = seq
        tra_data = Data_Train(merged_train, args)
        tra_loader = tra_data.get_pytorch_dataloaders()
        args.coverage_candidate_items = {
            item for sequence in merged_train.values() for item in sequence
        }
        args.coverage_candidate_items.update(
            item for targets in data_raw['val'].values() for item in targets
        )
        args.train_item_popularity = Counter(
            item for sequence in merged_train.values() for item in sequence
        )
        args.experiment_tracker = ExperimentTracker(args.dataset, "DiffuRec")

        diffu_rec = create_model_diffu(args)
        model = Att_Diffuse_model(diffu_rec, args).to(args.device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        from plotting import TrainingPlotter
        plotter = TrainingPlotter(save_dir=args.log_file + args.dataset,
                                model_name=f"DiffuRec_final_{time.strftime('%Y%m%d_%H%M%S')}",
                                metrics=['loss'])
        
        for epoch in range(1, args.epochs+1):
            model.train()
            total_loss = 0
            for batch in tqdm(tra_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", unit="batch"):
                batch = [x.to(args.device) for x in batch]
                optimizer.zero_grad()
                _, rep_diffu, _, _, _, _ = model(batch[0], batch[1], train_flag=True)
                loss = model.loss_diffu_ce(rep_diffu, batch[1])
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(tra_loader)
            plotter.update(epoch=epoch, loss=avg_loss)
            args.experiment_tracker.log_epoch(epoch, train_loss=avg_loss)
            if checkpoint_due(epoch - 1, args.epochs):
                model_path = checkpoint_path("DiffuRec", args.dataset, args.max_len, args.random_seed)
                saved_args = {key: value for key, value in vars(args).items() if key != "experiment_tracker"}
                save_torch_checkpoint({"model_state_dict": model.state_dict(), "args": saved_args}, model_path)
            if epoch % 10 == 0:
                plotter.plot(save=True, show=False, suffix=f'_epoch{epoch}')
            print(f"Epoch {epoch:03d} loss {avg_loss:.4f}")
        plotter.plot(save=True, show=False, suffix='_final')
        
        # Сохраняем модель
        model_path = checkpoint_path("DiffuRec", args.dataset, args.max_len, args.random_seed)
        saved_args = {key: value for key, value in vars(args).items() if key != "experiment_tracker"}
        save_torch_checkpoint({"model_state_dict": model.state_dict(), "args": saved_args}, model_path)
        print(f"Final model saved to {model_path}")
        
        # Оценка на тесте
        evaluate_and_print(model, test_data_loader, args, logger, description="test", save_recs=True)
        return
    best_model, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_diffu_joint_model, args, logger)
    
    evaluate_and_print(best_model, test_data_loader, args, logger, description="test")

# def main(args):    
#     fix_random_seed_as(args.random_seed)
#     torch.backends.cudnn.benchmark = True
#     # path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
#     # with open(path_data, 'rb') as f:
#     #     data_raw = pickle.load(f)
#     data_raw = load_and_split_gts(quantiles=(0.7, 0.8, 0.9))
    
#     print("Train users:", len(data_raw['train']))
#     print("Sample sequence length:", len(list(data_raw['train'].values())[0]))
#     print("Item vocab size:", len(data_raw['smap']))
    
#     data_raw_for_val = {
#         'train': data_raw['val_seq'],   # подменяем: для валидации история — это полная последовательность
#         'val': data_raw['val'],
#         'test': data_raw['test'],
#         'smap': data_raw['smap']
#         }

#     # Для теста: история = test_seq, дополнительная история (u2seq_add) = пусто
#     data_raw_for_test = {
#         'train': data_raw['test_seq'],
#         # 'val': data_raw['val']
#         'val': {uid: [] for uid in data_raw['test_seq']},
#         'test': data_raw['test'],
#         'smap': data_raw['smap']
#         }
    
#         # Создаём baseline-последовательности (без адаптации)
#     baseline_test_seq = {}
#     for uid in data_raw['test_seq'].keys():
#         full_seq = data_raw['test_seq'][uid]          # train + val_all + adapt + test_history
#         adapt_items = set(data_raw['adapt_seq'].get(uid, []))
#         # Удаляем адаптационные айтемы, сохраняя порядок
#         baseline_seq = [item for item in full_seq if item not in adapt_items]
#         baseline_test_seq[uid] = baseline_seq

#     data_raw_for_test_baseline = {
#         'train': baseline_test_seq,
#         'val': {uid: [] for uid in baseline_test_seq},   # пустые дополнительные истории
#         'test': data_raw['test'],
#         'smap': data_raw['smap']
#     }
#     baseline_test_data = Data_Test(data_raw_for_test_baseline['train'],
#                                 data_raw_for_test_baseline['val'],
#                                 data_raw_for_test_baseline['test'], args)
#     baseline_test_loader = baseline_test_data.get_pytorch_dataloaders()
        
    
#     # cold_hot_long_short(data_raw, args.dataset)
#     args = item_num_create(args, len(data_raw['smap']))

#     tra_data = Data_Train(data_raw['train'], args)   # обучение без изменений

#     # ВАЖНО: для валидации передаём подменённый словарь
#     val_data = Data_Val(data_raw_for_val['train'], data_raw_for_val['val'], args)

#     # Для теста передаём подменённый словарь (u2seq = test_seq, u2seq_add = пустой)
#     # Data_Test ожидает три аргумента: data_train, data_val, data_test.
#     test_data = Data_Test(data_raw_for_test['train'], data_raw_for_test['val'], data_raw_for_test['test'], args)
    
#     # args = item_num_create(args, len(data_raw['smap']))
#     # tra_data = Data_Train(data_raw['train'], args)
#     # val_data = Data_Val(data_raw['train'], data_raw['val'], args)
#     # test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    
    
#     tra_data_loader = tra_data.get_pytorch_dataloaders()
#     val_data_loader = val_data.get_pytorch_dataloaders()
#     test_data_loader = test_data.get_pytorch_dataloaders()
#     diffu_rec = create_model_diffu(args)
#     rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)
    
#     best_model, test_results = model_train(tra_data_loader, val_data_loader, test_data_loader, rec_diffu_joint_model, args, logger)
    # Baseline инференс
    # evaluate_and_print(best_model, baseline_test_loader, args, logger, description="baseline")
    # # Adaptation инференс (уже есть test_data_loader)
    # evaluate_and_print(best_model, test_data_loader, args, logger, description="adaptation")

    if args.long_head:
        cold_hot_dict, len_seq_dict, split_hotcold, split_length, list_len, list_num = cold_hot_long_short(data_raw, args.dataset)
        cold_data = Data_CHLS(cold_hot_dict['cold'], args)
        cold_data_loader = cold_data.get_pytorch_dataloaders()
        print('--------------Cold item-----------------------')
        LSHT_inference(best_model, args, cold_data_loader)

        hot_data = Data_CHLS(cold_hot_dict['hot'], args)
        hot_data_loader = hot_data.get_pytorch_dataloaders()
        print('--------------hot item-----------------------')
        LSHT_inference(best_model, args, hot_data_loader)

        short_data = Data_CHLS(len_seq_dict['short'], args)
        short_data_loader = short_data.get_pytorch_dataloaders()
        print('--------------Short-----------------------')
        LSHT_inference(best_model, args, short_data_loader)

        mid_short_data = Data_CHLS(len_seq_dict['mid_short'], args)
        mid_short_data_loader = mid_short_data.get_pytorch_dataloaders()
        print('--------------Mid_short-----------------------')
        LSHT_inference(best_model, args, mid_short_data_loader)

        mid_data = Data_CHLS(len_seq_dict['mid'], args)
        mid_data_loader = mid_data.get_pytorch_dataloaders()
        print('--------------Mid-----------------------')
        LSHT_inference(best_model, args, mid_data_loader)

        mid_long_data = Data_CHLS(len_seq_dict['mid_long'], args)
        mid_long_data_loader = mid_long_data.get_pytorch_dataloaders()
        print('--------------Mid_long-----------------------')
        LSHT_inference(best_model, args, mid_long_data_loader)

        long_data = Data_CHLS(len_seq_dict['long'], args)
        long_data_loader = long_data.get_pytorch_dataloaders()
        print('--------------Long-----------------------')
        LSHT_inference(best_model, args, long_data_loader)
    

if __name__ == '__main__':
    main(args)
