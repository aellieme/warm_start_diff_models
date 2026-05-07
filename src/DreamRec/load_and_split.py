import pandas as pd
import numpy as np
from polara import get_movielens_data
from utility import pad_history         


def load_and_preprocess_ml1m():
    mldata = get_movielens_data(include_time=True)
    userid_col = 'userid'
    itemid_col = 'movieid'
    time_col = 'timestamp'

    unique_users = mldata[userid_col].unique()
    unique_items = mldata[itemid_col].unique()
    user2idx = {u: i for i, u in enumerate(unique_users)}
    item2idx = {it: i for i, it in enumerate(unique_items)}

    mldata[userid_col] = mldata[userid_col].map(user2idx)
    mldata[itemid_col] = mldata[itemid_col].map(item2idx)

    data_index = {
        'users': pd.Index(unique_users),
        'items': pd.Index(unique_items)
    }

    all_data_sorted = mldata.sort_values(time_col).reset_index(drop=True)

    n_users = len(unique_users)
    n_items = len(unique_items)

    return all_data_sorted, data_index, n_users, n_items, userid_col, itemid_col, time_col

def global_temporal_split(all_data_sorted, time_col='timestamp',
                          train_ratio=0.7, val_ratio=0.1):
    quantiles = all_data_sorted[time_col].quantile([train_ratio, train_ratio + val_ratio])
    T_train = quantiles.iloc[0]
    T_val = quantiles.iloc[1]

    train_data = all_data_sorted[all_data_sorted[time_col] <= T_train].copy()
    val_data = all_data_sorted[(all_data_sorted[time_col] > T_train) &
                               (all_data_sorted[time_col] <= T_val)].copy()
    test_data = all_data_sorted[all_data_sorted[time_col] > T_val].copy()

    return train_data, val_data, test_data, T_train, T_val

def build_sequences(user_data, userid_col, itemid_col, time_col, max_seq_len,
                    filter_test_users=None, keep_last_only=False):
    sequences = []
    for uid, group in user_data.groupby(userid_col):
        if filter_test_users is not None and uid not in filter_test_users:
            continue
        group = group.sort_values(time_col)
        items = group[itemid_col].tolist()
        if keep_last_only:
            if len(items) < 2:
                continue
            seq = items[:-1]
            target = items[-1]
            if len(seq) > max_seq_len:
                seq = seq[-max_seq_len:]
            sequences.append((seq, len(seq), target))
        else:
            for i in range(1, len(items)):
                seq = items[:i]
                target = items[i]
                if len(seq) > max_seq_len:
                    seq = seq[-max_seq_len:]
                sequences.append((seq, len(seq), target))
    return sequences

def pad_and_format(seq_list, max_seq_len, pad_item):
    padded = []
    for seq, length, target in seq_list:
        padded_seq = pad_history(seq[:], max_seq_len, pad_item)
        padded.append((padded_seq, length, target))
    return pd.DataFrame(padded, columns=['seq', 'len_seq', 'next'])

def prepare_dreamrec_data(train_data, val_data, test_data, userid_col, itemid_col, time_col,
                          max_seq_len=50, pad_item=None):
    """
    pad_item: int, индекс паддинга (обычно item_num). Если не передан, определяется как число уникальных item.
    """
    all_items = pd.concat([train_data[itemid_col], val_data[itemid_col], test_data[itemid_col]])
    if pad_item is None:
        pad_item = all_items.max() + 1  
    train_seq = build_sequences(train_data, userid_col, itemid_col, time_col, max_seq_len, keep_last_only=False)

    train_val_data = pd.concat([train_data, val_data]).sort_values([userid_col, time_col])
    val_seq = build_sequences(train_val_data, userid_col, itemid_col, time_col, max_seq_len, filter_test_users=set(train_data[userid_col].unique()), keep_last_only=True)

    all_data = pd.concat([train_data, val_data, test_data]).sort_values([userid_col, time_col])
    test_seq = build_sequences(all_data, userid_col, itemid_col, time_col, max_seq_len, filter_test_users=set(train_data[userid_col].unique()), keep_last_only=True)

    train_df = pad_and_format(train_seq, max_seq_len, pad_item)
    val_df   = pad_and_format(val_seq,   max_seq_len, pad_item)
    test_df  = pad_and_format(test_seq,  max_seq_len, pad_item)

    return train_df, val_df, test_df, pad_item