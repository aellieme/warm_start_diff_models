import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def to_numeric_id(data, field):
    idx_data = data[field].astype("category")
    idx = idx_data.cat.codes
    idx_map = idx_data.cat.categories.rename(field)
    return idx, idx_map

def transform_indices(data, users, items):
    data_index = {}
    for entity, field in zip(['users', 'items'], [users, items]):
        idx, idx_map = to_numeric_id(data, field)
        data_index[entity] = idx_map
        data.loc[:, field] = idx
    return data, data_index

def matrix_from_data(data, data_description, dtype=None):
    user_idx = data[data_description['users']].values
    item_idx = data[data_description['items']].values
    feedback_data = data_description.get('feedback', None)
    if feedback_data is not None:
        feedback = data[feedback_data].values
    else:
        feedback = np.ones(len(user_idx))
    shape = (data_description['n_users'], data_description['n_items'])
    return csr_matrix((feedback, (user_idx, item_idx)), shape=shape, dtype=dtype)

def data_to_sequences(data, data_description):
    userid = data_description['users']
    itemid = data_description['items']
    order = data_description['order']
    sequences = (
        data.sort_values([userid, order])
        .groupby(userid, sort=False)[itemid].apply(list)
    )
    return sequences

def split_per_user_leave_k(df, user_col='userid', time_col='timestamp', k=3):
    df = df.sort_values([user_col, time_col])
    train_parts = []
    future_parts = []
    for uid, user_df in df.groupby(user_col):
        if len(user_df) <= k:
            continue
        train_parts.append(user_df.iloc[:-k])
        future_parts.append(user_df.iloc[-k:])
    train_df = pd.concat(train_parts, ignore_index=True)
    future_df = pd.concat(future_parts, ignore_index=True)
    return train_df, future_df

def split_future_for_eval(future_df, user_col='userid', time_col='timestamp'):
    future_df = future_df.sort_values([user_col, time_col])
    adapt_parts = []
    holdout_parts = []
    for uid, user_df in future_df.groupby(user_col):
        if len(user_df) < 2:
            continue
        adapt_parts.append(user_df.iloc[:-1])
        holdout_parts.append(user_df.iloc[-1:])
    adapt_df = pd.concat(adapt_parts, ignore_index=True)
    holdout_df = pd.concat(holdout_parts, ignore_index=True)
    return adapt_df, holdout_df