import numpy as np
from fileinput import filename
import random
import torch
import torch.utils.data as data
import scipy.sparse as sp
import copy
import os
import json
# import torch.sparse as sp
from torch.utils.data import Dataset


def _pair_matrix(pairs, shape, values=None):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    if len(pairs) == 0:
        return sp.csr_matrix(shape, dtype='float64')
    if values is None:
        values = np.ones(len(pairs), dtype=np.float64)
    return sp.csr_matrix(
        (values, (pairs[:, 0], pairs[:, 1])), shape=shape, dtype='float64'
    )


def _weighted_pair_matrix(pairs, shape, w_min, w_max):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    weights = np.empty(len(pairs), dtype=np.float64)
    by_user = {}
    for index, (uid, _) in enumerate(pairs):
        by_user.setdefault(int(uid), []).append(index)
    for indices in by_user.values():
        weights[indices] = np.linspace(w_min, w_max, len(indices))
    return _pair_matrix(pairs, shape, weights)


def load_warm_start_data(data_path, w_min, w_max, include_test=True):
    """Load preprocessed last-item data for the shared warm-start protocol."""
    required = [
        'train_list.npy', 'valid_list.npy', 'test_list.npy',
        'valid_history.npy', 'valid_targets.npy', 'protocol_meta.json',
    ]
    if include_test:
        required.extend(['test_history.npy', 'test_targets.npy'])
    missing = [name for name in required if not os.path.exists(os.path.join(data_path, name))]
    if missing:
        raise FileNotFoundError(
            "T-DiffRec warm-start files are missing: " + ", ".join(missing)
            + ". Re-run split_load_data_dp.py for this dataset."
        )
    with open(os.path.join(data_path, 'protocol_meta.json'), encoding='utf-8') as handle:
        meta = json.load(handle)
    if meta.get('protocol') != 'warm_start_known_catalog_v2':
        raise ValueError("Unsupported T-DiffRec preprocessing protocol")

    shape = (int(meta['n_user']), int(meta['n_item']))
    train_pairs = np.load(os.path.join(data_path, 'train_list.npy'))
    valid_pairs = np.load(os.path.join(data_path, 'valid_list.npy'))
    valid_history_pairs = np.load(os.path.join(data_path, 'valid_history.npy'))

    train_binary = _pair_matrix(train_pairs, shape)
    valid_binary = _pair_matrix(valid_pairs, shape)
    train_weighted = _weighted_pair_matrix(train_pairs, shape, w_min, w_max)
    train_val_pairs = np.vstack([train_pairs, valid_pairs])
    train_val_weighted = _weighted_pair_matrix(train_val_pairs, shape, w_min, w_max)
    valid_history = _pair_matrix(valid_history_pairs, shape)
    train_candidates = np.asarray(train_binary.sum(axis=0)).ravel() > 0
    train_val_candidates = np.asarray(
        (train_binary + valid_binary).sum(axis=0)
    ).ravel() > 0
    valid_history = valid_history.multiply(train_candidates).tocsr()

    result = {
        'train_weighted': train_weighted,
        'train_binary': train_binary,
        'valid_binary': valid_binary,
        'train_val_weighted': train_val_weighted,
        'valid_input': train_weighted + valid_history,
        'valid_mask': train_binary + valid_history,
        'valid_targets': np.load(os.path.join(data_path, 'valid_targets.npy')),
        'train_candidates': train_candidates,
        'train_val_candidates': train_val_candidates,
        'n_user': shape[0],
        'n_item': shape[1],
    }
    if include_test:
        test_history_pairs = np.load(os.path.join(data_path, 'test_history.npy'))
        test_history = _pair_matrix(test_history_pairs, shape)
        result.update({
            'test_input': train_val_weighted + test_history,
            'test_mask': train_binary + valid_binary + test_history,
            'test_targets': np.load(os.path.join(data_path, 'test_targets.npy')),
        })
    return result


def select_eligible_rows(input_data, history_mask, targets, candidate_mask):
    """Select last-item rows with known target and non-empty known history."""
    targets = np.asarray(targets, dtype=np.int64)
    valid_target_ids = (targets >= 0) & (targets < len(candidate_mask))
    known_targets = np.zeros_like(valid_target_ids)
    known_targets[valid_target_ids] = candidate_mask[targets[valid_target_ids]]
    nonempty_history = np.asarray(history_mask.getnnz(axis=1)).ravel() > 0
    eligible = valid_target_ids & known_targets & nonempty_history
    user_ids = np.flatnonzero(eligible)
    return (
        input_data[user_ids],
        history_mask[user_ids],
        targets[user_ids],
        user_ids,
    )

def data_load(train_path, valid_path, test_path, w_min, w_max):
    train_list = np.load(train_path, allow_pickle=True)
    valid_list = np.load(valid_path, allow_pickle=True)
    test_list = np.load(test_path, allow_pickle=True)

    uid_max = 0
    iid_max = 0
    train_dict = {}

    for uid, iid in train_list:
        if uid not in train_dict:
            train_dict[uid] = []
        train_dict[uid].append(iid)
        if uid > uid_max:
            uid_max = uid
        if iid > iid_max:
            iid_max = iid
    
    n_user = uid_max + 1
    n_item = iid_max + 1
    print(f'user num: {n_user}')
    print(f'item num: {n_item}')


    train_weight = []
    train_list = []
    for uid in train_dict:
        int_num = len(train_dict[uid])
        weight = np.linspace(w_min, w_max, int_num)
        train_weight.extend(weight)
        for iid in train_dict[uid]:
            train_list.append([uid, iid])
    train_list = np.array(train_list)
    train_data_temp = sp.csr_matrix((train_weight, \
                (train_list[:, 0], train_list[:, 1])), dtype='float64', \
                shape=(n_user, n_item))

    train_data_ori = sp.csr_matrix((np.ones_like(train_list[:, 0]),
                 (train_list[:, 0], train_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))

    valid_y_data = sp.csr_matrix((np.ones_like(valid_list[:, 0]),
                 (valid_list[:, 0], valid_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # valid_groundtruth

    test_y_data = sp.csr_matrix((np.ones_like(test_list[:, 0]),
                 (test_list[:, 0], test_list[:, 1])), dtype='float64',
                 shape=(n_user, n_item))  # test_groundtruth
    
    return train_data_temp, train_data_ori, valid_y_data, test_y_data, n_user, n_item


class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)
