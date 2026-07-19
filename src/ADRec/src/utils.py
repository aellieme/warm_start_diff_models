import torch.utils.data as data_utils
import torch
from tqdm import tqdm
from einops import pack,unpack
import torch.backends.cudnn as cudnn
import numpy as np
from collections import Counter
import random
from scipy.stats import beta


def build_candidate_mask(candidate_items, num_items, device):
    """Return a boolean mask for real recommendable items; 0 is padding."""
    mask = torch.zeros(num_items, dtype=torch.bool, device=device)
    valid_items = sorted({
        int(item) for item in candidate_items
        if 0 < int(item) < num_items
    })
    if valid_items:
        indices = torch.as_tensor(valid_items, dtype=torch.long, device=device)
        mask[indices] = True
    return mask


def filter_history_to_candidates(sequences, candidate_mask):
    """Remove items unavailable at the evaluation boundary from histories."""
    valid_ids = (sequences > 0) & (sequences < candidate_mask.numel())
    safe_ids = sequences.clamp(min=0, max=candidate_mask.numel() - 1)
    known_items = valid_ids & candidate_mask[safe_ids]
    return sequences.masked_fill(~known_items, 0)


def prepare_model_history(full_history, candidate_mask, max_len):
    """Filter the full history, then compact its last known items for the model."""
    filtered = filter_history_to_candidates(full_history, candidate_mask)
    compact = torch.zeros(
        (filtered.shape[0], max_len), dtype=filtered.dtype, device=filtered.device
    )
    for row_index, row in enumerate(filtered):
        known = row[row > 0][-max_len:]
        if known.numel():
            compact[row_index, -known.numel():] = known
    return compact


def eligible_warm_start_rows(sequences, targets, candidate_mask):
    """Select examples with a known target and a non-empty known-item history."""
    targets = targets[..., -1]
    valid_ids = (targets > 0) & (targets < candidate_mask.numel())
    safe_targets = targets.clamp(min=0, max=candidate_mask.numel() - 1)
    return valid_ids & candidate_mask[safe_targets] & (sequences > 0).any(dim=1)


def mask_ranking_scores(scores, sequences, candidate_mask):
    """Mask padding, out-of-catalogue items and all observed items in-place."""
    if scores.shape[-1] != candidate_mask.numel():
        raise ValueError("Candidate mask and score vocabulary sizes differ")
    scores.masked_fill_(~candidate_mask.unsqueeze(0), -torch.inf)
    valid_seen = (sequences > 0) & (sequences < scores.shape[-1])
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    rows = rows.expand_as(sequences)
    scores[rows[valid_seen], sequences[valid_seen]] = -torch.inf
    return scores


def mask_training_scores(scores, candidate_mask):
    """Exclude unavailable items from classification losses."""
    if scores.shape[-1] != candidate_mask.numel():
        raise ValueError("Candidate mask and score vocabulary sizes differ")
    return scores.masked_fill(
        ~candidate_mask.view(*([1] * (scores.ndim - 1)), -1),
        torch.finfo(scores.dtype).min,
    )


def build_final_train_sequences(data_raw):
    """Build complete train+validation sequences, including validation-only users."""
    sequences = []
    user_ids = set(data_raw['train_dict']) | set(data_raw['val_seq_dict'])
    for uid in sorted(user_ids):
        if uid in data_raw['val_seq_dict'] and uid in data_raw['val_tgt_dict']:
            # val_seq_dict already contains the user's train history.
            sequence = list(data_raw['val_seq_dict'][uid])
            sequence.append(data_raw['val_tgt_dict'][uid])
        else:
            sequence = list(data_raw['train_dict'][uid])
        sequences.append(sequence)
    return sequences


# import polara
# from polara.datasets.movielens import get_movielens_data
from sklearn.preprocessing import LabelEncoder

# def load_and_split_gts(quantiles=(0.7, 0.8)):

#     df = get_movielens_data(include_time=True)
    
#     user_enc = LabelEncoder()
#     item_enc = LabelEncoder()
#     # df['userid'] = user_enc.fit_transform(df['userid'])
#     # df['movieid'] = item_enc.fit_transform(df['movieid'])
#     # item_smap = {idx: orig for idx, orig in enumerate(item_enc.classes_)}
#     df['userid'] = user_enc.fit_transform(df['userid'])
#     df['movieid'] = item_enc.fit_transform(df['movieid']) + 1  # сдвиг, 0 резервирован под паддинг
#     item_smap = {idx+1: orig for idx, orig in enumerate(item_enc.classes_)}  # сдвинутое сопоставление
#     # item_count остаётся len(item_enc.classes_) – 3706, макс. ID = 3706
    
#     df = df.sort_values('timestamp').reset_index(drop=True)
    
#     T_valid = df['timestamp'].quantile(quantiles[0])   # 0.7
#     T_test  = df['timestamp'].quantile(quantiles[1])   # 0.8
    
#     train_dict = {}
#     val_seq_dict = {}
#     val_tgt_dict = {}
#     test_seq_dict = {}
#     test_tgt_dict = {}
    
#     for uid, group in df.groupby('userid'):
#         group = group.sort_values('timestamp')
#         items = group['movieid'].tolist()
#         times = group['timestamp'].tolist()
        
#         train_seq = [item for item, ts in zip(items, times) if ts <= T_valid]
#         if len(train_seq) > 0:
#             train_dict[uid] = train_seq
        
#         val_window = [(item, ts) for item, ts in zip(items, times) if T_valid < ts <= T_test]
#         if val_window:
#             val_tgt = val_window[-1][0]
#             val_hist = [item for item, _ in val_window[:-1]]
#             val_seq_dict[uid] = train_seq + val_hist
#             val_tgt_dict[uid] = val_tgt
        
#         test_window = [(item, ts) for item, ts in zip(items, times) if ts > T_test]
#         if test_window:
#             test_tgt = test_window[-1][0]
#             test_hist = [item for item, _ in test_window[:-1]]
            
#             full_val_seq = [item for item, _ in val_window] if val_window else []
#             test_seq_dict[uid] = train_seq + full_val_seq + test_hist
#             test_tgt_dict[uid] = test_tgt
    
    # val_seq_list = [val_seq_dict[uid] for uid in sorted(val_seq_dict.keys())]
    # val_tgt_list = [val_tgt_dict[uid] for uid in sorted(val_seq_dict.keys())]
    
    # test_seq_list = [test_seq_dict[uid] for uid in sorted(test_seq_dict.keys())]
    # test_tgt_list = [test_tgt_dict[uid] for uid in sorted(test_seq_dict.keys())]
    
    # return {
    #     'train': list(train_dict.values()),          
    #     'val_seq': val_seq_list,                     
    #     'val_tgt': val_tgt_list,                     
    #     'test_seq': test_seq_list,
    #     'test_tgt': test_tgt_list,
    #     'item_smap': item_smap,
    #     'item_count': len(item_enc.classes_)
    # }
    # return {
    #     'train': list(train_dict.values()),
    #     'val_seq': val_seq_list,
    #     'val_tgt': val_tgt_list,
    #     'test_seq': test_seq_list,
    #     'test_tgt': test_tgt_list,
    #     'item_smap': item_smap,
    #     'item_count': len(item_enc.classes_),
    #     'train_dict': train_dict,
    #     'val_seq_dict': val_seq_dict,
    #     'val_tgt_dict': val_tgt_dict,
    #     'test_seq_dict': test_seq_dict,
    #     'test_tgt_dict': test_tgt_dict,
    # }


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, id2seq, max_len,parallel_ag=False):
        self.id2seq = id2seq
        self.max_len = max_len
        self.parallel = parallel_ag
    def __len__(self):
        return len(self.id2seq)
    def __getitem__(self, index):
        seq = self._getseq(index)
        hist = seq[:-1]
        hist = hist[-self.max_len:]
        mask_len = self.max_len - len(hist)
        hist_pad = [0] * mask_len + hist
        if self.parallel is True:
            target = [0] * mask_len + seq[-len(hist):]
        else:
            target = [0] * (self.max_len-1) + [seq[-1]]

        hist_pad = hist_pad[-self.max_len:]
        target = target[-self.max_len:]

        return torch.LongTensor(hist_pad), torch.LongTensor(target)
    # def __getitem__(self, index):
    #     seq = self._getseq(index)
    #     hist = seq[:-1]
    #     hist = hist[-self.max_len:]
    #     mask_len = self.max_len - len(hist)
    #     hist_pad = [0] * mask_len + hist
    #     if self.parallel is True:
    #         # mask_len = self.max_len - len(target)
    #         target = [0] * mask_len + seq[-len(hist):]
    #         # assert sum([i>0 for i in hist_pad]) == sum([i>0 for i in target])
    #     else:
    #         target = [0] * (self.max_len-1) + [seq[-1]]

    #     return torch.LongTensor(hist_pad), torch.LongTensor(target)

    def _getseq(self, idx):
        return self.id2seq[idx]


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        # A sequential training example needs at least one history item and a target.
        self.id_seq = [sequence for sequence in data_train if len(sequence) >= 2]
        self.split = args.split_onebyone
        self.parallel_ag = args.parallel_ag
        if self.split:
            print('splitting data onebyone ...')
            self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        idx = 0
        for seq_temp in self.u2seq:
            # Ограничиваем длину исходной последовательности до max_len+1
            if len(seq_temp) > self.max_len + 1:
                seq_temp = seq_temp[-self.max_len-1:]
            # Генерируем подпоследовательности
            for star in range(len(seq_temp) - 1):
                subseq = seq_temp[:star+2]
                # Дополнительно обрезаем, если вдруг subseq длиннее max_len+1
                if len(subseq) > self.max_len + 1:
                    subseq = subseq[-self.max_len-1:]
                self.id_seq[idx] = subseq
                idx += 1
    
    # def split_onebyone(self):
    #     self.id_seq = {}
    #     idx = 0
    #     for seq_temp in self.u2seq:
    #         seq_temp = seq_temp[-self.max_len-1:]
    #         # 只能从预截取长度后进行子序列切分
    #         # 加一是为了包含tgt
    #         for star in range(len(seq_temp) - 1):
    #             self.id_seq[idx] = seq_temp[:star + 2]
    #             idx += 1

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.max_len,self.parallel_ag)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.full_history_len = max(1, max(len(seq) for seq in self.u2seq))
    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        # user = self.users[index]
        full_seq = self.u2seq[index]
        hist = full_seq[-self.max_len:]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        # answer_pad = [0] * padding_len + seq[-(len(hist)-1):] + self.u2answer[index]
        answer_pad = [0] * padding_len + seq[-(len(hist)-1):] + [self.u2answer[index]]
        # assert sum([i>0 for i in hist_pad]) == sum([i>0 for i in answer_pad])
        hist_pad = hist_pad[-self.max_len:]
        answer_pad = answer_pad[-self.max_len:]
        full_seq = [0] * (self.full_history_len - len(full_seq)) + full_seq
        return (
            torch.LongTensor(hist_pad),
            torch.LongTensor(answer_pad),
            torch.LongTensor(full_seq),
        )


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        # self.parallel_ag = True if args.model == 'adrec' else False

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        # self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.full_history_len = max(
            1,
            max(len(seq) + len(extra) for seq, extra in zip(self.u2seq, self.u2seq_add)),
        )

    def __len__(self):
        return len(self.u2seq)

    def __getitem__(self, index):
        # user = self.users[index]
        full_seq = self.u2seq[index] + self.u2seq_add[index]
        # seq = self.u2seq[user]
        hist = full_seq[-self.max_len:]
        padding_len = self.max_len - len(hist)
        hist_pad = [0] * padding_len + hist
        # answer_pad = [0] * padding_len + seq[-(len(hist)-1):] + self.u2answer[index]
        answer_pad = [0] * padding_len + full_seq[-(len(hist)-1):] + [self.u2answer[index]]
        # assert sum([i>0 for i in hist_pad]) == sum([i>0 for i in answer_pad])
        hist_pad = hist_pad[-self.max_len:]
        answer_pad = answer_pad[-self.max_len:]
        full_seq = [0] * (self.full_history_len - len(full_seq)) + full_seq
        return (
            torch.LongTensor(hist_pad),
            torch.LongTensor(answer_pad),
            torch.LongTensor(full_seq),
        )


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader



def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """

    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))
def exists(v):
    return v is not None
def identity(t, *args, **kwargs):
    return t
def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

# tensor helpers

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def safe_div(num, den, eps = 1e-5):
    return num / den.clamp(min = eps)

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim

    if padding_dims <= 0:
        return t

    return t.view(*t.shape, *((1,) * padding_dims))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern = None):
        unpacked, = unpack(to_unpack, ps, default(unpack_pattern, pattern))
        return unpacked

    return packed, unpack_one


def exponential_mapping(x, v):
    # exp_x[v] = cos(||v||) * x + sin(||v||) * (v / ||v||)
    norm_v = v.norm(p=2, dim=-1, keepdim=True)  # L2 norm of v
    v_unit = v / (norm_v + 1e-8)  # Normalize v to unit vector
    cos_v = torch.cos(norm_v)  # Cosine of the norm of v
    sin_v = torch.sin(norm_v)  # Sine of the norm of v
    return cos_v * x + sin_v * v_unit  # Geodesic exponential map


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


