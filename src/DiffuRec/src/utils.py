import torch.utils.data as data_utils
import torch
from bisect import bisect_right


def encode_item_ids_with_padding(values):
    """Encode real items as 1..N and reserve 0 exclusively for padding."""
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    encoded = encoder.fit_transform(values) + 1
    mapping = {
        index + 1: original
        for index, original in enumerate(encoder.classes_)
    }
    return encoded, mapping


def build_candidate_mask(candidate_items, num_items, device):
    """Return a device mask for recommendable item IDs; 0 is always padding."""
    mask = torch.zeros(num_items, dtype=torch.bool, device=device)
    valid_items = sorted({
        int(item) for item in candidate_items
        if 0 < int(item) < num_items
    })
    if valid_items:
        mask[torch.as_tensor(valid_items, dtype=torch.long, device=device)] = True
    return mask


def mask_ranking_scores(scores, sequences, candidate_mask):
    """Mask padding, out-of-catalogue items and the user's observed history."""
    if scores.shape[-1] != candidate_mask.numel():
        raise ValueError("Candidate mask and score vocabulary sizes differ")
    scores.masked_fill_(~candidate_mask.unsqueeze(0), -torch.inf)

    valid_seen = (sequences > 0) & (sequences < scores.shape[-1])
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1)
    rows = rows.expand_as(sequences)
    scores[rows[valid_seen], sequences[valid_seen]] = -torch.inf
    return scores


def eligible_warm_start_rows(sequences, targets, candidate_mask):
    """Keep examples with non-empty history and a target in the known catalogue."""
    targets = targets.squeeze(-1)
    valid_target_id = (targets > 0) & (targets < candidate_mask.numel())
    safe_targets = targets.clamp(min=0, max=candidate_mask.numel() - 1)
    return valid_target_id & candidate_mask[safe_targets] & (sequences > 0).any(dim=1)


def filter_history_to_candidates(sequences, candidate_mask):
    """Replace padding and items outside the known catalogue with padding."""
    valid_id = (sequences > 0) & (sequences < candidate_mask.numel())
    safe_ids = sequences.clamp(min=0, max=candidate_mask.numel() - 1)
    known = valid_id & candidate_mask[safe_ids]
    return sequences.masked_fill(~known, 0)


class TrainDataset(data_utils.Dataset):
    def __init__(self, sequences, max_len):
        # Keep one reference per user sequence instead of materializing every
        # prefix.  On ML-1M the old representation copied hundreds of millions
        # of Python integers before the first epoch even started.
        self.sequences = tuple(sequences)
        self.max_len = max_len
        self.cumulative_examples = []
        total = 0
        for sequence in self.sequences:
            total += max(0, len(sequence) - 1)
            self.cumulative_examples.append(total)

    def __len__(self):
        return self.cumulative_examples[-1] if self.cumulative_examples else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)

        sequence_index = bisect_right(self.cumulative_examples, index)
        previous_total = self.cumulative_examples[sequence_index - 1] if sequence_index else 0
        target_position = index - previous_total + 1
        sequence = self.sequences[sequence_index]

        labels = [sequence[target_position]]
        tokens = sequence[max(0, target_position - self.max_len):target_position]
        mask_len = self.max_len - len(tokens)
        tokens = [0] * mask_len + tokens
        return torch.LongTensor(tokens), torch.LongTensor(labels)


class Data_Train():
    def __init__(self, data_train, args):
        self.u2seq = data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.num_workers = getattr(args, 'num_workers', 2)
        self.pin_memory = getattr(args, 'device', 'cpu') == 'cuda'

    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.u2seq.values(), self.max_len)
        return data_utils.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, max_len):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq),  torch.LongTensor(answer)


class Data_Val():
    def __init__(self, data_train, data_val, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer, max_len):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user] + self.u2seq_add[user]
        # seq = self.u2seq[user]
        answer = self.u2answer[user]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_Test():
    def __init__(self, data_train, data_val, data_test, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.max_len = args.max_len

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
        return dataloader


class CHLSDataset(data_utils.Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        data_temp = self.data[index]
        seq = data_temp[:-1]
        answer = [data_temp[-1]]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq
        return torch.LongTensor(seq), torch.LongTensor(answer)


class Data_CHLS():
    def __init__(self, data, args):
        self.batch_size = args.batch_size
        self.max_len = args.max_len
        self.data = data

    def get_pytorch_dataloaders(self):
        dataset = CHLSDataset(self.data, self.max_len)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=2)
        return dataloader
