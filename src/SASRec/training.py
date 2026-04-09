import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from data_utils import data_to_sequences

def random_neq(l, r, s, random_state):
    t = random_state.randint(l, r)
    while t in s:
        t = random_state.randint(l, r)
    return t

def sequential_batch_sampler(user_train, usernum, itemnum, batch_size, maxlen, seed, pad_token=None):
    if pad_token is None:
        pad_token = itemnum

    def sample(random_state):
        user = random_state.randint(usernum)
        while len(user_train.get(user, [])) <= 1:
            user = random_state.randint(usernum)
        user_items = user_train[user]
        seq = np.full(maxlen, pad_token, dtype=np.int32)
        pos = np.full(maxlen, pad_token, dtype=np.int32)
        neg = np.full(maxlen, pad_token, dtype=np.int32)
        nxt = user_items[-1]
        idx = maxlen - 1
        ts = set(user_items)
        for i in reversed(user_items[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(0, itemnum, ts, random_state)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        return (user, seq, pos, neg)

    random_state = np.random.RandomState(seed)
    while True:
        yield zip(*(sample(random_state) for _ in range(batch_size)))

def prepare_sasrec_model(config, data, data_description):
    n_users = data_description['n_users']
    n_items = data_description['n_items']
    from model import SASRec
    model = SASRec(n_items, config)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=model.pad_token)
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    train_sequences = data_to_sequences(data, data_description)
    sampler = sequential_batch_sampler(
        train_sequences, n_users, n_items,
        batch_size=config['batch_size'],
        maxlen=config['maxlen'],
        seed=config['sampler_seed'],
        pad_token=model.pad_token
    )
    n_batches = len(train_sequences) // config['batch_size']
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98)
    )
    return model, sampler, n_batches, criterion, optimizer

def train_sasrec_epoch(model, num_batch, l2_emb, sampler, optimizer, criterion, device):
    model.train()
    losses = []
    for _ in range(num_batch):
        _, seq_data, pos_data, _ = next(sampler)
        seq = torch.LongTensor(np.array(seq_data)).to(device)
        pos = torch.LongTensor(np.array(pos_data)).to(device)
        optimizer.zero_grad()
        logits = model(seq)
        logits_flat = logits.view(-1, logits.size(-1))
        pos_flat = pos.view(-1)
        loss = criterion(logits_flat, pos_flat)
        if l2_emb != 0:
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.norm(param)**2
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

def build_sasrec_model(config, data, data_description):
    model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(config, data, data_description)
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
    losses = {}
    for epoch in tqdm(range(config['num_epochs'])):
        losses[epoch] = train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizer, criterion, device
        )
    return model, losses

def sasrec_model_scoring(model, data, data_description):
    model.eval()
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    test_sequences = data_to_sequences(data, data_description)

    scores = []
    user_order = []

    for uid, seq in test_sequences.items():
        with torch.no_grad():
            predictions = model.score(tensor(seq))
        scores.append(predictions.detach().cpu().numpy())
        user_order.append(uid)

    return np.concatenate(scores, axis=0), user_order