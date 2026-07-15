import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm

from plotting import TrainingPlotter
from data_utils import data_to_sequences
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tracking import ExperimentTracker

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

# def build_sasrec_model(config, data, data_description):
#     model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(config, data, data_description)
#     device = 'cpu'
#     if torch.cuda.is_available():
#         device = torch.device(f'cuda:{torch.cuda.current_device()}')
#     losses = {}
#     for epoch in tqdm(range(config['num_epochs'])):
#         losses[epoch] = train_sasrec_epoch(
#             model, n_batches, config['l2_emb'], sampler, optimizer, criterion, device
#         )
#     return model, losses

def build_sasrec_model(config, train_data, val_data, data_description, patience=5):
    """
    Обучает модель с early stopping на val_data (last-item стратегия).
    Возвращает лучшую модель (по HR@10) и словарь потерь.
    """
    import time

    plotter = TrainingPlotter(
        save_dir='./log/',
        model_name=f"SASRec_{time.strftime('%Y%m%d_%H%M%S')}",
        metrics=['loss', 'recall@10']
        )
    model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(config, train_data, data_description)
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{torch.cuda.current_device()}')
        model = model.to(device)

    best_hr = 0.0
    best_model_state = None
    epochs_no_improve = 0
    losses = {}

    for epoch in tqdm(range(config['num_epochs'])):
        # Обучаем одну эпоху
        epoch_loss = train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizer, criterion, device
        )
        losses[epoch] = epoch_loss

        # Валидация после эпохи
        hr, mrr = validate_last_item(model, val_data, train_data, data_description, topn=10)
        avg_loss = np.mean(epoch_loss)
        plotter.update(epoch=epoch, loss=avg_loss, val_recall=hr)
        if epoch % 5 == 0:
            plotter.plot(save=True, show=False, suffix=f'_epoch{epoch}')
        print(f"Epoch {epoch}: loss={np.mean(epoch_loss):.4f}, val_HR@10={hr:.4f}, val_MRR={mrr:.4f}")

        # Early stopping
        if hr > best_hr:
            best_hr = hr
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break
    plotter.plot(save=True, show=False, suffix='_final')
    # Восстанавливаем лучшую модель
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device)
        print(f"Restored best model with val_HR@10={best_hr:.4f}")

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

def validate_last_item(model, val_data, train_data, data_description, topn=10):
    model.eval()
    device = next(model.parameters()).device
    tensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
    from data_utils import data_to_sequences   

    userid = data_description['users']
    itemid = data_description['items']

    train_seq_dict = data_to_sequences(train_data, data_description)

    hits = 0
    reciprocal_ranks = []
    n_users = 0

    with torch.no_grad():
        for _, row in val_data.iterrows():
            uid = row[userid]
            target = row[itemid]
            test_history = row['history']   # из future_data
            # Полная история = train + future (до таргета)
            full_history = train_seq_dict.get(uid, []) + test_history
            if len(full_history) == 0:
                continue
            # Пользователь должен быть в train 
            if uid not in train_seq_dict:
                continue

            seq_tensor = tensor(full_history)
            scores = model.score(seq_tensor).cpu().numpy()
            if scores.ndim == 2:
                scores = scores[0]

            seen = set(full_history)
            for it in seen:
                if it < len(scores):
                    scores[it] = -np.inf

            top_idx = np.argpartition(scores, -topn)[-topn:]
            top_idx = top_idx[np.argsort(-scores[top_idx])]

            if target in top_idx[:topn]:
                hits += 1
                rank = np.where(top_idx == target)[0][0] + 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
            n_users += 1

    hr = hits / n_users if n_users > 0 else 0.0
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    return hr, mrr

def build_final_sasrec_model(config, train_val_data, data_description, num_epochs=None, tracker=None):
    if num_epochs is None:
        num_epochs = config['num_epochs']

    model, sampler, n_batches, criterion, optimizer = prepare_sasrec_model(
        config, train_val_data, data_description
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    plotter = TrainingPlotter(
        save_dir='./log/',
        model_name=f"SASRec_FINAL_{time.strftime('%Y%m%d_%H%M%S')}",
        metrics=['loss']           
    )

    for epoch in tqdm(range(num_epochs), desc='Final training'):
        epoch_loss = train_sasrec_epoch(
            model, n_batches, config['l2_emb'], sampler, optimizer, criterion, device
        )
        avg_loss = np.mean(epoch_loss)
        plotter.update(epoch=epoch, loss=avg_loss)
        if tracker is not None:
            tracker.log_epoch(epoch, train_loss=avg_loss)

        if (epoch % 5 == 0) or (epoch == num_epochs - 1):
            plotter.plot(save=True, show=False, suffix=f'_epoch{epoch}')

    plotter.plot(save=True, show=False, suffix='_final')
    print(f"Final training completed. Loss plot saved in {plotter.save_dir}")
    return model
