"""
Run full experiment - train + predict.
"""

import time
import os
import gzip
import json

import hydra
import numpy as np
import pandas as pd

# from clearml import Task
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary, TQDMProgressBar)
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
from polara import get_movielens_data

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertModel

from plotting import TrainingPlotter
from pytorch_lightning.callbacks import Callback

from datasets import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn, MaskedLMDataset, MaskedLMPredictionDataset, LastEvaluationDataset
# from datasets import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn, MaskedLMDataset, MaskedLMPredictionDataset
from metrics import Evaluator
from modules import SeqRecHuggingface, SeqRec
from models import SASRec, BERT4Rec
from postprocess import preds2recs
from preprocess import add_time_idx


@hydra.main(version_base=None, config_path="configs", config_name="GPT_train_predict")
def main(config):
    import random
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    train, validation, test, item_count = prepare_data(config)
    
    if config.get('final_train', False):
        train_val = pd.concat([train, validation], ignore_index=True)
        train_val = add_time_idx(train_val)

        train_dataset = CausalLMDataset(train_val, **config['dataset'])
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.dataloader.batch_size,
            shuffle=True,
            num_workers=config.dataloader.num_workers,
            collate_fn=PaddingCollateFn()
        )

        model = create_model(config, item_count=item_count)

        seqrec_module, trainer = final_training(model, train_loader, config)

        torch.save(seqrec_module.model.state_dict(), "final_model.pt")

        history_before_test = pd.concat([train, validation], ignore_index=True)
        history_before_test = add_time_idx(history_before_test)
        start_time_inf = time.perf_counter()
        recs = predict(trainer, seqrec_module, history_before_test, config,
                       test_data=test, last_evaluation=True)
        inf_time = time.perf_counter() - start_time_inf
        # print(f"Inference time: {inf_time:.4f} seconds")
        print(f"Inference time: {inf_time:.4f} seconds", flush=True)
        recs.to_csv('recommendations.csv', index=False)
        print("Recommendations saved to recommendations.csv")
        
        test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
        # Объединяем все данные для корректного coverage (полный каталог)
        all_items_df = pd.concat([train, validation, test])
        metrics = evaluate(recs, test_last, all_items_df, config, prefix='test_last')

        # test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
        # metrics = evaluate(recs, test_last, train, config, prefix='test_last')

        # print("Final test metrics:")
        # for key in sorted(metrics.keys()):
        #     if any(metric in key for metric in ['recall', 'ndcg', 'mrr', 'coverage']):
        #         print(f"{key}: {metrics[key]:.6f}")
        #     else:
        #         print(f"{key}: {metrics[key]}")
        # Формируем таблицу для top_k
        top_k_list = config.evaluator.top_k  # [10, 20, 100]
        print("\nFinal test metrics:")
        print(f"{'k':<5} {'recall':<12} {'ndcg':<12} {'mrr':<12} {'coverage':<12}")
        print("-" * 55)
        for k in top_k_list:
            recall_key = f'test_last_recall@{k}'
            ndcg_key   = f'test_last_ndcg@{k}'
            mrr_key    = f'test_last_mrr@{k}'
            cov_key    = f'test_last_coverage@{k}'
            recall = metrics.get(recall_key, 0.0)
            ndcg   = metrics.get(ndcg_key, 0.0)
            mrr    = metrics.get(mrr_key, 0.0)
            cov    = metrics.get(cov_key, 0.0)
            print(f"{k:<5} {recall:<12.6f} {ndcg:<12.6f} {mrr:<12.6f} {cov:<12.6f}")
        
        
        # print("Final test metrics:", metrics)

        # summary = {
        #     'Recall@10': metrics.get('test_last_recall@10', 0),
        #     'NDCG@10': metrics.get('test_last_ndcg@10', 0),
        #     'Coverage': metrics.get('test_last_coverage@10', 0),
        #     'MRR': metrics.get('test_last_mrr@10', 0),
        #     'Latency (s)': time.perf_counter() - start_time_inf,
        # }
        # print(pd.DataFrame([summary]).to_string(index=False))
        return   
    
    else:
        train_loader, eval_loader = create_dataloaders(train, validation, config)
        model = create_model(config, item_count=item_count)
        start_time = time.time()
        trainer, seqrec_module = training(model, train_loader, eval_loader, config)
        training_time = time.time() - start_time
        print('training_time', training_time)

        if config.test_metrics:
            history_before_test = pd.concat([train, validation], ignore_index=True)
            history_before_test = add_time_idx(history_before_test)
            start_time_inf = time.perf_counter()
            recs = predict(trainer, seqrec_module, history_before_test, config, test_data=test, last_evaluation=True)

            # recs = predict(trainer, seqrec_module, train, config, test_data=test, last_evaluation=True)
            # recs = predict(trainer, seqrec_module, train, config)
            baseline_latency = time.perf_counter() - start_time_inf
        else:
            start_time_inf = time.perf_counter()
            # recs = predict(trainer, seqrec_module, train[train.user_id.isin(validation.user_id.unique())], config,last_evaluation=True )
            recs = predict(trainer, seqrec_module, train[train.user_id.isin(validation.user_id.unique())], config, test_data=validation, last_evaluation=True)
            val_last = validation.sort_values('time_idx').groupby('user_id').last().reset_index()
            evaluate(recs, val_last, train, config, prefix='val_last')
            baseline_latency = time.perf_counter() - start_time_inf
        
        if hasattr(config, 'optuna_metrics'):
            # val_metrics = evaluate(recs, validation[validation.time_idx == 0], train,  config, prefix='val')
            val_last = validation.sort_values('time_idx').groupby('user_id').last().reset_index()
            val_metrics = evaluate(recs, val_last, train, config, prefix='val_last')
            return val_metrics[val_metrics['metric_name'] == config.optuna_metrics]['metric_value'].values
        else:
            evaluate(recs, validation, train,  config, prefix='val')
        
        
        if config.test_metrics:
            # metrics_baseline = evaluate(recs, test, train, config, prefix='test')
            test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
            metrics_baseline = evaluate(recs, test_last, train, config, prefix='test_last')
        else:
            val_last = validation.sort_values('time_idx').groupby('user_id').last().reset_index()
            metrics_baseline = evaluate(recs, val_last, train, config, prefix='val_last')

        baseline_prefix = 'test_last_' if config.get('test_metrics', True) else 'val_last_'
        summary = {
            f'Recall@10 ': metrics_baseline.get(f'{baseline_prefix}recall@10', 0),
            f'NDCG@10 ': metrics_baseline.get(f'{baseline_prefix}ndcg@10', 0),
            f'Coverage': metrics_baseline.get(f'{baseline_prefix}coverage@10', 0),
            f'MRR ': metrics_baseline.get(f'{baseline_prefix}mrr@10', 0),
            'Latency (s)': baseline_latency,
        }
        summary_df = pd.DataFrame([summary])
        print(summary_df.to_string(index=False))

        torch.save(seqrec_module.model.state_dict(), "best_model.pt")


def load_amazon(dataset_name, data_dir='../data/amazon'):
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
    #item_id начинаются с 1 для padding=0
    df['user_id'] = pd.Categorical(df['user_id']).codes
    df['item_id'] = pd.Categorical(df['item_id']).codes + 1
    return df

def prepare_data(config):
    dataset_name = config.get('dataset_name', 'ml-1m')
    
    if dataset_name == 'ml-1m':
        data = get_movielens_data(include_time=True)
        data = data.rename(columns={'userid': 'user_id', 'movieid': 'item_id'})
        # унифицируем индексацию (item_id с 1)
        data['user_id'] = pd.Categorical(data['user_id']).codes
        data['item_id'] = pd.Categorical(data['item_id']).codes + 1
    else:
        amazon_data_dir = config.get('amazon_data_dir', '../data/amazon')
        data = load_amazon(dataset_name, amazon_data_dir)
    
    # data = get_movielens_data(include_time=True)   
    # data = data.rename(columns={'userid': 'user_id', 'movieid': 'item_id'})
    
    
    print('GTS')
    global_time_col = getattr(config, 'global_time_col', 'timestamp')

    ratios = getattr(config, 'split_ratios', [0.7, 0.1, 0.2])
    assert len(ratios) == 3 #and abs(sum(ratios) - 1.0) < 1e-6

    data = data.sort_values(global_time_col)
    

    time_values = data[global_time_col]
    train_cutoff = time_values.quantile(ratios[0])
    val_cutoff   = time_values.quantile(ratios[0] + ratios[1])
    # adapt_cutoff = time_values.quantile(ratios[0] + ratios[1] + ratios[2])
    print(f'time_values {time_values}, train_cutoff {train_cutoff}, val cutoff {val_cutoff}')

    train = data[data[global_time_col] <= train_cutoff].copy()
    validation = data[(data[global_time_col] > train_cutoff) & (data[global_time_col] <= val_cutoff)].copy()
    # adapt = data[(data[global_time_col] > val_cutoff) & (data[global_time_col] <= adapt_cutoff)].copy()
    test = data[data[global_time_col] > val_cutoff].copy()

    train = add_time_idx(train)
    validation = add_time_idx(validation)
    # adapt = add_time_idx(adapt)
    test = add_time_idx(test)

    # validation_full как в старом коде 
    train2 = train[train.user_id.isin(validation.user_id.unique())]
    # первое взаимодействие каждого пользователя в validation (time_idx == 0)
    validation2 = validation[validation.time_idx == 0]
    validation_full = pd.concat([train2, validation2])
    validation_full = add_time_idx(validation_full)

    item_count = data.item_id.max()
    print(f'item count {item_count}')

    return train, validation, test, item_count


def create_dataloaders(train, validation, config):

    validation_size = config.dataloader.validation_size
    validation_users = validation.user_id.unique()
    if validation_size and (validation_size < len (validation_users)):
        
        np.random.seed(42)
        validation_users = np.random.choice(validation_users, size=validation_size, replace=False)
        validation = validation[validation.user_id.isin(validation_users)]
    
    user_seq_len = validation.groupby('user_id').size()
    users_with_enough = user_seq_len[user_seq_len >= 2].index
    validation = validation[validation.user_id.isin(users_with_enough)]
    
    train_dataset = MaskedLMDataset(train, **config['dataset']) if config.model == 'BERT4Rec' else CausalLMDataset(train, **config['dataset'])
    max_len = config.dataset.max_length
    if config.generation:
        max_len = max_len - max(config.evaluator.top_k)
    eval_dataset = LastEvaluationDataset(
        train_data=train[train.user_id.isin(validation.user_id.unique())],
        test_data=validation,
        max_length=max_len,
        user_col='user_id', item_col='item_id', time_col='time_idx'
    )
    collate_fn = PaddingCollateFn(left_padding=config.generation)
    # eval_dataset = MaskedLMPredictionDataset(validation, max_length=config.dataset.max_length, validation_mode=True) if config.model == 'BERT4Rec' else CausalLMPredictionDataset(validation, max_length=config.dataset.max_length, validation_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=config.dataloader.batch_size,
                              shuffle=True, num_workers=config.dataloader.num_workers,
                              collate_fn=PaddingCollateFn())
    eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
                             shuffle=False, num_workers=config.dataloader.num_workers,
                             collate_fn=collate_fn)
    # eval_loader = DataLoader(eval_dataset, batch_size=config.dataloader.test_batch_size,
    #                          shuffle=False, num_workers=config.dataloader.num_workers,
    #                          collate_fn=PaddingCollateFn())

    return train_loader, eval_loader


def create_model(config, item_count, weights_path=None):

    if config.model == 'GPT-2':
        gpt2_config = GPT2Config(vocab_size=item_count + 1, **config.model_params)
        model = GPT2LMHeadModel(gpt2_config)
    elif config.model == 'SASRec':
        model = SASRec(item_num=item_count, **config.model_params)
    elif config.model == 'BERT4Rec':
        model = BERT4Rec(vocab_size=item_count + 1, add_head=True, tie_weights=True, bert_config=config.model_params) #######################?
        
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    return model

class PlottingCallback(Callback):
    def __init__(self, plotter, save_every=5):
        self.plotter = plotter
        self.save_every = save_every
    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = metrics.get('train_loss')
        val_loss = metrics.get('val_loss')
        val_recall = metrics.get('val_hit_rate')
        
        # Преобразование тензоров в числа (если они есть)
        if train_loss is not None:
            train_loss = train_loss.item() if hasattr(train_loss, 'item') else train_loss
        if val_loss is not None:
            val_loss = val_loss.item() if hasattr(val_loss, 'item') else val_loss
        if val_recall is not None:
            val_recall = val_recall.item() if hasattr(val_recall, 'item') else val_recall
        
        self.plotter.update(
            epoch=epoch,
            loss=train_loss,
            val_loss=val_loss,
            recall=val_recall
        )
        if (epoch % self.save_every == 0) or (epoch == trainer.max_epochs - 1):
            self.plotter.plot(save=True, show=False)


def training(model, train_loader, eval_loader, config):

    if config.model == 'GPT-2':
        seqrec_module = SeqRecHuggingface(model, **config['seqrec_module'])
    elif config.model == 'SASRec':
        seqrec_module = SeqRec(model, **config['seqrec_module'])
    elif config.model == 'BERT4Rec':
        seqrec_module = SeqRec(model, **config['seqrec_module'])

    early_stopping = EarlyStopping(monitor="val_ndcg", mode="max",
                                   patience=config.patience, verbose=False)
    model_summary = ModelSummary(max_depth=4)
    checkpoint = ModelCheckpoint(save_top_k=1, monitor="val_ndcg",
                                 mode="max", save_weights_only=True)
    progress_bar = TQDMProgressBar(refresh_rate=100)
    callbacks=[early_stopping, model_summary, checkpoint, progress_bar]

    from datetime import datetime

    os.makedirs("./log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./log/{timestamp}"
    plotter = TrainingPlotter(save_dir, model_name=config.model, metrics=['loss', 'val_loss', 'recall'])
    plotting_callback = PlottingCallback(plotter, save_every=5)

    callbacks.append(plotting_callback)   # добавляем в существующий список
    
    trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=True,
                         **config['trainer_params'])

    trainer.fit(model=seqrec_module,
            train_dataloaders=train_loader,
            val_dataloaders=eval_loader)

    seqrec_module.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

    return trainer, seqrec_module

def predict(trainer, seqrec_module, data, config, test_data=None, last_evaluation=False):
    if last_evaluation and test_data is not None:
        #  Last Evaluation
        max_len = config.dataset.max_length
        if config.generation:
            max_len = max_len - max(config.evaluator.top_k)
        dataset = LastEvaluationDataset(
            train_data=data,
            test_data=test_data,
            max_length=max_len,
            user_col='user_id',
            item_col='item_id',
            time_col='time_idx'
        )
        collate_fn = PaddingCollateFn(left_padding=config.generation)
        loader = DataLoader(
            dataset,
            shuffle=False,
            collate_fn=collate_fn,
            batch_size=config.dataloader.test_batch_size,
            num_workers=config.dataloader.num_workers
        )
        
        if config.model == 'GPT-2':
            seqrec_module.set_predict_mode(
                generate=config.generation,
                mode=config.mode,
                **config.generation_params
            )
        
        seqrec_module.predict_top_k = max(config.evaluator.top_k)
        preds = trainer.predict(model=seqrec_module, dataloaders=loader)
        recs = preds2recs(preds)
        print('recs shape', recs.shape)
        return recs

    # Стандартный режим (без изменений)
    if config.model == 'GPT-2':
        if config.generation:
            predict_dataset = CausalLMPredictionDataset(
                data, max_length=config.dataset.max_length - max(config.evaluator.top_k))
            predict_loader = DataLoader(
                predict_dataset, shuffle=False,
                collate_fn=PaddingCollateFn(left_padding=True),
                batch_size=config.dataloader.test_batch_size,
                num_workers=config.dataloader.num_workers)
            seqrec_module.set_predict_mode(generate=True, mode=config.mode, **config.generation_params)
        else:
            predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset.max_length)
            predict_loader = DataLoader(
                predict_dataset, shuffle=False,
                collate_fn=PaddingCollateFn(),
                batch_size=config.dataloader.test_batch_size,
                num_workers=config.dataloader.num_workers)
            seqrec_module.set_predict_mode(generate=False)
        
    elif config.model == 'SASRec':
        predict_dataset = CausalLMPredictionDataset(data, max_length=config.dataset.max_length)
        predict_loader = DataLoader(
            predict_dataset, shuffle=False,
            collate_fn=PaddingCollateFn(),
            batch_size=config.dataloader.test_batch_size,
            num_workers=config.dataloader.num_workers)
        
    elif config.model == 'BERT4Rec':
        predict_dataset = MaskedLMPredictionDataset(data, max_length=config.dataset.max_length)
        predict_loader = DataLoader(
            predict_dataset, shuffle=False,
            collate_fn=PaddingCollateFn(),
            batch_size=config.dataloader.test_batch_size,
            num_workers=config.dataloader.num_workers)

    seqrec_module.predict_top_k = max(config.evaluator.top_k)
    preds = trainer.predict(model=seqrec_module, dataloaders=predict_loader)
    recs = preds2recs(preds)
    print('recs shape', recs.shape)
    return recs


def evaluate(recs, test, train,  config, prefix='test'):

    evaluator = Evaluator(**config['evaluator'])

    metrics = evaluator.compute_metrics(test, recs, train)
    metrics = {prefix + '_' + key: value for key, value in metrics.items()}
    print(f'{prefix} metrics\n', metrics)

    compute_by_time_idx_flag = test['time_idx'].nunique() > 1
    if compute_by_time_idx_flag: #подробные метрики я сохраняю в файлы csv в папку метрикс csv, она создается если ее нет 
        metrics_by_time_idx = evaluator.compute_metrics_by_time_idx(test, recs)
        metrics_by_time_idx_top_k_gt = evaluator.compute_metrics_by_time_idx(test, recs, top_k_gt=True)
        os.makedirs("metrics_csv", exist_ok=True)
        metrics_by_time_idx.to_csv(f"metrics_csv/{prefix}_metrics_by_time_idx.csv", index=False)
        metrics_by_time_idx_top_k_gt.to_csv(f"metrics_csv/{prefix}_metrics_by_time_idx_top_k_gt.csv", index=False)
        print(f"\nSaved metrics_by_time_idx to metrics_csv/{prefix}_metrics_by_time_idx.csv")
        print(f"\nSaved metrics_by_time_idx_top_k_gt to metrics_csv/{prefix}_metrics_by_time_idx_top_k_gt.csv")

    return metrics

def final_training(model, train_loader, config):
    if config.model == 'GPT-2':
        seqrec_module = SeqRecHuggingface(model, **config['seqrec_module'])
    elif config.model == 'SASRec':
        seqrec_module = SeqRec(model, **config['seqrec_module'])
    elif config.model == 'BERT4Rec':
        seqrec_module = SeqRec(model, **config['seqrec_module'])
    else:
        raise ValueError(f"Unknown model: {config.model}")

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plotter = TrainingPlotter(save_dir=f"./log/{timestamp}",
                              model_name=config.model,
                              metrics=['loss'])

    class FinalLossCallback(pl.Callback):
        def __init__(self, plotter):
            self.plotter = plotter
        def on_train_epoch_end(self, trainer, pl_module):
            loss = trainer.callback_metrics.get('train_loss')
            if loss is not None:
                self.plotter.update(epoch=trainer.current_epoch,
                                    loss=loss.item())

    loss_callback = FinalLossCallback(plotter)
    progress_bar = TQDMProgressBar(refresh_rate=100)

    trainer_params = dict(config.get('trainer_params', {}))
    trainer_params.pop('max_epochs', None)

    trainer = pl.Trainer(
        callbacks=[progress_bar, loss_callback],
        max_epochs=config.get('final_epochs', 80),
        enable_checkpointing=False,
        **trainer_params
    )

    trainer.fit(model=seqrec_module, train_dataloaders=train_loader)

    plotter.plot(save=True, show=False)
    print(f"График сохранён в: {plotter.save_dir}") 
    return seqrec_module, trainer

if __name__ == "__main__":

    main()
    
