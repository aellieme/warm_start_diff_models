"""
Run full experiment - train + predict.
"""

import time
import os

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

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, BertConfig, BertModel

from datasets import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn, MaskedLMDataset, MaskedLMPredictionDataset, LastEvaluationDataset
# from datasets import CausalLMDataset, CausalLMPredictionDataset, PaddingCollateFn, MaskedLMDataset, MaskedLMPredictionDataset
from metrics import Evaluator
from modules import SeqRecHuggingface, SeqRec
from models import SASRec, BERT4Rec
from postprocess import preds2recs
from preprocess import add_time_idx


@hydra.main(version_base=None, config_path="configs", config_name="GPT_train_predict")
def main(config):

    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    # if hasattr(config, 'project_name'):
    #     if hasattr(config, 'seed'):
    #         Task.set_random_seed(config.seed)
    #     else:
    #         Task.set_random_seed(None)
    #     task = Task.init(project_name=config.project_name, task_name=config.task_name,
    #                     reuse_last_task_id=False)
    #     task.connect(OmegaConf.to_container(config))
    # else:
    #     task = None

    # train, validation, validation_full, test, item_count = prepare_data(config)
    train, validation, adapt, test, item_count = prepare_data(config)
    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, item_count=item_count)
    start_time = time.time()
    trainer, seqrec_module = training(model, train_loader, eval_loader, config)
    training_time = time.time() - start_time
    print('training_time', training_time)

    if config.test_metrics:
        start_time_inf = time.perf_counter()
        recs = predict(trainer, seqrec_module, train, config, test_data=test, last_evaluation=True)
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
        # metrics_baseline = evaluate(recs, test, train,  config, prefix='test')
        test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
        metrics_baseline = evaluate(recs, test_last, train, config, prefix='test_last')
        
        if adapt is not None and len(adapt) > 0: #адаптация
            print("\nStarting adaptation\n")
            
            #объединяем train и adapt, пересчитываем time_idx
            train_adapt = pd.concat([train, adapt], ignore_index=True)
            train_adapt = add_time_idx(train_adapt)   # пересортировка и новый time_idx
            
            start_time_adapt = time.perf_counter()
            #генерируем рекомендации на основе обновл истории
            # recs_adapt = predict(trainer, seqrec_module, train_adapt, config, last_evaluation=True)
            recs_adapt = predict(trainer, seqrec_module, train_adapt, config, test_data=test, last_evaluation=True)
            adapt_latency = time.perf_counter() - start_time_adapt
            
            #оцениваем на тех же тестовых данных
            print("\nAdaptation metrics on test")
            # metrics_adapt = evaluate(recs_adapt, test, train_adapt, config, prefix='test_adapt')
            metrics_adapt = evaluate(recs_adapt, test_last, train_adapt, config, prefix='test_adapt_last')
            
            
            baseline_prefix = 'test_last_' if config.get('test_metrics', True) else 'val_last_'
            summary = {
                f'Recall@10 (Baseline)': metrics_baseline.get(f'{baseline_prefix}recall@10', 0),
                f'NDCG@10 (Baseline)': metrics_baseline.get(f'{baseline_prefix}ndcg@10', 0),
                f'Coverage (Baseline)': metrics_baseline.get(f'{baseline_prefix}coverage@10', 0),
                f'MRR (Baseline)': metrics_baseline.get(f'{baseline_prefix}mrr@10', 0),
                'Recall (adaptation)': metrics_adapt.get('test_adapt_last_recall@10', 0),
                'NDCG (adaptation)': metrics_adapt.get('test_adapt_last_ndcg@10', 0),
                'Coverage (adaptation)': metrics_adapt.get('test_adapt_last_coverage@10', 0),
                'MRR (adaptation)': metrics_adapt.get('test_adapt_last_mrr@10', 0),
                'Latency (baseline, s)': baseline_latency,
                'Latency (adaptation, s)': adapt_latency,
            }
            
            # summary = {
            # 'Recall@10 (Baseline)': metrics_baseline.get('test_recall@10', 0),
            # 'NDCG@10 (Baseline)': metrics_baseline.get('test_ndcg@10', 0),
            # 'Coverage (Baseline)': metrics_baseline.get('test_coverage@10', 0),
            # 'MRR (Baseline)': metrics_baseline.get('test_mrr@10', 0),
            # 'Recall (adaptation)': metrics_adapt.get('test_adapt_recall@10', 0),
            # 'NDCG (adaptation)': metrics_adapt.get('test_adapt_ndcg@10', 0),
            # 'Coverage (adaptation)': metrics_adapt.get('test_adapt_coverage@10', 0),
            # 'MRR (adaptation)': metrics_adapt.get('test_adapt_mrr@10', 0),
            # 'Latency (baseline, s)': baseline_latency,
            # 'Latency (adaptation, s)': adapt_latency,
            # }

            summary_df = pd.DataFrame([summary])
            print(summary_df.to_string(index=False))
    # if task is not None:
    #     task.get_logger().report_single_value('training_time', training_time)
    #     task.upload_artifact('recs', recs)
    #     task.close()
    torch.save(seqrec_module.model.state_dict(), "best_model.pt")

def prepare_data(config):
    
    # from csv
    # data = pd.read_csv(config.data_path)
    from polara import get_movielens_data
    data = get_movielens_data(include_time=True)   # загружаем датасет
    data = data.rename(columns={'userid': 'user_id', 'movieid': 'item_id'})
    
    
    print('GTS')
    global_time_col = getattr(config, 'global_time_col', 'timestamp')

    ratios = getattr(config, 'split_ratios', [0.7, 0.1, 0.1, 0.1])
    assert len(ratios) == 4 and abs(sum(ratios) - 1.0) < 1e-6

    data = data.sort_values(global_time_col)
    

    time_values = data[global_time_col]
    train_cutoff = time_values.quantile(ratios[0])
    val_cutoff   = time_values.quantile(ratios[0] + ratios[1])
    adapt_cutoff = time_values.quantile(ratios[0] + ratios[1] + ratios[2])
    print(f'time_values {time_values}, train_cutoff {train_cutoff}, val cutoff {val_cutoff}, adapt cutoff {adapt_cutoff}, adapt {adapt_cutoff}')

    train = data[data[global_time_col] <= train_cutoff].copy()
    validation = data[(data[global_time_col] > train_cutoff) & (data[global_time_col] <= val_cutoff)].copy()
    adapt = data[(data[global_time_col] > val_cutoff) & (data[global_time_col] <= adapt_cutoff)].copy()
    test = data[data[global_time_col] > adapt_cutoff].copy()

    train = add_time_idx(train)
    validation = add_time_idx(validation)
    adapt = add_time_idx(adapt)
    test = add_time_idx(test)

    # validation_full как в старом коде 
    train2 = train[train.user_id.isin(validation.user_id.unique())]
    # первое взаимодействие каждого пользователя в validation (time_idx == 0)
    validation2 = validation[validation.time_idx == 0]
    validation_full = pd.concat([train2, validation2])
    validation_full = add_time_idx(validation_full)

    item_count = data.item_id.max()
    print(f'item count {item_count}')

    return train, validation, adapt, test, item_count
    # return train, validation, validation_full, adapt, test, item_count


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

    # if task:

        # clearml_logger = task.get_logger()

        # for key, value in metrics.items():
        #     clearml_logger.report_single_value(key, value)

        # if compute_by_time_idx_flag:
        #     for metric_name in metrics_by_time_idx.columns:
        #         for i, value in metrics_by_time_idx[metric_name].to_dict().items():
        #             clearml_logger.report_scalar(title=prefix + '_' + metric_name,
        #                                          series='by_time_idx', value=value, iteration=i)
        #         for i, value in metrics_by_time_idx_top_k_gt[metric_name].to_dict().items():
        #             clearml_logger.report_scalar(title=prefix + '_' + metric_name,
        #                                          series='by_time_idx_top_k_gt',
        #                                          value=value, iteration=i)

        # metrics = pd.Series(metrics).to_frame().reset_index()
        # metrics.columns = ['metric_name', 'metric_value']
        # clearml_logger.report_table(title=f'{prefix}_metrics', series='dataframe',
        #                             table_plot=metrics)
        # task.upload_artifact(f'{prefix}_metrics', metrics)

        # if compute_by_time_idx_flag:
        #     clearml_logger.report_table(title=f'{prefix}_metrics_by_time_idx', series='dataframe',
        #                                 table_plot=metrics_by_time_idx)
        #     task.upload_artifact(f'{prefix}_metrics_by_time_idx', metrics_by_time_idx)
        #     clearml_logger.report_table(title=f'{prefix}_metrics_by_time_idx_top_k_gt',
        #                                 series='dataframe',
        #                                 table_plot=metrics_by_time_idx_top_k_gt)
        #     task.upload_artifact(f'{prefix}_metrics_by_time_idx_top_k_gt',
        #                       metrics_by_time_idx_top_k_gt)
    return metrics







if __name__ == "__main__":

    main()
    
