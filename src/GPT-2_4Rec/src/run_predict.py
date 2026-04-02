"""
Predict and evaluate with trained model.
"""

import time
import os

import torch
import hydra
import numpy as np
import pandas as pd
# from clearml import Task 

import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar
from modules import SeqRecHuggingface
from run_train_predict import prepare_data, create_model, predict, evaluate


from run_train_predict import ( prepare_data, create_model, predict, evaluate, add_time_idx)
from modules import SeqRecHuggingface


@hydra.main(version_base=None, config_path="configs", config_name="GPT_predict")
def main(config):
    print(OmegaConf.to_yaml(config))

    if hasattr(config, 'cuda_visible_devices'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

    train, validation, validation_full, adapt, test, item_count = prepare_data(config)

    model = create_model(config, item_count=item_count)
    state_dict = torch.load(config.model_checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)

    seqrec_module = SeqRecHuggingface(model, **config.seqrec_module)

    if config.model == 'GPT-2':
        if config.generation:
            seqrec_module.set_predict_mode(
                generate=True,
                mode=config.mode,
                **config.generation_params
            )
        else:
            seqrec_module.set_predict_mode(generate=False)

    trainer = pl.Trainer(
        callbacks=[TQDMProgressBar(refresh_rate=100)],
        enable_checkpointing=False,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )

    print("\nBaseline inference")
    start_time_inf = time.perf_counter()
    recs = predict(trainer, seqrec_module, train, config)
    baseline_latency = time.perf_counter() - start_time_inf
    print(f"Baseline inference latency: {baseline_latency:.4f} seconds")

    # oценка baseline на test
    if config.get('test_metrics', True):
        metrics_baseline = evaluate(recs, test, train, config, prefix='test')
    else:
        metrics_baseline = evaluate(recs, validation, train, config, prefix='val')

    if adapt is not None and len(adapt) > 0:
        print("\nStarting adaptation")
        train_adapt = pd.concat([train, adapt], ignore_index=True)
        train_adapt = add_time_idx(train_adapt)

        start_time_adapt = time.perf_counter()
        recs_adapt = predict(trainer, seqrec_module, train_adapt, config)
        adapt_latency = time.perf_counter() - start_time_adapt
        print(f"Adaptation inference latency: {adapt_latency:.4f} seconds")

        metrics_adapt = evaluate(recs_adapt, test, train_adapt, config, prefix='test_adapt')

        summary = {
            'Recall@10 (Baseline)': metrics_baseline.get('test_recall@10', 0),
            'NDCG@10 (Baseline)': metrics_baseline.get('test_ndcg@10', 0),
            'Coverage (Baseline)': metrics_baseline.get('test_coverage@10', 0),
            'MRR (Baseline)': metrics_baseline.get('test_mrr@10', 0),
            'Recall (adaptation)': metrics_adapt.get('test_adapt_recall@10', 0),
            'NDCG (adaptation)': metrics_adapt.get('test_adapt_ndcg@10', 0),
            'Coverage (adaptation)': metrics_adapt.get('test_adapt_coverage@10', 0),
            'MRR (adaptation)': metrics_adapt.get('test_adapt_mrr@10', 0),
            'Latency (baseline, s)': baseline_latency,
            'Latency (adaptation, s)': adapt_latency,
        }
        summary_df = pd.DataFrame([summary])
        print(summary_df.to_string(index=False))
    else:
        print("\nNo adaptation data available.")


if __name__ == "__main__":
    main()


# @hydra.main(version_base=None, config_path="configs", config_name="GPT_predict")
# def main(config):

#     train_task = Task.get_task(task_id=config.train_task)
#     train_config = OmegaConf.create(train_task.get_configuration_object('OmegaConf'))
#     config = OmegaConf.merge(train_config, config)
#     print(OmegaConf.to_yaml(config))

#     if hasattr(config, 'cuda_visible_devices'):
#         os.environ['CUDA_VISIBLE_DEVICES'] = str(config.cuda_visible_devices)

#     if hasattr(config, 'project_name'):
#         if hasattr(config, 'seed'):
#             Task.set_random_seed(config.seed)
#         else:
#             Task.set_random_seed(None)
#         task = Task.init(project_name=config.project_name, task_name=config.task_name,
#                         reuse_last_task_id=False)
#         task.connect(OmegaConf.to_container(config))
#     else:
#         task = None

#     train, validation, validation_full, test, item_count = prepare_data(config)
#     model = create_model(config, item_count=item_count,
#                          weights_path=train_task.artifacts['model'].get_local_copy())
#     trainer, seqrec_module = create_trainer(model, config)

#     start_time = time.time()
#     if config.test_metrics:
#         recs = predict(trainer, seqrec_module, train, config)
#     else:
#         recs = predict(trainer, seqrec_module,
#                        train[train.user_id.isin(validation.user_id.unique())], config)
#     prediction_time = time.time() - start_time
#     print('prediction_time', prediction_time)
    
#     if hasattr(config, 'optuna_metrics'):
#         val_metrics = evaluate(recs, validation, train, task, config, prefix='val')
#         return val_metrics[val_metrics['metric_name'] == config.optuna_metrics]['metric_value'].values

#     evaluate(recs, validation, train, task, config, prefix='val')
#     if config.test_metrics:
#         evaluate(recs, test, train, task, config, prefix='test')

#     if task is not None:
#         task.get_logger().report_single_value('prediction_time', prediction_time)
#         task.upload_artifact('recs', recs)
#         task.close()


# def create_trainer(model, config):

#     seqrec_module = SeqRecHuggingface(model, **config['seqrec_module'])

#     progress_bar = TQDMProgressBar(refresh_rate=100)
#     callbacks=[progress_bar]

#     trainer = pl.Trainer(callbacks=callbacks, enable_checkpointing=False,
#                          **config['trainer_params'])

#     return trainer, seqrec_module


# if __name__ == "__main__":

#     main()
