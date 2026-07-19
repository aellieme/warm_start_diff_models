"""
Predict and evaluate with trained model.
"""

import time
import os
import sys
from pathlib import Path

import torch
import hydra
import numpy as np
import pandas as pd
# from clearml import Task 

import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.callbacks import TQDMProgressBar
from modules import SeqRecHuggingface

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import checkpoint_path
from run_train_predict import prepare_data, create_model, predict, evaluate


from run_train_predict import ( prepare_data, create_model, predict, evaluate, add_time_idx)
from modules import SeqRecHuggingface


@hydra.main(version_base=None, config_path="configs", config_name="GPT_predict")
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
    test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()

    model = create_model(config, item_count=item_count)
    model_checkpoint = Path(config.model_checkpoint)
    if not model_checkpoint.exists():
        model_checkpoint = checkpoint_path(
            "GPTRec", config.dataset_name, int(config.dataset.max_length), 42
        )
    if not model_checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_checkpoint}")
    print(f"Loading model from {model_checkpoint}")
    state_dict = torch.load(model_checkpoint, map_location='cpu')
    model.load_state_dict(state_dict)

    history_before_test = pd.concat([train, validation], ignore_index=True)
    history_before_test = add_time_idx(history_before_test)
    seqrec_module = SeqRecHuggingface(
        model,
        **config.seqrec_module,
        candidate_items=history_before_test.item_id.unique().tolist(),
    )

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
    # recs = predict(trainer, seqrec_module, train, config)
    start_time_inf = time.perf_counter()
    recs = predict(trainer, seqrec_module, history_before_test, config, test_data=test, last_evaluation=True)
    # recs = predict(trainer, seqrec_module, train, config, test_data=test, last_evaluation=True)
    baseline_latency = time.perf_counter() - start_time_inf
    print(f"Baseline inference latency: {baseline_latency:.4f} seconds")

    # oценка baseline на test
    if config.get('test_metrics', True):
        # metrics_baseline = evaluate(recs, test, train, config, prefix='test')
        test_last = test.sort_values('time_idx').groupby('user_id').last().reset_index()
        metrics_baseline = evaluate(
            recs, test_last, history_before_test, config, prefix='test_last'
        )
    else:
        # metrics_baseline = evaluate(recs, validation, train, config, prefix='val')
        val_last = validation.sort_values('time_idx').groupby('user_id').last().reset_index()
        metrics_baseline = evaluate(recs, val_last, train, config, prefix='val_last')
        
        
        
    baseline_prefix = 'test_last_' if config.get('test_metrics', True) else 'val_last_'
    summary = {
        f'Recall@10 (Baseline)': metrics_baseline.get(f'{baseline_prefix}recall@10', 0),
        f'NDCG@10 (Baseline)': metrics_baseline.get(f'{baseline_prefix}ndcg@10', 0),
        f'Coverage (Baseline)': metrics_baseline.get(f'{baseline_prefix}coverage@10', 0),
        f'MRR (Baseline)': metrics_baseline.get(f'{baseline_prefix}mrr@10', 0),
        'Latency (baseline, s)': baseline_latency,
    }
    summary_df = pd.DataFrame([summary])
    print(summary_df.to_string(index=False))    
    # adapt = None
    # if adapt is not None and len(adapt) > 0:
    #     print("\nStarting adaptation")
    #     train_adapt = pd.concat([train, validation, adapt], ignore_index=True)
    #     train_adapt = add_time_idx(train_adapt)

    #     start_time_adapt = time.perf_counter()
    #     recs_adapt = predict(trainer, seqrec_module, train_adapt, config, test_data=test, last_evaluation=True)
    #     adapt_latency = time.perf_counter() - start_time_adapt
    #     print(f"Adaptation inference latency: {adapt_latency:.4f} seconds")

    #     # metrics_adapt = evaluate(recs_adapt, test, train_adapt, config, prefix='test_adapt')
    #     metrics_adapt = evaluate(recs_adapt, test_last, train_adapt, config, prefix='test_adapt_last')

    #     baseline_prefix = 'test_last_' if config.get('test_metrics', True) else 'val_last_'
    # summary = {
    #         f'Recall@10 (Baseline)': metrics_baseline.get(f'{baseline_prefix}recall@10', 0),
    #         f'NDCG@10 (Baseline)': metrics_baseline.get(f'{baseline_prefix}ndcg@10', 0),
    #         f'Coverage (Baseline)': metrics_baseline.get(f'{baseline_prefix}coverage@10', 0),
    #         f'MRR (Baseline)': metrics_baseline.get(f'{baseline_prefix}mrr@10', 0),
    #         'Recall (adaptation)': metrics_adapt.get('test_adapt_last_recall@10', 0),
    #         'NDCG (adaptation)': metrics_adapt.get('test_adapt_last_ndcg@10', 0),
    #         'Coverage (adaptation)': metrics_adapt.get('test_adapt_last_coverage@10', 0),
    #         'MRR (adaptation)': metrics_adapt.get('test_adapt_last_mrr@10', 0),
    #         'Latency (baseline, s)': baseline_latency,
    #         'Latency (adaptation, s)': adapt_latency,
    #     }
    # summary_df = pd.DataFrame([summary])
    # print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()

