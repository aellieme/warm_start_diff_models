import time
import torch
import pandas as pd
import numpy as np
from model import save_sasrec_model, get_model_path, generate_model_name
from training import build_sasrec_model
from load_evaluate_pipeline import (
    prepare_data_and_description,
    run_inference_pipeline,
    print_example_user
)

def main():
    (train_data, val_data, adapt_data, test_data, test_last,
     data_index, data_description, userid_col, itemid_col, time_col) = prepare_data_and_description()

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, "
          f"Adapt: {len(adapt_data)}, Test: {len(test_data)}, "
          f"len test_last: {len(test_last)}")

    config = {
        'num_epochs': 500,
        'maxlen': 200,
        'hidden_units': 128,
        'dropout_rate': 0.5,
        'num_blocks': 2,
        'num_heads': 2,
        'batch_size': 128,
        'sampler_seed': 99,
        'manual_seed': 111,
        'learning_rate': 1e-3,
        'l2_emb': 0.0,
    }

    print("Training SASRec...")
    model, losses = build_sasrec_model(config, train_data, val_data, data_description, patience=10)

    # Сохранение модели
    model_filename = generate_model_name(config, suffix='best')
    model_path = get_model_path(model_filename)
    save_sasrec_model(model, config, data_description, data_index, model_path)
    print(f"Model saved to {model_path}")

    # Baseline 
    print("\nbaseline")
    recs, users, metrics, inf_time = run_inference_pipeline(
        model, train_data, train_data, test_last,
        data_description, userid_col, itemid_col, time_col, topn=10
    )
    precisions, recalls, ndcgs, mrrs, covs = metrics

    print(f"Total inference time: {inf_time:.4f} sec")
    print(f"Evaluated users: {len(users)}")
    for k, p, r, ndcg, mrr, cov in zip([10], precisions, recalls, ndcgs, mrrs, covs):
        print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")

    # Adaptation (train + adapt) 
    print("\nadaptation")
    inference_history = pd.concat([train_data, adapt_data], ignore_index=True)
    recs_adapt, users_adapt, metrics_adapt, inf_time_adapt = run_inference_pipeline(
        model, inference_history, train_data, test_last,
        data_description, userid_col, itemid_col, time_col, topn=10
    )
    precisions_a, recalls_a, ndcgs_a, mrrs_a, covs_a = metrics_adapt

    print(f"Inference latency (total): {inf_time_adapt:.4f} seconds")
    print(f"Evaluated users: {len(users_adapt)}")
    for k, p, r, ndcg, mrr, cov in zip([10], precisions_a, recalls_a, ndcgs_a, mrrs_a, covs_a):
        print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")

    #  Пример для одного пользователя 
    example_user = users_adapt[1]
    print_example_user(
        example_user, users_adapt, recs_adapt,
        train_data, adapt_data, test_last,
        data_index, data_description,
        userid_col, itemid_col, time_col
    )

if __name__ == "__main__":
    main()