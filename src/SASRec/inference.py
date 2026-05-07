import time
import pandas as pd
import torch
from model import load_sasrec_model, get_latest_model_path
from load_evaluate_pipeline import (
    prepare_data_and_description,
    run_inference_pipeline,
    print_example_user
)
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
def main():
    (train_data, val_data, test_data, test_examples, data_index, data_description, userid_col, itemid_col, time_col, val_seq_dict, _) = prepare_data_and_description()
    #  Загрузка модели 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = get_latest_model_path()
    print(f"Loading latest model from {model_path}")
    model, config, _, _ = load_sasrec_model(model_path, device=device)
    print(f"Loaded config: {config}")

    # Baseline 
    # print("\nbaseline")
    recs, users, metrics, inf_time = run_inference_pipeline(
    model, train_data, train_data, test_examples,
        data_description, userid_col, itemid_col, time_col, val_seq_dict, topn=10
    )
    precisions, recalls, ndcgs, mrrs, covs = metrics

    print(f"Total inference time: {inf_time:.4f} sec")
    print(f"Evaluated users: {len(users)}")
    for k, p, r, ndcg, mrr, cov in zip([10], precisions, recalls, ndcgs, mrrs, covs):
        print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")

    # Adaptation 
    # print("\nadaptation")
    # inference_history = pd.concat([train_data, adapt_data], ignore_index=True)
    # recs_adapt, users_adapt, metrics_adapt, inf_time_adapt = run_inference_pipeline(
    #     model, inference_history, train_data, test_examples,
    #     data_description, userid_col, itemid_col, time_col, val_seq_dict, topn=10
    # )
    # precisions_a, recalls_a, ndcgs_a, mrrs_a, covs_a = metrics_adapt

    # print(f"Total inference time: {inf_time_adapt:.4f} sec")
    # print(f"Evaluated users: {len(users_adapt)}")
    # for k, p, r, ndcg, mrr, cov in zip([10], precisions_a, recalls_a, ndcgs_a, mrrs_a, covs_a):
    #     print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")

    # Пример для пользователя 
    example_user = users[1]
    print_example_user(
        example_user, users, recs,
        train_data, test_examples,
        data_index, data_description,
        userid_col, itemid_col, time_col
    )

if __name__ == "__main__":
    main()