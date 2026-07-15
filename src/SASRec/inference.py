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
import argparse 
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_tools.experiment_tracking import checkpoint_path

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty','amazon_Toys_and_Games',
                                 'amazon_Sports_and_Outdoors' ],)
    parser.add_argument('--maxlen', type=int, default=50)
    parser.add_argument('--checkpoint', type=Path, default=None)
    args = parser.parse_args()
    (train_data, val_data, test_data, test_examples,
     data_index, data_description, userid_col, itemid_col, time_col,
     val_seq_dict, _) = prepare_data_and_description(args.dataset)
    # (train_data, val_data, test_data, test_examples, data_index, data_description, userid_col, itemid_col, time_col, val_seq_dict, _) = prepare_data_and_description()
    #  Загрузка модели 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = args.checkpoint or checkpoint_path("SASRec", args.dataset, args.maxlen, seed)
    if not model_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    print(f"Loading model from {model_path}")
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

    
    example_user = users[1]
    print_example_user(
        example_user, users, recs,
        train_data, test_examples,
        data_index, data_description,
        userid_col, itemid_col, time_col, args.dataset
    )

if __name__ == "__main__":
    main()
