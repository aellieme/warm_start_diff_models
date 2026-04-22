"""
Train model.
"""

import time
import os

import hydra
import numpy as np
import pandas as pd
import torch
# from clearml import Task
from omegaconf import OmegaConf
from run_train_predict import prepare_data, create_dataloaders, create_model, training

# import argparse


@hydra.main(version_base=None, config_path="configs", config_name="GPT_train")
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

    train, validation,  test, item_count = prepare_data(config)
    train_loader, eval_loader = create_dataloaders(train, validation, config)
    model = create_model(config, item_count=item_count)
    start_time = time.time()
    trainer, seqrec_module = training(model, train_loader, eval_loader, config)
    training_time = time.time() - start_time
    print('training_time', training_time)
    torch.save(seqrec_module.model.state_dict(), "best_model.pt")



if __name__ == "__main__":

    main()
