import os
import glob
import torch
import logging
from argparse import Namespace

from main import load_and_split_gts, item_num_create, fix_random_seed_as
from model import create_model_diffu, Att_Diffuse_model
from utils import Data_Test
from trainer import evaluate_and_print

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# загрузка последней модели из папки best_models 
list_of_files = glob.glob('best_models/model_*.pt')
if not list_of_files:
    raise FileNotFoundError("No model found in 'best_models' directory. Train a model first.")

latest_model_path = max(list_of_files, key=os.path.getctime)
print(f"Loading model from {latest_model_path}")

checkpoint = torch.load(latest_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
args_dict = checkpoint['args']
args = Namespace(**args_dict)

fix_random_seed_as(args.random_seed)
data_raw = load_and_split_gts(quantiles=(0.7, 0.8, 0.9))
args = item_num_create(args, len(data_raw['smap']))

test_data = Data_Test(data_raw['test_seq'], {uid: [] for uid in data_raw['test_seq']}, data_raw['test'], args)
test_loader = test_data.get_pytorch_dataloaders()

baseline_test_seq = {}
for uid in data_raw['test_seq'].keys():
    full_seq = data_raw['test_seq'][uid]
    adapt_items = set(data_raw.get('adapt_seq', {}).get(uid, []))
    baseline_seq = [item for item in full_seq if item not in adapt_items]
    baseline_test_seq[uid] = baseline_seq
baseline_test_data = Data_Test(baseline_test_seq, {uid: [] for uid in baseline_test_seq}, data_raw['test'], args)
baseline_loader = baseline_test_data.get_pytorch_dataloaders()

diffu_rec = create_model_diffu(args)
model = Att_Diffuse_model(diffu_rec, args)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()

print("\n Inference using saved model")
evaluate_and_print(model, baseline_loader, args, logger, description="baseline")
evaluate_and_print(model, test_loader, args, logger, description="adaptation")

params_json_path = latest_model_path.replace('.pt', '.json').replace('model_', 'params_')
if os.path.exists(params_json_path):
    import json
    with open(params_json_path, 'r') as f:
        best_params = json.load(f)
    print("\nHyperparameters used in this model:")
    for k, v in best_params.items():
        print(f"{k}: {v}")