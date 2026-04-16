import os
import glob
import torch
import logging
from argparse import Namespace
import pandas as pd
import random   

from main import load_and_split_gts, item_num_create, fix_random_seed_as
from model import create_model_diffu, Att_Diffuse_model
from utils import Data_Test
from trainer import evaluate_and_print

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_movies_metadata(data_dir='../data/info'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        if not os.path.exists(os.path.join(data_dir, 'movies.dat')):
            os.system('wget -O ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip')
            os.system('unzip -o ml-1m.zip -d ../data/')
            os.system('mv ../data/ml-1m/movies.dat ' + data_dir)
    movies_meta = pd.read_csv(
        os.path.join(data_dir, 'movies.dat'),
        sep='::', engine='python',
        encoding='ISO-8859-1',
        names=['movieId', 'title', 'genres']
    )
    return dict(zip(movies_meta.movieId, movies_meta.title))

def decode_item(inner_id, smap, movie_dict):
    original_id = smap[inner_id]
    return movie_dict.get(original_id, str(original_id))

def recommend_for_user(model, history_seq, target_item, args, topk=10):
    model.eval()
    device = args.device
    max_len = args.max_len
    seq = history_seq[-max_len:]
    pad_len = max_len - len(seq)
    seq = [0] * pad_len + seq
    seq_tensor = torch.LongTensor(seq).unsqueeze(0).to(device)
    tag_tensor = torch.LongTensor([[0]]).to(device)   # фиктивный tag
    with torch.no_grad():
        _, rep_diffu, _, _, _, _ = model(seq_tensor, tag_tensor, train_flag=False)
        scores = model.diffu_rep_pre(rep_diffu)
        _, topk_indices = torch.topk(scores, k=topk, dim=-1)
    return topk_indices[0].cpu().tolist()


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

#  Демонстрация для случайного пользователя
movie_dict = load_movies_metadata()
smap = data_raw['smap']

test_users = list(data_raw['test_seq'].keys())
if test_users:
    rand_uid = random.choice(test_users)
    print(f"\n--- Пример для пользователя {rand_uid} ---")
    
    full_seq = data_raw['test_seq'][rand_uid]
    adapt_items = set(data_raw.get('adapt_seq', {}).get(rand_uid, []))
    baseline_history = [item for item in full_seq if item not in adapt_items]
    target = data_raw['test'][rand_uid][0]
    
    recs_baseline = recommend_for_user(model, baseline_history, target, args, topk=10)
    recs_adapt = recommend_for_user(model, full_seq, target, args, topk=10)
    
    print("\nИстория (baseline, последние 10):")
    for idx in baseline_history[-10:]:
        print(f"  {decode_item(idx, smap, movie_dict)}")
    
    print(f"\nПравильный ответ: {decode_item(target, smap, movie_dict)}")
    
    print("\nРекомендации baseline (top-10):")
    for rank, rec in enumerate(recs_baseline, 1):
        print(f"  {rank}. {decode_item(rec, smap, movie_dict)}")
    
    print("\nРекомендации c адаптацией (top-10):")
    for rank, rec in enumerate(recs_adapt, 1):
        print(f"  {rank}. {decode_item(rec, smap, movie_dict)}")
else:
    print("Нет пользователей для демонстрации")

params_json_path = latest_model_path.replace('.pt', '.json').replace('model_', 'params_')
if os.path.exists(params_json_path):
    import json
    with open(params_json_path, 'r') as f:
        best_params = json.load(f)
    print("\nHyperparameters used in this model:")
    for k, v in best_params.items():
        print(f"{k}: {v}")