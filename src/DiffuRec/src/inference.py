import os
import glob
import torch
import logging
from argparse import Namespace
import pandas as pd
import random   
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import checkpoint_path
from main import args as cli_args, load_and_split_gts, item_num_create, fix_random_seed_as
from model import create_model_diffu, Att_Diffuse_model
from utils import (Data_Test, build_candidate_mask, filter_history_to_candidates,
                   mask_ranking_scores, prepare_model_history)
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
    candidate_mask = build_candidate_mask(
        args.coverage_candidate_items, args.item_num + 1, device
    )
    full_history = torch.LongTensor(history_seq).unsqueeze(0).to(device)
    full_history = filter_history_to_candidates(full_history, candidate_mask)
    seq_tensor = prepare_model_history(full_history, candidate_mask, max_len)
    tag_tensor = torch.LongTensor([[0]]).to(device)   # фиктивный tag
    with torch.no_grad():
        _, rep_diffu, _, _, _, _ = model(seq_tensor, tag_tensor, train_flag=False)
        scores = model.diffu_rep_pre(rep_diffu)
        mask_ranking_scores(scores, full_history, candidate_mask)
        _, topk_indices = torch.topk(scores, k=topk, dim=-1)
    return topk_indices[0].cpu().tolist()


latest_model_path = Path(os.environ.get(
    "MODEL_CHECKPOINT",
    checkpoint_path("DiffuRec", cli_args.dataset, cli_args.max_len, cli_args.random_seed),
))
if not latest_model_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {latest_model_path}")
print(f"Loading model from {latest_model_path}")

checkpoint = torch.load(
    latest_model_path,
    map_location='cuda' if torch.cuda.is_available() else 'cpu',
    weights_only=False,
)
args_dict = checkpoint['args']
if args_dict.get('item_id_offset') != 1:
    raise ValueError(
        "Legacy DiffuRec checkpoint uses item 0 as both data and padding. "
        "Retrain it with warm_start_known_catalog_v2 before inference."
    )
args = Namespace(**args_dict)
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

fix_random_seed_as(args.random_seed)
# data_raw = load_and_split_gts(quantiles=(0.7, 0.8, 0.9))
data_raw = load_and_split_gts(quantiles=(0.7, 0.8))
args = item_num_create(args, len(data_raw['smap']))

# После загрузки чекпоинта и args
diffu_rec = create_model_diffu(args)
model = Att_Diffuse_model(diffu_rec, args)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(args.device)
model.eval()

# Загрузка метаданных и словаря
movie_dict = load_movies_metadata()
smap = data_raw['smap']

# Один тестовый лоадер
test_data = Data_Test(data_raw['test_seq'], {uid: [] for uid in data_raw['test_seq']}, data_raw['test'], args)
test_loader = test_data.get_pytorch_dataloaders()

print("\nInference using saved model")
evaluate_and_print(model, test_loader, args, logger, description="test")

# Демонстрация
test_users = list(data_raw['test_seq'].keys())
if test_users:
    rand_uid = random.choice(test_users)
    print(f"\n--- Пример для пользователя {rand_uid} ---")
    full_seq = data_raw['test_seq'][rand_uid]
    target = data_raw['test'][rand_uid][0]
    recs = recommend_for_user(model, full_seq, target, args, topk=10)
    print("\nИстория (последние 10):")
    for idx in full_seq[-10:]:
        print(f"  {decode_item(idx, smap, movie_dict)}")
    print(f"\nПравильный ответ: {decode_item(target, smap, movie_dict)}")
    print("\nРекомендации (top-10):")
    for rank, rec in enumerate(recs, 1):
        print(f"  {rank}. {decode_item(rec, smap, movie_dict)}")
else:
    print("Нет пользователей для демонстрации")

# diffu_rec = create_model_diffu(args)
# model = Att_Diffuse_model(diffu_rec, args)
# model.load_state_dict(checkpoint['model_state_dict'])
# model = model.to(args.device)
# model.eval()

# print("\n Inference using saved model")
# evaluate_and_print(model, baseline_loader, args, logger, description="baseline")
# evaluate_and_print(model, test_loader, args, logger, description="adaptation")

# #  Демонстрация для случайного пользователя
# movie_dict = load_movies_metadata()
# smap = data_raw['smap']

# test_users = list(data_raw['test_seq'].keys())
# if test_users:
#     rand_uid = random.choice(test_users)
#     print(f"\n--- Пример для пользователя {rand_uid} ---")
    
#     full_seq = data_raw['test_seq'][rand_uid]
#     adapt_items = set(data_raw.get('adapt_seq', {}).get(rand_uid, []))
#     baseline_history = [item for item in full_seq if item not in adapt_items]
#     target = data_raw['test'][rand_uid][0]
    
#     recs_baseline = recommend_for_user(model, baseline_history, target, args, topk=10)
#     recs_adapt = recommend_for_user(model, full_seq, target, args, topk=10)
    
#     print("\nИстория (baseline, последние 10):")
#     for idx in baseline_history[-10:]:
#         print(f"  {decode_item(idx, smap, movie_dict)}")
    
#     print(f"\nПравильный ответ: {decode_item(target, smap, movie_dict)}")
    
#     print("\nРекомендации baseline (top-10):")
#     for rank, rec in enumerate(recs_baseline, 1):
#         print(f"  {rank}. {decode_item(rec, smap, movie_dict)}")
    
#     print("\nРекомендации c адаптацией (top-10):")
#     for rank, rec in enumerate(recs_adapt, 1):
#         print(f"  {rank}. {decode_item(rec, smap, movie_dict)}")
# else:
#     print("Нет пользователей для демонстрации")

params_json_path = latest_model_path.with_suffix('.json')
if os.path.exists(params_json_path):
    import json
    with open(params_json_path, 'r') as f:
        best_params = json.load(f)
    print("\nHyperparameters used in this model:")
    for k, v in best_params.items():
        print(f"{k}: {v}")
