import pandas as pd
import argparse
import sys
from pathlib import Path
from ast import parse
import os
import time
import numpy as np
import copy
import multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import models.gaussian_diffusion as gd
from models.DNN import DNN
# import evaluate_utils
import evaluate_topk_dp as eval_metrics
import data_utils
from copy import deepcopy

from plotting import TrainingPlotter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import (ExperimentTracker, checkpoint_due, checkpoint_path,
                                                  recommendation_popularity, save_dataset_popularity,
                                                  save_torch_checkpoint)

import random
# random_seed = 1
random_seed = 42
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) # gpu
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms
torch.backends.cudnn.deterministic=True # cudnn

def worker_init_fn(worker_id):
        np.random.seed(random_seed + worker_id)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

if __name__ == '__main__':
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m_noisy', help='choose the dataset')
    parser.add_argument('--final_train', action='store_true', help='Train on train+val without validation')
    parser.add_argument('--data_path', type=str, default='../../data/ml-1m/', help='load data path')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
    parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
    parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
    parser.add_argument('--cuda', action='store_true', help='use CUDA')
    parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
    parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
    parser.add_argument('--log_name', type=str, default='log', help='the log name')
    parser.add_argument('--round', type=int, default=1, help='record the experiment')

    parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
    parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

    # params for the model
    parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
    parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
    parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
    parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

    # params for diffusion
    parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
    parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
    parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
    parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
    parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
    parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
    parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
    parser.add_argument('--sampling_steps', type=int, default=100, help='steps of the forward process during inference')
    parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

    args = parser.parse_args()

    # Автоматическая подстановка пути для Amazon (должно быть ДО загрузки данных)
    if args.dataset != 'ml-1m' and args.dataset != 'ml-1m_noisy':
        args.data_path = f'../../data/{args.dataset}/'

    # Настройка гиперпараметров для разных датасетов
    if args.dataset == 'amazon-book_clean':
        args.steps = 10
        args.sampling_steps = 10   
        args.noise_scale = 0.0005
        args.noise_min = 0.001
        args.noise_max = 0.005
        args.w_min = 0.1
        args.w_max = 1.0
    elif args.dataset == 'yelp_clean':
        args.steps = 5
        args.sampling_steps = 5    
        args.noise_scale = 0.005
        args.noise_min = 0.001
        args.noise_max = 0.01
        args.w_min = 0.5
        args.w_max = 1.0
    elif args.dataset == 'ml-1m' or args.dataset == 'ml-1m_noisy':
        args.steps = 100
        args.noise_scale = 0.1
        args.noise_min = 0.0001
        args.noise_max = 0.02
        args.w_min = 0.1
        args.w_max = 1.0
        args.sampling_steps = args.steps
        print("args:", args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if args.cuda else "cpu")

    print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

    ### DATA LOAD ###
    train_path = args.data_path + 'train_list.npy'
    valid_path = args.data_path + 'valid_list.npy'
    test_path = args.data_path + 'test_list.npy'

    train_data, train_data_ori, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path, args.w_min, args.w_max)
    train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    if args.tst_w_val:
        tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()) + torch.FloatTensor(valid_y_data.toarray()))
        test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
    mask_tv = train_data_ori + valid_y_data

    # Режим финального обучения на train+val
    if args.final_train:
        print("FINAL TRAINING MODE: train+val (no validation)")
        train_val_weighted = train_data + valid_y_data
        train_val_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_val_weighted.toarray()))
        train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2)
        # Маска для теста остаётся mask_tv (train+val)

    train_set = set(zip(*train_data_ori.nonzero()))
    test_set = set(zip(*test_y_data.nonzero()))
    print(f"Пересечение train и test: {len(train_set & test_set)}")
    print('data ready.')

    ### Build Gaussian Diffusion ###
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)

    diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

    ### Build MLP ###
    out_dims = eval(args.dims) + [n_item]
    in_dims = out_dims[::-1]
    model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)
    checkpoint_payload = lambda: {
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "in_dims": in_dims,
            "out_dims": out_dims,
            "emb_size": args.emb_size,
            "time_type": "cat",
            "norm": args.norm,
        },
        "args": vars(args),
    }

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print("models ready.")

    param_num = 0
    mlp_num = sum([param.nelement() for param in model.parameters()])
    diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
    param_num = mlp_num + diff_num
    print("Number of all parameters:", param_num)
    
    def print_results(loss, valid_result, test_result):
        if loss is not None:
            print("[Train]: loss: {:.4f}".format(loss))
        if valid_result is not None:
            print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {} Coverage: {}".format(
                '-'.join([f"{x:.4f}" for x in valid_result[0]]),
                '-'.join([f"{x:.4f}" for x in valid_result[1]]),
                '-'.join([f"{x:.4f}" for x in valid_result[2]]),
                '-'.join([f"{x:.4f}" for x in valid_result[3]]),
                '-'.join([f"{x:.4f}" for x in valid_result[4]])
            ))
        if test_result is not None:
            print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {} Coverage: {}".format(
                '-'.join([f"{x:.4f}" for x in test_result[0]]),
                '-'.join([f"{x:.4f}" for x in test_result[1]]),
                '-'.join([f"{x:.4f}" for x in test_result[2]]),
                '-'.join([f"{x:.4f}" for x in test_result[3]]),
                '-'.join([f"{x:.4f}" for x in test_result[4]])
            ))

    def print_table_metrics(metrics, topN_list):
        print("\n[Test]")
        print(f"{'k':<5} {'recall':<12} {'ndcg':<12} {'mrr':<12} {'coverage':<12}")
        print("-" * 55)
        for i, k in enumerate(topN_list):
            print(f"{k:<5} {metrics[1][i]:<12.6f} {metrics[2][i]:<12.6f} {metrics[3][i]:<12.6f} {metrics[4][i]:<12.6f}")

    def load_movie_titles(movies_path='../../data/info/movies.dat'):
        id2title = {}
        with open(movies_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                if len(parts) >= 2:
                    movie_id = int(parts[0])
                    title = parts[1]
                    id2title[movie_id] = title
        return id2title

    id2title = load_movie_titles()
    
    def evaluate(data_loader, data_te, mask_his, topN):
        model.eval()
        e_idxlist = list(range(mask_his.shape[0]))
        e_N = mask_his.shape[0]

        predict_items = []
        target_items = []
        for i in range(e_N):
            target_items.append(data_te[i, :].nonzero()[1].tolist())
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
                batch = batch.to(device)
                prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
                prediction[his_data.nonzero()] = -np.inf

                _, indices = torch.topk(prediction, topN[-1])
                indices = indices.cpu().numpy().tolist()
                predict_items.extend(indices)
                
                # Выводим пользователей из разных батчей: первый, середина, последний
                debug_batches = [0, len(data_loader)//2, len(data_loader)-1]
                if batch_idx in debug_batches:
                    u = 0  # берём первого пользователя в батче
                    if u < len(indices):
                        true_items = target_items[batch_idx*args.batch_size + u][:5]
                        if true_items:
                            true_names = [id2title.get(i, f"ID_{i}") for i in true_items]
                            rec_items = indices[u][:10]
                            rec_names = [id2title.get(i, f"ID_{i}") for i in rec_items]
                            print(f"\n[DEBUG] Batch {batch_idx}, User {batch_idx*args.batch_size + u}:")
                            print(f"  True: {true_names}")
                            print(f"  Rec : {rec_names}")
                
                
        candidate_items = set(
            np.asarray(mask_his.sum(axis=0)).ravel().nonzero()[0].tolist()
        )
        precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(
            target_items,
            predict_items,
            topN,
            len(candidate_items),
            candidate_items=candidate_items,
        )
        return (precisions, recalls, ndcgs, mrrs, covs), predict_items

    best_recall, best_epoch = -100, 0
    best_results = None
    best_test_results = None
    
    plotter = TrainingPlotter(
            save_dir='./log/' + args.dataset,
            model_name=f"T-DiffRec_{time.strftime('%Y%m%d_%H%M%S')}",
            metrics=['loss', 'recall@10']
        )
    tracker = ExperimentTracker(args.dataset, "T-DiffRec")
    train_item_counts = np.asarray(train_data_ori.sum(axis=0)).ravel()
    train_item_popularity = {i: int(v) for i, v in enumerate(train_item_counts) if v > 0}
    save_dataset_popularity(args.dataset, train_item_popularity)
    
    print("Start training...")
    for epoch in range(1, args.epochs + 1):
        # Ранняя остановка только если не финальное обучение
        if not args.final_train and epoch - best_epoch >= 25:
            print('-'*18)
            print('Exiting from training early')
            break

        model.train()
        start_time = time.time()

        batch_count = 0
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", unit="batch")):
            batch = batch.to(device)
            batch_count += 1
            optimizer.zero_grad()
            losses = diffusion.training_losses(model, batch, args.reweight)
            loss = losses["loss"].mean()
            total_loss += loss
            loss.backward()
            optimizer.step()

        # avg_loss = total_loss / len(train_loader)
        avg_loss = (total_loss / len(train_loader)).item()
        plotter.update(epoch=epoch, loss=avg_loss)
        tracker.log_epoch(epoch, train_loss=avg_loss)
        if checkpoint_due(epoch - 1, args.epochs):
            periodic_path = checkpoint_path("T-DiffRec", args.dataset, seed=random_seed, extension=".pth")
            save_torch_checkpoint(checkpoint_payload(), periodic_path)
        
        # Валидация и сохранение модели только при обычном обучении (не final_train)
        if not args.final_train and epoch % 5 == 0:
            valid_results, _ = evaluate(test_loader, valid_y_data, train_data_ori, eval(args.topN))
            recall10 = valid_results[1][0]  # recall@10
            plotter.update(epoch=epoch, val_recall=recall10)
            idx10 = eval(args.topN).index(10) if 10 in eval(args.topN) else 0
            tracker.log_epoch(epoch, **{
                "val_recall@10": valid_results[1][idx10],
                "val_ndcg@10": valid_results[2][idx10],
                "val_mrr@10": valid_results[3][idx10],
            })
            plotter.plot(save=True, show=False, suffix=f'_epoch{epoch}')
            if args.tst_w_val:
                test_results, _ = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
            else:
                test_results, _ = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
            print_results(None, valid_results, test_results)

            if valid_results[1][1] > best_recall: # recall@20 as selection
                best_recall, best_epoch = valid_results[1][1], epoch
                best_results = valid_results
                best_test_results = test_results

                if not os.path.exists(args.save_path):
                    os.makedirs(args.save_path)
                torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}.pth' \
                    .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
                    args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.w_min, args.w_max, args.log_name))
        
        print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        print('---'*18)

    if args.final_train:
        # print("\n" + "="*50)
        print("Evaluating final model on test set...")
        start_time_inf = time.perf_counter()
        test_loader_final = DataLoader(train_val_dataset, batch_size=args.batch_size, shuffle=False)
        test_results, test_preds = evaluate(test_loader_final, test_y_data, mask_tv, eval(args.topN))
        inf_time = time.perf_counter() - start_time_inf  # <-- добавить
        print(f"Inference time: {inf_time:.4f} seconds")
        
        # Сохраняем рекомендации 
        recs_df = pd.DataFrame({
            'user_id': list(range(len(test_preds))),
            'recommendations': [list(map(int, rec)) for rec in test_preds]
        })
        recs_df.to_csv('recommendations.csv', index=False)
        print("Recommendations saved to recommendations.csv")
        
        # Выводим таблицу метрик с точностью .6f
        print_table_metrics(test_results, eval(args.topN))
        tracker.log_final_metrics(
            {k: {"recall": test_results[1][i], "ndcg": test_results[2][i],
                 "mrr": test_results[3][i], "coverage": test_results[4][i]}
             for i, k in enumerate(eval(args.topN))},
            split="global_temporal_70_10_20", mask_seen=True,
            seed=getattr(args, "random_seed", 42), inference_total_sec=inf_time,
            n_users=len(test_preds), maxlen=None,
            popularity_bias=recommendation_popularity(test_preds, train_item_popularity, eval(args.topN)),
        )
        
        # Сохраняем финальную модель
        final_model_path = checkpoint_path("T-DiffRec", args.dataset, seed=random_seed, extension=".pth")
        save_torch_checkpoint(checkpoint_payload(), final_model_path)
        print(f"Final model saved to {final_model_path}")
        plotter.plot(save=True, show=False, suffix='_final')
        tracker.close()
        print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
        exit(0)  #

    print("End. Best Epoch {:03d} ".format(best_epoch))
    plotter.plot(save=True, show=False, suffix='_final')
    if best_test_results is not None:
        tracker.log_final_metrics(
            {k: {"recall": best_test_results[1][i], "ndcg": best_test_results[2][i],
                 "mrr": best_test_results[3][i], "coverage": best_test_results[4][i]}
             for i, k in enumerate(eval(args.topN))},
            split="global_temporal_70_10_20", mask_seen=True,
            seed=getattr(args, "random_seed", 42), maxlen=None,
        )
    tracker.close()
    print_results(None, best_results, best_test_results)   
    print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
# import argparse
# from ast import parse
# import os
# import time
# import numpy as np
# import copy
# import multiprocessing
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data
# from torch.utils.data import DataLoader
# import torch.backends.cudnn as cudnn
# import torch.nn.functional as F

# import models.gaussian_diffusion as gd
# from models.DNN import DNN
# # import evaluate_utils
# import evaluate_topk_dp as eval_metrics
# import data_utils
# from copy import deepcopy

# from plotting import TrainingPlotter

# import random
# # random_seed = 1
# random_seed = 42
# torch.manual_seed(random_seed) # cpu
# torch.cuda.manual_seed(random_seed) # gpu
# np.random.seed(random_seed) # numpy
# random.seed(random_seed) # random and transforms
# torch.backends.cudnn.deterministic=True # cudnn

# def worker_init_fn(worker_id):
#         np.random.seed(random_seed + worker_id)
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)

# if __name__ == '__main__':
#     multiprocessing.freeze_support()

#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='ml-1m_noisy', help='choose the dataset')
#     parser.add_argument('--final_train', action='store_true', help='Train on train+val without validation')
#     parser.add_argument('--data_path', type=str, default='../../data/ml-1m/', help='load data path')
#     parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
#     parser.add_argument('--weight_decay', type=float, default=0.0)
#     parser.add_argument('--batch_size', type=int, default=400)
#     parser.add_argument('--epochs', type=int, default=1000, help='upper epoch limit')
#     parser.add_argument('--topN', type=str, default='[10, 20, 50, 100]')
#     parser.add_argument('--tst_w_val', action='store_true', help='test with validation')
#     parser.add_argument('--cuda', action='store_true', help='use CUDA')
#     parser.add_argument('--gpu', type=str, default='0', help='gpu card ID')
#     parser.add_argument('--save_path', type=str, default='./saved_models/', help='save model path')
#     parser.add_argument('--log_name', type=str, default='log', help='the log name')
#     parser.add_argument('--round', type=int, default=1, help='record the experiment')

#     parser.add_argument('--w_min', type=float, default=0.1, help='the minimum weight for interactions')
#     parser.add_argument('--w_max', type=float, default=1., help='the maximum weight for interactions')

#     # params for the model
#     parser.add_argument('--time_type', type=str, default='cat', help='cat or add')
#     parser.add_argument('--dims', type=str, default='[1000]', help='the dims for the DNN')
#     parser.add_argument('--norm', type=bool, default=False, help='Normalize the input or not')
#     parser.add_argument('--emb_size', type=int, default=10, help='timestep embedding size')

#     # params for diffusion
#     parser.add_argument('--mean_type', type=str, default='x0', help='MeanType for diffusion: x0, eps')
#     parser.add_argument('--steps', type=int, default=100, help='diffusion steps')
#     parser.add_argument('--noise_schedule', type=str, default='linear-var', help='the schedule for noise generating')
#     parser.add_argument('--noise_scale', type=float, default=0.1, help='noise scale for noise generating')
#     parser.add_argument('--noise_min', type=float, default=0.0001, help='noise lower bound for noise generating')
#     parser.add_argument('--noise_max', type=float, default=0.02, help='noise upper bound for noise generating')
#     parser.add_argument('--sampling_noise', type=bool, default=False, help='sampling with noise or not')
#     parser.add_argument('--sampling_steps', type=int, default=100, help='steps of the forward process during inference')
#     parser.add_argument('--reweight', type=bool, default=True, help='assign different weight to different timestep or not')

#     args = parser.parse_args()
#     # После строки args = parser.parse_args()
#     if args.dataset == 'amazon-book_clean':
#         args.steps = 10
#         args.sampling_steps = 10   
#         args.noise_scale = 0.0005
#         args.noise_min = 0.001
#         args.noise_max = 0.005
#         args.w_min = 0.1
#         args.w_max = 1.0
#     elif args.dataset == 'yelp_clean':
#         args.steps = 5
#         args.sampling_steps = 5    
#         args.noise_scale = 0.005
#         args.noise_min = 0.001
#         args.noise_max = 0.01
#         args.w_min = 0.5
#         args.w_max = 1.0
#     elif args.dataset == 'ml-1m':
#         args.steps = 100
#         # args.sampling_steps = 100
#         args.noise_scale = 0.1
#         args.noise_min = 0.0001
#         args.noise_max = 0.02
#         args.w_min = 0.1
#         args.w_max = 1.0
#         # Для ml-1m steps=100 по умолчанию, sampling_steps тоже должен быть 100
#         args.sampling_steps = args.steps   # <-- добавить
#         print("args:", args)

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
#     device = torch.device("cuda:0" if args.cuda else "cpu")

#     print("Starting time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

#     ### DATA LOAD ###
#     train_path = args.data_path + 'train_list.npy'
#     valid_path = args.data_path + 'valid_list.npy'
#     test_path = args.data_path + 'test_list.npy'

#     train_data, train_data_ori, valid_y_data, test_y_data, n_user, n_item = data_utils.data_load(train_path, valid_path, test_path, args.w_min, args.w_max)
#     # train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.A)) #old method
#     train_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()))
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)
#     test_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

#     if args.tst_w_val:
#         tv_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_data.toarray()) + torch.FloatTensor(valid_y_data.toarray()))
#         test_twv_loader = DataLoader(tv_dataset, batch_size=args.batch_size, shuffle=False)
#     mask_tv = train_data_ori + valid_y_data
#     if args.dataset != 'ml-1m':
#         args.data_path = f'../../data/{args.dataset}/'
        
#     if args.final_train:
#          print("FINAL TRAINING MODE: train+val (no validation)")
#         # Объединяем train и val с весами
#         # train_data уже содержит веса (linspace), а valid_y_data – бинарные 1
#         train_val_weighted = train_data + valid_y_data  # сохраняет веса train и добавляет валидацию с весом 1
#         train_val_dataset = data_utils.DataDiffusion(torch.FloatTensor(train_val_weighted.toarray()))
#         train_loader = DataLoader(train_val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=2)
#         # Маска для инференса (train+val+test – чтобы не рекомендовать уже виденное)
#         mask_final = train_data_ori + valid_y_data + test_y_data  # для теста, но обычно только train+val
#         # Но в evaluate маска передаётся отдельно – позже используем mask_tv (train+val) или расширенную
        
#         # Отключаем раннюю остановку и валидацию
#         best_epoch = 0
#         best_recall = -100
#         best_model_state = None
    
#     train_set = set(zip(*train_data_ori.nonzero()))
#     test_set = set(zip(*test_y_data.nonzero()))
#     print(f"Пересечение train и test: {len(train_set & test_set)}")
    
#     print('data ready.')


#     ### Build Gaussian Diffusion ###
#     if args.mean_type == 'x0':
#         mean_type = gd.ModelMeanType.START_X
#     elif args.mean_type == 'eps':
#         mean_type = gd.ModelMeanType.EPSILON
#     else:
#         raise ValueError("Unimplemented mean type %s" % args.mean_type)

#     diffusion = gd.GaussianDiffusion(mean_type, args.noise_schedule, \
#             args.noise_scale, args.noise_min, args.noise_max, args.steps, device).to(device)

#     ### Build MLP ###
#     out_dims = eval(args.dims) + [n_item]
#     in_dims = out_dims[::-1]
#     model = DNN(in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm).to(device)

#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     print("models ready.")

#     param_num = 0
#     mlp_num = sum([param.nelement() for param in model.parameters()])
#     diff_num = sum([param.nelement() for param in diffusion.parameters()])  # 0
#     param_num = mlp_num + diff_num
#     print("Number of all parameters:", param_num)
    
#     def print_results(loss, valid_result, test_result):
#         if loss is not None:
#             print("[Train]: loss: {:.4f}".format(loss))
#         if valid_result is not None:
#             print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {} Coverage: {}".format(
#                 '-'.join([f"{x:.4f}" for x in valid_result[0]]),
#                 '-'.join([f"{x:.4f}" for x in valid_result[1]]),
#                 '-'.join([f"{x:.4f}" for x in valid_result[2]]),
#                 '-'.join([f"{x:.4f}" for x in valid_result[3]]),
#                 '-'.join([f"{x:.4f}" for x in valid_result[4]])
#             ))
#         if test_result is not None:
#             print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {} Coverage: {}".format(
#                 '-'.join([f"{x:.4f}" for x in test_result[0]]),
#                 '-'.join([f"{x:.4f}" for x in test_result[1]]),
#                 '-'.join([f"{x:.4f}" for x in test_result[2]]),
#                 '-'.join([f"{x:.4f}" for x in test_result[3]]),
#                 '-'.join([f"{x:.4f}" for x in test_result[4]])
#             ))
#     def print_table_metrics(metrics, topN_list):
#         print("\n[Test]")
#         print(f"{'k':<5} {'recall':<12} {'ndcg':<12} {'mrr':<12} {'coverage':<12}")
#         print("-" * 55)
#         for i, k in enumerate(topN_list):
#             print(f"{k:<5} {metrics[1][i]:<12.6f} {metrics[2][i]:<12.6f} {metrics[3][i]:<12.6f} {metrics[4][i]:<12.6f}")

#     def load_movie_titles(movies_path='../../data/info/movies.dat'):
#         id2title = {}
#         with open(movies_path, 'r', encoding='latin-1') as f:
#             for line in f:
#                 parts = line.strip().split('::')
#                 if len(parts) >= 2:
#                     movie_id = int(parts[0])
#                     title = parts[1]
#                     id2title[movie_id] = title
#         return id2title

#     id2title = load_movie_titles()
    
#     def evaluate(data_loader, data_te, mask_his, topN):
#         model.eval()
#         e_idxlist = list(range(mask_his.shape[0]))
#         e_N = mask_his.shape[0]

#         predict_items = []
#         target_items = []
#         for i in range(e_N):
#             target_items.append(data_te[i, :].nonzero()[1].tolist())
        
#         with torch.no_grad():
#             for batch_idx, batch in enumerate(data_loader):
#                 his_data = mask_his[e_idxlist[batch_idx*args.batch_size:batch_idx*args.batch_size+len(batch)]]
#                 batch = batch.to(device)
#                 prediction = diffusion.p_sample(model, batch, args.sampling_steps, args.sampling_noise)
#                 prediction[his_data.nonzero()] = -np.inf

#                 _, indices = torch.topk(prediction, topN[-1])
#                 indices = indices.cpu().numpy().tolist()
#                 predict_items.extend(indices)
                
#                 # Выводим пользователей из разных батчей: первый, середина, последний
#                 debug_batches = [0, len(data_loader)//2, len(data_loader)-1]
#                 if batch_idx in debug_batches:
#                     u = 0  # берём первого пользователя в батче
#                     if u < len(indices):
#                         true_items = target_items[batch_idx*args.batch_size + u][:5]
#                         if true_items:
#                             true_names = [id2title.get(i, f"ID_{i}") for i in true_items]
#                             rec_items = indices[u][:10]
#                             rec_names = [id2title.get(i, f"ID_{i}") for i in rec_items]
#                             print(f"\n[DEBUG] Batch {batch_idx}, User {batch_idx*args.batch_size + u}:")
#                             print(f"  True: {true_names}")
#                             print(f"  Rec : {rec_names}")
                
                
#         precisions, recalls, ndcgs, mrrs, covs = eval_metrics.compute_all_metrics(target_items, predict_items, topN, n_item)
#         # return (precisions, recalls, ndcgs, mrrs, covs)
#         return (precisions, recalls, ndcgs, mrrs, covs), predict_items

#     best_recall, best_epoch = -100, 0
#     best_test_result = None
    
#     plotter = TrainingPlotter(
#             save_dir='./log/' + args.dataset,
#             model_name=f"T-DiffRec_{time.strftime('%Y%m%d_%H%M%S')}",
#             metrics=['loss', 'recall@10']
#         )
    
    
#     print("Start training...")
#     for epoch in range(1, args.epochs + 1):
#         if epoch - best_epoch >= 25:
#             print('-'*18)
#             print('Exiting from training early')
#             break

#         model.train()
#         start_time = time.time()

#         batch_count = 0
#         total_loss = 0.0
        
#         # for batch_idx, batch in enumerate(train_loader):
#         for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch:03d}/{args.epochs}", unit="batch")):
#             batch = batch.to(device)
#             batch_count += 1
#             optimizer.zero_grad()
#             losses = diffusion.training_losses(model, batch, args.reweight)
#             loss = losses["loss"].mean()
#             total_loss += loss
#             loss.backward()
#             optimizer.step()

#         avg_loss = total_loss / len(train_loader)
#         plotter.update(epoch=epoch, loss=avg_loss)
        
#         if epoch % 5 == 0:
#             # valid_results = evaluate(test_loader, valid_y_data, train_data, eval(args.topN))
#             valid_results = evaluate(test_loader, valid_y_data, train_data_ori, eval(args.topN))
#             recall10 = valid_results[1][0]  # recall@10
#             plotter.update(epoch=epoch, val_recall=recall10)
#             plotter.plot(save=True, show=False, suffix=f'_epoch{epoch}')
#             if args.tst_w_val:
#                 test_results = evaluate(test_twv_loader, test_y_data, mask_tv, eval(args.topN))
#             else:
#                 test_results = evaluate(test_loader, test_y_data, mask_tv, eval(args.topN))
#             print_results(None, valid_results, test_results)

#             if valid_results[1][1] > best_recall: # recall@20 as selection
#                 best_recall, best_epoch = valid_results[1][1], epoch
#                 best_results = valid_results
#                 best_test_results = test_results

#                 if not os.path.exists(args.save_path):
#                     os.makedirs(args.save_path)
#                 torch.save(model, '{}{}_lr{}_wd{}_bs{}_dims{}_emb{}_{}_steps{}_scale{}_min{}_max{}_sample{}_reweight{}_wmin{}_wmax{}_{}.pth' \
#                     .format(args.save_path, args.dataset, args.lr, args.weight_decay, args.batch_size, args.dims, args.emb_size, args.mean_type, \
#                     args.steps, args.noise_scale, args.noise_min, args.noise_max, args.sampling_steps, args.reweight, args.w_min, args.w_max, args.log_name))
        
#         print("Runing Epoch {:03d} ".format(epoch) + 'train loss {:.4f}'.format(total_loss) + " costs " + time.strftime(
#                             "%H: %M: %S", time.gmtime(time.time()-start_time)))
#         print('---'*18)

#     print('==='*18)
#     print("End. Best Epoch {:03d} ".format(best_epoch))
#     plotter.plot(save=True, show=False, suffix='_final')
#     print_results(None, best_results, best_test_results)   
#     print("End time: ", time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))





