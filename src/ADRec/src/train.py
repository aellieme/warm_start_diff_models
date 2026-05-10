import os
import sys
import time
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import yaml

from trainer import model_train
from logger import make_logger
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_and_split_gts, fix_random_seed_as, Data_Train, Data_Test
from trainer import optimizers, choose_model
from evaluate_topk_dp import compute_all_metrics
from plotting import TrainingPlotter
from pcgrad import PCGrad
from trainer import downvote_seen_items

# # Функция extend_argparse удалена, так как не используется и содержала синтаксическую ошибку

# def final_training(args):
#     # fix_random_seed_as(args.seed)
#     fix_random_seed_as(args.random_seed)

#     print("Loading data...")
#     data_raw = load_and_split_gts(quantiles=(0.7, 0.8))
#     args.item_num = data_raw['item_count']

#     train_combined = []
#     for uid in sorted(data_raw['train_dict'].keys()):
#         if uid in data_raw['val_seq_dict'] and uid in data_raw['val_tgt_dict']:
#             combined_seq = data_raw['train_dict'][uid] + data_raw['val_seq_dict'][uid] + [data_raw['val_tgt_dict'][uid]]
#             train_combined.append(combined_seq)
#     print(f"Total training sequences after merging train+val: {len(train_combined)}")

#     test_seq = data_raw['test_seq']
#     test_tgt = data_raw['test_tgt']
#     test_data = Data_Test(test_seq, [[] for _ in test_tgt], test_tgt, args)

#     tra_data = Data_Train(train_combined, args)
#     train_loader = tra_data.get_pytorch_dataloaders()
#     test_loader = test_data.get_pytorch_dataloaders()

#     print("Creating model...")
#     model = choose_model(args)
#     device = args.device
#     model = model.to(device)

#     if args.pcgrad:
#         base_optim = optimizers(model, args)
#         optimizer = PCGrad(base_optim, args)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(base_optim, T_max=args.epochs)
#     else:
#         optimizer = optimizers(model, args)
#         scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     plotter = TrainingPlotter(
#         save_dir=os.path.join(args.log_file, args.model, args.dataset),
#         model_name=f"{args.model}_{args.dataset}_final",
#         metrics=['loss']
#     )

#     print(f"Starting final training for {args.epochs} epochs...")
#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         total_loss = 0.0
#         pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
#         for batch in pbar:
#             batch = [x.to(device) for x in batch]
#             optimizer.zero_grad()
#             out_seq, last_item, *dif_loss = model(batch[0], batch[1], train_flag=True)
#             if dif_loss:
#                 dif_loss = dif_loss[0]
#             else:
#                 dif_loss = torch.zeros(1, device=device)

#             ce_loss = model.calculate_loss(out_seq, batch[1])
#             if args.model == 'adrec' and args.loss == 'mse':
#                 losses = [ce_loss, args.loss_scale * dif_loss]
#             elif args.model == 'dreamrec':
#                 losses = [dif_loss]
#             else:
#                 losses = [ce_loss]

#             if args.pcgrad:
#                 optimizer.pc_backward(losses)
#             else:
#                 loss = sum(losses)
#                 loss.backward()
#             optimizer.step()
#             total_loss += ce_loss.item()
#         scheduler.step()

#         avg_loss = total_loss / len(train_loader)
#         plotter.update(epoch=epoch, loss=avg_loss)

#         if epoch % args.eval_interval == 0 or epoch == args.epochs:
#             print(f"Epoch {epoch}: train loss = {avg_loss:.4f}")

#     saved_dir = os.path.join('saved', args.model, args.dataset)
#     os.makedirs(saved_dir, exist_ok=True)
#     model_path = os.path.join(saved_dir, f'final_{args.description}_{time.strftime("%Y%m%d_%H%M%S")}.pth')
#     torch.save(model.state_dict(), model_path)
#     print(f"Model saved to {model_path}")

#     print("Evaluating on test set...")
#     model.eval()
#     all_actual = []
#     all_predicted = []
#     start_time = time.time()
#     with torch.no_grad():
#         for batch in tqdm(test_loader, desc='Testing'):
#             batch = [x.to(device) for x in batch]
#             out_seq, last_item, *_ = model(batch[0], batch[1], train_flag=False)
#             scores = model.calculate_score(last_item)
#             if args.mask_seen:
#                 scores = downvote_seen_items(scores, batch[0])
#             _, topk_idx = torch.topk(scores, k=max(args.metric_ks), dim=-1)
#             for i in range(len(batch[1])):
#                 all_actual.append([batch[1][i, -1].item()])
#                 all_predicted.append(topk_idx[i].cpu().tolist())
#     elapsed = time.time() - start_time
#     recalls, ndcgs, mrrs, covs = None, None, None, None
#     if args.metric_ks:
#         _, recalls, ndcgs, mrrs, covs = compute_all_metrics(
#             all_actual, all_predicted, args.metric_ks, args.item_num
#         )
#     print(f"\nTest inference time: {elapsed:.2f}s")
#     print("Test metrics:")
#     for k, r, n, m, c in zip(args.metric_ks, recalls, ndcgs, mrrs, covs):
#         print(f"k={k}: Recall={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}, Coverage={c:.4f}")

#     log_dir = os.path.join(args.log_file, args.model, args.dataset)
#     os.makedirs(log_dir, exist_ok=True)
#     log_path = os.path.join(log_dir, f"final_{args.description}.log")
#     with open(log_path, 'a') as f:
#         f.write(f"Run at {time.ctime()}\n")
#         f.write(f"Arguments: {args}\n")
#         f.write(f"Test recall@10: {recalls[args.metric_ks.index(10)] if 10 in args.metric_ks else 'N/A'}\n")
#         for k, r, n, m, c in zip(args.metric_ks, recalls, ndcgs, mrrs, covs):
#             f.write(f"k={k}: Recall={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}, Cov={c:.4f}\n")
#         f.write("\n")
#     print(f"Log saved to {log_path}")

#     plotter.plot(save=True, show=False, suffix='_final')
#     return model, (recalls, ndcgs, mrrs, covs)


def main():
    def load_config(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    temp_parser = argparse.ArgumentParser(add_help=False)
    temp_parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    temp_args, _ = temp_parser.parse_known_args()

    # Загрузить конфиг, если указан
    config_dict = {}
    if temp_args.config:
        with open(temp_args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
    parser = argparse.ArgumentParser(description="Final training script for recommendation models")

    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--dataset', type=str, default=config_dict.get('dataset', 'ml-1m'))
    parser.add_argument('--metric_ks', nargs='+', type=int, default=config_dict.get('metric_ks', [5,10,20]))
    parser.add_argument('--random_seed', type=int, default=config_dict.get('random_seed', 42))
    parser.add_argument('--log_file', type=str, default=config_dict.get('log_file', 'logs/'))
    parser.add_argument('--description', type=str, default=config_dict.get('description', '_final'))
    parser.add_argument('--epochs', type=int, default=config_dict.get('epochs', 150))
    parser.add_argument('--eval_interval', type=int, default=config_dict.get('eval_interval', 10))
    parser.add_argument('--max_len', type=int, default=config_dict.get('max_len', 50))
    parser.add_argument('--device', type=str, default=config_dict.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--num_gpu', type=int, default=config_dict.get('num_gpu', 1))
    parser.add_argument('--batch_size', type=int, default=config_dict.get('batch_size', 512))
    parser.add_argument('--hidden_size', type=int, default=config_dict.get('hidden_size', 128))
    parser.add_argument('--dropout', type=float, default=config_dict.get('dropout', 0.1))
    parser.add_argument('--emb_dropout', type=float, default=config_dict.get('emb_dropout', 0.3))
    parser.add_argument('--num_blocks', type=int, default=config_dict.get('num_blocks', 4))
    parser.add_argument('--diffusion_steps', type=int, default=config_dict.get('diffusion_steps', 32))
    parser.add_argument('--lambda_uncertainty', type=float, default=config_dict.get('lambda_uncertainty', 0.001))
    parser.add_argument('--noise_schedule', type=str, default=config_dict.get('noise_schedule', 'trunc_lin'))
    parser.add_argument('--schedule_sampler_name', type=str, default=config_dict.get('schedule_sampler_name', 'uniform'))
    parser.add_argument('--optimizer', type=str, default=config_dict.get('optimizer', 'Adam'))
    parser.add_argument('--lr', type=float, default=config_dict.get('lr', 0.001))
    parser.add_argument('--weight_decay', type=float, default=config_dict.get('weight_decay', 0.0))
    parser.add_argument('--momentum', type=float, default=config_dict.get('momentum', None))
    parser.add_argument('--model', type=str, default=config_dict.get('model', 'adrec'))
    parser.add_argument('--loss', type=str, default=config_dict.get('loss', 'mse'))
    parser.add_argument('--loss_scale', type=float, default=config_dict.get('loss_scale', 1.0))
    parser.add_argument('--cfg_scale', type=float, default=config_dict.get('cfg_scale', 1.0))
    parser.add_argument('--geodesic', action='store_true', default=config_dict.get('geodesic', False))
    parser.add_argument('--independent', action='store_true', default=config_dict.get('independent', False))
    parser.add_argument('--pcgrad', action='store_true', default=config_dict.get('pcgrad', False))
    parser.add_argument('--is_causal', action='store_true', default=config_dict.get('is_causal', False))
    parser.add_argument('--parallel_ag', action='store_true', default=config_dict.get('parallel_ag', False))
    parser.add_argument('--split_onebyone', action='store_true', default=config_dict.get('split_onebyone', False))
    parser.add_argument('--dif_decoder', type=str, default=config_dict.get('dif_decoder', 'att'))
    parser.add_argument('--dif_objective', type=str, default=config_dict.get('dif_objective', 'pred_x0'))
    parser.add_argument('--beta_a', type=float, default=config_dict.get('beta_a', 0.3))
    parser.add_argument('--beta_b', type=float, default=config_dict.get('beta_b', 10.0))
    parser.add_argument('--mask_seen', action='store_true', default=config_dict.get('mask_seen', False))
    parser.add_argument('--pretrained', action='store_true', default=config_dict.get('pretrained', False))
    parser.add_argument('--freeze_emb', action='store_true', default=config_dict.get('freeze_emb', False))
    parser.add_argument('--rescale_timesteps', action='store_true', default=config_dict.get('rescale_timesteps', True))
    
    args = parser.parse_args()

    args.hidden_act = 'gelu'
    args.loss_lambda = args.lambda_uncertainty
    args.lambda_schedule = False
    args.lambda_beta_a = 0.0
    args.lambda_beta_b = 0.0
    args.diversity_measure = False
    args.epoch_time_avg = False
    args.long_head = False
    if not hasattr(args, 'description'):
        args.description = '_final'
    if not hasattr(args, 'log_file'):
        args.log_file = 'logs/'

    args.pretrained = getattr(args, 'pretrained', False)
    args.freeze_emb = getattr(args, 'freeze_emb', False)
    args.rescale_timesteps = getattr(args, 'rescale_timesteps', True)
    args.is_causal = getattr(args, 'is_causal', False)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    train_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    
    print("Loading data...")
    data_raw = load_and_split_gts(quantiles=(0.7, 0.8))
    args.item_num = data_raw['item_count']

    train_combined = []
    for uid in sorted(data_raw['train_dict'].keys()):
        if uid in data_raw['val_seq_dict'] and uid in data_raw['val_tgt_dict']:
            combined_seq = data_raw['train_dict'][uid] + data_raw['val_seq_dict'][uid] + [data_raw['val_tgt_dict'][uid]]
            train_combined.append(combined_seq)

    test_seq = data_raw['test_seq']
    test_tgt = data_raw['test_tgt']
    test_data = Data_Test(test_seq, [[] for _ in test_tgt], test_tgt, args)

    tra_data = Data_Train(train_combined, args)
    train_loader = tra_data.get_pytorch_dataloaders()
    test_loader = test_data.get_pytorch_dataloaders()

    model = choose_model(args).to(args.device)

    best_model, test_metrics = model_train(
        model,
        train_loader,
        None,          
        test_loader,
        args,
        logger,
        train_time
    )


if __name__ == '__main__':
    main()