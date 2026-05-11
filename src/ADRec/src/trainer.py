import time
import datetime
import os
import torch
from tqdm import tqdm
import copy
from metrics import *
from utils import *
from model import Att_Diffuse_model
from pcgrad import PCGrad
from torch import optim
from sasrec import SASRec
from evaluate_topk_dp import compute_all_metrics
from plotting import TrainingPlotter

def downvote_seen_items(scores, hist_pad):
    for i in range(scores.shape[0]):
        seen = torch.unique(hist_pad[i][hist_pad[i] > 0])
        scores[i, seen] = -float('inf')
    return scores

def evaluate_and_print(model, data_loader, args, logger, description="Validation", mask_seen=False):
    device = args.device
    metric_ks = args.metric_ks
    model.eval()
    all_actual = []
    all_predicted = []
    start_time = time.time()
    with torch.no_grad():
        # for batch in tqdm.tqdm(data_loader, leave=False, desc=f'{description}'):
        for batch in tqdm(data_loader, leave=False, desc=f'{description}'):
            batch = [x.to(device) for x in batch]
            out_seq, last_item, *_ = model(batch[0], batch[1], train_flag=False)
            scores = model.calculate_score(last_item)
            if mask_seen:
                scores = downvote_seen_items(scores, batch[0])
            _, topk_idx = torch.topk(scores, k=max(metric_ks), dim=-1)
            for i in range(len(batch[1])):
                all_actual.append([batch[1][i, -1].item()])
                all_predicted.append(topk_idx[i].cpu().tolist())
    elapsed = time.time() - start_time
    precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
        all_actual, all_predicted, metric_ks, args.item_num
    )
    print(f'{description} results ({elapsed:.2f}s):')
    logger.info(f'{description} results ({elapsed:.2f}s):')
    for k, r, n, m, c in zip(metric_ks, recalls, ndcgs, mrrs, covs):
        print(f"k={k}: Recall={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}, Cov={c:.4f}")
        logger.info(f"k={k}: Recall={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}, Cov={c:.4f}")
    return recalls[metric_ks.index(10)] if 10 in metric_ks else 0.0


# from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup
def extract(data):
    seq= data[0]
    diff_loss = data[1] if len(data) == 2 else torch.zeros(1,device=seq.device)
    return seq, seq[:,-1], diff_loss

def item_num_create(args):
    length = {"ml-1m": 3706,
              "ml-100k":1682,#1008
              'yelp': 64669,
              'sports':12301,
              'baby':4731,
              'toys':7309,
              'beauty':6086
              }
    args.item_num = length[args.dataset]
    return args
def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        if args.model == 'adrec':
            opt= optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        else:
            opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        opt= optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError
    return opt

def choose_model(args):
    device = args.device
    if args.model in ['diffurec','adrec','dreamrec']:
        if args.model == 'adrec':
            
            pretrain_path = os.path.join('saved', 'pretrain', args.dataset, 'pretrain.pth')
            if os.path.exists(pretrain_path):
                args.pretrained = True
                args.freeze_emb = True
            # если файла нет – оставляем pretrained=False 
            # args.pcgrad=True
            # args.pretrained=True
            # args.freeze_emb=True
            pass
        if args.model == 'diffurec':
            args.split_onebyone=True
            args.parallel_ag = False
            args.is_causal = False
        model = Att_Diffuse_model(args)
    elif args.model == 'sasrec' or args.model == 'pretrain':
        args.parallel_ag = False
        model = SASRec(args)
    else:
        model=None
    return model.to(device)
# ("bert4rec" "core" "eulerformer" "fearec" "gru4rec" "trimlp")
def load_data(args):

    path_data = '../datasets/data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = data_raw['item_count']
    tra_data = Data_Train(data_raw['train'], args)
    # val_data = Data_Val(data_raw['train'], data_raw['val'], args)
    val_data = Data_Val(data_raw['val_seq'], data_raw['val_tgt'], args)
    # test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args)
    test_data = Data_Test(data_raw['test_seq'], [[] for _ in data_raw['test_tgt']], data_raw['test_tgt'], args) 
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    # args.item_num = data_raw['item_count']

    return tra_data_loader, val_data_loader, test_data_loader


def model_train(model_joint, tra_data_loader, val_data_loader, test_data_loader, args, logger, train_time):
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    torch.set_float32_matmul_precision('high')
    plotter = TrainingPlotter(
        save_dir=os.path.join(args.log_file, args.model, args.dataset),
        model_name=f"{args.model}_{args.dataset}_{train_time}",
        metrics=['loss', 'recall@10']
    )
    optimizer = PCGrad(optimizers(model_joint, args), args)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer.optim, T_max=500)
    best_metrics_dict = {'Best_Recall@5': 0, 'Best_NDCG@5': 0, 'Best_Recall@10': 0, 'Best_NDCG@10': 0, 'Best_Recall@20': 0, 'Best_NDCG@20': 0}
    best_epoch = {'Best_epoch_Recall@5': 0, 'Best_epoch_NDCG@5': 0, 'Best_epoch_Recall@10': 0, 'Best_epoch_NDCG@10': 0, 'Best_epoch_Recall@20': 0, 'Best_epoch_NDCG@20': 0}
    best_recall10 = -1.0
    best_model = None

    for epoch_temp in range(epochs):
        model_joint.train()
        if epoch_temp == 5 and args.model == 'adrec':
            print(f'warm up finished in epoch {epoch_temp}')
            logger.info(f'warm up finished in epoch {epoch_temp}')
            model_joint.item_embedding.weight.requires_grad = True
        ce_losses = []
        dif_losses = []
        pbr_train = tqdm(enumerate(tra_data_loader), desc='Epoch: {}'.format(epoch_temp), leave=False, total=len(tra_data_loader))
        for index_temp, train_batch in pbr_train:
            train_batch = [x.to(device) for x in train_batch]
            optimizer.zero_grad()
            out_seq, last_item, *dif_loss = model_joint(train_batch[0], train_batch[1], train_flag=True)
            if len(dif_loss) > 0:
                dif_loss = dif_loss[0]
            else:
                dif_loss = torch.zeros(1, device=args.device)
            ce_loss = model_joint.calculate_loss(out_seq, train_batch[1])
            if args.model == 'adrec' and args.loss == 'mse':
                losses = [ce_loss, args.loss_scale * dif_loss]
            elif args.model == 'dreamrec':
                losses = [dif_loss]
            else:
                losses = [ce_loss]
            optimizer.pc_backward(losses)
            ce_losses.append(ce_loss.item())
            dif_losses.append(dif_loss.item())
            optimizer.step()
            pbr_train.set_postfix_str(f'loss={ce_losses[-1]:.3f}')

        print(f"loss in epoch {epoch_temp}: ce_loss {sum(ce_losses)/len(ce_losses):.3f}, dif_loss {sum(dif_losses)/len(dif_losses):.3f}")
        logger.info(f"loss in epoch {epoch_temp}: ce_loss {sum(ce_losses)/len(ce_losses):.3f}, dif_loss {sum(dif_losses)/len(dif_losses):.3f}")
        avg_loss = sum(ce_losses) / len(ce_losses)
        plotter.update(epoch=epoch_temp, loss=avg_loss)
        lr_scheduler.step()
        # plotter.plot(save=True, show=False, suffix=f'_epoch{epoch_temp}')

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            if val_data_loader is not None:
                all_actual = []
                all_predicted = []
                model_joint.eval()
                with torch.no_grad():
                    for val_batch in tqdm(val_data_loader, leave=False, desc='Denoising..., Epoch: {}'.format(epoch_temp)):
                        val_batch = [x.to(device) for x in val_batch]
                        out_seq, last_item, *_ = model_joint(val_batch[0], val_batch[1], train_flag=False)
                        scores = model_joint.calculate_score(last_item)
                        if getattr(args, 'mask_seen', False):
                            scores = downvote_seen_items(scores, val_batch[0])  
                    
                        _, topk_idx = torch.topk(scores, k=max(metric_ks), dim=-1)
                        for i in range(len(val_batch[1])):
                            all_actual.append([val_batch[1][i, -1].item()])
                            all_predicted.append(topk_idx[i].cpu().tolist())

                precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
                    all_actual, all_predicted, metric_ks, args.item_num
                )
                idx10 = metric_ks.index(10) if 10 in metric_ks else 0
                recall10 = recalls[idx10]

                if recall10 > best_recall10:
                    best_recall10 = recall10
                    best_model = copy.deepcopy(model_joint)
                
                plotter.plot(save=True, show=False, suffix=f'_epoch{epoch_temp}')

                for k, r, n, m, c in zip(metric_ks, recalls, ndcgs, mrrs, covs):
                    if r > best_metrics_dict.get('Best_Recall@{}'.format(k), 0):
                        best_metrics_dict['Best_Recall@{}'.format(k)] = r
                        best_epoch['Best_epoch_Recall@{}'.format(k)] = epoch_temp
                    if n > best_metrics_dict.get('Best_NDCG@{}'.format(k), 0):
                        best_metrics_dict['Best_NDCG@{}'.format(k)] = n
                        best_epoch['Best_epoch_NDCG@{}'.format(k)] = epoch_temp

                for k, r, n, m, c in zip(metric_ks, recalls, ndcgs, mrrs, covs):
                    print(f"k={k}: Recall={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}, Cov={c:.4f}")
                    logger.info(f"k={k}: Recall={r:.4f}, NDCG={n:.4f}, MRR={m:.4f}, Cov={c:.4f}")
    plotter.plot(save=True, show=False, suffix='_final')
    # сли никакая модель не сохранилась
    if best_model is None:
        best_model = copy.deepcopy(model_joint)

    saved_dir = os.path.join('saved', args.model, args.dataset)
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)
    output_path = os.path.join(saved_dir, str(train_time) + args.description + '.pth')
    # Специальное сохранение для pretrain модели
    if args.model == 'pretrain':
        pretrain_saved_dir = os.path.join('saved', 'pretrain', args.dataset)
        os.makedirs(pretrain_saved_dir, exist_ok=True)
        pretrain_output_path = os.path.join(pretrain_saved_dir, 'pretrain.pth')
        torch.save(best_model.state_dict(), pretrain_output_path)
        print(f"Pretrained embeddings saved to {pretrain_output_path}")
    torch.save(best_model.state_dict(), str(output_path))
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    # all_actual = []
    # all_predicted = []
    test_metrics_dict_mean = {}
    if test_data_loader is not None:
        print('start testing: ', datetime.datetime.now())
        logger.info('start testing: {}'.format(datetime.datetime.now()))
        top_100_item = []
        all_actual = []
        all_predicted = []
        best_model.eval()
        start_test_time = time.time()
        with torch.no_grad():
            for test_batch in tqdm(test_data_loader, leave=False):
                test_batch = [x.to(device) for x in test_batch]
                out_seq, last_item, *_ = best_model(test_batch[0], test_batch[1], train_flag=False)
                scores = best_model.calculate_score(last_item)
                if getattr(args, 'mask_seen', False):
                    scores = downvote_seen_items(scores, test_batch[0])
                _, topk_idx = torch.topk(scores, k=max(metric_ks), dim=-1)
                for i in range(len(test_batch[1])):
                    all_actual.append([test_batch[1][i, -1].item()])
                    all_predicted.append(topk_idx[i].cpu().tolist())
                if args.diversity_measure:
                    _, top100 = torch.topk(scores, k=100, dim=-1)
                    top_100_item.append(top100.cpu())
        test_elapsed = time.time() - start_test_time
        logger.info(f"Test inference time: {test_elapsed:.2f}s")
        print(f"Test inference time: {test_elapsed:.2f}s")
        precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
            all_actual, all_predicted, metric_ks, args.item_num
        )
        test_metrics_dict_mean = {}
        for k, r, n, m, c in zip(metric_ks, recalls, ndcgs, mrrs, covs):
            test_metrics_dict_mean['Recall@{}'.format(k)] = r
            test_metrics_dict_mean['NDCG@{}'.format(k)] = n
            test_metrics_dict_mean['MRR@{}'.format(k)] = m
            test_metrics_dict_mean['Coverage@{}'.format(k)] = c

        print('Test------------------------------------------------------')
        logger.info('Test------------------------------------------------------')
        print(test_metrics_dict_mean)
        logger.info(test_metrics_dict_mean)
        if best_recall10 >= 0:
            print('Best Eval---------------------------------------------------------')
            print(f"Best Recall@10: {best_recall10:.4f}")
            logger.info('Best Eval---------------------------------------------------------')
            logger.info(f"Best Recall@10: {best_recall10:.4f}")
        # print('Best Eval---------------------------------------------------------')
        # print(f"Best Recall@10: {best_recall10:.4f}")
        # logger.info(f"Best Recall@10: {best_recall10:.4f}")
        print(args)

    if args.diversity_measure:
        pass

    return best_model, test_metrics_dict_mean
