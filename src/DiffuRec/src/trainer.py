import torch.nn as nn
import torch.optim as optim
import datetime
import torch
import numpy as np
import copy
import time
import pickle
import sys
from pathlib import Path
from tqdm import tqdm

from evaluate_topk_dp import compute_all_metrics
from plotting import TrainingPlotter
from utils import (build_candidate_mask, eligible_warm_start_rows,
                   filter_history_to_candidates, mask_ranking_scores)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiment_tools.experiment_tracking import ExperimentTracker, recommendation_popularity

def optimizers(model, args):
    if args.optimizer.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError


def cal_hr(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = label == topk_predict
    hr = [hit[:, :ks[i]].sum().item()/label.size()[0] for i in range(len(ks))]
    return hr


def cal_ndcg(label, predict, ks):
    max_ks = max(ks)
    _, topk_predict = torch.topk(predict, k=max_ks, dim=-1)
    hit = (label == topk_predict).int()
    ndcg = []
    for k in ks:
        max_dcg = dcg(torch.tensor([1] + [0] * (k-1)))
        predict_dcg = dcg(hit[:, :k])
        ndcg.append((predict_dcg/max_dcg).mean().item())
    return ndcg


def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel


def hrs_and_ndcgs_k(scores, labels, ks):
    metrics = {}
    ndcg = cal_ndcg(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    hr = cal_hr(labels.clone().detach().to('cpu'), scores.clone().detach().to('cpu'), ks)
    for k, ndcg_temp, hr_temp in zip(ks, ndcg, hr):
        metrics['HR@%d' % k] = hr_temp
        metrics['NDCG@%d' % k] = ndcg_temp
    return metrics  


def LSHT_inference(model_joint, args, data_loader):
    device = args.device
    model_joint = model_joint.to(device)
    candidate_mask = build_candidate_mask(
        args.coverage_candidate_items,
        args.item_num + 1,
        device,
    )
    with torch.no_grad():
        test_metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        test_metrics_dict_mean = {}
        for test_batch in data_loader:
            test_batch = [x.to(device) for x in test_batch]
            test_batch[0] = filter_history_to_candidates(
                test_batch[0], candidate_mask
            )
            
            scores_rec, rep_diffu, _, _, _, _ = model_joint(test_batch[0], test_batch[1], train_flag=False)
            scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)
            valid_rows = eligible_warm_start_rows(
                test_batch[0], test_batch[1], candidate_mask
            )
            if not valid_rows.any():
                continue
            mask_ranking_scores(
                scores_rec_diffu, test_batch[0], candidate_mask
            )
            metrics = hrs_and_ndcgs_k(
                scores_rec_diffu[valid_rows], test_batch[1][valid_rows], [5, 10, 20]
            )
            for k, v in metrics.items():
                test_metrics_dict[k].append(v)
    for key_temp, values_temp in test_metrics_dict.items():
        values_mean = round(np.mean(values_temp) * 100, 4)
        test_metrics_dict_mean[key_temp] = values_mean
    print(test_metrics_dict_mean)

def evaluate_and_print(model, data_loader, args, logger, description="evaluation", save_recs=False):
    """ инференс на data_loader, выводит время и метрики @10"""
    device = args.device
    use_amp = getattr(args, 'amp', False) and device == 'cuda'
    candidate_items = set(args.coverage_candidate_items)
    candidate_items.discard(0)
    candidate_mask = build_candidate_mask(
        candidate_items,
        args.item_num + 1,
        device,
    )
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type='cuda', dtype=torch.float16, enabled=use_amp
    ):
        all_actual = []
        all_predicted = []
        excluded_examples = 0
        start_time = time.time()
        for batch in data_loader:
            batch = [x.to(device, non_blocking=True) for x in batch]
            batch[0] = filter_history_to_candidates(batch[0], candidate_mask)
            _, rep_diffu, _, _, _, _ = model(batch[0], batch[1], train_flag=False)
            scores = model.diffu_rep_pre(rep_diffu)
            seq = batch[0]  # [batch_size, max_len]
            valid_rows = eligible_warm_start_rows(seq, batch[1], candidate_mask)
            excluded_examples += (~valid_rows).sum().item()
            mask_ranking_scores(scores, seq, candidate_mask)
            k_max = max(args.metric_ks)
            _, topk = torch.topk(scores, k=k_max, dim=-1)
            for i in valid_rows.nonzero(as_tuple=False).squeeze(-1).tolist():
                all_actual.append([batch[1][i].item()])
                all_predicted.append(topk[i].cpu().tolist())
        inference_time = time.time() - start_time
        num_users = len(all_actual)
        if num_users == 0:
            raise ValueError("No eligible warm-start examples remain for evaluation")
        if excluded_examples:
            message = (
                f"{description}: excluded {excluded_examples} examples with an "
                "empty history or a target outside the training catalogue"
            )
            print(message)
            logger.info(message)
        print(f"{description} inference time: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms per user")
        logger.info(f"{description} inference time: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms per user")
        
        topN_list = args.metric_ks
        prec, rec, ndcg, mrr, cov = compute_all_metrics(
            all_actual,
            all_predicted,
            topN_list,
            len(candidate_items),
            candidate_items=candidate_items,
        )
        print("\nTest results:")
        print(f"{'k':<5} {'recall':<12} {'ndcg':<12} {'mrr':<12} {'coverage':<12}")
        for i, k in enumerate(topN_list):
            print(f"{k:<5} {rec[i]:<12.6f} {ndcg[i]:<12.6f} {mrr[i]:<12.6f} {cov[i]:<12.6f}")
        tracker = getattr(args, "experiment_tracker", None)
        if tracker is not None and description.lower() == "test":
            tracker.log_final_metrics(
                {k: {"recall": rec[i], "ndcg": ndcg[i], "mrr": mrr[i], "coverage": cov[i]}
                 for i, k in enumerate(topN_list)},
                split="global_temporal_70_10_20",
                mask_seen=True,
                seed=args.random_seed,
                inference_total_sec=inference_time,
                n_users=num_users,
                maxlen=args.max_len,
                popularity_bias=recommendation_popularity(
                    all_predicted, getattr(args, "train_item_popularity", {}), topN_list
                ),
            )
            tracker.close()

        if save_recs:
            import pandas as pd
            recs_df = pd.DataFrame({'user_id': list(range(len(all_actual))), 'recommendations': all_predicted})
            recs_df.to_csv('recommendations.csv', index=False)
            print("Recommendations saved to recommendations.csv")
        
        # Логирование (опционально, без ошибок):
        logger.info(f"{description} inference: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms/user")

def model_train(tra_data_loader, val_data_loader, test_data_loader, model_joint, args, logger):
    tracker = ExperimentTracker(args.dataset, "DiffuRec")
    args.experiment_tracker = tracker
    plotter = TrainingPlotter(
        save_dir=args.log_file + args.dataset,
        model_name=f"{args.description}_{time.strftime('%Y%m%d_%H%M%S')}",
        metrics=['loss', 'recall@10']
        )
    epochs = args.epochs
    device = args.device
    metric_ks = args.metric_ks
    model_joint = model_joint.to(device)
    candidate_mask = build_candidate_mask(
        args.coverage_candidate_items,
        args.item_num + 1,
        device,
    )
    is_parallel = args.num_gpu > 1
    if is_parallel:
        model_joint = nn.DataParallel(model_joint)
    optimizer = optimizers(model_joint, args)
    use_amp = getattr(args, 'amp', False) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.gamma)
    
    best_metrics_dict = {}      # логирование, будет заполняться динамически для всех метрик
    best_epoch = {}             # аналогично
    best_recall10 = -1.0        # для early stopping
    best_model = None
    bad_count = 0
    
    for epoch_temp in range(epochs):        
        print('Epoch: {}'.format(epoch_temp))
        logger.info('Epoch: {}'.format(epoch_temp))
        model_joint.train()
    
        flag_update = 0
        # for index_temp, train_batch in enumerate(tra_data_loader):
        for index_temp, train_batch in enumerate(tqdm(tra_data_loader, desc=f"Epoch {epoch_temp:03d}/{epochs}", unit="batch")):
            train_batch = [x.to(device, non_blocking=True) for x in train_batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                scores, diffu_rep, weights, t, item_rep_dis, seq_rep_dis = model_joint(train_batch[0], train_batch[1], train_flag=True)
                loss_diffu_value = model_joint.loss_diffu_ce(diffu_rep, train_batch[1])  ## use this not above
                loss_all = loss_diffu_value
            scaler.scale(loss_all).backward()
            scaler.step(optimizer)
            scaler.update()
            if index_temp % int(len(tra_data_loader) / 5 + 1) == 0:
                print('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
                logger.info('[%d/%d] Loss: %.4f' % (index_temp, len(tra_data_loader), loss_all.item()))
        print("loss in epoch {}: {}".format(epoch_temp, loss_all.item()))
        plotter.update(epoch=epoch_temp, loss=loss_all.item())
        tracker.log_epoch(epoch_temp, train_loss=loss_all.item())
        lr_scheduler.step()

        if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
            print('start predicting: ', datetime.datetime.now())
            logger.info('start predicting: {}'.format(datetime.datetime.now()))
            model_joint.eval()
            
            # Собираем все предсказания и истинные метки по батчам
            all_actual = []      # список списков (каждый список содержит целевой айтем)
            all_predicted = []   # список списков индексов top‑k_max (k_max = max(metric_ks))
            
            with torch.no_grad():
                for val_batch in val_data_loader:
                    val_batch = [x.to(device) for x in val_batch]
                    val_batch[0] = filter_history_to_candidates(
                        val_batch[0], candidate_mask
                    )
                    _, rep_diffu, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
                    scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)   # [batch_size, num_items]
                    valid_rows = eligible_warm_start_rows(
                        val_batch[0], val_batch[1], candidate_mask
                    )
                    mask_ranking_scores(
                        scores_rec_diffu, val_batch[0], candidate_mask
                    )
                    # Получаем top‑k_max индексов для каждого пользователя в батче
                    k_max = max(args.metric_ks)
                    _, topk_indices = torch.topk(scores_rec_diffu, k=k_max, dim=-1)  # [batch_size, k_max]
                    
                    # Сохраняем
                    for i in valid_rows.nonzero(as_tuple=False).squeeze(-1).tolist():
                        all_actual.append([val_batch[1][i].item()])
                        all_predicted.append(topk_indices[i].cpu().tolist())
            
            # Вычисляем метрики
            topN_list = args.metric_ks
            if not all_actual:
                raise ValueError("No eligible warm-start validation examples remain")
            precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
                all_actual,
                all_predicted,
                topN_list,
                len(args.coverage_candidate_items),
                candidate_items=args.coverage_candidate_items,
            )
            
            # Формируем словарь для логирования (precision не нужен)
            metrics_dict = {}
            for k, rec, nd, mrr, cov in zip(topN_list, recalls, ndcgs, mrrs, covs):
                if k == 10:
                    metrics_dict[f'Recall@{k}'] = rec
                    metrics_dict[f'NDCG@{k}'] = nd
                    metrics_dict[f'MRR@{k}'] = mrr
                    metrics_dict[f'Coverage@{k}'] = cov
            
            # Обновление best_metrics_dict и early stopping 
            flag_update = 0
            # Обновление best_metrics_dict для логирования (все метрики)
            for key_temp, values_temp in metrics_dict.items():
                if 'Best_' + key_temp not in best_metrics_dict or values_temp > best_metrics_dict['Best_' + key_temp]:
                    best_metrics_dict['Best_' + key_temp] = values_temp
                    best_epoch['Best_epoch_' + key_temp] = epoch_temp

            # Early stopping только по recall@10
            recall10 = metrics_dict.get('Recall@10', 0.0)
            if recall10 > best_recall10:
                best_recall10 = recall10
                bad_count = 0
                best_model = copy.deepcopy(model_joint)
                print(f"New best recall@10: {recall10:.4f}")
                # Также выводим все лучшие метрики
                print(best_metrics_dict)
                print(best_epoch)
                logger.info(best_metrics_dict)
                logger.info(best_epoch)
            else:
                bad_count += 1
            # plotter.update(
            #     epoch=epoch_temp,
            #     loss=loss_all.item(),
            #     val_recall=recall10
            # )
            plotter.update(epoch=epoch_temp, val_recall=recall10)
            idx10 = topN_list.index(10) if 10 in topN_list else 0
            tracker.log_epoch(epoch_temp, **{
                "val_recall@10": recalls[idx10],
                "val_ndcg@10": ndcgs[idx10],
                "val_mrr@10": mrrs[idx10],
            })
            plotter.plot(save=True, show=False, suffix=f'_epoch{epoch_temp}')
            # Сохранять график каждые eval_interval для отслеживания прогресса
            # plotter.plot(save=True, show=False, suffix=f'_epoch{epoch_temp}')
            if bad_count >= args.patience:
                break
            # for key_temp, values_temp in metrics_dict.items():
            #     values_mean = values_temp  # уже число
            #     if values_mean > best_metrics_dict.get('Best_' + key_temp, -1):
            #         flag_update = 1
            #         bad_count = 0
            #         best_metrics_dict['Best_' + key_temp] = values_mean
            #         best_epoch['Best_epoch_' + key_temp] = epoch_temp
            
            # if flag_update == 0:
            #     bad_count += 1
            # else:
            #     print(best_metrics_dict)
            #     print(best_epoch)
            #     logger.info(best_metrics_dict)
            #     logger.info(best_epoch)
            #     best_model = copy.deepcopy(model_joint)
            
            # if bad_count >= args.patience:
            #     break
        
        # if epoch_temp != 0 and epoch_temp % args.eval_interval == 0:
        #     print('start predicting: ', datetime.datetime.now())
        #     logger.info('start predicting: {}'.format(datetime.datetime.now()))
        #     model_joint.eval()
        #     with torch.no_grad():
        #         metrics_dict = {'HR@5': [], 'NDCG@5': [], 'HR@10': [], 'NDCG@10': [], 'HR@20': [], 'NDCG@20': []}
        #         # metrics_dict_mean = {}
        #         for val_batch in val_data_loader:
        #             val_batch = [x.to(device) for x in val_batch]
        #             scores_rec, rep_diffu, _, _, _, _ = model_joint(val_batch[0], val_batch[1], train_flag=False)
        #             scores_rec_diffu = model_joint.diffu_rep_pre(rep_diffu)    ### inner_production
        #             # scores_rec_diffu = model_joint.routing_rep_pre(rep_diffu)   ### routing_rep_pre
        #             metrics = hrs_and_ndcgs_k(scores_rec_diffu, val_batch[1], metric_ks)
        #             for k, v in metrics.items():
        #                 metrics_dict[k].append(v)
                        
        #     for key_temp, values_temp in metrics_dict.items():
        #         values_mean = round(np.mean(values_temp) * 100, 4)
        #         if values_mean > best_metrics_dict['Best_' + key_temp]:
        #             flag_update = 1
        #             bad_count = 0
        #             best_metrics_dict['Best_' + key_temp] = values_mean
        #             best_epoch['Best_epoch_' + key_temp] = epoch_temp
                    
        #     if flag_update == 0:
        #         bad_count += 1
        #     else:
        #         print(best_metrics_dict)
        #         print(best_epoch)
        #         logger.info(best_metrics_dict)
        #         logger.info(best_epoch)
        #         best_model = copy.deepcopy(model_joint)
        #     if bad_count >= args.patience:
        #         break
            
    plotter.plot(save=True, show=False, suffix='_final')
    logger.info(best_metrics_dict)
    logger.info(best_epoch)
    # Гарантируем, что best_model не None (если улучшений не было, берём последнюю модель)
    if best_model is None:
        best_model = copy.deepcopy(model_joint)
    # if args.eval_interval > epochs:
    #     best_model = copy.deepcopy(model_joint)
    
    # # Тестирование на лучшей модели
    
    # top_100_item = []   # для diversity_measure, если нужно
    # with torch.no_grad():
    #     all_actual = []
    #     all_predicted = []
    #     start_time = time.time()
    #     for test_batch in test_data_loader:
    #         test_batch = [x.to(device) for x in test_batch]
    #         _, rep_diffu, _, _, _, _ = best_model(test_batch[0], test_batch[1], train_flag=False)
    #         scores_rec_diffu = best_model.diffu_rep_pre(rep_diffu)
    #         k_max = max(args.metric_ks)
    #         _, topk_indices = torch.topk(scores_rec_diffu, k=k_max, dim=-1)
    #         for i in range(len(test_batch[1])):
    #             all_actual.append([test_batch[1][i].item()])
    #             all_predicted.append(topk_indices[i].cpu().tolist())
    #         # Для diversity_measure (если нужно) собираем top-100
        #     if args.diversity_measure:
        #         _, top100 = torch.topk(scores_rec_diffu, k=100, dim=-1)
        #         top_100_item.append(top100.cpu())
        # inference_time = time.time() - start_time
        # num_users = len(all_actual)
        # print(f"Inference latency: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms per user")
        # logger.info(f"Inference latency: total {inference_time:.2f} sec, avg {inference_time/num_users*1000:.2f} ms per user")
        # # Вычисляем метрики
        # precisions, recalls, ndcgs, mrrs, covs = compute_all_metrics(
        #     all_actual, all_predicted, args.metric_ks, args.item_num
        # )
        # test_metrics_dict_mean = {}
        # for k, rec, nd, mrr, cov in zip(args.metric_ks, recalls, ndcgs, mrrs, covs):
        #     if k == 10:
        #         test_metrics_dict_mean[f'Recall@{k}'] = round(rec, 4)
        #         test_metrics_dict_mean[f'NDCG@{k}'] = round(nd, 4)
        #         test_metrics_dict_mean[f'MRR@{k}'] = round(mrr, 4)
        #         test_metrics_dict_mean[f'Coverage@{k}'] = round(cov, 4)

        
        # print('Test------------------------------------------------------')
        # logger.info('Test------------------------------------------------------')
        # print(test_metrics_dict_mean)
        # logger.info(test_metrics_dict_mean)


    # print(best_metrics_dict)
    # print(best_epoch)
    # logger.info(best_metrics_dict)
    # logger.info(best_epoch)

    print(args)

    # if args.diversity_measure:
    #     path_data = '../datasets/data/category/' + args.dataset +'/id_category_dict.pkl'
    #     with open(path_data, 'rb') as f:
    #         id_category_dict = pickle.load(f)
    #     id_top_100 = torch.cat(top_100_item, dim=0).tolist()
    #     category_list_100 = []
    #     for id_top_100_temp in id_top_100:
    #         category_temp_list = [] 
    #         for id_temp in id_top_100_temp:
    #             category_temp_list.append(id_category_dict[id_temp])
    #         category_list_100.append(category_temp_list)
    #     category_list_100.append(category_list_100)
    #     path_data_category = '../datasets/data/category/' + args.dataset +'/DiffuRec_top100_category.pkl'
    #     with open(path_data_category, 'wb') as f:
    #         pickle.dump(category_list_100, f)
            

    return best_model, None
    
