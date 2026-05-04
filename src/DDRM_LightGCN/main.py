import world
import utils
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
import dataloader
from os.path import join
from parse import parse_args
import torch.utils.data as data
import diffusion as gd
import register
import random
from scipy.sparse import csr_matrix 
import json, sys

from plotting import TrainingPlotter

if __name__ == '__main__':
    utils.set_seed(world.seed)
    print(">>SEED:", world.seed)
    
    # define dataset
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    args = parse_args()
    dataset = dataloader.DiffData(path = args.data_path) 
    train_loader = data.DataLoader(dataset,
            batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    if args.final:
        try:
            with open('best_params.json', 'r') as f:
                best_params = json.load(f)
            for key, value in best_params.items():
                setattr(args, key, value)
                world.config[key] = value
            print("Загружены лучшие параметры из best_params.json")
        except FileNotFoundError:
            print("ОШИБКА: best_params.json не найден. Сначала запустите tune.py")
            sys.exit(1)

        print("[FINAL MODE] Объединяем train и valid выборки ...")
        combined_train = np.concatenate([dataset.train_list, dataset.valid_list], axis=0)

        # Перестроим train_dict
        dataset.train_dict = {}
        for uid, iid in combined_train:
            dataset.train_dict.setdefault(uid, []).append(iid)

        # Новые массивы пользователей и предметов
        trainUniqueUsers, trainUser, trainItem = [], [], []
        dataset.traindataSize = 0
        for uid in dataset.train_dict:
            items = dataset.train_dict[uid]
            trainUniqueUsers.append(uid)
            trainUser.extend([uid] * len(items))
            trainItem.extend(items)
            dataset.traindataSize += len(items)

        dataset.trainUniqueUsers = np.array(trainUniqueUsers)
        dataset.trainUser = np.array(trainUser)
        dataset.trainItem = np.array(trainItem)

        # n_user и m_item уже установлены по максимуму из всех файлов (train+valid+test)
        # Перестраиваем разреженную матрицу взаимодействий
        dataset.UserItemNet = csr_matrix(
            (np.ones(len(dataset.trainUser)), (dataset.trainUser, dataset.trainItem)),
            shape=(dataset.n_user, dataset.m_item)
        )
        dataset.users_D = np.array(dataset.UserItemNet.sum(axis=1)).squeeze()
        dataset.users_D[dataset.users_D == 0.] = 1
        dataset.items_D = np.array(dataset.UserItemNet.sum(axis=0)).squeeze()
        dataset.items_D[dataset.items_D == 0.] = 1.

        # Обновляем кэш положительных элементов (теперь включает train+val)
        dataset._allPos = dataset.getUserPosItems(list(range(dataset.n_user)))
        print(f"Объединённый обучающий набор: {dataset.traindataSize} взаимодействий")

    # define rec mdoel
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)

    # define diffusion reverse model
    out_dims = eval(args.dims) + [args.recdim]
    in_dims = out_dims[::-1]
    # w_out_dims = eval(args.w_dims) + [args.recdim]
    # w_in_dims = w_out_dims[::-1]
    # user_reverse_model = register.DIFF_MODELS['transformer'](in_dims, out_dims, w_in_dims, w_out_dims, norm=args.norm)
    # user_reverse_model = user_reverse_model.to(world.device)

    # item_reverse_model = register.DIFF_MODELS['transformer'](in_dims, out_dims, w_in_dims, w_out_dims, norm=args.norm)
    # item_reverse_model = item_reverse_model.to(world.device)

    user_reverse_model = register.DIFF_MODELS['dnn'](in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm)
    user_reverse_model = user_reverse_model.to(world.device)

    item_reverse_model = register.DIFF_MODELS['dnn'](in_dims, out_dims, args.emb_size, time_type="cat", norm=args.norm)
    item_reverse_model = item_reverse_model.to(world.device)

    # define Gaussian Diffusion
    if args.mean_type == 'x0':
        mean_type = gd.ModelMeanType.START_X
    elif args.mean_type == 'eps':
        mean_type = gd.ModelMeanType.EPSILON
    else:
        raise ValueError("Unimplemented mean type %s" % args.mean_type)
    diffusion = gd.GaussianDiffusion(world.config, mean_type, args.noise_schedule, \
            args.noise_scale, args.noise_min, args.noise_max, args.steps, world.device).to(world.device)

    # define bpr
    bpr = utils.BPRLoss(Recmodel, user_reverse_model, item_reverse_model, diffusion, world.config)

    weight_file, user_weight_file, item_weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")

    # path = './pretrain_checkpoint/' + args.dataset + '_LightGCN_checkpoint.tar'
    # Recmodel.load_state_dict(torch.load(path,map_location=torch.device('cpu')))
    # print(f"loaded model weights from {path}")

    Neg_k = 1

    # get config
    config = world.config

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        print("not enable tensorflowboard")

    import os
    from datetime import datetime

    os.makedirs("./log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./log/{timestamp}"
    # plotter = TrainingPlotter(save_dir, model_name="DDRM-LightGCN", metrics=['loss', 'val_recall'])
    if args.final:
        plotter = TrainingPlotter(save_dir, model_name="DDRM-LightGCN", metrics=['loss'])
    else:
        plotter = TrainingPlotter(save_dir, model_name="DDRM-LightGCN", metrics=['loss', 'val_recall'])
    
    try:
        iter = 0   # счётчик итераций для call_bpr
        if args.final:
            # ---------- ФИНАЛЬНЫЙ РЕЖИМ ----------
            print(f"[final training] Запуск на {world.TRAIN_epochs} эпох ...")
            for epoch in range(world.TRAIN_epochs):
                Recmodel.train()
                user_reverse_model.train()
                item_reverse_model.train()
                train_loader.dataset.get_pair_bpr()
                aver_loss = 0.
                idx = 0
                for batch_users, batch_pos, batch_neg in train_loader:
                    batch_users = batch_users.to(world.device)
                    batch_pos = batch_pos.to(world.device)
                    batch_neg = batch_neg.to(world.device)
                    loss = bpr.call_bpr(batch_users, batch_pos, batch_neg, iter)
                    aver_loss += loss
                    idx += 1
                    iter += 1

                aver_loss = aver_loss / idx
                plotter.update(epoch=epoch, loss=aver_loss)
                print(f'EPOCH [{epoch+1}/{world.TRAIN_epochs}] loss: {aver_loss:.4f}')

                if (epoch + 1) % 5 == 0 or epoch == world.TRAIN_epochs - 1:
                    plotter.plot(save=True, show=False)

            # Сохраняем финальные веса
            torch.save(Recmodel.state_dict(), weight_file)
            torch.save(user_reverse_model.state_dict(), user_weight_file)
            torch.save(item_reverse_model.state_dict(), item_weight_file)

            # Оценка на тесте (история = train+val)
            test_users = list(dataset.test_dict.keys())
            ground_truth_dict = dataset.test_dict
            allPos_baseline = dataset.getUserPosItems(test_users)

            metrics = Procedure.Test_all(
                dataset, Recmodel, user_reverse_model, item_reverse_model, diffusion,
                users=test_users,
                allPos=allPos_baseline,
                ground_truth_dict=ground_truth_dict,
                multicore=world.config['multicore']
            )

            k_index = 0
            print("\nфинальные результаты")
            print(f"Recall@10: {metrics['recall'][k_index]:.4f}")
            print(f"NDCG@10:  {metrics['ndcg'][k_index]:.4f}")
            print(f"Coverage@10: {metrics['coverage'][k_index]:.4f}")
            print(f"MRR@10:   {metrics['mrr'][k_index]:.4f}")
            print(f"Latency:  {metrics['latency']:.4f} сек")

        else:
            best_recall = 0
            best_epoch = 0
            recall_list = []
            cnt = 0
            for epoch in range(world.TRAIN_epochs):
                Recmodel.train()
                user_reverse_model.train()
                item_reverse_model.train()
                train_loader.dataset.get_pair_bpr()
                aver_loss = 0.
                idx = 0
                for batch_users, batch_pos, batch_neg in train_loader:
                    batch_users = batch_users.to(world.device)
                    batch_pos = batch_pos.to(world.device)
                    batch_neg = batch_neg.to(world.device)
                    loss = bpr.call_bpr(batch_users, batch_pos, batch_neg, iter)
                    aver_loss += loss
                    idx += 1
                    iter += 1

                aver_loss = aver_loss / idx
                plotter.update(epoch=epoch, loss=aver_loss)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] loss:{aver_loss}')

                if (epoch+1) % 5 == 0:
                    results = Procedure.Test(dataset, Recmodel, user_reverse_model,
                                             item_reverse_model, diffusion, epoch,
                                             w, world.config['multicore'])
                    val_recall_at_10 = results[1][0]
                    plotter.update(epoch=epoch, val_recall=val_recall_at_10)
                    Procedure.print_results(results)
                    if results[1][0] > best_recall:
                        best_epoch = epoch
                        best_recall = results[1][0]
                        best_v = results
                        torch.save(Recmodel.state_dict(), weight_file)
                        torch.save(user_reverse_model.state_dict(), user_weight_file)
                        torch.save(item_reverse_model.state_dict(), item_weight_file)
                    if epoch == 30:
                        recall_list.append((epoch, results[1][0]))
                    if epoch > 30:
                        recall_list.append((epoch, results[1][0]))
                        if results[1][0] < best_recall:
                            cnt += 1
                        else:
                            cnt = 1
                        if cnt >= 6:
                            break
                    if (epoch+1) % 5 == 0 or epoch == world.TRAIN_epochs - 1:
                        plotter.plot(save=True, show=False)

            print("End train and valid. Best validation epoch is {:03d}. ".format(best_epoch))
            # Загружаем лучшую модель и тестируем
            Recmodel.load_state_dict(torch.load(weight_file, map_location=torch.device('cpu')))
            user_reverse_model.load_state_dict(torch.load(user_weight_file, map_location=torch.device('cpu')))
            item_reverse_model.load_state_dict(torch.load(item_weight_file, map_location=torch.device('cpu')))

            test_users = list(dataset.test_dict.keys())
            ground_truth_dict = dataset.test_dict
            allPos_baseline = dataset.getUserPosItems(test_users)   # только train

            metrics = Procedure.Test_all(
                dataset, Recmodel, user_reverse_model, item_reverse_model, diffusion,
                users=test_users,
                allPos=allPos_baseline,
                ground_truth_dict=ground_truth_dict,
                multicore=world.config['multicore']
            )

            k_index = 0
            print("\nфинальные результаты")
            print(f"Recall@10: {metrics['recall'][k_index]:.4f}")
            print(f"NDCG@10: {metrics['ndcg'][k_index]:.4f}")
            print(f"Coverage@10: {metrics['coverage'][k_index]:.4f}")
            print(f"MRR@10: {metrics['mrr'][k_index]:.4f}")
            print(f"Latency: {metrics['latency']:.4f} seconds")

    finally:
        if world.tensorboard:
            w.close()