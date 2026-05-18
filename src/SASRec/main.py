import torch
import numpy as np
import pandas as pd 
import random
import argparse
from model import save_sasrec_model, get_model_path, generate_model_name
from training import build_final_sasrec_model   # используем финальное обучение без валидации
from load_evaluate_pipeline import (
    prepare_data_and_description,
    run_inference_pipeline,
    print_example_user
)
import os
import glob
import matplotlib.pyplot as plt

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-1m',
                        choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty',
                                 'amazon_Sports_and_Outdoors', 'amazon_Toys_and_Games'])
    args = parser.parse_args()

    # Получаем все данные, включая train_val_data (80% до T_test)
    (train_data, val_data, test_data, test_examples,
     data_index, data_description, userid_col, itemid_col, time_col,
     val_seq_dict, train_val_data) = prepare_data_and_description(args.dataset)

    print(f"Train+Val (80%): {len(train_val_data)} interactions")
    print(f"Test examples: {len(test_examples)} users")
    
    # seqs = data_to_sequences(train_val_data, data_description)
    # print(np.mean([len(s) for s in seqs]))

    # Фиксированные гиперпараметры
    config = {
        'num_epochs': 250,
        'maxlen': 50,
        'hidden_units': 256,
        'dropout_rate': 0.2,
        'num_blocks': 2,
        'num_heads': 2,
        'batch_size': 128,
        'sampler_seed': 42,
        'manual_seed': 42,
        'learning_rate': 1e-3,
        'l2_emb': 1e-4,
    }

    print("Training SASRec on train+val (80%)...")
    # Используем build_final_sasrec_model – она обучает на всех переданных данных без валидации
    model = build_final_sasrec_model(config, train_val_data, data_description, num_epochs=config['num_epochs'])
    


    log_dir = './log/'
    # Ищем все файлы с суффиксом '_final.png'
    pattern = os.path.join(log_dir, '*_training_curves_final.png')
    files = glob.glob(pattern)

    if files:
        latest_file = max(files, key=os.path.getmtime)
        print(f"Последний график: {latest_file}")
        img = plt.imread(latest_file)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    else:
        print("График не найден. Проверьте папку ./log/")

    # Сохранение модели
    model_filename = generate_model_name(config, suffix='trainval80')
    model_path = get_model_path(model_filename)
    save_sasrec_model(model, config, data_description, data_index, model_path)
    print(f"Model saved to {model_path}")

    # Инференс на тесте
    print("\nEvaluation on test set...")
    recs, users, metrics, inf_time = run_inference_pipeline(
        model,
        history_data=train_val_data,      # полная история (train+val)
        train_data=train_val_data,        # для фильтрации пользователей
        test_examples=test_examples,
        data_description=data_description,
        userid_col=userid_col,
        itemid_col=itemid_col,
        time_col=time_col,
        val_seq_dict={},                  # val_seq_dict не нужен, т.к. история уже полная
        topn=100
    )

    precisions, recalls, ndcgs, mrrs, covs = metrics
    print(f"Inference time: {inf_time:.4f} sec")
    print(f"Evaluated users: {len(users)}")
    # for k, p, r, ndcg, mrr, cov in zip([10], precisions, recalls, ndcgs, mrrs, covs):
    #     print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")
    for k, p, r, ndcg, mrr, cov in zip([10, 20, 100], precisions, recalls, ndcgs, mrrs, covs):
        print(f"k={k}: Recall(HR)={r:.4f}, NDCG={ndcg:.4f}, MRR={mrr:.4f}, Coverage={cov:.4f}")

    # Пример рекомендаций
    if users:
        example_user = users[0]
        print_example_user(
            example_user, users, recs,
            train_val_data, test_examples,
            data_index, data_description,
            userid_col, itemid_col, time_col, args.dataset
        )
    # Сохраняем рекомендации
    recommendations_df = pd.DataFrame({
        'userid': users,
        'recommendations': [list(rec) for rec in recs]   # каждый rec - массив из topn айтемов
    })
    recommendations_df.to_csv('recommendations_top20.csv', index=False)
    print("Recommendations saved to recommendations_top20.csv")
if __name__ == "__main__":
    main()
# import time
# import torch
# import pandas as pd
# import numpy as np
# from model import save_sasrec_model, get_model_path, generate_model_name
# from training import build_sasrec_model
# from load_evaluate_pipeline import (
#     prepare_data_and_description,
#     run_inference_pipeline,
#     print_example_user
# )
# import random
# import torch
# import argparse

# seed = 42
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='ml-1m',
#                         choices=['ml-1m', 'amazon_Baby', 'amazon_Beauty',
#                                  'amazon_Sports_and_Outdoors', 'amazon_Toys_and_Games'],
#                         help='Dataset to use')
#     args = parser.parse_args()

#     (train_data, val_data, test_data, test_examples,
#      data_index, data_description, userid_col, itemid_col, time_col,
#      val_seq_dict, _) = prepare_data_and_description(args.dataset)
#     # (train_data, val_data, test_data, test_examples, data_index, data_description, userid_col, itemid_col, time_col, val_seq_dict, _) = prepare_data_and_description()


#     print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, len test_examples: {len(test_examples)}")

#     config = {
#         'num_epochs': 200,
#         'maxlen': 200,
#         'hidden_units': 128,
#         'dropout_rate': 0.5,
#         'num_blocks': 2,
#         'num_heads': 2,
#         'batch_size': 128,
#         'sampler_seed': 42, #99
#         'manual_seed': 42, #111
#         'learning_rate': 1e-3,
#         'l2_emb': 0.0,
#     }

#     print("Training SASRec...")
#     model, losses = build_sasrec_model(config, train_data, val_data, data_description, patience=10)

#     # Сохранение модели
#     model_filename = generate_model_name(config, suffix='best')
#     model_path = get_model_path(model_filename)
#     save_sasrec_model(model, config, data_description, data_index, model_path)
#     print(f"Model saved to {model_path}")

#     # Baseline 
#     print("\nbaseline")
#     recs, users, metrics, inf_time = run_inference_pipeline(
#         model, train_data, train_data, test_examples,
#         data_description, userid_col, itemid_col, time_col, val_seq_dict, topn=10
#     )
#     precisions, recalls, ndcgs, mrrs, covs = metrics

#     print(f"Total inference time: {inf_time:.4f} sec")
#     print(f"Evaluated users: {len(users)}")
#     for k, p, r, ndcg, mrr, cov in zip([10], precisions, recalls, ndcgs, mrrs, covs):
#         print(f"k={k}: Recall(HR)={r:.4f}, MRR={mrr:.4f}, NDCG={ndcg:.4f}, Coverage={cov:.4f}")


#     #  Пример для одного пользователя 
#     example_user = users[1]
#     print_example_user(
#         example_user, users, recs,
#         train_data, test_examples,
#         data_index, data_description,
#         userid_col, itemid_col, time_col, args.dataset
#     )

# if __name__ == "__main__":
#     main()