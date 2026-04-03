# warm_start_diff_models


## for DDRM LightGCN:


!git clone https://github.com/aellieme/warm_start_diff_models.git


%cd warm_start_diff_models/src/DDRM_LightGCN


!mkdir log


!mkdir -p code/checkpoints


!python train.py


##for GPT2 for Rec

!git clone https://github.com/aellieme/warm_start_diff_models.git


%cd /content/warm_start_diff_models/src/GPT-2_4Rec


!python src/run_train_predict.py --config-name=GPT_train_predict data_path=data/ml-20m.csv task_name=ml-1m_GPT_train dataloader.test_batch_size=256 model_params.n_embd=256

!python src/run_predict.py --config-name=GPT_predict data_path=data/ml-20m.csv dataloader.test_batch_size=256 model_params.n_embd=256