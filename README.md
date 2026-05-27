# Warm-Start Diffusion Models for Sequential Recommendation

This repository contains the experimental code and configurations for the comparative analysis of **diffusion-based** and **transformer-based** sequential recommender systems.

The main goal is to evaluate whether diffusion models (DiffuRec, ADRec, T‑DiffRec) can outperform classical transformers (SASRec, GPTRec) in terms of **accuracy**, **catalogue coverage**, and **inference latency** under a strictly controlled, reproducible protocol (global temporal split, fixed random seed, limited compute budget).

**Key findings** (in brief):
- Diffusion models (especially DiffuRec and ADRec) consistently achieve higher Recall@K, NDCG@K, and MRR@K than SASRec and GPTRec on MovieLens‑1M, Amazon Baby, and Amazon Toys.
- GPTRec is the fastest but least accurate.
- DiffuRec offers the best trade‑off between quality and inference time.
- T‑DiffRec shows strong bias toward popular items (high NDCG/MRR but low Recall).

For full details, please refer to the [thesis document](https://github.com/aellieme/warm_start_diff_models/blob/main/PekerskayaDaniela_diploma2026.pdf) (in Russian).


## Repository structure

```
warm_start_diff_models/
├── src/
│   ├── DiffuRec/            # DiffuRec model (denoises item embeddings)
│   ├── ADRec/               # Auto‑regressive diffusion model
│   ├── T-DiffRec/           # Time‑weighted DiffRec
│   ├── SASRec/              # SASRec with full cross‑entropy
│   ├── GPTRec/              # GPTRec (GPT‑2 based, greedy decoding)
│   ├── TopPopular/          # Non‑personalized baseline
│   ├── RandomRecs/          # Random baseline
│   └── download_amazon_data.py
├── README.md
└── requirements.txt
```

All models are implemented as independent subprojects with their own `main.py` and configuration files. The repository is designed to allow launching each model separately with a unified interface (common argument names where possible).

---

## Setup & installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/aellieme/warm_start_diff_models.git
   cd warm_start_diff_models
   ```

2. **Install dependencies**  
   The recommended way is to use `pip` with the provided requirements (you may need to create a virtual environment first):
   ```bash
   pip install -r requirements.txt
   ```
   Key packages: `torch`, `numpy`, `pandas`, `scipy`, `numba`, `optuna`, `polara`, `hydra-core`, `pytorch_lightning`, `clearml`, `wheel`.

   For the Amazon datasets, you also need to download the data (see below).

3. **Download Amazon datasets** (Baby and Toys)  
   From the `src/` folder run:
   ```bash
   python download_amazon_data.py
   ```
   This will download the 2023 version of the Amazon Reviews datasets and place them in the appropriate subfolders.

   MovieLens‑1M will be downloaded automatically by the respective model scripts.

---

## Running experiments

Each model is launched from its own subdirectory inside `src/`.  
All commands assume you are currently in the root of the repository and change directory appropriately.

### Common command line arguments (unified across models)

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset name: `ml-1m`, `amazon_Baby`, `amazon_Toys_and_Games` | `--dataset ml-1m` |
| `--final_train` | Use train+validation for final training and evaluate on test | `--final_train` |
| `--max_len` | Max sequence length (history truncation) | `--max_len 50` |
| `--batch_size` | Training batch size | `--batch_size 512` |
| `--epochs` / `--num_epochs` | Number of training epochs | `--epochs 250` |
| `--metric_ks` | List of K values for evaluation | `--metric_ks 10 20 100` |
| `--hidden_size` | Embedding / hidden dimension | `--hidden_size 64` |
| `--num_blocks` | Number of transformer/attention blocks | `--num_blocks 2` |
| `--random_seed` | Fixed seed for reproducibility | `--random_seed 42` |

> Note: Not every model supports all arguments. Check the model’s own `main.py` for exact parameter names (e.g., GPTRec uses `dataset.max_length=50`). Below we provide the exact commands used in the experiments.

### 1. DiffuRec samples

```bash
cd src/DiffuRec/src

# Train on ml-1m, 100 epochs
python main.py --dataset ml-1m --final_train --max_len 50 --batch_size 1024 --epochs 100 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --random_seed 42

# Amazon Baby, max_len=50, 150 epochs
python main.py --dataset amazon_Baby --final_train --max_len 50 --batch_size 512 --epochs 150 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --random_seed 42

# Amazon Toys, max_len=100, batch_size=256, epochs=150
python main.py --dataset amazon_Toys_and_Games --final_train --max_len 100 --batch_size 256 --epochs 150 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --random_seed 42
```

Training curves and final metrics are saved in `./log/`.

### 2. ADRec

```bash
cd src/ADRec/src

# ml-1m, max_len=100, 250 epochs
python main.py --dataset ml-1m --final --max_len 100 --epochs 250 --metric_ks 10 20 100 --mask_seen True

# Amazon Baby, max_len=100
python main.py --dataset baby --final --max_len 100 --epochs 250 --batch_size 512 --metric_ks 10 20 100 --mask_seen True

# Amazon Toys, max_len=50
python main.py --dataset toys --final --max_len 50 --epochs 250 --batch_size 1024 --metric_ks 10 20 100 --mask_seen True
```

> Note: ADRec uses `--mask_seen True` to filter already interacted items.

### 3. T‑DiffRec

```bash
cd src/DiffRec/T-DiffRec

# First, preprocess the dataset (generate train/val/test splits)
python split_load_data_dp.py --dataset ml-1m
python split_load_data_dp.py --dataset amazon_Baby
python split_load_data_dp.py --dataset amazon_Toys_and_Games

# Train and evaluate
python main.py --dataset ml-1m --final_train --epochs 250 --topN "[10,20,100]"
python main.py --dataset amazon_Baby --final_train --epochs 250 --topN "[10,20,100]" --cuda
python main.py --dataset amazon_Toys_and_Games --final_train --epochs 250 --topN "[10,20,100]" --cuda
```

### 4. SASRec

```bash
cd src/SASRec

# ml-1m, default max_len=50
python main.py --dataset ml-1m

# amazon Baby, default max_len=50
python main.py --dataset amazon_Baby

# amazon Toys with max_len=100
python main.py --dataset amazon_Toys_and_Games --maxlen 100
```

SASRec uses the scalable cross‑entropy loss from [Mezentsev et al., RecSys 2024].  
Training curves are saved as `*_training_curves_final.png` in `./log/`.

### 5. GPTRec

```bash
cd src/GPTRec/src

# ml-1m, max_len=50, 250 epochs
python run_train_predict.py dataset_name=ml-1m final_epochs=250 dataset.max_length=50 evaluator.top_k="[10,20,100]"

# Amazon Baby, max_len=50
python run_train_predict.py dataset_name=amazon_Baby final_epochs=250 dataset.max_length=50 evaluator.top_k="[10,20,100]"

# Amazon Toys, max_len=100
python run_train_predict.py dataset_name=amazon_Toys_and_Games final_epochs=250 dataset.max_length=100 evaluator.top_k="[10,20,100]"
```

### 6. Baselines (Top‑Popular & Random)

```bash
cd src/TopPopular
python TopPopular_model.py --dataset ml-1m --topk_list 10 20 100
python TopPopular_model.py --dataset baby --topk_list 10 20 100
python TopPopular_model.py --dataset toys --topk_list 10 20 100

cd ../RandomRecs
python RandomRecsModel.py --dataset ml-1m
python RandomRecsModel.py --dataset toys
python RandomRecsModel.py --dataset baby
```

---

## Viewing results

After training any model, the final training curves (loss, Recall@K, NDCG@K vs epoch) are saved as PNG images.  
To display the most recent one automatically, you can use the following Python snippet (works inside a notebook or script):

```python
import os, glob
from IPython.display import Image, display

log_dir = './log/'
files = glob.glob(os.path.join(log_dir, '**', '*_training_curves_final.png'), recursive=True)
if files:
    latest = max(files, key=os.path.getmtime)
    display(Image(filename=latest))
else:
    print("No final curves found.")
```

Metrics are also printed to the console at the end of training and saved in log files.

---

## Reproducibility notes

- **Global temporal split**: All experiments use the split recommended by [Gusak et al. (RecSys 2025)](https://dl.acm.org/doi/10.1145/3705328.3748164): 70% train, 10% validation, 20% test, sorted by global timestamp.
- **Fixed random seed**: 42 is used wherever possible
- **Hyperparameter search**: For each model we performed a grid search on the validation set using Recall@10 as the target metric. The best configuration was then used for the final training (train+validation) and test evaluation.
- **Compute budget**: Training was limited to 4 hours per model on an NVIDIA Tesla T4 GPU.

---

## Citation

If you use this code or results in your own work, please cite the original papers of the respective models and the thesis:

```
@thesis{Pekerskaya2026,
  author = {Pekerskaya, Daniela M.},
  title = {Fast recommendations in the warm-start scenario using diffusion models},
  university = {HSE University, Nizhny Novgorod},
  year = {2026}
}
```

Also consider citing the methodological paper:

```
@inproceedings{Gusak2025,
  author = {Gusak, D. and Volodkevich, A. and Klenitskiy, A. and Vasilev, A. and Frolov, E.},
  title = {Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders},
  booktitle = {RecSys '25},
  year = {2025}
}
```

---

## License

This project is released under the MIT License. The original repositories of the included models may have their own licenses – please refer to them separately.
**Maintainer**: [aellieme](https://github.com/aellieme)  
