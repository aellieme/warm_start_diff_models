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
   pip install -r src/requirements.txt
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

For a complete Google Colab workflow (setup, all models, TensorBoard, plots, result tables, and ZIP download), open the root-level [`demo.ipynb`](demo.ipynb).

### Common command line arguments (unified across models)

| Argument | Description | Example |
|----------|-------------|---------|
| `--dataset` | Dataset name: `ml-1m`, `amazon_Baby`, `amazon_Toys_and_Games` | `--dataset ml-1m` |
| `--final_train` | Use train+validation for final training and evaluate on test | `--final_train` |
| `--max_len` | Max sequence length (history truncation) | `--max_len 50` |
| `--batch_size` | Training batch size | `--batch_size 512` |
| `--amp` | CUDA mixed precision for faster DiffuRec training | `--amp` |
| `--eval_repeats` | Fixed stochastic DiffuRec inference runs averaged during validation | `--eval_repeats 5` |
| `--resume_checkpoint` | Resume DiffuRec tuning from a saved path, or from `latest` matching the current configuration | `--resume_checkpoint latest` |
| `--epochs` / `--num_epochs` | Number of training epochs | `--epochs 250` |
| `--metric_ks` | List of K values for evaluation | `--metric_ks 10 20 100` |
| `--hidden_size` | Embedding / hidden dimension | `--hidden_size 64` |
| `--num_blocks` | Number of transformer/attention blocks | `--num_blocks 2` |
| `--random_seed` | Fixed seed for reproducibility | `--random_seed 42` |

> Note: Not every model supports all arguments. Check the model’s own `main.py` for exact parameter names (e.g., GPTRec uses `dataset.max_length=50`). Below we provide the exact commands used in the experiments.

### 1. DiffuRec samples

```bash
cd src/DiffuRec/src

# Tuned fast configuration selected on ML-1M validation (26 final epochs)
python main.py --dataset ml-1m --final_train --max_len 50 --batch_size 1024 --epochs 26 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --lr 0.003 --noise_schedule cosine --random_seed 42

# Same DiffuRec configuration with CUDA mixed precision
python main.py --dataset ml-1m --final_train --max_len 50 --batch_size 1024 --epochs 26 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --lr 0.003 --noise_schedule cosine --random_seed 42 --device cuda --amp

# Run only fixed dataset/maxlen-specific configurations.
# No --epochs is needed: each preset contains its fixed epoch count.
# Every completed run is added to the results registry.
python src/experiment_tools/run_demo_experiments.py --models DiffuRec --datasets ml-1m --maxlens 50 --tuned-diffurec --amp --prepare-data --run
python src/experiment_tools/run_demo_experiments.py --models DiffuRec --datasets amazon_Baby --maxlens 50 100 --tuned-diffurec --amp --prepare-data --run
python src/experiment_tools/run_demo_experiments.py --models DiffuRec --datasets amazon_Toys_and_Games --maxlens 50 --tuned-diffurec --amp --prepare-data --run

# Amazon Baby, max_len=50: validation-selected batch_size=256, 100 epochs
python main.py --dataset amazon_Baby --final_train --max_len 50 --batch_size 256 --epochs 100 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --lr 0.001 --noise_schedule trunc_lin --random_seed 42 --device cuda --amp

# Amazon Baby, max_len=100: validation-selected batch_size=128, 130 epochs
python main.py --dataset amazon_Baby --final_train --max_len 100 --batch_size 128 --epochs 130 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --lr 0.001 --noise_schedule trunc_lin --random_seed 42 --device cuda --amp

# Amazon Toys, max_len=50: validation-selected batch_size=256, 70 epochs
python main.py --dataset amazon_Toys_and_Games --final_train --max_len 50 --batch_size 256 --epochs 70 --metric_ks 10 20 100 --hidden_size 64 --num_blocks 2 --lr 0.003 --noise_schedule cosine --random_seed 42 --device cuda --amp

```

Training curves and final metrics are saved in `./log/`.

Non-final DiffuRec tuning runs save a resumable checkpoint every 10 completed
epochs. Each exact hyperparameter configuration retains only its two newest
checkpoints. The files contain the model, optimizer, LR scheduler, AMP scaler,
validation selection, plots/history, and RNG states. Set
`EXPERIMENT_OUTPUT_DIR` to persistent storage (for example,
`/content/drive/MyDrive/experiment_results/logs` in Colab) before training;
checkpoints are written to the sibling `checkpoints/DiffuRec/tuning` directory.
After an interruption, repeat the same command with
`--resume_checkpoint latest`. Changing a training or validation-selection
parameter is rejected when resuming instead of silently mixing configurations.

The `--tuned-diffurec` presets currently fix ML-1M at `max_len=50`, Amazon Baby
at `max_len=50/100`, and Amazon Toys at `max_len=50`. A requested combination
without a fixed preset is rejected instead of silently receiving unrelated
fallback parameters. All listed configurations were selected on validation;
final test metrics are not used by the runner for preset selection. Use
`--diffurec-lr` only when an explicit LR override is intended.

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


Before the first launch, install the dependencies for the graphs: `python -m pip install -r src/visualization_requirements.txt`. Then run the training with the usual commands above — no additional flag is needed, the graphs are built automatically.

The results of each local run are stored in `experiment_results/logs/<dataset>/<model>/<run_id>/`. Depending on the training mode, `plots/` contains `loss.png`, `validation_ranking.png`, `metrics_by_k.png`, and `popularity_bias.png`. Reusable model checkpoints are stored in `experiment_results/checkpoints/<model>/`.

To view the loss and ranking metrics of all launches in TensorBoard, run `tensorboard --logdir experiment_results/logs` from the root of the project and open the address shown in the terminal.

### Automatic result tables

Each completed run automatically updates `experiment_results/results_registry.csv`; no metrics need to be copied by hand. To keep logs, tables, and checkpoints between Google Colab sessions, mount Google Drive and set the output directory before training:

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['EXPERIMENT_OUTPUT_DIR'] = '/content/drive/MyDrive/experiment_results/logs'
```

After any number of model runs, generate all 9 tables (3 datasets × K = 10, 20, 100) with `python src/experiment_tools/generate_result_tables.py`. The files `experimental_results.md` and `experimental_results.xlsx` will be saved in `MyDrive/experiment_results/reports/tables/`; missing experiments are shown as `—`.

If Google Drive was not used, download the local Colab results before disconnecting the runtime:

```python
!python src/experiment_tools/package_experiment_results.py --output /content/experiment_results.zip

from google.colab import files
files.download("experiment_results.zip")
```

Extract the downloaded archive into the root of the local project in VS Code. It creates one local `experiment_results/` folder containing logs, tables, and checkpoints.

---

## Reproducibility notes

- **Global temporal split**: All experiments use the split recommended by [Gusak et al. (RecSys 2025)](https://dl.acm.org/doi/10.1145/3705328.3748164): 70% train, 10% validation, 20% test, sorted by global timestamp.
- **Shared warm-start catalogue**: For every model, validation ranks only items observed in train and final test ranks only items observed in train+validation. Classification/reconstruction losses exclude unavailable item dimensions. Padding, out-of-catalogue items, and every item in the full available history are masked before top-k; `max_len` limits only the model input, not the seen-item mask.
- **Shared sequential target**: Accuracy metrics use exactly one target per user: the raw last event in the validation/test window. The input contains train+validation events and chronologically earlier test events, filtered to the eligible catalogue. A row is excluded when its filtered history is empty or its raw target is outside the catalogue; user identity itself is not an eligibility condition.
- **Expected ML-1M final cohort**: The train+validation catalogue contains 3,662 items and final last-item evaluation contains 1,775 eligible users. DiffuRec, ADRec, SASRec, GPTRec, T-DiffRec, TopPopular, and RandomRecs must report this same cohort.
- **Item IDs and padding**: DiffuRec, ADRec, GPTRec, TopPopular, and RandomRecs reserve `0` for padding. SASRec and T-DiffRec retain their native indexing but mask their padding/out-of-catalogue dimensions explicitly.
- **T-DiffRec preprocessing**: Protocol v2 adds `valid_history.npy`, `valid_targets.npy`, `test_history.npy`, `test_targets.npy`, and `protocol_meta.json`, and maps items from train+validation. Re-run `split_load_data_dp.py` after updating; legacy split files and checkpoints are intentionally rejected by the new inference path.
- **Checkpoint compatibility**: Models must be retrained after this protocol change. Scores from legacy checkpoints are not comparable to protocol-v2 results.
- **Validation-only selection**: Non-final runs, including GPTRec configs that still contain the legacy `test_metrics` field, cannot evaluate the test split. Test metrics are produced only by the final train+validation run.
- **DiffuRec stochastic evaluation**: Validation metrics are averaged over five predeclared reverse-diffusion seeds (`random_seed ... random_seed + 4`) to stabilize checkpoint selection. The final test is evaluated exactly once with `random_seed`; test seeds are never searched or selected. Recommendation lists are never unioned across seeds. Evaluation runs in an isolated PyTorch RNG context, so changing `eval_interval` cannot change subsequent training updates.
- **Fixed random seed**: 42 is used wherever possible.
- **Hyperparameter and checkpoint search**: Selection uses validation only. DiffuRec chooses one checkpoint lexicographically by Recall@10, then NDCG@10, MRR@10, and Coverage@10; it never combines per-metric maxima from different epochs. The selected number of completed epochs is then used for fresh final training on train+validation.
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
