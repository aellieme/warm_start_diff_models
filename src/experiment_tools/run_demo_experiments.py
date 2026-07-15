"""Build or run the complete experiment command matrix used by demo.ipynb."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[2]
ALL_MODELS = ("DiffuRec", "ADRec", "T-DiffRec", "SASRec", "GPTRec", "TopPopular", "Random")
ALIASES = {
    "ml-1m": {"adrec": "ml-1m", "baseline": "ml-1m"},
    "amazon_Baby": {"adrec": "baby", "baseline": "baby"},
    "amazon_Toys_and_Games": {"adrec": "toys", "baseline": "toys"},
}


def build_commands(datasets, maxlens, epochs, models, seed=42):
    commands = []
    use_cuda = torch.cuda.is_available()

    def add(model, cwd, args):
        if model in models:
            commands.append((model, Path(cwd), [str(value) for value in args]))

    for dataset in datasets:
        alias = ALIASES[dataset]
        for maxlen in maxlens:
            add("DiffuRec", REPO / "src/DiffuRec/src", [
                sys.executable, "main.py", "--dataset", dataset, "--final_train",
                "--max_len", maxlen, "--epochs", epochs, "--metric_ks", 10, 20, 100,
                "--random_seed", seed, "--device", "cuda" if use_cuda else "cpu",
            ])
            add("ADRec", REPO / "src/ADRec/src", [
                sys.executable, "main.py", "--dataset", alias["adrec"], "--final",
                "--max_len", maxlen, "--epochs", epochs, "--metric_ks", 10, 20, 100,
                "--mask_seen", "True", "--random_seed", seed,
                "--device", "cuda:0" if use_cuda else "cpu",
            ])
            add("SASRec", REPO / "src/SASRec", [
                sys.executable, "main.py", "--dataset", dataset,
                "--maxlen", maxlen, "--num_epochs", epochs,
            ])
            add("GPTRec", REPO / "src/GPTRec/src", [
                sys.executable, "run_train_predict.py", f"dataset_name={dataset}",
                f"final_epochs={epochs}", f"dataset.max_length={maxlen}",
                "evaluator.top_k=[10,20,100]",
            ])

        if "T-DiffRec" in models:
            commands.append(("T-DiffRec preprocessing", REPO / "src/DiffRec/T-DiffRec", [
                sys.executable, "split_load_data_dp.py", "--dataset", dataset,
            ]))
        tdiff_args = [
            sys.executable, "main.py", "--dataset", dataset, "--final_train",
            "--epochs", epochs, "--topN", "[10,20,100]",
        ]
        if use_cuda:
            tdiff_args.append("--cuda")
        add("T-DiffRec", REPO / "src/DiffRec/T-DiffRec", tdiff_args)
        add("TopPopular", REPO / "src/TopPopular", [
            sys.executable, "TopPopular_model.py", "--dataset", alias["baseline"],
            "--topk_list", 10, 20, 100,
        ])
        add("Random", REPO / "src/RandomRecs", [
            sys.executable, "RandomRecsModel.py", "--dataset", alias["baseline"],
            "--topk_list", 10, 20, 100,
        ])
    return commands


def prepare_data(datasets, models):
    if any(name.startswith("amazon_") for name in datasets):
        subprocess.check_call([sys.executable, "download_amazon_data.py"], cwd=REPO / "src")
    if "ADRec" in models:
        required = [
            REPO / "src/ADRec/datasets/data" / ALIASES[name]["adrec"] / "dataset.pkl"
            for name in datasets
        ]
        if not all(path.exists() for path in required):
            print("Preparing ADRec datasets (the source script downloads all supported subsets).")
            subprocess.check_call([sys.executable, "get_data.py"], cwd=REPO / "src/ADRec/src")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=ALIASES, default=["ml-1m"])
    parser.add_argument("--maxlens", nargs="+", type=int, default=[50])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--run", action="store_true", help="Execute commands; otherwise only print them")
    args = parser.parse_args()

    commands = build_commands(args.datasets, args.maxlens, args.epochs, set(args.models), args.seed)
    for model, cwd, command in commands:
        print(f"{model:24s} {cwd} {shlex.join(command)}")
    print(f"Total commands: {len(commands)}")
    if not args.run:
        print("Dry run only. Add --run to start training.")
        return

    if args.prepare_data:
        prepare_data(args.datasets, set(args.models))
    failures = []
    for model, cwd, command in commands:
        print("\n" + "=" * 90 + f"\n{model}\n" + "=" * 90, flush=True)
        result = subprocess.run(command, cwd=cwd, env=os.environ.copy())
        if result.returncode:
            failures.append((model, result.returncode))
    if failures:
        raise SystemExit(f"Failed commands: {failures}")


if __name__ == "__main__":
    main()
