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

# Fixed final-training budgets used by the demo. They intentionally live here
# rather than in notebook commands so rerunning the benchmark cannot silently
# give different models different ad-hoc budgets.
FIXED_MODEL_EPOCHS = {
    "ADRec": 250,
    "SASRec": 250,
    "GPTRec": 250,
    "T-DiffRec": 250,
}

# Fixed DiffuRec configurations used by the demo runner. Keep these
# dataset/maxlen-specific: applying the ML-1M schedule to Amazon caused
# materially different convergence and catalogue coverage. Selection
# provenance is documented in README.md.
DIFFUREC_TUNED_PRESETS = {
    ("ml-1m", 50): {
        "batch_size": 1024,
        "epochs": 26,
        "hidden_size": 64,
        "num_blocks": 2,
        "lr": 0.003,
        "noise_schedule": "cosine",
    },
    ("ml-1m", 100): {
        "batch_size": 512,
        "epochs": 55,
        "hidden_size": 64,
        "num_blocks": 2,
        "lr": 0.003,
        "noise_schedule": "trunc_lin",
    },
    ("amazon_Baby", 50): {
        "batch_size": 128,
        "epochs": 60,
        "hidden_size": 64,
        "num_blocks": 2,
        "lr": 0.001,
        "noise_schedule": "trunc_lin",
    },
    ("amazon_Baby", 100): {
        "batch_size": 128,
        "epochs": 130,
        "hidden_size": 64,
        "num_blocks": 2,
        "lr": 0.001,
        "noise_schedule": "trunc_lin",
    },
    ("amazon_Toys_and_Games", 50): {
        "batch_size": 128,
        "epochs": 60,
        "hidden_size": 64,
        "num_blocks": 2,
        "lr": 0.003,
        "noise_schedule": "trunc_lin",
    },
    ("amazon_Toys_and_Games", 100): {
        "batch_size": 256,
        "epochs": 100,
        "hidden_size": 64,
        "num_blocks": 2,
        "lr": 0.003,
        "noise_schedule": "trunc_lin",
    },
}


def build_commands(
    datasets, maxlens, models, seed=42,
    amp=False, diffurec_eval_repeats=5,
    diffurec_lr=None,
):
    commands = []
    use_cuda = torch.cuda.is_available()

    def add(model, cwd, args):
        if model in models:
            commands.append((model, Path(cwd), [str(value) for value in args]))

    for dataset in datasets:
        alias = ALIASES[dataset]
        for maxlen in maxlens:
            if "DiffuRec" in models:
                preset_key = (dataset, maxlen)
                if preset_key not in DIFFUREC_TUNED_PRESETS:
                    raise ValueError(
                        "No fixed DiffuRec preset for "
                        f"dataset={dataset}, maxlen={maxlen}. "
                        "Select it on validation before adding it to the demo."
                    )
                diffurec_config = dict(DIFFUREC_TUNED_PRESETS[preset_key])
                if diffurec_lr is not None:
                    diffurec_config["lr"] = diffurec_lr
                diffurec_args = [
                    sys.executable, "main.py", "--dataset", dataset, "--final_train",
                    "--max_len", maxlen, "--epochs", diffurec_config["epochs"],
                    "--metric_ks", 10, 20, 100,
                    "--random_seed", seed,
                    "--device", "cuda" if use_cuda else "cpu",
                    "--eval_repeats", diffurec_eval_repeats,
                ]
                diffurec_args.extend([
                    "--batch_size", diffurec_config["batch_size"],
                    "--hidden_size", diffurec_config["hidden_size"],
                    "--num_blocks", diffurec_config["num_blocks"],
                    "--lr", diffurec_config["lr"],
                    "--noise_schedule", diffurec_config["noise_schedule"],
                ])
                if amp:
                    diffurec_args.append("--amp")
                add("DiffuRec", REPO / "src/DiffuRec/src", diffurec_args)
            add("ADRec", REPO / "src/ADRec/src", [
                sys.executable, "main.py", "--dataset", alias["adrec"], "--final",
                "--max_len", maxlen, "--epochs", FIXED_MODEL_EPOCHS["ADRec"],
                "--metric_ks", 10, 20, 100,
                "--mask_seen", "True", "--random_seed", seed,
                "--device", "cuda:0" if use_cuda else "cpu",
            ])
            add("SASRec", REPO / "src/SASRec", [
                sys.executable, "main.py", "--dataset", dataset,
                "--maxlen", maxlen, "--num_epochs", FIXED_MODEL_EPOCHS["SASRec"],
            ])
            add("GPTRec", REPO / "src/GPTRec/src", [
                sys.executable, "run_train_predict.py", f"dataset_name={dataset}",
                f"final_epochs={FIXED_MODEL_EPOCHS['GPTRec']}",
                f"dataset.max_length={maxlen}",
                "evaluator.top_k=[10,20,100]",
            ])

        if "T-DiffRec" in models:
            commands.append(("T-DiffRec preprocessing", REPO / "src/DiffRec/T-DiffRec", [
                sys.executable, "split_load_data_dp.py", "--dataset", dataset,
            ]))
        tdiff_args = [
            sys.executable, "main.py", "--dataset", dataset, "--final_train",
            "--epochs", FIXED_MODEL_EPOCHS["T-DiffRec"],
            "--topN", "[10,20,100]",
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
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable AMP for DiffuRec")
    parser.add_argument(
        "--diffurec-eval-repeats", type=int, default=5,
        help="Fixed reverse-diffusion inference runs averaged for DiffuRec validation",
    )
    parser.add_argument(
        "--diffurec-lr", type=float,
        help="Override the learning rate stored in the fixed DiffuRec preset",
    )
    parser.add_argument(
        "--tuned-diffurec", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fast-diffurec", dest="tuned_diffurec", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--run", action="store_true", help="Execute commands; otherwise only print them")
    args = parser.parse_args()

    commands = build_commands(
        args.datasets, args.maxlens, set(args.models), args.seed,
        amp=args.amp,
        diffurec_eval_repeats=args.diffurec_eval_repeats,
        diffurec_lr=args.diffurec_lr,
    )
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
