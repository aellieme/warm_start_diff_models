import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR.parents[1]))

from experiment_tools.experiment_tracking import tuning_files_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ml-1m")
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--topN", default="[10,20,100]")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def selection_command(args):
    command = [
        sys.executable,
        "main.py",
        "--dataset",
        args.dataset,
        "--epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--topN",
        args.topN,
    ]
    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        command.append("--cuda")
    return command


def latest_selection(dataset, previous):
    root = tuning_files_dir("T-DiffRec", dataset)
    candidates = [
        path for path in root.glob("*/validation_selection.json")
        if path not in previous
    ]
    if not candidates:
        raise RuntimeError(
            "T-DiffRec validation run did not create validation_selection.json"
        )
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    return path, json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    root = tuning_files_dir("T-DiffRec", args.dataset)
    previous = set(root.glob("*/validation_selection.json"))
    subprocess.run(selection_command(args), cwd=SOURCE_DIR, check=True)
    selection_path, selection = latest_selection(args.dataset, previous)
    selected_epoch = int(selection["selected_epoch"])
    if selected_epoch <= 0:
        raise ValueError(f"Invalid selected epoch: {selected_epoch}")

    print(f"Validation selection: {selection_path}")
    print(f"Selected epochs: {selected_epoch}")


if __name__ == "__main__":
    main()
