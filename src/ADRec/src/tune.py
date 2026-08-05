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
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.3)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda:0", "cuda:1"],
    )
    return parser.parse_args()


def selection_command(args):
    command = [
        sys.executable,
        "main.py",
        "--dataset",
        args.dataset,
        "--max_len",
        str(args.max_len),
        "--epochs",
        str(args.max_epochs),
        "--patience",
        str(args.patience),
        "--random_seed",
        str(args.random_seed),
        "--batch_size",
        str(args.batch_size),
        "--hidden_size",
        str(args.hidden_size),
        "--lr",
        str(args.lr),
        "--dropout",
        str(args.dropout),
        "--emb_dropout",
        str(args.emb_dropout),
        "--device",
        args.device,
        "--metric_ks",
        "10",
        "20",
        "100",
        "--mask_seen",
        "True",
    ]
    if args.resume_checkpoint:
        command.extend(["--resume_checkpoint", args.resume_checkpoint])
    return command


def latest_selection(dataset, previous):
    root = tuning_files_dir("ADRec", dataset)
    candidates = [
        path for path in root.glob("*/validation_selection.json")
        if path not in previous
    ]
    if not candidates:
        raise RuntimeError("ADRec validation run did not create validation_selection.json")
    path = max(candidates, key=lambda item: item.stat().st_mtime_ns)
    return path, json.loads(path.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    root = tuning_files_dir("ADRec", args.dataset)
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
