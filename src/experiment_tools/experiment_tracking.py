"""Lightweight local experiment logging shared by all model subprojects.

The module only consumes already computed values.  It never runs evaluation or
changes model, optimizer, data split, masking, or checkpoint selection logic.
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # Scalars and training must keep working without optional plots.
    plt = None

try:
    from tensorboardX import SummaryWriter
except ImportError:  # Logging must never make training unavailable.
    SummaryWriter = None


REPO_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_FIELDS = [
    "dataset", "model", "maxlen", "k", "recall", "ndcg", "mrr", "coverage",
    "latency_sec", "latency_ms_per_user", "n_users", "seed", "split", "mask_seen",
    "run_id", "created_at", "summary_path",
]

DATASET_NAMES = {
    "ml-1m": "ML-1M",
    "baby": "Amazon Baby",
    "amazon_baby": "Amazon Baby",
    "toys": "Amazon Toys",
    "amazon_toys_and_games": "Amazon Toys",
}

MODEL_NAMES = {
    "adrec": "ADRec",
    "diffurec": "DiffuRec",
    "t-diffrec": "T-DiffRec",
    "tdiffrec": "T-DiffRec",
    "sasrec": "SASRec",
    "gpt-2": "GPTRec",
    "gptrec": "GPTRec",
    "toppopular": "TopPopular",
    "random": "Random",
    "randomrecs": "Random",
}


def output_root() -> Path:
    default = REPO_ROOT / "experiment_results" / "logs"
    return Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", default)).resolve()


def checkpoint_path(
    model: str,
    dataset: str,
    maxlen: int | None = None,
    seed: int = 42,
    extension: str = ".pt",
) -> Path:
    """Return a stable checkpoint path beside logs and reports."""
    safe = lambda value: str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")
    model_dir = output_root().parent / "checkpoints" / safe(normalize_model_name(model))
    model_dir.mkdir(parents=True, exist_ok=True)
    parts = [safe(dataset)]
    if maxlen is not None:
        parts.append(f"maxlen{int(maxlen)}")
    parts.append(f"seed{int(seed)}")
    if not extension.startswith("."):
        extension = f".{extension}"
    return model_dir / ("_".join(parts) + extension)


def checkpoint_due(epoch: int, total_epochs: int, every: int | None = None) -> bool:
    """Save infrequently and overwrite one file to keep Drive I/O and storage small."""
    interval = every if every is not None else int(os.environ.get("CHECKPOINT_EVERY", "25"))
    completed = int(epoch) + 1
    return completed == int(total_epochs) or (interval > 0 and completed % interval == 0)


def save_torch_checkpoint(payload, path: Path) -> None:
    """Atomically replace a checkpoint so an interrupted write keeps the old file."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def make_run_dir(dataset: str, model: str, run_id: str | None = None) -> Path:
    safe = lambda value: str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root() / safe(dataset) / safe(model) / safe(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_dataset_name(value: str) -> str:
    return DATASET_NAMES.get(str(value).lower(), str(value))


def normalize_model_name(value: str) -> str:
    return MODEL_NAMES.get(str(value).lower(), str(value))


def registry_path() -> Path:
    return output_root().parent / "results_registry.csv"


class ExperimentTracker:
    """Append-only scalar history plus inexpensive local plots."""

    def __init__(self, dataset: str, model: str, run_id: str | None = None):
        self.dataset = str(dataset)
        self.model = str(model)
        self.run_dir = make_run_dir(dataset, model, run_id)
        self.run_id = self.run_dir.name
        self.plot_dir = self.run_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        self.history_path = self.run_dir / "history.csv"
        self.summary_path = self.run_dir / "summary.json"
        self.started = time.perf_counter()
        self.rows: list[dict] = []
        self.writer = SummaryWriter(str(self.run_dir / "tensorboard")) if SummaryWriter else None

    def log_epoch(self, epoch: int, **metrics: float | None) -> None:
        epoch = int(epoch)
        row = next((existing for existing in reversed(self.rows) if existing["epoch"] == epoch), None)
        if row is None:
            row = {"epoch": epoch, "elapsed_sec": time.perf_counter() - self.started}
            self.rows.append(row)
        for name, value in metrics.items():
            if value is None:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            row[name] = value
            if self.writer:
                self.writer.add_scalar(name.replace("@", "_at_"), value, epoch)
        self._write_history()

    def log_final_metrics(self, metrics_by_k: Mapping[int, Mapping[str, float]], **metadata) -> None:
        created_at = datetime.now().isoformat(timespec="seconds")
        n_users = int(metadata.get("n_users") or 0)
        latency_sec = metadata.get("inference_total_sec")
        latency_ms_per_user = (
            float(latency_sec) * 1000.0 / n_users if latency_sec is not None and n_users else None
        )
        payload = {
            "dataset": self.dataset,
            "model": self.model,
            "run_id": self.run_id,
            "created_at": created_at,
            "run_dir": str(self.run_dir),
            "final_metrics": {str(k): dict(values) for k, values in metrics_by_k.items()},
            "latency_sec": latency_sec,
            "latency_ms_per_user": latency_ms_per_user,
            **metadata,
        }
        self.summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        for k, values in metrics_by_k.items():
            for name, value in values.items():
                if self.writer:
                    self.writer.add_scalar(f"final/{name}_at_{k}", float(value), 0)
        self.plot_metrics_by_k(metrics_by_k)
        if metadata.get("popularity_bias"):
            self.plot_popularity_bias(metadata["popularity_bias"])
        self._update_registry(metrics_by_k, payload)

    def log_validation_selection(
        self, epoch: int, metrics: Mapping[str, float] | None, **metadata,
    ) -> None:
        """Persist the one validation checkpoint selected for final retraining."""
        payload = {
            "dataset": self.dataset,
            "model": self.model,
            "run_id": self.run_id,
            "selected_epoch": int(epoch),
            "selected_metrics": dict(metrics) if metrics is not None else None,
            **metadata,
        }
        path = self.run_dir / "validation_selection.json"
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def plot_training(self) -> None:
        if plt is None or not self.rows:
            return
        series = self._series()
        losses = [name for name in series if "loss" in name.lower()]
        ranking = [
            name for name in (
                "val_recall@10", "val_ndcg@10", "val_mrr@10", "val_coverage@10"
            )
            if name in series
        ]
        if losses:
            self._line_plot(losses, "Training loss", "Loss", self.plot_dir / "loss.png")
        if ranking:
            fig, axes = plt.subplots(1, len(ranking), figsize=(5 * len(ranking), 4), squeeze=False)
            for ax, name in zip(axes[0], ranking):
                x, y = series[name]
                ax.plot(x, y, marker="o", linewidth=1.5)
                ax.set(title=name.replace("val_", "Validation ").upper(), xlabel="Epoch", ylabel=name.split("_")[-1])
                ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(self.plot_dir / "validation_ranking.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    def plot_metrics_by_k(self, metrics_by_k: Mapping[int, Mapping[str, float]]) -> None:
        if plt is None or not metrics_by_k:
            return
        ks = sorted(int(k) for k in metrics_by_k)
        names = [name for name in ("recall", "ndcg", "mrr", "coverage") if any(name in metrics_by_k[k] for k in ks)]
        if not names:
            return
        fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4), squeeze=False)
        for ax, name in zip(axes[0], names):
            values = [metrics_by_k[k].get(name, np.nan) for k in ks]
            ax.plot(ks, values, marker="o")
            ax.set(title=name.upper(), xlabel="K", ylabel=name.capitalize(), xticks=ks)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "metrics_by_k.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_popularity_bias(self, values_by_k: Mapping[int, Mapping[str, float]]) -> None:
        if plt is None:
            return
        ks = sorted(int(k) for k in values_by_k)
        if not ks:
            return
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)
        for ax, name, label in zip(
            axes[0], ("head_exposure", "avg_train_popularity"),
            ("Head exposure", "Average train popularity"),
        ):
            ax.plot(ks, [values_by_k[k][name] for k in ks], marker="o")
            ax.set(title=label, xlabel="K", ylabel=label, xticks=ks)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.plot_dir / "popularity_bias.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    def close(self) -> None:
        self.plot_training()
        if self.writer:
            self.writer.flush()
            self.writer.close()

    def _series(self):
        result = defaultdict(lambda: ([], []))
        for row in self.rows:
            for name, value in row.items():
                if name not in {"epoch", "elapsed_sec"}:
                    result[name][0].append(row["epoch"])
                    result[name][1].append(value)
        return result

    def _line_plot(self, names, title, ylabel, path):
        series = self._series()
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in names:
            x, y = series[name]
            ax.plot(x, y, label=name, linewidth=1.5)
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _write_history(self):
        fields = ["epoch", "elapsed_sec"] + sorted({key for row in self.rows for key in row} - {"epoch", "elapsed_sec"})
        with self.history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)

    def _update_registry(self, metrics_by_k, payload):
        path = registry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            with path.open(newline="", encoding="utf-8") as handle:
                existing = list(csv.DictReader(handle))
        maxlen = payload.get("maxlen")
        key_prefix = (
            normalize_dataset_name(self.dataset), normalize_model_name(self.model),
            "" if maxlen is None else str(int(maxlen)), self.run_id,
        )
        new_rows = []
        for k, values in metrics_by_k.items():
            new_rows.append({
                "dataset": key_prefix[0],
                "model": key_prefix[1],
                "maxlen": key_prefix[2],
                "k": int(k),
                "recall": values.get("recall", ""),
                "ndcg": values.get("ndcg", ""),
                "mrr": values.get("mrr", ""),
                "coverage": values.get("coverage", ""),
                "latency_sec": payload.get("latency_sec", ""),
                "latency_ms_per_user": payload.get("latency_ms_per_user", ""),
                "n_users": payload.get("n_users", ""),
                "seed": payload.get("seed", ""),
                "split": payload.get("split", ""),
                "mask_seen": payload.get("mask_seen", ""),
                "run_id": self.run_id,
                "created_at": payload.get("created_at", ""),
                "summary_path": str(self.summary_path),
            })
        replace_keys = {(row["dataset"], row["model"], row["maxlen"], row["run_id"], str(row["k"])) for row in new_rows}
        existing = [row for row in existing if (
            row.get("dataset"), row.get("model"), row.get("maxlen", ""),
            row.get("run_id"), row.get("k"),
        ) not in replace_keys]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
            writer.writeheader()
            writer.writerows(existing + new_rows)


def popularity_from_sequences(sequences: Iterable[Iterable[int]], padding: int = 0) -> Counter:
    return Counter(int(item) for seq in sequences for item in seq if int(item) != padding)


def save_dataset_popularity(dataset: str, popularity: Mapping[int, int]) -> Path | None:
    if plt is None or not popularity:
        return None
    report_dir = output_root().parent / "reports" / "datasets" / str(dataset)
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = np.asarray(sorted(popularity.values(), reverse=True), dtype=float)
    with (report_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "n_items", "n_interactions", "head_fraction"])
        writer.writerow([dataset, len(counts), int(counts.sum()), 0.1])
    fig, ax = plt.subplots(figsize=(8, 5))
    ranks = np.arange(1, len(counts) + 1)
    ax.plot(ranks, counts)
    head_end = max(1, int(np.ceil(0.1 * len(counts))))
    ax.axvline(head_end, linestyle="--", linewidth=1, label="Top 10% items (head)")
    ax.set(title=f"{dataset}: train item popularity", xlabel="Item rank", ylabel="Train interactions", yscale="log")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = report_dir / "popularity_long_tail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def recommendation_popularity(
    recommendations: Sequence[Sequence[int]], popularity: Mapping[int, int], ks: Sequence[int]
) -> dict[int, dict[str, float]]:
    if not popularity:
        return {}
    ranked = sorted(popularity, key=popularity.get, reverse=True)
    head = set(ranked[: max(1, int(np.ceil(0.1 * len(ranked))))])
    result = {}
    for k in ks:
        items = [int(item) for recs in recommendations for item in recs[: int(k)] if int(item) in popularity]
        result[int(k)] = {
            "head_exposure": sum(item in head for item in items) / len(items) if items else 0.0,
            "avg_train_popularity": float(np.mean([popularity[item] for item in items])) if items else 0.0,
        }
    return result
