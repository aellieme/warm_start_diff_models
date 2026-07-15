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


REPO_ROOT = Path(__file__).resolve().parent.parent


def output_root() -> Path:
    return Path(os.environ.get("EXPERIMENT_OUTPUT_DIR", REPO_ROOT / "logs")).resolve()


def make_run_dir(dataset: str, model: str, run_id: str | None = None) -> Path:
    safe = lambda value: str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_root() / safe(dataset) / safe(model) / safe(run_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


class ExperimentTracker:
    """Append-only scalar history plus inexpensive local plots."""

    def __init__(self, dataset: str, model: str, run_id: str | None = None):
        self.dataset = str(dataset)
        self.model = str(model)
        self.run_dir = make_run_dir(dataset, model, run_id)
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
        payload = {
            "dataset": self.dataset,
            "model": self.model,
            "run_dir": str(self.run_dir),
            "final_metrics": {str(k): dict(values) for k, values in metrics_by_k.items()},
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

    def plot_training(self) -> None:
        if plt is None or not self.rows:
            return
        series = self._series()
        losses = [name for name in series if "loss" in name.lower()]
        ranking = [name for name in ("val_recall@10", "val_ndcg@10", "val_mrr@10") if name in series]
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
