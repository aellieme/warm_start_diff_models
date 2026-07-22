#!/usr/bin/env python3
"""Create presentation-ready Bradley-Terry plots from saved CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


KS = (10, 20, 100)


def plot_ranking(ranking: pd.DataFrame, k: int, output_path: Path) -> None:
    required = {"model", "beta_mean", "beta_2.5%", "beta_97.5%"}
    missing = required - set(ranking.columns)
    if missing:
        raise ValueError(f"K={k} ranking is missing columns: {sorted(missing)}")

    ranking = ranking.sort_values("rank").reset_index(drop=True)
    means = ranking["beta_mean"].to_numpy(dtype=float)
    lower = ranking["beta_2.5%"].to_numpy(dtype=float)
    upper = ranking["beta_97.5%"].to_numpy(dtype=float)
    errors = np.vstack((means - lower, upper - means))
    y = np.arange(len(ranking))

    fig, ax = plt.subplots(figsize=(10, 6.5))
    colors = plt.cm.viridis_r(np.linspace(0.12, 0.88, len(ranking)))
    for index in range(len(ranking)):
        ax.errorbar(
            means[index], y[index],
            xerr=errors[:, index].reshape(2, 1),
            fmt="o", markersize=7, capsize=4,
            color=colors[index], ecolor=colors[index], linewidth=1.8,
        )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.65)
    ax.set_yticks(y, ranking["model"])
    ax.invert_yaxis()
    ax.set(
        title=f"Bradley–Terry ranking by Recall@{k}",
        xlabel="Posterior mean ability β (95% credible interval)",
        ylabel="Model",
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_probabilities(
    probabilities: pd.DataFrame,
    k: int,
    output_path: Path,
) -> None:
    if probabilities.shape[0] != probabilities.shape[1]:
        raise ValueError(f"K={k} probability matrix must be square")

    values = probabilities.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(probabilities.columns)), probabilities.columns)
    ax.set_yticks(np.arange(len(probabilities.index)), probabilities.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value < 0.2 or value > 0.8 else "black"
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=7)

    ax.set(
        title=f"Posterior pairwise win probabilities, Recall@{k}",
        xlabel="Opponent model",
        ylabel="Model",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("P(row model beats column model)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def create_plots(input_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for k in KS:
        ranking_path = input_dir / f"combined_k{k}_bt_ranking.csv"
        probability_path = input_dir / f"combined_k{k}_bt_probabilities.csv"
        if not ranking_path.exists() or not probability_path.exists():
            raise FileNotFoundError(
                f"Bradley-Terry CSV files for K={k} were not found in {input_dir}"
            )

        ranking = pd.read_csv(ranking_path)
        probabilities = pd.read_csv(probability_path, index_col=0)
        ranking_plot = output_dir / f"combined_k{k}_bt_ranking.png"
        probability_plot = output_dir / f"combined_k{k}_bt_probabilities.png"
        plot_ranking(ranking, k, ranking_plot)
        plot_probabilities(probabilities, k, probability_plot)
        outputs.extend((ranking_plot, probability_plot))
        print(ranking_plot)
        print(probability_plot)

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/content"))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir
    create_plots(args.input_dir, output_dir)


if __name__ == "__main__":
    main()
