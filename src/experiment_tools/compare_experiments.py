"""Build cross-model plots from compatible local summary.json files."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .experiment_tracking import normalize_dataset_name, output_root
except ImportError:
    from experiment_tracking import normalize_dataset_name, output_root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, default=None)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    summaries = []
    log_roots = (
        [args.logdir]
        if args.logdir is not None
        else list((output_root() / "service_files" / "models").glob("*/*/logs"))
    )
    for log_root in log_roots:
        for path in Path(log_root).glob("*/summary.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            if (
                normalize_dataset_name(data.get("dataset", ""))
                == normalize_dataset_name(args.dataset)
                and data.get("final_metrics")
            ):
                summaries.append(data)
    if not summaries:
        raise SystemExit(f"No completed runs found for {args.dataset!r}")
    signatures = {
        (tuple(sorted(x["final_metrics"])), x.get("split"), x.get("mask_seen"), x.get("seed"))
        for x in summaries
    }
    if len(signatures) != 1:
        raise SystemExit("Runs are not comparable: K, split, mask_seen, or seed differ")
    out = output_root() / "graphics" / "evaluation" / "comparisons" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    names = ("recall", "ndcg", "mrr", "coverage")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4), squeeze=False)
    for run in summaries:
        ks = sorted(int(k) for k in run["final_metrics"])
        for ax, metric in zip(axes[0], names):
            ax.plot(ks, [run["final_metrics"][str(k)].get(metric, float("nan")) for k in ks], marker="o", label=run["model"])
            ax.set(title=metric.upper(), xlabel="K", ylabel=metric.capitalize(), xticks=ks)
            ax.grid(alpha=0.25)
    axes[0][-1].legend()
    fig.tight_layout()
    fig.savefig(out / "metrics_by_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    with (out / "compatible_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "run_dir"])
        writer.writerows((run["model"], run.get("run_dir", "")) for run in summaries)
    popularity_runs = [run for run in summaries if run.get("popularity_bias")]
    if popularity_runs:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
        for run in popularity_runs:
            values = run["popularity_bias"]
            ks = sorted(int(k) for k in values)
            axes[0][0].plot(ks, [values[str(k)]["head_exposure"] for k in ks], marker="o", label=run["model"])
            axes[0][1].plot(ks, [values[str(k)]["avg_train_popularity"] for k in ks], marker="o", label=run["model"])
        for ax, title in zip(axes[0], ("Head exposure", "Average train popularity")):
            ax.set(title=title, xlabel="K", ylabel=title)
            ax.grid(alpha=0.25)
        axes[0][1].legend()
        fig.tight_layout()
        fig.savefig(out / "popularity_bias.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(out / "metrics_by_k.png")


if __name__ == "__main__":
    main()
