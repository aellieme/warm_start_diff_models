from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


BT_KS = (10, 20, 100)
PLOT_NAMES = (
    "loss.png",
    "validation_ranking.png",
    "metrics_by_k.png",
    "popularity_bias.png",
)

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


def safe(value: str) -> str:
    return str(value).replace("/", "-").replace("\\", "-").replace(" ", "_")


def normalize_dataset(value: str) -> str:
    return DATASET_NAMES.get(str(value).lower(), str(value))


def normalize_model(value: str) -> str:
    return MODEL_NAMES.get(str(value).lower(), str(value))


def candidate_result_roots() -> list[Path]:
    candidates: list[Path] = []
    output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR")
    if output_dir:
        candidates.append(Path(output_dir).resolve().parent)
    candidates.append(Path("/content/drive/MyDrive/experiment_results"))
    shortcut_root = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_root.exists():
        candidates.extend(shortcut_root.glob("*/experiment_results"))
    return candidates


def locate_results_root(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if root.exists():
            return root
        raise FileNotFoundError(f"Experiment results folder not found: {root}")
    roots = candidate_result_roots()
    root = next((path for path in roots if (path / "results_registry.csv").exists()), None)
    if root is not None:
        return root
    checked = "\n".join(str(path) for path in roots)
    raise FileNotFoundError("experiment_results was not found. Checked:\n" + checked)


def read_registry(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_latest_runs(rows: list[dict], seed: int = 42) -> dict[tuple[str, str, str], list[dict]]:
    grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for row in rows:
        if str(row.get("seed", "")) != str(seed):
            continue
        key = (row.get("dataset", ""), row.get("model", ""), row.get("maxlen", ""))
        grouped.setdefault(key, {}).setdefault(row.get("run_id", ""), []).append(row)

    selected: dict[tuple[str, str, str], list[dict]] = {}
    for key, runs in grouped.items():
        latest_id = max(
            runs,
            key=lambda run_id: max(
                (item.get("created_at", "") for item in runs[run_id]),
                default="",
            ),
        )
        selected[key] = runs[latest_id]
    return selected


def copy_if_exists(source: Path, destination: Path, copied: list[Path]) -> bool:
    if not source.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied.append(destination)
    return True


def copy_regular_tables(results_root: Path, output_root: Path, copied: list[Path]) -> None:
    source = results_root / "reports" / "tables"
    destination = output_root / "01_ordinary_results" / "tables"
    for name in ("experimental_results.xlsx", "experimental_results.md"):
        copy_if_exists(source / name, destination / name, copied)


def copy_latest_run_plots(
    results_root: Path,
    output_root: Path,
    copied: list[Path],
    notes: list[str],
) -> None:
    registry = results_root / "results_registry.csv"
    latest = select_latest_runs(read_registry(registry))
    summaries: dict[tuple[str, str, str], Path] = {
        key: Path(run_rows[0].get("summary_path", ""))
        for key, run_rows in latest.items()
    }

    # A run directory may still exist even when its registry rows were lost.
    # Include the newest such summary so available ordinary graphs are not omitted.
    discovered: dict[tuple[str, str, str], tuple[str, Path]] = {}
    for path in (results_root / "logs").rglob("summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(payload.get("seed", "")) != "42":
            continue
        maxlen = payload.get("maxlen")
        maxlen_key = "" if maxlen in (None, "") else str(int(maxlen))
        key = (
            normalize_dataset(payload.get("dataset", "")),
            normalize_model(payload.get("model", "")),
            maxlen_key,
        )
        created_at = str(payload.get("created_at", ""))
        if key not in discovered or created_at > discovered[key][0]:
            discovered[key] = (created_at, path)

    for key, (_, path) in discovered.items():
        if key not in summaries:
            summaries[key] = path
            notes.append(
                "Recovered plots from an unregistered run: "
                f"{key[0]} / {key[1]} / maxlen={key[2]}"
            )

    graph_root = output_root / "01_ordinary_results" / "graphs" / "latest_runs"
    summary_root = output_root / "03_reproducibility" / "latest_run_summaries"

    for (dataset, model, maxlen), summary in sorted(summaries.items()):
        label = safe(model) + (f"_maxlen{maxlen}" if maxlen else "")
        relative = Path(safe(dataset)) / label
        if not summary.exists():
            notes.append(f"Missing summary: {dataset} / {model} / maxlen={maxlen}")
            continue

        run_dir = summary.parent
        copy_if_exists(summary, summary_root / relative / "summary.json", copied)
        copy_if_exists(run_dir / "history.csv", summary_root / relative / "history.csv", copied)
        found_plot = False
        for name in PLOT_NAMES:
            found_plot |= copy_if_exists(
                run_dir / "plots" / name,
                graph_root / relative / name,
                copied,
            )
        if not found_plot:
            notes.append(f"No run plots: {dataset} / {model} / maxlen={maxlen}")


def copy_report_graphs(results_root: Path, output_root: Path, copied: list[Path]) -> None:
    report_root = results_root / "reports"
    destination = output_root / "01_ordinary_results" / "graphs" / "reports"
    for subfolder in ("comparisons", "datasets"):
        source = report_root / subfolder
        if not source.exists():
            continue
        for path in source.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".png", ".csv"}:
                copy_if_exists(path, destination / subfolder / path.relative_to(source), copied)


def find_bt_file(name: str, bt_source: Path, results_root: Path) -> Path | None:
    candidates = (
        bt_source / name,
        results_root / "reports" / "bradley_terry" / "tables" / name,
        results_root / "reports" / "bradley_terry" / "graphs" / name,
    )
    return next((path for path in candidates if path.exists()), None)


def copy_bt_results(
    bt_source: Path,
    results_root: Path,
    output_root: Path,
    copied: list[Path],
    notes: list[str],
) -> None:
    table_root = output_root / "02_bradley_terry" / "tables"
    graph_root = output_root / "02_bradley_terry" / "graphs"
    for k in BT_KS:
        table_names = (
            f"combined_k{k}.csv",
            f"combined_k{k}_bt_ranking.csv",
            f"combined_k{k}_bt_probabilities.csv",
            f"combined_k{k}_win_matrix.csv",
        )
        graph_names = (
            f"combined_k{k}_bt_ranking.png",
            f"combined_k{k}_bt_probabilities.png",
        )
        for name in table_names:
            source = find_bt_file(name, bt_source, results_root)
            if source is None:
                notes.append(f"Missing Bradley-Terry table: {name}")
            else:
                copy_if_exists(source, table_root / name, copied)
        for name in graph_names:
            source = find_bt_file(name, bt_source, results_root)
            if source is None:
                notes.append(f"Missing Bradley-Terry graph: {name}")
            else:
                copy_if_exists(source, graph_root / name, copied)


def copy_reproducibility_files(
    repo_root: Path,
    results_root: Path,
    output_root: Path,
    copied: list[Path],
) -> None:
    destination = output_root / "03_reproducibility"
    copy_if_exists(
        results_root / "results_registry.csv",
        destination / "results_registry.csv",
        copied,
    )
    method_dir = destination / "method"
    for name in (
        "prepare_bt_inputs.py",
        "paper_bt.py",
        "plot_bt_results.py",
        "prepare_folder.py",
        "Bradly_Terry_demo.ipynb",
    ):
        copy_if_exists(repo_root / name, method_dir / name, copied)


def write_readme(output_root: Path, notes: list[str], copied_count: int) -> None:
    text = f"""РЕЗУЛЬТАТЫ ДИПЛОМНЫХ ЭКСПЕРИМЕНТОВ

01_ordinary_results
  Итоговая книга с 9 таблицами и графики последних запусков моделей.

02_bradley_terry
  Рейтинги, матрицы вероятностей и графики Bradley–Terry для K=10, 20, 100.

03_reproducibility
  Реестр экспериментов, summary/history последних запусков и код анализа.

Протокол: global temporal split 70/10/20; финальное обучение train+validation;
оценка на test; просмотренные items маскируются; seed=42; K=10,20,100.

Примечание: отсутствовавшие Recall ADRec maxlen=100 для Amazon Baby были
восстановлены только для Bradley–Terry из ранее сохранённых таблиц.

Скопировано файлов: {copied_count}
"""
    if notes:
        text += "\nПРЕДУПРЕЖДЕНИЯ\n" + "\n".join(f"- {note}" for note in notes) + "\n"
    (output_root / "00_README.txt").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--bt-source", type=Path, default=Path("/content"))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    results_root = locate_results_root(args.results_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output or (results_root / f"results_package_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=False)

    copied: list[Path] = []
    notes: list[str] = []
    copy_regular_tables(results_root, output_root, copied)
    copy_latest_run_plots(results_root, output_root, copied, notes)
    copy_report_graphs(results_root, output_root, copied)
    copy_bt_results(args.bt_source, results_root, output_root, copied, notes)
    copy_reproducibility_files(repo_root, results_root, output_root, copied)
    write_readme(output_root, notes, len(copied))

    print(f"Results package created: {output_root}")
    print(f"Copied files: {len(copied)}")
    if notes:
        print("Warnings:")
        for note in notes:
            print(f"- {note}")


if __name__ == "__main__":
    main()
