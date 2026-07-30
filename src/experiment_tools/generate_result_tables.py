"""Generate the nine fixed thesis tables from the accumulated run registry."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

try:
    from .experiment_tracking import (
        normalize_dataset_name, normalize_model_name, output_root, registry_path,
    )
except ImportError:  # Allow direct execution: python src/experiment_tools/generate_result_tables.py
    from experiment_tracking import (
        normalize_dataset_name, normalize_model_name, output_root, registry_path,
    )


DATASETS = ("ML-1M", "Amazon Toys", "Amazon Baby")
KS = (10, 20, 100)
ROWS = (
    ("ADRec", "50"), ("DiffuRec", "50"), ("SASRec", "50"), ("GPTRec", "50"),
    ("ADRec", "100"), ("DiffuRec", "100"), ("SASRec", "100"), ("GPTRec", "100"),
    ("T-DiffRec", ""), ("TopPopular", ""), ("Random", ""),
)
METRICS = ("recall", "ndcg", "mrr", "coverage", "latency_sec")


def read_registry(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Result registry not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection_value(value) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    return None


def select_runs(rows: list[dict], seed: int = 42) -> dict[tuple[str, str, str], list[dict]]:
    eligible = [row for row in rows if str(row.get("seed", "")) == str(seed)]
    grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for row in eligible:
        key = (
            normalize_dataset_name(row.get("dataset", "")),
            normalize_model_name(row.get("model", "")),
            row.get("maxlen", ""),
        )
        grouped.setdefault(key, {}).setdefault(row.get("run_id", ""), []).append(row)
    selected = {}
    for key, runs in grouped.items():
        selected_ids = [
            run_id for run_id, run_rows in runs.items()
            if any(_selection_value(row.get("selected")) is True for row in run_rows)
        ]
        if len(selected_ids) > 1:
            raise ValueError(
                f"Several runs are selected for dataset={key[0]}, model={key[1]}, "
                f"maxlen={key[2]}: {selected_ids}"
            )
        if selected_ids:
            selected[key] = runs[selected_ids[0]]
            continue
        managed = any(
            _selection_value(row.get("selected")) is False
            for run_rows in runs.values()
            for row in run_rows
        )
        if not managed:
            latest_id = max(
                runs,
                key=lambda run_id: max(
                    (item.get("created_at", "") for item in runs[run_id]), default=""
                ),
            )
            selected[key] = runs[latest_id]
    return selected


def build_tables(rows: list[dict], seed: int = 42):
    selected = select_runs(rows, seed)
    tables = []
    number = 1
    for dataset in DATASETS:
        for k in KS:
            body = []
            for model, maxlen in ROWS:
                run_rows = selected.get((dataset, model, maxlen), [])
                result = _result_for_k(run_rows, k)
                body.append({
                    "Model": model,
                    "Maxlen": maxlen or "-",
                    "Recall": _format(result, "recall", 6),
                    "NDCG": _format(result, "ndcg", 6),
                    "MRR": _format(result, "mrr", 6),
                    "Coverage": _format(result, "coverage", 6),
                    "Latency": _format(result, "latency_sec", 4),
                })
            tables.append({"number": number, "dataset": dataset, "k": k, "rows": body})
            number += 1
    return tables


def _result_for_k(run_rows: list[dict], k: int) -> dict | None:
    result = next((row for row in run_rows if row.get("k") == str(k)), None)
    if result is not None:
        return result
    if not run_rows:
        return None
    wide = run_rows[0]
    if not any(wide.get(f"{metric}@{k}", "") not in ("", None) for metric in METRICS):
        return None
    return {
        "recall": wide.get(f"recall@{k}", ""),
        "ndcg": wide.get(f"ndcg@{k}", ""),
        "mrr": wide.get(f"mrr@{k}", ""),
        "coverage": wide.get(f"coverage@{k}", ""),
        "latency_sec": wide.get("latency_sec", ""),
    }


def _format(row: dict | None, key: str, digits: int) -> str:
    if not row or row.get(key, "") in ("", None):
        return "—"
    return f"{float(row[key]):.{digits}f}"


def write_markdown(tables, path: Path):
    lines = []
    for table in tables:
        lines.extend([
            f"### Таблица {table['number']}: {table['dataset']} | k = {table['k']}",
            "",
            "| Model | Maxlen | Recall | NDCG | MRR | Coverage | Latency |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for row in table["rows"]:
            lines.append("| " + " | ".join(str(row[key]) for key in (
                "Model", "Maxlen", "Recall", "NDCG", "MRR", "Coverage", "Latency"
            )) + " |")
        lines.extend(["", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_excel(tables, path: Path):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise RuntimeError("Install src/visualization_requirements.txt to create Excel tables") from exc
    workbook = Workbook()
    workbook.remove(workbook.active)
    headers = ("Model", "Maxlen", "Recall", "NDCG", "MRR", "Coverage", "Latency")
    for table in tables:
        sheet = workbook.create_sheet(f"{table['dataset']}_k{table['k']}"[:31])
        sheet.append([f"Таблица {table['number']}: {table['dataset']} | k = {table['k']}"])
        sheet["A1"].font = Font(bold=True)
        sheet.append(list(headers))
        for cell in sheet[2]:
            cell.font = Font(bold=True)
        for row in table["rows"]:
            sheet.append([row[key] for key in headers])
        sheet.freeze_panes = "A3"
        for column, width in zip("ABCDEFG", (16, 10, 13, 13, 13, 13, 13)):
            sheet.column_dimensions[column].width = width
    workbook.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = args.registry or registry_path()
    destination = args.output_dir or (output_root() / "experiment_tables")
    destination.mkdir(parents=True, exist_ok=True)
    tables = build_tables(read_registry(source), args.seed)
    markdown_path = destination / "experimental_results.md"
    excel_path = destination / "experimental_results.xlsx"
    write_markdown(tables, markdown_path)
    write_excel(tables, excel_path)
    print(markdown_path)
    print(excel_path)


if __name__ == "__main__":
    main()
