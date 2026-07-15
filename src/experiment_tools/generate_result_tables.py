"""Generate the nine fixed thesis tables from the accumulated run registry."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

try:
    from .experiment_tracking import output_root, registry_path
except ImportError:  # Allow direct execution: python src/experiment_tools/generate_result_tables.py
    from experiment_tracking import output_root, registry_path


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


def select_latest_runs(rows: list[dict], seed: int = 42) -> dict[tuple[str, str, str], list[dict]]:
    eligible = [row for row in rows if str(row.get("seed", "")) == str(seed)]
    grouped: dict[tuple[str, str, str], dict[str, list[dict]]] = {}
    for row in eligible:
        key = (row.get("dataset", ""), row.get("model", ""), row.get("maxlen", ""))
        grouped.setdefault(key, {}).setdefault(row.get("run_id", ""), []).append(row)
    selected = {}
    for key, runs in grouped.items():
        latest_id = max(
            runs,
            key=lambda run_id: max((item.get("created_at", "") for item in runs[run_id]), default=""),
        )
        selected[key] = runs[latest_id]
    return selected


def build_tables(rows: list[dict], seed: int = 42):
    selected = select_latest_runs(rows, seed)
    tables = []
    number = 1
    for dataset in DATASETS:
        for k in KS:
            body = []
            for model, maxlen in ROWS:
                run_rows = selected.get((dataset, model, maxlen), [])
                result = next((row for row in run_rows if row.get("k") == str(k)), None)
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
    destination = args.output_dir or (output_root().parent / "reports" / "tables")
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
