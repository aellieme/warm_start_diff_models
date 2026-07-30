from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd


DATASETS = ("ML-1M", "Amazon Toys", "Amazon Baby")
KS = (10, 20, 100)

# Recall values from the nine legacy tables supplied for ADRec recovery.
ADREC_RECALL_FALLBACK = {
    ("ML-1M", 10, 50): 0.146943,
    ("ML-1M", 10, 100): 0.163208,
    ("ML-1M", 20, 50): 0.218732,
    ("ML-1M", 20, 100): 0.240606,
    ("ML-1M", 100, 50): 0.440269,
    ("ML-1M", 100, 100): 0.465508,
    ("Amazon Toys", 10, 50): 0.026932,
    ("Amazon Toys", 10, 100): 0.035536,
    ("Amazon Toys", 20, 50): 0.037661,
    ("Amazon Toys", 20, 100): 0.048364,
    ("Amazon Toys", 100, 50): 0.081125,
    ("Amazon Toys", 100, 100): 0.101219,
    ("Amazon Baby", 10, 50): 0.010807,
    ("Amazon Baby", 10, 100): 0.015001,
    ("Amazon Baby", 20, 50): 0.019661,
    ("Amazon Baby", 20, 100): 0.024342,
    ("Amazon Baby", 100, 50): 0.064453,
    ("Amazon Baby", 100, 100): 0.075007,
}


def candidate_workbooks() -> list[Path]:
    """Return likely Drive locations in priority order."""
    candidates: list[Path] = []
    output_dir = os.environ.get("EXPERIMENT_OUTPUT_DIR")
    if output_dir:
        results_root = Path(output_dir).resolve().parent
        candidates.append(
            results_root / "reports" / "tables" / "experimental_results.xlsx"
        )

    candidates.append(
        Path(
            "/content/drive/MyDrive/experiment_results/reports/tables/"
            "experimental_results.xlsx"
        )
    )

    shortcut_root = Path("/content/drive/.shortcut-targets-by-id")
    if shortcut_root.exists():
        candidates.extend(
            shortcut_root.glob(
                "*/experiment_results/reports/tables/experimental_results.xlsx"
            )
        )
    return candidates


def locate_workbook(explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Result workbook not found: {path}")

    candidates = candidate_workbooks()
    workbook = next((path for path in candidates if path.exists()), None)
    if workbook is not None:
        return workbook

    checked = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "experimental_results.xlsx was not found. Checked paths:\n" + checked
    )


def normalize_maxlen(value) -> int | None:
    if pd.isna(value) or str(value).strip() == "-":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def model_name(row: pd.Series) -> str:
    maxlen = normalize_maxlen(row["Maxlen"])
    return str(row["Model"]) if maxlen is None else f"{row['Model']} {maxlen}"


def read_sheet(
    workbook: Path,
    dataset: str,
    k: int,
    allow_adrec_fallback: bool,
) -> pd.DataFrame:
    sheet = f"{dataset}_k{k}"
    table = pd.read_excel(workbook, sheet_name=sheet, header=1)
    required = {"Model", "Maxlen", "Recall"}
    missing_columns = required - set(table.columns)
    if missing_columns:
        raise ValueError(f"{sheet}: missing columns {sorted(missing_columns)}")
    if len(table) != 11:
        raise ValueError(f"{sheet}: expected 11 model rows, found {len(table)}")

    scores = pd.to_numeric(table["Recall"], errors="coerce")
    unresolved: list[str] = []
    for index in table.index[scores.isna()]:
        model = str(table.at[index, "Model"]).strip()
        maxlen_raw = table.at[index, "Maxlen"]
        maxlen = normalize_maxlen(maxlen_raw)
        fallback_key = (dataset, k, maxlen)

        if (
            allow_adrec_fallback
            and model == "ADRec"
            and fallback_key in ADREC_RECALL_FALLBACK
        ):
            scores.at[index] = ADREC_RECALL_FALLBACK[fallback_key]
            print(
                f"WARNING: {sheet}: ADRec maxlen={maxlen} "
                "was restored from the legacy table"
            )
        else:
            unresolved.append(f"{model} maxlen={maxlen_raw}")

    if unresolved:
        raise ValueError(
            f"{sheet}: missing Recall without an allowed fallback: "
            + ", ".join(unresolved)
        )

    return pd.DataFrame(
        {
            "dataset": dataset,
            "model": table.apply(model_name, axis=1),
            "score": scores,
        }
    )


def prepare_inputs(
    workbook: Path,
    output_dir: Path,
    allow_adrec_fallback: bool = True,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    for k in KS:
        parts = [
            read_sheet(workbook, dataset, k, allow_adrec_fallback)
            for dataset in DATASETS
        ]
        combined = pd.concat(parts, ignore_index=True)
        if len(combined) != 33:
            raise ValueError(f"k={k}: expected 33 rows, found {len(combined)}")
        if combined.duplicated(["dataset", "model"]).any():
            raise ValueError(f"k={k}: duplicate dataset/model rows found")

        output_path = output_dir / f"combined_k{k}.csv"
        combined.to_csv(output_path, index=False)
        outputs.append(output_path)
        print(f"Saved {output_path} ({len(combined)} rows)")

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content") if Path("/content").exists() else Path("."),
    )
    parser.add_argument(
        "--no-adrec-fallback",
        action="store_true",
        help="Fail instead of restoring missing ADRec Recall values.",
    )
    args = parser.parse_args()

    workbook = locate_workbook(args.input)
    print(f"Using result workbook: {workbook}")
    prepare_inputs(
        workbook,
        args.output_dir,
        allow_adrec_fallback=not args.no_adrec_fallback,
    )
    print("All nine sheets are ready for Bradley-Terry.")


if __name__ == "__main__":
    main()
