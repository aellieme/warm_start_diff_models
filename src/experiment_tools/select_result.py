"""Find experiment runs and explicitly select results for thesis tables."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from .experiment_tracking import registry_path
except ImportError:
    from experiment_tracking import registry_path


DEFAULT_FIELDS = (
    "run_id", "selected", "created_at", "model", "dataset", "maxlen",
    "epochs", "run_type", "recall@10", "ndcg@10", "checkpoint",
)
TRUE_VALUES = {"true", "1", "yes"}
FALSE_VALUES = {"false", "0", "no"}


def _read_registry(path: Path) -> tuple[list[dict], list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Result registry not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _write_registry(path: Path, rows: Sequence[Mapping], fields: Sequence[str]) -> None:
    fieldnames = list(fields)
    if "selected" not in fieldnames:
        fieldnames.append("selected")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _selection_value(value) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return None


def _flatten(value, prefix: str = "") -> dict:
    result = {}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(nested, name))
    elif isinstance(value, (str, int, float, bool)) or value is None:
        result[prefix] = value
    return result


def _history_info(summary_path: str) -> dict:
    if not summary_path:
        return {}
    history_path = Path(summary_path).parent / "history.csv"
    if not history_path.exists():
        return {}
    with history_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metric_names = {name for row in rows for name in row}
    has_loss = any("loss" in name.lower() for name in metric_names)
    return {
        "epochs": len({row.get("epoch", "") for row in rows if row.get("epoch", "") != ""}),
        "run_type": "training" if has_loss else "evaluation",
    }


def _summary_info(summary_path: str) -> dict:
    if not summary_path:
        return {}
    path = Path(summary_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    flattened = _flatten(payload)
    run_type = payload.get("run_type")
    if run_type is None and payload.get("checkpoint"):
        run_type = "inference"
    if run_type is not None:
        flattened["run_type"] = run_type
    config_path = path.parent / "config.json"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            flattened.update(_flatten(config, "config"))
        except (OSError, json.JSONDecodeError):
            pass
    return flattened


def load_experiments(registry: Path | None = None) -> list[dict]:
    path = Path(registry) if registry is not None else registry_path()
    rows, _ = _read_registry(path)
    grouped: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in rows:
        key = (
            row.get("dataset", ""), row.get("model", ""), row.get("maxlen", ""),
            row.get("seed", ""), row.get("run_id", ""),
        )
        grouped.setdefault(key, []).append(row)

    experiments = []
    for run_rows in grouped.values():
        first = dict(run_rows[0])
        experiment = {
            key: value for key, value in first.items()
            if key not in {"k", "recall", "ndcg", "mrr", "coverage"}
        }
        values = [_selection_value(row.get("selected")) for row in run_rows]
        experiment["selected"] = (
            True if True in values else False if False in values else None
        )
        for row in run_rows:
            k = row.get("k", "")
            for metric in ("recall", "ndcg", "mrr", "coverage"):
                if row.get(metric, "") != "":
                    experiment[f"{metric}@{k}"] = row[metric]
        summary = _summary_info(first.get("summary_path", ""))
        history = _history_info(first.get("summary_path", ""))
        for key, value in summary.items():
            experiment.setdefault(key, value)
        experiment.update(history)
        if "run_type" not in experiment:
            experiment["run_type"] = "evaluation"
        experiments.append(experiment)
    return experiments


def _get_value(experiment: Mapping, field: str):
    aliases = {"date": "created_at", "max_len": "maxlen"}
    requested = aliases.get(field, field)
    if requested in experiment:
        return experiment[requested]
    lowered = requested.lower()
    exact = [key for key in experiment if key.lower() == lowered]
    if exact:
        return experiment[exact[0]]
    suffix = [key for key in experiment if key.lower().endswith(f".{lowered}")]
    if len(suffix) == 1:
        return experiment[suffix[0]]
    return None


def _matches(actual, expected, field: str) -> bool:
    if actual is None:
        return False
    actual_text = str(actual).strip()
    expected_text = str(expected).strip()
    if field in {"date", "created_at"}:
        return actual_text.startswith(expected_text)
    try:
        return float(actual_text) == float(expected_text)
    except ValueError:
        return actual_text.casefold() == expected_text.casefold()


def print_experiments(
    experiments: Iterable[Mapping],
    display_fields: Sequence[str] | None = None,
) -> None:
    fields = tuple(display_fields or DEFAULT_FIELDS)
    found = False
    for experiment in experiments:
        found = True
        print(" | ".join(
            f"{field}={_get_value(experiment, field)}" for field in fields
        ))
    if not found:
        print("No matching experiments found.")


def find_experiments(
    filters: Mapping[str, object] | None = None,
    display_fields: Sequence[str] | None = None,
    registry: Path | None = None,
    print_results: bool = True,
) -> list[dict]:
    filters = {key: value for key, value in (filters or {}).items() if value is not None}
    experiments = [
        experiment for experiment in load_experiments(registry)
        if all(
            _matches(_get_value(experiment, field), expected, field)
            for field, expected in filters.items()
        )
    ]
    experiments.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    if print_results:
        print_experiments(experiments, display_fields)
    return experiments


def get_last_training_run(
    display_fields: Sequence[str] | None = None,
    registry: Path | None = None,
    print_result: bool = True,
) -> dict | None:
    training = [
        experiment for experiment in load_experiments(registry)
        if str(experiment.get("run_type", "")).lower() in {"training", "final_training"}
    ]
    latest = max(training, key=lambda item: str(item.get("created_at", "")), default=None)
    if print_result:
        print_experiments([latest] if latest else [], display_fields)
    return latest


def _resolve_run(
    run_id: str,
    experiments: Sequence[dict],
    model: str | None = None,
    dataset: str | None = None,
    maxlen: int | str | None = None,
) -> dict:
    filters = {"model": model, "dataset": dataset, "maxlen": maxlen}
    matches = [
        experiment for experiment in experiments
        if str(experiment.get("run_id", "")) == str(run_id)
        and all(
            expected is None or _matches(_get_value(experiment, field), expected, field)
            for field, expected in filters.items()
        )
    ]
    if not matches:
        raise ValueError(f"Run not found: {run_id}")
    if len(matches) > 1:
        raise ValueError(
            f"Run ID {run_id} is ambiguous. Add model, dataset or maxlen."
        )
    return matches[0]


def do_selected(
    run_id: str,
    registry: Path | None = None,
    model: str | None = None,
    dataset: str | None = None,
    maxlen: int | str | None = None,
) -> dict:
    path = Path(registry) if registry is not None else registry_path()
    rows, fields = _read_registry(path)
    selected = _resolve_run(run_id, load_experiments(path), model, dataset, maxlen)
    scope = (
        selected.get("dataset", ""), selected.get("model", ""),
        selected.get("maxlen", ""), selected.get("seed", ""),
    )
    for row in rows:
        row_scope = (
            row.get("dataset", ""), row.get("model", ""),
            row.get("maxlen", ""), row.get("seed", ""),
        )
        if row_scope == scope:
            row["selected"] = (
                "true" if row.get("run_id", "") == selected["run_id"] else "false"
            )
    _write_registry(path, rows, fields)
    selected["selected"] = True
    return selected


def remove_selected(
    run_id: str,
    registry: Path | None = None,
    model: str | None = None,
    dataset: str | None = None,
    maxlen: int | str | None = None,
) -> dict:
    path = Path(registry) if registry is not None else registry_path()
    rows, fields = _read_registry(path)
    selected = _resolve_run(run_id, load_experiments(path), model, dataset, maxlen)
    for row in rows:
        if (
            row.get("dataset", "") == selected.get("dataset", "")
            and row.get("model", "") == selected.get("model", "")
            and row.get("maxlen", "") == selected.get("maxlen", "")
            and row.get("seed", "") == selected.get("seed", "")
            and row.get("run_id", "") == selected.get("run_id", "")
        ):
            row["selected"] = "false"
    _write_registry(path, rows, fields)
    selected["selected"] = False
    return selected


def _fields(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(field.strip() for field in value.split(",") if field.strip())


def _where(values: Sequence[str]) -> dict:
    filters = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Filter must have the form field=value: {value}")
        field, expected = value.split("=", 1)
        filters[field.strip()] = expected.strip()
    return filters


def _confirm(action: str, experiment: Mapping, fields: Sequence[str] | None) -> bool:
    print_experiments([experiment], fields)
    answer = input(f"{action}? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    select_last = commands.add_parser("select_last")
    select_last.add_argument("--show")
    last = commands.add_parser("last")
    last.add_argument("--show")

    find = commands.add_parser("find")
    find.add_argument("--model")
    find.add_argument("--dataset")
    find.add_argument("--maxlen")
    find.add_argument("--epochs")
    find.add_argument("--date")
    find.add_argument("--where", action="append", default=[])
    find.add_argument("--show")

    for name in ("select", "unselect"):
        command = commands.add_parser(name)
        command.add_argument("run_id")
        command.add_argument("--model")
        command.add_argument("--dataset")
        command.add_argument("--maxlen")
        command.add_argument("--show")
        command.add_argument("--yes", action="store_true")

    args = parser.parse_args()
    if args.command in {"select_last", "last"}:
        get_last_training_run(_fields(args.show), args.registry)
        return
    if args.command == "find":
        filters = _where(args.where)
        filters.update({
            "model": args.model, "dataset": args.dataset, "maxlen": args.maxlen,
            "epochs": args.epochs, "date": args.date,
        })
        find_experiments(filters, _fields(args.show), args.registry)
        return

    experiments = load_experiments(args.registry)
    experiment = _resolve_run(
        args.run_id, experiments, args.model, args.dataset, args.maxlen
    )
    action = "Select this experiment" if args.command == "select" else "Unselect this experiment"
    if not args.yes and not _confirm(action, experiment, _fields(args.show)):
        print("Cancelled.")
        return
    if args.command == "select":
        do_selected(
            args.run_id, args.registry, args.model, args.dataset, args.maxlen
        )
        print(f"Selected run: {args.run_id}")
    else:
        remove_selected(
            args.run_id, args.registry, args.model, args.dataset, args.maxlen
        )
        print(f"Unselected run: {args.run_id}")


if __name__ == "__main__":
    main()
