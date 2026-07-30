"""Copy legacy experiment results into the normalized directory layout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path


DATASET_ALIASES = {
    "baby": "amazon_baby",
    "amazon_baby": "amazon_baby",
    "amazonbaby": "amazon_baby",
    "toys": "amazon_toys",
    "amazon_toys": "amazon_toys",
    "amazon_toys_and_games": "amazon_toys",
    "amazontoys": "amazon_toys",
    "amazontoysandgames": "amazon_toys",
    "ml_1m": "ml-1m",
    "ml-1m": "ml-1m",
}

MODEL_ALIASES = {
    "adrec": "ADRec",
    "diffurec": "DiffuRec",
    "t_diffrec": "T-DiffRec",
    "t-diffrec": "T-DiffRec",
    "tdiffrec": "T-DiffRec",
    "sasrec": "SASRec",
    "gpt_2": "GPTRec",
    "gpt-2": "GPTRec",
    "gptrec": "GPTRec",
    "toppopular": "TopPopular",
    "random": "Random",
    "randomrecs": "Random",
}

RUN_FIELDS = [
    "run_id", "selected", "created_at", "run_type", "model", "dataset", "maxlen",
    "epochs", "seed", "split", "mask_seen", "ranking_protocol", "batch_size",
    "hidden_size", "embedding_dim", "num_layers", "num_heads", "dropout",
    "learning_rate", "weight_decay", "optimizer", "scheduler", "patience",
    "diffusion_steps", "sampling_steps", "noise_schedule", "loss", "loss_lambda",
    "recall@10", "ndcg@10", "mrr@10", "coverage@10",
    "recall@20", "ndcg@20", "mrr@20", "coverage@20",
    "recall@100", "ndcg@100", "mrr@100", "coverage@100",
    "latency_sec", "latency_ms_per_user", "n_users", "checkpoint", "summary_path",
]

CHECKPOINT_FIELDS = [
    "checkpoint", "metadata", "model", "dataset", "maxlen", "seed", "type",
    "created_at", "size_bytes", "sha256", "epochs", "batch_size", "hidden_size",
    "embedding_dim", "num_layers", "num_heads", "dropout", "learning_rate",
    "weight_decay", "optimizer", "scheduler", "patience", "diffusion_steps",
    "sampling_steps", "noise_schedule", "loss", "loss_lambda",
]

SKIP_CONFIG_KEYS = {
    "coverage_candidate_items", "train_item_popularity", "experiment_tracker",
    "data_index", "data_description",
}


def canonical_dataset(value) -> str:
    text = str(value or "").strip()
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return DATASET_ALIASES.get(key, text or "unknown")


def canonical_model(value) -> str:
    text = str(value or "").strip()
    key = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return MODEL_ALIASES.get(key, text or "unknown")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value):
    if isinstance(value, dict):
        return {
            str(key): json_value(nested)
            for key, nested in value.items()
            if str(key) not in SKIP_CONFIG_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(json_value(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def maxlen_folder(value) -> str:
    if value in (None, "", "None"):
        return "maxlen_not_applicable"
    if str(value).lower() == "unknown":
        return "maxlen_unknown"
    return f"maxlen_{int(float(value))}"


def normalize_maxlen(value):
    if value in (None, "", "None"):
        return ""
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "unknown"


class Migration:
    def __init__(self, source: Path, destination: Path):
        self.source = source.resolve()
        self.destination = destination.resolve()
        self.manifest: list[dict] = []
        self.destination_hashes: dict[str, str] = {}
        self.summary_destinations: dict[str, str] = {}
        self.checkpoint_destinations: dict[str, str] = {}
        self.checkpoint_rows: list[dict] = []
        self.legacy_rows: list[dict] = []
        self.run_records: dict[tuple[str, str, str, str], dict] = {}
        self.selected_runs: dict[tuple[str, str, str], str] = {}
        self.registry_maxlen: dict[tuple[str, str, str], str] = {}

    def relative_source(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.source))

    def relative_destination(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.destination))

    def record(
        self, source: Path, destination: Path, category: str, status: str,
        source_hash: str | None = None, destination_hash: str | None = None,
    ) -> None:
        self.manifest.append({
            "source": self.relative_source(source),
            "destination": self.relative_destination(destination),
            "category": category,
            "status": status,
            "source_sha256": source_hash or sha256(source),
            "destination_sha256": destination_hash or (
                sha256(destination) if destination.exists() else ""
            ),
        })

    def copy(self, source: Path, destination: Path, category: str) -> Path:
        source_hash = sha256(source)
        if destination.exists():
            destination_hash = sha256(destination)
            if source_hash == destination_hash:
                self.record(
                    source, destination, category, "duplicate", source_hash, destination_hash
                )
                return destination
            destination = destination.with_name(
                f"{destination.stem}__{source_hash[:8]}{destination.suffix}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        destination_hash = sha256(destination)
        if source_hash != destination_hash:
            raise RuntimeError(f"Checksum mismatch after copying {source}")
        self.destination_hashes[source_hash] = self.relative_destination(destination)
        self.record(source, destination, category, "copied", source_hash, destination_hash)
        return destination

    def copy_unique(self, source: Path, destination: Path, category: str) -> Path:
        source_hash = sha256(source)
        existing = self.destination_hashes.get(source_hash)
        if existing:
            existing_path = self.destination / existing
            self.record(
                source, existing_path, category, "duplicate_content",
                source_hash, source_hash,
            )
            return existing_path
        return self.copy(source, destination, category)

    def transform_json(
        self, source: Path, destination: Path, payload: dict, category: str
    ) -> Path:
        write_json(destination, payload)
        self.record(source, destination, category, "transformed")
        return destination

    def load_legacy_registries(self) -> None:
        for path in sorted(self.source.glob("results_registry*.csv")):
            with path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    normalized = dict(row)
                    normalized["dataset"] = canonical_dataset(row.get("dataset"))
                    normalized["model"] = canonical_model(row.get("model"))
                    normalized["maxlen"] = normalize_maxlen(row.get("maxlen"))
                    self.legacy_rows.append(normalized)
                    key = (
                        normalized["dataset"], normalized["model"],
                        normalized.get("run_id", ""),
                    )
                    value = normalized["maxlen"]
                    if value:
                        self.registry_maxlen[key] = value

    def discover_selected_runs(self) -> None:
        log_hashes: dict[str, list[Path]] = defaultdict(list)
        for path in (self.source / "logs").rglob("metrics_by_k.png"):
            log_hashes[sha256(path)].append(path)
        packages = sorted(self.source.glob("results_package_*"))
        for package in packages:
            for path in package.rglob("metrics_by_k.png"):
                parts = path.parts
                if "Графики по последним запускам" not in parts:
                    continue
                index = parts.index("Графики по последним запускам")
                if len(parts) <= index + 2:
                    continue
                dataset = canonical_dataset(parts[index + 1])
                label = parts[index + 2]
                match = re.match(r"(.+)_maxlen(\d+)$", label)
                if match:
                    model = canonical_model(match.group(1))
                    maxlen = match.group(2)
                else:
                    model = canonical_model(label)
                    maxlen = ""
                matches = log_hashes.get(sha256(path), [])
                if not matches:
                    continue
                run_id = sorted(
                    (candidate.parent.parent.name for candidate in matches)
                )[-1]
                self.selected_runs[(dataset, model, maxlen)] = run_id

    def checkpoint_config(self, path: Path, model: str) -> dict:
        if model == "GPTRec":
            yaml_path = path.with_suffix(".yaml")
            if yaml_path.exists():
                try:
                    import yaml

                    value = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
                    return json_value(value if isinstance(value, dict) else {})
                except (ImportError, OSError, ValueError):
                    return {"yaml_config": yaml_path.read_text(encoding="utf-8")}
            return {}
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:
            return {"metadata_error": str(error)}
        if not isinstance(payload, dict):
            return {}
        config = {}
        for key in ("args", "config", "model_kwargs"):
            value = payload.get(key)
            if isinstance(value, dict):
                config[key] = json_value(value)
            elif hasattr(value, "__dict__"):
                config[key] = json_value(vars(value))
        return config

    def parameter(self, config: dict, *names):
        matches = {}

        def visit(value):
            if not isinstance(value, dict):
                return
            for key, nested in value.items():
                lowered = str(key).lower()
                if lowered in names and lowered not in matches:
                    matches[lowered] = nested
                visit(nested)

        visit(config)
        for name in names:
            if name in matches:
                return matches[name]
        return ""

    def migrate_checkpoints(self) -> None:
        root = self.source / "checkpoints"
        primary_files = [
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt", ".pth"}
        ]
        for source in sorted(primary_files):
            relative = source.relative_to(root)
            model = canonical_model(relative.parts[0])
            name = source.stem
            dataset_text = re.split(r"_maxlen\d+|_seed\d+", name, maxsplit=1)[0]
            dataset = canonical_dataset(dataset_text)
            maxlen_match = re.search(r"_maxlen(\d+)", name)
            seed_match = re.search(r"_seed(\d+)", name)
            maxlen = maxlen_match.group(1) if maxlen_match else ""
            seed = seed_match.group(1) if seed_match else "42"
            is_tuning = "tuning" in [part.lower() for part in relative.parts]
            if is_tuning:
                target = (
                    self.destination / "service_files" / "models" / model / dataset
                    / "tuning_files"
                    / "resume_checkpoints" / source.name
                )
                self.copy(source, target, "tuning_checkpoint")
                continue
            suffix_start = seed_match.end() if seed_match else len(name)
            extra = name[suffix_start:]
            target_name = (
                f"{dataset}__{maxlen_folder(maxlen)}__seed_{seed}{extra}{source.suffix.lower()}"
            )
            target = self.destination / "checkpoints" / model / target_name
            copied = self.copy(source, target, "checkpoint")
            self.checkpoint_destinations[str(source.resolve())] = str(copied)
            config = self.checkpoint_config(source, model)
            source_hash = sha256(source)
            metadata_path = copied.with_name(f"{copied.stem}.metadata.json")
            metadata = {
                "model": model,
                "dataset": dataset,
                "maxlen": int(maxlen) if maxlen else None,
                "seed": int(seed),
                "type": "final",
                "created_at": datetime.fromtimestamp(source.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
                "checkpoint": self.relative_destination(copied),
                "source_checkpoint": self.relative_source(source),
                "config": config,
            }
            write_json(metadata_path, metadata)
            yaml_source = source.with_suffix(".yaml")
            if yaml_source.exists():
                self.copy(
                    yaml_source, copied.with_suffix(".yaml"), "checkpoint_config"
                )
            row = {
                "checkpoint": self.relative_destination(copied),
                "metadata": self.relative_destination(metadata_path),
                "model": model,
                "dataset": dataset,
                "maxlen": maxlen,
                "seed": seed,
                "type": "final",
                "created_at": metadata["created_at"],
                "size_bytes": source.stat().st_size,
                "sha256": source_hash,
                "epochs": self.parameter(config, "epochs", "num_epochs", "final_epochs"),
                "batch_size": self.parameter(config, "batch_size", "train_batch_size"),
                "hidden_size": self.parameter(
                    config, "hidden_size", "hidden_units", "hidden_dim"
                ),
                "embedding_dim": self.parameter(
                    config, "embedding_dim", "embedding_size", "emb_size"
                ),
                "num_layers": self.parameter(
                    config, "num_layers", "n_layers", "num_blocks"
                ),
                "num_heads": self.parameter(config, "num_heads", "n_heads"),
                "dropout": self.parameter(
                    config, "dropout", "dropout_rate", "dropout_prob"
                ),
                "learning_rate": self.parameter(config, "learning_rate", "lr"),
                "weight_decay": self.parameter(config, "weight_decay"),
                "optimizer": self.parameter(config, "optimizer"),
                "scheduler": self.parameter(config, "scheduler"),
                "patience": self.parameter(config, "patience"),
                "diffusion_steps": self.parameter(config, "diffusion_steps", "steps"),
                "sampling_steps": self.parameter(config, "sampling_steps"),
                "noise_schedule": self.parameter(config, "noise_schedule"),
                "loss": self.parameter(config, "loss", "loss_type"),
                "loss_lambda": self.parameter(config, "loss_lambda", "loss_scale"),
            }
            self.checkpoint_rows.append(row)

    def run_maxlen(self, dataset: str, model: str, run_id: str, summary: dict) -> str:
        value = normalize_maxlen(summary.get("maxlen"))
        if value:
            return value
        value = self.registry_maxlen.get((dataset, model, run_id), "")
        if value:
            return value
        if model in {"T-DiffRec", "TopPopular", "Random"}:
            return ""
        return "unknown"

    def history_epochs(self, path: Path) -> tuple[int | str, bool]:
        if not path.exists():
            return "", False
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        epochs = {row.get("epoch", "") for row in rows if row.get("epoch", "") != ""}
        fields = {field for row in rows for field in row}
        return len(epochs), any("loss" in field.lower() for field in fields)

    def migrate_logs(self) -> None:
        root = self.source / "logs"
        for dataset_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            dataset = canonical_dataset(dataset_dir.name)
            for model_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
                model = canonical_model(model_dir.name)
                for run_dir in sorted(path for path in model_dir.iterdir() if path.is_dir()):
                    run_id = run_dir.name
                    summary_source = run_dir / "summary.json"
                    summary = read_json(summary_source)
                    maxlen = self.run_maxlen(dataset, model, run_id, summary)
                    history_source = run_dir / "history.csv"
                    epochs, has_loss = self.history_epochs(history_source)
                    validation_source = run_dir / "validation_selection.json"
                    has_final_metrics = bool(summary.get("final_metrics"))
                    if has_final_metrics and has_loss:
                        run_type = "final_training"
                    elif has_final_metrics and model in {"TopPopular", "Random"}:
                        run_type = "evaluation"
                    elif has_final_metrics:
                        run_type = "inference"
                    else:
                        run_type = "tuning"
                    model_root = (
                        self.destination / "service_files" / "models" / model / dataset
                    )
                    log_target = (
                        model_root / "tuning_files" / run_id
                        if run_type == "tuning"
                        else model_root / "logs" / run_id
                    )
                    for source in sorted(path for path in run_dir.rglob("*") if path.is_file()):
                        relative = source.relative_to(run_dir)
                        if relative.parts[0] == "plots" and source.suffix.lower() == ".png":
                            if source.name in {"metrics_by_k.png", "popularity_bias.png"}:
                                graphic_type = "evaluation"
                            elif run_type == "tuning":
                                graphic_type = "tuning"
                            else:
                                graphic_type = "training"
                            target = (
                                self.destination / "graphics" / graphic_type / model
                                / dataset
                                / f"{run_id}__{maxlen_folder(maxlen)}__{source.name}"
                            )
                            self.copy(source, target, f"{graphic_type}_graphic")
                            continue
                        if relative.parts[0] == "tensorboard":
                            target = (
                                log_target / "tensorboard"
                                / Path(*relative.parts[1:])
                            )
                            self.copy(source, target, "tensorboard")
                            continue
                        if source.name == "recommendations.csv":
                            target = (
                                model_root / "recommendations" / f"{run_id}.csv"
                            )
                            self.copy(source, target, "recommendations")
                            continue
                        target = log_target / relative
                        if source.name == "summary.json":
                            normalized = dict(summary)
                            normalized["dataset"] = dataset
                            normalized["model"] = model
                            normalized["maxlen"] = (
                                int(maxlen) if maxlen not in {"", "unknown"} else None
                            )
                            normalized["run_type"] = run_type
                            normalized["run_dir"] = str(log_target)
                            old_checkpoint = normalized.get("checkpoint")
                            if old_checkpoint:
                                resolved = str(Path(old_checkpoint).resolve())
                                normalized["checkpoint"] = self.checkpoint_destinations.get(
                                    resolved, old_checkpoint
                                )
                            self.transform_json(
                                source, target, normalized, "summary"
                            )
                            self.summary_destinations[str(source.resolve())] = str(target)
                        else:
                            self.copy(source, target, "service_log")
                    key = (dataset, model, maxlen, run_id)
                    record = {
                        field: "" for field in RUN_FIELDS
                    }
                    record.update({
                        "run_id": run_id,
                        "selected": "true" if self.selected_runs.get(
                            (dataset, model, maxlen)
                        ) == run_id else "false",
                        "created_at": summary.get("created_at", ""),
                        "run_type": run_type,
                        "model": model,
                        "dataset": dataset,
                        "maxlen": maxlen,
                        "epochs": epochs,
                        "seed": summary.get("seed", 42),
                        "split": summary.get("split", ""),
                        "mask_seen": summary.get("mask_seen", ""),
                        "ranking_protocol": summary.get("ranking_protocol", ""),
                        "latency_sec": summary.get(
                            "latency_sec", summary.get("inference_total_sec", "")
                        ),
                        "latency_ms_per_user": summary.get(
                            "latency_ms_per_user", ""
                        ),
                        "n_users": summary.get("n_users", ""),
                        "checkpoint": summary.get("checkpoint", ""),
                        "summary_path": self.summary_destinations.get(
                            str(summary_source.resolve()), ""
                        ),
                    })
                    for k, metrics in summary.get("final_metrics", {}).items():
                        for metric in ("recall", "ndcg", "mrr", "coverage"):
                            if metric in metrics:
                                record[f"{metric}@{k}"] = metrics[metric]
                    self.run_records[key] = record

    def merge_legacy_registry(self) -> None:
        for row in self.legacy_rows:
            dataset = row.get("dataset", "")
            model = row.get("model", "")
            maxlen = row.get("maxlen", "")
            run_id = row.get("run_id", "")
            key = (dataset, model, maxlen, run_id)
            record = self.run_records.setdefault(
                key, {field: "" for field in RUN_FIELDS}
            )
            record.update({
                "run_id": run_id,
                "model": model,
                "dataset": dataset,
                "maxlen": maxlen,
                "seed": row.get("seed", record.get("seed", "")),
                "split": row.get("split", record.get("split", "")),
                "mask_seen": row.get("mask_seen", record.get("mask_seen", "")),
                "created_at": row.get("created_at", record.get("created_at", "")),
                "latency_sec": row.get(
                    "latency_sec", record.get("latency_sec", "")
                ),
                "latency_ms_per_user": row.get(
                    "latency_ms_per_user",
                    record.get("latency_ms_per_user", ""),
                ),
                "n_users": row.get("n_users", record.get("n_users", "")),
            })
            if not record.get("run_type"):
                record["run_type"] = "legacy_evaluation"
            if not record.get("selected"):
                record["selected"] = "true" if self.selected_runs.get(
                    (dataset, model, maxlen)
                ) == run_id else "false"
            k = row.get("k", "")
            for metric in ("recall", "ndcg", "mrr", "coverage"):
                if row.get(metric, "") != "":
                    record[f"{metric}@{k}"] = row[metric]
            old_summary = row.get("summary_path", "")
            if old_summary:
                try:
                    resolved = str(Path(old_summary).resolve())
                    record["summary_path"] = self.summary_destinations.get(
                        resolved, record.get("summary_path", "")
                    )
                except OSError:
                    pass

    def migrate_datasets(self) -> None:
        root = self.source / "reports" / "datasets"
        preferred = {
            "amazon_baby": "amazon_Baby",
            "amazon_toys": "amazon_Toys_and_Games",
            "ml-1m": "ml-1m",
        }
        groups: dict[str, list[Path]] = defaultdict(list)
        if root.exists():
            for directory in root.iterdir():
                if directory.is_dir():
                    groups[canonical_dataset(directory.name)].append(directory)
        for dataset, directories in groups.items():
            chosen = next(
                (
                    directory for directory in directories
                    if directory.name == preferred.get(dataset)
                ),
                sorted(directories)[0],
            )
            target_root = self.destination / "datasets" / dataset
            for source in sorted(path for path in chosen.rglob("*") if path.is_file()):
                target = target_root / source.name
                if source.name == "summary.csv":
                    with source.open(newline="", encoding="utf-8-sig") as handle:
                        rows = list(csv.reader(handle))
                    if len(rows) > 1 and rows[1]:
                        rows[1][0] = dataset
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("w", newline="", encoding="utf-8") as handle:
                        csv.writer(handle).writerows(rows)
                    self.record(source, target, "dataset_summary", "transformed")
                else:
                    self.copy(source, target, "dataset_graphic")
            for directory in directories:
                if directory == chosen:
                    continue
                for source in sorted(path for path in directory.rglob("*") if path.is_file()):
                    destination = target_root / source.name
                    self.record(
                        source, destination, "dataset_alias",
                        "merged_alias", sha256(source),
                        sha256(destination) if destination.exists() else "",
                    )

    def migrate_reports(self) -> None:
        reports = self.source / "reports"
        tables = reports / "tables"
        if tables.exists():
            for source in tables.rglob("*"):
                if source.is_file():
                    self.copy(
                        source,
                        self.destination / "experiment_tables" / source.name,
                        "experiment_table",
                    )
        bt = reports / "bradley_terry"
        if bt.exists():
            for source in bt.rglob("*"):
                if not source.is_file():
                    continue
                relative = source.relative_to(bt)
                self.copy(
                    source, self.destination / "bt" / relative, "bradley_terry"
                )
        share = reports / "share"
        if share.exists():
            for source in share.rglob("*"):
                if source.is_file():
                    self.copy(
                        source,
                        self.destination / "service_files" / "other_service_files"
                        / source.name,
                        "other_service_file",
                    )
    def package_target(self, relative: Path) -> tuple[Path, str] | None:
        parts = relative.parts
        if not parts:
            return None
        if parts[0] == "Bradley_Terry":
            if len(parts) >= 3 and parts[1] == "graphs":
                return self.destination / "bt" / "graphics" / relative.name, "bradley_terry"
            if len(parts) >= 3 and parts[1] == "tables":
                return self.destination / "bt" / "tables" / relative.name, "bradley_terry"
        if parts[0] == "results":
            if relative.name in {"experimental_results.md", "experimental_results.xlsx"}:
                return (
                    self.destination / "experiment_tables" / relative.name,
                    "experiment_table",
                )
            if "Информация о датасетах" in parts:
                try:
                    dataset_index = parts.index("datasets") + 1
                    dataset = canonical_dataset(parts[dataset_index])
                    return (
                        self.destination / "datasets" / dataset / relative.name,
                        "dataset",
                    )
                except (ValueError, IndexError):
                    return None
            if "Графики по последним запускам" in parts:
                index = parts.index("Графики по последним запускам")
                try:
                    dataset = canonical_dataset(parts[index + 1])
                    label = parts[index + 2]
                except IndexError:
                    return None
                match = re.match(r"(.+)_maxlen(\d+)$", label)
                if match:
                    model = canonical_model(match.group(1))
                    maxlen = match.group(2)
                else:
                    model = canonical_model(label)
                    maxlen = ""
                return (
                    self.destination / "graphics" / "evaluation" / model / dataset
                    / f"selected__{maxlen_folder(maxlen)}__{relative.name}",
                    "evaluation_graphic",
                )
        return None

    def migrate_packages(self) -> None:
        for package in sorted(self.source.glob("results_package_*")):
            for source in sorted(path for path in package.rglob("*") if path.is_file()):
                mapping = self.package_target(source.relative_to(package))
                if mapping is None:
                    self.copy_unique(
                        source,
                        self.destination / "other_files" / package.name
                        / source.relative_to(package),
                        "other_file",
                    )
                else:
                    self.copy_unique(source, mapping[0], mapping[1])

    def write_registries(self) -> None:
        registry = self.destination / "service_files" / "all_experiments.csv"
        registry.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(
            self.run_records.values(),
            key=lambda row: (
                row.get("dataset", ""), row.get("model", ""),
                row.get("maxlen", ""), row.get("created_at", ""), row.get("run_id", ""),
            ),
        )
        with registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        checkpoint_registry = self.destination / "checkpoints" / "checkpoints_registry.csv"
        checkpoint_registry.parent.mkdir(parents=True, exist_ok=True)
        with checkpoint_registry.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CHECKPOINT_FIELDS)
            writer.writeheader()
            writer.writerows(self.checkpoint_rows)

    def verify(self) -> None:
        failures = []
        for row in self.manifest:
            if row["status"] not in {"copied", "duplicate", "duplicate_content"}:
                continue
            destination = self.destination / row["destination"]
            if not destination.exists():
                failures.append(f"Missing destination: {destination}")
                continue
            if sha256(destination) != row["destination_sha256"]:
                failures.append(f"Checksum mismatch: {destination}")
        if failures:
            raise RuntimeError("\n".join(failures))

    def run(self) -> None:
        if not self.source.is_dir():
            raise FileNotFoundError(self.source)
        if self.destination.exists():
            raise FileExistsError(
                f"Destination already exists and will not be overwritten: {self.destination}"
            )
        for name in (
            "service_files", "datasets", "bt", "graphics/training", "graphics/tuning",
            "graphics/evaluation", "experiment_tables", "checkpoints", "other_files",
            "service_files/other_service_files",
        ):
            (self.destination / name).mkdir(parents=True, exist_ok=True)
        self.load_legacy_registries()
        self.discover_selected_runs()
        self.migrate_checkpoints()
        self.migrate_logs()
        self.merge_legacy_registry()
        self.migrate_datasets()
        self.migrate_reports()
        self.migrate_packages()
        self.write_registries()
        self.verify()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    migration = Migration(args.source, args.destination)
    migration.run()
    print(args.destination.resolve())


if __name__ == "__main__":
    main()
