import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_tools.experiment_tracking import (ExperimentTracker, checkpoint_due, checkpoint_path,
                                 recommendation_popularity, save_dataset_popularity,
                                 save_torch_checkpoint, plt)
from experiment_tools.generate_result_tables import build_tables, write_markdown


class ExperimentTrackingTests(unittest.TestCase):
    def test_checkpoint_schedule_and_atomic_round_trip(self):
        self.assertFalse(checkpoint_due(23, 250, every=25))
        self.assertTrue(checkpoint_due(24, 250, every=25))
        self.assertTrue(checkpoint_due(249, 250, every=25))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_torch_checkpoint({"weight": torch.tensor([1.0])}, path)
            save_torch_checkpoint({"weight": torch.tensor([2.0])}, path)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(payload["weight"].item(), 2.0)
            self.assertFalse(path.with_suffix(".pt.tmp").exists())

    def test_checkpoint_path_is_stable_and_beside_logs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"EXPERIMENT_OUTPUT_DIR": str(Path(directory) / "logs")}
        ):
            path = checkpoint_path("SASRec", "ml-1m", maxlen=50, seed=42)
            self.assertEqual(
                path,
                Path(directory).resolve() / "checkpoints" / "SASRec" / "ml-1m_maxlen50_seed42.pt",
            )
            self.assertTrue(path.parent.is_dir())

    def test_tracker_writes_local_artifacts(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"EXPERIMENT_OUTPUT_DIR": str(Path(directory) / "logs")}
        ):
            tracker = ExperimentTracker("tiny", "model", run_id="run")
            tracker.log_epoch(0, train_loss=2.0)
            tracker.log_epoch(1, train_loss=1.0, **{
                "val_recall@10": 0.4,
                "val_ndcg@10": 0.3,
                "val_mrr@10": 0.2,
                "val_coverage@10": 0.5,
            })
            tracker.log_validation_selection(
                1,
                {"Recall@10": 0.4, "NDCG@10": 0.3, "Coverage@10": 0.5},
                rule="recall_then_coverage",
                inference_seeds=(42, 43),
            )
            tracker.log_final_metrics({
                10: {"recall": 0.4, "ndcg": 0.3, "mrr": 0.2, "coverage": 0.5}
            }, split="synthetic", mask_seen=True, seed=42, maxlen=50,
               inference_total_sec=2.0, n_users=4)
            tracker.close()

            run = Path(directory) / "logs" / "tiny" / "model" / "run"
            self.assertTrue((run / "history.csv").exists())
            self.assertTrue((run / "summary.json").exists())
            self.assertTrue((run / "validation_selection.json").exists())
            if plt is not None:
                self.assertTrue((run / "plots" / "loss.png").exists())
                self.assertTrue((run / "plots" / "validation_ranking.png").exists())
                self.assertTrue((run / "plots" / "metrics_by_k.png").exists())
            payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["mask_seen"])
            self.assertEqual(payload["latency_ms_per_user"], 500.0)
            selection = json.loads(
                (run / "validation_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selection["selected_epoch"], 1)
            self.assertEqual(selection["inference_seeds"], [42, 43])
            with (run / "history.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            with (Path(directory) / "results_registry.csv").open(encoding="utf-8") as handle:
                registry = list(csv.DictReader(handle))
            self.assertEqual(len(registry), 1)
            self.assertEqual(registry[0]["dataset"], "tiny")
            self.assertEqual(registry[0]["maxlen"], "50")
            self.assertEqual(registry[0]["latency_ms_per_user"], "500.0")

    def test_popularity_uses_train_counts_only(self):
        popularity = {1: 100, 2: 10, 3: 1}
        bias = recommendation_popularity([[1, 2], [1, 3]], popularity, [1, 2])
        self.assertEqual(bias[1]["head_exposure"], 1.0)
        self.assertEqual(bias[2]["head_exposure"], 0.5)
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"EXPERIMENT_OUTPUT_DIR": str(Path(directory) / "logs")}
        ):
            path = save_dataset_popularity("tiny", popularity)
            if plt is not None:
                self.assertIsNotNone(path)
                self.assertTrue(path.exists())

    def test_fixed_tables_use_latest_matching_run_and_expected_order(self):
        rows = []
        for run_id, created_at, recall in (("old", "2026-01-01T00:00:00", "0.1"),
                                            ("new", "2026-01-02T00:00:00", "0.2")):
            for k in (10, 20, 100):
                rows.append({
                    "dataset": "ML-1M", "model": "ADRec", "maxlen": "50", "k": str(k),
                    "recall": recall, "ndcg": "0.3", "mrr": "0.2", "coverage": "0.4",
                    "latency_sec": "1.25", "seed": "42", "run_id": run_id,
                    "created_at": created_at,
                })
        tables = build_tables(rows, seed=42)
        self.assertEqual(len(tables), 9)
        self.assertEqual((tables[0]["dataset"], tables[0]["k"]), ("ML-1M", 10))
        self.assertEqual((tables[3]["dataset"], tables[3]["k"]), ("Amazon Toys", 10))
        self.assertEqual(tables[0]["rows"][0]["Recall"], "0.200000")
        self.assertEqual(tables[0]["rows"][8]["Model"], "T-DiffRec")
        self.assertEqual(tables[0]["rows"][8]["Maxlen"], "-")
        self.assertEqual(tables[0]["rows"][8]["Recall"], "—")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tables.md"
            write_markdown(tables, path)
            text = path.read_text(encoding="utf-8")
            self.assertIn("Таблица 1: ML-1M | k = 10", text)
            self.assertIn("Таблица 9: Amazon Baby | k = 100", text)


if __name__ == "__main__":
    unittest.main()
