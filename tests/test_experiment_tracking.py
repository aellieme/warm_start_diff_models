import csv
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiment_tracking import (ExperimentTracker, recommendation_popularity,
                                 save_dataset_popularity, plt)


class ExperimentTrackingTests(unittest.TestCase):
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
            })
            tracker.log_final_metrics({
                10: {"recall": 0.4, "ndcg": 0.3, "mrr": 0.2, "coverage": 0.5}
            }, split="synthetic", mask_seen=True)
            tracker.close()

            run = Path(directory) / "logs" / "tiny" / "model" / "run"
            self.assertTrue((run / "history.csv").exists())
            self.assertTrue((run / "summary.json").exists())
            if plt is not None:
                self.assertTrue((run / "plots" / "loss.png").exists())
                self.assertTrue((run / "plots" / "validation_ranking.png").exists())
                self.assertTrue((run / "plots" / "metrics_by_k.png").exists())
            payload = json.loads((run / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(payload["mask_seen"])
            with (run / "history.csv").open(encoding="utf-8") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

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


if __name__ == "__main__":
    unittest.main()
