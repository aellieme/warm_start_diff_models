"""CPU-only end-to-end smoke tests for TopPopular and RandomRecs."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    source_dir = str(path.parent)
    previous_metrics = sys.modules.pop("evaluate_topk_dp", None)
    sys.path.insert(0, source_dir)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop("evaluate_topk_dp", None)
        if previous_metrics is not None:
            sys.modules["evaluate_topk_dp"] = previous_metrics
        sys.path.remove(source_dir)


TOP_POPULAR = load_module(
    "top_popular_smoke_model",
    ROOT / "src" / "TopPopular" / "TopPopular_model.py",
)
RANDOM_RECS = load_module(
    "random_recs_smoke_model",
    ROOT / "src" / "RandomRecs" / "RandomRecsModel.py",
)


class TopPopularCpuSmokeTest(unittest.TestCase):
    def test_tiny_end_to_end_inference_and_coverage(self):
        histories = [[1, 2], [2, 3]]
        popular_items = [2, 3, 4, 1]
        ground_truth = [[3], [4]]
        TOP_POPULAR.ground_truth = ground_truth

        recommendations = TOP_POPULAR.get_top_k_recommendations(
            histories,
            popular_items,
            k=2,
        )
        results = TOP_POPULAR.run_experiment(histories, popular_items, [2])

        self.assertEqual(recommendations, [[3, 4], [4, 1]])
        for history, recs in zip(histories, recommendations):
            self.assertTrue(set(history).isdisjoint(recs))
        self.assertEqual(results["recalls"], [1.0])
        self.assertEqual(results["covs"], [3 / 4])


class RandomRecsCpuSmokeTest(unittest.TestCase):
    def test_tiny_end_to_end_inference_and_coverage(self):
        histories = [[1, 2], [2, 3]]
        candidate_items = [1, 2, 3, 4]
        ground_truth = [[3], [4]]
        RANDOM_RECS.ground_truth = ground_truth

        recommendations = RANDOM_RECS.get_random_recommendations(
            histories,
            candidate_items,
            k=2,
            rng=np.random.default_rng(42),
        )
        results = RANDOM_RECS.run_experiment(
            histories,
            candidate_items,
            [2],
            rng=np.random.default_rng(42),
        )

        self.assertEqual(len(recommendations), len(histories))
        for history, recs in zip(histories, recommendations):
            self.assertEqual(len(recs), 2)
            self.assertTrue(set(history).isdisjoint(recs))
            self.assertTrue(set(recs).issubset(candidate_items))
        self.assertEqual(len(results["recalls"]), 1)
        self.assertGreaterEqual(results["covs"][0], 0.0)
        self.assertLessEqual(results["covs"][0], 1.0)


if __name__ == "__main__":
    unittest.main()
