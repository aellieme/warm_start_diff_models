"""Contract tests for catalogue coverage in every evaluated model.

The model directories are not Python packages (and T-DiffRec contains a
hyphen), so the metric modules are loaded directly from their file paths.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]

METRIC_MODULES = {
    "TopPopular": ROOT / "src" / "TopPopular" / "evaluate_topk_dp.py",
    "RandomRecs": ROOT / "src" / "RandomRecs" / "evaluate_topk_dp.py",
    "SASRec": ROOT / "src" / "SASRec" / "evaluate_topk_dp.py",
    "T-DiffRec": ROOT / "src" / "DiffRec" / "T-DiffRec" / "evaluate_topk_dp.py",
    "DiffuRec": ROOT / "src" / "DiffuRec" / "src" / "evaluate_topk_dp.py",
    "GPTRec": ROOT / "src" / "GPTRec" / "src" / "evaluate_topk_dp.py",
    "ADRec": ROOT / "src" / "ADRec" / "src" / "evaluate_topk_dp.py",
}


def load_module(model_name: str, path: Path):
    module_name = f"coverage_metrics_{model_name.lower().replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics for {model_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULES = {
    model_name: load_module(model_name, path)
    for model_name, path in METRIC_MODULES.items()
}


class CommonCoverageContractTests(unittest.TestCase):
    def test_all_models_count_unique_recommended_items(self):
        predicted = [[1, 2, 2], [2, 3]]

        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                self.assertAlmostEqual(metrics.coverage(predicted, 10), 3 / 10)

    def test_all_models_return_zero_for_empty_catalogue(self):
        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                self.assertEqual(metrics.coverage([[1, 2]], 0), 0.0)

    def test_compute_all_metrics_applies_each_top_k_to_coverage(self):
        actual = [[9], [8]]
        predicted = [[1, 2], [1, 3]]

        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                *_, coverages = metrics.compute_all_metrics(
                    actual,
                    predicted,
                    [1, 2],
                    10,
                )
                self.assertEqual(len(coverages), 2)
                self.assertAlmostEqual(coverages[0], 1 / 10)
                self.assertAlmostEqual(coverages[1], 3 / 10)

    def test_coverage_is_bounded_when_recommendations_belong_to_catalogue(self):
        predicted = [[1, 2, 3], [3, 4, 5]]

        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                value = metrics.coverage(predicted, 5)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_all_models_ignore_items_outside_explicit_candidate_catalogue(self):
        candidate_items = {1, 2, 3}
        predicted = [[1, 4], [2, 5]]

        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                value = metrics.coverage(
                    predicted,
                    n_items=999,
                    candidate_items=candidate_items,
                )
                self.assertAlmostEqual(value, 2 / 3)

    def test_compute_all_metrics_forwards_candidate_catalogue(self):
        actual = [[9], [8]]
        predicted = [[1, 4], [2, 5]]
        candidate_items = {1, 2, 3}

        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                *_, coverages = metrics.compute_all_metrics(
                    actual,
                    predicted,
                    [2],
                    n_items=999,
                    candidate_items=candidate_items,
                )
                self.assertAlmostEqual(coverages[0], 2 / 3)

    def test_empty_explicit_candidate_catalogue_returns_zero(self):
        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                value = metrics.coverage(
                    [[1, 2]],
                    n_items=10,
                    candidate_items=set(),
                )
                self.assertEqual(value, 0.0)


class ExistingPaddingConventionTests(unittest.TestCase):
    """Preserve the model-specific 0/-1 conventions requested by the project."""

    def test_top_popular_and_random_recs_ignore_minus_one(self):
        predicted = [[1, -1], [2, -1]]

        for model_name in ("TopPopular", "RandomRecs"):
            with self.subTest(model=model_name):
                self.assertAlmostEqual(
                    MODULES[model_name].coverage(predicted, 10),
                    2 / 10,
                )

    def test_gptrec_and_adrec_ignore_zero(self):
        predicted = [[1, 0], [2, 0]]

        for model_name in ("GPTRec", "ADRec"):
            with self.subTest(model=model_name):
                self.assertAlmostEqual(
                    MODULES[model_name].coverage(predicted, 10),
                    2 / 10,
                )

    def test_models_without_padding_filter_keep_current_behaviour(self):
        predicted = [[1, 0], [2, -1]]

        for model_name in ("SASRec", "T-DiffRec", "DiffuRec"):
            with self.subTest(model=model_name):
                self.assertAlmostEqual(
                    MODULES[model_name].coverage(predicted, 10),
                    4 / 10,
                )


if __name__ == "__main__":
    unittest.main()
