"""Small CPU-only temporal-split tests for the unified Coverage contract."""

from __future__ import annotations

import unittest

from test_coverage import MODULES


# Tiny MovieLens-like fragment after model-side ID encoding.
# Item 40 appears for the first time in test and is therefore cold/test-only.
MOVIELENS_MINI = {
    "train": [(1, 10, 100), (2, 20, 110), (1, 20, 120)],
    "validation": [(1, 30, 130)],
    "test": [(1, 20, 140), (2, 40, 150)],
}

# Tiny Amazon-like fragment after ASINs have been converted to internal IDs.
# Item 104 appears for the first time in test.
AMAZON_MINI = {
    "train": [("u1", 101, 100), ("u2", 102, 110)],
    "validation": [("u1", 103, 120)],
    "test": [("u1", 102, 130), ("u2", 104, 140)],
}


def item_catalog(interactions):
    return {item for _, item, _ in interactions}


class MiniTemporalDatasetCoverageTests(unittest.TestCase):
    def assert_all_models_coverage(
        self,
        *,
        actual,
        predicted,
        candidate_items,
        expected,
    ):
        for model_name, metrics in MODULES.items():
            with self.subTest(model=model_name):
                *_, coverages = metrics.compute_all_metrics(
                    actual,
                    predicted,
                    [2],
                    n_items=len(candidate_items),
                    candidate_items=candidate_items,
                )
                self.assertAlmostEqual(coverages[0], expected)
                self.assertLessEqual(coverages[0], 1.0)

    def test_movielens_tuning_uses_train_catalogue_only(self):
        train_catalog = item_catalog(MOVIELENS_MINI["train"])

        self.assertEqual(train_catalog, {10, 20})
        self.assert_all_models_coverage(
            actual=[[30]],
            predicted=[[10, 30]],
            candidate_items=train_catalog,
            expected=1 / 2,
        )

    def test_movielens_final_test_uses_train_plus_validation_catalogue(self):
        final_catalog = item_catalog(
            MOVIELENS_MINI["train"] + MOVIELENS_MINI["validation"]
        )

        self.assertEqual(final_catalog, {10, 20, 30})
        self.assertNotIn(40, final_catalog)
        self.assert_all_models_coverage(
            actual=[[20], [40]],
            predicted=[[10, 20], [30, 40]],
            candidate_items=final_catalog,
            expected=3 / 3,
        )

    def test_amazon_tuning_uses_train_catalogue_only(self):
        train_catalog = item_catalog(AMAZON_MINI["train"])

        self.assertEqual(train_catalog, {101, 102})
        self.assert_all_models_coverage(
            actual=[[103]],
            predicted=[[101, 103]],
            candidate_items=train_catalog,
            expected=1 / 2,
        )

    def test_amazon_final_test_uses_train_plus_validation_catalogue(self):
        final_catalog = item_catalog(
            AMAZON_MINI["train"] + AMAZON_MINI["validation"]
        )

        self.assertEqual(final_catalog, {101, 102, 103})
        self.assertNotIn(104, final_catalog)
        self.assert_all_models_coverage(
            actual=[[102], [104]],
            predicted=[[101, 102], [103, 104]],
            candidate_items=final_catalog,
            expected=3 / 3,
        )


if __name__ == "__main__":
    unittest.main()
