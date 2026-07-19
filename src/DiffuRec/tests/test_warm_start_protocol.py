import sys
import unittest
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from utils import (  # noqa: E402
    ValDataset,
    build_candidate_mask,
    eligible_warm_start_rows,
    filter_history_to_candidates,
    mask_ranking_scores,
    prepare_model_history,
)


class WarmStartProtocolTest(unittest.TestCase):
    def test_dataset_keeps_full_history_separate_from_model_input(self):
        sequence, target, full_history = ValDataset(
            {7: [1, 2, 3]}, {7: [4]}, max_len=2
        )[0]
        self.assertEqual(sequence.tolist(), [2, 3])
        self.assertEqual(target.tolist(), [4])
        self.assertEqual(full_history.tolist(), [1, 2, 3])

    def test_full_history_is_filtered_compacted_and_masked(self):
        candidate_mask = build_candidate_mask({1, 3, 4}, 6, "cpu")
        raw_history = torch.tensor([[1, 5, 3]])
        full_history = filter_history_to_candidates(raw_history, candidate_mask)
        model_history = prepare_model_history(full_history, candidate_mask, max_len=2)
        self.assertEqual(model_history.tolist(), [[1, 3]])
        self.assertEqual(
            eligible_warm_start_rows(full_history, torch.tensor([[4]]), candidate_mask).tolist(),
            [True],
        )

        scores = torch.zeros((1, 6))
        mask_ranking_scores(scores, full_history, candidate_mask)
        self.assertTrue(torch.isneginf(scores[0, [0, 1, 2, 3, 5]]).all())


if __name__ == "__main__":
    unittest.main()
