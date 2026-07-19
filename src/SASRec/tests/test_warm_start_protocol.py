import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

from model import SASRec  # noqa: E402
from warm_start import (  # noqa: E402
    filter_history_to_candidates,
    is_eligible_warm_start_example,
    mask_ranking_scores,
    topn_from_masked_scores,
)


class WarmStartProtocolTest(unittest.TestCase):
    def test_ranking_uses_only_known_unseen_items(self):
        candidates = {0, 2, 3}
        history = filter_history_to_candidates([0, 1, 4], candidates)
        self.assertEqual(history, [0])
        self.assertTrue(is_eligible_warm_start_example(history, 2, candidates))
        self.assertFalse(is_eligible_warm_start_example(history, 1, candidates))

        scores = np.arange(6, dtype=np.float32)
        mask_ranking_scores(scores, history, candidates, pad_token=5)
        recs = topn_from_masked_scores(scores[None, :], topn=4)[0]
        self.assertEqual(recs.tolist(), [3, 2, -1, -1])

    def test_empty_known_history_is_ineligible(self):
        candidates = {0, 2}
        history = filter_history_to_candidates([1, 3], candidates)
        self.assertFalse(is_eligible_warm_start_example(history, 2, candidates))

    def test_training_logits_exclude_future_items_and_padding(self):
        config = {
            "maxlen": 2,
            "hidden_units": 4,
            "dropout_rate": 0.0,
            "num_blocks": 1,
            "num_heads": 1,
            "manual_seed": 42,
        }
        model = SASRec(4, config, candidate_items={0, 2})
        logits = model(torch.tensor([[model.pad_token, 0]]))

        excluded = logits[0, -1, [1, 3, model.pad_token]]
        self.assertEqual(
            excluded.tolist(),
            [torch.finfo(logits.dtype).min] * 3,
        )


if __name__ == "__main__":
    unittest.main()
