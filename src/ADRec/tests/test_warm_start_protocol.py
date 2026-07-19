import sys
import unittest
from pathlib import Path

import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from utils import (  # noqa: E402
    build_candidate_mask,
    build_final_train_sequences,
    eligible_warm_start_rows,
    filter_history_to_candidates,
    mask_ranking_scores,
    mask_training_scores,
    prepare_model_history,
)


class WarmStartProtocolTest(unittest.TestCase):
    def test_final_sequences_include_validation_only_users(self):
        data_raw = {
            "train_dict": {1: [1, 2]},
            "val_seq_dict": {1: [1, 2, 3], 2: [4]},
            "val_tgt_dict": {1: 4, 2: 5},
        }

        self.assertEqual(
            build_final_train_sequences(data_raw),
            [[1, 2, 3, 4], [4, 5]],
        )

    def test_ranking_masks_padding_seen_and_future_items(self):
        candidate_mask = build_candidate_mask({1, 3, 4}, 6, "cpu")
        histories = torch.tensor([[0, 1, 5], [0, 5, 0], [0, 3, 0]])
        targets = torch.tensor([[0, 0, 3], [0, 0, 3], [0, 0, 5]])
        histories = filter_history_to_candidates(histories, candidate_mask)

        eligible = eligible_warm_start_rows(histories, targets, candidate_mask)
        self.assertEqual(eligible.tolist(), [True, False, False])

        scores = torch.arange(6, dtype=torch.float32).repeat(3, 1)
        mask_ranking_scores(scores, histories, candidate_mask)
        self.assertTrue(torch.isneginf(scores[0, [0, 1, 2, 5]]).all())
        self.assertEqual(set(torch.topk(scores[0], 2).indices.tolist()), {3, 4})

    def test_training_loss_has_no_gradient_for_future_items(self):
        candidate_mask = build_candidate_mask({1, 3, 4}, 6, "cpu")
        scores = torch.zeros((1, 6), requires_grad=True)
        masked_scores = mask_training_scores(scores, candidate_mask)
        loss = torch.nn.functional.cross_entropy(masked_scores, torch.tensor([3]))
        loss.backward()

        self.assertEqual(scores.grad[0, [0, 2, 5]].tolist(), [0.0, 0.0, 0.0])
        self.assertNotEqual(scores.grad[0, 1].item(), 0.0)

    def test_model_input_is_truncated_but_seen_mask_uses_full_history(self):
        candidate_mask = build_candidate_mask({1, 2, 3, 4}, 6, "cpu")
        full_history = torch.tensor([[1, 2, 3]])
        model_history = prepare_model_history(full_history, candidate_mask, max_len=2)
        self.assertEqual(model_history.tolist(), [[2, 3]])

        scores = torch.zeros((1, 6))
        mask_ranking_scores(scores, full_history, candidate_mask)
        self.assertTrue(torch.isneginf(scores[0, [1, 2, 3]]).all())


if __name__ == "__main__":
    unittest.main()
