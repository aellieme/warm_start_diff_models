import sys
import unittest
from pathlib import Path

import pandas as pd
import torch


SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_DIR))

from datasets import LastEvaluationDataset  # noqa: E402
from metrics import Evaluator  # noqa: E402
from modules import CandidateItemsProcessor, SeqRecBase  # noqa: E402


class DummySequenceModel(torch.nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError


class WarmStartProtocolTest(unittest.TestCase):
    def test_last_evaluation_uses_known_items_not_known_user_identity(self):
        train = pd.DataFrame({
            "user_id": [1, 1, 2],
            "item_id": [1, 2, 3],
            "time_idx": [0, 1, 0],
        })
        test = pd.DataFrame({
            "user_id": [1, 1, 9, 9, 10, 11, 11, 12, 12],
            "item_id": [3, 2, 1, 2, 4, 1, 4, 4, 2],
            "time_idx": [0, 1, 0, 1, 0, 0, 1, 0, 1],
        })

        dataset = LastEvaluationDataset(train, test, max_length=1)
        samples = {sample["user_id"]: sample for sample in dataset.samples}

        self.assertEqual(set(samples), {1, 9})
        self.assertEqual(samples[9]["input_ids"].tolist(), [1])
        self.assertEqual(samples[9]["target"], 2)
        self.assertEqual(samples[1]["input_ids"].tolist(), [3])
        self.assertEqual(samples[1]["full_history"], [1, 2, 3])

    def test_logits_and_generation_are_restricted_to_candidates(self):
        module = SeqRecBase(
            DummySequenceModel(), candidate_items={1, 3, 4}, predict_top_k=2
        )
        logits = torch.zeros((1, 6), requires_grad=True)
        masked = module.mask_candidate_logits(logits)
        loss = torch.nn.functional.cross_entropy(masked, torch.tensor([3]))
        loss.backward()
        self.assertEqual(logits.grad[0, [0, 2, 5]].tolist(), [0.0, 0.0, 0.0])

        processor = CandidateItemsProcessor(module.candidate_item_ids)
        generated_scores = processor(
            torch.tensor([[1]]), torch.arange(6, dtype=torch.float32).unsqueeze(0)
        )
        self.assertTrue(torch.isneginf(generated_scores[0, [0, 2, 5]]).all())

    def test_metrics_reject_out_of_catalogue_recommendations(self):
        evaluator = Evaluator(top_k=[1])
        train = pd.DataFrame({"user_id": [1], "item_id": [1]})
        test = pd.DataFrame({"user_id": [1], "item_id": [1]})
        recs = pd.DataFrame({"user_id": [1], "item_id": [2]})

        with self.assertRaisesRegex(ValueError, "outside"):
            evaluator.compute_metrics(test, recs, train)


if __name__ == "__main__":
    unittest.main()
