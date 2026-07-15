"""One-batch CPU training smoke test for the real SASRec model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

try:
    from test_coverage import MODULES
except ModuleNotFoundError:
    from tests.test_coverage import MODULES


ROOT = Path(__file__).resolve().parents[1]


def load_sasrec_class():
    path = ROOT / "src" / "SASRec" / "model.py"
    spec = importlib.util.spec_from_file_location("sasrec_smoke_model", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SASRec from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SASRec


SASRec = load_sasrec_class()


class SASRecCpuSmokeTest(unittest.TestCase):
    def test_one_training_epoch_and_coverage_on_tiny_dataset(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

        config = {
            "maxlen": 4,
            "hidden_units": 8,
            "dropout_rate": 0.0,
            "num_blocks": 1,
            "num_heads": 1,
            "manual_seed": 42,
        }
        item_count = 5
        pad = item_count
        model = SASRec(item_count, config).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad)

        # Two users, already converted to SASRec's internal 0-based item IDs.
        sequences = torch.tensor(
            [
                [pad, 0, 1, 2],
                [pad, 1, 2, 3],
            ],
            dtype=torch.long,
        )
        next_items = torch.tensor(
            [
                [pad, 1, 2, 3],
                [pad, 2, 3, 4],
            ],
            dtype=torch.long,
        )

        weights_before = model.item_emb.weight.detach().clone()
        model.train()
        optimizer.zero_grad()
        logits = model(sequences)
        loss = criterion(logits.reshape(-1, logits.size(-1)), next_items.reshape(-1))
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.equal(weights_before, model.item_emb.weight.detach()))

        model.eval()
        histories = ([0, 1, 2], [1, 2, 3])
        actual = [[3], [4]]
        predicted = []
        with torch.no_grad():
            for history in histories:
                scores = model.score(torch.tensor(history, dtype=torch.long)).squeeze(0)
                scores = scores[:item_count]
                scores[list(history)] = -torch.inf
                predicted.append(torch.topk(scores, k=2).indices.tolist())

        candidate_items = set(range(item_count))
        *_, coverages = MODULES["SASRec"].compute_all_metrics(
            actual,
            predicted,
            [2],
            n_items=len(candidate_items),
            candidate_items=candidate_items,
        )

        self.assertEqual(len(predicted), 2)
        self.assertTrue(all(len(recs) == 2 for recs in predicted))
        self.assertGreaterEqual(coverages[0], 0.0)
        self.assertLessEqual(coverages[0], 1.0)


if __name__ == "__main__":
    unittest.main()
