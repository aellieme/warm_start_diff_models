"""One-batch CPU training smoke test for GPTRec's GPT-2 backbone."""

from __future__ import annotations

import unittest

import torch
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from test_coverage import MODULES
except ModuleNotFoundError:
    from tests.test_coverage import MODULES


class GPTRecCpuSmokeTest(unittest.TestCase):
    def test_one_training_batch_prediction_and_coverage(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

        item_count = 6
        padding_id = 0
        config = GPT2Config(
            vocab_size=item_count + 1,
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            pad_token_id=padding_id,
        )
        model = GPT2LMHeadModel(config).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        input_ids = torch.tensor(
            [[1, 2, 3, 0], [2, 3, 4, 0]],
            dtype=torch.long,
        )
        attention_mask = (input_ids != padding_id).long()
        labels = input_ids.clone()
        labels[labels == padding_id] = -100

        weights_before = model.transformer.wte.weight.detach().clone()
        model.train()
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(
            torch.equal(weights_before, model.transformer.wte.weight.detach())
        )

        model.eval()
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits
            last_positions = attention_mask.sum(dim=1) - 1
            rows = torch.arange(input_ids.size(0))
            scores = logits[rows, last_positions]
            scores[:, padding_id] = -torch.inf
            for row, history in enumerate(input_ids):
                seen = history[history != padding_id]
                scores[row, seen] = -torch.inf
            predicted = torch.topk(scores, k=2, dim=-1).indices.tolist()

        candidate_items = set(range(1, item_count + 1))
        actual = [[4], [5]]
        *_, coverages = MODULES["GPTRec"].compute_all_metrics(
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
