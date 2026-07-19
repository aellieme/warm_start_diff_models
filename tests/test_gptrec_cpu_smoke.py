"""One-batch CPU training smoke test for GPTRec's GPT-2 backbone."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config, GPT2LMHeadModel

try:
    from test_coverage import MODULES
except ModuleNotFoundError:
    from tests.test_coverage import MODULES


class GPTRecCpuSmokeTest(unittest.TestCase):
    def test_prepare_data_returns_python_int_item_count(self):
        source_path = (
            Path(__file__).resolve().parents[1]
            / "src" / "GPTRec" / "src" / "run_train_predict.py"
        )
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        prepare_data_node = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "prepare_data"
        )
        isolated_module = ast.Module(body=[prepare_data_node], type_ignores=[])
        ast.fix_missing_locations(isolated_module)

        frame = pd.DataFrame({
            "user_id": [0] * 10,
            "item_id": np.asarray(range(1, 11), dtype=np.int16),
            "timestamp": range(1, 11),
        })

        def add_time_idx(dataframe):
            result = dataframe.copy()
            result["time_idx"] = result.groupby("user_id").cumcount()
            return result

        namespace = {
            "pd": pd,
            "load_amazon": lambda *_: frame.copy(),
            "add_time_idx": add_time_idx,
        }
        exec(compile(isolated_module, source_path, "exec"), namespace)

        class Config(SimpleNamespace):
            def get(self, name, default=None):
                return getattr(self, name, default)

        config = Config(
            dataset_name="amazon_synthetic",
            amazon_data_dir="unused",
            global_time_col="timestamp",
            split_ratios=[0.7, 0.1, 0.2],
        )
        *_, item_count = namespace["prepare_data"](config)

        self.assertIs(type(item_count), int)
        self.assertEqual(item_count, 10)

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
