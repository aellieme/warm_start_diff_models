"""One-batch CPU training smoke test for the real DiffuRec model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch

try:
    from test_coverage import MODULES
except ModuleNotFoundError:
    from tests.test_coverage import MODULES


ROOT = Path(__file__).resolve().parents[1]


def load_diffurec_components():
    source_dir = ROOT / "src" / "DiffuRec" / "src"
    path = source_dir / "model.py"
    sys.path.insert(0, str(source_dir))
    try:
        spec = importlib.util.spec_from_file_location("diffurec_smoke_model", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DiffuRec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        diffurec_module = sys.modules["diffurec"]
        return (
            module.create_model_diffu,
            module.Att_Diffuse_model,
            diffurec_module.MultiHeadedAttention,
        )
    finally:
        sys.path.remove(str(source_dir))


create_model_diffu, AttDiffuseModel, MultiHeadedAttention = load_diffurec_components()


def load_train_dataset():
    source_dir = ROOT / "src" / "DiffuRec" / "src"
    path = source_dir / "utils.py"
    spec = importlib.util.spec_from_file_location("diffurec_smoke_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DiffuRec utilities from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (
        module.TrainDataset,
        module.build_candidate_mask,
        module.eligible_warm_start_rows,
        module.encode_item_ids_with_padding,
        module.filter_history_to_candidates,
        module.mask_ranking_scores,
    )


(
    TrainDataset,
    build_candidate_mask,
    eligible_warm_start_rows,
    encode_item_ids_with_padding,
    filter_history_to_candidates,
    mask_ranking_scores,
) = load_train_dataset()


class DiffuRecCpuSmokeTest(unittest.TestCase):
    def test_item_encoding_reserves_zero_for_padding(self):
        encoded, mapping = encode_item_ids_with_padding([20, 10, 20, 30])

        self.assertEqual(encoded.tolist(), [2, 1, 2, 3])
        self.assertEqual(mapping, {1: 10, 2: 20, 3: 30})
        self.assertNotIn(0, encoded)
        self.assertNotIn(0, mapping)

    def test_ranking_masks_padding_seen_and_unknown_items(self):
        scores = torch.arange(14, dtype=torch.float32).reshape(2, 7)
        sequences = torch.tensor([[0, 1, 2], [0, 0, 4]])
        targets = torch.tensor([[3], [6]])
        candidate_mask = build_candidate_mask({1, 2, 3, 4, 5}, 7, 'cpu')

        filtered = filter_history_to_candidates(sequences, candidate_mask)
        valid = eligible_warm_start_rows(filtered, targets, candidate_mask)
        mask_ranking_scores(scores, filtered, candidate_mask)

        self.assertEqual(valid.tolist(), [True, False])
        self.assertTrue(torch.isneginf(scores[:, 0]).all())
        self.assertTrue(torch.isneginf(scores[0, 1:3]).all())
        self.assertTrue(torch.isneginf(scores[1, 4]))
        self.assertTrue(torch.isneginf(scores[:, 6]).all())

    def test_attention_ignores_padding_keys(self):
        torch.manual_seed(42)
        attention = MultiHeadedAttention(2, 8, 0.0).eval()
        hidden = torch.randn(1, 3, 8)
        changed_padding = hidden.clone()
        changed_padding[:, 0] = 1000.0
        mask = torch.tensor([[0.0, 1.0, 1.0]])

        actual = attention(hidden, hidden, hidden, mask)
        changed = attention(
            changed_padding, changed_padding, changed_padding, mask
        )

        self.assertTrue(torch.allclose(actual[:, 1:], changed[:, 1:], atol=1e-5))

    def test_lazy_train_dataset_matches_prefix_expansion(self):
        dataset = TrainDataset([[1, 2, 3], [4, 5, 6, 7]], max_len=3)
        actual = [(tokens.tolist(), label.tolist()) for tokens, label in dataset]
        expected = [
            ([0, 0, 1], [2]),
            ([0, 1, 2], [3]),
            ([0, 0, 4], [5]),
            ([0, 4, 5], [6]),
            ([4, 5, 6], [7]),
        ]
        self.assertEqual(actual, expected)

    def test_one_training_batch_denoising_and_coverage(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

        args = SimpleNamespace(
            item_num=6,
            hidden_size=8,
            max_len=3,
            emb_dropout=0.0,
            dropout=0.0,
            num_blocks=1,
            lambda_uncertainty=0.001,
            schedule_sampler_name="uniform",
            diffusion_steps=2,
            noise_schedule="trunc_lin",
            rescale_timesteps=True,
            coverage_candidate_items={1, 2, 3, 4, 5},
        )
        diffusion = create_model_diffu(args)
        model = AttDiffuseModel(diffusion, args).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        sequences = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
        targets = torch.tensor([[3], [4]], dtype=torch.long)

        weights_before = model.item_embeddings.weight.detach().clone()
        model.train()
        optimizer.zero_grad()
        _, representation, _, _, _, _ = model(sequences, targets, train_flag=True)
        loss = model.loss_diffu_ce(representation, targets)
        loss.backward()
        self.assertTrue(torch.equal(
            model.item_embeddings.weight.grad[6], torch.zeros(8)
        ))
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(
            torch.equal(weights_before, model.item_embeddings.weight.detach())
        )
        self.assertTrue(torch.equal(model.item_embeddings.weight[0], torch.zeros(8)))

        model.eval()
        with torch.no_grad():
            _, representation, _, _, _, _ = model(
                sequences,
                targets,
                train_flag=False,
            )
            scores = model.diffu_rep_pre(representation)
            scores[:, 0] = -torch.inf
            for row, history in enumerate(sequences):
                seen = history[history > 0]
                scores[row, seen] = -torch.inf
            predicted = torch.topk(scores, k=2, dim=-1).indices.tolist()

        candidate_items = set(range(1, args.item_num + 1))
        actual = [[3], [4]]
        *_, coverages = MODULES["DiffuRec"].compute_all_metrics(
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
