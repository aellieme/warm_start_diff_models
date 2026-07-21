"""One-batch CPU training smoke test for the real DiffuRec model."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
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


def load_evaluation_helpers():
    source_dir = ROOT / "src" / "DiffuRec" / "src"
    path = source_dir / "trainer.py"
    sys.path.insert(0, str(source_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "diffurec_smoke_trainer", path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load DiffuRec trainer from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return (
            module.evaluation_seeds,
            module.isolated_torch_rng,
            module.evaluate_stochastic_ranking,
            module._capture_rng_state,
            module._restore_rng_state,
            module.list_tuning_checkpoints,
            module.save_tuning_checkpoint,
            module.model_train,
        )
    finally:
        sys.path.remove(str(source_dir))


(
    TrainDataset,
    build_candidate_mask,
    eligible_warm_start_rows,
    encode_item_ids_with_padding,
    filter_history_to_candidates,
    mask_ranking_scores,
) = load_train_dataset()
(
    evaluation_seeds,
    isolated_torch_rng,
    evaluate_stochastic_ranking,
    capture_rng_state,
    restore_rng_state,
    list_tuning_checkpoints,
    save_tuning_checkpoint,
    model_train,
) = load_evaluation_helpers()


class DiffuRecCpuSmokeTest(unittest.TestCase):
    @staticmethod
    def tuning_args(batch_size=512):
        return SimpleNamespace(
            dataset="ml-1m",
            random_seed=42,
            max_len=100,
            device="cpu",
            num_gpu=1,
            batch_size=batch_size,
            hidden_size=64,
            dropout=0.1,
            emb_dropout=0.3,
            hidden_act="gelu",
            num_blocks=2,
            decay_step=100,
            gamma=0.1,
            metric_ks=[10, 20, 100],
            optimizer="Adam",
            lr=0.003,
            loss_lambda=0.001,
            weight_decay=0,
            momentum=None,
            schedule_sampler_name="lossaware",
            diffusion_steps=32,
            lambda_uncertainty=0.001,
            noise_schedule="trunc_lin",
            rescale_timesteps=True,
            eval_interval=5,
            patience=8,
            eval_repeats=5,
            amp=False,
        )

    def test_tuning_checkpoints_keep_only_two_newest_per_configuration(self):
        args = self.tuning_args(batch_size=512)
        other_args = self.tuning_args(batch_size=256)
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ,
            {"EXPERIMENT_OUTPUT_DIR": str(Path(directory) / "logs")},
        ):
            for epoch in (10, 20, 30):
                save_tuning_checkpoint({"completed_epoch": epoch}, args, epoch)
            save_tuning_checkpoint(
                {"completed_epoch": 10}, other_args, 10,
            )

            checkpoints = list_tuning_checkpoints(args)
            other_checkpoints = list_tuning_checkpoints(other_args)

            self.assertEqual(
                [path.stem[-9:] for path in checkpoints],
                ["epoch0020", "epoch0030"],
            )
            self.assertEqual(len(other_checkpoints), 1)
            self.assertTrue(all(path.is_file() for path in checkpoints))

    def test_training_rng_state_round_trip(self):
        import random

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        state = capture_rng_state()
        expected = (random.random(), np.random.rand(), torch.rand(3))

        random.seed(999)
        np.random.seed(999)
        torch.manual_seed(999)
        restore_rng_state(state)
        actual = (random.random(), np.random.rand(), torch.rand(3))

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))

    def test_tuning_resume_restores_optimizer_and_training_epoch(self):
        class TinyDiffuRec(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([0.0, 0.1, 0.2, 0.3]))

            def forward(self, sequence, target, train_flag=False):
                representation = self.logits.unsqueeze(0).expand(sequence.shape[0], -1)
                return None, representation, None, None, None, None

            def loss_diffu_ce(self, representation, target):
                return torch.nn.functional.cross_entropy(
                    representation, target.squeeze(-1)
                )

        class NullLogger:
            def info(self, *args, **kwargs):
                pass

        def make_args(root, epochs, resume_checkpoint=None):
            args = self.tuning_args(batch_size=2)
            args.epochs = epochs
            args.resume_checkpoint = resume_checkpoint
            args.log_file = str(Path(root) / "plots") + os.sep
            args.description = "resume_test"
            args.item_num = 3
            args.coverage_candidate_items = {1, 2, 3}
            args.train_item_popularity = {1: 1, 2: 1, 3: 1}
            args.eval_interval = 100
            return args

        sequence = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        target = torch.tensor([[2], [3]], dtype=torch.long)
        train_loader = [(sequence, target)]

        with tempfile.TemporaryDirectory() as directory:
            continuous_root = Path(directory) / "continuous"
            split_root = Path(directory) / "split"

            torch.manual_seed(42)
            continuous_model = TinyDiffuRec()
            continuous_args = make_args(continuous_root, epochs=20)
            with mock.patch.dict(
                os.environ,
                {"EXPERIMENT_OUTPUT_DIR": str(continuous_root / "logs")},
            ):
                model_train(
                    train_loader, [], None, continuous_model,
                    continuous_args, NullLogger(),
                )
                continuous_checkpoint = torch.load(
                    list_tuning_checkpoints(continuous_args)[-1],
                    map_location="cpu",
                    weights_only=False,
                )

            torch.manual_seed(42)
            first_model = TinyDiffuRec()
            first_args = make_args(split_root, epochs=10)
            with mock.patch.dict(
                os.environ,
                {"EXPERIMENT_OUTPUT_DIR": str(split_root / "logs")},
            ):
                model_train(
                    train_loader, [], None, first_model,
                    first_args, NullLogger(),
                )
                resumed_model = TinyDiffuRec()
                resumed_args = make_args(
                    split_root, epochs=20, resume_checkpoint="latest",
                )
                model_train(
                    train_loader, [], None, resumed_model,
                    resumed_args, NullLogger(),
                )
                resumed_checkpoint = torch.load(
                    list_tuning_checkpoints(resumed_args)[-1],
                    map_location="cpu",
                    weights_only=False,
                )

            self.assertEqual(continuous_checkpoint["completed_epoch"], 20)
            self.assertEqual(resumed_checkpoint["completed_epoch"], 20)
            self.assertTrue(torch.equal(
                continuous_checkpoint["model_state_dict"]["logits"],
                resumed_checkpoint["model_state_dict"]["logits"],
            ))

    def test_evaluation_seeds_are_fixed_and_predeclared(self):
        args = SimpleNamespace(random_seed=42, eval_repeats=5)
        self.assertEqual(
            evaluation_seeds(args),
            (42, 43, 44, 45, 46),
        )
        self.assertEqual(evaluation_seeds(args, repeats=1), (42,))

    def test_evaluation_rng_is_repeatable_and_does_not_advance_training_rng(self):
        torch.manual_seed(123)
        expected_before = torch.randn(4)
        expected_after = torch.randn(4)

        torch.manual_seed(123)
        actual_before = torch.randn(4)
        with isolated_torch_rng(999, "cpu"):
            evaluation_sample_one = torch.randn(4)
        actual_after = torch.randn(4)
        with isolated_torch_rng(999, "cpu"):
            evaluation_sample_two = torch.randn(4)

        self.assertTrue(torch.equal(actual_before, expected_before))
        self.assertTrue(torch.equal(actual_after, expected_after))
        self.assertTrue(torch.equal(evaluation_sample_one, evaluation_sample_two))

    def test_stochastic_ranking_repeats_are_reproducible(self):
        class StochasticRanker(torch.nn.Module):
            def forward(self, sequence, target, train_flag=False):
                representation = torch.randn(sequence.shape[0], 6)
                return None, representation, None, None, None, None

            def diffu_rep_pre(self, representation):
                return representation

        args = SimpleNamespace(
            device="cpu",
            random_seed=42,
            eval_repeats=3,
            metric_ks=[1, 2],
        )
        sequences = torch.tensor([[0, 1, 2], [0, 1, 3]])
        targets = torch.tensor([[3], [4]])
        data_loader = [(sequences, targets, sequences.clone())]
        candidate_items = {1, 2, 3, 4, 5}
        candidate_mask = build_candidate_mask(candidate_items, 6, "cpu")
        model = StochasticRanker().train()

        first = evaluate_stochastic_ranking(
            model, data_loader, args, candidate_items, candidate_mask
        )
        second = evaluate_stochastic_ranking(
            model, data_loader, args, candidate_items, candidate_mask
        )

        self.assertEqual(first["seeds"], [42, 43, 44])
        self.assertEqual(first["means"], second["means"])
        self.assertEqual(first["stds"], second["stds"])
        self.assertEqual(
            first["canonical_predicted"], second["canonical_predicted"]
        )
        self.assertTrue(model.training)

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
