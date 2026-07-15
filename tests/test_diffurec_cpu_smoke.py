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
        return module.create_model_diffu, module.Att_Diffuse_model
    finally:
        sys.path.remove(str(source_dir))


create_model_diffu, AttDiffuseModel = load_diffurec_components()


class DiffuRecCpuSmokeTest(unittest.TestCase):
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
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(
            torch.equal(weights_before, model.item_embeddings.weight.detach())
        )

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
