"""One-batch CPU training smoke test for the real T-DiffRec components."""

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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODEL_DIR = ROOT / "src" / "DiffRec" / "T-DiffRec" / "models"
DNN = load_module("tdiffrec_smoke_dnn", MODEL_DIR / "DNN.py").DNN
DIFFUSION = load_module(
    "tdiffrec_smoke_diffusion", MODEL_DIR / "gaussian_diffusion.py"
)


class TDiffRecCpuSmokeTest(unittest.TestCase):
    def test_one_training_batch_sampling_and_coverage(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        device = torch.device("cpu")

        item_count = 6
        model = DNN(
            in_dims=[item_count, 8],
            out_dims=[8, item_count],
            emb_size=4,
            dropout=0.0,
        ).to(device)
        diffusion = DIFFUSION.GaussianDiffusion(
            DIFFUSION.ModelMeanType.START_X,
            noise_schedule="linear",
            noise_scale=0.1,
            noise_min=0.0001,
            noise_max=0.02,
            steps=3,
            device=device,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Binary interaction vectors for two users. Item 5 is test-only.
        histories = torch.tensor(
            [
                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        weights_before = next(model.parameters()).detach().clone()

        model.train()
        optimizer.zero_grad()
        losses = diffusion.training_losses(model, histories, reweight=True)
        loss = losses["loss"].mean()
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.equal(weights_before, next(model.parameters()).detach()))

        model.eval()
        with torch.no_grad():
            scores = diffusion.p_sample(
                model,
                histories,
                steps=2,
                sampling_noise=False,
            )
            scores[histories.nonzero(as_tuple=True)] = -torch.inf
            predicted = torch.topk(scores, k=2, dim=-1).indices.tolist()

        candidate_items = {0, 1, 2, 3, 4}
        actual = [[3], [5]]
        *_, coverages = MODULES["T-DiffRec"].compute_all_metrics(
            actual,
            predicted,
            [2],
            n_items=len(candidate_items),
            candidate_items=candidate_items,
        )

        self.assertEqual(scores.shape, histories.shape)
        self.assertEqual(len(predicted), 2)
        self.assertTrue(all(len(recs) == 2 for recs in predicted))
        self.assertGreaterEqual(coverages[0], 0.0)
        self.assertLessEqual(coverages[0], 1.0)


if __name__ == "__main__":
    unittest.main()
