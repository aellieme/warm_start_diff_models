"""One-batch CPU training smoke test for the real ADRec model."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
import unittest

import torch

try:
    from test_coverage import MODULES
except ModuleNotFoundError:
    from tests.test_coverage import MODULES


ROOT = Path(__file__).resolve().parents[1]


def load_adrec_model_class():
    source_dir = ROOT / "src" / "ADRec" / "src"
    path = source_dir / "model.py"
    sys.path.insert(0, str(source_dir))
    unused_diffurec = ModuleType("diffurec")
    unused_diffurec.DiffuRec = object
    unused_dreamrec = ModuleType("dreamrec")
    unused_dreamrec.DreamRec = object
    previous_diffurec = sys.modules.get("diffurec")
    previous_dreamrec = sys.modules.get("dreamrec")
    sys.modules["diffurec"] = unused_diffurec
    sys.modules["dreamrec"] = unused_dreamrec
    try:
        spec = importlib.util.spec_from_file_location("adrec_smoke_model", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load ADRec from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Att_Diffuse_model
    finally:
        if previous_diffurec is None:
            sys.modules.pop("diffurec", None)
        else:
            sys.modules["diffurec"] = previous_diffurec
        if previous_dreamrec is None:
            sys.modules.pop("dreamrec", None)
        else:
            sys.modules["dreamrec"] = previous_dreamrec
        sys.path.remove(str(source_dir))


AttDiffuseModel = load_adrec_model_class()


class ADRecCpuSmokeTest(unittest.TestCase):
    def test_one_training_batch_denoising_and_coverage(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)

        args = SimpleNamespace(
            model="adrec",
            item_num=6,
            hidden_size=8,
            pretrained=False,
            freeze_emb=False,
            emb_dropout=0.0,
            dropout=0.0,
            geodesic=False,
            dif_decoder="mlp",
            lambda_uncertainty=0.001,
            schedule_sampler_name="uniform",
            diffusion_steps=2,
            noise_schedule="trunc_lin",
            beta_a=0.3,
            beta_b=10,
            rescale_timesteps=True,
            independent=True,
            cfg_scale=1.0,
            is_causal=True,
            dif_blocks=1,
        )
        model = AttDiffuseModel(args).cpu()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        sequences = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
        targets = torch.tensor([[0, 2, 3], [0, 3, 4]], dtype=torch.long)

        weights_before = model.item_embedding.weight.detach().clone()
        model.train()
        optimizer.zero_grad()
        out_seq, _, diffusion_loss = model(sequences, targets, train_flag=True)
        recommendation_loss = model.calculate_loss(out_seq, targets)
        loss = recommendation_loss + diffusion_loss
        loss.backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertFalse(torch.equal(weights_before, model.item_embedding.weight.detach()))

        model.eval()
        with torch.no_grad():
            _, last_item, _ = model(sequences, targets, train_flag=False)
            scores = model.calculate_score(last_item)
            for row, history in enumerate(sequences):
                seen = history[history > 0]
                scores[row, seen] = -torch.inf
            predicted = torch.topk(scores, k=2, dim=-1).indices.tolist()

        candidate_items = set(range(1, args.item_num + 1))
        actual = [[3], [4]]
        *_, coverages = MODULES["ADRec"].compute_all_metrics(
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
