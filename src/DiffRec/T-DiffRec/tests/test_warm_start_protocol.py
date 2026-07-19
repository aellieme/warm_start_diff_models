import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


SOURCE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_DIR))

import data_utils  # noqa: E402
import models.gaussian_diffusion as gd  # noqa: E402
from split_load_data_dp import GlobalTemporalSplitter  # noqa: E402


class BiasModel(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(size))

    def forward(self, values, timesteps):
        return self.bias.unsqueeze(0).expand_as(values)


class WarmStartProtocolTest(unittest.TestCase):
    def test_mapping_uses_train_validation_catalog_and_preserves_cold_target(self):
        df = pd.DataFrame({
            "userid": [1, 1, 2, 2, 3, 3, 4, 4, 4, 4],
            "movieid": [10, 11, 12, 13, 14, 15, 16, 99, 10, 100],
            "timestamp": list(range(1, 11)),
        })
        result = GlobalTemporalSplitter(df).split(train_p=0.7, val_p=0.1)
        _, item_map = result["maps"]

        self.assertIn(10, item_map)
        self.assertEqual(len(item_map), 8)
        mapped_user = result["maps"][0][4]
        self.assertEqual(result["test_targets"][mapped_user], -1)
        self.assertEqual(
            result["test_history"].tolist(),
            [[mapped_user, item_map[10]]],
        )

    def test_only_known_targets_with_nonempty_history_are_selected(self):
        inputs = sp.csr_matrix([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        history = inputs.copy()
        targets = np.array([2, 4, -1])
        candidates = np.array([True, True, True, False, False])

        selected, selected_history, selected_targets, users = (
            data_utils.select_eligible_rows(inputs, history, targets, candidates)
        )
        self.assertEqual(users.tolist(), [0])
        self.assertEqual(selected_targets.tolist(), [2])
        self.assertEqual(selected.shape, (1, 5))
        self.assertEqual(selected_history.getnnz(), 1)

    def test_diffusion_loss_ignores_future_dimensions(self):
        diffusion = gd.GaussianDiffusion(
            gd.ModelMeanType.START_X,
            "linear-var",
            0.1,
            0.0001,
            0.02,
            5,
            torch.device("cpu"),
        )
        model = BiasModel(4)
        values = torch.zeros((2, 4))
        candidate_mask = torch.tensor([True, True, False, False])
        loss = diffusion.training_losses(
            model, values, reweight=True, candidate_mask=candidate_mask
        )["loss"].mean()
        loss.backward()

        self.assertNotEqual(model.bias.grad[0].item(), 0.0)
        self.assertEqual(model.bias.grad[2:].tolist(), [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
