import sys
import unittest
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from experiment_tools.warm_start import build_last_item_examples  # noqa: E402


class LastItemExamplesTest(unittest.TestCase):
    def test_known_history_and_target_define_eligibility(self):
        history = pd.DataFrame({
            "user": [1, 1, 2],
            "item": [1, 2, 3],
            "time": [1, 2, 1],
        })
        test = pd.DataFrame({
            "user": [1, 1, 9, 9, 10, 11, 11],
            "item": [3, 2, 1, 2, 4, 1, 4],
            "time": [3, 4, 3, 4, 3, 3, 4],
        })

        users, histories, targets = build_last_item_examples(
            history, test, "user", "item", "time", {1, 2, 3}
        )

        self.assertEqual(users, [1, 9])
        self.assertEqual(histories, [[1, 2, 3], [1]])
        self.assertEqual(targets, [[2], [2]])


if __name__ == "__main__":
    unittest.main()
