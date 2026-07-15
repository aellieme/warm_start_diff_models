import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_tools.run_demo_experiments import ALL_MODELS, build_commands


class DemoWorkflowTests(unittest.TestCase):
    def test_one_epoch_command_matrix_contains_every_model(self):
        commands = build_commands(["ml-1m"], [50], 1, set(ALL_MODELS), seed=42)
        names = {model for model, _, _ in commands}
        self.assertEqual(names, set(ALL_MODELS) | {"T-DiffRec preprocessing"})
        self.assertTrue(all(Path(cwd).is_dir() for _, cwd, _ in commands))
        for model, _, command in commands:
            if model not in {"TopPopular", "Random", "T-DiffRec preprocessing"}:
                joined = " ".join(command)
                self.assertTrue("--epochs 1" in joined or "--num_epochs 1" in joined or "final_epochs=1" in joined)

    def test_full_matrix_has_expected_number_of_commands(self):
        commands = build_commands(
            ["ml-1m", "amazon_Baby", "amazon_Toys_and_Games"],
            [50, 100], 250, set(ALL_MODELS), seed=42,
        )
        self.assertEqual(len(commands), 36)


if __name__ == "__main__":
    unittest.main()
