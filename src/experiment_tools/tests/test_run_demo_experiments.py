import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from run_demo_experiments import build_commands  # noqa: E402


class DemoRunnerTest(unittest.TestCase):
    @patch("run_demo_experiments.torch.cuda.is_available", return_value=True)
    def test_ml_1m_diffurec_presets(self, _):
        commands = build_commands(
            ["ml-1m"], [50, 100], {"DiffuRec"}, seed=42,
            amp=True, diffurec_eval_repeats=3,
            diffurec_lr=0.0015,
        )
        self.assertEqual(len(commands), 2)
        commands_by_maxlen = {
            command[command.index("--max_len") + 1]: command
            for _, _, command in commands
        }
        expected = {
            "50": {
                "batch_size": "1024", "epochs": "26", "schedule": "cosine"
            },
            "100": {
                "batch_size": "512", "epochs": "55", "schedule": "trunc_lin"
            },
        }
        for maxlen, values in expected.items():
            command = commands_by_maxlen[maxlen]
            self.assertIn("--amp", command)
            self.assertEqual(
                command[command.index("--batch_size") + 1], values["batch_size"]
            )
            self.assertEqual(command[command.index("--epochs") + 1], values["epochs"])
            self.assertEqual(command[command.index("--hidden_size") + 1], "64")
            self.assertEqual(command[command.index("--num_blocks") + 1], "2")
            self.assertEqual(command[command.index("--lr") + 1], "0.0015")
            self.assertEqual(
                command[command.index("--noise_schedule") + 1], values["schedule"]
            )
            self.assertEqual(command[command.index("--eval_repeats") + 1], "3")
            self.assertEqual(command[command.index("--device") + 1], "cuda")

    @patch("run_demo_experiments.torch.cuda.is_available", return_value=True)
    def test_amazon_baby_presets(self, _):
        commands = build_commands(
            ["amazon_Baby"], [50, 100], {"DiffuRec"}, seed=42,
            amp=True,
        )
        self.assertEqual(len(commands), 2)

        commands_by_maxlen = {}
        for _, _, command in commands:
            commands_by_maxlen[command[command.index("--max_len") + 1]] = command

        expected = {
            "50": {"batch_size": "128", "epochs": "60"},
            "100": {"batch_size": "128", "epochs": "130"},
        }
        for maxlen, values in expected.items():
            command = commands_by_maxlen[maxlen]
            self.assertEqual(
                command[command.index("--batch_size") + 1], values["batch_size"]
            )
            self.assertEqual(command[command.index("--epochs") + 1], values["epochs"])
            self.assertEqual(command[command.index("--hidden_size") + 1], "64")
            self.assertEqual(command[command.index("--num_blocks") + 1], "2")
            self.assertEqual(command[command.index("--lr") + 1], "0.001")
            self.assertEqual(
                command[command.index("--noise_schedule") + 1], "trunc_lin"
            )
            self.assertIn("--amp", command)

    @patch("run_demo_experiments.torch.cuda.is_available", return_value=False)
    def test_amazon_toys_presets(self, _):
        commands = build_commands(
            ["amazon_Toys_and_Games"], [50, 100], {"DiffuRec"}, amp=True,
        )
        self.assertEqual(len(commands), 2)
        commands_by_maxlen = {
            command[command.index("--max_len") + 1]: command
            for _, _, command in commands
        }
        expected = {
            "50": {"batch_size": "128", "epochs": "60"},
            "100": {"batch_size": "256", "epochs": "100"},
        }
        for maxlen, values in expected.items():
            command = commands_by_maxlen[maxlen]
            self.assertEqual(
                command[command.index("--batch_size") + 1], values["batch_size"]
            )
            self.assertEqual(command[command.index("--epochs") + 1], values["epochs"])
            self.assertEqual(command[command.index("--hidden_size") + 1], "64")
            self.assertEqual(command[command.index("--num_blocks") + 1], "2")
            self.assertEqual(command[command.index("--lr") + 1], "0.003")
            self.assertEqual(
                command[command.index("--noise_schedule") + 1], "trunc_lin"
            )
            self.assertEqual(command[command.index("--device") + 1], "cpu")
            self.assertIn("--amp", command)

    @patch("run_demo_experiments.torch.cuda.is_available", return_value=False)
    def test_unknown_diffurec_combination_is_rejected(self, _):
        with self.assertRaisesRegex(ValueError, "No fixed DiffuRec preset"):
            build_commands(["ml-1m"], [25], {"DiffuRec"})

    @patch("run_demo_experiments.torch.cuda.is_available", return_value=True)
    def test_non_diffurec_epoch_budgets_are_internal(self, _):
        commands = build_commands(
            ["ml-1m"], [50], {"ADRec", "SASRec", "GPTRec", "T-DiffRec"}
        )
        by_model = {model: command for model, _, command in commands}
        self.assertEqual(
            by_model["ADRec"][by_model["ADRec"].index("--epochs") + 1], "250"
        )
        self.assertEqual(
            by_model["SASRec"][by_model["SASRec"].index("--num_epochs") + 1],
            "250",
        )
        self.assertIn("final_epochs=250", by_model["GPTRec"])
        self.assertEqual(
            by_model["T-DiffRec"][by_model["T-DiffRec"].index("--epochs") + 1],
            "250",
        )


if __name__ == "__main__":
    unittest.main()
