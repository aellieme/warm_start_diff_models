import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from run_demo_experiments import build_commands  # noqa: E402


class DemoRunnerTest(unittest.TestCase):
    @patch("run_demo_experiments.torch.cuda.is_available", return_value=True)
    def test_tuned_diffurec_command(self, _):
        commands = build_commands(
            ["ml-1m"], [50], 50, {"DiffuRec"}, seed=42,
            amp=True, tuned_diffurec=True, diffurec_eval_repeats=3,
            diffurec_lr=0.0015,
        )
        self.assertEqual(len(commands), 1)
        _, _, command = commands[0]
        self.assertIn("--amp", command)
        self.assertEqual(command[command.index("--batch_size") + 1], "1024")
        self.assertEqual(command[command.index("--hidden_size") + 1], "64")
        self.assertEqual(command[command.index("--num_blocks") + 1], "2")
        self.assertEqual(command[command.index("--lr") + 1], "0.0015")
        self.assertEqual(command[command.index("--noise_schedule") + 1], "cosine")
        self.assertEqual(command[command.index("--epochs") + 1], "26")
        self.assertEqual(command[command.index("--eval_repeats") + 1], "3")
        self.assertEqual(command[command.index("--device") + 1], "cuda")

    @patch("run_demo_experiments.torch.cuda.is_available", return_value=True)
    def test_amazon_baby_presets(self, _):
        commands = build_commands(
            ["amazon_Baby"], [50, 100], 250, {"DiffuRec"}, seed=42,
            amp=True, tuned_diffurec=True,
        )
        self.assertEqual(len(commands), 2)

        commands_by_maxlen = {}
        for _, _, command in commands:
            commands_by_maxlen[command[command.index("--max_len") + 1]] = command

        expected = {
            "50": {"batch_size": "256", "epochs": "100"},
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
    def test_amazon_toys_maxlen_50_preset(self, _):
        commands = build_commands(
            ["amazon_Toys_and_Games"], [50], 250, {"DiffuRec"},
            amp=True, tuned_diffurec=True,
        )
        command = commands[0][2]
        self.assertEqual(command[command.index("--batch_size") + 1], "256")
        self.assertEqual(command[command.index("--epochs") + 1], "70")
        self.assertEqual(command[command.index("--hidden_size") + 1], "64")
        self.assertEqual(command[command.index("--num_blocks") + 1], "2")
        self.assertEqual(command[command.index("--lr") + 1], "0.003")
        self.assertEqual(
            command[command.index("--noise_schedule") + 1], "cosine"
        )
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertIn("--amp", command)

    @patch("run_demo_experiments.torch.cuda.is_available", return_value=False)
    def test_unknown_tuned_combination_is_rejected(self, _):
        with self.assertRaisesRegex(
            ValueError, "No fixed DiffuRec preset"
        ):
            build_commands(
                ["amazon_Toys_and_Games"], [100], 250, {"DiffuRec"},
                tuned_diffurec=True,
            )


if __name__ == "__main__":
    unittest.main()
