import sys
import unittest
from pathlib import Path
from unittest.mock import patch


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from run_demo_experiments import build_commands  # noqa: E402


class DemoRunnerTest(unittest.TestCase):
    @patch("run_demo_experiments.torch.cuda.is_available", return_value=True)
    def test_fast_diffurec_command(self, _):
        commands = build_commands(
            ["ml-1m"], [50], 50, {"DiffuRec"}, seed=42,
            amp=True, fast_diffurec=True,
        )
        self.assertEqual(len(commands), 1)
        _, _, command = commands[0]
        self.assertIn("--amp", command)
        self.assertEqual(command[command.index("--batch_size") + 1], "1024")
        self.assertEqual(command[command.index("--hidden_size") + 1], "64")
        self.assertEqual(command[command.index("--num_blocks") + 1], "2")
        self.assertEqual(command[command.index("--lr") + 1], "0.003")
        self.assertEqual(command[command.index("--noise_schedule") + 1], "cosine")
        self.assertEqual(command[command.index("--device") + 1], "cuda")


if __name__ == "__main__":
    unittest.main()
