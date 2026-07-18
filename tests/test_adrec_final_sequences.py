"""Regression tests for ADRec final train+validation sequences."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADREC_SOURCE = ROOT / "src" / "ADRec" / "src"


def load_adrec_utils_module():
    sys.path.insert(0, str(ADREC_SOURCE))
    try:
        spec = importlib.util.spec_from_file_location(
            "adrec_utils_for_sequence_test", ADREC_SOURCE / "utils.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load ADRec utils.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(ADREC_SOURCE))


ADREC_UTILS = load_adrec_utils_module()


class ADRecFinalSequenceTest(unittest.TestCase):
    def test_train_history_is_not_duplicated_when_validation_exists(self):
        data_raw = {
            "train_dict": {7: [1, 2]},
            # get_data.py stores train history together with validation history.
            "val_seq_dict": {7: [1, 2, 3]},
            "val_tgt_dict": {7: 4},
        }

        self.assertEqual(
            ADREC_UTILS.build_final_train_sequences(data_raw), [[1, 2, 3, 4]]
        )

    def test_user_without_validation_keeps_train_history(self):
        data_raw = {
            "train_dict": {7: [1, 2]},
            "val_seq_dict": {},
            "val_tgt_dict": {},
        }

        self.assertEqual(ADREC_UTILS.build_final_train_sequences(data_raw), [[1, 2]])

    def test_test_interactions_are_never_added(self):
        data_raw = {
            "train_dict": {7: [1, 2]},
            "val_seq_dict": {7: [1, 2, 3]},
            "val_tgt_dict": {7: 4},
            "test_seq_dict": {7: [1, 2, 3, 4, 5]},
            "test_tgt_dict": {7: 6},
        }

        sequences = ADREC_UTILS.build_final_train_sequences(data_raw)

        self.assertEqual(sequences, [[1, 2, 3, 4]])
        self.assertNotIn(5, sequences[0])
        self.assertNotIn(6, sequences[0])

    def test_real_preprocessing_produces_clean_train_plus_validation(self):
        spec = importlib.util.spec_from_file_location(
            "adrec_get_data_for_sequence_test", ADREC_SOURCE / "get_data.py"
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load ADRec get_data.py")
        get_data = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(get_data)
        frame = pd.DataFrame({
            "userid": ["user"] * 10,
            "movieid": list(range(1, 11)),
            "timestamp": list(range(1, 11)),
        })

        with tempfile.TemporaryDirectory() as directory:
            path = get_data.save_dataset_with_gts(
                "synthetic", frame, output_dir=directory
            )
            with open(path, "rb") as handle:
                data_raw = pickle.load(handle)

        sequences = ADREC_UTILS.build_final_train_sequences(data_raw)

        self.assertEqual(data_raw["train_dict"][0], list(range(1, 8)))
        self.assertEqual(data_raw["val_seq_dict"][0], list(range(1, 8)))
        self.assertEqual(data_raw["val_tgt_dict"][0], 8)
        self.assertEqual(sequences, [list(range(1, 9))])
        self.assertNotIn(9, sequences[0])
        self.assertNotIn(10, sequences[0])


if __name__ == "__main__":
    unittest.main()
