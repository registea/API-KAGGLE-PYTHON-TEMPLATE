"""
Verify scoring includes the entire test set in sample-submission order.
"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

# ------------------------------------------------------------------------------
# Import Local Functionality

from training.job.execute import main
from training.job.score import run as score
from training.job.utils.utils import get_config


class ScoreTests(unittest.TestCase):
    """
    Group checks for score behaviour.
    """

    def setUp(self):
        """
        Create isolated input files and output paths for each test.
        """
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = self.root / "raw"
        self.output = self.root / "output"
        self.raw.mkdir()
        self.config = get_config(
            Path(__file__).parents[1] / "job/cfg/training.yaml"
        )
        pd.DataFrame(
            {
                "TransactionID": range(40),
                "isFraud": [0, 1] * 20,
                "amount": [float(i % 7) for i in range(40)],
            }
        ).to_csv(self.raw / "train_transaction.csv", index=False)
        pd.DataFrame({"TransactionID": range(40), "device": ["a"] * 40}).to_csv(
            self.raw / "train_identity.csv",
            index=False,
        )
        pd.DataFrame(
            {"TransactionID": range(100, 112), "amount": range(12)}
        ).to_csv(
            self.raw / "test_transaction.csv",
            index=False,
        )
        pd.DataFrame(
            {
                "TransactionID": list(reversed(range(100, 112))),
                "isFraud": [0.0] * 12,
            }
        ).to_csv(
            self.raw / "sample_submission.csv",
            index=False,
        )

    def train(self):
        """
        Create a saved model from the temporary training inputs.
        """
        main(
            [
                "--nodes",
                "data_process",
                "fit",
                "--input-dir",
                str(self.raw),
                "--output-dir",
                str(self.output),
                "--max-rows",
                "20",
            ]
        )

    def test_end_to_end_scoring_uses_all_test_rows_and_sample_order(self):
        """
        Verify end to end scoring uses all test rows and sample order.
        """
        main(
            [
                "--nodes",
                "data_process",
                "fit",
                "score",
                "--input-dir",
                str(self.raw),
                "--output-dir",
                str(self.output),
                "--max-rows",
                "10",
                "--run-id",
                "test-run",
            ]
        )
        submission = pd.read_csv(self.output / "submission.csv")
        self.assertEqual(list(submission.columns), ["TransactionID", "isFraud"])
        self.assertEqual(
            submission["TransactionID"].tolist(),
            list(reversed(range(100, 112))),
        )
        self.assertTrue(submission["isFraud"].between(0, 1).all())
        manifest = json.loads(
            (self.output / "submission-manifest.json").read_text()
        )
        self.assertEqual(manifest["run_id"], "test-run")
        self.assertEqual(manifest["rows"], 12)
        self.assertEqual(
            manifest["sha256"],
            hashlib.sha256(
                (self.output / "submission.csv").read_bytes()
            ).hexdigest(),
        )
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(metrics["refit_rows"], 10)

    def test_score_only_uses_saved_model(self):
        """
        Verify score only uses saved model.
        """
        self.train()
        destination = self.root / "predictions"
        main(
            [
                "--nodes",
                "score",
                "--model-path",
                str(self.output / "model.joblib"),
                "--input-dir",
                str(self.raw),
                "--output-dir",
                str(destination),
            ]
        )
        self.assertTrue((destination / "submission.csv").exists())
        self.assertFalse((destination / "model.joblib").exists())

    def test_mismatched_test_identifiers_fail(self):
        """
        Verify mismatched test identifiers fail.
        """
        self.train()
        pd.DataFrame({"TransactionID": [999], "isFraud": [0]}).to_csv(
            self.raw / "sample_submission.csv",
            index=False,
        )
        with self.assertRaisesRegex(ValueError, "identifiers differ"):
            score(
                self.output / "model.joblib", self.raw, self.output, self.config
            )
        self.assertFalse((self.output / "submission.csv").exists())

    def test_duplicate_test_identifiers_fail(self):
        """
        Verify duplicate test identifiers fail.
        """
        self.train()
        pd.DataFrame({"TransactionID": [100, 100], "amount": [1, 2]}).to_csv(
            self.raw / "test_transaction.csv",
            index=False,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            score(
                self.output / "model.joblib", self.raw, self.output, self.config
            )

    def test_identity_predictor_column_names_are_normalised(self):
        # Exercise IEEE-CIS test identity names id-01 versus training id_01.
        """
        Verify identity predictor column names are normalised.
        """
        train = pd.read_csv(self.raw / "train_transaction.csv").drop(
            columns="amount"
        )
        train.to_csv(self.raw / "train_transaction.csv", index=False)
        pd.DataFrame({"TransactionID": range(40), "id_01": range(40)}).to_csv(
            self.raw / "train_identity.csv",
            index=False,
        )
        pd.DataFrame(
            {"TransactionID": range(100, 112), "id-01": range(12)}
        ).to_csv(
            self.raw / "test_identity.csv",
            index=False,
        )
        self.train()
        score(self.output / "model.joblib", self.raw, self.output, self.config)
        self.assertEqual(len(pd.read_csv(self.output / "submission.csv")), 12)
