"""
Exercise processing, fitting and selective routing with synthetic competition
data.
"""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

# ------------------------------------------------------------------------------
# Import Local Functionality

from training.job.data_process import run as process_data
from training.job.execute import main
from training.job.fit import run as fit_model
from training.job.utils.utils import get_config
from training.main import get_options
from training.job.utils.arguments import pipeline_argv


class PipelineTests(unittest.TestCase):
    """
    Group checks for pipeline behaviour.
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
        pd.DataFrame(
            {
                "TransactionID": range(20),
                "device": ["desktop"] * 20,
            }
        ).to_csv(self.raw / "train_identity.csv", index=False)

    def test_processing_preserves_unmatched_transactions(self):
        """
        Verify processing preserves unmatched transactions.
        """
        result = process_data(self.raw, self.output, self.config, max_rows=30)
        frame = pd.read_csv(result)
        self.assertEqual(len(frame), 30)
        self.assertEqual(frame["device"].isna().sum(), 10)

    def test_duplicate_identity_ids_fail(self):
        """
        Verify duplicate identity ids fail.
        """
        pd.DataFrame({"TransactionID": [0, 0], "device": ["a", "b"]}).to_csv(
            self.raw / "train_identity.csv",
            index=False,
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            process_data(self.raw, self.output, self.config)

    def test_both_nodes_run_in_dependency_order(self):
        """
        Verify both nodes run in dependency order.
        """
        main(
            [
                "--nodes",
                "fit",
                "data_process",
                "--input-dir",
                str(self.raw),
                "--output-dir",
                str(self.output),
            ]
        )
        self.assertTrue((self.output / "training.csv").is_file())
        metrics = json.loads((self.output / "metrics.json").read_text())
        self.assertEqual(metrics["feature"], "amount")
        self.assertEqual(metrics["train_rows"] + metrics["validation_rows"], 40)
        model = joblib.load(self.output / "model.joblib")
        self.assertEqual(
            model.predict_proba(pd.DataFrame({"amount": [2.0]})).shape, (1, 2)
        )

    def test_processing_only_does_not_fit(self):
        """
        Verify processing only does not fit.
        """
        main(
            [
                "--nodes",
                "data_process",
                "--input-dir",
                str(self.raw),
                "--output-dir",
                str(self.output),
                "--max-rows",
                "12",
            ]
        )
        self.assertEqual(len(pd.read_csv(self.output / "training.csv")), 12)
        self.assertFalse((self.output / "model.joblib").exists())

    def test_fit_only_reads_existing_data_without_raw_inputs(self):
        """
        Verify fit only reads existing data without raw inputs.
        """
        prepared = process_data(self.raw, self.output, self.config)
        models = self.root / "models"
        main(
            [
                "--nodes",
                "fit",
                "--processed-data",
                str(prepared),
                "--input-dir",
                str(self.root / "missing"),
                "--output-dir",
                str(models),
            ]
        )
        self.assertTrue((models / "model.joblib").is_file())
        self.assertFalse((models / "training.csv").exists())

    def test_missing_processed_data_has_actionable_error(self):
        """
        Verify missing processed data has actionable error.
        """
        with self.assertRaisesRegex(FileNotFoundError, "--processed-data"):
            fit_model(self.root / "missing.csv", self.output, self.config)

    def test_conflicting_processed_path_is_rejected(self):
        """
        Verify conflicting processed path is rejected.
        """
        with self.assertRaisesRegex(ValueError, "fit-only"):
            main(["--nodes", "data_process", "--processed-data", "other.csv"])

    def test_pipeline_arguments_roundtrip(self):
        """
        Verify pipeline arguments roundtrip.
        """
        options = get_options(
            [
                "prepare",
                "--nodes",
                "fit",
                "--processed-data",
                "/kaggle/input/data-job/training.csv",
                "--max-rows",
                "0",
            ]
        )
        self.assertEqual(
            pipeline_argv(options),
            [
                "--nodes",
                "fit",
                "--processed-data",
                "/kaggle/input/data-job/training.csv",
                "--max-rows",
                "0",
            ],
        )
