"""
Verify that packaged Kaggle source runs independently of the local source tree.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

# ------------------------------------------------------------------------------
# Import Local Functionality

from training.utils.package import prepare_kernel


# ------------------------------------------------------------------------------
# Remote packaging tests


class PackagingTests(unittest.TestCase):
    """
    Group checks for package behaviour.
    """

    def test_generated_kernel_runs_standalone(self):
        # ----------------------------------------------------------------------
        # Prepare a temporary project and package its remote source

        """
        Verify generated kernel runs standalone.
        """
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            shutil.copytree(root / "src", staging / "src")
            (staging / ".env").write_text(
                "KAGGLE_API_TOKEN=excluded-test-secret", encoding="utf-8"
            )
            metadata = {"id": "example/training-template", "is_private": True}
            raw = staging / "raw"
            raw.mkdir()
            (raw / "train_transaction.csv").write_text(
                "TransactionID,isFraud,amount\n1,0,10\n2,1,20\n",
                encoding="utf-8",
            )
            (raw / "train_identity.csv").write_text(
                "TransactionID,device\n1,desktop\n", encoding="utf-8"
            )
            folder = prepare_kernel(
                staging,
                metadata,
                job_args=[
                    "--nodes",
                    "data_process",
                    "--input-dir",
                    str(raw),
                    "--output-dir",
                    str(staging / "results"),
                ],
            )

            # ------------------------------------------------------------------
            # Execute without importing the original project checkout

            result = subprocess.run(
                [sys.executable, "-I", str(folder / "run.py")],
                cwd=staging,
                capture_output=True,
                text=True,
                check=True,
            )

            # ------------------------------------------------------------------
            # Verify configuration, output and source exclusions

            self.assertIn("Selected pipeline nodes completed.", result.stdout)
            self.assertTrue((staging / "results/training.csv").is_file())
            saved = json.loads((folder / "kernel-metadata.json").read_text())
            self.assertTrue(saved["is_private"])
            self.assertEqual(saved["code_file"], "run.py")
            script = (folder / "run.py").read_text()
            self.assertIn("cfg/training.yaml", script)
            self.assertNotIn("def prepare_kernel", script)
            self.assertNotIn("YOUR_KAGGLE_USERNAME", script)
            self.assertNotIn("excluded-test-secret", script)
            self.assertNotIn("class Settings", script)
            self.assertIn("job/utils/logging.py", script)

    def test_custom_build_directory(self):
        # ----------------------------------------------------------------------
        # Verify the configured build destination is respected

        """
        Verify custom build directory.
        """
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "custom"
            self.assertEqual(prepare_kernel(root, {}, destination), destination)
            self.assertTrue((destination / "run.py").is_file())

    def test_missing_entry_point_is_rejected(self):
        # ----------------------------------------------------------------------
        # Reject a project without executable remote source

        """
        Verify missing entry point is rejected.
        """
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                ValueError, "Missing remote entry point"
            ):
                prepare_kernel(Path(directory), {})
