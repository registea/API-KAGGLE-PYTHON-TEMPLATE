"""
Verify only successful, matching run artifacts reach leaderboard submission.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

# ------------------------------------------------------------------------------
# Import Local Functionality

from training.main import main
from training.utils.competition import (
    submit_predictions,
    validate_artifact,
    wait_for_success,
)


class CompetitionTests(unittest.TestCase):
    """
    Group checks for competition behaviour.
    """

    def make_artifacts(self, root, run_id="current"):
        """
        Create a prediction CSV and matching manifest for submission tests.
        """
        data = b"TransactionID,isFraud\n1,0.25\n"
        (root / "submission.csv").write_bytes(data)
        manifest = {
            "run_id": run_id,
            "competition": "ieee-fraud-detection",
            "rows": 1,
            "columns": ["TransactionID", "isFraud"],
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        (root / "submission-manifest.json").write_text(json.dumps(manifest))

    def test_stale_or_modified_artifact_is_rejected(self):
        """
        Verify stale or modified artifact is rejected.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_artifacts(root)
            with self.assertRaisesRegex(ValueError, "do not match"):
                validate_artifact(root, "another-run", "ieee-fraud-detection")
            (root / "submission.csv").write_text("modified")
            with self.assertRaisesRegex(ValueError, "checksum"):
                validate_artifact(root, "current", "ieee-fraud-detection")

    def test_wait_polls_until_complete(self):
        """
        Verify wait polls until complete.
        """
        responses = [
            subprocess.CompletedProcess(
                [], 0, 'x has status "KernelWorkerStatus.RUNNING"'
            ),
            subprocess.CompletedProcess(
                [], 0, 'x has status "KernelWorkerStatus.COMPLETE"'
            ),
        ]
        with patch(
            "training.utils.competition.subprocess.run", side_effect=responses
        ) as run:
            with patch("training.utils.competition.time.sleep"):
                wait_for_success("owner/kernel", Path.cwd(), 60)
        self.assertEqual(run.call_count, 2)

    def test_failed_job_does_not_download_or_submit(self):
        """
        Verify failed job does not download or submit.
        """
        with patch(
            "training.utils.competition.wait_for_success",
            side_effect=ValueError("ERROR"),
        ):
            with patch("training.utils.competition.subprocess.run") as run:
                with self.assertRaises(ValueError):
                    submit_predictions(
                        "a/b",
                        Path.cwd(),
                        Path.cwd(),
                        "current",
                        "ieee-fraud-detection",
                        "message",
                        60,
                    )
                run.assert_not_called()

    def test_unknown_and_failed_statuses_are_rejected(self):
        """
        Verify unknown and failed statuses are rejected.
        """
        for output in ('x has status "ERROR"', "unrecognised response"):
            with patch(
                "training.utils.competition.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, output),
            ):
                with self.assertRaises(ValueError):
                    wait_for_success("a/b", Path.cwd(), 60)

    def test_verified_artifact_is_submitted_once(self):
        """
        Verify verified artifact is submitted once.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_cli(command, **kwargs):
                """
                Simulate Kaggle commands without making a remote submission.
                """
                if command[1:3] == ["kernels", "output"]:
                    self.make_artifacts(Path(command[command.index("-p") + 1]))
                return subprocess.CompletedProcess(command, 0)

            with patch("training.utils.competition.wait_for_success"):
                with patch(
                    "training.utils.competition.subprocess.run",
                    side_effect=fake_cli,
                ) as run:
                    submit_predictions(
                        "a/b",
                        root,
                        root,
                        "current",
                        "ieee-fraud-detection",
                        "example",
                        60,
                    )
            self.assertEqual(run.call_count, 2)
            self.assertEqual(
                run.call_args.args[0][:4],
                ["kaggle", "competitions", "submit", "ieee-fraud-detection"],
            )

    def test_stale_download_cannot_reach_competition_submission(self):
        """
        Verify stale download cannot reach competition submission.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_cli(command, **kwargs):
                """
                Simulate Kaggle commands without making a remote submission.
                """
                self.make_artifacts(
                    Path(command[command.index("-p") + 1]), run_id="old"
                )

            with patch("training.utils.competition.wait_for_success"):
                with patch(
                    "training.utils.competition.subprocess.run",
                    side_effect=fake_cli,
                ) as run:
                    with self.assertRaisesRegex(ValueError, "do not match"):
                        submit_predictions(
                            "a/b",
                            root,
                            root,
                            "current",
                            "ieee-fraud-detection",
                            "example",
                            60,
                        )
            self.assertEqual(run.call_count, 1)

    def test_launcher_flag_selects_all_nodes_and_invokes_submission_helper(
        self,
    ):
        """
        Verify launcher flag selects all nodes and invokes submission helper.
        """
        config = {
            "environment": {"metadata_path": "metadata.json"},
            "submission": {
                "build_path": "build/kaggle",
                "output_path": "outputs",
            },
        }
        metadata = {
            "id": "example/kernel",
            "title": "Kernel",
            "competition_sources": ["ieee-fraud-detection"],
        }
        with patch.dict(
            os.environ,
            {"CI": "true", "KAGGLE_API_TOKEN": "test-token"},
            clear=True,
        ):
            with patch(
                "training.main.get_config",
                side_effect=[
                    config,
                    metadata,
                    {"competition": "ieee-fraud-detection"},
                ],
            ):
                with patch(
                    "training.main.prepare_kernel",
                    return_value=Path("build/kaggle"),
                ) as package:
                    with patch("training.main.subprocess.run"):
                        with patch(
                            "training.main.submit_predictions"
                        ) as submit:
                            main(
                                [
                                    "submit",
                                    "--submit-predictions",
                                    "--no-follow",
                                ]
                            )
        arguments = package.call_args.kwargs["job_args"]
        self.assertEqual(
            arguments[:4], ["--nodes", "data_process", "fit", "score"]
        )
        self.assertIn("--run-id", arguments)
        submit.assert_called_once()
        self.assertEqual(
            submit.call_args.args[3], arguments[arguments.index("--run-id") + 1]
        )
