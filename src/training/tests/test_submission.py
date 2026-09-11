"""
Verify submission follows logs and respects log-viewing options.
"""

import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.main import main


# ----------------------------------------------------------------------------------------------------------------------
# Submission and log streaming tests


class SubmissionTests(unittest.TestCase):
    """
    Group checks for submission behaviour.
    """

    def invoke(self, arguments, submit_error=None, stream_error=None):
        """
        Run the launcher with mocked packaging, submission and log streaming.
        """
        root = Path.cwd()
        config = {
            "environment": {"metadata_path": "metadata.json"},
            "submission": {
                "build_path": "build/kaggle",
                "output_path": "outputs",
            },
        }
        metadata = {"id": "example/hello", "title": "Hello"}
        with patch.dict(
            os.environ,
            {"CI": "true", "KAGGLE_API_TOKEN": "test-token"},
            clear=True,
        ):
            with patch(
                "training.main.get_config", side_effect=[config, metadata]
            ):
                with patch(
                    "training.main.prepare_kernel",
                    return_value=root / "build/kaggle",
                ):
                    with patch(
                        "training.main.subprocess.run", side_effect=submit_error
                    ) as run:
                        with patch(
                            "training.main.stream_logs",
                            side_effect=stream_error,
                        ) as stream:
                            try:
                                main(arguments)
                            except SystemExit as exc:
                                return run, stream, exc.code
                            return run, stream, 0

    def test_submit_streams_filtered_logs_by_default(self):
        """
        Verify submit streams filtered logs by default.
        """
        run, stream, code = self.invoke(["submit"])
        self.assertEqual(code, 0)
        self.assertEqual(
            run.call_args.args[0][:3], ["kaggle", "kernels", "push"]
        )
        stream.assert_called_once_with("example/hello", Path.cwd(), raw=False)

    def test_raw_logs_option(self):
        """
        Verify raw logs option.
        """
        run, stream, code = self.invoke(["submit", "--raw-logs"])
        self.assertEqual(code, 0)
        stream.assert_called_once_with("example/hello", Path.cwd(), raw=True)

    def test_no_follow_returns_after_submission(self):
        """
        Verify no follow returns after submission.
        """
        run, stream, code = self.invoke(["submit", "--no-follow"])
        self.assertEqual(code, 0)
        run.assert_called_once()
        stream.assert_not_called()

    def test_failed_submission_does_not_start_log_viewer(self):
        """
        Verify failed submission does not start log viewer.
        """
        run, stream, code = self.invoke(
            ["submit"], submit_error=subprocess.CalledProcessError(1, "kaggle")
        )
        self.assertEqual(code, 1)
        run.assert_called_once()
        stream.assert_not_called()

    def test_stream_failure_does_not_resubmit(self):
        """
        Verify stream failure does not resubmit.
        """
        run, stream, code = self.invoke(
            ["submit"], stream_error=subprocess.CalledProcessError(2, "kaggle")
        )
        self.assertEqual(code, 2)
        run.assert_called_once()
        stream.assert_called_once()

    def test_interrupt_stops_viewing_without_remote_cancellation(self):
        """
        Verify interrupt stops viewing without remote cancellation.
        """
        run, stream, code = self.invoke(
            ["submit"], stream_error=KeyboardInterrupt()
        )
        self.assertEqual(code, 130)
        run.assert_called_once()
        stream.assert_called_once()
