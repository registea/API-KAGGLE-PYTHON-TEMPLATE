"""
Verify noise filtering preserves job output, warnings and failures.
"""

import io
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.utils.logs import filter_logs, stream_logs


# ----------------------------------------------------------------------------------------------------------------------
# Log filtering tests


class LogTests(unittest.TestCase):
    """
    Group checks for logs behaviour.
    """

    def test_known_renderer_noise_is_hidden(self):
        """
        Verify known renderer noise is hidden.
        """
        lines = [
            "Hello, world!\n",
            "/usr/local/lib/python3.12/dist-packages/mistune.py:435: "
            "SyntaxWarning: invalid escape sequence '\\|'\n",
            "  cells[i][c] = re.sub('pattern', 'replacement', cell)\n",
            "/usr/local/lib/python3.12/dist-packages/nbconvert/filters/"
            "filter_links.py:36: "
            "SyntaxWarning: invalid escape sequence '\\_'\n",
            "  text = re.sub('pattern', 'replacement', text)\n",
            "[NbConvertApp] Converting notebook __script__.ipynb to html\n",
            "[NbConvertApp] Writing 292113 bytes to __results__.html\n",
        ]
        self.assertEqual(list(filter_logs(lines)), ["Hello, world!\n"])

    def test_job_errors_and_unknown_platform_messages_are_preserved(self):
        """
        Verify job errors and unknown platform messages are preserved.
        """
        lines = [
            "Traceback (most recent call last):\n",
            "  File '/tmp/training/job/execute.py', line 42, in main\n",
            "ValueError: invalid input\n",
            "/tmp/training/job/model.py:1: SyntaxWarning: "
            "invalid escape sequence\n",
            "[NbConvertApp] ERROR: rendering failed\n",
            "UserWarning: check your dataset\n",
        ]
        self.assertEqual(list(filter_logs(lines)), lines)

    def test_warning_does_not_swallow_unrelated_next_line(self):
        """
        Verify warning does not swallow unrelated next line.
        """
        lines = [
            "/usr/local/lib/python3.12/dist-packages/mistune.py:435: "
            "SyntaxWarning: invalid escape sequence\n",
            "RuntimeError: training failed\n",
        ]
        self.assertEqual(list(filter_logs(lines)), [lines[1]])

    def test_raw_mode_and_filtering(self):
        """
        Verify raw mode and filtering.
        """
        source = (
            "Hello, world!\n"
            "[NbConvertApp] Writing 123 bytes to __results__.html\n"
        )
        for raw in (False, True):
            with self.subTest(raw=raw):
                process = MagicMock()
                process.stdout = io.StringIO(source)
                process.wait.return_value = 0
                process.poll.return_value = 0
                with patch(
                    "training.utils.logs.subprocess.Popen", return_value=process
                ):
                    with patch(
                        "training.utils.logs.sys.stdout",
                        new_callable=io.StringIO,
                    ) as output:
                        stream_logs("example/hello", Path.cwd(), raw=raw)
                        self.assertEqual(
                            output.getvalue(),
                            source if raw else "Hello, world!\n",
                        )

    def test_nonzero_exit_is_preserved(self):
        """
        Verify nonzero exit is preserved.
        """
        process = MagicMock()
        process.stdout = io.StringIO("Connection error\n")
        process.wait.return_value = 2
        process.poll.return_value = 2
        with patch(
            "training.utils.logs.subprocess.Popen", return_value=process
        ):
            with patch(
                "training.utils.logs.sys.stdout", new_callable=io.StringIO
            ):
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    stream_logs("example/hello", Path.cwd())
        self.assertEqual(error.exception.returncode, 2)
