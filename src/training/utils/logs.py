"""
Stream Kaggle logs while hiding recognised notebook-rendering noise.
"""

from collections.abc import Iterable, Iterator
import os
from pathlib import Path
import re
import subprocess
import sys


# ----------------------------------------------------------------------------------------------------------------------
# Known Kaggle rendering messages

_RENDER_WARNING = re.compile(
    r"^/[^\r\n]*/(?:site|dist)-packages/"
    r"(?:mistune\.py|nbconvert/filters/filter_links\.py):\d+: "
    r"SyntaxWarning: invalid escape sequence"
)
_RENDER_PROGRESS = re.compile(
    r"^\[NbConvertApp\] (?:Converting notebook __script__\.ipynb to html"
    r"|Writing \d+ bytes to __results__\.html)\s*$"
)


def filter_logs(lines: Iterable[str]) -> Iterator[str]:
    """
    Filter recognised Kaggle notebook-rendering messages.

    Remove only known warning headers, their matching source lines and renderer
    progress. Preserve unrecognised messages and application errors.

    :param lines: Incoming log lines, including their original line endings.

    :return: Iterator yielding retained lines unchanged.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Filter warning headers and their specific accompanying source lines

    # Track whether the next line can be the source excerpt for a known warning
    pending_source = False
    for line in lines:
        if pending_source:
            # Suppress a source excerpt only immediately after its recognised warning.
            pending_source = False
            if line[:1].isspace() and line.strip().startswith(
                ("cells[i][c] = re.sub(", "text = re.sub(")
            ):
                continue

        # A recognised warning may be followed by one indented library source line
        if _RENDER_WARNING.match(line):
            pending_source = True
            continue

        # Hide Kaggle's HTML conversion progress while retaining job output
        if _RENDER_PROGRESS.match(line):
            continue

        yield line


def stream_logs(kernel_id: str, root: Path, raw: bool = False) -> None:
    """
    Forward Kaggle job logs to the local console.

    Optionally filter known rendering noise and propagate CLI failures. Interrupting the viewer stops the local process
    without cancelling the remote job.

    :param kernel_id: Kaggle kernel identifier in owner/slug form.
    :param root: Working directory for the Kaggle CLI.
    :param raw: Whether to display all lines without filtering.

    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Start the local log viewer with line-buffered Python output

    # Build the command as tokens so the kernel identifier is never shell-interpreted
    command = ["kaggle", "kernels", "logs", kernel_id, "--follow"]
    environment = dict(os.environ, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(
        command,
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Forward raw or filtered lines immediately

    try:
        assert process.stdout is not None

        # Raw mode remains available when platform diagnostics are needed
        lines = process.stdout if raw else filter_logs(process.stdout)
        for line in lines:
            sys.stdout.write(line)
            sys.stdout.flush()
        returncode = process.wait()

        # Surface viewer failures after all available output is forwarded
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    finally:
        # Stop only the local viewer when interrupted; never cancel the Kaggle job.
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process.stdout is not None:
            process.stdout.close()
