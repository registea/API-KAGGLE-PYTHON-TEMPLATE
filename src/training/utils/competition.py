"""
Wait for a successful kernel run and submit its verified prediction artifact.
"""

import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

# ------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger


def wait_for_success(kernel_id: str, root: Path, timeout: int) -> None:
    """
    Wait for the remote kernel to complete successfully.

    Poll queued and running states until the deadline. Reject failed or unknown
    states before any prediction download or
    submission.

    :param kernel_id: Kaggle kernel identifier in owner/slug form.
    :param root: Working directory for the Kaggle CLI.
    :param timeout: Maximum wait duration in seconds.

    :return: None.
    """
    # --------------------------------------------------------------------------
    # Poll Kaggle within a bounded wait period

    logger = setup_logger()

    # A monotonic clock keeps the timeout stable if the system clock changes.
    deadline = time.monotonic() + timeout
    previous = None
    while time.monotonic() < deadline:
        # Request the current status without hiding CLI failures
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        # Extract the status from the human-readable Kaggle CLI response
        match = re.search(r'has status "([^"]+)"', result.stdout)
        if not match:
            raise ValueError(
                "Could not determine Kaggle job status; predictions "
                "were not submitted."
            )
        # Accept both plain status names and enum-qualified names returned by
        # the CLI.
        status = match.group(1).rsplit(".", 1)[-1].upper()
        # Log transitions once so polling does not flood the console
        if status != previous:
            logger.info("Kaggle job status: %s", status)
            previous = status
        # Continue only for active states and stop immediately after success
        if status == "COMPLETE":
            return
        if status not in {"QUEUED", "RUNNING"}:
            raise ValueError(
                f"Kaggle job status is {status}; predictions were "
                "not submitted."
            )
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    raise TimeoutError(
        "Timed out waiting for Kaggle; predictions were not submitted."
    )


def validate_artifact(folder: Path, run_id: str, competition: str) -> Path:
    """
    Validate downloaded predictions against their run manifest.

    Require one manifest, matching run and competition identifiers, and matching
    CSV checksum, columns and row count.

    :param folder: Directory containing downloaded prediction artifacts.
    :param run_id: Expected identifier assigned to this submission run.
    :param competition: Expected competition slug.

    :return: Path to the verified submission CSV.
    """
    # --------------------------------------------------------------------------
    # Locate and identify this run's prediction artifacts

    # Multiple manifests are ambiguous; never guess which predictions should be
    # uploaded.
    manifests = list(folder.rglob("submission-manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            "Expected exactly one submission manifest from the scoring node."
        )
    # Load the proof written alongside the predictions by the scoring node
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    if (
        manifest.get("run_id") != run_id
        or manifest.get("competition") != competition
    ):
        raise ValueError(
            "Downloaded predictions do not match this run and "
            "competition; refusing submission."
        )
    # Bind the CSV bytes to this run before trusting its shape or submitting it.
    output = manifests[0].parent / "submission.csv"
    if not output.is_file() or hashlib.sha256(
        output.read_bytes()
    ).hexdigest() != manifest.get("sha256"):
        raise ValueError(
            "Prediction file is missing or its checksum does not match."
        )
    # --------------------------------------------------------------------------
    # Validate the submission schema and row count

    with output.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != manifest.get("columns"):
            raise ValueError(
                "Prediction CSV columns do not match the manifest."
            )
        rows = sum(1 for _ in reader)
    if rows < 1 or rows != manifest.get("rows"):
        raise ValueError(
            "Prediction CSV row count does not match the manifest."
        )
    return output


def submit_predictions(
    kernel_id: str,
    root: Path,
    output_root: Path,
    run_id: str,
    competition: str,
    message: str,
    timeout: int,
) -> None:
    """
    Submit verified predictions after successful remote execution.

    Download into a fresh directory and verify the scoring manifest before
    invoking competition submission once. Failed uploads are not retried
    automatically.

    :param kernel_id: Kaggle kernel identifier in owner/slug form.
    :param root: Working directory for Kaggle CLI commands.
    :param output_root: Parent directory for a fresh artifact download
                        directory.
    :param run_id: Expected scoring-run identifier.
    :param competition: Competition slug to receive the predictions.
    :param message: Description attached to the leaderboard submission.
    :param timeout: Maximum time in seconds to wait for kernel success.

    :return: None.
    """
    # --------------------------------------------------------------------------
    # Wait independently of console log streaming

    # Do not download artifacts from a failed or incomplete remote run
    wait_for_success(kernel_id, root, timeout)

    # --------------------------------------------------------------------------
    # Download into a fresh directory so local stale files cannot be reused

    output_root.mkdir(parents=True, exist_ok=True)
    download_dir = Path(
        tempfile.mkdtemp(prefix="predictions-", dir=output_root)
    )
    subprocess.run(
        [
            "kaggle",
            "kernels",
            "output",
            kernel_id,
            "-p",
            str(download_dir),
            "--file-pattern",
            r"(^|/)(submission\.csv|submission-manifest\.json)$",
        ],
        cwd=root,
        check=True,
    )
    # Ensure only predictions from this exact run can reach the leaderboard
    predictions = validate_artifact(download_dir, run_id, competition)

    # --------------------------------------------------------------------------
    # Make the explicit leaderboard submission without automatic retries

    setup_logger().info(
        "Submitting verified predictions to competition %s", competition
    )
    subprocess.run(
        [
            "kaggle",
            "competitions",
            "submit",
            competition,
            "-f",
            str(predictions),
            "-m",
            message,
        ],
        cwd=root,
        check=True,
    )
