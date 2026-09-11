"""
Prepare and submit locally developed Python jobs to Kaggle.
"""

import argparse
from pathlib import Path, PurePosixPath
import re
import subprocess
import uuid


# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.arguments import add_pipeline_arguments, pipeline_argv
from training.job.utils.logging import setup_logger
from training.job.utils.utils import get_config
from training.utils.competition import submit_predictions
from training.utils.logs import stream_logs
from training.utils.metadata import resolve_metadata
from training.utils.package import prepare_kernel
from training.utils.settings import Settings


def get_options(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse local Kaggle actions, metadata overrides and pipeline arguments. Pipeline options share their definitions
    with the remote entry point.

    :param argv: Arguments to parse; None reads the current process command line.

    :return: Parsed submission and pipeline options.
    """
    parser = argparse.ArgumentParser(description=__doc__)

    # ------------------------------------------------------------------------------------------------------------------
    # Initial pipeline and configuration parameters

    # What action this script can complete
    parser.add_argument(
        "action", choices=["prepare", "submit", "status", "output"]
    )

    # Control kaggle logs
    parser.add_argument(
        "--follow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stream Kaggle logs after submission (default); "
            "--no-follow returns immediately."
        ),
    )

    # Show all logs, including those which Kaggle produces when converting the code to a script
    parser.add_argument(
        "--raw-logs",
        action="store_true",
        help="Show all Kaggle rendering messages.",
    )

    # Capture the root directory of the project for further configuration
    parser.add_argument("--project-root", type=Path, default=Path.cwd())

    # Location of the configuration path
    parser.add_argument(
        "--config-path", type=Path, default=Path("src/training/cfg/config.yaml")
    )

    # Conditional override arg - Save the metadata to a different path stated in the job/config
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="Override the configured metadata path.",
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Kernel identity and compute parameters

    # Unique id of the kernel being used for the run
    parser.add_argument(
        "--kernel-id",
        help="Full Kaggle username/kernel-slug; overrides environment.",
    )

    # A unique title which I might choose to give the run
    parser.add_argument("--title")

    # Flag whether the kernel is private of not in Kaggle
    parser.add_argument(
        "--private", dest="is_private", action=argparse.BooleanOptionalAction
    )

    # Determine whether to run this job using GPU compute
    parser.add_argument("--enable-gpu", action=argparse.BooleanOptionalAction)

    # Determine whether to enable the internet
    parser.add_argument(
        "--enable-internet", action=argparse.BooleanOptionalAction
    )

    # Determine the Kaggle accelerator
    parser.add_argument(
        "--machine-shape",
        help="Kaggle accelerator identifier supported by your account.",
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Attached data parameters

    # Providing a flag replaces its configured list; a flag without values clears it.
    for flag in (
        "dataset-sources",
        "competition-sources",
        "kernel-sources",
        "model-sources",
    ):
        parser.add_argument(f"--{flag}", nargs="*", default=None)

    # ------------------------------------------------------------------------------------------------------------------
    # Optional competition leaderboard submission

    # Determine if this run should submit any generated predictions
    parser.add_argument(
        "--submit-predictions",
        action="store_true",
        help=(
            "Wait for success, download scored predictions and submit them to the competition."
        ),
    )

    # Determine the competition this run is associated with
    parser.add_argument(
        "--competition",
        help="Competition slug; defaults to the single attached competition.",
    )

    # Provide a unique message associated with submission
    parser.add_argument(
        "--submission-message", default="Template pipeline predictions"
    )

    # How long to wait for the submission to register
    parser.add_argument(
        "--submission-timeout",
        type=int,
        default=43200,
        help="Status wait timeout in seconds.",
    )

    # Custom function to add relevant args to the job as a pass through
    add_pipeline_arguments(parser)

    # Return
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Run the requested Kaggle action and optional competition submission. Load local settings, validate overrides and
    package the remote job. Prediction submission waits for successful execution and verifies the downloaded artifact.

    :param argv: Arguments to parse; None reads the current process command line.

    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Set up logging and capture options

    # Set up our logger
    logger = setup_logger()

    # Get arguments passed to the script
    options = get_options(argv)

    # Extract the project root
    root = options.project_root.resolve()

    try:
        # --------------------------------------------------------------------------------------------------------------
        # Load environment variables and validate remote credentials early

        settings = Settings(logger=logger, project_root=root)
        if options.action != "prepare":
            settings.validate_authentication()

        # --------------------------------------------------------------------------------------------------------------
        # Resolve configuration and metadata overrides

        config = get_config(root / options.config_path)
        metadata_path = options.metadata_path or Path(
            config["environment"]["metadata_path"]
        )
        metadata = resolve_metadata(
            get_config(root / metadata_path), options, settings
        )
        kernel_id = metadata["id"]

        # --------------------------------------------------------------------------------------------------------------
        # Resolve optional leaderboard submission before starting a remote job

        competition: str | None = None
        if options.submit_predictions:
            if options.action != "submit":
                raise ValueError(
                    "--submit-predictions is only valid with the submit action."
                )
            if options.submission_timeout < 1:
                raise ValueError("--submission-timeout must be positive.")
            sources = metadata.get("competition_sources", [])
            competition = options.competition or (
                sources[0] if len(sources) == 1 else None
            )
            if not competition or not re.fullmatch(
                r"[A-Za-z0-9_-]+", competition
            ):
                raise ValueError(
                    "Specify --competition or exactly one competition source."
                )
            training_config = get_config(
                root / "src/training/job/cfg/training.yaml"
            )
            if competition != training_config.get("competition"):
                raise ValueError(
                    "--competition must match competition in "
                    "job/cfg/training.yaml."
                )
            if options.nodes is None:
                options.nodes = ["data_process", "fit", "score"]
            if "score" not in options.nodes:
                raise ValueError(
                    "--submit-predictions requires the score node."
                )
            if options.output_dir is not None:
                remote_output = PurePosixPath(options.output_dir.as_posix())
                if (
                    ".." in remote_output.parts
                    or not remote_output.is_relative_to("/kaggle/working")
                ):
                    raise ValueError(
                        "Automatic prediction submission requires outputs "
                        "under /kaggle/working."
                    )
            options.run_id = uuid.uuid4().hex

        # --------------------------------------------------------------------------------------------------------------
        # Prepare remote source code and select the Kaggle command

        if options.action in ("prepare", "submit"):
            folder = prepare_kernel(
                root,
                metadata,
                root / config["submission"]["build_path"],
                job_args=pipeline_argv(options),
            )
            logger.info("Prepared Kaggle script and metadata in %s", folder)
            if options.action == "prepare":
                return
            command = ["kaggle", "kernels", "push", "-p", str(folder)]
        elif options.action == "status":
            command = ["kaggle", "kernels", "status", kernel_id]
        else:
            output = root / config["submission"]["output_path"]
            command = [
                "kaggle",
                "kernels",
                "output",
                kernel_id,
                "-p",
                str(output),
            ]

        # --------------------------------------------------------------------------------------------------------------
        # Execute the command using inherited environment credentials

        logger.info("Running Kaggle action: %s", options.action)
        subprocess.run(command, cwd=root, check=True)

        # --------------------------------------------------------------------------------------------------------------
        # Stream remote execution logs after a successful submission

        if options.action == "submit" and options.follow:
            logger.info(
                "Job submitted. Streaming Kaggle logs; Ctrl+C stops "
                "viewing, not the remote job."
            )
            try:
                stream_logs(kernel_id, root, raw=options.raw_logs)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    "Submission succeeded, but log streaming failed. "
                    "Check the job on Kaggle or use the status command "
                    "before resubmitting."
                )
                raise SystemExit(exc.returncode) from exc
            except OSError as exc:
                logger.error(
                    "Submission succeeded, but the log viewer could not start."
                )
                raise SystemExit(1) from exc
            except KeyboardInterrupt:
                logger.info(
                    "Stopped viewing logs. The remote Kaggle job has "
                    "not been cancelled."
                )
                raise SystemExit(130) from None

        # --------------------------------------------------------------------------------------------------------------
        # Submit predictions only when explicitly requested and verified against this run

        if options.action == "submit" and options.submit_predictions:
            try:
                assert competition is not None
                submit_predictions(
                    kernel_id,
                    root,
                    root / config["submission"]["output_path"],
                    options.run_id,
                    competition,
                    options.submission_message,
                    options.submission_timeout,
                )
            except (OSError, ValueError, subprocess.CalledProcessError) as exc:
                logger.error(
                    "The compute job was submitted, but the prediction "
                    "submission workflow failed: %s. Check Kaggle before "
                    "retrying; no automatic resubmission was attempted.",
                    exc,
                )
                raise SystemExit(1) from exc
            except KeyboardInterrupt:
                logger.info(
                    "Stopped waiting for prediction submission; check "
                    "Kaggle for current state."
                )
                raise SystemExit(130) from None

    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.error("Unable to process Kaggle job: %s", exc)
        raise SystemExit(1) from exc
    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle command failed; see CLI output above.")
        raise SystemExit(exc.returncode) from exc


# ----------------------------------------------------------------------------------------------------------------------
# Script entry point

if __name__ == "__main__":
    main()
