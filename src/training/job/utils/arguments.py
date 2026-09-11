"""
Shared pipeline options for local execution and remote submission.
"""

import argparse
from pathlib import Path

# ------------------------------------------------------------------------------
# Pipeline command-line arguments


def add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register shared pipeline command-line options.

    Mutate the supplied parser so local execution and remote submission accept
    the same node and artifact options.

    :param parser: Argument parser to extend with pipeline options.

    :return: None.
    """
    # --------------------------------------------------------------------------
    # Register node selection and artifact overrides

    # Select complete nodes while keeping dependency ordering inside execute.py
    parser.add_argument(
        "--nodes",
        nargs="+",
        choices=["data_process", "fit", "score"],
        help=(
            "Nodes to execute; defaults to training YAML. "
            "Dependencies always run first."
        ),
    )
    # Allow each pipeline artifact path to be overridden for partial runs
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Raw transaction and identity CSV directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Stage outputs; defaults to Kaggle working/local outputs.",
    )
    parser.add_argument(
        "--processed-data",
        type=Path,
        help="Existing training.csv for a fit-only run.",
    )
    parser.add_argument(
        "--model-path", type=Path, help="Saved model for a score-only run."
    )
    # Inject a hidden run identifier so downloaded artifacts can be verified
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument(
        "--max-rows", type=int, help="Transaction row limit; 0 means all rows."
    )


def pipeline_argv(options: argparse.Namespace) -> list[str]:
    """
    Serialise pipeline options for the remote entry point.

    Forward explicitly supplied values, including zero, using portable path
    separators. Submission settings and credentials are not included.

    :param options: Parsed namespace containing the shared pipeline options.

    :return: Command-line tokens embedded in the generated Kaggle script.
    """
    # --------------------------------------------------------------------------
    # Serialise remote pipeline arguments

    # Forward only shared pipeline options; launcher credentials stay in the
    # local environment.
    arguments = []

    # Preserve the complete node selection as one multi-value option
    if options.nodes is not None:
        arguments.extend(["--nodes", *options.nodes])
    # Forward each explicitly supplied artifact or row-limit override
    for name in (
        "input_dir",
        "output_dir",
        "processed_data",
        "max_rows",
        "model_path",
        "run_id",
    ):
        value = getattr(options, name)
        # Zero is a valid row limit override, so do not use a truthiness check
        # here.
        if value is not None:
            arguments.extend(
                [
                    "--" + name.replace("_", "-"),
                    value.as_posix() if isinstance(value, Path) else str(value),
                ]
            )
    return arguments
