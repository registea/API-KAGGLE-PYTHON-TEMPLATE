"""
Route the selected nodes.
"""

import argparse
from pathlib import Path

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.arguments import add_pipeline_arguments
from training.job.utils.logging import setup_logger
from training.job.utils.utils import get_config


def get_options(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse the options accepted by the pipeline entry point.

    The same parser accepts arguments supplied locally or embedded by the
    launcher.

    :param argv: Arguments to parse; None reads the current process command
                 line.

    :return: Parsed pipeline options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(__file__).parent / "cfg/training.yaml",
    )
    add_pipeline_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """
    Execute the selected pipeline nodes in dependency order.

    Resolve paths and configuration before running processing, fitting and
    scoring. Existing artifacts can be supplied when earlier nodes are omitted.

    :param argv: Arguments to parse; None reads the current process command
                 line.

    :return: None.
    """
    # ------------------------------------------------------------------------------------------------------------------
    # Load configuration and resolve pipeline switches

    # Configure objects
    logger = setup_logger()
    options = get_options(argv)
    config = get_config(options.config_path)

    # Determine whether to pull the nodes from config or override with script args
    nodes = options.nodes if options.nodes is not None else config["nodes"]

    # ------------------------------------------------------------------------------------------------------------------
    # Defensive checks

    # Ensure a valid set of nodes have been passed
    if (
        not isinstance(nodes, list)
        or not nodes
        or any(node not in {"data_process", "fit", "score"} for node in nodes)
    ):
        raise ValueError(
            "Select at least one supported node: data_process, fit or score."
        )

    # Ensure a unique set of nodes have been passed
    if len(nodes) != len(set(nodes)):
        raise ValueError("Each pipeline node can only be selected once.")

    # Ensure only processing or using previously processed data - Both are not permitted
    if options.processed_data is not None and "data_process" in nodes:
        raise ValueError(
            "--processed-data is for fit-only runs; omit it when "
            "running data_process."
        )

    # Ensure there is a clear direction for where to fit models
    if options.model_path is not None and "fit" in nodes:
        raise ValueError(
            "--model-path is for score-only runs; omit it when "
            "fitting a new model."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Extract parameters

    # Pull directories
    input_dir = options.input_dir or Path(config["input_dir"])
    output_dir = options.output_dir or (
        Path("/kaggle/working")
        if Path("/kaggle/working").is_dir()
        else Path("outputs")
    )
    processed_data = options.processed_data or output_dir / "training.csv"
    model_path = options.model_path or output_dir / "model.joblib"

    # Determine where to get max rows from, config or script arg
    max_rows = (
        options.max_rows
        if options.max_rows is not None
        else config.get("max_rows")
    )
    if max_rows is not None and max_rows < 0:
        raise ValueError("--max-rows must be zero or positive.")
    max_rows = max_rows or None

    # ------------------------------------------------------------------------------------------------------------------
    # Data processing node

    if "data_process" in nodes:
        # Only load when required
        from training.job.data_process import run as process_data

        logger.info("Starting node: data_process")
        processed_data = process_data(
            input_dir, output_dir, config, max_rows=max_rows
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Model fitting node

    if "fit" in nodes:
        # Only load when required
        from training.job.fit import run as fit_model

        logger.info("Starting node: fit")
        model_path = fit_model(processed_data, output_dir, config)

    # ------------------------------------------------------------------------------------------------------------------
    # Test-set scoring node

    if "score" in nodes:
        # Only load when required
        from training.job.score import run as score_model

        logger.info("Starting node: score")
        score_model(
            model_path, input_dir, output_dir, config, run_id=options.run_id
        )

    logger.info("Selected pipeline nodes completed.")


# ----------------------------------------------------------------------------------------------------------------------
# Script entry point

if __name__ == "__main__":
    main()
