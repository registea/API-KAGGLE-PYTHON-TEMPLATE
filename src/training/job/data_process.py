"""
Join competition transaction and identity tables into a flat training file.
"""

import pandas as pd
from pathlib import Path


# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger


def run(
    input_dir: Path, output_dir: Path, config: dict, max_rows: int | None = None
) -> Path:
    """
    Join transaction and identity data and save the training table.

    Validate unique join keys before a left join so transactions without identity data remain available for training.

    :param input_dir: Directory containing the configured raw CSV files.
    :param output_dir: Directory in which to write training.csv.
    :param config: Training configuration containing file names, join key and target.
    :param max_rows: Maximum transaction rows to read; None loads all rows.

    :return: Path to the processed training CSV.
    """
    logger = setup_logger()

    # Load source data
    transaction_path = input_dir / config["transaction_file"]
    identity_path = input_dir / config["identity_file"]

    # Ensure an explicitly supplied row limit can return at least one record
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive or null.")

    # Load all identity rows so transaction matches are not affected by the training limit
    transactions = pd.read_csv(transaction_path, nrows=max_rows)
    identity = pd.read_csv(identity_path)
    join_key = config["join_key"]
    target = config["target_column"]

    # Ensure both datasets can participate safely in a one-to-one join
    for name, frame in (("transactions", transactions), ("identity", identity)):
        if join_key not in frame:
            raise ValueError(f"Missing join column {join_key} in {name}.")
        if frame[join_key].isna().any() or frame[join_key].duplicated().any():
            raise ValueError(
                f"{name} must contain unique, non-null {join_key} values."
            )

    # Ensure the transaction data contains the label required by the fitting node
    if target not in transactions:
        raise ValueError(f"Missing target column {target} in transactions.")

    # ------------------------------------------------------------------------------------------------------------------
    # Join data without dropping transactions that have no identity record

    # Reject ambiguous names rather than accepting pandas-generated suffixes
    overlap = (set(transactions.columns) & set(identity.columns)) - {join_key}
    if overlap:
        raise ValueError(f"Unexpected overlapping columns: {sorted(overlap)}")

    # Retain every transaction because identity information is optional
    flat = transactions.merge(
        identity, on=join_key, how="left", validate="one_to_one"
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Persist the stage output for this run or a later training job

    # Create the Kaggle working directory or its local equivalent when required
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training.csv"
    flat.to_csv(output_path, index=False)
    logger.info(
        "Data processing saved %s rows and %s columns to %s",
        len(flat),
        len(flat.columns),
        output_path,
    )

    return output_path
