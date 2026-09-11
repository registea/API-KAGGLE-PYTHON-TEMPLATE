"""
Score the complete competition test set and write a validated submission CSV.
"""

import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger


def run(
    model_path: Path,
    input_dir: Path,
    output_dir: Path,
    config: dict,
    run_id: str | None = None,
) -> Path:
    """
    Score the complete test set using the saved model.

    Load only required predictors, align probabilities with the sample submission and save a manifest that identifies
    and checksums this run's CSV.

    :param model_path: Path to the model produced by the fitting node.
    :param input_dir: Directory containing test inputs and the sample submission.
    :param output_dir: Directory in which to save predictions and their manifest.
    :param config: Competition configuration containing file names, join key and target.
    :param run_id: Optional launcher identifier recorded for artifact verification.

    :return: Path to submission.csv.
    """
    logger = setup_logger()

    # ------------------------------------------------------------------------------------------------------------------
    # Load model features and the test transaction identifiers

    # Load an artifact produced by the fitting node, including its fitted preprocessing.
    model = joblib.load(model_path)
    features = list(model.feature_names_in_)
    key, target = config["join_key"], config["target_column"]
    transaction_path = input_dir / config["test_transaction_file"]

    # Read the header first so transaction and identity predictors can be separated
    headers = pd.read_csv(transaction_path, nrows=0).columns
    transaction_features = [
        feature for feature in features if feature in headers
    ]

    # Scoring deliberately reads all test rows, regardless of the training row limit.
    test = pd.read_csv(transaction_path, usecols=[key, *transaction_features])
    missing = [feature for feature in features if feature not in headers]

    # ------------------------------------------------------------------------------------------------------------------
    # Join identity predictors when the model requires them

    if missing:
        identity_path = input_dir / config["test_identity_file"]
        identity_headers = pd.read_csv(identity_path, nrows=0).columns

        # IEEE-CIS test identity names use id- where the training files use id_.
        names = {
            column.replace("id-", "id_"): column for column in identity_headers
        }

        # Stop before scoring when a fitted predictor is absent from both test files
        if any(feature not in names for feature in missing):
            raise ValueError("Test inputs do not contain every fitted feature.")

        # Load only the identity columns used by the saved estimator
        identity = pd.read_csv(
            identity_path,
            usecols=[key, *[names[feature] for feature in missing]],
        )
        identity = identity.rename(
            columns={value: name for name, value in names.items()}
        )

        # Protect the test row count from duplicate or missing identity keys
        if identity[key].isna().any() or identity[key].duplicated().any():
            raise ValueError("Test identity keys must be unique and non-null.")
        test = test.merge(identity, on=key, how="left", validate="one_to_one")

    # ------------------------------------------------------------------------------------------------------------------
    # Score all test rows and align with the competition's sample submission

    # Use the sample submission as the contract for column names and row ordering
    sample = pd.read_csv(input_dir / config["sample_submission_file"])
    if list(sample.columns) != [key, target]:
        raise ValueError(
            f"Expected sample submission columns: {key}, {target}."
        )

    # Ensure each prediction maps unambiguously to one competition row
    for name, frame in (("test", test), ("sample", sample)):
        if (
            frame.empty
            or frame[key].isna().any()
            or frame[key].duplicated().any()
        ):
            raise ValueError(
                f"{name} identifiers must be non-empty, unique and non-null."
            )

    # Require exactly the same identifiers even when the files use different orders
    if set(test[key]) != set(sample[key]):
        raise ValueError("Test and sample submission identifiers differ.")

    # Guard against accidental target leakage or identifier-based fitting
    if target in features or key in features:
        raise ValueError(
            "Model predictors must exclude the target and transaction ID."
        )

    # Convert predictors consistently and leave missing values for the fitted imputer
    values = (
        test[features]
        .apply(pd.to_numeric, errors="raise")
        .replace([np.inf, -np.inf], np.nan)
    )
    classes = list(model.classes_)

    # Ensure the estimator exposes the positive class required by the competition
    if 1 not in classes:
        raise ValueError(
            "Model must provide probabilities for positive class 1."
        )

    # Locate the positive class explicitly rather than assuming its probability column.
    probabilities = model.predict_proba(values)[:, classes.index(1)]

    # Reject invalid probabilities before creating a submission artifact
    if (
        not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError(
            "Predictions must be finite probabilities between zero and one."
        )

    # Match by transaction ID so a different sample-submission order remains valid.
    predictions = pd.Series(probabilities, index=test[key])
    sample[target] = sample[key].map(predictions)

    # ------------------------------------------------------------------------------------------------------------------
    # Save predictions and proof identifying this job's output

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    sample.to_csv(output_path, index=False)

    # The local launcher uses this manifest to reject stale or altered downloads.
    manifest = {
        "run_id": run_id,
        "competition": config["competition"],
        "rows": len(sample),
        "columns": list(sample.columns),
        "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    (output_dir / "submission-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Scored all %s test rows; saved %s", len(sample), output_path)

    return output_path
