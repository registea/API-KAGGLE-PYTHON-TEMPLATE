"""
Fit a one-feature binary classifier from a previously prepared CSV.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger


def run(input_file: Path, output_dir: Path, config: dict) -> Path:
    """
    Fit and evaluate a demonstration binary classifier.

    Select the first eligible numeric predictor and evaluate on a stratified holdout. Refit preprocessing and the
    model on all labelled rows before saving the model and evaluation metrics.

    :param input_file: Processed training CSV containing predictors and the target.
    :param output_dir: Directory in which to save the model and metrics.
    :param config: Training settings including target, join key, split size and seed.

    :return: Path to the fitted model.joblib artifact.
    """
    logger = setup_logger()

    # ------------------------------------------------------------------------------------------------------------------
    # Load prepared data and select the first eligible numeric column

    # Provide an actionable failure when the processing node has not produced an input
    if not input_file.is_file():
        raise FileNotFoundError(
            f"Prepared data not found: {input_file}. Run data_process "
            "first or pass --processed-data."
        )

    # Load the flat training table produced locally or by the processing node
    data = pd.read_csv(input_file)
    target = config["target_column"]

    # Ensure the configured target is present before selecting predictors
    if target not in data:
        raise ValueError(f"Missing target column {target}.")

    # Neither the transaction identifier nor the label is a valid baseline predictor.
    excluded = {target, config["join_key"]}
    features = [
        name
        for name in data.select_dtypes(include="number").columns
        if name not in excluded
    ]

    # Require at least one numeric column for the demonstration estimator
    if not features:
        raise ValueError(
            "No numeric predictor remains after excluding the target "
            "and join ID."
        )

    # Use one predictor deliberately to keep the template example quick and transparent
    feature = features[0]
    x = data[[feature]].replace([np.inf, -np.inf], np.nan)
    y = data[target]

    # Require enough examples from both binary classes for a stratified holdout
    if (
        y.isna().any()
        or set(y.unique()) != {0, 1}
        or y.value_counts().min() < 2
    ):
        raise ValueError(
            "The target must contain binary 0/1 values and at least "
            "two rows per class."
        )
    test_size = config["test_size"]

    # Ensure the configured fraction leaves data available on both sides of the split
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    # ------------------------------------------------------------------------------------------------------------------
    # Split before fitting preprocessing to avoid validation leakage

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=config["random_seed"],
        stratify=y,
    )

    # Median imputation requires at least one observed training value
    if x_train[feature].notna().sum() == 0:
        raise ValueError(
            f"Selected feature {feature} has no finite training values."
        )

    # Fit imputation and scaling inside the pipeline using only the training split.
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=config["random_seed"],
        ),
    )
    model.fit(x_train, y_train)

    # Retain holdout probabilities for metrics before refitting the final estimator
    probabilities = model.predict_proba(x_test)[:, 1]

    # ------------------------------------------------------------------------------------------------------------------
    # Refit on all available labelled rows after held-out evaluation, then save

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"

    # Keep holdout probabilities for metrics, then use all labelled rows for final scoring.
    model.fit(x, y)
    joblib.dump(model, model_path)

    # Record enough context to interpret the demonstration validation scores
    metrics = {
        "feature": feature,
        "target": target,
        "train_rows": len(x_train),
        "validation_rows": len(x_test),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "random_seed": config["random_seed"],
        "refit_rows": len(x),
        "note": (
            "One-feature demonstration with a random stratified split; "
            "not a competition-ready model."
        ),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Fitted feature %s; ROC AUC %.4f; saved model to %s",
        feature,
        metrics["roc_auc"],
        model_path,
    )

    return model_path
