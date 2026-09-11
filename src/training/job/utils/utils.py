"""
Shared configuration helpers for local submission and remote execution.
"""

import json
from pathlib import Path
from typing import Any
import yaml


# ------------------------------------------------------------------------------
# Configuration loading


def get_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a JSON or YAML configuration mapping.

    Select JSON by its file extension and otherwise parse YAML. Reject
    non-mapping roots so callers can access configuration by key.

    :param config_path: Path to the configuration file.

    :return: Dictionary containing the configuration values.
    """
    # --------------------------------------------------------------------------
    # Read configuration from disk

    # Normalise string inputs before selecting the parser from the file suffix
    path = Path(config_path)

    # Use safe YAML loading; JSON metadata retains its native JSON parser
    # Open explicitly as UTF-8 so local and Kaggle parsing behave consistently
    with path.open(encoding="utf-8") as stream:
        try:
            config = (
                json.load(stream)
                if path.suffix == ".json"
                else yaml.safe_load(stream)
            )
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML configuration: {path}") from exc

    # --------------------------------------------------------------------------
    # Validate the parsed configuration

    # Fail early when the file does not describe named configuration settings
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return config
