"""
Resolve and validate Kaggle metadata without changing the source file.
"""

import argparse
import re

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.utils.settings import Settings


# ----------------------------------------------------------------------------------------------------------------------
# Metadata resolution


def resolve_metadata(
    metadata: dict, options: argparse.Namespace, settings: Settings
) -> dict:
    """
    Resolve and validate Kaggle kernel metadata.

    Apply command-line overrides and environment identity before the JSON defaults. Return a copy so preparing a
    run does not alter the template metadata.

    :param metadata: Source kernel metadata mapping.
    :param options: Parsed launcher arguments containing optional overrides.
    :param settings: Loaded environment identity settings.

    :return: Validated metadata for the requested action.
    """
    # Copy the source mapping so command-line overrides do not edit the template file
    result = dict(metadata)

    # Resolve a full ID, or compose it from environment username and slug.
    # Prefer a complete command-line or environment identity when one is available
    kernel_id = options.kernel_id or settings.kernel_id

    # Compose the identity only when both individual environment values are present
    if not kernel_id and (settings.username or settings.kernel_slug):
        if not settings.username or not settings.kernel_slug:
            raise ValueError(
                "Set both KAGGLE_USERNAME and KAGGLE_KERNEL_SLUG, "
                "or KAGGLE_KERNEL_ID."
            )
        kernel_id = f"{settings.username}/{settings.kernel_slug}"
    if kernel_id:
        result["id"] = kernel_id

        # Kaggle prioritises numeric id_no over id; avoid targeting a stale kernel.
        result.pop("id_no", None)

    # ------------------------------------------------------------------------------------------------------------------
    # Apply scalar and list overrides

    # Preserve explicit false values and empty lists by checking against None
    for field in (
        "title",
        "is_private",
        "enable_gpu",
        "enable_internet",
        "machine_shape",
        "dataset_sources",
        "competition_sources",
        "kernel_sources",
        "model_sources",
    ):
        value = getattr(options, field)
        if value is not None:
            result[field] = value

    # ------------------------------------------------------------------------------------------------------------------
    # Validate metadata before packaging or contacting Kaggle

    # Ensure the final identifier is safe to pass directly to the Kaggle CLI
    identity = result.get("id")
    if not isinstance(identity, str) or not re.fullmatch(
        r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", identity
    ):
        raise ValueError("Kernel id must have the form username/kernel-slug.")
    if options.action != "prepare" and identity.upper().startswith("YOUR_"):
        raise ValueError(
            "Replace the placeholder kernel id using --kernel-id "
            "or environment settings."
        )

    # Numeric Kaggle identifiers can silently override the intended owner and slug
    if result.get("id_no") is not None:
        raise ValueError(
            "Remove id_no from metadata; this launcher manages kernels by id."
        )
    if not isinstance(result.get("title"), str) or not result["title"].strip():
        raise ValueError("Kernel title must be a non-empty string.")

    # Require JSON boolean values rather than truthy strings such as false
    for field in ("is_private", "enable_gpu", "enable_internet"):
        if field in result and not isinstance(result[field], bool):
            raise ValueError(f"{field} must be a JSON boolean.")

    # Validate attached-source lists before packaging the metadata file
    for field in (
        "dataset_sources",
        "competition_sources",
        "kernel_sources",
        "model_sources",
    ):
        value = result.get(field, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise ValueError(f"{field} must be a list of non-empty strings.")

    return result
