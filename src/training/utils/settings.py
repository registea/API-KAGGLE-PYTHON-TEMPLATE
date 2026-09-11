"""
Load local dotenv settings or credentials injected by a build agent.
"""

from dotenv import load_dotenv
import logging
import os
from pathlib import Path


# ----------------------------------------------------------------------------------------------------------------------
# Environment settings


class Settings:
    """
    Load submission settings from local dotenv files or build-agent variables.

    :param logger: Shared project logger.
    :param project_root: Repository root containing the optional .env file.
    """

    def __init__(
        self,
        logger: logging.Logger | None = None,
        project_root: Path | None = None,
    ):
        """
        Load environment settings for the submission launcher.Local dotenv values do not override existing environment
        variables. Build agents use injected variables without loading the local file.

        :param logger: Optional logger used to report the execution context.
        :param project_root: Repository root containing .env; defaults to the
                             working directory.

        :return: None.
        """

        # --------------------------------------------------------------------------------------------------------------
        # Capture logging and execution context

        # Reuse the caller's logger so setup messages share a consistent format
        self.logger = (
            logger if logger is not None else logging.getLogger(__name__)
        )
        root = project_root if project_root is not None else Path.cwd()

        # Detect both Azure DevOps and generic continuous-integration environments
        self.devops_run = bool(os.environ.get("BUILD_REASON"))
        ci_run = os.environ.get("CI", "").lower() in {"true", "1", "yes"}
        self.build_agent_run = self.devops_run or ci_run

        # --------------------------------------------------------------------------------------------------------------
        # Load local settings or use build-agent variables

        # Local runs may read .env; automated jobs must receive settings explicitly
        if self.build_agent_run:
            self.logger.info(
                "Running on build agent; using injected environment variables."
            )
        else:
            self.logger.info(
                "Running locally; loading environment variables from .env."
            )
            # Existing shell/CI values take precedence over values in the local file.
            load_dotenv(
                dotenv_path=root / ".env", override=False, interpolate=False
            )

        # --------------------------------------------------------------------------------------------------------------
        # Capture non-secret submission identity

        # Never retain the API token on the settings object
        self.kernel_id = os.environ.get("KAGGLE_KERNEL_ID", "").strip()
        self.username = os.environ.get("KAGGLE_USERNAME", "").strip()
        self.kernel_slug = os.environ.get("KAGGLE_KERNEL_SLUG", "").strip()

    def validate_authentication(self) -> None:
        """
        Check that the Kaggle API token is present and usable in form.

        Reject blank values, template placeholders and whitespace without
        logging the token. Kaggle verifies the credentials when the CLI runs.

        :return: None.
        """
        # --------------------------------------------------------------------------------------------------------------
        # Check credentials without logging or placing them in command arguments

        # Read the token only for validation and never include it in errors or logs
        token = os.environ.get("KAGGLE_API_TOKEN", "")
        placeholders = {
            "your-token",
            "your-api-token",
            "your-kaggle-api-token",
            "your-token-from-kaggle",
        }

        # Treat template values and whitespace-containing tokens as unconfigured
        if (
            not token.strip()
            or token.lower() in placeholders
            or any(char.isspace() for char in token)
        ):
            raise EnvironmentError(
                "Set a non-empty, non-placeholder KAGGLE_API_TOKEN "
                "without whitespace in .env or the environment before "
                "running submit, status or output."
            )
