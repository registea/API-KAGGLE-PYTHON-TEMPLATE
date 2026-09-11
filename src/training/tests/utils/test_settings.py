"""
Verify local dotenv loading and build-agent credential handling.
"""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.utils.settings import Settings


# ----------------------------------------------------------------------------------------------------------------------
# Environment settings tests


class SettingsTests(unittest.TestCase):
    """
    Group checks for settings behaviour.
    """

    def test_local_dotenv_reaches_child_environment(self):
        # --------------------------------------------------------------------------------------------------------------
        # Load a synthetic token without accessing user credentials

        """
        Verify local dotenv reaches child environment.
        """
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {}, clear=True),
        ):
            root = Path(directory)
            (root / ".env").write_text(
                "KAGGLE_API_TOKEN=test-only-token\n", encoding="utf-8"
            )
            settings = Settings(project_root=root)
            settings.validate_authentication()
            self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "test-only-token")

    def test_shell_values_take_precedence(self):
        # --------------------------------------------------------------------------------------------------------------
        # Keep credentials supplied by the caller

        """
        Verify shell values take precedence.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "KAGGLE_API_TOKEN=file-token\n", encoding="utf-8"
            )
            with patch.dict(
                os.environ, {"KAGGLE_API_TOKEN": "shell-token"}, clear=True
            ):
                Settings(project_root=root)
                self.assertEqual(os.environ["KAGGLE_API_TOKEN"], "shell-token")

    def test_build_agent_skips_dotenv(self):
        # --------------------------------------------------------------------------------------------------------------
        # Exercise both DevOps detection and generic CI detection

        """
        Verify build agent skips dotenv.
        """
        for agent in ({"BUILD_REASON": "Manual"}, {"CI": "true"}):
            with (
                self.subTest(agent=agent),
                patch.dict(os.environ, agent, clear=True),
            ):
                with patch("training.utils.settings.load_dotenv") as loader:
                    settings = Settings()
                    loader.assert_not_called()
                    with self.assertRaisesRegex(
                        EnvironmentError, "KAGGLE_API_TOKEN"
                    ):
                        settings.validate_authentication()

    def test_prepare_settings_do_not_require_credentials(self):
        # --------------------------------------------------------------------------------------------------------------
        # Local preparation remains possible before credentials are configured

        """
        Verify prepare settings do not require credentials.
        """
        with patch.dict(os.environ, {"CI": "true"}, clear=True):
            settings = Settings()
            self.assertEqual(settings.kernel_id, "")
