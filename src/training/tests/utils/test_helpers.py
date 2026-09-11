"""
Check shared configuration loading and repeated logger setup.
"""

from pathlib import Path
import tempfile
import unittest

# ------------------------------------------------------------------------------
# Import Local Functionality

from training.job.utils.logging import setup_logger
from training.job.utils.utils import get_config


# ------------------------------------------------------------------------------
# Shared helper tests


class HelperTests(unittest.TestCase):
    """
    Group checks for helpers behaviour.
    """

    def test_yaml_and_json_configuration(self):
        # ----------------------------------------------------------------------
        # Verify equivalent YAML and JSON values

        """
        Verify yaml and json configuration.
        """
        with tempfile.TemporaryDirectory() as directory:
            for suffix, content in (
                (".yaml", "enabled: false\nseed: 42\n"),
                (".json", '{"enabled": false, "seed": 42}'),
            ):
                path = Path(directory) / ("config" + suffix)
                path.write_text(content, encoding="utf-8")
                self.assertEqual(
                    get_config(path), {"enabled": False, "seed": 42}
                )

    def test_non_mapping_configuration_is_rejected(self):
        # ----------------------------------------------------------------------
        # Reject configuration without named settings

        """
        Verify non mapping configuration is rejected.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text("- value\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mapping"):
                get_config(path)

    def test_logger_setup_does_not_duplicate_handlers(self):
        # ----------------------------------------------------------------------
        # Preserve handler count when the logger is requested again

        """
        Verify logger setup does not duplicate handlers.
        """
        logger = setup_logger()
        handler_count = len(logger.handlers)
        self.assertIs(setup_logger(), logger)
        self.assertEqual(len(logger.handlers), handler_count)
