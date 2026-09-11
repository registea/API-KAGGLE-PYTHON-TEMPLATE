"""
Verify identity precedence, metadata overrides and early authentication checks.
"""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

# ----------------------------------------------------------------------------------------------------------------------
# Import Local Functionality

from training.main import get_options, main
from training.utils.metadata import resolve_metadata
from training.utils.settings import Settings


# ----------------------------------------------------------------------------------------------------------------------
# Metadata and launcher tests


class MetadataTests(unittest.TestCase):
    """
    Group checks for metadata behaviour.
    """

    def resolve(self, args, environment=None, metadata=None):
        """
        Resolve metadata using isolated environment values and command-line
        options.
        """
        with patch.dict(
            os.environ, {"CI": "true", **(environment or {})}, clear=True
        ):
            return resolve_metadata(
                metadata
                or {
                    "id": "example/default",
                    "title": "Default",
                    "enable_gpu": True,
                },
                get_options(args),
                Settings(),
            )

    def test_cli_identity_wins_and_stale_numeric_id_is_removed(self):
        """
        Verify cli identity wins and stale numeric id is removed.
        """
        result = self.resolve(
            ["prepare", "--kernel-id", "cli/new"],
            {"KAGGLE_KERNEL_ID": "env/old"},
            {"id": "file/old", "id_no": 123, "title": "Example"},
        )
        self.assertEqual(result["id"], "cli/new")
        self.assertNotIn("id_no", result)

    def test_environment_full_id_wins_over_components(self):
        """
        Verify environment full id wins over components.
        """
        result = self.resolve(
            ["prepare"],
            {
                "KAGGLE_KERNEL_ID": "env/full",
                "KAGGLE_USERNAME": "other",
                "KAGGLE_KERNEL_SLUG": "component",
            },
        )
        self.assertEqual(result["id"], "env/full")

    def test_environment_components_are_combined(self):
        """
        Verify environment components are combined.
        """
        result = self.resolve(
            ["prepare"],
            {
                "KAGGLE_USERNAME": "example",
                "KAGGLE_KERNEL_SLUG": "hello",
            },
        )
        self.assertEqual(result["id"], "example/hello")

    def test_partial_components_fail(self):
        """
        Verify partial components fail.
        """
        with self.assertRaisesRegex(ValueError, "both"):
            self.resolve(["prepare"], {"KAGGLE_USERNAME": "example"})

    def test_json_identity_is_the_fallback(self):
        """
        Verify json identity is the fallback.
        """
        self.assertEqual(self.resolve(["prepare"])["id"], "example/default")

    def test_false_boolean_and_empty_list_override_defaults(self):
        """
        Verify false boolean and empty list override defaults.
        """
        result = self.resolve(
            [
                "prepare",
                "--no-enable-gpu",
                "--dataset-sources",
                "--title",
                "New Title",
            ],
            metadata={
                "id": "example/default",
                "title": "Default",
                "enable_gpu": True,
                "dataset_sources": ["owner/data"],
            },
        )
        self.assertFalse(result["enable_gpu"])
        self.assertEqual(result["dataset_sources"], [])
        self.assertEqual(result["title"], "New Title")

    def test_explicit_sources_replace_defaults(self):
        """
        Verify explicit sources replace defaults.
        """
        result = self.resolve(
            ["prepare", "--dataset-sources", "owner/one", "owner/two"]
        )
        self.assertEqual(result["dataset_sources"], ["owner/one", "owner/two"])

    def test_invalid_remote_identity_fails(self):
        """
        Verify invalid remote identity fails.
        """
        for identity in (
            "YOUR_KAGGLE_USERNAME/template",
            "missing-slash",
            "owner/",
        ):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                self.resolve(["submit", "--kernel-id", identity])

    def test_invalid_credentials_stop_before_packaging_or_subprocess(self):
        """
        Verify invalid credentials stop before packaging or subprocess.
        """
        for token in ("", "   ", "your-kaggle-api-token", "token with spaces"):
            with self.subTest(token=token):
                with patch.dict(
                    os.environ,
                    {"CI": "true", "KAGGLE_API_TOKEN": token},
                    clear=True,
                ):
                    with patch("training.main.prepare_kernel") as package:
                        with patch("training.main.subprocess.run") as run:
                            with self.assertRaises(SystemExit):
                                main(["submit"])
                            package.assert_not_called()
                            run.assert_not_called()

    def test_prepare_needs_no_credentials_and_preserves_source_metadata(self):
        # --------------------------------------------------------------------------------------------------------------
        # Prepare in a temporary checkout without reading real credentials

        """
        Verify prepare needs no credentials and preserves source metadata.
        """
        import json
        import shutil

        root = Path(__file__).resolve().parents[3]
        with TemporaryDirectory() as directory:
            staging = Path(directory)
            shutil.copytree(root / "src", staging / "src")
            metadata_path = (
                staging / "src/training/environment/kernel-metadata.json"
            )
            original = metadata_path.read_bytes()
            with patch.dict(os.environ, {"CI": "true"}, clear=True):
                main(
                    [
                        "prepare",
                        "--project-root",
                        str(staging),
                        "--kernel-id",
                        "example/hello",
                        "--enable-gpu",
                    ]
                )
            generated = json.loads(
                (staging / "build/kaggle/kernel-metadata.json").read_text()
            )
            self.assertEqual(generated["id"], "example/hello")
            self.assertTrue(generated["enable_gpu"])
            self.assertEqual(metadata_path.read_bytes(), original)
