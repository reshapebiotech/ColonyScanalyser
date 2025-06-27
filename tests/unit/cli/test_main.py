"""
Simple unit tests for the CLI main module.
"""

from unittest.mock import MagicMock, patch

import pytest

from colonyscanalyser.cli.main import main


class TestMainCLI:
    """Basic test cases for the main CLI function."""

    def test_help_works(self):
        """Test that help flag works."""
        with patch("sys.argv", ["colonyscanalyser", "--help"]):
            with pytest.raises(SystemExit):
                main()

    def test_missing_path_fails(self):
        """Test that missing path argument fails."""
        with patch("sys.argv", ["colonyscanalyser"]):
            with pytest.raises(SystemExit):
                main()

    def test_nonexistent_path_fails(self):
        """Test that nonexistent path fails with proper error."""
        with patch("sys.argv", ["colonyscanalyser", "/nonexistent"]):
            with pytest.raises(EnvironmentError, match="could not be found"):
                main()

    def test_basic_workflow_with_mocks(self):
        """Test basic workflow with minimal mocking."""
        with patch("sys.argv", ["colonyscanalyser", "/test", "--image-align", "none"]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch(
                    "colonyscanalyser.cli.main.ImageFileCollection.from_path"
                ) as mock_ifc:
                    with patch(
                        "colonyscanalyser.cli.main.PlateCollection.from_image"
                    ) as mock_plates:
                        # Setup mocks
                        mock_path.return_value.resolve.return_value.exists.return_value = True

                        mock_collection = MagicMock()
                        mock_collection.count = 5
                        mock_ifc.return_value = mock_collection

                        mock_plates_result = MagicMock()
                        mock_plates_result.count = 2
                        mock_plates.return_value = mock_plates_result

                        # Should complete without error
                        try:
                            main()
                        except SystemExit:
                            pass  # Expected
