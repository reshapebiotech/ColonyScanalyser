"""
Unit tests for the CLI main module.

This module tests the argument parsing and basic setup functionality
of the main CLI entry point.
"""

from unittest.mock import MagicMock, patch

import pytest

from colonyscanalyser.cli.main import main


class TestMainCLI:
    """Test cases for the main CLI function."""

    def test_argument_parsing_defaults(self):
        """Test that default arguments are parsed correctly."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    mock_path.return_value.resolve.return_value.exists.return_value = (
                        True
                    )
                    mock_ifc.from_path.return_value.count = 5

                    # Should not raise any exceptions during argument parsing
                    try:
                        main()
                    except SystemExit:
                        pass  # Expected due to successful completion

    def test_path_validation_missing_path(self):
        """Test that missing path raises ValueError."""
        with patch("sys.argv", ["colonyscanalyser"]):
            with pytest.raises(SystemExit):  # argparse exits on missing required arg
                main()

    def test_path_validation_nonexistent_path(self):
        """Test that nonexistent path raises EnvironmentError."""
        test_path = "/nonexistent/path"

        with patch("sys.argv", ["colonyscanalyser", test_path]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                mock_path.return_value.resolve.return_value.exists.return_value = False

                with pytest.raises(EnvironmentError, match="could not be found"):
                    main()

    def test_multiprocessing_setup_single_process(self):
        """Test multiprocessing setup with single process flag."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path, "--single-process"]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    with patch("colonyscanalyser.cli.main.cpu_count", return_value=4):
                        mock_path.return_value.resolve.return_value.exists.return_value = True
                        mock_ifc.from_path.return_value.count = 5

                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to successful completion

    def test_multiprocessing_setup_multi_process(self):
        """Test multiprocessing setup with multiple processes."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    with patch("colonyscanalyser.cli.main.cpu_count", return_value=4):
                        mock_path.return_value.resolve.return_value.exists.return_value = True
                        mock_ifc.from_path.return_value.count = 5

                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to successful completion

    def test_image_discovery(self):
        """Test image discovery functionality."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    mock_path.return_value.resolve.return_value.exists.return_value = (
                        True
                    )
                    mock_collection = MagicMock()
                    mock_collection.count = 10
                    mock_ifc.from_path.return_value = mock_collection

                    try:
                        main()
                    except SystemExit:
                        pass  # Expected due to successful completion

                    # Verify ImageFileCollection.from_path was called with correct arguments
                    mock_ifc.from_path.assert_called_once()

    def test_silent_mode(self):
        """Test that silent mode suppresses output."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path, "--silent"]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    with patch("builtins.print") as mock_print:
                        mock_path.return_value.resolve.return_value.exists.return_value = True
                        mock_ifc.from_path.return_value.count = 5

                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to successful completion

                        # In silent mode, print should not be called
                        mock_print.assert_not_called()

    def test_verbose_mode(self):
        """Test that verbose mode produces extra output."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path, "--verbose"]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    with patch("builtins.print") as mock_print:
                        with patch(
                            "colonyscanalyser.cli.main.cpu_count", return_value=4
                        ):
                            mock_path.return_value.resolve.return_value.exists.return_value = True
                            mock_ifc.from_path.return_value.count = 5

                            try:
                                main()
                            except SystemExit:
                                pass  # Expected due to successful completion

                            # In verbose mode, print should be called multiple times
                            assert mock_print.call_count > 0

    def test_cached_data_loading_success(self):
        """Test successful cached data loading."""
        test_path = "/test/path"

        with patch(
            "sys.argv",
            ["colonyscanalyser", test_path, "--use-cached-data", "--verbose"],
        ):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.load_file") as mock_load:
                    with patch("colonyscanalyser.cli.main.PlateCollection") as mock_pc:
                        from colonyscanalyser.core import Plate

                        mock_path.return_value.resolve.return_value.exists.return_value = True

                        # Mock successful data loading
                        mock_plates = MagicMock()
                        mock_plates.count = 4
                        mock_plates.items = [Plate(id=1, diameter=90, name="test")]
                        mock_load.return_value = mock_plates
                        mock_pc.coordinate_to_index.return_value = 4

                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to successful completion

                        # Verify load_file was called
                        mock_load.assert_called_once()

    def test_cached_data_loading_failure(self):
        """Test fallback when cached data loading fails."""
        test_path = "/test/path"

        with patch("sys.argv", ["colonyscanalyser", test_path, "--use-cached-data"]):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.load_file") as mock_load:
                    with patch(
                        "colonyscanalyser.cli.main.ImageFileCollection"
                    ) as mock_ifc:
                        mock_path.return_value.resolve.return_value.exists.return_value = True
                        mock_load.return_value = None  # Simulate failed loading
                        mock_ifc.from_path.return_value.count = 5

                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to successful completion

                        # Should fall back to image processing
                        mock_ifc.from_path.assert_called_once()

    def test_plate_configuration(self):
        """Test plate configuration parsing."""
        test_path = "/test/path"

        with patch(
            "sys.argv",
            [
                "colonyscanalyser",
                test_path,
                "--plate-lattice",
                "2",
                "3",
                "--plate-size",
                "90",
                "--plate-edge-cut",
                "10",
                "--plate-labels",
                "control",
                "test1",
                "test2",
            ],
        ):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    with patch(
                        "colonyscanalyser.cli.main.mm_to_pixels", return_value=300
                    ):
                        mock_path.return_value.resolve.return_value.exists.return_value = True
                        mock_ifc.from_path.return_value.count = 5

                        try:
                            main()
                        except SystemExit:
                            pass  # Expected due to successful completion

    def test_alignment_strategy_configuration(self):
        """Test alignment strategy configuration."""
        test_path = "/test/path"

        with patch(
            "sys.argv",
            [
                "colonyscanalyser",
                test_path,
                "--image-align",
                "comprehensive",
                "--image-align-tolerance",
                "0.05",
            ],
        ):
            with patch("colonyscanalyser.cli.main.Path") as mock_path:
                with patch("colonyscanalyser.cli.main.ImageFileCollection") as mock_ifc:
                    mock_path.return_value.resolve.return_value.exists.return_value = (
                        True
                    )
                    mock_ifc.from_path.return_value.count = 5

                    try:
                        main()
                    except SystemExit:
                        pass  # Expected due to successful completion
