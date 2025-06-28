"""
Comprehensive tests for pipeline stages and processor.

Tests the new stage-based pipeline architecture to ensure clean
separation of concerns and reliable execution.
"""

import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from colonyscanalyser.models.colony import Timepoint
from colonyscanalyser.models.config import (
    PipelineConfig,
)
from colonyscanalyser.models.image import ImageCollection, ImageFile
from colonyscanalyser.models.plate import Plate, PlateCollection
from colonyscanalyser.pipeline.processor_clean import (
    ColonyPipelineProcessor,
    SimplePipelineProcessor,
    create_minimal_pipeline,
    create_standard_pipeline,
    create_visualization_only_pipeline,
)
from colonyscanalyser.pipeline.stages import (
    CacheLoadStage,
    ColonyTrackingStage,
    DataSaveStage,
    ImageAlignmentStage,
    ImageDiscoveryStage,
    PlateDetectionStage,
    TimePointExtractionStage,
    VisualizationStage,
)


@pytest.fixture
def sample_config():
    """Create a sample pipeline configuration."""
    return PipelineConfig(
        input_dir=Path("/tmp/test"),
        output_dir=Path("/tmp/test/output"),
        cache_dir=Path("/tmp/test/cache"),
        enable_caching=False,
        enable_alignment=True,
        enable_visualization=True,
        parallel_processing=False,
        image_pattern="*.jpg",
        plate_diameter_mm=90.0,
        plate_edge_cut_mm=5.0,
    )


@pytest.fixture
def sample_image_files():
    """Create sample image files for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create dummy image files
        image_paths = []
        for i in range(3):
            img_file = temp_path / f"image_{i:03d}.jpg"
            img_file.touch()
            image_paths.append(img_file)

        # Create ImageFile objects
        images = []
        for i, path in enumerate(image_paths):
            # Mock the file existence check for testing
            with patch.object(Path, "exists", return_value=True):
                img = ImageFile(
                    file_path=path,
                    timestamp=i * 10,  # 10 second intervals
                )
                images.append(img)

        yield ImageCollection(images)


@pytest.fixture
def sample_plates():
    """Create sample plates for testing."""
    plates = PlateCollection()
    for i in range(4):
        plate = Plate(
            id=i + 1,
            diameter=100,
            name=f"Plate_{i + 1}",
            center=(50 + i * 100, 50 + (i // 2) * 100),
        )
        plates.add(plate)

    return plates


@pytest.fixture
def sample_timepoints():
    """Create sample timepoints for testing."""
    timepoints = []
    for i in range(5):
        tp = Timepoint(
            timestamp=timedelta(seconds=i * 60),
            area=100 + i * 10,
            center=(25.0 + i, 25.0 + i),
            diameter=10.0 + i,
            perimeter=30.0 + i * 2,
        )
        timepoints.append(tp)

    return timepoints


class TestPipelineStages:
    """Test individual pipeline stages."""

    def test_cache_load_stage_no_cache(self, sample_config):
        """Test cache loading when no cache exists."""
        stage = CacheLoadStage(sample_config)
        context = {}

        result = stage.execute(context)

        assert result["cache_loaded"] is False

    def test_cache_load_stage_disabled(self, sample_config):
        """Test cache loading when disabled in config."""
        sample_config.enable_caching = False
        stage = CacheLoadStage(sample_config)
        context = {}

        result = stage.execute(context)

        assert result["cache_loaded"] is False

    def test_image_discovery_stage_no_images(self, sample_config):
        """Test image discovery when no images exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.input_dir = Path(temp_dir)
            stage = ImageDiscoveryStage(sample_config)
            context = {}

            with pytest.raises(RuntimeError, match="No images found"):
                stage.execute(context)

    @patch("colonyscanalyser.io.find_image_files")
    def test_image_discovery_stage_success(self, mock_find_files, sample_config):
        """Test successful image discovery."""
        # Mock image file discovery
        mock_paths = [Path("/test/img1.jpg"), Path("/test/img2.jpg")]
        mock_find_files.return_value = mock_paths

        stage = ImageDiscoveryStage(sample_config)
        context = {}

        # Mock file existence check for ImageFile creation
        with patch.object(Path, "exists", return_value=True):
            result = stage.execute(context)

        assert "image_files" in result
        assert isinstance(result["image_files"], ImageCollection)
        assert len(result["image_files"]) == 2

    def test_image_alignment_stage_disabled(self, sample_config, sample_image_files):
        """Test image alignment when disabled."""
        sample_config.enable_alignment = False
        stage = ImageAlignmentStage(sample_config)
        context = {"image_files": sample_image_files}

        result = stage.execute(context)

        # Should return unchanged context
        assert result == context

    @patch("colonyscanalyser.io.load_image")
    @patch("colonyscanalyser.services.alignment.align_images_fft")
    def test_image_alignment_stage_fft(
        self, mock_align, mock_load, sample_config, sample_image_files
    ):
        """Test FFT-based image alignment."""
        sample_config.enable_alignment = True

        # Mock image loading and alignment
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_load.return_value = mock_image
        mock_align.return_value = mock_image

        stage = ImageAlignmentStage(sample_config)
        context = {"image_files": sample_image_files}

        result = stage.execute(context)

        assert result["images_aligned"] is True

    def test_plate_detection_stage(self, sample_config, sample_image_files):
        """Test plate detection stage."""
        with patch("colonyscanalyser.io.load_image") as mock_load:
            # Mock image loading
            mock_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
            mock_load.return_value = mock_image

            stage = PlateDetectionStage(sample_config)
            context = {"image_files": sample_image_files}

            result = stage.execute(context)

            assert "plates" in result
            assert "noise_masks" in result
            assert isinstance(result["plates"], PlateCollection)
            assert len(result["plates"]) >= 1  # At least one plate

    def test_timepoint_extraction_stage(
        self, sample_config, sample_image_files, sample_plates
    ):
        """Test timepoint extraction stage."""
        with patch("colonyscanalyser.io.load_image") as mock_load:
            with patch(
                "colonyscanalyser.services.segmentation.segment_plate_image"
            ) as mock_segment:
                with patch(
                    "colonyscanalyser.services.detection.timepoints_from_image"
                ) as mock_timepoints:
                    # Setup mocks
                    mock_load.return_value = np.random.randint(
                        0, 255, (200, 200, 3), dtype=np.uint8
                    )
                    mock_segment.return_value = np.zeros((100, 100), dtype=int)
                    mock_timepoints.return_value = [MagicMock()]

                    # Create noise masks
                    noise_masks = {
                        plate.id: np.ones((100, 100), dtype=bool)
                        for plate in sample_plates
                    }

                    stage = TimePointExtractionStage(sample_config)
                    context = {
                        "image_files": sample_image_files,
                        "plates": sample_plates,
                        "noise_masks": noise_masks,
                    }

                    result = stage.execute(context)

                    assert "timepoints" in result
                    assert isinstance(result["timepoints"], dict)

    def test_colony_tracking_stage(
        self, sample_config, sample_plates, sample_timepoints
    ):
        """Test colony tracking stage."""
        with patch(
            "colonyscanalyser.services.tracking.create_colonies_from_timepoints"
        ) as mock_create:
            with patch(
                "colonyscanalyser.services.tracking.filter_colonies"
            ) as mock_filter:
                # Setup mocks
                mock_colonies = [MagicMock()]
                mock_create.return_value = mock_colonies
                mock_filter.return_value = mock_colonies

                # Create timepoints dict
                timepoints = {plate.id: sample_timepoints for plate in sample_plates}

                stage = ColonyTrackingStage(sample_config)
                context = {
                    "plates": sample_plates,
                    "timepoints": timepoints,
                }

                result = stage.execute(context)

                assert "plates" in result
                # Verify colonies were added to plates
                for plate in result["plates"]:
                    assert hasattr(plate, "colonies")

    def test_visualization_stage_disabled(
        self, sample_config, sample_plates, sample_image_files
    ):
        """Test visualization stage when disabled."""
        sample_config.enable_visualization = False
        stage = VisualizationStage(sample_config)
        context = {
            "plates": sample_plates,
            "image_files": sample_image_files,
        }

        result = stage.execute(context)

        # Should return unchanged context
        assert result == context

    def test_data_save_stage(self, sample_config, sample_plates):
        """Test data saving stage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.output_dir = Path(temp_dir)

            with patch("colonyscanalyser.io.save_data_pickle") as mock_save:
                with patch("colonyscanalyser.io.export_colony_data") as mock_export:
                    stage = DataSaveStage(sample_config)
                    context = {"plates": sample_plates}

                    result = stage.execute(context)

                    assert result["data_saved"] is True
                    assert mock_save.called
                    assert mock_export.called


class TestPipelineProcessor:
    """Test the main pipeline processor."""

    def test_processor_initialization(self, sample_config):
        """Test processor initialization."""
        processor = SimplePipelineProcessor(sample_config)

        assert processor.config == sample_config
        assert len(processor.stages) > 0
        assert isinstance(processor.context, dict)

    def test_processor_stage_management(self, sample_config):
        """Test adding and removing stages."""
        processor = SimplePipelineProcessor(sample_config)
        initial_count = len(processor.stages)

        # Remove a stage type
        processor.remove_stage(VisualizationStage)
        assert len(processor.stages) == initial_count - 1

        # Add it back
        viz_stage = VisualizationStage(sample_config)
        processor.add_stage(viz_stage)
        assert len(processor.stages) == initial_count

        # Get a stage
        found_stage = processor.get_stage(VisualizationStage)
        assert found_stage is viz_stage

    def test_processor_configuration_based_stages(self, sample_config):
        """Test that stages are configured based on settings."""
        # Disable visualization
        sample_config.enable_visualization = False
        processor = SimplePipelineProcessor(sample_config)

        viz_stage = processor.get_stage(VisualizationStage)
        assert viz_stage is None

    @patch.object(CacheLoadStage, "execute")
    @patch.object(ImageDiscoveryStage, "execute")
    def test_processor_execution_flow(self, mock_discovery, mock_cache, sample_config):
        """Test processor execution flow."""
        # Setup mocks
        mock_cache.return_value = {"cache_loaded": False}
        mock_discovery.return_value = {
            "cache_loaded": False,
            "image_files": MagicMock(),
        }

        processor = SimplePipelineProcessor(sample_config)

        # Run with minimal stages for testing
        processor.stages = processor.stages[:2]  # Only cache and discovery

        result = processor.run()

        assert isinstance(result, dict)
        assert mock_cache.called
        assert mock_discovery.called

    def test_processor_early_termination_cache_loaded(self, sample_config):
        """Test early termination when cache is loaded."""
        with patch.object(CacheLoadStage, "execute") as mock_cache:
            mock_cache.return_value = {"cache_loaded": True, "plates": MagicMock()}

            processor = SimplePipelineProcessor(sample_config)
            initial_stage_count = len(processor.stages)

            # Mock the context update
            processor.context = {"cache_loaded": True}

            # Check early termination logic
            should_terminate = processor._should_terminate_early()
            assert should_terminate is False  # Should continue to visualization

    def test_processor_error_handling(self, sample_config):
        """Test processor error handling."""
        processor = SimplePipelineProcessor(sample_config)

        # Create a stage that will fail
        class FailingStage:
            def execute(self, context):
                raise ValueError("Test error")

        processor.stages = [FailingStage()]

        with pytest.raises(RuntimeError, match="Pipeline failed"):
            processor.run()

    def test_backward_compatibility_alias(self, sample_config):
        """Test that ColonyPipelineProcessor is a working alias."""
        processor = ColonyPipelineProcessor(sample_config)
        assert isinstance(processor, SimplePipelineProcessor)


class TestPipelineFactories:
    """Test pipeline factory functions."""

    def test_create_standard_pipeline(self, sample_config):
        """Test standard pipeline creation."""
        processor = create_standard_pipeline(sample_config)

        assert isinstance(processor, SimplePipelineProcessor)
        assert len(processor.stages) > 0

        # Should have all major stages
        assert processor.get_stage(ImageDiscoveryStage) is not None
        assert processor.get_stage(PlateDetectionStage) is not None
        assert processor.get_stage(ColonyTrackingStage) is not None

    def test_create_minimal_pipeline(self, sample_config):
        """Test minimal pipeline creation."""
        processor = create_minimal_pipeline(sample_config)

        assert isinstance(processor, SimplePipelineProcessor)

        # Should not have optional stages
        assert processor.get_stage(ImageAlignmentStage) is None
        assert processor.get_stage(VisualizationStage) is None
        assert processor.get_stage(DataSaveStage) is None

    def test_create_visualization_only_pipeline(self, sample_config):
        """Test visualization-only pipeline creation."""
        processor = create_visualization_only_pipeline(sample_config)

        assert isinstance(processor, SimplePipelineProcessor)

        # Should only have specific stages
        assert len(processor.stages) == 3
        assert processor.get_stage(CacheLoadStage) is not None
        assert processor.get_stage(ImageDiscoveryStage) is not None
        assert processor.get_stage(VisualizationStage) is not None


class TestPipelineIntegration:
    """Integration tests for complete pipeline workflows."""

    def test_cache_workflow(self, sample_config):
        """Test workflow when cached data is available."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.input_dir = Path(temp_dir)
            sample_config.enable_caching = True

            # Create a mock cache file
            cache_dir = Path(temp_dir) / "data"
            cache_dir.mkdir()
            cache_file = cache_dir / "cached_analysis.pkl"

            # Mock successful cache loading by directly setting context
            processor = create_visualization_only_pipeline(sample_config)

            # Create mock plates data
            mock_plates = PlateCollection()
            test_plate = Plate(id=1, diameter=100, name="Test Plate", center=(50, 50))
            mock_plates.add(test_plate)

            # Set up the context as if cache was loaded
            processor.context = {"cache_loaded": True, "plates": mock_plates}

            with patch("colonyscanalyser.io.find_image_files") as mock_find:
                # Mock image discovery for visualization
                mock_find.return_value = [Path(temp_dir) / "test.jpg"]

                with patch.object(Path, "exists", return_value=True):
                    # Should complete without error
                    result = processor.run()
                    assert isinstance(result, dict)

    def test_full_pipeline_workflow_mock(self, sample_config):
        """Test complete pipeline workflow with mocked services."""
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_config.input_dir = Path(temp_dir)

            # Create dummy image files
            img_dir = Path(temp_dir)
            for i in range(2):
                (img_dir / f"test_{i}.jpg").touch()

            # Mock all the services
            with patch("colonyscanalyser.io.find_image_files") as mock_find:
                with patch("colonyscanalyser.io.load_image") as mock_load_img:
                    with patch(
                        "colonyscanalyser.services.segmentation.segment_plate_image"
                    ) as mock_segment:
                        with patch(
                            "colonyscanalyser.services.detection.timepoints_from_image"
                        ) as mock_tp:
                            with patch(
                                "colonyscanalyser.services.tracking.create_colonies_from_timepoints"
                            ) as mock_create:
                                with patch(
                                    "colonyscanalyser.services.tracking.filter_colonies"
                                ) as mock_filter:
                                    # Setup mocks
                                    mock_find.return_value = [
                                        img_dir / "test_0.jpg",
                                        img_dir / "test_1.jpg",
                                    ]
                                    mock_load_img.return_value = np.random.randint(
                                        0, 255, (200, 200, 3), dtype=np.uint8
                                    )
                                    mock_segment.return_value = np.zeros(
                                        (100, 100), dtype=int
                                    )
                                    mock_tp.return_value = []
                                    mock_create.return_value = []
                                    mock_filter.return_value = []

                                    processor = create_minimal_pipeline(sample_config)

                                    # Should complete successfully
                                    result = processor.run()
                                    assert isinstance(result, dict)
                                    assert "plates" in result


class TestPipelineProgress:
    """Test progress reporting functionality."""

    def test_progress_callback(self, sample_config):
        """Test progress callback functionality."""
        progress_calls = []

        def progress_callback(current, total, message=""):
            progress_calls.append((current, total, message))

        processor = SimplePipelineProcessor(
            sample_config, progress_callback=progress_callback
        )

        # Create a simple stage that reports progress
        class ProgressStage:
            def __init__(self, config, progress_callback=None):
                self.progress_callback = progress_callback

            def execute(self, context):
                if self.progress_callback:
                    self.progress_callback(1, 2, "Test progress")
                return context

        processor.stages = [ProgressStage(sample_config, progress_callback)]

        result = processor.run()

        assert len(progress_calls) > 0
        assert progress_calls[0] == (1, 2, "Test progress")


class TestPipelineLogging:
    """Test pipeline logging functionality."""

    def test_default_logger_creation(self, sample_config):
        """Test default logger creation."""
        processor = SimplePipelineProcessor(sample_config)

        assert processor.logger is not None
        assert processor.logger.name.endswith("SimplePipelineProcessor")

    def test_custom_logger(self, sample_config):
        """Test using custom logger."""
        import logging

        custom_logger = logging.getLogger("test_logger")

        processor = SimplePipelineProcessor(sample_config, logger=custom_logger)

        assert processor.logger == custom_logger

    def test_logging_levels(self, sample_config):
        """Test different logging levels based on configuration."""
        # Test basic processor creation
        processor = SimplePipelineProcessor(sample_config)

        # Logger should exist
        assert processor.logger is not None


class TestPipelineConfiguration:
    """Test pipeline behavior with different configurations."""

    def test_multiprocessing_configuration(self, sample_config):
        """Test pipeline behavior with multiprocessing enabled."""
        sample_config.parallel_processing = True

        processor = SimplePipelineProcessor(sample_config)

        # Should initialize without error
        assert processor.config.parallel_processing is True

    def test_plate_lattice_configuration(self, sample_config):
        """Test different plate lattice configurations."""
        sample_config.plate_diameter_mm = 90.0

        stage = PlateDetectionStage(sample_config)

        with patch("colonyscanalyser.io.load_image") as mock_load:
            mock_image = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
            mock_load.return_value = mock_image

            context = {"image_files": MagicMock()}
            result = stage.execute(context)

            # Should create plates based on configuration
            assert len(result["plates"]) >= 1

    def test_image_format_configuration(self, sample_config):
        """Test different image format configurations."""
        sample_config.image_pattern = "*.png"

        with patch("colonyscanalyser.io.find_image_files") as mock_find:
            mock_find.return_value = []

            stage = ImageDiscoveryStage(sample_config)
            context = {}

            with pytest.raises(RuntimeError):
                stage.execute(context)

            # Verify function was called
            mock_find.assert_called()
