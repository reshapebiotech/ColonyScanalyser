"""
Clean pipeline processor for ColonyScanalyser.

This module provides a modern, stage-based pipeline processor that
orchestrates colony analysis workflows with clean separation of concerns.
"""

import logging
import sys
import time
from typing import Any, Dict, List, Optional, Type

from ..models.config import PipelineConfig
from .stages import (
    CacheLoadStage,
    ColonyTrackingStage,
    DataSaveStage,
    ImageAlignmentStage,
    ImageDiscoveryStage,
    PipelineStage,
    PlateDetectionStage,
    ProgressCallback,
    TimePointExtractionStage,
    VisualizationStage,
)


class SimplePipelineProcessor:
    """
    Clean pipeline processor using composable stages.

    This processor orchestrates colony analysis by executing a series
    of pipeline stages in order, maintaining clean state and providing
    comprehensive error handling and progress reporting.
    """

    def __init__(
        self,
        config: PipelineConfig,
        logger: Optional[logging.Logger] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        """
        Initialize pipeline processor.

        Args:
            config: Complete pipeline configuration
            logger: Optional logger for detailed logging
            progress_callback: Optional callback for progress updates
        """
        self.config = config
        self.logger = logger or self._create_default_logger()
        self.progress_callback = progress_callback
        self.context: Dict[str, Any] = {}
        self.stages: List[PipelineStage] = []

        # Initialize stages based on configuration
        self._initialize_stages()

    def run(self) -> Dict[str, Any]:
        """
        Execute the complete pipeline.

        Returns:
            Final pipeline context containing all results

        Raises:
            RuntimeError: If pipeline execution fails
        """
        try:
            self._log_info("Starting ColonyScanalyser analysis")
            self._log_info(f"Working directory: {self.config.input_dir}")

            start_time = time.time()

            # Execute pipeline stages
            for i, stage in enumerate(self.stages):
                stage_name = stage.__class__.__name__
                self._log_info(
                    f"Executing stage {i + 1}/{len(self.stages)}: {stage_name}"
                )

                try:
                    stage_start = time.time()
                    self.context = stage.execute(self.context)
                    stage_duration = time.time() - stage_start

                    self._log_debug(f"{stage_name} completed in {stage_duration:.2f}s")

                    # Check for early termination conditions
                    if self._should_terminate_early():
                        break

                except Exception as e:
                    self._log_error(f"Error in {stage_name}: {e}")
                    raise RuntimeError(f"Pipeline failed at stage {stage_name}") from e

            total_duration = time.time() - start_time
            self._log_info(f"Pipeline completed successfully in {total_duration:.2f}s")

            # Log final results
            self._log_results()

            return self.context

        except KeyboardInterrupt:
            self._log_info("Pipeline interrupted by user")
            sys.exit(1)
        except Exception as e:
            self._log_error(f"Pipeline execution failed: {e}")
            # Always show traceback for debugging
            if True:
                import traceback

                traceback.print_exc()
            raise

    def add_stage(self, stage: PipelineStage) -> None:
        """
        Add a stage to the pipeline.

        Args:
            stage: Pipeline stage to add
        """
        self.stages.append(stage)

    def remove_stage(self, stage_type: Type[PipelineStage]) -> None:
        """
        Remove stages of a specific type.

        Args:
            stage_type: Type of stage to remove
        """
        self.stages = [s for s in self.stages if not isinstance(s, stage_type)]

    def get_stage(self, stage_type: Type[PipelineStage]) -> Optional[PipelineStage]:
        """
        Get the first stage of a specific type.

        Args:
            stage_type: Type of stage to find

        Returns:
            Stage instance or None if not found
        """
        for stage in self.stages:
            if isinstance(stage, stage_type):
                return stage
        return None

    def _initialize_stages(self) -> None:
        """Initialize pipeline stages based on configuration."""
        # Create stage instances with shared configuration
        stage_kwargs = {
            "config": self.config,
            "progress_callback": self.progress_callback,
        }

        # Standard pipeline stages
        self.stages = [
            CacheLoadStage(**stage_kwargs),
            ImageDiscoveryStage(**stage_kwargs),
            ImageAlignmentStage(**stage_kwargs),
            PlateDetectionStage(**stage_kwargs),
            TimePointExtractionStage(**stage_kwargs),
            ColonyTrackingStage(**stage_kwargs),
            VisualizationStage(**stage_kwargs),
            DataSaveStage(**stage_kwargs),
        ]

        # Remove stages based on configuration
        if not self.config.enable_visualization:
            self.remove_stage(VisualizationStage)

        if not self.config.enable_caching:
            self.remove_stage(DataSaveStage)

        if not self.config.enable_alignment:
            self.remove_stage(ImageAlignmentStage)

    def _should_terminate_early(self) -> bool:
        """
        Check if pipeline should terminate early.

        Returns:
            True if pipeline should stop, False otherwise
        """
        # If cache was loaded successfully, skip to visualization
        if self.context.get("cache_loaded", False):
            # Remove processing stages that aren't needed
            self.remove_stage(ImageDiscoveryStage)
            self.remove_stage(ImageAlignmentStage)
            self.remove_stage(PlateDetectionStage)
            self.remove_stage(TimePointExtractionStage)
            self.remove_stage(ColonyTrackingStage)
            self.remove_stage(DataSaveStage)

            # Continue with visualization if enabled
            return False

        # Check if no colonies were found
        plates = self.context.get("plates")
        if plates is not None:
            total_colonies = sum(len(plate.colonies) for plate in plates)
            if total_colonies == 0:
                self._log_info("No colonies found in any plates")
                return True

        return False

    def _log_results(self) -> None:
        """Log final pipeline results."""
        plates = self.context.get("plates")
        if plates:
            total_colonies = sum(len(plate.colonies) for plate in plates)
            self._log_info(
                f"Analysis complete: {total_colonies} colonies found across {len(plates)} plates"
            )

            for plate in plates:
                colony_count = len(plate.colonies)
                self._log_info(
                    f"Plate {plate.id} ({plate.name}): {colony_count} colonies"
                )

        # Log other results
        if "visualization_files" in self.context:
            viz_count = len(self.context["visualization_files"])
            self._log_info(f"Generated {viz_count} visualization files")

        if self.context.get("data_saved", False):
            self._log_info("Analysis data saved for future use")

    def _create_default_logger(self) -> logging.Logger:
        """Create a default logger for pipeline operations."""
        logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Set level to INFO by default
        logger.setLevel(logging.INFO)

        # Create console handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _log_info(self, message: str) -> None:
        """Log info message and print."""
        self.logger.info(message)
        print(message)

    def _log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def _log_error(self, message: str) -> None:
        """Log error message and print."""
        self.logger.error(message)
        print(f"ERROR: {message}")


class ColonyPipelineProcessor(SimplePipelineProcessor):
    """
    Backward-compatible alias for the clean pipeline processor.

    This maintains compatibility with existing code while providing
    the new clean architecture.
    """

    pass


def create_standard_pipeline(config: PipelineConfig) -> SimplePipelineProcessor:
    """
    Create a standard colony analysis pipeline.

    Args:
        config: Pipeline configuration

    Returns:
        Configured pipeline processor
    """
    return SimplePipelineProcessor(config)


def create_minimal_pipeline(config: PipelineConfig) -> SimplePipelineProcessor:
    """
    Create a minimal pipeline for testing or quick analysis.

    Args:
        config: Pipeline configuration

    Returns:
        Minimal pipeline processor
    """
    processor = SimplePipelineProcessor(config)

    # Remove optional stages
    processor.remove_stage(ImageAlignmentStage)
    processor.remove_stage(VisualizationStage)
    processor.remove_stage(DataSaveStage)

    return processor


def create_visualization_only_pipeline(
    config: PipelineConfig,
) -> SimplePipelineProcessor:
    """
    Create a pipeline that only generates visualizations from cached data.

    Args:
        config: Pipeline configuration

    Returns:
        Visualization-only pipeline processor
    """
    processor = SimplePipelineProcessor(config)

    # Keep only cache loading and visualization
    processor.stages = [
        CacheLoadStage(config),
        ImageDiscoveryStage(config),  # Needed for visualization
        VisualizationStage(config),
    ]

    return processor
