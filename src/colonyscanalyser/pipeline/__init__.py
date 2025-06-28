"""
Pipeline processing package for ColonyScanalyser.

This package contains the main pipeline orchestration classes
for running the complete colony analysis workflow.
"""

# Legacy processor (for backward compatibility)
# from .processor import ColonyPipelineProcessor as LegacyColonyPipelineProcessor

# New clean architecture
from .processor_clean import (
    ColonyPipelineProcessor,
    SimplePipelineProcessor,
    create_minimal_pipeline,
    create_standard_pipeline,
    create_visualization_only_pipeline,
)
from .stages import (
    CacheLoadStage,
    ColonyTrackingStage,
    DataSaveStage,
    ImageAlignmentStage,
    ImageDiscoveryStage,
    PipelineStage,
    PlateDetectionStage,
    TimePointExtractionStage,
    VisualizationStage,
)

__all__ = [
    # Main processors
    "ColonyPipelineProcessor",
    "SimplePipelineProcessor",
    # Factory functions
    "create_standard_pipeline",
    "create_minimal_pipeline",
    "create_visualization_only_pipeline",
    # Individual stages
    "PipelineStage",
    "CacheLoadStage",
    "ImageDiscoveryStage",
    "ImageAlignmentStage",
    "PlateDetectionStage",
    "TimePointExtractionStage",
    "ColonyTrackingStage",
    "VisualizationStage",
    "DataSaveStage",
    # Legacy
    # "LegacyColonyPipelineProcessor",
]
