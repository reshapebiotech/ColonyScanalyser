"""Configuration data models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ProcessingConfig:
    """Configuration for image processing parameters."""

    # Detection parameters
    min_colony_area: int = 50
    max_colony_area: int = 10000
    min_colony_circularity: float = 0.5

    # Tracking parameters
    max_distance_tolerance: float = 15.0
    min_timepoints: int = 3
    min_growth_factor: float = 1.4

    # Image processing
    gaussian_blur_sigma: float = 1.0
    threshold_method: str = "otsu"
    edge_detection_method: str = "canny"

    # Filtering
    timestamp_diff_std_max: float = 10.0


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""

    # Input/Output paths
    input_dir: Path
    output_dir: Path
    cache_dir: Optional[Path] = None

    # Processing options
    enable_caching: bool = True
    enable_alignment: bool = True
    enable_visualization: bool = True
    parallel_processing: bool = False

    # File patterns
    image_pattern: str = "*.tif"
    output_format: str = "csv"

    # Plate detection
    plate_diameter_mm: float = 90.0
    plate_edge_cut_mm: float = 5.0

    def __post_init__(self):
        """Ensure directories are Path objects."""
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
