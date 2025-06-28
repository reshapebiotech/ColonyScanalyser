"""
Configuration data structures for ColonyScanalyser.

This module provides type-safe configuration classes that encapsulate
all the settings and parameters used throughout the ColonyScanalyser pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..align.strategy import AlignStrategy


@dataclass
class ProcessingConfig:
    """Configuration for image processing and analysis pipeline."""

    # Core paths and files
    base_path: Path
    use_cached_data: bool = False

    # Image processing
    image_formats: List[str] = field(
        default_factory=lambda: ["tif", "tiff", "png", "bmp", "jpeg", "jpg"]
    )
    dots_per_inch: int = 300

    # Image alignment
    image_align_strategy: AlignStrategy = AlignStrategy.quick
    image_align_tolerance: float = 3.0

    # Plate detection
    plate_lattice: Tuple[int, int] = (3, 2)
    plate_size_mm: float = 90.0
    plate_size_pixels: Optional[int] = None  # Calculated from mm and DPI
    plate_edge_cut_percent: float = 5.0
    plate_edge_cut_pixels: Optional[int] = None  # Calculated from plate size
    plate_labels: Dict[int, str] = field(default_factory=dict)

    # Colony tracking
    colony_distance_max: float = 2.0
    colony_timepoints_min: int = 3
    colony_timestamp_diff_max: float = 10.0
    colony_growth_factor_min: float = 4.0
    colony_first_area_max: float = 200.0

    # Performance
    single_process: bool = False
    pool_max: Optional[int] = None  # Calculated based on single_process and CPU count

    # Output control
    silent: bool = False
    verbose: bool = False
    animation: bool = False
    plots: bool = True

    def __post_init__(self):
        """Calculate derived values after initialization."""
        from multiprocessing import cpu_count

        from ..utils.geometry import mm_to_pixels

        # Calculate pixel values from millimeters
        self.plate_size_pixels = int(
            mm_to_pixels(self.plate_size_mm, self.dots_per_inch)
        )
        self.plate_edge_cut_pixels = int(
            round(self.plate_size_pixels * (self.plate_edge_cut_percent / 100))
        )

        # Calculate pool size
        if self.single_process:
            self.pool_max = 1
        else:
            self.pool_max = cpu_count() - 1 if cpu_count() > 1 else 1

    @classmethod
    def from_args(cls, args) -> "ProcessingConfig":
        """Create configuration from parsed command line arguments."""
        return cls(
            base_path=Path(args.path).resolve(),
            use_cached_data=args.use_cached_data,
            image_formats=args.image_formats,
            dots_per_inch=args.dots_per_inch,
            image_align_strategy=AlignStrategy[args.image_align],
            image_align_tolerance=args.image_align_tolerance,
            plate_lattice=tuple(args.plate_lattice),
            plate_size_mm=args.plate_size,
            plate_edge_cut_percent=args.plate_edge_cut,
            plate_labels={
                plate_id: label
                for plate_id, label in enumerate(args.plate_labels, start=1)
            },
            single_process=args.single_process,
            silent=args.silent,
            verbose=args.verbose,
            animation=args.animation,
            plots=not args.no_plots,
        )


@dataclass
class VisualizationConfig:
    """Configuration for colony visualizations."""

    enabled: bool = False
    output_dir: str = "visualizations"
    types: List[str] = field(default_factory=lambda: ["ids", "comprehensive"])
    use_full_image: bool = True
    save_plate_only: bool = False

    # Visualization appearance
    font_size: int = 10
    line_width: float = 1.0
    mask_alpha: float = 0.3

    @classmethod
    def from_args(cls, args) -> "VisualizationConfig":
        """Create visualization config from parsed arguments."""
        return cls(
            enabled=args.visualize,
            output_dir=args.visualization_dir,
            types=args.visualization_types,
        )


@dataclass
class PipelineConfig:
    """Complete configuration for the ColonyScanalyser pipeline."""

    processing: ProcessingConfig
    visualization: VisualizationConfig

    @classmethod
    def from_args(cls, args) -> "PipelineConfig":
        """Create complete pipeline configuration from command line arguments."""
        return cls(
            processing=ProcessingConfig.from_args(args),
            visualization=VisualizationConfig.from_args(args),
        )

    def validate(self) -> None:
        """Validate the configuration and raise errors for invalid settings."""
        if not self.processing.base_path.exists():
            raise FileNotFoundError(
                f"Working directory not found: {self.processing.base_path}"
            )

        if self.processing.plate_size_mm <= 0:
            raise ValueError("Plate size must be positive")

        if not (0 <= self.processing.plate_edge_cut_percent <= 100):
            raise ValueError("Plate edge cut must be between 0 and 100 percent")

        if self.processing.image_align_tolerance < 0:
            raise ValueError("Image alignment tolerance must be non-negative")

        valid_viz_types = {"masks", "ids", "outlines", "comprehensive"}
        for viz_type in self.visualization.types:
            if viz_type not in valid_viz_types:
                raise ValueError(
                    f"Invalid visualization type: {viz_type}. Must be one of {valid_viz_types}"
                )
