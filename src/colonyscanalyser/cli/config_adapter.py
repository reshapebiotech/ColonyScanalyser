"""
Configuration adapter for ColonyScanalyser CLI.

This module provides an adapter to convert legacy configuration
to the new configuration format, enabling CLI integration with
the new pipeline architecture.
"""

from typing import List

from ..core.config import PipelineConfig as LegacyPipelineConfig
from ..models.config import PipelineConfig as NewPipelineConfig


class ConfigAdapter:
    """Adapter to convert legacy configuration to new configuration format."""

    @staticmethod
    def convert_legacy_to_new(legacy_config: LegacyPipelineConfig) -> NewPipelineConfig:
        """
        Convert legacy PipelineConfig to new PipelineConfig format.

        Args:
            legacy_config: Legacy configuration object

        Returns:
            New configuration object with mapped values
        """
        # Map basic paths
        input_dir = legacy_config.processing.base_path
        output_dir = input_dir / "output"
        cache_dir = (
            input_dir / ".cache" if legacy_config.processing.use_cached_data else None
        )

        # Map image formats to pattern
        image_pattern = ConfigAdapter._convert_image_formats_to_pattern(
            legacy_config.processing.image_formats
        )

        # Map boolean flags
        enable_caching = legacy_config.processing.use_cached_data
        enable_alignment = legacy_config.processing.image_align_strategy.name != "none"
        enable_visualization = getattr(legacy_config, "visualization", None) is not None
        parallel_processing = not legacy_config.processing.single_process

        # Map plate parameters
        plate_diameter_mm = legacy_config.processing.plate_size_mm
        plate_edge_cut_mm = (
            legacy_config.processing.plate_edge_cut_percent / 100.0 * plate_diameter_mm
        )

        # Create new config with additional extensions attribute
        new_config = NewPipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            enable_caching=enable_caching,
            enable_alignment=enable_alignment,
            enable_visualization=enable_visualization,
            parallel_processing=parallel_processing,
            image_pattern=image_pattern,
            output_format="csv",  # Default format
            plate_diameter_mm=plate_diameter_mm,
            plate_edge_cut_mm=plate_edge_cut_mm,
        )

        # Store extensions for pipeline stages to use
        new_config.image_extensions = legacy_config.processing.image_formats

        return new_config

    @staticmethod
    def _convert_image_formats_to_pattern(formats: List[str]) -> str:
        """
        Convert list of image formats to a file pattern.

        Args:
            formats: List of file extensions (e.g., ["tif", "png"])

        Returns:
            File pattern string (e.g., "*.{tif,png}")
        """
        if not formats:
            return "*.tif"  # Default pattern

        if len(formats) == 1:
            return f"*.{formats[0]}"

        # Create pattern for multiple formats
        formats_str = ",".join(formats)
        return f"*.{{{formats_str}}}"

    @staticmethod
    def get_image_extensions_from_legacy(
        legacy_config: LegacyPipelineConfig,
    ) -> List[str]:
        """
        Extract image extensions from legacy configuration.

        Args:
            legacy_config: Legacy configuration object

        Returns:
            List of image file extensions
        """
        return legacy_config.processing.image_formats

    @staticmethod
    def create_output_directories(config: NewPipelineConfig) -> None:
        """
        Ensure output directories exist.

        Args:
            config: New configuration object
        """
        config.output_dir.mkdir(parents=True, exist_ok=True)

        if config.cache_dir:
            config.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_legacy_visualization_config(legacy_config: LegacyPipelineConfig) -> dict:
        """
        Extract visualization configuration from legacy config.

        Args:
            legacy_config: Legacy configuration object

        Returns:
            Dictionary with visualization settings
        """
        viz_config = {
            "enable_plots": True,
            "enable_animation": False,
            "visualization_types": ["ids", "comprehensive"],
            "output_dir": "visualizations",
            "dpi": 300,
        }

        # Extract from legacy config if available
        if hasattr(legacy_config, "visualization"):
            viz = legacy_config.visualization
            viz_config.update(
                {
                    "enable_plots": not getattr(viz, "no_plots", False),
                    "enable_animation": getattr(viz, "animation", False),
                    "visualization_types": getattr(
                        viz, "types", ["ids", "comprehensive"]
                    ),
                    "output_dir": getattr(viz, "output_dir", "visualizations"),
                    "dpi": getattr(viz, "dots_per_inch", 300),
                }
            )

        return viz_config
