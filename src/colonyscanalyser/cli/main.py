"""
Main CLI entry point for ColonyScanalyser.

This module provides a clean command line interface entry point that
delegates to the pipeline processor for all actual work.
"""

from colonyscanalyser.cli.args import create_parser
from colonyscanalyser.cli.config_adapter import ConfigAdapter
from colonyscanalyser.core.config import PipelineConfig as LegacyPipelineConfig
from colonyscanalyser.pipeline import SimplePipelineProcessor


def main():
    """
    Main entry point for the ColonyScanalyser CLI.

    Parses arguments, creates configuration, and runs the pipeline processor.
    """
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Create and validate legacy configuration
    legacy_config = LegacyPipelineConfig.from_args(args)
    legacy_config.validate()

    # Convert to new configuration format
    config = ConfigAdapter.convert_legacy_to_new(legacy_config)

    # Ensure output directories exist
    ConfigAdapter.create_output_directories(config)

    # Create and run new pipeline processor
    processor = SimplePipelineProcessor(config)
    processor.run()
