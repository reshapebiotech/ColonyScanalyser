"""
Main CLI entry point for ColonyScanalyser.

This module provides a clean command line interface entry point that
delegates to the pipeline processor for all actual work.
"""

from ..core.config import PipelineConfig
from ..pipeline import ColonyPipelineProcessor
from .args import create_parser


def main():
    """
    Main entry point for the ColonyScanalyser CLI.

    Parses arguments, creates configuration, and runs the pipeline processor.
    """
    # Parse command line arguments
    parser = create_parser()
    args = parser.parse_args()

    # Create and validate configuration
    config = PipelineConfig.from_args(args)
    config.validate()

    # Create and run pipeline processor
    processor = ColonyPipelineProcessor(config)
    processor.run()
