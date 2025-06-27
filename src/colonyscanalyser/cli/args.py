"""
Command line argument parsing for ColonyScanalyser.

This module provides argument parsing functionality for the ColonyScanalyser CLI.
"""

import argparse

try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python, use importlib-metadata package
    import importlib_metadata as metadata

from .. import config
from ..align.strategy import AlignStrategy


def create_parser(*args, **kwargs) -> argparse.ArgumentParser:
    """
    Create an ArgumentParser instance with the standard package arguments.

    :param args: positional arguments to pass to the ArgumentParser initialiser
    :param kwargs: keyword arguments to pass to the ArgumentParser initialiser
    :returns: an ArgumentParser instance with the standard package arguments
    """
    parser = argparse.ArgumentParser(
        prog="colonyscanalyser '/image/file/path/' [OPTIONS]",
        description="An image analysis tool for measuring microorganism colony growth",
        *args,
        **kwargs,
    )

    # Mutually exclusive options
    output = parser.add_mutually_exclusive_group()

    parser.add_argument("path", type=str, help="Image files location", default=None)

    parser.add_argument(
        "-a",
        "--animation",
        action="store_true",
        help="Output animated plots and videos",
    )
    parser.add_argument(
        "-d",
        "--dots-per-inch",
        type=int,
        default=config.DOTS_PER_INCH,
        metavar="N",
        help="The image DPI (dots per inch) setting",
    )
    parser.add_argument(
        "--image-align",
        nargs="?",
        default=AlignStrategy.quick.name,
        const=AlignStrategy.quick.name,
        choices=[strategy.name for strategy in AlignStrategy],
        help="The strategy used for aligning images for analysis",
    )
    parser.add_argument(
        "--image-align-tolerance",
        type=float,
        default=config.ALIGNMENT_TOLERANCE,
        help="The tolerance value allowed when aligning images. 0 means the images must match exactly",
        metavar="N",
    )
    parser.add_argument(
        "--image-formats",
        default=config.SUPPORTED_FORMATS,
        action="version",
        version=str(config.SUPPORTED_FORMATS),
        help="The supported image formats",
    )
    parser.add_argument(
        "--no-plots", action="store_true", help="Prevent output of plot images to disk"
    )
    parser.add_argument(
        "--plate-edge-cut",
        type=int,
        default=config.PLATE_EDGE_CUT,
        help="The exclusion area from the plate edge, as a percentage of the plate diameter",
        metavar="N",
    )
    parser.add_argument(
        "--plate-labels",
        type=str,
        nargs="*",
        default=list(),
        metavar="LABEL",
        help="A list of labels to identify each plate. Plates are ordered from top left, in rows. Example usage: --plate_labels plate1 plate2",
    )
    parser.add_argument(
        "--plate-lattice",
        type=int,
        nargs=2,
        default=config.PLATE_LATTICE,
        metavar=("ROW", "COL"),
        help="The row and column co-ordinate layout of plates. Example usage: --plate_lattice 3 3",
    )
    parser.add_argument(
        "--plate-size",
        type=int,
        default=config.PLATE_SIZE,
        help="The plate diameter, in millimetres",
        metavar="N",
    )
    output.add_argument(
        "-s", "--silent", action="store_true", help="Silence all output to console"
    )
    parser.add_argument(
        "--single-process",
        action="store_true",
        help="Use only a single CPU core, slower but less resource intensive",
    )
    parser.add_argument(
        "-u",
        "--use-cached-data",
        action="store_true",
        help="Allow use of previously calculated data",
    )
    output.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Output extra information to console",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"ColonyScanalyser {metadata.version('colonyscanalyser')}",
        help="The package version number",
    )

    return parser
