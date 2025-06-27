"""
Main CLI entry point for ColonyScanalyser.

This module provides the main command line interface entry point for the
ColonyScanalyser tool, handling argument parsing and delegating to the
appropriate analysis functions.
"""

from multiprocessing import cpu_count
from pathlib import Path

from .. import config
from ..align.strategy import AlignStrategy
from ..core import ImageFileCollection, Plate, PlateCollection
from ..io import CompressionMethod, load_file
from ..processing.imaging import mm_to_pixels
from .args import create_parser


def main():
    """
    Main entry point for the ColonyScanalyser CLI.
    """
    parser = create_parser()
    args = parser.parse_args()

    # --- Argument parsing and setup ---
    BASE_PATH = args.path
    ANIMATION = args.animation
    IMAGE_ALIGN_STRATEGY = AlignStrategy[args.image_align]
    IMAGE_ALIGN_TOLERANCE = args.image_align_tolerance
    IMAGE_FORMATS = args.image_formats
    PLOTS = not args.no_plots
    PLATE_LABELS = {
        plate_id: label for plate_id, label in enumerate(args.plate_labels, start=1)
    }
    PLATE_LATTICE = tuple(args.plate_lattice)
    PLATE_SIZE = int(mm_to_pixels(args.plate_size, dots_per_inch=args.dots_per_inch))
    PLATE_EDGE_CUT = int(round(PLATE_SIZE * (args.plate_edge_cut / 100)))
    SILENT = args.silent
    USE_CACHED = args.use_cached_data
    VERBOSE = args.verbose
    POOL_MAX = 1
    if not args.single_process:
        POOL_MAX = cpu_count() - 1 if cpu_count() > 1 else 1

    if not SILENT:
        print("Starting ColonyScanalyser analysis")
    if VERBOSE and POOL_MAX > 1:
        print(
            f"Multiprocessing enabled, utilising {POOL_MAX} of {cpu_count()} processors"
        )

    # Resolve working directory
    if BASE_PATH is None:
        raise ValueError("A path to a working directory must be supplied")
    else:
        BASE_PATH = Path(args.path).resolve()
    if not BASE_PATH.exists():
        raise EnvironmentError(
            f"The supplied folder path could not be found: {BASE_PATH}"
        )
    if not SILENT:
        print(f"Working directory: {BASE_PATH}")

    # Check if processed image data is already stored and can be loaded
    plates = None
    if USE_CACHED:
        if not SILENT:
            print("Attempting to load cached data")
        plates = load_file(
            BASE_PATH.joinpath(config.DATA_DIR, config.CACHED_DATA_FILE_NAME),
            CompressionMethod.LZMA,
            pickle=True,
        )
        # Check that segmented image data has been loaded for all plates
        # Also that data is not from an older format (< v0.4.0)
        if (
            VERBOSE
            and plates is not None
            and plates.count == PlateCollection.coordinate_to_index(PLATE_LATTICE)
            and isinstance(plates.items[0], Plate)
        ):
            print("Successfully loaded cached data")
            image_files = None
        else:
            print("Unable to load cached data, starting image processing")
            plates = None

    if not USE_CACHED or plates is None:
        # Find images in working directory. Raises IOError if images not loaded correctly
        image_files = ImageFileCollection.from_path(
            BASE_PATH, IMAGE_FORMATS, cache_images=False
        )
        if not SILENT:
            print(f"{image_files.count} images found")

        # TODO: Add image alignment logic here
        # TODO: Add plate detection logic here
        # TODO: Add colony segmentation logic here
        # TODO: Add colony tracking logic here

    # TODO: Add data persistence logic here
    # TODO: Add visualization logic here

    if not SILENT:
        print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")
