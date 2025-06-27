"""
Main CLI entry point for ColonyScanalyser.

This module provides the main command line interface entry point for the
ColonyScanalyser tool, handling argument parsing and delegating to the
appropriate analysis functions.
"""

import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List

from .. import config
from ..align.strategy import (
    AlignStrategy,
    apply_align_transform,
    calculate_transformation_strategy,
)
from ..core import ImageFileCollection, Plate, PlateCollection, timepoints_from_image
from ..io import CompressionMethod, load_file
from ..processing.imaging import mm_to_pixels
from ..processing.segmentation import segment_image
from ..utils.utilities import dicts_merge, progress_bar
from .args import create_parser


def image_file_to_timepoints(
    image_file, plates: PlateCollection, plate_noise_masks: Dict[int, "ndarray"]
) -> Dict[int, List]:
    """
    Get Timepoint object data from a plate image

    :param image_file: an ImageFile object
    :param plates: a PlateCollection of Plate instances
    :param plate_noise_masks: a dict of plate images to use as noise masks
    :returns: a Dict of lists containing Timepoints, with the plate number as keys
    """
    from collections import defaultdict

    from skimage.color import rgb2gray

    plate_timepoints = defaultdict(list)

    # Split image into individual plates
    with image_file as img:
        plate_images = plates.slice_plate_images(img.image)

        for plate_id, plate_image in plate_images.items():
            plate_image_gray = rgb2gray(plate_image)

            # Segment each image
            segmented_image = segment_image(
                plate_image_gray,
                plate_mask=plate_image_gray > 0,
                plate_noise_mask=plate_noise_masks[plate_id],
                area_min=1.5,
            )

            # Create Timepoint objects for each plate
            plate_timepoints[plate_id].extend(
                timepoints_from_image(
                    segmented_image, img.timestamp_elapsed, image=plate_image
                )
            )

    return plate_timepoints


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

        # Verify image alignment
        if IMAGE_ALIGN_STRATEGY != AlignStrategy.none:
            if not SILENT:
                print(
                    f"Verifying image alignment with '{IMAGE_ALIGN_STRATEGY.name}' strategy. This process will take some time"
                )

            # Initialise the model and determine which images need alignment
            align_model, image_files_align = calculate_transformation_strategy(
                image_files.items, IMAGE_ALIGN_STRATEGY, tolerance=IMAGE_ALIGN_TOLERANCE
            )

            # Apply image alignment according to selected strategy
            if len(image_files_align) > 0:
                if not SILENT:
                    print(
                        f"{len(image_files_align)} of {image_files.count} images require alignment"
                    )

                with Pool(processes=POOL_MAX) as pool:
                    results = list()
                    job = pool.imap_unordered(
                        func=partial(apply_align_transform, align_model=align_model),
                        iterable=image_files_align,
                        chunksize=2,
                    )
                    # Store results and update progress bar
                    for i, result in enumerate(job, start=1):
                        results.append(result)
                        if not SILENT:
                            progress_bar(
                                (i / len(image_files_align)) * 100,
                                message="Correcting image alignment",
                            )

                    image_files.update(results)

        # Process images to Timepoint data objects
        plate_images_mask = None
        plate_timepoints = defaultdict(list)

        if not SILENT:
            print("Preprocessing images to locate plates")

        # Load the first image to get plate coordinates and mask
        with image_files.items[0] as image_file:
            # Only find centers using first image. Assume plates do not move
            if plates is None:
                if VERBOSE:
                    print(f"Locating plate centres in image: {image_file.file_path}")

                # Create new Plate instances to store the information
                plates = PlateCollection.from_image(
                    shape=PLATE_LATTICE,
                    image=image_file.image_gray,
                    diameter=PLATE_SIZE,
                    search_radius=PLATE_SIZE // 20,
                    edge_cut=PLATE_EDGE_CUT,
                    labels=PLATE_LABELS,
                )

                if not plates.count > 0:
                    if not SILENT:
                        print(
                            f"Unable to locate plates in image: {image_file.file_path}"
                        )
                        print("Processing unable to continue")
                    sys.exit()

                if VERBOSE:
                    for plate in plates.items:
                        print(f"Plate {plate.id} center: {plate.center}")

            # Use the first plate image as a noise mask
            plate_noise_masks = plates.slice_plate_images(image_file.image_gray)

        if not SILENT:
            print("Processing colony data from all images")

        # Process images to Timepoints
        with Pool(processes=POOL_MAX) as pool:
            results = list()
            job = pool.imap(
                func=partial(
                    image_file_to_timepoints,
                    plates=plates,
                    plate_noise_masks=plate_noise_masks,
                ),
                iterable=image_files.items,
                chunksize=2,
            )
            # Store results and update progress bar
            for i, result in enumerate(job, start=1):
                results.append(result)
                if not SILENT:
                    progress_bar(
                        (i / image_files.count) * 100, message="Processing images"
                    )
            plate_timepoints = dicts_merge(list(results))

        # TODO: Add colony tracking logic here

    # TODO: Add data persistence logic here
    # TODO: Add visualization logic here

    if not SILENT:
        print(f"ColonyScanalyser analysis completed for: {BASE_PATH}")
