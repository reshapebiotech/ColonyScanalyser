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
from typing import Dict, List, Optional, Tuple

from .. import config
from ..align.strategy import (
    AlignStrategy,
    apply_align_transform,
    calculate_transformation_strategy,
)
from ..core import ImageFileCollection, Plate, PlateCollection, timepoints_from_image
from ..core.colony import colonies_filtered, colonies_from_timepoints
from ..io import CompressionMethod, load_file
from ..processing.imaging import mm_to_pixels
from ..processing.segmentation import segment_image
from ..utils.utilities import dicts_merge, progress_bar
from ..visualization import save_colony_visualizations
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


def plates_colonies_from_timepoints(
    plates: PlateCollection,
    timepoints: Dict[int, List],
    timepoints_distance: float = 1,
    timestamp_diff_std: float = 10,
    pool_size=1,
) -> PlateCollection:
    """
    Group a list of Timepoints to Colony objects, and populate in a Plate instance.

    :param plates: a PlateCollection instance associated with the Timepoints
    :param timepoints: a dict of lists of Timepoint instances, with keys corresponding to Plate.id numbers
    :param timepoints_distance: the maximum distance allowed for Colony grouping
    :param timestamp_diff_std: the maximum allowed deviation in timestamps
    :param pool_size: the number of logical processors available for multiprocessing
    :returns: the collection with each plate instance populated with a collection of Colony instances
    """
    # Assemble data to a single iterable for starmap
    timepoints_iter = [
        (plates[plate_id], timepoints_list, timepoints_distance, timestamp_diff_std)
        for plate_id, timepoints_list in timepoints.items()
    ]

    # Process and filter Timepoints to Colony objects in parallel
    with Pool(processes=pool_size) as pool:
        plates.items = pool.starmap(
            func=_plate_colonies_from_timepoints_filtered, iterable=timepoints_iter
        )

    return plates


def _plate_colonies_from_timepoints_filtered(
    plate: Plate,
    timepoints: List,
    timepoints_distance: float,
    timestamp_diff_std: float,
) -> Plate:
    """
    Group a list of Timepoints to Colony objects, and filter to return only valid colonies.

    :param plate: a Plate instance associated with the Timepoints
    :param timepoints: a list of Timepoint instances
    :param timepoints_distance: the maximum distance allowed for Colony grouping
    :param timestamp_diff_std: the maximum allowed deviation in timestamps
    :returns: the plate instance with a collection of Colony instances
    """
    if len(timepoints) > 0:
        # Group Timepoints by Euclidean distance
        plate.items = colonies_from_timepoints(
            timepoints, distance_tolerance=timepoints_distance
        )

        # Filter colonies to remove noise, background objects and merged colonies
        plate.items = colonies_filtered(plate.items, timestamp_diff_std)

    return plate


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = create_parser()
    args = parser.parse_args()

    if args.path is None:
        raise ValueError("A path to a working directory must be supplied")

    return args


def setup_processing_environment(args) -> Tuple[Path, dict]:
    """
    Set up the processing environment and extract configuration.

    :param args: parsed command line arguments
    :returns: tuple of (base_path, config_dict)
    """
    # Extract all configuration from args
    config_dict = {
        "BASE_PATH": Path(args.path).resolve(),
        "ANIMATION": args.animation,
        "IMAGE_ALIGN_STRATEGY": AlignStrategy[args.image_align],
        "IMAGE_ALIGN_TOLERANCE": args.image_align_tolerance,
        "IMAGE_FORMATS": args.image_formats,
        "PLOTS": not args.no_plots,
        "PLATE_LABELS": {
            plate_id: label for plate_id, label in enumerate(args.plate_labels, start=1)
        },
        "PLATE_LATTICE": tuple(args.plate_lattice),
        "PLATE_SIZE": int(
            mm_to_pixels(args.plate_size, dots_per_inch=args.dots_per_inch)
        ),
        "PLATE_EDGE_CUT": None,  # Will be calculated
        "SILENT": args.silent,
        "USE_CACHED": args.use_cached_data,
        "VERBOSE": args.verbose,
        "VISUALIZE": args.visualize,
        "VISUALIZATION_DIR": args.visualization_dir,
        "VISUALIZATION_TYPES": args.visualization_types,
        "POOL_MAX": 1
        if args.single_process
        else (cpu_count() - 1 if cpu_count() > 1 else 1),
    }

    # Calculate derived values
    config_dict["PLATE_EDGE_CUT"] = int(
        round(config_dict["PLATE_SIZE"] * (args.plate_edge_cut / 100))
    )

    # Validate paths
    if not config_dict["BASE_PATH"].exists():
        raise EnvironmentError(
            f"The supplied folder path could not be found: {config_dict['BASE_PATH']}"
        )

    return config_dict["BASE_PATH"], config_dict


def load_or_discover_images(
    base_path: Path, config_dict: dict
) -> Tuple[Optional[PlateCollection], ImageFileCollection]:
    """
    Load cached data or discover images from the file system.

    :param base_path: working directory path
    :param config_dict: configuration dictionary
    :returns: tuple of (plates, image_files) - plates may be None if not cached
    """
    plates = None

    if config_dict["USE_CACHED"]:
        if not config_dict["SILENT"]:
            print("Attempting to load cached data")

        plates = load_file(
            base_path.joinpath(config.DATA_DIR, config.CACHED_DATA_FILE_NAME),
            CompressionMethod.LZMA,
            pickle=True,
        )

        # Validate cached data
        if (
            config_dict["VERBOSE"]
            and plates is not None
            and plates.count
            == PlateCollection.coordinate_to_index(config_dict["PLATE_LATTICE"])
            and isinstance(plates.items[0], Plate)
        ):
            print("Successfully loaded cached data")
            return plates, None
        else:
            print("Unable to load cached data, starting image processing")
            plates = None

    # Find images in working directory
    image_files = ImageFileCollection.from_path(
        base_path, config_dict["IMAGE_FORMATS"], cache_images=False
    )

    if not config_dict["SILENT"]:
        print(f"{image_files.count} images found")

    return plates, image_files


def align_images(
    image_files: ImageFileCollection, config_dict: dict
) -> ImageFileCollection:
    """
    Align images if alignment strategy is specified.

    :param image_files: collection of images to align
    :param config_dict: configuration dictionary
    :returns: updated image file collection
    """
    if config_dict["IMAGE_ALIGN_STRATEGY"] == AlignStrategy.none:
        return image_files

    if not config_dict["SILENT"]:
        print(
            f"Verifying image alignment with '{config_dict['IMAGE_ALIGN_STRATEGY'].name}' strategy. This process will take some time"
        )

    # Determine which images need alignment
    align_model, image_files_align = calculate_transformation_strategy(
        image_files.items,
        config_dict["IMAGE_ALIGN_STRATEGY"],
        tolerance=config_dict["IMAGE_ALIGN_TOLERANCE"],
    )

    # Apply alignment if needed
    if len(image_files_align) > 0:
        if not config_dict["SILENT"]:
            print(
                f"{len(image_files_align)} of {image_files.count} images require alignment"
            )

        with Pool(processes=config_dict["POOL_MAX"]) as pool:
            results = []
            job = pool.imap_unordered(
                func=partial(apply_align_transform, align_model=align_model),
                iterable=image_files_align,
                chunksize=2,
            )

            for i, result in enumerate(job, start=1):
                results.append(result)
                if not config_dict["SILENT"]:
                    progress_bar(
                        (i / len(image_files_align)) * 100,
                        message="Correcting image alignment",
                    )

            image_files.update(results)

    return image_files


def detect_plates(
    image_files: ImageFileCollection, config_dict: dict
) -> Tuple[PlateCollection, dict]:
    """
    Detect plates in the first image and create noise masks.

    :param image_files: collection of aligned images
    :param config_dict: configuration dictionary
    :returns: tuple of (plates, plate_noise_masks)
    """
    if not config_dict["SILENT"]:
        print("Preprocessing images to locate plates")

    # Load the first image to get plate coordinates and mask
    with image_files.items[0] as image_file:
        if config_dict["VERBOSE"]:
            print(f"Locating plate centres in image: {image_file.file_path}")

        # Create new Plate instances
        plates = PlateCollection.from_image(
            shape=config_dict["PLATE_LATTICE"],
            image=image_file.image_gray,
            diameter=config_dict["PLATE_SIZE"],
            search_radius=config_dict["PLATE_SIZE"] // 20,
            edge_cut=config_dict["PLATE_EDGE_CUT"],
            labels=config_dict["PLATE_LABELS"],
        )

        if not plates.count > 0:
            if not config_dict["SILENT"]:
                print(f"Unable to locate plates in image: {image_file.file_path}")
                print("Processing unable to continue")
            sys.exit()

        if config_dict["VERBOSE"]:
            for plate in plates.items:
                print(f"Plate {plate.id} center: {plate.center}")

        # Use the first plate image as a noise mask
        plate_noise_masks = plates.slice_plate_images(image_file.image_gray)

    return plates, plate_noise_masks


def process_images_to_timepoints(
    image_files: ImageFileCollection,
    plates: PlateCollection,
    plate_noise_masks: dict,
    config_dict: dict,
) -> dict:
    """
    Process all images to extract colony timepoints.

    :param image_files: collection of images to process
    :param plates: detected plates
    :param plate_noise_masks: noise masks for each plate
    :param config_dict: configuration dictionary
    :returns: dictionary of timepoints grouped by plate
    """
    if not config_dict["SILENT"]:
        print("Processing colony data from all images")

    # Process images to Timepoints
    with Pool(processes=config_dict["POOL_MAX"]) as pool:
        results = []
        job = pool.imap(
            func=partial(
                image_file_to_timepoints,
                plates=plates,
                plate_noise_masks=plate_noise_masks,
            ),
            iterable=image_files.items,
            chunksize=2,
        )

        for i, result in enumerate(job, start=1):
            results.append(result)
            if not config_dict["SILENT"]:
                progress_bar((i / image_files.count) * 100, message="Processing images")

        plate_timepoints = dicts_merge(list(results))

    return plate_timepoints


def track_colonies(
    plates: PlateCollection,
    plate_timepoints: dict,
    image_files: ImageFileCollection,
    config_dict: dict,
) -> PlateCollection:
    """
    Track colonies across timepoints by grouping timepoints into colony trajectories.

    :param plates: detected plates
    :param plate_timepoints: timepoints grouped by plate
    :param image_files: image collection for timestamp calculation
    :param config_dict: configuration dictionary
    :returns: plates with tracked colonies
    """
    if not config_dict["SILENT"]:
        print("Calculating colony properties")

    # Calculate deviation in timestamps
    from numpy import diff

    timestamp_diff_std = diff(
        [img.timestamp_elapsed.total_seconds() for img in image_files.items[1:]]
    ).std()
    timestamp_diff_std += config.COLONY_TIMESTAMP_DIFF_MAX

    # Group and consolidate Timepoints into Colony instances
    plates = plates_colonies_from_timepoints(
        plates,
        plate_timepoints,
        config.COLONY_DISTANCE_MAX,
        timestamp_diff_std,
        config_dict["POOL_MAX"],
    )

    # Validate results
    if not any([plate.count for plate in plates.items]):
        if not config_dict["SILENT"]:
            print("Unable to locate any colonies in the images provided")
            print(
                f"ColonyScanalyser analysis completed for: {config_dict['BASE_PATH']}"
            )
        sys.exit()
    elif not config_dict["SILENT"]:
        for plate in plates.items:
            print(f"{plate.count} colonies identified on plate {plate.id}")

    return plates


def generate_visualizations(
    plates: PlateCollection,
    image_files: ImageFileCollection,
    base_path: Path,
    config_dict: dict,
) -> None:
    """
    Generate colony visualizations if requested.

    :param plates: plates with tracked colonies
    :param image_files: collection of processed images
    :param base_path: working directory
    :param config_dict: configuration dictionary
    """
    if not config_dict["VISUALIZE"]:
        return

    if not config_dict["SILENT"]:
        print("Generating colony visualizations")

    visualization_path = base_path / config_dict["VISUALIZATION_DIR"]
    visualization_path.mkdir(exist_ok=True)

    # Generate visualizations for selected timepoints (first, middle, last)
    viz_indices = [0, len(image_files.items) // 2, -1]
    viz_images = [image_files.items[i] for i in viz_indices]

    for i, image_file in enumerate(viz_images):
        if config_dict["VERBOSE"]:
            print(
                f"Creating visualizations for timepoint {i + 1}/3: {image_file.file_path.name}"
            )

        # Determine visualization types
        viz_types = config_dict["VISUALIZATION_TYPES"]
        save_masks = "masks" in viz_types
        save_ids = "ids" in viz_types
        save_outlines = "outlines" in viz_types
        save_comprehensive = "comprehensive" in viz_types

        saved_files = save_colony_visualizations(
            plates=plates,
            image_file=image_file,
            output_dir=visualization_path,
            save_masks=save_masks,
            save_ids=save_ids,
            save_outlines=save_outlines,
            save_comprehensive=save_comprehensive,
            use_full_image=True,
            save_plate_only=False,
        )

        if config_dict["VERBOSE"] and saved_files:
            print(f"  Saved {len(saved_files)} visualization files")

    if not config_dict["SILENT"]:
        print(f"Colony visualizations saved to: {visualization_path}")


def main():
    """
    Main entry point for the ColonyScanalyser CLI.
    """
    try:
        # Parse arguments and set up environment
        args = parse_arguments()
        base_path, config_dict = setup_processing_environment(args)

        if not config_dict["SILENT"]:
            print("Starting ColonyScanalyser analysis")

        if config_dict["VERBOSE"] and config_dict["POOL_MAX"] > 1:
            print(
                f"Multiprocessing enabled, utilising {config_dict['POOL_MAX']} of {cpu_count()} processors"
            )

        if not config_dict["SILENT"]:
            print(f"Working directory: {base_path}")

        # Load cached data or discover images
        plates, image_files = load_or_discover_images(base_path, config_dict)

        # If we have cached data, skip processing
        if plates is not None and image_files is None:
            # TODO: Add data persistence logic here
            if not config_dict["SILENT"]:
                print(f"ColonyScanalyser analysis completed for: {base_path}")
            return

        # Process images through the pipeline
        image_files = align_images(image_files, config_dict)
        plates, plate_noise_masks = detect_plates(image_files, config_dict)
        plate_timepoints = process_images_to_timepoints(
            image_files, plates, plate_noise_masks, config_dict
        )
        plates = track_colonies(plates, plate_timepoints, image_files, config_dict)

        # Generate visualizations if requested
        generate_visualizations(plates, image_files, base_path, config_dict)

        # TODO: Add data persistence logic here

        if not config_dict["SILENT"]:
            print(f"ColonyScanalyser analysis completed for: {base_path}")

    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
