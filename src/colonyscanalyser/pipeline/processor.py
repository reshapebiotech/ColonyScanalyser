"""
Pipeline processor for ColonyScanalyser.

This module contains the main pipeline processor that orchestrates
the entire colony analysis workflow from image discovery through
colony tracking and visualization.
"""

import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from typing import Dict, Optional

from numpy import diff

from .. import config as default_config
from ..align.strategy import apply_align_transform, calculate_transformation_strategy
from ..core import ImageFileCollection, PlateCollection
from ..core.colony import colonies_filtered, colonies_from_timepoints
from ..core.config import PipelineConfig
from ..io import CompressionMethod, load_file
from ..processing.segmentation import segment_image
from ..utils.utilities import dicts_merge, progress_bar
from ..visualization import save_colony_visualizations


class ColonyPipelineProcessor:
    """
    Main pipeline processor for colony analysis.

    This class orchestrates the entire workflow from image discovery
    through colony tracking and visualization generation.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the pipeline processor.

        :param config: Complete pipeline configuration
        """
        self.config = config
        self.plates: Optional[PlateCollection] = None
        self.image_files: Optional[ImageFileCollection] = None
        self.plate_timepoints: Optional[Dict] = None

    def run(self) -> None:
        """Run the complete colony analysis pipeline."""
        try:
            self._print("Starting ColonyScanalyser analysis")
            self._print_verbose(
                f"Working directory: {self.config.processing.base_path}"
            )

            if self.config.processing.pool_max > 1:
                self._print_verbose(
                    f"Multiprocessing enabled, utilising {self.config.processing.pool_max} processors"
                )

            # Pipeline stages
            if self._load_cached_data():
                self._print("Successfully loaded cached data")
                self._generate_visualizations_if_requested()
            else:
                self._discover_images()
                self._align_images()
                self._detect_plates()
                self._process_images_to_timepoints()
                self._track_colonies()
                self._generate_visualizations_if_requested()
                # TODO: Save processed data

            self._print(
                f"ColonyScanalyser analysis completed for: {self.config.processing.base_path}"
            )

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"Error during processing: {e}")
            if self.config.processing.verbose:
                import traceback

                traceback.print_exc()
            sys.exit(1)

    def _load_cached_data(self) -> bool:
        """
        Attempt to load cached analysis data.

        :returns: True if cached data was loaded successfully, False otherwise
        """
        if not self.config.processing.use_cached_data:
            return False

        self._print("Attempting to load cached data")

        cached_file = self.config.processing.base_path.joinpath(
            default_config.DATA_DIR, default_config.CACHED_DATA_FILE_NAME
        )

        self.plates = load_file(cached_file, CompressionMethod.LZMA, pickle=True)

        # Validate cached data
        if (
            self.plates is not None
            and self.plates.count
            == PlateCollection.coordinate_to_index(self.config.processing.plate_lattice)
            and len(self.plates.items) > 0
        ):
            return True
        else:
            self._print("Unable to load cached data, starting image processing")
            self.plates = None
            return False

    def _discover_images(self) -> None:
        """Discover and load image files from the working directory."""
        self.image_files = ImageFileCollection.from_path(
            self.config.processing.base_path,
            self.config.processing.image_formats,
            cache_images=False,
        )

        if self.image_files.count == 0:
            raise RuntimeError("No images found in the specified directory")

        self._print(f"{self.image_files.count} images found")

    def _align_images(self) -> None:
        """Align images according to the configured strategy."""
        from ..align.strategy import AlignStrategy

        if self.config.processing.image_align_strategy == AlignStrategy.none:
            return

        self._print(
            f"Verifying image alignment with '{self.config.processing.image_align_strategy.name}' strategy. "
            "This process will take some time"
        )

        # Determine which images need alignment
        align_model, image_files_align = calculate_transformation_strategy(
            self.image_files.items,
            self.config.processing.image_align_strategy,
            tolerance=self.config.processing.image_align_tolerance,
        )

        if len(image_files_align) == 0:
            return

        self._print(
            f"{len(image_files_align)} of {self.image_files.count} images require alignment"
        )

        # Apply alignment
        with Pool(processes=self.config.processing.pool_max) as pool:
            results = []
            job = pool.imap_unordered(
                func=partial(apply_align_transform, align_model=align_model),
                iterable=image_files_align,
                chunksize=2,
            )

            for i, result in enumerate(job, start=1):
                results.append(result)
                if not self.config.processing.silent:
                    progress_bar(
                        (i / len(image_files_align)) * 100,
                        message="Correcting image alignment",
                    )

            self.image_files.update(results)

    def _detect_plates(self) -> None:
        """Detect plates in the first image and create noise masks."""
        self._print("Preprocessing images to locate plates")

        # Use first image for plate detection
        with self.image_files.items[0] as image_file:
            self._print_verbose(
                f"Locating plate centres in image: {image_file.file_path}"
            )

            # Create plate collection
            self.plates = PlateCollection.from_image(
                shape=self.config.processing.plate_lattice,
                image=image_file.image_gray,
                diameter=self.config.processing.plate_size_pixels,
                search_radius=self.config.processing.plate_size_pixels // 20,
                edge_cut=self.config.processing.plate_edge_cut_pixels,
                labels=self.config.processing.plate_labels,
            )

            if self.plates.count == 0:
                self._print(f"Unable to locate plates in image: {image_file.file_path}")
                self._print("Processing unable to continue")
                sys.exit(1)

            # Log detected plates
            for plate in self.plates.items:
                self._print_verbose(f"Plate {plate.id} center: {plate.center}")

            # Create noise masks
            self.plate_noise_masks = self.plates.slice_plate_images(
                image_file.image_gray
            )

    def _process_images_to_timepoints(self) -> None:
        """Process all images to extract colony timepoints."""
        self._print("Processing colony data from all images")

        with Pool(processes=self.config.processing.pool_max) as pool:
            results = []
            job = pool.imap(
                func=partial(
                    self._image_file_to_timepoints,
                    plates=self.plates,
                    plate_noise_masks=self.plate_noise_masks,
                ),
                iterable=self.image_files.items,
                chunksize=2,
            )

            for i, result in enumerate(job, start=1):
                results.append(result)
                if not self.config.processing.silent:
                    progress_bar(
                        (i / self.image_files.count) * 100, message="Processing images"
                    )

            self.plate_timepoints = dicts_merge(list(results))

    def _track_colonies(self) -> None:
        """Track colonies across timepoints by grouping timepoints into trajectories."""
        self._print("Calculating colony properties")

        # Calculate timestamp deviation
        timestamp_diff_std = diff(
            [
                img.timestamp_elapsed.total_seconds()
                for img in self.image_files.items[1:]
            ]
        ).std()
        timestamp_diff_std += default_config.COLONY_TIMESTAMP_DIFF_MAX

        # Group timepoints into colonies
        self.plates = self._plates_colonies_from_timepoints(
            self.plates,
            self.plate_timepoints,
            default_config.COLONY_DISTANCE_MAX,
            timestamp_diff_std,
            self.config.processing.pool_max,
        )

        # Validate results
        if not any(plate.count for plate in self.plates.items):
            self._print("Unable to locate any colonies in the images provided")
            self._print(
                f"ColonyScanalyser analysis completed for: {self.config.processing.base_path}"
            )
            sys.exit(0)

        # Report results
        for plate in self.plates.items:
            self._print(f"{plate.count} colonies identified on plate {plate.id}")

    def _generate_visualizations_if_requested(self) -> None:
        """Generate colony visualizations if enabled in configuration."""
        if not self.config.visualization.enabled or self.plates is None:
            return

        if self.image_files is None:
            # If we loaded from cache, we need to reload images for visualization
            self._discover_images()

        self._print("Generating colony visualizations")

        visualization_path = (
            self.config.processing.base_path / self.config.visualization.output_dir
        )
        visualization_path.mkdir(exist_ok=True)

        # Select timepoints for visualization (first, middle, last)
        viz_indices = [0, len(self.image_files.items) // 2, -1]
        viz_images = [self.image_files.items[i] for i in viz_indices]

        for i, image_file in enumerate(viz_images):
            self._print_verbose(
                f"Creating visualizations for timepoint {i + 1}/3: {image_file.file_path.name}"
            )

            # Determine visualization types
            viz_types = self.config.visualization.types
            saved_files = save_colony_visualizations(
                plates=self.plates,
                image_file=image_file,
                output_dir=visualization_path,
                save_masks="masks" in viz_types,
                save_ids="ids" in viz_types,
                save_outlines="outlines" in viz_types,
                save_comprehensive="comprehensive" in viz_types,
                use_full_image=self.config.visualization.use_full_image,
                save_plate_only=self.config.visualization.save_plate_only,
            )

            if saved_files:
                self._print_verbose(f"  Saved {len(saved_files)} visualization files")

        self._print(f"Colony visualizations saved to: {visualization_path}")

    def _image_file_to_timepoints(self, image_file, plates, plate_noise_masks) -> Dict:
        """Extract timepoints from a single image file."""
        from skimage.color import rgb2gray

        from ..core import timepoints_from_image

        plate_timepoints = defaultdict(list)

        with image_file as img:
            plate_images = plates.slice_plate_images(img.image)

            for plate_id, plate_image in plate_images.items():
                plate_image_gray = rgb2gray(plate_image)

                # Segment the image
                segmented_image = segment_image(
                    plate_image_gray,
                    plate_mask=plate_image_gray > 0,
                    plate_noise_mask=plate_noise_masks[plate_id],
                    area_min=1.5,
                )

                # Create timepoints
                plate_timepoints[plate_id].extend(
                    timepoints_from_image(
                        segmented_image, img.timestamp_elapsed, image=plate_image
                    )
                )

        return plate_timepoints

    def _plates_colonies_from_timepoints(
        self, plates, timepoints, distance_tolerance, timestamp_diff_std, pool_size
    ) -> PlateCollection:
        """Group timepoints into colony objects across all plates."""
        # Prepare data for parallel processing
        timepoints_iter = [
            (plates[plate_id], timepoints_list, distance_tolerance, timestamp_diff_std)
            for plate_id, timepoints_list in timepoints.items()
        ]

        # Process in parallel
        with Pool(processes=pool_size) as pool:
            plates.items = pool.starmap(
                func=self._plate_colonies_from_timepoints_filtered,
                iterable=timepoints_iter,
            )

        return plates

    def _plate_colonies_from_timepoints_filtered(
        self, plate, timepoints, distance_tolerance, timestamp_diff_std
    ):
        """Group timepoints into colonies for a single plate."""
        if len(timepoints) > 0:
            # Group timepoints by distance
            plate.items = colonies_from_timepoints(
                timepoints, distance_tolerance=distance_tolerance
            )

            # Filter colonies
            plate.items = colonies_filtered(plate.items, timestamp_diff_std)

        return plate

    def _print(self, message: str) -> None:
        """Print message unless in silent mode."""
        if not self.config.processing.silent:
            print(message)

    def _print_verbose(self, message: str) -> None:
        """Print message only in verbose mode."""
        if self.config.processing.verbose:
            print(message)
