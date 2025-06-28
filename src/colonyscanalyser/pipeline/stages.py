"""
Pipeline stages for ColonyScanalyser.

This module provides individual pipeline stages that can be composed
to create different analysis workflows. Each stage has a single
responsibility and clean interfaces.
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from numpy import diff

from ..io import load_data
from ..models.config import PipelineConfig
from ..models.image import ImageCollection, ImageFile
from ..models.plate import Plate, PlateCollection
from ..services.detection import timepoints_from_image
from ..services.tracking import create_colonies_from_timepoints, filter_colonies


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def __call__(self, current: int, total: int, message: str = "") -> None:
        """Report progress."""
        ...


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    def __init__(
        self,
        config: PipelineConfig,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.config = config
        self.progress_callback = progress_callback
        self.log = logging.getLogger("")

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute this pipeline stage.

        Args:
            context: Pipeline context containing data from previous stages

        Returns:
            Updated context with this stage's results
        """
        pass

    def _report_progress(self, current: int, total: int, message: str = "") -> None:
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(current, total, message)


class CacheLoadStage(PipelineStage):
    """Stage for loading cached analysis data."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Try to load cached data."""
        if not self.config.enable_caching:
            context["cache_loaded"] = False
            return context

        cache_file = self.config.input_dir / "data" / "cached_analysis.pkl"

        try:
            if cache_file.exists():
                plates = load_data(cache_file)
                if self._validate_cached_plates(plates):
                    context["plates"] = plates
                    context["cache_loaded"] = True
                    return context
        except Exception:
            pass

        context["cache_loaded"] = False
        return context

    def _validate_cached_plates(self, plates: PlateCollection) -> bool:
        """Validate that cached plates data is usable."""
        if not isinstance(plates, PlateCollection):
            return False

        # For now, just check that we have some plates
        # In a full implementation, this would use actual plate configuration
        expected_count = 1

        return len(plates) == expected_count and len(plates) > 0


class ImageDiscoveryStage(PipelineStage):
    """Stage for discovering and loading image files."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Discover image files in the working directory."""
        from ..io import find_image_files

        # Find image files using extensions from config
        extensions = getattr(
            self.config, "image_extensions", ["jpg", "png", "tif", "tiff"]
        )
        image_paths = find_image_files(
            self.config.input_dir,
            extensions=extensions,
            recursive=False,
        )

        if not image_paths:
            raise RuntimeError(f"No images found in {self.config.input_dir}")

        # Create ImageFile objects
        images = []
        for i, path in enumerate(image_paths):
            # Extract timestamp from filename or use index
            timestamp = self._extract_timestamp(path, i)

            image_file = ImageFile(
                file_path=path,
                timestamp=timestamp,
            )
            images.append(image_file)

        # Create collection and sort by timestamp
        image_collection = ImageCollection(images)
        image_collection.sort_by_timestamp()

        context["image_files"] = image_collection
        return context

    def _extract_timestamp(self, path: Path, index: int) -> int:
        """Extract timestamp from filename or use index."""
        # Simple implementation - use index as timestamp
        # In real implementation, this would parse actual timestamps
        return index


class ImageAlignmentStage(PipelineStage):
    """Stage for aligning images."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Align images if alignment is enabled."""
        image_files = context["image_files"]

        if not self.config.enable_alignment or len(image_files) < 2:
            return context

        from ..io import load_image
        from ..services.alignment import align_image_simple, create_alignment_strategy

        # Load reference image (first image)
        reference_image = load_image(image_files[0].file_path, as_rgb=True)

        # Create alignment strategy based on image size
        strategy = create_alignment_strategy("fast", reference_image.shape)

        # Align all subsequent images to the reference
        aligned_results = []
        for i, image_file in enumerate(image_files):
            if i == 0:
                # Reference image doesn't need alignment
                aligned_results.append({"aligned": True, "quality": 1.0})
                continue

            # Load target image
            target_image = load_image(image_file.file_path, as_rgb=True)

            # Perform alignment
            try:
                aligned_image = align_image_simple(
                    target_image, reference_image, **strategy
                )
                if aligned_image is not None:
                    aligned_results.append({"aligned": True, "quality": 0.8})
                else:
                    aligned_results.append({"aligned": False, "quality": 0.0})
            except Exception as e:
                # Log error but continue processing
                aligned_results.append(
                    {"aligned": False, "quality": 0.0, "error": str(e)}
                )

        # Store alignment results in context
        context["alignment_results"] = aligned_results
        context["images_aligned"] = True
        return context


class PlateDetectionStage(PipelineStage):
    """Stage for detecting plates in images."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Detect plates in the first image."""
        image_files = context["image_files"]

        # Use first image for plate detection
        first_image_file = image_files[0]

        from ..io import load_image

        image = load_image(first_image_file.file_path, ensure_rgb=False)

        # Create plates based on lattice configuration
        plates = self._create_plates_from_lattice(image)

        self.log.info("Plates detected")
        # Create noise masks for each plate
        noise_masks = self._create_noise_masks(image, plates)

        context["plates"] = plates
        context["noise_masks"] = noise_masks
        return context

    def _create_plates_from_lattice(self, image: np.ndarray) -> PlateCollection:
        """Create plates based on configured lattice."""
        height, width = image.shape[:2]

        # Simple single plate detection for now
        # In a full implementation, this would detect multiple plates
        plate_diameter = min(height, width) * 0.8  # Use 80% of image size
        center_x = width / 2
        center_y = height / 2
        edge_cut = plate_diameter * 0.05  # 5% edge cut

        plate = Plate(
            id=1,
            diameter=plate_diameter,
            name="Plate_1",
            center=(center_x, center_y),
            edge_cut=edge_cut,
        )

        plates = PlateCollection()
        plates.add(plate)

        return plates

    def _create_noise_masks(
        self, image: np.ndarray, plates: PlateCollection
    ) -> Dict[int, np.ndarray]:
        """Create noise masks for each plate."""
        noise_masks = {}

        for plate in plates:
            # Create a simple noise mask (in real implementation, this would be more sophisticated)
            mask = np.ones_like(image, dtype=bool)
            noise_masks[plate.id] = mask

        return noise_masks


class TimePointExtractionStage(PipelineStage):
    """Stage for extracting colony timepoints from images."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract timepoints from all images."""
        image_files = context["image_files"]
        plates = context["plates"]
        noise_masks = context["noise_masks"]

        all_timepoints = defaultdict(list)

        # Process images in parallel if configured
        if self.config.parallel_processing:
            timepoints = self._process_images_parallel(image_files, plates, noise_masks)
        else:
            timepoints = self._process_images_sequential(
                image_files, plates, noise_masks
            )

        # Merge results
        for plate_timepoints in timepoints:
            for plate_id, tp_list in plate_timepoints.items():
                all_timepoints[plate_id].extend(tp_list)

        context["timepoints"] = dict(all_timepoints)
        return context

    def _process_images_parallel(
        self,
        image_files: ImageCollection,
        plates: PlateCollection,
        noise_masks: Dict[int, np.ndarray],
    ) -> List[Dict[int, List]]:
        """Process images in parallel."""
        with Pool(processes=2) as pool:  # Use 2 processes for parallel mode
            func = partial(
                self._extract_timepoints_from_image,
                plates=plates,
                noise_masks=noise_masks,
            )

            results = []
            for i, result in enumerate(
                pool.imap(func, image_files.images, chunksize=2)
            ):
                results.append(result)
                self._report_progress(i + 1, len(image_files), "Extracting timepoints")

        return results

    def _process_images_sequential(
        self,
        image_files: ImageCollection,
        plates: PlateCollection,
        noise_masks: Dict[int, np.ndarray],
    ) -> List[Dict[int, List]]:
        """Process images sequentially."""
        results = []

        for i, image_file in enumerate(image_files.images):
            result = self._extract_timepoints_from_image(
                image_file, plates, noise_masks
            )
            results.append(result)
            self._report_progress(i + 1, len(image_files), "Extracting timepoints")

        return results

    def _extract_timepoints_from_image(
        self,
        image_file: ImageFile,
        plates: PlateCollection,
        noise_masks: Dict[int, np.ndarray],
    ) -> Dict[int, List]:
        """Extract timepoints from a single image."""
        from ..io import load_image
        from ..services.segmentation import segment_plate_image

        # Load image
        image = load_image(image_file.file_path, ensure_rgb=True)

        plate_timepoints = defaultdict(list)

        # Process each plate
        for plate in plates:
            # Extract plate region
            plate_image = self._extract_plate_region(image, plate)

            # Segment the plate image
            segmented = segment_plate_image(
                plate_image, noise_mask=noise_masks.get(plate.id), min_area=2.0
            )

            # Extract timepoints
            timepoints = timepoints_from_image(
                segmented, timestamp=image_file.timestamp
            )

            plate_timepoints[plate.id].extend(timepoints)

        return dict(plate_timepoints)

    def _extract_plate_region(self, image: np.ndarray, plate: Plate) -> np.ndarray:
        """Extract the region of image corresponding to a plate."""
        height, width = image.shape[:2]
        center_x, center_y = plate.center
        radius = plate.radius

        # Calculate bounding box
        x1 = max(0, int(center_x - radius))
        y1 = max(0, int(center_y - radius))
        x2 = min(width, int(center_x + radius))
        y2 = min(height, int(center_y + radius))

        return image[y1:y2, x1:x2]


class ColonyTrackingStage(PipelineStage):
    """Stage for tracking colonies across timepoints."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Track colonies by grouping timepoints."""
        plates = context["plates"]
        timepoints = context["timepoints"]

        # Calculate timestamp standard deviation for filtering
        timestamp_std = self._calculate_timestamp_std(timepoints)

        # Process each plate
        for plate in plates:
            plate_timepoints = timepoints.get(plate.id, [])

            if not plate_timepoints:
                continue

            # Group timepoints into colonies
            colonies = create_colonies_from_timepoints(
                plate_timepoints,
                distance_tolerance=15.0,  # Default distance tolerance
            )

            # Filter colonies
            filtered_colonies = filter_colonies(
                colonies,
                min_timepoints=2,
                min_area=5,
            )

            # Add colonies to plate
            plate.colonies = filtered_colonies

        context["plates"] = plates
        return context

    def _calculate_timestamp_std(self, timepoints: Dict[int, List]) -> float:
        """Calculate standard deviation of timestamps for filtering."""
        all_timestamps = []

        for plate_timepoints in timepoints.values():
            for tp in plate_timepoints:
                all_timestamps.append(tp.timestamp.total_seconds())

        if len(all_timestamps) < 2:
            return 10.0  # Default value

        timestamp_diffs = diff(sorted(all_timestamps))
        return float(timestamp_diffs.std()) + 5.0  # Add buffer


class VisualizationStage(PipelineStage):
    """Stage for generating visualizations."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate colony visualizations."""
        if not self.config.enable_visualization:
            return context

        plates = context["plates"]
        image_files = context["image_files"]

        # Create output directory
        output_dir = self.config.output_dir / "visualizations"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Select images for visualization
        viz_images = self._select_visualization_images(image_files)

        saved_files = []
        for i, image_file in enumerate(viz_images):
            files = self._create_visualizations_for_image(
                image_file, plates, output_dir
            )
            saved_files.extend(files)

            self._report_progress(i + 1, len(viz_images), "Creating visualizations")

        context["visualization_files"] = saved_files
        return context

    def _select_visualization_images(
        self, image_files: ImageCollection
    ) -> List[ImageFile]:
        """Select which images to create visualizations for."""
        if len(image_files) <= 3:
            return list(image_files.images)

        # Select first, middle, and last images
        indices = [0, len(image_files) // 2, -1]
        return [image_files.images[i] for i in indices]

    def _create_visualizations_for_image(
        self, image_file: ImageFile, plates: PlateCollection, output_dir: Path
    ) -> List[Path]:
        """Create visualizations for a single image."""
        from ..io import load_image
        from ..visualization import (
            create_colony_visualization,
            create_visualization_filename,
            save_image,
        )

        # Load image
        image = load_image(image_file.file_path, ensure_rgb=True)

        saved_files = []

        # Create visualizations for each plate
        for plate in plates:
            if not plate.colonies:
                continue

            # Extract plate region
            plate_image = self._extract_plate_region(image, plate)

            # Create visualization
            viz_image = create_colony_visualization(
                plate_image,
                plate,
                timestamp=image_file.timestamp,
                show_masks=True,
                show_ids=True,
                show_plate=True,
            )

            # Save visualization
            filename = create_visualization_filename(
                "colony_viz",
                plate_id=plate.id,
                timestamp=int(image_file.timestamp.total_seconds()),
                visualization_type="comprehensive",
            )

            output_path = output_dir / f"{filename}.png"
            save_image(viz_image, output_path)
            saved_files.append(output_path)

        return saved_files

    def _extract_plate_region(self, image: np.ndarray, plate: Plate) -> np.ndarray:
        """Extract the region of image corresponding to a plate."""
        height, width = image.shape[:2]
        center_x, center_y = plate.center
        radius = plate.radius

        # Calculate bounding box
        x1 = max(0, int(center_x - radius))
        y1 = max(0, int(center_y - radius))
        x2 = min(width, int(center_x + radius))
        y2 = min(height, int(center_y + radius))

        return image[y1:y2, x1:x2]


class DataSaveStage(PipelineStage):
    """Stage for saving processed data."""

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Save processed data for future use."""
        plates = context["plates"]

        # Create data directory
        data_dir = self.config.output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save plates data
        from ..io import save_data_pickle

        cache_file = data_dir / "cached_analysis.pkl"
        save_data_pickle(plates, cache_file)

        # Export colony data to CSV
        from ..io import export_colony_data

        export_colony_data(
            [colony for plate in plates for colony in plate.colonies],
            data_dir,
            formats=["csv"],
            prefix="colony_analysis",
        )

        context["data_saved"] = True
        return context
