"""
Export utilities for ColonyScanalyser visualization.

This module handles saving and exporting visualization images and plots,
separating file I/O concerns from the core visualization logic.
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from numpy import ndarray


def save_image(
    image: ndarray,
    file_path: Path,
    create_dirs: bool = True,
) -> Path:
    """
    Save an image array to disk.

    :param image: Image array to save
    :param file_path: Path where to save the image
    :param create_dirs: Whether to create parent directories
    :returns: Path to saved file
    """
    from skimage.io import imsave

    if create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure image is in proper format for saving
    if image.dtype == np.float64:
        # Convert float images to uint8
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    imsave(str(file_path), image)
    return file_path


def save_plot(
    save_path: Path,
    filename: str,
    dpi: int = 300,
    format: str = "png",
    bbox_inches: str = "tight",
    create_dirs: bool = True,
) -> Path:
    """
    Save the current matplotlib plot to disk.

    :param save_path: Directory to save the plot
    :param filename: Filename (without extension)
    :param dpi: Resolution for saved image
    :param format: Image format (png, jpg, svg, etc.)
    :param bbox_inches: Bounding box setting for matplotlib
    :param create_dirs: Whether to create parent directories
    :returns: Path to saved file
    """
    import matplotlib.pyplot as plt

    if create_dirs:
        save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / f"{filename}.{format}"
    plt.savefig(
        str(file_path),
        dpi=dpi,
        format=format,
        bbox_inches=bbox_inches,
    )
    return file_path


def create_visualization_filename(
    base_name: str,
    plate_id: Optional[int] = None,
    timestamp: Optional[Union[str, int]] = None,
    visualization_type: Optional[str] = None,
) -> str:
    """
    Create standardized filename for visualization exports.

    :param base_name: Base name for the file
    :param plate_id: Optional plate ID to include
    :param timestamp: Optional timestamp to include
    :param visualization_type: Type of visualization (masks, ids, outlines, etc.)
    :returns: Formatted filename
    """
    parts = [base_name]

    if plate_id is not None:
        parts.append(f"plate{plate_id}")

    if timestamp is not None:
        parts.append(f"t{timestamp}")

    if visualization_type is not None:
        parts.append(visualization_type)

    return "_".join(parts)


def save_visualization_series(
    images: List[ndarray],
    output_dir: Path,
    base_filename: str,
    file_format: str = "png",
) -> List[Path]:
    """
    Save a series of visualization images.

    :param images: List of image arrays to save
    :param output_dir: Directory to save images
    :param base_filename: Base filename for the series
    :param file_format: Image format
    :returns: List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i, image in enumerate(images):
        filename = f"{base_filename}_{i:04d}.{file_format}"
        file_path = output_dir / filename
        saved_path = save_image(image, file_path, create_dirs=False)
        saved_paths.append(saved_path)

    return saved_paths


def create_output_directory(
    base_path: Path,
    subdirectory: Optional[str] = None,
) -> Path:
    """
    Create output directory for visualizations.

    :param base_path: Base output path
    :param subdirectory: Optional subdirectory name
    :returns: Created directory path
    """
    if subdirectory:
        output_dir = base_path / subdirectory
    else:
        output_dir = base_path

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing/replacing invalid characters.

    :param filename: Original filename
    :returns: Sanitized filename safe for filesystem
    """
    # Replace common problematic characters
    replacements = {
        " ": "_",
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }

    for old, new in replacements.items():
        filename = filename.replace(old, new)

    return filename


def export_visualization_batch(
    visualization_data: List[dict],
    output_dir: Path,
    base_name: str = "visualization",
) -> List[Path]:
    """
    Export a batch of visualizations with consistent naming.

    :param visualization_data: List of dicts with 'image' and metadata
    :param output_dir: Output directory
    :param base_name: Base name for files
    :returns: List of saved file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for i, data in enumerate(visualization_data):
        image = data["image"]
        metadata = data.get("metadata", {})

        # Create filename from metadata
        filename = create_visualization_filename(
            base_name,
            plate_id=metadata.get("plate_id"),
            timestamp=metadata.get("timestamp"),
            visualization_type=metadata.get("type"),
        )

        # Add index if needed
        if len(visualization_data) > 1:
            filename = f"{filename}_{i:03d}"

        file_path = output_dir / f"{filename}.png"
        saved_path = save_image(image, file_path, create_dirs=False)
        saved_paths.append(saved_path)

    return saved_paths
