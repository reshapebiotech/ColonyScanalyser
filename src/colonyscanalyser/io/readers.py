"""Image and data reading services."""

from pathlib import Path
from typing import List, Optional

import numpy as np
from numpy import ndarray
from skimage.io import imread


def load_image(
    file_path: Path,
    as_gray: bool = False,
    as_rgb: bool = True,
    **kwargs,
) -> ndarray:
    """
    Load an image from file.

    Args:
        file_path: Path to image file
        as_gray: Load as grayscale
        as_rgb: Convert to RGB (ignored if as_gray=True)
        **kwargs: Additional arguments for imread

    Returns:
        Image as numpy array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file can't be loaded as image
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Image file not found: {file_path}")

    # Handle legacy parameter name
    if "ensure_rgb" in kwargs:
        as_rgb = kwargs.pop("ensure_rgb")

    try:
        image = imread(str(file_path), as_gray=as_gray, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to load image {file_path}: {e}")

    # Convert to RGB if requested and not grayscale
    if not as_gray and as_rgb:
        image = ensure_rgb(image)

    return image


def ensure_rgb(image: ndarray) -> ndarray:
    """
    Ensure image is in RGB format.

    Args:
        image: Input image array

    Returns:
        RGB image array
    """
    from skimage.color import gray2rgb, rgba2rgb

    # If image has no color channels, it's grayscale
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        return gray2rgb(image.squeeze())

    # If image has alpha channel, convert to RGB
    if len(image.shape) == 3 and image.shape[2] == 4:
        return rgba2rgb(image)

    # Already RGB or other format
    return image


def find_image_files(
    directory: Path,
    extensions: Optional[List[str]] = None,
    recursive: bool = False,
) -> List[Path]:
    """
    Find image files in a directory.

    Args:
        directory: Directory to search
        extensions: File extensions to include (default: common image formats)
        recursive: Search subdirectories recursively

    Returns:
        List of image file paths, sorted by name
    """
    if extensions is None:
        extensions = ["tif", "tiff", "png", "jpg", "jpeg", "bmp"]

    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files = []
    for ext in extensions:
        pattern = f"**/*.{ext}" if recursive else f"*.{ext}"
        files.extend(directory.glob(pattern))

    # Filter out hidden files and sort
    files = [f for f in files if not f.name.startswith(".")]
    return sorted(files)


def load_data(
    file_path: Path,
    format: str = "auto",
    **kwargs,
) -> np.ndarray:
    """
    Load data from various formats.

    Args:
        file_path: Path to data file
        format: Data format ('numpy', 'pickle', 'csv', 'auto')
        **kwargs: Additional loading arguments

    Returns:
        Loaded data as numpy array

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If format not supported or loading fails
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Auto-detect format from extension
    if format == "auto":
        suffix = file_path.suffix.lower()
        if suffix in [".npy", ".npz"]:
            format = "numpy"
        elif suffix in [".pkl", ".pickle"]:
            format = "pickle"
        elif suffix in [".csv", ".txt"]:
            format = "csv"
        else:
            raise ValueError(f"Cannot auto-detect format for {file_path}")

    try:
        if format == "numpy":
            return np.load(file_path, **kwargs)
        elif format == "pickle":
            import pickle

            with open(file_path, "rb") as f:
                return pickle.load(f)
        elif format == "csv":
            return np.loadtxt(file_path, delimiter=",", **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")

    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}")


def file_exists_with_data(file_path: Path) -> bool:
    """
    Check if file exists and contains data.

    Args:
        file_path: Path to check

    Returns:
        True if file exists and has size > 0
    """
    return file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0


def get_file_info(file_path: Path) -> dict:
    """
    Get basic information about a file.

    Args:
        file_path: Path to file

    Returns:
        Dictionary with file information

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = file_path.stat()
    return {
        "name": file_path.name,
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "suffix": file_path.suffix,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
    }


def find_files_by_pattern(
    directory: Path,
    pattern: str,
    recursive: bool = False,
) -> List[Path]:
    """
    Find files matching a glob pattern.

    Args:
        directory: Directory to search
        pattern: Glob pattern (e.g., "*.tif", "frame_*.png")
        recursive: Search subdirectories recursively

    Returns:
        List of matching file paths, sorted by name
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    if recursive:
        pattern = f"**/{pattern}"

    files = list(directory.glob(pattern))
    # Filter out hidden files and sort
    files = [f for f in files if not f.name.startswith(".")]
    return sorted(files)
