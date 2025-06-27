"""Data and image writing services."""

import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy import ndarray
from skimage.io import imsave


def save_image(
    image: ndarray,
    file_path: Path,
    **kwargs,
) -> Path:
    """
    Save an image to file.

    Args:
        image: Image array to save
        file_path: Output file path
        **kwargs: Additional arguments for imsave

    Returns:
        Path to saved file

    Raises:
        ValueError: If image cannot be saved
    """
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        imsave(str(file_path), image, **kwargs)
    except Exception as e:
        raise ValueError(f"Failed to save image to {file_path}: {e}")

    return file_path


def save_data_csv(
    data: List[Any],
    file_path: Path,
    headers: Optional[List[str]] = None,
    delimiter: str = ",",
) -> Path:
    """
    Save data to CSV file.

    Args:
        data: Data to save (list of rows)
        file_path: Output file path
        headers: Optional column headers
        delimiter: CSV delimiter

    Returns:
        Path to saved file

    Raises:
        ValueError: If data cannot be saved
    """
    # Ensure .csv extension
    file_path = file_path.with_suffix(".csv")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter=delimiter)

            # Write headers if provided
            if headers:
                writer.writerow(headers)

            # Write data rows
            for row in data:
                if hasattr(row, "__iter__") and not isinstance(row, str):
                    writer.writerow(row)
                else:
                    writer.writerow([row])

    except Exception as e:
        raise ValueError(f"Failed to save CSV to {file_path}: {e}")

    return file_path


def save_data_dict_csv(
    data: List[Dict[str, Any]],
    file_path: Path,
    fieldnames: Optional[List[str]] = None,
    delimiter: str = ",",
) -> Path:
    """
    Save dictionary data to CSV file.

    Args:
        data: List of dictionaries to save
        file_path: Output file path
        fieldnames: Field names for CSV columns (auto-detected if None)
        delimiter: CSV delimiter

    Returns:
        Path to saved file

    Raises:
        ValueError: If data cannot be saved
    """
    if not data:
        raise ValueError("No data to save")

    # Auto-detect fieldnames from first dictionary
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    # Ensure .csv extension
    file_path = file_path.with_suffix(".csv")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, delimiter=delimiter, extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(data)

    except Exception as e:
        raise ValueError(f"Failed to save CSV to {file_path}: {e}")

    return file_path


def save_data_numpy(
    data: ndarray,
    file_path: Path,
    compressed: bool = False,
) -> Path:
    """
    Save numpy array to file.

    Args:
        data: Numpy array to save
        file_path: Output file path
        compressed: Use compressed format (.npz)

    Returns:
        Path to saved file

    Raises:
        ValueError: If data cannot be saved
    """
    # Set appropriate extension
    if compressed:
        file_path = file_path.with_suffix(".npz")
    else:
        file_path = file_path.with_suffix(".npy")

    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if compressed:
            np.savez_compressed(file_path, data=data)
        else:
            np.save(file_path, data)

    except Exception as e:
        raise ValueError(f"Failed to save numpy data to {file_path}: {e}")

    return file_path


def save_data_pickle(
    data: Any,
    file_path: Path,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> Path:
    """
    Save data using pickle.

    Args:
        data: Data to save
        file_path: Output file path
        protocol: Pickle protocol version

    Returns:
        Path to saved file

    Raises:
        ValueError: If data cannot be saved
    """
    # Ensure .pkl extension
    file_path = file_path.with_suffix(".pkl")
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "wb") as f:
            pickle.dump(data, f, protocol=protocol)

    except Exception as e:
        raise ValueError(f"Failed to save pickle to {file_path}: {e}")

    return file_path


def create_safe_filename(
    parts: List[str],
    separator: str = "_",
    max_length: int = 255,
) -> str:
    """
    Create a safe filename from parts.

    Args:
        parts: List of strings to join
        separator: Separator between parts
        max_length: Maximum filename length

    Returns:
        Safe filename string
    """
    # Remove empty parts and clean each part
    clean_parts = []
    for part in parts:
        if part:
            # Replace spaces and problematic characters
            clean = str(part).replace(" ", separator)
            # Remove or replace other problematic characters
            clean = "".join(c for c in clean if c.isalnum() or c in "._-")
            if clean:
                clean_parts.append(clean)

    filename = separator.join(clean_parts)

    # Truncate if too long
    if len(filename) > max_length:
        filename = filename[:max_length]

    return filename


def ensure_directory(directory: Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        directory: Directory path to create

    Returns:
        Path to directory

    Raises:
        OSError: If directory cannot be created
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise OSError(f"Failed to create directory {directory}: {e}")

    return directory


def export_colony_data(
    colonies: List[Any],
    output_dir: Path,
    prefix: str = "colonies",
    formats: Optional[List[str]] = None,
) -> List[Path]:
    """
    Export colony data in multiple formats.

    Args:
        colonies: List of colony objects
        output_dir: Output directory
        prefix: Filename prefix
        formats: Export formats ('csv', 'pickle', 'numpy')

    Returns:
        List of created file paths
    """
    if formats is None:
        formats = ["csv"]

    ensure_directory(output_dir)
    created_files = []

    # Convert colonies to exportable format
    if not colonies:
        return created_files

    # Try to extract data from colonies
    try:
        # Assume colonies have an __iter__ method for CSV export
        colony_data = [
            list(colony) for colony in colonies if hasattr(colony, "__iter__")
        ]
    except Exception:
        # Fallback: just save as pickle
        if "pickle" not in formats:
            formats.append("pickle")
        colony_data = []

    for format_type in formats:
        if format_type == "csv" and colony_data:
            headers = [
                "id",
                "name",
                "time_of_appearance",
                "center",
                "timepoint_count",
                "area_first",
                "area_last",
            ]
            file_path = output_dir / f"{prefix}.csv"
            created_files.append(save_data_csv(colony_data, file_path, headers))

        elif format_type == "pickle":
            file_path = output_dir / f"{prefix}.pkl"
            created_files.append(save_data_pickle(colonies, file_path))

        elif format_type == "numpy" and colony_data:
            data_array = np.array(colony_data, dtype=object)
            file_path = output_dir / f"{prefix}.npy"
            created_files.append(save_data_numpy(data_array, file_path))

    return created_files


def create_timestamped_filename(
    base_name: str,
    extension: str = "",
    timestamp_format: str = "%Y%m%d_%H%M%S",
) -> str:
    """
    Create filename with timestamp.

    Args:
        base_name: Base filename
        extension: File extension (with or without dot)
        timestamp_format: Timestamp format string

    Returns:
        Timestamped filename
    """
    from datetime import datetime

    timestamp = datetime.now().strftime(timestamp_format)

    # Ensure extension starts with dot
    if extension and not extension.startswith("."):
        extension = f".{extension}"

    return f"{base_name}_{timestamp}{extension}"
