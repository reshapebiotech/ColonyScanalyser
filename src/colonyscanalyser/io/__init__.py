"""
Clean I/O operations for ColonyScanalyser.

This package provides focused services for:
- Reading: Load images and data from files
- Writing: Save data and images in various formats
- Caching: Simple file-based caching for processed data
"""

# Reading operations
# Caching operations
from .cache import SimpleCache, cached_result, clear_cache, get_cache_size
from .readers import (
    file_exists_with_data,
    find_files_by_pattern,
    find_image_files,
    load_data,
    load_image,
)

# Writing operations
from .writers import (
    create_safe_filename,
    ensure_directory,
    export_colony_data,
    save_data_csv,
    save_data_dict_csv,
    save_data_numpy,
    save_data_pickle,
    save_image,
)

__all__ = [
    # Reading
    "load_image",
    "load_data",
    "find_image_files",
    "find_files_by_pattern",
    "file_exists_with_data",
    # Writing
    "save_image",
    "save_data_csv",
    "save_data_dict_csv",
    "save_data_numpy",
    "save_data_pickle",
    "export_colony_data",
    "create_safe_filename",
    "ensure_directory",
    # Caching
    "SimpleCache",
    "cached_result",
    "clear_cache",
    "get_cache_size",
]
