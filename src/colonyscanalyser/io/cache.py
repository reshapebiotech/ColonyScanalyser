"""Simple caching services for processed data and images."""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np


def create_cache_key(*args: Any) -> str:
    """
    Create a unique cache key from arguments.

    Args:
        *args: Arguments to create key from

    Returns:
        Hexadecimal cache key string
    """
    # Convert args to string representation
    key_parts = []
    for arg in args:
        if isinstance(arg, (str, int, float, bool)):
            key_parts.append(str(arg))
        elif isinstance(arg, Path):
            key_parts.append(str(arg.absolute()))
        elif isinstance(arg, np.ndarray):
            # Use array shape and hash of first/last elements for arrays
            key_parts.append(
                f"array_{arg.shape}_{hash(str(arg.flat[0]) + str(arg.flat[-1]))}"
            )
        else:
            key_parts.append(str(hash(str(arg))))

    # Create hash from combined parts
    combined = "_".join(key_parts)
    return hashlib.md5(combined.encode()).hexdigest()


def get_cache_path(cache_dir: Path, key: str, extension: str = ".pkl") -> Path:
    """
    Get full cache file path.

    Args:
        cache_dir: Cache directory
        key: Cache key
        extension: File extension

    Returns:
        Full path to cache file
    """
    if not extension.startswith("."):
        extension = f".{extension}"

    return cache_dir / f"{key}{extension}"


def cache_exists(cache_dir: Path, key: str, extension: str = ".pkl") -> bool:
    """
    Check if cached data exists.

    Args:
        cache_dir: Cache directory
        key: Cache key
        extension: File extension

    Returns:
        True if cache file exists and has data
    """
    cache_file = get_cache_path(cache_dir, key, extension)
    return cache_file.exists() and cache_file.stat().st_size > 0


def save_to_cache(
    data: Any,
    cache_dir: Path,
    key: str,
    extension: str = ".pkl",
) -> Path:
    """
    Save data to cache.

    Args:
        data: Data to cache
        cache_dir: Cache directory
        key: Cache key
        extension: File extension

    Returns:
        Path to cached file

    Raises:
        OSError: If cache cannot be written
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = get_cache_path(cache_dir, key, extension)

    try:
        if extension in [".pkl", ".pickle"]:
            with open(cache_file, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
        elif extension in [".npy"]:
            np.save(cache_file, data)
        elif extension in [".npz"]:
            np.savez_compressed(cache_file, data=data)
        else:
            # Default to pickle for unknown extensions
            with open(cache_file, "wb") as f:
                pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        raise OSError(f"Failed to save cache {cache_file}: {e}")

    return cache_file


def load_from_cache(
    cache_dir: Path,
    key: str,
    extension: str = ".pkl",
) -> Optional[Any]:
    """
    Load data from cache.

    Args:
        cache_dir: Cache directory
        key: Cache key
        extension: File extension

    Returns:
        Cached data or None if not found/invalid
    """
    cache_file = get_cache_path(cache_dir, key, extension)

    if not cache_file.exists():
        return None

    try:
        if extension in [".pkl", ".pickle"]:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        elif extension in [".npy"]:
            return np.load(cache_file, allow_pickle=True)
        elif extension in [".npz"]:
            data = np.load(cache_file, allow_pickle=True)
            return data["data"]
        else:
            # Default to pickle
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    except Exception:
        # If cache is corrupted, remove it
        try:
            cache_file.unlink()
        except Exception:
            pass
        return None


def clear_cache(cache_dir: Path, pattern: str = "*") -> int:
    """
    Clear cache files matching pattern.

    Args:
        cache_dir: Cache directory
        pattern: Glob pattern for files to remove

    Returns:
        Number of files removed
    """
    if not cache_dir.exists():
        return 0

    removed_count = 0
    for cache_file in cache_dir.glob(pattern):
        if cache_file.is_file():
            try:
                cache_file.unlink()
                removed_count += 1
            except Exception:
                pass

    return removed_count


def get_cache_size(cache_dir: Path) -> int:
    """
    Get total cache size in bytes.

    Args:
        cache_dir: Cache directory

    Returns:
        Total size in bytes
    """
    if not cache_dir.exists():
        return 0

    total_size = 0
    for cache_file in cache_dir.rglob("*"):
        if cache_file.is_file():
            try:
                total_size += cache_file.stat().st_size
            except Exception:
                pass

    return total_size


def cleanup_old_cache(cache_dir: Path, max_age_days: int = 30) -> int:
    """
    Remove cache files older than specified days.

    Args:
        cache_dir: Cache directory
        max_age_days: Maximum age in days

    Returns:
        Number of files removed
    """
    if not cache_dir.exists():
        return 0

    import time

    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    removed_count = 0

    for cache_file in cache_dir.rglob("*"):
        if cache_file.is_file():
            try:
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
            except Exception:
                pass

    return removed_count


class SimpleCache:
    """Simple file-based cache for processed data."""

    def __init__(self, cache_dir: Path, enabled: bool = True):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache files
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str, default: Any = None) -> Any:
        """Get item from cache."""
        if not self.enabled:
            return default

        data = load_from_cache(self.cache_dir, key)
        return data if data is not None else default

    def set(self, key: str, value: Any) -> None:
        """Set item in cache."""
        if self.enabled:
            save_to_cache(value, self.cache_dir, key)

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.enabled:
            return False
        return cache_exists(self.cache_dir, key)

    def clear(self) -> int:
        """Clear all cache."""
        if not self.enabled:
            return 0
        return clear_cache(self.cache_dir)

    def size(self) -> int:
        """Get cache size in bytes."""
        if not self.enabled:
            return 0
        return get_cache_size(self.cache_dir)

    def cleanup(self, max_age_days: int = 30) -> int:
        """Remove old cache files."""
        if not self.enabled:
            return 0
        return cleanup_old_cache(self.cache_dir, max_age_days)


def cached_result(cache_dir: Path, enabled: bool = True):
    """
    Decorator for caching function results.

    Args:
        cache_dir: Cache directory
        enabled: Whether caching is enabled

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            # Create cache key from function name and arguments
            key_parts = [func.__name__] + list(args)
            for k, v in sorted(kwargs.items()):
                key_parts.extend([k, v])

            cache_key = create_cache_key(*key_parts)

            # Try to load from cache
            result = load_from_cache(cache_dir, cache_key)
            if result is not None:
                return result

            # Compute and cache result
            result = func(*args, **kwargs)
            save_to_cache(result, cache_dir, cache_key)

            return result

        return wrapper

    return decorator
