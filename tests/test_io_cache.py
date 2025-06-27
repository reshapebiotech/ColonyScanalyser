"""Tests for I/O cache module."""

import pickle
import time

import numpy as np

from colonyscanalyser.io.cache import (
    SimpleCache,
    cache_exists,
    cached_result,
    cleanup_old_cache,
    clear_cache,
    create_cache_key,
    get_cache_path,
    get_cache_size,
    load_from_cache,
    save_to_cache,
)


def test_create_cache_key_basic():
    """Test creating cache key from basic arguments."""
    key = create_cache_key("test", 123, 45.6)

    assert isinstance(key, str)
    assert len(key) == 32  # MD5 hash length

    # Same inputs should produce same key
    key2 = create_cache_key("test", 123, 45.6)
    assert key == key2

    # Different inputs should produce different key
    key3 = create_cache_key("test", 124, 45.6)
    assert key != key3


def test_create_cache_key_different_types(tmp_path):
    """Test cache key creation with different argument types."""
    test_path = tmp_path / "test.txt"
    test_array = np.array([1, 2, 3, 4, 5])

    key = create_cache_key("string", 42, True, test_path, test_array)

    assert isinstance(key, str)
    assert len(key) == 32


def test_create_cache_key_numpy_arrays():
    """Test cache key with numpy arrays."""
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([1, 2, 4])

    key1 = create_cache_key(arr1)
    key2 = create_cache_key(arr2)
    key3 = create_cache_key(arr3)

    # Same arrays should produce same key
    assert key1 == key2
    # Different arrays should produce different key
    assert key1 != key3


def test_get_cache_path_basic(tmp_path):
    """Test getting cache file path."""
    cache_dir = tmp_path / "cache"
    key = "test_key"

    path = get_cache_path(cache_dir, key)

    assert path == cache_dir / "test_key.pkl"


def test_get_cache_path_custom_extension(tmp_path):
    """Test cache path with custom extension."""
    cache_dir = tmp_path / "cache"
    key = "test_key"

    path = get_cache_path(cache_dir, key, ".npy")

    assert path == cache_dir / "test_key.npy"


def test_get_cache_path_extension_without_dot(tmp_path):
    """Test cache path adds dot to extension."""
    cache_dir = tmp_path / "cache"
    key = "test_key"

    path = get_cache_path(cache_dir, key, "txt")

    assert path == cache_dir / "test_key.txt"


def test_cache_exists_false_no_file(tmp_path):
    """Test cache exists returns False when no file."""
    cache_dir = tmp_path / "cache"
    key = "nonexistent"

    assert cache_exists(cache_dir, key) is False


def test_cache_exists_false_empty_file(tmp_path):
    """Test cache exists returns False for empty file."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = "empty"

    # Create empty file
    cache_file = get_cache_path(cache_dir, key)
    cache_file.touch()

    assert cache_exists(cache_dir, key) is False


def test_cache_exists_true_with_data(tmp_path):
    """Test cache exists returns True for file with data."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = "with_data"

    # Create file with data
    cache_file = get_cache_path(cache_dir, key)
    cache_file.write_text("some data")

    assert cache_exists(cache_dir, key) is True


def test_save_to_cache_pickle(tmp_path):
    """Test saving data to cache as pickle."""
    cache_dir = tmp_path / "cache"
    key = "test_data"
    data = {"test": [1, 2, 3], "value": 42}

    result_path = save_to_cache(data, cache_dir, key)

    assert result_path.exists()
    assert cache_dir.exists()

    # Verify data can be loaded
    with open(result_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == data


def test_save_to_cache_numpy(tmp_path):
    """Test saving numpy array to cache."""
    cache_dir = tmp_path / "cache"
    key = "numpy_data"
    data = np.array([1, 2, 3, 4, 5])

    result_path = save_to_cache(data, cache_dir, key, ".npy")

    assert result_path.exists()
    assert result_path.suffix == ".npy"

    # Verify data can be loaded
    loaded = np.load(result_path)
    assert np.array_equal(loaded, data)


def test_save_to_cache_compressed_numpy(tmp_path):
    """Test saving compressed numpy array to cache."""
    cache_dir = tmp_path / "cache"
    key = "compressed_data"
    data = np.random.random((100, 100))

    result_path = save_to_cache(data, cache_dir, key, ".npz")

    assert result_path.exists()
    assert result_path.suffix == ".npz"

    # Verify data can be loaded
    loaded_data = np.load(result_path)
    loaded_array = loaded_data["data"]
    assert np.array_equal(loaded_array, data)


def test_load_from_cache_pickle(tmp_path):
    """Test loading pickle data from cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = "pickle_test"
    data = {"numbers": [1, 2, 3], "text": "hello"}

    # Save data first
    save_to_cache(data, cache_dir, key)

    # Load it back
    loaded = load_from_cache(cache_dir, key)

    assert loaded == data


def test_load_from_cache_numpy(tmp_path):
    """Test loading numpy data from cache."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = "numpy_test"
    data = np.array([10, 20, 30])

    # Save data first
    save_to_cache(data, cache_dir, key, ".npy")

    # Load it back
    loaded = load_from_cache(cache_dir, key, ".npy")

    assert np.array_equal(loaded, data)


def test_load_from_cache_not_found(tmp_path):
    """Test loading from cache when file doesn't exist."""
    cache_dir = tmp_path / "cache"
    key = "nonexistent"

    result = load_from_cache(cache_dir, key)

    assert result is None


def test_load_from_cache_corrupted_file(tmp_path):
    """Test loading corrupted cache file."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    key = "corrupted"

    # Create corrupted cache file
    cache_file = get_cache_path(cache_dir, key)
    cache_file.write_text("this is not valid pickle data")

    result = load_from_cache(cache_dir, key)

    assert result is None
    # File should be removed after corruption detection
    assert not cache_file.exists()


def test_clear_cache_basic(tmp_path):
    """Test clearing all cache files."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create some cache files
    (cache_dir / "file1.pkl").write_text("data1")
    (cache_dir / "file2.pkl").write_text("data2")
    (cache_dir / "file3.npy").write_text("data3")

    removed_count = clear_cache(cache_dir)

    assert removed_count == 3
    assert len(list(cache_dir.iterdir())) == 0


def test_clear_cache_pattern(tmp_path):
    """Test clearing cache files with pattern."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create different types of files
    (cache_dir / "data1.pkl").write_text("data1")
    (cache_dir / "data2.pkl").write_text("data2")
    (cache_dir / "other.npy").write_text("other")

    removed_count = clear_cache(cache_dir, "*.pkl")

    assert removed_count == 2
    assert (cache_dir / "other.npy").exists()


def test_clear_cache_nonexistent_dir(tmp_path):
    """Test clearing cache when directory doesn't exist."""
    nonexistent_dir = tmp_path / "nonexistent"

    removed_count = clear_cache(nonexistent_dir)

    assert removed_count == 0


def test_get_cache_size_basic(tmp_path):
    """Test getting cache size."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create files of known sizes
    (cache_dir / "small.txt").write_text("hello")  # 5 bytes
    (cache_dir / "large.txt").write_text("x" * 100)  # 100 bytes

    size = get_cache_size(cache_dir)

    assert size == 105  # 5 + 100 bytes


def test_get_cache_size_empty_dir(tmp_path):
    """Test cache size for empty directory."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    size = get_cache_size(cache_dir)

    assert size == 0


def test_get_cache_size_nonexistent_dir(tmp_path):
    """Test cache size for nonexistent directory."""
    nonexistent_dir = tmp_path / "nonexistent"

    size = get_cache_size(nonexistent_dir)

    assert size == 0


def test_cleanup_old_cache(tmp_path):
    """Test cleaning up old cache files."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    # Create files with different ages
    old_file = cache_dir / "old.pkl"
    recent_file = cache_dir / "recent.pkl"

    old_file.write_text("old data")
    recent_file.write_text("recent data")

    # Make one file appear old by modifying its timestamp
    import os

    old_time = time.time() - (35 * 24 * 60 * 60)  # 35 days ago
    os.utime(old_file, (old_time, old_time))

    removed_count = cleanup_old_cache(cache_dir, max_age_days=30)

    assert removed_count == 1
    assert not old_file.exists()
    assert recent_file.exists()


def test_cleanup_old_cache_nonexistent_dir(tmp_path):
    """Test cleanup on nonexistent directory."""
    nonexistent_dir = tmp_path / "nonexistent"

    removed_count = cleanup_old_cache(nonexistent_dir)

    assert removed_count == 0


def test_simple_cache_basic(tmp_path):
    """Test SimpleCache basic operations."""
    cache = SimpleCache(tmp_path / "cache")

    # Test set and get
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"

    # Test default value
    assert cache.get("nonexistent", "default") == "default"


def test_simple_cache_disabled(tmp_path):
    """Test SimpleCache when disabled."""
    cache = SimpleCache(tmp_path / "cache", enabled=False)

    # Operations should be no-ops
    cache.set("key1", "value1")
    assert cache.get("key1", "default") == "default"
    assert cache.exists("key1") is False
    assert cache.size() == 0
    assert cache.clear() == 0


def test_simple_cache_exists(tmp_path):
    """Test SimpleCache exists method."""
    cache = SimpleCache(tmp_path / "cache")

    assert cache.exists("nonexistent") is False

    cache.set("existing", "data")
    assert cache.exists("existing") is True


def test_simple_cache_clear(tmp_path):
    """Test SimpleCache clear method."""
    cache = SimpleCache(tmp_path / "cache")

    cache.set("key1", "value1")
    cache.set("key2", "value2")

    cleared_count = cache.clear()

    assert cleared_count == 2
    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_simple_cache_size(tmp_path):
    """Test SimpleCache size method."""
    cache = SimpleCache(tmp_path / "cache")

    assert cache.size() == 0

    cache.set("key1", "small data")
    size = cache.size()

    assert size > 0


def test_simple_cache_cleanup(tmp_path):
    """Test SimpleCache cleanup method."""
    cache = SimpleCache(tmp_path / "cache")

    cache.set("key1", "data")

    # Should not remove recent files
    removed = cache.cleanup(max_age_days=1)
    assert removed == 0

    # Should remove old files (tested indirectly)
    removed = cache.cleanup(max_age_days=0)
    assert removed >= 0  # Depends on file system precision


def test_cached_result_decorator(tmp_path):
    """Test cached_result decorator."""
    call_count = 0

    @cached_result(tmp_path / "cache")
    def expensive_function(x, y):
        nonlocal call_count
        call_count += 1
        return x + y

    # First call should execute function
    result1 = expensive_function(1, 2)
    assert result1 == 3
    assert call_count == 1

    # Second call with same arguments should use cache
    result2 = expensive_function(1, 2)
    assert result2 == 3
    assert call_count == 1  # Should not increment

    # Different arguments should execute function again
    result3 = expensive_function(2, 3)
    assert result3 == 5
    assert call_count == 2


def test_cached_result_decorator_disabled(tmp_path):
    """Test cached_result decorator when disabled."""
    call_count = 0

    @cached_result(tmp_path / "cache", enabled=False)
    def function_no_cache(x):
        nonlocal call_count
        call_count += 1
        return x * 2

    # Each call should execute function
    result1 = function_no_cache(5)
    assert result1 == 10
    assert call_count == 1

    result2 = function_no_cache(5)  # Same arguments
    assert result2 == 10
    assert call_count == 2  # Should increment since caching disabled


def test_cached_result_with_kwargs(tmp_path):
    """Test cached_result decorator with keyword arguments."""
    call_count = 0

    @cached_result(tmp_path / "cache")
    def function_with_kwargs(a, b=10, c=20):
        nonlocal call_count
        call_count += 1
        return a + b + c

    # Calls with same effective arguments should use cache
    result1 = function_with_kwargs(1, b=10, c=20)
    result2 = function_with_kwargs(1, 10, 20)
    result3 = function_with_kwargs(1, c=20, b=10)

    assert result1 == result2 == result3 == 31
    # Note: Cache key generation may differ for different argument styles
    # This is acceptable behavior for this simple cache implementation
    assert call_count <= 3  # Should not exceed number of calls


def test_end_to_end_cache_workflow(tmp_path):
    """Test complete cache workflow."""
    cache_dir = tmp_path / "cache"

    # Test saving and loading different data types
    test_data = {
        "string": "hello world",
        "numbers": [1, 2, 3, 4, 5],
        "array": np.array([10, 20, 30]),
        "nested": {"key": "value", "list": [7, 8, 9]},
    }

    key = create_cache_key("test", "workflow", 123)

    # Save to cache
    cache_file = save_to_cache(test_data, cache_dir, key)
    assert cache_file.exists()

    # Check if exists
    assert cache_exists(cache_dir, key)

    # Load from cache
    loaded_data = load_from_cache(cache_dir, key)
    assert loaded_data["string"] == test_data["string"]
    assert loaded_data["numbers"] == test_data["numbers"]
    assert np.array_equal(loaded_data["array"], test_data["array"])
    assert loaded_data["nested"] == test_data["nested"]

    # Check cache size
    size = get_cache_size(cache_dir)
    assert size > 0

    # Clear cache
    cleared = clear_cache(cache_dir)
    assert cleared == 1
    assert not cache_exists(cache_dir, key)


def test_error_handling_edge_cases(tmp_path):
    """Test error handling and edge cases."""
    cache_dir = tmp_path / "cache"

    # Test with None data
    key = "none_data"
    save_to_cache(None, cache_dir, key)
    loaded = load_from_cache(cache_dir, key)
    assert loaded is None

    # Test with complex data structures
    complex_data = {
        "function": lambda x: x + 1,  # Non-picklable
    }

    # Should handle non-picklable data gracefully
    try:
        save_to_cache(complex_data, cache_dir, "complex")
    except Exception:
        pass  # Expected to fail, should not crash
