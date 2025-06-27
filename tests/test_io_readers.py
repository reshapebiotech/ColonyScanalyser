"""Tests for I/O readers module."""

import csv
import pickle
from pathlib import Path

import numpy as np
import pytest
from skimage.io import imsave

from colonyscanalyser.io.readers import (
    ensure_rgb,
    file_exists_with_data,
    find_files_by_pattern,
    find_image_files,
    get_file_info,
    load_data,
    load_image,
)


def create_test_image(tmp_path: Path, name: str = "test.tif") -> Path:
    """Helper to create a test image file."""
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    image_path = tmp_path / name
    imsave(str(image_path), image)
    return image_path


def test_load_image_basic(tmp_path):
    """Test basic image loading."""
    image_path = create_test_image(tmp_path)

    image = load_image(image_path)

    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 3  # Should be RGB
    assert image.shape[2] == 3


def test_load_image_grayscale(tmp_path):
    """Test loading image as grayscale."""
    image_path = create_test_image(tmp_path)

    image = load_image(image_path, as_gray=True)

    assert isinstance(image, np.ndarray)
    assert len(image.shape) == 2  # Should be grayscale


def test_load_image_file_not_found(tmp_path):
    """Test error when image file doesn't exist."""
    missing_path = tmp_path / "missing.tif"

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image(missing_path)


def test_ensure_rgb_grayscale():
    """Test converting grayscale to RGB."""
    gray_image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

    rgb_image = ensure_rgb(gray_image)

    assert len(rgb_image.shape) == 3
    assert rgb_image.shape[2] == 3


def test_ensure_rgb_rgba():
    """Test converting RGBA to RGB."""
    rgba_image = np.random.randint(0, 255, (50, 50, 4), dtype=np.uint8)

    rgb_image = ensure_rgb(rgba_image)

    assert len(rgb_image.shape) == 3
    assert rgb_image.shape[2] == 3


def test_ensure_rgb_already_rgb():
    """Test RGB image passes through unchanged."""
    rgb_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

    result = ensure_rgb(rgb_image)

    assert result.shape == rgb_image.shape
    assert np.array_equal(result, rgb_image)


def test_find_image_files_basic(tmp_path):
    """Test finding image files in directory."""
    # Create test image files
    create_test_image(tmp_path, "image1.tif")
    create_test_image(tmp_path, "image2.png")
    create_test_image(tmp_path, "image3.jpg")

    # Create non-image file
    (tmp_path / "data.txt").write_text("not an image")

    files = find_image_files(tmp_path)

    assert len(files) == 3
    assert all(f.suffix in [".tif", ".png", ".jpg"] for f in files)
    assert all(f.name.startswith("image") for f in files)


def test_find_image_files_custom_extensions(tmp_path):
    """Test finding files with custom extensions."""
    create_test_image(tmp_path, "test.tif")
    create_test_image(tmp_path, "test.png")

    files = find_image_files(tmp_path, extensions=["tif"])

    assert len(files) == 1
    assert files[0].suffix == ".tif"


def test_find_image_files_directory_not_found(tmp_path):
    """Test error when directory doesn't exist."""
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        find_image_files(missing_dir)


def test_find_image_files_recursive(tmp_path):
    """Test recursive file search."""
    # Create images in subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    create_test_image(tmp_path, "root.tif")
    create_test_image(subdir, "sub.tif")

    files = find_image_files(tmp_path, recursive=True)

    assert len(files) == 2

    # Non-recursive should only find root image
    files_nonrecursive = find_image_files(tmp_path, recursive=False)
    assert len(files_nonrecursive) == 1


def test_load_data_numpy(tmp_path):
    """Test loading numpy data."""
    data = np.array([1, 2, 3, 4, 5])
    data_path = tmp_path / "data.npy"
    np.save(data_path, data)

    loaded = load_data(data_path, format="numpy")

    assert np.array_equal(loaded, data)


def test_load_data_pickle(tmp_path):
    """Test loading pickle data."""
    data = {"test": [1, 2, 3], "value": 42}
    data_path = tmp_path / "data.pkl"

    with open(data_path, "wb") as f:
        pickle.dump(data, f)

    loaded = load_data(data_path, format="pickle")

    assert loaded == data


def test_load_data_csv(tmp_path):
    """Test loading CSV data."""
    data = [[1, 2, 3], [4, 5, 6]]
    data_path = tmp_path / "data.csv"

    with open(data_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    loaded = load_data(data_path, format="csv")

    assert loaded.shape == (2, 3)
    assert np.array_equal(loaded, np.array(data))


def test_load_data_auto_format(tmp_path):
    """Test auto-detecting data format."""
    data = np.array([1, 2, 3])

    # Test .npy auto-detection
    npy_path = tmp_path / "data.npy"
    np.save(npy_path, data)

    loaded = load_data(npy_path)  # format="auto" is default
    assert np.array_equal(loaded, data)


def test_load_data_file_not_found(tmp_path):
    """Test error when data file doesn't exist."""
    missing_path = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError, match="Data file not found"):
        load_data(missing_path)


def test_load_data_unsupported_format(tmp_path):
    """Test error for unsupported format."""
    data_path = tmp_path / "data.xyz"
    data_path.write_text("some data")

    with pytest.raises(ValueError, match="Cannot auto-detect format"):
        load_data(data_path)


def test_file_exists_with_data_exists(tmp_path):
    """Test file exists check with data."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("some data")

    assert file_exists_with_data(test_file) is True


def test_file_exists_with_data_empty(tmp_path):
    """Test file exists but is empty."""
    test_file = tmp_path / "empty.txt"
    test_file.touch()  # Creates empty file

    assert file_exists_with_data(test_file) is False


def test_file_exists_with_data_missing(tmp_path):
    """Test file doesn't exist."""
    missing_file = tmp_path / "missing.txt"

    assert file_exists_with_data(missing_file) is False


def test_get_file_info_basic(tmp_path):
    """Test getting file information."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello world")

    info = get_file_info(test_file)

    assert info["name"] == "test.txt"
    assert info["size"] > 0
    assert info["suffix"] == ".txt"
    assert info["is_file"] is True
    assert info["is_dir"] is False


def test_get_file_info_directory(tmp_path):
    """Test getting directory information."""
    test_dir = tmp_path / "testdir"
    test_dir.mkdir()

    info = get_file_info(test_dir)

    assert info["name"] == "testdir"
    assert info["is_file"] is False
    assert info["is_dir"] is True


def test_get_file_info_not_found(tmp_path):
    """Test error when file doesn't exist."""
    missing_file = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="File not found"):
        get_file_info(missing_file)


def test_find_files_by_pattern_basic(tmp_path):
    """Test finding files by pattern."""
    # Create test files
    (tmp_path / "frame_001.tif").touch()
    (tmp_path / "frame_002.tif").touch()
    (tmp_path / "other.png").touch()

    files = find_files_by_pattern(tmp_path, "frame_*.tif")

    assert len(files) == 2
    assert all("frame_" in f.name for f in files)
    assert all(f.suffix == ".tif" for f in files)


def test_find_files_by_pattern_recursive(tmp_path):
    """Test recursive pattern search."""
    # Create files in subdirectory
    subdir = tmp_path / "subdir"
    subdir.mkdir()
    (tmp_path / "test.txt").touch()
    (subdir / "test.txt").touch()

    files = find_files_by_pattern(tmp_path, "test.txt", recursive=True)

    assert len(files) == 2

    # Non-recursive should find only root file
    files_nonrecursive = find_files_by_pattern(tmp_path, "test.txt", recursive=False)
    assert len(files_nonrecursive) == 1


def test_find_files_by_pattern_directory_not_found(tmp_path):
    """Test error when directory doesn't exist."""
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError, match="Directory not found"):
        find_files_by_pattern(missing_dir, "*.txt")


def test_find_files_excludes_hidden(tmp_path):
    """Test that hidden files are excluded."""
    # Create regular and hidden files
    (tmp_path / "visible.txt").touch()
    (tmp_path / ".hidden.txt").touch()

    files = find_files_by_pattern(tmp_path, "*.txt")

    assert len(files) == 1
    assert files[0].name == "visible.txt"
