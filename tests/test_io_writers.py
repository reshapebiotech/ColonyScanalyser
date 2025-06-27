"""Tests for I/O writers module."""

import csv
import pickle
from datetime import datetime

import numpy as np
import pytest
from skimage.io import imread

from colonyscanalyser.io.writers import (
    create_safe_filename,
    create_timestamped_filename,
    ensure_directory,
    export_colony_data,
    save_data_csv,
    save_data_dict_csv,
    save_data_numpy,
    save_data_pickle,
    save_image,
)


def test_save_image_basic(tmp_path):
    """Test basic image saving."""
    # Create test image
    image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
    output_path = tmp_path / "test_image.png"

    result_path = save_image(image, output_path)

    assert result_path == output_path
    assert output_path.exists()

    # Verify image can be loaded back
    loaded = imread(str(output_path))
    assert loaded.shape == image.shape


def test_save_image_creates_directory(tmp_path):
    """Test that save_image creates parent directories."""
    image = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
    nested_path = tmp_path / "subdir" / "deep" / "image.tif"

    result_path = save_image(image, nested_path)

    assert result_path == nested_path
    assert nested_path.exists()
    assert nested_path.parent.exists()


def test_save_image_invalid_data(tmp_path):
    """Test error when saving invalid image data."""
    invalid_data = "not an image"
    output_path = tmp_path / "invalid.png"

    with pytest.raises(ValueError, match="Failed to save image"):
        save_image(invalid_data, output_path)


def test_save_data_csv_basic(tmp_path):
    """Test basic CSV saving."""
    data = [["apple", 5, 1.2], ["banana", 3, 0.8], ["cherry", 10, 2.1]]
    headers = ["fruit", "count", "price"]
    output_path = tmp_path / "test_data.csv"

    result_path = save_data_csv(data, output_path, headers)

    assert result_path.suffix == ".csv"
    assert result_path.exists()

    # Verify CSV content
    with open(result_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert rows[0] == headers
        assert rows[1] == ["apple", "5", "1.2"]


def test_save_data_csv_no_headers(tmp_path):
    """Test CSV saving without headers."""
    data = [[1, 2, 3], [4, 5, 6]]
    output_path = tmp_path / "no_headers"

    result_path = save_data_csv(data, output_path)

    assert result_path.suffix == ".csv"

    with open(result_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[0] == ["1", "2", "3"]


def test_save_data_csv_single_values(tmp_path):
    """Test CSV saving with single values (not iterables)."""
    data = [42, "hello", 3.14]
    output_path = tmp_path / "singles.csv"

    result_path = save_data_csv(data, output_path)

    with open(result_path, "r") as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert rows[0] == ["42"]
        assert rows[1] == ["hello"]
        assert rows[2] == ["3.14"]


def test_save_data_dict_csv_basic(tmp_path):
    """Test saving dictionary data to CSV."""
    data = [
        {"name": "Alice", "age": 30, "city": "NYC"},
        {"name": "Bob", "age": 25, "city": "LA"},
        {"name": "Charlie", "age": 35, "city": "Chicago"},
    ]
    output_path = tmp_path / "dict_data.csv"

    result_path = save_data_dict_csv(data, output_path)

    assert result_path.exists()

    with open(result_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["name"] == "Alice"
        assert rows[0]["age"] == "30"


def test_save_data_dict_csv_custom_fieldnames(tmp_path):
    """Test saving dict CSV with custom field names."""
    data = [{"a": 1, "b": 2, "c": 3}]
    fieldnames = ["c", "a"]  # Only include some fields, different order
    output_path = tmp_path / "custom_fields.csv"

    result_path = save_data_dict_csv(data, output_path, fieldnames)

    with open(result_path, "r") as f:
        content = f.read()
        lines = content.strip().split("\n")
        assert lines[0] == "c,a"  # Header
        assert lines[1] == "3,1"  # Data


def test_save_data_dict_csv_empty_data(tmp_path):
    """Test error when saving empty dict data."""
    output_path = tmp_path / "empty.csv"

    with pytest.raises(ValueError, match="No data to save"):
        save_data_dict_csv([], output_path)


def test_save_data_numpy_basic(tmp_path):
    """Test saving numpy array."""
    data = np.array([1, 2, 3, 4, 5])
    output_path = tmp_path / "array_data"

    result_path = save_data_numpy(data, output_path)

    assert result_path.suffix == ".npy"
    assert result_path.exists()

    # Verify data can be loaded back
    loaded = np.load(result_path)
    assert np.array_equal(loaded, data)


def test_save_data_numpy_compressed(tmp_path):
    """Test saving compressed numpy array."""
    data = np.random.random((100, 100))
    output_path = tmp_path / "compressed_array"

    result_path = save_data_numpy(data, output_path, compressed=True)

    assert result_path.suffix == ".npz"
    assert result_path.exists()

    # Verify data can be loaded back
    loaded_data = np.load(result_path)
    loaded_array = loaded_data["data"]
    assert np.array_equal(loaded_array, data)


def test_save_data_pickle_basic(tmp_path):
    """Test saving data with pickle."""
    data = {"numbers": [1, 2, 3], "text": "hello", "nested": {"key": "value"}}
    output_path = tmp_path / "pickle_data"

    result_path = save_data_pickle(data, output_path)

    assert result_path.suffix == ".pkl"
    assert result_path.exists()

    # Verify data can be loaded back
    with open(result_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == data


def test_save_data_pickle_custom_protocol(tmp_path):
    """Test saving pickle with custom protocol."""
    data = [1, 2, 3, 4, 5]
    output_path = tmp_path / "protocol_test"

    result_path = save_data_pickle(data, output_path, protocol=2)

    assert result_path.exists()

    with open(result_path, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == data


def test_create_safe_filename_basic():
    """Test creating safe filename from parts."""
    parts = ["experiment", "2024", "plate_1", "data"]

    filename = create_safe_filename(parts)

    assert filename == "experiment_2024_plate_1_data"


def test_create_safe_filename_with_spaces():
    """Test safe filename with spaces and special characters."""
    parts = ["my experiment", "data file", "v1.0"]

    filename = create_safe_filename(parts)

    assert filename == "my_experiment_data_file_v1.0"


def test_create_safe_filename_custom_separator():
    """Test safe filename with custom separator."""
    parts = ["part1", "part2", "part3"]

    filename = create_safe_filename(parts, separator="-")

    assert filename == "part1-part2-part3"


def test_create_safe_filename_empty_parts():
    """Test safe filename with empty parts."""
    parts = ["valid", "", "also_valid", None, "last"]

    filename = create_safe_filename(parts)

    assert filename == "valid_also_valid_last"


def test_create_safe_filename_max_length():
    """Test filename truncation when too long."""
    parts = ["very_long_part"] * 20

    filename = create_safe_filename(parts, max_length=50)

    assert len(filename) <= 50


def test_ensure_directory_creates_new(tmp_path):
    """Test creating new directory."""
    new_dir = tmp_path / "new_directory"

    result = ensure_directory(new_dir)

    assert result == new_dir
    assert new_dir.exists()
    assert new_dir.is_dir()


def test_ensure_directory_exists(tmp_path):
    """Test with existing directory."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()

    result = ensure_directory(existing_dir)

    assert result == existing_dir
    assert existing_dir.exists()


def test_ensure_directory_nested(tmp_path):
    """Test creating nested directories."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"

    result = ensure_directory(nested_dir)

    assert result == nested_dir
    assert nested_dir.exists()
    assert nested_dir.is_dir()


def test_create_timestamped_filename():
    """Test creating timestamped filename."""
    base_name = "experiment"
    extension = "csv"

    filename = create_timestamped_filename(base_name, extension)

    assert filename.startswith("experiment_")
    assert filename.endswith(".csv")
    assert len(filename) > len("experiment_.csv")


def test_create_timestamped_filename_custom_format():
    """Test timestamped filename with custom format."""
    base_name = "data"
    extension = ".txt"

    filename = create_timestamped_filename(
        base_name, extension, timestamp_format="%Y%m%d"
    )

    # Should contain current date in YYYYMMDD format
    today = datetime.now().strftime("%Y%m%d")
    assert today in filename
    assert filename.endswith(".txt")


def test_create_timestamped_filename_no_extension():
    """Test timestamped filename without extension."""
    base_name = "output"

    filename = create_timestamped_filename(base_name)

    assert filename.startswith("output_")
    assert "." not in filename.split("_")[-1]  # No extension added


def test_export_colony_data_csv(tmp_path):
    """Test exporting colony data as CSV."""

    # Create mock colonies that implement __iter__
    class MockColony:
        def __init__(self, data):
            self.data = data

        def __iter__(self):
            return iter(self.data)

    colonies = [
        MockColony([1, "Colony1", 10, "(10,10)", 5, 100, 200]),
        MockColony([2, "Colony2", 20, "(20,20)", 3, 50, 150]),
    ]

    created_files = export_colony_data(colonies, tmp_path, formats=["csv"])

    assert len(created_files) == 1
    assert created_files[0].suffix == ".csv"
    assert created_files[0].exists()


def test_export_colony_data_pickle(tmp_path):
    """Test exporting colony data as pickle."""
    colonies = [{"id": 1, "name": "test"}, {"id": 2, "name": "test2"}]

    created_files = export_colony_data(colonies, tmp_path, formats=["pickle"])

    assert len(created_files) == 1
    assert created_files[0].suffix == ".pkl"

    # Verify pickle can be loaded
    with open(created_files[0], "rb") as f:
        loaded = pickle.load(f)
    assert loaded == colonies


class MockColony:
    def __iter__(self):
        return iter([1, "test", 10])


def test_export_colony_data_multiple_formats(tmp_path):
    """Test exporting in multiple formats."""

    colonies = [MockColony()]

    created_files = export_colony_data(colonies, tmp_path, formats=["csv", "pickle"])

    assert len(created_files) == 2
    extensions = [f.suffix for f in created_files]
    assert ".csv" in extensions
    assert ".pkl" in extensions


def test_export_colony_data_empty_list(tmp_path):
    """Test exporting empty colony list."""
    created_files = export_colony_data([], tmp_path)

    assert len(created_files) == 0


def test_export_colony_data_custom_prefix(tmp_path):
    """Test exporting with custom filename prefix."""
    colonies = [{"test": "data"}]

    created_files = export_colony_data(
        colonies, tmp_path, prefix="my_colonies", formats=["pickle"]
    )

    assert len(created_files) == 1
    assert created_files[0].name.startswith("my_colonies")


def test_writers_error_handling(tmp_path):
    """Test error handling in writer functions."""
    # Test save_image with completely invalid data type
    with pytest.raises(ValueError):
        save_image("not an array at all", tmp_path / "bad.png")

    # Test CSV error with invalid data
    with pytest.raises(ValueError):
        save_data_csv(None, tmp_path / "test.csv")


def test_file_extension_handling(tmp_path):
    """Test that functions correctly handle file extensions."""
    # CSV functions should add .csv extension
    result = save_data_csv([[1, 2]], tmp_path / "test")
    assert result.suffix == ".csv"

    # Pickle should add .pkl extension
    result = save_data_pickle({"test": 1}, tmp_path / "test")
    assert result.suffix == ".pkl"

    # NumPy should add appropriate extension
    result = save_data_numpy(np.array([1, 2]), tmp_path / "test")
    assert result.suffix == ".npy"

    result = save_data_numpy(np.array([1, 2]), tmp_path / "test", compressed=True)
    assert result.suffix == ".npz"
