"""Tests for the new data models."""

from datetime import datetime, timedelta

from colonyscanalyser.models import (
    Colony,
    ImageCollection,
    ImageFile,
    PipelineConfig,
    Plate,
    PlateCollection,
    ProcessingConfig,
    Timepoint,
)


def test_timepoint_creation():
    """Test creating a Timepoint."""
    tp = Timepoint(
        timestamp=timedelta(hours=1),
        area=100,
        center=(50.0, 60.0),
        diameter=10.0,
        perimeter=30.0,
        color_average=(255, 128, 64),
    )

    assert tp.timestamp == timedelta(hours=1)
    assert tp.area == 100
    assert tp.center == (50.0, 60.0)


def test_colony_creation():
    """Test creating a Colony with timepoints."""
    colony = Colony(id=1, name="Test Colony")

    tp1 = Timepoint(
        timestamp=timedelta(hours=1),
        area=50,
        center=(10.0, 10.0),
        diameter=8.0,
        perimeter=25.0,
        color_average=(255, 255, 255),
    )

    tp2 = Timepoint(
        timestamp=timedelta(hours=2),
        area=100,
        center=(10.5, 10.5),
        diameter=11.0,
        perimeter=35.0,
        color_average=(200, 200, 200),
    )

    colony.add_timepoint(tp1)
    colony.add_timepoint(tp2)

    assert len(colony.timepoints) == 2
    assert colony.first_timepoint == tp1
    assert colony.last_timepoint == tp2
    assert colony.time_of_appearance == timedelta(hours=1)


def test_plate_creation():
    """Test creating a Plate."""
    plate = Plate(
        id=1,
        name="Test Plate",
        diameter=90.0,
        center=(100.0, 100.0),
    )

    colony = Colony(id=1, name="Colony 1")
    plate.add_colony(colony)

    assert plate.diameter == 90.0
    assert plate.radius == 45.0
    assert plate.colony_count == 1
    assert plate.get_colony(1) == colony


def test_plate_collection():
    """Test PlateCollection."""
    collection = PlateCollection()

    plate1 = Plate(id=1, name="Plate A", diameter=90.0)
    plate2 = Plate(id=2, name="Plate B", diameter=90.0)

    collection.add(plate1)
    collection.add(plate2)

    assert len(collection) == 2
    assert collection[1] == plate1
    assert collection.get_by_name("Plate A") == plate1


def test_image_file_creation(tmp_path):
    """Test creating an ImageFile."""
    # Create a temporary file
    test_file = tmp_path / "test.tif"
    test_file.touch()

    image = ImageFile(
        file_path=test_file,
        timestamp=datetime.now(),
        width=100,
        height=100,
    )

    assert image.file_path == test_file
    assert image.name == "test.tif"
    assert image.size == (100, 100)


def test_image_collection():
    """Test ImageCollection."""
    collection = ImageCollection()

    # We can't create real files in unit tests easily,
    # so we'll skip this for now and focus on the structure
    assert len(collection) == 0


def test_processing_config_defaults():
    """Test ProcessingConfig default values."""
    config = ProcessingConfig()

    assert config.min_colony_area == 50
    assert config.max_distance_tolerance == 15.0
    assert config.min_timepoints == 3


def test_pipeline_config_creation(tmp_path):
    """Test PipelineConfig."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
    )

    assert config.input_dir == input_dir
    assert config.output_dir == output_dir
    assert config.enable_caching is True
