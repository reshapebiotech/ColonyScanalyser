"""Tests for detection services."""

from datetime import timedelta

import numpy as np
import pytest
from skimage.measure import label

from colonyscanalyser.models import Timepoint
from colonyscanalyser.services.detection import timepoints_from_image


def test_timepoints_from_image_basic():
    """Test basic timepoint extraction from segmented image."""
    # Create a simple segmented image with two regions
    segmented = np.zeros((100, 100), dtype=int)
    segmented[20:40, 20:40] = 1  # First colony
    segmented[60:80, 60:80] = 2  # Second colony

    timestamp = timedelta(hours=1)

    timepoints = timepoints_from_image(segmented, timestamp)

    assert len(timepoints) == 2
    assert all(isinstance(tp, Timepoint) for tp in timepoints)
    assert all(tp.timestamp == timestamp for tp in timepoints)
    assert all(tp.area > 0 for tp in timepoints)


def test_timepoints_from_image_with_color():
    """Test timepoint extraction with color image."""
    # Create segmented image
    segmented = np.zeros((50, 50), dtype=int)
    segmented[10:30, 10:30] = 1

    # Create color image
    color_image = np.zeros((50, 50, 3), dtype=np.uint8)
    color_image[10:30, 10:30] = [255, 128, 64]  # Orange colony

    timestamp = timedelta(minutes=30)

    timepoints = timepoints_from_image(segmented, timestamp, color_image)

    assert len(timepoints) == 1
    tp = timepoints[0]
    assert tp.timestamp == timestamp
    assert tp.color_average != (0.0, 0.0, 0.0)  # Should have color


def test_timepoints_from_image_no_regions():
    """Test with empty segmented image."""
    segmented = np.zeros((50, 50), dtype=int)
    timestamp = timedelta(hours=2)

    timepoints = timepoints_from_image(segmented, timestamp)

    assert len(timepoints) == 0


def test_timepoints_from_image_dimension_mismatch():
    """Test error when image dimensions don't match."""
    segmented = np.zeros((50, 50), dtype=int)
    color_image = np.zeros((40, 40, 3), dtype=np.uint8)
    timestamp = timedelta(hours=1)

    with pytest.raises(ValueError, match="same dimensions"):
        timepoints_from_image(segmented, timestamp, color_image)


def test_timepoints_properties():
    """Test that timepoint properties are correctly set."""
    # Create a circular-ish region
    segmented = np.zeros((100, 100), dtype=int)
    y, x = np.ogrid[:100, :100]
    mask = (x - 50) ** 2 + (y - 50) ** 2 <= 20**2
    segmented[mask] = 1

    timestamp = timedelta(hours=3)

    timepoints = timepoints_from_image(segmented, timestamp)

    assert len(timepoints) == 1
    tp = timepoints[0]

    # Check basic properties
    assert tp.timestamp == timestamp
    assert tp.area > 0
    assert tp.diameter > 0
    assert tp.perimeter > 0
    assert len(tp.center) == 2
    assert tp.center[0] > 0 and tp.center[1] > 0
    assert tp.bbox is not None
    assert len(tp.bbox) == 4
    assert tp.label == 1


def test_timepoints_from_labeled_image():
    """Test with a properly labeled image using skimage.label."""
    # Create binary image
    binary = np.zeros((60, 60), dtype=bool)
    binary[10:20, 10:20] = True  # Small square
    binary[40:55, 40:55] = True  # Larger square

    # Label the regions
    labeled = label(binary)

    timestamp = timedelta(minutes=45)

    timepoints = timepoints_from_image(labeled, timestamp)

    assert len(timepoints) == 2
    # Should have different areas
    areas = [tp.area for tp in timepoints]
    assert len(set(areas)) == 2  # Two different areas
