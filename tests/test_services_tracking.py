"""Tests for tracking services."""

from datetime import timedelta

import pytest

from colonyscanalyser.models import Colony, Timepoint
from colonyscanalyser.services.tracking import (
    create_colonies_from_timepoints,
    filter_colonies,
    group_timepoints_by_center,
)


def create_test_timepoint(
    timestamp_hours: float, area: int, center: tuple
) -> Timepoint:
    """Helper to create test timepoints."""
    return Timepoint(
        timestamp=timedelta(hours=timestamp_hours),
        area=area,
        center=center,
        diameter=float(area**0.5),
        perimeter=float(area**0.5 * 4),
        color_average=(255, 255, 255),
    )


def create_test_colony(colony_id: int, timepoint_data: list) -> Colony:
    """Helper to create test colonies."""
    colony = Colony(id=colony_id)
    for timestamp_hours, area, center in timepoint_data:
        tp = create_test_timepoint(timestamp_hours, area, center)
        colony.add_timepoint(tp)
    return colony


def test_filter_colonies_valid():
    """Test filtering keeps valid colonies."""
    # Create valid colony with growth
    colony = create_test_colony(
        1,
        [
            (1, 50, (10, 10)),
            (2, 75, (10, 10)),
            (3, 100, (10, 10)),
            (4, 125, (10, 10)),
        ],
    )

    filtered = filter_colonies([colony])

    assert len(filtered) == 1
    assert filtered[0] == colony


def test_filter_colonies_too_few_timepoints():
    """Test filtering removes colonies with too few timepoints."""
    colony = create_test_colony(
        1,
        [
            (1, 100, (10, 10)),
            (2, 120, (10, 10)),
        ],
    )

    filtered = filter_colonies([colony], min_timepoints=3)

    assert len(filtered) == 0


def test_filter_colonies_too_small():
    """Test filtering removes colonies that are too small."""
    colony = create_test_colony(
        1,
        [
            (1, 20, (10, 10)),
            (2, 25, (10, 10)),
            (3, 30, (10, 10)),
        ],
    )

    filtered = filter_colonies([colony], min_area=50)

    assert len(filtered) == 0


def test_filter_colonies_no_growth():
    """Test filtering removes colonies with insufficient growth."""
    colony = create_test_colony(
        1,
        [
            (1, 100, (10, 10)),
            (2, 101, (10, 10)),
            (3, 102, (10, 10)),
            (4, 103, (10, 10)),
        ],
    )

    filtered = filter_colonies([colony], min_growth_rate=2.0)

    assert len(filtered) == 0


def test_filter_colonies_empty_list():
    """Test filtering empty list returns empty list."""
    filtered = filter_colonies([])
    assert len(filtered) == 0


def test_filter_colonies_mixed():
    """Test filtering with mix of valid and invalid colonies."""
    valid_colony = create_test_colony(
        1,
        [
            (1, 50, (10, 10)),
            (2, 75, (10, 10)),
            (3, 100, (10, 10)),
        ],
    )

    invalid_colony = create_test_colony(
        2,
        [
            (1, 20, (20, 20)),  # Too small
            (2, 22, (20, 20)),
            (3, 24, (20, 20)),
        ],
    )

    filtered = filter_colonies([valid_colony, invalid_colony])

    assert len(filtered) == 1
    assert filtered[0] == valid_colony


def test_create_colonies_from_timepoints_basic():
    """Test basic colony creation from timepoints."""
    timepoints = [
        create_test_timepoint(1, 50, (10, 10)),
        create_test_timepoint(2, 75, (10, 10)),  # Same location
        create_test_timepoint(1, 60, (50, 50)),  # Different location
        create_test_timepoint(2, 80, (50, 50)),  # Same as above
    ]

    colonies = create_colonies_from_timepoints(timepoints, distance_tolerance=5.0)

    assert len(colonies) == 2
    assert all(len(colony.timepoints) == 2 for colony in colonies)


def test_create_colonies_from_timepoints_empty():
    """Test error with empty timepoints."""
    with pytest.raises(ValueError, match="No timepoints provided"):
        create_colonies_from_timepoints([])


def test_create_colonies_from_timepoints_single():
    """Test with single timepoint."""
    timepoints = [create_test_timepoint(1, 50, (10, 10))]

    colonies = create_colonies_from_timepoints(timepoints)

    assert len(colonies) == 1
    assert len(colonies[0].timepoints) == 1


def test_group_timepoints_by_center_close():
    """Test grouping timepoints that are close together."""
    timepoints = [
        create_test_timepoint(1, 50, (10, 10)),
        create_test_timepoint(2, 75, (12, 11)),  # Close to first
        create_test_timepoint(1, 60, (50, 50)),  # Far from others
    ]

    groups = group_timepoints_by_center(timepoints, max_distance=5.0)

    assert len(groups) == 2
    # First group should have 2 timepoints, second should have 1
    group_sizes = [len(group) for group in groups]
    assert sorted(group_sizes) == [1, 2]


def test_group_timepoints_by_center_far():
    """Test grouping timepoints that are far apart."""
    timepoints = [
        create_test_timepoint(1, 50, (10, 10)),
        create_test_timepoint(2, 75, (100, 100)),
        create_test_timepoint(3, 60, (200, 200)),
    ]

    groups = group_timepoints_by_center(timepoints, max_distance=5.0)

    assert len(groups) == 3  # All separate
    assert all(len(group) == 1 for group in groups)


def test_group_timepoints_by_center_all_close():
    """Test grouping when all timepoints are close."""
    timepoints = [
        create_test_timepoint(1, 50, (10, 10)),
        create_test_timepoint(2, 75, (12, 12)),
        create_test_timepoint(3, 60, (14, 14)),
    ]

    groups = group_timepoints_by_center(timepoints, max_distance=10.0)

    assert len(groups) == 1  # All in one group
    assert len(groups[0]) == 3


def test_group_timepoints_by_center_empty():
    """Test grouping empty list."""
    groups = group_timepoints_by_center([])
    assert len(groups) == 0


def test_create_colonies_assigns_ids():
    """Test that created colonies get proper IDs."""
    timepoints = [
        create_test_timepoint(1, 50, (10, 10)),
        create_test_timepoint(1, 60, (50, 50)),
        create_test_timepoint(1, 70, (100, 100)),
    ]

    colonies = create_colonies_from_timepoints(timepoints, distance_tolerance=5.0)

    assert len(colonies) == 3
    colony_ids = [colony.id for colony in colonies]
    assert colony_ids == [1, 2, 3]


def test_integration_full_workflow():
    """Test complete workflow: create colonies then filter them."""
    # Create timepoints for two colonies - one valid, one invalid
    timepoints = [
        # Valid colony (grows over time)
        create_test_timepoint(1, 50, (10, 10)),
        create_test_timepoint(2, 75, (10, 10)),
        create_test_timepoint(3, 100, (10, 10)),
        # Invalid colony (too small)
        create_test_timepoint(1, 20, (50, 50)),
        create_test_timepoint(2, 22, (50, 50)),
        create_test_timepoint(3, 24, (50, 50)),
    ]

    # Create colonies from timepoints
    colonies = create_colonies_from_timepoints(timepoints, distance_tolerance=5.0)
    assert len(colonies) == 2

    # Filter colonies
    valid_colonies = filter_colonies(colonies, min_area=50)
    assert len(valid_colonies) == 1
    assert valid_colonies[0].timepoints[0].center == (10, 10)
