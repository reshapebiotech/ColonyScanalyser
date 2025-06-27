"""Colony tracking services."""

from math import dist
from typing import List

import numpy as np

from ...models import Colony, Timepoint


def filter_colonies(
    colonies: List[Colony],
    min_timepoints: int = 3,
    min_area: int = 50,
    min_growth_rate: float = 1.4,
) -> List[Colony]:
    """
    Filter colonies to keep only valid ones.

    Args:
        colonies: List of colonies to filter
        min_timepoints: Minimum number of timepoints required
        min_area: Minimum maximum area in pixels
        min_growth_rate: Minimum mean growth per timepoint

    Returns:
        Filtered list of valid colonies
    """
    if not colonies:
        return colonies

    def is_valid_colony(colony: Colony) -> bool:
        """Check if colony meets validity criteria."""
        timepoints = colony.timepoints

        if len(timepoints) < min_timepoints:
            return False

        areas = np.array([tp.area for tp in timepoints])

        # Must have significant maximum area
        if areas.max() <= min_area:
            return False

        # Must show growth over time
        if len(areas) > 1 and np.diff(areas).mean() <= min_growth_rate:
            return False

        return True

    return [colony for colony in colonies if is_valid_colony(colony)]


def create_colonies_from_timepoints(
    timepoints: List[Timepoint], distance_tolerance: float = 15.0
) -> List[Colony]:
    """
    Group timepoints into colonies based on spatial proximity.

    Args:
        timepoints: List of timepoints to group
        distance_tolerance: Maximum distance between timepoint centers

    Returns:
        List of colonies with grouped timepoints
    """
    if not timepoints:
        raise ValueError("No timepoints provided")

    # Group timepoints by center proximity
    timepoint_groups = group_timepoints_by_center(timepoints, distance_tolerance)

    colonies = []
    for i, group in enumerate(timepoint_groups, start=1):
        colony = Colony(id=i)
        for timepoint in group:
            colony.add_timepoint(timepoint)
        colonies.append(colony)

    return colonies


def group_timepoints_by_center(
    timepoints: List[Timepoint], max_distance: float = 15.0
) -> List[List[Timepoint]]:
    """
    Group timepoints by euclidean distance between centers.

    Args:
        timepoints: List of timepoints to group
        max_distance: Maximum distance between centers in same group

    Returns:
        List of timepoint groups
    """
    groups = []
    remaining = timepoints.copy()

    while remaining:
        # Start new group with first remaining timepoint
        current_group = [remaining.pop(0)]

        # Find all timepoints within distance of any point in current group
        i = 0
        while i < len(remaining):
            # Check if this timepoint is close to any in current group
            is_close = any(
                dist(remaining[i].center, group_tp.center) <= max_distance
                for group_tp in current_group
            )

            if is_close:
                current_group.append(remaining.pop(i))
            else:
                i += 1

        groups.append(current_group)

    return groups
