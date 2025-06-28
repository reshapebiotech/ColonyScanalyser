"""Colony data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from numpy import ndarray
from skimage.measure._regionprops import RegionProperties


@dataclass
class Timepoint:
    """Colony measurements at a specific time."""

    timestamp: timedelta
    area: int
    center: Tuple[float, float]
    diameter: float
    perimeter: float
    color_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox: Optional[Tuple[int, int, int, int]] = None
    image: Optional[ndarray] = None
    label: Optional[int] = None
    region_props: Optional[RegionProperties] = None

    def __lt__(self, other: Timepoint) -> bool:
        """Enable sorting by timestamp."""
        return self.timestamp < other.timestamp


@dataclass
class Colony:
    """A colony with measurements over time."""

    id: int
    name: str = ""
    timepoints: List[Timepoint] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id <= 0:
            raise ValueError(f"ID must be a positive integer, got {self.id}")

    @property
    def first_timepoint(self) -> Optional[Timepoint]:
        """Get the earliest timepoint."""
        return min(self.timepoints) if self.timepoints else None

    @property
    def last_timepoint(self) -> Optional[Timepoint]:
        """Get the latest timepoint."""
        return max(self.timepoints) if self.timepoints else None

    @property
    def timepoint_last(self) -> Optional[Timepoint]:
        """Get the latest timepoint (alias for last_timepoint)."""
        return self.last_timepoint

    @property
    def time_of_appearance(self) -> Optional[timedelta]:
        """When the colony first appeared."""
        first = self.first_timepoint
        return first.timestamp if first else None

    @property
    def center(self) -> Optional[Tuple[float, float]]:
        """Average center position across all timepoints."""
        if not self.timepoints:
            return None

        x_coords = [tp.center[0] for tp in self.timepoints]
        y_coords = [tp.center[1] for tp in self.timepoints]
        return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

    @property
    def average_color(self) -> Optional[Tuple[float, float, float]]:
        """Average color across all timepoints."""
        if not self.timepoints:
            return None

        colors = [tp.color_average for tp in self.timepoints]
        r = sum(color[0] for color in colors) / len(colors)
        g = sum(color[1] for color in colors) / len(colors)
        b = sum(color[2] for color in colors) / len(colors)
        return (r, g, b)

    def add_timepoint(self, timepoint: Timepoint) -> None:
        """Add a timepoint to this colony."""
        self.timepoints.append(timepoint)
        self.timepoints.sort()

    def get_timepoint(self, timestamp: timedelta) -> Optional[Timepoint]:
        """Get timepoint at specific timestamp."""
        for tp in self.timepoints:
            if tp.timestamp == timestamp:
                return tp
        return None

    @property
    def growth_curve(self) -> "GrowthCurve":
        """Calculate growth curve from timepoints."""
        return GrowthCurve.from_timepoints(self.timepoints)


class GrowthCurve:
    """Growth curve calculated from colony timepoints."""

    def __init__(self, data: Dict[timedelta, int]):
        """Initialize with timestamp -> area mapping."""
        self.data = data

    @classmethod
    def from_timepoints(cls, timepoints: List[Timepoint]) -> "GrowthCurve":
        """Create growth curve from timepoints."""
        data = {}
        for tp in sorted(timepoints, key=lambda x: x.timestamp):
            data[tp.timestamp] = tp.area
        return cls(data)

    @property
    def lag_time(self) -> timedelta:
        """Estimated lag time before exponential growth."""
        if len(self.data) < 3:
            return timedelta(0)

        # Simple estimation: time to reach 2x initial area
        timestamps = sorted(self.data.keys())
        initial_area = self.data[timestamps[0]]
        target_area = initial_area * 2

        for timestamp in timestamps[1:]:
            if self.data[timestamp] >= target_area:
                return timestamp

        return timestamps[-1] if timestamps else timedelta(0)

    @property
    def lag_time_std(self) -> timedelta:
        """Standard deviation of lag time (simplified)."""
        return timedelta(minutes=5)  # Placeholder

    @property
    def growth_rate(self) -> float:
        """Growth rate per second."""
        if len(self.data) < 2:
            return 0.0

        timestamps = sorted(self.data.keys())

        # Convert areas to single values (handle both single values and lists)
        areas = []
        for ts in timestamps:
            value = self.data[ts]
            if isinstance(value, list):
                # Take median of list values
                from statistics import median

                areas.append(median(value) if value else 0)
            else:
                areas.append(value)

        # Calculate average growth rate
        total_rate = 0.0
        valid_intervals = 0

        for i in range(1, len(areas)):
            dt = (timestamps[i] - timestamps[i - 1]).total_seconds()
            if dt > 0 and areas[i - 1] > 0:
                rate = (areas[i] - areas[i - 1]) / (areas[i - 1] * dt)
                total_rate += rate
                valid_intervals += 1

        return total_rate / valid_intervals if valid_intervals > 0 else 0.0

    @property
    def doubling_time(self) -> timedelta:
        """Time for colony to double in size."""
        rate = self.growth_rate
        if rate <= 0:
            return timedelta(hours=24)  # Default fallback

        # Doubling time = ln(2) / growth_rate
        import math

        seconds = math.log(2) / rate if rate > 0 else 86400
        return timedelta(
            seconds=max(60, min(86400, seconds))
        )  # Clamp between 1min and 1day

    @property
    def carrying_capacity(self) -> int:
        """Maximum area reached by the colony."""
        if not self.data:
            return 0

        max_value = 0
        for value in self.data.values():
            if isinstance(value, list):
                # For aggregated data (lists of areas)
                if value:  # Check if list is not empty
                    max_value = max(max_value, max(value))
            else:
                # For individual colony data (single values)
                max_value = max(max_value, value)

        return max_value
