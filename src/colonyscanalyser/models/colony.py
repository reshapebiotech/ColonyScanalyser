"""Colony data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional, Tuple

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
