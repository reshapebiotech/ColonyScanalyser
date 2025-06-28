"""Plate data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from numpy import ndarray

from .base import IdentifiedCollection
from .colony import Colony


@dataclass
class Plate:
    """An agar plate containing colonies."""

    id: int
    diameter: float
    name: str = ""
    center: Tuple[float, float] = (0.0, 0.0)
    edge_cut: float = 0.0
    colonies: List[Colony] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id <= 0:
            raise ValueError(f"ID must be a positive integer, got {self.id}")

    @property
    def radius(self) -> float:
        """Plate radius."""
        return self.diameter / 2.0

    @property
    def colony_count(self) -> int:
        """Number of colonies on this plate."""
        return len(self.colonies)

    def add_colony(self, colony: Colony) -> None:
        """Add a colony to this plate."""
        self.colonies.append(colony)

    def get_colony(self, colony_id: int) -> Optional[Colony]:
        """Get colony by ID."""
        for colony in self.colonies:
            if colony.id == colony_id:
                return colony
        return None

    def slice_plate_image(
        self, image: ndarray, background_color: Tuple = (0,)
    ) -> ndarray:
        """
        Extract plate region from full image.

        Args:
            image: Full image as numpy array
            background_color: Color to use for background areas

        Returns:
            Cropped plate image
        """
        from ..utils.geometry import crop_circle_from_image

        return crop_circle_from_image(
            image,
            center=self.center,
            radius=self.radius - self.edge_cut,
            background_color=background_color,
        )

    @property
    def growth_curve(self) -> "GrowthCurve":
        """Calculate aggregated growth curve from all colonies on this plate."""
        from collections import defaultdict

        from .colony import GrowthCurve

        if not self.colonies:
            return GrowthCurve({})

        # Aggregate data from all colonies
        aggregated_data = defaultdict(list)

        for colony in self.colonies:
            colony_curve = colony.growth_curve
            for timestamp, area in colony_curve.data.items():
                aggregated_data[timestamp].append(area)

        # Keep the lists of areas for each timestamp for median calculation
        return GrowthCurve(dict(aggregated_data))


@dataclass
class PlateCollection(IdentifiedCollection[Plate]):
    """Collection of plates."""

    def get_by_name(self, name: str) -> Optional[Plate]:
        """Get plate by name."""
        for plate in self._items.values():
            if plate.name == name:
                return plate
        return None
