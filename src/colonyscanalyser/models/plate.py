"""Plate data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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


@dataclass
class PlateCollection(IdentifiedCollection[Plate]):
    """Collection of plates."""

    def get_by_name(self, name: str) -> Optional[Plate]:
        """Get plate by name."""
        for plate in self._items.values():
            if plate.name == name:
                return plate
        return None
