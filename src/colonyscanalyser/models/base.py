"""Simple base data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, TypeVar


@dataclass
class Identified:
    """Object with an ID."""

    id: int

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id <= 0:
            raise ValueError(f"ID must be a positive integer, got {self.id}")


@dataclass
class Named:
    """Object with a name."""

    name: str = ""


T = TypeVar("T", bound=Identified)


@dataclass
class IdentifiedCollection(Generic[T]):
    """Simple collection of objects with IDs."""

    _items: Dict[int, T] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items.values())

    def __getitem__(self, id: int) -> T:
        return self._items[id]

    def __contains__(self, item: T | int) -> bool:
        if isinstance(item, int):
            return item in self._items
        return item.id in self._items

    def add(self, item: T) -> None:
        """Add an item to the collection."""
        self._items[item.id] = item

    def remove(self, id: int) -> T:
        """Remove and return item by ID."""
        return self._items.pop(id)

    def get(self, id: int, default: Optional[T] = None) -> Optional[T]:
        """Get item by ID, return default if not found."""
        return self._items.get(id, default)

    @property
    def items(self) -> List[T]:
        """Return sorted list of items."""
        return sorted(self._items.values(), key=lambda x: x.id)
