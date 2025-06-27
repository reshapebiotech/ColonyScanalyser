"""Image data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class ImageFile:
    """An image file with metadata."""

    file_path: Path
    timestamp: datetime
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None

    def __post_init__(self):
        if not self.file_path.exists():
            raise FileNotFoundError(f"Image file not found: {self.file_path}")

    @property
    def name(self) -> str:
        """Image filename."""
        return self.file_path.name

    @property
    def size(self) -> Optional[tuple[int, int]]:
        """Image dimensions as (width, height)."""
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return None


@dataclass
class ImageCollection:
    """Collection of image files."""

    images: List[ImageFile] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.images)

    def __iter__(self):
        return iter(self.images)

    def __getitem__(self, index: int) -> ImageFile:
        return self.images[index]

    def add(self, image: ImageFile) -> None:
        """Add an image to the collection."""
        self.images.append(image)

    def sort_by_timestamp(self) -> None:
        """Sort images by timestamp."""
        self.images.sort(key=lambda img: img.timestamp)

    @property
    def sorted_by_time(self) -> List[ImageFile]:
        """Get images sorted by timestamp."""
        return sorted(self.images, key=lambda img: img.timestamp)

    def get_by_name(self, name: str) -> Optional[ImageFile]:
        """Get image by filename."""
        for image in self.images:
            if image.name == name:
                return image
        return None
