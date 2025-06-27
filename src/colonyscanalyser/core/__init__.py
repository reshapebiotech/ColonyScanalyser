"""
Core domain logic package for ColonyScanalyser.

This package contains the core domain classes and business logic for
colony analysis, including Colony, Plate, ImageFile, and related functionality.
"""

from .base import (
    Identified,
    IdentifiedCollection,
    Named,
    TimeStamped,
    TimeStampElapsed,
    Unique,
)
from .colony import (
    Colony,
    colonies_filtered,
    colonies_from_timepoints,
    timepoints_from_image,
)
from .growth_curve import GrowthCurve
from .image_file import ImageFile, ImageFileCollection
from .plate import Plate, PlateCollection

__all__ = [
    # Base classes
    "Identified",
    "IdentifiedCollection",
    "Named",
    "Unique",
    "TimeStamped",
    "TimeStampElapsed",
    # Core domain classes
    "Colony",
    "GrowthCurve",
    "ImageFile",
    "ImageFileCollection",
    "Plate",
    "PlateCollection",
    # Colony analysis functions
    "colonies_filtered",
    "colonies_from_timepoints",
    "timepoints_from_image",
]
