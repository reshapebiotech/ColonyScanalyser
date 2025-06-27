"""
ColonyScanalyser - An image analysis tool for measuring microorganism colony growth.

ColonyScanalyser will analyse and collate statistical data from agar plate images.
It provides fast, high-throughput image processing for microorganism colony analysis.
"""

__version__ = "0.6.2"

# Core exports
from .base import Identified, IdentifiedCollection
from .colony import Colony
from .growth_curve import GrowthCurve
from .image_file import ImageFile, ImageFileCollection
from .plate import Plate, PlateCollection

__all__ = [
    "__version__",
    "Identified",
    "IdentifiedCollection",
    "Colony",
    "Plate",
    "PlateCollection",
    "ImageFile",
    "ImageFileCollection",
    "GrowthCurve",
]
