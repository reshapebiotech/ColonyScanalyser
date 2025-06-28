"""
ColonyScanalyser - An image analysis tool for measuring microorganism colony growth.

ColonyScanalyser will analyse and collate statistical data from agar plate images.
It provides fast, high-throughput image processing for microorganism colony analysis.
"""

__version__ = "0.6.2"

# Model exports
from .models.base import Identified, IdentifiedCollection
from .models.colony import Colony, GrowthCurve
from .models.image import ImageCollection, ImageFile
from .models.plate import Plate, PlateCollection

__all__ = [
    "__version__",
    "Identified",
    "IdentifiedCollection",
    "Colony",
    "Plate",
    "PlateCollection",
    "ImageFile",
    "ImageCollection",
    "GrowthCurve",
]
