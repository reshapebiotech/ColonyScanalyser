"""
Services for colony analysis algorithms.

This package contains focused, single-responsibility services for:
- Detection: Extract colony data from images
- Tracking: Group and filter colonies over time
- Segmentation: Image segmentation algorithms
- Alignment: Image alignment algorithms
"""

from .alignment import (
    align_image_simple,
    align_images_features,
    align_images_fft,
    calculate_alignment_quality,
)
from .detection import timepoints_from_image
from .tracking import (
    create_colonies_from_timepoints,
    filter_colonies,
    group_timepoints_by_center,
)

__all__ = [
    # Alignment services
    "align_image_simple",
    "align_images_fft",
    "align_images_features",
    "calculate_alignment_quality",
    # Detection services
    "timepoints_from_image",
    # Tracking services
    "filter_colonies",
    "create_colonies_from_timepoints",
    "group_timepoints_by_center",
]
