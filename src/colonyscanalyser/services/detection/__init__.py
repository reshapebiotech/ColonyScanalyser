"""Colony detection services."""

from datetime import timedelta
from typing import List, Optional

from numpy import ndarray
from skimage.measure import regionprops

from ...models import Timepoint


def timepoints_from_image(
    image_segmented: ndarray,
    timestamp: timedelta,
    image_original: Optional[ndarray] = None,
) -> List[Timepoint]:
    """
    Extract colony timepoints from a segmented image.

    Args:
        image_segmented: Segmented and labeled image
        timestamp: Time when image was taken
        image_original: Original color image for color analysis

    Returns:
        List of Timepoint objects
    """
    if image_original is not None:
        if image_original.shape[:2] != image_segmented.shape[:2]:
            raise ValueError("Original and segmented images must have same dimensions")

    timepoints = []

    for region in regionprops(image_segmented):
        color_average = _calculate_color_average(region, image_original)

        timepoint = Timepoint(
            timestamp=timestamp,
            area=region.area,
            center=region.centroid,
            diameter=region.equivalent_diameter,
            perimeter=region.perimeter,
            color_average=color_average,
            bbox=region.bbox,
            image=region.image,
            label=region.label,
            region_props=region,
        )

        timepoints.append(timepoint)

    return timepoints


def _calculate_color_average(
    region, image_original: Optional[ndarray]
) -> tuple[float, float, float]:
    """Calculate average color of a colony region."""
    if image_original is None:
        return (0.0, 0.0, 0.0)

    # Get the region from the original image
    region_image = image_original[region.slice]

    # Create a mask for the colony pixels
    mask = region.image

    # Extract color values where mask is True
    if len(region_image.shape) == 3:
        # Color image
        colony_pixels = region_image[mask]
        if colony_pixels.size > 0:
            # Calculate mean color, take only RGB channels
            return tuple(colony_pixels.mean(axis=0)[:3])

    return (0.0, 0.0, 0.0)
