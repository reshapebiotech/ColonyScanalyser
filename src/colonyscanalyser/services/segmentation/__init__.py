"""Image segmentation services."""

from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border


def segment_image(
    image: ndarray,
    mask: Optional[ndarray] = None,
    min_area: int = 50,
    blur_sigma: float = 1.0,
    remove_border: bool = True,
) -> ndarray:
    """
    Segment an image to identify individual colonies.

    Args:
        image: Input grayscale image
        mask: Optional binary mask to restrict segmentation area
        min_area: Minimum area for detected objects (smaller ones removed)
        blur_sigma: Gaussian blur sigma for noise reduction
        remove_border: Whether to remove objects touching image border

    Returns:
        Labeled image with each colony having a unique integer label
    """
    # Apply Gaussian blur to reduce noise
    if blur_sigma > 0:
        image = gaussian(image, sigma=blur_sigma)

    # Apply mask if provided
    if mask is not None:
        image = image * mask

    # Threshold to create binary image
    threshold = threshold_otsu(image)
    binary = image > threshold

    # Remove small objects (noise)
    if min_area > 0:
        binary = remove_small_objects(binary, min_size=min_area)

    # Remove objects touching border
    if remove_border:
        binary = clear_border(binary)

    # Label connected components
    labeled = label(binary)

    return labeled


def create_circular_mask(
    image_shape: Tuple[int, int],
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
) -> ndarray:
    """
    Create a circular mask for plate detection.

    Args:
        image_shape: Shape of the image (height, width)
        center: Center of the circle (y, x). If None, uses image center
        radius: Radius of the circle. If None, uses 80% of min(height, width)/2

    Returns:
        Binary mask with circular region as True
    """
    height, width = image_shape

    if center is None:
        center = (height // 2, width // 2)

    if radius is None:
        radius = min(height, width) * 0.4  # 80% of half the smallest dimension

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Create circular mask
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) <= radius**2

    return mask


def remove_background(
    image: ndarray,
    sigma: float = 1.0,
    percentile: float = 10.0,
) -> ndarray:
    """
    Remove background from image using simple background subtraction.

    Args:
        image: Input image
        sigma: Gaussian blur sigma for background estimation
        percentile: Percentile for background intensity estimation

    Returns:
        Background-subtracted image
    """
    # Estimate background by heavy blurring
    background = gaussian(image, sigma=sigma * 10)

    # Subtract background
    result = image.astype(float) - background.astype(float)

    # Clip negative values
    result = np.clip(result, 0, None)

    return result.astype(image.dtype)
