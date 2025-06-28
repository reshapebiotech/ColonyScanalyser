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


def segment_plate_image(
    plate_image: ndarray,
    noise_mask: Optional[ndarray] = None,
    min_area: float = 2.0,
) -> ndarray:
    """
    Segment a plate image to identify colonies.

    Args:
        plate_image: Input plate image (grayscale or RGB)
        noise_mask: Optional noise mask to improve segmentation
        min_area: Minimum area for detected colonies

    Returns:
        Labeled image with each colony having a unique integer label
    """
    # Convert to grayscale if needed
    if len(plate_image.shape) == 3:
        from skimage.color import rgb2gray

        gray_image = rgb2gray(plate_image)
    else:
        gray_image = plate_image

    # Apply noise reduction if mask provided
    if noise_mask is not None:
        # Simple background subtraction using noise mask
        gray_image = remove_background(gray_image, sigma=1.0)

    # Create plate mask (circular)
    plate_mask = create_circular_mask(gray_image.shape)

    # Segment the image
    labeled = segment_image(
        gray_image,
        mask=plate_mask,
        min_area=int(min_area),
        blur_sigma=1.0,
        remove_border=True,
    )

    return labeled


def remove_background_mask(
    image: ndarray, smoothing: float = 1, sigmoid_cutoff: float = 0.4, **filter_args
) -> ndarray:
    """
    Separate the image foreground from the background.

    Returns a boolean mask of the image foreground.

    Args:
        image: An image as a numpy array
        smoothing: A sigma value for the gaussian filter
        sigmoid_cutoff: Cutoff for the sigmoid exposure function
        filter_args: Arguments to pass to the gaussian filter

    Returns:
        A boolean image mask of the foreground
    """
    from skimage import img_as_bool
    from skimage.exposure import adjust_sigmoid
    from skimage.filters import gaussian, threshold_triangle

    if image.size == 0:
        raise ValueError("The supplied image cannot be empty")

    image = image.astype("float64", copy=True)

    # Do not process the image if it is empty
    if not image.any():
        return img_as_bool(image)

    # Apply smoothing to reduce noise
    image = gaussian(image, smoothing, **filter_args)

    # Heighten contrast
    image = adjust_sigmoid(image, cutoff=sigmoid_cutoff, gain=10)

    # Find background threshold and return only foreground
    return image < threshold_triangle(image, nbins=10)
