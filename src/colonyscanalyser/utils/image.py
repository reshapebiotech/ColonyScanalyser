"""Image processing utilities for ColonyScanalyser."""

from typing import Optional, Tuple, Union

import numpy as np
from numpy import ndarray


def ensure_rgb(image: ndarray) -> ndarray:
    """
    Convert an image in any color mode to RGB.

    Args:
        image: Input image as numpy array

    Returns:
        Image converted to RGB format
    """
    from skimage.color import gray2rgb, rgba2rgb

    # If the image has no color channels, it must be grayscale
    if len(image.shape) < 3 or image.shape[-1] == 1:
        return gray2rgb(image.squeeze())

    # If the image has 4 channels, it must be RGBA
    elif image.shape[-1] == 4:
        return rgba2rgb(image)

    # If the image has 3 channels, it's already RGB
    elif image.shape[-1] == 3:
        return image

    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def image_as_rgb(image: ndarray) -> ndarray:
    """
    Convert an image in any color mode to RGB (legacy alias).

    Args:
        image: Input image as numpy array

    Returns:
        Image converted to RGB format
    """
    return ensure_rgb(image)


def crop_image(
    image: ndarray,
    crop_shape: Union[Tuple[int, int], Tuple[int, int, int]],
    center: Optional[Tuple[int, int]] = None,
) -> ndarray:
    """
    Get a subsection of an image.

    Optionally specify a center point to crop around.

    Args:
        image: Input image as numpy array
        crop_shape: Desired crop dimensions (height, width) or (height, width, channels)
        center: Center point to crop around (y, x). If None, uses image center

    Returns:
        Cropped image
    """
    if len(crop_shape) == 2:
        crop_h, crop_w = crop_shape
    else:
        crop_h, crop_w = crop_shape[:2]

    img_h, img_w = image.shape[:2]

    # Use image center if no center specified
    if center is None:
        center_y, center_x = img_h // 2, img_w // 2
    else:
        center_y, center_x = center

    # Calculate crop boundaries
    half_h, half_w = crop_h // 2, crop_w // 2

    y_start = max(0, center_y - half_h)
    y_end = min(img_h, center_y + half_h)
    x_start = max(0, center_x - half_w)
    x_end = min(img_w, center_x + half_w)

    # Ensure we don't exceed crop dimensions
    actual_h = y_end - y_start
    actual_w = x_end - x_start

    if actual_h < crop_h:
        # Adjust if we're at image boundary
        if y_start == 0:
            y_end = min(img_h, y_start + crop_h)
        else:
            y_start = max(0, y_end - crop_h)

    if actual_w < crop_w:
        # Adjust if we're at image boundary
        if x_start == 0:
            x_end = min(img_w, x_start + crop_w)
        else:
            x_start = max(0, x_end - crop_w)

    return image[y_start:y_end, x_start:x_end]


def normalize_image(image: ndarray) -> ndarray:
    """
    Normalize image values to 0-255 uint8 range.

    Args:
        image: Input image array

    Returns:
        Normalized image as uint8
    """
    if image.dtype == np.float64 or image.dtype == np.float32:
        # Handle float images that may be in 0-1 range
        if image.max() <= 1.0:
            image = image * 255

    # Clip to valid range and convert to uint8
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def resize_image(
    image: ndarray,
    output_shape: Tuple[int, int],
    preserve_range: bool = True,
    anti_aliasing: bool = True,
) -> ndarray:
    """
    Resize image to specified dimensions.

    Args:
        image: Input image
        output_shape: Desired output shape (height, width)
        preserve_range: Whether to preserve the original data range
        anti_aliasing: Whether to apply anti-aliasing

    Returns:
        Resized image
    """
    from skimage.transform import resize

    return resize(
        image,
        output_shape,
        preserve_range=preserve_range,
        anti_aliasing=anti_aliasing,
    ).astype(image.dtype)


def pad_image_to_size(
    image: ndarray,
    target_shape: Tuple[int, int],
    mode: str = "constant",
    constant_values: Union[int, float] = 0,
) -> ndarray:
    """
    Pad image to reach target shape.

    Args:
        image: Input image
        target_shape: Target shape (height, width)
        mode: Padding mode ('constant', 'edge', 'reflect', etc.)
        constant_values: Value to use for constant padding

    Returns:
        Padded image
    """
    current_h, current_w = image.shape[:2]
    target_h, target_w = target_shape

    if current_h >= target_h and current_w >= target_w:
        return image

    pad_h = max(0, target_h - current_h)
    pad_w = max(0, target_w - current_w)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if len(image.shape) == 3:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    else:
        padding = ((pad_top, pad_bottom), (pad_left, pad_right))

    return np.pad(image, padding, mode=mode, constant_values=constant_values)


def apply_circular_mask(
    image: ndarray,
    center: Optional[Tuple[float, float]] = None,
    radius: Optional[float] = None,
    background_value: Union[int, float] = 0,
) -> ndarray:
    """
    Apply a circular mask to an image.

    Args:
        image: Input image
        center: Center of circle (y, x). If None, uses image center
        radius: Radius of circle. If None, uses 80% of min dimension
        background_value: Value to use outside the circle

    Returns:
        Masked image
    """
    height, width = image.shape[:2]

    if center is None:
        center = (height // 2, width // 2)

    if radius is None:
        radius = min(height, width) * 0.4

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Create circular mask
    mask = ((x - center[1]) ** 2 + (y - center[0]) ** 2) <= radius**2

    # Apply mask
    result = image.copy()
    if len(image.shape) == 3:
        for channel in range(image.shape[2]):
            result[~mask, channel] = background_value
    else:
        result[~mask] = background_value

    return result


def enhance_contrast(
    image: ndarray,
    clip_limit: float = 0.01,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> ndarray:
    """
    Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Input image
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of the grid for local enhancement

    Returns:
        Contrast-enhanced image
    """
    from skimage.exposure import equalize_adapthist

    # Apply CLAHE
    if len(image.shape) == 3:
        # For color images, apply to each channel
        enhanced = np.zeros_like(image)
        for i in range(image.shape[2]):
            enhanced[:, :, i] = equalize_adapthist(
                image[:, :, i], clip_limit=clip_limit, nbins=256
            )
        return enhanced
    else:
        # For grayscale images
        return equalize_adapthist(image, clip_limit=clip_limit, nbins=256)


def create_thumbnail(
    image: ndarray,
    max_size: int = 256,
    maintain_aspect: bool = True,
) -> ndarray:
    """
    Create a thumbnail of the image.

    Args:
        image: Input image
        max_size: Maximum dimension of thumbnail
        maintain_aspect: Whether to maintain aspect ratio

    Returns:
        Thumbnail image
    """
    height, width = image.shape[:2]

    if maintain_aspect:
        # Calculate scaling factor
        scale = min(max_size / height, max_size / width)
        new_height = int(height * scale)
        new_width = int(width * scale)
    else:
        new_height = new_width = max_size

    return resize_image(image, (new_height, new_width))
