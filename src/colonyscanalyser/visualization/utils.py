"""
Visualization utilities for ColonyScanalyser.

This module provides utility functions for visualization operations,
including coordinate transformations, color management, and matplotlib helpers.
"""

from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from numpy import ndarray


def rc_to_xy(coordinate: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert row,column coordinate to x,y coordinate.

    :param coordinate: (row, column) coordinate tuple
    :returns: (x, y) coordinate tuple
    """
    return (coordinate[1], coordinate[0])


def xy_to_rc(coordinate: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert x,y coordinate to row,column coordinate.

    :param coordinate: (x, y) coordinate tuple
    :returns: (row, column) coordinate tuple
    """
    return (coordinate[1], coordinate[0])


def normalize_image(image: ndarray) -> ndarray:
    """
    Normalize image values to 0-255 uint8 range.

    :param image: Input image array
    :returns: Normalized image as uint8
    """
    if image.dtype == np.float64 or image.dtype == np.float32:
        # Handle float images that may be in 0-1 range
        if image.max() <= 1.0:
            image = image * 255

    # Clip to valid range and convert to uint8
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def ensure_rgb(image: ndarray) -> ndarray:
    """
    Ensure image is in RGB format.

    :param image: Input image (grayscale or RGB)
    :returns: RGB image
    """
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        return np.stack([image, image, image], axis=2)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert RGBA to RGB
        return image[:, :, :3]
    elif len(image.shape) == 3 and image.shape[2] == 3:
        # Already RGB
        return image
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def create_colormap_colors(n_colors: int, colormap_name: str = "tab20") -> List[str]:
    """
    Generate a list of colors from a matplotlib colormap.

    :param n_colors: Number of colors needed
    :param colormap_name: Name of matplotlib colormap
    :returns: List of color strings
    """
    import matplotlib.cm as cm

    colormap = cm.get_cmap(colormap_name)
    colors = []

    for i in range(n_colors):
        color = colormap(i / max(1, n_colors - 1))
        # Convert to hex string
        colors.append(
            f"#{int(color[0] * 255):02x}{int(color[1] * 255):02x}{int(color[2] * 255):02x}"
        )

    return colors


def label_bars(ax: Axes, bars: BarContainer, text_format: str, **kwargs):
    """
    Attach labels to bars in a bar chart.

    :param ax: Matplotlib Axes object
    :param bars: Bar container from bar plot
    :param text_format: Format string for bar labels
    :param kwargs: Additional arguments for ax.text
    """
    ys = [bar.get_y() for bar in bars]
    vertical = all(y == ys[0] for y in ys)

    if vertical:
        _label_bar_vertical(ax, bars, text_format, **kwargs)
    else:
        _label_bar_horizontal(ax, bars, text_format, **kwargs)


def _label_bar_vertical(ax: Axes, bars: BarContainer, text_format: str, **kwargs):
    """Label vertical bars."""
    max_y_value = ax.get_ylim()[1]
    inside_distance = max_y_value * 0.05
    outside_distance = max_y_value * 0.01

    for bar in bars:
        text = text_format.format(bar.get_height())
        text_x = bar.get_x() + bar.get_width() / 2

        is_inside = bar.get_height() >= max_y_value * 0.15
        if is_inside:
            color = "white"
            text_y = bar.get_height() - inside_distance
        else:
            color = "black"
            text_y = bar.get_height() + outside_distance

        ax.text(text_x, text_y, text, ha="center", va="bottom", color=color, **kwargs)


def _label_bar_horizontal(ax: Axes, bars: BarContainer, text_format: str, **kwargs):
    """Label horizontal bars."""
    max_x_value = ax.get_xlim()[1]
    distance = max_x_value * 0.0025

    for bar in bars:
        text = text_format.format(bar.get_width())
        text_x = bar.get_width() + distance
        text_y = bar.get_y() + bar.get_height() / 2
        ax.text(text_x, text_y, text, va="center", **kwargs)


def axis_minutes_to_hours(labels: Union[List[int], List[float]]) -> List[str]:
    """
    Convert axis labels from minutes to hours.

    :param labels: List of time labels in minutes
    :returns: List of formatted hour labels
    """
    return [f"{x // 60:.0f}" for x in labels]


def axis_seconds_to_hours(labels: Union[List[int], List[float]]) -> List[str]:
    """
    Convert axis labels from seconds to hours.

    :param labels: List of time labels in seconds
    :returns: List of formatted hour labels
    """
    return [f"{x / 3600:.1f}" for x in labels]


def setup_matplotlib_defaults():
    """Set up default matplotlib parameters for consistent styling."""
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 1.5,
            "lines.markersize": 6,
        }
    )


def create_figure_with_size(width: float, height: float, dpi: int = 100) -> Tuple:
    """
    Create matplotlib figure with specific size.

    :param width: Width in inches
    :param height: Height in inches
    :param dpi: Dots per inch
    :returns: (figure, axes) tuple
    """
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    return fig, ax


def cleanup_matplotlib_figure(fig=None):
    """
    Clean up matplotlib figure to prevent memory leaks.

    :param fig: Figure to clean up (None for current figure)
    """
    if fig is not None:
        plt.close(fig)
    else:
        plt.close()


def calculate_grid_layout(n_items: int, max_cols: int = 4) -> Tuple[int, int]:
    """
    Calculate optimal grid layout for n items.

    :param n_items: Number of items to arrange
    :param max_cols: Maximum number of columns
    :returns: (rows, cols) tuple
    """
    if n_items <= max_cols:
        return 1, n_items

    cols = min(n_items, max_cols)
    rows = (n_items + cols - 1) // cols  # Ceiling division
    return rows, cols


def apply_image_mask(image: ndarray, mask: ndarray, alpha: float = 0.5) -> ndarray:
    """
    Apply a mask overlay to an image.

    :param image: Base image
    :param mask: Boolean mask or labeled image
    :param alpha: Transparency of mask overlay
    :returns: Image with mask applied
    """
    if mask.dtype == bool:
        # Simple boolean mask
        overlay = np.zeros_like(image)
        overlay[mask] = [255, 0, 0]  # Red overlay
    else:
        # Labeled mask - use different colors
        from skimage.color import label2rgb

        overlay = label2rgb(mask, image, alpha=alpha, bg_label=0)
        return (
            (overlay * 255).astype(np.uint8)
            if overlay.max() <= 1.0
            else overlay.astype(np.uint8)
        )

    # Blend with original image
    result = image.copy().astype(float)
    result = (1 - alpha) * result + alpha * overlay
    return np.clip(result, 0, 255).astype(np.uint8)


def pad_image_to_square(image: ndarray, pad_value: int = 0) -> ndarray:
    """
    Pad image to make it square.

    :param image: Input image
    :param pad_value: Value to use for padding
    :returns: Square image
    """
    height, width = image.shape[:2]
    max_dim = max(height, width)

    pad_h = (max_dim - height) // 2
    pad_w = (max_dim - width) // 2

    if len(image.shape) == 3:
        padding = (
            (pad_h, max_dim - height - pad_h),
            (pad_w, max_dim - width - pad_w),
            (0, 0),
        )
    else:
        padding = ((pad_h, max_dim - height - pad_h), (pad_w, max_dim - width - pad_w))

    return np.pad(image, padding, mode="constant", constant_values=pad_value)


def resize_image_proportional(image: ndarray, max_size: Tuple[int, int]) -> ndarray:
    """
    Resize image while maintaining aspect ratio.

    :param image: Input image
    :param max_size: Maximum (width, height)
    :returns: Resized image
    """
    from skimage.transform import resize

    height, width = image.shape[:2]
    max_width, max_height = max_size

    # Calculate scaling factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale

    if scale < 1.0:
        new_height = int(height * scale)
        new_width = int(width * scale)
        return resize(
            image, (new_height, new_width), preserve_range=True, anti_aliasing=True
        ).astype(np.uint8)

    return image


def create_progress_callback(total_items: int, description: str = "Processing"):
    """
    Create a simple progress callback function.

    :param total_items: Total number of items to process
    :param description: Description for progress display
    :returns: Callback function
    """

    def callback(current_item: int):
        if total_items > 0:
            percent = (current_item / total_items) * 100
            print(
                f"\r{description}: {percent:.1f}% ({current_item}/{total_items})",
                end="",
                flush=True,
            )
        if current_item >= total_items:
            print()  # New line when complete

    return callback
