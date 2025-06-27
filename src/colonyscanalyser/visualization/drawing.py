"""
Drawing utilities for ColonyScanalyser visualization.

This module provides functions for drawing colony masks, IDs, and other
visualizations on images without necessarily saving to disk.
"""

from datetime import timedelta
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from numpy import ndarray
from skimage.color import label2rgb

from ..models.colony import Colony
from ..models.plate import Plate


def draw_colony_masks(
    image: ndarray,
    colonies: List[Colony],
    timestamp: timedelta,
    alpha: float = 0.3,
    colormap=None,
) -> ndarray:
    """
    Draw colony masks overlaid on an image.

    :param image: the base image to draw on
    :param colonies: list of Colony objects to draw
    :param timestamp: the timestamp to find the appropriate timepoint
    :param alpha: transparency of the overlay (0.0 to 1.0)
    :param colormap: colors for colony overlay (None for automatic colors)
    :returns: image with colony masks overlaid
    """
    if len(colonies) == 0:
        return image.copy()

    # Create label image
    labels = np.zeros(image.shape[:2], dtype=int)

    for colony in colonies:
        # Find timepoint for this timestamp
        timepoint = colony.get_timepoint(timestamp)
        if timepoint is not None and timepoint.bbox is not None:
            min_row, min_col, max_row, max_col = timepoint.bbox
            if timepoint.image is not None:
                # Use the actual colony mask if available
                colony_mask = timepoint.image * colony.id
                labels[min_row:max_row, min_col:max_col] = np.maximum(
                    labels[min_row:max_row, min_col:max_col], colony_mask
                )

    # Convert to RGB overlay
    if len(labels[labels > 0]) > 0:
        overlay = label2rgb(labels, image, alpha=alpha, bg_label=0, colors=colormap)
        return (
            (overlay * 255).astype(np.uint8)
            if overlay.max() <= 1.0
            else overlay.astype(np.uint8)
        )
    else:
        return image.copy()


def draw_colony_ids(
    image: ndarray,
    colonies: List[Colony],
    timestamp: timedelta,
    font_size: int = 12,
    text_color: str = "white",
    bg_color: str = "black",
    show_centers: bool = True,
    center_color: str = "red",
    center_marker: str = "+",
) -> ndarray:
    """
    Draw colony IDs and optionally center markers on an image.

    :param image: the base image to draw on
    :param colonies: list of Colony objects to draw
    :param timestamp: the timestamp to find the appropriate timepoint
    :param font_size: size of the ID text
    :param text_color: color of the ID text
    :param bg_color: background color for the ID text
    :param show_centers: whether to show center markers
    :param center_color: color of the center markers
    :param center_marker: marker style for centers
    :returns: image with colony IDs drawn
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(colonies) == 0:
        return image.copy()

    # Create matplotlib figure matching image size
    height, width = image.shape[:2]
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

    for colony in colonies:
        # Find timepoint for this timestamp
        timepoint = colony.get_timepoint(timestamp)
        if timepoint is not None:
            y, x = timepoint.center  # Note: center is (row, col) format

            if show_centers:
                ax.text(
                    x,
                    y,
                    center_marker,
                    color=center_color,
                    fontsize=font_size,
                    ha="center",
                    va="center",
                    weight="bold",
                )

            # Draw colony ID
            radius = getattr(timepoint, "diameter", 20) / 2
            ax.text(
                x + radius * 0.7,
                y - radius * 0.9,
                str(colony.id),
                color=text_color,
                backgroundcolor=bg_color,
                fontsize=font_size * 0.8,
                ha="left",
                va="top",
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.7),
            )

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)

    return buf


def draw_colony_outlines(
    image: ndarray,
    colonies: List[Colony],
    timestamp: timedelta,
    outline_color: str = "red",
    line_width: float = 1.0,
    show_ids: bool = False,
) -> ndarray:
    """
    Draw colony outlines on an image.

    :param image: the base image to draw on
    :param colonies: list of Colony objects to draw
    :param timestamp: the timestamp to find the appropriate timepoint
    :param outline_color: color of the outline
    :param line_width: width of the outline
    :param show_ids: whether to show colony IDs
    :returns: image with colony outlines drawn
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(colonies) == 0:
        return image.copy()

    # Create matplotlib figure matching image size
    height, width = image.shape[:2]
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

    for colony in colonies:
        # Find timepoint for this timestamp
        timepoint = colony.get_timepoint(timestamp)
        if timepoint is not None:
            y, x = timepoint.center  # Note: center is (row, col) format
            radius = getattr(timepoint, "diameter", 20) / 2

            # Draw circle outline
            circle = Circle(
                (x, y),
                radius,
                fill=False,
                edgecolor=outline_color,
                linewidth=line_width,
                alpha=0.8,
            )
            ax.add_patch(circle)

            if show_ids:
                ax.text(
                    x + radius * 0.7,
                    y - radius * 0.9,
                    str(colony.id),
                    color="white",
                    backgroundcolor="black",
                    fontsize=8,
                    ha="left",
                    va="top",
                    alpha=0.85,
                )

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)

    return buf


def draw_plate_overlay(
    image: ndarray,
    plate: Plate,
    show_boundary: bool = True,
    show_detection_area: bool = True,
    show_label: bool = True,
    boundary_color: str = "purple",
    detection_color: str = "white",
    label_color: str = "white",
) -> ndarray:
    """
    Draw plate boundary and labels on an image.

    :param image: the base image to draw on
    :param plate: Plate object to draw
    :param show_boundary: whether to show the plate boundary
    :param show_detection_area: whether to show the colony detection area
    :param show_label: whether to show the plate label
    :param boundary_color: color of the plate boundary
    :param detection_color: color of the detection area boundary
    :param label_color: color of the plate label text
    :returns: image with plate overlay drawn
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    # Create matplotlib figure matching image size
    height, width = image.shape[:2]
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

    center_y, center_x = plate.center

    if show_boundary:
        # Draw detected plate boundary
        boundary_circle = Circle(
            (center_x, center_y),
            radius=plate.radius,
            fill=False,
            edgecolor=boundary_color,
            linewidth=2.5,
            linestyle="-",
            alpha=0.8,
        )
        ax.add_patch(boundary_circle)

    if show_detection_area:
        # Draw colony detection area
        detection_circle = Circle(
            (center_x, center_y),
            radius=plate.radius - plate.edge_cut,
            fill=False,
            edgecolor=detection_color,
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(detection_circle)

    if show_label:
        # Draw plate label
        ax.text(
            center_x,
            center_y - plate.radius - (plate.edge_cut * 1.4),
            f"Plate #{plate.id}".upper(),
            color=label_color,
            backgroundcolor="black",
            fontsize=16,
            ha="center",
            va="center",
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
        )

        if len(plate.name) > 0:
            ax.text(
                center_x,
                center_y - plate.radius - (plate.edge_cut * 0.6),
                plate.name,
                color=label_color,
                backgroundcolor="black",
                fontsize=12,
                ha="center",
                va="center",
                alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)

    return buf


def create_colony_visualization(
    image: ndarray,
    plate: Plate,
    timestamp: timedelta,
    show_masks: bool = True,
    show_ids: bool = True,
    show_outlines: bool = False,
    show_plate: bool = True,
    mask_alpha: float = 0.3,
) -> ndarray:
    """
    Create a comprehensive colony visualization combining multiple overlays.

    :param image: the base image to draw on
    :param plate: Plate object containing colonies
    :param timestamp: the timestamp to visualize
    :param show_masks: whether to show colony masks
    :param show_ids: whether to show colony IDs
    :param show_outlines: whether to show colony outlines
    :param show_plate: whether to show plate boundaries
    :param mask_alpha: transparency of colony masks
    :returns: image with all requested overlays
    """
    result = image.copy()

    if show_plate:
        result = draw_plate_overlay(result, plate)

    if len(plate.colonies) > 0:
        if show_masks:
            result = draw_colony_masks(
                result, plate.colonies, timestamp, alpha=mask_alpha
            )

        if show_outlines:
            result = draw_colony_outlines(
                result, plate.colonies, timestamp, show_ids=False
            )

        if show_ids:
            result = draw_colony_ids(result, plate.colonies, timestamp)

    return result


def save_colony_visualizations(
    plates: "PlateCollection",
    image_file: "ImageFile",
    output_dir: "Path",
    save_masks: bool = True,
    save_ids: bool = True,
    save_outlines: bool = False,
    save_comprehensive: bool = True,
    use_full_image: bool = True,
    save_plate_only: bool = False,
) -> List["Path"]:
    """
    Save colony visualization images for all plates at a specific timepoint.

    :param plates: PlateCollection containing tracked colonies
    :param image_file: ImageFile for the timepoint to visualize
    :param output_dir: Directory to save visualization images
    :param save_masks: Whether to save colony mask overlays
    :param save_ids: Whether to save colony ID overlays
    :param save_outlines: Whether to save colony outline overlays
    :param save_comprehensive: Whether to save comprehensive visualization
    :param use_full_image: Whether to use the full image (True) or crop to plates (False)
    :param save_plate_only: Whether to also save plate-only versions
    :returns: List of saved file paths
    """
    from pathlib import Path

    from skimage.io import imsave

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []

    with image_file as img:
        base_name = image_file.file_path.stem

        # Create visualizations with full image if requested
        if use_full_image:
            # Create separate full image visualizations for each plate
            for plate in plates.items:
                if len(plate.items) == 0:
                    continue

                plate_suffix = f"_plate{plate.id}_full"

                if save_ids:
                    # Create full image with single plate overlay and colony IDs
                    full_ids_overlay = draw_plate_overlay(
                        img.image, plate, show_label=True, show_boundary=True
                    )
                    full_ids_overlay = draw_colony_ids_on_full_image(
                        full_ids_overlay, plate, img.timestamp_elapsed
                    )
                    full_ids_file = output_dir / f"{base_name}{plate_suffix}_ids.png"
                    imsave(str(full_ids_file), full_ids_overlay)
                    saved_files.append(full_ids_file)

                if save_comprehensive:
                    # Create full image with single plate overlay and colony outlines
                    full_comp_overlay = draw_plate_overlay(
                        img.image, plate, show_label=True, show_boundary=True
                    )
                    full_comp_overlay = draw_colony_outlines_on_full_image(
                        full_comp_overlay, plate, img.timestamp_elapsed, show_ids=True
                    )
                    full_comp_file = (
                        output_dir / f"{base_name}{plate_suffix}_comprehensive.png"
                    )
                    imsave(str(full_comp_file), full_comp_overlay)
                    saved_files.append(full_comp_file)

        # Create plate-only visualizations if requested
        if save_plate_only or not use_full_image:
            for plate in plates.items:
                if len(plate.items) == 0:
                    continue

                # Extract plate image
                plate_image = plate.slice_plate_image(img.image)
                plate_suffix = f"_plate{plate.id}" + (
                    "_cropped" if use_full_image else ""
                )

                if save_masks:
                    mask_overlay = draw_colony_masks(
                        plate_image, plate.items, img.timestamp_elapsed, alpha=0.4
                    )
                    mask_file = output_dir / f"{base_name}{plate_suffix}_masks.png"
                    imsave(str(mask_file), mask_overlay)
                    saved_files.append(mask_file)

                if save_ids:
                    ids_overlay = draw_colony_ids(
                        plate_image, plate.items, img.timestamp_elapsed, font_size=10
                    )
                    ids_file = output_dir / f"{base_name}{plate_suffix}_ids.png"
                    imsave(str(ids_file), ids_overlay)
                    saved_files.append(ids_file)

                if save_outlines:
                    outlines_overlay = draw_colony_outlines(
                        plate_image, plate.items, img.timestamp_elapsed, show_ids=True
                    )
                    outlines_file = (
                        output_dir / f"{base_name}{plate_suffix}_outlines.png"
                    )
                    imsave(str(outlines_file), outlines_overlay)
                    saved_files.append(outlines_file)

                if save_comprehensive:
                    comprehensive = create_colony_visualization(
                        plate_image,
                        plate,
                        img.timestamp_elapsed,
                        show_masks=False,
                        show_ids=True,
                        show_outlines=True,
                        show_plate=False,  # Already cropped to plate
                    )
                    comp_file = (
                        output_dir / f"{base_name}{plate_suffix}_comprehensive.png"
                    )
                    imsave(str(comp_file), comprehensive)
                    saved_files.append(comp_file)

    return saved_files


def draw_colony_ids_on_full_image(
    image: ndarray,
    plate: Plate,
    timestamp: timedelta,
) -> ndarray:
    """
    Draw colony IDs on full image, adjusting coordinates for plate position.

    :param image: the full image to draw on
    :param plate: Plate object containing colonies
    :param timestamp: the timestamp to visualize
    :returns: image with colony IDs drawn
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(plate.items) == 0:
        return image

    # Create matplotlib figure matching image size
    height, width = image.shape[:2]
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

    # Calculate offset to convert plate-relative coordinates to full image coordinates
    center_y, center_x = plate.center
    offset_y = center_y - plate.radius + plate.edge_cut
    offset_x = center_x - plate.radius + plate.edge_cut

    for colony in plate.items:
        # Find timepoint for this timestamp
        timepoint = colony.get_timepoint(timestamp)
        if timepoint is not None:
            # Convert plate-relative coordinates to full image coordinates
            rel_y, rel_x = timepoint.center  # Colony center relative to plate image
            full_x = offset_x + rel_x
            full_y = offset_y + rel_y

            # Draw center marker
            ax.text(
                full_x,
                full_y,
                "+",
                color="red",
                fontsize=8,
                ha="center",
                va="center",
                weight="bold",
            )

            # Draw colony ID
            radius = getattr(timepoint, "diameter", 20) / 2
            ax.text(
                full_x + radius * 0.7,
                full_y - radius * 0.9,
                str(colony.id),
                color="white",
                backgroundcolor="black",
                fontsize=8,
                ha="left",
                va="top",
                alpha=0.85,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)

    return buf


def draw_colony_outlines_on_full_image(
    image: ndarray,
    plate: Plate,
    timestamp: timedelta,
    show_ids: bool = True,
) -> ndarray:
    """
    Draw colony outlines on full image, adjusting coordinates for plate position.

    :param image: the full image to draw on
    :param plate: Plate object containing colonies
    :param timestamp: the timestamp to visualize
    :param show_ids: whether to show colony IDs
    :returns: image with colony outlines drawn
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(plate.items) == 0:
        return image

    # Create matplotlib figure matching image size
    height, width = image.shape[:2]
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

    # Calculate offset to convert plate-relative coordinates to full image coordinates
    center_y, center_x = plate.center
    offset_y = center_y - plate.radius + plate.edge_cut
    offset_x = center_x - plate.radius + plate.edge_cut

    for colony in plate.items:
        # Find timepoint for this timestamp
        timepoint = colony.get_timepoint(timestamp)
        if timepoint is not None:
            # Convert plate-relative coordinates to full image coordinates
            rel_y, rel_x = timepoint.center  # Colony center relative to plate image
            full_x = offset_x + rel_x
            full_y = offset_y + rel_y
            radius = getattr(timepoint, "diameter", 20) / 2

            # Draw circle outline
            circle = Circle(
                (full_x, full_y),
                radius,
                fill=False,
                edgecolor="red",
                linewidth=1.0,
                alpha=0.8,
            )
            ax.add_patch(circle)

            if show_ids:
                ax.text(
                    full_x + radius * 0.7,
                    full_y - radius * 0.9,
                    str(colony.id),
                    color="white",
                    backgroundcolor="black",
                    fontsize=6,
                    ha="left",
                    va="top",
                    alpha=0.85,
                )

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)

    return buf


def draw_all_plates_overlay(
    image: ndarray,
    plates: List[Plate],
) -> ndarray:
    """
    Draw overlays for all plates on the full image.

    :param image: the full image to draw on
    :param plates: list of Plate objects to draw
    :returns: image with all plate overlays drawn
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if len(plates) == 0:
        return image.copy()

    # Create matplotlib figure matching image size
    height, width = image.shape[:2]
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert y-axis to match image coordinates

    for plate in plates:
        center_y, center_x = plate.center

        # Draw detected plate boundary
        boundary_circle = Circle(
            (center_x, center_y),
            radius=plate.radius,
            fill=False,
            edgecolor="purple",
            linewidth=2.5,
            linestyle="-",
            alpha=0.8,
        )
        ax.add_patch(boundary_circle)

        # Draw colony detection area
        detection_circle = Circle(
            (center_x, center_y),
            radius=plate.radius - plate.edge_cut,
            fill=False,
            edgecolor="white",
            linewidth=1.5,
            linestyle="--",
            alpha=0.8,
        )
        ax.add_patch(detection_circle)

        # Draw plate label
        ax.text(
            center_x,
            center_y - plate.radius - (plate.edge_cut * 1.4),
            f"Plate #{plate.id}".upper(),
            color="white",
            backgroundcolor="black",
            fontsize=16,
            ha="center",
            va="center",
            alpha=0.9,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7),
        )

        if len(plate.name) > 0:
            ax.text(
                center_x,
                center_y - plate.radius - (plate.edge_cut * 0.6),
                plate.name,
                color="white",
                backgroundcolor="black",
                fontsize=12,
                ha="center",
                va="center",
                alpha=0.9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            )

    # Convert figure to numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Convert RGBA to RGB
    plt.close(fig)

    return buf
