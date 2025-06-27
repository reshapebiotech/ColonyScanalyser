"""
Visualization package for ColonyScanalyser.

This package provides plotting and visualization functionality for the
ColonyScanalyser tool, including growth curve plots, colony maps,
plate visualizations, and animations.
"""

from .drawing import (
    create_colony_visualization,
    draw_colony_ids,
    draw_colony_masks,
    draw_colony_outlines,
    draw_plate_overlay,
    save_colony_visualizations,
)
from .export import (
    create_visualization_filename,
    save_image,
    save_plot,
    save_visualization_series,
)
from .plots import (
    plot_appearance_frequency,
    plot_colony_map,
    plot_doubling_map,
    plot_growth_curve,
    plot_plate_images_animation,
)
from .plotting import rc_to_xy
from .utils import (
    axis_minutes_to_hours,
    label_bars,
    normalize_image,
    setup_matplotlib_defaults,
)

__all__ = [
    # Main plotting functions
    "plot_appearance_frequency",
    "plot_colony_map",
    "plot_doubling_map",
    "plot_growth_curve",
    "plot_plate_images_animation",
    # Drawing utilities
    "create_colony_visualization",
    "draw_colony_ids",
    "draw_colony_masks",
    "draw_colony_outlines",
    "draw_plate_overlay",
    "save_colony_visualizations",
    # Export utilities
    "create_visualization_filename",
    "save_image",
    "save_plot",
    "save_visualization_series",
    # Plotting utilities
    "rc_to_xy",
    # Visualization utilities
    "axis_minutes_to_hours",
    "label_bars",
    "normalize_image",
    "setup_matplotlib_defaults",
]
