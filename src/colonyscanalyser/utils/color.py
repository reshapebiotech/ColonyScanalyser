"""Color utilities for ColonyScanalyser."""

from typing import Tuple, Union


def rgb_to_name(
    color_rgb: Union[Tuple[int, int, int], Tuple[float, float, float]],
    color_spec: str = "css3",
) -> str:
    """
    Convert an RGB tuple to the closest named web colour.

    For a full list of colours, see: https://www.w3.org/TR/css-color-3/

    Args:
        color_rgb: A red, green, blue colour value tuple
        color_spec: A color spec from the webcolors module

    Returns:
        A named colour string
    """
    try:
        import webcolors
    except ImportError:
        # Fallback if webcolors not available
        return f"rgb({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]})"

    # Ensure RGB values are integers
    if isinstance(color_rgb[0], float):
        color_rgb = tuple(int(c * 255) if c <= 1.0 else int(c) for c in color_rgb)
    else:
        color_rgb = tuple(int(c) for c in color_rgb)

    # Default to CSS3 color spec if none specified
    color_spec = color_spec.lower()

    try:
        # Try to get exact match first
        return webcolors.rgb_to_name(color_rgb, spec=color_spec)
    except ValueError:
        # If no exact match, find closest color
        return _find_closest_color(color_rgb, color_spec)


def _find_closest_color(rgb: Tuple[int, int, int], color_spec: str = "css3") -> str:
    """
    Find the closest named color to the given RGB value.

    Args:
        rgb: RGB tuple
        color_spec: Color specification to use

    Returns:
        Name of the closest color
    """
    try:
        import webcolors
    except ImportError:
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

    min_distance = float("inf")
    closest_name = "black"

    # Get all colors for the specified spec
    try:
        if color_spec == "css3":
            color_dict = webcolors.CSS3_HEX_TO_NAMES
        elif color_spec == "css21":
            color_dict = webcolors.CSS21_HEX_TO_NAMES
        elif color_spec == "html4":
            color_dict = webcolors.HTML4_HEX_TO_NAMES
        else:
            color_dict = webcolors.CSS3_HEX_TO_NAMES
    except AttributeError:
        # Fallback for different webcolors versions
        return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

    # Calculate distance to each color
    for hex_color, name in color_dict.items():
        try:
            color_rgb = webcolors.hex_to_rgb(hex_color)
            distance = _calculate_color_distance(rgb, color_rgb)
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        except ValueError:
            continue

    return closest_name


def _calculate_color_distance(
    color1: Tuple[int, int, int], color2: Tuple[int, int, int]
) -> float:
    """
    Calculate Euclidean distance between two RGB colors.

    Args:
        color1: First RGB color
        color2: Second RGB color

    Returns:
        Distance between colors
    """
    return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) ** 0.5


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """
    Convert RGB tuple to hex color string.

    Args:
        rgb: RGB color tuple

    Returns:
        Hex color string (e.g., "#FF0000")
    """
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert hex color string to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF0000" or "FF0000")

    Returns:
        RGB color tuple
    """
    # Remove # if present
    hex_color = hex_color.lstrip("#")

    # Validate length
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")

    try:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        raise ValueError(f"Invalid hex color: {hex_color}")


def normalize_rgb(rgb: Tuple[Union[int, float], ...]) -> Tuple[float, float, float]:
    """
    Normalize RGB values to 0-1 range.

    Args:
        rgb: RGB tuple with values in any range

    Returns:
        RGB tuple with values in 0-1 range
    """
    # If values are already in 0-1 range, return as is
    if all(0 <= c <= 1 for c in rgb):
        return tuple(float(c) for c in rgb)

    # Otherwise assume 0-255 range and normalize
    return tuple(c / 255.0 for c in rgb)
