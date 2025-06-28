from abc import ABC, abstractmethod
from math import pi
from typing import Tuple, Union


class Shape(ABC):
    """
    An abstract class to provide the fundamental properties of a surface
    """

    @property
    @abstractmethod
    def area(self) -> float:
        raise NotImplementedError(
            "This property must be implemented in a derived class"
        )

    @property
    def center(self) -> Union[Tuple[float, float], Tuple[float, float, float]]:
        try:
            return self._center
        except AttributeError:
            return None

    @center.setter
    def center(self, val: Union[Tuple[float, float], Tuple[float, float, float]]):
        self._center = val

    @property
    def depth(self) -> float:
        try:
            return self._depth
        except AttributeError:
            return 0

    @depth.setter
    def depth(self, val: float):
        self._depth = val

    @property
    def height(self) -> float:
        try:
            return self._height
        except AttributeError:
            return 0

    @height.setter
    def height(self, val: float):
        self._height = val

    @property
    @abstractmethod
    def perimeter(self) -> float:
        raise NotImplementedError(
            "This property must be implemented in a derived class"
        )

    @property
    def width(self) -> float:
        try:
            return self._width
        except AttributeError:
            return 0

    @width.setter
    def width(self, val: float):
        self._width = val


class Circle(Shape):
    """
    An object to generate the properties of a circle
    """

    def __init__(self, diameter: float):
        self.diameter = diameter

    @property
    def area(self) -> float:
        return pi * self.radius * self.radius

    @property
    def circumference(self) -> float:
        return self.perimeter

    @property
    def diameter(self) -> float:
        return self._diameter

    @diameter.setter
    def diameter(self, val: float):
        if val < 0:
            raise ValueError("The diameter must be a number greater than zero")

        self._diameter = val

    @property
    def height(self) -> float:
        return self.diameter

    @property
    def perimeter(self) -> float:
        return pi * self.diameter

    @property
    def radius(self) -> float:
        return self.diameter / 2


def crop_circle_from_image(
    image, center: Tuple[float, float], radius: float, background_color: Tuple = (0,)
):
    """
    Extract circular region from image.

    Args:
        image: Input image as numpy array
        center: Center coordinates (x, y)
        radius: Radius of circle to extract
        background_color: Color for areas outside the circle

    Returns:
        Cropped circular image
    """
    import numpy as np

    # Get image dimensions
    height, width = image.shape[:2]

    # Calculate bounding box
    cx, cy = center
    x_min = max(0, int(cx - radius))
    x_max = min(width, int(cx + radius))
    y_min = max(0, int(cy - radius))
    y_max = min(height, int(cy + radius))

    # Create cropped image
    cropped_height = y_max - y_min
    cropped_width = x_max - x_min

    if len(image.shape) == 3:
        cropped = np.full(
            (cropped_height, cropped_width, image.shape[2]),
            background_color[0],
            dtype=image.dtype,
        )
    else:
        cropped = np.full(
            (cropped_height, cropped_width), background_color[0], dtype=image.dtype
        )

    # Create circular mask
    y_coords, x_coords = np.ogrid[:cropped_height, :cropped_width]
    center_x_rel = cx - x_min
    center_y_rel = cy - y_min

    mask = (
        (x_coords - center_x_rel) ** 2 + (y_coords - center_y_rel) ** 2
    ) <= radius**2

    # Apply the original image within the circular mask
    if len(image.shape) == 3:
        for channel in range(image.shape[2]):
            cropped[mask, channel] = image[y_min:y_max, x_min:x_max, channel][mask]
    else:
        cropped[mask] = image[y_min:y_max, x_min:x_max][mask]

    return cropped


def mm_to_pixels(
    millimeters: float,
    dots_per_inch: float = 300,
    pixels_per_mm: Union[float, None] = None,
) -> float:
    """
    Convert a measurement in millimetres to image pixels.

    Args:
        millimeters: The measurement to convert
        dots_per_inch: The conversion factor
        pixels_per_mm: Optional conversion factor, instead of DPI

    Returns:
        A value in pixels
    """
    if (
        millimeters <= 0
        or dots_per_inch <= 0
        or (pixels_per_mm is not None and pixels_per_mm <= 0)
    ):
        raise ValueError("All supplied arguments must be positive values")

    factor = dots_per_inch / 25.4

    if pixels_per_mm is not None:
        factor = pixels_per_mm

    return int(millimeters * factor)

    @property
    def width(self) -> float:
        return self.diameter


def circularity(area: float, perimeter: float) -> float:
    """
    Calculate how closely the shape of an object approaches that of a mathematically perfect circle

    A mathematically perfect circle has a circularity of 1

    :param area: the size of the region enclosed by the perimeter
    :param perimeter: the total distance along the edge of a shape
    :returns: a ratio of area to perimiter as a float
    """
    return (4 * pi * area) / (perimeter * perimeter)
