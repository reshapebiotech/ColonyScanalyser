"""
Image segmentation module for ColonyScanalyser.

This module provides functionality for segmenting images to identify and
label colonies on plates for analysis.
"""

from numpy import ndarray


def segment_image(
    plate_image: ndarray,
    plate_mask: ndarray = None,
    plate_noise_mask: ndarray = None,
    area_min: float = 1,
) -> ndarray:
    """
    Attempts to separate and label all colonies on a plate.

    :param plate_image: an image containing colonies
    :param plate_mask: a boolean image mask to remove from the original image
    :param plate_noise_mask: a black and white image as a numpy array
    :param area_min: the minimum area for a colony, in pixels
    :returns: a segmented and labelled image as a numpy array
    """
