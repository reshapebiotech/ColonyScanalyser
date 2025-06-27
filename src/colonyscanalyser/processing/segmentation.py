"""
Image segmentation module for ColonyScanalyser.

This module provides functionality for segmenting images to identify and
label colonies on plates for analysis.
"""

from numpy import ndarray


def remove_background_mask(
    image: ndarray, smoothing: float = 1, sigmoid_cutoff: float = 0.4, **filter_args
) -> ndarray:
    """
    Separate the image foreground from the background.

    Returns a boolean mask of the image foreground.

    :param image: an image as a numpy array
    :param smoothing: a sigma value for the gaussian filter
    :param sigmoid_cutoff: cutoff for the sigmoid exposure function
    :param filter_args: arguments to pass to the gaussian filter
    :returns: a boolean image mask of the foreground
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
    from numpy import clip, isin, unique, zeros
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.measure import label
    from skimage.morphology import binary_erosion, remove_small_objects
    from skimage.segmentation import clear_border, watershed

    diff = clip(plate_noise_mask - plate_image, 0, 255)

    plate_image = ~remove_background_mask(diff, smoothing=1, sigmoid_cutoff=0)

    if plate_mask is not None:
        # Remove mask from image
        plate_image = plate_image & plate_mask
        # Remove objects touching the mask border
        plate_image = clear_border(
            plate_image, bgval=0, mask=binary_erosion(plate_mask)
        )
    else:
        # Remove objects touching the image border
        plate_image = clear_border(plate_image, buffer_size=2, bgval=0)

    distance = ndi.distance_transform_edt(plate_image)
    local_max_coords = peak_local_max(distance, min_distance=7)
    local_max_mask = zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = label(local_max_mask)

    labels = watershed(-distance, markers, mask=plate_image)

    # Remove background noise
    if len(unique(labels)) > 1:
        labels = remove_small_objects(labels, min_size=area_min)

    # Remove colonies that have grown on top of image artefacts or static objects
    if plate_noise_mask is not None:
        plate_noise_image = remove_background_mask(
            plate_noise_mask, smoothing=1, sigmoid_cutoff=0.7
        )
        if len(unique(plate_noise_mask)) > 1:
            noise_mask = remove_small_objects(plate_noise_image, min_size=area_min)
        # Remove all objects where there is an existing static object
        exclusion = unique(labels[noise_mask])
        exclusion_mask = isin(labels, exclusion[exclusion > 0])
        labels[exclusion_mask] = 0

    return labels
