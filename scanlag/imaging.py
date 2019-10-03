def crop_image(image, crop_shape, center = None):
    """
    Get a subsection of an image

    Optionally specify a center point to crop around

    :param image: an image as a numpy array
    :param crop_shape: a row, column tuple array size to crop
    :param center: a row, column tuple co-ordinate point
    :returns: an image as a numpy array
    """
    import operator

    img = image.copy()

    if any(x < 0 for x in crop_shape) or len(image.shape) < len(crop_shape):
        raise ValueError("The crop shape must be positive integers and the same dimensions as the image to crop")
    if any(r < c for r, c in zip(img.shape, crop_shape)):
        raise ValueError("The crop shape cannot be larger than the image to crop")

    if center is None:
        # Use the center of the image
        start = tuple(map(lambda a, da: a // 2 - da // 2, img.shape, crop_shape))
    else:
        # Use a custom center point
        start = tuple(map(lambda a, da: a - da // 2, center, crop_shape))

    end = tuple(map(operator.add, start, crop_shape))
    
    if any(x < 0 for x in start) or any(x > y for x in end for y in img.shape):
        raise ValueError("The crop area cannot be outside the original image")

    slices = tuple(map(slice, start, end))

    return img[slices]

def cut_image_circle(image, **kwargs):
    """
    Get the circular area of an image

    A center and radius can be specified to return a smaller part of the image

    :param image: an image as a numpy array
    :param center: a row, column tuple co-ordinate point
    :param radius: an integer length
    :param inverse: specify whether to return the inside or the outside area of the circle
    :returns: an image as a numpy array
    """
    import numpy as np

    img = image.copy()
    center = kwargs.get('center', None)
    radius = kwargs.get('radius', None)
    inverse = kwargs.get('inverse', False)

    # Either use the entire image or crop to a radius
    if radius is None:
        radius = image.shape[0] // 2
    else:
        if any(radius * 2 > x for x in img.shape):
            raise ValueError("The circle radius cannot be larger than the supplied image")
        # Crop the image around the center point (if provided)
        crop_area = (radius * 2) + 1
        img = crop_image(img, (crop_area, crop_area), center)
        
    (cy, cx) = map(lambda x: x // 2, img.shape)
        
    # Calculate distances from center
    dist_x = np.vstack([(np.arange(img.shape[1]) - cx)] * (img.shape[0]))
    dist_y = np.vstack([(np.arange(img.shape[0]) - cy)] * (img.shape[1])).T
    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
    
    if inverse:
        # Remove image information in area inside boundary
        img[dist <= radius] = 0
    else:
        # Remove image information in area outside boundary
        img[dist > radius] = 0

    return img


def remove_background_mask(image, mask):
    """
    Process an image by removing a background mask
    """
    from skimage import exposure
    from skimage.filters import gaussian
    
    background = image[mask & (image > 0.05)].mean()
    ind = gaussian(image, 0.5) > background + 0.03

    return mask & ind


def watershed_separation(image, smoothing = 0.5):
    """
    Returns a labelled image where merged objects are separated
    """
    img = image.copy()
    # Estimate smoothed distance from peaks to background
    distance = ndimage.distance_transform_edt(img)
    distance = gaussian(distance, smoothing)
    
    # Find image peaks, returned as a boolean array (indices = False)
    local_maxi = peak_local_max(distance, indices = False, footprint = np.ones((3, 3)), labels = img)

    # Find the borders around the peaks
    img = watershed(-distance, label(local_maxi), mask = img)

    return img


def standardise_labels_timeline(images_list, start_at_end = True, count_offset = 1000):
    """
    Replace labels on similar images to allow tracking over time

    :param images_list: a list of segmented and lablled images as numpy arrays
    :param start_at_end: relabels the images beginning at the end of the list
    :param count_offset: an int greater than the total number of expected labels in a single image
    :returns: a list of relablled images as numpy arrays
    """
    import numpy as np

    images = list(images_list)
    if start_at_end:
        images.reverse()

    # Relabel all images to ensure there are no duplicates
    for image in images:
        for label in np.unique(image):
            if label > 0:
                count_offset += 1
                image[image == label] = count_offset
        
    # Ensure labels are propagated through image timeline
    for i, image in enumerate(images):
        labels = get_labelled_centers(image)
        
        # Apply labels to all subsequent images
        for j in range(i, len(images)):
            images[j] = replace_image_point_labels(images[j], labels)

    if start_at_end:
        images.reverse()

    return images


def get_labelled_centers(image):
    """
    Builds a list of labels and their centers

    :param image: a segmented and labelled image as a numpy array
    :returns: a list of label, co-ordinate tuples
    """
    from skimage.measure import regionprops

    # Find all labelled areas, disable caching so properties are only calculated if required
    rps = regionprops(image, cache = False)
    
    return [(r.label, r.centroid) for r in rps]


def replace_image_point_labels(image, labels):
    """
    Replace the labelled at a list of points with new labels

    :param image: a segmented and lablled image as a numpy array
    :param labels: a list of label, co-ordinate tuples
    :returns: a relabelled image as a numpy array
    """
    img = image.copy()
    for label, point in labels:
        row, col = point
        # Find the existing label at the point
        index = img[int(row), int(col)]
        # Replace the existing label with new, excluding background
        if index > 0:
            img[img == index] = label

    return img