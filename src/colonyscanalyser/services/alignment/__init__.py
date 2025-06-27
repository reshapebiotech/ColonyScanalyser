"""Image alignment services."""

from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp


def align_images_fft(
    image_target: ndarray,
    image_reference: ndarray,
    upsample_factor: int = 100,
) -> Tuple[ndarray, np.ndarray]:
    """
    Align images using FFT-based phase correlation.

    Args:
        image_target: Image to be aligned
        image_reference: Reference image to align to
        upsample_factor: Upsampling factor for sub-pixel precision

    Returns:
        Tuple of (aligned_image, shift_vector)
    """
    # Calculate shift using phase cross correlation
    shift, error, diffphase = phase_cross_correlation(
        image_reference, image_target, upsample_factor=upsample_factor
    )

    # Apply shift to align the image
    aligned = np.roll(image_target, shift.astype(int), axis=(0, 1))

    return aligned, shift


def align_images_features(
    image_target: ndarray,
    image_reference: ndarray,
    n_keypoints: int = 500,
    match_threshold: float = 0.6,
) -> Optional[ndarray]:
    """
    Align images using feature matching.

    Args:
        image_target: Image to be aligned
        image_reference: Reference image to align to
        n_keypoints: Number of keypoints to detect
        match_threshold: Threshold for feature matching

    Returns:
        Aligned image or None if alignment fails
    """
    # Initialize ORB detector
    detector = ORB(n_keypoints=n_keypoints)

    # Detect keypoints and descriptors
    try:
        detector.detect_and_extract(image_reference)
        keypoints_ref = detector.keypoints
        descriptors_ref = detector.descriptors
    except RuntimeError:
        # No features found in reference image
        return None

    try:
        detector.detect_and_extract(image_target)
        keypoints_target = detector.keypoints
        descriptors_target = detector.descriptors
    except RuntimeError:
        # No features found in target image
        return None

    # Match features
    if descriptors_ref is None or descriptors_target is None:
        return None

    matches = match_descriptors(
        descriptors_ref, descriptors_target, cross_check=True, max_ratio=match_threshold
    )

    if len(matches) < 4:  # Need at least 4 points for similarity transform
        return None

    # Get matched keypoints
    src_pts = keypoints_ref[matches[:, 0]]
    dst_pts = keypoints_target[matches[:, 1]]

    # Estimate transformation using RANSAC
    try:
        model, inliers = ransac(
            (src_pts, dst_pts),
            SimilarityTransform,
            min_samples=4,
            residual_threshold=2.0,
            max_trials=100,
        )

        if model is None:
            return None

        # Apply transformation
        aligned = warp(image_target, model.inverse, output_shape=image_reference.shape)
        return aligned

    except Exception:
        return None


def calculate_alignment_quality(
    image_aligned: ndarray,
    image_reference: ndarray,
) -> float:
    """
    Calculate alignment quality using normalized cross correlation.

    Args:
        image_aligned: Aligned image
        image_reference: Reference image

    Returns:
        Correlation coefficient (higher is better, max 1.0)
    """
    # Flatten images
    img1_flat = image_aligned.flatten()
    img2_flat = image_reference.flatten()

    # Calculate normalized cross correlation
    correlation = np.corrcoef(img1_flat, img2_flat)[0, 1]

    # Handle NaN case
    if np.isnan(correlation):
        return 0.0

    return correlation


def align_image_simple(
    image_target: ndarray,
    image_reference: ndarray,
    method: str = "fft",
    **kwargs,
) -> Optional[ndarray]:
    """
    Simple interface for image alignment.

    Args:
        image_target: Image to be aligned
        image_reference: Reference image to align to
        method: Alignment method ("fft" or "features")
        **kwargs: Additional arguments for specific methods

    Returns:
        Aligned image or None if alignment fails
    """
    if method == "fft":
        aligned, _ = align_images_fft(image_target, image_reference, **kwargs)
        return aligned
    elif method == "features":
        return align_images_features(image_target, image_reference, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")
