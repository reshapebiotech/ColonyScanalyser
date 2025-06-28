"""Image alignment services."""

from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from skimage.feature import ORB, match_descriptors
from skimage.measure import ransac
from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp


def align_images_fft_fast(
    image_target: ndarray,
    image_reference: ndarray,
    scale_factor: float = 0.25,
    numiter: int = 1,
    constraints: Optional[dict] = None,
) -> Tuple[ndarray, dict]:
    """
    Fast FFT-based alignment with scaling for performance.

    Args:
        image_target: Image to be aligned
        image_reference: Reference image to align to
        scale_factor: Scale factor for faster processing (0.1-1.0)
        numiter: Number of iterations for precision
        constraints: Optional constraints for alignment

    Returns:
        Tuple of (aligned_image, transform_params)
    """
    try:
        from imreg_dft import similarity, transform_img
        from skimage.color import rgb2gray
        from skimage.transform import rescale
    except ImportError:
        # Fallback to basic FFT alignment
        return align_images_fft(image_target, image_reference)

    # Convert to grayscale if needed
    if len(image_reference.shape) > 2:
        ref_gray = rgb2gray(image_reference)
    else:
        ref_gray = image_reference

    if len(image_target.shape) > 2:
        target_gray = rgb2gray(image_target)
    else:
        target_gray = image_target

    # Scale images for faster processing
    if scale_factor < 1.0:
        ref_scaled = rescale(
            ref_gray, scale_factor, anti_aliasing=True, preserve_range=True
        )
        target_scaled = rescale(
            target_gray, scale_factor, anti_aliasing=True, preserve_range=True
        )
    else:
        ref_scaled = ref_gray
        target_scaled = target_gray

    # Perform alignment on scaled images
    transform_params = similarity(
        ref_scaled, target_scaled, numiter=numiter, constraints=constraints
    )

    # Scale transform parameters back to original size
    if scale_factor < 1.0:
        transform_params["tvec"] = transform_params["tvec"] / scale_factor

    # Apply transformation to original image
    aligned = transform_img(
        image_target,
        transform_params["scale"],
        transform_params["angle"],
        transform_params["tvec"],
        bgval=0,
    )

    return aligned, transform_params


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


def get_optimal_scale_factor(image_shape: Tuple[int, int]) -> float:
    """
    Calculate optimal scale factor for alignment based on image size.

    Args:
        image_shape: Shape of the image (height, width)

    Returns:
        Optimal scale factor for fast processing
    """
    height, width = image_shape[:2]
    total_pixels = height * width

    # Scale factor based on image size for optimal performance
    if total_pixels > 4000000:  # > 4MP
        return 0.1
    elif total_pixels > 1000000:  # > 1MP
        return 0.25
    elif total_pixels > 250000:  # > 0.25MP
        return 0.5
    else:
        return 1.0


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
    elif method == "fft_fast":
        aligned, _ = align_images_fft_fast(image_target, image_reference, **kwargs)
        return aligned
    elif method == "features":
        return align_images_features(image_target, image_reference, **kwargs)
    else:
        raise ValueError(f"Unknown alignment method: {method}")


def create_alignment_strategy(
    strategy_name: str = "fast",
    image_shape: Optional[Tuple[int, int]] = None,
) -> dict:
    """
    Create alignment strategy configuration.

    Args:
        strategy_name: Strategy name ("fast", "accurate", "balanced")
        image_shape: Optional image shape for auto-scaling

    Returns:
        Dictionary with alignment configuration
    """
    strategies = {
        "fast": {
            "method": "fft_fast",
            "scale_factor": 0.1,
            "numiter": 1,
        },
        "balanced": {
            "method": "fft_fast",
            "scale_factor": 0.25,
            "numiter": 2,
        },
        "accurate": {
            "method": "fft",
            "upsample_factor": 100,
        },
        "features": {
            "method": "features",
            "n_keypoints": 500,
            "match_threshold": 0.6,
        },
    }

    if strategy_name not in strategies:
        strategy_name = "balanced"

    config = strategies[strategy_name].copy()

    # Auto-adjust scale factor based on image size
    if image_shape and "scale_factor" in config:
        optimal_scale = get_optimal_scale_factor(image_shape)
        config["scale_factor"] = min(config["scale_factor"], optimal_scale)

    return config
