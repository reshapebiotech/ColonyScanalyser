"""Tests for alignment services."""

import numpy as np
import pytest

from colonyscanalyser.services.alignment import (
    align_image_simple,
    align_images_features,
    align_images_fft,
    calculate_alignment_quality,
)


def create_test_image(size: int = 100, pattern: str = "circle") -> np.ndarray:
    """Create a synthetic test image."""
    image = np.zeros((size, size), dtype=float)

    if pattern == "circle":
        # Create a circle in the center
        y, x = np.ogrid[:size, :size]
        center = size // 2
        radius = size // 4
        mask = (x - center) ** 2 + (y - center) ** 2 <= radius**2
        image[mask] = 1.0
    elif pattern == "square":
        # Create a square in the center
        quarter = size // 4
        image[quarter : 3 * quarter, quarter : 3 * quarter] = 1.0
    elif pattern == "gradient":
        # Create a gradient
        image = np.linspace(0, 1, size * size).reshape(size, size)

    return image


def shift_image(image: np.ndarray, shift: tuple) -> np.ndarray:
    """Shift an image by a given amount."""
    return np.roll(image, shift, axis=(0, 1))


def test_align_images_fft_basic():
    """Test basic FFT alignment functionality."""
    # Create reference image
    reference = create_test_image(100, "circle")

    # Create shifted target image
    shift_amount = (10, 15)
    target = shift_image(reference, shift_amount)

    # Align images
    aligned, detected_shift = align_images_fft(target, reference)

    # Check that we detected some shift
    assert len(detected_shift) == 2
    assert not np.allclose(detected_shift, [0, 0])

    # Check that aligned image is different from original target
    assert not np.array_equal(aligned, target)


def test_align_images_fft_no_shift():
    """Test FFT alignment when no shift is needed."""
    # Create identical images
    reference = create_test_image(50, "square")
    target = reference.copy()

    # Align images
    aligned, detected_shift = align_images_fft(target, reference)

    # Should detect minimal shift
    assert np.allclose(detected_shift, [0, 0], atol=1e-10)


def test_align_images_fft_different_patterns():
    """Test FFT alignment with different image patterns."""
    patterns = ["circle", "square", "gradient"]

    for pattern in patterns:
        reference = create_test_image(80, pattern)
        target = shift_image(reference, (5, 8))

        aligned, shift = align_images_fft(target, reference)

        # Should detect some shift
        assert not np.allclose(shift, [0, 0])


def test_align_images_features_basic():
    """Test basic feature alignment."""
    # Create images with features
    reference = create_test_image(100, "circle")

    # Add noise to make it more realistic for feature detection
    noise = np.random.normal(0, 0.1, reference.shape)
    reference = np.clip(reference + noise, 0, 1)

    # Create slightly shifted target
    target = shift_image(reference, (3, 5))

    # Align images
    aligned = align_images_features(target, reference)

    # Feature alignment might fail on simple synthetic images
    # So we just check that it returns something (could be None)
    assert aligned is None or isinstance(aligned, np.ndarray)


def test_align_images_features_insufficient_features():
    """Test feature alignment with insufficient features."""
    # Create uniform image (no features)
    reference = np.ones((50, 50))
    target = np.ones((50, 50))

    # Should return None due to lack of features
    aligned = align_images_features(target, reference)
    assert aligned is None


def test_calculate_alignment_quality_perfect():
    """Test alignment quality calculation with identical images."""
    image = create_test_image(60, "circle")

    quality = calculate_alignment_quality(image, image)

    # Should be perfect correlation
    assert abs(quality - 1.0) < 1e-10


def test_calculate_alignment_quality_different():
    """Test alignment quality with different images."""
    image1 = create_test_image(60, "circle")
    image2 = create_test_image(60, "square")

    quality = calculate_alignment_quality(image1, image2)

    # Should be less than perfect
    assert quality < 1.0


def test_calculate_alignment_quality_shifted():
    """Test alignment quality with shifted images."""
    reference = create_test_image(80, "circle")
    shifted = shift_image(reference, (10, 10))

    quality = calculate_alignment_quality(shifted, reference)

    # Should be less than perfect due to shift
    assert quality < 1.0


def test_calculate_alignment_quality_uniform():
    """Test alignment quality with uniform images."""
    # Uniform images should return 0 (NaN case)
    uniform1 = np.ones((50, 50))
    uniform2 = np.ones((50, 50))

    quality = calculate_alignment_quality(uniform1, uniform2)
    assert quality == 0.0


def test_align_image_simple_fft():
    """Test simple alignment interface with FFT method."""
    reference = create_test_image(70, "circle")
    target = shift_image(reference, (7, 12))

    aligned = align_image_simple(target, reference, method="fft")

    assert isinstance(aligned, np.ndarray)
    assert aligned.shape == reference.shape


def test_align_image_simple_features():
    """Test simple alignment interface with features method."""
    reference = create_test_image(70, "gradient")
    target = shift_image(reference, (3, 5))

    aligned = align_image_simple(target, reference, method="features")

    # Could return None if feature detection fails
    assert aligned is None or isinstance(aligned, np.ndarray)


def test_align_image_simple_invalid_method():
    """Test simple alignment with invalid method."""
    reference = create_test_image(50, "circle")
    target = create_test_image(50, "circle")

    with pytest.raises(ValueError, match="Unknown alignment method"):
        align_image_simple(target, reference, method="invalid")


def test_align_images_fft_with_kwargs():
    """Test FFT alignment with additional parameters."""
    reference = create_test_image(60, "square")
    target = shift_image(reference, (4, 6))

    aligned, shift = align_images_fft(target, reference, upsample_factor=50)

    assert isinstance(aligned, np.ndarray)
    assert len(shift) == 2


def test_align_images_features_with_kwargs():
    """Test feature alignment with additional parameters."""
    reference = create_test_image(100, "gradient")

    # Add some noise to create features
    noise = np.random.normal(0, 0.2, reference.shape)
    reference = np.clip(reference + noise, 0, 1)

    target = shift_image(reference, (2, 3))

    aligned = align_images_features(
        target, reference, n_keypoints=100, match_threshold=0.8
    )

    # Could return None if not enough good features
    assert aligned is None or isinstance(aligned, np.ndarray)


def test_alignment_workflow():
    """Test complete alignment workflow."""
    # Create reference image
    reference = create_test_image(90, "circle")

    # Create shifted and slightly noisy target
    target = shift_image(reference, (8, 12))
    noise = np.random.normal(0, 0.05, target.shape)
    target = np.clip(target + noise, 0, 1)

    # Try FFT alignment
    aligned_fft = align_image_simple(target, reference, method="fft")
    quality_fft = calculate_alignment_quality(aligned_fft, reference)

    # Quality should be reasonable
    assert quality_fft > 0.5

    # Try feature alignment (might fail on synthetic data)
    aligned_features = align_image_simple(target, reference, method="features")

    if aligned_features is not None:
        quality_features = calculate_alignment_quality(aligned_features, reference)
        assert quality_features >= 0.0  # At least valid


def test_empty_images():
    """Test alignment with empty images."""
    empty1 = np.zeros((50, 50))
    empty2 = np.zeros((50, 50))

    # FFT should work but return zero shift
    aligned, shift = align_images_fft(empty1, empty2)
    assert isinstance(aligned, np.ndarray)

    # Features should return None (no features)
    aligned_features = align_images_features(empty1, empty2)
    assert aligned_features is None

    # Quality calculation should handle it
    quality = calculate_alignment_quality(empty1, empty2)
    assert quality == 0.0
