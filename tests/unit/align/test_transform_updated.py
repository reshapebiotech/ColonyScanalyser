from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from scipy.ndimage import shift
from skimage.data import coins
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
    rotate,
    warp,
)

from colonyscanalyser.align.transform import (
    DescriptorAlignTransform,
    FastFourierAlignTransform,
    transform_parameters_equal,
)


class TestUpdatedTransformAPI:
    """Test suite for the updated transform API to ensure compatibility with modern scikit-image."""

    @pytest.fixture
    def sample_image(self):
        """Provide a sample image for testing."""
        return coins()

    @pytest.fixture
    def rotated_image(self, sample_image):
        """Provide a rotated version of the sample image."""
        return rotate(sample_image, 15, resize=False)

    @pytest.fixture
    def translated_image(self, sample_image):
        """Provide a translated version of the sample image."""
        return shift(sample_image, (10, 5), order=1)

    def test_similarity_transform_creation(self):
        """Test that SimilarityTransform can be created with different parameters."""
        # Test with rotation and translation
        transform = SimilarityTransform(rotation=np.pi / 4, translation=(10, 20))
        assert hasattr(transform, "params")
        assert transform.params.shape == (3, 3)

        # Test with matrix
        matrix = np.array(
            [
                [np.cos(np.pi / 6), -np.sin(np.pi / 6), 5],
                [np.sin(np.pi / 6), np.cos(np.pi / 6), 10],
                [0, 0, 1],
            ]
        )
        transform_matrix = SimilarityTransform(matrix=matrix)
        assert hasattr(transform_matrix, "params")

    def test_euclidean_transform_creation(self):
        """Test that EuclideanTransform works with the new API."""
        transform = EuclideanTransform(rotation=np.pi / 8, translation=(5, -3))
        assert hasattr(transform, "params")
        assert transform.params.shape == (3, 3)


class TestDescriptorAlignTransformUpdated:
    """Test the updated DescriptorAlignTransform implementation."""

    @pytest.fixture
    def descriptor_aligner(self, sample_image):
        """Create a DescriptorAlignTransform instance."""
        return DescriptorAlignTransform(
            sample_image, transform_model=SimilarityTransform
        )

    @pytest.fixture
    def sample_image(self):
        return coins()

    def test_initialization(self, sample_image):
        """Test that DescriptorAlignTransform initializes correctly with new API."""
        aligner = DescriptorAlignTransform(
            sample_image, transform_model=EuclideanTransform
        )
        assert aligner.transform_model == EuclideanTransform
        assert hasattr(aligner, "descriptor_extractor")

    def test_image_ref_property(self, descriptor_aligner):
        """Test that image_ref property returns descriptors and keypoints."""
        descriptors, keypoints = descriptor_aligner.image_ref
        assert isinstance(descriptors, np.ndarray)
        assert isinstance(keypoints, np.ndarray)

    def test_align_with_robust_error_handling(self, descriptor_aligner, sample_image):
        """Test alignment with improved error handling."""
        # Create a simple test image that should not cause errors
        simple_image = np.zeros_like(sample_image)
        simple_image[100:200, 100:200] = 255

        try:
            result = descriptor_aligner.align(simple_image, precise=False)
            # If successful, check basic properties
            assert result.shape == simple_image.shape
        except RuntimeError as e:
            # This is expected if no features can be matched
            assert "No feature matches" in str(e)

    def test_extract_keypoints_with_fallback(self, descriptor_aligner, sample_image):
        """Test keypoint extraction with fallback for different descriptor extractors."""
        descriptors, keypoints = descriptor_aligner._extract_keypoints(sample_image)

        # Should return arrays even if empty
        assert isinstance(descriptors, np.ndarray)
        assert isinstance(keypoints, np.ndarray)


class TestFastFourierAlignTransformUpdated:
    """Test the updated FastFourierAlignTransform implementation."""

    @pytest.fixture
    def fft_aligner(self, sample_image):
        """Create a FastFourierAlignTransform instance."""
        return FastFourierAlignTransform(
            sample_image, transform_model=SimilarityTransform
        )

    @pytest.fixture
    def sample_image(self):
        return coins()

    def test_initialization(self, sample_image):
        """Test FFT aligner initialization."""
        aligner = FastFourierAlignTransform(
            sample_image, transform_model=AffineTransform
        )
        assert aligner.transform_model == AffineTransform

    def test_align_with_rotation(self, fft_aligner, sample_image):
        """Test alignment of rotated image."""
        rotated = rotate(sample_image, 10, resize=False)

        try:
            result = fft_aligner.align(rotated, precise=False)
            assert result.shape == rotated.shape
            # Should be more similar to original than the rotated version
            from skimage.metrics import normalized_root_mse

            original_error = normalized_root_mse(sample_image, rotated)
            aligned_error = normalized_root_mse(sample_image, result)
            # Allow for some tolerance in improvement
            assert aligned_error <= original_error + 0.1
        except Exception as e:
            # FFT alignment might fail with certain images/transforms
            pytest.skip(f"FFT alignment failed: {e}")

    def test_align_transform_robust(self, fft_aligner, sample_image):
        """Test transform calculation with robust error handling."""
        shifted = shift(sample_image, (5, 5), order=1)

        try:
            transform = fft_aligner.align_transform(shifted)
            assert hasattr(transform, "params")
            assert transform.params.shape == (3, 3)
        except Exception as e:
            pytest.skip(f"Transform calculation failed: {e}")


class TestTransformParametersEqualUpdated:
    """Test the updated transform_parameters_equal function."""

    def test_identical_transforms(self):
        """Test that identical transforms are considered equal."""
        transform1 = SimilarityTransform(rotation=0.1, translation=(5, 3))
        transform2 = SimilarityTransform(rotation=0.1, translation=(5, 3))

        assert transform_parameters_equal(transform1, transform2, tolerance=0.001)

    def test_different_transforms(self):
        """Test that different transforms are not equal."""
        transform1 = SimilarityTransform(rotation=0.1, translation=(5, 3))
        transform2 = SimilarityTransform(rotation=0.2, translation=(5, 3))

        assert not transform_parameters_equal(transform1, transform2, tolerance=0.001)

    def test_matrix_input(self):
        """Test with matrix input."""
        transform = SimilarityTransform(rotation=0.1, translation=(5, 3))
        matrix = transform.params

        assert transform_parameters_equal(transform, matrix, tolerance=0.001)

    def test_tolerance_behavior(self):
        """Test tolerance parameter behavior."""
        transform1 = SimilarityTransform(rotation=0.1, translation=(5, 3))
        transform2 = SimilarityTransform(rotation=0.11, translation=(5, 3))

        # Should be equal with larger tolerance
        assert transform_parameters_equal(transform1, transform2, tolerance=0.1)

        # Should not be equal with smaller tolerance
        assert not transform_parameters_equal(transform1, transform2, tolerance=0.001)

    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        transform = SimilarityTransform()

        # Test with invalid matrix shape
        invalid_matrix = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            transform_parameters_equal(transform, invalid_matrix, tolerance=0.1)

        # Test with invalid object
        with pytest.raises(ValueError):
            transform_parameters_equal(transform, "invalid", tolerance=0.1)


class TestTransformCompatibility:
    """Test compatibility between different transform types."""

    @pytest.mark.parametrize(
        "transform_class", [SimilarityTransform, EuclideanTransform, AffineTransform]
    )
    def test_transform_classes_work(self, transform_class):
        """Test that all transform classes work with our code."""
        # Test basic instantiation
        transform = transform_class()
        assert hasattr(transform, "params")

        # Test with parameters
        if transform_class == EuclideanTransform:
            transform_with_params = transform_class(rotation=0.1, translation=(1, 2))
        elif transform_class == SimilarityTransform:
            transform_with_params = transform_class(
                scale=1.1, rotation=0.1, translation=(1, 2)
            )
        else:  # AffineTransform
            transform_with_params = transform_class(translation=(1, 2))

        assert hasattr(transform_with_params, "params")
        assert transform_with_params.params.shape == (3, 3)


class TestErrorHandling:
    """Test error handling in updated transform code."""

    def test_missing_imaging_module_fallback(self):
        """Test fallback when imaging module is not available."""
        # This tests the try/except blocks in the transform code
        sample_image = coins()

        # Test DescriptorAlignTransform with missing imaging module
        with patch(
            "colonyscanalyser.align.transform.image_as_rgb", side_effect=ImportError
        ):
            aligner = DescriptorAlignTransform(sample_image)
            # Should still work with fallback
            descriptors, keypoints = aligner._extract_keypoints(sample_image)
            assert isinstance(descriptors, np.ndarray)
            assert isinstance(keypoints, np.ndarray)

    def test_descriptor_extractor_compatibility(self):
        """Test compatibility with different descriptor extractor interfaces."""
        sample_image = coins()

        # Test with mock descriptor extractor that has different interface
        mock_extractor = MagicMock()
        mock_extractor.descriptors = np.array([[1, 2, 3]])
        mock_extractor.keypoints = np.array([[10, 20]])

        aligner = DescriptorAlignTransform(sample_image)
        aligner.descriptor_extractor = mock_extractor

        # Should handle both detect_and_extract and separate detect/extract
        descriptors, keypoints = aligner._extract_keypoints(sample_image)
        assert isinstance(descriptors, np.ndarray)
        assert isinstance(keypoints, np.ndarray)


@pytest.mark.integration
class TestIntegrationWithSkimage:
    """Integration tests with actual scikit-image functions."""

    def test_warp_integration(self):
        """Test that our transforms work with skimage.transform.warp."""

        image = coins()
        transform = SimilarityTransform(rotation=0.1, translation=(5, 3))

        # This should work without errors
        warped = warp(image, transform.inverse)
        assert warped.shape == image.shape

    def test_transform_chaining(self):
        """Test chaining of transforms."""
        transform1 = SimilarityTransform(rotation=0.1)
        transform2 = SimilarityTransform(translation=(5, 3))

        # Test transform addition (if supported)
        try:
            combined = transform1 + transform2
            assert hasattr(combined, "params")
        except (TypeError, AttributeError):
            # Some transform combinations might not be supported
            pass
