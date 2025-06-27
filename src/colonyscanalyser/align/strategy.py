from enum import Enum, auto
from typing import List, Tuple, Union

from ..core.image_file import ImageFile
from .transform import AlignTransform


class AlignStrategy(Enum):
    quick = auto()
    verify = auto()
    complete = auto()
    none = auto()


def calculate_transformation_strategy(
    images: List[ImageFile],
    strategy: AlignStrategy,
    transform_type: str = "euclidean",
    tolerance: float = 0.1,
    **kwargs,
) -> Tuple[Union[AlignTransform, type, None], List[ImageFile]]:
    from skimage.transform import (
        AffineTransform,
        EuclideanTransform,
        ProjectiveTransform,
        SimilarityTransform,
    )
    from .transform import FastFourierAlignTransform
    
    shift_index = 0
    
    if len(images) <= 1 or strategy == AlignStrategy.none:
        return None, images
    
    TRANSFORM_CLASSES = {
        "euclidean": EuclideanTransform,
        "similarity": SimilarityTransform,
        "affine": AffineTransform,
        "projective": ProjectiveTransform,
    }
    
    transform_type = transform_type.lower()
    if transform_type not in TRANSFORM_CLASSES:
        raise ValueError(f"the transformation type {transform_type} is not implemented")
    transform_model = TRANSFORM_CLASSES[transform_type]
    
    align_model = FastFourierAlignTransform(images[0].image, transform_model)
    
    return align_model, images[shift_index:]


def apply_align_transform(
    image_file: ImageFile,
    align_model: Union[AlignTransform, type],
    replace_existing: bool = False,
    **kwargs,
) -> ImageFile:
    if image_file.alignment_transform is None or replace_existing:
        if isinstance(align_model, AlignTransform):
            align_model = align_model.align_transform(image_file.image, **kwargs)
        image_file.alignment_transform = align_model
    return image_file
