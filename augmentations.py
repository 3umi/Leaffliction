"""Shared augmentation definitions for Part 2.
"""
import cv2
import albumentations as A
import numpy as np

AUGMENTATIONS = {
    "Flip": A.HorizontalFlip(p=1.0),
    "Rotate": A.Rotate(
        limit=(-60, -30),
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    ),
    "Shear": A.Affine(
        shear={"x": (20, 30), "y": 0},
        border_mode=cv2.BORDER_REFLECT_101,
        p=1.0,
    ),
    "Zoom": A.RandomResizedCrop(
        size=(256, 256),
        scale=(0.5, 0.6),
        ratio=(0.9, 1.1),
        p=1.0,
    ),
    "Lighting": A.RandomBrightnessContrast(
        brightness_limit=(0.3, 0.6),
        contrast_limit=(0.3, 0.6),
        p=1.0,
    ),
    "Color": A.HueSaturationValue(
        hue_shift_limit=(30, 60),
        sat_shift_limit=(60, 100),
        val_shift_limit=(30, 60),
        p=1.0,
    ),
}

AUGMENTATION_NAMES = list(AUGMENTATIONS.keys())


def apply(
    name: str,
    image: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Apply named augmentation to an RGB numpy image.

    Args:
        name: one of AUGMENTATION_NAMES.
        image: HxWx3 numpy array in RGB order, uint8.
        seed: if provided, reseeds the transform's per-instance RNGs so
            the output is deterministic. Albumentations 2.x transforms
            each own a `numpy.random.default_rng()` that is *not*
            affected by `np.random.seed()`, so callers must reseed here
            to get reproducible bytes.

    Returns:
        Augmented HxWx3 numpy array in RGB order.

    Raises:
        KeyError: if name is not in AUGMENTATIONS.
    """
    if name not in AUGMENTATIONS:
        raise KeyError(
            f"Unknown augmentation {name!r}; "
            f"expected one of {AUGMENTATION_NAMES}"
        )
    transform = AUGMENTATIONS[name]
    if seed is not None:
        transform.set_random_seed(seed)
    return transform(image=image)["image"]
