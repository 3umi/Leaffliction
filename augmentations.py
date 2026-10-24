import cv2
import albumentations as A
import numpy as np

BORDER = cv2.BORDER_REFLECT_101

AUGMENTATIONS = {
    "Flip": A.HorizontalFlip(p=1.0),
    "Rotate": A.Rotate(
        limit=(-30, 30),
        border_mode=BORDER,
        p=1.0
    ),
    "Skew": A.Perspective(
        scale=(0.05, 0.1),
        keep_size=True,
        border_mode=BORDER,
        p=1.0
    ),
    "Shear": A.Affine(
        shear={"x": (-20, 20), "y": 0},
        border_mode=BORDER,
        p=1.0
    ),
    "Zoom": A.Affine(
        scale=(1.1, 1.25),
        border_mode=BORDER,
        p=1.0
    ),
    "Lighting": A.RandomBrightnessContrast(
        brightness_limit=(-0.2, 0.2),
        contrast_limit=(-0.2, 0.2),
        p=1.0
    ),
}

AUGMENTATION_NAMES = list(AUGMENTATIONS.keys())


def apply(
    name: str,
    image: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    if name not in AUGMENTATIONS:
        raise KeyError(
            f"Unknown augmentation {name!r}; "
            f"expected one of {AUGMENTATION_NAMES}"
        )
    transform = AUGMENTATIONS[name]
    if seed is not None:
        transform.set_random_seed(seed)
    return transform(image=image)["image"]
