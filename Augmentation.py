"""Part 2 — Generate 6 augmented copies of a leaf image.

Usage:
    ./Augmentation.py <image_path>

Produces 6 files in the same directory as the source image:
    <stem>_Flip.<ext>
    <stem>_Rotate.<ext>
    <stem>_Skew.<ext>
    <stem>_Shear.<ext>
    <stem>_Crop.<ext>
    <stem>_Distortion.<ext>

Also displays a 1x7 grid: Original + the 6 augmented images.
"""
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


# ---------------------------------------------------------------------------
# Augmentation functions  (each accepts and returns a PIL Image)
# ---------------------------------------------------------------------------

def augment_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def augment_rotate(img: Image.Image, angle: float = 45.0) -> Image.Image:
    return img.rotate(angle, expand=True, resample=Image.BICUBIC)


def augment_skew(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    offset = int(w * 0.25)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [offset, 0],
        [w - offset, 0],
        [w, h],
        [0, h],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out)


def augment_shear(img: Image.Image, shear: float = 0.3) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    M = np.float32([[1, shear, 0],
                    [0, 1,     0]])
    new_w = int(w + abs(shear) * h)
    out = cv2.warpAffine(arr, M, (new_w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return Image.fromarray(out)


def augment_crop(img: Image.Image, ratio: float = 0.8) -> Image.Image:
    w, h = img.size
    left = int(w * (1 - ratio) / 2)
    top = int(h * (1 - ratio) / 2)
    right = w - left
    bottom = h - top
    return img.crop((left, top, right, bottom)).resize((w, h), Image.BICUBIC)


def augment_distortion(img: Image.Image, strength: float = 0.4) -> Image.Image:
    arr = np.array(img)
    h, w = arr.shape[:2]
    k1, k2 = strength * 0.1, strength * 0.05
    fx = fy = max(w, h)
    cx, cy = w / 2.0, h / 2.0
    camera = np.array([[fx, 0, cx],
                       [0, fy, cy],
                       [0,  0,  1]], dtype=np.float64)
    dist = np.array([k1, k2, 0.0, 0.0], dtype=np.float64)
    out = cv2.undistort(arr, camera, dist)
    return Image.fromarray(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUGMENTATIONS: List[Tuple[str, callable]] = [
    ("Flip",       augment_flip),
    ("Rotate",     augment_rotate),
    ("Skew",       augment_skew),
    ("Shear",      augment_shear),
    ("Crop",       augment_crop),
    ("Distortion", augment_distortion),
]


def _output_path(src: str, label: str) -> str:
    """Build the output path: <dir>/<stem>_<label>.<ext>."""
    directory = os.path.dirname(src) or "."
    base = os.path.basename(src)
    stem, ext = os.path.splitext(base)
    return os.path.join(directory, f"{stem}_{label}{ext}")


def _display(original: Image.Image,
             augmented: List[Tuple[str, Image.Image]]) -> None:
    n = 1 + len(augmented)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax, (label, img) in zip(axes[1:], augmented):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(image_path: str) -> int:
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        return 1

    try:
        original = Image.open(image_path).convert("RGB")
    except Exception as exc:
        print(f"Error: cannot open image: {exc}", file=sys.stderr)
        return 1

    results: List[Tuple[str, Image.Image]] = []
    for label, fn in AUGMENTATIONS:
        augmented = fn(original)
        out_path = _output_path(image_path, label)
        augmented.save(out_path)
        print(f"Saved {out_path}")
        results.append((label, augmented))

    _display(original, results)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list) -> int:
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print(f"Usage: {argv[0]} <image_path>")
        return 0 if len(argv) > 1 else 1
    return run(argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
