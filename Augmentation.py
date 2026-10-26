#!/usr/bin/env python3
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

import augmentations

from utils.images import is_image


def _output_path(src: str, name: str) -> str:
    stem, ext = os.path.splitext(src)
    return f"{stem}_{name}{ext}"


def _display(original: np.ndarray,
             results: List[Tuple[str, np.ndarray]]) -> None:
    images = [("Original", original)] + results
    fig, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 4))
    for ax, (name, img) in zip(axes, images):
        ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def run(image_path: str) -> int:
    if not os.path.isfile(image_path) or not is_image(image_path):
        print("Error: not an image file (.jpg/.jpeg/.png)", file=sys.stderr)
        return 1

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot open image: {image_path}", file=sys.stderr)
        return 1
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results: List[Tuple[str, np.ndarray]] = []
    for name in augmentations.AUGMENTATION_NAMES:
        augmented = augmentations.apply(name, original)
        out_path = _output_path(image_path, name)
        bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(out_path, bgr):
            print(f"Error: failed to write {out_path}", file=sys.stderr)
            return 1
        print(f"Saved {out_path}")
        results.append((name, augmented))

    _display(original, results)
    return 0


def main(argv: list) -> int:
    if argv[1:] and argv[1] in ("-h", "--help"):
        print(f"Usage: {argv[0]} <image_path>")
        return 0
    if len(argv) != 2:
        print(f"Usage: {argv[0]} <image_path>", file=sys.stderr)
        return 2
    return run(argv[1])


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)
