"""Part 2 — Generate 6 augmented copies of a leaf image.

Usage:
    ./Augmentation.py <image_path>

Produces 6 files in the same directory as the source image:
    <stem>_<Label>.<ext>  for each label in augmentations.AUGMENTATION_NAMES

Also displays a 1x7 grid: Original + the 6 augmented images.
"""
import os
import sys
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import augmentations


_NON_INTERACTIVE_BACKENDS = {
    "agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_path(src: str, label: str) -> str:
    directory = os.path.dirname(src) or "."
    base = os.path.basename(src)
    stem, ext = os.path.splitext(base)
    return os.path.join(directory, f"{stem}_{label}{ext}")


def _display(original: np.ndarray,
             results: List[Tuple[str, np.ndarray]]) -> None:
    n = 1 + len(results)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 4))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[0].axis("off")
    for ax, (label, img) in zip(axes[1:], results):
        ax.imshow(img)
        ax.set_title(label)
        ax.axis("off")
    fig.tight_layout()
    if matplotlib.get_backend().lower() in _NON_INTERACTIVE_BACKENDS:
        out = "augmentation_result.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"No display available; saved grid to {out}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def run(image_path: str) -> int:
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        return 1

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: cannot open image: {image_path}", file=sys.stderr)
        return 1
    original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results: List[Tuple[str, np.ndarray]] = []
    for label in augmentations.AUGMENTATION_NAMES:
        augmented = augmentations.apply(label, original)
        out_path = _output_path(image_path, label)
        bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(out_path, bgr):
            print(f"Error: failed to write {out_path}", file=sys.stderr)
            return 1
        print(f"Saved {out_path}")
        results.append((label, augmented))

    _display(original, results)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
