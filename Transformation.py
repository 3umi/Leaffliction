#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv

from utils.images import is_image, list_images


def _color_histogram(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    channels = [
        ("blue",           img[:, :, 0]),
        ("green",         img[:, :, 1]),
        ("red",          img[:, :, 2]),
        ("hue",           cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 0]),
        ("saturation",    cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]),
        ("value",         cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]),
        ("lightness",     cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0]),
        ("green-magenta", cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]),
        ("blue-yellow",   cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 2]),
    ]
    colors = [
        "blue", "green", "red", "goldenrod", "gray",
        "black", "purple", "magenta", "cyan",
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    masked = mask > 0
    total = max(int(masked.sum()), 1)
    weights_template = np.full(total, 100.0 / total)
    for (name, ch), color in zip(channels, colors):
        pixels = ch[masked].flatten()
        ax.hist(pixels, bins=256, range=(0, 256), weights=weights_template,
                histtype="step", label=name, color=color, linewidth=1)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Proportion of pixels (%)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()

    fig.canvas.draw()
    pixels = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)


def _largest_blob(binary: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(binary)
    if cnts:
        biggest = max(cnts, key=cv2.contourArea)
        cv2.drawContours(out, [biggest], -1, 255, -1)
    return out


def _segment_leaf(img: np.ndarray) -> np.ndarray:
    masks = []
    for channel, otype in (("a", "dark"), ("b", "light")):
        gray = pcv.rgb2gray_lab(rgb_img=img, channel=channel)
        gray = pcv.gaussian_blur(img=gray, ksize=(5, 5), sigma_x=0)
        masks.append(pcv.threshold.triangle(gray_img=gray,
                                            object_type=otype))
    return _largest_blob(cv2.bitwise_or(masks[0], masks[1]))


def _pipeline(img: np.ndarray) -> dict:

    pcv.params.debug = None
    pcv.params.verbose = False

    binary = _segment_leaf(img)

    gray_hsv = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    gray_hsv[binary == 0] = 0
    blurred = pcv.gaussian_blur(img=gray_hsv, ksize=(5, 5), sigma_x=0)

    masked = pcv.apply_mask(img=img, mask=binary, mask_color="white")

    roi_img = img.copy()
    green_overlay = np.zeros_like(roi_img)
    green_overlay[binary > 0] = (0, 255, 0)
    roi_img = cv2.addWeighted(roi_img, 0.6, green_overlay, 0.4, 0)
    cnts, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        x0, y0, bw, bh = cv2.boundingRect(np.vstack(cnts))
        cv2.rectangle(roi_img, (x0, y0), (x0 + bw, y0 + bh), (255, 0, 0), 2)

    analyze_img = pcv.analyze.size(img=img, labeled_mask=binary, n_labels=1)

    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img, mask=binary
    )
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
        img=img, mask=binary
    )
    lm_img = img.copy()
    for pts, color in [
        (top,      (255, 0, 0)),
        (right,    (255, 0, 0)),
        (bottom,   (255, 0, 255)),
        (left,     (255, 0, 255)),
        (center_v, (0, 165, 255)),
        (center_h, (0, 165, 255)),
    ]:
        for pt in pts:
            cv2.circle(lm_img, (int(pt[0][0]), int(pt[0][1])), 3, color, -1)

    hist_arr = _color_histogram(img, binary)

    return {
        "Original": img,
        "Gaussian blur": blurred,
        "Mask": masked,
        "Roi objects": roi_img,
        "Analyze object": analyze_img,
        "Pseudolandmarks": lm_img,
        "Color histogram": hist_arr,
    }


def _show(results: dict) -> None:
    image_keys = [k for k in results if k != "Color histogram"]
    n = len(image_keys)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, key in zip(axes, image_keys):
        img = results[key]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(key)

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    if "Color histogram" in results:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.imshow(cv2.cvtColor(results["Color histogram"], cv2.COLOR_BGR2RGB))
        ax2.set_title("Color histogram")
        ax2.axis("off")
        fig2.tight_layout()

    plt.show()


def _save(results: dict, stem: str, dest_dir: str,
          include: set = None) -> None:
    for key in results:
        if include is not None and key not in include:
            continue
        img = results[key]
        label = key.lower().replace(" ", "_")
        out_path = os.path.join(dest_dir, f"{stem}_{label}.png")
        if not cv2.imwrite(out_path, img):
            print(f"Warning: failed to write {out_path}", file=sys.stderr)
        else:
            print(f"Saved {out_path}")


def run_single(image_path: str) -> int:
    if not os.path.isfile(image_path) or not is_image(image_path):
        print("Error: not an image file (.jpg/.jpeg/.png)", file=sys.stderr)
        return 1
    img, _, _ = pcv.readimage(filename=image_path, mode="rgb")
    results = _pipeline(img)
    _show(results)
    return 0


FLAG_TO_KEY = {
    "blur": "Gaussian blur",
    "mask": "Mask",
    "roi": "Roi objects",
    "analyze": "Analyze object",
    "landmarks": "Pseudolandmarks",
    "hist": "Color histogram",
}


def run_batch(src_dir: str, dst_dir: str,
              include: set = None) -> int:
    if not os.path.isdir(src_dir):
        print(f"Error: source is not a directory: {src_dir}", file=sys.stderr)
        return 1
    os.makedirs(dst_dir, exist_ok=True)

    images = list_images(src_dir)
    if not images:
        print(f"Error: no images found in {src_dir}", file=sys.stderr)
        return 1

    errors = 0
    for path in images:
        filename = os.path.basename(path)
        stem = os.path.splitext(filename)[0]
        try:
            img, _, _ = pcv.readimage(filename=path, mode="rgb")
            results = _pipeline(img)
            _save(results, stem, dst_dir, include=include)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to process {path}: {exc}", file=sys.stderr)
            errors += 1

    return 0 if errors == 0 else 1


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Transformation.py",
        description="Apply PlantCV image transformations to leaf images.",
        epilog=(
            "Examples:\n"
            "  ./Transformation.py image.JPG\n"
            "  ./Transformation.py -src Apple/apple_healthy/ -dst out/ -mask"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", nargs="?", help="Single image path.")
    parser.add_argument("-src", help="Source directory (batch).")
    parser.add_argument("-dst", help="Dest directory (batch).")
    for flag, key in FLAG_TO_KEY.items():
        parser.add_argument(
            f"-{flag}", action="store_true",
            help=f"(batch) Save the {key!r} panel.",
        )
    args = parser.parse_args(argv[1:])

    if args.image and (args.src or args.dst):
        parser.error("Provide either a single image or -src/-dst, not both.")
    if args.image and any(getattr(args, f) for f in FLAG_TO_KEY):
        parser.error("panel flags (-mask, ...) only work with -src/-dst.")
    if bool(args.src) != bool(args.dst):
        parser.error("-src and -dst must be used together.")
    if not args.image and not args.src:
        parser.error("give an image path, or -src and -dst.")
    return args


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.image:
        return run_single(args.image)

    selected = {FLAG_TO_KEY[f] for f in FLAG_TO_KEY if getattr(args, f)}
    include = ({"Original"} | selected) if selected else None
    return run_batch(args.src, args.dst, include=include)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
