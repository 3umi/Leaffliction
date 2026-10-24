"""Part 3 — Image transformation pipeline using PlantCV.

Single-image mode:
    ./Transformation.py <image_path>
    Displays 6 image panels + color histogram.

Batch mode:
    ./Transformation.py -src <source_dir> -dst <dest_dir> [-mask]
    Saves all transformation images to dest_dir.
"""
import argparse
import os
import sys
from typing import List

import io

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv

from utils.images import list_images

# ---------------------------------------------------------------------------
# Color histogram helper
# ---------------------------------------------------------------------------


def _color_histogram(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute a 9-channel color histogram and return it as an RGB array."""
    channels = [
        ("red",           img[:, :, 0]),
        ("green",         img[:, :, 1]),
        ("blue",          img[:, :, 2]),
        ("hue",           cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 0]),
        ("saturation",    cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 1]),
        ("value",         cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:, :, 2]),
        ("lightness",     cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0]),
        ("green-magenta", cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 1]),
        ("blue-yellow",   cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 2]),
    ]
    colors = [
        "red", "green", "blue", "goldenrod", "gray",
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

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    arr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------

def _segment_leaf(img: np.ndarray) -> np.ndarray:
    """Segment the leaf from the background.

    Computes three candidate masks and picks the largest plausible one:
      * LAB-a/dark:   isolates green leaves (originals + most cases)
      * LAB-b/light:  isolates yellow/brown leaves (Color augmentation)
      * LAB-L<253:    isolates any non-pure-white pixel (Lighting
                      augmentation bleaches the leaf near L=255)

    L<253 gives 0% on normal images (backgrounds aren't overexposed) but
    dominates when the Lighting augmentation blows out the highlights,
    which fragments the LAB triangle-threshold masks into swiss cheese.
    Picking the largest non-degenerate candidate handles all three cases
    without any per-augmentation branching.
    """
    candidates = []
    for channel, otype in (("a", "dark"), ("b", "light")):
        gray = pcv.rgb2gray_lab(rgb_img=img, channel=channel)
        gray = pcv.gaussian_blur(img=gray, ksize=(5, 5), sigma_x=0)
        m = pcv.threshold.triangle(gray_img=gray, object_type=otype)
        m = pcv.fill(bin_img=m, size=200)
        candidates.append(_isolate_leaf(m))

    gray = pcv.rgb2gray_lab(rgb_img=img, channel="l")
    gray = pcv.gaussian_blur(img=gray, ksize=(3, 3), sigma_x=0)
    m = pcv.threshold.binary(gray_img=gray, threshold=253, object_type="dark")
    m = pcv.fill(bin_img=m, size=100)
    candidates.append(_isolate_leaf(m))

    plausible = [c for c in candidates if 0.05 <= (c > 0).mean() <= 0.90]
    if plausible:
        return max(plausible, key=lambda c: (c > 0).sum())
    return candidates[0]


def _close_mask(binary: np.ndarray) -> np.ndarray:
    """Fill small internal holes with morphological closing.

    Overexposed leaves (Lighting augmentation) produce swiss-cheese masks
    where veins and highlights get excluded, which fragments the ROI /
    analyze / pseudolandmark panels. A small elliptical close bridges
    those pixels while leaving larger disease-spot holes intact.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _isolate_leaf(binary: np.ndarray) -> np.ndarray:
    """Force the binary mask to represent the leaf only.

    Triangle threshold can flip polarity on unusual color distributions,
    producing a "picture frame" mask that hugs every border and leaves the
    leaf as a hole. We invert only when the foreground clearly wraps the
    frame — high border density AND high overall area — then keep only the
    largest connected component so stray noise blobs don't leak in.
    """
    border = np.concatenate([
        binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
    ])
    border_density = (border > 0).mean()
    fg_density = (binary > 0).mean()
    # Only invert if the mask looks like it captured the background:
    # borders are almost fully foreground AND the area is majority foreground.
    if border_density > 0.9 and fg_density > 0.5:
        binary = cv2.bitwise_not(binary)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if n <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# PlantCV pipeline
# ---------------------------------------------------------------------------

def _pipeline(img: np.ndarray) -> dict:
    """Run the full PlantCV transformation pipeline on an RGB image.

    Returns a dict of named result images/figures.
    """
    pcv.params.debug = None
    pcv.params.verbose = False

    # 1. Segment the leaf — tries LAB-a first, falls back to LAB-b if the
    #    Color augmentation shifted the leaf away from green, then closes
    #    small internal holes from vein highlights.
    binary = _close_mask(_segment_leaf(img))
    # "Gaussian blur" panel: actual blur of the HSV-s channel, matching the
    # texture visible in subject figure IV.2 (not the binary mask).
    gray_hsv = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    blurred = pcv.gaussian_blur(img=gray_hsv, ksize=(5, 5), sigma_x=0)

    # 3. Apply mask to original
    masked = pcv.apply_mask(img=img, mask=binary, mask_color="white")

    # 4. ROI — full image rectangle
    h, w = img.shape[:2]
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=h, w=w)
    filtered_mask = pcv.roi.filter(mask=binary, roi=roi, roi_type="partial")

    # 5. ROI objects visualisation (green fill + blue bounding rect)
    roi_img = img.copy()
    green_overlay = np.zeros_like(roi_img)
    green_overlay[filtered_mask > 0] = (0, 255, 0)
    roi_img = cv2.addWeighted(roi_img, 0.6, green_overlay, 0.4, 0)
    cnts, _ = cv2.findContours(
        filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if cnts:
        x0, y0, bw, bh = cv2.boundingRect(np.vstack(cnts))
        cv2.rectangle(roi_img, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 255), 2)

    # 6. Analyze object — shape overlay (hull, axes, centroid)
    # PlantCV draws with BGR color values, so convert to BGR first so its
    # (255,0,0) perimeter renders as blue (not red) and (255,0,255) hull/axes
    # render as magenta, matching subject figure IV.5.
    bgr_for_analyze = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    analyze_result = pcv.analyze.size(
        img=bgr_for_analyze, labeled_mask=filtered_mask, n_labels=1
    )
    analyze_img = cv2.cvtColor(analyze_result, cv2.COLOR_BGR2RGB)

    # 7. Pseudolandmarks
    top, bottom, center_v = pcv.homology.x_axis_pseudolandmarks(
        img=img, mask=filtered_mask
    )
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
        img=img, mask=filtered_mask
    )
    lm_img = img.copy()
    for pts, color in [
        (top, (0, 0, 255)),
        (bottom, (255, 0, 255)),
        (center_v, (255, 165, 0)),
        (left, (255, 165, 0)),
        (right, (255, 165, 0)),
        (center_h, (255, 0, 255)),
    ]:
        for pt in pts:
            cv2.circle(lm_img, (int(pt[0][0]), int(pt[0][1])), 3, color, -1)

    # 8. Color histogram — 9 channels across RGB, HSV, LAB
    hist_arr = _color_histogram(img, filtered_mask)

    return {
        "Original": img,
        "Gaussian blur": blurred,
        "Mask": masked,
        "Roi objects": roi_img,
        "Analyze object": analyze_img,
        "Pseudolandmarks": lm_img,
        "Color histogram": hist_arr,
    }


# ---------------------------------------------------------------------------
# Display / save helpers
# ---------------------------------------------------------------------------

def _show(results: dict) -> None:
    """Display 6 image panels in one window, histogram in a second."""
    backend = matplotlib.get_backend().lower()
    non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}

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
            ax.imshow(img)
        ax.set_title(key)
        ax.axis("on")

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.tight_layout()

    if "Color histogram" in results:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.imshow(results["Color histogram"])
        ax2.set_title("Color histogram")
        ax2.axis("off")
        fig2.tight_layout()

    if backend in non_interactive:
        fig.savefig("transformation_result.png", bbox_inches="tight")
        print("No display available; saved panels to "
              "transformation_result.png")
        if "Color histogram" in results:
            fig2.savefig("transformation_histogram.png", bbox_inches="tight")
            print("Saved histogram to transformation_histogram.png")
    else:
        plt.show()


def _save(results: dict, stem: str, dest_dir: str,
          include: set = None) -> None:
    """Save each panel as <dest_dir>/<stem>_<label>.png.

    If `include` is given, only save panels whose title is in the set;
    otherwise save every panel.
    """
    for key in results:
        if include is not None and key not in include:
            continue
        img = results[key]
        label = key.lower().replace(" ", "_")
        out_path = os.path.join(dest_dir, f"{stem}_{label}.png")
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(out_path, bgr):
            print(f"Warning: failed to write {out_path}", file=sys.stderr)
        else:
            print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Single-image and batch modes
# ---------------------------------------------------------------------------


def run_single(image_path: str) -> int:
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        return 1
    img, _, _ = pcv.readimage(filename=image_path)
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
            img, _, _ = pcv.readimage(filename=path)
            results = _pipeline(img)
            _save(results, stem, dst_dir, include=include)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: failed to process {path}: {exc}", file=sys.stderr)
            errors += 1

    return 0 if errors == 0 else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    parser.add_argument("-src", dest="src", help="Source directory (batch).")
    parser.add_argument("-dst", dest="dst", help="Dest directory (batch).")
    for flag, key in FLAG_TO_KEY.items():
        parser.add_argument(
            f"-{flag}", action="store_true",
            help=f"(batch) Save the {key!r} panel.",
        )
    args = parser.parse_args(argv[1:])

    if args.image and (args.src or args.dst):
        parser.error("Provide either a single image or -src/-dst, not both.")
    if bool(args.src) != bool(args.dst):
        parser.error("-src and -dst must be used together.")
    if not args.image and not args.src:
        parser.print_help(sys.stderr)
        raise SystemExit(1)
    return args


def main(argv: List[str]) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0

    if args.image:
        return run_single(args.image)
    # Panel filter: if any -blur/-mask/... flag is set, save only those +
    # the Original. Otherwise save every panel (backward-compatible default).
    selected = {FLAG_TO_KEY[f] for f in FLAG_TO_KEY if getattr(args, f)}
    include = ({"Original"} | selected) if selected else None
    return run_batch(args.src, args.dst, include=include)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
