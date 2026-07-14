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

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# PlantCV pipeline
# ---------------------------------------------------------------------------

def _pipeline(img: np.ndarray) -> dict:
    """Run the full PlantCV transformation pipeline on an RGB image.

    Returns a dict of named result images/figures.
    """
    pcv.params.debug = None
    pcv.params.verbose = False

    # 1. Gaussian blur — HSV-s for display, LAB-b for segmentation
    gray_hsv = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    blurred = pcv.gaussian_blur(img=gray_hsv, ksize=(5, 5), sigma_x=0)
    gray_lab = pcv.rgb2gray_lab(rgb_img=img, channel="b")
    blurred_lab = pcv.gaussian_blur(img=gray_lab, ksize=(5, 5), sigma_x=0)

    # 2. Triangle threshold on LAB-b → clean mask
    binary = pcv.threshold.triangle(
        gray_img=blurred_lab, object_type="light"
    )
    binary = pcv.fill(bin_img=binary, size=200)

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
    analyze_img = pcv.analyze.size(
        img=img, labeled_mask=filtered_mask, n_labels=1
    )

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

    # 8. Color histogram figure
    hist_fig = pcv.visualize.histogram(img=img, mask=filtered_mask)

    return {
        "Original": img,
        "Gaussian blur": blurred,
        "Mask": masked,
        "Roi objects": roi_img,
        "Analyze object": analyze_img,
        "Pseudolandmarks": lm_img,
        "_hist_fig": hist_fig,
    }


# ---------------------------------------------------------------------------
# Display / save helpers
# ---------------------------------------------------------------------------

def _show(results: dict) -> None:
    """Display 6 image panels + color histogram."""
    panel_keys = [k for k in results if not k.startswith("_")]
    n = len(panel_keys)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = axes.flatten()

    for ax, key in zip(axes, panel_keys):
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

    backend = matplotlib.get_backend().lower()
    non_interactive = {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}
    if backend in non_interactive:
        out = "transformation_result.png"
        fig.savefig(out, bbox_inches="tight")
        print(f"No display available; saved panels to {out}")
    else:
        plt.show()

    hist_fig = results.get("_hist_fig")
    if hist_fig is not None:
        try:
            hist_fig.show()
        except Exception:  # noqa: BLE001
            hist_fig.save("transformation_histogram.png")


def _save(results: dict, stem: str, dest_dir: str) -> None:
    """Save each panel as <dest_dir>/<stem>_<label>.png."""
    panel_keys = [k for k in results if not k.startswith("_")]
    for key in panel_keys:
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

    hist_fig = results.get("_hist_fig")
    if hist_fig is not None:
        hist_path = os.path.join(dest_dir, f"{stem}_color_histogram.png")
        try:
            hist_fig.save(hist_path)
            print(f"Saved {hist_path}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not save histogram: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Single-image and batch modes
# ---------------------------------------------------------------------------

def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def run_single(image_path: str) -> int:
    if not os.path.isfile(image_path):
        print(f"Error: file not found: {image_path}", file=sys.stderr)
        return 1
    img, _, _ = pcv.readimage(filename=image_path)
    results = _pipeline(img)
    _show(results)
    return 0


def run_batch(src_dir: str, dst_dir: str) -> int:
    if not os.path.isdir(src_dir):
        print(f"Error: source is not a directory: {src_dir}", file=sys.stderr)
        return 1
    os.makedirs(dst_dir, exist_ok=True)

    images: List[str] = [
        f for f in sorted(os.listdir(src_dir)) if _is_image(f)
    ]
    if not images:
        print(f"Error: no images found in {src_dir}", file=sys.stderr)
        return 1

    errors = 0
    for filename in images:
        path = os.path.join(src_dir, filename)
        stem = os.path.splitext(filename)[0]
        try:
            img, _, _ = pcv.readimage(filename=path)
            results = _pipeline(img)
            _save(results, stem, dst_dir)
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
    parser.add_argument(
        "-mask", action="store_true", help="(batch) Include mask outputs."
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
    return run_batch(args.src, args.dst)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        sys.exit(130)
