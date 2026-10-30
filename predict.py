#!/usr/bin/env python3
"""Part 4 — Predict the disease of leaf images with the trained model.

Single image:
    ./predict.py <image>
    Displays the original and transformed image with the predicted class.

Directory:
    ./predict.py <directory> [-val]
    Prints the accuracy over every class folder in the directory. With
    -val, the directory is the original dataset given to train.py and
    only its held-out validation split is evaluated.
"""
import argparse
import os
import sys
from collections import Counter
from typing import List, Tuple

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from plantcv import plantcv as pcv
from torch.utils.data import DataLoader

from Distribution import class_label, discover_classes
from model import LeafCNN, LeafDataset, preprocess
from Transformation import _close_mask, _segment_leaf
from train import BATCH_SIZE, MODEL_PATH, SEED, _split_originals
from utils.images import is_image, list_images


def _device() -> str:
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def _load_model(path: str, device: str) -> Tuple[LeafCNN, List[str]]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    class_names = checkpoint["class_names"]
    model = LeafCNN(len(class_names)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, class_names


def _transformed(bgr: np.ndarray) -> np.ndarray:
    """Leaf isolated on a white background, as in Transformation's mask."""
    pcv.params.debug = None
    mask = _close_mask(_segment_leaf(bgr))
    return pcv.apply_mask(img=bgr, mask=mask, mask_color="white")


def _show(rgb: np.ndarray, transformed: np.ndarray, label: str) -> None:
    fig = plt.figure(figsize=(8, 5.5), facecolor="black")
    for i, (img, title) in enumerate(
        [(rgb, "Original"), (transformed, "Transformed")]
    ):
        ax = fig.add_axes([0.03 + i * 0.49, 0.28, 0.45, 0.65])
        ax.imshow(img)
        ax.set_title(title, color="white")
        ax.axis("off")
    fig.text(0.5, 0.17, "===      DL classification      ===",
             color="white", ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.07, f"Class predicted : {label}",
             color="limegreen", ha="center", fontsize=14)

    backend = matplotlib.get_backend().lower()
    if backend in {"agg", "cairo", "pdf", "pgf", "ps", "svg", "template"}:
        fig.savefig("predict_result.png", facecolor="black")
        print("No display available; saved figure to predict_result.png")
    else:
        plt.show()


def predict_image(path: str, model_path: str) -> int:
    device = _device()
    model, class_names = _load_model(model_path, device)
    with torch.no_grad():
        logits = model(preprocess(path).unsqueeze(0).to(device))
    label = class_names[int(logits.argmax(dim=1))]
    print(label)

    bgr = cv2.imread(path)
    _show(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
          cv2.cvtColor(_transformed(bgr), cv2.COLOR_BGR2RGB), label)
    return 0


def _labeled_images(root: str, val: bool) -> Tuple[List[str], List[str]]:
    if val:
        _, paths, _, labels = _split_originals(root, seed=SEED)
        return paths, labels
    paths, labels = [], []
    for key in discover_classes(root):
        label = class_label(key, root)
        for p in list_images(os.path.join(root, *key)):
            paths.append(p)
            labels.append(label)
    return paths, labels


def evaluate_directory(root: str, model_path: str, val: bool) -> int:
    device = _device()
    model, class_names = _load_model(model_path, device)
    paths, labels = _labeled_images(root, val)
    unknown = sorted(set(labels) - set(class_names))
    if unknown:
        print(f"Error: classes unknown to the model: {', '.join(unknown)}",
              file=sys.stderr)
        return 1

    loader = DataLoader(LeafDataset(paths, labels, class_names),
                        batch_size=BATCH_SIZE, shuffle=False)
    correct: Counter = Counter()
    with torch.no_grad():
        for images, targets in loader:
            preds = model(images.to(device)).argmax(dim=1).cpu()
            for target, pred in zip(targets.tolist(), preds.tolist()):
                correct[target] += int(target == pred)

    totals = Counter(loader.dataset.targets)
    for i, name in enumerate(class_names):
        if totals[i]:
            print(f"{name:<20} {correct[i]:5d}/{totals[i]:<5d} "
                  f"{correct[i] / totals[i]:.1%}")
    total = sum(totals.values())
    print(f"{'accuracy':<20} {sum(correct.values()):5d}/{total:<5d} "
          f"{sum(correct.values()) / total:.1%}")
    return 0


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "Predict leaf diseases with the model saved by train.py. An "
            "image is displayed with its prediction; a directory of class "
            "folders is evaluated and its accuracy printed."
        ),
        epilog=(
            "Examples:\n"
            "  ./predict.py 'images/Apple_healthy/image (1).JPG'\n"
            "  ./predict.py images/ -val"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Image, or directory of class folders.")
    parser.add_argument(
        "-model", default=MODEL_PATH,
        help=f"Model file saved by train.py (default: {MODEL_PATH}).",
    )
    parser.add_argument(
        "-val", action="store_true",
        help=(
            "Treat the directory as the dataset given to train.py and "
            "evaluate only its validation split."
        ),
    )

    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.path):
        parser.error(f"path does not exist: {args.path}")
    if not os.path.isfile(args.model):
        parser.error(f"model not found: {args.model} (run train.py first)")
    if args.val and not os.path.isdir(args.path):
        parser.error("-val needs the dataset directory given to train.py")
    return args


def main(argv: list) -> int:
    args = parse_args(argv)
    try:
        if os.path.isdir(args.path):
            return evaluate_directory(args.path, args.model, args.val)
        if not is_image(args.path):
            print(f"Error: not an image: {args.path}", file=sys.stderr)
            return 1
        return predict_image(args.path, args.model)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
