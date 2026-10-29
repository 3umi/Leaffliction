#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np

import augmentations
from Distribution import class_label, discover_classes

from utils.images import list_images


OUTPUT_DIR = "augmented_directory"


def _output_name(src: str, aug_name: str) -> str:
    stem, ext = os.path.splitext(os.path.basename(src))
    return f"{stem}_{aug_name}{ext}"


def _class_dir(root: str, components: Tuple[str, ...]) -> str:
    return os.path.join(root, *components)


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_rgb(path: str, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(path, bgr):
        raise OSError(f"failed to write {path}")


def _print_plan(counts, target):
    print(f"{'class':<16} {'images':>6} {'to add':>7}")
    for label in sorted(counts):
        n = counts[label]
        print(f"{label:<16} {n:>6} {target - n:>+7}")
    print(f"target: {target} images per class")


def generate_for_class(
    label: str,
    sources: List[str],
    needed: int,
    dest_dir: str,
    rng: random.Random,
) -> Tuple[int, int]:
    successes = 0
    failures = 0

    taken = {os.path.basename(s) for s in sources}
    pairs = [
        (source, aug)
        for source in sources
        for aug in augmentations.AUGMENTATION_NAMES
        if _output_name(source, aug) not in taken
    ]

    if needed > len(pairs):
        print(f"Warning: {label}: only {len(pairs)} unique augmentations "
              f"possible, {needed - len(pairs)} short of the target",
              file=sys.stderr)
        failures = needed - len(pairs)
        needed = len(pairs)

    rng.shuffle(pairs)

    for src, aug_name in pairs[:needed]:
        try:
            image = _load_rgb(src)
            out_img = augmentations.apply(aug_name, image,
                                          seed=rng.getrandbits(63))
            _save_rgb(os.path.join(dest_dir, _output_name(src, aug_name)),
                      out_img)
            successes += 1
        except (ValueError, OSError) as exc:
            print(f"Warning: {label}: {src} + {aug_name}: {exc}",
                  file=sys.stderr)
            failures += 1
    return successes, failures


def copy_originals(sources: List[str], dest_dir: str) -> int:
    copied = 0
    for src in sources:
        try:
            shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
            copied += 1
        except OSError as exc:
            print(
                f"Warning: failed to copy {src}: {exc}",
                file=sys.stderr,
            )
    return copied


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="balance.py",
        description=(
            "Balance a leaf-image dataset by generating augmentations "
            "until every class reaches the size of the largest class."
        ),
        epilog="Example: ./balance.py images/ --seed 42",
    )
    parser.add_argument(
        "directory",
        help="Dataset root directory (e.g. images/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without writing any files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed - the same seed gives "
             "identical output (default: 42).",
    )
    args = parser.parse_args(argv[1:])

    if not os.path.exists(args.directory):
        print(
            f"Error: path does not exist: {args.directory}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if not os.path.isdir(args.directory):
        print(
            f"Error: not a directory: {args.directory}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return args


def main(argv: List[str]) -> int:
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0

    try:
        classes = discover_classes(args.directory)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    counts = {class_label(c, args.directory): n for c, n in classes.items()}
    target = max(counts.values())
    _print_plan(counts, target)

    if args.dry_run:
        return 0

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    rng = random.Random(args.seed)

    label_to_components: Dict[str, Tuple[str, ...]] = {
        class_label(c, args.directory): c for c in classes
    }

    try:
        os.makedirs(OUTPUT_DIR)
    except OSError as exc:
        print(
            f"Error: could not create {OUTPUT_DIR}: {exc}",
            file=sys.stderr,
        )
        return 1

    total_failures = 0
    start = time.time()

    for label in sorted(label_to_components.keys()):
        components = label_to_components[label]
        class_path = _class_dir(args.directory, components)
        sources = list_images(class_path)
        if not sources:
            print(
                f"Warning: no source images in {class_path}; skipping.",
                file=sys.stderr,
            )
            continue

        dest_dir = os.path.join(OUTPUT_DIR, label)
        try:
            os.makedirs(dest_dir)
        except OSError as exc:
            print(
                f"Error: could not create {dest_dir}: {exc}",
                file=sys.stderr,
            )
            return 1

        copied = copy_originals(sources, dest_dir)
        total_failures += len(sources) - copied
        needed = target - copied

        if needed <= 0:
            continue

        successes, failures = generate_for_class(
            label, sources, needed, dest_dir, rng,
        )
        total_failures += failures
        print(f"{label}: {successes} created"
              + (f", {failures} FAILED" if failures else ""))

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s, total failures: {total_failures}.")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
