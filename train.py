#!/usr/bin/env python3
import random
import sys
import os
import argparse
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split

from Distribution import discover_classes, class_label
from balance import copy_originals, generate_for_class
from utils.images import list_images

TRAIN_DIR = "train_directory"
SEED = 42


def _split_originals(
    root: str,
    seed: int,
    val_ratio: float = 0.2,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    paths, labels = [], []
    for key in discover_classes(root):
        label = class_label(key, root)
        for p in list_images(os.path.join(root, *key)):
            paths.append(p)
            labels.append(label)
    X_train, X_val, y_train, y_val = train_test_split(
        paths,
        labels,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels
    )
    return X_train, X_val, y_train, y_val


def _group_by_label(
    paths: List[str], labels: List[str],
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for path, label in zip(paths, labels):
        groups.setdefault(label, []).append(path)
    return groups


def _build_train_set(
    groups: Dict[str, List[str]],
    out_dir: str,
    rng: random.Random
) -> int:
    target = max(len(paths) for paths in groups.values())
    total_failures = 0

    for label, paths in groups.items():
        dest_dir = os.path.join(out_dir, label)
        os.makedirs(dest_dir)
        copied = copy_originals(paths, dest_dir)
        total_failures += len(paths) - copied
        needed = target - copied

        if needed <= 0:
            continue

        _, failures = generate_for_class(
            label, paths, needed, dest_dir, rng,
        )
        total_failures += failures

    return total_failures


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="train.py",
        description=(
            "Train a CNN to classify leaf diseases from a dataset "
            "directory, and save the model with its images in a .zip."
        ),
        epilog="Example: ./train.py images/",
    )
    parser.add_argument(
        "directory",
        help=(
            "Dataset directory (flat or nested layout, or a single class dir)."
        ),
    )

    args = parser.parse_args(argv[1:])
    if not os.path.exists(args.directory):
        parser.error(f"path does not exist: {args.directory}")
    if not os.path.isdir(args.directory):
        parser.error(f"not a directory: {args.directory}")
    return args


def main(argv: list) -> int:
    args = parse_args(argv)

    try:
        X_train, X_val, y_train, y_val = _split_originals(args.directory,
                                                          seed=SEED)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if os.path.exists(TRAIN_DIR):
        print(
            f"Error: training directory exists at {TRAIN_DIR}; remove "
            f"it first or use a different working directory.",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(SEED)

    groups = _group_by_label(X_train, y_train)

    try:
        failures = _build_train_set(groups, TRAIN_DIR, rng)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if failures:
        print(f"Error: {failures} training image(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Error: unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)
