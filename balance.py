"""Part 2 — Balance a leaf dataset via augmentation.

Discovers classes (via Distribution.py), groups them by plant, and
generates augmented images per class until every class within a plant
reaches that plant's max class count. Writes to `augmented_directory/`,
which mirrors the input structure with one subdirectory per class.
"""
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
from Distribution import class_label, discover_classes, group_by_plant


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
OUTPUT_DIR = "augmented_directory"
MAX_PAIR_COUNT = 3
MAX_REROLLS = 50


def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS


def _class_dir(root: str, components: Tuple[str, ...]) -> str:
    """Absolute directory holding the source images for a class."""
    if not components:
        return root
    return os.path.join(root, *components)


def _list_sources(class_path: str) -> List[str]:
    return sorted(
        e.path for e in os.scandir(class_path)
        if not e.name.startswith(".")
        and e.is_file(follow_symlinks=False)
        and _is_image(e.name)
    )


def _load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _save_rgb(path: str, image: np.ndarray) -> None:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(path, bgr)
    if not ok:
        raise IOError(f"failed to write {path}")


def _weighted_choice(
    sources: List[str],
    used: Dict[str, int],
    rng: random.Random,
) -> str:
    """Pick a source weighted by 1 / (1 + used_count)."""
    weights = [1.0 / (1 + used.get(s, 0)) for s in sources]
    return rng.choices(sources, weights=weights, k=1)[0]


def _compute_targets(
    groups: Dict[str, Dict[str, int]],
    override: int,
) -> Dict[str, int]:
    """Compute per-class target count.

    If `override` is a positive int, every class targets that count.
    Otherwise (None) the target for a class is the max count within
    its plant group.
    """
    targets: Dict[str, int] = {}
    for plant, class_counts in groups.items():
        plant_max = max(class_counts.values())
        target = override if override else plant_max
        for label in class_counts:
            targets[label] = target
    return targets


def _print_plan(
    groups: Dict[str, Dict[str, int]],
    targets: Dict[str, int],
) -> None:
    """Print the per-class balancing plan to stdout."""
    for plant in sorted(groups.keys()):
        for label in sorted(groups[plant].keys()):
            current = groups[plant][label]
            target = targets[label]
            if current >= target:
                print(f"{label}: {current} (at target)")
            else:
                needed = target - current
                print(
                    f"{label}: {current} → {target} "
                    f"(+{needed} augmented)"
                )


def _generate_for_class(
    label: str,
    sources: List[str],
    needed: int,
    dest_dir: str,
    rng: random.Random,
) -> Tuple[int, int]:
    """Generate `needed` augmentations into `dest_dir`.

    Returns (success_count, failure_count).
    """
    used: Dict[str, int] = {}
    pair: Dict[Tuple[str, str], int] = {}
    technique_idx: Dict[str, int] = {}
    successes = 0
    failures = 0

    for _ in range(needed):
        src = ""
        aug_name = ""
        found = False
        for _attempt in range(MAX_REROLLS):
            src = _weighted_choice(sources, used, rng)
            aug_name = rng.choice(augmentations.AUGMENTATION_NAMES)
            if pair.get((src, aug_name), 0) < MAX_PAIR_COUNT:
                found = True
                break
        if not found:
            print(
                f"Warning: could not find a fresh (source, augmentation) "
                f"pair for {label} after {MAX_REROLLS} rerolls; skipping.",
                file=sys.stderr,
            )
            failures += 1
            continue

        try:
            image = _load_rgb(src)
            aug_seed = rng.getrandbits(63)
            out_img = augmentations.apply(aug_name, image, seed=aug_seed)
        except Exception as exc:  # noqa: BLE001
            print(
                f"Warning: failed to augment {src} with "
                f"{aug_name}: {exc}",
                file=sys.stderr,
            )
            failures += 1
            continue

        base = os.path.basename(src)
        stem, ext = os.path.splitext(base)
        idx = technique_idx.get(aug_name, 0) + 1
        technique_idx[aug_name] = idx
        out_name = f"{stem}_{aug_name}_{idx:04d}{ext}"
        out_path = os.path.join(dest_dir, out_name)
        try:
            _save_rgb(out_path, out_img)
        except (IOError, OSError) as exc:
            print(
                f"Warning: failed to save {out_path}: {exc}",
                file=sys.stderr,
            )
            failures += 1
            continue

        used[src] = used.get(src, 0) + 1
        pair[(src, aug_name)] = pair.get((src, aug_name), 0) + 1
        successes += 1

    return successes, failures


def _copy_originals(sources: List[str], dest_dir: str) -> int:
    """Copy every source file into dest_dir. Returns count copied."""
    copied = 0
    for src in sources:
        try:
            shutil.copy2(src, os.path.join(dest_dir, os.path.basename(src)))
            copied += 1
        except (IOError, OSError) as exc:
            print(
                f"Warning: failed to copy {src}: {exc}",
                file=sys.stderr,
            )
    return copied


def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse and validate CLI args."""
    parser = argparse.ArgumentParser(
        prog="balance.py",
        description=(
            "Balance a leaf-image dataset by generating augmentations "
            "until each class within a plant reaches that plant's max."
        ),
        epilog="Example: ./balance.py images/ --seed 42",
    )
    parser.add_argument(
        "directory",
        help="Dataset root directory (e.g. images/).",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        help=(
            "Override per-class target count. Defaults to the per-plant "
            "max."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without writing any files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for reproducibility (default: time-based).",
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
    if args.target is not None and args.target <= 0:
        print("Error: --target must be a positive integer.", file=sys.stderr)
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

    groups = group_by_plant(classes, args.directory)

    targets = _compute_targets(groups, args.target)
    _print_plan(groups, targets)

    if args.dry_run:
        return 0

    if os.path.exists(OUTPUT_DIR):
        print(
            f"Error: output directory exists at {OUTPUT_DIR}; remove "
            f"it first or use a different working directory.",
            file=sys.stderr,
        )
        return 1

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

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
        sources = _list_sources(class_path)
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

        _copy_originals(sources, dest_dir)

        needed = targets[label] - len(sources)
        if needed <= 0:
            print(f"[{label}] at target ({len(sources)}).")
            continue

        print(
            f"[{label}] generating {needed} augmentations from "
            f"{len(sources)} sources..."
        )
        successes, failures = _generate_for_class(
            label, sources, needed, dest_dir, rng,
        )
        total_failures += failures
        print(
            f"[{label}] done: {successes} created, {failures} failed."
        )

    elapsed = time.time() - start
    print(f"Total time: {elapsed:.1f}s, total failures: {total_failures}.")
    return 0 if total_failures == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
