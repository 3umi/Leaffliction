#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

from utils.images import list_images


def _iter_subdirs(path: str) -> List[os.DirEntry]:
    return [
        e for e in os.scandir(path)
        if not e.name.startswith(".")
        and e.is_dir(follow_symlinks=False)
    ]


def _count_all_images(path: str) -> int:
    total = len(list_images(path))
    for sub in _iter_subdirs(path):
        total += _count_all_images(sub.path)
    return total


def discover_classes(root: str) -> Dict[Tuple[str, ...], int]:
    classes: Dict[Tuple[str, ...], int] = {}

    n = len(list_images(root))
    if n > 0:
        if any(_count_all_images(s.path) for s in _iter_subdirs(root)):
            raise ValueError(
                f"{n} loose image(s) directly in {root}: move them "
                f"into a class folder or remove them"
            )
        classes[()] = n
        return classes

    for s in _iter_subdirs(root):
        n = len(list_images(s.path))
        if n > 0:
            classes[(s.name,)] = n
            continue
        for c in _iter_subdirs(s.path):
            n = len(list_images(c.path))
            if n > 0:
                classes[(s.name, c.name)] = n

    total = _count_all_images(root)
    found = sum(classes.values())
    if total == 0:
        raise ValueError(f"no images found under {root}")
    if found != total:
        raise ValueError(
            f"{total - found} image(s) under {root} are not in a class "
            f"folder (expected a single, flat or nested layout)"
        )
    return classes


def class_label(components: Tuple[str, ...], root: str) -> str:
    if not components:
        return os.path.basename(os.path.normpath(root)) or root
    return "_".join(components)


def _plant_of(components: Tuple[str, ...], root: str) -> str:
    if len(components) == 2:
        return components[0]
    return _plant_from_flat_name(class_label(components, root))


def _plant_from_flat_name(name: str) -> str:
    for sep in ("_", "-"):
        idx = name.find(sep)
        if idx > 0:
            return name[:idx]
    return name


def group_by_plant(
    classes: Dict[Tuple[str, ...], int],
    root: str,
) -> Dict[str, Dict[str, int]]:
    groups: Dict[str, Dict[str, int]] = {}
    for key, count in classes.items():
        plant = _plant_of(key, root)
        label = class_label(key, root)
        bucket = groups.setdefault(plant, {})
        bucket[label] = count
    return groups


def make_charts(plant: str, class_counts: Dict[str, int]) -> plt.Figure:
    labels = sorted(class_counts.keys())
    counts = [class_counts[name] for name in labels]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % cmap.N) for i in range(len(labels))]

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))
    title = f"{plant} class distribution"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_pie.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax_pie.set_aspect("equal")

    bars = ax_bar.bar(labels, counts, color=colors)
    ax_bar.set_ylabel("Image count")
    ax_bar.tick_params(axis="x", rotation=30)
    for tick in ax_bar.get_xticklabels():
        tick.set_horizontalalignment("right")
    for rect, value in zip(bars, counts):
        ax_bar.text(
            rect.get_x() + rect.get_width() / 2,
            rect.get_height(),
            str(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="Distribution.py",
        description=(
            "Analyze a leaf-image dataset and display a pie and bar chart "
            "per plant type."
        ),
        epilog="Example: ./Distribution.py images/",
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


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    try:
        classes = discover_classes(args.directory)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    groups = group_by_plant(classes, args.directory)

    for plant in sorted(groups):
        make_charts(plant, groups[plant])

    plt.show()
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
