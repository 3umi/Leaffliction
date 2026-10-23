import argparse
import os
import sys
from typing import Dict, Iterable, List, Tuple

import matplotlib

import matplotlib.pyplot as plt

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

def _is_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in IMAGE_EXTS

def _iter_subdirs(path: str) -> List[os.DirEntry]:
    return [
        e for e in os.scandir(path)
        if not e.name.startswith(".")
        and e.is_dir(follow_symlinks=False)
    ]

def _count_images(path: str) -> int:
    return sum(
        1 for e in os.scandir(path)
        if not e.name.startswith(".")
        and e.is_file(follow_symlinks=False)
        and _is_image(e.name)
    )

def _count_all_images(path: str) -> int:
    total = _count_images(path)
    for sub in _iter_subdirs(path):
        total += _count_all_images(sub.path)
    return total

def discover_classes(root: str) -> Dict[Tuple[str, ...], int]:
    classes: Dict[Tuple[str, ...], int] = {}

    n = _count_images(root)
    if n > 0:
        if any(_count_all_images(s.path) for s in _iter_subdirs(root)):
            raise ValueError(
                f"{n} loose image(s) directly in {root}: move them "
                f"into a class folder or remove them"
            )
        classes[()] = n
        return classes

    for s in _iter_subdirs(root):
        n = _count_images(s.path)
        if n > 0:
            classes[(s.name,)] = n
            continue
        for c in _iter_subdirs(s.path):
            n = _count_images(c.path)
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

def _flatten_label(components: Tuple[str, ...], root: str) -> str:
    """Flatten relative path components to an underscore-joined display label.

    If the discovery root *is* the single class directory (no components),
    fall back to the basename of `root` itself.
    """
    if not components:
        return os.path.basename(os.path.normpath(root)) or root
    return "_".join(components)


def _plant_of(components: Tuple[str, ...], root: str) -> str:
    """Derive the plant group for a class, per the two-tier rule (§5)."""
    if not components:
        # Root itself is the class -> its own plant group.
        name = os.path.basename(os.path.normpath(root)) or root
        return _plant_from_flat_name(name)
    if len(components) > 1:
        return components[0]
    return _plant_from_flat_name(components[0])


def _plant_from_flat_name(name: str) -> str:
    """
    Plant token = substring before first '_' or '-'; whole name if neither.
    """
    for sep in ("_", "-"):
        idx = name.find(sep)
        if idx > 0:
            return name[:idx]
    return name


def group_by_plant(
    classes: Dict[Tuple[str, ...], int],
    root: str,
) -> Dict[str, Dict[str, int]]:
    """
    Group discovered classes by plant. Returns {plant: {display_label: count}}.
    """
    groups: Dict[str, Dict[str, int]] = {}
    for components, count in classes.items():
        plant = _plant_of(components, root)
        label = _flatten_label(components, root)
        bucket = groups.setdefault(plant, {})
        if label in bucket:
            # Should be impossible (relative paths are unique);
            raise ValueError(f"Duplicate class label detected: {label}")
        bucket[label] = count
    return groups


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------
def make_charts(plant: str, class_counts: Dict[str, int]) -> plt.Figure:
    """Build one figure with a pie + bar chart for a single plant group."""
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


_NON_INTERACTIVE_BACKENDS = {
    "agg", "cairo", "pdf", "pgf", "ps", "svg", "template",
}


def _backend_is_interactive() -> bool:
    """True if the active matplotlib backend can actually display a window."""
    try:
        backend = matplotlib.get_backend().lower()
        return backend not in _NON_INTERACTIVE_BACKENDS
    except Exception:  # noqa: BLE001
        return False


def _save_figures(figures: List[Tuple[str, "plt.Figure"]]) -> None:
    """
    Save each figure as `<plant>_distribution.png` in the current directory.
    """
    for plant, fig in figures:
        out = f"{plant}_distribution.png"
        try:
            fig.savefig(out, bbox_inches="tight")
            print(f"Saved {out}")
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: could not save {out}: {exc}", file=sys.stderr)


def show_or_save(figures: List[Tuple[str, "plt.Figure"]]) -> None:
    """
    Display figures if possible, otherwise save them to disk.
    """
    if not _backend_is_interactive():
        _save_figures(figures)
        return
    try:
        plt.show()
    except Exception as exc:  # noqa: BLE001
        print(
            f"Display unavailable ({exc}); saving figures instead.",
            file=sys.stderr,
        )
        _save_figures(figures)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: List[str]) -> argparse.Namespace:
    """Parse CLI args. Exits cleanly (no traceback) on validation failure."""
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
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as exc:
        # argparse already printed a clean message; propagate its code.
        raise SystemExit(exc.code)

    if not os.path.exists(args.directory):
        print(f"Error: path does not exist: {args.directory}", file=sys.stderr)
        raise SystemExit(1)
    if not os.path.isdir(args.directory):
        print(f"Error: not a directory: {args.directory}", file=sys.stderr)
        raise SystemExit(1)
    return args


def main(argv: List[str]) -> int:
    """Program entry point."""
    try:
        args = parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 0

    try:
        classes = discover_classes(args.directory)
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not classes:
        print(
            f"Error: no class directories with images found under "
            f"{args.directory}",
            file=sys.stderr,
        )
        return 1

    try:
        groups = group_by_plant(classes, args.directory)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    figures: List[Tuple[str, plt.Figure]] = []
    for plant in sorted(groups.keys()):
        fig = make_charts(plant, groups[plant])
        figures.append((plant, fig))

    show_or_save(figures)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        sys.exit(130)
