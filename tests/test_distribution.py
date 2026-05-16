"""
Tests for Distribution.py — synthetic structures, CLI errors, real-data lock.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

import Distribution as D


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cb\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "Distribution.py"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------
def write_image(path: Path, name: str) -> None:
    """Write a dummy image file (extension matters, content doesn't)."""
    (path / name).write_bytes(PNG_BYTES)


def make_class_dir(
    base: Path, rel_name: str, count: int, ext: str = ".JPG"
) -> Path:
    """
    Create a class directory (possibly nested via '/') with `count` images.
    """
    d = base / rel_name
    d.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        write_image(d, f"img{i:03d}{ext}")
    return d


def run_cli(*args: str, cwd: Path = None) -> subprocess.CompletedProcess:
    """
    Invoke Distribution.py as a real subprocess and capture exit + streams.
    """
    env = {**os.environ, "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(cwd) if cwd else None,
        env=env,
    )


# ---------------------------------------------------------------------------
# §3.1 — Flat structure (synthetic mirror of real dataset shape)
# ---------------------------------------------------------------------------
def test_flat_structure_counts_and_groups(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple_healthy", 3)
    make_class_dir(root, "Apple_scab", 5)
    make_class_dir(root, "Grape_Esca", 2)
    make_class_dir(root, "Grape_healthy", 4)

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    assert set(groups.keys()) == {"Apple", "Grape"}
    assert groups["Apple"] == {"Apple_healthy": 3, "Apple_scab": 5}
    assert groups["Grape"] == {"Grape_Esca": 2, "Grape_healthy": 4}


# ---------------------------------------------------------------------------
# §3.2 — Nested structure (subject's example shape)
# ---------------------------------------------------------------------------
def test_nested_structure_uses_first_path_component(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple/apple_healthy", 3)
    make_class_dir(root, "Apple/apple_scab", 5)
    make_class_dir(root, "Grape/grape_healthy", 2)

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    # Containers `Apple` and `Grape` are not classes themselves.
    assert all(len(k) == 2 for k in classes.keys())
    assert set(groups.keys()) == {"Apple", "Grape"}
    # Labels are flattened relative paths.
    assert groups["Apple"] == {
        "Apple_apple_healthy": 3,
        "Apple_apple_scab": 5,
    }
    assert groups["Grape"] == {"Grape_grape_healthy": 2}


# ---------------------------------------------------------------------------
# §3.3 — Basename collision must not silently merge
# ---------------------------------------------------------------------------
def test_basename_collision_not_merged(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple/healthy", 7)
    make_class_dir(root, "Grape/healthy", 11)

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    # Two distinct internal keys.
    assert len(classes) == 2
    # Counts preserved separately; labels disambiguated.
    assert groups["Apple"]["Apple_healthy"] == 7
    assert groups["Grape"]["Grape_healthy"] == 11


# ---------------------------------------------------------------------------
# §3.4 — Single class directory (root IS the class)
# ---------------------------------------------------------------------------
def test_single_class_directory(tmp_path):
    root = tmp_path / "Apple_healthy"
    root.mkdir()
    for i in range(6):
        write_image(root, f"img{i}.JPG")

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    assert classes == {(): 6}
    assert groups == {"Apple": {"Apple_healthy": 6}}


# ---------------------------------------------------------------------------
# §3.5 — Container with a stray loose image (Rule C §3.3)
# ---------------------------------------------------------------------------
def test_stray_image_in_container_is_ignored(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple_healthy", 3)
    make_class_dir(root, "Apple_scab", 5)
    write_image(root, "stray.JPG")

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    # No phantom class for the stray; only the real classes counted.
    assert set(groups.keys()) == {"Apple"}
    assert groups["Apple"] == {"Apple_healthy": 3, "Apple_scab": 5}
    # Defensive: total of all classes equals only the real images.
    assert sum(classes.values()) == 8


# ---------------------------------------------------------------------------
# §3.6 — Hidden / system files present
# ---------------------------------------------------------------------------
def test_hidden_entries_ignored(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple_healthy", 3)
    # Hidden file at root level (mirrors .DS_Store in the real dataset).
    (root / ".DS_Store").write_bytes(b"junk")
    # Hidden subdir that DOES contain images — must still be ignored.
    hidden = root / ".hidden_dir"
    hidden.mkdir()
    write_image(hidden, "img.JPG")

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    assert groups == {"Apple": {"Apple_healthy": 3}}


# ---------------------------------------------------------------------------
# §3.7 — Empty / no-image directories
# ---------------------------------------------------------------------------
def test_empty_sibling_directory_is_skipped(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple_healthy", 3)
    (root / "empty_dir").mkdir()

    classes = D.discover_classes(str(root))
    assert classes == {("Apple_healthy",): 3}


def test_wholly_empty_root_yields_nothing_and_does_not_crash(tmp_path):
    root = tmp_path / "empty_root"
    root.mkdir()

    classes = D.discover_classes(str(root))
    assert classes == {}

    # Also exercise the full CLI: must NOT traceback, must signal an error.
    result = run_cli(str(root))
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in combined
    assert combined.strip() != ""


# ---------------------------------------------------------------------------
# §3.8 — Mixed-case image extensions + non-image files
# ---------------------------------------------------------------------------
def test_mixed_case_extensions_all_counted_non_image_ignored(tmp_path):
    root = tmp_path / "data"
    cls = root / "Apple_healthy"
    cls.mkdir(parents=True)
    for name in ["a.JPG", "b.jpg", "c.JPEG", "d.png", "e.PNG"]:
        (cls / name).write_bytes(PNG_BYTES)
    (cls / "notes.txt").write_text("ignore me")
    (cls / "README.md").write_text("ignore me too")

    classes = D.discover_classes(str(root))
    assert classes == {("Apple_healthy",): 5}


# ---------------------------------------------------------------------------
# §3.9 — Class name with no separator (degenerate plant grouping)
# ---------------------------------------------------------------------------
def test_class_name_without_separator_is_own_plant(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "leaves", 4)

    classes = D.discover_classes(str(root))
    groups = D.group_by_plant(classes, str(root))

    assert groups == {"leaves": {"leaves": 4}}


# ---------------------------------------------------------------------------
# §3.10 — Bounded recursion / deep junk doesn't crash or hang
# ---------------------------------------------------------------------------
def test_deep_empty_chain_contributes_nothing(tmp_path):
    root = tmp_path / "data"
    make_class_dir(root, "Apple_healthy", 3)
    # Pointless deep empty chain — no images anywhere along it.
    (root / "a" / "b" / "c" / "d" / "e").mkdir(parents=True)

    classes = D.discover_classes(str(root))
    assert classes == {("Apple_healthy",): 3}


# ---------------------------------------------------------------------------
# §4 — CLI / error edge cases (auto-0 protectors)
# ---------------------------------------------------------------------------
def _assert_clean_cli_error(result: subprocess.CompletedProcess) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"Expected non-zero exit, got 0. Output:\n{combined}"
    )
    assert combined.strip() != "", (
        "Expected some user-facing message on stdout/stderr"
    )
    assert "Traceback" not in combined, (
        f"Program produced a traceback (would score 0):\n{combined}"
    )


def test_cli_no_argument():
    _assert_clean_cli_error(run_cli())


def test_cli_nonexistent_path(tmp_path):
    _assert_clean_cli_error(run_cli(str(tmp_path / "no_such_dir")))


def test_cli_path_is_a_file(tmp_path):
    f = tmp_path / "image.JPG"
    f.write_bytes(PNG_BYTES)
    _assert_clean_cli_error(run_cli(str(f)))


def test_cli_too_many_arguments():
    _assert_clean_cli_error(run_cli("a", "b", "c"))


# ---------------------------------------------------------------------------
# §5 — Real-dataset sanity (skipped if dataset absent)
# ---------------------------------------------------------------------------
DATASET_PATH = os.environ.get(
    "LEAFFLICTION_DATASET",
    str(PROJECT_ROOT / "images"),
)
DATASET_AVAILABLE = os.path.isdir(DATASET_PATH)
_SKIP_REASON = f"Real dataset not available at {DATASET_PATH}"


@pytest.mark.skipif(not DATASET_AVAILABLE, reason=_SKIP_REASON)
def test_real_dataset_exact_counts():
    """Intentionally brittle — must be consciously updated if data changes."""
    classes = D.discover_classes(DATASET_PATH)
    groups = D.group_by_plant(classes, DATASET_PATH)
    assert groups["Apple"] == {
        "Apple_Black_rot": 620,
        "Apple_healthy": 1640,
        "Apple_rust": 275,
        "Apple_scab": 629,
    }
    assert groups["Grape"] == {
        "Grape_Black_rot": 1178,
        "Grape_Esca": 1382,
        "Grape_healthy": 422,
        "Grape_spot": 1075,
    }


@pytest.mark.skipif(not DATASET_AVAILABLE, reason=_SKIP_REASON)
def test_real_dataset_structural_invariants():
    classes = D.discover_classes(DATASET_PATH)
    groups = D.group_by_plant(classes, DATASET_PATH)

    # Two plant groups exactly.
    assert set(groups.keys()) == {"Apple", "Grape"}
    # Eight classes total.
    assert sum(len(c) for c in groups.values()) == 8

    expected_labels = {
        "Apple_Black_rot", "Apple_healthy", "Apple_rust", "Apple_scab",
        "Grape_Black_rot", "Grape_Esca", "Grape_healthy", "Grape_spot",
    }
    actual_labels = set().union(*(c.keys() for c in groups.values()))
    assert actual_labels == expected_labels

    # .DS_Store at the images/ root must not have produced a phantom class.
    assert not any(".DS_Store" in lbl for lbl in actual_labels)

    # Known imbalances hold (these motivate Part 2's augmentation).
    assert groups["Apple"]["Apple_rust"] < groups["Apple"]["Apple_healthy"]
    assert groups["Grape"]["Grape_healthy"] < groups["Grape"]["Grape_Esca"]


# ---------------------------------------------------------------------------
# §6 — Plot smoke test (Agg, non-blocking, runs without raising)
# ---------------------------------------------------------------------------
def test_make_charts_smoke(tmp_path):
    fig = D.make_charts(
        "Test", {"Test_alpha": 3, "Test_beta": 5, "Test_gamma": 2}
    )
    assert fig is not None
    out = tmp_path / "smoke.png"
    fig.savefig(str(out))
    assert out.exists() and out.stat().st_size > 0
