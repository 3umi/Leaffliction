#!/usr/bin/env python3
import random
import sys
import os
import argparse
import zipfile
import shutil
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model import LeafDataset, LeafCNN
from Distribution import discover_classes, class_label
from balance import copy_originals, generate_for_class
from utils.images import list_images

TRAIN_DIR = "train_directory"
SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
MODEL_PATH = "model.pt"
ZIP_PATH = "leaffliction.zip"


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


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    model.train()
    total = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
    return total / len(loader.dataset)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
    return correct / len(loader.dataset)


def _list_train_set(out_dir: str) -> Tuple[List[str], List[str]]:
    paths, labels = [], []
    for label in sorted(os.listdir(out_dir)):
        folder = os.path.join(out_dir, label)
        if label.startswith(".") or not os.path.isdir(folder):
            continue
        for p in list_images(folder):
            paths.append(p)
            labels.append(label)
    return paths, labels


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


def _make_zip(zip_path: str) -> None:
    paths, _ = _list_train_set(TRAIN_DIR)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(MODEL_PATH)
        for path in paths:
            zf.write(path)
    print(f"Saved {zip_path}")


def main(argv: list) -> int:
    args = parse_args(argv)

    try:
        (train_originals, X_val,
         train_original_labels, y_val) = _split_originals(
            args.directory, seed=SEED,
        )
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if len(set(train_original_labels)) < 2:
        print("Error: training needs at least 2 classes.", file=sys.stderr)
        return 1

    shutil.rmtree(TRAIN_DIR, ignore_errors=True)

    rng = random.Random(SEED)

    groups = _group_by_label(train_originals, train_original_labels)

    try:
        failures = _build_train_set(groups, TRAIN_DIR, rng)
    except OSError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    if failures:
        print(f"Error: {failures} training image(s) failed.", file=sys.stderr)
        return 1

    torch.manual_seed(SEED)
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    train_paths, train_labels = _list_train_set(TRAIN_DIR)
    class_names = sorted(set(train_labels))
    train_loader = DataLoader(
        LeafDataset(train_paths, train_labels, class_names),
        batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(
        LeafDataset(X_val, y_val, class_names),
        batch_size=BATCH_SIZE, shuffle=False)
    model = LeafCNN(len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"{len(train_paths)} training / {len(X_val)} validation images, "
          f"{len(class_names)} classes, device {device}")
    for epoch in range(1, EPOCHS + 1):
        loss = _train_one_epoch(model, train_loader, loss_fn,
                                optimizer, device)
        acc = _evaluate(model, val_loader, device)
        print(f"epoch {epoch:2d}  train loss {loss:.3f}  "
              f"val accuracy {acc:.1%}")
    torch.save({
        "state_dict": model.state_dict(),
        "class_names": class_names,
    }, MODEL_PATH)
    _make_zip(ZIP_PATH)
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
