# Leaffliction

Image classification of leaf diseases (Apple, Grape) — 42 project.

## Setup

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Programs

- `Distribution.py <dir>` — class distribution charts (Part 1)
- `Augmentation.py <image>` — 6 augmentations (Part 2)
- `Transformation.py <image | -src DIR -dst DIR>` — image transformations (Part 3)
- `train.py <dir>` — train classifier (Part 4)
- `predict.py <image | dir> [-val]` — predict disease for an image, or print
  accuracy over a directory; `predict.py images/ -val` scores the validation
  split held out by `train.py` (Part 4)

## Dataset

Not committed. SHA1 of the dataset zip lives in `signature.txt`.
