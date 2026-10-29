import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn


IMG_SIZE = 128


def preprocess(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)


class LeafDataset(Dataset):

    def __init__(self, paths, labels, class_names):
        index = {class_name: i for i, class_name in enumerate(class_names)}
        self.paths = paths
        self.targets = [index[label] for label in labels]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return preprocess(self.paths[i]), self.targets[i]


class LeafCNN(nn.Module):
    """4 conv blocks (pattern finders) + a linear classifier."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))
