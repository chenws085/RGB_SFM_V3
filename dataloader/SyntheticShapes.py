"""Small deterministic image dataset for end-to-end smoke tests."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class SyntheticShapesDataset(Dataset):
    """Binary circles-vs-squares dataset generated entirely in memory."""

    classes = ["circle", "square"]

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        del root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.size = 32 if train else 16
        self.seed_offset = 0 if train else 10_000

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int):
        label = index % 2
        rng = np.random.default_rng(self.seed_offset + index)
        background = tuple(int(v) for v in rng.integers(0, 25, size=3))
        foreground = (220, 45, 45) if label == 0 else (45, 220, 45)
        image = Image.new("RGB", (32, 32), background)
        draw = ImageDraw.Draw(image)
        jitter_x, jitter_y = (int(v) for v in rng.integers(-2, 3, size=2))
        box = (7 + jitter_x, 7 + jitter_y, 25 + jitter_x, 25 + jitter_y)
        if label == 0:
            draw.ellipse(box, fill=foreground)
        else:
            draw.rectangle(box, fill=foreground)

        if self.transform is not None:
            image = self.transform(image)
        target = torch.tensor(label, dtype=torch.long)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target
