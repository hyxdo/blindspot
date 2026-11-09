from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from blindspot.model import Classifier


class TinyNet(nn.Module):
    """Small stand-in with the same frozen-body, trainable-head shape as a ResNet."""

    def __init__(self, classes: int = 3) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(8, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.body(x).flatten(1))


def preprocess(image: torch.Tensor) -> torch.Tensor:
    resized = nn.functional.interpolate(image[None].float() / 255.0, size=(32, 32), mode="bilinear")
    return resized[0]


def make_classifier(seed: int = 0) -> Classifier:
    torch.manual_seed(seed)
    identity = {"arch": "tiny", "weights": f"seed{seed}", "checkpoint": None, "checkpoint_sha256": None}
    return Classifier(TinyNet(), preprocess, ["apple", "boat", "cat"], identity)


@pytest.fixture
def classifier() -> Classifier:
    return make_classifier()


@pytest.fixture
def image() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (64, 80, 3), dtype=np.uint8)
