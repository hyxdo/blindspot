"""A small wrapper that turns numpy images into ranked classifier predictions."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class Prediction:
    index: int
    label: str
    prob: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an HxWx3 uint8 array to the CxHxW tensor torchvision expects."""
    return torch.from_numpy(image.copy()).permute(2, 0, 1)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Classifier:
    """Wraps a model with its preprocessing, labels, and an identifying record."""

    def __init__(
        self,
        model: nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor],
        labels: Sequence[str],
        identity: dict[str, Any],
        device: str | torch.device = "cpu",
        batch_size: int = 32,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.eval().to(self.device)
        self.preprocess = preprocess
        self.labels = list(labels)
        self.identity = identity
        self.batch_size = batch_size

    @classmethod
    def pretrained(
        cls,
        arch: str = "resnet18",
        checkpoint: str | Path | None = None,
        device: str | None = None,
    ) -> "Classifier":
        """Load a torchvision ImageNet model, optionally with fine-tuned weights."""
        weights = models.get_model_weights(arch).DEFAULT
        model = models.get_model(arch, weights=weights)
        identity: dict[str, Any] = {
            "arch": arch,
            "weights": str(weights),
            "checkpoint": None,
            "checkpoint_sha256": None,
        }
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state)
            identity["checkpoint"] = str(checkpoint)
            identity["checkpoint_sha256"] = file_sha256(checkpoint)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, weights.transforms(), weights.meta["categories"], identity, device)

    @torch.inference_mode()
    def probabilities(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """Return an N x num_classes array of softmax probabilities."""
        batches = []
        for start in range(0, len(images), self.batch_size):
            batch = torch.stack(
                [self.preprocess(to_tensor(image)) for image in images[start : start + self.batch_size]]
            )
            batches.append(self.model(batch.to(self.device)).softmax(dim=1).cpu().numpy())
        if not batches:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        return np.concatenate(batches)

    def rank(self, probs: np.ndarray, k: int = 5) -> list[Prediction]:
        top = np.argsort(-probs, kind="stable")[:k]
        return [Prediction(int(i), self.labels[int(i)], float(probs[i])) for i in top]

    def predict(self, images: Sequence[np.ndarray], k: int = 5) -> list[list[Prediction]]:
        return [self.rank(probs, k) for probs in self.probabilities(images)]
