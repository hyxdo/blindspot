"""Run one image through a grid of perturbations and report what breaks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .cases import is_failure
from .model import Classifier, Prediction
from .perturb import Perturbation, apply, sweep_grid


@dataclass(frozen=True)
class SweepRow:
    perturbation: Perturbation
    top: list[Prediction]
    failed: bool


@dataclass(frozen=True)
class SweepResult:
    clean: list[Prediction]
    rows: list[SweepRow]

    @property
    def failures(self) -> list[SweepRow]:
        return [row for row in self.rows if row.failed]


def sweep(
    image: np.ndarray,
    classifier: Classifier,
    grid: Sequence[Perturbation] | None = None,
    true_label: str | None = None,
) -> SweepResult:
    grid = sweep_grid() if grid is None else list(grid)
    clean, *perturbed = classifier.predict([image, *[apply(image, p) for p in grid]])
    rows = [
        SweepRow(p, top, is_failure(clean, top, true_label)) for p, top in zip(grid, perturbed)
    ]
    return SweepResult(clean, rows)
