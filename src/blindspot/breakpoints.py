"""Find how far each setting can move before a model's prediction fails.

A fixed grid only says which of its settings failed. This search walks every
setting away from normal in both directions, stops at the first failure, then
bisects to find the smallest change that causes it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from .cases import is_failure
from .model import Classifier, Prediction
from .perturb import AXES, Perturbation, apply


@dataclass(frozen=True)
class Direction:
    """One way a setting can move: from its normal value toward a limit."""

    axis: str
    limit: float

    def value(self, severity: float) -> float:
        """The setting at `severity`, from 0 (normal) to 1 (limit), on the axis's step."""
        start = getattr(Perturbation(), self.axis)
        step = AXES[self.axis].step
        return round(round((start + severity * (self.limit - start)) / step) * step, 6)

    def at(self, severity: float) -> Perturbation:
        return replace(Perturbation(), **{self.axis: self.value(severity)})


def all_directions() -> list[Direction]:
    directions = []
    for axis, (low, high, _) in AXES.items():
        if axis == "motion_angle":  # only meaningful alongside motion blur
            continue
        start = getattr(Perturbation(), axis)
        directions += [Direction(axis, limit) for limit in (low, high) if limit != start]
    return directions


@dataclass(frozen=True)
class BreakPoint:
    direction: Direction
    severity: float | None  # None if the model held all the way to the limit
    perturbation: Perturbation | None
    top: list[Prediction] | None

    @property
    def broke(self) -> bool:
        return self.severity is not None


@dataclass(frozen=True)
class BreakpointResult:
    clean: list[Prediction]
    points: list[BreakPoint]  # most fragile first
    already_wrong: bool = False

    @property
    def broken(self) -> list[BreakPoint]:
        return [point for point in self.points if point.broke]


def find_breakpoints(
    image: np.ndarray,
    classifier: Classifier,
    true_label: str | None = None,
    steps: int = 6,
    refinements: int = 4,
) -> BreakpointResult:
    """Search every direction for the smallest change that makes the model fail."""
    clean = classifier.predict([image])[0]
    if is_failure(clean, clean, true_label):
        return BreakpointResult(clean, [], already_wrong=True)

    def run(settings: Sequence[Perturbation]) -> list[tuple[list[Prediction], bool]]:
        tops = classifier.predict([apply(image, p) for p in settings])
        return [(top, is_failure(clean, top, true_label)) for top in tops]

    directions = all_directions()
    severities = [(index + 1) / steps for index in range(steps)]
    outcomes = run([d.at(s) for d in directions for s in severities])

    # For each direction that failed: [last passing severity, first failing severity, top].
    spans: list[list | None] = []
    for index in range(len(directions)):
        row = outcomes[index * steps : (index + 1) * steps]
        first = next((j for j, (_, failed) in enumerate(row) if failed), None)
        if first is None:
            spans.append(None)
        else:
            spans.append([severities[first - 1] if first else 0.0, severities[first], row[first][0]])

    for _ in range(refinements):
        active = [index for index, span in enumerate(spans) if span is not None]
        if not active:
            break
        middles = [(spans[i][0] + spans[i][1]) / 2 for i in active]
        results = run([directions[i].at(m) for i, m in zip(active, middles)])
        for i, middle, (top, failed) in zip(active, middles, results):
            if failed:
                spans[i][1:] = [middle, top]
            else:
                spans[i][0] = middle

    points = [
        BreakPoint(d, None, None, None)
        if span is None
        else BreakPoint(d, span[1], d.at(span[1]), span[2])
        for d, span in zip(directions, spans)
    ]
    points.sort(key=lambda p: (p.severity is None, p.severity or 0.0))
    return BreakpointResult(clean, points)
