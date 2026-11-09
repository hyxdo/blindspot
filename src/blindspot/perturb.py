"""Deterministic image perturbations for lighting, blur, and framing.

Every perturbation is a pure function of the image and a `Perturbation` value.
There is no randomness here on purpose: the settings alone are enough to
regenerate the exact pixels that fooled a model.

Blur and shift amounts are fractions of the image size rather than pixels, so
a given setting means the same thing at any resolution.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Iterable, Mapping, NamedTuple

import cv2
import numpy as np

BORDER_MODE = cv2.BORDER_REFLECT_101

FAMILIES: dict[str, tuple[str, ...]] = {
    "lighting": ("exposure", "contrast", "gamma"),
    "blur": ("blur", "motion"),
    "framing": ("zoom", "shift_x", "shift_y", "rotate"),
}


class Axis(NamedTuple):
    low: float
    high: float
    step: float


# The range and resolution of each setting, shared by the sliders and searches.
AXES: dict[str, Axis] = {
    "exposure": Axis(-4.0, 4.0, 0.1),
    "contrast": Axis(0.0, 3.0, 0.05),
    "gamma": Axis(0.2, 4.0, 0.05),
    "blur": Axis(0.0, 0.1, 0.002),
    "motion": Axis(0.0, 0.3, 0.005),
    "motion_angle": Axis(0.0, 180.0, 5.0),
    "zoom": Axis(0.5, 3.0, 0.05),
    "shift_x": Axis(-0.5, 0.5, 0.01),
    "shift_y": Axis(-0.5, 0.5, 0.01),
    "rotate": Axis(-180.0, 180.0, 1.0),
}

# Single-axis severities used by sweeps and by the held-out evaluation.
SWEEP_AXES: dict[str, tuple[float, ...]] = {
    "exposure": (-3.0, -2.0, -1.0, 1.0, 2.0),
    "contrast": (0.2, 0.5, 1.5, 2.5),
    "gamma": (0.35, 0.6, 1.8, 3.0),
    "blur": (0.01, 0.02, 0.04),
    "motion": (0.05, 0.1, 0.2),
    "zoom": (1.3, 1.7, 2.5),
    "shift_x": (-0.3, -0.15, 0.15, 0.3),
    "shift_y": (-0.3, -0.15, 0.15, 0.3),
    "rotate": (-45.0, -20.0, 20.0, 45.0),
}


@dataclass(frozen=True)
class Perturbation:
    """Settings for one perturbed view of an image. Defaults change nothing."""

    exposure: float = 0.0  # photographic stops; +1 doubles linear brightness
    contrast: float = 1.0  # multiplier around the image's mean brightness
    gamma: float = 1.0  # above 1 darkens midtones, below 1 lifts them
    blur: float = 0.0  # Gaussian sigma as a fraction of the short side
    motion: float = 0.0  # streak length as a fraction of the short side
    motion_angle: float = 0.0  # streak direction in degrees
    zoom: float = 1.0  # above 1 crops in, below 1 pulls out
    shift_x: float = 0.0  # fraction of width, positive moves content right
    shift_y: float = 0.0  # fraction of height, positive moves content down
    rotate: float = 0.0  # degrees counter-clockwise

    def __post_init__(self) -> None:
        for field in fields(self):
            if not np.isfinite(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be finite")
        for name in ("contrast", "blur", "motion"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        for name in ("gamma", "zoom"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Perturbation":
        known = {field.name for field in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown perturbation settings: {sorted(unknown)}")
        return cls(**{key: float(value) for key, value in data.items()})

    @property
    def changed(self) -> dict[str, float]:
        """Settings that differ from the defaults."""
        default = Perturbation().to_dict()
        return {k: v for k, v in self.to_dict().items() if v != default[k]}

    @property
    def is_identity(self) -> bool:
        return not (set(self.changed) - {"motion_angle"})

    @property
    def family(self) -> str:
        """Which family this perturbation belongs to, or "mixed"."""
        active = set(self.changed) - {"motion_angle"}
        matches = [name for name, axes in FAMILIES.items() if active & set(axes)]
        if not matches:
            return "none"
        return matches[0] if len(matches) == 1 else "mixed"

    def describe(self) -> str:
        changed = self.changed
        if self.motion == 0:
            changed.pop("motion_angle", None)
        return ", ".join(f"{k}={v:g}" for k, v in changed.items()) or "none"


def sweep_grid(axes: Iterable[str] | None = None) -> list[Perturbation]:
    """Return one perturbation per axis and severity, varying a single setting."""
    names = list(SWEEP_AXES) if axes is None else list(axes)
    unknown = set(names) - set(SWEEP_AXES)
    if unknown:
        raise ValueError(f"unknown sweep axes: {sorted(unknown)}")
    base = Perturbation()
    return [replace(base, **{name: value}) for name in names for value in SWEEP_AXES[name]]


def apply(image: np.ndarray, p: Perturbation) -> np.ndarray:
    """Return a perturbed copy of an RGB uint8 image.

    The order follows how a photo forms: scene light, then blur, then framing.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected an HxWx3 uint8 image, got {image.shape} {image.dtype}")
    return _frame(_blur(_light(image, p), p), p)


def _light(image: np.ndarray, p: Perturbation) -> np.ndarray:
    if p.exposure == 0 and p.contrast == 1 and p.gamma == 1:
        return image.copy()
    x = image.astype(np.float32) / 255.0
    # Clip between stages: a blown-out highlight stays blown out, as on a sensor.
    x = np.clip(x * 2.0**p.exposure, 0.0, 1.0)
    mean = x.mean()
    x = np.clip((x - mean) * p.contrast + mean, 0.0, 1.0)
    x = x**p.gamma
    return np.rint(x * 255.0).astype(np.uint8)


def _blur(image: np.ndarray, p: Perturbation) -> np.ndarray:
    short_side = min(image.shape[:2])
    out = image
    if p.blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=p.blur * short_side, borderType=BORDER_MODE)
    if p.motion > 0:
        length = max(1, round(p.motion * short_side))
        out = cv2.filter2D(out, -1, _motion_kernel(length, p.motion_angle), borderType=BORDER_MODE)
    return out


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    size = length | 1  # the kernel needs a center pixel
    center = size // 2
    kernel = np.zeros((size, size), np.float32)
    kernel[center, :] = 1.0
    rotation = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (size, size), flags=cv2.INTER_LINEAR)
    return kernel / kernel.sum()


def _frame(image: np.ndarray, p: Perturbation) -> np.ndarray:
    if p.zoom == 1 and p.shift_x == 0 and p.shift_y == 0 and p.rotate == 0:
        return image
    height, width = image.shape[:2]
    center = ((width - 1) / 2.0, (height - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(center, p.rotate, p.zoom)
    matrix[0, 2] += p.shift_x * width
    matrix[1, 2] += p.shift_y * height
    # Reflecting the border avoids adding black wedges the model could key on.
    return cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=BORDER_MODE)
