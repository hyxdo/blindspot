"""Save model failures to disk and replay them later.

A case holds everything needed to regenerate a failure: a lossless copy of the
source image, the perturbation settings, which model produced it, what that
model predicted, and the library versions in use.

    cases/<id>.json
    images/<pixel hash>.png
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
import torchvision

from .model import Classifier, Prediction
from .perturb import Perturbation, apply

DEFAULT_ROOT = Path("blindspot_cases")
PROB_TOLERANCE = 0.01


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }


def pixel_hash(image: np.ndarray) -> str:
    """Hash decoded pixels, so re-encoding a file doesn't change its identity."""
    digest = hashlib.sha256()
    digest.update(repr(image.shape).encode())
    digest.update(image.tobytes())
    return digest.hexdigest()


def read_image(source: str | Path | bytes) -> np.ndarray:
    """Decode an image file to RGB uint8. Works with non-ASCII Windows paths."""
    data = source if isinstance(source, (bytes, bytearray)) else Path(source).read_bytes()
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def is_failure(
    clean: Sequence[Prediction], perturbed: Sequence[Prediction], true_label: str | None = None
) -> bool:
    """A failure is a wrong top-1 if the truth is known, else a changed top-1."""
    if true_label is not None:
        return perturbed[0].label != true_label
    return perturbed[0].index != clean[0].index


@dataclass(frozen=True)
class Case:
    id: str
    created_at: str
    image_sha256: str
    image_file: str
    true_label: str | None
    perturbation: dict[str, float]
    model: dict[str, Any]
    clean_top: list[dict[str, Any]]
    perturbed_top: list[dict[str, Any]]
    environment: dict[str, str]
    note: str = ""

    @property
    def settings(self) -> Perturbation:
        return Perturbation.from_dict(self.perturbation)


@dataclass(frozen=True)
class Replay:
    case_id: str
    reproduced: bool
    stored_top1: str
    replayed_top1: str
    max_prob_delta: float
    model_matches: bool
    environment_changes: dict[str, tuple[str, str]]


class CaseStore:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)
        self.case_dir = self.root / "cases"
        self.image_dir = self.root / "images"

    def save(
        self,
        image: np.ndarray,
        perturbation: Perturbation,
        classifier: Classifier,
        true_label: str | None = None,
        note: str = "",
    ) -> Case:
        """Run the model and record the result. Saving the same case twice is a no-op."""
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.image_dir.mkdir(parents=True, exist_ok=True)

        image_hash = pixel_hash(image)
        settings = perturbation.to_dict()
        key = json.dumps([image_hash, settings, classifier.identity, true_label], sort_keys=True)
        case_id = hashlib.sha256(key.encode()).hexdigest()[:12]
        if (self.case_dir / f"{case_id}.json").exists():
            return self.load(case_id)

        image_file = f"{image_hash}.png"
        self._write_image(self.image_dir / image_file, image)
        clean, perturbed = classifier.predict([image, apply(image, perturbation)])
        case = Case(
            id=case_id,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            image_sha256=image_hash,
            image_file=image_file,
            true_label=true_label,
            perturbation=settings,
            model=classifier.identity,
            clean_top=[p.to_dict() for p in clean],
            perturbed_top=[p.to_dict() for p in perturbed],
            environment=environment(),
            note=note,
        )
        (self.case_dir / f"{case_id}.json").write_text(
            json.dumps(asdict(case), indent=2), encoding="utf-8"
        )
        return case

    def load(self, case_id: str) -> Case:
        path = self.case_dir / f"{case_id}.json"
        if not path.exists():
            raise KeyError(f"no saved case {case_id!r} in {self.root}")
        return Case(**json.loads(path.read_text(encoding="utf-8")))

    def all(self) -> list[Case]:
        """Return every saved case, newest first."""
        cases = [self.load(path.stem) for path in self.case_dir.glob("*.json")]
        return sorted(cases, key=lambda case: (case.created_at, case.id), reverse=True)

    def image(self, case: Case) -> np.ndarray:
        """Load a case's source image, refusing it if the pixels have changed."""
        image = read_image(self.image_dir / case.image_file)
        if pixel_hash(image) != case.image_sha256:
            raise ValueError(f"stored image for case {case.id} no longer matches its hash")
        return image

    def replay(self, case: Case, classifier: Classifier) -> Replay:
        """Regenerate a case's image and check the model still does the same thing."""
        probs = classifier.probabilities([apply(self.image(case), case.settings)])[0]
        top = classifier.rank(probs, k=1)[0]
        delta = max(abs(float(probs[p["index"]]) - p["prob"]) for p in case.perturbed_top)
        stored = case.perturbed_top[0]
        now = environment()
        return Replay(
            case_id=case.id,
            reproduced=top.index == stored["index"] and delta <= PROB_TOLERANCE,
            stored_top1=stored["label"],
            replayed_top1=top.label,
            max_prob_delta=delta,
            model_matches=classifier.identity == case.model,
            environment_changes={
                k: (case.environment.get(k, "?"), v)
                for k, v in now.items()
                if case.environment.get(k) != v
            },
        )

    @staticmethod
    def _write_image(path: Path, image: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError("could not encode image")
        path.write_bytes(encoded.tobytes())
