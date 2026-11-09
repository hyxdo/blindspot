import numpy as np
import torch
from torch import nn

from blindspot.breakpoints import Direction, all_directions, find_breakpoints
from blindspot.model import Classifier
from blindspot.perturb import AXES, Perturbation, apply


class BrightnessNet(nn.Module):
    """Says "bright" when mean brightness is above 0.3, otherwise "dark"."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = (x.mean(dim=(1, 2, 3)) - 0.3) * 100
        return torch.stack([score, -score], dim=1)


def brightness_classifier() -> Classifier:
    identity = {"arch": "brightness", "weights": None, "checkpoint": None, "checkpoint_sha256": None}
    return Classifier(BrightnessNet(), lambda t: t.float() / 255.0, ["bright", "dark"], identity)


def gray(value: int = 128) -> np.ndarray:
    return np.full((40, 40, 3), value, dtype=np.uint8)


def point_for(result, axis: str, limit: float):
    return next(p for p in result.points if p.direction == Direction(axis, limit))


def test_every_direction_moves_away_from_normal():
    directions = all_directions()
    assert Direction("blur", 0.0) not in directions
    assert Direction("blur", AXES["blur"].high) in directions
    assert Direction("exposure", AXES["exposure"].low) in directions
    assert not any(d.axis == "motion_angle" for d in directions)


def test_values_sit_on_the_slider_grid():
    direction = Direction("shift_x", 0.5)
    assert direction.value(0.0) == 0.0
    assert direction.value(1.0) == 0.5
    assert direction.value(0.33) == 0.16


def test_finds_the_exact_exposure_where_the_model_fails():
    result = find_breakpoints(gray(), brightness_classifier())
    point = point_for(result, "exposure", -4.0)
    assert point.perturbation.exposure == -0.8
    assert point.top[0].label == "dark"


def test_the_setting_just_before_the_breaking_point_still_passes():
    classifier = brightness_classifier()
    image = gray()
    assert classifier.predict([apply(image, Perturbation(exposure=-0.7))])[0][0].label == "bright"
    assert classifier.predict([apply(image, Perturbation(exposure=-0.8))])[0][0].label == "dark"


def test_settings_that_cannot_break_this_model_hold_to_the_limit():
    result = find_breakpoints(gray(), brightness_classifier())
    held = point_for(result, "rotate", 180.0)
    assert not held.broke
    assert held.perturbation is None


def test_results_are_sorted_most_fragile_first_with_held_points_last():
    points = find_breakpoints(gray(), brightness_classifier()).points
    flags = [p.broke for p in points]
    assert flags == sorted(flags, reverse=True)
    severities = [p.severity for p in points if p.broke]
    assert severities == sorted(severities)


def test_true_label_decides_what_counts_as_failing():
    result = find_breakpoints(gray(), brightness_classifier(), true_label="bright")
    assert point_for(result, "exposure", -4.0).perturbation.exposure == -0.8


def test_stops_early_when_the_clean_image_is_already_wrong():
    result = find_breakpoints(gray(), brightness_classifier(), true_label="dark")
    assert result.already_wrong
    assert result.points == []
