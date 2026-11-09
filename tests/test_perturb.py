import numpy as np
import pytest

from blindspot.perturb import FAMILIES, SWEEP_AXES, Perturbation, apply, sweep_grid


def gray(value: int, height: int = 40, width: int = 60) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_defaults_change_nothing(image):
    out = apply(image, Perturbation())
    assert np.array_equal(out, image)
    assert out is not image


def test_input_is_not_mutated_and_output_is_repeatable(image):
    before = image.copy()
    p = Perturbation(exposure=1, blur=0.03, motion=0.1, zoom=1.4, rotate=12)
    first = apply(image, p)
    second = apply(image, p)
    assert np.array_equal(image, before)
    assert np.array_equal(first, second)


def test_one_stop_of_exposure_doubles_brightness():
    assert apply(gray(60), Perturbation(exposure=1)).mean() == pytest.approx(120, abs=1)
    assert apply(gray(60), Perturbation(exposure=-1)).mean() == pytest.approx(30, abs=1)


def test_gamma_above_one_darkens():
    assert apply(gray(128), Perturbation(gamma=2)).mean() < 128


def test_contrast_zero_flattens_to_mean(image):
    out = apply(image, Perturbation(contrast=0))
    assert out.std() < 1
    assert out.mean() == pytest.approx(image.mean(), abs=1)


@pytest.mark.parametrize("settings", [{"blur": 0.05}, {"motion": 0.2}])
def test_blur_reduces_detail(image, settings):
    assert apply(image, Perturbation(**settings)).std() < image.std() * 0.8


def test_blur_scale_is_relative_to_image_size():
    def edge_width(size: int) -> float:
        image = gray(0, size, size)
        image[:, size // 2 :] = 255
        row = apply(image, Perturbation(blur=0.05))[size // 2, :, 0]
        return np.count_nonzero((row > 25) & (row < 230)) / size

    assert edge_width(100) > 0.05
    assert edge_width(100) == pytest.approx(edge_width(200), rel=0.1)


def test_motion_angle_sets_streak_direction():
    image = gray(0, 51, 51)
    image[25, 25] = 255
    horizontal = apply(image, Perturbation(motion=0.2))[..., 0]
    vertical = apply(image, Perturbation(motion=0.2, motion_angle=90))[..., 0]
    assert np.ptp(np.flatnonzero(horizontal[25])) > 5
    assert np.flatnonzero(horizontal[:, 25]).size <= 3
    assert np.ptp(np.flatnonzero(vertical[:, 25])) > 5


def test_shift_moves_content_by_a_fraction_of_the_image():
    image = gray(0, 40, 60)
    image[10, 20] = 255
    out = apply(image, Perturbation(shift_x=0.5, shift_y=0.25))
    assert out[20, 50, 0] == 255


def test_zoom_keeps_the_center_fixed():
    image = gray(0, 41, 41)
    image[20, 20] = 255
    assert apply(image, Perturbation(zoom=2))[20, 20, 0] > 200


def test_half_turn_flips_the_image():
    image = gray(0, 41, 41)
    image[5, 10] = 255
    assert apply(image, Perturbation(rotate=180))[35, 30, 0] == 255


@pytest.mark.parametrize(
    "bad",
    [
        {"gamma": 0},
        {"zoom": -1},
        {"blur": -0.1},
        {"motion": -0.1},
        {"contrast": -1},
        {"exposure": float("nan")},
    ],
)
def test_invalid_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        Perturbation(**bad)


@pytest.mark.parametrize(
    "bad", [np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float32), np.zeros((4, 4, 4), np.uint8)]
)
def test_invalid_images_are_rejected(bad):
    with pytest.raises(ValueError):
        apply(bad, Perturbation())


def test_settings_round_trip_and_reject_unknown_keys():
    p = Perturbation(exposure=-2, rotate=15)
    assert Perturbation.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError):
        Perturbation.from_dict({"sharpen": 1})


def test_family_and_description():
    assert Perturbation().family == "none"
    assert Perturbation(motion_angle=45).is_identity
    assert Perturbation(gamma=2).family == "lighting"
    assert Perturbation(blur=0.1, rotate=4).family == "mixed"
    assert Perturbation(exposure=-2).describe() == "exposure=-2"
    assert Perturbation().describe() == "none"


def test_grid_covers_every_axis_with_real_changes():
    grid = sweep_grid()
    assert len(grid) == sum(len(values) for values in SWEEP_AXES.values())
    assert not any(p.is_identity for p in grid)
    assert {p.family for p in grid} == set(FAMILIES)
    with pytest.raises(ValueError):
        sweep_grid(["sharpen"])
