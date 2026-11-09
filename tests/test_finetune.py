import numpy as np
import pytest
import torch

from blindspot.finetune import (
    Example,
    PerturbedDataset,
    build_pool,
    evaluate,
    finetune,
    predict,
    select_examples,
)
from blindspot.perturb import Perturbation, sweep_grid

from conftest import TinyNet, preprocess

CLASSES = (0, 1)
GRID = [Perturbation(exposure=-1), Perturbation(blur=0.05), Perturbation(rotate=15)]


def colored_images(count: int = 12) -> tuple[list[np.ndarray], list[int]]:
    """Half red images labeled 0 and half blue images labeled 1."""
    rng = np.random.default_rng(1)
    images, labels = [], []
    for index in range(count):
        label = index % 2
        image = rng.integers(0, 40, (32, 32, 3), dtype=np.uint8)
        image[..., 0 if label == 0 else 2] = 220
        images.append(image)
        labels.append(label)
    return images, labels


def net() -> TinyNet:
    torch.manual_seed(0)
    return TinyNet()


def test_pool_gives_each_image_distinct_perturbations():
    pool = build_pool(5, sweep_grid(), 3, np.random.default_rng(0))
    assert len(pool) == 15
    for image in range(5):
        assert len({e.perturbation for e in pool if e.image == image}) == 3


def test_pool_caps_at_the_grid_size():
    assert len(build_pool(2, GRID, 10, np.random.default_rng(0))) == 6


def test_selection_is_equal_budget_and_hard_only_uses_mistakes():
    pool = [Example(i, Perturbation()) for i in range(20)]
    wrong = np.array([i < 8 for i in range(20)])
    hard = select_examples(pool, wrong, 5, "hard", np.random.default_rng(0))
    random = select_examples(pool, wrong, 5, "random", np.random.default_rng(0))
    assert len(hard) == len(random) == 5
    assert len({e.image for e in hard}) == 5
    assert all(e.image < 8 for e in hard)


def test_selection_rejects_impossible_budgets_and_strategies():
    pool = [Example(0, Perturbation())]
    with pytest.raises(ValueError, match="budget"):
        select_examples(pool, np.array([False]), 1, "hard", np.random.default_rng(0))
    with pytest.raises(ValueError, match="strategy"):
        select_examples(pool, np.array([True]), 1, "nope", np.random.default_rng(0))


def dataset(images, labels) -> PerturbedDataset:
    examples = [Example(i, p) for i in range(len(images)) for p in GRID]
    return PerturbedDataset(images, labels, examples, preprocess)


def test_finetune_learns_without_touching_the_base_or_frozen_layers():
    images, labels = colored_images()
    data = dataset(images, labels)
    base = net()
    before = {k: v.clone() for k, v in base.state_dict().items()}

    tuned = finetune(base, data, CLASSES, epochs=40, lr=0.05, unfrozen=("fc",))

    for key, value in base.state_dict().items():
        assert torch.equal(value, before[key])
    for key, value in tuned.state_dict().items():
        assert torch.equal(value, before[key]) == key.startswith("body")

    predicted, targets = predict(tuned, data, CLASSES)
    assert (predicted == targets).mean() == 1.0


def test_finetune_is_repeatable_for_a_seed():
    images, labels = colored_images()
    data = dataset(images, labels)
    first = finetune(net(), data, CLASSES, epochs=2, unfrozen=("fc",), seed=3)
    second = finetune(net(), data, CLASSES, epochs=2, unfrozen=("fc",), seed=3)
    for key, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[key])


def test_finetune_rejects_layers_that_do_not_exist():
    images, labels = colored_images()
    with pytest.raises(ValueError, match="no parameters"):
        finetune(net(), dataset(images, labels), CLASSES, unfrozen=("layer4",))


def test_evaluate_reports_clean_and_each_family():
    images, labels = colored_images()
    scores = evaluate(net(), images, labels, preprocess, CLASSES, GRID)
    assert set(scores) == {"clean", "perturbed", "lighting", "blur", "framing"}
    assert all(0 <= value <= 1 for value in scores.values())
    assert scores["perturbed"] == pytest.approx(np.mean([scores[k] for k in ("lighting", "blur", "framing")]))
