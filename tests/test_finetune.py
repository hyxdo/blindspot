import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision import models

from blindspot.finetune import (
    Example,
    ExperimentConfig,
    PerturbedDataset,
    bootstrap_difference,
    build_pool,
    compare_strategies,
    extract_features,
    finetune,
    join_resnet,
    predict,
    render_report,
    scores_by_image,
    select_indices,
    split_resnet,
    training_grid,
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


def dataset(images, labels) -> PerturbedDataset:
    examples = [Example(i, p) for i in range(len(images)) for p in GRID]
    return PerturbedDataset(images, labels, examples, preprocess)


def small_resnet() -> nn.Module:
    torch.manual_seed(0)
    return models.resnet18(weights=None).eval()


def test_pool_gives_each_image_distinct_perturbations():
    pool = build_pool(5, sweep_grid(), 3, np.random.default_rng(0))
    assert len(pool) == 15
    for image in range(5):
        assert len({e.perturbation for e in pool if e.image == image}) == 3


def test_pool_caps_at_the_grid_size():
    assert len(build_pool(2, GRID, 10, np.random.default_rng(0))) == 6


def test_holding_out_a_family_removes_only_that_family():
    allowed = training_grid(GRID, "blur")
    assert [p.family for p in allowed] == ["lighting", "framing"]
    assert training_grid(GRID, None) == GRID
    with pytest.raises(ValueError, match="family"):
        training_grid(GRID, "weather")


def test_selection_is_equal_budget_and_hard_only_uses_mistakes():
    wrong = np.array([i < 8 for i in range(20)])
    hard = select_indices(wrong, 5, "hard", np.random.default_rng(0))
    random = select_indices(wrong, 5, "random", np.random.default_rng(0))
    assert len(hard) == len(random) == 5
    assert len(set(hard.tolist())) == 5
    assert all(i < 8 for i in hard)


def test_selection_rejects_impossible_budgets_and_strategies():
    with pytest.raises(ValueError, match="budget"):
        select_indices(np.array([False]), 1, "hard", np.random.default_rng(0))
    with pytest.raises(ValueError, match="strategy"):
        select_indices(np.array([True]), 1, "nope", np.random.default_rng(0))


def test_finetune_learns_without_touching_the_base_or_frozen_layers():
    images, labels = colored_images()
    data = dataset(images, labels)
    base = net()
    before = {k: v.clone() for k, v in base.state_dict().items()}

    tuned = finetune(base, data, CLASSES, epochs=60, lr=0.05, unfrozen=("fc",))

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


def test_split_model_matches_the_whole_model():
    model = small_resnet()
    prefix, suffix = split_resnet(model)
    x = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(suffix(prefix(x)), model(x))


def test_cached_features_can_be_trained_on():
    prefix, suffix = split_resnet(small_resnet())
    data = TensorDataset(torch.rand(4, 3, 64, 64), torch.tensor([0, 1, 0, 1]))
    features = extract_features(prefix, data, batch_size=2)
    tuned = finetune(suffix, features, CLASSES, epochs=1, lr=0.01, batch_size=2)
    assert not torch.equal(tuned.fc.weight, suffix.fc.weight)


def test_join_uses_trained_layers_and_leaves_the_base_alone():
    base = small_resnet()
    _, suffix = split_resnet(base)
    tuned_suffix = finetune(
        suffix,
        TensorDataset(torch.rand(4, 256, 4, 4), torch.tensor([0, 1, 0, 1])),
        CLASSES,
        epochs=1,
        lr=0.01,
        batch_size=2,
    )
    original = base.fc.weight.clone()
    joined = join_resnet(base, tuned_suffix)
    assert torch.equal(joined.fc.weight, tuned_suffix.fc.weight)
    assert torch.equal(base.fc.weight, original)


def test_scores_by_image_groups_conditions():
    correct = np.array(
        [
            [True, True],  # clean
            [True, False],  # lighting
            [False, False],  # blur
            [True, True],  # framing
        ]
    )
    scores = scores_by_image(correct, GRID)
    assert scores["clean"].tolist() == [1.0, 1.0]
    assert scores["lighting"].tolist() == [1.0, 0.0]
    assert scores["blur"].tolist() == [0.0, 0.0]
    assert scores["perturbed"].tolist() == pytest.approx([2 / 3, 1 / 3])


def test_bootstrap_interval_contains_the_difference_and_detects_a_clear_gap():
    rng = np.random.default_rng(0)
    a = rng.random(200) * 0.1 + 0.9
    b = rng.random(200) * 0.1
    gap = bootstrap_difference(a, b, rng)
    assert gap["low"] <= gap["difference"] <= gap["high"]
    assert gap["low"] > 0.7


def test_bootstrap_interval_spans_zero_when_there_is_no_difference():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    same = bootstrap_difference(scores, rng.permutation(scores), rng)
    assert same["low"] < 0 < same["high"]


def resnet_preprocess(image: torch.Tensor) -> torch.Tensor:
    return nn.functional.interpolate(image[None].float() / 255.0, size=(64, 64), mode="bilinear")[0]


def run_small_comparison(holdout: str | None = None) -> dict:
    images, labels = colored_images(4)
    labels = [1 - label for label in labels]  # this untrained model is right on the originals
    config = ExperimentConfig(
        pool_images=4, per_image=2, eval_images=4, budget=2, epochs=1, seeds=2, batch_size=2,
        holdout_family=holdout,
    )
    return compare_strategies(
        small_resnet(), resnet_preprocess, images, labels, images, labels, CLASSES, config,
        grid=GRID, log=lambda _: None,
    )


def test_full_comparison_runs_and_reports_every_model():
    results = run_small_comparison()
    assert set(results["strategies"]) == {"hard", "random"}
    assert len(results["strategies"]["hard"]["runs"]) == 2
    assert results["train_examples"] == 2
    assert set(results["comparisons"]) == {"hard - random", "hard - baseline", "random - baseline"}
    json.dumps(results)

    report = render_report(results)
    for model in ("baseline", "hard", "random", "hard − random"):
        assert f"| {model} |" in report


def test_holdout_family_never_enters_the_training_pool():
    results = run_small_comparison(holdout="blur")
    assert results["pool_size"] == 8  # 4 images x the 2 remaining perturbations
    assert "blur (never trained on)" in render_report(results)
